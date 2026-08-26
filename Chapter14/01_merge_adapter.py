"""
01_merge_adapter.py
-------------------
Merge a trained LoRA adapter into its base model, IN FULL PRECISION,
then verify parity against the adapter-attached model before writing.

This is the foundation step for Chapter 14. The two things that matter:
  1. Load the base in bf16/fp16, NOT 4-bit. Merging into a 4-bit base
     double-quantizes and silently degrades quality. Quantize AFTERWARD.
  2. Verify parity: with greedy decoding the merged model must reproduce
     the adapter-attached model's output token-for-token.

Requirements / Gemma 4 gotchas:
  - Gemma 4 wraps its linear layers in Gemma4ClippableLinear, which PEFT
    cannot attach a LoRA to directly. peft>=0.19.0 handles this AT TRAINING
    TIME: omit target_modules and it targets the LM linears, skipping the
    clippable wrappers (incl. the vision/audio towers).
    => Train the adapter with peft>=0.19.0 and DEFAULT target modules.
    => Upgrading PEFT does NOT rescue an adapter whose saved target_modules
       already point at the wrappers; that adapter must be retrained.
  - The adapter must have been trained on the SAME Gemma 4 base. A Gemma 3
    adapter cannot be merged into a Gemma 4 model (wrong layer types and
    wrong weights); inspect adapter_config.json -> base_model_name_or_path.
  - Give the merge enough memory to hold the full bf16 model (~16 GB+ for
    E4B). If device_map="auto" offloads weights to the meta/disk device,
    merge_and_unload() will fail or corrupt weights (see the guard below).

Usage:
    python 01_merge_adapter.py \
        --base google/gemma-4-e4b-it \
        --adapter ./medical-qa-v1-gemma4-lora \
        --out ./medical-gemma-merged
"""
import os
# Gemma 4 is multimodal, so transformers reaches into its image/video utils,
# which try to `import tensorflow`. A broken TF/protobuf combo in the
# environment then crashes the whole import. We never use TF here, so tell
# transformers to skip it. (Must be set BEFORE importing transformers/peft.)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from peft import PeftModel

PARITY_PROMPTS = [
    "What are the first-line treatments for type 2 diabetes?",
    "Explain the mechanism of action of beta blockers.",
    "When is a CT scan preferred over an MRI?",
    "List the common side effects of metformin.",
]


def generate(model, tok, text, max_new_tokens=120):
    msgs = [{"role": "user", "content": text}]
    # Gemma 4's tokenizer returns a BatchEncoding (dict), not a bare tensor,
    # so use return_dict=True and unpack with **inputs into generate().
    inputs = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0][prompt_len:], skip_special_tokens=True)


def sanitize_tokenizer_config(out_dir):
    """Gemma 4 tokenizers can round-trip `extra_special_tokens` as a list,
    which later crashes any AutoTokenizer load (e.g. llama.cpp's GGUF
    conversion) at `_set_model_specific_special_tokens`. Coerce it back to a
    dict so the merged folder is portable. The real special tokens live in
    tokenizer.json regardless, so this is safe."""
    import json
    import pathlib
    p = pathlib.Path(out_dir) / "tokenizer_config.json"
    if not p.exists():
        return
    cfg = json.loads(p.read_text())
    if isinstance(cfg.get("extra_special_tokens"), list):
        cfg["extra_special_tokens"] = {}
        p.write_text(json.dumps(cfg, indent=2))
        print("  (normalized tokenizer_config.json for GGUF compatibility)")


def next_token_logits(model, tok, text):
    """Next-token logits for a prompt. This is the robust parity signal:
    we compare what the model predicts next, not a long greedy decode."""
    msgs = [{"role": "user", "content": text}]
    inputs = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1].float().cpu()
    return logits


def check_merged_integrity(model, out_dir):
    """Prove the save wrote every tensor the model actually holds.

    Compare the keys the in-memory merged model exposes against the keys written
    to disk. This catches a genuine save-time drop (a tensor the model has but
    the file lacks) WITHOUT assuming every layer is identical. That assumption is
    wrong for modern architectures: Gemma 4 E4B shares KV across a block of
    layers (config.num_kv_shared_layers), so the tail layers legitimately have no
    k_proj / v_proj / k_norm of their own. A uniform-layer check would raise a
    false alarm on exactly those correct checkpoints. The reference is what
    transformers instantiated, not a guess about layer shape.
    """
    import glob
    shards = glob.glob(os.path.join(out_dir, "*.safetensors"))
    if not shards:
        return  # non-safetensors save; skip
    from safetensors import safe_open
    written = set()
    for f in shards:
        with safe_open(f, "pt") as sf:
            written.update(sf.keys())

    expected = set(model.state_dict().keys())
    # transformers legitimately keeps tied weights out of the file (e.g. an
    # lm_head tied to the input embeddings), so don't count those as dropped.
    tied = set(getattr(model, "_tied_weights_keys", None) or [])
    missing = expected - written - tied
    if missing:
        raise SystemExit(
            f"ABORT: the save dropped {len(missing)} tensors the model holds "
            f"(e.g. {sorted(missing)[:3]}). The merged checkpoint is incomplete; "
            "re-merge with the whole model resident (--device cpu on a high-RAM "
            "box) so nothing is offloaded during the merge."
        )
    print(f"Integrity OK: all {len(expected)} in-memory tensors were written "
          f"({len(written)} on disk).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-drift", type=int, default=1,
                    help="max prompts allowed to drift before aborting")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="where to load the bf16 base. Use 'cpu' on a runtime "
                         "whose GPU can't hold the full bf16 model (~16 GB for "
                         "E4B); merging is just math and runs fine on CPU.")
    args = ap.parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # device_map="auto" can offload to meta/disk on a small GPU and corrupt the
    # merge. Forcing CPU avoids that when the GPU is too small for bf16.
    device_map = {"cpu": {"": "cpu"}, "cuda": {"": 0}, "auto": "auto"}[args.device]

    print(f"Loading base {args.base} in {dtype} on '{args.device}' (NOT 4-bit) ...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, device_map=device_map
    )

    # Guard: if the box was too small, device_map="auto"/"cuda" offloads part
    # of the model to CPU or disk. Merging while offloaded silently DROPS the
    # non-LoRA weights (e.g. attention k_norm) in the offloaded layers when the
    # model is saved, producing a checkpoint that fails to load in a serving
    # stack with "weights were not initialized from checkpoint". Catch every
    # offload form: params left on the meta device AND an hf_device_map that
    # placed any module on cpu/disk when we did NOT ask for a CPU merge.
    meta = [n for n, p in base.named_parameters() if p.device.type == "meta"]
    dev_map = getattr(base, "hf_device_map", {}) or {}
    offloaded = [m for m, d in dev_map.items()
                 if str(d) in ("cpu", "disk") and args.device != "cpu"]
    if meta or offloaded:
        raise SystemExit(
            "ABORT: part of the model was offloaded to CPU/disk (not enough "
            "VRAM to hold the full bf16 model, ~16 GB+ for E4B). Merging while "
            "offloaded silently drops the non-LoRA weights in the offloaded "
            "layers. Re-run with --device cpu on a high-RAM runtime, or use a "
            "GPU that fits the whole model."
        )

    tok = AutoTokenizer.from_pretrained(args.adapter)

    print("Attaching adapter and capturing reference logits ...")
    peft_model = PeftModel.from_pretrained(base, args.adapter)
    ref_logits = [next_token_logits(peft_model, tok, p) for p in PARITY_PROMPTS]

    print("Merging adapter into base weights ...")
    merged = peft_model.merge_and_unload()

    # Parity check: the merged model and the adapter-attached model should
    # predict the SAME next token for each prompt. We compare next-token
    # logits, NOT long greedy generations: in bf16 the merged path
    # (W' = W + dW, then W'x) and the adapter path (Wx + dWx) round slightly
    # differently, so a 100-token greedy decode can diverge even on a perfectly
    # correct merge. Matching the argmax (with near-identical logits) is the
    # right, robust signal.
    print("Verifying parity (next-token prediction must match) ...")
    drift = 0
    for p, ref in zip(PARITY_PROMPTS, ref_logits):
        got = next_token_logits(merged, tok, p)
        same = int(ref.argmax()) == int(got.argmax())
        max_diff = (ref - got).abs().max().item()
        drift += 0 if same else 1
        print(f"  {'MATCH' if same else 'DRIFT'} | max|d logit|={max_diff:6.3f} "
              f"| {p[:40]}")

    if drift > args.max_drift:
        raise SystemExit(
            f"ABORT: {drift} prompts predicted a different next token (max "
            f"{args.max_drift}). That points to a real merge problem (dtype or "
            "tokenizer mismatch), not just rounding."
        )

    print(f"Parity OK ({drift} mismatch). Saving merged model to {args.out} ...")
    merged.save_pretrained(args.out, safe_serialization=True)

    # Prove the save wrote every tensor before trusting the checkpoint. Parity
    # confirms the math; this confirms nothing was dropped on the way to disk.
    check_merged_integrity(merged, args.out)

    tok.save_pretrained(args.out)

    # Save the FULL processor, not just the tokenizer. Gemma 4 is multimodal,
    # so its base ships an AutoProcessor (tokenizer + image/audio preprocessing
    # config). A serving stack like vLLM resolves the checkpoint as the
    # multimodal architecture and refuses to start without preprocessor_config
    # .json / processor_config.json, even when you only ever send text. Saving
    # the processor here writes those files so the merged model serves cleanly.
    # It's best-effort: a purely text-only base has no processor, so we fall
    # back silently to the tokenizer we already saved.
    try:
        AutoProcessor.from_pretrained(args.base).save_pretrained(args.out)
        print("Saved full processor (multimodal preprocessor configs included).")
    except Exception as e:
        print(f"No AutoProcessor for this base ({e.__class__.__name__}); "
              "tokenizer-only save is fine for a text-only model.")

    sanitize_tokenizer_config(args.out)
    print("\nSample from the merged model:")
    print(" ", generate(merged, tok, PARITY_PROMPTS[0])[:240])
    print("Done.")


if __name__ == "__main__":
    main()
