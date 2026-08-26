"""
00_finetune_gemma4.py
----------------------
Fine-tune Gemma 4 (E4B) on the medical Q&A dataset with QLoRA, producing the
LoRA adapter that the rest of this chapter merges and quantizes.

Dataset: https://huggingface.co/datasets/lmoroney/synthdata
    Alpaca-style columns: instruction, input, output (+ metadata, format).
    One "train" split, ~5,321 examples pooled from three teacher models.

Why this script exists in Chapter 14: a LoRA trained on an OLD base (e.g.
google/gemma-3-4b-it) cannot be merged into a Gemma 4 model. If you're
migrating to Gemma 4, retrain the adapter here first, then run
01_merge_adapter.py.

The Gemma 4 gotcha, handled correctly below: Gemma 4 wraps its linear layers
in Gemma4ClippableLinear, which PEFT can't attach to directly. The fix is to
OMIT target_modules so peft>=0.19.0 applies its Gemma-4-aware defaults (the
language-model linears, skipping the clippable wrappers). Do NOT pass an
explicit target_modules list.

Requirements (CUDA GPU, ~16 GB VRAM is enough for E4B in 4-bit):
    pip install -U "peft>=0.19.0" "trl>=0.12" transformers datasets \
        accelerate bitsandbytes

Usage:
    python 00_finetune_gemma4.py                 # vanilla base
    python 00_finetune_gemma4.py --qat           # start from QAT checkpoint (§14.7)
    python 00_finetune_gemma4.py --out ./medical-qa-v1-gemma4-lora --epochs 1
"""
import os
# Gemma 4 is multimodal, so transformers reaches into its image/video utils,
# which try to `import tensorflow`. A broken TF/protobuf combo crashes the
# import; we don't use TF, so skip it. (Set BEFORE importing transformers.)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

VANILLA_BASE = "google/gemma-4-e4b-it"
# Starting from the unquantized QAT checkpoint makes the eventual 4-bit
# quant nearly lossless (see section 14.7).
QAT_BASE = "google/gemma-4-e4b-it-qat-q4_0-unquantized"


def build_dataset(tokenizer, dataset_id, val_frac=0.05, seed=42):
    """Load the Alpaca-style medical set and render each row with the Gemma
    chat template into a single 'text' field for supervised fine-tuning."""
    ds = load_dataset(dataset_id, split="train")

    def to_chat(ex):
        # Fold the optional 'input' field into the user turn.
        user = ex["instruction"]
        if ex.get("input"):
            user = f"{user}\n\n{ex['input']}"
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": ex["output"]},
        ]
        # Gemma 4 ships a "thinking" reasoner mode. Our targets are direct
        # answers with no thought block, so train (and later serve) with
        # thinking OFF so the template matches the data. Otherwise the model
        # is prompted to think but never shown a thought block, and at
        # inference it emits a reasoning preamble before answering.
        return {"text": tokenizer.apply_chat_template(
            messages, tokenize=False, enable_thinking=False)}

    ds = ds.map(to_chat, remove_columns=ds.column_names)
    # Small validation slice for loss monitoring. Keep your real evaluation
    # TEST set (Chapter 9 / section 14.8) separate and disjoint from this.
    return ds.train_test_split(test_size=val_frac, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None, help="override the base model id")
    ap.add_argument("--qat", action="store_true",
                    help="start from the unquantized QAT checkpoint (§14.7)")
    ap.add_argument("--dataset", default="lmoroney/synthdata")
    ap.add_argument("--out", default="./medical-qa-v1-gemma4-lora")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    args = ap.parse_args()

    base_id = args.base or (QAT_BASE if args.qat else VANILLA_BASE)
    print(f"Base model: {base_id}")

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    splits = build_dataset(tokenizer, args.dataset)
    print(f"train={len(splits['train'])}  val={len(splits['test'])}")

    # QLoRA: load the base in 4-bit NF4 to fit a single GPU. (Training in
    # 4-bit is fine; you'll RELOAD the base in bf16 for merging in script 01.)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_id, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # LoRA config — NOTE: no target_modules. peft>=0.19.0 fills in the
    # Gemma-4-aware defaults, which is what avoids the Gemma4ClippableLinear
    # "is not supported" error.
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training config. A couple of arg names drift across library versions:
    #   - transformers <4.41 used `evaluation_strategy` (now `eval_strategy`)
    #   - SFTConfig: newer TRL uses `max_length`; older TRL used `max_seq_length`
    # If you hit a TypeError on those, rename to match your installed version.
    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="epoch",
        max_length=args.max_seq_len,   # older TRL: max_seq_length=...
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
        peft_config=lora,            # SFTTrainer applies LoRA + k-bit prep
        processing_class=tokenizer,  # older TRL: use tokenizer=tokenizer
    )

    trainer.train()

    # Save ONLY the adapter (small). 01_merge_adapter.py folds it into a
    # fresh bf16 base.
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"\nSaved LoRA adapter to {args.out}")
    print("Next: python 01_merge_adapter.py --base", base_id,
          "--adapter", args.out, "--out ./medical-gemma-merged")


if __name__ == "__main__":
    main()
