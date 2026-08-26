"""
04_kv_cache_quant.py
--------------------
Two things:
  1. size_kv_cache(): compute the KV-cache footprint from first principles
     so you can budget memory for long context.
  2. A transformers generation that stores the cache in quantized form,
     letting you fit much longer context in the same memory.

See also the llama.cpp equivalent (runtime flags), shown at the bottom.
"""
import os
# Skip transformers' optional TensorFlow import (broken TF/protobuf in some
# environments crashes it); we don't use TF here.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def size_kv_cache(n_layers, n_kv_heads, head_dim, seq_len,
                  batch=1, bytes_per_value=2):
    """Bytes held by the KV cache. The leading 2 is keys + values.
    Note n_kv_heads (not n_attention_heads): GQA shrinks this directly."""
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per_value


def cache_budget_report():
    # Example: a 4B-class model with GQA.
    cfg = dict(n_layers=34, n_kv_heads=8, head_dim=256)
    print(f"{'ctx tokens':>12s} {'16-bit':>10s} {'8-bit':>10s}")
    for seq in (512, 4000, 8000, 16000):
        b16 = size_kv_cache(**cfg, seq_len=seq, bytes_per_value=2) / 1e9
        b8 = size_kv_cache(**cfg, seq_len=seq, bytes_per_value=1) / 1e9
        print(f"{seq:12d} {b16:8.2f}GB {b8:8.2f}GB")


def generate_quantized_cache(model_dir, prompt, nbits=4):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="auto")
    # Gemma 4's tokenizer returns a BatchEncoding (dict); use return_dict=True
    # and unpack with **inputs into generate().
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    out = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        cache_implementation="quantized",       # store KV in quantized form
        cache_config={"backend": "quanto", "nbits": nbits},
    )
    return tok.decode(out[0][prompt_len:], skip_special_tokens=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="HF model dir for a live demo")
    args = ap.parse_args()

    print("=== KV-cache budget (GQA, 4B-class model) ===")
    cache_budget_report()

    if args.model:
        print("\n=== generation with a 4-bit quantized cache ===")
        print(generate_quantized_cache(
            args.model, "Summarize the key risk factors for stroke."))

# llama.cpp equivalent (8-bit keys and values at serve time):
#   ./llama-cli -m medical-gemma-Q4_K_M.gguf \
#       --cache-type-k q8_0 --cache-type-v q8_0 -c 8192 -p "..."
