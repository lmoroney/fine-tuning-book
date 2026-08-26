"""
06_use_qat_checkpoint.py
------------------------
Claim most of the benefit of Quantization-Aware Training without running
the expensive QAT pass yourself: fine-tune on top of Google's UNQUANTIZED
QAT checkpoint, then quantize the merged model as usual.

The only change from the normal flow is the base model id. Everything
downstream (LoRA training in ch. 7-10, merge in 01_merge_adapter.py,
GGUF quantize in 02_convert_to_gguf.sh) is identical.

Why this works: the QAT-unquantized weights were trained to sit cleanly on
the 4-bit grid, so your fine-tune inherits that robustness and the final
q4_0 conversion costs far less accuracy than PTQ on a vanilla base.
"""
import os
# Skip transformers' optional TensorFlow import (broken TF/protobuf in some
# environments crashes it); we don't use TF here.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import argparse

# Vanilla base (PTQ path) vs. unquantized QAT checkpoint (QAT path).
VANILLA_BASE = "google/gemma-4-e4b-it"
QAT_BASE = "google/gemma-4-e4b-it-qat-q4_0-unquantized"


def pick_base(use_qat: bool) -> str:
    """Return the base model id to fine-tune from."""
    return QAT_BASE if use_qat else VANILLA_BASE


def load_base_for_finetuning(use_qat: bool):
    """Load the chosen base in bf16, ready to attach a LoRA adapter.
    NOTE: load the *unquantized* QAT checkpoint, not a pre-quantized GGUF —
    you need full-precision weights to train and merge correctly."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_id = pick_base(use_qat)
    print(f"Fine-tuning base: {base_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(base_id)
    return model, tok

    # From here: attach your LoRA config and run SFTTrainer exactly as in
    # Chapter 8, then merge (01_merge_adapter.py) and quantize to q4_0
    # (02_convert_to_gguf.sh). No other step changes.


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qat", action="store_true",
                    help="start from the unquantized QAT checkpoint")
    args = ap.parse_args()
    print("Recommended base ->", pick_base(args.qat))
    # load_base_for_finetuning(args.qat)   # uncomment to actually load
