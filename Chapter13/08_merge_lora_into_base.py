"""
chapter13/code/08_merge_lora_into_base.py

Bake a LoRA back into the base UNet weights so you can ship a single
.safetensors file with no adapter dependency. After merging:
  - inference no longer needs PEFT installed
  - the cross_attention_kwargs scale knob is gone (frozen at 1.0)
  - the resulting model is the same size as the base

Use this at the END of iteration, once you've settled on a strength.

Run:
    python 07_merge_lora_into_base.py \
        --lora ./mybrand_lora \
        --out  ./mybrand_full
"""

import argparse
import torch
from diffusers import StableDiffusionPipeline

MODEL_ID = "runwayml/stable-diffusion-v1-5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="effective LoRA strength to bake in")
    args = ap.parse_args()

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, safety_checker=None)
    pipe.load_lora_weights(args.lora)
    pipe.fuse_lora(lora_scale=args.scale)
    pipe.unload_lora_weights()       # drop the adapter; the weights are fused now
    pipe.save_pretrained(args.out)
    print(f"merged + saved to {args.out}")


if __name__ == "__main__":
    main()
