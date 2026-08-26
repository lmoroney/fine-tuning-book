"""
chapter13/code/06_compose_multiple_loras.py

Load two or more LoRA adapters into a single pipeline, give each one a
weight, and generate with all of them active. This is how you ship a
single inference call that carries a brand style AND a specific subject
AND a particular lighting recipe.

The diffusers helper does the bookkeeping for you:
    pipe.load_lora_weights(path, adapter_name=...)
    pipe.set_adapters(names, adapter_weights=weights)

Internally, each LoRA layer accumulates its delta:
    y = W·x + Σ scaleᵢ · Bᵢ·Aᵢ·x

Run:
    python 05_compose_multiple_loras.py \
        --style ./mybrand_lora \
        --subject ./acme_mascot_lora \
        --light ./studio_lighting_lora \
        --prompt "a portrait of sks mascot, studio lighting, in mybrand style"
"""

import argparse
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

MODEL_ID = "runwayml/stable-diffusion-v1-5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--light", default=None)
    ap.add_argument("--style_w", type=float, default=0.8)
    ap.add_argument("--subject_w", type=float, default=1.0)
    ap.add_argument("--light_w", type=float, default=0.4)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative_prompt",
                    default="blurry, low quality, watermark, text, extra limbs")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="composed.png")
    args = ap.parse_args()

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Each load_lora_weights call attaches that adapter under the given name.
    pipe.load_lora_weights(args.style, adapter_name="style")
    pipe.load_lora_weights(args.subject, adapter_name="subject")
    names = ["style", "subject"]
    weights = [args.style_w, args.subject_w]

    if args.light:
        pipe.load_lora_weights(args.light, adapter_name="light")
        names.append("light")
        weights.append(args.light_w)

    pipe.set_adapters(names, adapter_weights=weights)
    pipe = pipe.to("cuda")

    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=gen,
    ).images[0]
    image.save(args.out)
    print(f"wrote {args.out}")
    print(f"active adapters: {names} with weights {weights}")


if __name__ == "__main__":
    main()
