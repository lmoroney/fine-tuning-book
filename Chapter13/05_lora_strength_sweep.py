"""
chapter13/code/05_lora_strength_sweep.py

Generate a grid of images at multiple LoRA strengths so you can pick the
strength that looks best for production. Fix the seed, fix the prompt,
sweep the scale.

The pivotal argument is cross_attention_kwargs={"scale": s}. Set s=0 to
see the base model exactly; set s=1.0 to see what you trained; go higher
to push the style further (at the risk of artifacts).

Run:
    python 04_lora_strength_sweep.py \
        --lora ./mybrand_lora \
        --prompt "a steaming cup of tea on a wooden table, in mybrand style" \
        --out grid.png
"""

import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

MODEL_ID = "runwayml/stable-diffusion-v1-5"


def make_grid(images, cols=4):
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * w, rows * h), "white")
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        grid.paste(im, (c * w, r * h))
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative_prompt",
                    default="blurry, low quality, watermark, text")
    ap.add_argument("--scales", nargs="+", type=float,
                    default=[0.0, 0.3, 0.6, 0.8, 1.0, 1.2, 1.5])
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--out", default="grid.png")
    args = ap.parse_args()

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(args.lora)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    images = []
    for s in args.scales:
        gen = torch.Generator(device="cuda").manual_seed(args.seed)
        img = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=gen,
            cross_attention_kwargs={"scale": s},
        ).images[0]
        # annotate so the grid is self-documenting
        img.load()
        images.append(img)
        print(f"scale={s:>4.2f}  -> done")

    grid = make_grid(images, cols=min(4, len(images)))
    grid.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
