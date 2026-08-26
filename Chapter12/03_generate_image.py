"""
chapter12/code/03_generate_image.py

Generate an image with Stable Diffusion 1.5 using a chosen scheduler
and a fixed seed for reproducibility. Useful for "before vs. after"
comparisons against a fine-tuned model.

Setup:
    pip install --upgrade \\
        "diffusers>=0.30" "transformers>=4.44" "accelerate>=0.30" \\
        "safetensors>=0.4" "huggingface_hub>=0.24" torch

Run:
    python 03_generate_image.py "a corgi astronaut on Mars, cinematic"
"""

import sys
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

MODEL_ID = "runwayml/stable-diffusion-v1-5"
SEED = 42
STEPS = 25
GUIDANCE = 7.5


def main(prompt: str, out_path: str = "out.png") -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    # Swap the slow default scheduler for a fast multistep solver.
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(SEED)

    image = pipe(
        prompt=prompt,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=generator,
    ).images[0]

    image.save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python 03_generate_image.py "your prompt"')
        sys.exit(1)
    main(sys.argv[1])
