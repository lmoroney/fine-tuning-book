"""
chapter12/code/06_load_adapter_and_generate.py

Apply a previously-trained LoRA adapter to a base Stable Diffusion
pipeline and generate samples with your rare-token prompt.

Setup:
    pip install --upgrade \\
        "diffusers>=0.30" "transformers>=4.44" "accelerate>=0.30" \\
        "peft>=0.11" "safetensors>=0.4" "huggingface_hub>=0.24" torch

Run:
    python 06_load_adapter_and_generate.py ./lora_out
"""

import sys
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

MODEL_ID = "runwayml/stable-diffusion-v1-5"
SEED = 7


def main(lora_path: str, prompt: str = "a photo of sks dog wearing a top hat") -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lora_path)
    pipe = pipe.to(device)

    g = torch.Generator(device=device).manual_seed(SEED)
    img = pipe(prompt, num_inference_steps=25, guidance_scale=7.5, generator=g).images[0]
    img.save("subject_generation.png")
    print("Saved subject_generation.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 06_load_adapter_and_generate.py <lora_path> [prompt]")
        sys.exit(1)
    prompt = sys.argv[2] if len(sys.argv) > 2 else "a photo of sks dog wearing a top hat"
    main(sys.argv[1], prompt)
