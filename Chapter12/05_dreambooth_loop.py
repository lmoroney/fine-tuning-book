"""
chapter12/code/05_dreambooth_loop.py

A minimal, readable DreamBooth-style training loop with LoRA. This is
the *core* of the training — not a production-grade script. It exists
so you can read every line and see exactly what's happening.

Inputs you supply:
- A folder of 3–8 photos of your subject (e.g. ./subject/*.jpg)
- An instance prompt that includes a rare token, e.g. "a photo of sks dog"
- An optional folder of class-prior images and a class prompt, e.g. "a photo of a dog"

Setup:
    pip install --upgrade \
        "diffusers>=0.30" "transformers>=4.44" "accelerate>=0.30" \
        "peft>=0.11" "safetensors>=0.4" "huggingface_hub>=0.24" torch torchvision

Run:
    python 05_dreambooth_loop.py \
        --subject_dir ./subject \
        --instance_prompt "a photo of sks dog" \
        --output_dir ./lora_out
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model

MODEL_ID = "runwayml/stable-diffusion-v1-5"
IMG_SIZE = 512


class SubjectDataset(Dataset):
    def __init__(self, image_dir: Path, prompt: str, tokenizer):
        self.paths = sorted([p for p in image_dir.glob("*")
                             if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if not self.paths:
            raise ValueError(f"No images in {image_dir}")
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.tfm = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.paths) * 100  # epochs-via-repetition

    def __getitem__(self, idx):
        path = self.paths[idx % len(self.paths)]
        pixel = self.tfm(Image.open(path).convert("RGB"))
        tok = self.tokenizer(
            self.prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        return pixel, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject_dir", required=True)
    ap.add_argument("--instance_prompt", required=True)
    ap.add_argument("--output_dir", default="./lora_out")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Load components individually so we can attach LoRA to the UNet only.
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder", torch_dtype=dtype).to(device).eval()
    vae = AutoencoderKL.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=dtype).to(device).eval()
    unet = UNet2DConditionModel.from_pretrained(
        MODEL_ID, subfolder="unet", torch_dtype=torch.float32).to(device)
    noise_sched = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    # Freeze text encoder + VAE.
    for m in (text_encoder, vae):
        for p in m.parameters():
            p.requires_grad_(False)

    # LoRA on the UNet's attention projections.
    for p in unet.parameters():
        p.requires_grad_(False)
    lora_cfg = LoraConfig(r=8, lora_alpha=8,
                          target_modules=["to_q", "to_k", "to_v", "to_out.0"])
    unet = get_peft_model(unet, lora_cfg)
    unet.train()

    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad], lr=args.lr)

    ds = SubjectDataset(Path(args.subject_dir), args.instance_prompt, tokenizer)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=2)

    step = 0
    for pixel, input_ids in loader:
        pixel = pixel.to(device, dtype=dtype)
        input_ids = input_ids.to(device)

        with torch.no_grad():
            # Encode pixels into the VAE latent and scale by SD's convention.
            latents = vae.encode(pixel).latent_dist.sample() * vae.config.scaling_factor
            encoder_hidden_states = text_encoder(input_ids.unsqueeze(0))[0]

        # Sample a random timestep and matching noise.
        bsz = latents.shape[0]
        t = torch.randint(0, noise_sched.config.num_train_timesteps,
                          (bsz,), device=device).long()
        noise = torch.randn_like(latents)
        noisy = noise_sched.add_noise(latents, noise, t)

        # Predict the noise. This is the entire learning signal.
        pred = unet(noisy.float(), t, encoder_hidden_states.float()).sample
        loss = torch.nn.functional.mse_loss(pred.float(), noise.float())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 25 == 0:
            print(f"step {step:>4} | loss {loss.item():.4f}")
        step += 1
        if step >= args.steps:
            break

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(out)
    print(f"\nSaved LoRA adapter to {out.resolve()}")


if __name__ == "__main__":
    main()
