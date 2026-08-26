"""
chapter12/code/02_forward_noise.py

Visualise the forward diffusion process. Given an image, sample noisy
versions of it at several timesteps and save them as a horizontal strip.
This is the q(x_t | x_0) closed-form sampling formula in action.

x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

Run:
    python 02_forward_noise.py path/to/photo.jpg
"""

import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from diffusers import DDPMScheduler

IMG_SIZE = 256
TIMESTEPS_TO_SHOW = [0, 100, 250, 500, 750, 999]


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),                  # [0,1]
        transforms.Normalize([0.5]*3, [0.5]*3), # [-1,1]
    ])
    return tfm(img)  # (3, H, W)


def to_pil(x: torch.Tensor) -> Image.Image:
    x = (x.clamp(-1, 1) + 1) / 2.0
    arr = (x * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def main(path: str) -> None:
    x0 = load_image(path)
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",   # the SD-default
        beta_start=0.00085,
        beta_end=0.012,
    )

    frames = []
    for t in TIMESTEPS_TO_SHOW:
        # Sample one noise tensor per timestep so each step shows
        # a different realization, not a cumulative blur.
        eps = torch.randn_like(x0)
        timesteps = torch.tensor([t])
        xt = scheduler.add_noise(x0.unsqueeze(0), eps.unsqueeze(0), timesteps)[0]
        frames.append(to_pil(xt))

    strip = Image.new("RGB", (IMG_SIZE * len(frames), IMG_SIZE), "white")
    for i, f in enumerate(frames):
        strip.paste(f, (i * IMG_SIZE, 0))

    out = Path("forward_noise_strip.png")
    strip.save(out)
    print(f"Saved {out} — timesteps {TIMESTEPS_TO_SHOW}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 02_forward_noise.py <image_path>")
        sys.exit(1)
    main(sys.argv[1])
