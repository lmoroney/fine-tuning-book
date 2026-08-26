"""
chapter13/code/03b_train_textual_inversion.py

Textual Inversion is the lightest-weight diffusion fine-tuning recipe.
You don't change any model weights at all. You only learn the embedding
for a single new token that the text encoder can then use.

The result is a ~3 KB file containing one 768-dimensional vector (SD 1.5)
or two vectors (SDXL, which has two text encoders). It's perfectly
composable with everything else, never damages the base model, and can
be shared and stacked freely.

The trade-off is fidelity: because you only get a single embedding to
work with, complex subjects (faces, products with fine detail) won't
look as close to the originals as a DreamBooth LoRA would.

Run:
    python 03b_train_textual_inversion.py \
        --subject_dir ./subject \
        --placeholder "<my-dog>" \
        --initializer "dog" \
        --out_dir ./my_dog_ti \
        --steps 3000
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from safetensors.torch import save_file

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

MODEL_ID = "runwayml/stable-diffusion-v1-5"
IMG_SIZE = 512

TEMPLATES = [
    "a photo of {}",
    "a rendering of {}",
    "a cropped photo of {}",
    "a close-up photo of {}",
    "a bright photo of {}",
    "a photo of one {}",
    "a clean photo of {}",
]


class TextualInversionDataset(Dataset):
    def __init__(self, folder, placeholder, tokenizer):
        self.paths = sorted([p for p in folder.glob("*")
                             if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        self.placeholder = placeholder
        self.tokenizer = tokenizer
        self.tfm = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.paths) * 100  # epochs-via-repetition

    def __getitem__(self, idx):
        path = self.paths[idx % len(self.paths)]
        pixel = self.tfm(Image.open(path).convert("RGB"))
        template = TEMPLATES[idx % len(TEMPLATES)]
        caption = template.format(self.placeholder)
        ids = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        return pixel, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject_dir", required=True)
    ap.add_argument("--placeholder", required=True,
                    help="new token to learn, e.g. '<my-dog>'")
    ap.add_argument("--initializer", required=True,
                    help="existing token to initialize from, e.g. 'dog'")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=5e-4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder", torch_dtype=torch.float32).to(device)
    vae = AutoencoderKL.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=dtype).to(device).eval()
    unet = UNet2DConditionModel.from_pretrained(
        MODEL_ID, subfolder="unet", torch_dtype=dtype).to(device).eval()
    noise_sched = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    # Add the new placeholder token. This grows the tokenizer's vocabulary
    # by 1 and adds one new row to the text encoder's embedding matrix.
    num_added = tokenizer.add_tokens(args.placeholder)
    if num_added != 1:
        raise SystemExit(f"placeholder {args.placeholder} already exists in vocab")

    placeholder_id = tokenizer.convert_tokens_to_ids(args.placeholder)
    init_id = tokenizer.encode(args.initializer, add_special_tokens=False)
    if len(init_id) != 1:
        raise SystemExit(
            f"initializer must be a single token; "
            f"'{args.initializer}' is {len(init_id)} tokens")
    init_id = init_id[0]

    text_encoder.resize_token_embeddings(len(tokenizer))
    embeds = text_encoder.get_input_embeddings()
    with torch.no_grad():
        embeds.weight[placeholder_id] = embeds.weight[init_id].clone()

    # Freeze everything except the input embedding matrix.
    for p in unet.parameters():
        p.requires_grad_(False)
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in text_encoder.parameters():
        p.requires_grad_(False)
    embeds.weight.requires_grad_(True)

    optimizer = torch.optim.AdamW([embeds.weight], lr=args.lr)

    ds = TextualInversionDataset(Path(args.subject_dir), args.placeholder, tokenizer)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=2)

    step = 0
    for pixel, input_ids in loader:
        pixel = pixel.to(device, dtype=dtype)
        input_ids = input_ids.to(device)

        with torch.no_grad():
            latents = vae.encode(pixel).latent_dist.sample() * vae.config.scaling_factor

        # The forward pass through the text encoder MUST include the embedding
        # layer in the computation graph, so we do NOT wrap text_encoder in no_grad.
        encoder_hidden_states = text_encoder(input_ids)[0]

        t = torch.randint(0, noise_sched.config.num_train_timesteps,
                          (1,), device=device).long()
        noise = torch.randn_like(latents)
        noisy = noise_sched.add_noise(latents, noise, t)

        pred = unet(noisy, t, encoder_hidden_states.to(dtype)).sample
        loss = F.mse_loss(pred.float(), noise.float())

        loss.backward()

        # CRUCIAL: zero out gradients for every OTHER token in the embedding
        # matrix. We only want to update the one row we added.
        with torch.no_grad():
            grad = embeds.weight.grad
            mask = torch.zeros_like(grad)
            mask[placeholder_id] = 1
            grad.mul_(mask)

        optimizer.step()
        optimizer.zero_grad()

        if step % 100 == 0:
            print(f"step {step:>5} | loss {loss.item():.4f}")
        step += 1
        if step >= args.steps:
            break

    # Save just the one learned vector.
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    learned = {args.placeholder: embeds.weight[placeholder_id].detach().cpu()}
    save_file(learned, out / "learned_embeds.safetensors")
    print(f"\nSaved learned embedding to {out / 'learned_embeds.safetensors'}")
    print(f"File size: {(out / 'learned_embeds.safetensors').stat().st_size} bytes")
    print(f"Load at inference with pipe.load_textual_inversion(out_dir)")


if __name__ == "__main__":
    main()
