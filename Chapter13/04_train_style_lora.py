"""
chapter13/code/04_train_style_lora.py

Train a style LoRA on top of Stable Diffusion 1.5. This script is the
DreamBooth loop from Chapter 12 with the prior-preservation pieces
removed and the dataset broadened from a single subject to many subjects
sharing a single visual style.

Inputs:
    --data_dir : a folder containing PNG/JPG files plus a metadata.jsonl
                 where each row is {"file_name": "...", "text": "..."}.
                 (Produced by 01b_generate_style_dataset.py, or by
                 01_curate_dataset.py + 02_caption_with_blip.py for your
                 own real-world style set.)

Outputs:
    --out_dir : the trained LoRA adapter, saved as a single
                pytorch_lora_weights.safetensors file in diffusers-native
                key format.

Defaults are tuned for a ~100-image dataset on a single 16 GB GPU.

Logging:
    Diffusion training loss is intrinsically noisy because every step
    samples a random timestep t. We log three things every 25 steps:
      - the raw per-step loss (for completeness)
      - a 50-step moving-average loss (less noisy)
    And every --eval_every steps:
      - val_loss on a fixed set of (image, t, noise) tuples (the clean
        signal you should actually plot)

Run:
    python 04_train_style_lora.py \
        --data_dir ./style_set \
        --out_dir ./mybrand_lora \
        --steps 1500 \
        --lr 1e-4 \
        --rank 16
"""

import argparse
import json
import math
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file

MODEL_ID = "runwayml/stable-diffusion-v1-5"
IMG_SIZE = 512


class StyleDataset(Dataset):
    """One image, one caption per item. Caption already includes the trigger."""

    def __init__(self, folder: Path, tokenizer):
        meta = (folder / "metadata.jsonl")
        if not meta.exists():
            raise FileNotFoundError(meta)
        self.rows = [json.loads(line) for line in meta.open()]
        self.folder = folder
        self.tokenizer = tokenizer
        self.tfm = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(self.folder / row["file_name"]).convert("RGB")
        pixel = self.tfm(img)
        ids = self.tokenizer(
            row["text"],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        return pixel, ids


def build_validation_set(loader, n, vae, text_encoder, scheduler,
                          device, dtype, seed=12345):
    """Pre-cache N (latent, text-embedding, timestep, noise) tuples for clean
    periodic evaluation. The variance from random t and random noise sampling
    is the dominant noise floor in diffusion training loss; fixing those two
    things across evaluations turns the eval loss into a clean signal that
    can actually be plotted."""
    g_cpu = torch.Generator(device="cpu").manual_seed(seed)
    items = []
    it = iter(loader)
    for _ in range(n):
        try:
            pixel, ids = next(it)
        except StopIteration:
            it = iter(loader)
            pixel, ids = next(it)
        pixel = pixel.to(device, dtype=dtype)
        ids = ids.to(device)
        with torch.no_grad():
            # .mode() instead of .sample() removes the VAE's stochasticity.
            latent = vae.encode(pixel).latent_dist.mode() * vae.config.scaling_factor
            txt = text_encoder(ids)[0]
        t = torch.randint(0, scheduler.config.num_train_timesteps,
                          (1,), generator=g_cpu).long().to(device)
        noise = torch.randn(latent.shape, generator=g_cpu,
                            dtype=torch.float32).to(device)
        items.append({"latent": latent.detach(),
                      "txt": txt.detach(),
                      "t": t,
                      "noise": noise.detach()})
    return items


@torch.no_grad()
def evaluate(val_items, unet, scheduler):
    """Mean MSE over the fixed validation set. Deterministic for a given UNet."""
    unet.eval()
    losses = []
    for item in val_items:
        noisy = scheduler.add_noise(item["latent"].float(),
                                    item["noise"], item["t"])
        pred = unet(noisy, item["t"], item["txt"].float()).sample
        losses.append(F.mse_loss(pred.float(), item["noise"]).item())
    unet.train()
    return sum(losses) / len(losses)


def save_lora_for_diffusers(unet, save_dir):
    """Save the PEFT-wrapped UNet's LoRA in diffusers-native format.

    Bypasses both `unet.save_pretrained()` (which writes PEFT-format files
    that confuse pipe.load_lora_weights) and the diffusers
    `convert_state_dict_to_diffusers` helper (which behaves differently
    across versions). Instead, we explicitly rewrite the keys ourselves:
    strip PEFT's `base_model.model.` prefix, strip the `.default.` adapter
    name segment, and add diffusers' `unet.` prefix.

    Also clears any stale adapter files in `save_dir` first, so the
    directory ends up with a single canonical adapter file. Otherwise an
    older training run's `adapter_config.json` can shadow the new save
    and confuse the loader.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for stale in [
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "pytorch_lora_weights.safetensors",
        "pytorch_lora_weights.bin",
    ]:
        (save_dir / stale).unlink(missing_ok=True)

    raw = get_peft_model_state_dict(unet)

    cleaned = {}
    for k, v in raw.items():
        new_k = k
        if new_k.startswith("base_model.model."):
            new_k = new_k[len("base_model.model."):]
        new_k = new_k.replace(".lora_A.default.weight", ".lora_A.weight")
        new_k = new_k.replace(".lora_B.default.weight", ".lora_B.weight")
        cleaned[f"unet.{new_k}"] = v

    bad = [k for k in cleaned if "base_model" in k or ".default." in k]
    if bad:
        raise RuntimeError(
            f"Key cleanup failed; {len(bad)} keys still PEFT-shaped. "
            f"Example: {bad[0]}")

    save_file(cleaned, str(save_dir / "pytorch_lora_weights.safetensors"))

    samples = list(cleaned.keys())[:3]
    print(f"  wrote {len(cleaned)} tensors to "
          f"{save_dir / 'pytorch_lora_weights.safetensors'}")
    for s in samples:
        print(f"    key: {s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_window", type=int, default=50,
                    help="length of the moving-average window for loss logs")
    ap.add_argument("--checkpoint_every", type=int, default=300,
                    help="save an intermediate adapter every N steps; 0 disables")
    ap.add_argument("--val_size", type=int, default=32,
                    help="number of fixed (image,t,noise) tuples for validation")
    ap.add_argument("--eval_every", type=int, default=100,
                    help="run validation every N steps; 0 disables")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder", torch_dtype=weight_dtype).to(device).eval()
    vae = AutoencoderKL.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=weight_dtype).to(device).eval()
    unet = UNet2DConditionModel.from_pretrained(
        MODEL_ID, subfolder="unet", torch_dtype=torch.float32).to(device)
    noise_sched = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    for m in (text_encoder, vae):
        for p in m.parameters():
            p.requires_grad_(False)
    for p in unet.parameters():
        p.requires_grad_(False)

    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )
    unet = get_peft_model(unet, lora_cfg)
    unet.train()

    trainable = [p for p in unet.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"trainable params: {n_train/1e6:.2f}M "
          f"({100*n_train/sum(p.numel() for p in unet.parameters()):.2f}% of UNet)")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    ds = StyleDataset(Path(args.data_dir), tokenizer)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, drop_last=True)

    # Build a fixed validation set so we can report a low-variance loss curve.
    val_items = None
    if args.eval_every > 0 and args.val_size > 0:
        val_loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
        val_items = build_validation_set(
            val_loader, args.val_size, vae, text_encoder, noise_sched,
            device, weight_dtype, seed=args.seed + 1)
        print(f"built validation set of {len(val_items)} fixed (image,t,noise) tuples")

    epochs = math.ceil(args.steps * args.grad_accum / max(1, len(loader)))
    print(f"requested {args.steps} optimizer steps; "
          f"samples/epoch={len(loader)}; running up to {epochs} epoch(s)")

    # Moving-average windows. We accumulate per-MICRO-step losses, not
    # per-optimizer-step, so the window covers a lot more random-t samples.
    micro_win = deque(maxlen=args.log_window * args.grad_accum)

    step = 0
    accum = 0
    for epoch in range(epochs):
        for pixel, input_ids in loader:
            pixel = pixel.to(device, dtype=weight_dtype)
            input_ids = input_ids.to(device)

            with torch.no_grad():
                latents = vae.encode(pixel).latent_dist.sample() * vae.config.scaling_factor
                encoder_hidden_states = text_encoder(input_ids)[0]

            bsz = latents.shape[0]
            t = torch.randint(0, noise_sched.config.num_train_timesteps,
                              (bsz,), device=device).long()
            noise = torch.randn_like(latents)
            noisy = noise_sched.add_noise(latents, noise, t)

            pred = unet(noisy.float(), t, encoder_hidden_states.float()).sample
            loss = F.mse_loss(pred.float(), noise.float())
            (loss / args.grad_accum).backward()
            micro_win.append(loss.item())
            accum += 1

            if accum % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                step += 1

                if step % 25 == 0:
                    last_loss = micro_win[-1]
                    avg = sum(micro_win) / len(micro_win)
                    n_micro = len(micro_win)
                    # The moving average covers up to log_window optimizer
                    # steps worth of micro-steps. Report both numbers.
                    print(f"step {step:>5} | last {last_loss:.4f} "
                          f"(avg over {n_micro} micro-steps: {avg:.4f})")

                # Periodic fixed-validation-set evaluation.
                if (val_items is not None and args.eval_every > 0
                        and step % args.eval_every == 0):
                    val_loss = evaluate(val_items, unet, noise_sched)
                    print(f"  -> val_loss @ step {step}: {val_loss:.4f}")

                # Intermediate checkpoints.
                if (args.checkpoint_every > 0 and step > 0
                        and step % args.checkpoint_every == 0
                        and step < args.steps):
                    ckpt = Path(args.out_dir) / f"checkpoint-{step}"
                    save_lora_for_diffusers(unet, ckpt)
                    print(f"  -> saved checkpoint to {ckpt}")

                if step >= args.steps:
                    break
        if step >= args.steps:
            break

    out = Path(args.out_dir)
    save_lora_for_diffusers(unet, out)
    print(f"\nSaved final LoRA adapter to {out.resolve()} after {step} steps")
    print("Load it with pipe.load_lora_weights(out_dir)")


if __name__ == "__main__":
    main()
