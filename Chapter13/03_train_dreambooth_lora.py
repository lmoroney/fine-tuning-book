"""
chapter13/code/03_train_dreambooth_lora.py

Train a DreamBooth LoRA on Stable Diffusion 1.5. This script teaches the
model a SINGLE subject (your dog, your product, your face). It is the
sibling of 04_train_style_lora.py, which teaches a visual style.

Two ideas drive DreamBooth:
1. Rare-token binding. Use a token the model has almost never seen
   (e.g. "sks") so that fine-tuning can bind your subject's visual
   features to that token without fighting an existing prior.
2. Prior preservation. Generate ~200 images of the broader class
   (e.g. "a photo of a dog") with the base model and train on them
   alongside your subject images. This prevents the model from
   collapsing the general concept onto your specific subject.

Inputs:
    --subject_dir       folder of 3-8 subject photos (any size). The easiest
                        way to populate this is to run 00_download_dog_example.py,
                        which stages the canonical diffusers/dog-example
                        dataset (5 photos of one corgi) for you.
    --class_dir         folder of ~200 auto-generated class images. If empty,
                        this script will populate it on first run.
    --instance_prompt   caption for the subject images,
                        e.g. "a photo of sks dog"
    --class_prompt      caption for the class-prior images,
                        e.g. "a photo of a dog"

Outputs:
    --out_dir           the trained LoRA adapter, saved as safetensors.

Run (with the dog-example dataset):
    python 00_download_dog_example.py --out_dir ./dog_subject
    python 03_train_dreambooth_lora.py \
        --subject_dir ./dog_subject \
        --class_dir   ./class_dog \
        --instance_prompt "a photo of sks dog" \
        --class_prompt "a photo of a dog" \
        --out_dir     ./dog_lora \
        --steps 1200
"""

import argparse
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
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file

MODEL_ID = "runwayml/stable-diffusion-v1-5"
IMG_SIZE = 512


class CaptionedFolderDataset(Dataset):
    """Read every image in a folder and pair each with a fixed caption."""

    def __init__(self, folder: Path, prompt: str, tokenizer, repeat: int = 1):
        self.paths = sorted([p for p in folder.glob("*")
                             if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if not self.paths:
            raise ValueError(f"no images in {folder}")
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.repeat = repeat
        self.tfm = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.paths) * self.repeat

    def __getitem__(self, idx):
        path = self.paths[idx % len(self.paths)]
        pixel = self.tfm(Image.open(path).convert("RGB"))
        ids = self.tokenizer(
            self.prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        return pixel, ids


def maybe_generate_class_images(class_dir: Path, class_prompt: str,
                                target_count: int, device: str):
    """If the class folder is empty, generate prior images using the base model.
    Skip if the folder already has enough images."""
    class_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in class_dir.glob("*.png"))
    if len(existing) >= target_count:
        print(f"class folder already has {len(existing)} images; skipping generation")
        return
    print(f"generating {target_count - len(existing)} class images for "
          f"prior preservation ({class_prompt!r})")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, safety_checker=None).to(device)
    pipe.set_progress_bar_config(disable=True)
    for i in range(len(existing), target_count):
        img = pipe(class_prompt, num_inference_steps=20).images[0]
        img.save(class_dir / f"class_{i:04d}.png")
    del pipe
    torch.cuda.empty_cache()


def build_validation_set(loader, n, vae, text_encoder, scheduler,
                          device, dtype, seed=12345):
    """Pre-cache N (latent, text-embedding, timestep, noise) tuples for clean
    periodic evaluation. The variance from random t and random noise sampling
    is the dominant noise floor in diffusion training loss; fixing those two
    things across evaluations turns the eval loss into a clean monotonic-ish
    signal that can actually be plotted."""
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


def save_lora_for_diffusers(unet, save_dir):
    """Save the PEFT-wrapped UNet's LoRA in diffusers-native format.

    This avoids two failure modes:

    1) `unet.save_pretrained()` writes PEFT-format adapters whose key names
       and adapter_config.json paths confuse `pipe.load_lora_weights()`.
    2) `diffusers.utils.convert_state_dict_to_diffusers` is supposed to strip
       PEFT's `base_model.model.` prefix and the `.default.` adapter-name
       segment, but its behavior varies across diffusers versions. On some
       installs it leaves the prefix in place, which makes inference fail
       with a "Target modules ... not found in the base model" error.

    To sidestep both, this helper rewrites the keys explicitly and writes the
    safetensors file directly. After saving, `save_dir` contains exactly one
    file (`pytorch_lora_weights.safetensors`) whose key naming convention is
    the one `pipe.load_lora_weights()` expects: each key starts with `unet.`,
    contains no `base_model.model.`, and has no `.default.` adapter-name
    segment.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Remove any pre-existing adapter files so there's only one canonical
    # source of weights left when we're done. This covers both the PEFT-native
    # format (`adapter_*`) and the diffusers-native format
    # (`pytorch_lora_weights.*`), in case the directory was used by a prior run.
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
        # 1) Strip PEFT's wrapper prefix.
        if new_k.startswith("base_model.model."):
            new_k = new_k[len("base_model.model."):]
        # 2) Strip the adapter-name segment if PEFT inserted it.
        new_k = new_k.replace(".lora_A.default.weight", ".lora_A.weight")
        new_k = new_k.replace(".lora_B.default.weight", ".lora_B.weight")
        # 3) Add the diffusers `unet.` prefix.
        cleaned[f"unet.{new_k}"] = v

    # Defensive sanity check: anything that still looks PEFT-shaped is a bug.
    bad = [k for k in cleaned if "base_model" in k or ".default." in k]
    if bad:
        raise RuntimeError(
            f"Key cleanup failed; {len(bad)} keys still PEFT-shaped. "
            f"Example: {bad[0]}")

    save_file(cleaned, str(save_dir / "pytorch_lora_weights.safetensors"))

    # Print a few sample keys so you can verify the format from the run log.
    samples = list(cleaned.keys())[:3]
    print(f"  wrote {len(cleaned)} tensors to "
          f"{save_dir / 'pytorch_lora_weights.safetensors'}")
    for s in samples:
        print(f"    key: {s}")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject_dir", required=True)
    ap.add_argument("--class_dir", required=True)
    ap.add_argument("--instance_prompt", required=True)
    ap.add_argument("--class_prompt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--prior_weight", type=float, default=1.0)
    ap.add_argument("--class_count", type=int, default=200)
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
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Auto-generate class-prior images if needed.
    maybe_generate_class_images(
        Path(args.class_dir), args.class_prompt, args.class_count, device)

    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder", torch_dtype=dtype).to(device).eval()
    vae = AutoencoderKL.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=dtype).to(device).eval()
    unet = UNet2DConditionModel.from_pretrained(
        MODEL_ID, subfolder="unet", torch_dtype=torch.float32).to(device)
    noise_sched = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    for m in (text_encoder, vae):
        for p in m.parameters():
            p.requires_grad_(False)
    for p in unet.parameters():
        p.requires_grad_(False)

    # Subject training is a narrow concept; a small rank is enough.
    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )
    unet = get_peft_model(unet, lora_cfg)
    unet.train()

    trainable = [p for p in unet.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable)/1e6:.2f}M")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    # repeat=1 here; the outer epoch loop below handles re-iteration.
    subj_ds = CaptionedFolderDataset(
        Path(args.subject_dir), args.instance_prompt, tokenizer, repeat=1)
    prior_ds = CaptionedFolderDataset(
        Path(args.class_dir), args.class_prompt, tokenizer)

    subj_loader = DataLoader(subj_ds, batch_size=1, shuffle=True, num_workers=2)
    prior_loader = DataLoader(prior_ds, batch_size=1, shuffle=True, num_workers=2)
    prior_iter = iter(prior_loader)

    # Build a fixed validation set so we can report a low-variance loss curve
    # that's actually plottable. The dominant noise in per-step loss comes
    # from t-sampling and noise-sampling; freezing both removes it.
    val_items = None
    if args.eval_every > 0 and args.val_size > 0:
        # Use a small sequential loader so we always get the same images first.
        val_loader = DataLoader(subj_ds, batch_size=1, shuffle=False, num_workers=0)
        val_items = build_validation_set(
            val_loader, args.val_size, vae, text_encoder, noise_sched,
            device, dtype, seed=args.seed + 1)
        print(f"built validation set of {len(val_items)} fixed (image,t,noise) tuples")

    # An "epoch" here is one full pass over the (already-repeated) subject set.
    # We wrap the loader in an outer loop so training can keep going past one
    # epoch worth of samples. Without this, the for-loop would exit silently
    # the moment subj_loader exhausts, regardless of --steps.
    samples_per_epoch = max(1, len(subj_loader))
    epochs = math.ceil(args.steps / samples_per_epoch)
    print(f"requested {args.steps} steps; subj samples/epoch={samples_per_epoch}; "
          f"running {epochs} epoch(s)")

    # Rolling windows so we can log meaningful loss averages instead of the
    # wildly noisy per-step values. Diffusion training samples a random
    # timestep per step; per-step loss bounces around accordingly.
    subj_win = deque(maxlen=args.log_window)
    prior_win = deque(maxlen=args.log_window)

    step = 0
    for epoch in range(epochs):
        for pixel_s, ids_s in subj_loader:
            try:
                pixel_p, ids_p = next(prior_iter)
            except StopIteration:
                prior_iter = iter(prior_loader)
                pixel_p, ids_p = next(prior_iter)

            pixel = torch.cat([pixel_s, pixel_p]).to(device, dtype=dtype)
            ids = torch.cat([ids_s, ids_p]).to(device)

            with torch.no_grad():
                latents = vae.encode(pixel).latent_dist.sample() * vae.config.scaling_factor
                txt = text_encoder(ids)[0]

            bsz = latents.shape[0]
            t = torch.randint(0, noise_sched.config.num_train_timesteps,
                              (bsz,), device=device).long()
            noise = torch.randn_like(latents)
            noisy = noise_sched.add_noise(latents, noise, t)

            pred = unet(noisy.float(), t, txt.float()).sample

            # First half of the batch is the subject; second half is the class prior.
            pred_subj, pred_prior = pred.chunk(2, dim=0)
            noise_subj, noise_prior = noise.chunk(2, dim=0)

            loss_subj = F.mse_loss(pred_subj.float(), noise_subj.float())
            loss_prior = F.mse_loss(pred_prior.float(), noise_prior.float())
            loss = loss_subj + args.prior_weight * loss_prior

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            subj_win.append(loss_subj.item())
            prior_win.append(loss_prior.item())

            if step % 25 == 0:
                # Moving-average loss is much more informative than per-step.
                subj_avg = sum(subj_win) / len(subj_win)
                prior_avg = sum(prior_win) / len(prior_win)
                print(f"step {step:>4} | subj {loss_subj.item():.4f} "
                      f"(avg{len(subj_win):>3} {subj_avg:.4f}) | "
                      f"prior {loss_prior.item():.4f} "
                      f"(avg{len(prior_win):>3} {prior_avg:.4f})")

            step += 1

            # Periodic fixed-validation-set evaluation. This is the clean
            # signal you should actually plot; per-step training loss is
            # too noisy to be useful, even with a 50-step moving average.
            if (val_items is not None and args.eval_every > 0
                    and step > 0 and step % args.eval_every == 0):
                val_loss = evaluate(val_items, unet, noise_sched)
                print(f"  -> val_loss @ step {step}: {val_loss:.4f}")

            # Periodic checkpoints so you can recover an earlier adapter
            # if the run starts overfitting before --steps is reached.
            if (args.checkpoint_every > 0 and step > 0
                    and step % args.checkpoint_every == 0 and step < args.steps):
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


if __name__ == "__main__":
    main()
