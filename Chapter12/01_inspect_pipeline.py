"""
chapter12/code/01_inspect_pipeline.py

Inspect a Stable Diffusion pipeline at three levels of detail:

  1) The four components of the pipeline (text encoder, VAE, UNet,
     scheduler) and their parameter counts.
  2) The UNet's down/mid/up block structure with channel counts, so
     you can verify the layer-by-layer claims in section 12.5.1.
  3) A live trace of tensor shapes flowing through the UNet, so you
     can see the U-shape (64x64 -> 32x32 -> 16x16 -> 8x8 -> 16x16
     -> 32x32 -> 64x64) actually happening.

Setup:
    # Always upgrade diffusers and transformers together. Older diffusers
    # versions reference symbols (like FLAX_WEIGHTS_NAME) that newer
    # transformers releases removed, which produces import errors.
    pip install --upgrade \\
        "diffusers>=0.30" "transformers>=4.44" "accelerate>=0.30" \\
        "safetensors>=0.4" "huggingface_hub>=0.24" torch

Run:
    python 01_inspect_pipeline.py
"""

import torch
from diffusers import StableDiffusionPipeline

MODEL_ID = "runwayml/stable-diffusion-v1-5"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)


def count_params(module):
    return sum(p.numel() for p in module.parameters())


# ----------------------------------------------------------------------
# 1) Pipeline components
# ----------------------------------------------------------------------
print(f"\nLoaded pipeline: {MODEL_ID}")
print(f"Device: {device}, dtype: {dtype}\n")
print(f"{'Component':<18}{'Class':<32}{'Params':>14}")
print("-" * 64)
print(f"{'text_encoder':<18}{type(pipe.text_encoder).__name__:<32}"
      f"{count_params(pipe.text_encoder)/1e6:>12.1f} M")
print(f"{'tokenizer':<18}{type(pipe.tokenizer).__name__:<32}"
      f"{'-':>14}")
print(f"{'vae':<18}{type(pipe.vae).__name__:<32}"
      f"{count_params(pipe.vae)/1e6:>12.1f} M")
print(f"{'unet':<18}{type(pipe.unet).__name__:<32}"
      f"{count_params(pipe.unet)/1e6:>12.1f} M")
print(f"{'scheduler':<18}{type(pipe.scheduler).__name__:<32}"
      f"{'(config only)':>14}")

# ----------------------------------------------------------------------
# 2) UNet structure: down blocks, mid block, up blocks
# ----------------------------------------------------------------------
unet = pipe.unet
print("\n\n=== UNet structure ===")
print(f"latent in:  {unet.conv_in.in_channels} channels")
print(f"latent out: {unet.conv_out.out_channels} channels")
print(f"block_out_channels: {list(unet.config.block_out_channels)}")


def has_attention(block):
    return getattr(block, "attentions", None) is not None and len(block.attentions) > 0


def fmt_block(idx, block, kind):
    name = type(block).__name__
    in_ch = block.resnets[0].in_channels
    out_ch = block.resnets[-1].out_channels
    attn = "yes" if has_attention(block) else "no "
    resampler = "downsamplers" if kind == "down" else "upsamplers"
    has_resampler = getattr(block, resampler, None) is not None
    flag = "↓" if (kind == "down" and has_resampler) else (
        "↑" if (kind == "up" and has_resampler) else "·")
    return (f"  [{idx}] {flag} {name:<26} "
            f"in={in_ch:>4}  out={out_ch:>4}  attn={attn}")


print(f"\nDown blocks ({len(unet.down_blocks)}):")
for i, block in enumerate(unet.down_blocks):
    print(fmt_block(i, block, "down"))

print(f"\nMid block:")
mid = unet.mid_block
print(f"  {'·':<3} {type(mid).__name__:<26} "
      f"in={mid.resnets[0].in_channels:>4}  "
      f"out={mid.resnets[-1].out_channels:>4}")

print(f"\nUp blocks ({len(unet.up_blocks)}):")
for i, block in enumerate(unet.up_blocks):
    print(fmt_block(i, block, "up"))

# ----------------------------------------------------------------------
# 3) Trace tensor shapes through the UNet with forward hooks
# ----------------------------------------------------------------------
print("\n\n=== Tensor shape trace through the UNet ===")

trace = []


def make_hook(label):
    def hook(_module, _inputs, outputs):
        # Block outputs can be a tensor or a tuple of (hidden_states, skips, ...)
        primary = outputs[0] if isinstance(outputs, tuple) else outputs
        trace.append((label, tuple(primary.shape)))
    return hook


handles = []
for i, block in enumerate(unet.down_blocks):
    handles.append(block.register_forward_hook(make_hook(f"down[{i}]")))
handles.append(unet.mid_block.register_forward_hook(make_hook("mid")))
for i, block in enumerate(unet.up_blocks):
    handles.append(block.register_forward_hook(make_hook(f"up[{i}]")))

# Fake inputs at SD 1.5's native latent size and text embedding shape.
fake_latent = torch.zeros(1, 4, 64, 64, device=device, dtype=dtype)
fake_t = torch.tensor([500], device=device, dtype=torch.long)
fake_text = torch.zeros(1, 77, 768, device=device, dtype=dtype)

with torch.no_grad():
    _ = unet(fake_latent, fake_t, fake_text)

for h in handles:
    h.remove()

print(f"{'block':<10}{'output shape':<30}")
print("-" * 40)
for name, shape in trace:
    print(f"{name:<10}{str(shape):<30}")

print("\nThe spatial dims should walk 64 -> 32 -> 16 -> 8 -> 16 -> 32 -> 64.")
print("Channel counts should follow the block_out_channels you printed above.")
