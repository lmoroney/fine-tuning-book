"""
chapter12/code/04_attach_lora.py

Attach a LoRA to Stable Diffusion's UNet using PEFT — the same library
you used in Chapters 7–8 for the language model. The target modules are
the attention projection layers (to_q, to_k, to_v, to_out.0), which is
where text conditioning enters the network.

This prints the parameter counts so you can see exactly how few weights
you'll actually be training compared to the full model.

Setup:
    pip install --upgrade \\
        "diffusers>=0.30" "transformers>=4.44" "accelerate>=0.30" \\
        "peft>=0.11" "safetensors>=0.4" "huggingface_hub>=0.24" torch

Run:
    python 04_attach_lora.py
"""

import torch
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model

MODEL_ID = "runwayml/stable-diffusion-v1-5"


def count_trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,  # train in fp32 for stable gradients
)
unet = pipe.unet

total_before = sum(p.numel() for p in unet.parameters())
print(f"UNet total params:    {total_before/1e6:>10.2f} M")

# Freeze everything first; LoRA will toggle its own layers back on.
for p in unet.parameters():
    p.requires_grad_(False)

# LoRA on the attention projections — both self-attn and cross-attn
# layers share these names, so a single target_modules list covers both.
lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.0,
    bias="none",
)

unet_lora = get_peft_model(unet, lora_config)

trainable_after = count_trainable(unet_lora)
print(f"LoRA trainable params:{trainable_after/1e6:>10.2f} M "
      f"({100*trainable_after/total_before:.2f}% of UNet)")

# Sanity check: see one of the freshly attached LoRA layers
for name, module in unet_lora.named_modules():
    if "lora_A" in name:
        print("Example LoRA layer:", name)
        break
