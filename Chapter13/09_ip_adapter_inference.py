"""
chapter13/code/09_ip_adapter_inference.py

IP-Adapter is the zero-shot alternative to DreamBooth. Instead of
training per-subject, you load a small pre-trained adapter (about 50 MB)
that contains an image encoder plus a cross-attention pathway. At
inference you pass a REFERENCE image alongside your prompt, and the
adapter biases the cross-attention to pull from the reference.

There are three common flavors:
- IP-Adapter (general image conditioning)
- IP-Adapter FaceID (specialized on face identity)
- IP-Adapter Plus (higher fidelity, slower)

You don't train anything yourself. The trade-off is that you can't pick
up stylistic nuance the way a trained LoRA can; IP-Adapter is for
"render something that resembles THIS specific reference image."

This script shows the general-purpose flavor. The FaceID variant is
nearly identical; only the model name and the call signature change.

Run:
    python 09_ip_adapter_inference.py \
        --reference ./reference.png \
        --prompt "a portrait in soft afternoon light" \
        --out portrait.png
"""

import argparse
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

MODEL_ID = "runwayml/stable-diffusion-v1-5"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_FILE = "ip-adapter_sd15.bin"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True,
                    help="path to the reference image you want to condition on")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative_prompt",
                    default="blurry, low quality, watermark, text")
    ap.add_argument("--scale", type=float, default=0.7,
                    help="how strongly the reference influences output (0..1)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--out", default="ip_adapter_out.png")
    args = ap.parse_args()

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder="models", weight_name=IP_ADAPTER_FILE)
    pipe.set_ip_adapter_scale(args.scale)
    pipe = pipe.to("cuda")

    ref = Image.open(args.reference).convert("RGB")

    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        ip_adapter_image=ref,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=gen,
    ).images[0]

    image.save(args.out)
    print(f"wrote {args.out}")
    print(f"reference: {args.reference}  scale: {args.scale}")


if __name__ == "__main__":
    main()
