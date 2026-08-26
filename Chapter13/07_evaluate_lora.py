"""
chapter13/code/07_evaluate_lora.py

Three quantitative checks for a style LoRA:

1. Prompt fidelity. Does the generated image still match what the prompt
   describes? Measured as CLIP cosine similarity between the generated
   image embedding and the prompt's text embedding.

2. Style similarity. Does the generated image carry the trained style?
   Measured as CLIP image-image cosine similarity between the generated
   image and the *style reference set* (the held-out training images).
   We average over the reference set so noisy individual references
   don't dominate.

3. Subject preservation. Does the trained model still know how to draw
   ordinary subjects? Measured by running both the base and trained
   models with the SAME prompt + seed, then taking the CLIP cosine.
   Wildly different outputs (cosine < 0.5) suggest the LoRA collapsed
   the base model's general knowledge.

All three together give you a single CSV row per evaluation prompt that
you can graph over training checkpoints.

Run:
    python 06_evaluate_style_lora.py \
        --lora ./mybrand_lora \
        --style_refs ./style_set_holdout \
        --prompts prompts.txt \
        --out scores.csv
"""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPProcessor, CLIPModel

MODEL_ID = "runwayml/stable-diffusion-v1-5"
CLIP_ID = "openai/clip-vit-base-patch32"


def cos(a, b):
    return F.cosine_similarity(a, b).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--style_refs", required=True,
                    help="folder of held-out images that exemplify the style")
    ap.add_argument("--prompts", required=True,
                    help="text file with one evaluation prompt per line")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", default="scores.csv")
    args = ap.parse_args()

    device = "cuda"

    # CLIP for all three measurements.
    clip = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
    cprc = CLIPProcessor.from_pretrained(CLIP_ID)

    # Reference style embedding = mean of all held-out reference images.
    refs = sorted(Path(args.style_refs).glob("*.png")) + \
           sorted(Path(args.style_refs).glob("*.jpg"))
    ref_imgs = [Image.open(p).convert("RGB") for p in refs]
    ref_inputs = cprc(images=ref_imgs, return_tensors="pt").to(device)
    with torch.no_grad():
        ref_feats = clip.get_image_features(**ref_inputs)
        ref_feats = F.normalize(ref_feats, dim=-1)
        style_anchor = ref_feats.mean(dim=0, keepdim=True)
        style_anchor = F.normalize(style_anchor, dim=-1)

    # Base pipeline (no LoRA).
    base_pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, safety_checker=None)
    base_pipe.scheduler = DPMSolverMultistepScheduler.from_config(base_pipe.scheduler.config)
    base_pipe = base_pipe.to(device)

    # Trained pipeline (load the LoRA into a separate pipe so we can compare).
    lora_pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, safety_checker=None)
    lora_pipe.scheduler = DPMSolverMultistepScheduler.from_config(lora_pipe.scheduler.config)
    lora_pipe.load_lora_weights(args.lora)
    lora_pipe = lora_pipe.to(device)

    prompts = [line.strip() for line in open(args.prompts) if line.strip()]

    rows = []
    for p in prompts:
        gen_a = torch.Generator(device=device).manual_seed(args.seed)
        base_img = base_pipe(p, num_inference_steps=args.steps,
                             generator=gen_a).images[0]
        gen_b = torch.Generator(device=device).manual_seed(args.seed)
        lora_img = lora_pipe(
            p, num_inference_steps=args.steps, generator=gen_b,
            cross_attention_kwargs={"scale": args.scale}).images[0]

        with torch.no_grad():
            inputs = cprc(text=[p], images=[base_img, lora_img],
                          return_tensors="pt", padding=True).to(device)
            img_feats = F.normalize(clip.get_image_features(
                pixel_values=inputs["pixel_values"]), dim=-1)
            txt_feats = F.normalize(clip.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]), dim=-1)

        base_feat, lora_feat = img_feats[0:1], img_feats[1:2]
        fidelity = cos(lora_feat, txt_feats)
        style    = cos(lora_feat, style_anchor)
        preserve = cos(lora_feat, base_feat)

        rows.append({
            "prompt": p,
            "prompt_fidelity": round(fidelity, 4),
            "style_match":     round(style, 4),
            "subject_drift":   round(1 - preserve, 4),
        })

        print(f"{p[:60]:60s}  fid={fidelity:.3f}  style={style:.3f}  "
              f"drift={(1-preserve):.3f}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
