"""
chapter13/code/01b_generate_style_dataset.py

Generate a synthetic style training set using SDXL.

We use SDXL (not SD 1.5) deliberately. The style training script in
this chapter fine-tunes an SD 1.5 LoRA. If we generated this dataset
with SD 1.5 itself, we'd be training the model on its own outputs, and
the fine-tuned LoRA would be enforcing a look the base model already
makes. By generating with a stronger, architecturally different model
(SDXL has a larger UNet, two text encoders, and was trained on a
different dataset), the resulting LoRA actually transfers a new style
into SD 1.5's output distribution.

The flow this script implements:
1. SDXL renders each subject in a defined visual style at 1024x1024.
2. We resize each image to 512x512 (the resolution our SD 1.5 LoRA
   will train at).
3. We caption each image with a SHORT trigger phrase, not the long
   style description. The long description goes into generation;
   the short trigger is what the LoRA learns to bind to the look.
4. We write metadata.jsonl, ready for 04_train_style_lora.py.

VRAM: SDXL at fp16 with attention slicing fits in roughly 9 to 10 GB.
Comfortable on a Colab T4 (16 GB).

Time: about 12 to 15 seconds per image on a T4 at 30 inference steps,
so the default 80 subjects takes roughly 16 to 20 minutes end to end.

Run:
    python 01b_generate_style_dataset.py --out_dir ./style_set
"""

import argparse
import json
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from PIL import Image

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# The descriptive style prompt SDXL receives at generation time.
# Edit this if you want to teach a different style.
DEFAULT_STYLE_SUFFIX = (
    "retrofuturist gouache illustration, soft pastel palette, "
    "clean lines, soft shading, 1960s magazine cover aesthetic, "
    "isometric perspective"
)

# The short trigger phrase the LoRA will bind to that style. Pick
# something the base SD 1.5 model doesn't already have strong opinions
# about so the gradient has something fresh to latch onto.
DEFAULT_TRIGGER = "in mybrand style"

# Varied subjects so the LoRA learns to factor style out from subject.
# About 80 prompts spanning people, animals, food, environments, objects.
SUBJECTS = [
    "a steaming cup of tea on a wooden table",
    "a vintage bicycle leaning against a brick wall",
    "a sliced grapefruit on a marble counter",
    "a young woman laughing in afternoon light",
    "an old fisherman mending a net",
    "a cat curled on a denim cushion",
    "a small bird perched on a window sill",
    "a dog sitting on a beach at sunrise",
    "a fox running through a snowy field",
    "a butterfly resting on a leaf",
    "a portrait of an elderly man with glasses",
    "a child blowing dandelion seeds into the air",
    "two friends sharing an umbrella in the rain",
    "a chef chopping vegetables in a kitchen",
    "a musician playing a saxophone on a street corner",
    "an astronaut floating in space",
    "a robot watering a plant",
    "a vintage car parked on a cobblestone street",
    "a rocket launching from a desert pad",
    "a sailing boat on a calm sea",
    "a hot air balloon over a meadow",
    "a wooden bridge over a forest stream",
    "a lighthouse at dusk",
    "a snow-capped mountain peak",
    "a desert oasis with palm trees",
    "a tropical waterfall surrounded by ferns",
    "a meadow of wildflowers at golden hour",
    "an empty road through a redwood forest",
    "a cottage in an autumn forest",
    "a city skyline at night",
    "a busy market square at noon",
    "a quiet library reading nook",
    "an empty subway station platform",
    "a cozy attic bedroom with a window seat",
    "a desk with an open notebook and a coffee mug",
    "a bowl of ramen with a soft egg",
    "a slice of cheesecake with raspberry sauce",
    "a glass of red wine beside an open book",
    "a basket of fresh bread on a kitchen counter",
    "a pile of vintage suitcases",
    "an antique typewriter on a writing desk",
    "a record player with vinyl albums",
    "a brass telescope on a balcony at night",
    "a pocket watch on a velvet cloth",
    "a stack of leather-bound books",
    "a pair of running shoes on a stone path",
    "a wooden rocking chair on a porch",
    "a knitted scarf draped over a wooden chair",
    "a vase of sunflowers on a windowsill",
    "a hummingbird hovering near a flower",
    "a school of fish swimming over a coral reef",
    "a starfish on a sandy beach",
    "a snail on a wet leaf",
    "a ladybug on a blade of grass",
    "a frog on a lily pad",
    "an owl perched on a moonlit branch",
    "a wolf howling at the moon",
    "a deer drinking from a forest stream",
    "a herd of horses running across a plain",
    "a koi pond at a temple",
    "a Japanese garden with stone lanterns",
    "a Greek temple at sunset",
    "a Moroccan courtyard with a tiled fountain",
    "an Italian villa with cypress trees",
    "a Scandinavian cabin in winter",
    "a Caribbean beach hut at dawn",
    "a vintage train station with a steam locomotive",
    "an old wooden ship at sea",
    "a biplane flying over green fields",
    "a vintage carousel at night",
    "a circus tent under a full moon",
    "a hot spring surrounded by snow",
    "a Greek statue covered in moss",
    "a clockmaker's workshop",
    "a botanical garden conservatory",
    "a planetarium projection of stars",
    "a wizard's study with floating books",
    "a fairy tale castle on a hilltop",
    "a witch's cauldron in a moonlit forest",
    "a deep-sea submarine porthole view",
    "a blacksmith forging at an anvil",
    "an ice cream cart on a sunny boardwalk",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./style_set")
    ap.add_argument("--style_suffix", default=DEFAULT_STYLE_SUFFIX,
                    help="long descriptive style prompt sent to SDXL")
    ap.add_argument("--trigger", default=DEFAULT_TRIGGER,
                    help="short trigger phrase written into metadata.jsonl")
    ap.add_argument("--gen_size", type=int, default=1024,
                    help="SDXL generation resolution")
    ap.add_argument("--out_size", type=int, default=512,
                    help="downscaled resolution for SD 1.5 training")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=0,
                    help="base seed; each image uses seed + index")
    ap.add_argument("--max_subjects", type=int, default=80,
                    help="generate only the first N subjects from the built-in list")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: SDXL is impractical on CPU; this would take many hours.")

    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"loading {MODEL_ID} ({dtype})...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
        use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    # Keep peak VRAM low enough for a T4.
    pipe.enable_attention_slicing()
    pipe.enable_vae_tiling()
    pipe.set_progress_bar_config(disable=True)

    subjects = SUBJECTS[:args.max_subjects]
    print(f"generating {len(subjects)} images at {args.gen_size}x{args.gen_size}, "
          f"saving at {args.out_size}x{args.out_size}")
    print(f"style suffix used for generation: {args.style_suffix!r}")
    print(f"trigger written into captions:   {args.trigger!r}\n")

    rows = []
    for i, subj in enumerate(subjects):
        prompt = f"{subj}, {args.style_suffix}"
        gen = torch.Generator(device=device).manual_seed(args.seed + i)
        img = pipe(
            prompt=prompt,
            negative_prompt="blurry, low quality, watermark, text, signature, "
                            "deformed, extra limbs",
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            width=args.gen_size,
            height=args.gen_size,
            generator=gen,
        ).images[0]
        # Downscale to SD 1.5's training resolution.
        if args.out_size != args.gen_size:
            img = img.resize((args.out_size, args.out_size), Image.LANCZOS)
        out_path = out / f"{i:03d}.png"
        img.save(out_path)
        # Caption uses the short trigger, not the long style suffix.
        rows.append({
            "file_name": out_path.name,
            "text": f"{subj}, {args.trigger}",
        })
        if (i + 1) % 5 == 0 or i == len(subjects) - 1:
            print(f"  {i+1}/{len(subjects)} generated")

    with (out / "metadata.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nWrote {len(rows)} style training images to {out.resolve()}")
    print(f"  Generated with SDXL at {args.gen_size}x{args.gen_size}")
    print(f"  Downscaled to {args.out_size}x{args.out_size}")
    print(f"  Style trigger in captions: {args.trigger!r}")
    print("\nBefore training, hold out ~10% of these for evaluation:")
    print(f"  mkdir -p {out}_holdout && "
          f"mv {out}/00[0-7].png {out}_holdout/")
    print("\nThen train:")
    print(f"  python 04_train_style_lora.py \\")
    print(f"      --data_dir {out} \\")
    print(f"      --out_dir  ./mybrand_lora \\")
    print(f"      --steps 1500 --rank 16 --lr 1e-4")


if __name__ == "__main__":
    main()
