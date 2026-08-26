"""
chapter13/code/01_curate_dataset.py

Take a folder of raw images (either a style set with many subjects, or a
DreamBooth subject set with one subject) and produce a clean training set:
- resize to a target edge length (preserving aspect ratio, then center-crop
  to a square if you want a fixed shape)
- convert everything to RGB
- strip EXIF/metadata
- drop near-duplicates using a perceptual hash
- write a metadata.jsonl file with empty captions (caption step is separate)

For a DreamBooth subject set, pass --square because subjects benefit from
fixed-aspect crops centered on the subject.

Run:
    python 01_curate_dataset.py \
        --in_dir ./raw_images \
        --out_dir ./training_set \
        --size 512
"""

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def phash(img: Image.Image, hash_size: int = 16) -> int:
    """Tiny perceptual hash. Resize to NxN greyscale, threshold by mean,
    pack into an int. Good enough to catch obvious dupes."""
    g = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = list(g.getdata())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for i, p in enumerate(pixels):
        if p > avg:
            bits |= 1 << i
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--square", action="store_true",
                    help="center-crop to a square; otherwise keep aspect ratio")
    ap.add_argument("--dup_threshold", type=int, default=6,
                    help="hamming distance below which two images are dupes")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes: list[int] = []
    rows: list[dict] = []
    dropped_dupes = 0

    paths = sorted([p for p in in_dir.glob("*") if p.suffix.lower() in EXTS])
    if not paths:
        raise SystemExit(f"No images in {in_dir}")

    for i, p in enumerate(paths):
        try:
            img = Image.open(p)
        except Exception as e:
            print(f"skip {p.name}: cannot open ({e})")
            continue

        img = ImageOps.exif_transpose(img).convert("RGB")

        # dedupe
        h = phash(img)
        if any(hamming(h, s) < args.dup_threshold for s in seen_hashes):
            dropped_dupes += 1
            continue
        seen_hashes.append(h)

        # resize
        if args.square:
            img = ImageOps.fit(img, (args.size, args.size), Image.LANCZOS)
        else:
            img.thumbnail((args.size, args.size), Image.LANCZOS)

        # strip metadata by writing a fresh file
        out_name = f"{i:04d}.png"
        out_path = out_dir / out_name
        img.save(out_path, "PNG", optimize=True)

        rows.append({"file_name": out_name, "text": ""})

    # metadata.jsonl is the schema Hugging Face's ImageFolder loader expects.
    with (out_dir / "metadata.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"kept   {len(rows)} images")
    print(f"dropped {dropped_dupes} duplicates")
    print(f"wrote  {out_dir / 'metadata.jsonl'}")


if __name__ == "__main__":
    main()
