"""
chapter13/code/00_download_dog_example.py

Download the canonical DreamBooth reference dataset, `diffusers/dog-example`,
from the Hugging Face Hub and stage it into a folder this chapter's training
scripts can read.

The dataset is 5 JPEG photos of one specific corgi (the same dog used in
the original DreamBooth paper). It's the standard "hello world" subject
for diffusion fine-tuning tutorials, which means you can compare your
training results against every other recipe on the internet that uses it.

After running this script, you'll have a `./dog_subject/` folder containing
the 5 reference photos and a ready-to-use `metadata.jsonl`. You can then
hand that folder to 03_train_dreambooth_lora.py via --subject_dir.

Run:
    python 00_download_dog_example.py --out_dir ./dog_subject
"""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

# The dataset repo. This is a public, MIT-style licensed tutorial dataset.
REPO_ID = "diffusers/dog-example"
INSTANCE_PROMPT = "a photo of sks dog"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./dog_subject",
                    help="where to stage the dataset")
    ap.add_argument("--instance_prompt", default=INSTANCE_PROMPT,
                    help="caption to attach to every subject photo")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Download the whole repo into the HF cache, then copy just the images out.
    print(f"downloading {REPO_ID} from the Hugging Face Hub...")
    cached = snapshot_download(repo_id=REPO_ID, repo_type="dataset")
    cached_path = Path(cached)

    # The dataset ships JPEGs at the root (and possibly under data/).
    images = sorted(cached_path.glob("*.jpeg")) + sorted(cached_path.glob("*.jpg"))
    if not images:
        images = sorted(cached_path.rglob("*.jpeg")) + sorted(cached_path.rglob("*.jpg"))
    if not images:
        raise SystemExit(f"no images found inside {cached_path}")

    rows = []
    for i, src in enumerate(images):
        dst = out / f"{i:02d}.jpg"
        shutil.copyfile(src, dst)
        rows.append({"file_name": dst.name, "text": args.instance_prompt})

    with (out / "metadata.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nstaged {len(rows)} images in {out.resolve()}")
    print(f"instance prompt: {args.instance_prompt!r}")
    print("\nnext step:")
    print(f"  python 03_train_dreambooth_lora.py \\")
    print(f"      --subject_dir {out} \\")
    print(f"      --class_dir   ./class_dog \\")
    print(f"      --instance_prompt {args.instance_prompt!r} \\")
    print(f"      --class_prompt  'a photo of a dog' \\")
    print(f"      --out_dir     ./dog_lora")


if __name__ == "__main__":
    main()
