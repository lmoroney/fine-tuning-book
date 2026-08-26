"""
chapter13/code/02_caption_with_blip.py

Auto-caption every image in a curated style dataset using BLIP-2, then
append a style trigger phrase. The trigger is what the LoRA will bind
the style vocabulary to. Pick something the base model doesn't already
recognize: 'in mybrand style', 'awxstyle illustration', etc.

Notes:
- Describe the SUBJECT, not the style. The model already has a strong
  prior for words like 'watercolor' or 'oil painting'. If your captions
  contain those phrases the LoRA learns 'subject + watercolor' instead
  of 'subject + your trigger'.
- The trigger goes at the end so the rest of the caption stays general.

Run:
    python 02_caption_with_blip.py \
        --dir ./style_set \
        --trigger "in mybrand style"
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

MODEL_ID = "Salesforce/blip2-opt-2.7b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="curated dataset folder")
    ap.add_argument("--trigger", required=True, help="style trigger phrase")
    ap.add_argument("--prompt_template",
                    default="Question: describe the subject of this photo in one short phrase. Answer:",
                    help="BLIP-2 prompt; we only want the subject, not the style")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    proc = Blip2Processor.from_pretrained(MODEL_ID)
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype).to(device)
    model.eval()

    folder = Path(args.dir)
    meta_path = folder / "metadata.jsonl"
    if not meta_path.exists():
        raise SystemExit(f"missing {meta_path}; run the curation script first")

    rows = [json.loads(line) for line in meta_path.open()]

    for row in rows:
        img = Image.open(folder / row["file_name"]).convert("RGB")
        inputs = proc(images=img, text=args.prompt_template,
                      return_tensors="pt").to(device, dtype=dtype)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30)
        subject = proc.tokenizer.decode(out[0], skip_special_tokens=True)
        subject = subject.replace(args.prompt_template, "").strip().rstrip(".").lower()

        # Compose the final caption.
        row["text"] = f"{subject}, {args.trigger}"

    with meta_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    # Print a few samples so you can sanity-check the captions.
    print("Sample captions:")
    for row in rows[:5]:
        print(f"  {row['file_name']}: {row['text']}")


if __name__ == "__main__":
    main()
