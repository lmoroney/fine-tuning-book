"""
Chapter 6: Preparing Your Dataset for Fine Tuning
=================================================
Section: Using OpenAI's Templates

Converts the Alpaca-format synthetic dataset from Chapter 5 into the
OpenAI fine-tuning 'messages' format — one JSON object per line, each
with a messages list of role/content dictionaries. You never write the
special tokens (<|im_start|> etc.) yourself; the API applies the template.

The data is pulled from the book's hosted dataset on the Hugging Face
Hub (https://huggingface.co/datasets/lmoroney/synthdata), so you can
run this without having generated your own data in Chapter 5. If you
DID generate your own, point DATASET_NAME at your local JSONL instead:
    load_dataset("json", data_files="medical_qa_synthetic.jsonl", split="train")

Run this BEFORE chapter06_trl_training.py: it writes the three files
that script (and the chapter's 85/10/5 split guidance) expects:

    data/formatted/medical_qa_openai.jsonl   (85% — training)
    data/formatted/medical_qa_val.jsonl      (10% — validation)
    data/formatted/medical_qa_test.jsonl     ( 5% — held-out test)

After converting, validate the output with the script in OpenAI's
cookbook: the Chat_finetuning_data_prep.ipynb notebook at
https://github.com/openai/openai-cookbook — the most common error it
catches is an assistant message that's empty after stripping whitespace.
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

DATASET_NAME = "lmoroney/synthdata"

SYSTEM_PROMPT = """You are a board-certified physician providing educational \
medical information. Your answers are accurate, practical, and always recommend \
consulting a healthcare provider for personalized advice."""


def alpaca_to_messages(example: dict) -> dict | None:
    """Convert one Alpaca record to the messages format, or None to skip."""
    # Skip examples with an empty output — these slip through when
    # the teacher model returned a blank response (see Chapter 5)
    if not example.get("output", "").strip():
        return None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]

    # Include input field if present and non-empty
    if example.get("input", "").strip():
        messages[1]["content"] = (
            f"{example['instruction']}\n\n{example['input']}"
        )
    return {"messages": messages}


def alpaca_to_openai(train_path: Path, val_path: Path,
                     test_path: Path) -> None:
    """Convert the hosted Alpaca-format dataset to OpenAI fine-tuning
    JSONL, split 85/10/5 into train/validation/test files."""
    data = load_dataset(DATASET_NAME, split="train")
    converted, skipped = [], 0
    for example in data:
        record = alpaca_to_messages(example)
        if record is None:
            skipped += 1
        else:
            converted.append(record)

    # Shuffle with a fixed seed so the split is reproducible, then
    # cut 85% / 10% / 5% (see 'Splitting and Validating your Dataset')
    random.Random(42).shuffle(converted)
    n = len(converted)
    train_end = int(n * 0.85)
    val_end = int(n * 0.95)
    splits = [
        (train_path, converted[:train_end]),
        (val_path, converted[train_end:val_end]),
        (test_path, converted[val_end:]),
    ]

    for path, records in splits:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as outfile:
            for record in records:
                outfile.write(json.dumps(record) + "\n")
        print(f"Wrote {len(records)} examples to {path}")

    if skipped:
        print(f"Skipped {skipped} examples with empty outputs — "
              f"check your Chapter 5 filtering step!")


if __name__ == "__main__":
    alpaca_to_openai(
        Path("data/formatted/medical_qa_openai.jsonl"),
        Path("data/formatted/medical_qa_val.jsonl"),
        Path("data/formatted/medical_qa_test.jsonl"),
    )
