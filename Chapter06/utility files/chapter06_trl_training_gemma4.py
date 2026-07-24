"""
Chapter 6: Preparing Your Dataset for Fine Tuning
=================================================
Sections: Formatting for TRL / Using Alpaca with SFTTrainer

A complete TRL training run on the medical Q&A data. Two paths to the
'messages' format SFTTrainer expects:

  1. Load a pre-converted OpenAI-format JSONL (see
     chapter06_alpaca_to_openai.py), or
  2. Convert the raw Alpaca-format JSONL on the fly with
     alpaca_to_messages() below.

SFTTrainer applies the model's native chat template automatically,
using whatever tokenizer you pass — you never hand-format the special
tokens.

This version uses Gemma 4 E4B ("effective 4B": 8B total parameters,
~4.5B effective via per-layer embeddings). Its chat template accepts
the system role in our messages data natively — no folding into the
user turn needed, unlike Gemma 2/3. A full fine-tune of an 8B model
still wants a data-center GPU; for consumer hardware, add LoRA (see
Chapter 7) or use the Unsloth variant of this script.
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "google/gemma-4-E4B-it"

SYSTEM_PROMPT = """You are a board-certified physician providing educational \
medical information. Your answers are accurate, practical, and always recommend \
consulting a healthcare provider for personalized advice."""


def alpaca_to_messages(example: dict) -> dict:
    """Convert a single Alpaca record to the messages format SFTTrainer expects."""
    user_content = example["instruction"]
    if example.get("input", "").strip():
        user_content = f"{example['instruction']}\n\n{example['input']}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["output"]},
        ]
    }


# ── Load model and tokenizer ─────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# ── Path 1: load a dataset already in messages format ────────────────
dataset = load_dataset("json", data_files={
    "train": "data/formatted/medical_qa_openai.jsonl",
    "validation": "data/formatted/medical_qa_val.jsonl",
})

# ── Path 2 (alternative): convert the hosted Alpaca data on the fly ──
# Uncomment to pull the book's dataset straight from the Hugging Face
# Hub (https://huggingface.co/datasets/lmoroney/synthdata) instead.
# Note this doesn't split into train/validation — see
# chapter06_split_and_validate.py.
#
# raw = load_dataset("lmoroney/synthdata", split="train")
# converted = raw.map(alpaca_to_messages, remove_columns=raw.column_names)
# split = converted.train_test_split(test_size=0.1, seed=42)
# dataset = {"train": split["train"], "validation": split["test"]}

# ── Configure and run the trainer ────────────────────────────────────
config = SFTConfig(
    output_dir="checkpoints/medical-qa-gemma4-v1",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_ratio=0.05,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    bf16=True,               # Use bfloat16 if your GPU supports it
    max_length=2048,
)

trainer = SFTTrainer(
    model=model,
    args=config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
)

if __name__ == "__main__":
    trainer.train()
