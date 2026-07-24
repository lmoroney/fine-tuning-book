"""
Chapter 6: Preparing Your Dataset for Fine Tuning
=================================================
Section: Other Frameworks — Unsloth

Speed- and memory-optimized fine-tuning of Gemma 4 E4B on the medical
Q&A data. With 4-bit loading, the 8B-parameter model works on a free
Google Colab T4 instance (16GB VRAM). Unsloth's FastLanguageModel is a
drop-in replacement for the HuggingFace loading step; the SFTTrainer
interface is the same as TRL.

Gemma is gated on the Hugging Face Hub — accept the license on the
model page and log in (hf auth login) first.
"""

from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

SYSTEM_PROMPT = """You are a board-certified physician providing educational \
medical information. Your answers are accurate, practical, and always recommend \
consulting a healthcare provider for personalized advice."""

# ── Unsloth handles the model loading with optimization enabled ──────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-4-E4B-it",
    max_seq_length=2048,
    load_in_4bit=True,       # 4-bit quantization — 8B params in ~5GB
)

# ── Apply LoRA — covered in detail in Chapter 7 ──────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                        # LoRA rank
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
)


# ── Prepare the data: Alpaca → messages, then split ──────────────────
# Gemma 4 accepts a standalone system role natively (earlier Gemma
# generations did not — they required folding the system prompt into
# the first user turn).
def alpaca_to_messages(example: dict) -> dict:
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


# Pull the book's hosted dataset from the Hugging Face Hub
raw = load_dataset("lmoroney/synthdata", split="train")
dataset = raw.map(alpaca_to_messages, remove_columns=raw.column_names)
split = dataset.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = split["train"], split["test"]


# Unsloth's SFTTrainer requires an explicit formatting_func — it does
# not auto-detect the messages format the way vanilla TRL does. We
# apply the tokenizer's template here to produce the formatted string.
# If training fails or the model doesn't learn, look HERE first: the
# output must match the model's expected format exactly.
def formatting_func(example):
    texts = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return texts if isinstance(texts, list) else [texts]


# ── The SFTTrainer call is identical to the TRL example ──────────────
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir="checkpoints/medical-qa-unsloth",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        bf16=False,
        max_length=2048,
    ),
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
    formatting_func=formatting_func,
)

# Loss masking: Unsloth provides train_on_responses_only() as a native
# alternative to TRL's collator approach. Call it after constructing
# the trainer, passing the exact strings that open the instruction and
# response turns in the chat template. These are GEMMA 4's turn tokens —
# earlier Gemma generations used <start_of_turn>user / <start_of_turn>model,
# and using those here would silently mask nothing. Verify against
# print(tokenizer.apply_chat_template(...)) whenever you change models!
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|turn>user\n",
    response_part="<|turn>model\n",
)

if __name__ == "__main__":
    trainer.train()
