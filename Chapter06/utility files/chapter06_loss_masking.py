"""
Chapter 6: Preparing Your Dataset for Fine Tuning
=================================================
Section: Loss Masking

Without masking, the loss is computed over EVERY token — system prompt,
instruction, and role markers included — so part of your training budget
is spent teaching the model to reproduce questions. Loss masking sets
the labels for non-response tokens to -100 (PyTorch's 'ignore this token
in the loss' convention) so the gradient comes only from the answer.

NOTE ON TRL VERSIONS: older TRL provided DataCollatorForCompletionOnlyLM
for this; it was removed in recent versions. The modern approach is
built into SFTTrainer: structure the data as PROMPT-COMPLETION pairs and
set completion_only_loss=True in SFTConfig. Everything in the prompt is
masked; only completion tokens produce gradient.

This script converts the Alpaca data to prompt-completion form, shows
the trainer configuration, then reproduces the masking locally so you
can inspect it: every token prints with TRAIN or MASK status. Only the
assistant's response tokens should say TRAIN.
"""

from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "openai/gpt-oss-20b"

SYSTEM_PROMPT = """You are a board-certified physician providing educational \
medical information. Your answers are accurate, practical, and always recommend \
consulting a healthcare provider for personalized advice."""

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ── Convert Alpaca records to conversational prompt-completion ───────
# The 'prompt' holds everything the model reads as context (system +
# user turns); the 'completion' holds only the assistant's answer.
# With completion_only_loss=True, SFTTrainer masks the whole prompt.
def to_prompt_completion(example: dict) -> dict:
    user_content = example["instruction"]
    if example.get("input", "").strip():
        user_content = f"{example['instruction']}\n\n{example['input']}"
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "completion": [
            {"role": "assistant", "content": example["output"]},
        ],
    }


# Pull the book's hosted dataset from the Hugging Face Hub
raw = load_dataset("lmoroney/synthdata", split="train")
pc_dataset = raw.map(to_prompt_completion, remove_columns=raw.column_names)

# ── The training setup (see chapter06_trl_training.py for the rest) ──
#
# from trl import SFTTrainer, SFTConfig
#
# config = SFTConfig(
#     output_dir="checkpoints/medical-qa-v1",
#     completion_only_loss=True,   # mask the prompt; train on the answer
#     max_length=2048,
#     ...
# )
# trainer = SFTTrainer(
#     model=model,
#     args=config,
#     train_dataset=pc_dataset,
#     processing_class=tokenizer,
# )

# ── Verify the masking by reproducing it for one example ─────────────
# This mirrors what the trainer does internally: tokenize the rendered
# prompt, tokenize prompt+completion, and label the prompt tokens -100.
if __name__ == "__main__":
    example = pc_dataset[0]

    prompt_text = tokenizer.apply_chat_template(
        example["prompt"], tokenize=False, add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        example["prompt"] + example["completion"],
        tokenize=False, add_generation_prompt=False,
    )

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

    # Print the token / label pairs
    for token_id, label in zip(full_ids, labels):
        token = tokenizer.decode([token_id])
        status = "TRAIN" if label != -100 else "MASK "
        print(f"{status} | {repr(token)}")

    # If you see TRAIN on system prompt or instruction tokens, the
    # prompt/completion boundary is wrong — check that your prompt list
    # ends with the user turn and the completion holds only the answer.
