"""
Chapter 6: Preparing Your Dataset for Fine Tuning
=================================================
Section: Chat Templates — Using Templates with HuggingFace

Demonstrates how a model's built-in chat template turns a structured
messages list into the flat, special-token-delimited string the model
actually consumes. Uses tokenize=False so you can see the intermediate
string — the single most useful debugging trick in this chapter.
"""

from transformers import AutoTokenizer

# Load the tokenizer for the model you're fine-tuning.
# Swap in any model to see its template — e.g. "meta-llama/Llama-3.2-3B-Instruct",
# "Qwen/Qwen2.5-7B-Instruct", "google/gemma-3-1b-it".
MODEL_NAME = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

messages = [
    {"role": "system", "content": "You are a helpful medical assistant providing "
     "educational information. Always recommend consulting a healthcare provider."},
    {"role": "user", "content": "What are the warning signs of high blood pressure?"},
    {"role": "assistant", "content": "High blood pressure rarely causes noticeable "
     "symptoms, which is why it's often called the silent killer..."},
]

# tokenize=False gives you the formatted string before tokenization
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)
print("=== Formatted string (training form: full assistant turn included) ===")
print(formatted)

# add_generation_prompt=True appends the OPENING of the assistant turn with no
# closing token — the signal for the model to keep generating. This is what
# you want at INFERENCE time, never in training data (where the full
# assistant response is already present).
inference_messages = messages[:-1]  # drop the assistant answer
formatted_for_inference = tokenizer.apply_chat_template(
    inference_messages,
    tokenize=False,
    add_generation_prompt=True,
)
print("\n=== Formatted string (inference form: open assistant turn) ===")
print(formatted_for_inference)

# With tokenize=True you get the token IDs that are ready for training.
token_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
)
print(f"\nToken count after tokenization: {len(token_ids)}")

# Bonus: the template itself is just a Jinja2 string shipped with the
# tokenizer. Print it to see the source of truth for this model's format.
print("\n=== The model's raw Jinja2 chat template ===")
print(tokenizer.chat_template)
