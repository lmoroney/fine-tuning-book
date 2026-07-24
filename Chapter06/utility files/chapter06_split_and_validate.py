"""
Chapter 6: Preparing Your Dataset for Fine Tuning
=================================================
Sections: Splitting and Validating your Dataset / Format Validation /
          Inspecting Tokenized Examples

This is the full validation code referenced in the chapter ("much too
long to put into print"). It:

  1. Splits the dataset 85/10/5 into train/validation/test
  2. Validates every example against Gemma-oriented format rules
  3. Reports token-length percentiles (p50/p90/p99/max)
  4. Lets you eyeball individual formatted examples

Expected output looks like:

    Valid examples: 580/580
    Token length — p50: 435, p90: 510, p99: 560, max: 700
    Over limit (2048 tokens): 0
"""

from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "google/gemma-4-E4B-it"
MAX_TOKENS = 2048

SYSTEM_PROMPT = """You are a board-certified physician providing educational \
medical information. Your answers are accurate, practical, and always recommend \
consulting a healthcare provider for personalized advice."""

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ──────────────────────────────────────────────────────────────
# Template helper for models without a standalone system role
# ──────────────────────────────────────────────────────────────

def _apply_template_safe(tokenizer, messages, **kwargs):
    """Apply the chat template, folding the system message into the
    first user turn for models whose template rejects a standalone
    system role. (Gemma 4 accepts the system role natively, so for our
    model the fallback never triggers — but earlier Gemma generations
    and some other families need it, so it's a useful general tool.)"""
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        if messages and messages[0]["role"] == "system":
            system = messages[0]["content"]
            rest = [dict(m) for m in messages[1:]]
            if rest and rest[0]["role"] == "user":
                rest[0]["content"] = f"{system}\n\n{rest[0]['content']}"
                return tokenizer.apply_chat_template(rest, **kwargs)
        raise  # a different problem — let the caller see it


# ──────────────────────────────────────────────────────────────
# Format validation rules (a good starting point for any model)
# ──────────────────────────────────────────────────────────────

def normalize(example: dict) -> list | None:
    """Accept either a 'messages' key or both 'instruction' and
    'output' keys (Alpaca). Anything else fails (returns None)."""
    if "messages" in example:
        return example["messages"]
    if "instruction" in example and "output" in example:
        user_content = example["instruction"]
        if example.get("input", "").strip():
            user_content = f"{user_content}\n\n{example['input']}"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["output"]},
        ]
    return None


def validate_example(example: dict) -> tuple[bool, list, int]:
    """Returns (is_valid, list_of_issues, token_count)."""
    issues = []
    token_count = 0

    messages = normalize(example)
    if messages is None:
        return False, ["needs 'messages', or both 'instruction' and 'output'"], 0

    # The messages list should not be empty
    if not messages:
        return False, ["messages list is empty"], 0

    # The last message should be the model's turn
    # (Gemma calls the assistant role 'model'; messages-format data
    # uses 'assistant' and the template maps it — accept either)
    if messages[-1]["role"] not in ("assistant", "model"):
        issues.append(f"last message role is '{messages[-1]['role']}', "
                      f"expected the model's turn")

    # At least one message should come from the user
    if not any(m["role"] == "user" for m in messages):
        issues.append("no user message in conversation")

    # There should be no empty messages
    for i, m in enumerate(messages):
        if not m.get("content", "").strip():
            issues.append(f"message {i} (role={m.get('role')}) is empty")

    # The template must apply without throwing, and the formatted
    # example must fit within the model's maximum token length
    try:
        formatted = _apply_template_safe(tokenizer, messages, tokenize=False)
        token_count = len(tokenizer.encode(formatted))
        if token_count > MAX_TOKENS:
            issues.append(f"too long: {token_count} > {MAX_TOKENS} tokens")
    except Exception as e:
        issues.append(f"chat template failed: {e}")

    return len(issues) == 0, issues, token_count


# ──────────────────────────────────────────────────────────────
# Token-length statistics
# ──────────────────────────────────────────────────────────────

def length_percentiles(lengths: list) -> dict:
    """p50/p90/p99/max over token lengths. A big gap between p90 and
    p99 flags a handful of unusually long examples — a red flag."""
    s = sorted(lengths)
    pick = lambda p: s[min(int(len(s) * p), len(s) - 1)] if s else 0
    return {"p50": pick(0.50), "p90": pick(0.90),
            "p99": pick(0.99), "max": s[-1] if s else 0}


# ──────────────────────────────────────────────────────────────
# Eyeball inspection of individual examples
# ──────────────────────────────────────────────────────────────

def inspect_example(dataset, tokenizer, index: int = 0) -> None:
    """Print a tokenized example with its formatted string."""
    example = dataset[index]
    messages = normalize(example)

    print(f"=== Example {index} ===")
    for msg in messages:
        print(f"\n[{msg['role'].upper()}]")
        print(msg["content"])

    # Show the full formatted string as the model sees it
    formatted = _apply_template_safe(tokenizer, messages, tokenize=False)
    print(f"\n=== Formatted String ({len(formatted)} chars) ===")
    print(formatted)

    # Show the token count
    tokens = tokenizer.encode(formatted)
    print(f"\nTotal tokens: {len(tokens)}")


# ──────────────────────────────────────────────────────────────
# Load, split 85/10/5, validate, report
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Load the raw dataset into 'all_data' from the book's hosted
    # dataset on the Hugging Face Hub. (To use your own Chapter 5
    # output instead: load_dataset("json",
    #     data_files={"data": "medical_qa_synthetic.jsonl"}, split="data"))
    all_data = load_dataset("lmoroney/synthdata", split="train")

    # 2. Extract 85% for Training, leaving 15% as a remainder
    train_and_remainder = all_data.train_test_split(test_size=0.15, seed=42)
    train_set = train_and_remainder["train"]
    remainder = train_and_remainder["test"]

    # 3. Split the 15% remainder: 10% for Validation, 5% for Test
    # (Since 5% is 1/3 of the 15% remainder, we use test_size=1/3)
    val_test_split = remainder.train_test_split(test_size=1 / 3, seed=42)
    val_set = val_test_split["train"]
    test_set = val_test_split["test"]

    print(f"Split sizes — train: {len(train_set)}, "
          f"validation: {len(val_set)}, test: {len(test_set)}")

    # NOTE: this is a naive random split. With synthetic data, near-
    # duplicate seeds (e.g. 'asthma / treatment options') can land in
    # both train and test, inflating your generalization estimate. If
    # that happens, split by category instead, putting whole categories
    # into train or test.

    # 4. Validate every example in the training set
    valid_count = 0
    lengths = []
    over_limit = 0
    for i in range(len(train_set)):
        ok, issues, token_count = validate_example(train_set[i])
        lengths.append(token_count)
        if token_count > MAX_TOKENS:
            over_limit += 1
        if ok:
            valid_count += 1
        else:
            print(f"Example {i} FAILED: {'; '.join(issues)}")

    p = length_percentiles(lengths)
    print(f"\nValid examples: {valid_count}/{len(train_set)}")
    print(f"Token length — p50: {p['p50']}, p90: {p['p90']}, "
          f"p99: {p['p99']}, max: {p['max']}")
    print(f"Over limit ({MAX_TOKENS} tokens): {over_limit}")

    # 5. Spot-check a few examples — beginning, middle, and end
    for i in [0, len(train_set) // 2, len(train_set) - 1]:
        inspect_example(train_set, tokenizer, i)
        print()
