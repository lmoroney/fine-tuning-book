"""
Chapter 5: Synthetic Data — Manufacturing Training Fuel
========================================================
Complete code for generating synthetic medical Q&A training data
using a teacher model (GPT-OSS 20B) and preparing it for fine-tuning.

This version uses GPT-OSS 20B running LOCALLY via Ollama, through
Ollama's OpenAI-compatible API. Before running:

    1. Install Ollama (https://ollama.com) and make sure it's running.
    2. Pull the model:  ollama pull gpt-oss:20b

No API key is required — everything runs on your own machine.
"""

import random
import json
import time
from collections import Counter
from openai import OpenAI

# ──────────────────────────────────────────────────────────────
# 5.3.2 — Seed-Based Generation
# ──────────────────────────────────────────────────────────────

medical_topics = [
    {"category": "Cardiology", "conditions": ["hypertension", "atrial fibrillation",
     "heart failure", "angina", "hyperlipidemia"]},
    {"category": "Endocrinology", "conditions": ["type 2 diabetes", "hypothyroidism",
     "hyperthyroidism", "PCOS", "adrenal insufficiency"]},
    {"category": "Pulmonology", "conditions": ["asthma", "COPD", "pneumonia",
     "pulmonary embolism", "sleep apnea"]},
    {"category": "Gastroenterology", "conditions": ["GERD", "IBS", "Crohn's disease",
     "celiac disease", "gallstones"]},
    {"category": "Neurology", "conditions": ["migraine", "epilepsy", "multiple sclerosis",
     "Parkinson's disease", "peripheral neuropathy"]},
    {"category": "Dermatology", "conditions": ["eczema", "psoriasis", "acne",
     "rosacea", "melanoma screening"]},
    {"category": "Orthopedics", "conditions": ["osteoarthritis", "rotator cuff injury",
     "plantar fasciitis", "herniated disc", "carpal tunnel syndrome"]},
    {"category": "Psychiatry", "conditions": ["generalized anxiety", "major depression",
     "insomnia", "ADHD in adults", "panic disorder"]},
]

question_types = [
    "symptoms and presentation",
    "treatment options",
    "lifestyle modifications",
    "when to seek emergency care",
    "diagnostic workup",
    "medication side effects",
    "prognosis and long-term management",
    "prevention strategies",
]

# Generate 1,000 unique seed combinations
seeds = []
for _ in range(1000):
    topic = random.choice(medical_topics)
    condition = random.choice(topic["conditions"])
    q_type = random.choice(question_types)
    seeds.append({
        "category": topic["category"],
        "condition": condition,
        "question_type": q_type,
    })


# ──────────────────────────────────────────────────────────────
# 5.3.3 — Prompting the Teacher
# ──────────────────────────────────────────────────────────────

# Uses GPT-OSS 20B served locally by Ollama via its OpenAI-compatible endpoint.
# Ollama listens on port 11434 by default; the api_key is required by the client
# library but ignored by Ollama, so any placeholder string works.
OLLAMA_BASE_URL = "http://localhost:11434/v1"
TEACHER_MODEL = "gpt-oss:20b"  # Ollama model tag — see `ollama list`

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

# Fail fast with a helpful message if Ollama isn't running or the model is missing
try:
    available_models = [m.id for m in client.models.list().data]
    if TEACHER_MODEL not in available_models:
        raise SystemExit(
            f"Model '{TEACHER_MODEL}' not found in Ollama. "
            f"Run: ollama pull {TEACHER_MODEL}\n"
            f"Available models: {', '.join(available_models) or '(none)'}"
        )
except SystemExit:
    raise
except Exception as e:
    raise SystemExit(
        f"Could not reach Ollama at {OLLAMA_BASE_URL} — is it running? ({e})"
    )

system_prompt = """You are a board-certified physician creating educational medical
content. Generate a realistic patient question and a detailed, accurate medical answer.

Rules:
- The question should sound like a real patient asking their doctor — natural language,
  not clinical jargon.
- The answer should be medically accurate, cite specific guidance where appropriate,
  and be 150-300 words.
- Include practical advice the patient can act on.
- Always recommend consulting a healthcare provider for personalized advice.
- Never fabricate drug names, dosages, or clinical trial results."""


def generate_pair(seed):
    user_prompt = f"""Generate a medical Q&A pair about {seed['condition']}
({seed['category']}), focusing on {seed['question_type']}.

Return valid JSON with exactly two keys: "question" and "answer"."""

    response = client.chat.completions.create(
        model=TEACHER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.9,
        # GPT-OSS is a reasoning model: part of the token budget is spent on
        # hidden "thinking" before the final answer, so we allow more headroom
        # here than the 600 tokens the visible answer actually needs.
        max_tokens=1200,
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        # The model spent its whole budget thinking and produced no answer —
        # raise so this seed is logged as an error and retried later
        raise ValueError("Empty response from teacher (token budget exhausted?)")
    # Strip markdown fences if the model wraps its response in ```json ... ```
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]  # Remove opening ```json line
        content = content.rsplit("```", 1)[0]  # Remove closing ```
    # Parse the JSON, allowing control characters (newlines inside strings)
    pair = json.loads(content, strict=False)
    pair["metadata"] = seed  # Keep the seed for traceability
    return pair


# Generate all 1,000 pairs with error handling
# (No rate limits locally, but the model can still return malformed JSON)
# Results are written to a JSONL file incrementally so nothing is lost on failure
output_file = "medical_qa_pairs.jsonl"
error_file = "medical_qa_errors.jsonl"
results = []
errors = []

# Check if we already have generated data — skip generation if so
import os
if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
    with open(output_file, "r") as f:
        results = [json.loads(line) for line in f if line.strip()]
    print(f"Found existing {output_file} with {len(results)} pairs — skipping generation.")
    print(f"Delete {output_file} if you want to regenerate from scratch.")
else:
    with open(output_file, "a") as f_out, open(error_file, "a") as f_err:
        for i, seed in enumerate(seeds):
            try:
                print(f"[{i+1}/1000] Generating: {seed['condition']} ({seed['category']}) "
                      f"— {seed['question_type']}...", end=" ", flush=True)
                pair = generate_pair(seed)
                results.append(pair)
                f_out.write(json.dumps(pair) + "\n")
                f_out.flush()  # Force write to disk immediately
                print("OK")
            except Exception as e:
                print(f"FAILED: {e}")
                error_entry = {"seed_index": i, "seed": seed, "error": str(e)}
                errors.append(error_entry)
                f_err.write(json.dumps(error_entry) + "\n")
                f_err.flush()
                time.sleep(1)  # Brief pause before the next attempt
                continue

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(results)} pairs")
    print(f"Errors: {len(errors)}")
    print(f"Results saved to: {output_file}")
    if errors:
        print(f"Failed seeds: {[e['seed_index'] for e in errors]}")
        print(f"Error details saved to: {error_file}")


# ──────────────────────────────────────────────────────────────
# 5.3.4 — Batching for Cost Efficiency
# ──────────────────────────────────────────────────────────────

def generate_batch(seed_batch, batch_size=5):
    seed_descriptions = "\n".join([
        f"{i+1}. {s['condition']} ({s['category']}): {s['question_type']}"
        for i, s in enumerate(seed_batch)
    ])

    user_prompt = f"""Generate {batch_size} distinct medical Q&A pairs for these topics:

{seed_descriptions}

Return a JSON array of objects, each with "question" and "answer" keys."""

    response = client.chat.completions.create(
        model=TEACHER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.9,
        # Extra headroom: 5 answers per call, plus the reasoning tokens
        max_tokens=4500,
    )
    content = (response.choices[0].message.content or "").strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        content = content.rsplit("```", 1)[0]
    return json.loads(content, strict=False)


# ──────────────────────────────────────────────────────────────
# 5.4 — Evol-Instruct: Evolving Complexity
# ──────────────────────────────────────────────────────────────

evolution_prompts = {
    "add_constraints": """Take a medical question and make it more complex by
adding 1-2 patient-specific constraints (e.g., age, comorbidities, allergies,
pregnancy, current medications). Return the evolved question only.

Example:
Original: What are the treatment options for GERD?
Evolved: What are the treatment options for GERD in a 72-year-old who is
already taking clopidogrel after a stent placement?

Example:
Original: How is asthma diagnosed?
Evolved: How is asthma diagnosed in a patient who smokes and whose symptoms
only appear during exercise?

Now evolve this question:
Original: {question}""",

    "deepen_reasoning": """Take a medical question and evolve it to require
deeper clinical reasoning — the answer should need to explain underlying
mechanisms, weigh trade-offs, or consider multiple factors. Return the
evolved question only.

Example:
Original: Should I take a statin for high cholesterol?
Evolved: How do statins reduce cardiovascular risk beyond lowering my LDL
number, and how should I weigh that benefit against the risk of muscle
side effects?

Example:
Original: What is the treatment for insomnia?
Evolved: Why do doctors recommend cognitive behavioral therapy before sleep
medication for chronic insomnia, and what are the trade-offs if I start
with medication instead?

Now evolve this question:
Original: {question}""",

    "increase_specificity": """Take a medical question and make it more specific
— add lab values, specific staging, or particular clinical scenarios that
narrow the question. Return the evolved question only.

Example:
Original: What do my thyroid test results mean?
Evolved: My TSH came back at 6.8 with a normal free T4 — does subclinical
hypothyroidism like this need treatment, or should we just monitor it?

Example:
Original: How serious is my kidney disease?
Evolved: My eGFR has dropped from 55 to 48 over the past year and my urine
albumin is 40 mg/g — what does stage 3a kidney disease with moderate
albuminuria mean for my long-term outlook?

Now evolve this question:
Original: {question}""",
}


def evolve_question(question, evolution_type):
    prompt = evolution_prompts[evolution_type].format(question=question)
    response = client.chat.completions.create(
        model=TEACHER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        # The evolved question itself is short, but leave room for reasoning
        max_tokens=600,
    )
    return (response.choices[0].message.content or "").strip()


# ──────────────────────────────────────────────────────────────
# 5.5 — Generating Chain-of-Thought Reasoning
# ──────────────────────────────────────────────────────────────

cot_system_prompt = """You are a board-certified physician working through a
clinical reasoning exercise. When answering the patient's question:

1. First, identify the key clinical considerations (2-3 sentences).
2. Then, work through the relevant differential or decision points step by step.
3. Finally, provide your recommendation with clear rationale.

Format your answer with these labeled sections:
CLINICAL CONSIDERATIONS: ...
REASONING: ...
RECOMMENDATION: ...

Be medically accurate. Never fabricate drug names or dosages."""


# ──────────────────────────────────────────────────────────────
# 5.6.1 — Hallucination Detection
# ──────────────────────────────────────────────────────────────

def verify_medical_accuracy(question, answer):
    verification_prompt = f"""Review this medical Q&A pair for accuracy.

Question: {question}
Answer: {answer}

Check for:
1. Fabricated drug names or medications that don't exist
2. Incorrect dosages or treatment protocols
3. Outdated or contradicted medical guidance
4. Claims that are misleading or could cause patient harm

Respond with JSON: {{"is_accurate": true/false, "issues": ["list of problems"]}}"""

    response = client.chat.completions.create(
        model=TEACHER_MODEL,
        messages=[{"role": "user", "content": verification_prompt}],
        temperature=0.1,  # Low temp for consistent verification
    )
    content = (response.choices[0].message.content or "").strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        content = content.rsplit("```", 1)[0]
    return json.loads(content, strict=False)


# ──────────────────────────────────────────────────────────────
# 5.6.2 — Diversity Metrics
# ──────────────────────────────────────────────────────────────

def ngram_diversity(texts, n=3):
    all_ngrams = []
    for text in texts:
        words = text.lower().split()
        ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
        all_ngrams.extend(ngrams)
    unique = len(set(all_ngrams))
    total = len(all_ngrams)
    return unique / total if total > 0 else 0


# Load results from disk (so this section can run independently of generation)
if not results:
    print(f"Loading previously generated pairs from {output_file}...")
    with open(output_file, "r") as f:
        results = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(results)} pairs from disk")

questions = [r["question"] for r in results]
answers = [r["answer"] for r in results]

print(f"Question trigram diversity: {ngram_diversity(questions):.3f}")
print(f"Answer trigram diversity: {ngram_diversity(answers):.3f}")


# ──────────────────────────────────────────────────────────────
# 5.6.3 — Filtering the Final Dataset
# ──────────────────────────────────────────────────────────────

def filter_synthetic_example(pair, verification_result):
    # Remove hallucinated content
    if not verification_result["is_accurate"]:
        return False
    # Length sanity checks
    if len(pair["answer"]) < 200 or len(pair["answer"]) > 2000:
        return False
    # Ensure the answer actually addresses the question
    # (Sometimes the teacher generates a generic response that ignores the seed)
    if pair["metadata"]["condition"].lower() not in pair["answer"].lower():
        return False
    return True


# ──────────────────────────────────────────────────────────────
# Run the verification and filtering pipeline
# ──────────────────────────────────────────────────────────────

verified_file = "medical_qa_verified.jsonl"
filtered_file = "medical_qa_filtered.jsonl"

# Step 1: Verify each pair for medical accuracy
print(f"\n{'='*60}")
print("Running verification on {len(results)} pairs...")

if os.path.exists(verified_file) and os.path.getsize(verified_file) > 0:
    with open(verified_file, "r") as f:
        verified_pairs = [json.loads(line) for line in f if line.strip()]
    print(f"Found existing {verified_file} with {len(verified_pairs)} entries — skipping verification.")
else:
    verified_pairs = []
    with open(verified_file, "a") as f_out:
        for i, pair in enumerate(results):
            try:
                print(f"[{i+1}/{len(results)}] Verifying: {pair['metadata']['condition']} "
                      f"— {pair['metadata']['question_type']}...", end=" ", flush=True)
                verification = verify_medical_accuracy(pair["question"], pair["answer"])
                entry = {**pair, "verification": verification}
                verified_pairs.append(entry)
                f_out.write(json.dumps(entry) + "\n")
                f_out.flush()
                status = "PASS" if verification["is_accurate"] else f"FLAGGED: {verification['issues']}"
                print(status)
            except Exception as e:
                print(f"ERROR: {e}")
                # If verification fails, keep the pair but mark it as unverified
                entry = {**pair, "verification": {"is_accurate": False, "issues": [f"Verification error: {e}"]}}
                verified_pairs.append(entry)
                f_out.write(json.dumps(entry) + "\n")
                f_out.flush()
                time.sleep(5)

    flagged = sum(1 for p in verified_pairs if not p["verification"].get("is_accurate", False))
    print(f"\nVerification complete: {flagged}/{len(verified_pairs)} flagged for issues")

# Step 2: Apply filters
print(f"\n{'='*60}")
print("Filtering dataset...")

filtered_pairs = []
removed_reasons = {"hallucination": 0, "length": 0, "off_topic": 0}

for pair in verified_pairs:
    verification = pair["verification"]
    if not verification.get("is_accurate", False):
        removed_reasons["hallucination"] += 1
        continue
    if len(pair["answer"]) < 200 or len(pair["answer"]) > 2000:
        removed_reasons["length"] += 1
        continue
    if pair["metadata"]["condition"].lower() not in pair["answer"].lower():
        removed_reasons["off_topic"] += 1
        continue
    filtered_pairs.append(pair)

# Write filtered dataset
with open(filtered_file, "w") as f:
    for pair in filtered_pairs:
        # Write clean pairs without the verification metadata
        clean = {"question": pair["question"], "answer": pair["answer"], "metadata": pair["metadata"]}
        f.write(json.dumps(clean) + "\n")

print(f"Filtered dataset: {len(filtered_pairs)}/{len(verified_pairs)} pairs kept")
print(f"Removed — hallucination: {removed_reasons['hallucination']}, "
      f"length: {removed_reasons['length']}, off-topic: {removed_reasons['off_topic']}")
print(f"Final dataset saved to: {filtered_file}")

# Step 3: Topic coverage check
print(f"\n{'='*60}")
print("Topic coverage in filtered dataset:")
from collections import defaultdict
coverage = defaultdict(set)
for pair in filtered_pairs:
    coverage[pair["metadata"]["category"]].add(pair["metadata"]["condition"])
for category in sorted(coverage.keys()):
    conditions = sorted(coverage[category])
    print(f"  {category}: {len(conditions)} conditions — {', '.join(conditions)}")


# ──────────────────────────────────────────────────────────────
# 5.7 — Mixing Real and Synthetic Data
# ──────────────────────────────────────────────────────────────

from datasets import Dataset, concatenate_datasets

# Load the filtered synthetic data
synthetic_examples = []
with open(filtered_file, "r") as f:
    for line in f:
        if line.strip():
            pair = json.loads(line)
            synthetic_examples.append({"question": pair["question"], "answer": pair["answer"]})

synthetic_ds = Dataset.from_list(synthetic_examples)
print(f"\nLoaded {len(synthetic_ds)} synthetic examples into HuggingFace Dataset")

# If you have real examples, uncomment and provide them:
# real_examples = [...]  # list of {"question": ..., "answer": ...} dicts
# real_ds = Dataset.from_list(real_examples)

# Approach 1: Uniform mix
# combined = concatenate_datasets([real_ds, synthetic_ds]).shuffle(seed=42)

# Approach 2: Upsample real data 4x
# real_upsampled = concatenate_datasets([real_ds] * 4)
# combined_upsampled = concatenate_datasets([real_upsampled, synthetic_ds]).shuffle(seed=42)

# Approach 3: Curriculum (two separate datasets for two training stages)
# stage1_data = synthetic_ds.shuffle(seed=42)
# stage2_data = real_ds.shuffle(seed=42)

print(f"\n{'='*60}")
print("Pipeline complete!")
print(f"  Generated: {len(results)} pairs")
print(f"  After filtering: {len(filtered_pairs)} pairs")
print(f"  Final dataset: {filtered_file}")
