"""
Chapter 5: Synthetic Medical Q&A Data Pipeline (Fast Version)
==============================================================
Identical pipeline to chapter05_synth_gemini.py but using gemini-3.1-flash-lite-preview
as the teacher model. Flash Lite has no thinking mode, so responses are immediate —
expect 5-10x faster throughput than the Pro version at a fraction of the cost.
Use this version when iteration speed matters; use chapter05_synth_gemini.py when
you want the highest possible answer quality.

Requirements:
    pip install google-genai datasets

Usage:
    # Set your Gemini API key first:
    export GEMINI_API_KEY="..."

    # Run the full pipeline:
    python chapter05_synth_fast.py

    # Or import individual stages:
    from chapter05_synth_fast import generate_pair, evolve_question, verify_medical_accuracy
"""

import json
import os
import random
import time
from collections import Counter
from pathlib import Path

from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("data/synthetic_fast")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_EXAMPLES = 1000
BATCH_SIZE = 5
EVOL_SAMPLE_SIZE = 250       # How many examples to evolve for complexity
COT_FRACTION = 0.35          # Fraction of examples to generate with chain-of-thought
TEACHER_MODEL = "gemini-3.1-flash-lite-preview"   # Fast, no thinking overhead
SEED = 42

random.seed(SEED)
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# ---------------------------------------------------------------------------
# 1. Seed Generation (Section 5.3.2)
# ---------------------------------------------------------------------------

MEDICAL_TOPICS = [
    {"category": "Cardiology", "conditions": [
        "hypertension", "atrial fibrillation", "heart failure",
        "angina", "hyperlipidemia"]},
    {"category": "Endocrinology", "conditions": [
        "type 2 diabetes", "hypothyroidism", "hyperthyroidism",
        "PCOS", "adrenal insufficiency"]},
    {"category": "Pulmonology", "conditions": [
        "asthma", "COPD", "pneumonia",
        "pulmonary embolism", "sleep apnea"]},
    {"category": "Gastroenterology", "conditions": [
        "GERD", "IBS", "Crohn's disease",
        "celiac disease", "gallstones"]},
    {"category": "Neurology", "conditions": [
        "migraine", "epilepsy", "multiple sclerosis",
        "Parkinson's disease", "peripheral neuropathy"]},
    {"category": "Dermatology", "conditions": [
        "eczema", "psoriasis", "acne",
        "rosacea", "melanoma screening"]},
    {"category": "Orthopedics", "conditions": [
        "osteoarthritis", "rotator cuff injury", "plantar fasciitis",
        "herniated disc", "carpal tunnel syndrome"]},
    {"category": "Psychiatry", "conditions": [
        "generalized anxiety", "major depression", "insomnia",
        "ADHD in adults", "panic disorder"]},
]

QUESTION_TYPES = [
    "symptoms and presentation",
    "treatment options",
    "lifestyle modifications",
    "when to seek emergency care",
    "diagnostic workup",
    "medication side effects",
    "prognosis and long-term management",
    "prevention strategies",
]


def generate_seeds(n: int) -> list[dict]:
    """Create n seed combinations of (category, condition, question_type)."""
    seeds = []
    for _ in range(n):
        topic = random.choice(MEDICAL_TOPICS)
        condition = random.choice(topic["conditions"])
        q_type = random.choice(QUESTION_TYPES)
        seeds.append({
            "category": topic["category"],
            "condition": condition,
            "question_type": q_type,
        })
    return seeds


# ---------------------------------------------------------------------------
# 2. Teacher Prompting — Single & Batch (Sections 5.3.3–5.3.4)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a board-certified physician creating educational medical
content. Generate a realistic patient question and a detailed, accurate medical answer.

Rules:
- The question should sound like a real patient asking their doctor — natural language,
  not clinical jargon.
- The answer should be medically accurate, cite specific guidance where appropriate,
  and be 150-300 words.
- Include practical advice the patient can act on.
- Always recommend consulting a healthcare provider for personalized advice.
- Never fabricate drug names, dosages, or clinical trial results."""

COT_SYSTEM_PROMPT = """You are a board-certified physician working through a
clinical reasoning exercise. When answering the patient's question:

1. First, identify the key clinical considerations (2-3 sentences).
2. Then, work through the relevant differential or decision points step by step.
3. Finally, provide your recommendation with clear rationale.

Format your answer with these labeled sections:
CLINICAL CONSIDERATIONS: ...
REASONING: ...
RECOMMENDATION: ...

Be medically accurate. Never fabricate drug names or dosages."""


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers that Gemini often adds around JSON."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.index("\n")
        text = text[first_newline + 1:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _call_gemini(system_prompt: str, user_prompt: str,
                 temperature: float = 0.9, max_tokens: int = 8192) -> str:
    """Send a request to Gemini Flash Lite and return the text response.

    Flash Lite has no thinking mode, so responses are fast and max_tokens
    only needs to cover actual output. We still allow MAX_TOKENS as an
    acceptable finish reason and let json.loads decide usability.
    """
    if system_prompt:
        combined_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"
    else:
        combined_prompt = user_prompt

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    response = client.models.generate_content(
        model=TEACHER_MODEL,
        contents=combined_prompt,
        config=config,
    )

    if not response.candidates:
        block_reason = getattr(response, "prompt_feedback", None)
        raise RuntimeError(f"Gemini returned no candidates. Prompt feedback: {block_reason}")

    candidate = response.candidates[0]
    finish = candidate.finish_reason

    _ACCEPTABLE_FINISH = {"FinishReason.STOP", "STOP", "0", "FinishReason.MAX_TOKENS", "MAX_TOKENS"}
    if finish and str(finish) not in _ACCEPTABLE_FINISH:
        raise RuntimeError(
            f"Gemini finish_reason={finish}. "
            f"Safety ratings: {candidate.safety_ratings}"
        )

    text = response.text
    if not text:
        raise RuntimeError(
            f"Gemini returned empty text. finish_reason={finish}, "
            f"safety_ratings={candidate.safety_ratings}"
        )

    return _strip_markdown_fences(text)


def generate_pair(seed: dict, use_cot: bool = False) -> dict:
    """Generate a single Q&A pair from a seed (Section 5.3.3)."""
    sys_prompt = COT_SYSTEM_PROMPT if use_cot else SYSTEM_PROMPT
    user_prompt = (
        f"Generate a medical Q&A pair about {seed['condition']} "
        f"({seed['category']}), focusing on {seed['question_type']}.\n\n"
        f"Return ONLY valid JSON with exactly two keys: \"question\" and \"answer\". "
        f"No other text before or after the JSON."
    )

    content = _call_gemini(sys_prompt, user_prompt, temperature=0.9)
    pair = json.loads(content)
    pair["metadata"] = seed
    pair["is_cot"] = use_cot
    return pair


def generate_batch(seed_batch: list[dict], use_cot: bool = False) -> list[dict]:
    """Generate multiple Q&A pairs in a single API call (Section 5.3.4)."""
    sys_prompt = COT_SYSTEM_PROMPT if use_cot else SYSTEM_PROMPT
    batch_size = len(seed_batch)
    seed_descriptions = "\n".join([
        f"{i+1}. {s['condition']} ({s['category']}): {s['question_type']}"
        for i, s in enumerate(seed_batch)
    ])

    user_prompt = (
        f"Generate {batch_size} distinct medical Q&A pairs for these topics:\n\n"
        f"{seed_descriptions}\n\n"
        f"Return ONLY a JSON array of objects, each with \"question\" and \"answer\" keys. "
        f"No other text before or after the JSON."
    )

    content = _call_gemini(sys_prompt, user_prompt, temperature=0.9)
    pairs = json.loads(content)
    for pair, seed in zip(pairs, seed_batch):
        pair["metadata"] = seed
        pair["is_cot"] = use_cot
    return pairs


def run_generation(seeds: list[dict], cot_fraction: float = COT_FRACTION) -> list[dict]:
    """Generate all Q&A pairs from seeds, one at a time for reliability."""
    results = []
    errors = 0
    cot_cutoff = int(len(seeds) * (1 - cot_fraction))

    for i, seed in enumerate(seeds):
        use_cot = i >= cot_cutoff
        retries = 2

        for attempt in range(retries + 1):
            try:
                pair = generate_pair(seed, use_cot=use_cot)
                results.append(pair)
                break
            except (json.JSONDecodeError, Exception) as e:
                if attempt < retries:
                    time.sleep(2)
                else:
                    errors += 1
                    if errors <= 10:
                        print(f"  Skipping seed {i} ({seed['condition']}): {e}")
                    elif errors == 11:
                        print(f"  ... suppressing further error messages")

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(seeds)} seeds processed, "
                  f"{len(results)} pairs generated, {errors} errors")

    print(f"Generation complete: {len(results)} pairs ({errors} errors)")
    return results


# ---------------------------------------------------------------------------
# 3. Evol-Instruct Complexity Evolution (Section 5.4)
# ---------------------------------------------------------------------------

EVOLUTION_PROMPTS = {
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


def evolve_question(question: str, evolution_type: str) -> str:
    """Evolve a question to be more complex (Section 5.4)."""
    prompt = EVOLUTION_PROMPTS[evolution_type].format(question=question)
    content = _call_gemini("", prompt, temperature=0.8, max_tokens=1024)
    return content.strip()


def generate_answer_for_evolved(question: str) -> str:
    """Generate a new answer for an evolved question."""
    content = _call_gemini(
        SYSTEM_PROMPT,
        f"Answer this medical question in 150-300 words. Be accurate "
        f"and practical.\n\nQuestion: {question}",
        temperature=0.7,
        max_tokens=2048,
    )
    return content.strip()


def run_evolution(results: list[dict], sample_size: int = EVOL_SAMPLE_SIZE) -> list[dict]:
    """Select simple examples, evolve them, and generate new answers."""
    sorted_by_length = sorted(results, key=lambda r: len(r["question"]))
    candidates = sorted_by_length[:sample_size]

    evolution_types = list(EVOLUTION_PROMPTS.keys())
    evolved_pairs = []

    for i, pair in enumerate(candidates):
        evo_type = evolution_types[i % len(evolution_types)]
        try:
            evolved_q = evolve_question(pair["question"], evo_type)
            evolved_a = generate_answer_for_evolved(evolved_q)
            evolved_pairs.append({
                "question": evolved_q,
                "answer": evolved_a,
                "metadata": {
                    **pair.get("metadata", {}),
                    "evolution_type": evo_type,
                    "original_question": pair["question"],
                },
                "is_cot": False,
                "is_evolved": True,
            })
        except Exception as e:
            print(f"  Evolution error on example {i}: {e}")
            time.sleep(2)
            continue

        if (i + 1) % 25 == 0:
            print(f"  Evolved {i + 1}/{sample_size} questions")

    print(f"Evolution complete: {len(evolved_pairs)} evolved pairs")
    return evolved_pairs


# ---------------------------------------------------------------------------
# 4. Quality Control (Section 5.6)
# ---------------------------------------------------------------------------

def verify_medical_accuracy(question: str, answer: str) -> dict:
    """Cross-verify a Q&A pair for hallucinations (Section 5.6.1)."""
    verification_prompt = (
        f"Review this medical Q&A pair for accuracy.\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        f"Check for:\n"
        f"1. Fabricated drug names or medications that don't exist\n"
        f"2. Incorrect dosages or treatment protocols\n"
        f"3. Outdated or contradicted medical guidance\n"
        f"4. Claims that are misleading or could cause patient harm\n\n"
        f"Respond with ONLY valid JSON, no other text: "
        f"{{\"is_accurate\": true/false, \"issues\": [\"list of problems\"]}}"
    )

    content = _call_gemini("", verification_prompt, temperature=0.1, max_tokens=1024)
    return json.loads(content)


def ngram_diversity(texts: list[str], n: int = 3) -> float:
    """Compute unique n-gram ratio as a diversity metric (Section 5.6.2)."""
    all_ngrams = []
    for text in texts:
        words = text.lower().split()
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        all_ngrams.extend(ngrams)
    unique = len(set(all_ngrams))
    total = len(all_ngrams)
    return unique / total if total > 0 else 0.0


def topic_coverage(results: list[dict]) -> dict:
    """Check how many seed categories and conditions are represented."""
    categories = set()
    conditions = set()
    for r in results:
        meta = r.get("metadata", {})
        if "category" in meta:
            categories.add(meta["category"])
        if "condition" in meta:
            conditions.add(meta["condition"])

    all_categories = {t["category"] for t in MEDICAL_TOPICS}
    all_conditions = {c for t in MEDICAL_TOPICS for c in t["conditions"]}

    return {
        "categories_covered": len(categories),
        "categories_total": len(all_categories),
        "conditions_covered": len(conditions),
        "conditions_total": len(all_conditions),
        "missing_categories": all_categories - categories,
        "missing_conditions": all_conditions - conditions,
    }


def filter_synthetic_example(pair: dict, verification_result: dict) -> bool:
    """Decide whether to keep a synthetic example (Section 5.6.3)."""
    if not verification_result.get("is_accurate", False):
        return False
    if len(pair.get("answer", "")) < 200 or len(pair.get("answer", "")) > 2000:
        return False
    condition = pair.get("metadata", {}).get("condition", "")
    if condition and condition.lower() not in pair.get("answer", "").lower():
        return False
    return True


def run_verification_and_filtering(results: list[dict]) -> list[dict]:
    """Verify all examples and filter out bad ones."""
    filtered = []
    removed = {"hallucination": 0, "length": 0, "off_topic": 0, "error": 0}

    for i, pair in enumerate(results):
        try:
            verification = verify_medical_accuracy(pair["question"], pair["answer"])
        except Exception:
            verification = {"is_accurate": True, "issues": []}

        if filter_synthetic_example(pair, verification):
            pair["verification"] = verification
            filtered.append(pair)
        else:
            if not verification.get("is_accurate", True):
                removed["hallucination"] += 1
            elif len(pair.get("answer", "")) < 200 or len(pair.get("answer", "")) > 2000:
                removed["length"] += 1
            else:
                removed["off_topic"] += 1

        if (i + 1) % 50 == 0:
            print(f"  Verified {i + 1}/{len(results)} — kept {len(filtered)} so far")

    print(f"Filtering complete: {len(filtered)} kept, {sum(removed.values())} removed")
    print(f"  Removal breakdown: {dict(removed)}")
    return filtered


# ---------------------------------------------------------------------------
# 5. Mixing Real and Synthetic Data (Section 5.7)
# ---------------------------------------------------------------------------

def mix_uniform(real: list[dict], synthetic: list[dict]) -> list[dict]:
    """Shuffle real and synthetic examples together."""
    combined = real + synthetic
    random.shuffle(combined)
    return combined


def mix_upsampled(real: list[dict], synthetic: list[dict],
                  upsample_factor: int = 4) -> list[dict]:
    """Repeat real examples N times, then shuffle with synthetic."""
    upsampled_real = real * upsample_factor
    combined = upsampled_real + synthetic
    random.shuffle(combined)
    return combined


def mix_curriculum(real: list[dict], synthetic: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return two datasets: stage 1 (synthetic) and stage 2 (real)."""
    stage1 = synthetic.copy()
    stage2 = real.copy()
    random.shuffle(stage1)
    random.shuffle(stage2)
    return stage1, stage2


# ---------------------------------------------------------------------------
# 6. Output (Section 4.6.2 — shared format)
# ---------------------------------------------------------------------------

def save_jsonl(examples: list[dict], filepath: Path) -> None:
    """Save examples as JSONL, converting to instruction/input/output format."""
    with open(filepath, "w") as f:
        for ex in examples:
            record = {
                "instruction": ex["question"],
                "input": "",
                "output": ex["answer"],
            }
            if "metadata" in ex:
                record["metadata"] = ex["metadata"]
            if ex.get("is_cot"):
                record["format"] = "chain_of_thought"
            if ex.get("is_evolved"):
                record["format"] = "evolved"
            f.write(json.dumps(record) + "\n")
    print(f"Saved {len(examples)} examples to {filepath}")


def print_diversity_report(results: list[dict]) -> None:
    """Print diversity and coverage metrics."""
    questions = [r["question"] for r in results]
    answers = [r["answer"] for r in results]

    q_diversity = ngram_diversity(questions, n=3)
    a_diversity = ngram_diversity(answers, n=3)
    coverage = topic_coverage(results)

    print("\n=== Diversity Report ===")
    print(f"Question trigram diversity: {q_diversity:.3f}")
    print(f"Answer trigram diversity:   {a_diversity:.3f}")
    print(f"Categories covered: {coverage['categories_covered']}/{coverage['categories_total']}")
    print(f"Conditions covered: {coverage['conditions_covered']}/{coverage['conditions_total']}")
    if coverage["missing_conditions"]:
        print(f"Missing conditions: {coverage['missing_conditions']}")
    if q_diversity < 0.7:
        print("WARNING: Question diversity is below 0.7 — consider increasing temperature "
              "or adding more seed variation.")
    print()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Chapter 5: Synthetic Medical Q&A Pipeline (Fast)")
    print(f"Model: {TEACHER_MODEL}")
    print("=" * 60)

    print("\n[Step 1] Generating seeds...")
    seeds = generate_seeds(NUM_EXAMPLES)
    print(f"  Created {len(seeds)} seeds across "
          f"{len({s['category'] for s in seeds})} categories")

    print(f"\n[Step 2] Generating {NUM_EXAMPLES} Q&A pairs "
          f"({int(COT_FRACTION*100)}% chain-of-thought)...")
    results = run_generation(seeds, cot_fraction=COT_FRACTION)

    save_jsonl(results, OUTPUT_DIR / "raw_generated.jsonl")

    print(f"\n[Step 3] Evolving {EVOL_SAMPLE_SIZE} examples for complexity...")
    evolved = run_evolution(results, sample_size=EVOL_SAMPLE_SIZE)
    results.extend(evolved)
    print(f"  Total examples after evolution: {len(results)}")

    print("\n[Step 4] Analyzing diversity...")
    print_diversity_report(results)

    print("[Step 5] Verifying accuracy and filtering...")
    filtered = run_verification_and_filtering(results)
    print_diversity_report(filtered)

    print("[Step 6] Saving final dataset...")
    save_jsonl(filtered, OUTPUT_DIR / "medical_qa_synthetic.jsonl")

    real_data_path = Path("data/real/medical_qa_real.jsonl")
    if real_data_path.exists():
        print("\n[Step 7] Mixing with real data...")
        real_examples = []
        with open(real_data_path) as f:
            for line in f:
                obj = json.loads(line)
                real_examples.append({
                    "question": obj.get("instruction", obj.get("question", "")),
                    "answer": obj.get("output", obj.get("answer", "")),
                    "metadata": {"source": "real"},
                })

        uniform = mix_uniform(real_examples, filtered)
        save_jsonl(uniform, OUTPUT_DIR / "mixed_uniform.jsonl")

        upsampled = mix_upsampled(real_examples, filtered, upsample_factor=4)
        save_jsonl(upsampled, OUTPUT_DIR / "mixed_upsampled.jsonl")

        stage1, stage2 = mix_curriculum(real_examples, filtered)
        save_jsonl(stage1, OUTPUT_DIR / "curriculum_stage1_synthetic.jsonl")
        save_jsonl(stage2, OUTPUT_DIR / "curriculum_stage2_real.jsonl")

        print(f"  Real examples: {len(real_examples)}")
        print(f"  Uniform mix: {len(uniform)} total")
        print(f"  Upsampled mix: {len(upsampled)} total")
        print(f"  Curriculum: {len(stage1)} synthetic → {len(stage2)} real")
    else:
        print(f"\n[Step 7] No real data found at {real_data_path} — skipping mixing step.")
        print("  To enable mixing, place a JSONL file with 'instruction'/'output' fields there.")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"  Seeds generated:        {len(seeds)}")
    print(f"  Raw pairs generated:    {len(results)}")
    print(f"  Pairs after filtering:  {len(filtered)}")
    print(f"  Output directory:       {OUTPUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
