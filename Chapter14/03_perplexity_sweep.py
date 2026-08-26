"""
03_perplexity_sweep.py
----------------------
Measure the damage. For each quantized GGUF variant, compute:
  - perplexity on a held-out text set  (tripwire: lower is better)
  - task accuracy on the medical test set (the number that matters)
and print a table so you can see the quality knee.

Requires:  pip install llama-cpp-python
The medical test set must be DISJOINT from training and calibration data.

Usage:
    python 03_perplexity_sweep.py --test-set test_set.jsonl
"""
import argparse
import json
import math
import os
from llama_cpp import Llama

# Map a label -> GGUF path. Add/remove rows to match what you built.
VARIANTS = {
    "f16  (baseline)": "medical-gemma-f16.gguf",
    "Q6_K":            "medical-gemma-Q6_K.gguf",
    "Q5_K_M":          "medical-gemma-Q5_K_M.gguf",
    "Q4_K_M":          "medical-gemma-Q4_K_M.gguf",
    "Q3_K_M":          "medical-gemma-Q3_K_M.gguf",
}


def load(path, n_gpu_layers=-1, logits_all=False):
    # n_gpu_layers=-1 offloads ALL layers to the GPU, which is the difference
    # between minutes and an hour. It requires a CUDA build of llama-cpp-python:
    #   CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall \
    #       --no-cache-dir llama-cpp-python
    # On a CPU-only build the flag is simply ignored (it stays on CPU).
    # logits_all=True is needed ONLY for perplexity and is expensive on a big
    # vocab, so we leave it off unless perplexity is actually requested.
    return Llama(model_path=path, n_ctx=2048, logits_all=logits_all,
                 n_gpu_layers=n_gpu_layers, verbose=False)


def perplexity(llm, texts):
    """Token-level perplexity over a list of held-out strings."""
    total_nll, total_tokens = 0.0, 0
    for text in texts:
        out = llm(text, max_tokens=0, echo=True, logprobs=0)
        lps = out["choices"][0]["logprobs"]["token_logprobs"]
        lps = [lp for lp in lps if lp is not None]
        total_nll += -sum(lps)
        total_tokens += len(lps)
    return math.exp(total_nll / max(total_tokens, 1))


def _token_f1(pred, ref):
    """Token-overlap F1 between a prediction and a reference string (the
    SQuAD-style metric). Returns a value in [0, 1]. This is a soft score that
    rewards saying the right things without demanding a verbatim match, so it
    actually moves as quantization degrades the answers, unlike exact match."""
    import re
    tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
    p, r = tok(pred), tok(ref)
    if not p or not r:
        return 0.0
    common = {}
    for w in p:
        if w in r:
            common[w] = min(p.count(w), r.count(w))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def medical_accuracy(llm, test_set):
    """Mean token-overlap F1 against the reference answers. For open-ended
    answers this is far more informative than exact/contains match (which is
    ~0 because the model paraphrases). For an authoritative score on answer
    quality, swap in an LLM-as-judge; F1 is the dependency-free stand-in that
    tracks relative quality across quantization levels."""
    total = 0.0
    for ex in test_set:
        prompt = f"<start_of_turn>user\n{ex['question']}<end_of_turn>\n<start_of_turn>model\n"
        out = llm(prompt, max_tokens=256, temperature=0.0)
        ans = out["choices"][0]["text"]
        total += _token_f1(ans, ex["answer"])
    return total / len(test_set)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-set", required=True)
    ap.add_argument("--heldout-text", default=None,
                    help="text file for perplexity; defaults to test answers")
    ap.add_argument("--n-gpu-layers", type=int, default=-1,
                    help="layers to offload to GPU (-1 = all; 0 = CPU only). "
                         "Needs a CUDA build of llama-cpp-python.")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of test rows graded, to keep it quick")
    ap.add_argument("--no-ppl", action="store_true",
                    help="skip the slow perplexity pass; grade with F1 only")
    ap.add_argument("--ppl-samples", type=int, default=15,
                    help="held-out texts to score for perplexity (when enabled)")
    args = ap.parse_args()

    want_ppl = not args.no_ppl

    test_set = [json.loads(l) for l in open(args.test_set)]
    if args.limit:
        test_set = test_set[:args.limit]
    if args.heldout_text:
        heldout = [l for l in open(args.heldout_text) if l.strip()][:args.ppl_samples]
    else:
        # Cap count AND truncate length: perplexity cost scales with both.
        heldout = [ex["answer"][:600] for ex in test_set[:args.ppl_samples]]

    print(f"{'variant':18s} {'size':>8s}  {'ppl':>7s}  {'ansF1':>6s}")
    print("-" * 46)
    for name, path in VARIANTS.items():
        if not os.path.exists(path):
            print(f"{name:18s}  (missing: {path})")
            continue
        llm = load(path, args.n_gpu_layers, logits_all=want_ppl)
        ppl = perplexity(llm, heldout) if want_ppl else None
        acc = medical_accuracy(llm, test_set)
        size_gb = os.path.getsize(path) / 1e9
        ppl_s = f"{ppl:7.3f}" if ppl is not None else f"{'—':>7s}"
        print(f"{name:18s} {size_gb:6.2f}GB  {ppl_s}  {acc:5.1%}")
        del llm

    print("\nPick the smallest variant whose accuracy is still above the knee.")


if __name__ == "__main__":
    main()
