"""
05_compression_pipeline.py
--------------------------
End-to-end, gated compression pipeline:

    adapter + base
        -> MERGE (fp16)          + parity check
        -> CONVERT to GGUF f16
        -> QUANTIZE to Q4_K_M
        -> EVALUATE vs fp16 baseline
        -> GATE on accuracy drop  (fail loudly if over budget)
        -> SHIP (artifact + eval report)

The point of the gate: compression isn't done when the file is smaller,
it's done when the file is smaller AND you've proven it's still good enough.

This script wires together the helpers shown elsewhere in the chapter:
  - 01_merge_adapter.py
  - 02_convert_to_gguf.sh
  - 03_perplexity_sweep.py
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


class QualityGateError(RuntimeError):
    pass


def run(cmd):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def merge_adapter(base, adapter, out="medical-gemma-merged"):
    run([sys.executable, "01_merge_adapter.py",
         "--base", base, "--adapter", adapter, "--out", out])
    return out


def convert_and_quantize(merged, calib, quant="Q4_K_M"):
    run(["bash", "02_convert_to_gguf.sh", merged, calib, quant])
    return f"medical-gemma-{quant}.gguf"


def task_accuracy(gguf_path, test_set):
    """Thin wrapper around the grader in 03_perplexity_sweep.py."""
    from importlib import import_module
    sweep = import_module("03_perplexity_sweep")  # noqa: keep filename import
    llm = sweep.load(gguf_path)
    return sweep.medical_accuracy(llm, test_set)


def write_report(path, quant, base_acc, quant_acc, drop):
    report = Path("compression_report.md")
    report.write_text(
        f"# Compression Report\n\n"
        f"- Artifact: `{path}`\n"
        f"- Quant: **{quant}**\n"
        f"- fp16 baseline accuracy: **{base_acc:.1%}**\n"
        f"- quantized accuracy: **{quant_acc:.1%}**\n"
        f"- accuracy drop: **{drop:.1%}**\n"
    )
    print(f"Wrote {report}")


def compress(base, adapter, calib, test_set_path,
             quant="Q4_K_M", max_acc_drop=0.02):
    test_set = [json.loads(l) for l in open(test_set_path)]

    merged = merge_adapter(base, adapter)                 # §14.2 (parity inside)
    quantized = convert_and_quantize(merged, calib, quant)  # §14.6

    base_gguf = "medical-gemma-f16.gguf"                  # produced by convert
    base_acc = task_accuracy(base_gguf, test_set)         # fp16 reference
    quant_acc = task_accuracy(quantized, test_set)        # §14.7
    drop = base_acc - quant_acc

    if drop > max_acc_drop:
        raise QualityGateError(
            f"{quant} dropped accuracy by {drop:.1%} "
            f"(budget {max_acc_drop:.1%}). Try a higher bit width "
            f"(e.g. Q5_K_M).")

    write_report(quantized, quant, base_acc, quant_acc, drop)
    print(f"PASS: shipping {quantized} (drop {drop:.1%} within "
          f"{max_acc_drop:.1%} budget).")
    return quantized


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--calib", required=True, help="in-domain text for imatrix")
    ap.add_argument("--test-set", required=True, help="held-out jsonl")
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--max-acc-drop", type=float, default=0.02)
    args = ap.parse_args()

    try:
        compress(args.base, args.adapter, args.calib, args.test_set,
                 args.quant, args.max_acc_drop)
    except QualityGateError as e:
        print(f"QUALITY GATE FAILED: {e}", file=sys.stderr)
        sys.exit(1)
