#!/usr/bin/env bash
# 02_convert_to_gguf.sh
# ---------------------
# Convert a merged HF model to GGUF and quantize it to Q4_K_M, the
# portable path that runs on CPU, Apple Silicon, and GPU via llama.cpp.
#
# Steps:
#   1. (one-time) clone + set up llama.cpp tooling
#   2. HF model  -> 16-bit GGUF (translation only, no compression)
#   3. (optional) build an importance matrix on in-domain text
#   4. 16-bit GGUF -> 4-bit Q4_K_M (this is what actually shrinks it)
#   5. verify with a quick generation
#
# Usage:
#   ./02_convert_to_gguf.sh ./medical-gemma-merged medical_calibration.txt

set -euo pipefail

MERGED="${1:?path to merged HF model dir}"
CALIB="${2:-}"            # optional in-domain text file for imatrix
QUANT="${3:-Q4_K_M}"
LLAMA_DIR="${LLAMA_DIR:-./llama.cpp}"

# 1. One-time setup -----------------------------------------------------
if [ ! -d "$LLAMA_DIR" ]; then
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
  cmake -B "$LLAMA_DIR/build" -S "$LLAMA_DIR"
  cmake --build "$LLAMA_DIR/build" --config Release -j
  pip install -r "$LLAMA_DIR/requirements.txt"
fi

F16="medical-gemma-f16.gguf"
OUT="medical-gemma-${QUANT}.gguf"

# 2. HF -> 16-bit GGUF (no compression) ---------------------------------
python "$LLAMA_DIR/convert_hf_to_gguf.py" "$MERGED" \
    --outfile "$F16" --outtype f16

# 3. Optional importance matrix (recommended for <=4-bit) ---------------
IMATRIX_ARGS=()
if [ -n "$CALIB" ] && [ -f "$CALIB" ]; then
  "$LLAMA_DIR/build/bin/llama-imatrix" \
      -m "$F16" -f "$CALIB" -o medical.imatrix
  IMATRIX_ARGS=(--imatrix medical.imatrix)
fi

# 4. Quantize -> Q4_K_M (the step that shrinks ~8GB -> ~2.5GB) ----------
"$LLAMA_DIR/build/bin/llama-quantize" \
    "${IMATRIX_ARGS[@]}" "$F16" "$OUT" "$QUANT"

# 5. Verify -------------------------------------------------------------
echo "=== sanity generation ==="
"$LLAMA_DIR/build/bin/llama-cli" -m "$OUT" -n 120 \
    -p "What are the first-line treatments for type 2 diabetes?"

ls -lh "$F16" "$OUT"
echo "Done: $OUT"
