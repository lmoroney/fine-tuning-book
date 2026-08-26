#!/usr/bin/env bash
# 03_vllm_multi_lora.sh
# -----------------------------------------------------------------------------
# Serve MANY experts from ONE base model. This is the callback to Chapter 14:
# when you have several fine-tunes of the same base (one adapter per specialty
# or per customer), do NOT merge and ship N full copies. Keep one base in GPU
# memory and hot-swap small LoRA adapters per request.
#
#   bash 03_vllm_multi_lora.sh
#
# Each request names which adapter it wants in the "model" field. vLLM applies
# the right adapter on the fly. Storage and memory cost scales with the base
# once, plus a few tens of MB per adapter, instead of a full model per expert.
# -----------------------------------------------------------------------------
set -euo pipefail

BASE="${BASE:-google/gemma-4-e4b-it}"
PORT="${PORT:-8000}"

# Register each adapter as name=path. These are the LoRA dirs from earlier
# chapters; they must have been trained on the SAME base you serve here.
vllm serve "$BASE" \
    --enable-lora \
    --max-loras 4 \
    --max-lora-rank 16 \
    --lora-modules \
        cardiology=./adapters/cardiology-lora \
        oncology=./adapters/oncology-lora \
        general-med=./medical-qa-v1-gemma4-lora \
    --host 0.0.0.0 --port "$PORT" --max-model-len 8192

# Route a request to a specific expert by naming the adapter as the model:
#
#   curl http://localhost:8000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{"model":"oncology",
#          "messages":[{"role":"user","content":"Stage a T2N1M0 tumor."}]}'
#
# Switch "oncology" to "cardiology" and the next request uses a different
# adapter over the exact same base weights, with no reload.
