#!/usr/bin/env bash
# 07_vllm_runtime_lora.sh
# -----------------------------------------------------------------------------
# Serve the UNMODIFIED base + apply the medical-qa LoRA at runtime on vLLM.
#
# Why this exists: a *merged* Gemma 4 E4B currently fails to load on vLLM,
# because a faithful merge drops the KV-shared tail layers' K/V weights and this
# vLLM version doesn't yet implement KV sharing (see the version note in 15.4.1).
# But the STOCK base loads fine, because its checkpoint physically ships those
# weights. So instead of merging, load the base and attach the small adapter at
# request time. You never write a stripped checkpoint, so you never hit the gap.
#
#   pip install vllm
#   bash 07_vllm_runtime_lora.sh
# -----------------------------------------------------------------------------
set -euo pipefail

BASE="${BASE:-google/gemma-4-e4b-it}"
ADAPTER_PATH="${ADAPTER_PATH:-./medical-qa-v1-gemma4-lora}"   # the trained LoRA dir
PORT="${PORT:-8000}"

# --max-lora-rank must be >= the adapter's r (see adapter_config.json). 16 is a
# safe ceiling for the r=4/8/16 adapters in this book; raising it only costs a
# little memory.
vllm serve "$BASE" \
    --enable-lora \
    --lora-modules medical-qa="$ADAPTER_PATH" \
    --max-lora-rank 16 \
    --served-model-name gemma-base \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90

# Route a request to the fine-tune by naming the ADAPTER as the model:
#   curl http://localhost:8000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{"model":"medical-qa",
#          "messages":[{"role":"user","content":"First-line treatments for T2DM?"}]}'
#
# Name "gemma-base" instead to hit the un-adapted base, so you can A/B the two
# and confirm the adapter is actually changing the answers.
#
# Or reuse the chapter's client:
#   python 01_openai_client.py --base http://localhost:8000/v1 --model medical-qa
