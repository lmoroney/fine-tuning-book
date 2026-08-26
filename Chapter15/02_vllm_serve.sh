#!/usr/bin/env bash
# 02_vllm_serve.sh
# -----------------------------------------------------------------------------
# Serve your merged expert model with vLLM as an OpenAI-compatible API.
#
# vLLM is the throughput play: continuous batching + PagedAttention (Chapter 14)
# let one GPU serve many concurrent users far more efficiently than a naive
# loop. Use this when you are serving a product, not a single developer.
#
#   pip install vllm
#   bash 02_vllm_serve.sh
#
# Serve the MERGED bf16 model for maximum quality on a GPU with room, or point
# MODEL at a GPTQ / compressed-tensors build to fit a smaller card (see §15.4).
# -----------------------------------------------------------------------------
set -euo pipefail

MODEL="${MODEL:-./medical-gemma-merged}"     # merged fp16 dir from Chapter 14
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-8192}"                    # max context; sizes the KV cache

vllm serve "$MODEL" \
    --served-model-name medical-gemma \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16

# The server now speaks the same /v1 contract as Ollama and OpenAI:
#   curl http://localhost:8000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{"model":"medical-gemma",
#          "messages":[{"role":"user","content":"Explain beta blockers."}]}'
#
# Point 01_openai_client.py at it with:
#   python 01_openai_client.py --base http://localhost:8000/v1 --model medical-gemma
