#!/usr/bin/env bash
set -euo pipefail

export PPU_SDK=/usr/local/PPU_SDK
export CUDA_PATH=$PPU_SDK/CUDA_SDK
export CUDA_HOME=$CUDA_PATH
export PATH=$PPU_SDK/bin:$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$PPU_SDK/lib:$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}

MODEL_NAME="${MODEL_NAME:-/mnt/ws_nas/models/Qwen3.8-27B}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8002}"

CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

API_KEY="${API_KEY:-EMPTY}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

echo "============================================================"
echo "Starting Qwen with vLLM"
echo "MODEL_NAME=${MODEL_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "TP_SIZE=${TP_SIZE}"
echo "HOST=${HOST}"
echo "PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "============================================================"

exec vllm serve "${MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --dtype auto \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --api-key "${API_KEY}" \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --enforce-eager