#!/usr/bin/env bash
set -euo pipefail

REAL_PYTHON="${RL_KERNEL_REAL_PYTHON:?RL_KERNEL_REAL_PYTHON must name the real Python executable}"

if [[ "${1:-}" == "train.py" || "${1:-}" == */train.py ]]; then
  exec "${REAL_PYTHON}" "$@" \
    --seed 1234 \
    --rollout-seed 42 \
    --vllm-enable-deterministic-inference \
    --vllm-attention-backend flash_attn \
    --vllm-disable-custom-all-reduce \
    --deterministic-mode \
    --accumulate-allreduce-grads-in-fp32
fi

exec "${REAL_PYTHON}" "$@"
