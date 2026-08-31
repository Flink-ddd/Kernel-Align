#!/usr/bin/env bash
set -euo pipefail

REAL_PYTHON="${RL_KERNEL_REAL_PYTHON:?RL_KERNEL_REAL_PYTHON must name the real Python executable}"
RL_KERNEL_ROOT="${RL_KERNEL_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
export RL_KERNEL_ROOT
export PYTHONPATH="${RL_KERNEL_ROOT}:${PYTHONPATH:-}"

if [[ "${1:-}" == "train.py" || "${1:-}" == */train.py ]]; then
  # Ray job submission does not always forward the shell exports used by the
  # outer launcher. A strict linear-logp job must still install the matching
  # vLLM hooks; otherwise it silently falls back to native attention/FFN and
  # loses the R/R performance path. Preserve explicit ablation selections.
  strict_linear_logp=0
  previous_arg=""
  for current_arg in "$@"; do
    if [[ "${previous_arg}" == "--linear-logp-provider-mode" && "${current_arg}" == "strict" ]]; then
      strict_linear_logp=1
      break
    fi
    previous_arg="${current_arg}"
  done
  if [[ "${strict_linear_logp}" == "1" ]]; then
    export RL_KERNEL_VLLM_INTEGRATION="${RL_KERNEL_VLLM_INTEGRATION:-1}"
    export RL_KERNEL_CUDA_ONLY="${RL_KERNEL_CUDA_ONLY:-1}"
    export VIME_RL_KERNEL_STRICT="${VIME_RL_KERNEL_STRICT:-1}"
    export RL_KERNEL_ATTENTION_CASE="${RL_KERNEL_ATTENTION_CASE:-R/R}"
    export RL_KERNEL_FFN_CASE="${RL_KERNEL_FFN_CASE:-R/R}"
    export RL_KERNEL_LOGP_CASE="${RL_KERNEL_LOGP_CASE:-R/R}"
  fi
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
