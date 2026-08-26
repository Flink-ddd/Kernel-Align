#!/usr/bin/env bash
set -euo pipefail

REAL_PYTHON="${RL_KERNEL_REAL_PYTHON:?RL_KERNEL_REAL_PYTHON must name the real Python executable}"
MODE="${RL_KERNEL_MODE:-strict}"
ALIGNED="${RL_KERNEL_ALIGNED:-0}"

case "${MODE}" in
  strict|audit|auto|off) ;;
  *)
    echo "RL_KERNEL_MODE must be strict, audit, auto, or off; got ${MODE}" >&2
    exit 2
    ;;
esac

if [[ "${1:-}" == "train.py" || "${1:-}" == */train.py ]]; then
  args=()
  while (( $# )); do
    case "$1" in
      --selected-logprob-provider|--selected-logprob-provider-mode|--custom-megatron-init-path)
        if (( $# < 2 )); then
          echo "$1 requires a value" >&2
          exit 2
        fi
        if [[ "${MODE}" != off ]]; then
          args+=("$1")
          if [[ "$1" == --selected-logprob-provider ]]; then
            args+=("rl_engine.integrations.vime.linear_logp.provider")
          else
            args+=("$2")
          fi
        fi
        shift 2
        ;;
      --selected-logprob-provider=*|--selected-logprob-provider-mode=*|--custom-megatron-init-path=*)
        if [[ "${MODE}" != off ]]; then
          case "$1" in
            --selected-logprob-provider=*)
              args+=("--selected-logprob-provider=rl_engine.integrations.vime.linear_logp.provider")
              ;;
            *) args+=("$1") ;;
          esac
        fi
        shift
        ;;
      *) args+=("$1"); shift ;;
    esac
  done

  if [[ "${ALIGNED}" == 1 ]]; then
    args+=(
      --seed 1234
      --rollout-seed 42
      --vllm-enable-deterministic-inference
      --vllm-attention-backend flash_attn
      --vllm-disable-custom-all-reduce
      --deterministic-mode
      --accumulate-allreduce-grads-in-fp32
    )
  fi

  case "${MODE}" in
    strict|audit)
      export VIME_RL_KERNEL_STRICT=1
      ;;
    auto)
      export VIME_RL_KERNEL_STRICT=0
      ;;
    off)
      exec env \
        -u RL_KERNEL_VLLM_INTEGRATION \
        -u VIME_RL_KERNEL_STRICT \
        -u RL_KERNEL_DET_GEMM_SM90_ONLY \
        -u RL_KERNEL_CUDA_ONLY \
        "${REAL_PYTHON}" "${args[@]}"
      ;;
  esac
  exec "${REAL_PYTHON}" "${args[@]}"
fi

exec "${REAL_PYTHON}" "$@"
