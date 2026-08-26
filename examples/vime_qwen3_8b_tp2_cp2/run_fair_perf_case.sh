#!/usr/bin/env bash
set -euo pipefail

REQUESTED_MODE="${1:-}"
case "${REQUESTED_MODE}" in
  strict|audit|auto|off|pp|pp-aligned|rr|rr-aligned) ;;
  *)
    echo "usage: $0 {strict|audit|auto|off|pp|pp-aligned|rr|rr-aligned}" >&2
    exit 2
    ;;
esac

MODE="${REQUESTED_MODE}"
case_id=R/R
aligned=1
provider_mode=strict
case "${REQUESTED_MODE}" in
  strict) MODE=strict ;;
  audit) MODE=audit ;;
  auto) MODE=auto; case_id=P/P; provider_mode=auto ;;
  off) MODE=off; case_id=P/P; provider_mode=auto ;;
  rr-aligned) MODE=strict ;;
  rr) MODE=strict; aligned=0 ;;
  pp-aligned) MODE=off; case_id=P/P; provider_mode=auto ;;
  pp) MODE=off; case_id=P/P; provider_mode=auto; aligned=0 ;;
esac

VIME_ROOT="${VIME_ROOT:-/home/ellm/ljj/vime-debug-main}"
RL_KERNEL_ROOT="${RL_KERNEL_ROOT:-/home/ellm/ljj/RL-Kernel-issue335-fix}"
PYTHON_BIN="${PYTHON_BIN:-/home/ellm/workspace/ljj/.conda/envs/rlk-attention-engines/bin/python}"
RAY_BIN="${RAY_BIN:-$(dirname "${PYTHON_BIN}")/ray}"
TE_SITE="${TE_SITE:-/home/ellm/ljj/.te-2.11.0-site-packages}"
CUDA_PYTHON_SITE="${CUDA_PYTHON_SITE:-/home/ellm/ljj/.cuda-python-12.8-site-packages}"
CUDA12_COMPAT_LIB="${CUDA12_COMPAT_LIB:-/home/ellm/ljj/.cuda12-compat-lib}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="${RUN_ID:-fair-${REQUESTED_MODE}-${STAMP}}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${VIME_ROOT}/outputs/rlkernel/${RUN_ID}}"
TRACE_MODE="${TRACE_MODE:-0}"

NVIDIA_LIBS="$(find "$(dirname "$(dirname "${PYTHON_BIN}")")/lib/python3.11/site-packages/nvidia" -type d -name lib -print 2>/dev/null | paste -sd: -)"
export PYTHONPATH="${CUDA_PYTHON_SITE}:${TE_SITE}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CUDA12_COMPAT_LIB}:${TE_SITE}/transformer_engine/wheel_lib${NVIDIA_LIBS:+:${NVIDIA_LIBS}}:${LD_LIBRARY_PATH:-}"

cp_comm_type=all_gather
ci_test=1
validate_artifacts="${VALIDATE_ARTIFACTS:-1}"
if [[ "${case_id}" == P/P ]]; then
  cp_comm_type=p2p
  ci_test=0
  validate_artifacts=0
fi
if [[ "${MODE}" == audit ]]; then
  validate_artifacts=0
fi
launcher="${VIME_ROOT}/scripts/codex-debug-qwen3-8B-production-pp-tp2-cp2.sh"
run_python="${RL_KERNEL_ROOT}/examples/vime_qwen3_8b_tp2_cp2/aligned_python_entrypoint.sh"
[[ -x "${run_python}" ]] || {
  echo "RL-Kernel Python entrypoint is not executable: ${run_python}" >&2
  exit 3
}
adapter_env=(
  RL_KERNEL_REAL_PYTHON="${PYTHON_BIN}"
  RL_KERNEL_MODE="${MODE}"
  RL_KERNEL_ALIGNED="${aligned}"
)
if [[ "${aligned}" == 1 ]]; then
  adapter_env+=(
    VLLM_BATCH_INVARIANT=1
    NCCL_ALGO=Ring
    NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
    CUBLAS_WORKSPACE_CONFIG=:16:8
    CUBLASLT_WORKSPACE_SIZE=1
  )
fi
if [[ "${MODE}" == audit ]]; then
  adapter_env+=(RL_KERNEL_ROUTE_REPORT_ALL_RANKS=1)
fi

mkdir -p "${ARTIFACT_DIR}"
run_case=(env "${adapter_env[@]}" \
  RL_KERNEL_MODE="${MODE}" \
  RL_KERNEL_ROOT="${RL_KERNEL_ROOT}" \
  PYTHON_BIN="${run_python}" \
  RAY_BIN="${RAY_BIN}" \
  RL_KERNEL_RUN_ID="${RUN_ID}" \
  RL_KERNEL_ARTIFACT_DIR="${ARTIFACT_DIR}" \
  RAY_TEMP_DIR="/tmp/rlk-fair-${REQUESTED_MODE}" \
  VIME_CKPT="${ARTIFACT_DIR}/unused-checkpoint" \
  RL_KERNEL_ATTENTION_CASE="${case_id}" \
  RL_KERNEL_FFN_CASE="${case_id}" \
  RL_KERNEL_LOGP_CASE="${case_id}" \
  SELECTED_LOGPROB_PROVIDER_MODE="${provider_mode}" \
  TRANSFORMER_IMPL=transformer_engine \
  ATTENTION_BACKEND="${ATTENTION_BACKEND:-auto}" \
  VLLM_ENFORCE_EAGER=0 \
  TP_SIZE=2 \
  CP_SIZE=2 \
  CP_COMM_TYPE="${cp_comm_type}" \
  ACTOR_GPUS=4 \
  ROLLOUT_GPUS=4 \
  NUM_GPUS=8 \
  ROLLOUT_GPUS_PER_ENGINE=2 \
  ROLLOUT_BATCH_SIZE=1 \
  N_SAMPLES_PER_PROMPT=2 \
  MAX_RESPONSE_LEN=128 \
  GLOBAL_BATCH_SIZE=2 \
  MAX_TOKENS_PER_GPU=2048 \
  NUM_ROLLOUT="${NUM_ROLLOUT:-3}" \
  CI_TEST="${ci_test}" \
  SAVE_OPTIM=0 \
  SAVE_CHECKPOINT=0 \
  VALIDATE_ARTIFACTS=0 \
  TRACE_MODE=0 \
  bash "${launcher}")

if [[ "${TRACE_MODE}" == 0 ]]; then
  "${run_case[@]}" >"${ARTIFACT_DIR}/run.log" 2>&1
elif [[ "${TRACE_MODE}" == 1 ]]; then
  command -v nsys >/dev/null || {
    echo "TRACE_MODE=1 requires nsys" >&2
    exit 3
  }
  trace_base="${ARTIFACT_DIR}/trace/full-stack"
  session="rlkfair${REQUESTED_MODE//-/}$$"
  launch_pid=""
  cleanup_trace() {
    nsys stop --session="${session}" >/dev/null 2>&1 || true
    [[ -z "${launch_pid}" ]] || kill "${launch_pid}" >/dev/null 2>&1 || true
  }
  mkdir -p "${ARTIFACT_DIR}/trace"
  trap cleanup_trace EXIT
  nsys launch \
    --session-new="${session}" \
    --trace=cuda,nvtx \
    --cuda-graph-trace=node \
    --trace-fork-before-exec=true \
    --wait=primary \
    --show-output=true \
    "${run_case[@]}" >"${ARTIFACT_DIR}/run.log" 2>&1 &
  launch_pid=$!
  sleep 1
  nsys start \
    --session="${session}" \
    --sample=none \
    --cpuctxsw=none \
    --force-overwrite=true \
    --output="${trace_base}"
  set +e
  wait "${launch_pid}"
  run_status=$?
  set -e
  launch_pid=""
  nsys stop --session="${session}" || true
  trap - EXIT
  (( run_status == 0 )) || exit "${run_status}"
  nsys stats --report cuda_gpu_kern_sum --format csv \
    "${trace_base}.nsys-rep" >"${trace_base}-kernels.csv"
  grep -Eq '^"?[0-9]+([.][0-9]+)?"?,' "${trace_base}-kernels.csv" || {
    echo "Nsight trace contains no CUDA kernel data" >&2
    exit 4
  }
else
  echo "TRACE_MODE must be 0 or 1" >&2
  exit 2
fi

if [[ "${validate_artifacts}" == 1 ]]; then
  "${PYTHON_BIN}" \
    "${RL_KERNEL_ROOT}/examples/vime_qwen3_8b_tp2_cp2/validate_artifacts.py" \
    --readback-dir "${ARTIFACT_DIR}/readbacks" \
    --train-data-dir "${ARTIFACT_DIR}/train-data" \
    --output "${ARTIFACT_DIR}/bitwise-validation.json" \
    --runtime-ci-zero-threshold-passed \
    >>"${ARTIFACT_DIR}/run.log"
fi

printf 'MODE=%s\nREQUESTED_MODE=%s\nRUN_ID=%s\nARTIFACT_DIR=%s\n' \
  "${MODE}" "${REQUESTED_MODE}" "${RUN_ID}" "${ARTIFACT_DIR}"
