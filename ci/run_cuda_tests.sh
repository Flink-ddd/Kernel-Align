#!/usr/bin/env bash
# ci/run_cuda_tests.sh — portable CUDA test runner
#
# Runs on any machine that already has CUDA and Python. Invoked three ways:
#
#   1. Self-hosted GitHub/GitLab runner (security-isolated):
#        PR_REPO_URL=https://... PR_SHA=<sha> bash ci/run_cuda_tests.sh
#      The script clones the PR commit into an isolated /tmp workspace so
#      untrusted fork code never executes with host-level privileges.
#
#   2. Hosted CUDA runner (called from ci/run_gpu_ci.sh; the orchestrator
#      uploads this script via scp and sets PR_REPO_URL + PR_SHA in the
#      environment, so the isolation clone below runs on the remote host):
#        PR_REPO_URL=... PR_SHA=... bash /tmp/run_cuda_tests.sh
#
#   3. Local developer machine (no clone — runs in the current working tree):
#        bash ci/run_cuda_tests.sh          # from repo root
#
# Environment variables (all optional):
#   PR_REPO_URL           git remote to clone (skip if unset → use cwd)
#   PR_SHA                commit SHA to detach to after clone
#   GPU_COUNT             number of GPUs for distributed tests (auto-detected)
#   TORCH_CUDA_ARCH_LIST  NVCC arch targets  (default: 8.6)
#   FORCE_CUDA            force CUDA ext build (default: 1)
#   MAX_JOBS              parallel build jobs (default: 8)
#   KERNEL_ALIGN_FORCE_SM90  build SM90/TMA kernels (default: 0)
#   CI_INSTALL_FLASHINFER    install flashinfer before tests (default: 0)
#   CI_UPGRADE_BUILD_TOOLS   upgrade pip/setuptools/wheel first (default: 0)
#   FLASHINFER_WHEEL_INDEX   flashinfer find-links URL
#   PYTEST_ARGS           passed verbatim to pytest

set -euo pipefail

FLASHINFER_WHEEL_INDEX="${FLASHINFER_WHEEL_INDEX:-https://flashinfer.ai/whl/cu124/torch2.4/index.html}"
PYTEST_ARGS="${PYTEST_ARGS:-tests/ rl_engine/tests/ -v}"

# --- Isolation: clone PR commit when coordinates are provided ----------------
if [ -n "${PR_REPO_URL:-}" ] && [ -n "${PR_SHA:-}" ]; then
  CUDA_WORK_DIR="${CUDA_WORK_DIR:-/tmp/rl-kernel-cuda-ci}"
  rm -rf "${CUDA_WORK_DIR}"
  git clone "${PR_REPO_URL}" "${CUDA_WORK_DIR}" \
    || { echo "[cuda-ci] FATAL: git clone failed for ${PR_REPO_URL}"; exit 1; }
  cd "${CUDA_WORK_DIR}"
  git fetch origin "${PR_SHA}" \
    || { echo "[cuda-ci] FATAL: git fetch failed for ${PR_SHA}"; exit 1; }
  git checkout --detach "${PR_SHA}"
  echo "[cuda-ci] Running PR code from ${PR_REPO_URL} @ ${PR_SHA:0:7}"
fi

# --- Interpreter discovery ---------------------------------------------------
PY="${PYTHON:-$(command -v python3.11 || command -v python3 || true)}"
if [ -z "$PY" ]; then
  echo "[cuda-ci] FATAL: python not found in PATH"
  exit 127
fi
if ! "$PY" -c "import torch" >/dev/null 2>&1; then
  for cand in python3.11 python3.10 python3; do
    p=$(command -v "$cand" 2>/dev/null) || continue
    if "$p" -c "import torch" >/dev/null 2>&1; then PY="$p"; break; fi
  done
fi
echo "[cuda-ci] Using interpreter: $PY"

# --- Build env ---------------------------------------------------------------
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export FORCE_CUDA="${FORCE_CUDA:-1}"
export MAX_JOBS="${MAX_JOBS:-8}"
export KERNEL_ALIGN_FORCE_SM90="${KERNEL_ALIGN_FORCE_SM90:-0}"

PIP_INSTALL_ARGS=(--timeout 60 --retries 5)

if [ "${CI_UPGRADE_BUILD_TOOLS:-0}" = "1" ]; then
  "$PY" -m pip install "${PIP_INSTALL_ARGS[@]}" -U pip setuptools wheel
fi

if [ "${CI_INSTALL_FLASHINFER:-0}" = "1" ]; then
  "$PY" -m pip install "${PIP_INSTALL_ARGS[@]}" flashinfer-python -f "$FLASHINFER_WHEEL_INDEX"
  "$PY" -m pip install "${PIP_INSTALL_ARGS[@]}" --no-build-isolation -e ".[cuda,test,hf]"
else
  "$PY" -m pip install "${PIP_INSTALL_ARGS[@]}" nvidia-ml-py
  "$PY" -m pip install "${PIP_INSTALL_ARGS[@]}" --no-build-isolation -e ".[test,hf]"
fi

# --- Hardware / dispatch preflight ------------------------------------------
nvidia-smi
"$PY" - <<'PY'
import sys
import torch
from rl_engine.kernels.registry import kernel_registry

if not torch.cuda.is_available():
    raise SystemExit("[cuda-ci] FATAL: CUDA is unavailable after installation")

print(
    f"[cuda-ci] python={sys.version.split()[0]} "
    f"torch={torch.__version__} torch_cuda={torch.version.cuda}"
)
op = kernel_registry.get_op("logp")
backend = op.__class__.__name__
print(f"[cuda-ci] logp backend={backend}")
if not backend.startswith("FusedLogp"):
    raise SystemExit(
        "[cuda-ci] strict fused logp preflight failed: "
        f"dispatch selected {backend}, not a FusedLogp backend"
    )
PY

"$PY" examples/grpo_single_gpu.py \
  --device cuda \
  --require-fused-logp \
  --steps 2 \
  --num-prompts 1 \
  --samples-per-prompt 2 \
  --prompt-len 2 \
  --completion-len 3 \
  --vocab-size 16 \
  --hidden-dim 8

# --- NCCL smoke test (multi-GPU only) ----------------------------------------
GPU_COUNT="${GPU_COUNT:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}"
if [ "${GPU_COUNT}" -gt 1 ]; then
  cat >/tmp/rl_kernel_nccl_smoke.py <<'PY'
import os
import torch
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
world_size = dist.get_world_size()
device = torch.device("cuda", local_rank)
value = torch.tensor([local_rank + 1], device=device, dtype=torch.float32)
dist.all_reduce(value, op=dist.ReduceOp.SUM)
expected = world_size * (world_size + 1) / 2
if value.item() != expected:
    raise SystemExit(f"unexpected all-reduce value: got {value.item()}, expected {expected}")
if dist.get_rank() == 0:
    print(f"[cuda-ci] NCCL all-reduce smoke passed on {world_size} GPUs")
dist.destroy_process_group()
PY
  "$PY" -m torch.distributed.run --nproc_per_node="$GPU_COUNT" /tmp/rl_kernel_nccl_smoke.py
fi

# --- Test suite --------------------------------------------------------------
# PYTEST_ARGS is intentionally word-split here for multi-arg support.
# shellcheck disable=SC2086
"$PY" -m pytest $PYTEST_ARGS
