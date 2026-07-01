#!/usr/bin/env bash
set -euo pipefail

# When invoked from GitHub Actions (pull_request_target), the workflow checks
# out the base branch (trusted CI scripts) and passes the fork's coordinates
# via PR_REPO_URL + PR_SHA.  We clone the PR code into an isolated /tmp
# workspace here so that untrusted fork code never executes on the self-hosted
# runner with elevated host privileges.
if [ -n "${PR_REPO_URL:-}" ] && [ -n "${PR_SHA:-}" ]; then
  ROCM_WORK_DIR="${ROCM_WORK_DIR:-/tmp/rl-kernel-rocm-ci}"
  rm -rf "${ROCM_WORK_DIR}"
  git clone "${PR_REPO_URL}" "${ROCM_WORK_DIR}" \
    || { echo "[rocm-ci] FATAL: git clone failed for ${PR_REPO_URL}"; exit 1; }
  cd "${ROCM_WORK_DIR}"
  git fetch origin "${PR_SHA}" \
    || { echo "[rocm-ci] FATAL: git fetch failed for ${PR_SHA}"; exit 1; }
  git checkout --detach "${PR_SHA}"
  echo "[rocm-ci] Running PR code from ${PR_REPO_URL} @ ${PR_SHA:0:7}"
fi

PY="${PYTHON:-$(command -v python3 || command -v python || true)}"
if [ -z "$PY" ]; then
  echo "[rocm-ci] FATAL: python not found in PATH"
  exit 127
fi

export MAX_JOBS="${MAX_JOBS:-8}"

echo "[rocm-ci] Using interpreter: $PY"
"$PY" - <<'PY'
import os
import sys

import torch

print("python:", sys.version.replace("\n", " "))
print("torch:", torch.__version__)
print("torch hip:", torch.version.hip)
print("PYTORCH_ROCM_ARCH:", os.environ.get("PYTORCH_ROCM_ARCH", "(unset)"))
print("MAX_JOBS:", os.environ.get("MAX_JOBS", "(unset)"))
print("cuda api available:", torch.cuda.is_available())
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        gcn = getattr(props, "gcnArchName", "unknown")
        print("device", index, torch.cuda.get_device_name(index), "gcn:", gcn)
if torch.version.hip is None:
    raise SystemExit("[rocm-ci] FATAL: PyTorch is not a ROCm build")
if not torch.cuda.is_available():
    raise SystemExit("[rocm-ci] FATAL: ROCm device is not visible to PyTorch")
PY

ATTN_BACKEND="${RL_KERNEL_ROCM_ATTN_BACKEND:-sdpa}"
FLASH_AUTO_INSTALL="${RL_KERNEL_ROCM_FLASH_ATTN_AUTO_INSTALL:-1}"
PYTEST_ARGS="${PYTEST_ARGS:-rl_engine/tests/test_dispatch.py tests/test_kernel_registry.py tests/test_attention_correctness.py tests/test_linear_logp.py tests/test_ratio_kl.py -q -rs}"
PIP_INSTALL_ARGS=(--timeout 60 --retries 5)

"$PY" -m pip install "${PIP_INSTALL_ARGS[@]}" -U pip setuptools wheel
"$PY" -m pip install "${PIP_INSTALL_ARGS[@]}" --no-build-isolation -e ".[test]"

case "${ATTN_BACKEND}" in
  flash_attn|flash-attn|flash_attention)
    "$PY" -m pip install "${PIP_INSTALL_ARGS[@]}" ninja packaging wheel psutil einops
    if ! "$PY" - <<'PY'
import os

os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"
try:
    from flash_attn import flash_attn_func
except Exception as exc:
    raise SystemExit(f"missing flash-attn ROCm backend: {exc}")
if flash_attn_func is None:
    raise SystemExit("missing flash_attn_func")
PY
    then
      if [ "${FLASH_AUTO_INSTALL}" != "1" ]; then
        echo "[rocm-ci] FATAL: flash-attn is unavailable and auto-install is disabled"
        exit 1
      fi
      FLASH_ATTN_REF="${RL_KERNEL_FLASH_ATTN_REF:-v2.8.3}"
      FLASH_ATTN_DIR="${RL_KERNEL_FLASH_ATTN_DIR:-/tmp/rl-kernel-flash-attention}"
      rm -rf "${FLASH_ATTN_DIR}"
      git clone --depth 1 --branch "${FLASH_ATTN_REF}" --recurse-submodules \
        https://github.com/Dao-AILab/flash-attention.git "${FLASH_ATTN_DIR}"
      FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
        "$PY" -m pip install "${PIP_INSTALL_ARGS[@]}" --no-build-isolation --no-deps "${FLASH_ATTN_DIR}"
    fi
    "$PY" scripts/check_rocm_env.py
    ;;
  native|pytorch|sdpa)
    echo "[rocm-ci] Using PyTorch SDPA ROCm attention fallback"
    ;;
  *)
    echo "[rocm-ci] FATAL: unsupported RL_KERNEL_ROCM_ATTN_BACKEND=${ATTN_BACKEND}"
    exit 1
    ;;
esac

export RL_KERNEL_ROCM_ATTN_BACKEND="${ATTN_BACKEND}"
"$PY" - <<'PY'
from rl_engine.kernels.registry import kernel_registry

for op_name in ("logp", "attn", "linear_logp", "ratio_kl"):
    op = kernel_registry.get_op(op_name)
    print(f"[rocm-ci] backend {op_name}: {op.__class__.__name__}")
PY

# PYTEST_ARGS is intentionally split into pytest argv for local override support.
# shellcheck disable=SC2086
"$PY" -m pytest $PYTEST_ARGS
