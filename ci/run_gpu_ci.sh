#!/usr/bin/env bash
set -uo pipefail

# TP=2 (override via env; GPU_ID/GPU_COUNT accepted as matrix-friendly aliases)
PRIMARY_GPU_ID="${PRIMARY_GPU_ID:-${GPU_ID:-NVIDIA RTX A4000}}"
PRIMARY_GPU_COUNT="${PRIMARY_GPU_COUNT:-${GPU_COUNT:-2}}"

# TP=1
FALLBACK_GPU_ID="${FALLBACK_GPU_ID:-NVIDIA A40}"
FALLBACK_GPU_COUNT="${FALLBACK_GPU_COUNT:-1}"

# Optional arch override; asserted against the pod's real cap in the remote build so a
# cross-arch resource fallback cannot build mismatched SASS.
TARGET_SM="${TARGET_SM:-}"

# Forwarded to the remote build; setup.py compiles the Hopper (sm90) kernels only when "1".
KERNEL_ALIGN_FORCE_SM90="${KERNEL_ALIGN_FORCE_SM90:-}"

CI_IMAGE="${CI_IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
DISK_GB=40
PR_SHA="${PR_SHA:-$(date +%s)}"
POD_NAME="rl-kernel-ci-${PR_SHA:0:7}"
READY_RETRIES=60

POD_ID=""

cleanup() {
  trap - EXIT INT TERM

  if [ -n "$POD_ID" ]; then
    echo ""
    echo "[ci] ========================================================"
    echo "[ci] === AUTOMATIC CLEANUP: Removing pod $POD_ID ==="
    echo "[ci] ========================================================"

    REMOVE_OUT=$(runpodctl pod remove "$POD_ID" 2>&1)
    if echo "$REMOVE_OUT" | grep -qi "not found"; then
      echo "[ci] Pod $POD_ID was already cleared from the cloud. Safe to exit."
    else
      echo "$REMOVE_OUT"
    fi
  fi
}
trap cleanup EXIT INT TERM

GPU_ID=$PRIMARY_GPU_ID
GPU_COUNT=$PRIMARY_GPU_COUNT

echo "[ci] Attempt 1: create pod: ${GPU_COUNT}x ${GPU_ID}"
CREATE_OUT=$(runpodctl pod create \
  --name "$POD_NAME" \
  --gpu-id "$GPU_ID" \
  --gpu-count "$GPU_COUNT" \
  --image "$CI_IMAGE" \
  --container-disk-in-gb "$DISK_GB" \
  --cloud-type SECURE \
  --ports "22/tcp" 2>&1)

# Fallback 触发
if echo "$CREATE_OUT" | grep -qi "no longer any instances available"; then
  echo "[ci] WARN: ${GPU_COUNT}x ${GPU_ID} sold out! Triggering elastic Fallback..."

  GPU_ID=$FALLBACK_GPU_ID
  GPU_COUNT=$FALLBACK_GPU_COUNT

  echo "[ci] Attempt 2 (Fallback): create pod: ${GPU_COUNT}x ${GPU_ID}"
  CREATE_OUT=$(runpodctl pod create \
    --name "$POD_NAME" \
    --gpu-id "$GPU_ID" \
    --gpu-count "$GPU_COUNT" \
    --image "$CI_IMAGE" \
    --container-disk-in-gb "$DISK_GB" \
    --cloud-type SECURE \
    --ports "22/tcp" 2>&1)

  if echo "$CREATE_OUT" | grep -qi "no longer any instances available"; then
    echo "[ci] FATAL: Alternatives (${GPU_COUNT}x ${GPU_ID}) have also been exhausted. Please try CI again later."
    exit 1
  fi
fi

POD_ID=$(echo "$CREATE_OUT" | grep -oE '"id":\s*"[a-z0-9]{8,}"' | cut -d '"' -f4 | head -1)
if [ -z "$POD_ID" ]; then
  POD_ID=$(echo "$CREATE_OUT" | grep -oE '"[a-z0-9]{8,}"' | tr -d '"' | head -1)
fi

if [ -z "$POD_ID" ]; then
  echo "[ci] ERROR: Unable to resolve pod id. Output: $CREATE_OUT"
  exit 1
fi
echo "[ci] Successfully rented pod: $POD_ID"

echo "[ci] Waiting for pod network infrastructure to be fully ready..."
SSH_IP=""
SSH_PORT=""

for i in $(seq 1 "$READY_RETRIES"); do
  POD_INFO=$(runpodctl pod get "$POD_ID" -o json)

  SSH_IP=$(echo "$POD_INFO" | grep -iE '"ip"|"publicIp"|"address"' | grep -oE '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}' | head -1 || true)
  SSH_PORT=$(echo "$POD_INFO" | grep -iE '"port"|"externalPort"|"publicPort"' | grep -oE '[0-9]+' | grep -v '^22$' | head -1 || true)

  if [ -n "$SSH_IP" ] && [ -n "$SSH_PORT" ] && ! echo "$POD_INFO" | grep -qi "not ready"; then
    echo "[ci] Pod infrastructure is 100% READY!"
    break
  fi

  if [ "$i" -eq "$READY_RETRIES" ]; then
    echo "[ci] ERROR: Pod network/SSH infrastructure initialization timed out."
    exit 1
  fi

  echo "[ci] Pod layer status: RUNNING, but network routing is initializing... waiting 10s (Attempt $i/$READY_RETRIES)"
  sleep 10
done

echo "[ci] Target Establish -> root@$SSH_IP:$SSH_PORT"

SSH_OPTIONS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p $SSH_PORT"

if [ "${GPU_COUNT}" -gt 1 ]; then
  TEST_CMD='"$PY" -m torch.distributed.run --nproc_per_node='"${GPU_COUNT}"' -m pytest tests/ -v'
else
  TEST_CMD='"$PY" -m pytest tests/ -v'
fi

REMOTE_CMD='set -e
PY=$(command -v python3.11 || command -v python3)
if [ -z "$PY" ]; then echo "[remote] FATAL: python not found in PATH"; exit 127; fi
if ! "$PY" -c "import torch" >/dev/null 2>&1; then
  for cand in python3.11 python3.10 python3; do
    p=$(command -v "$cand" 2>/dev/null) || continue
    if "$p" -c "import torch" >/dev/null 2>&1; then PY="$p"; break; fi
  done
fi
echo "[remote] Using interpreter: $PY"
export FORCE_CUDA=1
export MAX_JOBS=8
export KERNEL_ALIGN_FORCE_SM90="'"${KERNEL_ALIGN_FORCE_SM90}"'"

# normalize_sm: compact (90) or dotted (9.0) compute cap -> torch dotted form, keeping +PTX.
normalize_sm() {
  sm_in="$1"; sm_ptx=""
  case "$sm_in" in *+PTX) sm_ptx="+PTX"; sm_in="${sm_in%+PTX}";; esac
  case "$sm_in" in
    *.*) : ;;
    [0-9][0-9]|[0-9][0-9][0-9]) sm_major="${sm_in%?}"; sm_in="${sm_major}.${sm_in#$sm_major}" ;;
    *) return 1 ;;
  esac
  case "$sm_in" in [0-9]*.[0-9]|[0-9]*.[0-9][0-9]) echo "${sm_in}${sm_ptx}" ;; *) return 1 ;; esac
}

ACTUAL_SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d "[:space:]")
[ -z "$ACTUAL_SM" ] && ACTUAL_SM=$("$PY" -c "import torch;a,b=torch.cuda.get_device_capability();print(f\"{a}.{b}\")" 2>/dev/null || true)
[ -z "$ACTUAL_SM" ] && { echo "[remote] FATAL: cannot determine GPU compute capability"; exit 3; }

REQUESTED_SM="'"${TARGET_SM}"'"
if [ -n "$REQUESTED_SM" ]; then
  NORM_REQ=$(normalize_sm "$REQUESTED_SM") || { echo "[remote] FATAL: unsupported TARGET_SM=$REQUESTED_SM"; exit 3; }
  NORM_REQ_BASE="${NORM_REQ%+PTX}"
  if [ "$NORM_REQ_BASE" != "$ACTUAL_SM" ]; then
    echo "[remote] FATAL: requested TARGET_SM=$REQUESTED_SM (sm_$NORM_REQ_BASE) but provisioned GPU is sm_$ACTUAL_SM."
    echo "[remote]        Refusing to build mismatched kernels (likely a cross-arch resource fallback)."
    exit 3
  fi
  BUILD_SM="$NORM_REQ_BASE"
else
  BUILD_SM=$(normalize_sm "$ACTUAL_SM") || { echo "[remote] FATAL: unsupported detected arch $ACTUAL_SM"; exit 3; }
fi
# BUILD_SM is always bare here (both paths strip +PTX); +PTX gives forward-compat JIT.
export TORCH_CUDA_ARCH_LIST="${BUILD_SM}+PTX"
echo "[remote] Detected GPU sm_$ACTUAL_SM; building _C for TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

cd /workspace
git clone '"${PR_REPO_URL:-https://github.com/RL-Align/RL-Kernel.git}"' repo
cd repo
git fetch origin '"${PR_SHA}"'
git checkout --detach '"${PR_SHA}"'
"$PY" -c "import torch;print(f\"[remote] image torch {torch.__version__} cuda {torch.version.cuda}\")"
# Pin torch (cu124, matching the CI image) so the extension is built against the exact
# runtime torch, not a non-deterministic bare-install upgrade of the 2.4.0 in the image.
TORCH_SPEC="${TORCH_SPEC:-torch==2.4.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
"$PY" -m pip install --no-cache-dir "$TORCH_SPEC" --index-url "$TORCH_INDEX_URL"
"$PY" -c "import torch;print(f\"[remote] pinned torch {torch.__version__} cuda {torch.version.cuda}\")"
# --no-build-isolation: torch must be visible to setup.py, else the extension is silently skipped.
# --no-deps: keep the pinned torch; do not let the editable install re-resolve it.
"$PY" -m pip install --no-build-isolation --no-deps -e .
"$PY" -m pip install --no-cache-dir numpy tabulate accelerate "transformers==5.13.1" pytest
nvidia-smi
# Fail fast if _C did not build or cannot launch, instead of silently using native fallbacks.
"$PY" scripts/ci_smoke.py
# Enforce _C in the pytest suite too (test_extension_smoke.py skips unless this is set).
export RL_KERNEL_REQUIRE_EXT=1
'"${TEST_CMD}"

echo "[ci] Launching remote test suite on GPU pod (Distributed Execution Mode: TP=${GPU_COUNT})..."
ssh $SSH_OPTIONS root@"$SSH_IP" "bash -lc '$REMOTE_CMD'"
TEST_EXIT=$?

echo "[ci] Remote execution finished with exit code = $TEST_EXIT"
exit $TEST_EXIT
