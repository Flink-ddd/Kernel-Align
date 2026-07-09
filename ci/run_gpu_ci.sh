#!/usr/bin/env bash
# ci/run_gpu_ci.sh — ephemeral RunPod CUDA runner orchestration
#
# Starts a temporary RunPod pod, runs ci/run_cuda_tests.sh remotely, then
# removes the pod. All test logic lives in run_cuda_tests.sh, which can also
# be executed directly on any CUDA machine.
#
# Requires an authenticated runpodctl and an SSH key registered with RunPod.
#
# The GitHub workflow overrides GPU selection per matrix row via env vars;
# defaults below work for ad-hoc local invocation.

set -euo pipefail

PRIMARY_GPU_ID="${RUNPOD_GPU_ID:-NVIDIA RTX A4000}"
PRIMARY_GPU_COUNT="${RUNPOD_GPU_COUNT:-2}"
FALLBACK_GPU_ID="${RUNPOD_FALLBACK_GPU_ID:-NVIDIA A40}"
FALLBACK_GPU_COUNT="${RUNPOD_FALLBACK_GPU_COUNT:-1}"

DEFAULT_CI_IMAGE="runpod/pytorch:0.7.2-dev-cu1241-torch241-ubuntu2204"
if [ -n "${GITHUB_REPOSITORY:-}" ] && [ "${RUNPOD_USE_GHCR_IMAGE:-1}" = "1" ]; then
  DEFAULT_CI_IMAGE="ghcr.io/${GITHUB_REPOSITORY,,}/rl-kernel-ci:cuda"
fi
CI_IMAGE="${CI_IMAGE:-$DEFAULT_CI_IMAGE}"
DISK_GB="${RUNPOD_DISK_GB:-40}"
PR_SHA="${PR_SHA:-main}"
PR_SHA_FOR_NAME="${PR_SHA}"
if [ "$PR_SHA" = "main" ]; then
  PR_SHA_FOR_NAME="$(date +%s)"
fi
PROFILE_SLUG=$(printf "%s" "${RUNPOD_PROFILE_NAME:-gpu}" | tr -c "[:alnum:]-" "-")
POD_NAME="rl-kernel-ci-${PR_SHA_FOR_NAME:0:7}-${PROFILE_SLUG}"
READY_RETRIES="${RUNPOD_READY_RETRIES:-60}"
SSH_READY_RETRIES="${RUNPOD_SSH_READY_RETRIES:-30}"
REMOTE_ATTEMPTS="${RUNPOD_REMOTE_ATTEMPTS:-2}"
RUNPOD_MIN_CUDA_VERSION="${RUNPOD_MIN_CUDA_VERSION:-12.4}"
RUNPOD_TERMINATE_AFTER="${RUNPOD_TERMINATE_AFTER:-$(date -u -d '+2 hours' '+%Y-%m-%dT%H:%M:%SZ')}"

POD_ID=""
RUNPOD_LOCATION_ARGS=()
if [ -n "${RUNPOD_DATA_CENTER_IDS:-}" ]; then
  RUNPOD_LOCATION_ARGS+=(--data-center-ids "$RUNPOD_DATA_CENTER_IDS")
fi
if [ -n "${RUNPOD_COUNTRY_CODE:-}" ]; then
  RUNPOD_LOCATION_ARGS+=(--country-code "$RUNPOD_COUNTRY_CODE")
fi

_FINAL_EXIT=0
cleanup() {
  local exit_code=$?
  [ "$_FINAL_EXIT" -ne 0 ] && exit_code=$_FINAL_EXIT
  trap - EXIT INT TERM
  if [ -n "$POD_ID" ]; then
    echo ""
    echo "[ci] ========================================================"
    echo "[ci] === AUTOMATIC CLEANUP: Releasing RunPod pod $POD_ID ==="
    echo "[ci] ========================================================"
    REMOVE_OUT=$(runpodctl pod remove "$POD_ID" 2>&1 || true)
    if echo "$REMOVE_OUT" | grep -qiE "unknown command|unknown subcommand"; then
      REMOVE_OUT=$(runpodctl pod delete "$POD_ID" 2>&1 || true)
    fi
    if echo "$REMOVE_OUT" | grep -qi "not found"; then
      echo "[ci] RunPod pod $POD_ID was already released. Safe to exit."
    else
      echo "$REMOVE_OUT"
    fi
  fi
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

# --- RunPod pod provisioning ----------------------------------------------
GPU_ID=$PRIMARY_GPU_ID
GPU_COUNT=$PRIMARY_GPU_COUNT

echo "[ci] Attempt 1: create RunPod pod: ${GPU_COUNT}x ${GPU_ID}"
CREATE_STATUS=0
CREATE_OUT=$(runpodctl pod create \
  --name "$POD_NAME" \
  --gpu-id "$GPU_ID" \
  --gpu-count "$GPU_COUNT" \
  --image "$CI_IMAGE" \
  --container-disk-in-gb "$DISK_GB" \
  --cloud-type SECURE \
  --min-cuda-version "$RUNPOD_MIN_CUDA_VERSION" \
  --terminate-after "$RUNPOD_TERMINATE_AFTER" \
  --ports "22/tcp" \
  "${RUNPOD_LOCATION_ARGS[@]}" 2>&1) || CREATE_STATUS=$?

if [ "$CREATE_STATUS" -ne 0 ] || echo "$CREATE_OUT" | grep -qi "no longer any instances available"; then
  echo "[ci] WARN: ${GPU_COUNT}x ${GPU_ID} unavailable; trying fallback GPU shape."
  GPU_ID=$FALLBACK_GPU_ID
  GPU_COUNT=$FALLBACK_GPU_COUNT

  echo "[ci] Attempt 2 (fallback): create RunPod pod: ${GPU_COUNT}x ${GPU_ID}"
  CREATE_STATUS=0
  CREATE_OUT=$(runpodctl pod create \
    --name "$POD_NAME" \
    --gpu-id "$GPU_ID" \
    --gpu-count "$GPU_COUNT" \
    --image "$CI_IMAGE" \
    --container-disk-in-gb "$DISK_GB" \
    --cloud-type SECURE \
    --min-cuda-version "$RUNPOD_MIN_CUDA_VERSION" \
    --terminate-after "$RUNPOD_TERMINATE_AFTER" \
    --ports "22/tcp" \
    "${RUNPOD_LOCATION_ARGS[@]}" 2>&1) || CREATE_STATUS=$?

  if [ "$CREATE_STATUS" -ne 0 ] || echo "$CREATE_OUT" | grep -qi "no longer any instances available"; then
    echo "[ci] ERROR: Fallback GPU shape (${GPU_COUNT}x ${GPU_ID}) is also unavailable. Please try CI again later."
    exit 1
  fi
fi

if [ "$CREATE_STATUS" -ne 0 ]; then
  echo "[ci] ERROR: Failed to create RunPod pod. Output: $CREATE_OUT"
  exit "$CREATE_STATUS"
fi

# runpodctl may print human-readable text around (or instead of) a JSON
# payload, so try a structured jq parse first and fall back to pattern
# matching on the raw output.
POD_ID=$(printf '%s' "$CREATE_OUT" | jq -r '.id // empty' 2>/dev/null || true)
if [ -z "$POD_ID" ]; then
  POD_ID=$(printf '%s' "$CREATE_OUT" | grep -oE '"id":[[:space:]]*"[a-z0-9]{8,}"' | head -1 | cut -d '"' -f4)
fi
if [ -z "$POD_ID" ]; then
  # Plain-text form: pod "abc123def456" created ...
  POD_ID=$(printf '%s' "$CREATE_OUT" | grep -oE 'pod "[a-z0-9]{8,}"' | head -1 | grep -oE '[a-z0-9]{8,}')
fi
if [ -z "$POD_ID" ]; then
  echo "[ci] ERROR: Unable to resolve RunPod pod id. Output: $CREATE_OUT"
  exit 1
fi
echo "[ci] RunPod pod created: $POD_ID"

# --- Wait for network --------------------------------------------------------
# Resolve the pod's public SSH endpoint from `runpodctl pod get -o json`.
# Layouts differ across runpodctl versions, so try in order:
#   1. runpodctl >= 2.6: a top-level `.ssh {ip, port}` convenience block.
#   2. older layouts: a runtime port-mapping object with privatePort == 22.
#   3. regex fallback for non-JSON / unknown layouts (`"port"` is included
#      because layout 1 calls the external port just `port`).
parse_ssh_endpoint() {
  SSH_IP=$(printf '%s' "$POD_INFO" | jq -r '.ssh.ip // empty' 2>/dev/null || true)
  SSH_PORT=$(printf '%s' "$POD_INFO" | jq -r '.ssh.port // empty' 2>/dev/null || true)
  if [ -n "$SSH_IP" ] && [ -n "$SSH_PORT" ]; then
    return
  fi

  local ssh_mapping
  ssh_mapping=$(printf '%s' "$POD_INFO" | jq -c '
    [.. | objects | select(.privatePort? == 22)]
    | map(select(.isIpPublic? != false))
    | first // empty
  ' 2>/dev/null || true)
  if [ -n "$ssh_mapping" ]; then
    SSH_IP=$(printf '%s' "$ssh_mapping" | jq -r '.ip // .publicIp // empty')
    SSH_PORT=$(printf '%s' "$ssh_mapping" | jq -r '.publicPort // .externalPort // empty')
    return
  fi

  SSH_IP=$(printf '%s' "$POD_INFO" | grep -iE '"ip"|"publicIp"|"address"' | grep -oE '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}' | head -1 || true)
  SSH_PORT=$(printf '%s' "$POD_INFO" | grep -iE '"port"|"publicPort"|"externalPort"' | grep -oE '[0-9]+' | grep -v '^22$' | head -1 || true)
}

echo "[ci] Waiting for RunPod pod network routing..."
SSH_IP=""
SSH_PORT=""

for i in $(seq 1 "$READY_RETRIES"); do
  POD_INFO=$(runpodctl pod get "$POD_ID" -o json)
  parse_ssh_endpoint

  if [ -n "$SSH_IP" ] && [ -n "$SSH_PORT" ] && ! echo "$POD_INFO" | grep -qi "not ready"; then
    echo "[ci] RunPod pod network is ready."
    break
  fi

  if [ "$i" -eq "$READY_RETRIES" ]; then
    echo "[ci] ERROR: RunPod pod network/SSH initialization timed out."
    exit 1
  fi

  echo "[ci] Network still initializing; waiting 10s (attempt $i/$READY_RETRIES)"
  sleep 10
done

echo "[ci] Remote target: root@$SSH_IP:$SSH_PORT"

# --- SSH options -------------------------------------------------------------
RUNPOD_SSH_KEY_PATH="${RUNPOD_SSH_KEY_PATH:-}"
SSH_OPTIONS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p $SSH_PORT"
# scp uses -P (capital) for port; build a separate option string for it.
SCP_OPTIONS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -P $SSH_PORT"
if [ -n "$RUNPOD_SSH_KEY_PATH" ]; then
  if [ ! -r "$RUNPOD_SSH_KEY_PATH" ]; then
    echo "[ci] ERROR: RUNPOD_SSH_KEY_PATH is set but not readable: $RUNPOD_SSH_KEY_PATH"
    exit 1
  fi
  SSH_OPTIONS="$SSH_OPTIONS -o IdentitiesOnly=yes -i $RUNPOD_SSH_KEY_PATH"
  SCP_OPTIONS="$SCP_OPTIONS -o IdentitiesOnly=yes -i $RUNPOD_SSH_KEY_PATH"
fi

echo "[ci] Verifying SSH daemon readiness..."
for i in $(seq 1 "$SSH_READY_RETRIES"); do
  if ssh $SSH_OPTIONS root@"$SSH_IP" true >/dev/null 2>&1; then
    echo "[ci] SSH daemon is ready."
    break
  fi
  if [ "$i" -eq "$SSH_READY_RETRIES" ]; then
    echo "[ci] ERROR: SSH daemon did not become ready."
    exit 1
  fi
  echo "[ci] SSH not ready yet... waiting 10s (Attempt $i/$SSH_READY_RETRIES)"
  sleep 10
done

# --- Upload test script and run it remotely ----------------------------------
# run_cuda_tests.sh is uploaded rather than embedded so it stays the single
# source of truth for test logic regardless of how CI is triggered.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCP_OK=0
for i in $(seq 1 3); do
  if scp $SCP_OPTIONS "${SCRIPT_DIR}/run_cuda_tests.sh" root@"${SSH_IP}:/tmp/run_cuda_tests.sh"; then
    SCP_OK=1; break
  fi
  echo "[ci] WARN: scp attempt $i failed; retrying after 10s..."
  sleep 10
done
if [ "$SCP_OK" -eq 0 ]; then
  echo "[ci] ERROR: scp upload of run_cuda_tests.sh failed after 3 attempts."
  _FINAL_EXIT=1; exit 1
fi

printf -v REMOTE_ENV \
  "GPU_COUNT=%q PR_REPO_URL=%q PR_SHA=%q TORCH_CUDA_ARCH_LIST=%q FORCE_CUDA=%q MAX_JOBS=%q KERNEL_ALIGN_FORCE_SM90=%q PYTEST_ARGS=%q FLASHINFER_WHEEL_INDEX=%q CI_UPGRADE_BUILD_TOOLS=%q CI_INSTALL_FLASHINFER=%q" \
  "$GPU_COUNT" \
  "${PR_REPO_URL:-https://github.com/RL-Align/RL-Kernel.git}" \
  "$PR_SHA" \
  "${TORCH_CUDA_ARCH_LIST:-8.6}" \
  "${FORCE_CUDA:-1}" \
  "${MAX_JOBS:-8}" \
  "${KERNEL_ALIGN_FORCE_SM90:-0}" \
  "${PYTEST_ARGS:-tests/ rl_engine/tests/ -v}" \
  "${FLASHINFER_WHEEL_INDEX:-https://flashinfer.ai/whl/cu124/torch2.4/index.html}" \
  "${CI_UPGRADE_BUILD_TOOLS:-0}" \
  "${CI_INSTALL_FLASHINFER:-0}"

run_remote_suite() {
  ssh $SSH_OPTIONS root@"$SSH_IP" "$REMOTE_ENV bash /tmp/run_cuda_tests.sh"
}

echo "[ci] Launching remote CUDA test suite (GPU_COUNT=${GPU_COUNT})..."
TEST_EXIT=0
for attempt in $(seq 1 "$REMOTE_ATTEMPTS"); do
  if [ "$REMOTE_ATTEMPTS" -gt 1 ]; then
    echo "[ci] Remote execution attempt $attempt/$REMOTE_ATTEMPTS"
  fi

  set +e
  run_remote_suite
  TEST_EXIT=$?
  set -e

  if [ "$TEST_EXIT" -eq 255 ] && [ "$attempt" -lt "$REMOTE_ATTEMPTS" ]; then
    echo "[ci] WARN: SSH remote execution disconnected; retrying after 10s..."
    sleep 10
    continue
  fi

  break
done

echo "[ci] Remote execution finished with exit code = $TEST_EXIT"
_FINAL_EXIT=$TEST_EXIT
exit $TEST_EXIT
