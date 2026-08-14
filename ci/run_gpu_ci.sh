#!/usr/bin/env bash
# Run one GPU CI job on a shared, local 8x H100 SXM host.
#
# Lock files are deliberately persistent.  Removing a lock file while another
# process still holds its inode can split the lock domain.  A GPU pair is
# released by closing its flock file descriptor; the optional owner file is
# only diagnostic metadata and is removed during cleanup.
set -Eeuo pipefail

readonly GPU_PAIRS=("0,1" "2,3" "4,5" "6,7")
# Fixed descriptors keep this script compatible with Bash versions that do not
# support `exec {var}>file`.  fd 9 serializes allocation; fds 8..5 are the
# four pair-lock descriptors in the same order as GPU_PAIRS.
readonly SCHEDULER_FD=9
readonly GPU_PAIR_FDS=(8 7 6 5)
readonly GPU_CI_LOCK_DIR="${GPU_CI_LOCK_DIR:-/var/tmp/rl-kernel-gpu-ci/locks}"
readonly GPU_CI_WORK_ROOT="${GPU_CI_WORK_ROOT:-/var/tmp/rl-kernel-gpu-ci/workspaces}"
readonly GPU_CI_WAIT_SECONDS="${GPU_CI_WAIT_SECONDS:-5}"
readonly EXPECTED_GPU_NAME="${EXPECTED_GPU_NAME:-H100}"
readonly EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-8}"
readonly TARGET_SM="${TARGET_SM:-9.0}"
readonly KERNEL_ALIGN_FORCE_SM90="${KERNEL_ALIGN_FORCE_SM90:-1}"

GPU_PAIR_FD=""
GPU_PAIR=""
GPU_PAIR_LABEL=""
GPU_PAIR_OWNER_FILE=""
WORK_DIR=""
SOURCE_DIR=""
PYTHON=""
TEST_PID=""

die() {
  echo "[gpu-ci] FATAL: $*" >&2
  exit 1
}

release_scheduler_lock() {
  flock -u "$SCHEDULER_FD" || true
}

release_gpu_pair() {
  if [[ -n "$GPU_PAIR_FD" ]]; then
    rm -f -- "$GPU_PAIR_OWNER_FILE" || true
    flock -u "$GPU_PAIR_FD" || true
    echo "[gpu-ci] Released GPU pair ${GPU_PAIR}."
    GPU_PAIR_FD=""
    GPU_PAIR=""
  fi
}

close_lock_files() {
  exec 5>&- || true
  exec 6>&- || true
  exec 7>&- || true
  exec 8>&- || true
  exec 9>&- || true
}

stop_test_process_group() {
  if [[ -z "$TEST_PID" ]]; then
    return
  fi

  if kill -0 "$TEST_PID" 2>/dev/null; then
    echo "[gpu-ci] Stopping test process group ${TEST_PID}."
    kill -TERM -- "-${TEST_PID}" 2>/dev/null || true
    for _ in $(seq 1 10); do
      kill -0 "$TEST_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-${TEST_PID}" 2>/dev/null || true
  fi
  wait "$TEST_PID" 2>/dev/null || true
  TEST_PID=""
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM HUP

  stop_test_process_group
  release_gpu_pair
  release_scheduler_lock
  close_lock_files

  if [[ -n "$WORK_DIR" && -d "$WORK_DIR" ]]; then
    rm -rf -- "$WORK_DIR" || true
  fi
  exit "$status"
}

on_signal() {
  local signal="$1"
  local status="$2"
  echo "[gpu-ci] Received ${signal}; cleaning up GPU lease."
  exit "$status"
}

trap cleanup EXIT
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
trap 'on_signal HUP 129' HUP

require_host_prerequisites() {
  command -v flock >/dev/null || die "flock is required for GPU-pair scheduling."
  command -v git >/dev/null || die "git is required to fetch the PR revision."
  command -v setsid >/dev/null || die "setsid is required to terminate distributed tests safely."
  command -v nvidia-smi >/dev/null || die "nvidia-smi is required on the local GPU runner."
  command -v python3 >/dev/null || die "python3 is required on the local GPU runner."

  local detected_count
  detected_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d '[:space:]')
  [[ "$detected_count" == "$EXPECTED_GPU_COUNT" ]] || die \
    "expected ${EXPECTED_GPU_COUNT} GPUs, found ${detected_count}. Refusing to allocate an unknown host."

  local gpu_index gpu_name gpu_sm
  for gpu_index in $(seq 0 7); do
    gpu_name=$(nvidia-smi --id="$gpu_index" --query-gpu=name --format=csv,noheader | xargs)
    gpu_sm=$(nvidia-smi --id="$gpu_index" --query-gpu=compute_cap --format=csv,noheader | xargs)
    [[ "$gpu_name" == *"$EXPECTED_GPU_NAME"* ]] || die \
      "GPU ${gpu_index} is '${gpu_name}', expected a ${EXPECTED_GPU_NAME}."
    [[ "$gpu_sm" == "$TARGET_SM" ]] || die \
      "GPU ${gpu_index} is sm_${gpu_sm}, expected sm_${TARGET_SM}."
  done
}

acquire_gpu_pair() {
  local scheduler_lock_file="${GPU_CI_LOCK_DIR}/scheduler.lock"
  local pair pair_index candidate_fd

  mkdir -p "$GPU_CI_LOCK_DIR"
  [[ -d "$GPU_CI_LOCK_DIR" && -w "$GPU_CI_LOCK_DIR" ]] || die \
    "GPU_CI_LOCK_DIR is not writable: ${GPU_CI_LOCK_DIR}"

  # Serialize only the probe-and-claim transaction.  The selected pair lock is
  # retained while tests run, so unrelated jobs can immediately claim another pair.
  exec 9>>"$scheduler_lock_file"
  exec 8>>"${GPU_CI_LOCK_DIR}/pair-0-1.lock"
  exec 7>>"${GPU_CI_LOCK_DIR}/pair-2-3.lock"
  exec 6>>"${GPU_CI_LOCK_DIR}/pair-4-5.lock"
  exec 5>>"${GPU_CI_LOCK_DIR}/pair-6-7.lock"

  while true; do
    flock "$SCHEDULER_FD"

    for pair_index in "${!GPU_PAIRS[@]}"; do
      pair="${GPU_PAIRS[$pair_index]}"
      candidate_fd="${GPU_PAIR_FDS[$pair_index]}"
      if flock -n "$candidate_fd"; then
        GPU_PAIR="$pair"
        GPU_PAIR_LABEL="${pair/,/-}"
        GPU_PAIR_FD="$candidate_fd"
        GPU_PAIR_OWNER_FILE="${GPU_CI_LOCK_DIR}/pair-${GPU_PAIR_LABEL}.owner"
        printf 'pid=%s\nrun_id=%s\nrepository=%s\nacquired_at=%s\n' \
          "$$" "${GITHUB_RUN_ID:-local}" "${GITHUB_REPOSITORY:-local}" \
          "$(date -u +%FT%TZ)" > "$GPU_PAIR_OWNER_FILE"
        release_scheduler_lock
        echo "[gpu-ci] Acquired GPU pair ${GPU_PAIR}."
        return
      fi
    done

    release_scheduler_lock
    echo "[gpu-ci] All GPU pairs are busy; waiting ${GPU_CI_WAIT_SECONDS}s before retrying."
    sleep "$GPU_CI_WAIT_SECONDS"
  done
}

prepare_pr_worktree() {
  mkdir -p "$GPU_CI_WORK_ROOT"
  [[ -d "$GPU_CI_WORK_ROOT" && -w "$GPU_CI_WORK_ROOT" ]] || die \
    "GPU_CI_WORK_ROOT is not writable: ${GPU_CI_WORK_ROOT}"

  WORK_DIR=$(mktemp -d "${GPU_CI_WORK_ROOT%/}/rl-kernel-gpu-ci.XXXXXX")
  SOURCE_DIR="${WORK_DIR}/repo"

  if [[ -n "${PR_REPO_URL:-}" && -n "${PR_SHA:-}" ]]; then
    echo "[gpu-ci] Fetching PR revision ${PR_SHA} into an isolated worktree."
    git clone --no-checkout "$PR_REPO_URL" "$SOURCE_DIR"
    git -C "$SOURCE_DIR" fetch --depth=1 origin "$PR_SHA"
    git -C "$SOURCE_DIR" checkout --detach FETCH_HEAD
  else
    # This branch is useful for local reproduction only.  GitHub Actions always
    # supplies PR_REPO_URL and PR_SHA, so production CI never tests the base checkout.
    [[ -n "${GITHUB_WORKSPACE:-}" ]] || die "PR_REPO_URL/PR_SHA or GITHUB_WORKSPACE is required."
    SOURCE_DIR="$GITHUB_WORKSPACE"
  fi
}

create_job_venv() {
  local base_python="${PYTHON_BIN:-python3}"
  local venv_dir="${WORK_DIR}/venv"

  "$base_python" -m venv --system-site-packages "$venv_dir"
  PYTHON="${venv_dir}/bin/python"
  "$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA-enabled torch is required"'

  # Keep package installation job-local; concurrent CI jobs must not mutate the
  # self-hosted runner's global Python environment.
  "$PYTHON" -m pip install --no-build-isolation --no-deps -e "$SOURCE_DIR"
  "$PYTHON" -m pip install --no-cache-dir numpy tabulate accelerate "transformers==5.13.1" pytest
}

configure_gpu_isolation() {
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="$GPU_PAIR"
  export FORCE_CUDA=1
  export MAX_JOBS="${MAX_JOBS:-8}"
  export KERNEL_ALIGN_FORCE_SM90
  export TORCH_CUDA_ARCH_LIST="${TARGET_SM}+PTX"
  export RL_KERNEL_REQUIRE_EXT=1

  echo "[gpu-ci] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
}

run_tests() {
  "$PYTHON" - <<'PY'
import os
import torch

assert torch.cuda.is_available(), "CUDA is not available after GPU allocation"
assert torch.cuda.device_count() == 2, (
    f"expected exactly 2 visible GPUs, got {torch.cuda.device_count()} "
    f"for CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
)
for local_index in range(2):
    name = torch.cuda.get_device_name(local_index)
    print(f"[gpu-ci] local cuda:{local_index} -> {name}")
    assert "H100" in name, f"unexpected visible device: {name}"
PY

  (
    cd "$SOURCE_DIR"
    "$PYTHON" scripts/ci_smoke.py
  )

  echo "[gpu-ci] Starting TP=2 test suite on physical GPUs ${GPU_PAIR}."
  # setsid gives the signal handler an isolated process group to terminate on
  # workflow cancellation before the GPU-pair flock is released.
  setsid "$PYTHON" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    -m pytest "$SOURCE_DIR/tests" -v &
  TEST_PID=$!

  local test_status
  set +e
  wait "$TEST_PID"
  test_status=$?
  set -e
  TEST_PID=""

  echo "[gpu-ci] Test suite exited with code ${test_status}."
  return "$test_status"
}

main() {
  require_host_prerequisites
  acquire_gpu_pair
  configure_gpu_isolation
  prepare_pr_worktree
  create_job_venv
  run_tests
}

main "$@"
