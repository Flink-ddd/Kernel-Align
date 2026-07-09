#!/usr/bin/env bash
set -euo pipefail
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-rl-kernel-ci:rocm}"
LOG="${LOG:-${ROOT}/rocm-ci-container.log}"
DOCKER="${DOCKER:-docker}"

require_rocm_devices() {
  if [[ ! -e /dev/kfd ]]; then
    echo "[rocm-container] FATAL: /dev/kfd not found; ROCm device is not visible" >&2
    exit 1
  fi
  if [[ ! -e /dev/dri ]]; then
    echo "[rocm-container] FATAL: /dev/dri not found; ROCm DRM device is not visible" >&2
    exit 1
  fi
}

docker_device_group_args() {
  local seen=" "
  local path gid
  for path in /dev/kfd /dev/dri/renderD* /dev/dri/card*; do
    [[ -e "${path}" ]] || continue
    gid="$(stat -c '%g' "${path}" 2>/dev/null || true)"
    [[ -n "${gid}" ]] || continue
    case "${seen}" in
      *" ${gid} "*) ;;
      *)
        printf '%s\n' "--group-add=${gid}"
        seen="${seen}${gid} "
        ;;
    esac
  done
}

run_container_ci() {
  local docker_group_args=()
  local docker_env_args=()
  local arg
  while IFS= read -r arg; do
    docker_group_args+=("${arg}")
  done < <(docker_device_group_args)
  docker_env_args+=(-e "RL_KERNEL_ROCM_ATTN_BACKEND=${RL_KERNEL_ROCM_ATTN_BACKEND:-sdpa}")
  docker_env_args+=(-e "MAX_JOBS=${MAX_JOBS:-8}")
  [[ -n "${PYTORCH_ROCM_ARCH:-}" ]] && docker_env_args+=(-e "PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}")
  [[ -n "${PYTEST_ARGS:-}" ]] && docker_env_args+=(-e "PYTEST_ARGS=${PYTEST_ARGS}")

  echo "===== host info ====="
  date -Is
  uname -a
  id
  command -v rocminfo >/dev/null && rocminfo | grep -E "Name:|gfx" | head -80 || true
  command -v rocm-smi >/dev/null && rocm-smi || true

  echo "===== run ROCm CI container ====="
  echo "image=${IMAGE}"
  if ((${#docker_group_args[@]})); then
    echo "device group args=${docker_group_args[*]}"
  else
    echo "device group args=(none)"
  fi

  "${DOCKER}" run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    "${docker_group_args[@]}" \
    --ipc=host \
    --shm-size=16g \
    --security-opt seccomp=unconfined \
    -v "${ROOT}:/workspace/RL-Kernel" \
    -w /workspace/RL-Kernel \
    "${docker_env_args[@]}" \
    "${IMAGE}" \
    bash ci/run_rocm_ci.sh
}

require_rocm_devices
run_container_ci 2>&1 | tee "${LOG}"
exit_code="${PIPESTATUS[0]}"
echo "exit_code=${exit_code}" | tee -a "${LOG}"
exit "${exit_code}"
