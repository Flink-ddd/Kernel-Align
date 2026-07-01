# Testing

RL-Kernel uses focused tests for dispatch behavior and operator accuracy.

## Docker Images

Build the source-build images from the repository root:

```bash
docker build -f docker/Dockerfile.cuda -t rl-kernel-ci:cuda .
docker build -f docker/Dockerfile.rocm -t rl-kernel-ci:rocm .
```

The CUDA image is based on a CUDA-enabled PyTorch devel image and installs the
compiler, CMake, Ninja, and Python test tooling needed for editable source
builds. The ROCm image is based on the official ROCm PyTorch image and includes
the same source-build tooling plus common FlashAttention ROCm build helpers.

Run CUDA tests in the image with an NVIDIA runtime:

```bash
docker run --rm --gpus all \
  -v "$PWD:/workspace/RL-Kernel" \
  -w /workspace/RL-Kernel \
  rl-kernel-ci:cuda \
  bash -lc 'pip install -e ".[cuda,test]" && python -m pytest tests/test_kernel_registry.py -q'
```

Run ROCm tests on an AMD host with ROCm devices exposed:

```bash
IMAGE=rl-kernel-ci:rocm bash ci/run_rocm_container.sh
```

The container helper mounts the current checkout, exposes `/dev/kfd` and
`/dev/dri`, adds the numeric device GIDs used by the host, and writes
`rocm-ci-container.log`. Numeric GIDs are intentional: some ROCm hosts expose
the render device without a matching `render` group name inside Docker.

Do not report a ROCm pass from a CUDA or CPU-only machine. `ci/run_rocm_ci.sh`
checks for a ROCm PyTorch build before running hardware tests.

The ROCm CI helper defaults to the PyTorch SDPA fallback path:

```bash
bash ci/run_rocm_ci.sh
```

To validate the external ROCm FlashAttention path on a machine where the longer
source build is acceptable:

```bash
RL_KERNEL_ROCM_ATTN_BACKEND=flash_attn bash ci/run_rocm_ci.sh
```

## Dispatch Tests

```bash
python -m pytest rl_engine/tests/test_dispatch.py -v
python -m pytest tests/test_kernel_registry.py -q
```

## Operator Accuracy

```bash
python tests/test_op_accuracy.py
```

## Documentation Build

```bash
pip install -r requirements-docs.txt
mkdocs build --strict -f mkdocs.yaml
```

Run the documentation build whenever adding a new operator page or changing navigation.

## Hardware CI

Default pull-request CI runs linting, documentation, CPU tests, and mocked
hardware dispatch tests. The Docker image workflow builds both CUDA and ROCm
images on pull requests and pushes `rl-kernel-ci:cuda` / `rl-kernel-ci:rocm` to
GHCR after merges to `main`.

### CUDA — three paths, same test script

All CUDA hardware paths run the same `ci/run_cuda_tests.sh`. Choose whichever
suits your setup:

**Path A — local machine or self-hosted runner (no cloud account)**

```bash
# Run directly on any machine with CUDA drivers and Python:
bash ci/run_cuda_tests.sh

# Or point it at a specific PR commit for isolated testing:
PR_REPO_URL=https://github.com/RL-Align/RL-Kernel.git \
PR_SHA=<commit-sha> \
bash ci/run_cuda_tests.sh
```

In GitHub Actions, add the label `needs-gpu-ci-self-hosted` to your PR and
register a runner with the tag `cuda`. This path executes the PR commit on the
self-hosted runner; only use it for PRs that are allowed to run on that machine.

**Path B — hosted CUDA runner**

Add the label `needs-gpu-ci` to the PR on GitHub. This runs
`ci/run_cuda_tests.sh` on an ephemeral NVIDIA GPU instance via
`ci/run_gpu_ci.sh`, then releases the instance. Configure the hosted-provider
API and SSH secrets in GitHub Actions before enabling this path.

For Hopper-specific coverage, add `needs-gpu-ci-sm90`. This runs a separate
H100 job with `KERNEL_ALIGN_FORCE_SM90=1` so SM90/TMA kernels are compiled and
the fallback tests run on real Hopper hardware. It is intentionally separate
from `needs-gpu-ci` because H100 capacity is more expensive and less available.

**Path C — Docker (local validation without GPU CI secrets)**

```bash
docker run --rm --gpus all \
  -v "$PWD:/workspace/RL-Kernel" \
  -w /workspace/RL-Kernel \
  rl-kernel-ci:cuda \
  bash ci/run_cuda_tests.sh
```

### ROCm — self-hosted runner

Add `needs-rocm-ci` to the PR. Requires a self-hosted runner with the `rocm`
tag. The script also runs standalone on any ROCm machine:

```bash
bash ci/run_rocm_ci.sh

# Override only for mixed-architecture hosts or explicit cross-builds:
PYTORCH_ROCM_ARCH=gfx1100 bash ci/run_rocm_ci.sh

# With the FlashAttention ROCm backend:
RL_KERNEL_ROCM_ATTN_BACKEND=flash_attn bash ci/run_rocm_ci.sh
```

When `PYTORCH_ROCM_ARCH` is unset or `auto`, `ci/run_rocm_ci.sh` detects the
visible ROCm device architectures and scopes HIP extension builds accordingly.

Like the CUDA self-hosted path, this executes the PR commit on the configured
ROCm machine; only use it for PRs that are allowed to run on that runner.

### Required GitHub Actions secrets

The following secrets are only needed for the hosted CUDA paths (`needs-gpu-ci`
and `needs-gpu-ci-sm90`). The self-hosted and local paths require no secrets.

| Secret | Purpose |
|--------|---------|
| `RUNPOD_API_KEY` | Authenticates hosted GPU instance creation/removal for CUDA CI |
| `RUNPOD_SSH_PRIVATE_KEY` | Ed25519 private key for SSH access to hosted GPU instances |

The **public key** counterpart of `RUNPOD_SSH_PRIVATE_KEY` must be registered
with the hosted GPU provider before GPU CI can connect to the instance.
