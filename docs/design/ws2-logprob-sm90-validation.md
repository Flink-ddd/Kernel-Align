# WS2 Logprob PR2 SM90 Validation

This document records the Hopper SM90 validation procedure for the PR2 single-GPU
logprob comparison harness from issue #241. It is a validation note for maintainers;
the cloud setup wrapper used during development is intentionally kept outside the
repository.

## Prerequisites

The validation host must provide:

- Python 3.10 or newer;
- CUDA-enabled PyTorch;
- an NVIDIA Hopper GPU with compute capability 9.0, such as H100, H800, or H200;
- `nvidia-smi` and `nvcc`;
- a CUDA development environment capable of compiling the RL-Kernel extension.

The CUDA version reported by `nvcc` must match `torch.version.cuda`. A runtime-only
image is insufficient because it normally does not include the CUDA compiler.

## Build

Activate an environment containing the repository dependencies and a CUDA-enabled
PyTorch installation, then build the editable extension with SM90 enabled:

```bash
export FORCE_CUDA=1
export KERNEL_ALIGN_FORCE_SM90=1
export TORCH_CUDA_ARCH_LIST="9.0+PTX"
export MAX_JOBS=2

python -m pip install --no-build-isolation --no-deps -e .
```

Verify the extension and SM90 symbol after the build. Import PyTorch first so its
runtime libraries are available to the extension loader:

```bash
python - <<'PY'
import torch
from rl_engine import _C

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("extension:", _C.__file__)
print("SM90 symbol:", hasattr(_C, "batch_invariant_logp_sm90"))
PY
```

The final line must report `SM90 symbol: True`.

## Validation commands

Run the focused PR2 tests and the complete batch-invariant logprob suite:

```bash
python -m pytest \
  tests/test_logprob_comparison.py \
  tests/test_operator_inputs.py \
  tests/test_op_checks.py -q

python -m pytest tests/test_batch_invariant_logp.py -q
```

Run the two explicit SM90 comparisons:

```bash
python scripts/compare_logprob.py \
  --candidate cuda-sm90 \
  --device cuda \
  --dtype bf16 \
  --batch 2 \
  --seq 8 \
  --vocab 1024 \
  --prompt-tokens 3 \
  --seed 7

python scripts/compare_logprob.py \
  --candidate cuda-sm90 \
  --device cuda \
  --dtype bf16 \
  --batch 2 \
  --seq 16 \
  --vocab 151936 \
  --prompt-tokens 8 \
  --seed 241
```

The comparison command writes the JSON report to stdout. Diagnostic log messages are
written to stderr so stdout can be redirected directly to a `.json` file.

## Expected report

The report must identify the requested and actual backend as `cuda-sm90`, use the
`BatchInvariantLogpSM90Op` implementation, and record:

```text
tp_world=1
communication=none
lse_source=direct
```

LSE drift is measured over all logical token rows. Selected-logprob drift is measured
only over active response/action tokens. Each drift section includes maximum, mean,
p95, p99, and active-count values.

## Validation result

The procedure was validated on:

```text
GPU: NVIDIA H800 PCIe
Compute capability: 9.0
Python: 3.11.15
PyTorch: 2.11.0+cu128
CUDA toolkit / nvcc: 12.8
Triton: 3.6.0
```

Results:

```text
PR2 focused tests: 41 passed
Complete batch-invariant logprob suite: 67 passed
```

Observed BF16 SM90 drift against the PyTorch reference:

| Shape | LSE max abs | dlogp max abs |
| --- | ---: | ---: |
| `[2, 8, 1024]` | `4.76837158203125e-07` | `4.76837158203125e-07` |
| `[2, 16, 151936]` | `9.5367431640625e-07` | `9.5367431640625e-07` |

Both runs used TP=1, no communication, and the explicit SM90 backend without fallback.
