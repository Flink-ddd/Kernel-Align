# SiLU / SwiGLU Activation

The activation operators are the element-wise core of the Qwen3/Llama gated MLP. They
implement the WS1 dual-path contract (issue #108): pure-PyTorch fp32 ground truth, plus
CUDA, Triton, and Ascend C candidates that validate against it.

- **SiLU** (`NativeSiLUOp` / `SiLUCudaOp` / `TritonSiLUOp`): `silu(x) = x * sigmoid(x)` —
  the `hidden_act="silu"` gate.
- **SwiGLU** (`NativeSwiGLUOp` / `SwiGLUCudaOp` / `TritonSwiGLUOp` /
  `SwiGLUAscendOp`):
  `swiglu(gate, up) = silu(gate) * up` — the gated MLP middle stage. `gate` / `up` are the
  `gate_proj` / `up_proj` outputs (already at the intermediate width); the following
  `down_proj` is a plain Matmul and is **not** part of this operator.

```text
hidden --gate_proj--> gate --\
                              swiglu --> down_proj --> hidden
hidden --up_proj----> up ----/
```

## Entry Point
```python
from rl_engine.kernels.registry import kernel_registry

silu = kernel_registry.get_op("silu")
swiglu = kernel_registry.get_op("swiglu")

# SiLU: single element-wise activation
y = silu(x)                       # [..., N]  ->  [..., N]

# SwiGLU: gated activation (gate and up must share shape)
h = swiglu(gate, up)              # [..., I], [..., I]  ->  [..., I]
```

All backends expose the WS1 dual-path contract:

- `forward(...)` — computes in fp32, casts back to the input dtype (Axis-B accuracy
  candidate / dtype-behavior path).
- `forward_fp32(...)` — computes and returns fp32 (the ground-truth golden path).

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| PyTorch fallback | `NativeSiLUOp` / `NativeSwiGLUOp` | None | fp32 ground-truth reference; CPU and any GPU. |
| CUDA | `SiLUCudaOp` / `SwiGLUCudaOp` | `_C.silu_*` / `_C.swiglu_*` | General CUDA (fp16/bf16/fp32); math in fp32. |
| Triton | `TritonSiLUOp` / `TritonSwiGLUOp` | Triton JIT | Portable GPU baseline; same fp32 math contract. |
| Ascend C | `SwiGLUAscendOp` | `_C_npu.swiglu_ascend_*` | NPU SwiGLU forward/backward (fp16/bf16/fp32); math in fp32. |

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `x` (SiLU) | `[..., N]` | float (fp16/bf16/fp32) | Any shape; last dim arbitrary (Qwen3-8B `I=12288`). |
| `gate` (SwiGLU) | `[..., I]` | float | `gate_proj` output. |
| `up` (SwiGLU) | `[..., I]` | float | `up_proj` output; **must share `gate`'s shape, dtype, and device**. |
| output | same as input | `forward`: input dtype · `forward_fp32`: float32 | Same shape as input. |

Element-wise and shape-agnostic: the Qwen3-8B intermediate dim `I=12288` is just one valid
last-dim size, not a hard requirement. Pure functions — no randomness, no in-place
mutation, device/dtype follow the inputs.

## Dispatch Behavior

`kernel_registry.get_op("silu" | "swiglu")` resolves through the `OpBackend` priority map:

| Platform | Priority |
| --- | --- |
| `cuda` | CUDA → Triton → PyTorch native |
| `rocm` | Triton → PyTorch native |
| `cpu` | PyTorch native |
| `npu` | Ascend C SwiGLU → PyTorch native (SwiGLU); PyTorch native (SiLU) |

If an accelerated extension is not built (or its symbols are missing), the registry moves
to the next available candidate and ultimately falls back to the native gold.

## Accuracy

Reference semantics (`forward_fp32`, fp32 accumulation):

```python
# SiLU
out = x.float() * torch.sigmoid(x.float())

# SwiGLU
gate_f = gate.float()
out = gate_f * torch.sigmoid(gate_f) * up.float()
```

- **Ground truth**: `forward_fp32` always accumulates in and returns fp32.
- **Dtype path**: `forward` runs the same fp32 math, then casts back to the input dtype.
- **Axis A — batch invariance**: element-wise and row-independent, so a row's output is
  bitwise-identical regardless of batch size or padding (`torch.equal`, `atol=0`).
- **Axis B — tolerance**: as `elementwise` ops, low-precision tolerance follows the
  `elementwise` row of the WS1 numerical contract (`tolerance_contract.json`).

## Ground-truth harness

CUDA and Triton candidates are registered in `OP_SPECS` and can be checked with the
shared issue-#108 CLI:

```bash
python scripts/check_operator.py --op silu --candidate cuda --dtype bf16 --device cuda
python scripts/check_operator.py --op swiglu --candidate triton --dtype bf16 --device cuda --check-grad
python scripts/check_operator.py --op silu --candidate pytorch --dtype fp32 --device cpu --check-grad
```

Gold path: `NativeSiLUOp.forward_fp32` / `NativeSwiGLUOp.forward_fp32`.

## Performance Notes

Element-wise kernels with a fixed 1-D grid (CUDA), `BLOCK=1024` (Triton), or aligned
multi-core 1-D tiles of at most 5120 elements (Ascend C). Suitable as the standalone WS1
activation path; fused bias+SiLU MLP kernels remain a separate future work item and should
continue to validate against this reference.

## Tests

```bash
python -m pytest tests/test_swiglu.py tests/test_swiglu_ascend.py -v
```

Covers: correctness vs an independent fp32 formula, dtype paths, Axis-A batch invariance
(slice + padding), input purity, gradient flow, the SwiGLU shape guard, CUDA/Triton vs
native forward+backward, Ascend C forward/backward and tail tiles when NPU hardware is
available, registry dispatch, and the issue-#108 `OP_SPECS` harness.

## Implementation Files

- `rl_engine/kernels/ops/pytorch/activation/swiglu.py` — gold
- `rl_engine/kernels/ops/cuda/activation/swiglu.py` — CUDA wrappers
- `rl_engine/kernels/ops/triton/activation/swiglu.py` — Triton kernels
- `rl_engine/kernels/ops/ascend/activation/swiglu.py` — Ascend wrapper/autograd
- `csrc/cuda/activation.cu` — CUDA kernels
- `csrc/ascend/swiglu_ascend.asc` — Ascend C forward/backward kernels
- `rl_engine/kernels/registry.py`
- `rl_engine/kernels/gtest/operator_specs.py`
- `tests/test_swiglu.py`
- `tests/test_swiglu_ascend.py`

## Known Limitations

- SwiGLU requires `gate` and `up` to share shape, dtype, and device; no broadcasting.
- No fused `bias + SiLU` or `chunk(y,2) + silu_and_mul` variant yet (vLLM-style
  `SiluAndMul` on a packed gate/up tensor). Callers that hold a packed tensor should
  split first, then call `swiglu`.
