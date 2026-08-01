# SiLU / SwiGLU Activation

The activation operators are the element-wise core of the Qwen3/Llama gated MLP. The
pure-PyTorch implementations are **WS1 ground-truth references** (issue #108):
fp32-accumulating definitions of the correct answer for optimized backends.

- **SiLU** (`NativeSiLUOp`): `silu(x) = x * sigmoid(x)` — the `hidden_act="silu"` gate.
- **SwiGLU** (`NativeSwiGLUOp`): `swiglu(gate, up) = silu(gate) * up` — the gated MLP
  middle stage. The following `down_proj` is a separate operator.

```text
hidden --gate_proj--> gate --\
                              swiglu --> down_proj --> hidden
hidden --up_proj----> up ----/
```

## Entry point

```python
from rl_engine.kernels.registry import kernel_registry

silu = kernel_registry.get_op("silu")
swiglu = kernel_registry.get_op("swiglu")

y = silu(x)
h = swiglu(gate, up)
```

The native reference ops expose the WS1 dual-path contract:

- `forward(...)` computes in fp32 and casts back to the input dtype.
- `forward_fp32(...)` computes and returns fp32 ground truth.

## Fused Qwen3 forward contract

The optimized operator is the local activation boundary in issue #239's Qwen3-8B TP=2
pipeline:

```text
gate_local [M_local, 6144] --\
                                  SiLU(gate) * up --> hidden_local [M_local, 6144]
up_local   [M_local, 6144] --/
```

| Tensor | Shape | Dtype | Layout/device |
| --- | --- | --- | --- |
| `gate_local` | `[..., I/TP]` | BF16 | CUDA; arbitrary input strides accepted |
| `up_local` | same as `gate_local` | BF16 | same shape and device as `gate_local` |
| `hidden_local` | same as inputs | BF16 | contiguous CUDA output |

Both optimized backends compute every coordinate in FP32 and round once when storing BF16:

```text
sigmoid_gate = 1 / (1 + exp(-float(gate)))
hidden = float(gate) * sigmoid_gate * float(up)
```

The fused activation has no collective, reduction, random state, in-place mutation, or Down
GEMM. It runs on the caller's current stream and returns a tensor on the inputs' device.

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| CUDA SM90 | `SwiGLUSM90Op` | `swiglu_forward_sm90` | BF16x2 fixed mapping with scalar odd tail |
| Triton | `TritonSwiGLUOp` | None | Fixed block size; no autotune |
| PyTorch | `NativeSiLUOp` / `NativeSwiGLUOp` | None | FP32 ground-truth reference and fallback |

On CUDA, registry dispatch prefers the compiled CUDA implementation on SM90, then Triton,
then the PyTorch reference. Constructing a backend class directly provides explicit backend
selection for validation.

## Accuracy and invariance

Reference semantics are:

```python
# SiLU
out = x.float() * torch.sigmoid(x.float())

# SwiGLU
gate_f = gate.float()
out = gate_f * torch.sigmoid(gate_f) * up.float()
```

- The native dtype path is bitwise equal to its fp32 formula cast to the input dtype.
- Every backend must be bitwise batch/chunk/padding invariant: a coordinate is independent
  of unrelated rows.
- Optimized output is currently checked against the independent fp32 oracle with issue
  #108's elementwise threshold.
- CUDA/Triton cross-backend bitwise equality is not currently part of the implementation
  contract because their exponential implementations may differ. This must be confirmed
  with the integration owner before the contract is frozen.

## Build and validation

Build the CUDA backend with:

```bash
KERNEL_ALIGN_ACTIVATION_SM90=1 pip install --no-build-isolation -e .
```

Validate the Qwen3-8B TP-local width on an H100:

```bash
python scripts/check_operator.py --op swiglu --candidate cuda-sm90 \
  --device cuda --dtype bf16 --batch 1 --seq 4096 --intermediate-dim 6144 --arch-key sm90
python scripts/check_operator.py --op swiglu --candidate triton \
  --device cuda --dtype bf16 --batch 1 --seq 4096 --intermediate-dim 6144 --arch-key sm90
python -m pytest tests/test_swiglu.py tests/test_swiglu_forward_backends.py -v
```

The forward benchmark defaults to `[M_local, 6144]`:

```bash
python benchmarks/benchmark_swiglu.py --rows 4096 --width 6144
```

## Implementation files

- `rl_engine/kernels/ops/pytorch/activation/swiglu.py`
- `rl_engine/kernels/ops/cuda/activation/swiglu.py`
- `rl_engine/kernels/ops/triton/activation/swiglu.py`
- `csrc/cuda/activation/swiglu_sm90.cu`
- `tests/test_swiglu.py`
- `tests/test_swiglu_forward_backends.py`

Backward for the optimized CUDA and Triton backends is intentionally outside this
forward-stage implementation.
