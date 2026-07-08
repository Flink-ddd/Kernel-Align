# Gumbel-Softmax

Gumbel-Softmax provides differentiable sampling from token logits for rollout and
alignment experiments. It returns either soft probabilities or straight-through
one-hot samples while preserving gradients through the softmax path.

## Entry Point

```python
from rl_engine.kernels.registry import kernel_registry

op = kernel_registry.get_op("gumbel_softmax")
samples = op(logits, tau=1.0, hard=True)
```

Backend classes:

- `NativeGumbelSoftmaxOp`
- `TritonGumbelSoftmaxOp`

## Definition

For logits `x`, Gumbel noise `g`, and temperature `tau`:

```text
z = (x + g) / tau
y_soft = softmax(z)
```

When `hard=False`, the output is `y_soft`. When `hard=True`, the forward output
is one-hot at `argmax(y_soft)` and the backward pass uses the straight-through
softmax gradient.

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `logits` | `[..., V]` | fp32/fp16/bf16 | Floating-point tensor, `V` is vocab size. |
| `tau` | scalar | Python float | Must be positive. |
| `hard` | scalar | Python bool | Enables straight-through one-hot output. |
| `gumbels` | `[..., V]` or `None` | Floating point | Optional deterministic noise for tests/repro. |
| `seed` | scalar or `None` | Python int | Triton-only seed for backend-internal Gumbel noise when `gumbels=None`. |
| Output | `[..., V]` | Same as `logits` | Probabilities or one-hot rows over vocab. |

The Triton backend flattens leading dimensions to `[N, V]` and launches one
program per row. It supports vocab sizes up to 131072 in this implementation.

## Backends

| Backend | Status | Notes |
| --- | --- | --- |
| PyTorch | Reference | Uses PyTorch autograd end to end. |
| Triton | GPU optimized | Uses Triton forward, a no-grad hard-sampling fast path, and fused softmax backward. |

For deterministic correctness tests, pass the same precomputed `gumbels` tensor
to both backends. If `gumbels=None`, each backend samples its own Gumbel noise.
The Triton backend accepts `seed` to make backend-internal Gumbel generation
reproducible for smoke tests and benchmarks.

## Tests and Benchmarks

```bash
python -m pytest tests/test_gumbel_softmax.py
python benchmarks/benchmark_gumbel_softmax.py
```

The benchmark reports forward latency, forward+backward latency, speedup, and
peak forward VRAM for PyTorch and Triton on a single GPU.
