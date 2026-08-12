# Batch-Invariant LogP

Batch-Invariant LogP computes selected token log-probabilities from already
materialized logits:

```text
out[row] = logits[row, target_ids[row]] - logsumexp(logits[row, :])
```

It targets RL post-training paths where policy log-probs are compared across
different packing, padding, and batch layouts. The key contract is
batch-invariance: for a fixed row of logits and target id, the result must not
change when that row is evaluated alone, at a different batch position, or with
different neighboring rows.

Unlike `linear_logp`, this operator does not fuse the LM-head projection. It
takes `[*, V]` logits as input and returns one selected log-probability per row.

## Entry Point

```python
from rl_engine.kernels.registry import kernel_registry

batch_invariant_logp = kernel_registry.get_op("batch_invariant_logp")

logp = batch_invariant_logp(
    logits,       # [B, T, V] or [N, V], differentiable
    target_ids,   # [B, T] or [N], int
    ignore_index=-100,
    validate=False,  # Triton fast path; use True to debug-check target range
)                # -> [B, T] or [N], float32

logp.sum().backward()  # gradients flow into logits only
```

## Backends

| Backend | Wrapper | Status |
| --- | --- | --- |
| CUDA (SM90 TMA) | `BatchInvariantLogpSM90Op` | Hopper TMA online-softmax forward. |
| Ascend (CANN) | `BatchInvariantLogpAscendOp` | Ascend C two-pass streaming forward; PyTorch-formula backward. |
| CUDA / ROCm (Triton) | `TritonBatchInvariantLogpOp` | Triton online-softmax forward and tile-wise backward. Requires a GPU tensor. |
| PyTorch native | `NativeBatchInvariantLogpOp` | FP32 reference path; CPU fallback and Triton-less fallback. |

Current dispatch:

```text
CUDA (Hopper, SM90 kernel compiled): CUDA (SM90 TMA) -> Triton -> PyTorch
CUDA / ROCm (otherwise):             Triton -> PyTorch
Ascend NPU:                          Ascend -> PyTorch
CPU:                                 PyTorch
```

The SM90 backend is hardware-gated: it is only inserted at the front of the
CUDA priority list when the extension exposes `_C.batch_invariant_logp_sm90`
(built with `KERNEL_ALIGN_FORCE_SM90=1`) on an SM90 device. On any other build
or device, dispatch is unchanged (Triton -> PyTorch).

The Ascend backend lives on the `npu` platform key and is available when the
extension exposes `_C_npu.batch_invariant_logp_ascend` (built with
`KERNEL_ALIGN_FORCE_ASCEND=1` on a CANN + torch_npu host; `npu-arch` defaults
to `dav-2201`, override with `KERNEL_ALIGN_ASCEND_ARCH`). When the extension is
not compiled, instantiation fails and dispatch falls through to PyTorch native.
bf16/fp32 NPU tensors run the Ascend C kernel; anything else (e.g. fp16)
silently falls back to the native op.

## Benchmarks

`benchmarks/benchmark_batch_invariant_logp.py` compares Native, Triton, and the
CUDA SM90 backend (forward latency and peak VRAM across a vocab sweep, bf16):

```bash
python benchmarks/benchmark_batch_invariant_logp.py
python benchmarks/benchmark_batch_invariant_logp.py --configs "4096,128256;8192,151936"
```

The CUDA column is only shown when the SM90 kernel is compiled in; otherwise the
benchmark reports Native vs Triton only.

### Measured results

Environment: NVIDIA H200 (Hopper, SM90, cc 9.0), CUDA 12.8 / `nvcc` 12.8.93,
PyTorch 2.11.0+cu128, `KERNEL_ALIGN_FORCE_SM90=1`. dtype bf16, 20 iters + 5
warmup. "MB" is peak extra device memory above baseline. Both tables are
reproduced by `benchmarks/benchmark_batch_invariant_logp.py --backward`.

**Forward**

| shape (N x V) | native ms | triton ms | cuda ms | cuda vs native | cuda vs triton | native MB | triton MB | cuda MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 x 32768 | 1.355 | 0.148 | 0.091 | 14.9x | 1.63x | 1536 | 0 | 0 |
| 4096 x 128256 | 5.011 | 0.567 | 0.324 | 15.5x | 1.75x | 6012 | 0 | 0 |
| 4096 x 151936 | 5.961 | 0.669 | 0.384 | 15.5x | 1.74x | 7122 | 0 | 0 |
| 8192 x 128256 | 9.991 | 1.056 | 0.597 | 16.7x | 1.77x | 12024 | 0 | 0 |

**Forward + backward**

| shape (N x V) | native ms | triton ms | cuda ms | cuda vs native | cuda vs triton | native MB | triton MB | cuda MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 x 32768 | 3.400 | 0.305 | 0.242 | 14.1x | 1.26x | 1536 | 256 | 256 |
| 4096 x 128256 | 12.581 | 1.117 | 0.871 | 14.5x | 1.28x | 6012 | 1002 | 1002 |
| 4096 x 151936 | 14.943 | 1.319 | 1.032 | 14.5x | 1.28x | 7122 | 1188 | 1188 |
| 8192 x 128256 | 25.036 | 2.144 | 1.684 | 14.9x | 1.27x | 12024 | 2004 | 2004 |

- Forward: ~1.7x vs Triton, ~15x vs native, with ~0 extra VRAM — the vocab is
  reduced to per-row scalars, so no `[N, V]` intermediate is materialized.
- Forward + backward: ~1.27x vs Triton, ~14x vs native, with memory equal to
  Triton. The backward's `[N, V]` cost is `grad_logits` itself (one gradient per
  input logit, unavoidable for any backend); the streamed backends avoid native's
  extra `[N, V]` `softmax` / `log_softmax` intermediates by recomputing from the
  saved per-row `lse`.

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `logits` | `[N, V]` / `[B, T, V]` / `[*lead, V]` | fp32 / fp16 / bf16 | Differentiable input; last dimension is vocab. |
| `target_ids` | `[N]` / `[B, T]` / `[*lead]` | int | Same leading shape as `logits`; non-ignored values in `[0, V)`. |
| `ignore_index` | scalar int | Python int | Default `-100`. Ignored rows output zero and receive zero gradient. |
| Output | `[N]` / `[B, T]` / `[*lead]` | float32 | Selected log-probability per row. |

`target_ids` is integer and non-differentiable. Gradients flow only into
`logits`.

## Reference Semantics

For non-ignored rows:

```python
logits_2d = logits.reshape(-1, logits.size(-1)).float()
target_1d = target_ids.reshape(-1).long()

log_probs = torch.log_softmax(logits_2d, dim=-1)
selected = torch.gather(
    log_probs,
    dim=-1,
    index=target_1d.unsqueeze(-1),
).squeeze(-1)

out = selected.reshape(target_ids.shape)
```

For ignored rows:

```text
target_ids[row] == ignore_index
out[row] = 0.0
grad_logits[row, :] = 0.0
```

Non-ignored target ids outside `[0, V)` are invalid. In particular,
`target=-1` is invalid unless `ignore_index=-1`.

The PyTorch native backend validates target ranges by default. The Triton
backend defaults to `validate=False` to avoid CUDA stream synchronization in
training hot paths. Use `validate=True` during debugging or in tests when
calling the Triton backend with untrusted targets.

## Batch-Invariance

The operator is designed so each row is computed independently:

- The PyTorch path reshapes to `[N, V]` and applies row-wise reductions.
- The Triton forward uses `grid=(num_tokens,)`, so one program owns exactly one
  row.
- Triton vocab traversal uses a fixed `_BLOCK_V=1024` and does not autotune by
  batch size.
- Triton forward scans vocab tiles left-to-right using online logsumexp.
- Triton backward uses `grid=(num_tokens, vocab_tiles)` and writes one row tile
  per program. It reuses the forward-saved per-row `lse`, so no backward
  reduction crosses row boundaries.
- The Ascend forward strides rows across blocks, so one AI core block owns
  exactly one row; the vocab is scanned left-to-right in fixed
  `TILE_LENGTH=4096` tiles with a two-pass (max, then sum-exp) fixed-order
  reduction.
- No atomic writes are used.

These constraints ensure the result for a row depends only on that row's logits
and target id, not on batch size, row position, or neighboring rows.

## Accuracy

Both backends accumulate reductions in float32 and return float32 outputs. Tests
compare against `torch.log_softmax(...).gather(...)` with dtype-appropriate
tolerances:

```text
fp32 forward: atol around 1e-5
fp16/bf16 forward: atol around 1e-4
fp16/bf16 backward: checked against fp32 reference with relaxed tolerance
```

CPU-vs-CUDA comparisons use tolerance-based checks; batch-invariance checks
within the same backend use exact equality where appropriate.

## Minimal Example

```python
import torch

from rl_engine.kernels.registry import kernel_registry

op = kernel_registry.get_op("batch_invariant_logp")

logits = torch.randn(2, 4, 300, device="cuda", dtype=torch.bfloat16)
target_ids = torch.randint(0, 300, (2, 4), device="cuda")
target_ids[0, 0] = -100

out = op(logits, target_ids, ignore_index=-100)
assert out.shape == target_ids.shape
assert out.dtype == torch.float32
assert out[0, 0].item() == 0.0

out.sum().backward()
```

## Tests

```bash
python -m pytest tests/test_batch_invariant_logp.py -q -rs
```

All backends (Native, Triton, SM90, Ascend) are tested in a single file.
Coverage includes: correctness, leading-shape preservation, batch-invariance
(bitwise), validation, ignore-index behavior, backward correctness, CUDA smoke
cases, registry dispatch, and Triton-specific fp32/fp16/bf16 correctness, large
vocab, backward gradient batch-invariance, and ignored-row zero gradients.

Triton tests skip when Triton or CUDA is unavailable. SM90 tests skip without a
Hopper build; Ascend tests skip without an NPU + `_C_npu` build. On Windows, run
via WSL/Linux with CUDA.

## Implementation Files

- `rl_engine/kernels/ops/pytorch/loss/batch_invariant_logp.py`
- `rl_engine/kernels/ops/triton/loss/batch_invariant_logp.py`
- `rl_engine/kernels/ops/cuda/loss/batch_invariant_logp.py`
- `rl_engine/kernels/ops/ascend/loss/batch_invariant_logp.py`
- `csrc/cuda/batch_invariant_logp_kernel_sm90.cu`
- `csrc/ascend/batch_invariant_logp_ascend.asc`
- `rl_engine/kernels/registry.py`
- `rl_engine/platforms/device.py`
- `tests/test_batch_invariant_logp.py`
- `benchmarks/benchmark_batch_invariant_logp.py`
