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
| CUDA / ROCm (Triton) | `TritonBatchInvariantLogpOp` | Triton online-softmax forward and tile-wise backward. Requires a GPU tensor. |
| PyTorch native | `NativeBatchInvariantLogpOp` | FP32 reference path; CPU fallback and Triton-less fallback. |

Current dispatch:

```text
CUDA / ROCm: Triton -> PyTorch
CPU:         PyTorch
```

A compiled CUDA backend and benchmark suite are planned follow-up work.
Benchmarks are not included in this PR; they will be added alongside the CUDA
backend in a subsequent PR.

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

All backends (Native, Triton) are tested in a single file. Coverage includes:
correctness, leading-shape preservation, batch-invariance (bitwise), validation,
ignore-index behavior, backward correctness, CUDA smoke cases, registry
dispatch, and Triton-specific fp32/fp16/bf16 correctness, large vocab, backward
gradient batch-invariance, and ignored-row zero gradients.

Triton tests skip when Triton or CUDA is unavailable. On Windows, run via
WSL/Linux with CUDA.

## Implementation Files

- `rl_engine/kernels/ops/pytorch/loss/batch_invariant_logp.py`
- `rl_engine/kernels/ops/triton/loss/batch_invariant_logp.py`
- `rl_engine/kernels/registry.py`
- `tests/test_batch_invariant_logp.py`
