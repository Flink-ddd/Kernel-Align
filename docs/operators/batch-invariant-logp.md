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
| CUDA / ROCm (Triton) | `TritonBatchInvariantLogpOp` | Triton online-softmax forward and tile-wise backward. Requires a GPU tensor. |
| PyTorch native | `NativeBatchInvariantLogpOp` | FP32 reference path; CPU fallback and Triton-less fallback. |

Current dispatch:

```text
CUDA (Hopper, SM90 kernel compiled): CUDA (SM90 TMA) -> Triton -> PyTorch
CUDA / ROCm (otherwise):             Triton -> PyTorch
CPU:                                 PyTorch
```

The SM90 backend is hardware-gated: it is only inserted at the front of the
CUDA priority list when the extension exposes `_C.batch_invariant_logp_sm90`
(built with `KERNEL_ALIGN_FORCE_SM90=1`) on an SM90 device. On any other build
or device, dispatch is unchanged (Triton -> PyTorch).

## Tensor Parallel

`VocabParallelLogprobOp`
(`rl_engine/kernels/ops/pytorch/loss/vocab_parallel_logp.py`)
defines a cross-TP bitwise contract for TP=1, TP=2, and TP=4 when
`num_vocab_tiles` is fixed and every vocabulary-shard boundary is tile-aligned.
The complete BF16 CUDA/NCCL validation matrix for this contract is tracked by
issue #241 PR4.

1. Split the padded vocabulary into `num_vocab_tiles` fixed tiles.
2. Each rank computes fp32 `(max, sumexp)` for the tiles it owns. Every tile
   is reduced as the same contiguous `[n, tile]` shape, on any rank.
3. All tile partials are shared with `all_gather`. The collective only moves
   bytes; it never does math, so it cannot round anything.
4. Every rank merges all tiles in the same fixed order, over the same
   `[n, num_vocab_tiles]` shape. `LSE = M + log(sum(s_t * exp(m_t - M)))`.
5. The target logit is copied from the rank that owns it (never summed).
6. `logp = target_logit - LSE`. Inactive rows become `0.0`.

Usage goes through the contract-aware entry point:

```python
from rl_engine.kernels.registry import kernel_registry

result = kernel_registry.get_logprob_op(contract)   # LogprobContract from
op = result.op                                      # rl_engine.kernels.logprob_contract
logp, lse = op(local_logits, target_ids, contract=contract, tp_group=tp_group)
```

### Vime CP=2 runtime provider

The optional Vime adapter is owned by RL-Kernel and can be selected without
patching Megatron or vLLM:

```text
--selected-logprob-provider rl_engine.integrations.vime.logp.provider
--selected-logprob-provider-mode strict
```

Vime passes the local `[T, V_local]` logits, shifted targets, TP subgroup,
and CP row-ownership metadata. The provider builds the same `LogprobContract`
used by the distributed report, dispatches the explicit
`pytorch-vocab-parallel-logp-ws2` backend, and returns selected logp as `[T, 1]`.
When entropy is requested, it uses the same fixed TP-rank order and returns
full-vocabulary entropy for the existing loss surface. CP rank/layout are
recorded in provenance and never participate in the vocabulary LSE merge.

The provider fails closed for undeclared real/padded vocabulary sizes, TP/CP
metadata mismatches, unsupported top-p replay masks, and backend fallback.
`auto` mode may then use Vime's native path; `strict` mode reports the
configuration error. This adapter does not import Vime.

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

## TP=1 Comparison Harness

The single-GPU comparison harness is the TP=1 registration and regression guard
for issue #241. It uses the batch-invariant PyTorch implementation as the
reference and compares exact `pytorch`, `triton`, or `cuda-sm90` backends before
distributed communication is introduced.

Each backend exposes a diagnostic-only entry point while the production contract
remains unchanged:

```text
op(logits, target_ids)                  -> logp
op.forward_with_lse(logits, target_ids) -> (logp, lse)
```

The harness reports LSE drift over every logical token row and selected-logprob
drift over active response/action tokens only. Drift summaries contain max,
mean, p95, p99, and the number of compared values. Reports also record requested
and actual backends, implementation, direct-LSE provenance, input shape and
dtype, `tp_world=1`, and `communication=none`.

Backend selection is exact and does not use registry fallback. In particular,
an explicit `cuda-sm90` comparison fails unless the compiled SM90 extension,
Hopper hardware, input dtype, and vocab row stride satisfy the kernel contract.

Run the PyTorch TP=1 guard directly from the kernel-specific testing module:

```bash
python rl_engine/testing/logprob_comparison.py \
  --candidate pytorch \
  --device cpu \
  --dtype fp32 \
  --batch 2 \
  --seq 16 \
  --vocab 257
```

On a GPU, repeat `--candidate` to compare multiple exact backends:

```bash
python rl_engine/testing/logprob_comparison.py \
  --candidate triton \
  --candidate cuda-sm90 \
  --device cuda \
  --dtype bf16 \
  --batch 2 \
  --seq 16 \
  --vocab 151936
```

The command writes structured JSON to stdout and routes backend diagnostics to
stderr. The harness does not implement vocab sharding, collective communication,
cross-rank LSE merging, or CP reconstruction.

### SM90 validation

SM90 validation requires a Hopper GPU, CUDA-enabled PyTorch, and an `nvcc`
toolkit matching `torch.version.cuda`. Build the extension with:

```bash
export FORCE_CUDA=1
export KERNEL_ALIGN_FORCE_SM90=1
export TORCH_CUDA_ARCH_LIST="9.0+PTX"

python -m pip install --no-build-isolation --no-deps -e .
```

Run the focused harness tests, the complete operator suite, and an explicit
SM90 comparison:

```bash
python -m pytest \
  tests/test_logprob_comparison.py \
  tests/test_operator_inputs.py \
  tests/test_op_checks.py -q

python -m pytest tests/test_batch_invariant_logp.py -q

python rl_engine/testing/logprob_comparison.py \
  --candidate cuda-sm90 \
  --device cuda \
  --dtype bf16 \
  --batch 2 \
  --seq 16 \
  --vocab 151936 \
  --prompt-tokens 8 \
  --seed 241
```

The PR2 path was validated on an NVIDIA H800 PCIe with PyTorch 2.11.0+cu128,
CUDA 12.8, and Triton 3.6.0. The focused tests passed 41 cases and the complete
batch-invariant suite passed 67 cases. For BF16 shape `[2, 16, 151936]`, both
LSE and active-token dlogp had maximum absolute drift
`9.5367431640625e-07` against the PyTorch reference, with no backend fallback.

## Distributed WS2 Drift Report

The issue #241 PR4 runner materializes one TP/CP topology per `torchrun`
invocation. TP partitions the vocabulary and is the only numerical merge axis;
CP partitions token rows and is recorded in provenance without participating in
the vocab-domain LSE merge. For global rank `r`:

```text
tp_rank = r % tp_world_size
cp_rank = r // tp_world_size
```

Every case generates the same seeded FP32 logical logits, targets, and active
mask. The candidate receives a BF16 token/vocab shard through the explicit
`pytorch-vocab-parallel-logp-ws2` backend, while the independent oracle computes
`torch.logsumexp` over the complete real-vocab FP32 token slice. Distributed
dispatch rejects `auto`, capability fallback, topology mismatches, non-tileable
vocabularies, and incomplete materialization.

Reports follow the issue #116 fields and contain per-rank and aggregate LSE and
active-token dlogp summaries: max/mean/p95/p99 absolute drift, max relative
drift, worst global token position, target id, target owner rank, #108 tolerance,
and pass/fail. Provenance includes TP/CP topology, dtype, shard bounds, backend
capability, contract fingerprint, reduction spec, merge order, transport, and
the exact launch command. Replicated TP outputs are checked bitwise before one
representative per CP shard is included in aggregate statistics.

Print the scoped TP=1/2/4 x CP=1/2 launch matrix without starting workers:

```bash
python rl_engine/testing/distributed_logprob_comparison.py \
  --plan \
  --device cuda \
  --dtype bf16 \
  --output artifacts/ws2-logprob/report.json
```

Run one TP=2, CP=2 Qwen3-vocab case on four local GPUs:

```bash
torchrun --standalone --nproc-per-node=4 \
  rl_engine/testing/distributed_logprob_comparison.py \
  --tp 2 \
  --cp 2 \
  --dtype bf16 \
  --backend pytorch-vocab-parallel-logp-ws2 \
  --real-vocab 151936 \
  --padded-vocab 151936 \
  --num-vocab-tiles 64 \
  --batch 2 \
  --seq 16 \
  --prompt-tokens 8 \
  --output artifacts/ws2-logprob/tp2-cp2.json
```

The full matrix requires up to eight ranks for TP=4, CP=2. CPU/Gloo cases are
available for topology and artifact validation; the scoped numerical gate is
BF16 on CUDA/NCCL.

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

All production backends are tested in a single file. Coverage includes
correctness, leading-shape preservation, batch-invariance (bitwise), validation,
ignore-index behavior, backward correctness, CUDA smoke cases, registry
dispatch, and Triton-specific fp32/fp16/bf16 correctness, large vocab, backward
gradient batch-invariance, and ignored-row zero gradients. The focused
`tests/test_logprob_comparison.py` suite covers TP=1 bitwise regression, direct
LSE identity, active-token drift statistics, structured serialization, exact
backend diagnostics, and fail-closed provenance.
`tests/test_distributed_logprob_comparison.py` covers topology planning, TP/CP
rank mapping, token/vocab sharding, explicit backend materialization, #116 JSON
artifacts, and a real four-process TP=2, CP=2 Gloo smoke case.

Triton tests skip when Triton or CUDA is unavailable. On Windows, run via
WSL/Linux with CUDA.

## Implementation Files

- `rl_engine/kernels/ops/pytorch/loss/batch_invariant_logp.py`
- `rl_engine/kernels/ops/triton/loss/batch_invariant_logp.py`
- `rl_engine/kernels/ops/cuda/loss/batch_invariant_logp.py`
- `csrc/cuda/batch_invariant_logp_kernel_sm90.cu`
- `rl_engine/kernels/registry.py`
- `tests/test_batch_invariant_logp.py`
- `tests/test_logprob_comparison.py`
- `rl_engine/testing/logprob_drift.py`
- `rl_engine/testing/distributed_logprob_comparison.py`
- `tests/test_distributed_logprob_comparison.py`
- `benchmarks/benchmark_batch_invariant_logp.py`
- `rl_engine/kernels/ops/pytorch/loss/vocab_parallel_logp.py`
- `rl_engine/kernels/logprob_contract.py`
- `tests/test_vocab_parallel_logp.py`
