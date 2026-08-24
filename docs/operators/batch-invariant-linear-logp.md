# Batch-Invariant Fused Linear LogP

Batch-Invariant Fused Linear LogP computes the selected token log-probability
directly from hidden states and LM-head weights:

```text
logp[row] = log_softmax(hidden[row] @ weight.T + bias)[target_ids[row]]
```

The operator does not materialize the full `[N, V]` logits tensor. Unlike the
throughput-oriented [`linear_logp`](linear-logp.md), its floating-point reduction
topology is independent of the token-row count `N`. This gives a bitwise
single-card batch-invariance contract for rollout and train-inference alignment.

## Prerequisite

Build the extension on a Hopper host after installing CUDA-enabled PyTorch:

```bash
RL_KERNEL_REQUIRE_EXT=1 KERNEL_ALIGN_FORCE_SM90=1 \
  python -m pip install --no-build-isolation -e .
```

## Entry Point

```python
import torch

from rl_engine.kernels.registry import kernel_registry

op = kernel_registry.get_op("batch_invariant_linear_logp", device="cuda")

with torch.no_grad():
    logp = op(
        hidden,          # [B, S, D] or [N, D], bf16
        lm_head_weight,  # [V, D], bf16
        target_ids,      # [B, S] or [N], integer
        bias=None,       # optional [V]
        validate=False,  # async; invalid targets produce NaN
    )                   # [B, S] or [N], float32
```

## Contract

For fixed `hidden[row]`, `weight`, `target_ids[row]`, and optional `bias`, the
output bits do not depend on:

- batch size or flattened token-row count;
- the row's position in the batch;
- neighboring row contents;
- splitting the batch dimension into chunks and concatenating the results;
- repeated execution on the same SM90 device and software stack.

The contract is stronger than repeated-run determinism. It does not promise
bitwise equality across GPU architectures, CUDA versions, compiler versions,
different vocabulary sizes, or different hidden dimensions.

## Why the Existing Fused Path Is Separate

The regular SM90 `linear_logp` path chooses the number of split-V CTAs using
the number of row blocks and the device SM count. That occupancy heuristic is
useful for throughput, but changing `N` can change both vocab partitioning and
the final log-sum-exp merge order.

The batch-invariant entry point instead uses:

- fixed `BM=256`, `BN=64`, and `BK=32` tiles;
- fixed ascending hidden-dimension traversal with FP32 tensor-core accumulation;
- no split-K reduction;
- a split-V schedule derived only from `V`, capped at 32 contiguous ranges;
- an ascending, fixed-order merge of per-split online-softmax states.

Both entry points reuse the same validated TMA + `mma.sync` device kernel. Their
host launch policies are intentionally distinct so performance tuning cannot
silently weaken the batch-invariance contract.

## Tensor Contract

| Value | Shape | Dtype | Device and layout |
| --- | --- | --- | --- |
| `hidden` | `[*lead, D]`, at least 2-D, non-empty | BF16 | Hopper CUDA device; non-contiguous layouts are copied |
| `lm_head_weight` | `[V, D]`, `V > 0` | BF16 | Same device; non-contiguous layouts are copied |
| `target_ids` | `[*lead]` | `uint8`, `int8`, `int16`, `int32`, or `int64` | Same device; converted to contiguous int32 without wrapping extreme int64 values |
| `bias` | Optional `[V]` | FP16, BF16, or FP32 | Same device; consumed as FP32 |
| output | `[*lead]` | FP32 | Same device |

`D` must be positive and divisible by 32. A contiguous tensor can still have a
misaligned non-zero storage offset; the CUDA owner detects this and conditionally
clones only the affected TMA input to a 16-byte-aligned allocation.

Target IDs must be in `[0, V)`. The default `validate=False` stays asynchronous
and writes `NaN` for an invalid row instead of silently returning a finite but
wrong log-probability. Pass `validate=True` at trust boundaries to raise a
detailed `ValueError`; that diagnostic mode performs a GPU-to-CPU synchronization.
Padding and ignore-index values should normally be filtered before the call.

Backward, tensor parallelism, and fallback are deliberately unsupported. The
wrapper rejects differentiable execution while autograd is enabled, and registry
dispatch fails closed outside Hopper or when the compiled symbol is absent.

Model parameters may still have `requires_grad=True` during rollout; call the
operator inside `torch.no_grad()` or `torch.inference_mode()`.

## Memory

The forward never stores `[N, V]` logits or probabilities. Its partial-reduction
workspace is:

```text
3 * num_vocab_splits * N * sizeof(float)
```

for the partial max, exponential sum, and selected target logit. The workspace
grows with the number of vocab splits until the 32-split cap; its worst-case bound
is therefore `O(N)` and independent of `V`. The entry point also allocates FP32
`logp` and `lse` outputs (`2 * N` values), `O(N)` target conversion/sentinel
buffers for non-int32 or non-contiguous IDs, and up to `V` FP32 values when a
non-FP32 or non-contiguous bias must be converted. Layout or TMA-alignment fixes
can additionally create conditional input copies; aligned contiguous inputs stay
zero-copy.

## Accuracy

The hidden projection uses BF16 tensor-core inputs with FP32 accumulation. The
online softmax and split-state merge are FP32. Correctness tests compare against:

```python
from rl_engine.kernels.ops.pytorch.loss.linear_logp import NativeLinearLogpOp

expected = NativeLinearLogpOp().forward_fp32(hidden, weight, target_ids, bias)
```

Reference parity is tolerance-based because the reference uses a different GEMM
and log-sum-exp reduction. Batch-invariance checks within the SM90 backend use
exact `torch.equal` comparisons.

## Tests

```bash
RL_KERNEL_REQUIRE_EXT=1 \
  python -m pytest tests/test_batch_invariant_linear_logp.py -q -rs
```

The explicit extension gate prevents a Hopper build that omitted the SM90
symbol from reporting these feature checks as hardware skips. Ordinary test
runs without that flag still skip the optional suite when the symbol is absent.

GPU coverage includes:

- correctness with and without bias;
- repeated-run bitwise determinism;
- a fixed probe row at multiple batch positions;
- unrelated neighboring-row noise;
- batch size 1 versus 4096 rows;
- full-batch versus chunk sizes around `BM=256`;
- hidden-tile traversals from `D=32` through production-style `D=4096`;
- vocab-tile and split-cap boundaries at `V=63/64/65` and `V=2048/2049`;
- a production-style `V=50257` with edge and partial vocab tiles;
- strided public inputs and contiguous inputs with misaligned storage offsets;
- adversarial mutation, temporary copies, launch, and dependent consumption on a non-default stream;
- supported bias dtypes and rejection of unsupported bias dtypes;
- every supported target dtype, extreme int64 sentinels, asynchronous invalid-target
  `NaN`, opt-in range errors, and the forward-only boundary.

The batch-size test uses a vocabulary large enough to cross the adaptive split-V
boundary in the regular fused path.

The shared operator checker can run forward correctness on Hopper:

```bash
python scripts/check_operator.py \
  --op batch_invariant_linear_logp \
  --candidate cuda-sm90 \
  --device cuda \
  --arch-key sm90 \
  --dtype bf16 \
  --batch 4 \
  --seq 32 \
  --normalized-dim 128 \
  --vocab 4096
```

Do not pass `--check-grad`; backward is outside this operator's contract.

## Benchmark

```bash
python benchmarks/benchmark_batch_invariant_linear_logp.py
python benchmarks/benchmark_batch_invariant_linear_logp.py \
  --configs "4096,4096,151936"
```

This is a manual comparative microbenchmark over the same logical values, so it
is intentionally separate from the profiler's single-workload registry. The two
fused symbols share prepared BF16 tensors and exclude Python validation from
timing. The materialized batch-invariant LM-head + logp reference uses FP32 copies
created after the fused measurements. Each fused path reports maximum absolute
error against that reference; memory columns are incremental allocations above
each path's prepared-input baseline. Default cases include `N=1`, `32`, and `256`
to expose the fixed split-V schedule's small-batch occupancy tradeoff. Custom
configs require `D % 32 == 0` for the fused kernels and `V % 4 == 0` so the
materialized FP32 logp comparator remains on its SM90 TMA backend.

## Implementation Files

- `csrc/cuda/fused_linear_logp_sm90.cu`
- `csrc/ops.cpp`
- `rl_engine/kernels/ops/cuda/loss/batch_invariant_linear_logp.py`
- `rl_engine/kernels/registry.py`
- `tests/test_batch_invariant_linear_logp.py`
- `benchmarks/benchmark_batch_invariant_linear_logp.py`
