# WS2 Single-GPU Logprob Comparison Harness

This harness is the TP=1 registration and regression guard for issue #241. It compares
selected-token logprob implementations before any distributed communication is introduced.

## Contract

For each logical token row, every backend returns direct FP32 values:

```text
LSE  = logsumexp(logits[..., vocab])
logp = selected_logit - LSE
```

The harness uses the merged WS1 batch-invariant PyTorch implementation as its reference.
Reference logp is obtained through the unchanged production call, while reference LSE is
obtained through the diagnostic entry point. The TP=1 PyTorch candidate follows the same
core computation and must be bitwise equal. This is a regression guard, not new
distributed mathematics.

LSE drift is reported over every logical token row. Selected-token dlogp drift is reported
only over active response/action tokens. Both reports contain max, mean, p95, p99, and the
number of compared values.

## Exact Backend Selection

Supported backend names are:

- `pytorch`
- `triton`
- `cuda-sm90`

The comparison path does not use registry fallback. An explicitly requested backend must
run exactly or raise `LogprobBackendUnavailable`. In particular, `cuda-sm90` requires a
compiled SM90 extension, Hopper hardware, BF16/FP32 logits, and a compatible vocab row
stride. The production operator may retain its normal fallback behavior outside the
harness.

Each backend exposes a diagnostic-only `forward_with_lse` method. Existing production
calls remain unchanged:

```text
op(logits, target_ids)                  -> logp
op.forward_with_lse(logits, target_ids) -> (logp, lse)
```

## Usage

CPU TP=1 regression guard:

```bash
python scripts/compare_logprob.py \
  --candidate pytorch \
  --device cpu \
  --dtype fp32 \
  --batch 2 \
  --seq 16 \
  --vocab 257
```

GPU comparison:

```bash
python scripts/compare_logprob.py \
  --candidate triton \
  --candidate cuda-sm90 \
  --device cuda \
  --dtype bf16 \
  --batch 2 \
  --seq 16 \
  --vocab 151936
```

The command prints a structured JSON report containing input dtype/shape, active-token
count, TP world size, communication mode, requested and actual backends, direct-LSE
provenance, bitwise logp status, and LSE/dlogp drift statistics.

## Scope Boundary

This harness is intentionally single-GPU and records `tp_world=1` and
`communication=none`. It does not implement vocab-shard metadata, all-gather transport,
fixed-order cross-rank LSE merging, CP reconstruction, or distributed artifacts. Those
belong to the later PR3 and PR4 work in issue #241.

## Tests

```bash
python -m pytest tests/test_logprob_comparison.py -q
```

The focused tests cover bitwise TP=1 regression, direct LSE identity, active-token-only
percentiles, zero active tokens, invalid ignore-index usage, structured serialization,
generic operator-harness registration, exact GPU backend diagnostics, and fail-closed
backend provenance.
