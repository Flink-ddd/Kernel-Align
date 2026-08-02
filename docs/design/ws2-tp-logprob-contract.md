# WS2 TP-Aware Logprob Contract

Status: PR1 contract and dispatch metadata

Tracking and shared contracts:

- [#241: TP-aware deterministic logprob](https://github.com/RL-Align/RL-Kernel/issues/241)
- [#83: WS2 roadmap](https://github.com/RL-Align/RL-Kernel/issues/83)
- [#108: WS1 numerical contract](https://github.com/RL-Align/RL-Kernel/issues/108)
- [#111: WS2 cross-config alignment](https://github.com/RL-Align/RL-Kernel/issues/111)
- [#116: WS2 tolerance and drift-report format](https://github.com/RL-Align/RL-Kernel/issues/116)
- [Cross-config logprob drift contract](ws2_cross_config_logprob_drift_contract.md)

## Scope

This contract describes the logical inputs and deterministic reduction semantics for
selected-token log-probability under vocab-parallel tensor parallelism (TP):

```text
selected_logp[t] = logits[t, target[t]] - logsumexp_vocab(logits[t, :])
```

Under vocab-parallel TP each rank holds one vocabulary shard, so the vocabulary-wide
`logsumexp` requires a cross-rank reduction. This contract lets runtime dispatch reject a
backend whose numerical semantics do not match the requested layout.

This PR1 layer does not shard tensors, launch a collective, merge `(max, sumexp)` partial
states, or implement a kernel. The single-GPU harness registration, the deterministic
vocab-parallel TP reference, and the cross-config integration belong to later PRs in #241.

Context parallelism (CP) is a declared non-merge axis. CP partitions tokens, never the
vocabulary, so the logprob reduction spans TP vocab shards only. CP rank metadata is carried
for provenance and must never widen the merge.

## Contract Objects

`rl_engine.kernels.logprob_contract` defines:

- `LogprobContract`: role, logits dtype, mask, sharding, reduction, output surface, and
  LSE export, plus a rank-independent `cross_rank_fingerprint()`;
- `ShardingSpec`: per-rank vocab-shard bounds, padded-vs-real vocabulary, TP/CP rank
  metadata, and target-token ownership;
- `MaskSpec`: active-token mask and ignore index;
- `ReductionSpec`: fixed `(max, sumexp)` merge semantics and the requested determinism
  scope;
- `LogprobOutputSpec`: the output surface — fp32 selected logprob and fp32 vocab-domain
  LSE, replicated across the TP group;
- `LogprobBackendCapability`: the layouts and semantics a backend explicitly supports,
  including its mask modes and determinism scopes.

Construction performs validation immediately. A structurally valid contract means that the
request is complete and internally consistent; it does not mean that an installed backend can
materialize it.

`ShardingSpec.vocab_shard_bounds` lists every TP rank's half-open `[start, end)` vocab range
indexed by TP rank. The full table is required on every rank: it defines target ownership
and the fixed merge order without any collective, and it makes an incomplete or overlapping
partition a loud construction-time error instead of a silent runtime divergence.
`ShardingSpec.owner_rank(token_id)` resolves the unique owning rank for a real-vocab token
and rejects everything else.

`padded_vocab_size` is the shard-covered (weight) vocabulary; `real_vocab_size` is the
tokenizer vocabulary. Padding columns occupy `[real_vocab_size, padded_vocab_size)` and must
be excluded from the logsumexp by any conforming implementation. The two sizes are equal
when the vocabulary is unpadded.

Inactive tokens (prompt, padding, masked-out response positions) are excluded from every
drift aggregate and are exempt from the exactly-one-owner target gather; their targets may
legally hold `ignore_index`. `ignore_index` must not collide with the real vocabulary.

## Qwen3-8B TP=2 BF16 Example

```python
from rl_engine.kernels.logprob_contract import (
    LogprobContract,
    MaskSpec,
    ReductionSpec,
    ShardingSpec,
)

sharding = ShardingSpec(
    tp_rank=0,
    tp_world_size=2,
    vocab_shard_bounds=((0, 76032), (76032, 152064)),
    real_vocab_size=151936,
    padded_vocab_size=152064,
    cp_rank=0,
    cp_world_size=2,
)

contract = LogprobContract(
    role="train",
    dtype="bf16",
    mask=MaskSpec(
        num_tokens=8,
        active_mask=(False, False, True, True, True, True, True, False),
        ignore_index=-100,
    ),
    sharding=sharding,
    reduction=ReductionSpec(),
)
```

Each rank owns one contiguous vocab shard; the 128 padding columns at the end of rank 1's
shard are outside the real vocabulary and never contribute to the logsumexp. The two leading
prompt tokens and the trailing padding token are inactive.

## Reduction Semantics

The only PR1 reduction contract is:

```text
partial state: (local_max, local_sumexp), fp32
merge: max_sumexp
merge_axis: tp_vocab
order: global_vocab_shard_index
transport: all_gather
downcast_at: final_write
engine: in_op_reference
```

Every rank first masks every local column whose global id lies in
`[real_vocab_size, padded_vocab_size)` to `-inf`, so padding never contributes to the
logsumexp, then computes `m_l = max(local_logits)` and `s_l = sum(exp(local_logits - m_l))`
in fp32. The partials travel by all-gather (collectives are transport only, never a
numerical reduction), and every rank merges in fixed global vocab-shard index order:

```text
M = max_l(m_l)
S = sum_l(s_l * exp(m_l - M))
LSE = M + log(S)
selected_logp = target_logit - LSE
```

The selected target logit comes from a masked single-owner gather: exactly one rank holds
each active token's target column. Downcast happens only at the final write. Because the
merge order is fixed by shard index, the result is deterministic and reproducible at every
TP degree by construction. Cross-degree bitwise equality (TP=2 equal to TP=1, the #241 PR 3
acceptance target) requires one further condition: the entire reduction must follow a
global tile-level structure that is independent of TP partitioning — a fixed tile
decomposition of the vocabulary plus a fixed merge order and rescaling tree over those
tiles, identical at every TP degree, so that the TP degree only selects which rank computes
which tiles and never changes the floating-point grouping. A TP-degree-independent
decomposition inside each shard is not sufficient on its own, because shard boundaries
would still group the combines differently across degrees. Providing that global structure
is an obligation of the deterministic reference implementation; a backend without it is
still deterministic per degree, and its cross-degree drift is judged against the #108
tolerance table instead. The contract expresses this distinction as
`ReductionSpec.determinism_scope`: `cross_tp_bitwise` (the #241 target and the default)
versus `fixed_topology`. A backend declares the scopes it honors in
`LogprobBackendCapability.determinism_scopes`, and dispatch rejects a backend that cannot
honor the requested scope — prose obligations are not enough; the guarantee is part of the
typed contract.

A shard may lie entirely inside the padded region, and a row's local columns may all be
`-inf` after masking. The identity partial for these cases is defined as
`(m_l, s_l) = (-inf, 0)`: a partial with `s_l = 0` contributes nothing to the merge
regardless of its `m_l`, and implementations must use this identity directly rather than
evaluating `exp(-inf - (-inf))`, which would poison the merge with NaN.

Averaging per-rank logsumexp values or letting a collective reduce numerically is never
conformant, at either determinism scope.

The acceptable LSE and selected-token drift thresholds remain owned by #108, and drift
reports follow the #116 format. This contract does not introduce another tolerance table.
The selected-token metric remains the cross-config convention:

```text
dlogp = training-side recomputed logp - rollout-side old logp
```

computed over active response tokens only.

## Contract-Aware Dispatch

Legacy callers continue to use `KernelRegistry.get_op()`. WS2 callers use:

```python
result = kernel_registry.get_logprob_op(contract)
op = result.op
provenance = result.provenance
```

Dispatch considers only backends with a `LogprobBackendCapability`. It checks role, dtype,
TP/CP degree, padded-vs-real vocab masking, explicit active-mask support, vocab-domain LSE
export, and the requested determinism scope. An undeclared or incompatible backend is
skipped with an explicit rejection reason; there is no silent fallback.

`requested_backend` accepts a case-insensitive policy keyword (`auto` | `production` |
`reference`; default `auto`) or an exact, case-sensitive stable backend id. The keywords
select an implementation tier; determinism is not a tier — it is requested through
`ReductionSpec.determinism_scope`, so `requested_backend="deterministic"` raises a loud
error instead of silently matching nothing. Strictness comes from the contract's
capability checks, not from the policy string. A backend id may never shadow a reserved
keyword; capability construction rejects that. The provenance `fallback` flag reports only
capability or load rejections of policy-eligible candidates — a candidate excluded by the
caller's own policy never counts as a fallback, even if it would also have failed
capability checks.

WS2 dispatch resolves from its own candidate list, seeded from but decoupled from the legacy
`batch_invariant_logp` priority list: registering a TP-vocab backend for WS2 dispatch does
not change what legacy `get_op("batch_invariant_logp")` returns to WS1 callers.

The current WS1 batch-invariant logp implementations are single-shard (TP=1) references:
they accept full-vocabulary logits with ignore-index masking but carry no vocab-shard
metadata, no padded-vs-real vocab distinction, and no public vocab-domain LSE export. Strict
WS2 requests therefore fail clearly today. The later deterministic vocab-parallel reference
becomes selectable through `KernelRegistry.register_logprob_backend(backend, capability)`
by declaring a capability that truthfully describes those features; no controller branch or
silent fallback is required.

Successful dispatch provenance records:

- requested and actual backend ids;
- platform and fallback status;
- prior candidate rejection reasons;
- the complete dispatch-relevant contract, including shard bounds, padded and real vocab
  sizes, merge and output semantics, the explicit `cp_is_merge_axis: false` declaration,
  and the active-mask digest (`active_mask_sha256`) — the mask's identity without its
  per-token payload;
- the selected backend capability descriptor.

### Distributed dispatch safety

`get_logprob_op` resolves locally on each rank, so `requested_backend="auto"` is not
distributed-safe on its own: a load failure on one rank can resolve a different backend
than its peers, which for a collective-bearing implementation means divergent numerical
schedules or a deadlock. For `tp_world_size > 1` a caller must either request an exact
backend id or run a preflight agreement before any collective: all-gather the resolved
backend id together with `LogprobContract.cross_rank_fingerprint()` — a rank-independent
hash covering the shard-bounds table, vocab sizes, reduction/output semantics, and the
active-mask digest, excluding rank-local fields — and abort on any mismatch. Implementing
this preflight is an obligation of the #241 PR 3/PR 4 work; the backend invocation
protocol (how the contract and mask reach the implementation) is likewise defined there,
against this contract.

## Validation

Contract and dispatch behavior are covered by:

```bash
python -m pytest tests/test_logprob_contract.py -q
```

The tests include Qwen3-8B TP=2 BF16 construction with padded vocab, the TP=1/2/4 sweep
shapes, incomplete/overlapping shard-bound rejection, owner-rank resolution, active-mask and
ignore-index validation, fp32-accumulation and merge-semantics enforcement, undeclared
backend rejection, no incompatible fallback, and JSON-compatible provenance.
