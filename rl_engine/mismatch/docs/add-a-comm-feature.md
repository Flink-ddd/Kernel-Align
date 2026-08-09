<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 RL-Kernel Contributors -->

# Add a communication feature

Mismatch comes from floating-point addition not being associative, and the
accumulation order is almost entirely decided by collective communication — so
collectives are declared objects here, not an implementation detail. Six factors
across gemm, attention, logprob and MoE are instances of one semantic model;
written separately they drift apart.

This covers only what differs from
[add-a-kernel-factor.md](add-a-kernel-factor.md), which you should read first.
Worked example: `gemm.forward_reduce`.

## Describe the collective, do not just name it

```python
ORDERED_REDUCE_SCATTER = CollectiveContract(
    op=CollectiveOp.REDUCE_SCATTER,
    group=ParallelDim.TENSOR,
    group_size=TP_SIZE,
    reduction_order=ReductionOrder.GLOBAL_RANK_INDEX,
    accumulate_precision=Precision.FP32,
    downcast_at=DowncastPoint.FINAL_WRITE,
    determinism=DeterminismLevel.STABLE_ACROSS_TOPOLOGY,
    backend="rl_kernel",
)
```

| field | what it decides |
|---|---|
| `op` | which collective. `NONE` is a real value — record the single-device path rather than leaving it blank |
| `reduction_order` | **the direct root of mismatch.** `ARRIVAL` and `NCCL_ALGORITHM` are unstable; the `GLOBAL_*_INDEX` orders are fixed |
| `accumulate_precision` + `downcast_at` | how much error the accumulation keeps |
| `determinism` | how strong a reproducibility guarantee this offers |
| `backend` | `nccl` / `vllm_custom_ipc` / `mnnvl` / `transformer_engine` / `rl_kernel` |

**One combination is rejected before anything runs**: claiming
`STABLE_ACROSS_TOPOLOGY` while reducing with `NCCL_ALGORITHM` or `ARRIVAL`
produces numbers that mean nothing. Do not weaken the claim to get past
`reject_contradictory_factors()`; fix whichever half is wrong.

## Pin the contract so the planner can see it

`declared_collectives()` finds your contract among the reference's
`required_settings`, so it is pinned like any other setting, with the channel
that actually delivers it:

```python
required_settings=(
    RequiredSetting("forward_reduce_contract", ORDERED_REDUCE_SCATTER,
                    SettingChannel.CALL_ARG,
                    readback="module.last_collective_contract"),
    RequiredSetting("NCCL_ALGO", "Ring", SettingChannel.ENV_VAR,
                    readback="os.environ", guards="nccl_algo_unpinned"),
)
```

Communication is one of the few places where `SELF_WRITTEN` is the honest
answer: neither TE nor FlashInfer exposes a reduction whose order is fixed across
topologies. Say that in the PR rather than leaving the tier unexplained.

## What differs in the factor declaration

See `operator_checks/gemm/factors/forward_reduce.py`. Four things are specific to
a comm factor:

- **Indexed paths reach into the collective** — `collectives[0].reduction_order`
  resolves through the tuple, so no operator-specific comparison code is needed.
- **`backend` is `RECORD_ONLY`.** Two backends may legitimately differ; what must
  agree is the order. Comparing it buries the real finding.
- **`min_gpu_count=2` and `PROCESS_GROUP_REBUILD`.** The factor is identically
  zero on one device and belongs at `SHARDED_SINGLE_NODE` or above.
- **`call_sites`** records one factor acting in several physical places —
  attention's O-linear, the MLP's down-linear and the MoE output are all row
  parallel linears with the same accumulation-order problem. One factor, three
  sites.

`compare_contracts()` adds a check you do not declare: two sides promising
different `DeterminismLevel`s emit `DETERMINISM_INCOMPATIBLE`, because comparing
against something not reproducible across runs measures the weaker side's noise
rather than the gap.

## The cheapest strong check

An implementation claiming `STABLE_ACROSS_TOPOLOGY` must produce bitwise
identical results when NCCL uses a different algorithm. No cross-framework
comparison, just reruns — the highest value per minute in the framework.

```python
repeat_under={"NCCL_ALGO": ("Ring", "Tree"), "NCCL_PROTO": ("Simple", "LL")}
```

The runner expands the cartesian product and asserts bitwise agreement; a
disagreement sets `ERROR`, which gate 1 turns into `VARIANT_DID_NOT_APPLY`.

This protects **the premise of the self-check gate**: `both_reference` can only
anchor the other arms if the fixed-order implementation really did fix the order.
Without it, `REFERENCE_ITSELF_IS_BROKEN` is itself untrustworthy.

To use `repeat_under` you pass the arms explicitly through
`MismatchFactor.variants` — a non-empty tuple is returned as-is by
`build_variants`.

## Observe what actually ran

`build_contract()` says what was asked for; `observe_collectives()` says what
happened. vLLM switching between custom IPC, MNNVL and NCCL by world size and
topology is exactly where the two disagree, and only the second is evidence. It
feeds `COLLECTIVE_CONTRACT`, which gate 2 requires.

## Rewrites

"Mathematically identical, unequal in floating point" is its own type;
`schema/collectives.py` declares two:

```python
ALL_REDUCE_AS_SCATTER_GATHER  # all_reduce -> reduce_scatter + all_gather
ALL_TO_ALL_AS_GATHER_SLICE    # all_to_all -> all_gather + local slice
```

`preserves_bitwise` is `False` on both and always will be — that is the entire
problem. Megatron applying the first with sequence parallelism on while the
rollout side does not *is* `gemm.forward_reduce`. If your factor is "one side
rewrites this collective", add a constant next to those two rather than
describing the rewrite in prose.

## Adding an enum value

A new `CollectiveOp`, `ParallelDim` or `ReductionOrder` **is** a framework
change. Justified when the semantics cannot be expressed by the existing values —
a new parallel dimension, or a fixed order keyed on something that is neither
rank, block, nor vocab shard. Not justified for a new backend (that is the
`backend` string) or a new library version (that is `LibraryPin`).

If you do add one: put it in `schema/collectives.py`; extend
`_NON_DETERMINISTIC_ORDERS` in `pipeline/planner.py` if the order is not stable,
so the contradiction check keeps working; add a test; and say in the PR why an
existing value could not express it.

## Checklist

- [ ] Every `CollectiveContract` field comes from what the code does, not what the config requests.
- [ ] `determinism` and `reduction_order` do not contradict each other.
- [ ] `backend` is `RECORD_ONLY`; `group_size` is `MUST_MATCH_BITWISE`.
- [ ] `min_gpu_count` ≥ 2, floor `SHARDED_SINGLE_NODE` or above.
- [ ] `NCCL_ALGO` / `NCCL_PROTO` pinned **with readback**.
- [ ] Any topology-independence claim is backed by a `repeat_under` arm.
- [ ] `observe_collectives()` returns the trace; `COLLECTIVE_CONTRACT` is required evidence.
- [ ] `call_sites` lists every place this one factor acts.
