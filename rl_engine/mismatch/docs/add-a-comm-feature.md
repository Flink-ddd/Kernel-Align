<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 RL-Kernel Contributors -->

# Tutorial: add a communication feature

Mismatch exists because floating-point addition is not associative, and **the
accumulation order is almost entirely decided by collective communication**. So
collectives are not an implementation detail here — they are a first-class
declared object, and `gemm.forward_reduce`, `gemm.dgrad_reduce`,
`attn.cp_split_k_merge_order`, `logp.reduce_topology`, `moe.tp_forces_sp_reduce`
and `gemm.rollout_all_reduce_backend` are six instances of *one* semantic model.
Written separately they drift apart; written against `CollectiveContract` they
cannot.

This tutorial adds one: `gemm.forward_reduce` — RowParallel forward reduction,
where the training side does `all_reduce` without sequence parallelism and
`reduce_scatter + all_gather` with it, and the two paths accumulate in different
orders.

Read [add-a-kernel-factor.md](add-a-kernel-factor.md) first. The file layout, the
four arms, and the gates are identical; this tutorial only covers what is
different when the suspect is a collective.

## Step 1 · Describe the collective, do not just name it

`CollectiveContract` is the full numerical semantics of one collective. Every
field is load-bearing:

```python
from rl_engine.mismatch.schema import (
    CollectiveContract,
    CollectiveOp,
    DeterminismLevel,
    DowncastPoint,
    ParallelDim,
    Precision,
    ReductionOrder,
)

ORDERED_REDUCE_SCATTER = CollectiveContract(
    op=CollectiveOp.REDUCE_SCATTER,
    group=ParallelDim.TENSOR,
    group_size=2,
    reduction_order=ReductionOrder.GLOBAL_RANK_INDEX,
    accumulate_precision=Precision.FP32,
    downcast_at=DowncastPoint.FINAL_WRITE,
    determinism=DeterminismLevel.STABLE_ACROSS_TOPOLOGY,
    backend="rl_kernel",
)
```

| field | what it decides |
|---|---|
| `op` | which collective. `NONE` is a real value — record the single-device path explicitly rather than leaving it blank |
| `group` | which parallel dimension: `TENSOR` / `SEQUENCE` / `CONTEXT` / `EXPERT` / `PIPELINE` / `DATA` |
| `reduction_order` | **the direct root of mismatch.** `ARRIVAL` and `NCCL_ALGORITHM` are order-unstable; `GLOBAL_RANK_INDEX`, `GLOBAL_BLOCK_INDEX` and `GLOBAL_VOCAB_SHARD_INDEX` are the fixed orders |
| `accumulate_precision` + `downcast_at` | how much error the accumulation keeps. `PER_BLOCK` is the largest, `FINAL_WRITE` the smallest |
| `determinism` | how strong a reproducibility guarantee this implementation offers |
| `backend` | `"nccl"` / `"vllm_custom_ipc"` / `"mnnvl"` / `"transformer_engine"` / `"rl_kernel"` |

**One combination is rejected before anything runs**: claiming
`determinism=STABLE_ACROSS_TOPOLOGY` while reducing with `NCCL_ALGORITHM` or
`ARRIVAL` produces numbers that mean nothing. `reject_contradictory_factors()`
raises `ContradictoryFactor` at planning time — a static check, no GPU, no wasted
run. Do not weaken the claim to get past it; fix whichever half is wrong.

## Step 2 · Pin the contract so the planner can see it

The planner finds your contract through `declared_collectives(factor)`, which
collects `CollectiveContract` values out of the reference's `required_settings`.
So the contract is pinned like any other setting — with the channel that actually
delivers it:

```python
# operator_checks/gemm/_common.py
from rl_engine.mismatch.schema import (
    ExecutionPath,
    LibraryPin,
    ReferenceAuthority,
    ReferenceImplementation,
    RequiredSetting,
    SettingChannel,
)

DETERMINISTIC_REDUCE_REFERENCE = ReferenceImplementation(
    name="rl_kernel",
    tier=ReferenceAuthority.SELF_WRITTEN,   # justify this in the PR body
    training_impl="rl_engine.kernels.collectives.ordered_reduce_scatter",
    rollout_impl="rl_engine.kernels.collectives.ordered_reduce_scatter",
    covers_paths=(
        ExecutionPath.TRAINING_FULL_PREFILL,
        ExecutionPath.ROLLOUT_FULL_PREFILL,
    ),
    required_settings=(
        RequiredSetting(
            "forward_reduce_contract",
            ORDERED_REDUCE_SCATTER,          # the contract itself, pinned as a value
            SettingChannel.CALL_ARG,
            readback="module.last_collective_contract",
        ),
        RequiredSetting(
            "NCCL_ALGO", "Ring", SettingChannel.ENV_VAR,
            readback="os.environ", guards="nccl_algo_unpinned",
        ),
        RequiredSetting(
            "NCCL_PROTO", "Simple", SettingChannel.ENV_VAR,
            readback="os.environ", guards="nccl_algo_unpinned",
        ),
    ),
    pinned_libraries=(LibraryPin("torch", "2.6.0", container_digest="sha256:..."),),
)
```

Communication is one of the few places where `SELF_WRITTEN` is the honest answer:
neither TE nor FlashInfer exposes a reduction whose order is fixed across
topologies, so a deterministic `all_reduce` / `reduce_scatter + all_gather` has to
be written. Say that in the PR rather than leaving the tier unexplained.

The channel is not cosmetic — it decides when a setting can take effect and
therefore the rebind cost: `ENV_VAR` and `TORCH_GLOBAL` need a process restart,
`ENGINE_ARG` an engine rebuild, `CALL_ARG` nothing.

## Step 3 · The factor, with the collective in the comparison rules

```python
# operator_checks/gemm/factors/forward_reduce.py
from rl_engine.mismatch.operator_checks.gemm._common import DETERMINISTIC_REDUCE_REFERENCE
from rl_engine.mismatch.schema import (
    COLLECTIVE_CONTRACT,
    ComparisonRule,
    Evidence,
    FactorCategory,
    FailureMode,
    KnownPitfall,
    MismatchFactor,
    NoiseFloor,
    PolicyRole,
    Prerequisites,
    RebindCost,
    Switch,
)

FACTOR = MismatchFactor(
    id="gemm.forward_reduce",
    operator="gemm",
    category=FactorCategory.SHARDING_AND_REDUCTION,
    question=(
        "Does the RowParallel forward reduction differ because sequence "
        "parallelism rewrites all_reduce into reduce_scatter + all_gather?"
    ),
    switch=Switch(
        path="gemm.forward_reduce",
        rebind_cost=RebindCost.PROCESS_GROUP_REBUILD,
        applies_to=(PolicyRole.ROLLOUT, PolicyRole.TRAINING),
        allowed_values=("native", "rl_kernel"),
    ),
    comparison_rules={
        "collectives[0].op": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "collectives[0].reduction_order": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "collectives[0].accumulate_precision": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "collectives[0].group_size": ComparisonRule.MUST_MATCH_BITWISE,
        "collectives[0].backend": ComparisonRule.RECORD_ONLY,
    },
    prerequisites=Prerequisites(required_ops=("ordered_reduce_scatter",), min_gpu_count=2),
    required_evidence=(Evidence.EFFECTIVE_CONFIG_READBACK.value, COLLECTIVE_CONTRACT),
    reference=DETERMINISTIC_REDUCE_REFERENCE,
    call_sites=("attention.o_linear", "mlp.down_linear", "moe.output"),
    pitfalls=(
        KnownPitfall(
            id="nccl_algo_unpinned",
            mode=FailureMode.SILENT_FALSE_NEGATIVE,
            symptom="the reduction-order conclusion looks stable",
            actual_cause="NCCL picks ring or tree per run, so the conclusion is noise",
            guard="pin NCCL_ALGO/NCCL_PROTO and rerun; results must be bitwise identical",
            guard_runs_at=NoiseFloor.SHARDED_SINGLE_NODE,
        ),
    ),
)
```

Four things are specific to a comm factor:

- **Indexed paths reach into the collective**: `collectives[0].reduction_order`
  resolves through the tuple, so no operator-specific comparison code is needed.
- **`backend` is `RECORD_ONLY`.** Two backends may legitimately differ; what must
  agree is the *order*, not the library. Declaring `backend` as `MUST_MATCH_*`
  buries the real finding under a difference you already knew about.
- **`min_gpu_count=2` plus `PROCESS_GROUP_REBUILD`.** The factor is identically
  zero on one device, and `suggested_floor_is_lowest()` will tell you it belongs
  at `SHARDED_SINGLE_NODE` or above. Running it at the anchor floor wastes time.
- **`call_sites`** records that one factor acts in several physical places —
  attention's O-linear, the MLP's down-linear and the MoE output are all row
  parallel linears eating the same accumulation-order problem. One factor, three
  sites, not three factors.

`compare_contracts()` adds one check you do not declare: if the two sides'
collectives promise different `DeterminismLevel`s, it emits
`DETERMINISM_INCOMPATIBLE`. Comparing a topology-independent implementation
against one that is not even reproducible across runs measures the weaker side's
noise, not the gap between the sides.

## Step 4 · The cheapest strong check: rerun under different NCCL settings

An implementation claiming `STABLE_ACROSS_TOPOLOGY` must produce **bitwise
identical** results when NCCL is told to use a different algorithm. This needs no
cross-framework comparison at all — one side, rerun — which makes it the highest
value-per-minute check in the whole framework.

Declare it on the arm with `repeat_under`, the single exception to "one variant,
one execution":

```python
from rl_engine.mismatch.schema import ExpectedOutcome, FactorVariant, PolicyRole

TOPOLOGY_INVARIANCE = FactorVariant(
    name="both_reference",
    switch_values={"gemm.forward_reduce": "rl_kernel"},
    replace_on={
        PolicyRole.ROLLOUT: "rl_engine.kernels.collectives.ordered_reduce_scatter",
        PolicyRole.TRAINING: "rl_engine.kernels.collectives.ordered_reduce_scatter",
    },
    expected=ExpectedOutcome.BITWISE_IDENTICAL,
    repeat_under={"NCCL_ALGO": ("Ring", "Tree"), "NCCL_PROTO": ("Simple", "LL")},
    why="a fixed order must survive NCCL choosing a different algorithm",
)
```

The runner expands the cartesian product (four runs here) and asserts they agree
bitwise; a disagreement sets `SwitchStatus.ERROR`, which gate 1 turns into
`VARIANT_DID_NOT_APPLY` rather than a numeric verdict.

What this protects is **the premise of the self-check gate**: `both_reference` can
only anchor the other arms if the fixed-order implementation really did fix the
order. Without it, `REFERENCE_ITSELF_IS_BROKEN` is itself untrustworthy.

To use `repeat_under` you pass the arm explicitly through `MismatchFactor.variants`
(a non-empty `variants` tuple is returned as-is by `build_variants`), so declare
all four arms there when you need it on one of them.

## Step 5 · Observe what actually ran

```python
# operator_checks/gemm/adapter.py
def observe_collectives(role: PolicyRole, adapter: Any) -> tuple[CollectiveContract, ...]:
    """The collectives this operator really performed, this run."""
    return tuple(adapter.collective_trace())     # not what the config asked for
```

This feeds the `COLLECTIVE_CONTRACT` evidence item that the factor declares as
required, and gate 2 refuses a verdict without it. The distinction that matters:
`build_contract()` says what was *asked for*, `observe_collectives()` says what
*happened*. vLLM switching between custom IPC, MNNVL and NCCL by world size and
topology is exactly the case where the two disagree, and only the second one is
evidence.

## Declaring a rewrite

"Mathematically identical, unequal in floating point" is common enough to be its
own type. Two are already declared in `schema/collectives.py`:

```python
ALL_REDUCE_AS_SCATTER_GATHER  # all_reduce -> reduce_scatter + all_gather
ALL_TO_ALL_AS_GATHER_SLICE    # all_to_all -> all_gather + local slice
```

`preserves_bitwise` is `False` on both, and always will be — that is the entire
problem. Megatron applying the first rewrite when sequence parallelism is on,
while the rollout side does not, *is* `gemm.forward_reduce`. If your factor is
"one side rewrites this collective and the other does not", add a
`CollectiveRewrite` constant next to those two and reference it from the factor's
`question`, rather than describing the rewrite in prose.

## When you genuinely need a new enum value

Adding a `CollectiveOp`, `ParallelDim` or `ReductionOrder` value **is** a
framework change, and the plugin seam exists so that this is rare. It is
justified when the semantics cannot be expressed by the existing values — a new
parallel dimension, or a fixed order keyed on something that is neither rank,
block, nor vocab shard. It is *not* justified for a new backend (that is the
`backend` string) or a new library version (that is `LibraryPin`).

If you do add one:

1. Put it in `schema/collectives.py` with a comment saying what orders it.
2. Extend `_NON_DETERMINISTIC_ORDERS` in `pipeline/planner.py` if the new order is
   not stable, so the contradiction check keeps working.
3. Add a case to `tests/test_mismatch_framework.py`.
4. Say in the PR why an existing value could not express it.

## Checklist before the PR

- [ ] Every field of `CollectiveContract` is filled from what the code does, not from what the config requests.
- [ ] `determinism` and `reduction_order` do not contradict each other (the planner will tell you, but know why).
- [ ] `backend` is `RECORD_ONLY`; `group_size` is `MUST_MATCH_BITWISE`.
- [ ] `min_gpu_count` ≥ 2 and the suggested floor is `SHARDED_SINGLE_NODE` or above.
- [ ] `NCCL_ALGO` / `NCCL_PROTO` are pinned as `RequiredSetting`s **with readback**.
- [ ] Any topology-independence claim is backed by a `repeat_under` arm.
- [ ] `observe_collectives()` returns the trace, and `COLLECTIVE_CONTRACT` is in `required_evidence`.
- [ ] `call_sites` lists every physical place this one factor acts.
- [ ] A `SELF_WRITTEN` reference is justified against `SHARED_BACKEND` in the PR body.
