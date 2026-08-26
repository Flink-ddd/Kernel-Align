# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Values shared by logprob's factors: contracts, shard maps, the reference.

The swap reference is WS2's deterministic vocab-parallel selected-logprob
(issue #241 PR3): per-shard (max, sumexp) partials merged in global vocab-shard
order, fp32 accumulation, one downcast at the final write. The sweep factors
need no reference and scan a parameter instead.
"""

from __future__ import annotations

from rl_engine.mismatch.schema import (
    CollectiveContract,
    CollectiveOp,
    DeterminismLevel,
    DowncastPoint,
    ExecutionPath,
    LibraryPin,
    ParallelDim,
    Precision,
    ReductionOrder,
    ReferenceAuthority,
    ReferenceImplementation,
    RequiredSetting,
    SettingChannel,
)

# vLLM computes logits at the model dtype; Megatron can keep the head in fp32.
HEAD_DTYPES: dict[str, Precision] = {
    "bf16": Precision.BF16,
    "fp32": Precision.FP32,
}

DOWNCAST_POINTS: dict[str, DowncastPoint] = {
    "final_write": DowncastPoint.FINAL_WRITE,
    "per_partial": DowncastPoint.PER_PARTIAL,
}

TP_SIZE = 2

QWEN3_REAL_VOCAB = 151936
QWEN3_PADDED_VOCAB = 152064


def even_vocab_shard_bounds(padded_vocab: int, tp_world_size: int) -> tuple[tuple[int, int], ...]:
    """The even split both frameworks produce when padding already divides evenly.

    MCore's ``VocabUtility`` and vLLM's ``_get_indices`` can still disagree once
    per-shard padding rules differ, which is why the effective map is read back
    per side rather than assumed equal.
    """

    shard = padded_vocab // tp_world_size
    return tuple(
        (rank * shard, padded_vocab if rank == tp_world_size - 1 else (rank + 1) * shard)
        for rank in range(tp_world_size)
    )


# WS2 reference: all_gather the per-shard (max, sumexp) partials, then every
# rank merges them locally in vocab-shard-index order. The gather concatenates
# by rank index, so the merge order is fixed regardless of NCCL's choices.
REFERENCE_LSE_MERGE = CollectiveContract(
    op=CollectiveOp.ALL_GATHER,
    group=ParallelDim.TENSOR,
    group_size=TP_SIZE,
    reduction_order=ReductionOrder.GLOBAL_VOCAB_SHARD_INDEX,
    accumulate_precision=Precision.FP32,
    downcast_at=DowncastPoint.FINAL_WRITE,
    determinism=DeterminismLevel.STABLE_ACROSS_TOPOLOGY,
    backend="rl_kernel",
)

# Megatron's vocab-parallel cross entropy: all_reduce of the partial max and
# partial sumexp over the TP group, ordered by whatever NCCL picks.
NATIVE_TRAINING_LSE_MERGE = CollectiveContract(
    op=CollectiveOp.ALL_REDUCE,
    group=ParallelDim.TENSOR,
    group_size=TP_SIZE,
    reduction_order=ReductionOrder.NCCL_ALGORITHM,
    accumulate_precision=Precision.FP32,
    downcast_at=DowncastPoint.FINAL_WRITE,
    determinism=DeterminismLevel.NONE,
    backend="nccl",
)

# vLLM: gather the full logits to one rank and reduce locally in one pass --
# a different floating-point association than merging per-shard partials.
NATIVE_ROLLOUT_LSE_MERGE = CollectiveContract(
    op=CollectiveOp.ALL_GATHER,
    group=ParallelDim.TENSOR,
    group_size=TP_SIZE,
    reduction_order=ReductionOrder.NCCL_ALGORITHM,
    accumulate_precision=Precision.FP32,
    downcast_at=DowncastPoint.FINAL_WRITE,
    determinism=DeterminismLevel.NONE,
    backend="vllm_custom_ipc",
)

# SELF_WRITTEN because neither TE nor FlashInfer offers a vocab-parallel
# selected-logprob whose partial-LSE merge order is fixed across topologies.
DETERMINISTIC_LSE_REFERENCE = ReferenceImplementation(
    name="rl_kernel",
    tier=ReferenceAuthority.SELF_WRITTEN,
    training_impl="rl_engine.kernels.ops.pytorch.loss.vocab_parallel_logp.VocabParallelLogprobOp",
    rollout_impl="rl_engine.kernels.ops.pytorch.loss.vocab_parallel_logp.VocabParallelLogprobOp",
    covers_paths=(
        ExecutionPath.TRAINING_FULL_PREFILL,
        ExecutionPath.ROLLOUT_FULL_PREFILL,
    ),
    required_settings=(
        # The op takes its typed LogprobContract as a call argument and echoes
        # the resolved reduction semantics back through dispatch provenance.
        RequiredSetting(
            "logp.reduction_contract",
            REFERENCE_LSE_MERGE,
            SettingChannel.CALL_ARG,
            readback="dispatch.provenance['contract']['reduction']",
        ),
    ),
    pinned_libraries=(LibraryPin("torch", "2.6.0"),),
)
