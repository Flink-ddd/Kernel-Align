# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Collective contracts and the ordered reference, shared by gemm's factors."""

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

TP_SIZE = 2

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

# Megatron with sequence parallelism on: all_reduce rewritten as
# reduce_scatter + all_gather, ordered by whatever NCCL picks.
NATIVE_TRAINING_REDUCE = CollectiveContract(
    op=CollectiveOp.REDUCE_SCATTER,
    group=ParallelDim.TENSOR,
    group_size=TP_SIZE,
    reduction_order=ReductionOrder.NCCL_ALGORITHM,
    accumulate_precision=Precision.FP32,
    downcast_at=DowncastPoint.FINAL_WRITE,
    determinism=DeterminismLevel.NONE,
    backend="nccl",
)

# vLLM without sequence parallelism: a plain all_reduce, backend chosen at
# runtime from world size and topology.
NATIVE_ROLLOUT_REDUCE = CollectiveContract(
    op=CollectiveOp.ALL_REDUCE,
    group=ParallelDim.TENSOR,
    group_size=TP_SIZE,
    reduction_order=ReductionOrder.NCCL_ALGORITHM,
    accumulate_precision=Precision.FP32,
    downcast_at=DowncastPoint.PER_PARTIAL,
    determinism=DeterminismLevel.NONE,
    backend="vllm_custom_ipc",
)

# SELF_WRITTEN because neither TE nor FlashInfer exposes a reduction whose order
# is fixed across topologies.
DETERMINISTIC_REDUCE_REFERENCE = ReferenceImplementation(
    name="rl_kernel",
    tier=ReferenceAuthority.SELF_WRITTEN,
    training_impl="rl_engine.kernels.collectives.ordered_reduce_scatter",
    rollout_impl="rl_engine.kernels.collectives.ordered_reduce_scatter",
    covers_paths=(
        ExecutionPath.TRAINING_FULL_PREFILL,
        ExecutionPath.ROLLOUT_FULL_PREFILL,
    ),
    required_settings=(
        RequiredSetting(
            "forward_reduce_contract",
            ORDERED_REDUCE_SCATTER,
            SettingChannel.CALL_ARG,
            readback="module.last_collective_contract",
        ),
        RequiredSetting(
            "NCCL_ALGO",
            "Ring",
            SettingChannel.ENV_VAR,
            readback="os.environ",
            guards="nccl_algo_unpinned",
        ),
        RequiredSetting(
            "NCCL_PROTO",
            "Simple",
            SettingChannel.ENV_VAR,
            readback="os.environ",
            guards="nccl_algo_unpinned",
        ),
    ),
    pinned_libraries=(LibraryPin("torch", "2.6.0"),),
)
