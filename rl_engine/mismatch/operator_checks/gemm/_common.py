# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Shared across the gemm factors: collective contracts and the ordered reference."""

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

# What the reference pins: a reduce_scatter whose accumulation order is keyed on
# global rank, which is what makes the result independent of topology.
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

# Megatron with sequence parallelism on: all_reduce is rewritten into
# reduce_scatter + all_gather, and NCCL picks the order.
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

# vLLM without sequence parallelism: a plain all_reduce, and the backend is
# chosen by world size and topology at runtime.
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

DETERMINISTIC_REDUCE_REFERENCE = ReferenceImplementation(
    name="rl_kernel",
    # SELF_WRITTEN because neither TE nor FlashInfer exposes a reduction whose
    # order is fixed across topologies -- see the PR body.
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
            ORDERED_REDUCE_SCATTER,  # the contract itself, pinned as a value
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
