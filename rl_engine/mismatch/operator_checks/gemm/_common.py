# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Contracts, references and normalization shared by GEMM factors."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

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
QWEN3_8B_HIDDEN_SIZE = 4096
QWEN3_8B_TP2_INTERMEDIATE_SIZE = 6144

FFN_STAGE_OUTPUTS = "ffn_stage_outputs"

DTYPES: dict[str, Precision] = {
    "bf16": Precision.BF16,
    "bfloat16": Precision.BF16,
    "fp16": Precision.FP16,
    "float16": Precision.FP16,
    "fp32": Precision.FP32,
    "float32": Precision.FP32,
}

DOWNCAST_POINTS: dict[str, DowncastPoint] = {
    "never": DowncastPoint.NEVER,
    "per_block": DowncastPoint.PER_BLOCK,
    "per_partial": DowncastPoint.PER_PARTIAL,
    "final_write": DowncastPoint.FINAL_WRITE,
}

COLLECTIVE_OPS: dict[str, CollectiveOp] = {item.value: item for item in CollectiveOp}
PARALLEL_DIMS: dict[str, ParallelDim] = {item.value: item for item in ParallelDim}
REDUCTION_ORDERS: dict[str, ReductionOrder] = {item.value: item for item in ReductionOrder}
DETERMINISM_LEVELS: dict[str, DeterminismLevel] = {item.value: item for item in DeterminismLevel}


class GemmContractError(ValueError):
    """Runtime GEMM metadata cannot prove the declared contract."""


def precision(value: Any, field: str) -> Precision:
    if isinstance(value, Precision):
        return value
    key = str(value).lower()
    if key not in DTYPES:
        raise GemmContractError(f"{field} must be one of {tuple(DTYPES)}, got {value!r}")
    return DTYPES[key]


def downcast_point(value: Any, field: str) -> DowncastPoint:
    if isinstance(value, DowncastPoint):
        return value
    key = str(value).lower()
    if key not in DOWNCAST_POINTS:
        raise GemmContractError(f"{field} must be one of {tuple(DOWNCAST_POINTS)}, got {value!r}")
    return DOWNCAST_POINTS[key]


def positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise GemmContractError(f"{field} must be a positive integer, got {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise GemmContractError(f"{field} must be a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise GemmContractError(f"{field} must be a positive integer, got {value!r}")
    return parsed


def strict_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise GemmContractError(f"{field} must be a bool, got {value!r}")
    return value


def reference_selected(value: Any, role: Any, reference_name: str, field: str) -> bool:
    if value in (None, "native", "fast"):
        return False
    if value == reference_name:
        return True
    role_value = getattr(role, "value", str(role))
    one_sided = (f"{reference_name}@training", f"{reference_name}@rollout")
    if value in one_sided:
        return value.endswith(f"@{role_value}")
    raise GemmContractError(
        f"unknown {field} value {value!r}; expected 'fast', {reference_name!r}, "
        f"'{reference_name}@training' or '{reference_name}@rollout'"
    )


def normalize_collectives(
    raw: Any,
    *,
    tp_world_size: int,
) -> tuple[CollectiveContract, ...]:
    """Normalize the collective trace that actually ran.

    Absence remains absence. Requested reduction policy is not runtime evidence.
    """

    if raw is None:
        return ()
    records = raw if isinstance(raw, (tuple, list)) else (raw,)
    return tuple(_normalize_collective(record, tp_world_size=tp_world_size) for record in records)


def _normalize_collective(raw: Any, *, tp_world_size: int) -> CollectiveContract:
    if isinstance(raw, CollectiveContract):
        contract = raw
    elif isinstance(raw, Mapping):
        contract = CollectiveContract(
            op=_enum_member(COLLECTIVE_OPS, raw.get("op"), "gemm.forward_collective.op"),
            group=_enum_member(
                PARALLEL_DIMS,
                raw.get("group", ParallelDim.TENSOR.value),
                "gemm.forward_collective.group",
            ),
            group_size=positive_int(
                raw.get("group_size", tp_world_size),
                "gemm.forward_collective.group_size",
            ),
            reduction_order=_enum_member(
                REDUCTION_ORDERS,
                raw.get("reduction_order"),
                "gemm.forward_collective.reduction_order",
            ),
            accumulate_precision=precision(
                raw.get("accumulate_precision"),
                "gemm.forward_collective.accumulate_precision",
            ),
            downcast_at=downcast_point(
                raw.get("downcast_at"),
                "gemm.forward_collective.downcast_at",
            ),
            determinism=_enum_member(
                DETERMINISM_LEVELS,
                raw.get("determinism"),
                "gemm.forward_collective.determinism",
            ),
            backend=_non_empty_string(raw.get("backend"), "gemm.forward_collective.backend"),
        )
    else:
        raise GemmContractError(
            "gemm.forward_collective must be a CollectiveContract, mapping, or sequence"
        )
    if contract.group_size != tp_world_size:
        raise GemmContractError(
            "GEMM collective group size does not match gemm.tp_world_size: "
            f"{contract.group_size} != {tp_world_size}"
        )
    return contract


def _enum_member(options: Mapping[str, Any], value: Any, field: str) -> Any:
    for member in options.values():
        if value is member:
            return member
    key = str(value).lower()
    if key not in options:
        raise GemmContractError(f"{field} must be one of {tuple(options)}, got {value!r}")
    return options[key]


def _non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GemmContractError(f"{field} must be a non-empty string, got {value!r}")
    return value.strip()


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


# SELF_WRITTEN because neither Megatron nor vLLM exposes one shared Qwen3 FFN
# path with fixed-order GEMMs and deterministic SwiGLU on both sides. The
# builder selects CUDA by default; Triton is an alternate consistent backend
# whose effective provenance is recorded by the same adapter.
FFN_CONSISTENT_REFERENCE = ReferenceImplementation(
    name="consistent",
    tier=ReferenceAuthority.SELF_WRITTEN,
    training_impl="rl_engine.kernels.ffn.build_qwen3_ffn",
    rollout_impl="rl_engine.kernels.ffn.build_qwen3_ffn",
    covers_paths=(
        ExecutionPath.TRAINING_FULL_PREFILL,
        ExecutionPath.ROLLOUT_FULL_PREFILL,
        ExecutionPath.ROLLOUT_CHUNKED_PREFILL,
    ),
    required_settings=(
        RequiredSetting(
            "gemm.ffn_path",
            "consistent",
            SettingChannel.CALL_ARG,
            readback="module.provenance.path",
        ),
        RequiredSetting(
            "gemm.ffn_backend",
            "cuda.det_gemm",
            SettingChannel.CALL_ARG,
            readback="module.provenance.gemm_backend",
        ),
    ),
    pinned_libraries=(LibraryPin("torch", "2.6.0"),),
)


def inferred_forward_collectives(
    role: Any,
    switch_values: Mapping[str, Any],
    *,
    tp_world_size: int,
) -> tuple[CollectiveContract, ...]:
    """Build the declared reduction contract; runtime observation still wins."""

    raw = switch_values.get("gemm.forward_collective")
    if raw is not None:
        return normalize_collectives(raw, tp_world_size=tp_world_size)
    # The Qwen3 FFN factor covers only local Gate/Up/SwiGLU/Down arithmetic.
    # A TP reduction belongs to gemm.forward_reduce and must not be invented
    # just because the model metadata says tp_world_size > 1.
    if "gemm.forward_reduce" not in switch_values:
        return ()
    if tp_world_size == 1:
        return ()
    if reference_selected(
        switch_values.get("gemm.forward_reduce"),
        role,
        DETERMINISTIC_REDUCE_REFERENCE.name,
        "gemm.forward_reduce",
    ):
        return (replace(ORDERED_REDUCE_SCATTER, group_size=tp_world_size),)
    native = (
        NATIVE_TRAINING_REDUCE
        if getattr(role, "value", role) == "training"
        else NATIVE_ROLLOUT_REDUCE
    )
    return (replace(native, group_size=tp_world_size),)


__all__ = [
    "DETERMINISTIC_REDUCE_REFERENCE",
    "FFN_CONSISTENT_REFERENCE",
    "FFN_STAGE_OUTPUTS",
    "GemmContractError",
    "NATIVE_ROLLOUT_REDUCE",
    "NATIVE_TRAINING_REDUCE",
    "ORDERED_REDUCE_SCATTER",
    "QWEN3_8B_HIDDEN_SIZE",
    "QWEN3_8B_TP2_INTERMEDIATE_SIZE",
    "TP_SIZE",
    "downcast_point",
    "inferred_forward_collectives",
    "normalize_collectives",
    "positive_int",
    "precision",
    "reference_selected",
    "strict_bool",
]
