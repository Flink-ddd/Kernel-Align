# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""GEMM's operator-level contract, readback and implementation adapter."""

from __future__ import annotations

import importlib
from typing import Any, Callable, Mapping

from rl_engine.mismatch.operator_checks.gemm._common import (
    FFN_CONSISTENT_REFERENCE,
    QWEN3_8B_HIDDEN_SIZE,
    QWEN3_8B_TP2_INTERMEDIATE_SIZE,
    TP_SIZE,
    GemmContractError,
    downcast_point,
    inferred_forward_collectives,
    normalize_collectives,
    positive_int,
    precision,
    reference_selected,
    strict_bool,
)
from rl_engine.mismatch.schema import (
    CollectiveContract,
    ImplementationResolution,
    OperatorContract,
    PolicyRole,
    Precision,
    PrecisionProfile,
    RejectedCandidate,
)

GemmAdapterError = GemmContractError


def _ffn_path(value: Any, role: PolicyRole) -> str:
    selected = reference_selected(
        value,
        role,
        FFN_CONSISTENT_REFERENCE.name,
        "gemm.ffn_path",
    )
    return "consistent" if selected else "fast"


def build_contract(role: PolicyRole, switch_values: Mapping[str, Any]) -> OperatorContract:
    """Build a GEMM contract from effective runtime state.

    The local Qwen3 FFN owns no communication. ``gemm.forward_collective`` is
    retained for the existing RowParallel reduction factor and stays outside
    the FFN implementation factor.
    """

    if not isinstance(role, PolicyRole):
        raise GemmAdapterError(f"role must be a PolicyRole, got {role!r}")
    if not isinstance(switch_values, Mapping):
        raise GemmAdapterError("GEMM effective config must be a mapping")

    compute = precision(switch_values.get("gemm.compute_dtype", "bf16"), "gemm.compute_dtype")
    accumulate = precision(
        switch_values.get("gemm.accumulate_dtype", "fp32"),
        "gemm.accumulate_dtype",
    )
    if accumulate is not Precision.FP32:
        raise GemmAdapterError("the Qwen3 FFN contract requires FP32 GEMM accumulation")
    downcast = downcast_point(
        switch_values.get("gemm.downcast_at", "per_partial"),
        "gemm.downcast_at",
    )

    hidden_size = positive_int(
        switch_values.get("gemm.hidden_size", QWEN3_8B_HIDDEN_SIZE),
        "gemm.hidden_size",
    )
    intermediate_size = positive_int(
        switch_values.get(
            "gemm.intermediate_size",
            QWEN3_8B_TP2_INTERMEDIATE_SIZE,
        ),
        "gemm.intermediate_size",
    )
    tp_world_size = positive_int(
        switch_values.get("gemm.tp_world_size", TP_SIZE),
        "gemm.tp_world_size",
    )
    path = _ffn_path(switch_values.get("gemm.ffn_path", "fast"), role)

    default_gemm_backend = "cuda.det_gemm" if path == "consistent" else "pytorch.matmul"
    default_activation_backend = (
        "cuda.swiglu" if path == "consistent" else "torch.nn.functional.silu"
    )
    gemm_backend = _non_empty_string(
        switch_values.get("gemm.ffn_backend", default_gemm_backend),
        "gemm.ffn_backend",
    )
    activation_backend = _non_empty_string(
        switch_values.get("gemm.activation_backend", default_activation_backend),
        "gemm.activation_backend",
    )
    batch_invariant = strict_bool(
        switch_values.get("gemm.batch_invariant", path == "consistent"),
        "gemm.batch_invariant",
    )

    collectives = inferred_forward_collectives(
        role,
        switch_values,
        tp_world_size=tp_world_size,
    )
    extra: dict[str, Any] = {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "tp_world_size": tp_world_size,
        "weight_layout": switch_values.get("gemm.weight_layout", "A[M,K]@B[K,N]"),
        "gate_up_packed": strict_bool(
            switch_values.get("gemm.gate_up_packed", False),
            "gemm.gate_up_packed",
        ),
        "has_bias": strict_bool(
            switch_values.get("gemm.has_bias", False),
            "gemm.has_bias",
        ),
        "ffn_path": path,
        "gemm_backend": gemm_backend,
        "activation_backend": activation_backend,
        "batch_invariant": batch_invariant,
    }
    if "gemm.stage_output_digests" in switch_values:
        extra["stage_output_digests"] = switch_values["gemm.stage_output_digests"]

    return OperatorContract(
        operator="gemm",
        role=role,
        precision=PrecisionProfile(
            compute=compute,
            accumulate=accumulate,
            downcast_at=downcast,
        ),
        collectives=collectives,
        extra=extra,
    )


def read_effective_config(role: PolicyRole, adapter: Any) -> Mapping[str, Any]:
    """Read what ran, rejecting requested-only configuration as evidence."""

    adapter_role = getattr(adapter, "role", None)
    if adapter_role is not None:
        try:
            normalized_role = PolicyRole(adapter_role)
        except (TypeError, ValueError) as exc:
            raise GemmAdapterError(f"invalid adapter role {adapter_role!r}") from exc
        if normalized_role is not role:
            raise GemmAdapterError(
                f"adapter plays {normalized_role.value!r} but was queried as {role.value!r}"
            )

    reader = getattr(adapter, "read_effective_config", None)
    if callable(reader):
        value = reader()
    elif isinstance(adapter, Mapping):
        value = adapter
    else:
        value = getattr(adapter, "effective_config", None)
    if not isinstance(value, Mapping):
        raise GemmAdapterError(
            f"cannot read effective GEMM config from {type(adapter).__name__}: expected "
            "read_effective_config(), a mapping, or an effective_config mapping"
        )

    config = dict(value)
    requested_only = config.pop("requested_config", None)
    if requested_only is not None and not any(key.startswith("gemm.") for key in config):
        raise GemmAdapterError(
            "engine returned requested_config but no effective gemm.* runtime readback"
        )
    return config


def observe_collectives(role: PolicyRole, adapter: Any) -> tuple[CollectiveContract, ...]:
    """Return the actual forward collective trace; local FFN arithmetic has none."""

    config = read_effective_config(role, adapter)
    tp_world_size = positive_int(
        config.get("gemm.tp_world_size", TP_SIZE),
        "gemm.tp_world_size",
    )
    return normalize_collectives(
        config.get("gemm.forward_collective"),
        tp_world_size=tp_world_size,
    )


def resolve_implementation(
    factor_id: str, role: PolicyRole, impl_name: str
) -> tuple[Callable[..., Any] | None, ImplementationResolution]:
    """Resolve a replacement and retain every rejection reason."""

    del factor_id, role
    rejected: list[RejectedCandidate] = []
    parsed = _import_target(impl_name)
    if parsed is None:
        rejected.append(
            RejectedCandidate(
                name=impl_name,
                reason="not a dotted or module:attribute import path",
            )
        )
        return None, _failed_resolution(impl_name, rejected)

    module_name, attribute = parsed
    try:
        module = importlib.import_module(module_name)
    except (ImportError, OSError) as exc:
        rejected.append(RejectedCandidate(name=impl_name, reason=f"import failed: {exc}"))
        return None, _failed_resolution(impl_name, rejected)

    resolved: Any = module
    for part in attribute.split("."):
        resolved = getattr(resolved, part, None)
        if resolved is None:
            break
    if resolved is None:
        rejected.append(
            RejectedCandidate(
                name=impl_name,
                reason=f"{module_name} has no attribute {attribute!r}",
            )
        )
        return None, _failed_resolution(impl_name, rejected)
    if isinstance(resolved, type):
        try:
            resolved = resolved()
        except Exception as exc:  # noqa: BLE001 - recorded as provenance
            rejected.append(
                RejectedCandidate(
                    name=impl_name,
                    reason=f"instantiation failed: {exc}",
                )
            )
            return None, _failed_resolution(impl_name, rejected)
    if not callable(resolved):
        rejected.append(RejectedCandidate(name=impl_name, reason="resolved object is not callable"))
        return None, _failed_resolution(impl_name, rejected)

    return resolved, ImplementationResolution(
        requested=impl_name,
        resolved=impl_name,
        rejected=tuple(rejected),
    )


def _import_target(value: str) -> tuple[str, str] | None:
    if not isinstance(value, str):
        return None
    if ":" in value:
        module_name, attribute = value.split(":", 1)
    else:
        module_name, separator, attribute = value.rpartition(".")
        if not separator:
            return None
    if not module_name or not attribute:
        return None
    return module_name, attribute


def _failed_resolution(
    requested: str,
    rejected: list[RejectedCandidate],
) -> ImplementationResolution:
    return ImplementationResolution(
        requested=requested,
        resolved=None,
        rejected=tuple(rejected),
    )


def _non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GemmAdapterError(f"{field} must be a non-empty string, got {value!r}")
    return value.strip()


__all__ = [
    "GemmAdapterError",
    "build_contract",
    "observe_collectives",
    "read_effective_config",
    "resolve_implementation",
]
