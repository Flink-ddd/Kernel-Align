# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""GEMM's operator-level contract, readback and implementation adapter."""

from __future__ import annotations

import importlib
from typing import Any, Callable, Mapping

from rl_engine.mismatch.operator_checks.gemm._common import (
    FFN_CONSISTENT_REFERENCE,
    FFN_STAGE_NAMES,
    FFN_STAGE_OUTPUTS,
    TP_SIZE,
    GemmContractError,
    downcast_point,
    inferred_forward_collectives,
    non_empty_string,
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

    ffn_observed = "gemm.ffn_path" in switch_values
    compute = precision(
        _required(switch_values, "gemm.compute_dtype") if ffn_observed else "bf16",
        "gemm.compute_dtype",
    )
    accumulate = precision(
        _required(switch_values, "gemm.accumulate_dtype") if ffn_observed else "fp32",
        "gemm.accumulate_dtype",
    )
    if accumulate is not Precision.FP32:
        raise GemmAdapterError("the Qwen3 FFN contract requires FP32 GEMM accumulation")
    downcast = downcast_point(
        _required(switch_values, "gemm.downcast_at") if ffn_observed else "per_partial",
        "gemm.downcast_at",
    )

    tp_world_size = positive_int(
        _required(switch_values, "gemm.tp_world_size") if ffn_observed else TP_SIZE,
        "gemm.tp_world_size",
    )
    collectives = inferred_forward_collectives(
        role,
        switch_values,
        tp_world_size=tp_world_size,
    )
    extra: dict[str, Any] = {}
    if ffn_observed:
        extra = _ffn_runtime_metadata(role, switch_values, tp_world_size)

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
    evidence = config.get("evidence", ())
    if isinstance(evidence, str):
        evidence = (evidence,)
    if FFN_STAGE_OUTPUTS in evidence:
        _stage_output_digests(config.get("gemm.stage_output_digests"))
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
    except Exception as exc:  # noqa: BLE001 - import failure is retained as provenance
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


def _required(values: Mapping[str, Any], field: str) -> Any:
    if field not in values:
        raise GemmAdapterError(f"missing effective runtime readback for {field}")
    return values[field]


def _stage_output_digests(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise GemmAdapterError("gemm.stage_output_digests must be a mapping")
    missing = [stage for stage in FFN_STAGE_NAMES if stage not in value]
    if missing:
        raise GemmAdapterError(
            "gemm.stage_output_digests is missing required FFN stages: " + ", ".join(missing)
        )
    return {
        stage: non_empty_string(value[stage], f"gemm.stage_output_digests.{stage}")
        for stage in FFN_STAGE_NAMES
    }


def _ffn_runtime_metadata(
    role: PolicyRole,
    switch_values: Mapping[str, Any],
    tp_world_size: int,
) -> dict[str, Any]:
    """Normalize only observed FFN state; never synthesize runtime evidence."""

    return {
        "hidden_size": positive_int(
            _required(switch_values, "gemm.hidden_size"),
            "gemm.hidden_size",
        ),
        "intermediate_size": positive_int(
            _required(switch_values, "gemm.intermediate_size"),
            "gemm.intermediate_size",
        ),
        "tp_world_size": tp_world_size,
        "weight_layout": non_empty_string(
            _required(switch_values, "gemm.weight_layout"),
            "gemm.weight_layout",
        ),
        "gate_up_packed": strict_bool(
            _required(switch_values, "gemm.gate_up_packed"),
            "gemm.gate_up_packed",
        ),
        "has_bias": strict_bool(
            _required(switch_values, "gemm.has_bias"),
            "gemm.has_bias",
        ),
        "ffn_path": _ffn_path(_required(switch_values, "gemm.ffn_path"), role),
        "gemm_backend": non_empty_string(
            _required(switch_values, "gemm.ffn_backend"),
            "gemm.ffn_backend",
        ),
        "activation_backend": non_empty_string(
            _required(switch_values, "gemm.activation_backend"),
            "gemm.activation_backend",
        ),
        "batch_invariant": strict_bool(
            _required(switch_values, "gemm.batch_invariant"),
            "gemm.batch_invariant",
        ),
        "stage_output_digests": _stage_output_digests(
            _required(switch_values, "gemm.stage_output_digests")
        ),
    }


__all__ = [
    "GemmAdapterError",
    "build_contract",
    "observe_collectives",
    "read_effective_config",
    "resolve_implementation",
]
