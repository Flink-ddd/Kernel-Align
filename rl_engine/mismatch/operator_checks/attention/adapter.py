# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Attention's operator-level contract, readback and implementation adapter."""

from __future__ import annotations

import importlib
from typing import Any, Callable, Mapping

from rl_engine.mismatch.operator_checks.attention._common import (
    ATTENTION_LSE_DOMAIN,
    ATTENTION_MERGE_STATE,
    CP_MERGE_REFERENCE,
    SPLIT_KV_REFERENCE,
    AttentionContractError,
    downcast_point,
    normalize_collective,
    normalize_cp_block_manifest,
    normalize_split_kv_plan_set,
    positive_int,
    precision,
    reference_collective,
    reference_selected,
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


# One public error type for adapter and contract normalization failures. Callers
# should not have to know which validation layer rejected the runtime record.
AttentionAdapterError = AttentionContractError


def build_contract(role: PolicyRole, switch_values: Mapping[str, Any]) -> OperatorContract:
    """Build a contract from effective runtime state, never requested policy alone.

    Actual Split-KV plans, CP ownership and post-RoPE digests are intentionally
    absent when the engine did not report them. Their factor rules then produce
    ``REQUIRED_FIELD_MISSING`` instead of allowing a requested setting to pass as
    runtime evidence.
    """

    if not isinstance(role, PolicyRole):
        raise AttentionAdapterError(f"role must be a PolicyRole, got {role!r}")
    if not isinstance(switch_values, Mapping):
        raise AttentionAdapterError("Attention effective config must be a mapping")

    compute = precision(switch_values.get("attn.compute_dtype", "bf16"), "attn.compute_dtype")
    accumulate = precision(
        switch_values.get("attn.accumulate_dtype", "fp32"), "attn.accumulate_dtype"
    )
    if accumulate is not Precision.FP32:
        raise AttentionAdapterError(
            "Attention softmax, Split-KV and CP (out, lse) merges must accumulate in fp32"
        )
    downcast = downcast_point(
        _role_value(
            switch_values,
            role,
            common="attn.downcast_at",
            training="attn.training_downcast_at",
            rollout="attn.rollout_downcast_at",
            default="final_write",
        ),
        "attn.downcast_at",
    )

    batch_size = positive_int(switch_values.get("attn.batch_size", 1), "attn.batch_size")
    tp_world_size = positive_int(
        switch_values.get("attn.tp_world_size", 1), "attn.tp_world_size"
    )
    cp_world_size = positive_int(
        switch_values.get("attn.cp_world_size", 1), "attn.cp_world_size"
    )

    split_reference = reference_selected(
        switch_values.get("attn.split_kv"), role, SPLIT_KV_REFERENCE.name, "attn.split_kv"
    )
    cp_reference = reference_selected(
        switch_values.get("attn.cp_merge"), role, CP_MERGE_REFERENCE.name, "attn.cp_merge"
    )

    raw_collective = switch_values.get("attn.cp_collective")
    if raw_collective is None and cp_reference:
        collectives = reference_collective(cp_world_size)
    else:
        collectives = normalize_collective(raw_collective, cp_world_size=cp_world_size)

    plan = normalize_split_kv_plan_set(
        switch_values.get("attn.actual_split_kv_plan_set")
    )
    if plan is not None:
        topology = (plan["batch_size"], plan["tp_world_size"], plan["cp_world_size"])
        expected = (batch_size, tp_world_size, cp_world_size)
        if topology != expected:
            raise AttentionAdapterError(
                "Split-KV runtime plan topology does not match the Attention invocation: "
                f"plan={topology}, attention={expected}"
            )

    manifest = normalize_cp_block_manifest(
        switch_values.get("attn.cp_block_manifest"),
        tp_world_size=tp_world_size,
        cp_world_size=cp_world_size,
    )

    lse_domain = switch_values.get("attn.lse_domain")
    export_lse = switch_values.get("attn.export_lse")
    merge_state = switch_values.get("attn.merge_state")
    if lse_domain is not None and lse_domain != ATTENTION_LSE_DOMAIN:
        raise AttentionAdapterError(
            f"attn.lse_domain must be {ATTENTION_LSE_DOMAIN!r}, got {lse_domain!r}"
        )
    if export_lse is not None and not isinstance(export_lse, bool):
        raise AttentionAdapterError("attn.export_lse must be a bool")
    if merge_state is not None and merge_state != ATTENTION_MERGE_STATE:
        raise AttentionAdapterError(
            f"attn.merge_state must be {ATTENTION_MERGE_STATE!r}, got {merge_state!r}"
        )

    extra: dict[str, Any] = {
        "batch_size": batch_size,
        "tp_world_size": tp_world_size,
        "cp_world_size": cp_world_size,
        "requested_split_kv_policy": switch_values.get("attn.requested_split_kv_policy"),
        "requested_split_kv_size": switch_values.get("attn.requested_split_kv_size"),
        "split_kv_reference_selected": split_reference,
        "cp_reference_selected": cp_reference,
        "fusion_boundary": switch_values.get("attn.fusion_boundary"),
    }

    _copy_if_present(extra, switch_values, "rope_theta", "attn.rope_theta")
    _copy_if_present(extra, switch_values, "position_ids_digest", "attn.position_ids_digest")
    _copy_if_present(extra, switch_values, "post_rope_qk_digest", "attn.post_rope_qk_digest")
    _copy_if_present(extra, switch_values, "q_rope_state", "attn.q_rope_state")
    _copy_if_present(extra, switch_values, "k_rope_state", "attn.k_rope_state")
    _copy_if_present(extra, switch_values, "k_cache_rope_state", "attn.k_cache_rope_state")

    if plan is not None:
        extra.update(
            {
                "total_kv_tokens": plan["total_kv_tokens"],
                "split_kv_coordinates": plan["coordinates"],
                "split_kv_owner_ranges": plan["owner_ranges"],
                "split_kv_boundaries": plan["boundaries"],
                "split_kv_merge_order": plan["merge_order"],
                "split_kv_accumulate_precision": plan["accumulate_precision"],
                "split_kv_downcast_at": plan["downcast_at"],
                "split_kv_fallback": plan["fallback"],
                "split_kv_runtime_plan_set": plan["canonical"],
                "split_kv_backend": plan["backend"],
                "split_kv_plan_source": plan["source"],
            }
        )
    if manifest is not None:
        extra["cp_block_manifest"] = manifest
        extra["cp_owner_ranges"] = tuple(
            (block_index, start, end, owner_cp, owner_tp)
            for block_index, start, end, owner_cp, owner_tp in manifest
        )
    if lse_domain is not None:
        extra["lse_domain"] = lse_domain
    if export_lse is not None:
        extra["export_lse"] = export_lse
    if merge_state is not None:
        extra["merge_state"] = merge_state

    return OperatorContract(
        operator="attention",
        role=role,
        precision=PrecisionProfile(
            compute=compute,
            accumulate=accumulate,
            softmax_accumulate=accumulate,
            downcast_at=downcast,
        ),
        collectives=collectives,
        extra=extra,
    )


def read_effective_config(role: PolicyRole, adapter: Any) -> Mapping[str, Any]:
    """Read the state that ran, rejecting requested-only configuration objects."""

    adapter_role = getattr(adapter, "role", None)
    if adapter_role is not None:
        try:
            normalized_role = PolicyRole(adapter_role)
        except (TypeError, ValueError) as exc:
            raise AttentionAdapterError(f"invalid adapter role {adapter_role!r}") from exc
        if normalized_role is not role:
            raise AttentionAdapterError(
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
        raise AttentionAdapterError(
            f"cannot read effective Attention config from {type(adapter).__name__}: expected "
            "read_effective_config(), a mapping, or an effective_config mapping"
        )

    config = dict(value)
    requested_only = config.pop("requested_config", None)
    if requested_only is not None and not any(key.startswith("attn.") for key in config):
        raise AttentionAdapterError(
            "engine returned requested_config but no effective attn.* runtime readback"
        )
    return config


def observe_collectives(role: PolicyRole, adapter: Any) -> tuple[CollectiveContract, ...]:
    """Return the CP collective that actually ran, or no evidence when absent."""

    config = read_effective_config(role, adapter)
    cp_world_size = positive_int(
        config.get("attn.cp_world_size", 1), "attn.cp_world_size"
    )
    return normalize_collective(config.get("attn.cp_collective"), cp_world_size=cp_world_size)


def resolve_implementation(
    factor_id: str, role: PolicyRole, impl_name: str
) -> tuple[Callable[..., Any] | None, ImplementationResolution]:
    """Resolve every candidate in order and preserve every rejection reason."""

    candidates = _implementation_candidates(factor_id, role, impl_name)
    rejected: list[RejectedCandidate] = []
    for candidate in candidates:
        parsed = _import_target(candidate)
        if parsed is None:
            rejected.append(
                RejectedCandidate(name=candidate, reason="not a dotted or module:attribute path")
            )
            continue
        module_name, attribute = parsed
        try:
            module = importlib.import_module(module_name)
        except (ImportError, OSError) as exc:
            rejected.append(
                RejectedCandidate(name=candidate, reason=f"import failed: {exc}")
            )
            continue

        resolved: Any = module
        for part in attribute.split("."):
            resolved = getattr(resolved, part, None)
            if resolved is None:
                break
        if resolved is None:
            rejected.append(
                RejectedCandidate(
                    name=candidate,
                    reason=f"{module_name} has no attribute {attribute!r}",
                )
            )
            continue
        if isinstance(resolved, type):
            try:
                resolved = resolved()
            except Exception as exc:  # noqa: BLE001 - recorded in provenance
                rejected.append(
                    RejectedCandidate(
                        name=candidate,
                        reason=f"instantiation failed: {exc}",
                    )
                )
                continue
        if not callable(resolved):
            rejected.append(
                RejectedCandidate(name=candidate, reason="resolved object is not callable")
            )
            continue
        return resolved, ImplementationResolution(
            requested=impl_name,
            resolved=candidate,
            rejected=tuple(rejected),
        )

    return None, ImplementationResolution(
        requested=impl_name,
        resolved=None,
        rejected=tuple(rejected),
    )


def _role_value(
    values: Mapping[str, Any],
    role: PolicyRole,
    *,
    common: str,
    training: str,
    rollout: str,
    default: Any,
) -> Any:
    role_key = training if role is PolicyRole.TRAINING else rollout
    return values.get(role_key, values.get(common, default))


def _copy_if_present(
    target: dict[str, Any], source: Mapping[str, Any], target_key: str, source_key: str
) -> None:
    if source_key in source:
        target[target_key] = source[source_key]


def _implementation_candidates(
    factor_id: str, role: PolicyRole, impl_name: str
) -> tuple[str, ...]:
    if factor_id == "attn.rope_fusion" and role is PolicyRole.ROLLOUT:
        fallback = "vllm.model_executor.layers.rotary_embedding.get_rope"
        return (impl_name, fallback) if impl_name != fallback else (impl_name,)
    return (impl_name,)


def _import_target(value: str) -> tuple[str, str] | None:
    if ":" in value:
        module_name, attribute = value.rsplit(":", 1)
    elif "." in value:
        module_name, _, attribute = value.rpartition(".")
    else:
        return None
    if not module_name or not attribute:
        return None
    return module_name, attribute


__all__ = [
    "AttentionAdapterError",
    "build_contract",
    "observe_collectives",
    "read_effective_config",
    "resolve_implementation",
]
