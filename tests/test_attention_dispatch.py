# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Contract-aware attention dispatch: registration, policy, and fail-closed.

These cases use a fresh ``KernelRegistry`` so they never mutate the process
singleton, and they assert the property that motivates a separate dispatch
entry point: a WS2 attention caller must never be served by a backend that
declares different reduction, Split-KV, or LSE-export semantics.
"""

from __future__ import annotations

import pytest

from rl_engine.kernels.attention_contract import (
    AttentionBackendCapability,
    AttentionContract,
    AttentionContractError,
    AttentionDType,
    AttentionMode,
    AttentionRole,
    ReductionSpec,
    ShardingSpec,
    SplitKVSpec,
)
from rl_engine.kernels.registry import KernelRegistry, OpBackend, _rocm_strict_attention_available


def _contract(*, cp_world_size: int = 1, seq_len: int = 128) -> AttentionContract:
    sharding = ShardingSpec(
        tp_rank=0,
        tp_world_size=1,
        cp_rank=0,
        cp_world_size=cp_world_size,
        global_q_heads=32,
        global_kv_heads=8,
        local_q_head_start=0,
        local_q_heads=32,
        local_kv_head_start=0,
        local_kv_heads=8,
        global_sequence_length=seq_len,
        local_sequence_length=seq_len,
        global_block_indices=(0,),
        global_block_token_starts=(0,),
        local_block_offsets=(0, seq_len),
    )
    return AttentionContract(
        role=AttentionRole.TRAIN,
        mode=AttentionMode.PREFILL,
        dtype=AttentionDType.BF16,
        batch_size=1,
        query_sequence_length=seq_len,
        head_dim=128,
        causal=True,
        causal_offsets=(0,),
        sharding=sharding,
        reduction=ReductionSpec(),
        split_kv=SplitKVSpec.disabled(),
        export_lse=True,
    )


def _capability(**overrides) -> AttentionBackendCapability:
    fields = {
        "backend_id": "test.strict.core",
        "roles": frozenset({AttentionRole.TRAIN, AttentionRole.INFER}),
        "modes": frozenset({AttentionMode.PREFILL}),
        "dtypes": frozenset({AttentionDType.BF16}),
        "cp_world_sizes": (1,),
        "exports_attention_lse": True,
        "reports_actual_split_kv_plan": True,
        "implementation_kind": "production",
    }
    fields.update(overrides)
    return AttentionBackendCapability(**fields)


class _FakeCore:
    pass


@pytest.fixture()
def registry(monkeypatch):
    fresh = KernelRegistry()
    # Serve a stand-in instance so dispatch never imports a vendor stack.
    monkeypatch.setattr(fresh, "_get_or_create_backend", lambda backend: _FakeCore())
    return fresh


def _platform(registry) -> str:
    return registry._platform()


def test_only_rocm_autoregisters_and_only_when_the_vendor_stack_loads():
    """Attention dispatch is opt-in per platform.

    The strict ROCm core is the single auto-registration, and it appears only
    when ``aiter.ops.mha`` really loaded.  No other platform gains an attention
    candidate implicitly, so a CUDA or CPU WS2 caller still fails loudly rather
    than being served something with different semantics.
    """

    fresh = KernelRegistry()

    for platform, candidates in fresh._attention_candidates.items():
        if platform == "rocm":
            expected = (
                [OpBackend.ROCM_STRICT_ATTENTION] if _rocm_strict_attention_available() else []
            )
            assert candidates == expected
        else:
            assert candidates == []


def test_registered_backend_resolves_with_provenance(registry):
    registry.register_attention_backend(
        OpBackend.ROCM_STRICT_ATTENTION, _capability(), platform=_platform(registry)
    )

    result = registry.get_attention_op(_contract(), requested_backend="test.strict.core")

    assert result.capability.backend_id == "test.strict.core"
    assert result.provenance["actual_backend"] == "test.strict.core"
    assert result.provenance["fallback"] is False
    assert result.provenance["requested_backend"] == "test.strict.core"
    assert result.provenance["contract"]["lse_domain"] == "attention"


def test_unregistered_contract_fails_loudly_instead_of_falling_back(registry):
    # Drop every auto-registered candidate: a platform with no attention
    # backend must raise rather than resolve something from the legacy lists.
    for candidates in registry._attention_candidates.values():
        candidates.clear()

    with pytest.raises(RuntimeError, match="No attention backend supports"):
        registry.get_attention_op(_contract(), requested_backend="auto")


def test_explicit_backend_id_never_resolves_to_a_different_backend(registry):
    registry.register_attention_backend(
        OpBackend.ROCM_STRICT_ATTENTION, _capability(), platform=_platform(registry)
    )

    with pytest.raises(RuntimeError, match="does not match requested_backend"):
        registry.get_attention_op(_contract(), requested_backend="some.other.backend")


def test_capability_mismatch_is_rejected_rather_than_approximated(registry):
    # A backend that cannot export attention-domain LSE must not serve a
    # contract that requires it.
    registry.register_attention_backend(
        OpBackend.ROCM_STRICT_ATTENTION,
        _capability(exports_attention_lse=False),
        platform=_platform(registry),
    )

    with pytest.raises(RuntimeError, match="LSE export is unsupported"):
        registry.get_attention_op(_contract(), requested_backend="test.strict.core")


def test_cp_contract_requires_deterministic_merge_support(registry):
    registry.register_attention_backend(
        OpBackend.ROCM_STRICT_ATTENTION,
        _capability(cp_world_sizes=(1, 2)),
        platform=_platform(registry),
    )

    with pytest.raises(RuntimeError, match="deterministic CP"):
        registry.get_attention_op(_contract(cp_world_size=2), requested_backend="test.strict.core")


def test_auto_is_rejected_under_context_parallelism(registry):
    registry.register_attention_backend(
        OpBackend.ROCM_STRICT_ATTENTION, _capability(), platform=_platform(registry)
    )

    with pytest.raises(AttentionContractError, match="Unsafe dispatch"):
        registry.get_attention_op(_contract(cp_world_size=2), requested_backend="auto")


def test_deterministic_is_not_a_dispatch_policy(registry):
    with pytest.raises(AttentionContractError, match="not a dispatch policy"):
        registry.get_attention_op(_contract(), requested_backend="deterministic")


def test_implementation_kind_policy_filters_without_marking_fallback(registry):
    registry.register_attention_backend(
        OpBackend.ROCM_STRICT_ATTENTION, _capability(), platform=_platform(registry)
    )

    result = registry.get_attention_op(_contract(), requested_backend="production")
    assert result.provenance["fallback"] is False

    with pytest.raises(RuntimeError, match="does not satisfy requested_backend=reference"):
        registry.get_attention_op(_contract(), requested_backend="reference")


def test_reregistration_replaces_capability_without_duplicating(registry):
    platform = _platform(registry)
    registry.register_attention_backend(
        OpBackend.ROCM_STRICT_ATTENTION, _capability(), platform=platform
    )
    registry.register_attention_backend(
        OpBackend.ROCM_STRICT_ATTENTION,
        _capability(implementation_kind="reference"),
        platform=platform,
    )

    assert registry._attention_candidates[platform] == [OpBackend.ROCM_STRICT_ATTENTION]
    capability = registry._attention_capabilities[platform][OpBackend.ROCM_STRICT_ATTENTION]
    assert capability.implementation_kind == "reference"


def test_register_rejects_wrong_types_and_unknown_platforms(registry):
    with pytest.raises(AttentionContractError, match="must be an OpBackend"):
        registry.register_attention_backend("not-a-backend", _capability())
    with pytest.raises(AttentionContractError, match="must be an AttentionBackendCapability"):
        registry.register_attention_backend(OpBackend.ROCM_STRICT_ATTENTION, object())
    with pytest.raises(AttentionContractError, match="unsupported platform"):
        registry.register_attention_backend(
            OpBackend.ROCM_STRICT_ATTENTION, _capability(), platform="quantum"
        )


def test_legacy_priority_map_is_untouched_by_attention_registration(registry):
    platform = _platform(registry)
    before = {op: list(v) for op, v in registry._priority_map[platform].items()}

    registry.register_attention_backend(
        OpBackend.ROCM_STRICT_ATTENTION, _capability(), platform=platform
    )

    after = {op: list(v) for op, v in registry._priority_map[platform].items()}
    assert before == after
    # The strict core must not become reachable through legacy get_op keys.
    for candidates in after.values():
        assert OpBackend.ROCM_STRICT_ATTENTION not in candidates


def test_contract_fingerprint_is_rank_independent():
    left = _contract()
    right_sharding = ShardingSpec(
        tp_rank=1,
        tp_world_size=2,
        cp_rank=0,
        cp_world_size=1,
        global_q_heads=32,
        global_kv_heads=8,
        local_q_head_start=16,
        local_q_heads=16,
        local_kv_head_start=4,
        local_kv_heads=4,
        global_sequence_length=128,
        local_sequence_length=128,
        global_block_indices=(0,),
        global_block_token_starts=(0,),
        local_block_offsets=(0, 128),
    )
    left_tp2 = AttentionContract(
        role=AttentionRole.TRAIN,
        mode=AttentionMode.PREFILL,
        dtype=AttentionDType.BF16,
        batch_size=1,
        query_sequence_length=128,
        head_dim=128,
        causal=True,
        causal_offsets=(0,),
        sharding=ShardingSpec(
            tp_rank=0,
            tp_world_size=2,
            cp_rank=0,
            cp_world_size=1,
            global_q_heads=32,
            global_kv_heads=8,
            local_q_head_start=0,
            local_q_heads=16,
            local_kv_head_start=0,
            local_kv_heads=4,
            global_sequence_length=128,
            local_sequence_length=128,
            global_block_indices=(0,),
            global_block_token_starts=(0,),
            local_block_offsets=(0, 128),
        ),
        reduction=ReductionSpec(),
        split_kv=SplitKVSpec.disabled(),
        export_lse=True,
    )
    right_tp2 = AttentionContract(
        role=AttentionRole.TRAIN,
        mode=AttentionMode.PREFILL,
        dtype=AttentionDType.BF16,
        batch_size=1,
        query_sequence_length=128,
        head_dim=128,
        causal=True,
        causal_offsets=(0,),
        sharding=right_sharding,
        reduction=ReductionSpec(),
        split_kv=SplitKVSpec.disabled(),
        export_lse=True,
    )

    # Both TP ranks of one logical invocation agree; a different TP degree does not.
    assert left_tp2.cross_rank_fingerprint() == right_tp2.cross_rank_fingerprint()
    assert left.cross_rank_fingerprint() != left_tp2.cross_rank_fingerprint()
