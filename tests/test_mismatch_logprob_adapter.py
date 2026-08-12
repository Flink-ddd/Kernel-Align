# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Logprob operator plugin tests: the four adapter methods and the swap factor.

The framework's own logic is locked down in ``test_mismatch_framework.py`` with
fixture factors; here the subject is the logprob plugin's declarations and how
they map WS2's TP-aware reduction semantics onto the mismatch schema.
"""

from __future__ import annotations

import pytest

from rl_engine.mismatch.operator_checks.logprob import LogprobChecks, adapter
from rl_engine.mismatch.operator_checks.logprob._common import (
    DETERMINISTIC_LSE_REFERENCE,
    QWEN3_PADDED_VOCAB,
    even_vocab_shard_bounds,
)
from rl_engine.mismatch.operator_checks.logprob.factors.lse_merge_order import (
    FACTOR as LSE_MERGE_FACTOR,
)
from rl_engine.mismatch.pipeline import (
    build_variants,
    compare_contracts,
    reject_contradictory_factors,
)
from rl_engine.mismatch.schema import (
    CollectiveOp,
    ComparisonIssueCode,
    PolicyRole,
    Precision,
    ReductionOrder,
)

TP2 = {"logp.tp_world_size": 2}


# ------------------------------------------------------------ build_contract --


def test_native_contracts_map_each_framework_onto_the_schema():
    training = adapter.build_contract(PolicyRole.TRAINING, TP2)
    rollout = adapter.build_contract(PolicyRole.ROLLOUT, TP2)

    assert training.collectives[0].op is CollectiveOp.ALL_REDUCE
    assert training.collectives[0].backend == "nccl"
    assert rollout.collectives[0].op is CollectiveOp.ALL_GATHER
    assert rollout.collectives[0].backend == "vllm_custom_ipc"
    for side in (training, rollout):
        assert side.collectives[0].group_size == 2
        assert side.collectives[0].accumulate_precision is Precision.FP32
        assert side.collectives[0].reduction_order is ReductionOrder.NCCL_ALGORITHM
        assert side.extra["vocab_shard_map"] == even_vocab_shard_bounds(QWEN3_PADDED_VOCAB, 2)


def test_tp1_contract_declares_no_collectives():
    contract = adapter.build_contract(PolicyRole.TRAINING, {"logp.tp_world_size": 1})
    assert contract.collectives == ()


def test_reference_switch_pins_the_shard_order_merge_on_both_sides():
    switches = {**TP2, "logp.lse_merge": DETERMINISTIC_LSE_REFERENCE.name}
    training = adapter.build_contract(PolicyRole.TRAINING, switches)
    rollout = adapter.build_contract(PolicyRole.ROLLOUT, switches)

    for side in (training, rollout):
        assert side.collectives[0].reduction_order is ReductionOrder.GLOBAL_VOCAB_SHARD_INDEX
        assert side.collectives[0].backend == "rl_kernel"
        assert side.extra["lse_export"] is True


def test_one_sided_swap_replaces_only_the_named_side():
    switches = {**TP2, "logp.lse_merge": f"{DETERMINISTIC_LSE_REFERENCE.name}@training"}
    training = adapter.build_contract(PolicyRole.TRAINING, switches)
    rollout = adapter.build_contract(PolicyRole.ROLLOUT, switches)

    assert training.collectives[0].reduction_order is ReductionOrder.GLOBAL_VOCAB_SHARD_INDEX
    assert rollout.collectives[0].reduction_order is ReductionOrder.NCCL_ALGORITHM


def test_only_the_training_side_varies_the_head_dtype():
    switches = {**TP2, "logp.head_dtype": "fp32"}
    training = adapter.build_contract(PolicyRole.TRAINING, switches)
    rollout = adapter.build_contract(PolicyRole.ROLLOUT, switches)

    assert training.precision.lm_head is Precision.FP32
    assert rollout.precision.lm_head is Precision.BF16  # vLLM: the model dtype


def test_unknown_switch_values_fail_loudly():
    with pytest.raises(adapter.LogprobAdapterError, match="head_dtype"):
        adapter.build_contract(PolicyRole.TRAINING, {"logp.head_dtype": "fp8"})
    with pytest.raises(adapter.LogprobAdapterError, match="lse_merge"):
        adapter.build_contract(PolicyRole.TRAINING, {"logp.lse_merge": "fastest"})


# ------------------------------------------------- comparison with the factor --


def test_native_sides_disagree_semantically_on_the_merge():
    switches = {**TP2, "logp.lse_merge": "native"}
    issues = compare_contracts(
        adapter.build_contract(PolicyRole.ROLLOUT, switches),
        adapter.build_contract(PolicyRole.TRAINING, switches),
        (LSE_MERGE_FACTOR,),
    )

    codes = {issue.field_path: issue.code for issue in issues}
    assert codes["collectives[0].op"] is ComparisonIssueCode.SEMANTIC_MISMATCH
    # Identity fields agree, so the case is a finding rather than void.
    assert "collectives[0].group_size" not in codes
    assert "extra.vocab_shard_map" not in codes


def test_reference_on_both_sides_clears_every_contract_issue():
    switches = {**TP2, "logp.lse_merge": DETERMINISTIC_LSE_REFERENCE.name}
    issues = compare_contracts(
        adapter.build_contract(PolicyRole.ROLLOUT, switches),
        adapter.build_contract(PolicyRole.TRAINING, switches),
        (LSE_MERGE_FACTOR,),
    )
    assert issues == ()


def test_lse_merge_factor_expands_to_the_declared_four_arms_and_is_consistent():
    reject_contradictory_factors((LSE_MERGE_FACTOR,))
    names = [variant.name for variant in build_variants(LSE_MERGE_FACTOR)]
    assert names == [
        "both_native",
        "both_reference",
        "training_reference_only",
        "rollout_reference_only",
    ]


# ------------------------------------------- read_effective_config / observe --


def test_read_effective_config_accepts_the_three_adapter_shapes():
    as_mapping = adapter.read_effective_config(PolicyRole.TRAINING, {"logp.head_dtype": "fp32"})
    assert as_mapping == {"logp.head_dtype": "fp32"}

    class Engine:
        role = PolicyRole.ROLLOUT

        def read_effective_config(self):
            return {"logp.tp_world_size": 2}

    assert adapter.read_effective_config(PolicyRole.ROLLOUT, Engine()) == {"logp.tp_world_size": 2}

    class Bare:
        effective_config = {"logp.lse_merge": "native"}

    assert adapter.read_effective_config(PolicyRole.TRAINING, Bare()) == {
        "logp.lse_merge": "native"
    }


def test_read_effective_config_rejects_an_adapter_playing_the_other_role():
    class Engine:
        role = PolicyRole.ROLLOUT
        effective_config = {}

    with pytest.raises(adapter.LogprobAdapterError, match="plays 'rollout'"):
        adapter.read_effective_config(PolicyRole.TRAINING, Engine())


def test_observe_collectives_reflects_the_effective_config_not_the_request():
    observed = adapter.observe_collectives(
        PolicyRole.TRAINING, {"logp.tp_world_size": 2, "logp.lse_merge": "native"}
    )
    assert len(observed) == 1
    assert observed[0].op is CollectiveOp.ALL_REDUCE
    assert observed[0].group_size == 2

    assert adapter.observe_collectives(PolicyRole.TRAINING, {"logp.tp_world_size": 1}) == ()


# ------------------------------------------------------ resolve_implementation --


def test_resolution_failure_carries_the_trace_not_a_bare_none():
    impl, resolution = adapter.resolve_implementation(
        LSE_MERGE_FACTOR.id,
        PolicyRole.TRAINING,
        "rl_engine.kernels.ops.pytorch.loss.does_not_exist.MissingOp",
    )
    assert impl is None
    assert resolution.resolved is None
    assert "import failed" in resolution.rejected[0].reason

    impl, resolution = adapter.resolve_implementation(
        LSE_MERGE_FACTOR.id, PolicyRole.TRAINING, "math.no_such_attribute"
    )
    assert impl is None
    assert "no attribute" in resolution.rejected[0].reason

    impl, resolution = adapter.resolve_implementation(
        LSE_MERGE_FACTOR.id, PolicyRole.TRAINING, "math.pi"
    )
    assert impl is None
    assert "not callable" in resolution.rejected[0].reason


def test_resolution_success_returns_the_callable_and_a_clean_trace():
    impl, resolution = adapter.resolve_implementation(
        LSE_MERGE_FACTOR.id, PolicyRole.TRAINING, "math.sqrt"
    )
    assert impl(4.0) == 2.0
    assert resolution.resolved == "math.sqrt"
    assert resolution.rejected == ()


def test_ws2_reference_path_either_resolves_or_leaves_a_trace():
    """On a tree without issue #241 PR3 the WS2 op is absent; either way the
    resolution must be investigable, never a silent fallback."""

    impl, resolution = adapter.resolve_implementation(
        LSE_MERGE_FACTOR.id, PolicyRole.TRAINING, DETERMINISTIC_LSE_REFERENCE.training_impl
    )
    if impl is None:
        assert resolution.resolved is None
        assert resolution.rejected
    else:
        assert callable(impl)
        assert resolution.resolved == DETERMINISTIC_LSE_REFERENCE.training_impl


# ----------------------------------------------------------------- the plugin --


def test_plugin_wires_the_adapter_methods_and_discovers_both_factors():
    checks = LogprobChecks
    assert checks.build_contract is adapter.build_contract
    assert checks.read_effective_config is adapter.read_effective_config
    assert checks.observe_collectives is adapter.observe_collectives
    assert checks.resolve_implementation is adapter.resolve_implementation

    ids = [factor.id for factor in LogprobChecks().declare_factors()]
    assert ids == ["logp.lse_merge_order", "logp.precision_downcast"]
