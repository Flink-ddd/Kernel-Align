# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""GEMM mismatch adapter tests for the Qwen3 FFN and forward reduction."""

from __future__ import annotations

import pytest

from rl_engine.mismatch.operator_checks.gemm import GemmChecks, adapter
from rl_engine.mismatch.operator_checks.gemm._common import (
    FFN_CONSISTENT_REFERENCE,
    FFN_STAGE_OUTPUTS,
)
from rl_engine.mismatch.operator_checks.gemm.factors.ffn_implementation import FACTOR as FFN_FACTOR
from rl_engine.mismatch.operator_checks.gemm.factors.forward_reduce import (
    FACTOR as FORWARD_REDUCE_FACTOR,
)
from rl_engine.mismatch.pipeline import (
    build_variants,
    compare_contracts,
    reject_contradictory_factors,
)
from rl_engine.mismatch.schema import (
    MODEL_SHAPE,
    CollectiveOp,
    ComparisonIssueCode,
    DeterminismLevel,
    DowncastPoint,
    PolicyRole,
    Precision,
    ReductionOrder,
)


def _effective_ffn(**overrides):
    config = {
        "gemm.compute_dtype": "bf16",
        "gemm.accumulate_dtype": "fp32",
        "gemm.downcast_at": "per_partial",
        "gemm.hidden_size": 4096,
        "gemm.intermediate_size": 6144,
        "gemm.tp_world_size": 2,
        "gemm.ffn_path": "fast",
        "gemm.ffn_backend": "pytorch.matmul",
        "gemm.activation_backend": "torch.nn.functional.silu",
        "gemm.batch_invariant": False,
        "gemm.weight_layout": "A[M,K]@B[K,N]",
        "gemm.gate_up_packed": False,
        "gemm.has_bias": False,
        "gemm.stage_output_digests": {
            "gate": "sha256:gate",
            "up": "sha256:up",
            "hidden": "sha256:hidden",
            "output": "sha256:output",
        },
    }
    config.update(overrides)
    return config


def _reduce_scatter_trace(**overrides):
    trace = {
        "op": "reduce_scatter",
        "group": "tensor",
        "group_size": 2,
        "reduction_order": "nccl_algorithm",
        "accumulate_precision": "fp32",
        "downcast_at": "final_write",
        "determinism": "none",
        "backend": "nccl",
    }
    trace.update(overrides)
    return trace


# ------------------------------------------------------------ build_contract --


def test_fast_ffn_contract_records_qwen3_tp2_identity_and_no_local_collective():
    contract = adapter.build_contract(PolicyRole.TRAINING, _effective_ffn())

    assert contract.precision.compute is Precision.BF16
    assert contract.precision.accumulate is Precision.FP32
    assert contract.precision.downcast_at is DowncastPoint.PER_PARTIAL
    assert contract.collectives == ()
    assert contract.extra["hidden_size"] == 4096
    assert contract.extra["intermediate_size"] == 6144
    assert contract.extra["tp_world_size"] == 2
    assert contract.extra["ffn_path"] == "fast"
    assert contract.extra["batch_invariant"] is False
    assert contract.extra["stage_output_digests"]["hidden"] == "sha256:hidden"


def test_consistent_and_one_sided_switches_resolve_for_the_selected_role_only():
    both = adapter.build_contract(
        PolicyRole.ROLLOUT,
        {"gemm.ffn_path": "consistent"},
    )
    assert both.extra["ffn_path"] == "consistent"
    assert both.extra["gemm_backend"] == "cuda.det_gemm"
    assert both.extra["activation_backend"] == "cuda.swiglu"
    assert both.extra["batch_invariant"] is True

    switches = {"gemm.ffn_path": "consistent@training"}
    training = adapter.build_contract(PolicyRole.TRAINING, switches)
    rollout = adapter.build_contract(PolicyRole.ROLLOUT, switches)
    assert training.extra["ffn_path"] == "consistent"
    assert rollout.extra["ffn_path"] == "fast"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("gemm.compute_dtype", "fp8", "compute_dtype"),
        ("gemm.accumulate_dtype", "bf16", "requires FP32"),
        ("gemm.downcast_at", "whenever", "downcast_at"),
        ("gemm.hidden_size", 0, "hidden_size"),
        ("gemm.intermediate_size", -1, "intermediate_size"),
        ("gemm.tp_world_size", True, "tp_world_size"),
        ("gemm.ffn_path", "fastest", "ffn_path"),
        ("gemm.batch_invariant", "yes", "batch_invariant"),
        ("gemm.gate_up_packed", 1, "gate_up_packed"),
        ("gemm.has_bias", None, "has_bias"),
    ],
)
def test_invalid_effective_ffn_metadata_fails_closed(field, value, message):
    with pytest.raises(adapter.GemmAdapterError, match=message):
        adapter.build_contract(PolicyRole.TRAINING, _effective_ffn(**{field: value}))


# ---------------------------------------------------------- forward reduction --


def test_forward_reduce_contract_remains_supported_by_the_shared_adapter():
    native = adapter.build_contract(
        PolicyRole.TRAINING,
        {"gemm.tp_world_size": 2, "gemm.forward_reduce": "native"},
    )
    assert native.collectives[0].op is CollectiveOp.REDUCE_SCATTER
    assert native.collectives[0].reduction_order is ReductionOrder.NCCL_ALGORITHM

    reference = adapter.build_contract(
        PolicyRole.ROLLOUT,
        {"gemm.tp_world_size": 2, "gemm.forward_reduce": "rl_kernel"},
    )
    assert reference.collectives[0].op is CollectiveOp.REDUCE_SCATTER
    assert reference.collectives[0].reduction_order is ReductionOrder.GLOBAL_RANK_INDEX
    assert reference.collectives[0].determinism is DeterminismLevel.STABLE_ACROSS_TOPOLOGY


def test_observe_collectives_uses_only_the_actual_trace():
    assert adapter.observe_collectives(PolicyRole.TRAINING, _effective_ffn()) == ()

    effective = {
        "gemm.tp_world_size": 2,
        "gemm.forward_collective": _reduce_scatter_trace(),
    }
    observed = adapter.observe_collectives(PolicyRole.TRAINING, effective)
    assert len(observed) == 1
    assert observed[0].op is CollectiveOp.REDUCE_SCATTER
    assert observed[0].backend == "nccl"

    with pytest.raises(adapter.GemmAdapterError, match="group size"):
        adapter.observe_collectives(
            PolicyRole.TRAINING,
            {
                "gemm.tp_world_size": 2,
                "gemm.forward_collective": _reduce_scatter_trace(group_size=4),
            },
        )


# ------------------------------------------------- comparison with the factor --


def test_ffn_identity_matches_while_backend_representation_is_record_only():
    rollout_config = _effective_ffn(
        **{
            "gemm.weight_layout": "packed_nn_linear",
            "gemm.gate_up_packed": True,
        }
    )
    training_config = _effective_ffn(
        **{
            "gemm.ffn_path": "consistent",
            "gemm.ffn_backend": "cuda.det_gemm",
            "gemm.activation_backend": "cuda.swiglu",
            "gemm.batch_invariant": True,
        }
    )
    issues = compare_contracts(
        adapter.build_contract(PolicyRole.ROLLOUT, rollout_config),
        adapter.build_contract(PolicyRole.TRAINING, training_config),
        (FFN_FACTOR,),
    )
    assert issues == ()


def test_ffn_model_identity_mismatch_voids_the_ablation_arm():
    rollout = adapter.build_contract(PolicyRole.ROLLOUT, _effective_ffn())
    training = adapter.build_contract(
        PolicyRole.TRAINING,
        _effective_ffn(**{"gemm.intermediate_size": 12288}),
    )
    issues = compare_contracts(rollout, training, (FFN_FACTOR,))
    assert len(issues) == 1
    assert issues[0].field_path == "extra.intermediate_size"
    assert issues[0].code is ComparisonIssueCode.BITWISE_MISMATCH


def test_gemm_factors_expand_and_pass_registry_static_checks():
    reject_contradictory_factors((FFN_FACTOR, FORWARD_REDUCE_FACTOR))
    assert [variant.name for variant in build_variants(FFN_FACTOR)] == [
        "both_native",
        "both_reference",
        "training_reference_only",
        "rollout_reference_only",
    ]
    for variant in build_variants(FFN_FACTOR):
        value = variant.switch_values[FFN_FACTOR.switch.path]
        assert FFN_FACTOR.switch.parse(value) == value
    assert FFN_FACTOR.required_evidence == (
        "effective_config_readback",
        MODEL_SHAPE,
        FFN_STAGE_OUTPUTS,
    )
    assert FFN_FACTOR.call_sites == ("mlp.gate_up", "mlp.down")


# ------------------------------------------- read_effective_config / resolve --


def test_read_effective_config_accepts_runtime_shapes_and_rejects_requested_only():
    mapping = {"gemm.ffn_path": "fast"}
    assert adapter.read_effective_config(PolicyRole.TRAINING, mapping) == mapping

    class Engine:
        role = PolicyRole.ROLLOUT

        def read_effective_config(self):
            return {"gemm.ffn_path": "consistent"}

    assert adapter.read_effective_config(PolicyRole.ROLLOUT, Engine()) == {
        "gemm.ffn_path": "consistent"
    }

    class Bare:
        effective_config = {"gemm.ffn_path": "fast"}

    assert adapter.read_effective_config(PolicyRole.TRAINING, Bare()) == mapping

    with pytest.raises(adapter.GemmAdapterError, match="requested_config"):
        adapter.read_effective_config(
            PolicyRole.TRAINING,
            {"requested_config": {"gemm.ffn_path": "consistent"}},
        )

    with pytest.raises(adapter.GemmAdapterError, match="plays 'rollout'"):
        adapter.read_effective_config(PolicyRole.TRAINING, Engine())


def test_resolution_success_and_failures_retain_provenance():
    impl, resolution = adapter.resolve_implementation(
        FFN_FACTOR.id,
        PolicyRole.TRAINING,
        "math.sqrt",
    )
    assert impl(4.0) == 2.0
    assert resolution.resolved == "math.sqrt"
    assert resolution.rejected == ()

    for target, expected in (
        ("not_an_import_path", "not a dotted"),
        ("math.no_such_attribute", "no attribute"),
        ("math.pi", "not callable"),
    ):
        impl, resolution = adapter.resolve_implementation(
            FFN_FACTOR.id,
            PolicyRole.TRAINING,
            target,
        )
        assert impl is None
        assert expected in resolution.rejected[0].reason


def test_ws2_ffn_reference_path_resolves_or_reports_why_it_cannot():
    impl, resolution = adapter.resolve_implementation(
        FFN_FACTOR.id,
        PolicyRole.TRAINING,
        FFN_CONSISTENT_REFERENCE.training_impl,
    )
    if impl is None:
        assert resolution.resolved is None
        assert resolution.rejected
    else:
        assert callable(impl)
        assert resolution.resolved == FFN_CONSISTENT_REFERENCE.training_impl


# ----------------------------------------------------------------- the plugin --


def test_plugin_wires_adapter_and_discovers_both_gemm_factors():
    checks = GemmChecks
    assert checks.build_contract is adapter.build_contract
    assert checks.read_effective_config is adapter.read_effective_config
    assert checks.observe_collectives is adapter.observe_collectives
    assert checks.resolve_implementation is adapter.resolve_implementation

    ids = [factor.id for factor in GemmChecks().declare_factors()]
    assert ids == ["gemm.ffn_implementation", "gemm.forward_reduce"]
