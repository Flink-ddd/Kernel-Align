from dataclasses import dataclass

import pytest
import torch

from rl_engine.alignment.cross_config.operator_comparison import (
    OPERATOR_COMPARISON_SPECS,
    PHASE4_TARGET_OPERATORS,
    ForwardChainStep,
    OperatorPair,
    OperatorTolerance,
    build_single_card_batch_invariance_cases,
    build_strict_backend_admission_report,
    compare_batch_invariance,
    compare_operator_outputs,
    compare_operator_pair,
    iter_operator_comparison_specs,
    reference_rmsnorm,
    reference_selected_logprobs,
    run_deterministic_repeatability_check,
    run_forward_chain_comparison,
    run_reference_operator,
)


@dataclass(frozen=True)
class _BackendCapability:
    operator: str
    backend_id: str
    implementation_kind: str = "optimized"
    deterministic: bool = False
    batch_invariant: bool = True
    strict_fast_eligible: bool = False


def _backend(
    *,
    operator: str = "logp",
    backend_id: str = "rlk.logp.fake",
    deterministic: bool = False,
    batch_invariant: bool = True,
) -> _BackendCapability:
    return _BackendCapability(
        operator=operator,
        backend_id=backend_id,
        deterministic=deterministic,
        batch_invariant=batch_invariant,
    )


@pytest.mark.unit
def test_phase4_operator_registry_covers_targets_and_structured_unsupported():
    specs = {spec.op_name: spec for spec in iter_operator_comparison_specs()}

    assert tuple(specs) == PHASE4_TARGET_OPERATORS
    assert set(OPERATOR_COMPARISON_SPECS) == set(PHASE4_TARGET_OPERATORS)
    for spec in specs.values():
        assert spec.boundary
        assert spec.category in {"forward", "logp", "loss"}
        assert spec.batch_invariance_axes == (
            "single_sample_vs_mixed_batch",
            "padding_packing_layout",
            "row_position",
            "active_mask_density",
        )
        if spec.supported:
            assert spec.reference_impl is not None
        else:
            unsupported = spec.unsupported_result(metadata={"source": "unit"})
            assert unsupported.status == "unsupported"
            assert unsupported.unsupported_reason
            assert unsupported.metadata == {"source": "unit"}

    assert not specs["grpo_fragment"].supported
    assert not specs["dpo_fragment"].supported
    with pytest.raises(NotImplementedError, match="GRPO"):
        run_reference_operator("grpo_fragment", ppo_kl=torch.ones(1), advantages=torch.ones(1))


@pytest.mark.unit
def test_rmsnorm_reference_path_and_drift_metrics():
    hidden = torch.tensor([[1.0, -2.0, 3.0, -4.0], [0.5, 1.5, -2.5, 3.5]])
    weight = torch.tensor([1.0, 0.5, 2.0, -1.0])

    actual = reference_rmsnorm(hidden, weight, eps=1e-5)
    manual = hidden * torch.rsqrt(hidden.pow(2).mean(dim=-1, keepdim=True) + 1e-5) * weight
    torch.testing.assert_close(actual, manual, rtol=1e-6, atol=1e-6)

    pass_result = compare_operator_outputs("rmsnorm", actual, actual.clone())
    assert pass_result.status == "passed"
    assert pass_result.metrics["max_abs_error"] == pytest.approx(0.0)

    infer = actual.clone()
    infer[0, 1] += 5e-5
    fail_result = compare_operator_outputs(
        "rmsnorm",
        actual,
        infer,
        tolerance=OperatorTolerance(atol=1e-6, rtol=0.0),
    )

    assert fail_result.status == "failed"
    assert fail_result.failure_reason == "operator outputs exceeded tolerance"
    assert fail_result.metrics["max_abs_error"] == pytest.approx(5e-5, rel=1e-3)
    assert fail_result.metrics["p99_abs_error"] > 0.0


@pytest.mark.unit
def test_logp_comparison_reports_active_token_dlogp_contribution():
    logits = torch.tensor([[1.0, 0.0, -1.0], [0.5, 2.0, -0.5], [1.5, -0.5, 0.25]])
    target_ids = torch.tensor([0, 1, 2])
    train = reference_selected_logprobs(logits, target_ids)
    infer = train + torch.tensor([-0.1, 10.0, 0.2])
    active_mask = torch.tensor([1, 0, 1], dtype=torch.bool)

    result = compare_operator_outputs(
        "logp",
        train,
        infer,
        tolerance=OperatorTolerance(atol=1e-6, rtol=0.0),
        active_token_mask=active_mask,
    )

    assert result.status == "failed"
    assert result.metrics["active_token_count"] == pytest.approx(2.0)
    assert result.metrics["active_token_dlogp_abs_mean"] == pytest.approx(0.15, rel=1e-5)
    assert result.metrics["active_token_dlogp_signed_sum"] == pytest.approx(-0.1, rel=1e-5)
    assert result.metrics["max_abs_error"] == pytest.approx(10.0)


@pytest.mark.unit
def test_logp_reference_keeps_fp32_accumulation_dtype():
    logits = torch.tensor([[1.0, 0.0, -1.0]], dtype=torch.bfloat16)
    target_ids = torch.tensor([0])

    selected = reference_selected_logprobs(logits, target_ids)

    assert selected.dtype == torch.float32


@pytest.mark.unit
def test_ratio_kl_active_dlogp_metrics_use_only_dlogp_leaf():
    log_probs = torch.tensor([-1.0, -2.0, -3.0])
    old_log_probs = torch.tensor([-1.2, -1.5, -3.25])
    active_mask = torch.tensor([1, 0, 1], dtype=torch.bool)
    train = run_reference_operator(
        "ratio_kl", log_probs=log_probs, old_log_probs=old_log_probs, loss_mask=active_mask
    )
    infer = {key: value.clone() for key, value in train.items()}
    infer["dlogp"][0] += 0.2
    infer["ratio"][0] += 100.0

    result = compare_operator_outputs(
        "ratio_kl",
        train,
        infer,
        tolerance=OperatorTolerance(atol=1e-6, rtol=0.0),
        active_token_mask=active_mask,
    )

    assert result.status == "failed"
    assert "output.dlogp" in result.compared_tensors
    assert result.metrics["max_abs_error"] == pytest.approx(100.0)
    assert result.metrics["active_token_count"] == pytest.approx(2.0)
    assert result.metrics["active_token_dlogp_abs_mean"] == pytest.approx(0.1, rel=1e-5)


@pytest.mark.unit
def test_nonfinite_drift_and_bad_active_mask_fail_closed():
    nonfinite = compare_operator_outputs("logp", torch.tensor([float("nan")]), torch.tensor([0.0]))

    assert nonfinite.status == "failed"
    assert nonfinite.failure_reason == "operator outputs produced non-finite deltas"
    assert nonfinite.metrics["nonfinite_delta_count"] == pytest.approx(1.0)

    bad_mask = compare_operator_outputs(
        "logp",
        torch.zeros(2),
        torch.zeros(2),
        active_token_mask=torch.ones(3, dtype=torch.bool),
    )
    assert bad_mask.status == "failed"
    assert bad_mask.failure_reason == "active_token_mask did not align with compared tensor leaves"


@pytest.mark.unit
def test_operator_pair_runs_train_infer_role_pairing():
    hidden = torch.randn(3, 4)
    weight = torch.randn(5, 4)
    pair = OperatorPair(
        op_name="matmul_projection",
        train=lambda **kwargs: run_reference_operator("matmul_projection", **kwargs),
        infer=lambda **kwargs: run_reference_operator("matmul_projection", **kwargs),
        train_kwargs={"hidden": hidden, "weight": weight},
        infer_kwargs={"hidden": hidden.clone(), "weight": weight.clone()},
        metadata={"gradients_checked": True},
    )

    result = compare_operator_pair(pair)

    assert result.status == "passed"
    assert result.metadata["comparison"] == "train_vs_infer"
    assert result.metadata["gradients_checked"] is True


@pytest.mark.unit
def test_forward_chain_accumulates_small_drift_and_reports_first_boundary():
    initial = torch.tensor([1.0, -2.0])
    tolerance = OperatorTolerance(atol=1e-4, rtol=0.0)
    steps = (
        ForwardChainStep(
            "matmul_projection",
            train=lambda value: value + 1.0,
            infer=lambda value: value + 1.0 + 2e-5,
            tolerance=tolerance,
        ),
        ForwardChainStep(
            "rmsnorm",
            train=lambda value: value * 3.0,
            infer=lambda value: value * 3.0,
            tolerance=tolerance,
        ),
        ForwardChainStep(
            "swiglu",
            train=lambda value: value * 3.0,
            infer=lambda value: value * 3.0,
            tolerance=tolerance,
        ),
    )

    result = run_forward_chain_comparison(initial, initial.clone(), steps)

    assert result.status == "failed"
    assert result.first_drift_operator == "matmul_projection"
    assert result.steps[0].status == "passed"
    assert result.steps[-1].status == "failed"
    assert result.cumulative_metrics["failed_step_count"] == pytest.approx(1.0)
    assert result.cumulative_metrics["first_failed_step_index"] == pytest.approx(2.0)
    assert result.cumulative_metrics["max_abs_error"] > result.steps[0].metrics["max_abs_error"]


@pytest.mark.unit
def test_batch_invariance_cases_and_fixed_sample_comparison():
    cases = build_single_card_batch_invariance_cases("rmsnorm", sample_position=1)

    assert [case.case for case in cases] == [
        "same_sample_mixed_batch",
        "padding_packing_layout",
        "row_position",
        "active_mask_density",
    ]
    assert any("active_mask_density" in case.varied_axes for case in cases)

    weight = torch.tensor([0.75, -1.0, 1.5])
    fixed_sample = torch.tensor([1.0, -2.0, 0.5])
    neighbor = torch.tensor([-3.0, 0.25, 2.5])
    sample_alone = reference_rmsnorm(fixed_sample, weight)
    mixed = reference_rmsnorm(torch.stack([neighbor, fixed_sample]), weight)

    result = compare_batch_invariance(
        "rmsnorm", sample_alone, mixed, sample_position=1, case="row_position"
    )

    assert result.status == "passed"
    assert result.metadata["comparison"] == "batch_invariance"
    assert result.metadata["case"] == "row_position"


@pytest.mark.unit
def test_logp_batch_invariance_handles_padding_and_active_mask_density():
    fixed_logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 3.0, -2.0], [1.0, -1.0, 2.0]])
    fixed_targets = torch.tensor([0, 1, 2])
    neighbor_logits = torch.tensor([[-1.0, 2.0, 0.0], [3.0, -2.0, 1.0], [0.5, 0.25, -0.5]])
    neighbor_targets = torch.tensor([1, 0, 0])
    active_mask = torch.tensor([1, 0, 1], dtype=torch.bool)
    sample_alone = reference_selected_logprobs(fixed_logits, fixed_targets)
    mixed = reference_selected_logprobs(
        torch.stack([neighbor_logits, fixed_logits]),
        torch.stack([neighbor_targets, fixed_targets]),
    )

    result = compare_batch_invariance(
        "logp",
        sample_alone,
        mixed,
        sample_position=1,
        case="active_mask_density",
        active_token_mask=active_mask,
    )

    assert result.status == "passed"
    assert result.metrics["active_token_count"] == pytest.approx(2.0)
    assert result.metrics["active_token_dlogp_abs_max"] == pytest.approx(0.0)


@pytest.mark.unit
def test_deterministic_repeatability_is_gated_by_backend_capability():
    calls = 0

    def should_not_run():
        nonlocal calls
        calls += 1
        return torch.ones(2)

    unsupported = run_deterministic_repeatability_check(
        "logp", _backend(deterministic=False), should_not_run
    )

    assert unsupported.status == "unsupported"
    assert "does not advertise" in unsupported.unsupported_reason
    assert calls == 0

    deterministic = _backend(deterministic=True)
    passed = run_deterministic_repeatability_check(
        "logp", deterministic, lambda: {"logp": torch.ones(2)}
    )

    assert passed.status == "passed"
    assert passed.metrics["bitwise_mismatch_count"] == pytest.approx(0.0)


@pytest.mark.unit
def test_deterministic_repeatability_reports_bitwise_mismatch():
    counter = 0

    def run_once():
        nonlocal counter
        counter += 1
        return torch.tensor([float(counter)])

    result = run_deterministic_repeatability_check(
        "logp", _backend(deterministic=True), run_once, repetitions=3
    )

    assert result.status == "failed"
    assert result.metrics["repeat_count"] == pytest.approx(3.0)
    assert result.metrics["bitwise_mismatch_count"] == pytest.approx(2.0)


@pytest.mark.unit
def test_strict_admission_requires_batch_invariance_outputs_and_gradients():
    output = torch.ones(2)
    pass_result = compare_operator_outputs(
        "rmsnorm",
        output,
        output.clone(),
        metadata={"gradients_checked": True},
    )
    missing_gradient_result = compare_operator_outputs("rmsnorm", output, output.clone())
    fail_result = compare_operator_outputs(
        "rmsnorm",
        output,
        output + 1.0,
        tolerance=OperatorTolerance(atol=1e-6, rtol=0.0),
    )

    missing_batch_invariance = build_strict_backend_admission_report(
        _backend(operator="rmsnorm", batch_invariant=False),
        (pass_result,),
    )
    assert not missing_batch_invariance.strict_fast_eligible
    assert "batch_invariant_capability_missing" in missing_batch_invariance.reasons

    missing_gradient = build_strict_backend_admission_report(
        _backend(operator="rmsnorm", batch_invariant=True),
        (missing_gradient_result,),
    )
    assert not missing_gradient.strict_fast_eligible
    assert "gradient_comparison_missing" in missing_gradient.reasons

    failed = build_strict_backend_admission_report(_backend(operator="rmsnorm"), (fail_result,))
    assert not failed.strict_fast_eligible
    assert failed.failed_comparisons == ("rmsnorm",)

    missing_comparisons = build_strict_backend_admission_report(_backend(operator="rmsnorm"), ())
    assert not missing_comparisons.strict_fast_eligible
    assert "comparison_missing" in missing_comparisons.reasons

    admitted = build_strict_backend_admission_report(_backend(operator="rmsnorm"), (pass_result,))
    assert admitted.strict_fast_eligible
    assert admitted.reasons == ()
