# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Framework tests. No operator plugins involved.

The operators are written separately, so everything here uses fixture factors:
what is under test is the framework's own logic -- the gates, the matrix, the
ordering, the conflict detection.

The failure modes these lock down are the ones that make an attribution
framework worse than useless: a silently ignored switch read as "nothing here",
a missing shard read as a clean number, a broken reference quietly steering every
conclusion.
"""

from __future__ import annotations

import pytest

from rl_engine.mismatch.pipeline import (
    ContradictoryFactor,
    PluginRegistry,
    RegistrationError,
    RunContext,
    build_report,
    build_variants,
    compare_contracts,
    compute_metrics,
    diagnose,
    expand_repeats,
    missing_prerequisites,
    order_cases_by_rebind_cost,
    reject_contradictory_factors,
    render_summary,
    run_variant,
)
from rl_engine.mismatch.schema import (
    BatchPlacement,
    CollectiveContract,
    CollectiveOp,
    ComparisonIdentity,
    ComparisonIssueCode,
    ComparisonRule,
    DeterminismLevel,
    Diagnosis,
    DowncastPoint,
    DynamicSamplingDecision,
    Evidence,
    ExecutionPath,
    ExpectedOutcome,
    FactorCategory,
    FactorVariant,
    FailureMode,
    ImplementationResolution,
    KnownPitfall,
    LogprobShard,
    MismatchFactor,
    MismatchMetrics,
    NoiseFloor,
    OperatorContract,
    ParallelDim,
    PolicyRole,
    Precision,
    PrecisionProfile,
    Prerequisites,
    RebindCost,
    ReductionOrder,
    ReferenceAuthority,
    ReferenceImplementation,
    RejectedCandidate,
    ReuseKey,
    RolloutGroup,
    Switch,
    SwitchStatus,
    VariantResult,
    expected_range,
    is_silent_failure,
    reuse_level,
    tolerance_floor,
)
from tests.mismatch_cpu_backend import CpuScoringBackend

# ---------------------------------------------------------------- fixtures --


def make_identity(tokens: int = 8) -> ComparisonIdentity:
    return ComparisonIdentity(
        prompt_token_ids=tuple(range(tokens // 2)),
        response_token_ids=tuple(range(tokens)),
        active_mask=tuple([False] * (tokens // 2) + [True] * (tokens - tokens // 2)),
        position_ids=tuple(range(tokens)),
        checkpoint_id="fixture/model",
        checkpoint_revision="deadbeef",
        model_shape="L=1,H=8,Hq=2,Hkv=1,D=4",
        group=RolloutGroup(prompt_id="p0", rollout_ids=("r0",), group_size=1),
        batch_placement=BatchPlacement(
            data_parallel_rank=0, microbatch_index=0, position_in_microbatch=0
        ),
        sampling_decision=DynamicSamplingDecision(kept=True),
    )


def make_reference(name: str = "fixture_ref", **kwargs) -> ReferenceImplementation:
    return ReferenceImplementation(
        name=name,
        tier=ReferenceAuthority.SELF_WRITTEN,
        training_impl=f"{name}.training",
        rollout_impl=f"{name}.rollout",
        covers_paths=(
            ExecutionPath.TRAINING_FULL_PREFILL,
            ExecutionPath.ROLLOUT_FULL_PREFILL,
        ),
        **kwargs,
    )


def make_factor(
    factor_id: str = "fixture.swap",
    *,
    reference: ReferenceImplementation | None = None,
    rules: dict[str, ComparisonRule] | None = None,
    prerequisites: Prerequisites | None = None,
    required_evidence: tuple[str, ...] = (),
    rebind_cost: RebindCost = RebindCost.PER_REQUEST,
    allowed_values: tuple = ("native", "fixture_ref"),
) -> MismatchFactor:
    return MismatchFactor(
        id=factor_id,
        operator=factor_id.split(".")[0],
        category=FactorCategory.SHARDING_AND_REDUCTION,
        question="fixture",
        switch=Switch(
            path=factor_id,
            rebind_cost=rebind_cost,
            applies_to=(PolicyRole.ROLLOUT, PolicyRole.TRAINING),
            allowed_values=allowed_values,
        ),
        comparison_rules=rules if rules is not None else {},
        prerequisites=prerequisites or Prerequisites(),
        required_evidence=required_evidence,
        reference=reference if reference is not None else make_reference(),
    )


def make_contract(role: PolicyRole, **extra) -> OperatorContract:
    return OperatorContract(
        operator="fixture",
        role=role,
        precision=PrecisionProfile(
            compute=Precision.BF16,
            accumulate=Precision.FP32,
            downcast_at=DowncastPoint.FINAL_WRITE,
        ),
        collectives=(
            CollectiveContract(
                op=CollectiveOp.ALL_REDUCE,
                group=ParallelDim.TENSOR,
                group_size=2,
                reduction_order=extra.pop("reduction_order", ReductionOrder.GLOBAL_RANK_INDEX),
                accumulate_precision=Precision.FP32,
                downcast_at=DowncastPoint.FINAL_WRITE,
                determinism=extra.pop("determinism", DeterminismLevel.STABLE_ACROSS_RUNS),
                backend="fixture",
            ),
        ),
        extra=extra,
    )


def make_result(
    name: str,
    *,
    dlogp_mean: float = 0.0,
    dlogp_max: float = 0.0,
    clip_fraction: float = 0.0,
    status: SwitchStatus = SwitchStatus.APPLIED,
    evidence: frozenset[str] = frozenset(),
    expected: ExpectedOutcome = ExpectedOutcome.MEASURE_ONLY,
    shards: tuple[LogprobShard, ...] = (),
    resolution: ImplementationResolution | None = None,
) -> VariantResult:
    return VariantResult(
        variant=FactorVariant(name=name, switch_values={}, expected=expected),
        path=ExecutionPath.TRAINING_FULL_PREFILL,
        status=status,
        metrics=MismatchMetrics(
            active_token_count=4,
            dlogp_mean=dlogp_mean,
            dlogp_p99=dlogp_max,
            dlogp_max=dlogp_max,
            ratio_mean=1.0,
            ratio_max=1.0,
            clip_fraction=clip_fraction,
            approx_kl=0.0,
        ),
        evidence=evidence,
        effective_config={},
        logprob_shards=shards,
        resolution=resolution,
    )


def four_arms(**overrides) -> list[VariantResult]:
    """A standard four-arm set where nothing is wrong."""

    defaults = {
        "both_native": {"dlogp_mean": 0.02, "clip_fraction": 0.2},
        "both_reference": {"dlogp_max": 0.0, "expected": ExpectedOutcome.BITWISE_IDENTICAL},
        "training_reference_only": {"dlogp_mean": 0.02, "clip_fraction": 0.2},
        "rollout_reference_only": {"dlogp_mean": 0.02, "clip_fraction": 0.2},
    }
    for name, patch in overrides.items():
        defaults[name] = {**defaults.get(name, {}), **patch}
    return [make_result(name, **kwargs) for name, kwargs in defaults.items()]


# ------------------------------------------------------------ variant plan --


def test_swap_factor_expands_to_four_arms_not_two():
    """A single swap cannot attribute a side.

    Only a one-sided swap says which side is at fault, and only the two-sided
    swap proves the reference itself is sound -- so the standard set is four.
    """

    variants = build_variants(make_factor())
    assert [v.name for v in variants] == [
        "both_native",
        "both_reference",
        "training_reference_only",
        "rollout_reference_only",
    ]
    both = next(v for v in variants if v.name == "both_reference")
    assert both.expected is ExpectedOutcome.BITWISE_IDENTICAL
    assert both.replace_on == {
        PolicyRole.ROLLOUT: "fixture_ref.rollout",
        PolicyRole.TRAINING: "fixture_ref.training",
    }


def test_factor_without_reference_is_a_value_sweep():
    """No reference implementation means it is a parameter sweep.

    The distinction is derived from ``reference is None`` rather than stored in a
    separate field -- derivable state is state that can disagree with itself.
    """

    factor = make_factor(reference=None, allowed_values=(1, 2, 4))
    factor = MismatchFactor(**{**factor.__dict__, "reference": None})
    variants = build_variants(factor)
    assert [v.name for v in variants] == ["value_1", "value_2", "value_4"]


def test_fp64_oracle_arm_is_added_when_declared():
    factor = make_factor(reference=make_reference(fp64_oracle="fixture.fp64"))
    assert "fp64_oracle" in [v.name for v in build_variants(factor)]


def test_cases_are_ordered_cheapest_rebuild_first():
    """160 cases in random order restart the process for nearly every one.

    Ordering is what makes the whole run finish, not a nicety.
    """

    cheap = make_factor("a.cheap", rebind_cost=RebindCost.PER_REQUEST)
    expensive = make_factor("b.expensive", rebind_cost=RebindCost.PROCESS_RESTART)
    middle = make_factor("c.middle", rebind_cost=RebindCost.ENGINE_REBUILD)

    cases = [
        (expensive, build_variants(expensive)[0]),
        (cheap, build_variants(cheap)[0]),
        (middle, build_variants(middle)[0]),
    ]
    ordered = order_cases_by_rebind_cost(cases)
    assert [factor.id for factor, _ in ordered] == ["a.cheap", "c.middle", "b.expensive"]


# --------------------------------------------------------- static rejection --


def test_topology_independence_claim_with_nccl_order_is_rejected_before_running():
    """Claiming topology independence while reducing by NCCL's choice is
    self-contradictory -- reject at planning time, not after burning machine time."""

    from rl_engine.mismatch.schema import RequiredSetting, SettingChannel

    contradictory = CollectiveContract(
        op=CollectiveOp.ALL_REDUCE,
        group=ParallelDim.TENSOR,
        group_size=4,
        reduction_order=ReductionOrder.NCCL_ALGORITHM,
        accumulate_precision=Precision.FP32,
        downcast_at=DowncastPoint.FINAL_WRITE,
        determinism=DeterminismLevel.STABLE_ACROSS_TOPOLOGY,
        backend="nccl",
    )
    factor = make_factor(
        reference=make_reference(
            required_settings=(
                RequiredSetting(
                    key="collective",
                    value=contradictory,
                    channel=SettingChannel.CALL_ARG,
                ),
            )
        )
    )

    with pytest.raises(ContradictoryFactor, match="stable_across_topology"):
        reject_contradictory_factors([factor])


def test_prerequisites_report_what_is_missing_not_a_yes_no():
    """A whitelist probe returning the missing items, not an opaque boolean."""

    factor = make_factor(
        prerequisites=Prerequisites(
            required_ops=("rl_kernel.reduce_scatter",),
            min_gpu_count=2,
            blocked_by=("#247",),
        )
    )
    unmet = missing_prerequisites(factor, available_ops=frozenset(), gpu_count=0)
    reasons = " ".join(item.reason for item in unmet)
    assert "reduce_scatter" in reasons
    assert "2 devices" in reasons
    assert "#247" in reasons


# ------------------------------------------------------------- comparison ---


def test_record_only_fields_are_never_compared():
    """RECORD_ONLY exists so structural differences do not drown real findings.

    Packed QKV differs in form between the two sides while the arithmetic is
    identical; comparing it would turn every run red.
    """

    rollout = make_contract(PolicyRole.ROLLOUT, qkv_layout="packed")
    training = make_contract(PolicyRole.TRAINING, qkv_layout="split")
    factor = make_factor(rules={"extra.qkv_layout": ComparisonRule.RECORD_ONLY})

    assert compare_contracts(rollout, training, [factor]) == ()


def test_semantic_mismatch_is_reported_with_both_sides_values():
    rollout = make_contract(PolicyRole.ROLLOUT, rope_theta=10000.0)
    training = make_contract(PolicyRole.TRAINING, rope_theta=1000000.0)
    factor = make_factor(rules={"extra.rope_theta": ComparisonRule.MUST_MATCH_SEMANTICALLY})

    issues = compare_contracts(rollout, training, [factor])
    assert len(issues) == 1
    assert issues[0].code is ComparisonIssueCode.SEMANTIC_MISMATCH
    assert issues[0].values[PolicyRole.ROLLOUT] == 10000.0
    assert issues[0].values[PolicyRole.TRAINING] == 1000000.0


def test_missing_field_is_reported_rather_than_raising():
    factor = make_factor(rules={"extra.absent": ComparisonRule.MUST_MATCH_BITWISE})
    issues = compare_contracts(
        make_contract(PolicyRole.ROLLOUT), make_contract(PolicyRole.TRAINING), [factor]
    )
    assert issues[0].code is ComparisonIssueCode.REQUIRED_FIELD_MISSING


def test_indexed_path_reaches_into_collectives():
    rollout = make_contract(PolicyRole.ROLLOUT, reduction_order=ReductionOrder.ARRIVAL)
    training = make_contract(PolicyRole.TRAINING, reduction_order=ReductionOrder.GLOBAL_RANK_INDEX)
    factor = make_factor(
        rules={"collectives[0].reduction_order": ComparisonRule.MUST_MATCH_SEMANTICALLY}
    )
    issues = compare_contracts(rollout, training, [factor])
    assert issues[0].field_path == "collectives[0].reduction_order"


def test_one_side_promising_more_determinism_is_flagged():
    """Comparing a topology-independent implementation against a
    non-reproducible one measures the weaker side's noise, not the gap."""

    rollout = make_contract(PolicyRole.ROLLOUT, determinism=DeterminismLevel.STABLE_ACROSS_TOPOLOGY)
    training = make_contract(PolicyRole.TRAINING, determinism=DeterminismLevel.NONE)
    issues = compare_contracts(rollout, training, [make_factor()])
    assert any(i.code is ComparisonIssueCode.DETERMINISM_INCOMPATIBLE for i in issues)


# ---------------------------------------------------------------- metrics ---


def test_identical_logprobs_give_zero_clip_fraction():
    metrics = compute_metrics([-1.0, -2.0], [-1.0, -2.0], [True, True])
    assert metrics.dlogp_max == 0.0
    assert metrics.clip_fraction == 0.0


def test_clip_fraction_counts_tokens_past_the_grpo_clip_edge():
    """Past ln(1+eps) a token is clipped and its gradient signal is discarded.

    That is the actual mechanism by which mismatch breaks training, which is why
    this and not the mean is the headline number.
    """

    # 0.30 > ln(1.2) = 0.182, so one of the two is clipped.
    metrics = compute_metrics([-1.0, -1.0], [-0.70, -1.05], [True, True])
    assert metrics.clip_fraction == 0.5


def test_inactive_tokens_are_excluded():
    metrics = compute_metrics([-1.0, -1.0], [-9.0, -1.0], [False, True])
    assert metrics.active_token_count == 1
    assert metrics.dlogp_max == 0.0


def test_mean_within_band_but_tail_past_the_edge_is_a_silent_failure():
    """The most common false negative: the headline looks healthy while gradient
    signal is being thrown away."""

    metrics = MismatchMetrics(
        active_token_count=100,
        dlogp_mean=0.004,  # squarely inside the dense production band
        dlogp_p99=0.25,
        dlogp_max=0.4,
        ratio_mean=1.0,
        ratio_max=1.5,
        clip_fraction=0.02,
        approx_kl=0.001,
    )
    assert is_silent_failure(metrics)


# ------------------------------------------------------------- thresholds ---


def test_low_floor_expects_bitwise_not_the_production_band():
    """Reading the anchor floor against the production table hides real bugs.

    At the anchor floor the expectation is bitwise; judging it by 0.002-0.008
    would call a definite operator error "normal".
    """

    production = expected_range("dense", NoiseFloor.PRODUCTION)
    anchor = expected_range("dense", NoiseFloor.SINGLE_LAYER_ANCHOR)
    assert production.dlogp_mean == (0.002, 0.008)
    assert anchor.dlogp_mean == (0.0, 0.0)
    assert tolerance_floor("dense", NoiseFloor.SINGLE_LAYER_ANCHOR) < 1e-5


def test_large_moe_band_is_wider_and_not_a_bug():
    band = expected_range("large_moe", NoiseFloor.PRODUCTION, routing_replay=False)
    assert band.dlogp_mean == (0.01, 0.03)
    assert "do not file it as a bug" in band.note


# ---------------------------------------------------------------- the gates --


def test_a_variant_that_did_not_apply_never_reaches_the_matrix():
    """A silently reverted switch reads as "the deviation did not change", which
    reads as NOT_THIS_FACTOR. A false negative that looks like a clean result."""

    results = four_arms()
    results[2] = make_result(
        "training_reference_only",
        status=SwitchStatus.FELL_BACK,
        resolution=ImplementationResolution(
            requested="fixture_ref.training",
            resolved=None,
            rejected=(RejectedCandidate(name="fixture_ref.training", reason="library missing"),),
        ),
    )
    report = diagnose(make_factor(), results, noise_floor=NoiseFloor.PRODUCTION)
    assert report.diagnosis is Diagnosis.VARIANT_DID_NOT_APPLY
    assert "library missing" in report.diagnosis_reason


def test_missing_evidence_is_not_the_same_as_nothing_found():
    factor = make_factor(required_evidence=(Evidence.MODEL_STATE_FINGERPRINT.value,))
    report = diagnose(factor, four_arms(), noise_floor=NoiseFloor.PRODUCTION)
    assert report.diagnosis is Diagnosis.INSUFFICIENT_EVIDENCE


def test_incomplete_logprob_shards_block_the_verdict():
    """Under TP/CP each rank holds one slice. One slice short and the LSE
    denominator loses a chunk, so logp comes out systematically high -- wrong in
    a way that does not show."""

    partial = (LogprobShard(rank=0, world_size=4, selected_logprobs=[0.0]),)
    results = four_arms()
    results[0] = make_result("both_native", dlogp_mean=0.02, clip_fraction=0.2, shards=partial)
    report = diagnose(make_factor(), results, noise_floor=NoiseFloor.PRODUCTION)
    assert report.diagnosis is Diagnosis.INSUFFICIENT_EVIDENCE
    assert "1 of 4" in report.diagnosis_reason


def test_failed_pitfall_guard_blocks_the_verdict():
    guard = KnownPitfall(
        id="rope_hook_not_covered",
        mode=FailureMode.MISSING_INSTRUMENTATION,
        symptom="RoPE looks identical",
        actual_cause="the hook never captured it",
        guard="dump post-RoPE Q/K on both sides",
        guard_runs_at=NoiseFloor.SINGLE_LAYER_ANCHOR,
    )
    report = diagnose(
        make_factor(), four_arms(), noise_floor=NoiseFloor.PRODUCTION, failed_guards=[guard]
    )
    assert report.diagnosis is Diagnosis.INSUFFICIENT_EVIDENCE
    assert "rope_hook_not_covered" in report.diagnosis_reason


# ------------------------------------------------------- the matrix proper --


def test_broken_reference_voids_the_factor_rather_than_attributing_a_side():
    """Without this gate, one wrong reference quietly steers every attribution --
    worse than having no framework at all."""

    results = four_arms(both_reference={"dlogp_max": 0.5})
    report = diagnose(make_factor(), results, noise_floor=NoiseFloor.PRODUCTION)
    assert report.diagnosis is Diagnosis.REFERENCE_ITSELF_IS_BROKEN


def test_only_training_side_converging_attributes_the_training_side():
    results = four_arms(training_reference_only={"dlogp_mean": 0.0, "clip_fraction": 0.0})
    report = diagnose(make_factor(), results, noise_floor=NoiseFloor.PRODUCTION)
    assert report.diagnosis is Diagnosis.CAUSED_BY_TRAINING_SIDE


def test_only_rollout_side_converging_attributes_the_rollout_side():
    results = four_arms(rollout_reference_only={"dlogp_mean": 0.0, "clip_fraction": 0.0})
    report = diagnose(make_factor(), results, noise_floor=NoiseFloor.PRODUCTION)
    assert report.diagnosis is Diagnosis.CAUSED_BY_ROLLOUT_SIDE


def test_both_sides_converging_leaves_the_reference_as_the_only_anchor():
    results = four_arms(
        training_reference_only={"dlogp_mean": 0.0, "clip_fraction": 0.0},
        rollout_reference_only={"dlogp_mean": 0.0, "clip_fraction": 0.0},
    )
    report = diagnose(make_factor(), results, noise_floor=NoiseFloor.PRODUCTION)
    assert report.diagnosis is Diagnosis.CAUSED_BY_BOTH_SIDES


def test_neither_side_moving_means_look_upstream():
    report = diagnose(make_factor(), four_arms(), noise_floor=NoiseFloor.PRODUCTION)
    assert report.diagnosis is Diagnosis.NOT_THIS_FACTOR


def test_convergence_is_judged_on_clip_fraction_not_the_mean():
    """At every production floor the mean sits far below the clip edge, so
    judging on the mean would mark almost every factor NOT_THIS_FACTOR."""

    results = four_arms(
        both_native={"dlogp_mean": 0.005, "clip_fraction": 0.30},
        training_reference_only={"dlogp_mean": 0.005, "clip_fraction": 0.01},
    )
    report = diagnose(make_factor(), results, noise_floor=NoiseFloor.PRODUCTION)
    assert report.diagnosis is Diagnosis.CAUSED_BY_TRAINING_SIDE


# ------------------------------------------------------------- repeats ------


def test_repeat_under_expands_to_the_cartesian_product():
    """The one exception to "one variant, one execution".

    It verifies the self-check gate's own premise: both_reference can only anchor
    if the fixed-order implementation really did fix the order.
    """

    variant = FactorVariant(
        name="both_reference",
        switch_values={},
        repeat_under={"NCCL_ALGO": ("Ring", "Tree"), "NCCL_PROTO": ("Simple", "LL")},
    )
    assert len(expand_repeats(variant)) == 4


def test_variant_without_repeats_runs_once():
    assert expand_repeats(FactorVariant(name="x", switch_values={})) == ({},)


# ------------------------------------------------------------- registry ----


def test_duplicate_factor_id_is_rejected_at_registration():
    registry = PluginRegistry()

    class First:
        operator = "first"

        def declare_factors(self):
            return (make_factor("shared.id"),)

    class Second:
        operator = "second"

        def declare_factors(self):
            return (make_factor("shared.id"),)

    registry.register(First)
    with pytest.raises(RegistrationError, match="duplicate factor id"):
        registry.register(Second)


def test_same_contract_field_at_two_different_rules_is_rejected():
    """One field cannot be bitwise-required in one operator and record-only in
    another; the comparison would depend on which factor happened to run."""

    registry = PluginRegistry()

    class Strict:
        operator = "strict"

        def declare_factors(self):
            return (
                make_factor("strict.a", rules={"extra.shared": ComparisonRule.MUST_MATCH_BITWISE}),
            )

    class Loose:
        operator = "loose"

        def declare_factors(self):
            return (make_factor("loose.b", rules={"extra.shared": ComparisonRule.RECORD_ONLY}),)

    registry.register(Strict)
    with pytest.raises(RegistrationError, match="declared as"):
        registry.register(Loose)


def test_duplicate_factor_inside_one_plugin_is_rejected():
    registry = PluginRegistry()

    class Broken:
        operator = "broken"

        def declare_factors(self):
            return (make_factor("broken.same"), make_factor("broken.same"))

    with pytest.raises(RegistrationError, match="duplicate factor id"):
        registry.register(Broken)


def test_duplicate_switch_path_inside_one_plugin_is_rejected():
    registry = PluginRegistry()
    first = make_factor("broken.first")
    second = make_factor("broken.second")
    second = MismatchFactor(
        **{
            **second.__dict__,
            "switch": Switch(
                path=first.switch.path,
                rebind_cost=RebindCost.PER_REQUEST,
                applies_to=(PolicyRole.ROLLOUT, PolicyRole.TRAINING),
                allowed_values=("native", "fixture_ref"),
            ),
        }
    )

    class Broken:
        operator = "broken"

        def declare_factors(self):
            return (first, second)

    with pytest.raises(RegistrationError, match="duplicate switch path"):
        registry.register(Broken)


def test_registry_starts_empty_because_operators_ship_separately():
    assert PluginRegistry().operators() == ()


# ------------------------------------------------------------ fingerprints --


def test_reuse_level_returns_the_coarsest_thing_that_changed():
    base = ReuseKey(process="p", process_group="g", engine="e", request="r")
    assert reuse_level(base, base) is RebindCost.PER_REQUEST
    assert reuse_level(base, ReuseKey("p", "g", "e2", "r")) is RebindCost.ENGINE_REBUILD
    assert reuse_level(base, ReuseKey("p", "g2", "e", "r")) is RebindCost.PROCESS_GROUP_REBUILD
    assert reuse_level(base, ReuseKey("p2", "g", "e", "r")) is RebindCost.PROCESS_RESTART


# ---------------------------------------------------------------- runner ----


class _Checks:
    """Minimal plugin used to drive the runner. Not a real operator."""

    operator = "fixture"

    def __init__(self, resolvable: bool = True):
        self.resolvable = resolvable

    def declare_factors(self):
        return (make_factor(),)

    def build_contract(self, role, switch_values):
        return make_contract(role)

    def read_effective_config(self, role, adapter):
        return {}

    def observe_collectives(self, role, adapter):
        return ()

    def resolve_implementation(self, factor_id, role, impl_name):
        if not self.resolvable:
            return None, ImplementationResolution(
                requested=impl_name,
                resolved=None,
                rejected=(RejectedCandidate(name=impl_name, reason="not built on this host"),),
            )
        return (lambda *a, **k: None), ImplementationResolution(
            requested=impl_name, resolved=impl_name
        )


def test_runner_detects_a_one_sided_injected_deviation():
    """End to end on CPU: bias one side and the metrics must see exactly that."""

    identity = make_identity()
    backends = {
        PolicyRole.ROLLOUT: CpuScoringBackend(role=PolicyRole.ROLLOUT),
        PolicyRole.TRAINING: CpuScoringBackend(role=PolicyRole.TRAINING, bias=0.25),
    }
    factor = make_factor()
    variant = build_variants(factor)[0]  # both_native

    result = run_variant(factor, variant, _Checks(), backends, RunContext(identity=identity))
    assert result.status is SwitchStatus.APPLIED
    assert result.metrics.dlogp_mean == pytest.approx(0.25, abs=1e-9)


def test_runner_builds_contracts_from_effective_readback_not_requested_values():
    class ReadbackBackend(CpuScoringBackend):
        actual: str

        def __init__(self, *, role, actual):
            super().__init__(role=role)
            self.actual = actual

        def score(self, role, identity, switch_values, replacement):
            scores, readback = super().score(role, identity, switch_values, replacement)
            return scores, {**readback, "fixture.actual": self.actual}

    class ReadbackChecks(_Checks):
        def build_contract(self, role, switch_values):
            return make_contract(role, actual=switch_values["fixture.actual"])

        def read_effective_config(self, role, adapter):
            return dict(adapter)

    identity = make_identity()
    backends = {
        PolicyRole.ROLLOUT: ReadbackBackend(role=PolicyRole.ROLLOUT, actual="runtime-a"),
        PolicyRole.TRAINING: ReadbackBackend(role=PolicyRole.TRAINING, actual="runtime-b"),
    }
    factor = make_factor(
        rules={"extra.actual": ComparisonRule.MUST_MATCH_SEMANTICALLY}
    )
    variant = build_variants(factor)[0]

    result = run_variant(
        factor,
        variant,
        ReadbackChecks(),
        backends,
        RunContext(identity=identity),
    )
    assert result.comparison_issues[0].field_path == "extra.actual"
    assert result.effective_config["rollout.fixture.actual"] == "runtime-a"


def test_reference_swap_removes_the_injected_deviation():
    """both_reference puts both sides on one implementation, so an injected
    per-side bias must vanish -- this is the self-check gate working."""

    identity = make_identity()
    backends = {
        PolicyRole.ROLLOUT: CpuScoringBackend(role=PolicyRole.ROLLOUT),
        PolicyRole.TRAINING: CpuScoringBackend(role=PolicyRole.TRAINING, bias=0.25),
    }
    factor = make_factor()
    both_reference = build_variants(factor)[1]

    result = run_variant(factor, both_reference, _Checks(), backends, RunContext(identity=identity))
    assert result.metrics.dlogp_max == 0.0


def test_unresolvable_implementation_is_recorded_as_fell_back_with_a_trace():
    """A single fallback_reason string is not enough: you need to know which
    candidates were tried and why each was rejected."""

    identity = make_identity()
    backends = {
        PolicyRole.ROLLOUT: CpuScoringBackend(role=PolicyRole.ROLLOUT),
        PolicyRole.TRAINING: CpuScoringBackend(role=PolicyRole.TRAINING),
    }
    factor = make_factor()
    both_reference = build_variants(factor)[1]

    result = run_variant(
        factor, both_reference, _Checks(resolvable=False), backends, RunContext(identity=identity)
    )
    assert result.status is SwitchStatus.FELL_BACK
    assert result.resolution.resolved is None
    assert result.resolution.rejected[0].reason == "not built on this host"


def test_unstable_backend_fails_the_topology_independence_assertion():
    """A backend whose output moves with the environment must not pass as
    topology independent."""

    identity = make_identity()
    backends = {
        PolicyRole.ROLLOUT: CpuScoringBackend(
            role=PolicyRole.ROLLOUT, unstable_under=frozenset({"NCCL_ALGO"})
        ),
        PolicyRole.TRAINING: CpuScoringBackend(role=PolicyRole.TRAINING),
    }
    factor = make_factor()
    variant = FactorVariant(
        name="both_reference",
        switch_values={},
        expected=ExpectedOutcome.BITWISE_IDENTICAL,
        repeat_under={"NCCL_ALGO": ("Ring", "Tree")},
    )

    result = run_variant(factor, variant, _Checks(), backends, RunContext(identity=identity))
    assert result.status is SwitchStatus.ERROR


# ---------------------------------------------------------------- report ----


def test_report_ranks_hypotheses_and_summarises():
    from rl_engine.mismatch.model_meta import QWEN3_CORRESPONDENCES, QWEN3_EDGES

    attributed = diagnose(
        make_factor("mlp.forward_reduce"),
        four_arms(training_reference_only={"dlogp_mean": 0.0, "clip_fraction": 0.0}),
        noise_floor=NoiseFloor.SHARDED_SINGLE_NODE,
    )
    clean = diagnose(
        make_factor("attn.rope_fusion"),
        four_arms(),
        noise_floor=NoiseFloor.SHARDED_SINGLE_NODE,
    )

    report = build_report(
        [attributed, clean],
        noise_floor=NoiseFloor.SHARDED_SINGLE_NODE,
        correspondences=QWEN3_CORRESPONDENCES,
        edges=QWEN3_EDGES,
    )
    assert len(report.hypotheses) == 1
    assert report.hypotheses[0].rank == 1

    summary = render_summary(report)
    assert "caused_by_training_side: 1" in summary
    assert "not_this_factor: 1" in summary


def test_only_proven_equivalences_filter_findings():
    """An equivalence without a test proving it is not trusted -- otherwise
    "filtering false positives" quietly becomes "hiding real findings"."""

    from rl_engine.mismatch.pipeline import filter_known_equivalences
    from rl_engine.mismatch.schema import ModuleCorrespondence

    unproven = ModuleCorrespondence(
        semantic_name="mlp.gate_up",
        training_module="t",
        rollout_module="r",
        equivalence="concat_on_dim0",
        verified_by=None,
    )
    kept, filtered = filter_known_equivalences([unproven], ["mlp.gate_up"])
    assert kept == ("mlp.gate_up",)
    assert filtered == ()
