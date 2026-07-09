# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch

from rl_engine.kernels.gtest.op_checks import CandidateSpec, OperatorCase, run_operator_suite
from rl_engine.kernels.ops.pytorch.loss.logp import NativeLogpOp


def _logp_case(name: str, dtype: torch.dtype, *, seed: int = 0) -> OperatorCase:
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(2, 8, 257, dtype=dtype, generator=generator)
    token_ids = torch.randint(0, logits.size(-1), (2, 8), generator=generator)
    return OperatorCase(
        name=name,
        op_class="logprob",
        dtype=dtype,
        inputs={"logits": logits, "token_ids": token_ids},
        gold_fn=NativeLogpOp().forward_fp32,
    )


def _logp_backward_case(name: str, *, seed: int = 0) -> OperatorCase:
    case = _logp_case(name, torch.float32, seed=seed)
    return OperatorCase(
        name=case.name,
        op_class=case.op_class,
        dtype=case.dtype,
        inputs=case.inputs,
        gold_fn=case.gold_fn,
        grad_input_names=("logits",),
    )


def test_logp_native_candidate_suite_passes():
    report = run_operator_suite(
        "logp",
        candidates=[CandidateSpec(name="native-logp", backend="pytorch", fn=NativeLogpOp())],
        cases=[
            _logp_case("fp32", torch.float32, seed=1),
            _logp_case("bf16", torch.bfloat16, seed=2),
            _logp_case("fp16", torch.float16, seed=3),
        ],
    )

    assert report.passed
    assert report.pass_rate == 1.0
    assert report.candidates[0].passed_outputs == 3
    assert all(case.passed for case in report.candidates[0].cases)


def test_suite_reports_failure_for_bad_candidate():
    def bad_logp(logits, token_ids):
        del token_ids
        return torch.zeros(logits.shape[:-1], dtype=logits.dtype)

    report = run_operator_suite(
        "logp",
        candidates=[CandidateSpec(name="bad-logp", backend="test", fn=bad_logp)],
        cases=[_logp_case("fp32", torch.float32, seed=5)],
    )

    output = report.candidates[0].cases[0].outputs[0]
    assert not report.passed
    assert report.pass_rate == 0.0
    assert output.max_abs_error > 0.0


def test_suite_report_to_dict_contains_error_metrics():
    report = run_operator_suite(
        "logp",
        candidates=[CandidateSpec(name="native-logp", backend="pytorch", fn=NativeLogpOp())],
        cases=[_logp_case("fp32", torch.float32, seed=6)],
    )

    data = report.to_dict()
    output = data["candidates"][0]["cases"][0]["outputs"][0]
    assert data["suite_name"] == "logp"
    assert "max_abs_error" in output
    assert "atol" in output
    assert "passed" in output


def test_candidate_arch_key_uses_tolerance_override():
    def slightly_shifted_logp(logits, token_ids):
        return NativeLogpOp().forward_fp32(logits, token_ids) + 0.02

    contract = {
        "accuracy": {
            "default": {
                "logprob": {
                    "float32": {"atol": 1.0e-5, "rtol": 0.0},
                }
            },
            "arch_overrides": {
                "testarch": {
                    "logprob": {
                        "float32": {"atol": 5.0e-2, "rtol": 0.0},
                    }
                }
            },
        }
    }
    report = run_operator_suite(
        "logp",
        candidates=[
            CandidateSpec(
                name="shifted-logp",
                backend="test",
                fn=slightly_shifted_logp,
                arch_key="testarch",
            )
        ],
        cases=[_logp_case("fp32", torch.float32, seed=7)],
        contract=contract,
    )

    output = report.candidates[0].cases[0].outputs[0]
    assert report.passed
    assert output.atol == 5.0e-2


def test_logp_native_candidate_backward_suite_passes():
    report = run_operator_suite(
        "logp",
        candidates=[CandidateSpec(name="native-logp", backend="pytorch", fn=NativeLogpOp())],
        cases=[_logp_backward_case("fp32", seed=8)],
        check_grad=True,
    )

    assert report.passed
    assert report.candidates[0].passed_outputs == 2
    assert report.candidates[0].cases[0].outputs[1].message == "gradient:logits"


def test_backward_suite_reports_failure_for_bad_gradient():
    def bad_grad_logp(logits, token_ids):
        values = NativeLogpOp().forward_fp32(logits, token_ids)
        return values.detach() + logits.sum(dim=-1) * 0.0

    report = run_operator_suite(
        "logp",
        candidates=[CandidateSpec(name="bad-grad-logp", backend="test", fn=bad_grad_logp)],
        cases=[_logp_backward_case("fp32", seed=9)],
        check_grad=True,
    )

    gradient_output = report.candidates[0].cases[0].outputs[1]
    assert not report.passed
    assert gradient_output.message == "gradient:logits"
    assert gradient_output.max_abs_error > 0.0


def test_random_grad_mode_catches_nonuniform_upstream_gradient_bug():
    # Forward is identity, so only a non-uniform upstream gradient can expose
    # the intentionally wrong backward below.
    class MeanUpstreamIdentity(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.clone()

        @staticmethod
        def backward(ctx, grad_output):
            # Wrong for random upstream gradients, but correct when all values are 1.
            return grad_output.mean().expand_as(grad_output)

    def bad_identity(x):
        return MeanUpstreamIdentity.apply(x)

    case = OperatorCase(
        name="identity",
        op_class="elementwise",
        dtype=torch.float32,
        inputs={"x": torch.randn(8, dtype=torch.float32)},
        gold_fn=lambda x: x,
        grad_input_names=("x",),
    )

    ones_report = run_operator_suite(
        "identity",
        candidates=[CandidateSpec(name="bad-identity", backend="test", fn=bad_identity)],
        cases=[case],
        check_grad=True,
        grad_mode="ones",
    )
    # ones passes by design; random must fail and prove the stricter path works.
    random_report = run_operator_suite(
        "identity",
        candidates=[CandidateSpec(name="bad-identity", backend="test", fn=bad_identity)],
        cases=[case],
        check_grad=True,
        grad_mode="random",
        grad_seed=7,
    )

    assert ones_report.passed
    gradient_output = random_report.candidates[0].cases[0].outputs[1]
    assert not random_report.passed
    assert gradient_output.message == "gradient:x"
    assert gradient_output.max_abs_error > 0.0
