# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch

from rl_engine.kernels.gtest.tolerance import load_contract


@dataclass(frozen=True)
class OperatorCase:
    """One deterministic test object for an operator candidate."""

    name: str
    op_class: str
    dtype: torch.dtype
    inputs: Mapping[str, Any]
    gold_fn: Callable[..., Any]
    grad_input_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateSpec:
    """One implementation to validate against the gold path."""

    name: str
    fn: Callable[..., Any] | Any
    backend: str = "unknown"
    arch_key: str | None = None


@dataclass(frozen=True)
class OutputCheck:
    """Per-output comparison result."""

    output_index: int
    shape: tuple[int, ...]
    candidate_dtype: str
    gold_dtype: str
    atol: float
    rtol: float
    max_abs_error: float
    mean_abs_error: float
    max_rel_error: float
    passed: bool
    message: str = ""


@dataclass(frozen=True)
class CaseCheck:
    """Per-case result for one candidate."""

    case_name: str
    dtype: str
    op_class: str
    passed: bool
    outputs: list[OutputCheck]


@dataclass(frozen=True)
class CandidateReport:
    """Aggregate report for one candidate implementation."""

    candidate_name: str
    backend: str
    total_outputs: int
    passed_outputs: int
    pass_rate: float
    passed: bool
    cases: list[CaseCheck]


@dataclass(frozen=True)
class OperatorCheckReport:
    """Suite-level report across candidates."""

    suite_name: str
    total_candidates: int
    passed_candidates: int
    pass_rate: float
    passed: bool
    candidates: list[CandidateReport]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_operator_suite(
    suite_name: str,
    *,
    candidates: Sequence[CandidateSpec],
    cases: Sequence[OperatorCase],
    contract: Mapping[str, Any] | None = None,
    check_grad: bool = False,
    grad_mode: str = "random",
    grad_seed: int = 123,
) -> OperatorCheckReport:
    """Run candidates against gold outputs and return a structured report."""

    loaded_contract = dict(contract or load_contract())
    # run all test ops
    # cases : test object
    # camdidate : test instance
    # loaded_contract : tolerance table
    candidate_reports = [
        _run_candidate(
            candidate,
            cases,
            loaded_contract,
            check_grad=check_grad,
            grad_mode=grad_mode,
            grad_seed=grad_seed,
        )
        for candidate in candidates
    ]
    passed_candidates = sum(1 for report in candidate_reports if report.passed)
    total_candidates = len(candidate_reports)
    pass_rate = float(passed_candidates / total_candidates) if total_candidates else 0.0
    return OperatorCheckReport(
        suite_name=suite_name,
        total_candidates=total_candidates,
        passed_candidates=passed_candidates,
        pass_rate=pass_rate,
        passed=passed_candidates == total_candidates,
        candidates=candidate_reports,
    )


def _run_candidate(
    candidate: CandidateSpec,
    cases: Sequence[OperatorCase],
    contract: Mapping[str, Any],
    *,
    check_grad: bool,
    grad_mode: str,
    grad_seed: int,
) -> CandidateReport:
    if check_grad:
        case_checks = [
            _run_case_backward(
                candidate,
                case,
                contract,
                grad_mode=grad_mode,
                grad_seed=grad_seed,
            )
            for case in cases
        ]
    else:
        case_checks = [_run_case(candidate, case, contract) for case in cases]
    total_outputs = sum(len(case.outputs) for case in case_checks)
    passed_outputs = sum(1 for case in case_checks for output in case.outputs if output.passed)
    pass_rate = float(passed_outputs / total_outputs) if total_outputs else 0.0
    return CandidateReport(
        candidate_name=candidate.name,
        backend=candidate.backend,
        total_outputs=total_outputs,
        passed_outputs=passed_outputs,
        pass_rate=pass_rate,
        passed=passed_outputs == total_outputs,
        cases=case_checks,
    )


def _run_case(
    candidate: CandidateSpec,
    case: OperatorCase,
    contract: Mapping[str, Any],
) -> CaseCheck:
    candidate_outputs = _flatten_tensors(_call_candidate(candidate.fn, case.inputs))
    gold_outputs = _flatten_tensors(case.gold_fn(**case.inputs))
    return _compare_case_outputs(candidate, case, contract, candidate_outputs, gold_outputs)


def _run_case_backward(
    candidate: CandidateSpec,
    case: OperatorCase,
    contract: Mapping[str, Any],
    *,
    grad_mode: str,
    grad_seed: int,
) -> CaseCheck:
    if not case.grad_input_names:
        raise ValueError(f"case {case.name!r} does not declare gradient inputs")

    candidate_inputs = _clone_inputs_for_backward(case.inputs, case.grad_input_names)
    gold_inputs = _clone_inputs_for_backward(case.inputs, case.grad_input_names)
    candidate_outputs = _flatten_tensors(_call_candidate(candidate.fn, candidate_inputs))
    gold_outputs = _flatten_tensors(case.gold_fn(**gold_inputs))
    # Candidate and gold must use the same upstream gradients; otherwise we
    # would compare different vector-Jacobian products.
    # grad_mode="ones" is the old output.sum().backward() smoke path.
    # grad_mode="random" is closer to training, where dL/doutput is non-uniform.
    grad_outputs = _make_grad_outputs(candidate_outputs, grad_mode=grad_mode, seed=grad_seed)
    candidate_grads = _backward_grads(
        candidate_outputs,
        candidate_inputs,
        case.grad_input_names,
        grad_outputs=grad_outputs,
    )
    gold_grads = _backward_grads(
        gold_outputs,
        gold_inputs,
        case.grad_input_names,
        grad_outputs=_match_grad_outputs(grad_outputs, gold_outputs),
    )
    output_checks = _compare_case_outputs(
        candidate,
        case,
        contract,
        candidate_outputs,
        gold_outputs,
    ).outputs
    # Reuse the same tolerance class for gradients as for values. This is a
    # first conservative default; operator-specific gradient tolerances can be
    # split out later if a real backend shows different numerical behavior.
    atol, rtol = _resolve_tolerance(
        contract,
        op_class=case.op_class,
        dtype=case.dtype,
        arch_key=candidate.arch_key,
    )
    grad_checks = [
        _compare_output(
            candidate_grad,
            gold_grad,
            output_index=len(output_checks) + index,
            atol=atol,
            rtol=rtol,
            message=f"gradient:{name}",
        )
        for index, (name, candidate_grad, gold_grad) in enumerate(
            zip(case.grad_input_names, candidate_grads, gold_grads, strict=True)
        )
    ]
    checks = [*output_checks, *grad_checks]
    return CaseCheck(
        case_name=case.name,
        dtype=str(case.dtype),
        op_class=case.op_class,
        passed=all(output.passed for output in checks),
        outputs=checks,
    )


def _compare_case_outputs(
    candidate: CandidateSpec,
    case: OperatorCase,
    contract: Mapping[str, Any],
    candidate_outputs: list[torch.Tensor],
    gold_outputs: list[torch.Tensor],
) -> CaseCheck:
    if len(candidate_outputs) != len(gold_outputs):
        raise ValueError(
            f"candidate {candidate.name!r} returned {len(candidate_outputs)} outputs, "
            f"gold returned {len(gold_outputs)}"
        )
    atol, rtol = _resolve_tolerance(
        contract,
        op_class=case.op_class,
        dtype=case.dtype,
        arch_key=candidate.arch_key,
    )
    output_checks = [
        _compare_output(
            candidate_output,
            gold_output,
            output_index=index,
            atol=atol,
            rtol=rtol,
        )
        for index, (candidate_output, gold_output) in enumerate(
            zip(candidate_outputs, gold_outputs, strict=True)
        )
    ]
    return CaseCheck(
        case_name=case.name,
        dtype=str(case.dtype),
        op_class=case.op_class,
        passed=all(output.passed for output in output_checks),
        outputs=output_checks,
    )


# compatibility function or forward
def _call_candidate(candidate: Callable[..., Any] | Any, inputs: Mapping[str, Any]) -> Any:
    if hasattr(candidate, "forward") and callable(candidate.forward):
        return candidate.forward(**inputs)
    return candidate(**inputs)


def _clone_inputs_for_backward(
    inputs: Mapping[str, Any],
    grad_input_names: tuple[str, ...],
) -> dict[str, Any]:
    grad_names = set(grad_input_names)
    cloned: dict[str, Any] = {}
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor):
            tensor = value.detach().clone()
            if name in grad_names:
                if not tensor.is_floating_point():
                    raise TypeError(f"gradient input {name!r} must be floating point")
                tensor.requires_grad_(True)
            cloned[name] = tensor
        else:
            cloned[name] = value
    missing = grad_names.difference(cloned)
    if missing:
        raise ValueError(f"missing gradient inputs: {', '.join(sorted(missing))}")
    return cloned


def _backward_grads(
    outputs: list[torch.Tensor],
    inputs: Mapping[str, Any],
    grad_input_names: tuple[str, ...],
    *,
    grad_outputs: list[torch.Tensor],
) -> list[torch.Tensor]:
    if len(outputs) != len(grad_outputs):
        raise ValueError(f"got {len(grad_outputs)} upstream gradients for {len(outputs)} outputs")
    # `ones` makes this equivalent to output.sum().backward(); `random` tests a
    # stricter vector-Jacobian product.
    loss_terms = [
        (output.float() * grad_output.to(device=output.device).float()).sum()
        for output, grad_output in zip(outputs, grad_outputs, strict=True)
    ]
    if not loss_terms:
        raise ValueError("backward checks require at least one output")
    loss = loss_terms[0]
    for term in loss_terms[1:]:
        loss = loss + term
    loss.backward()
    grads: list[torch.Tensor] = []
    for name in grad_input_names:
        grad = inputs[name].grad
        if grad is None:
            raise ValueError(f"gradient for input {name!r} is None")
        grads.append(grad)
    return grads


def _make_grad_outputs(
    outputs: list[torch.Tensor],
    *,
    grad_mode: str,
    seed: int,
) -> list[torch.Tensor]:
    if grad_mode == "ones":
        # All-one upstream gradients make the scalar loss equal output.sum().
        return [torch.ones_like(output, dtype=torch.float32) for output in outputs]
    if grad_mode != "random":
        raise ValueError(f"unsupported grad_mode: {grad_mode}")

    grad_outputs: list[torch.Tensor] = []
    generators: dict[torch.device, torch.Generator] = {}
    for output in outputs:
        if output.device not in generators:
            # Generators are device-local; a CUDA generator cannot draw CPU tensors.
            generator = torch.Generator(device=output.device)
            generator.manual_seed(seed)
            generators[output.device] = generator
        # Random upstream gradients test a non-uniform dL/doutput. The same
        # tensors are later reused for gold so the comparison stays fair.
        grad_outputs.append(
            torch.randn(
                output.shape,
                generator=generators[output.device],
                device=output.device,
                dtype=torch.float32,
            )
        )
    return grad_outputs


def _match_grad_outputs(
    grad_outputs: list[torch.Tensor],
    outputs: list[torch.Tensor],
) -> list[torch.Tensor]:
    # Reuse upstream values for gold; only move device when needed.
    return [
        grad_output.to(device=output.device)
        for grad_output, output in zip(grad_outputs, outputs, strict=True)
    ]


def _flatten_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        outputs: list[torch.Tensor] = []
        for item in value:
            outputs.extend(_flatten_tensors(item))
        return outputs
    raise TypeError(f"operator output must be Tensor or sequence, got {type(value)!r}")


def _resolve_tolerance(
    contract: Mapping[str, Any],
    *,
    op_class: str,
    dtype: torch.dtype,
    arch_key: str | None = None,
) -> tuple[float, float]:
    dtype_name = _dtype_name(dtype)
    if arch_key is not None:
        arch_values = (
            contract["accuracy"]
            .get("arch_overrides", {})
            .get(arch_key, {})
            .get(op_class, {})
            .get(dtype_name)
        )
        if arch_values is not None:
            return float(arch_values["atol"]), float(arch_values.get("rtol", 0.0))

    values = contract["accuracy"]["default"][op_class][dtype_name]
    return float(values["atol"]), float(values.get("rtol", 0.0))


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.float32:
        return "float32"
    if dtype is torch.bfloat16:
        return "bfloat16"
    if dtype is torch.float16:
        return "float16"
    raise ValueError(f"unsupported dtype: {dtype}")


def _compare_output(
    candidate: torch.Tensor,
    gold: torch.Tensor,
    *,
    output_index: int,
    atol: float,
    rtol: float,
    message: str = "",
) -> OutputCheck:
    if candidate.shape != gold.shape:
        return OutputCheck(
            output_index=output_index,
            shape=tuple(candidate.shape),
            candidate_dtype=str(candidate.dtype),
            gold_dtype=str(gold.dtype),
            atol=atol,
            rtol=rtol,
            max_abs_error=float("inf"),
            mean_abs_error=float("inf"),
            max_rel_error=float("inf"),
            passed=False,
            message=f"shape mismatch: candidate={tuple(candidate.shape)} gold={tuple(gold.shape)}",
        )

    candidate_fp32 = candidate.float()
    gold_fp32 = gold.float()
    abs_error = (candidate_fp32 - gold_fp32).abs()
    if abs_error.numel() == 0:
        max_abs_error = 0.0
        mean_abs_error = 0.0
        max_rel_error = 0.0
    else:
        max_abs_error = float(abs_error.max().item())
        mean_abs_error = float(abs_error.mean().item())
        rel_error = abs_error / gold_fp32.abs().clamp_min(1e-12)
        max_rel_error = float(rel_error.max().item())

    return OutputCheck(
        output_index=output_index,
        shape=tuple(candidate.shape),
        candidate_dtype=str(candidate.dtype),
        gold_dtype=str(gold.dtype),
        atol=atol,
        rtol=rtol,
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        max_rel_error=max_rel_error,
        passed=bool(torch.allclose(candidate_fp32, gold_fp32, atol=atol, rtol=rtol)),
        message=message,
    )


__all__ = [
    "CandidateReport",
    "CandidateSpec",
    "CaseCheck",
    "OperatorCase",
    "OperatorCheckReport",
    "OutputCheck",
    "run_operator_suite",
]
