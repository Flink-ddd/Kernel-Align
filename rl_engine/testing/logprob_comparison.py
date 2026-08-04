# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Single-GPU WS2 selected-logprob cross-implementation harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch


class LogprobBackendUnavailable(RuntimeError):
    """Raised when an explicitly requested comparison backend cannot run exactly."""


@dataclass(frozen=True)
class LogprobComparisonInputs:
    """Logical TP=1 inputs shared by every comparison path."""

    logits: torch.Tensor
    target_ids: torch.Tensor
    active_token_mask: torch.Tensor | None = None
    ignore_index: int = -100


@dataclass(frozen=True)
class LogprobPathResult:
    """Direct selected-logprob and vocab-LSE outputs from one backend."""

    name: str
    logp: torch.Tensor
    lse: torch.Tensor
    provenance: dict[str, Any]


@dataclass(frozen=True)
class LogprobCandidate:
    """One exact backend materialization used by the harness."""

    name: str
    requested_backend: str
    actual_backend: str
    fn: Callable[[torch.Tensor, torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]]
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DriftStats:
    """Absolute drift statistics over a declared comparison population."""

    max_abs: float
    mean_abs: float
    p95_abs: float
    p99_abs: float
    active_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_abs": self.max_abs,
            "mean_abs": self.mean_abs,
            "p95_abs": self.p95_abs,
            "p99_abs": self.p99_abs,
            "active_count": self.active_count,
        }


@dataclass(frozen=True)
class LogprobPathDrift:
    """Candidate-vs-reference LSE and active-token dlogp drift."""

    candidate_name: str
    lse: DriftStats
    dlogp: DriftStats
    bitwise_logp: bool
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_name": self.candidate_name,
            "lse": self.lse.to_dict(),
            "dlogp": self.dlogp.to_dict(),
            "bitwise_logp": self.bitwise_logp,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class LogprobComparisonReport:
    """Structured single-GPU report consumed by later WS2 integration."""

    reference_name: str
    drifts: tuple[LogprobPathDrift, ...]
    input_provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_name": self.reference_name,
            "drifts": [drift.to_dict() for drift in self.drifts],
            "input_provenance": self.input_provenance,
        }


def make_logprob_candidate(backend: str) -> LogprobCandidate:
    """Materialize an exact built-in backend without registry fallback."""

    normalized = backend.strip().lower().replace("_", "-")
    if normalized in {"pytorch", "native"}:
        from rl_engine.kernels.ops.pytorch.loss.batch_invariant_logp import (
            NativeBatchInvariantLogpOp,
        )

        op = NativeBatchInvariantLogpOp()
        actual = "pytorch"
    elif normalized == "triton":
        try:
            from rl_engine.kernels.ops.triton.loss.batch_invariant_logp import (
                TritonBatchInvariantLogpOp,
            )

            op = TritonBatchInvariantLogpOp()
        except Exception as exc:
            raise LogprobBackendUnavailable(f"triton backend is unavailable: {exc}") from exc
        actual = "triton"
    elif normalized in {"cuda-sm90", "sm90"}:
        try:
            from rl_engine.kernels.ops.cuda.loss.batch_invariant_logp import (
                BatchInvariantLogpSM90Op,
            )

            op = BatchInvariantLogpSM90Op()
        except Exception as exc:
            raise LogprobBackendUnavailable(f"cuda-sm90 backend is unavailable: {exc}") from exc
        actual = "cuda-sm90"
    else:
        raise ValueError(
            f"unsupported logprob comparison backend {backend!r}; "
            "expected pytorch, triton, or cuda-sm90"
        )

    diagnostic = getattr(op, "forward_with_lse", None)
    if not callable(diagnostic):
        raise LogprobBackendUnavailable(
            f"backend {normalized!r} does not expose the required direct LSE diagnostic"
        )

    def run(
        logits: torch.Tensor, target_ids: torch.Tensor, ignore_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return diagnostic(logits, target_ids, ignore_index=ignore_index, validate=True)
        except (RuntimeError, NotImplementedError, OSError) as exc:
            raise LogprobBackendUnavailable(
                f"exact backend {normalized!r} cannot execute this input: {exc}"
            ) from exc

    return LogprobCandidate(
        name=f"{actual}-batch-invariant-logp",
        requested_backend=actual,
        actual_backend=actual,
        fn=run,
        provenance={
            "requested_alias": normalized,
            "implementation": f"{type(op).__module__}.{type(op).__qualname__}",
        },
    )


def compare_single_gpu_logprob(
    inputs: LogprobComparisonInputs,
    *,
    candidates: Sequence[str | LogprobCandidate] = ("pytorch",),
) -> LogprobComparisonReport:
    """Compare exact TP=1 implementations against the WS1 deterministic path."""

    active_mask, effective_targets = _validate_inputs(inputs)
    reference = _run_ws1_reference(inputs.logits, effective_targets, inputs.ignore_index)

    drifts = tuple(
        _compare_path(
            _run_candidate(
                (
                    candidate
                    if isinstance(candidate, LogprobCandidate)
                    else make_logprob_candidate(candidate)
                ),
                inputs.logits,
                effective_targets,
                inputs.ignore_index,
            ),
            reference,
            active_mask,
        )
        for candidate in candidates
    )
    return LogprobComparisonReport(
        reference_name=reference.name,
        drifts=drifts,
        input_provenance={
            "device": str(inputs.logits.device),
            "input_dtype": str(inputs.logits.dtype),
            "output_dtype": str(reference.logp.dtype),
            "shape": list(inputs.logits.shape),
            "ignore_index": inputs.ignore_index,
            "active_token_count": int(active_mask.sum().item()),
            "tp_world": 1,
            "communication": "none",
        },
    )


def _run_ws1_reference(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    ignore_index: int,
) -> LogprobPathResult:
    """Run the existing deterministic logp path and its direct-LSE diagnostic."""
    from rl_engine.kernels.ops.pytorch.loss.batch_invariant_logp import (
        NativeBatchInvariantLogpOp,
    )

    op = NativeBatchInvariantLogpOp()
    logp = op(logits, target_ids, ignore_index=ignore_index, validate=True)
    _, lse = op.forward_with_lse(
        logits, target_ids, ignore_index=ignore_index, validate=True
    )
    return LogprobPathResult(
        name="pytorch-batch-invariant-logp",
        logp=logp.detach(),
        lse=lse.detach(),
        provenance={
            "requested_backend": "pytorch",
            "actual_backend": "pytorch",
            "tp_world": 1,
            "communication": "none",
            "logp_source": "production",
            "lse_source": "direct",
        },
    )


def _run_candidate(
    candidate: LogprobCandidate,
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    ignore_index: int,
) -> LogprobPathResult:
    if candidate.requested_backend != candidate.actual_backend:
        raise LogprobBackendUnavailable(
            f"requested backend {candidate.requested_backend!r} materialized as "
            f"{candidate.actual_backend!r}; silent fallback is forbidden"
        )
    logp, lse = candidate.fn(logits, target_ids, ignore_index)
    expected_shape = logits.shape[:-1]
    for name, value in (("logp", logp), ("lse", lse)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"candidate {candidate.name!r} {name} must be a tensor")
        if value.shape != expected_shape:
            raise ValueError(
                f"candidate {candidate.name!r} {name} shape {tuple(value.shape)} "
                f"does not match {tuple(expected_shape)}"
            )
        if value.dtype != torch.float32:
            raise ValueError(f"candidate {candidate.name!r} {name} must be FP32")
    return LogprobPathResult(
        name=candidate.name,
        logp=logp.detach(),
        lse=lse.detach(),
        provenance={
            "requested_backend": candidate.requested_backend,
            "actual_backend": candidate.actual_backend,
            "tp_world": 1,
            "communication": "none",
            "lse_source": "direct",
            **candidate.provenance,
        },
    )


def _compare_path(
    candidate: LogprobPathResult,
    reference: LogprobPathResult,
    active_mask: torch.Tensor,
) -> LogprobPathDrift:
    return LogprobPathDrift(
        candidate_name=candidate.name,
        lse=_drift_stats(candidate.lse, reference.lse),
        dlogp=_drift_stats(candidate.logp, reference.logp, mask=active_mask),
        bitwise_logp=torch.equal(candidate.logp, reference.logp),
        provenance=candidate.provenance,
    )


def _drift_stats(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> DriftStats:
    if candidate.shape != reference.shape:
        raise ValueError(
            f"candidate shape {tuple(candidate.shape)} must match reference shape "
            f"{tuple(reference.shape)}"
        )
    diff = (candidate.float() - reference.float()).abs()
    values = diff.reshape(-1) if mask is None else diff[mask.to(device=diff.device)]
    count = int(values.numel())
    if count == 0:
        return DriftStats(0.0, 0.0, 0.0, 0.0, 0)
    return DriftStats(
        max_abs=float(values.max().item()),
        mean_abs=float(values.mean().item()),
        p95_abs=float(torch.quantile(values, 0.95).item()),
        p99_abs=float(torch.quantile(values, 0.99).item()),
        active_count=count,
    )


def _validate_inputs(
    inputs: LogprobComparisonInputs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if inputs.logits.dim() < 2:
        raise ValueError("logits must be at least 2-D [*lead, vocab]")
    if inputs.logits.shape[:-1] != inputs.target_ids.shape:
        raise ValueError("target_ids shape must match logits leading shape")
    if not inputs.logits.is_floating_point():
        raise ValueError("logits must be floating point")

    if inputs.active_token_mask is None:
        active = inputs.target_ids != inputs.ignore_index
    else:
        if inputs.active_token_mask.shape != inputs.target_ids.shape:
            raise ValueError("active_token_mask shape must match target_ids")
        if inputs.active_token_mask.dtype != torch.bool:
            raise ValueError("active_token_mask must be bool")
        active = inputs.active_token_mask.to(device=inputs.target_ids.device)
        if bool(((inputs.target_ids == inputs.ignore_index) & active).any().item()):
            raise ValueError("active target_ids cannot equal ignore_index")

    effective = inputs.target_ids.to(device=inputs.logits.device, dtype=torch.long).clone()
    active = active.to(device=inputs.logits.device, dtype=torch.bool)
    effective.masked_fill_(~active, inputs.ignore_index)
    valid = effective[active]
    vocab_size = inputs.logits.size(-1)
    if valid.numel() and ((valid < 0).any() or (valid >= vocab_size).any()):
        raise ValueError(f"active target_ids must be in [0, {vocab_size})")
    return active, effective


__all__ = [
    "DriftStats",
    "LogprobBackendUnavailable",
    "LogprobCandidate",
    "LogprobComparisonInputs",
    "LogprobComparisonReport",
    "LogprobPathDrift",
    "LogprobPathResult",
    "compare_single_gpu_logprob",
    "make_logprob_candidate",
]
