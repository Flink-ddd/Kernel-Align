# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Single-card RL-Kernel operator comparison and drift attribution helpers.

This module is intentionally an audit/admission surface. It does not import
framework code or dispatch production operator calls; callers provide the train
and inference implementations they want to compare.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import torch
import torch.nn.functional as F

OperatorComparisonStatus = Literal["passed", "failed", "unsupported"]
OperatorCategory = Literal["forward", "logp", "loss"]

RLK_OP_ATTENTION = "attention"
RLK_OP_LM_HEAD = "lm_head"
RLK_OP_LOGP = "logp"
RLK_OP_MATMUL_PROJECTION = "matmul_projection"
RLK_OP_RMSNORM = "rmsnorm"
RLK_OP_ROPE = "rope"
RLK_OP_SWIGLU = "swiglu"
RLK_OP_EMBEDDING = "embedding"
RLK_OP_RATIO_KL = "ratio_kl"
RLK_OP_PPO_FRAGMENT = "ppo_fragment"
RLK_OP_GRPO_FRAGMENT = "grpo_fragment"
RLK_OP_DPO_FRAGMENT = "dpo_fragment"

PHASE4_TARGET_OPERATORS = (
    RLK_OP_ATTENTION,
    RLK_OP_LM_HEAD,
    RLK_OP_LOGP,
    RLK_OP_MATMUL_PROJECTION,
    RLK_OP_RMSNORM,
    RLK_OP_ROPE,
    RLK_OP_SWIGLU,
    RLK_OP_EMBEDDING,
    RLK_OP_RATIO_KL,
    RLK_OP_PPO_FRAGMENT,
    RLK_OP_GRPO_FRAGMENT,
    RLK_OP_DPO_FRAGMENT,
)


@dataclass(frozen=True)
class OperatorTolerance:
    """Tolerance used for cross-role parity, separate from bitwise claims."""

    atol: float = 1e-6
    rtol: float = 1e-6

    def __post_init__(self) -> None:
        if self.atol < 0:
            raise ValueError(f"atol must be non-negative, got {self.atol!r}.")
        if self.rtol < 0:
            raise ValueError(f"rtol must be non-negative, got {self.rtol!r}.")

    def to_dict(self) -> dict[str, float]:
        return {"atol": float(self.atol), "rtol": float(self.rtol)}


@dataclass(frozen=True)
class OperatorComparisonSpec:
    op_name: str
    category: OperatorCategory
    boundary: str
    tolerance: OperatorTolerance = field(default_factory=OperatorTolerance)
    supports_grad: bool = False
    compares_logp: bool = False
    reference_impl: Callable[..., Any] | None = None
    unsupported_reason: str | None = None
    required_inputs: tuple[str, ...] = ()
    batch_invariance_axes: tuple[str, ...] = (
        "single_sample_vs_mixed_batch",
        "padding_packing_layout",
        "row_position",
        "active_mask_density",
    )

    def __post_init__(self) -> None:
        if bool(self.reference_impl is None) == bool(self.unsupported_reason is None):
            raise ValueError(
                f"Operator spec {self.op_name!r} must have exactly one of "
                "reference_impl or unsupported_reason."
            )
        object.__setattr__(self, "required_inputs", tuple(self.required_inputs))
        object.__setattr__(self, "batch_invariance_axes", tuple(self.batch_invariance_axes))

    @property
    def supported(self) -> bool:
        return self.reference_impl is not None

    def unsupported_result(
        self, *, metadata: Mapping[str, Any] | None = None
    ) -> OperatorComparisonResult:
        return OperatorComparisonResult(
            op_name=self.op_name,
            status="unsupported",
            unsupported_reason=self.unsupported_reason,
            metrics={},
            compared_tensors=(),
            tolerance=self.tolerance.to_dict(),
            metadata=_immutable_mapping(metadata),
        )


@dataclass(frozen=True)
class OperatorComparisonResult:
    op_name: str
    status: OperatorComparisonStatus
    metrics: Mapping[str, float] = field(default_factory=dict)
    compared_tensors: tuple[str, ...] = ()
    tolerance: Mapping[str, float] = field(default_factory=dict)
    unsupported_reason: str | None = None
    failure_reason: str | None = None
    first_drift_operator: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", _immutable_mapping(self.metrics))
        object.__setattr__(self, "compared_tensors", tuple(self.compared_tensors))
        object.__setattr__(self, "tolerance", _immutable_mapping(self.tolerance))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_name": self.op_name,
            "status": self.status,
            "metrics": dict(self.metrics),
            "compared_tensors": self.compared_tensors,
            "tolerance": dict(self.tolerance),
            "unsupported_reason": self.unsupported_reason,
            "failure_reason": self.failure_reason,
            "first_drift_operator": self.first_drift_operator,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class OperatorPair:
    """Train/infer role pairing supplied by an admission or audit test."""

    op_name: str
    train: Callable[..., Any]
    infer: Callable[..., Any]
    train_kwargs: Mapping[str, Any] = field(default_factory=dict)
    infer_kwargs: Mapping[str, Any] = field(default_factory=dict)
    tolerance: OperatorTolerance | None = None
    active_token_mask: torch.Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_kwargs", _immutable_mapping(self.train_kwargs))
        object.__setattr__(self, "infer_kwargs", _immutable_mapping(self.infer_kwargs))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True)
class ForwardChainStep:
    op_name: str
    train: Callable[[Any], Any]
    infer: Callable[[Any], Any]
    tolerance: OperatorTolerance | None = None
    active_token_mask: torch.Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True)
class ForwardChainComparisonResult:
    status: OperatorComparisonStatus
    steps: tuple[OperatorComparisonResult, ...]
    first_drift_operator: str | None
    cumulative_metrics: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "cumulative_metrics", _immutable_mapping(self.cumulative_metrics))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))

    @property
    def passed(self) -> bool:
        return self.status == "passed"


@dataclass(frozen=True)
class BatchInvarianceCase:
    op_name: str
    case: str
    varied_axes: tuple[str, ...]
    expected: str
    sample_position: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "varied_axes", tuple(self.varied_axes))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_name": self.op_name,
            "case": self.case,
            "varied_axes": self.varied_axes,
            "sample_position": self.sample_position,
            "expected": self.expected,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class StrictBackendAdmissionReport:
    backend_id: str
    operator: str
    strict_fast_eligible: bool
    reasons: tuple[str, ...]
    comparison_count: int
    failed_comparisons: tuple[str, ...]
    unsupported_comparisons: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "failed_comparisons", tuple(self.failed_comparisons))
        object.__setattr__(self, "unsupported_comparisons", tuple(self.unsupported_comparisons))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


def reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Reference scaled dot-product attention for single-card operator checks."""

    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head dimensions must match.")
    scale = (1.0 / math.sqrt(query.shape[-1])) if scale is None else float(scale)
    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
    if causal:
        query_len = scores.shape[-2]
        key_len = scores.shape[-1]
        causal_mask = torch.ones(
            (query_len, key_len), device=scores.device, dtype=torch.bool
        ).tril()
        scores = scores.masked_fill(~causal_mask, float("-inf"))
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores + mask.to(device=scores.device, dtype=scores.dtype)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, value.float()).to(dtype=query.dtype)


def reference_matmul_projection(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return F.linear(hidden, weight, bias)


def reference_lm_head(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return reference_matmul_projection(hidden, weight, bias)


def reference_selected_logprobs(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    temperature: float = 1.0,
    log_prob_keep_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature!r}.")
    scaled_logits = logits.float() / float(temperature)
    if log_prob_keep_mask is not None:
        scaled_logits = scaled_logits.masked_fill(
            ~log_prob_keep_mask.to(dtype=torch.bool), float("-inf")
        )
    selected = torch.gather(
        torch.log_softmax(scaled_logits, dim=-1),
        -1,
        target_ids.long().unsqueeze(-1),
    ).squeeze(-1)
    if mask is not None:
        selected = selected.masked_fill(~mask.to(dtype=torch.bool), 0.0)
    return selected


def reference_linear_logp(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target_ids: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return reference_selected_logprobs(reference_lm_head(hidden, lm_head_weight, bias), target_ids)


def reference_rmsnorm(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = hidden.float() * torch.rsqrt(variance + eps)
    return (normalized * weight.float()).to(dtype=hidden.dtype)


def reference_rope(
    hidden: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    if hidden.shape[-1] % 2 != 0:
        raise ValueError("RoPE reference expects an even last dimension.")
    even = hidden[..., 0::2].float()
    odd = hidden[..., 1::2].float()
    cos = cos.to(device=hidden.device, dtype=torch.float32)
    sin = sin.to(device=hidden.device, dtype=torch.float32)
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    out = torch.empty_like(hidden.float())
    out[..., 0::2] = rotated_even
    out[..., 1::2] = rotated_odd
    return out.to(dtype=hidden.dtype)


def reference_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate.float()).mul(up.float()).to(dtype=gate.dtype)


def reference_embedding(input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.embedding(input_ids.long(), weight)


def reference_ratio_kl(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    *,
    loss_mask: torch.Tensor | None = None,
    eps_clip: float = 0.2,
) -> dict[str, torch.Tensor]:
    dlogp = log_probs.float() - old_log_probs.float()
    ppo_kl = -dlogp
    ratio = (-ppo_kl).exp()
    approx_kl = ratio - 1.0 + ppo_kl
    clip_mask = ((ratio - 1.0).abs() > eps_clip).to(dtype=ratio.dtype)
    if loss_mask is not None:
        active = loss_mask.to(dtype=torch.bool)
        dlogp = dlogp.masked_fill(~active, 0.0)
        ppo_kl = ppo_kl.masked_fill(~active, 0.0)
        ratio = ratio.masked_fill(~active, 0.0)
        approx_kl = approx_kl.masked_fill(~active, 0.0)
        clip_mask = clip_mask.masked_fill(~active, 0.0)
    return {
        "dlogp": dlogp,
        "ppo_kl": ppo_kl,
        "ratio": ratio,
        "approx_kl": approx_kl,
        "clip_mask": clip_mask,
    }


def reference_ppo_fragment(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    *,
    eps_clip: float = 0.2,
    eps_clip_high: float | None = None,
    eps_clip_c: float | None = None,
) -> dict[str, torch.Tensor]:
    eps_clip_high = eps_clip if eps_clip_high is None else eps_clip_high
    ratio = (-ppo_kl.float()).exp()
    pg_losses1 = -ratio * advantages.float()
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages.float()
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    clipfrac = torch.gt(pg_losses2, pg_losses1).float()
    if eps_clip_c is not None:
        if eps_clip_c <= 1.0:
            raise ValueError(f"eps_clip_c must be greater than 1.0, got {eps_clip_c!r}.")
        pg_losses3 = -eps_clip_c * advantages.float()
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    else:
        pg_losses = clip_pg_losses1
    return {
        "pg_loss": pg_losses.to(dtype=ppo_kl.dtype),
        "clipfrac": clipfrac.to(dtype=ppo_kl.dtype),
        "ratio": ratio.to(dtype=ppo_kl.dtype),
    }


OPERATOR_COMPARISON_SPECS: Mapping[str, OperatorComparisonSpec] = MappingProxyType(
    {
        RLK_OP_ATTENTION: OperatorComparisonSpec(
            op_name=RLK_OP_ATTENTION,
            category="forward",
            boundary="transformer.self_attention",
            supports_grad=True,
            reference_impl=reference_attention,
            required_inputs=("query", "key", "value"),
        ),
        RLK_OP_LM_HEAD: OperatorComparisonSpec(
            op_name=RLK_OP_LM_HEAD,
            category="forward",
            boundary="decoder.output_layer",
            supports_grad=True,
            reference_impl=reference_lm_head,
            required_inputs=("hidden", "weight"),
        ),
        RLK_OP_LOGP: OperatorComparisonSpec(
            op_name=RLK_OP_LOGP,
            category="logp",
            boundary="selected_logprobs",
            supports_grad=True,
            compares_logp=True,
            reference_impl=reference_selected_logprobs,
            required_inputs=("logits", "target_ids"),
        ),
        RLK_OP_MATMUL_PROJECTION: OperatorComparisonSpec(
            op_name=RLK_OP_MATMUL_PROJECTION,
            category="forward",
            boundary="linear_projection",
            supports_grad=True,
            reference_impl=reference_matmul_projection,
            required_inputs=("hidden", "weight"),
        ),
        RLK_OP_RMSNORM: OperatorComparisonSpec(
            op_name=RLK_OP_RMSNORM,
            category="forward",
            boundary="normalization.rmsnorm",
            supports_grad=True,
            reference_impl=reference_rmsnorm,
            required_inputs=("hidden", "weight"),
        ),
        RLK_OP_ROPE: OperatorComparisonSpec(
            op_name=RLK_OP_ROPE,
            category="forward",
            boundary="position_encoding.rope",
            supports_grad=True,
            reference_impl=reference_rope,
            required_inputs=("hidden", "cos", "sin"),
        ),
        RLK_OP_SWIGLU: OperatorComparisonSpec(
            op_name=RLK_OP_SWIGLU,
            category="forward",
            boundary="mlp.swiglu",
            supports_grad=True,
            reference_impl=reference_swiglu,
            required_inputs=("gate", "up"),
        ),
        RLK_OP_EMBEDDING: OperatorComparisonSpec(
            op_name=RLK_OP_EMBEDDING,
            category="forward",
            boundary="embedding",
            supports_grad=True,
            reference_impl=reference_embedding,
            required_inputs=("input_ids", "weight"),
        ),
        RLK_OP_RATIO_KL: OperatorComparisonSpec(
            op_name=RLK_OP_RATIO_KL,
            category="loss",
            boundary="loss.ratio_kl",
            compares_logp=True,
            reference_impl=reference_ratio_kl,
            required_inputs=("log_probs", "old_log_probs"),
        ),
        RLK_OP_PPO_FRAGMENT: OperatorComparisonSpec(
            op_name=RLK_OP_PPO_FRAGMENT,
            category="loss",
            boundary="loss.ppo_policy_fragment",
            supports_grad=True,
            reference_impl=reference_ppo_fragment,
            required_inputs=("ppo_kl", "advantages"),
        ),
        RLK_OP_GRPO_FRAGMENT: OperatorComparisonSpec(
            op_name=RLK_OP_GRPO_FRAGMENT,
            category="loss",
            boundary="loss.grpo_fragment",
            unsupported_reason=(
                "GRPO currently reuses PPO policy-loss math after advantage construction; "
                "no standalone operator boundary is exposed for single-card comparison."
            ),
            required_inputs=("ppo_kl", "advantages"),
        ),
        RLK_OP_DPO_FRAGMENT: OperatorComparisonSpec(
            op_name=RLK_OP_DPO_FRAGMENT,
            category="loss",
            boundary="loss.dpo_fragment",
            unsupported_reason=(
                "DPO is not exposed as a framework operator-level training "
                "fragment in this stack."
            ),
            required_inputs=(),
        ),
    }
)


def iter_operator_comparison_specs() -> tuple[OperatorComparisonSpec, ...]:
    return tuple(OPERATOR_COMPARISON_SPECS[name] for name in PHASE4_TARGET_OPERATORS)


def get_operator_comparison_spec(op_name: str) -> OperatorComparisonSpec:
    try:
        return OPERATOR_COMPARISON_SPECS[op_name]
    except KeyError as exc:
        raise KeyError(f"unknown Phase 4 operator comparison target {op_name!r}") from exc


def run_reference_operator(op_name: str, **kwargs: Any) -> Any:
    spec = get_operator_comparison_spec(op_name)
    if spec.reference_impl is None:
        raise NotImplementedError(
            spec.unsupported_reason or f"{op_name!r} has no reference implementation."
        )
    missing = [name for name in spec.required_inputs if name not in kwargs]
    if missing:
        raise ValueError(f"{op_name!r} reference inputs missing required fields: {missing}.")
    return spec.reference_impl(**kwargs)


def compare_operator_pair(pair: OperatorPair) -> OperatorComparisonResult:
    train_output = pair.train(**pair.train_kwargs)
    infer_output = pair.infer(**pair.infer_kwargs)
    return compare_operator_outputs(
        pair.op_name,
        train_output,
        infer_output,
        tolerance=pair.tolerance,
        active_token_mask=pair.active_token_mask,
        metadata={"comparison": "train_vs_infer", **dict(pair.metadata)},
    )


def compare_operator_outputs(
    op_name: str,
    train_output: Any,
    infer_output: Any,
    *,
    tolerance: OperatorTolerance | None = None,
    active_token_mask: torch.Tensor | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> OperatorComparisonResult:
    spec = get_operator_comparison_spec(op_name)
    if not spec.supported:
        return spec.unsupported_result(metadata=metadata)
    tolerance = tolerance or spec.tolerance
    metadata = _immutable_mapping(metadata)

    try:
        pairs = _paired_tensor_leaves(train_output, infer_output)
    except ValueError as exc:
        return OperatorComparisonResult(
            op_name=op_name,
            status="failed",
            metrics={},
            compared_tensors=(),
            tolerance=tolerance.to_dict(),
            failure_reason=str(exc),
            metadata=metadata,
        )

    deltas: list[torch.Tensor] = []
    allowed_deltas: list[torch.Tensor] = []
    active_dlogp_deltas: list[torch.Tensor] = []
    compared_names: list[str] = []
    shape_failures = []
    nonfinite_delta_count = 0
    active_mask_mismatch_count = 0
    for name, train_tensor, infer_tensor in pairs:
        compared_names.append(name)
        if train_tensor.shape != infer_tensor.shape:
            shape_failures.append(
                f"{name}: train shape {tuple(train_tensor.shape)} != infer "
                f"shape {tuple(infer_tensor.shape)}"
            )
            continue
        train_float = train_tensor.detach().to(dtype=torch.float64)
        infer_float = infer_tensor.detach().to(device=train_tensor.device, dtype=torch.float64)
        delta = (train_float - infer_float).abs().flatten()
        allowed = (tolerance.atol + tolerance.rtol * infer_float.abs()).flatten()
        finite = torch.isfinite(delta)
        if not bool(finite.all().item()):
            nonfinite_delta_count += int((~finite).sum().item())
            delta = delta[finite]
            allowed = allowed[finite]
        deltas.append(delta.cpu())
        allowed_deltas.append(allowed.detach().cpu())
        active = _active_mask_for_tensor(active_token_mask, train_tensor)
        if active_token_mask is not None and active is None:
            active_mask_mismatch_count += 1
        if active is not None and _should_measure_active_dlogp(spec, name, len(pairs)):
            signed = (train_float - infer_float).flatten()[active.flatten()]
            active_dlogp_deltas.append(signed.detach().cpu())

    if shape_failures:
        return OperatorComparisonResult(
            op_name=op_name,
            status="failed",
            metrics={
                "compared_tensor_count": float(len(pairs)),
                "shape_mismatch_count": float(len(shape_failures)),
            },
            compared_tensors=tuple(compared_names),
            tolerance=tolerance.to_dict(),
            failure_reason="; ".join(shape_failures),
            metadata=metadata,
        )
    if not deltas:
        return OperatorComparisonResult(
            op_name=op_name,
            status="failed",
            metrics={"compared_tensor_count": 0.0},
            compared_tensors=tuple(compared_names),
            tolerance=tolerance.to_dict(),
            failure_reason="No tensor leaves were available for comparison.",
            metadata=metadata,
        )

    all_delta = torch.cat(deltas)
    all_allowed = torch.cat(allowed_deltas)
    metrics = _error_metrics(all_delta)
    metrics["compared_tensor_count"] = float(len(pairs))
    metrics["compared_element_count"] = float(all_delta.numel())
    metrics["nonfinite_delta_count"] = float(nonfinite_delta_count)
    if all_delta.numel():
        metrics["within_tolerance_fraction"] = float(
            (all_delta <= all_allowed).to(dtype=torch.float64).mean().item()
        )
    else:
        metrics["within_tolerance_fraction"] = 0.0
    if active_mask_mismatch_count:
        metrics["active_mask_mismatch_count"] = float(active_mask_mismatch_count)
    if active_dlogp_deltas:
        active_delta = torch.cat(active_dlogp_deltas)
        metrics.update(_active_dlogp_metrics(active_delta))
    failed = bool((all_delta > all_allowed).any().item()) if all_delta.numel() else False
    failed = failed or nonfinite_delta_count > 0 or active_mask_mismatch_count > 0
    failure_reason = None
    if failed:
        if nonfinite_delta_count:
            failure_reason = "operator outputs produced non-finite deltas"
        elif active_mask_mismatch_count:
            failure_reason = "active_token_mask did not align with compared tensor leaves"
        else:
            failure_reason = "operator outputs exceeded tolerance"
    return OperatorComparisonResult(
        op_name=op_name,
        status="failed" if failed else "passed",
        metrics=metrics,
        compared_tensors=tuple(compared_names),
        tolerance=tolerance.to_dict(),
        failure_reason=failure_reason,
        metadata=metadata,
    )


def run_forward_chain_comparison(
    initial_train_state: Any,
    initial_infer_state: Any,
    steps: Sequence[ForwardChainStep],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ForwardChainComparisonResult:
    train_state = initial_train_state
    infer_state = initial_infer_state
    results: list[OperatorComparisonResult] = []
    first_drift_operator: str | None = None
    first_failed_operator: str | None = None
    first_failed_index: int | None = None

    for index, step in enumerate(steps):
        train_state = step.train(train_state)
        infer_state = step.infer(infer_state)
        result = compare_operator_outputs(
            step.op_name,
            train_state,
            infer_state,
            tolerance=step.tolerance,
            active_token_mask=step.active_token_mask,
            metadata={"chain_index": index, **dict(step.metadata)},
        )
        results.append(result)
        if first_drift_operator is None and result.metrics.get("max_abs_error", 0.0) > 0.0:
            first_drift_operator = step.op_name
        if first_failed_operator is None and result.status == "failed":
            first_failed_operator = step.op_name
            first_failed_index = index

    cumulative = _chain_cumulative_metrics(results)
    if first_failed_operator is not None:
        cumulative["first_failed_step_index"] = float(
            first_failed_index if first_failed_index is not None else -1
        )
    return ForwardChainComparisonResult(
        status="failed" if first_failed_operator is not None else "passed",
        steps=tuple(results),
        first_drift_operator=first_drift_operator,
        cumulative_metrics=cumulative,
        metadata=metadata or {},
    )


def build_single_card_batch_invariance_cases(
    op_name: str,
    *,
    sample_position: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[BatchInvarianceCase, ...]:
    get_operator_comparison_spec(op_name)
    if sample_position < 0:
        raise ValueError(f"sample_position must be non-negative, got {sample_position!r}.")
    base = dict(metadata or {})
    return (
        BatchInvarianceCase(
            op_name=op_name,
            case="same_sample_mixed_batch",
            varied_axes=("batch_size", "neighboring_samples", "microbatch_membership"),
            sample_position=sample_position,
            expected="The fixed sample matches whether run alone or with neighbors.",
            metadata=base,
        ),
        BatchInvarianceCase(
            op_name=op_name,
            case="padding_packing_layout",
            varied_axes=("padding_side", "packed_order", "microbatch_offset"),
            sample_position=sample_position,
            expected="Padding and packing changes do not alter active-token operator outputs.",
            metadata=base,
        ),
        BatchInvarianceCase(
            op_name=op_name,
            case="row_position",
            varied_axes=("batch_row_index", "sequence_row_index"),
            sample_position=sample_position,
            expected="Moving the same sample to another row keeps its selected outputs invariant.",
            metadata=base,
        ),
        BatchInvarianceCase(
            op_name=op_name,
            case="active_mask_density",
            varied_axes=("active_mask_density", "neighbor_active_tokens"),
            sample_position=sample_position,
            expected="Neighboring active-token density does not change the fixed sample.",
            metadata=base,
        ),
    )


def compare_batch_invariance(
    op_name: str,
    sample_alone_output: Any,
    mixed_batch_output: Any,
    *,
    sample_position: int = 0,
    case: str = "same_sample_mixed_batch",
    tolerance: OperatorTolerance | None = None,
    active_token_mask: torch.Tensor | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> OperatorComparisonResult:
    if sample_position < 0:
        raise ValueError(f"sample_position must be non-negative, got {sample_position!r}.")
    fixed_sample_output = _select_batch_position(mixed_batch_output, sample_position)
    return compare_operator_outputs(
        op_name,
        sample_alone_output,
        fixed_sample_output,
        tolerance=tolerance,
        active_token_mask=active_token_mask,
        metadata={
            "comparison": "batch_invariance",
            "case": case,
            "sample_position": sample_position,
            **dict(metadata or {}),
        },
    )


def run_deterministic_repeatability_check(
    op_name: str,
    backend: Any,
    run_once: Callable[[], Any],
    *,
    repetitions: int = 2,
    metadata: Mapping[str, Any] | None = None,
) -> OperatorComparisonResult:
    backend_operator = _backend_field(backend, "operator", "")
    backend_id = _backend_field(backend, "backend_id", "unknown")
    if backend_operator != op_name:
        return OperatorComparisonResult(
            op_name=op_name,
            status="unsupported",
            unsupported_reason=(
                f"Backend {backend_id!r} advertises operator {backend_operator!r}, "
                f"not {op_name!r}."
            ),
            metadata=metadata or {},
        )
    if not _backend_field(backend, "deterministic", False):
        return OperatorComparisonResult(
            op_name=op_name,
            status="unsupported",
            unsupported_reason=(
                f"Backend {backend_id!r} does not advertise same-backend "
                "same-build deterministic behavior."
            ),
            metadata={"backend_id": backend_id, **dict(metadata or {})},
        )
    if repetitions < 2:
        raise ValueError(f"repetitions must be at least 2, got {repetitions!r}.")

    outputs = [run_once() for _ in range(repetitions)]
    baseline = outputs[0]
    compared_names: list[str] = []
    mismatch_count = 0
    for repeat_index, output in enumerate(outputs[1:], start=1):
        try:
            pairs = _paired_tensor_leaves(baseline, output)
        except ValueError:
            mismatch_count += 1
            continue
        for name, expected, actual in pairs:
            compared_names.append(f"repeat_{repeat_index}.{name}")
            if expected.shape != actual.shape or not torch.equal(
                expected.detach().cpu(), actual.detach().cpu()
            ):
                mismatch_count += 1

    return OperatorComparisonResult(
        op_name=op_name,
        status="failed" if mismatch_count else "passed",
        metrics={
            "repeat_count": float(repetitions),
            "compared_tensor_count": float(len(compared_names)),
            "bitwise_mismatch_count": float(mismatch_count),
        },
        compared_tensors=tuple(compared_names),
        tolerance={"bitwise": 0.0},
        failure_reason="deterministic backend produced non-bitwise-identical outputs"
        if mismatch_count
        else None,
        metadata={"backend_id": backend_id, **dict(metadata or {})},
    )


def build_strict_backend_admission_report(
    backend: Any,
    comparisons: Sequence[OperatorComparisonResult],
    *,
    require_batch_invariant_capability: bool = True,
    require_deterministic_for_logp: bool = False,
    require_gradients: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> StrictBackendAdmissionReport:
    backend_operator = _backend_field(backend, "operator", "")
    backend_id = _backend_field(backend, "backend_id", "unknown")
    failed = tuple(result.op_name for result in comparisons if result.status == "failed")
    unsupported = tuple(result.op_name for result in comparisons if result.status == "unsupported")
    reasons: list[str] = []
    if not comparisons:
        reasons.append("comparison_missing")
    if (
        backend_operator
        and comparisons
        and any(result.op_name != backend_operator for result in comparisons)
    ):
        reasons.append("comparison_operator_mismatch")
    if require_batch_invariant_capability and not _backend_field(backend, "batch_invariant", False):
        reasons.append("batch_invariant_capability_missing")
    if (
        require_deterministic_for_logp
        and backend_operator in {RLK_OP_LOGP, RLK_OP_RATIO_KL}
        and not _backend_field(backend, "deterministic", False)
    ):
        reasons.append("deterministic_logp_capability_missing")
    try:
        spec = get_operator_comparison_spec(backend_operator)
    except KeyError:
        spec = None
    if require_gradients and spec is not None and spec.supports_grad:
        if not any(result.metadata.get("gradients_checked") is True for result in comparisons):
            reasons.append("gradient_comparison_missing")
    if failed:
        reasons.append("comparison_failed")
    if unsupported:
        reasons.append("comparison_unsupported")
    strict_fast_eligible = not reasons and bool(comparisons)
    return StrictBackendAdmissionReport(
        backend_id=backend_id,
        operator=backend_operator,
        strict_fast_eligible=strict_fast_eligible,
        reasons=tuple(reasons),
        comparison_count=len(comparisons),
        failed_comparisons=failed,
        unsupported_comparisons=unsupported,
        metadata=metadata or {},
    )


def _backend_field(backend: Any, name: str, default: Any = None) -> Any:
    if isinstance(backend, Mapping):
        return backend.get(name, default)
    return getattr(backend, name, default)


def _error_metrics(delta: torch.Tensor) -> dict[str, float]:
    if delta.numel() == 0:
        return {
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "p99_abs_error": 0.0,
        }
    return {
        "max_abs_error": float(delta.max().item()),
        "mean_abs_error": float(delta.mean().item()),
        "p99_abs_error": float(_quantile(delta, 0.99).item()),
    }


def _active_dlogp_metrics(delta: torch.Tensor) -> dict[str, float]:
    abs_delta = delta.abs()
    metrics = {
        "active_token_count": float(delta.numel()),
        "active_token_dlogp_abs_max": float(abs_delta.max().item()) if delta.numel() else 0.0,
        "active_token_dlogp_abs_mean": float(abs_delta.mean().item()) if delta.numel() else 0.0,
        "active_token_dlogp_abs_p99": float(_quantile(abs_delta, 0.99).item())
        if delta.numel()
        else 0.0,
        "active_token_dlogp_signed_sum": float(delta.sum().item()) if delta.numel() else 0.0,
        "active_token_dlogp_abs_sum": float(abs_delta.sum().item()) if delta.numel() else 0.0,
    }
    return metrics


def _quantile(values: torch.Tensor, q: float) -> torch.Tensor:
    if values.numel() == 1:
        return values[0]
    return torch.quantile(values.to(dtype=torch.float64), q)


def _chain_cumulative_metrics(results: Sequence[OperatorComparisonResult]) -> dict[str, float]:
    if not results:
        return {
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "p99_abs_error": 0.0,
            "step_count": 0.0,
        }
    cumulative = {
        "max_abs_error": max(result.metrics.get("max_abs_error", 0.0) for result in results),
        "mean_abs_error": results[-1].metrics.get("mean_abs_error", 0.0),
        "p99_abs_error": results[-1].metrics.get("p99_abs_error", 0.0),
        "failed_step_count": float(sum(1 for result in results if result.status == "failed")),
        "unsupported_step_count": float(
            sum(1 for result in results if result.status == "unsupported")
        ),
        "step_count": float(len(results)),
    }
    return {key: float(value) for key, value in cumulative.items()}


def _paired_tensor_leaves(
    train_output: Any, infer_output: Any, prefix: str = "output"
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    if isinstance(train_output, torch.Tensor) and isinstance(infer_output, torch.Tensor):
        return [(prefix, train_output, infer_output)]
    if isinstance(train_output, Mapping) and isinstance(infer_output, Mapping):
        if set(train_output) != set(infer_output):
            raise ValueError(
                f"{prefix} mapping keys differ: train={sorted(train_output, key=str)}, "
                f"infer={sorted(infer_output, key=str)}."
            )
        pairs = []
        for key in sorted(train_output, key=str):
            pairs.extend(
                _paired_tensor_leaves(train_output[key], infer_output[key], f"{prefix}.{key}")
            )
        return pairs
    if _is_sequence(train_output) and _is_sequence(infer_output):
        if len(train_output) != len(infer_output):
            raise ValueError(
                f"{prefix} sequence lengths differ: train={len(train_output)}, "
                f"infer={len(infer_output)}."
            )
        pairs = []
        for index, (train_item, infer_item) in enumerate(
            zip(train_output, infer_output, strict=True)
        ):
            pairs.extend(_paired_tensor_leaves(train_item, infer_item, f"{prefix}.{index}"))
        return pairs
    if _is_number(train_output) and _is_number(infer_output):
        return [(prefix, torch.tensor(train_output), torch.tensor(infer_output))]
    raise ValueError(
        f"{prefix} has no comparable tensor leaves: train={type(train_output)!r}, "
        f"infer={type(infer_output)!r}."
    )


def _select_batch_position(output: Any, sample_position: int) -> Any:
    if isinstance(output, torch.Tensor):
        if output.ndim == 0:
            return output
        if sample_position >= output.shape[0]:
            raise IndexError(
                f"sample_position {sample_position} is outside batch dimension {output.shape[0]}."
            )
        return output[sample_position]
    if isinstance(output, Mapping):
        return {
            key: _select_batch_position(value, sample_position) for key, value in output.items()
        }
    if _is_sequence(output):
        return type(output)(_select_batch_position(value, sample_position) for value in output)
    return output


def _active_mask_for_tensor(mask: torch.Tensor | None, tensor: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    active = mask.detach().to(device=tensor.device, dtype=torch.bool)
    if active.shape == tensor.shape:
        return active
    if tensor.ndim > 0 and active.shape == tensor.shape[: active.ndim]:
        while active.ndim < tensor.ndim:
            active = active.unsqueeze(-1)
        return active.expand_as(tensor)
    if active.numel() == tensor.numel():
        return active.reshape(tensor.shape)
    return None


def _should_measure_active_dlogp(
    spec: OperatorComparisonSpec, leaf_name: str, pair_count: int
) -> bool:
    if not spec.compares_logp:
        return False
    if pair_count == 1 and leaf_name == "output":
        return True
    normalized = leaf_name.rsplit(".", 1)[-1].lower()
    return (
        normalized in {"dlogp", "logp", "log_probs", "selected_logprobs"} or "dlogp" in normalized
    )


def _immutable_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float | bool) and not isinstance(value, bool)
