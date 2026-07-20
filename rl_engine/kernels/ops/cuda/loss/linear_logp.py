# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import os
from typing import Any, Optional

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.pytorch.loss.linear_logp import (
    _merge_tp_local_logp,
    _require_distributed_initialized,
    _validate_global_targets,
    _validate_tp_targets_enabled,
    _validate_tp_vocab_partition_cached,
    chunked_linear_logp_backward,
    should_use_tensor_parallel_linear_logp,
    tensor_parallel_linear_logp,
    tensor_parallel_linear_logp_backward,
)
from rl_engine.utils.logger import logger

# Hidden-dim slice the SM90 forward streams per TMA load; D must be a multiple of
# it (mirrors `constexpr int BK` in csrc/cuda/fused_linear_logp_sm90.cu).
_SM90_BK = 32
_SM90_TP_PATH_LOGGED = False
_SM90_FUSED_BWD_PATH_LOGGED = False
_SM90_SAVE_PROBS_BF16_PATH_LOGGED = False
_SM90_SAVE_PROBS_BF16_SKIP_LOGGED = False
_SM90_FUSED_TILE_BF16_PATH_LOGGED = False


def _env_disabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"0", "false", "no", "off"}


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _fused_bwd_precision() -> str:
    return os.getenv("RL_KERNEL_LINEAR_LOGP_FUSED_BWD_PRECISION", "auto").strip().lower()


def _sm90_fused_backward_tf32_enabled() -> bool:
    return _fused_bwd_precision() != "fp32" and not _env_disabled(
        "RL_KERNEL_LINEAR_LOGP_FUSED_BWD_TF32"
    )


def _sm90_fused_backward_available() -> bool:
    return (
        _EXT_AVAILABLE
        and hasattr(_C, "fused_linear_logp_sm90_backward")
        and not _env_disabled("RL_KERNEL_LINEAR_LOGP_FUSED_BACKWARD")
    )


def _sm90_global_target_forward_available() -> bool:
    return _EXT_AVAILABLE and hasattr(_C, "fused_linear_logp_sm90_global_target")


def _sm90_save_probs_bf16_available() -> bool:
    return (
        _EXT_AVAILABLE
        and _env_enabled("RL_KERNEL_LINEAR_LOGP_SAVE_PROBS_BF16")
        and hasattr(_C, "linear_logp_probs_bf16_forward")
        and hasattr(_C, "linear_logp_probs_bf16_to_dlogits_")
    )


def _sm90_save_probs_bf16_tp_available() -> bool:
    return (
        _sm90_save_probs_bf16_available()
        and hasattr(_C, "linear_logp_local_probs_bf16_forward")
        and hasattr(_C, "linear_logp_local_probs_bf16_to_dlogits_")
    )


def _sm90_fused_tile_full_enabled() -> bool:
    value = os.getenv("RL_KERNEL_LINEAR_LOGP_FUSED_TILE_BWD_FULL", "").strip().lower()
    return value not in {"", "0", "false", "no", "off"}


def _sm90_fused_tile_bf16_forward_available(*, tp_path: bool) -> bool:
    if not (_EXT_AVAILABLE and _sm90_fused_tile_full_enabled()):
        return False
    if not _sm90_fused_backward_available():
        return False
    if not (
        hasattr(_C, "linear_logp_bf16_forward")
        and hasattr(_C, "linear_logp_logits_bf16_to_dlogits")
    ):
        return False
    if tp_path and not hasattr(_C, "linear_logp_local_bf16_forward"):
        return False
    return True


def _fused_tile_vocab_tile(vocab: int) -> int:
    raw = os.getenv("RL_KERNEL_LINEAR_LOGP_STREAMING_BWD_VOCAB_TILE")
    if raw is None or raw.strip() == "":
        return max(1, vocab)
    try:
        tile = int(raw)
    except ValueError:
        tile = vocab
    return max(1, min(tile, vocab))


def _log_sm90_save_probs_bf16_skip_once(
    *, tp_path: bool, details: dict[str, Any] | None = None, **conditions: bool
) -> None:
    global _SM90_SAVE_PROBS_BF16_SKIP_LOGGED
    if _SM90_SAVE_PROBS_BF16_SKIP_LOGGED or not _env_enabled(
        "RL_KERNEL_LINEAR_LOGP_SAVE_PROBS_BF16"
    ):
        return
    failed = [name for name, ok in conditions.items() if not ok]
    logger.info(
        "save-probs bf16 %slinear_logp fast path not selected; failed=%s conditions=%s details=%s",
        "tensor-parallel " if tp_path else "",
        failed,
        conditions,
        details,
    )
    _SM90_SAVE_PROBS_BF16_SKIP_LOGGED = True


def _sm90_linear_logp_backward(
    grad_logp: torch.Tensor,
    hidden_2d: torch.Tensor,
    weight: torch.Tensor,
    bias_t: torch.Tensor,
    target_1d: torch.Tensor,
    lse: torch.Tensor,
    *,
    has_bias: bool,
    lead_shape,
    hidden_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    bias_dtype,
    vocab_start_index: int = 0,
    tp_group: Any = None,
    compute_grad_hidden: bool = True,
    compute_grad_weight: bool = True,
    compute_grad_bias: bool = True,
):
    global _SM90_FUSED_BWD_PATH_LOGGED
    if not (compute_grad_hidden or compute_grad_weight or (has_bias and compute_grad_bias)):
        return None, None, None

    if _sm90_fused_backward_available():
        if not _SM90_FUSED_BWD_PATH_LOGGED:
            logger.info("Using fused_linear_logp_sm90_backward fast path.")
            _SM90_FUSED_BWD_PATH_LOGGED = True
        prev_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        enable_tf32 = _sm90_fused_backward_tf32_enabled()
        if enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
        try:
            grad_hidden_t, grad_weight_t, grad_bias_t = _C.fused_linear_logp_sm90_backward(
                grad_logp,
                hidden_2d,
                weight,
                target_1d,
                lse,
                bias_t if has_bias else None,
                int(vocab_start_index),
                bool(compute_grad_hidden),
                bool(compute_grad_weight),
                bool(has_bias and compute_grad_bias),
                bool(tp_group is not None),
            )
        finally:
            if enable_tf32:
                torch.backends.cuda.matmul.allow_tf32 = prev_allow_tf32
        grad_hidden = None
        if compute_grad_hidden:
            if tp_group is not None:
                dist = _require_distributed_initialized()
                dist.all_reduce(grad_hidden_t, op=dist.ReduceOp.SUM, group=tp_group)
            grad_hidden = grad_hidden_t.to(hidden_dtype).reshape(
                (*tuple(lead_shape), hidden_2d.size(1))
            )
        grad_weight = grad_weight_t.to(weight_dtype) if compute_grad_weight else None
        grad_bias = grad_bias_t.to(bias_dtype) if has_bias and compute_grad_bias else None
        return grad_hidden, grad_weight, grad_bias

    if tp_group is not None:
        return tensor_parallel_linear_logp_backward(
            grad_logp,
            hidden_2d,
            weight,
            bias_t,
            target_1d,
            lse,
            has_bias=has_bias,
            lead_shape=lead_shape,
            hidden_dtype=hidden_dtype,
            weight_dtype=weight_dtype,
            bias_dtype=bias_dtype,
            vocab_start_index=vocab_start_index,
            tp_group=tp_group,
            compute_grad_hidden=compute_grad_hidden,
            compute_grad_weight=compute_grad_weight,
            compute_grad_bias=compute_grad_bias,
        )
    return chunked_linear_logp_backward(
        grad_logp,
        hidden_2d,
        weight,
        target_1d,
        bias_t,
        has_bias=has_bias,
        lead_shape=lead_shape,
        hidden_dtype=hidden_dtype,
        weight_dtype=weight_dtype,
        bias_dtype=bias_dtype,
        compute_grad_hidden=compute_grad_hidden,
        compute_grad_weight=compute_grad_weight,
        compute_grad_bias=compute_grad_bias,
    )


class _LinearLogpSaveProbsBF16Function(torch.autograd.Function):
    """bf16 path that stores softmax probs instead of log-probs.

    It relies on cuBLAS for the linear GEMMs and uses the CUDA helper only for
    the softmax/probability transforms. When hidden needs gradients, backward
    also computes ``grad_hidden = dlogits @ weight``.
    """

    @staticmethod
    def forward(ctx, hidden, lm_head_weight, target_ids):
        hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous()
        weight = lm_head_weight.contiguous()
        target_1d = (
            target_ids.reshape(-1).to(device=hidden_2d.device, dtype=torch.long).contiguous()
        )
        logits = torch.nn.functional.linear(hidden_2d, weight, None)
        logp, probs = _C.linear_logp_probs_bf16_forward(logits, target_1d, 0)
        ctx.save_for_backward(probs, hidden_2d, weight, target_1d)
        ctx.lead_shape = hidden.shape[:-1]
        return logp.reshape(hidden.shape[:-1])

    @staticmethod
    def backward(ctx, grad_logp):
        probs, hidden_2d, weight, target_1d = ctx.saved_tensors
        grad_hidden = None
        grad_weight = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            _C.linear_logp_probs_bf16_to_dlogits_(probs, target_1d, grad_logp, 0)
        if ctx.needs_input_grad[1]:
            grad_weight = probs.t().matmul(hidden_2d)
        if ctx.needs_input_grad[0]:
            grad_hidden = probs.matmul(weight).reshape((*tuple(ctx.lead_shape), weight.size(1)))
        return grad_hidden, grad_weight, None


class _TensorParallelLinearLogpSaveProbsBF16Function(torch.autograd.Function):
    """TP bf16 path that saves local softmax probs.

    Each rank materializes only its local vocab shard logits/probs. The forward
    merges local ``target_logit`` and ``lse`` through the shared TP fast merge;
    backward converts local probs to global dlogits using the saved local/global
    lse, computes the local shard's ``grad_weight`` with cuBLAS, and all-reduces
    ``grad_hidden`` when hidden participates in training.
    """

    @staticmethod
    def forward(
        ctx, hidden, lm_head_weight, target_ids, vocab_start_index, global_vocab_size, tp_group
    ):
        hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous()
        weight = lm_head_weight.contiguous()
        target_1d = (
            target_ids.reshape(-1).to(device=hidden_2d.device, dtype=torch.long).contiguous()
        )
        vocab_start_index = int(vocab_start_index)
        global_vocab_size = _validate_tp_vocab_partition_cached(
            tp_group=tp_group,
            device=hidden_2d.device,
            vocab_start_index=vocab_start_index,
            local_vocab_size=weight.size(0),
            global_vocab_size=global_vocab_size,
        )
        if _validate_tp_targets_enabled():
            _validate_global_targets(target_1d, global_vocab_size, tp_group)

        logits = torch.nn.functional.linear(hidden_2d, weight, None)
        local_target_logit, local_lse, probs = _C.linear_logp_local_probs_bf16_forward(
            logits,
            target_1d,
            vocab_start_index,
        )
        log_prob, global_lse = _merge_tp_local_logp(
            local_lse,
            local_target_logit,
            tp_group=tp_group,
        )

        ctx.save_for_backward(probs, hidden_2d, weight, target_1d, local_lse, global_lse)
        ctx.lead_shape = hidden.shape[:-1]
        ctx.vocab_start_index = vocab_start_index
        ctx.tp_group = tp_group
        return log_prob.reshape(hidden.shape[:-1])

    @staticmethod
    def backward(ctx, grad_logp):
        probs, hidden_2d, weight, target_1d, local_lse, global_lse = ctx.saved_tensors
        grad_hidden = None
        grad_weight = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            _C.linear_logp_local_probs_bf16_to_dlogits_(
                probs,
                target_1d,
                grad_logp,
                local_lse,
                global_lse,
                ctx.vocab_start_index,
            )
        if ctx.needs_input_grad[1]:
            grad_weight = probs.t().matmul(hidden_2d)
        if ctx.needs_input_grad[0]:
            grad_hidden_2d = probs.matmul(weight)
            dist = _require_distributed_initialized()
            dist.all_reduce(grad_hidden_2d, op=dist.ReduceOp.SUM, group=ctx.tp_group)
            grad_hidden = grad_hidden_2d.reshape((*tuple(ctx.lead_shape), weight.size(1)))
        return grad_hidden, grad_weight, None, None, None, None


class _LinearLogpFusedTileBF16Function(torch.autograd.Function):
    """cuBLAS bf16 logits forward + tiled backward without saving probs."""

    @staticmethod
    def forward(ctx, hidden, lm_head_weight, target_ids):
        hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous()
        weight = lm_head_weight.contiguous()
        target_1d = (
            target_ids.reshape(-1).to(device=hidden_2d.device, dtype=torch.long).contiguous()
        )
        logits = torch.nn.functional.linear(hidden_2d, weight, None)
        logp, lse = _C.linear_logp_bf16_forward(logits, target_1d, 0)
        ctx.save_for_backward(logits, hidden_2d, weight, target_1d, lse)
        ctx.lead_shape = hidden.shape[:-1]
        ctx.hidden_dtype = hidden.dtype
        ctx.weight_dtype = lm_head_weight.dtype
        return logp.reshape(hidden.shape[:-1])

    @staticmethod
    def backward(ctx, grad_logp):
        logits, hidden_2d, weight, target_1d, lse = ctx.saved_tensors
        compute_grad_hidden = ctx.needs_input_grad[0]
        compute_grad_weight = ctx.needs_input_grad[1]
        grad_hidden_2d = None
        grad_weight = None
        if compute_grad_hidden or compute_grad_weight:
            n_tokens, vocab = logits.shape
            tile_v = _fused_tile_vocab_tile(vocab)
            use_inplace_logits = tile_v >= vocab
            dlogits_workspace = None
            if not use_inplace_logits:
                dlogits_workspace = torch.empty(
                    (n_tokens, min(tile_v, vocab)),
                    device=logits.device,
                    dtype=logits.dtype,
                )
            grad_hidden_initialized = False
            if compute_grad_hidden:
                grad_hidden_2d = torch.empty_like(hidden_2d)
            if compute_grad_weight:
                grad_weight = torch.empty_like(weight)
            for v0 in range(0, vocab, tile_v):
                vc = min(tile_v, vocab - v0)
                logits_tile = logits.narrow(1, v0, vc)
                if use_inplace_logits:
                    dlogits_tile = logits_tile
                else:
                    dlogits_tile = (
                        dlogits_workspace
                        if vc == dlogits_workspace.size(1)
                        else torch.empty((n_tokens, vc), device=logits.device, dtype=logits.dtype)
                    )
                _C.linear_logp_logits_bf16_to_dlogits(
                    logits_tile,
                    dlogits_tile,
                    target_1d,
                    grad_logp,
                    lse,
                    v0,
                )
                weight_tile = weight.narrow(0, v0, vc)
                if compute_grad_hidden:
                    if not grad_hidden_initialized:
                        torch.mm(dlogits_tile, weight_tile, out=grad_hidden_2d)
                        grad_hidden_initialized = True
                    else:
                        torch.addmm(
                            grad_hidden_2d,
                            dlogits_tile,
                            weight_tile,
                            out=grad_hidden_2d,
                        )
                if compute_grad_weight:
                    torch.mm(
                        dlogits_tile.transpose(0, 1),
                        hidden_2d,
                        out=grad_weight.narrow(0, v0, vc),
                    )
        grad_hidden = (
            None
            if grad_hidden_2d is None
            else grad_hidden_2d.to(ctx.hidden_dtype).reshape(
                (*tuple(ctx.lead_shape), hidden_2d.size(1))
            )
        )
        if grad_weight is not None:
            grad_weight = grad_weight.to(ctx.weight_dtype)
        return grad_hidden, grad_weight, None


class _TensorParallelLinearLogpFusedTileBF16Function(torch.autograd.Function):
    """TP cuBLAS bf16 logits forward + tiled local backward."""

    @staticmethod
    def forward(
        ctx, hidden, lm_head_weight, target_ids, vocab_start_index, global_vocab_size, tp_group
    ):
        hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous()
        weight = lm_head_weight.contiguous()
        target_1d = (
            target_ids.reshape(-1).to(device=hidden_2d.device, dtype=torch.long).contiguous()
        )
        vocab_start_index = int(vocab_start_index)
        global_vocab_size = _validate_tp_vocab_partition_cached(
            tp_group=tp_group,
            device=hidden_2d.device,
            vocab_start_index=vocab_start_index,
            local_vocab_size=weight.size(0),
            global_vocab_size=global_vocab_size,
        )
        if _validate_tp_targets_enabled():
            _validate_global_targets(target_1d, global_vocab_size, tp_group)

        logits = torch.nn.functional.linear(hidden_2d, weight, None)
        local_target_logit, local_lse = _C.linear_logp_local_bf16_forward(
            logits,
            target_1d,
            vocab_start_index,
        )
        log_prob, global_lse = _merge_tp_local_logp(
            local_lse,
            local_target_logit,
            tp_group=tp_group,
        )

        ctx.save_for_backward(logits, hidden_2d, weight, target_1d, global_lse)
        ctx.lead_shape = hidden.shape[:-1]
        ctx.hidden_dtype = hidden.dtype
        ctx.weight_dtype = lm_head_weight.dtype
        ctx.vocab_start_index = vocab_start_index
        ctx.tp_group = tp_group
        return log_prob.reshape(hidden.shape[:-1])

    @staticmethod
    def backward(ctx, grad_logp):
        logits, hidden_2d, weight, target_1d, global_lse = ctx.saved_tensors
        compute_grad_hidden = ctx.needs_input_grad[0]
        compute_grad_weight = ctx.needs_input_grad[1]
        grad_hidden_2d = None
        grad_weight = None
        if compute_grad_hidden or compute_grad_weight:
            n_tokens, vocab = logits.shape
            tile_v = _fused_tile_vocab_tile(vocab)
            use_inplace_logits = tile_v >= vocab
            dlogits_workspace = None
            if not use_inplace_logits:
                dlogits_workspace = torch.empty(
                    (n_tokens, min(tile_v, vocab)),
                    device=logits.device,
                    dtype=logits.dtype,
                )
            grad_hidden_initialized = False
            if compute_grad_hidden:
                grad_hidden_2d = torch.empty_like(hidden_2d)
            if compute_grad_weight:
                grad_weight = torch.empty_like(weight)
            for v0 in range(0, vocab, tile_v):
                vc = min(tile_v, vocab - v0)
                logits_tile = logits.narrow(1, v0, vc)
                if use_inplace_logits:
                    dlogits_tile = logits_tile
                else:
                    dlogits_tile = (
                        dlogits_workspace
                        if vc == dlogits_workspace.size(1)
                        else torch.empty((n_tokens, vc), device=logits.device, dtype=logits.dtype)
                    )
                _C.linear_logp_logits_bf16_to_dlogits(
                    logits_tile,
                    dlogits_tile,
                    target_1d,
                    grad_logp,
                    global_lse,
                    ctx.vocab_start_index + v0,
                )
                weight_tile = weight.narrow(0, v0, vc)
                if compute_grad_hidden:
                    if not grad_hidden_initialized:
                        torch.mm(dlogits_tile, weight_tile, out=grad_hidden_2d)
                        grad_hidden_initialized = True
                    else:
                        torch.addmm(
                            grad_hidden_2d,
                            dlogits_tile,
                            weight_tile,
                            out=grad_hidden_2d,
                        )
                if compute_grad_weight:
                    torch.mm(
                        dlogits_tile.transpose(0, 1),
                        hidden_2d,
                        out=grad_weight.narrow(0, v0, vc),
                    )
        grad_hidden = None
        if grad_hidden_2d is not None:
            dist = _require_distributed_initialized()
            dist.all_reduce(grad_hidden_2d, op=dist.ReduceOp.SUM, group=ctx.tp_group)
            grad_hidden = grad_hidden_2d.to(ctx.hidden_dtype).reshape(
                (*tuple(ctx.lead_shape), hidden_2d.size(1))
            )
        if grad_weight is not None:
            grad_weight = grad_weight.to(ctx.weight_dtype)
        return grad_hidden, grad_weight, None, None, None, None


def _sm90_support_details(hidden: torch.Tensor, lm_head_weight: torch.Tensor) -> dict[str, Any]:
    """Per-condition diagnostics for the bf16 TMA+MMA forward."""
    same_device = hidden.device == lm_head_weight.device
    cc_major = None
    cc_minor = None
    if hidden.is_cuda:
        try:
            cc_major, cc_minor = torch.cuda.get_device_capability(hidden.device)
        except Exception:
            pass
    return {
        "hidden_is_cuda": hidden.is_cuda,
        "weight_is_cuda": lm_head_weight.is_cuda,
        "same_device": same_device,
        "cc_major_9": cc_major == 9,
        "hidden_bf16": hidden.dtype == torch.bfloat16,
        "weight_bf16": lm_head_weight.dtype == torch.bfloat16,
        "hidden_dim_multiple_32": hidden.size(-1) % _SM90_BK == 0,
        "hidden_shape": tuple(hidden.shape),
        "hidden_dtype": str(hidden.dtype),
        "hidden_device": str(hidden.device),
        "hidden_requires_grad": bool(hidden.requires_grad),
        "weight_shape": tuple(lm_head_weight.shape),
        "weight_dtype": str(lm_head_weight.dtype),
        "weight_device": str(lm_head_weight.device),
        "weight_requires_grad": bool(lm_head_weight.requires_grad),
        "device_capability": None if cc_major is None else (cc_major, cc_minor),
    }


def _sm90_supported(hidden: torch.Tensor, lm_head_weight: torch.Tensor) -> bool:
    """Whether the bf16 TMA+MMA forward can run these inputs directly."""
    details = _sm90_support_details(hidden, lm_head_weight)
    return all(
        bool(details[name])
        for name in (
            "hidden_is_cuda",
            "weight_is_cuda",
            "same_device",
            "cc_major_9",
            "hidden_bf16",
            "weight_bf16",
            "hidden_dim_multiple_32",
        )
    )


def _fallback_op():
    """Portable op for inputs the SM90 forward cannot take (fp32/fp16, or a hidden
    dim not divisible by the kernel's K slice). Prefers Triton, else native."""
    try:
        from rl_engine.kernels.ops.triton.loss.linear_logp import TritonLinearLogpOp

        return TritonLinearLogpOp()
    except Exception:  # pragma: no cover - Triton missing
        from rl_engine.kernels.ops.pytorch.loss.linear_logp import NativeLinearLogpOp

        return NativeLinearLogpOp()


class _FusedLinearLogpSM90Function(torch.autograd.Function):
    """SM90 TMA+WGMMA fused forward + CUDA fused backward when available.

    The forward calls the compiled ``_C.fused_linear_logp_sm90`` kernel (logits
    never materialized). The fast backward is a CUDA extension path that
    materializes local dlogits once and uses GEMMs for the linear gradients,
    falling back to the deterministic chunked path when unavailable.
    """

    @staticmethod
    def forward(ctx, hidden, lm_head_weight, bias, target_ids):
        hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous()
        weight = lm_head_weight.contiguous()
        target_1d = (
            target_ids.reshape(-1).to(device=hidden_2d.device, dtype=torch.int32).contiguous()
        )
        logp, lse = _C.fused_linear_logp_sm90(hidden_2d, weight, target_1d, bias)

        ctx.save_for_backward(
            hidden_2d,
            weight,
            bias if bias is not None else hidden_2d,
            target_1d,
            lse,
        )
        ctx.has_bias = bias is not None
        ctx.lead_shape = hidden.shape[:-1]
        ctx.hidden_dtype = hidden.dtype
        ctx.weight_dtype = lm_head_weight.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        return logp.reshape(hidden.shape[:-1])

    @staticmethod
    def backward(ctx, grad_logp):
        hidden_2d, weight, bias_t, target_1d, lse = ctx.saved_tensors
        grad_hidden, grad_weight, grad_bias = _sm90_linear_logp_backward(
            grad_logp,
            hidden_2d,
            weight,
            bias_t,
            target_1d,
            lse,
            has_bias=ctx.has_bias,
            lead_shape=ctx.lead_shape,
            hidden_dtype=ctx.hidden_dtype,
            weight_dtype=ctx.weight_dtype,
            bias_dtype=ctx.bias_dtype,
            compute_grad_hidden=ctx.needs_input_grad[0],
            compute_grad_weight=ctx.needs_input_grad[1],
            compute_grad_bias=ctx.needs_input_grad[2],
        )
        # Inputs: hidden, lm_head_weight, bias, target_ids.
        return grad_hidden, grad_weight, grad_bias, None


class _FusedTensorParallelLinearLogpSM90Function(torch.autograd.Function):
    """SM90 local-shard forward with tensor-parallel logsumexp reduction.

    Each rank runs the fused SM90 kernel over its local vocab shard to get local
    log-sum-exp and the owned target logit. TP ranks then merge those states into
    the global selected log-prob. Backward intentionally reuses the existing
    the global selected log-prob. Backward uses the CUDA fused backward fast path
    when available and otherwise falls back to the portable TP helper.
    """

    @staticmethod
    def forward(
        ctx,
        hidden,
        lm_head_weight,
        bias,
        target_ids,
        vocab_start_index,
        global_vocab_size,
        tp_group,
    ):
        hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous()
        weight = lm_head_weight.contiguous()
        target_1d = (
            target_ids.reshape(-1).to(device=hidden_2d.device, dtype=torch.long).contiguous()
        )
        bias_t = bias.contiguous() if bias is not None else hidden_2d
        vocab_start_index = int(vocab_start_index)
        global_vocab_size = _validate_tp_vocab_partition_cached(
            tp_group=tp_group,
            device=hidden_2d.device,
            vocab_start_index=vocab_start_index,
            local_vocab_size=weight.size(0),
            global_vocab_size=global_vocab_size,
        )
        if _validate_tp_targets_enabled():
            _validate_global_targets(target_1d, global_vocab_size, tp_group)

        if _sm90_global_target_forward_available():
            local_target_logit, local_lse = _C.fused_linear_logp_sm90_global_target(
                hidden_2d,
                weight,
                target_1d,
                bias,
                vocab_start_index,
            )
        else:
            local_vocab = weight.size(0)
            local_target = target_1d - vocab_start_index
            owns_target = (local_target >= 0) & (local_target < local_vocab)
            kernel_target = torch.where(
                local_target >= 0, local_target, torch.zeros_like(local_target)
            )
            kernel_target = torch.where(
                kernel_target < local_vocab, kernel_target, torch.zeros_like(kernel_target)
            )
            kernel_target = kernel_target.to(torch.int32).contiguous()

            local_logp, local_lse = _C.fused_linear_logp_sm90(
                hidden_2d, weight, kernel_target, bias
            )
            local_target_logit = torch.where(
                owns_target, local_logp + local_lse, torch.zeros_like(local_lse)
            )
        log_prob, global_lse = _merge_tp_local_logp(
            local_lse,
            local_target_logit,
            tp_group=tp_group,
        )

        ctx.save_for_backward(hidden_2d, weight, bias_t, target_1d, global_lse)
        ctx.has_bias = bias is not None
        ctx.lead_shape = hidden.shape[:-1]
        ctx.hidden_dtype = hidden.dtype
        ctx.weight_dtype = lm_head_weight.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.vocab_start_index = vocab_start_index
        ctx.tp_group = tp_group
        return log_prob.reshape(hidden.shape[:-1])

    @staticmethod
    def backward(ctx, grad_logp):
        hidden_2d, weight, bias_t, target_1d, global_lse = ctx.saved_tensors
        grad_hidden, grad_weight, grad_bias = _sm90_linear_logp_backward(
            grad_logp,
            hidden_2d,
            weight,
            bias_t,
            target_1d,
            global_lse,
            has_bias=ctx.has_bias,
            lead_shape=ctx.lead_shape,
            hidden_dtype=ctx.hidden_dtype,
            weight_dtype=ctx.weight_dtype,
            bias_dtype=ctx.bias_dtype,
            vocab_start_index=ctx.vocab_start_index,
            tp_group=ctx.tp_group,
            compute_grad_hidden=ctx.needs_input_grad[0],
            compute_grad_weight=ctx.needs_input_grad[1],
            compute_grad_bias=ctx.needs_input_grad[2],
        )
        return grad_hidden, grad_weight, grad_bias, None, None, None, None


def _sm90_tensor_parallel_linear_logp(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target_ids: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    tp_group: Any,
    vocab_start_index: int,
    global_vocab_size: Optional[int],
) -> torch.Tensor:
    global _SM90_TP_PATH_LOGGED
    if not _SM90_TP_PATH_LOGGED:
        logger.info("Using fused_linear_logp_sm90 tensor-parallel local-shard path.")
        _SM90_TP_PATH_LOGGED = True
    return _FusedTensorParallelLinearLogpSM90Function.apply(
        hidden,
        lm_head_weight,
        bias,
        target_ids,
        int(vocab_start_index),
        None if global_vocab_size is None else int(global_vocab_size),
        tp_group,
    )


class FusedLinearLogpSM90Op:
    """SM90 (Hopper) TMA+WGMMA fused linear log-prob.

    Computes ``log_softmax(hidden @ W^T + b)[target]`` without materializing the
    ``[N, V]`` logits. Requires the extension built with ``KERNEL_ALIGN_FORCE_SM90=1``
    on an SM90 device; bfloat16 hidden/weight only.
    """

    def __init__(self) -> None:
        if not _EXT_AVAILABLE or not hasattr(_C, "fused_linear_logp_sm90"):
            raise RuntimeError(
                "fused_linear_logp_sm90 is not compiled into the extension. Rebuild with "
                "KERNEL_ALIGN_FORCE_SM90=1 on an SM90 (Hopper) device: 'pip install -e .'"
            )
        logger.info("Successfully linked to precompiled _C.fused_linear_logp_sm90 kernel.")

    def __call__(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        *,
        tp_group: Any = None,
        vocab_start_index: int = 0,
        global_vocab_size: Optional[int] = None,
    ) -> torch.Tensor:
        return self.apply(
            hidden,
            lm_head_weight,
            target_ids,
            bias,
            tp_group=tp_group,
            vocab_start_index=vocab_start_index,
            global_vocab_size=global_vocab_size,
        )

    def apply(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        *,
        tp_group: Any = None,
        vocab_start_index: int = 0,
        global_vocab_size: Optional[int] = None,
    ) -> torch.Tensor:
        global _SM90_SAVE_PROBS_BF16_PATH_LOGGED, _SM90_FUSED_TILE_BF16_PATH_LOGGED
        if lm_head_weight.size(-1) != hidden.size(-1):
            raise ValueError(
                f"hidden dim {hidden.size(-1)} must match lm_head_weight dim "
                f"{lm_head_weight.size(-1)}"
            )
        if lm_head_weight.device != hidden.device:
            raise ValueError(
                f"lm_head_weight device {lm_head_weight.device} must match hidden "
                f"device {hidden.device}"
            )
        n_tokens = hidden.numel() // hidden.size(-1)
        if target_ids.numel() != n_tokens:
            raise ValueError(
                f"target_ids must have one id per token: expected {n_tokens}, "
                f"got {target_ids.numel()}"
            )
        if bias is not None:
            if bias.numel() != lm_head_weight.size(0):
                raise ValueError(
                    f"bias must have V={lm_head_weight.size(0)} elements, got {bias.numel()}"
                )
            if bias.device != hidden.device:
                raise ValueError(
                    f"bias device {bias.device} must match hidden device {hidden.device}"
                )
        if should_use_tensor_parallel_linear_logp(
            tp_group,
            int(vocab_start_index),
            global_vocab_size,
            lm_head_weight.size(0),
        ):
            save_probs_tp_available = _sm90_save_probs_bf16_tp_available()
            sm90_supported = _sm90_supported(hidden, lm_head_weight)
            bias_none = bias is None
            hidden_no_grad = not hidden.requires_grad
            hidden_requires_grad = hidden.requires_grad
            weight_requires_grad = lm_head_weight.requires_grad
            any_grad_required = hidden_requires_grad or weight_requires_grad
            output_only_grad = hidden_no_grad and weight_requires_grad
            if save_probs_tp_available and sm90_supported and bias_none and output_only_grad:
                if not _SM90_SAVE_PROBS_BF16_PATH_LOGGED:
                    logger.info(
                        "Using save-probs bf16 output-only tensor-parallel linear_logp fast path."
                    )
                    _SM90_SAVE_PROBS_BF16_PATH_LOGGED = True
                return _TensorParallelLinearLogpSaveProbsBF16Function.apply(
                    hidden,
                    lm_head_weight,
                    target_ids,
                    int(vocab_start_index),
                    None if global_vocab_size is None else int(global_vocab_size),
                    tp_group,
                )
            _log_sm90_save_probs_bf16_skip_once(
                tp_path=True,
                details=_sm90_support_details(hidden, lm_head_weight),
                save_probs_tp_available=save_probs_tp_available,
                sm90_supported=sm90_supported,
                bias_none=bias_none,
                hidden_no_grad=hidden_no_grad,
                hidden_requires_grad=hidden_requires_grad,
                weight_requires_grad=weight_requires_grad,
                any_grad_required=any_grad_required,
                output_only_grad=output_only_grad,
            )
            fused_tile_forward_available = _sm90_fused_tile_bf16_forward_available(tp_path=True)
            if fused_tile_forward_available and sm90_supported and bias_none and any_grad_required:
                if not _SM90_FUSED_TILE_BF16_PATH_LOGGED:
                    logger.info(
                        "Using fused-tile bf16 %stensor-parallel linear_logp fast path.",
                        "output-only " if hidden_no_grad else "full-gradient ",
                    )
                    _SM90_FUSED_TILE_BF16_PATH_LOGGED = True
                return _TensorParallelLinearLogpFusedTileBF16Function.apply(
                    hidden,
                    lm_head_weight,
                    target_ids,
                    int(vocab_start_index),
                    None if global_vocab_size is None else int(global_vocab_size),
                    tp_group,
                )
            if sm90_supported:
                return _sm90_tensor_parallel_linear_logp(
                    hidden,
                    lm_head_weight,
                    target_ids,
                    bias,
                    tp_group=tp_group,
                    vocab_start_index=vocab_start_index,
                    global_vocab_size=global_vocab_size,
                )
            return tensor_parallel_linear_logp(
                hidden,
                lm_head_weight,
                target_ids,
                bias,
                tp_group=tp_group,
                vocab_start_index=vocab_start_index,
                global_vocab_size=global_vocab_size,
            )
        if not _sm90_supported(hidden, lm_head_weight):
            return _fallback_op()(hidden, lm_head_weight, target_ids, bias)
        vocab = lm_head_weight.size(0)
        if bool(((target_ids < 0) | (target_ids >= vocab)).any()):
            t_min, t_max = int(target_ids.min()), int(target_ids.max())
            raise ValueError(
                f"target_ids out of range: expected [0, {vocab - 1}], got [{t_min}, {t_max}]. "
                "Mask or filter padding / ignore-index values (e.g. -100) before this op."
            )
        save_probs_available = _sm90_save_probs_bf16_available()
        bias_none = bias is None
        hidden_no_grad = not hidden.requires_grad
        hidden_requires_grad = hidden.requires_grad
        weight_requires_grad = lm_head_weight.requires_grad
        any_grad_required = hidden_requires_grad or weight_requires_grad
        if save_probs_available and bias_none and any_grad_required:
            if not _SM90_SAVE_PROBS_BF16_PATH_LOGGED:
                logger.info(
                    "Using save-probs bf16 %slinear_logp fast path.",
                    "output-only " if hidden_no_grad else "full-gradient ",
                )
                _SM90_SAVE_PROBS_BF16_PATH_LOGGED = True
            return _LinearLogpSaveProbsBF16Function.apply(hidden, lm_head_weight, target_ids)
        _log_sm90_save_probs_bf16_skip_once(
            tp_path=False,
            details=_sm90_support_details(hidden, lm_head_weight),
            save_probs_available=save_probs_available,
            bias_none=bias_none,
            hidden_no_grad=hidden_no_grad,
            hidden_requires_grad=hidden_requires_grad,
            weight_requires_grad=weight_requires_grad,
            any_grad_required=any_grad_required,
        )
        fused_tile_forward_available = _sm90_fused_tile_bf16_forward_available(tp_path=False)
        if fused_tile_forward_available and bias_none and any_grad_required:
            if not _SM90_FUSED_TILE_BF16_PATH_LOGGED:
                logger.info(
                    "Using fused-tile bf16 %slinear_logp fast path.",
                    "output-only " if hidden_no_grad else "full-gradient ",
                )
                _SM90_FUSED_TILE_BF16_PATH_LOGGED = True
            return _LinearLogpFusedTileBF16Function.apply(hidden, lm_head_weight, target_ids)
        return _FusedLinearLogpSM90Function.apply(hidden, lm_head_weight, bias, target_ids)
