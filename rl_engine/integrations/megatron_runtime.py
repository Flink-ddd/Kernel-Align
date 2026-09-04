# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Megatron runtime hooks installed without editing Megatron source files."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Iterable
from types import MethodType
from typing import Any

import torch

from rl_engine.integrations.ablation import Implementation, IntegrationPlan
from rl_engine.integrations.framework_operators import (
    MegatronAttentionOperator,
    MegatronFFNOperator,
    MegatronLogpOperator,
    _fused_rms_norm_input,
)
from rl_engine.integrations.linear_logp import LinearLogpWrapper
from rl_engine.integrations.megatron import MegatronIntegration
from rl_engine.integrations.state import get_active_integration, set_active_integration
from rl_engine.integrations.vime.linear_logp_provider import _provider_impl

_PATCH_MARKER = "__rl_kernel_original_forward__"
_STRICT_ATTENTION_PATCH_MARKER = "__rl_kernel_original_strict_attention_init__"
_STRICT_ATTENTION_PROJECTION_MARKER = "__rl_kernel_strict_attention_projection__"
_STRICT_TE_RMS_NORM_PATCH_MARKER = "__rl_kernel_original_strict_rms_norm_forward__"
_STRICT_LOGP_OUTPUT_PATCH_MARKER = "__rl_kernel_original_strict_logp_output_forward__"
_STRICT_LOGP_REUSABLE_MARKER = "__rl_kernel_reusable_local_logits__"


class _DeterministicTensorParallelReduce(torch.autograd.Function):
    """Megatron ``reduce_from_tp`` semantics with the shared fixed tree."""

    @staticmethod
    def forward(ctx: Any, input_value: torch.Tensor, tp_group: Any) -> torch.Tensor:
        del ctx
        from rl_engine.distributed.collectives import collective_for_group

        collective = collective_for_group(
            tp_group,
            min_size_bytes=input_value.numel() * input_value.element_size(),
        )
        if collective is None:
            return input_value
        return collective.all_reduce(input_value.contiguous())

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        del ctx
        # ``reduce_from_tensor_model_parallel_region`` is an all-reduce in the
        # forward pass and identity in the backward pass.
        return grad_output, None


def _deterministic_reduce_from_tensor_model_parallel_region(
    input_value: torch.Tensor,
    tp_group: Any,
) -> torch.Tensor:
    return _DeterministicTensorParallelReduce.apply(input_value, tp_group)


class _DeterministicTPOutputProjection(torch.autograd.Function):
    """Materialize the strict TP LM head once and preserve its dgrad contract."""

    @staticmethod
    def forward(
        ctx: Any,
        input_value: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        tp_group: Any,
    ) -> torch.Tensor:
        from rl_engine.kernels.ops.cuda.matmul.det_gemm import det_gemm_linear

        if input_value.ndim == 3:
            input_2d = input_value.transpose(0, 1).contiguous().reshape(-1, input_value.shape[-1])
            ctx.batch_major = True
        else:
            input_2d = input_value.reshape(-1, input_value.shape[-1]).contiguous()
            ctx.batch_major = False
        weight_2d = weight.contiguous()
        output_2d = det_gemm_linear(input_2d, weight_2d)
        if bias is not None:
            output_2d = (output_2d.float() + bias.float().reshape(1, -1)).to(torch.bfloat16)
        ctx.save_for_backward(input_2d, weight_2d)
        ctx.input_shape = input_value.shape
        ctx.input_dtype = input_value.dtype
        ctx.weight_dtype = weight.dtype
        ctx.bias_dtype = None if bias is None else bias.dtype
        ctx.has_bias = bias is not None
        ctx.tp_group = tp_group
        if ctx.batch_major:
            return (
                output_2d.reshape(input_value.shape[1], input_value.shape[0], weight.size(0))
                .transpose(0, 1)
                .contiguous()
            )
        return output_2d.reshape(*input_value.shape[:-1], weight.size(0))

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        from rl_engine.kernels.ops.cuda.loss.linear_logp import _deterministic_tp_all_reduce_
        from rl_engine.kernels.ops.cuda.matmul.det_gemm import (
            det_gemm_linear_input_gradient,
            det_gemm_linear_weight_gradient,
        )

        input_2d, weight = ctx.saved_tensors
        if ctx.batch_major:
            dlogits = grad_output.transpose(0, 1).contiguous().reshape(-1, grad_output.shape[-1])
        else:
            dlogits = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        if dlogits.dtype != torch.bfloat16:
            raise TypeError("strict TP LM-head backward requires BF16 dlogits")
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = det_gemm_linear_input_gradient(dlogits, weight)
            _deterministic_tp_all_reduce_(grad_input, ctx.tp_group)
            if ctx.batch_major:
                grad_input = (
                    grad_input.reshape(ctx.input_shape[1], ctx.input_shape[0], ctx.input_shape[2])
                    .transpose(0, 1)
                    .contiguous()
                )
            else:
                grad_input = grad_input.reshape(ctx.input_shape)
            grad_input = grad_input.to(ctx.input_dtype)
        if ctx.needs_input_grad[1]:
            grad_weight = det_gemm_linear_weight_gradient(input_2d, dlogits).to(ctx.weight_dtype)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = dlogits.float().sum(dim=0).to(ctx.bias_dtype)
        return grad_input, grad_weight, grad_bias, None


def _optional_class(path: str) -> type[Any] | None:
    module_name, _, class_name = path.rpartition(".")
    try:
        value = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError):
        return None
    return value if isinstance(value, type) else None


def _discover_attention_classes() -> tuple[type[Any], ...]:
    paths = (
        "megatron.core.transformer.dot_product_attention.DotProductAttention",
        "megatron.core.extensions.transformer_engine.TEDotProductAttention",
        "megatron.core.extensions.transformer_engine.TEDotProductAttentionWithCP",
    )
    return tuple(value for path in paths if (value := _optional_class(path)) is not None)


def _discover_ffn_classes() -> tuple[type[Any], ...]:
    paths = ("megatron.core.transformer.mlp.MLP",)
    return tuple(value for path in paths if (value := _optional_class(path)) is not None)


def _unique_classes(values: Iterable[type[Any]]) -> tuple[type[Any], ...]:
    return tuple(dict.fromkeys(values))


def _patch_forward(
    cls: type[Any],
    *,
    integration: MegatronIntegration,
    module: str,
) -> None:
    if hasattr(cls, _PATCH_MARKER):
        raise RuntimeError(f"{cls.__module__}.{cls.__name__} is already RL-Kernel patched")
    original = cls.forward

    def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
        def native(_module: Any, *call_args: Any, **call_kwargs: Any) -> Any:
            return original(instance, *call_args, **call_kwargs)

        return integration.execute(module, native, instance, *args, **kwargs)

    wrapped.__name__ = getattr(original, "__name__", "forward")
    wrapped.__doc__ = getattr(original, "__doc__", None)
    setattr(cls, _PATCH_MARKER, original)
    cls.forward = wrapped


def _patch_strict_attention_projections(
    *,
    self_attention_cls: type[Any] | None = None,
    column_linear_cls: type[Any] | None = None,
    row_linear_cls: type[Any] | None = None,
    det_gemm: Any | None = None,
    copy_to_tp: Callable[[torch.Tensor], torch.Tensor] | None = None,
    reduce_from_tp: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> None:
    """Use the shared deterministic GEMM for Megatron Attention projections."""

    if self_attention_cls is None or column_linear_cls is None or row_linear_cls is None:
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
        from megatron.core.transformer.attention import SelfAttention

        self_attention_cls = SelfAttention
        column_linear_cls = ColumnParallelLinear
        row_linear_cls = RowParallelLinear
    if hasattr(self_attention_cls, _STRICT_ATTENTION_PATCH_MARKER):
        return
    if det_gemm is None:
        from rl_engine.kernels.ops.cuda.matmul.det_gemm import DetGemmOp

        det_gemm = DetGemmOp()

    attention_init = self_attention_cls.__init__
    column_forward_impl = column_linear_cls._forward_impl
    row_forward_impl = row_linear_cls._forward_impl

    def tp_mappings() -> tuple[
        Callable[[torch.Tensor], torch.Tensor],
        Callable[[torch.Tensor], torch.Tensor],
    ]:
        if copy_to_tp is not None and reduce_from_tp is not None:
            return copy_to_tp, reduce_from_tp
        from megatron.core.tensor_parallel.mappings import (
            copy_to_tensor_model_parallel_region,
            reduce_from_tensor_model_parallel_region,
        )

        return (
            copy_to_tensor_model_parallel_region,
            reduce_from_tensor_model_parallel_region,
        )

    def strict_tp_reduce(input_value: torch.Tensor) -> torch.Tensor:
        if reduce_from_tp is not None:
            return reduce_from_tp(input_value)
        from megatron.core import parallel_state

        tp_group = parallel_state.get_tensor_model_parallel_group()
        return _deterministic_reduce_from_tensor_model_parallel_region(
            input_value,
            tp_group,
        )

    def deterministic_projection(
        input_value: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        input_2d = input_value.reshape(-1, input_value.shape[-1])
        linear = getattr(det_gemm, "linear", None)
        output_2d = (
            linear(input_2d, weight)
            if linear is not None
            else det_gemm(input_2d, weight.t().contiguous())
        )
        output = output_2d.reshape(*input_value.shape[:-1], weight.shape[0])
        return output if bias is None else output + bias

    def attention_init_wrapped(instance: Any, *args: Any, **kwargs: Any) -> None:
        attention_init(instance, *args, **kwargs)
        qkv = instance.linear_qkv
        projection = instance.linear_proj
        setattr(qkv, _STRICT_ATTENTION_PROJECTION_MARKER, "qkv")
        setattr(projection, _STRICT_ATTENTION_PROJECTION_MARKER, "o_proj")
        if hasattr(qkv, "layer_norm_weight"):
            tp_copy, _native_tp_reduce = tp_mappings()

            def te_qkv_forward(module: Any, input_value: torch.Tensor) -> Any:
                normalized = _fused_rms_norm_input(module, input_value, "linear_qkv")
                normalized = tp_copy(normalized)
                return deterministic_projection(normalized, module.weight, None), None

            def te_projection_forward(module: Any, input_value: torch.Tensor) -> Any:
                output = deterministic_projection(input_value, module.weight, None)
                return strict_tp_reduce(output), None

            qkv.forward = MethodType(te_qkv_forward, qkv)
            projection.forward = MethodType(te_projection_forward, projection)
        else:
            # The local ColumnParallelLinear wrapper already owns TP dgrad.
            qkv.allreduce_dgrad = False

    def column_forward_impl_wrapped(
        instance: Any,
        input: torch.Tensor,
        weight: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if hasattr(instance, _STRICT_ATTENTION_PROJECTION_MARKER):
            return deterministic_projection(input, weight, kwargs.get("bias"))
        return column_forward_impl(instance, input, weight, *args, **kwargs)

    def row_forward_impl_wrapped(
        instance: Any,
        input: torch.Tensor,
        weight: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if hasattr(instance, _STRICT_ATTENTION_PROJECTION_MARKER):
            return deterministic_projection(input, weight, kwargs.get("bias"))
        return row_forward_impl(instance, input, weight, *args, **kwargs)

    setattr(self_attention_cls, _STRICT_ATTENTION_PATCH_MARKER, attention_init)
    self_attention_cls.__init__ = attention_init_wrapped
    column_linear_cls._forward_impl = column_forward_impl_wrapped
    row_linear_cls._forward_impl = row_forward_impl_wrapped


def _patch_strict_te_rms_norm(rms_norm_cls: type[Any] | None = None) -> None:
    """Match standalone TE RMSNorm modules to the strict rollout arithmetic."""

    if rms_norm_cls is None:
        try:
            from transformer_engine.pytorch import RMSNorm
        except ImportError:
            return

        rms_norm_cls = RMSNorm
    if hasattr(rms_norm_cls, _STRICT_TE_RMS_NORM_PATCH_MARKER):
        return
    original = rms_norm_cls.forward

    def wrapped(instance: Any, input_value: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if args or kwargs:
            raise RuntimeError("strict TE RMSNorm does not accept extra forward arguments")
        if bool(getattr(instance, "zero_centered_gamma", False)):
            raise RuntimeError("strict TE RMSNorm does not support zero-centered gamma")
        return torch.nn.functional.rms_norm(
            input_value,
            (input_value.shape[-1],),
            instance.weight,
            float(instance.eps),
        )

    setattr(rms_norm_cls, _STRICT_TE_RMS_NORM_PATCH_MARKER, original)
    rms_norm_cls.forward = wrapped


def _patch_strict_logp_output_layer(
    linear_cross_entropy_cls: type[Any] | None = None,
) -> None:
    """Replace Megatron's duplicate LM-head path with one reusable strict result."""

    if linear_cross_entropy_cls is None:
        from megatron.core.transformer.linear_cross_entropy import LinearCrossEntropyModule

        linear_cross_entropy_cls = LinearCrossEntropyModule
    if hasattr(linear_cross_entropy_cls, _STRICT_LOGP_OUTPUT_PATCH_MARKER):
        return
    original = linear_cross_entropy_cls.forward

    def wrapped(
        instance: Any,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
        output_cross_entropy_loss: bool = False,
        labels: torch.Tensor | None = None,
        reduction: str = "none",
        ignore_index: int = -100,
    ) -> Any:
        if output_cross_entropy_loss:
            return original(
                instance,
                input_,
                weight,
                runtime_gather_output,
                output_cross_entropy_loss,
                labels,
                reduction,
                ignore_index,
            )
        output_weight = instance.weight if weight is None else weight
        if output_weight is None:
            raise RuntimeError("strict TP LM head requires an explicit weight")
        gather_output = (
            instance.gather_output if runtime_gather_output is None else bool(runtime_gather_output)
        )
        if gather_output:
            raise RuntimeError("strict reusable TP LM head does not support gathered logits")
        if bool(getattr(instance, "sequence_parallel", False)):
            raise RuntimeError("strict reusable TP LM head does not support sequence parallelism")
        if bool(getattr(instance, "explicit_expert_comm", False)) or bool(
            getattr(instance, "disable_grad_reduce", False)
        ):
            raise RuntimeError("strict reusable TP LM head requires ordinary TP dgrad reduction")
        bias = instance.bias if not instance.skip_bias_add else None
        output = _DeterministicTPOutputProjection.apply(
            input_, output_weight, bias, instance.tp_group
        )
        instance._rl_kernel_local_logits = output
        output_bias = instance.bias if instance.skip_bias_add else None
        return output, output_bias

    wrapped.__name__ = getattr(original, "__name__", "forward")
    wrapped.__doc__ = getattr(original, "__doc__", None)
    setattr(linear_cross_entropy_cls, _STRICT_LOGP_OUTPUT_PATCH_MARKER, original)
    setattr(linear_cross_entropy_cls, _STRICT_LOGP_REUSABLE_MARKER, True)
    linear_cross_entropy_cls.forward = wrapped


def install_megatron_integration(
    plan: IntegrationPlan,
    *,
    attention_classes: Iterable[type[Any]] | None = None,
    ffn_classes: Iterable[type[Any]] | None = None,
) -> MegatronIntegration:
    """Install Attention, dense FFN and structural Logp routes in one actor."""

    existing = get_active_integration("megatron")
    if existing is not None:
        if not isinstance(existing, MegatronIntegration):
            raise RuntimeError("active Megatron integration has an unexpected type")
        if existing.plan != plan:
            raise RuntimeError("Megatron integration is already installed with another plan")
        return existing

    rl_kernel_operators: dict[str, Any] = {}
    if plan.implementation_for("attention", "training") is Implementation.RL_KERNEL:
        rl_kernel_operators["attention"] = MegatronAttentionOperator()
    if plan.implementation_for("ffn", "training") is Implementation.RL_KERNEL:
        rl_kernel_operators["ffn"] = MegatronFFNOperator()
    if plan.implementation_for("logp", "training") is Implementation.RL_KERNEL:
        rl_kernel_operators["logp"] = MegatronLogpOperator(
            _provider_impl,
            linear_logp=LinearLogpWrapper(),
        )
    integration = MegatronIntegration(plan, rl_kernel_operators=rl_kernel_operators)
    resolved_attention = _unique_classes(
        _discover_attention_classes() if attention_classes is None else attention_classes
    )
    resolved_ffn = _unique_classes(_discover_ffn_classes() if ffn_classes is None else ffn_classes)
    if (
        plan.implementation_for("attention", "training") is Implementation.RL_KERNEL
        and not resolved_attention
    ):
        raise RuntimeError("R/R Megatron Attention selected but no supported class was found")
    if plan.implementation_for("ffn", "training") is Implementation.RL_KERNEL and not resolved_ffn:
        raise RuntimeError("R/R Megatron FFN selected but no supported class was found")

    set_active_integration("megatron", integration)
    if plan.implementation_for("logp", "training") is Implementation.RL_KERNEL:
        _patch_strict_logp_output_layer()
    if plan.implementation_for("attention", "training") is Implementation.RL_KERNEL:
        _patch_strict_attention_projections()
        if plan.implementation_for("ffn", "training") is Implementation.RL_KERNEL:
            _patch_strict_te_rms_norm()
    for cls in resolved_attention:
        _patch_forward(cls, integration=integration, module="attention")
    if resolved_attention:
        integration.record_installed_hook(
            "attention",
            ",".join(f"{cls.__module__}.{cls.__name__}.forward" for cls in resolved_attention),
        )
    for cls in resolved_ffn:
        _patch_forward(cls, integration=integration, module="ffn")
    if resolved_ffn:
        integration.record_installed_hook(
            "ffn",
            ",".join(f"{cls.__module__}.{cls.__name__}.forward" for cls in resolved_ffn),
        )
    if plan.implementation_for("logp", "training") is Implementation.RL_KERNEL:
        integration.record_installed_hook(
            "logp",
            "rl_engine.integrations.vime.linear_logp_provider.provider,"
            "megatron.core.transformer.linear_cross_entropy.LinearCrossEntropyModule.forward",
        )
    return integration


def initialize_from_environment(_args: Any = None) -> MegatronIntegration:
    """Vime-compatible custom-init entry point backed by the shared plan env."""

    from rl_engine.integrations.ablation import integration_plan_from_environment

    plan = integration_plan_from_environment()
    integration = install_megatron_integration(plan)
    if plan.implementation_for("attention", "training") is Implementation.RL_KERNEL:
        _precompile_strict_attention_training(_args)
    return integration


def _precompile_strict_attention_training(args: Any) -> None:
    """Warm FA4 CuTe fwd/bwd JIT outside Vime's actor_train timer."""

    if args is None or torch.version.hip is not None or not torch.cuda.is_available():
        return
    if os.getenv("RL_KERNEL_PRECOMPILE_FA4", "1") == "0":
        return

    from rl_engine.kernels.ops.cuda.attention.flash_attn import StrictFlashAttention4Core

    attention_heads = int(getattr(args, "num_attention_heads", 0) or 0)
    query_groups = int(getattr(args, "num_query_groups", 0) or attention_heads)
    tp_size = int(getattr(args, "tensor_model_parallel_size", 1) or 1)
    head_dim = int(
        getattr(args, "kv_channels", 0)
        or (int(getattr(args, "hidden_size", 0) or 0) // attention_heads)
    )
    if (
        attention_heads <= 0
        or query_groups <= 0
        or tp_size <= 0
        or attention_heads % tp_size
        or query_groups % tp_size
        or head_dim <= 0
    ):
        return

    params_dtype = getattr(args, "params_dtype", None)
    dtype = params_dtype if params_dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    StrictFlashAttention4Core.precompile_training(
        q_heads=attention_heads // tp_size,
        kv_heads=query_groups // tp_size,
        head_dim=head_dim,
        dtype=dtype,
    )


__all__ = ["initialize_from_environment", "install_megatron_integration"]
