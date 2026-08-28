# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Megatron runtime hooks installed without editing Megatron source files."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import Any

import torch

from rl_engine.integrations.ablation import Implementation, IntegrationPlan
from rl_engine.integrations.framework_operators import (
    MegatronAttentionOperator,
    MegatronFFNOperator,
    MegatronLogpOperator,
)
from rl_engine.integrations.linear_logp import LinearLogpWrapper
from rl_engine.integrations.megatron import MegatronIntegration
from rl_engine.integrations.state import get_active_integration, set_active_integration
from rl_engine.integrations.vime.linear_logp_provider import _provider_impl

_PATCH_MARKER = "__rl_kernel_original_forward__"
_STRICT_ATTENTION_PATCH_MARKER = "__rl_kernel_original_strict_attention_init__"
_STRICT_ATTENTION_PROJECTION_MARKER = "__rl_kernel_strict_attention_projection__"


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
) -> None:
    """Use the shared deterministic GEMM for Megatron Attention projections."""

    if self_attention_cls is None or column_linear_cls is None or row_linear_cls is None:
        from megatron.core.tensor_parallel.layers import (
            ColumnParallelLinear,
            RowParallelLinear,
        )
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

    def deterministic_projection(
        input_value: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        input_2d = input_value.reshape(-1, input_value.shape[-1])
        output_2d = det_gemm(input_2d, weight.t().contiguous())
        output = output_2d.reshape(*input_value.shape[:-1], weight.shape[0])
        return output if bias is None else output + bias

    def attention_init_wrapped(instance: Any, *args: Any, **kwargs: Any) -> None:
        attention_init(instance, *args, **kwargs)
        setattr(instance.linear_qkv, _STRICT_ATTENTION_PROJECTION_MARKER, "qkv")
        setattr(instance.linear_proj, _STRICT_ATTENTION_PROJECTION_MARKER, "o_proj")
        # copy_to_tensor_model_parallel_region already owns this TP dgrad reduction.
        instance.linear_qkv.allreduce_dgrad = False

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

    integration = MegatronIntegration(
        plan,
        rl_kernel_operators={
            "attention": MegatronAttentionOperator(),
            "ffn": MegatronFFNOperator(),
            "logp": MegatronLogpOperator(
                _provider_impl,
                linear_logp=LinearLogpWrapper(),
            ),
        },
    )
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
    if plan.implementation_for("attention", "training") is Implementation.RL_KERNEL:
        _patch_strict_attention_projections()
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
    integration.record_installed_hook("logp", "rl_engine.integrations.vime.linear_logp_provider.provider")
    return integration


def initialize_from_environment(_args: Any = None) -> MegatronIntegration:
    """Vime-compatible custom-init entry point backed by the shared plan env."""

    from rl_engine.integrations.ablation import integration_plan_from_environment

    return install_megatron_integration(integration_plan_from_environment())


__all__ = ["initialize_from_environment", "install_megatron_integration"]
