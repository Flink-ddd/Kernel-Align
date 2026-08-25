# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Megatron runtime hooks installed without editing Megatron source files."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import Any

from rl_engine.integrations.ablation import Implementation, IntegrationPlan
from rl_engine.integrations.framework_operators import (
    MegatronAttentionOperator,
    MegatronFFNOperator,
    MegatronLogpOperator,
)
from rl_engine.integrations.linear_logp import LinearLogpWrapper
from rl_engine.integrations.megatron import MegatronIntegration
from rl_engine.integrations.state import get_active_integration, set_active_integration
from rl_engine.integrations.vime.logp import _provider_impl

_PATCH_MARKER = "__rl_kernel_original_forward__"


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
    integration.record_installed_hook("logp", "rl_engine.integrations.vime.logp.provider")
    return integration


def initialize_from_environment(_args: Any = None) -> MegatronIntegration:
    """Vime-compatible custom-init entry point backed by the shared plan env."""

    from rl_engine.integrations.ablation import integration_plan_from_environment

    return install_megatron_integration(integration_plan_from_environment())


__all__ = ["initialize_from_environment", "install_megatron_integration"]
