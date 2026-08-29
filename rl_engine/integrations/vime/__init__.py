# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Vime adapter entry points without a Vime runtime dependency."""

from .attention import AttentionProviderResult, AttentionProviderUnavailable, attention_provider
from .linear_logp_provider import LinearLogpProviderUnavailable, LinearLogpResult, provider

__all__ = [
    "AttentionProviderResult",
    "AttentionProviderUnavailable",
    "LinearLogpProviderUnavailable",
    "LinearLogpResult",
    "attention_provider",
    "provider",
]
