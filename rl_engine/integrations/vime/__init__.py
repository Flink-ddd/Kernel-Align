# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Vime adapter entry points without a Vime runtime dependency."""

from .attention import AttentionProviderResult, AttentionProviderUnavailable, attention_provider
from .logp import ProviderResult, SelectedLogprobProviderUnavailable, provider

__all__ = [
    "AttentionProviderResult",
    "AttentionProviderUnavailable",
    "ProviderResult",
    "SelectedLogprobProviderUnavailable",
    "attention_provider",
    "provider",
]
