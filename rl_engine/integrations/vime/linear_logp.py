# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Canonical Vime adapter for RL-Kernel's ``linear_logp`` operator."""

from .logp import ProviderResult, SelectedLogprobProviderUnavailable, provider

__all__ = ["ProviderResult", "SelectedLogprobProviderUnavailable", "provider"]
