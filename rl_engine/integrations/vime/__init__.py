# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Vime adapter entry points without a Vime runtime dependency."""

from .logp import ProviderResult, SelectedLogprobProviderUnavailable, provider

__all__ = ["ProviderResult", "SelectedLogprobProviderUnavailable", "provider"]

