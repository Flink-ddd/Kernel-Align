# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Wiring implementations in as reference implementations. No operator code here.

``kernels/`` and the external libraries own whether the arithmetic is right.
This package owns putting them into a deterministic mode and proving the setting
took effect, which is specific to this framework rather than to any operator.
"""

from rl_engine.mismatch.reference_adapters.settings import (
    SettingDeliveryError,
    apply_required_settings,
    verify_required_settings,
)
