# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The gemm operator plugin."""

from rl_engine.mismatch.operator_checks.gemm import adapter
from rl_engine.mismatch.pipeline import OPERATOR_CHECKS, discover_factors


@OPERATOR_CHECKS.register
class GemmChecks:
    operator = "gemm"

    def declare_factors(self):
        return discover_factors(__package__)

    build_contract = staticmethod(adapter.build_contract)
    read_effective_config = staticmethod(adapter.read_effective_config)
    observe_collectives = staticmethod(adapter.observe_collectives)
    resolve_implementation = staticmethod(adapter.resolve_implementation)
