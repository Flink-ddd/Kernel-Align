# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The two sides under test: process and engine lifetime, configuration readback.

Shared across operators -- all of attention's factors use one ``vllm.py``. Not to
be confused with ``reference_adapters/``, which wires in *reference*
implementations and contains no operator code.
"""

from rl_engine.mismatch.engines.cpu_reference import CpuScoringBackend
