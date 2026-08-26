# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The two sides under test: engine lifetime and configuration readback.

This package holds ``megatron.py`` and ``vllm.py`` and nothing else. Adding an
operator never adds a file here; all of attention's factors use the one
``vllm.py``.

Anything that merely satisfies ``ScoringBackend`` is a harness, not a side under
test, and lives in ``tests/`` -- see ``tests/mismatch_cpu_backend.py``.
"""

from rl_engine.mismatch.engines.megatron import MegatronBackend
from rl_engine.mismatch.engines.vllm import VllmBackend

__all__ = ["MegatronBackend", "VllmBackend"]
