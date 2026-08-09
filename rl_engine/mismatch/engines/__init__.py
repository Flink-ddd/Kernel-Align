# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The two sides under test: process and engine lifetime, configuration readback.

**This package holds exactly two things: ``megatron.py`` and ``vllm.py``.** They
are the training side and the rollout side as they really run -- how the engine
is constructed, how a switch is delivered to it, and how its *effective* value is
read back out.

Shared across operators: all of attention's factors use one ``vllm.py``, and
adding an operator never adds a file here.

Not to be confused with:

* ``reference_adapters/`` -- wires *reference* implementations in and contains no
  operator code;
* ``tests/mismatch_cpu_backend.py`` -- a harness that satisfies the same
  ``ScoringBackend`` protocol so the framework can be exercised without a real
  engine. It is not a side under test, so it does not live here.
"""

from rl_engine.mismatch.engines.megatron import MegatronBackend
from rl_engine.mismatch.engines.vllm import VllmBackend

__all__ = ["MegatronBackend", "VllmBackend"]
