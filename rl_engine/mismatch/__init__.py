# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Training-inference mismatch diagnosis, one factor at a time.

Rollout and training compute logprobs for the same tokens with the same weights
and still disagree. This turns "which of the dozens of possible causes is it"
into switches that can be flipped one at a time and attributed to a side.

Three dependency rules keep the plugin seam open:

1. ``schema/`` never imports ``pipeline/``, and inside ``schema/`` the imports go
   one way, with ``values.py`` importing nothing from the project.
2. ``pipeline/`` never imports ``operator_checks/``; it sees only what the
   registry hands it. Break this and adding an operator becomes changing the
   framework.
3. Only ``__main__`` imports ``operator_checks/``, to trigger self-registration.

See ``README.md`` for the layout and ``docs/`` for how to add a factor.
"""

from rl_engine.mismatch import pipeline, schema

__all__ = [
    "pipeline",
    "schema",
]
