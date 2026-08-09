# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Training-inference mismatch diagnosis, one factor at a time.

Rollout and training compute logprobs for the same tokens with the same weights
and still disagree. This framework turns "which of the dozens of possible causes
is it" into a set of switches that can be flipped one at a time and attributed.

Layout::

    schema/            pure data types, no behaviour
    pipeline/          the seven execution steps, free functions only
    engines/           the two sides under test
    reference_adapters/  wiring reference implementations in
    model_meta/        model shape and module correspondence
    operator_checks/   plugins, one directory per operator (empty by design)

Dependency rules:

1. ``schema/`` never imports ``pipeline/``; within ``schema/`` the imports go one
   way only, with ``values.py`` at the top importing nothing from the project.
2. ``pipeline/`` never imports ``operator_checks/`` -- it only sees what the
   registry hands it. Break this and adding an operator becomes changing the
   framework.
3. Only ``__main__`` imports ``operator_checks/``, to trigger self-registration.
"""

from rl_engine.mismatch import pipeline, schema
