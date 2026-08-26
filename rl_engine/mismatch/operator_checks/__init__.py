# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Operator plugins, one directory per operator.

Importing this package registers nothing. ``__main__`` imports the individual
operator packages, and that import triggers self-registration -- the framework
does not know which operators exist.

One operator's layout, and the conventions that are enforced::

    operator_checks/attention/
    |-- __init__.py     operator name + discover_factors
    |-- adapter.py      the four operator-level methods
    |-- _common.py      shared reference implementations and contract helpers
    `-- factors/        one file per factor, holding one FACTOR constant

A factor file's name must equal its factor id with the operator prefix stripped;
``discover_factors()`` fails at import otherwise. The four methods stay in
``adapter.py`` because they are operator-level, not factor-level.

Writing one: ``docs/add-a-kernel-factor.md``, or ``docs/add-a-comm-feature.md``
when the suspect is a collective.
"""
