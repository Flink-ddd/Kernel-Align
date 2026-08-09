# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Operator plugins, one directory per operator. **Empty by design.**

The framework ships without operators. Each operator is claimed and written
separately, and adding one changes nothing outside its own directory.

Importing this package registers nothing on its own -- ``__main__`` imports the
individual operator packages, and that import is what triggers
self-registration. The framework does not know which operators exist; the CLI
brings them in.

Layout of one operator
----------------------

An operator has more than a dozen factors, each with eight or so fields. Putting
them all in one file runs to 800 lines that nobody wants to edit, so it is **one
file per factor**, mirroring "one directory per operator"::

    operator_checks/attention/
    |-- __init__.py     ~15 lines: operator name + discover_factors, rest delegated
    |-- adapter.py      the four operator-level methods
    |-- _common.py      shared across factors: contract helpers, common
    |                   comparison rules, this operator's reference implementations
    `-- factors/        one file per factor, 30-50 lines each
        |-- provenance.py               attn.provenance
        |-- execution_path.py           attn.execution_path
        `-- ...

**File name equals the factor id with the operator prefix stripped.**
``discover_factors()`` enforces it, so renaming an id without renaming the file
fails at import rather than silently.

What a factor file contains
---------------------------

One ``FACTOR`` constant and nothing else -- declarations, no behaviour::

    FACTOR = MismatchFactor(
        id="attn.rope_fusion",
        operator="attention",
        category=FactorCategory.KERNEL_IMPLEMENTATION,
        question="...",
        switch=Switch(...),
        comparison_rules={...},
        prerequisites=Prerequisites(...),
        reference=TE_ROPE_REFERENCE,        # from _common.py
        pitfalls=(KnownPitfall(...),),
        required_evidence=(Evidence.EFFECTIVE_CONFIG_READBACK.value, POSITION_CACHE),
    )

What ``__init__.py`` contains
-----------------------------

::

    @OPERATOR_CHECKS.register
    class AttentionChecks:
        operator = "attention"

        def declare_factors(self):
            return discover_factors(__package__)   # scans factors/*.py

        build_contract         = adapter.build_contract
        read_effective_config  = adapter.read_effective_config
        observe_collectives    = adapter.observe_collectives
        resolve_implementation = adapter.resolve_implementation

Adding a factor means dropping a file into ``factors/``; ``__init__.py`` does not
change.

Why the four methods stay in ``adapter.py``
-------------------------------------------

They are **operator-level**, not factor-level: how to read configuration back
from an engine and how to observe collectives is one piece of logic shared by all
of that operator's factors. Pushing them down would make every factor file copy
them.
"""
