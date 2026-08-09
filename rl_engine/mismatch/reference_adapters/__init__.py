# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Wiring implementations in as reference implementations. **No operator code here.**

"Reference implementation" spans two directories. Of the three
``ReferenceAuthority`` levels, two are implemented in ``rl_engine/kernels/``:

===========================  =====================  ==============================
authority                    implementation lives   what this package does
===========================  =====================  ==============================
``FP64_ORACLE``              ``kernels/``           declare it, call it at the
                                                    lowest noise floor
``SHARED_BACKEND``           external library       put it in deterministic mode,
                                                    read back to verify
``SELF_WRITTEN``             ``kernels/``           wrap it as a
                                                    ``ReferenceImplementation``
===========================  =====================  ==============================

Split of duties: ``kernels/`` owns "is the arithmetic right"; this package owns
"how do we put it in a deterministic mode, and how do we prove the setting
actually took effect". The latter is specific to this diagnostic framework and
has nothing to do with the operator itself.
"""

from rl_engine.mismatch.reference_adapters.settings import (
    SettingDeliveryError,
    apply_required_settings,
    verify_required_settings,
)
