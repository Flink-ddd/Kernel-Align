# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""ROCm public surface for the shared deterministic Attention core."""

from rl_engine.kernels.ops.cuda.attention.deterministic_attn import (
    DeterministicAttentionCoreResult,
    DeterministicAttentionOp,
    RLKernelDeterministicAttentionCore,
)

__all__ = [
    "DeterministicAttentionCoreResult",
    "DeterministicAttentionOp",
    "RLKernelDeterministicAttentionCore",
]
