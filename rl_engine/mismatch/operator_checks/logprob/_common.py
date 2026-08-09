# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Values shared by logprob's factors.

No reference implementation yet: the rollout side has no deterministic
vocab-parallel reduction to swap in, so these factors are parameter sweeps.
"""

from __future__ import annotations

from rl_engine.mismatch.schema import DowncastPoint, Precision

# vLLM computes logits at the model dtype; Megatron can keep the head in fp32.
HEAD_DTYPES: dict[str, Precision] = {
    "bf16": Precision.BF16,
    "fp32": Precision.FP32,
}

DOWNCAST_POINTS: dict[str, DowncastPoint] = {
    "final_write": DowncastPoint.FINAL_WRITE,
    "per_partial": DowncastPoint.PER_PARTIAL,
}
