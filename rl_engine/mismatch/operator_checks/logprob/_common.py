# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Shared across the logprob factors.

No ``ReferenceImplementation`` here yet: the rollout side has no deterministic
vocab-parallel reduction to swap in, so today's logprob factors are parameter
sweeps. When the self-written reference (kernel (4) of the design doc) lands,
it goes here and the factors that use it become four-arm swaps without any
other file changing.
"""

from __future__ import annotations

from rl_engine.mismatch.schema import DowncastPoint, Precision

# vLLM computes logits at the model dtype and only upcasts for the softmax;
# Megatron can be told to keep the head in fp32. That difference is the switch.
HEAD_DTYPES: dict[str, Precision] = {
    "bf16": Precision.BF16,
    "fp32": Precision.FP32,
}

# Where the fp32 accumulator is written back to a lower precision.
DOWNCAST_POINTS: dict[str, DowncastPoint] = {
    "final_write": DowncastPoint.FINAL_WRITE,
    "per_partial": DowncastPoint.PER_PARTIAL,
}
