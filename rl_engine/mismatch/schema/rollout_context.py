# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The online RL context around one comparison.

All of these belong to ``INPUT_IDENTITY`` rather than to the environment: they
change how samples are grouped for reduction, so two runs that differ here are
not reproducible against each other.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RolloutGroup:
    """One GRPO group: K rollouts from the same prompt.

    Advantage is normalised within the group, so one rollout's logp deviating
    reaches all K through it. Averaging over tokens alone hides that. Offline
    ablation leaves this a placeholder; it means something only when wired into a
    real training loop.
    """

    prompt_id: str
    rollout_ids: tuple[str, ...]
    group_size: int  # GRPO's K


@dataclass(frozen=True)
class BatchPlacement:
    """Where one sample landed in this training step.

    Not ``BatchLayout``: in tensor land, layout means memory ordering. DP and
    microbatch splitting decide which samples reduce together, so moving a sample
    changes its accumulation order.
    """

    data_parallel_rank: int
    microbatch_index: int
    position_in_microbatch: int
    dropped_by_schedule: bool = False


@dataclass(frozen=True)
class DynamicSamplingDecision:
    """Record of dropping a group, which changes the batch composition."""

    kept: bool
    reason: str | None = None


@dataclass(frozen=True)
class ComparisonIdentity:
    """This comparison's input identity.

    If these differ, no numerical comparison means anything.
    """

    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    active_mask: tuple[bool, ...]  # loss mask: which tokens participate
    position_ids: tuple[int, ...]
    checkpoint_id: str
    checkpoint_revision: str
    model_shape: str  # a trimmed model is a different model
    group: RolloutGroup
    batch_placement: BatchPlacement
    sampling_decision: DynamicSamplingDecision


__all__ = [
    "BatchPlacement",
    "ComparisonIdentity",
    "DynamicSamplingDecision",
    "RolloutGroup",
]
