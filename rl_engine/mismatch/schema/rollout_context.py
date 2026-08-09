# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Online RL context.

The factor model targets **offline pairwise comparison**; real training is an
**online RL loop**. Offline these are placeholders and a factor author never
touches them. They matter only when this framework is wired into a real training
loop for online auditing -- and there, missing them causes two kinds of problem:

**One, results stop reproducing.** You get ``dlogp = 0.005`` today and ``0.008``
tomorrow from the same config, and eventually find the batch order changed and
one sample landed in a different microbatch -- **the group it all-reduces with
changed, so the accumulation order changed**. That is what ``BatchPlacement``
records: not "environment" but part of the **identity**. Changing where a sample
lands is changing the computation. ``DynamicSamplingDecision`` is the same story:
GRPO drops all-correct or all-wrong groups, and which ones were dropped changes
the whole batch composition.

**Two, metrics get misread.** GRPO normalises advantage within a group:
``A_i = (r_i - mean(r)) / (std(r) + eps)``. So **one rollout's logp drifting
affects all K rollouts in that group through the advantage** -- the error is not
independent per rollout, it amplifies within the group. Averaging over tokens
erases this completely: ``dlogp_mean`` looks fine while a few groups are skewed
wholesale. Hence metrics must be aggregatable by ``RolloutGroup``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RolloutGroup:
    """One GRPO group: K rollouts from the same prompt.

    Two modes, depending on who is running:

    * **offline ablation** (the main use): identity is built from a fixed token
      fixture and this is a placeholder (1 prompt x 1 rollout); group semantics
      are unused;
    * **online, wired into real training**: a real group comes from the training
      loop, and only then does aggregating by group mean anything.
    """

    prompt_id: str
    rollout_ids: tuple[str, ...]
    group_size: int  # GRPO's K


@dataclass(frozen=True)
class BatchPlacement:
    """Where one sample landed in this training step.

    Not ``BatchLayout`` -- in tensor land, layout means memory ordering (NCHW and
    friends); this is about placement.
    """

    data_parallel_rank: int
    microbatch_index: int
    position_in_microbatch: int
    dropped_by_schedule: bool = False


@dataclass(frozen=True)
class DynamicSamplingDecision:
    """Record of dropping a group (all-correct or all-wrong groups carry no
    learning signal).

    Dropping changes the batch composition and therefore the reduction grouping
    -- unrecorded, it cannot be reproduced.
    """

    kept: bool
    reason: str | None = None


@dataclass(frozen=True)
class ComparisonIdentity:
    """This comparison's input identity. If these differ, no numerical
    comparison means anything.

    Not ``ScoringIdentity`` -- in RL, "score" means reward scoring.
    """

    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    active_mask: tuple[bool, ...]  # loss mask: which tokens participate
    position_ids: tuple[int, ...]
    checkpoint_id: str
    checkpoint_revision: str
    model_shape: str  # "L=1,H=896,Hq=14,Hkv=2,D=64" -- a trimmed model is a different model
    group: RolloutGroup
    batch_placement: BatchPlacement
    sampling_decision: DynamicSamplingDecision


__all__ = [
    "BatchPlacement",
    "ComparisonIdentity",
    "DynamicSamplingDecision",
    "RolloutGroup",
]
