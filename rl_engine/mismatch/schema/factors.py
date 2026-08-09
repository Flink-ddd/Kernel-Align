# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The factor model: one suspected cause of training-inference mismatch."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping

from rl_engine.mismatch.schema.collectives import CollectiveContract, DeterminismLevel
from rl_engine.mismatch.schema.contracts import ComparisonRule
from rl_engine.mismatch.schema.pitfalls import KnownPitfall
from rl_engine.mismatch.schema.values import (
    ExecutionPath,
    LibraryPin,
    PolicyRole,
    RebindCost,
    RequiredSetting,
    choice_parser,
)


class FactorCategory(str, Enum):
    """Which family a factor belongs to -- decides where to look when it fires.

    Deliberately not ``Layer``: in this domain a layer is a model layer.
    """

    INPUT_IDENTITY = "input_identity"  # tokens / mask / position_ids / eps
    ENVIRONMENT = "environment"  # framework versions, NCCL, determinism switches
    KERNEL_IMPLEMENTATION = "kernel_implementation"  # backend choice, fusion, inner precision
    SHARDING_AND_REDUCTION = "sharding_and_reduction"  # TP/CP/SP, reduction order, split-K
    OUTPUT_NUMERICS = "output_numerics"  # logits / logp / (out, lse) / gradients


class Evidence(str, Enum):
    """Evidence **every** factor must have before a verdict is allowed.

    General items only. Operator-specific evidence does not belong in this enum:
    putting it here would turn "add an operator" into "change the framework".
    Plugins declare their own as plain strings (see the constants below).
    """

    EFFECTIVE_CONFIG_READBACK = "effective_config_readback"  # read back, not requested
    MODEL_STATE_FINGERPRINT = "model_state_fingerprint"  # all variants share one set of weights
    LIBRARY_VERSIONS = "library_versions"


# Operator-specific evidence: plugins define constants in their own module and
# the framework treats them as plain strings. Adding an operator adds a new set
# of constants; the framework does not change.
COLLECTIVE_CONTRACT = "collective_contract"
BATCH_PLACEMENT = "batch_placement"
MODEL_SHAPE = "model_shape"
POSITION_CACHE = "position_cache"
VOCAB_SHARD_MAP = "vocab_shard_map"
LSE_EXPORT = "lse_export"


class ReferenceAuthority(str, Enum):
    """Where a reference implementation comes from, most authoritative first.

    Not ``ReferenceTier``: "tier" says there are levels without saying what
    orders them. What orders these is authority.

    This is a decision order, not a description: look for a SHARED_BACKEND
    first, and only write SELF_WRITTEN when the first two cannot cover it.
    """

    FP64_ORACLE = "fp64_oracle"  # slow, mathematically exact, lowest noise floor only
    SHARED_BACKEND = "shared_backend"  # TransformerEngine / FlashInfer
    SELF_WRITTEN = "self_written"


@dataclass(frozen=True)
class Switch:
    """The one definition of a switch. Allowed values and parser declared once."""

    path: str  # "gemm.forward_reduce"
    rebind_cost: RebindCost
    applies_to: tuple[PolicyRole, ...]
    allowed_values: tuple[Any, ...] | None = None
    parse: Callable[[Any], Any] | None = None

    def __post_init__(self) -> None:
        if self.parse is None and self.allowed_values is not None:
            object.__setattr__(self, "parse", choice_parser(*self.allowed_values))


@dataclass(frozen=True)
class Prerequisites:
    """What this factor needs in order to run. A whitelist, not a blacklist.

    Capability probing itself is delegated: this only declares what is needed.
    """

    required_ops: tuple[str, ...] = ()
    min_gpu_count: int = 1
    required_packages: tuple[str, ...] = ()  # "transformer_engine>=2.0"
    required_model_traits: tuple[str, ...] = ()  # "moe" / "linear_attention"
    blocked_by: tuple[str, ...] = ()  # work this factor waits on


@dataclass(frozen=True)
class ReferenceImplementation:
    """What replaces the native implementation, and which execution paths it covers.

    **``covers_paths`` defines the shape of the self-check gate**, the field most
    easily overlooked: the gate is not "both sides use the same implementation",
    it is "**every path this reference covers must agree bitwise on the same
    sequence**".

    Most factors cover two paths (training full-prefill plus rollout
    full-prefill) and the gate holds across the two sides. But an attention
    FlashInfer reference also covers ROLLOUT_DECODE -- so the gate holds *inside
    the rollout side*: with full-prefill and decode both switched to FlashInfer
    and ``num_splits=1`` pinned, ``(out, lse)`` must agree on the same sequence.
    A disagreement is the reference's fault, unrelated to the training side.
    **No decode stub is needed on the training side.**

    Covering several paths also gives the attribution split directly:

        dlogp(training, real rollout)
          = dlogp(TRAINING_FULL_PREFILL, ROLLOUT_FULL_PREFILL)   cross-framework
          + dlogp(ROLLOUT_FULL_PREFILL,  ROLLOUT_DECODE)         rollout-side path

    The two terms are fixed in completely different ways: the first by swapping
    the kernel backend, the second by changing decode's softmax block size to
    match prefill. Without the middle path, attribution stops at the combined
    "cross-framework plus cross-path" answer.
    """

    name: str
    tier: ReferenceAuthority
    training_impl: str
    rollout_impl: str
    covers_paths: tuple[ExecutionPath, ...]
    fp64_oracle: str | None = None
    required_settings: tuple[RequiredSetting, ...] = ()
    pinned_libraries: tuple[LibraryPin, ...] = ()  # **required**, see LibraryPin


@dataclass(frozen=True)
class MismatchFactor:
    """One suspected cause of training-inference mismatch.

    Not ``DivergenceFactor``: in an RL context, divergence means KL divergence.

    A factor with ``reference is None`` is a parameter sweep; with a reference it
    is an implementation swap. There is no separate ``kind`` field -- derivable
    state is state that can disagree with itself.
    """

    id: str  # "gemm.forward_reduce", globally unique
    operator: str
    category: FactorCategory
    question: str  # what this factor answers, one line, goes into the docs
    switch: Switch
    comparison_rules: Mapping[str, ComparisonRule]  # contract field path -> rule
    prerequisites: Prerequisites
    required_evidence: tuple[str, ...] = ()  # Evidence values or plugin constants
    reference: ReferenceImplementation | None = None
    call_sites: tuple[str, ...] = ()  # one factor acting in several places
    pitfalls: tuple[KnownPitfall, ...] = ()
    variants: tuple[Any, ...] = ()  # empty -> expand the standard set
    owner: str = ""  # "" = unclaimed
    tracked_by: tuple[str, ...] = ()  # issues or PRs following this factor


def declared_collectives(factor: MismatchFactor) -> tuple[CollectiveContract, ...]:
    """Collectives a factor's reference implementation pins, if any.

    Used by the planner's static check; returns empty when the factor does not
    touch collective communication.
    """

    reference = factor.reference
    if reference is None:
        return ()
    pinned = tuple(
        setting.value
        for setting in reference.required_settings
        if isinstance(setting.value, CollectiveContract)
    )
    return pinned


def requires_fixed_order(contract: CollectiveContract) -> bool:
    """Whether this contract claims its result is independent of topology.

    Claiming it means it must survive the assertion in ``pipeline/planner.py``.
    """

    return contract.determinism is DeterminismLevel.STABLE_ACROSS_TOPOLOGY


__all__ = [
    "BATCH_PLACEMENT",
    "COLLECTIVE_CONTRACT",
    "Evidence",
    "FactorCategory",
    "LSE_EXPORT",
    "MODEL_SHAPE",
    "MismatchFactor",
    "POSITION_CACHE",
    "Prerequisites",
    "ReferenceAuthority",
    "ReferenceImplementation",
    "Switch",
    "VOCAB_SHARD_MAP",
    "declared_collectives",
    "requires_fixed_order",
]
