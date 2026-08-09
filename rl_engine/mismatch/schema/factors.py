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
    """Which family a factor belongs to, and so where to look when it fires."""

    INPUT_IDENTITY = "input_identity"  # tokens / mask / position_ids / eps
    ENVIRONMENT = "environment"  # framework versions, NCCL, determinism switches
    KERNEL_IMPLEMENTATION = "kernel_implementation"  # backend, fusion, inner precision
    SHARDING_AND_REDUCTION = "sharding_and_reduction"  # TP/CP/SP, reduction order, split-K
    OUTPUT_NUMERICS = "output_numerics"  # logits / logp / (out, lse) / gradients


class Evidence(str, Enum):
    """Evidence every factor must have before a verdict is allowed.

    Operator-specific evidence stays out of this enum: putting it here would turn
    "add an operator" into "change the framework". Plugins declare their own as
    plain strings, like the constants below.
    """

    EFFECTIVE_CONFIG_READBACK = "effective_config_readback"  # read back, not requested
    MODEL_STATE_FINGERPRINT = "model_state_fingerprint"
    LIBRARY_VERSIONS = "library_versions"


COLLECTIVE_CONTRACT = "collective_contract"
BATCH_PLACEMENT = "batch_placement"
MODEL_SHAPE = "model_shape"
POSITION_CACHE = "position_cache"
VOCAB_SHARD_MAP = "vocab_shard_map"
LSE_EXPORT = "lse_export"


class ReferenceAuthority(str, Enum):
    """Where a reference implementation comes from, most authoritative first.

    A decision order, not a description: look for a SHARED_BACKEND first, and
    write SELF_WRITTEN only when the first two cannot cover it.
    """

    FP64_ORACLE = "fp64_oracle"  # slow, exact, lowest noise floor only
    SHARED_BACKEND = "shared_backend"  # TransformerEngine / FlashInfer
    SELF_WRITTEN = "self_written"


@dataclass(frozen=True)
class Switch:
    """A switch's one definition: allowed values and parser declared together."""

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
    """What a factor needs in order to run. A whitelist, not a blacklist."""

    required_ops: tuple[str, ...] = ()
    min_gpu_count: int = 1
    required_packages: tuple[str, ...] = ()  # "transformer_engine>=2.0"
    required_model_traits: tuple[str, ...] = ()  # "moe" / "linear_attention"
    blocked_by: tuple[str, ...] = ()  # work this factor waits on


@dataclass(frozen=True)
class ReferenceImplementation:
    """What replaces the native implementation, and which paths it covers.

    ``covers_paths`` defines the shape of the self-check gate: every path this
    reference covers must agree bitwise on the same sequence. Covering two paths
    puts the gate across the two sides; a reference that also covers
    ``ROLLOUT_DECODE`` puts it inside the rollout side, and then no decode stub is
    needed on the training side.

    See ``docs/add-a-kernel-factor.md`` for how that shapes attribution.
    """

    name: str
    tier: ReferenceAuthority
    training_impl: str
    rollout_impl: str
    covers_paths: tuple[ExecutionPath, ...]
    fp64_oracle: str | None = None
    required_settings: tuple[RequiredSetting, ...] = ()
    pinned_libraries: tuple[LibraryPin, ...] = ()


@dataclass(frozen=True)
class MismatchFactor:
    """One suspected cause of training-inference mismatch.

    Not ``DivergenceFactor``: in an RL context, divergence means KL divergence.

    ``reference is None`` makes it a parameter sweep, otherwise an implementation
    swap. There is no separate ``kind`` field -- derivable state is state that can
    disagree with itself.
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


def declared_collectives(factor: MismatchFactor) -> tuple[CollectiveContract, ...]:
    """Collectives a factor's reference pins, for the planner's static check."""

    reference = factor.reference
    if reference is None:
        return ()
    return tuple(
        setting.value
        for setting in reference.required_settings
        if isinstance(setting.value, CollectiveContract)
    )


def requires_fixed_order(contract: CollectiveContract) -> bool:
    """Whether this contract claims its result is independent of topology."""

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
