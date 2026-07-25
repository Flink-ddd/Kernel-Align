# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Versioned alignment standard exported for downstream framework adapters."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from rl_engine.alignment.cross_config.comparison import compare_score_artifacts
from rl_engine.alignment.cross_config.schema import (
    RuntimeProvenance,
    ScoreArtifact,
    ScoreSide,
    ScorerSpec,
    SemanticIdentitySpec,
)
from rl_engine.kernels.gtest.tolerance import (
    resolve_logprob_threshold,
    tolerance_contract_fingerprint,
)

RLK_ALIGNMENT_PROFILE_ORDER = ("A0", "A1", "A2", "A3", "A4", "A5")
ALIGNMENT_STANDARD_ID = "rl_kernel.cross_config.alignment_standard"
ALIGNMENT_PROFILE_VERSION = "cross_config.alignment_profiles.v1"


@dataclass(frozen=True)
class AlignmentProfile:
    name: str
    description: str
    aligned_axes: tuple[str, ...]
    mismatched_axes: tuple[str, ...] = ()
    production_like: bool = False
    source: str = "rl_kernel"

    def __post_init__(self) -> None:
        if self.name not in RLK_ALIGNMENT_PROFILE_ORDER:
            raise ValueError(f"unknown A0-A5 profile {self.name!r}.")
        object.__setattr__(self, "aligned_axes", tuple(self.aligned_axes))
        object.__setattr__(self, "mismatched_axes", tuple(self.mismatched_axes))

    @property
    def order(self) -> int:
        return RLK_ALIGNMENT_PROFILE_ORDER.index(self.name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "aligned_axes": self.aligned_axes,
            "mismatched_axes": self.mismatched_axes,
            "production_like": self.production_like,
            "source": self.source,
        }


@dataclass(frozen=True)
class AlignmentStandard:
    source: str
    profiles: Mapping[str, AlignmentProfile]
    compare_score_artifacts: Callable[..., Any]
    resolve_logprob_threshold: Callable[[Any], float]
    schema_types: Mapping[str, Any]
    standard_id: str = ALIGNMENT_STANDARD_ID
    profile_version: str = ALIGNMENT_PROFILE_VERSION
    fingerprint: str = ""
    tolerance_fingerprint: str = ""
    issues: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        profiles = {name: profile for name, profile in self.profiles.items()}
        missing = [name for name in RLK_ALIGNMENT_PROFILE_ORDER if name not in profiles]
        if missing:
            raise ValueError(f"alignment standard is missing profiles: {missing!r}.")
        object.__setattr__(self, "profiles", MappingProxyType(profiles))
        object.__setattr__(self, "schema_types", MappingProxyType(dict(self.schema_types)))
        object.__setattr__(self, "issues", tuple(self.issues))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def iter_profiles(self) -> tuple[AlignmentProfile, ...]:
        return tuple(self.profiles[name] for name in RLK_ALIGNMENT_PROFILE_ORDER)

    def profile(self, name: str) -> AlignmentProfile:
        try:
            return self.profiles[name]
        except KeyError as exc:
            raise ValueError(f"unknown A0-A5 profile {name!r}.") from exc

    def to_metadata(self) -> dict[str, Any]:
        return {
            "alignment_standard_source": self.source,
            "alignment_standard_id": self.standard_id,
            "alignment_profile_version": self.profile_version,
            "alignment_standard_fingerprint": self.fingerprint,
            "alignment_tolerance_fingerprint": self.tolerance_fingerprint,
            **dict(self.metadata),
        }


DISTRIBUTED_ALIGNMENT_PROFILES: Mapping[str, AlignmentProfile] = MappingProxyType(
    {
        "A0": AlignmentProfile(
            name="A0",
            description="fully aligned reference",
            aligned_axes=(
                "arithmetic",
                "reduction_topology",
                "representation",
                "metadata",
                "fallback_behavior",
            ),
        ),
        "A1": AlignmentProfile(
            name="A1",
            description="arithmetic-only mismatch",
            aligned_axes=("reduction_topology", "representation", "metadata", "fallback_behavior"),
            mismatched_axes=("arithmetic",),
        ),
        "A2": AlignmentProfile(
            name="A2",
            description="reduction/topology-only mismatch",
            aligned_axes=("arithmetic", "representation", "metadata", "fallback_behavior"),
            mismatched_axes=("reduction_topology",),
        ),
        "A3": AlignmentProfile(
            name="A3",
            description="representation-only mismatch",
            aligned_axes=("arithmetic", "reduction_topology", "metadata", "fallback_behavior"),
            mismatched_axes=("representation",),
        ),
        "A4": AlignmentProfile(
            name="A4",
            description="pairwise mismatches",
            aligned_axes=("metadata", "fallback_behavior"),
            mismatched_axes=(
                "arithmetic+reduction_topology",
                "arithmetic+representation",
                "reduction_topology+representation",
            ),
        ),
        "A5": AlignmentProfile(
            name="A5",
            description="production mismatch",
            aligned_axes=(),
            mismatched_axes=(
                "arithmetic",
                "reduction_topology",
                "representation",
                "metadata",
                "fallback_behavior",
            ),
            production_like=True,
        ),
    }
)


def iter_alignment_profiles() -> tuple[AlignmentProfile, ...]:
    return tuple(DISTRIBUTED_ALIGNMENT_PROFILES[name] for name in RLK_ALIGNMENT_PROFILE_ORDER)


def alignment_profile_fingerprint() -> str:
    canonical = json.dumps(
        [profile.to_dict() for profile in iter_alignment_profiles()],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def get_alignment_standard() -> AlignmentStandard:
    return AlignmentStandard(
        source="rl_kernel",
        profiles=DISTRIBUTED_ALIGNMENT_PROFILES,
        compare_score_artifacts=compare_score_artifacts,
        resolve_logprob_threshold=resolve_logprob_threshold,
        schema_types={
            "RuntimeProvenance": RuntimeProvenance,
            "ScoreArtifact": ScoreArtifact,
            "ScorerSpec": ScorerSpec,
            "ScoreSide": ScoreSide,
            "SemanticIdentitySpec": SemanticIdentitySpec,
        },
        fingerprint=alignment_profile_fingerprint(),
        tolerance_fingerprint=tolerance_contract_fingerprint(),
        metadata={"alignment_profile_order": RLK_ALIGNMENT_PROFILE_ORDER},
    )


__all__ = [
    "ALIGNMENT_PROFILE_VERSION",
    "ALIGNMENT_STANDARD_ID",
    "DISTRIBUTED_ALIGNMENT_PROFILES",
    "AlignmentProfile",
    "AlignmentStandard",
    "RLK_ALIGNMENT_PROFILE_ORDER",
    "alignment_profile_fingerprint",
    "get_alignment_standard",
    "iter_alignment_profiles",
]
