# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import hashlib
import json

from rl_engine.alignment.cross_config import (
    ALIGNMENT_PROFILE_VERSION,
    ALIGNMENT_STANDARD_ID,
    DISTRIBUTED_ALIGNMENT_PROFILES,
    RLK_ALIGNMENT_PROFILE_ORDER,
    alignment_profile_fingerprint,
    compare_score_artifacts,
    get_alignment_standard,
    iter_alignment_profiles,
)
from rl_engine.alignment.cross_config.schema import (
    RuntimeProvenance,
    ScoreArtifact,
    ScoreSide,
    ScorerSpec,
    SemanticIdentitySpec,
)
from rl_engine.kernels.gtest.tolerance import tolerance_contract_fingerprint


def test_alignment_profiles_are_ordered_a0_to_a5():
    profiles = iter_alignment_profiles()

    assert [profile.name for profile in profiles] == list(RLK_ALIGNMENT_PROFILE_ORDER)
    assert profiles[0].description == "fully aligned reference"
    assert profiles[0].aligned_axes == (
        "arithmetic",
        "reduction_topology",
        "representation",
        "metadata",
        "fallback_behavior",
    )
    assert profiles[4].mismatched_axes == (
        "arithmetic+reduction_topology",
        "arithmetic+representation",
        "reduction_topology+representation",
    )
    assert profiles[-1].production_like


def test_alignment_profile_fingerprint_is_canonical_profile_sha256():
    canonical = json.dumps(
        [profile.to_dict() for profile in iter_alignment_profiles()],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    assert alignment_profile_fingerprint() == hashlib.sha256(canonical).hexdigest()
    assert len(alignment_profile_fingerprint()) == 64


def test_get_alignment_standard_exports_schema_comparator_and_metadata():
    standard = get_alignment_standard()

    assert standard.source == "rl_kernel"
    assert standard.standard_id == ALIGNMENT_STANDARD_ID
    assert standard.profile_version == ALIGNMENT_PROFILE_VERSION
    assert standard.profiles is not DISTRIBUTED_ALIGNMENT_PROFILES
    assert standard.profile("A0").name == "A0"
    assert standard.compare_score_artifacts is compare_score_artifacts
    assert standard.schema_types == {
        "RuntimeProvenance": RuntimeProvenance,
        "ScoreArtifact": ScoreArtifact,
        "ScorerSpec": ScorerSpec,
        "ScoreSide": ScoreSide,
        "SemanticIdentitySpec": SemanticIdentitySpec,
    }
    assert standard.fingerprint == alignment_profile_fingerprint()
    assert standard.tolerance_fingerprint == tolerance_contract_fingerprint()
    assert standard.to_metadata()["alignment_profile_order"] == RLK_ALIGNMENT_PROFILE_ORDER
