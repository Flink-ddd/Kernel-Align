# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Execution lifetime and identity.

Two sides of one question: identical identity is what makes reuse safe, and a
changed identity is what makes historical results stale.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from rl_engine.mismatch.schema.metrics import VariantResult
from rl_engine.mismatch.schema.values import LibraryPin, PolicyRole, RebindCost


@dataclass(frozen=True)
class ReuseKey:
    """Whether an already-built runtime can be reused.

    Four parts matching the four ``RebindCost`` levels, compared coarse to fine.
    """

    process: str  # env vars, determinism switches, compile-time flags
    process_group: str  # world size, TP/CP/PP split, comm backend
    engine: str  # dtype, backend choice, KV layout, operator implementation
    request: str  # batch size, sequence


@dataclass(frozen=True)
class EnvironmentFingerprint:
    """The execution environment. Change this layer and every number is stale."""

    python_version: str
    torch_version: str
    torch_build_hash: str  # hash of the build config (cuda/hip build, op set)
    driver_version: str
    device_model: str
    libraries: tuple[LibraryPin, ...]
    determinism_env: Mapping[str, str]  # NVTE_* / CUBLAS_* / NCCL_* / torch backends
    source_revision: str  # this framework's own version


@dataclass(frozen=True)
class ExecutionFingerprint:
    """One execution's full identity. Any part differing makes two runs
    incomparable.

    What goes in is the value read back, never the value requested: asking for
    ``num_splits=1`` and the backend using 1 are two different facts. Thresholds
    go in too, so changing one makes every historical pass/fail stale -- which is
    why thresholds are code constants, a configurable value cannot be pinned into
    an identity.
    """

    identity: str  # fingerprint of the ComparisonIdentity
    environment: EnvironmentFingerprint
    switch_binding: str  # effective switch values, read back
    implementation: Mapping[PolicyRole, str]  # what each side actually instantiated
    model_state: Mapping[PolicyRole, str]  # each side's weights
    collectives: tuple[str, ...]  # fingerprints of the collectives that ran
    threshold_table: str  # fingerprint of EXPECTED_RANGES


@dataclass(frozen=True)
class VariantRecord:
    """One variant's archived record: a ``VariantResult`` plus its identity.

    ``content_hash`` is its integrity seal. Only a record whose hash verifies may
    be reused on resume.
    """

    variant_name: str
    fingerprint: ExecutionFingerprint
    result: VariantResult
    content_hash: str


def canonical_fingerprint(payload: Any) -> str:
    """Hash a JSON-serialisable payload, normalising key order."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def reuse_level(previous: ReuseKey, current: ReuseKey) -> RebindCost:
    """How far a rebuild has to go between two cases. Earlier is costlier."""

    if previous.process != current.process:
        return RebindCost.PROCESS_RESTART
    if previous.process_group != current.process_group:
        return RebindCost.PROCESS_GROUP_REBUILD
    if previous.engine != current.engine:
        return RebindCost.ENGINE_REBUILD
    return RebindCost.PER_REQUEST


__all__ = [
    "EnvironmentFingerprint",
    "ExecutionFingerprint",
    "ReuseKey",
    "VariantRecord",
    "canonical_fingerprint",
    "reuse_level",
]
