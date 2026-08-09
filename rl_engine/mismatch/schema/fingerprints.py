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
    """Whether an already-built runtime can be reused. Four parts matching the
    four ``RebindCost`` levels, coarse to fine.

    Compared coarse to fine: a different ``process`` means restarting the whole
    process; equal process but different engine means only rebuilding the engine.
    """

    process: str  # env vars, determinism switches, compile-time flags
    process_group: str  # world size, TP/CP/PP split, comm backend
    engine: str  # dtype, backend choice, KV layout, operator implementation
    request: str  # batch size, sequence


@dataclass(frozen=True)
class EnvironmentFingerprint:
    """The execution environment. When this layer changes, every numerical
    conclusion has to be re-run."""

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

    Three rules:

    1. **What goes in is the value read back, not the value requested.**
       Requesting ``num_splits=1`` and the backend actually using 1 are two
       different facts.
    2. **Library versions belong here, not in a footnote.**
    3. **Thresholds go in too.** Change a threshold and every historical
       pass/fail must go stale -- which is why thresholds are code constants: a
       configurable value cannot be pinned into an identity.
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
    """One variant's immutable on-disk record. Append-only, never edited.

    Difference from ``VariantResult``: a Result is what was just computed in
    memory, a Record is archived -- it additionally carries the execution
    identity and a content hash, the latter being its integrity seal. Only a
    record whose hash verifies may be reused on resume.

    Not ``VariantArtifact`` -- in an ML context, "artifact" usually means model
    weights or checkpoints (MLflow artifacts); this stores an execution record.
    """

    variant_name: str
    fingerprint: ExecutionFingerprint
    result: VariantResult
    content_hash: str


def canonical_fingerprint(payload: Any) -> str:
    """Stable hash of a JSON-serialisable payload.

    Key order is normalised so the same content always hashes the same, whatever
    order the dict was built in.
    """

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
