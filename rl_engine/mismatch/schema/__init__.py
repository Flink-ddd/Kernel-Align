# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Pure data structures: public fields, frozen, no meaningful methods.

All behaviour lives in free functions under ``pipeline/``. This is a deliberate
trade-off, not an accident of using dataclasses:

======================  ==================  ==================
                        add a new function  add a new type
======================  ==================  ==================
data + procedural       easy                hard
objects + polymorphism  hard                easy
======================  ==================  ==================

This framework grows by **adding functions** (new diagnosis rules, new report
views, new ordering strategies, new evidence checks) while the set of types
stays stable -- so procedural is the correct side.

Two consequences:

* Do not hang methods on these structures. ``spec.requires_fixed_order()`` turns
  a data structure into a half-object hybrid, which is the worst of both worlds.
  Write ``requires_fixed_order(spec)`` instead.
* Chained field access such as ``factor.switch.path`` is **fine** here and does
  not violate the Law of Demeter -- Demeter constrains an object's internals,
  and plain data structures are supposed to expose their fields. Do not wrap
  them in getters for the sake of it.

Modules are ordered by dependency; each only imports from the ones above it.
"""

from rl_engine.mismatch.schema.collectives import (
    ALL_REDUCE_AS_SCATTER_GATHER,
    ALL_TO_ALL_AS_GATHER_SLICE,
    CollectiveContract,
    CollectiveOp,
    CollectiveRewrite,
    DeterminismLevel,
    ParallelDim,
    ReductionOrder,
)
from rl_engine.mismatch.schema.contracts import (
    ComparisonIssue,
    ComparisonIssueCode,
    ComparisonRule,
    OperatorContract,
)
from rl_engine.mismatch.schema.factors import (
    BATCH_PLACEMENT,
    COLLECTIVE_CONTRACT,
    LSE_EXPORT,
    MODEL_SHAPE,
    POSITION_CACHE,
    VOCAB_SHARD_MAP,
    Evidence,
    FactorCategory,
    MismatchFactor,
    Prerequisites,
    ReferenceAuthority,
    ReferenceImplementation,
    Switch,
    declared_collectives,
    requires_fixed_order,
)
from rl_engine.mismatch.schema.fingerprints import (
    EnvironmentFingerprint,
    ExecutionFingerprint,
    ReuseKey,
    VariantRecord,
    canonical_fingerprint,
    reuse_level,
)
from rl_engine.mismatch.schema.metrics import (
    DEFAULT_CLIP_EPS,
    FactorReport,
    ImplementationResolution,
    LogprobShard,
    MismatchMetrics,
    RejectedCandidate,
    VariantResult,
    WorstToken,
    is_silent_failure,
    missing_evidence,
)
from rl_engine.mismatch.schema.pitfalls import FailureMode, KnownPitfall
from rl_engine.mismatch.schema.rollout_context import (
    BatchPlacement,
    ComparisonIdentity,
    DynamicSamplingDecision,
    RolloutGroup,
)
from rl_engine.mismatch.schema.thresholds import (
    ANY_MODEL_FAMILY,
    EXPECTED_RANGES,
    ThresholdLookupError,
    expected_range,
    tolerance_floor,
)
from rl_engine.mismatch.schema.tracing import (
    MismatchReport,
    ModuleCorrespondence,
    PropagationEdge,
    RootCauseCategory,
    RootCauseHypothesis,
)
from rl_engine.mismatch.schema.values import (
    DowncastPoint,
    ExecutionPath,
    LibraryPin,
    PolicyRole,
    Precision,
    PrecisionProfile,
    RebindCost,
    RequiredSetting,
    SettingChannel,
    choice_parser,
    positive_int,
    strict_bool,
)
from rl_engine.mismatch.schema.variants import (
    Diagnosis,
    ExpectedOutcome,
    ExpectedRange,
    FactorVariant,
    NoiseFloor,
    SwitchStatus,
    VariantExpansion,
)
