# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Pure data structures: public fields, frozen, no meaningful methods.

Behaviour lives in free functions under ``pipeline/``. The framework grows by
adding functions while the set of types stays stable, which is the side
procedural code is good at -- so do not hang methods on these structures. Write
``requires_fixed_order(contract)``, not ``contract.requires_fixed_order()``.

Chained field access such as ``factor.switch.path`` is fine: Demeter constrains
an object's internals, and plain data is meant to expose its fields.

Modules are ordered by dependency; each imports only from the ones above it.
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
