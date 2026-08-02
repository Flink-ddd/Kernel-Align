# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Typed WS2 contract for TP-aware selected-token log-probability.

The objects in this module describe a vocab-parallel logprob invocation:

``selected_logp[t] = logits[t, target[t]] - logsumexp_vocab(logits[t, :])``

Under vocab-parallel tensor parallelism the vocabulary-wide ``logsumexp``
requires cross-rank reduction.  This module only *describes* that invocation
(shard ownership, merge semantics, mask/ignore-index metadata); it does not
shard tensors, launch collectives, or implement the ``(max, sumexp)`` merge.
Keeping description and materialization separate lets dispatch reject an
incompatible backend before any numerically different path is launched.

Context parallelism is a declared non-merge axis: CP partitions tokens, never
the vocabulary, so the logprob reduction spans TP vocab shards only.  CP rank
metadata is carried for provenance and must never widen the merge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

_EnumT = TypeVar("_EnumT", bound=Enum)

# Dispatch policy keywords accepted by KernelRegistry.get_logprob_op; a stable
# backend id must never shadow one of these, or it becomes unselectable by id.
RESERVED_DISPATCH_POLICIES = frozenset({"auto", "production", "reference", "deterministic"})


class LogprobContractError(ValueError):
    """Raised when logprob metadata does not describe a valid invocation."""


class LogprobRole(str, Enum):
    TRAIN = "train"
    INFER = "infer"


class LogprobDType(str, Enum):
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"


class LogprobMerge(str, Enum):
    """Merge primitive for per-shard partial states.

    Every rank contributes ``(local_max, local_sumexp)`` computed in the
    accumulation dtype; the merged result is
    ``M = max(m_l)``, ``S = sum(s_l * exp(m_l - M))``, ``LSE = M + log(S)``.
    """

    MAX_SUMEXP = "max_sumexp"


class MergeAxis(str, Enum):
    """The only reduction axis of this contract; CP is a non-merge axis."""

    TP_VOCAB = "tp_vocab"


class ReductionOrder(str, Enum):
    GLOBAL_VOCAB_SHARD_INDEX = "global_vocab_shard_index"


class ReductionTransport(str, Enum):
    """Collectives move partial states only; they never reduce numerically."""

    ALL_GATHER = "all_gather"


class DowncastPoint(str, Enum):
    FINAL_WRITE = "final_write"


class ReductionEngine(str, Enum):
    IN_OP_REFERENCE = "in_op_reference"


def _enum_value(enum_type: type[_EnumT], value: Any, field: str) -> _EnumT:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise LogprobContractError(f"{field} must be one of: {allowed}; got {value!r}") from exc


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LogprobContractError(f"{field} must be a positive integer; got {value!r}")
    return value


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LogprobContractError(f"{field} must be a non-negative integer; got {value!r}")
    return value


def _plain_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LogprobContractError(f"{field} must be an integer; got {value!r}")
    return value


@dataclass(frozen=True)
class ShardingSpec:
    """Logical vocab-parallel TP ownership for one logprob invocation.

    ``vocab_shard_bounds`` lists every TP rank's half-open ``[start, end)``
    vocab range indexed by TP rank.  The full table is required on every rank:
    it defines target-token ownership and the fixed global-shard-index merge
    order without any collective, and makes an incomplete partition a loud
    construction-time error instead of a silent runtime divergence.

    ``padded_vocab_size`` is the shard-covered (weight) vocabulary;
    ``real_vocab_size`` is the tokenizer vocabulary.  Padding columns occupy
    ``[real_vocab_size, padded_vocab_size)`` and must be excluded from the
    logsumexp by any conforming implementation.
    """

    tp_rank: int
    tp_world_size: int
    vocab_shard_bounds: tuple[tuple[int, int], ...]
    real_vocab_size: int
    padded_vocab_size: int
    cp_rank: int = 0
    cp_world_size: int = 1

    def __post_init__(self) -> None:
        tp_world_size = _positive_int(self.tp_world_size, "tp_world_size")
        tp_rank = _non_negative_int(self.tp_rank, "tp_rank")
        if tp_rank >= tp_world_size:
            raise LogprobContractError(
                f"tp_rank={tp_rank} must be smaller than tp_world_size={tp_world_size}"
            )
        cp_world_size = _positive_int(self.cp_world_size, "cp_world_size")
        cp_rank = _non_negative_int(self.cp_rank, "cp_rank")
        if cp_rank >= cp_world_size:
            raise LogprobContractError(
                f"cp_rank={cp_rank} must be smaller than cp_world_size={cp_world_size}"
            )

        real_vocab_size = _positive_int(self.real_vocab_size, "real_vocab_size")
        padded_vocab_size = _positive_int(self.padded_vocab_size, "padded_vocab_size")
        if padded_vocab_size < real_vocab_size:
            raise LogprobContractError(
                f"padded_vocab_size={padded_vocab_size} must not be smaller than "
                f"real_vocab_size={real_vocab_size}"
            )

        try:
            bounds = tuple((pair[0], pair[1]) for pair in self.vocab_shard_bounds)
        except (TypeError, IndexError) as exc:
            raise LogprobContractError(
                "vocab_shard_bounds must be an iterable of (start, end) integer pairs"
            ) from exc
        if len(bounds) != tp_world_size:
            raise LogprobContractError(
                "vocab_shard_bounds must declare exactly one (start, end) pair per TP rank; "
                f"got {len(bounds)} pairs for tp_world_size={tp_world_size}"
            )
        expected_start = 0
        for rank, (start, end) in enumerate(bounds):
            start = _plain_int(start, f"vocab_shard_bounds[{rank}][0]")
            end = _plain_int(end, f"vocab_shard_bounds[{rank}][1]")
            if end <= start:
                raise LogprobContractError(
                    f"vocab_shard_bounds[{rank}] must satisfy end > start; got [{start}, {end})"
                )
            if start != expected_start:
                raise LogprobContractError(
                    "vocab_shard_bounds must form a contiguous [0, padded_vocab_size) "
                    f"partition in TP-rank order; rank {rank} starts at {start}, "
                    f"expected {expected_start}"
                )
            expected_start = end
        if expected_start != padded_vocab_size:
            raise LogprobContractError(
                "vocab_shard_bounds must cover padded_vocab_size exactly; "
                f"covered {expected_start}, declared {padded_vocab_size}"
            )
        object.__setattr__(self, "vocab_shard_bounds", bounds)

    @property
    def local_vocab_start(self) -> int:
        return self.vocab_shard_bounds[self.tp_rank][0]

    @property
    def local_vocab_end(self) -> int:
        return self.vocab_shard_bounds[self.tp_rank][1]

    @property
    def local_vocab_size(self) -> int:
        start, end = self.vocab_shard_bounds[self.tp_rank]
        return end - start

    def owner_rank(self, token_id: int) -> int:
        """Return the unique TP rank owning ``token_id``; error outside real vocab."""

        token_id = _plain_int(token_id, "token_id")
        if token_id < 0 or token_id >= self.real_vocab_size:
            raise LogprobContractError(
                f"token_id={token_id} is outside the real vocabulary "
                f"[0, {self.real_vocab_size}); mask it as inactive instead"
            )
        for rank, (start, end) in enumerate(self.vocab_shard_bounds):
            if start <= token_id < end:
                return rank
        raise LogprobContractError(
            f"token_id={token_id} is not covered by any declared vocab shard"
        )


@dataclass(frozen=True)
class MaskSpec:
    """Active-token ownership for one logprob invocation.

    Inactive tokens are excluded from every drift aggregate and are exempt
    from the exactly-one-owner target gather; their targets may legally hold
    ``ignore_index``.
    """

    num_tokens: int
    active_mask: tuple[bool, ...]
    ignore_index: int = -100
    _active_token_count: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        num_tokens = _positive_int(self.num_tokens, "num_tokens")
        _plain_int(self.ignore_index, "ignore_index")
        try:
            active_mask = tuple(self.active_mask)
        except TypeError as exc:
            raise LogprobContractError("active_mask must be an iterable of booleans") from exc
        for index, value in enumerate(active_mask):
            if not isinstance(value, bool):
                raise LogprobContractError(f"active_mask[{index}] must be a bool; got {value!r}")
        if len(active_mask) != num_tokens:
            raise LogprobContractError(
                "active_mask must contain exactly one entry per token; "
                f"got {len(active_mask)} entries for num_tokens={num_tokens}"
            )
        object.__setattr__(self, "active_mask", active_mask)
        object.__setattr__(self, "_active_token_count", sum(active_mask))

    @property
    def active_token_count(self) -> int:
        return self._active_token_count


@dataclass(frozen=True)
class ReductionSpec:
    """Deterministic TP-vocab ``(max, sumexp)`` merge semantics."""

    merge: LogprobMerge = LogprobMerge.MAX_SUMEXP
    merge_axis: MergeAxis = MergeAxis.TP_VOCAB
    acc_dtype: LogprobDType = LogprobDType.FP32
    order: ReductionOrder = ReductionOrder.GLOBAL_VOCAB_SHARD_INDEX
    transport: ReductionTransport = ReductionTransport.ALL_GATHER
    downcast_at: DowncastPoint = DowncastPoint.FINAL_WRITE
    engine: ReductionEngine = ReductionEngine.IN_OP_REFERENCE

    def __post_init__(self) -> None:
        object.__setattr__(self, "merge", _enum_value(LogprobMerge, self.merge, "merge"))
        object.__setattr__(
            self, "merge_axis", _enum_value(MergeAxis, self.merge_axis, "merge_axis")
        )
        object.__setattr__(
            self, "acc_dtype", _enum_value(LogprobDType, self.acc_dtype, "acc_dtype")
        )
        object.__setattr__(self, "order", _enum_value(ReductionOrder, self.order, "order"))
        object.__setattr__(
            self, "transport", _enum_value(ReductionTransport, self.transport, "transport")
        )
        object.__setattr__(
            self, "downcast_at", _enum_value(DowncastPoint, self.downcast_at, "downcast_at")
        )
        object.__setattr__(self, "engine", _enum_value(ReductionEngine, self.engine, "engine"))
        if self.acc_dtype is not LogprobDType.FP32:
            raise LogprobContractError(
                f"TP logprob accumulation must be fp32; got {self.acc_dtype.value}"
            )


@dataclass(frozen=True)
class LogprobContract:
    """Complete semantic request consumed by contract-aware dispatch."""

    role: LogprobRole
    dtype: LogprobDType
    mask: MaskSpec
    sharding: ShardingSpec
    reduction: ReductionSpec
    export_lse: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _enum_value(LogprobRole, self.role, "role"))
        object.__setattr__(self, "dtype", _enum_value(LogprobDType, self.dtype, "dtype"))
        if not isinstance(self.mask, MaskSpec):
            raise LogprobContractError("mask must be a MaskSpec")
        if not isinstance(self.sharding, ShardingSpec):
            raise LogprobContractError("sharding must be a ShardingSpec")
        if not isinstance(self.reduction, ReductionSpec):
            raise LogprobContractError("reduction must be a ReductionSpec")
        if not isinstance(self.export_lse, bool) or not self.export_lse:
            raise LogprobContractError(
                "export_lse must be True for the WS2 vocab-domain LSE drift contract"
            )
        if 0 <= self.mask.ignore_index < self.sharding.real_vocab_size:
            raise LogprobContractError(
                f"ignore_index={self.mask.ignore_index} must not collide with the real "
                f"vocabulary [0, {self.sharding.real_vocab_size})"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return stable, JSON-compatible requested-contract provenance."""

        sharding = {
            "tp_rank": self.sharding.tp_rank,
            "tp_world_size": self.sharding.tp_world_size,
            "cp_rank": self.sharding.cp_rank,
            "cp_world_size": self.sharding.cp_world_size,
            "vocab_shard_bounds": [list(pair) for pair in self.sharding.vocab_shard_bounds],
            "real_vocab_size": self.sharding.real_vocab_size,
            "padded_vocab_size": self.sharding.padded_vocab_size,
            "local_vocab_start": self.sharding.local_vocab_start,
            "local_vocab_end": self.sharding.local_vocab_end,
        }
        reduction = {
            "merge": self.reduction.merge.value,
            "merge_axis": self.reduction.merge_axis.value,
            "acc_dtype": self.reduction.acc_dtype.value,
            "order": self.reduction.order.value,
            "transport": self.reduction.transport.value,
            "downcast_at": self.reduction.downcast_at.value,
            "engine": self.reduction.engine.value,
            "cp_is_merge_axis": False,
        }
        # The per-token mask is deliberately summarized: provenance exists for
        # logging/serialization and the raw mask would dominate its size.
        mask = {
            "num_tokens": self.mask.num_tokens,
            "active_token_count": self.mask.active_token_count,
            "ignore_index": self.mask.ignore_index,
        }
        return {
            "semantic_operator": "selected_token_logprob",
            "role": self.role.value,
            "dtype": self.dtype.value,
            "export_lse": self.export_lse,
            "lse_domain": "vocab",
            "mask": mask,
            "sharding": sharding,
            "reduction": reduction,
        }


@dataclass(frozen=True)
class LogprobBackendCapability:
    """Capabilities a concrete backend declares to contract-aware dispatch."""

    backend_id: str
    roles: frozenset[LogprobRole]
    dtypes: frozenset[LogprobDType]
    tp_world_sizes: tuple[int, ...] | None = None
    cp_world_sizes: tuple[int, ...] | None = None
    supports_vocab_padding: bool = False
    supports_inactive_tokens: bool = False
    exports_vocab_lse: bool = False
    deterministic_tp_merge: bool = False
    implementation_kind: str = "production"

    def __post_init__(self) -> None:
        if not isinstance(self.backend_id, str) or not self.backend_id.strip():
            raise LogprobContractError("backend_id must be a non-empty string")
        if self.backend_id.strip().lower() in RESERVED_DISPATCH_POLICIES:
            raise LogprobContractError(
                f"backend_id={self.backend_id!r} shadows a reserved dispatch policy keyword"
            )
        object.__setattr__(self, "backend_id", self.backend_id.strip())
        roles = frozenset(_enum_value(LogprobRole, value, "roles") for value in self.roles)
        dtypes = frozenset(_enum_value(LogprobDType, value, "dtypes") for value in self.dtypes)
        if not roles or not dtypes:
            raise LogprobContractError("backend roles and dtypes must not be empty")
        tp_world_sizes = self._validated_world_sizes(self.tp_world_sizes, "tp_world_sizes")
        cp_world_sizes = self._validated_world_sizes(self.cp_world_sizes, "cp_world_sizes")
        for flag_name in (
            "supports_vocab_padding",
            "supports_inactive_tokens",
            "exports_vocab_lse",
            "deterministic_tp_merge",
        ):
            if not isinstance(getattr(self, flag_name), bool):
                raise LogprobContractError(f"{flag_name} must be a bool")
        if self.implementation_kind not in {"production", "reference", "deterministic"}:
            raise LogprobContractError(
                "implementation_kind must be production, reference, or deterministic"
            )
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "dtypes", dtypes)
        object.__setattr__(self, "tp_world_sizes", tp_world_sizes)
        object.__setattr__(self, "cp_world_sizes", cp_world_sizes)

    @staticmethod
    def _validated_world_sizes(
        values: tuple[int, ...] | None, field: str
    ) -> tuple[int, ...] | None:
        if values is None:
            return None
        try:
            sizes = tuple(values)
        except TypeError as exc:
            raise LogprobContractError(f"{field} must be an iterable of integers") from exc
        if not sizes:
            raise LogprobContractError(f"{field} must not be empty; use None for unrestricted")
        for value in sizes:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise LogprobContractError(f"{field} must contain positive values; got {value!r}")
        if len(set(sizes)) != len(sizes):
            raise LogprobContractError(f"{field} must not contain duplicates")
        return sizes

    def incompatibilities(self, contract: LogprobContract) -> tuple[str, ...]:
        """Explain every reason this backend cannot materialize ``contract``."""

        reasons: list[str] = []
        if contract.role not in self.roles:
            reasons.append(f"role={contract.role.value} is unsupported")
        if contract.dtype not in self.dtypes:
            reasons.append(f"dtype={contract.dtype.value} is unsupported")
        tp_size = contract.sharding.tp_world_size
        cp_size = contract.sharding.cp_world_size
        if self.tp_world_sizes is not None and tp_size not in self.tp_world_sizes:
            reasons.append(f"TP={tp_size} is unsupported")
        if self.cp_world_sizes is not None and cp_size not in self.cp_world_sizes:
            reasons.append(f"CP={cp_size} is unsupported")
        if (
            contract.sharding.padded_vocab_size != contract.sharding.real_vocab_size
            and not self.supports_vocab_padding
        ):
            reasons.append("padded-vs-real vocab masking is unsupported")
        if (
            contract.mask.active_token_count != contract.mask.num_tokens
            and not self.supports_inactive_tokens
        ):
            reasons.append("inactive-token (ignore_index) masking is unsupported")
        if contract.export_lse and not self.exports_vocab_lse:
            reasons.append("vocab-domain LSE export is unsupported")
        if tp_size > 1 and not self.deterministic_tp_merge:
            reasons.append("deterministic TP (max, sumexp) merge is unsupported")
        return tuple(reasons)

    def supports(self, contract: LogprobContract) -> bool:
        return not self.incompatibilities(contract)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "roles": sorted(role.value for role in self.roles),
            "dtypes": sorted(dtype.value for dtype in self.dtypes),
            "tp_world_sizes": list(self.tp_world_sizes) if self.tp_world_sizes else None,
            "cp_world_sizes": list(self.cp_world_sizes) if self.cp_world_sizes else None,
            "supports_vocab_padding": self.supports_vocab_padding,
            "supports_inactive_tokens": self.supports_inactive_tokens,
            "exports_vocab_lse": self.exports_vocab_lse,
            "deterministic_tp_merge": self.deterministic_tp_merge,
            "implementation_kind": self.implementation_kind,
        }


@dataclass(frozen=True)
class LogprobDispatchResult:
    """A concrete backend plus the actual provenance bound to the request."""

    op: Any
    capability: LogprobBackendCapability
    provenance: dict[str, Any]


__all__ = [
    "RESERVED_DISPATCH_POLICIES",
    "DowncastPoint",
    "LogprobBackendCapability",
    "LogprobContract",
    "LogprobContractError",
    "LogprobDType",
    "LogprobDispatchResult",
    "LogprobMerge",
    "LogprobRole",
    "MaskSpec",
    "MergeAxis",
    "ReductionEngine",
    "ReductionOrder",
    "ReductionSpec",
    "ReductionTransport",
    "ShardingSpec",
]
