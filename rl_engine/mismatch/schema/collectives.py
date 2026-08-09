# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Collective communication as a first-class concept.

Mismatch comes from floating-point addition not being associative, and the
accumulation order is almost entirely decided by collective communication.
``gemm.forward_reduce``, ``gemm.dgrad_reduce``, ``attn.cp_split_k_merge_order``,
``logp.reduce_topology``, ``moe.tp_forces_sp_reduce`` and
``gemm.rollout_all_reduce_backend`` are six instances of *one* semantic model --
writing each separately guarantees they drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rl_engine.mismatch.schema.values import DowncastPoint, LibraryPin, Precision


class CollectiveOp(str, Enum):
    ALL_REDUCE = "all_reduce"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_GATHER = "all_gather"
    ALL_TO_ALL = "all_to_all"
    BROADCAST = "broadcast"
    POINT_TO_POINT = "point_to_point"
    NONE = "none"  # single-device path, recorded explicitly rather than left blank


class ParallelDim(str, Enum):
    """Which parallel dimension the communication happens on.

    Not ``ProcessGroupKind`` -- ``ProcessGroup`` is an existing
    ``torch.distributed`` type, and this describes a parallel dimension, not the
    process group object.
    """

    TENSOR = "tensor"
    SEQUENCE = "sequence"
    CONTEXT = "context"
    EXPERT = "expert"
    PIPELINE = "pipeline"
    DATA = "data"


class ReductionOrder(str, Enum):
    """Accumulation order -- the direct root of mismatch, not the collective."""

    ARRIVAL = "arrival"  # first to arrive is first added. Control group only
    NCCL_ALGORITHM = "nccl_algorithm"  # ring/tree, varies with world size and message size
    GLOBAL_RANK_INDEX = "global_rank_index"
    GLOBAL_BLOCK_INDEX = "global_block_index"  # CP / split-K merge
    GLOBAL_VOCAB_SHARD_INDEX = "global_vocab_shard_index"


class DeterminismLevel(str, Enum):
    """How strong a reproducibility guarantee an implementation offers."""

    NONE = "none"
    STABLE_WITHIN_PROCESS = "stable_within_process"
    STABLE_ACROSS_RUNS = "stable_across_runs"
    STABLE_ACROSS_TOPOLOGY = "stable_across_topology"


@dataclass(frozen=True)
class CollectiveContract:
    """The full numerical semantics of one collective.

    Called a contract rather than a spec: it shares the word with
    ``OperatorContract``, and it really is "what both sides must agree on",
    not merely a description.

    A self-contradictory combination must be rejected at planning time: claiming
    ``determinism == STABLE_ACROSS_TOPOLOGY`` while setting ``reduction_order``
    to ``NCCL_ALGORITHM`` or ``ARRIVAL`` produces numbers that mean nothing.
    Static check, no run needed -- see ``reject_contradictory_factors()``.
    """

    op: CollectiveOp
    group: ParallelDim
    group_size: int
    reduction_order: ReductionOrder
    accumulate_precision: Precision
    downcast_at: DowncastPoint
    determinism: DeterminismLevel
    backend: str  # "nccl" / "vllm_custom_ipc" / "mnnvl" / "transformer_engine" / "rl_kernel"
    pinned_libraries: tuple[LibraryPin, ...] = ()


@dataclass(frozen=True)
class CollectiveRewrite:
    """A mathematically identical rewrite of a collective -- equal in algebra,
    unequal in floating point.

    Megatron replacing ``all_reduce`` with ``reduce_scatter + all_gather`` when
    sequence parallelism is on is exactly this rule applied. Whether the two
    sides apply it is itself a source of mismatch, so it has to be declarable.
    """

    name: str
    original: tuple[CollectiveOp, ...]
    rewritten: tuple[CollectiveOp, ...]
    preserves_bitwise: bool = False  # always False -- that is the whole problem


ALL_REDUCE_AS_SCATTER_GATHER = CollectiveRewrite(
    name="all_reduce -> reduce_scatter + all_gather",
    original=(CollectiveOp.ALL_REDUCE,),
    rewritten=(CollectiveOp.REDUCE_SCATTER, CollectiveOp.ALL_GATHER),
)

ALL_TO_ALL_AS_GATHER_SLICE = CollectiveRewrite(
    name="all_to_all -> all_gather + local slice",
    original=(CollectiveOp.ALL_TO_ALL,),
    rewritten=(CollectiveOp.ALL_GATHER,),
)


__all__ = [
    "ALL_REDUCE_AS_SCATTER_GATHER",
    "ALL_TO_ALL_AS_GATHER_SLICE",
    "CollectiveContract",
    "CollectiveOp",
    "CollectiveRewrite",
    "DeterminismLevel",
    "ParallelDim",
    "ReductionOrder",
]
