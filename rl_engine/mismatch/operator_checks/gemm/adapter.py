# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The four operator-level methods for gemm. **Interface only.**

Whoever claims gemm implements these four; the factor files under ``factors/``
stay declarations and do not change.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from rl_engine.mismatch.schema import (
    CollectiveContract,
    ImplementationResolution,
    OperatorContract,
    PolicyRole,
)


def build_contract(role: PolicyRole, switch_values: Mapping[str, Any]) -> OperatorContract:
    """This side's switch values -> this side's numerical contract.

    For a reduction factor the load-bearing part is ``collectives``: the factors
    index into it by path (``collectives[0].reduction_order`` and friends), so
    the tuple order has to be stable and documented. ``_common.py`` already
    declares the three contracts this operator switches between.

    Remember the two native sides differ by construction: Megatron with sequence
    parallelism rewrites ``all_reduce`` into ``reduce_scatter + all_gather``,
    vLLM does a plain ``all_reduce``. That difference is the factor, not a bug in
    the contract.
    """

    raise NotImplementedError(
        "gemm.build_contract: return the OperatorContract for this side, with the "
        "collective this switch value selects at collectives[0]."
    )


def read_effective_config(role: PolicyRole, adapter: Any) -> Mapping[str, Any]:
    """Read the switches back **off the engine**. Requested is not actual."""

    raise NotImplementedError(
        "gemm.read_effective_config: read actual values off the live engine, "
        "including which all-reduce backend vLLM really chose."
    )


def observe_collectives(role: PolicyRole, adapter: Any) -> tuple[CollectiveContract, ...]:
    """The collectives this operator really performed, this run.

    Not what the config asked for: vLLM switches between custom IPC, MNNVL and
    NCCL by world size and topology, and only what actually ran is evidence.
    Feeds ``COLLECTIVE_CONTRACT``, which gate 2 requires before any verdict.
    """

    raise NotImplementedError(
        "gemm.observe_collectives: return the collective trace for this run."
    )


def resolve_implementation(
    factor_id: str, role: PolicyRole, impl_name: str
) -> tuple[Callable[..., Any] | None, ImplementationResolution]:
    """Resolve the name an arm asks for into a callable, with the rejection trace.

    ``gemm.forward_reduce`` names ``rl_engine.kernels.collectives.ordered_reduce_scatter``,
    the deterministic reduction that does not exist yet. Until it does, this must
    fail *loudly* -- return the trace so gate 1 reports ``VARIANT_DID_NOT_APPLY``
    rather than letting a silent fallback read as a clean result.
    """

    raise NotImplementedError(
        "gemm.resolve_implementation: import the requested implementation and "
        "return (callable, ImplementationResolution) -- including the rejection "
        "trace on failure."
    )
