# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""GEMM's four operator-level methods. Not implemented."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from rl_engine.mismatch.schema import (
    CollectiveContract,
    ImplementationResolution,
    OperatorContract,
    PolicyRole,
)


def build_contract(role: PolicyRole, switch_values: Mapping[str, Any]) -> OperatorContract:
    """Return this side's contract, with the collective the switch selects first.

    The factors index into ``collectives`` by position, so the order has to stay
    stable. The two native sides differ by construction -- Megatron with sequence
    parallelism rewrites all_reduce into reduce_scatter + all_gather, vLLM does a
    plain all_reduce -- and that difference is the factor, not a defect.
    """

    raise NotImplementedError


def read_effective_config(role: PolicyRole, adapter: Any) -> Mapping[str, Any]:
    raise NotImplementedError(
        "Read actual values off the live engine, including which all-reduce "
        "backend vLLM really chose."
    )


def observe_collectives(role: PolicyRole, adapter: Any) -> tuple[CollectiveContract, ...]:
    """Return the collective trace for this run.

    vLLM switches between custom IPC, MNNVL and NCCL by world size and topology,
    so only what ran is evidence.
    """

    raise NotImplementedError


def resolve_implementation(
    factor_id: str, role: PolicyRole, impl_name: str
) -> tuple[Callable[..., Any] | None, ImplementationResolution]:
    """Resolve an arm's implementation name, returning the trace even on failure.

    ``rl_engine.kernels.collectives.ordered_reduce_scatter`` does not exist yet,
    so this must fail loudly enough for gate 1 to report
    ``VARIANT_DID_NOT_APPLY`` rather than letting a silent fallback read as a
    clean result.
    """

    raise NotImplementedError
