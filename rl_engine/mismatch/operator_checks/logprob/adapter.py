# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Logprob's four operator-level methods. Not implemented."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from rl_engine.mismatch.schema import (
    CollectiveContract,
    ImplementationResolution,
    OperatorContract,
    PolicyRole,
)


def build_contract(role: PolicyRole, switch_values: Mapping[str, Any]) -> OperatorContract:
    """Return this side's contract, with the vocab shard map under ``extra``.

    Only the training side can vary the head dtype -- vLLM computes logits at the
    model dtype. Under TP, MCore's ``VocabUtility`` and vLLM's ``_get_indices``
    disagree on the shard boundaries for Qwen3, so comparing partial results
    without recording both maps is meaningless.
    """

    raise NotImplementedError


def read_effective_config(role: PolicyRole, adapter: Any) -> Mapping[str, Any]:
    raise NotImplementedError(
        "Read the head dtype, the transform chain and the vocab shard "
        "boundaries off the live engine."
    )


def observe_collectives(role: PolicyRole, adapter: Any) -> tuple[CollectiveContract, ...]:
    """Return the collective trace: empty at TP=1, the partial-LSE reduction otherwise."""

    raise NotImplementedError


def resolve_implementation(
    factor_id: str, role: PolicyRole, impl_name: str
) -> tuple[Callable[..., Any] | None, ImplementationResolution]:
    """Unused until a logprob factor gains a reference implementation."""

    raise NotImplementedError
