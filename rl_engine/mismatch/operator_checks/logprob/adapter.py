# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The four operator-level methods for logprob. **Interface only.**

Whoever claims logprob implements these four; the factor files under
``factors/`` stay declarations and do not change.
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

    Two asymmetries to encode rather than smooth over:

    * only the training side can vary the head dtype -- vLLM computes logits at
      the model dtype, which is why ``logp.precision_downcast`` declares
      ``applies_to=(TRAINING,)``;
    * under TP the vocabulary is sharded, and MCore's ``VocabUtility`` and vLLM's
      ``_get_indices`` do not agree on the boundaries for Qwen3. Record each
      rank's ``(vocab_start, vocab_end)`` in ``extra`` -- comparing partial
      results across different shard maps is meaningless.
    """

    raise NotImplementedError(
        "logprob.build_contract: return the OperatorContract for this side, with "
        "precision.lm_head set from the switch and the vocab shard map in extra."
    )


def read_effective_config(role: PolicyRole, adapter: Any) -> Mapping[str, Any]:
    """Read the switches back **off the engine**. Requested is not actual."""

    raise NotImplementedError(
        "logprob.read_effective_config: read the head dtype, the transform chain "
        "and the vocab shard boundaries off the live engine."
    )


def observe_collectives(role: PolicyRole, adapter: Any) -> tuple[CollectiveContract, ...]:
    """The collectives this operator really performed, this run.

    Empty at TP=1. Under vocab parallelism this is the partial-LSE reduction that
    ``logp.reduce_topology`` and ``logp.merge_order`` are about.
    """

    raise NotImplementedError(
        "logprob.observe_collectives: return the collective trace for this run "
        "(empty tuple at TP=1)."
    )


def resolve_implementation(
    factor_id: str, role: PolicyRole, impl_name: str
) -> tuple[Callable[..., Any] | None, ImplementationResolution]:
    """Resolve the name an arm asks for into a callable, with the rejection trace.

    A parameter sweep never asks for one, so this stays unused until a logprob
    factor gains a reference -- the deterministic vocab-parallel reduction, which
    is not written yet.
    """

    raise NotImplementedError(
        "logprob.resolve_implementation: import the requested implementation and "
        "return (callable, ImplementationResolution) -- including the rejection "
        "trace on failure."
    )
