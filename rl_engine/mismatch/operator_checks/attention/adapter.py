# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Attention's four operator-level methods. Not implemented.

They are operator-level rather than factor-level because reading configuration
back from an engine is the same logic for all of attention's factors.
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
    raise NotImplementedError(
        "Return this side's contract with each field at the path its factor "
        "declared in comparison_rules: precision.*, collectives[i].*, and "
        "attention's own state under a flat extra.*"
    )


def read_effective_config(role: PolicyRole, adapter: Any) -> Mapping[str, Any]:
    raise NotImplementedError(
        "Read actual values off the live engine. Megatron's AttnBackend=auto and "
        "vLLM's backend selection both pick per shape, so a requested value is "
        "not evidence."
    )


def observe_collectives(role: PolicyRole, adapter: Any) -> tuple[CollectiveContract, ...]:
    raise NotImplementedError("Return the collectives that actually ran.")


def resolve_implementation(
    factor_id: str, role: PolicyRole, impl_name: str
) -> tuple[Callable[..., Any] | None, ImplementationResolution]:
    """Resolve an arm's implementation name, returning the trace even on failure.

    A bare ``None`` produces ``FELL_BACK`` with nothing to investigate, and a
    fallen-back arm whose deviation did not change reads exactly like a clean
    ``NOT_THIS_FACTOR``.

    Candidates for the RoPE reference, in order:
    ``transformer_engine.pytorch.attention.rope:apply_rotary_pos_emb`` on the
    training side; ``flashinfer.rope:apply_rope`` then vLLM's
    ``rotary_embedding:get_rope`` on the rollout side. vLLM forwards to
    FlashInfer when it is available, so which one answered is itself evidence.
    """

    raise NotImplementedError(
        "Import the candidates in order and return (callable, resolution)."
    )
