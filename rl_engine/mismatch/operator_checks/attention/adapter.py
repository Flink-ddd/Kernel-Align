# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The four operator-level methods for attention. **Interface only.**

They are operator-level, not factor-level: how to read configuration back from an
engine and how to observe collectives is one piece of logic shared by all of
attention's factors. Pushing them down would make every factor file copy them.

Whoever claims attention implements these four; the factor files under
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

    Put each field at the path its factor declared in ``comparison_rules``:
    ``precision.*`` for the precision profile, ``collectives[i].*`` for
    communication, ``extra.*`` for everything attention-specific (rope_theta,
    post-RoPE Q/K digests, GQA head mapping, page layout). Keep ``extra`` flat --
    nested dicts make the paths long and unreadable.

    No "collect the comparable fields" helper is needed: the framework indexes
    both contracts by path and compares entry by entry.
    """

    raise NotImplementedError(
        "attention.build_contract: return the OperatorContract for this side, "
        "with fields at the paths declared in each factor's comparison_rules."
    )


def read_effective_config(role: PolicyRole, adapter: Any) -> Mapping[str, Any]:
    """Read the switches back **off the engine**.

    A requested value is not evidence. Megatron's ``AttnBackend=auto`` and vLLM's
    backend selection both pick per shape, so requested and actual must be
    recorded separately (pitfall ``requested_not_actual``).
    """

    raise NotImplementedError(
        "attention.read_effective_config: read actual values off the live engine "
        "and return them; anything with no readback path is UNOBSERVABLE."
    )


def observe_collectives(role: PolicyRole, adapter: Any) -> tuple[CollectiveContract, ...]:
    """Which collectives this operator really performed, this run.

    Feeds the ``COLLECTIVE_CONTRACT`` evidence item. RoPE performs none; CP
    attention performs several. Return what ran, not what was configured.
    """

    raise NotImplementedError(
        "attention.observe_collectives: return the collectives that actually ran "
        "(empty tuple for single-device paths)."
    )


def resolve_implementation(
    factor_id: str, role: PolicyRole, impl_name: str
) -> tuple[Callable[..., Any] | None, ImplementationResolution]:
    """Resolve the name an arm asks for into a callable.

    **Return the trace even when resolution fails** -- which candidates were
    tried and why each was rejected. A bare ``None`` produces
    ``SwitchStatus.FELL_BACK`` with nothing to investigate, and a fallen-back arm
    whose deviation "did not change" reads exactly like a clean
    ``NOT_THIS_FACTOR``.

    Candidates worth trying in order, for the RoPE reference:
    ``transformer_engine.pytorch.attention.rope:apply_rotary_pos_emb`` on the
    training side; ``flashinfer.rope:apply_rope`` then vLLM's
    ``rotary_embedding:get_rope`` on the rollout side -- vLLM forwards to
    FlashInfer itself when it is available, so which one answered is evidence.
    """

    raise NotImplementedError(
        "attention.resolve_implementation: import the candidates in order and "
        "return (callable, ImplementationResolution) -- including the rejection "
        "trace on failure."
    )
