# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Logprob's four operator-level methods."""

from __future__ import annotations

import importlib
from dataclasses import replace
from typing import Any, Callable, Mapping

from rl_engine.mismatch.operator_checks.logprob._common import (
    DETERMINISTIC_LSE_REFERENCE,
    DOWNCAST_POINTS,
    HEAD_DTYPES,
    NATIVE_ROLLOUT_LSE_MERGE,
    NATIVE_TRAINING_LSE_MERGE,
    QWEN3_PADDED_VOCAB,
    QWEN3_REAL_VOCAB,
    REFERENCE_LSE_MERGE,
    TP_SIZE,
    even_vocab_shard_bounds,
)
from rl_engine.mismatch.schema import (
    CollectiveContract,
    DowncastPoint,
    ImplementationResolution,
    OperatorContract,
    PolicyRole,
    Precision,
    PrecisionProfile,
    RejectedCandidate,
    positive_int,
)

# Both sides run the model at bf16; only the training head can deviate from it.
_MODEL_DTYPE = Precision.BF16


class LogprobAdapterError(ValueError):
    """A switch value or engine adapter this plugin cannot interpret."""


def _merge_choice(value: Any, role: PolicyRole) -> str:
    """Map a ``logp.lse_merge`` switch value to this side's implementation.

    ``<name>@training`` / ``<name>@rollout`` are the one-sided swap arms.
    """

    if value in (None, "native"):
        return "native"
    name = DETERMINISTIC_LSE_REFERENCE.name
    if value == name:
        return "reference"
    if value == f"{name}@training":
        return "reference" if role is PolicyRole.TRAINING else "native"
    if value == f"{name}@rollout":
        return "reference" if role is PolicyRole.ROLLOUT else "native"
    raise LogprobAdapterError(
        f"unknown logp.lse_merge value {value!r}; expected 'native', {name!r}, "
        f"'{name}@training' or '{name}@rollout'"
    )


def build_contract(role: PolicyRole, switch_values: Mapping[str, Any]) -> OperatorContract:
    """Return this side's contract, with the vocab shard map under ``extra``.

    Only the training side can vary the head dtype -- vLLM computes logits at the
    model dtype. Under TP, MCore's ``VocabUtility`` and vLLM's ``_get_indices``
    disagree on the shard boundaries for Qwen3, so comparing partial results
    without recording both maps is meaningless.
    """

    tp_world_size = positive_int(switch_values.get("logp.tp_world_size", TP_SIZE))

    head_key = switch_values.get("logp.head_dtype", "bf16")
    if head_key not in HEAD_DTYPES:
        raise LogprobAdapterError(
            f"unknown logp.head_dtype value {head_key!r}; expected one of {tuple(HEAD_DTYPES)}"
        )
    downcast_key = switch_values.get("logp.downcast_at", "final_write")
    if downcast_key not in DOWNCAST_POINTS:
        raise LogprobAdapterError(
            f"unknown logp.downcast_at value {downcast_key!r}; "
            f"expected one of {tuple(DOWNCAST_POINTS)}"
        )
    if role is PolicyRole.TRAINING:
        lm_head = HEAD_DTYPES[head_key]
        downcast_at = DOWNCAST_POINTS[downcast_key]
    else:
        lm_head = _MODEL_DTYPE
        downcast_at = DowncastPoint.FINAL_WRITE

    merge = _merge_choice(switch_values.get("logp.lse_merge"), role)
    collectives: tuple[CollectiveContract, ...]
    if tp_world_size == 1:
        collectives = ()
    elif merge == "reference":
        collectives = (replace(REFERENCE_LSE_MERGE, group_size=tp_world_size),)
    elif role is PolicyRole.TRAINING:
        collectives = (replace(NATIVE_TRAINING_LSE_MERGE, group_size=tp_world_size),)
    else:
        collectives = (replace(NATIVE_ROLLOUT_LSE_MERGE, group_size=tp_world_size),)

    return OperatorContract(
        operator="logprob",
        role=role,
        precision=PrecisionProfile(
            compute=_MODEL_DTYPE,
            accumulate=Precision.FP32,
            downcast_at=downcast_at,
            lm_head=lm_head,
        ),
        collectives=collectives,
        extra={
            "vocab_shard_map": even_vocab_shard_bounds(QWEN3_PADDED_VOCAB, tp_world_size),
            "real_vocab_size": QWEN3_REAL_VOCAB,
            "padded_vocab_size": QWEN3_PADDED_VOCAB,
            "logprobs_mode": (
                "vocab_parallel_cross_entropy" if role is PolicyRole.TRAINING else "raw_logprobs"
            ),
            "lse_export": merge == "reference",
        },
    )


def read_effective_config(role: PolicyRole, adapter: Any) -> Mapping[str, Any]:
    """Read the head dtype, transform chain and shard boundaries off the engine.

    Accepts an engine backend exposing ``read_effective_config()``, a plain
    mapping already read back, or an object carrying ``effective_config``. What
    comes back is actual state, never the requested switch values.
    """

    adapter_role = getattr(adapter, "role", None)
    if adapter_role is not None and adapter_role is not role:
        raise LogprobAdapterError(
            f"adapter plays {adapter_role.value!r} but was queried as {role.value!r}"
        )

    reader = getattr(adapter, "read_effective_config", None)
    if callable(reader):
        return dict(reader())
    if isinstance(adapter, Mapping):
        return dict(adapter)
    config = getattr(adapter, "effective_config", None)
    if config is not None:
        return dict(config)
    raise LogprobAdapterError(
        f"cannot read an effective config off {type(adapter).__name__}: expected "
        "read_effective_config(), a mapping, or an effective_config attribute"
    )


def observe_collectives(role: PolicyRole, adapter: Any) -> tuple[CollectiveContract, ...]:
    """Return the collective trace: empty at TP=1, the partial-LSE merge otherwise.

    Which contract applies is decided by the effective config read off the
    engine, not by what was requested.
    """

    config = read_effective_config(role, adapter)
    tp_world_size = positive_int(config.get("logp.tp_world_size", 1))
    if tp_world_size == 1:
        return ()
    merge = _merge_choice(config.get("logp.lse_merge"), role)
    if merge == "reference":
        return (replace(REFERENCE_LSE_MERGE, group_size=tp_world_size),)
    if role is PolicyRole.TRAINING:
        return (replace(NATIVE_TRAINING_LSE_MERGE, group_size=tp_world_size),)
    return (replace(NATIVE_ROLLOUT_LSE_MERGE, group_size=tp_world_size),)


def resolve_implementation(
    factor_id: str, role: PolicyRole, impl_name: str
) -> tuple[Callable[..., Any] | None, ImplementationResolution]:
    """Resolve an arm's dotted import path, returning the trace even on failure.

    The WS2 vocab-parallel reference ships behind issue #241 PR3, so on a tree
    without it the rejection record is the finding: ``FELL_BACK`` with the
    import error, not a silent ``None``.
    """

    rejected: list[RejectedCandidate] = []
    if "." not in impl_name:
        rejected.append(RejectedCandidate(name=impl_name, reason="not a dotted import path"))
        return None, ImplementationResolution(
            requested=impl_name, resolved=None, rejected=tuple(rejected)
        )

    module_name, _, attribute = impl_name.rpartition(".")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        rejected.append(RejectedCandidate(name=impl_name, reason=f"import failed: {exc}"))
        return None, ImplementationResolution(
            requested=impl_name, resolved=None, rejected=tuple(rejected)
        )

    resolved = getattr(module, attribute, None)
    if resolved is None:
        rejected.append(
            RejectedCandidate(
                name=impl_name, reason=f"{module_name} has no attribute {attribute!r}"
            )
        )
        return None, ImplementationResolution(
            requested=impl_name, resolved=None, rejected=tuple(rejected)
        )
    if isinstance(resolved, type):
        try:
            resolved = resolved()
        except Exception as exc:  # noqa: BLE001 - the reason goes into the trace
            rejected.append(
                RejectedCandidate(name=impl_name, reason=f"instantiation failed: {exc}")
            )
            return None, ImplementationResolution(
                requested=impl_name, resolved=None, rejected=tuple(rejected)
            )
    if not callable(resolved):
        rejected.append(RejectedCandidate(name=impl_name, reason="resolved object is not callable"))
        return None, ImplementationResolution(
            requested=impl_name, resolved=None, rejected=tuple(rejected)
        )

    return resolved, ImplementationResolution(
        requested=impl_name, resolved=impl_name, rejected=tuple(rejected)
    )
