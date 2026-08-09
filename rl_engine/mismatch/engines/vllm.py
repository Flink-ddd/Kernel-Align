# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The rollout side: vLLM. Not implemented.

Settings this backend must pin, and where it reads each one back:

    enable_chunked_prefill       False   llm_engine.scheduler_config.chunked_prefill_enabled
    max_num_batched_tokens       > len(prompt + response)
                                         llm_engine.scheduler_config.max_num_batched_tokens
    long_prefill_token_threshold 0       llm_engine.scheduler_config.long_prefill_token_threshold
    enable_prefix_caching        False   llm_engine.cache_config.enable_prefix_caching

``enable_chunked_prefill`` defaults to True, so a full-sequence prefill is
chunked unless it is turned off. ``prompt_logprobs`` already skips the prefix
cache, so unless prefix caching is off too, the full-prefill and decode paths run
against different cache state and are not comparable. Position 0 of
``prompt_logprobs`` is None, which is where the shift-by-one convention has to be
asserted against the training side rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from rl_engine.mismatch.schema import ComparisonIdentity, PolicyRole, ReuseKey


@dataclass
class VllmBackend:
    role: PolicyRole = PolicyRole.ROLLOUT
    engine: Any = None
    effective_config: dict[str, Any] = field(default_factory=dict)

    def score(
        self,
        role: PolicyRole,
        identity: ComparisonIdentity,
        switch_values: Mapping[str, Any],
        replacement: Callable[..., Any] | None,
    ) -> tuple[Sequence[float], Mapping[str, Any]]:
        raise NotImplementedError(
            "Build the engine with the settings pinned in this module's docstring, "
            "run the requested ExecutionPath, and return (logprobs, readback)."
        )

    def reuse_key(self, switch_values: Mapping[str, Any]) -> ReuseKey:
        raise NotImplementedError(
            "Group switches by tier: determinism env -> process, world size and "
            "TP/CP -> process_group, dtype and backend and KV layout -> engine, "
            "batch and sequence -> request."
        )

    def read_effective_config(self) -> Mapping[str, Any]:
        raise NotImplementedError(
            "Read each pinned setting off the live engine. A setting with no "
            "readback path can only be recorded UNOBSERVABLE."
        )


__all__ = ["VllmBackend"]
