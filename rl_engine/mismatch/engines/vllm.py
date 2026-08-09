# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The rollout side: vLLM. **Placeholder -- not wired up yet.**

What has to be built here, and why each part is not optional:

1. **Construction.** vLLM decides several things at engine-build time, so a
   switch at ``ENGINE_ARG`` level cannot be changed afterwards -- it costs a
   rebuild, which is what ``RebindCost.ENGINE_REBUILD`` prices in.
2. **Scoring.** ``score()`` returns the per-token logprobs for a fixed sequence.
   For the full-prefill path that means feeding ``prompt + response`` in as one
   prompt and reading ``prompt_logprobs``; for the decode path it means a real
   generation. Which one ran is ``ExecutionPath``, and the two are not equal in
   floating point.
3. **Readback.** Every value below must come back off the live engine object.
   A requested value is not evidence.

The settings that must be pinned and read back, with the pitfall each guards
(see ``schema/pitfalls.py`` and the reference's ``required_settings``):

    enable_chunked_prefill      -> False    guards chunked_prefill_default_on
      llm_engine.scheduler_config.chunked_prefill_enabled
      Defaults to True. Leave it and you believe you measured a full-sequence
      prefill while the engine chunked it.

    max_num_batched_tokens      -> > len(prompt + response)
      llm_engine.scheduler_config.max_num_batched_tokens

    long_prefill_token_threshold -> 0
      llm_engine.scheduler_config.long_prefill_token_threshold
      Non-zero splits long prefills by another route.

    enable_prefix_caching       -> False    guards prompt_logprobs_skips_prefix_cache
      llm_engine.cache_config.enable_prefix_caching
      prompt_logprobs already skips the prefix cache, so the full-prefill path
      has no cache while the decode path does -- unless this is off, the two
      paths are not comparable.

    all_reduce backend          -> record actual, not requested
      vLLM switches between custom IPC, MNNVL and NCCL by world size and
      topology. Only what it actually chose is evidence (gemm.rollout_all_reduce_backend).

Also to record: ``prompt_logprobs`` position 0 is ``None``, which is where the
shift-by-one convention has to be asserted against the training side rather than
assumed (pitfall ``shift_by_one``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from rl_engine.mismatch.schema import ComparisonIdentity, PolicyRole, ReuseKey


@dataclass
class VllmBackend:
    """Scores tokens on a live vLLM engine. **Every method is a stub.**"""

    role: PolicyRole = PolicyRole.ROLLOUT
    engine: Any = None  # vllm.LLM once constructed
    effective_config: dict[str, Any] = field(default_factory=dict)

    def score(
        self,
        role: PolicyRole,
        identity: ComparisonIdentity,
        switch_values: Mapping[str, Any],
        replacement: Callable[..., Any] | None,
    ) -> tuple[Sequence[float], Mapping[str, Any]]:
        raise NotImplementedError(
            "vLLM scoring is not wired up: build the engine with the settings "
            "pinned in this module's docstring, run the requested ExecutionPath, "
            "and return (logprobs, readback) with the effective values read off "
            "the engine."
        )

    def reuse_key(self, switch_values: Mapping[str, Any]) -> ReuseKey:
        raise NotImplementedError(
            "Group the switches by tier so order_cases_by_rebind_cost() can reuse "
            "engines: env/determinism -> process, world size and TP/CP -> "
            "process_group, dtype and backend and KV layout -> engine, batch and "
            "sequence -> request."
        )

    def read_effective_config(self) -> Mapping[str, Any]:
        raise NotImplementedError(
            "Read each pinned setting back off the live engine. A setting with no "
            "readback path can only be recorded UNOBSERVABLE."
        )


__all__ = ["VllmBackend"]
