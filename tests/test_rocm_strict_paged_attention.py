# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Decode-stage paged Attention on the strict ROCm runtime.

The core is injected, so these run without ROCm. What they pin is the part
that is ours: the page table decides logical KV order, the cached rows reach
the core exactly as a dense prefill over the same tokens would, and the
provenance never claims a native paged kernel.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from rl_engine.kernels.attention_contract import (
    STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID,
    STRICT_ATTENTION_ROCM_SCHEDULE_ID,
)
from rl_engine.kernels.ops.rocm.attention.strict_runtime import StrictRocmAttentionRuntime

_HEAD_DIM = 8
_PAGE_SIZE = 4


class _RecordingCore:
    """Dense core stand-in that records exactly what each launch consumed."""

    core_id = STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID
    strict_schedule = STRICT_ATTENTION_ROCM_SCHEDULE_ID
    backend_id = "aiter.rocm.ck_dense_mha"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def forward_with_lse(self, q, k, v, **kwargs) -> Any:
        self.calls.append(
            {
                "q": q,
                "k": k.clone(),
                "v": v.clone(),
                "causal": kwargs.get("causal"),
                "query_position_ids": kwargs.get("query_position_ids"),
                "key_position_ids": kwargs.get("key_position_ids"),
            }
        )

        class _Result:
            out = torch.zeros(q.size(0), q.size(1), q.size(2), _HEAD_DIM, dtype=q.dtype)
            lse = torch.zeros(q.size(0), q.size(1), q.size(2), dtype=torch.float32)
            provenance = {"attention_backend": "aiter.rocm.ck_dense_mha"}

        return _Result()


def _runtime() -> StrictRocmAttentionRuntime:
    return StrictRocmAttentionRuntime(core=_RecordingCore())


def _cache(pages: int, kv_heads: int = 1) -> torch.Tensor:
    total = pages * _PAGE_SIZE * kv_heads * _HEAD_DIM
    return (
        torch.arange(total, dtype=torch.float32)
        .reshape(pages, _PAGE_SIZE, kv_heads, _HEAD_DIM)
        .to(torch.bfloat16)
    )


def _paged_call(runtime, *, page_table, seqused_k, q_heads=1, kv_heads=1, pages=4):
    k_cache = _cache(pages, kv_heads)
    v_cache = _cache(pages, kv_heads) + 1
    q = torch.zeros(page_table.size(0), q_heads, 1, _HEAD_DIM, dtype=torch.bfloat16)
    return (
        runtime.forward_paged_with_lse(
            q,
            k_cache,
            v_cache,
            page_table=page_table,
            seqused_k=seqused_k,
            max_seqlen_k=page_table.size(1) * _PAGE_SIZE,
            scale=None,
        ),
        k_cache,
        v_cache,
    )


@pytest.fixture(autouse=True)
def _pretend_rocm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(StrictRocmAttentionRuntime, "_require_rocm", staticmethod(lambda t: None))


def test_paged_decode_gathers_kv_in_logical_not_physical_order() -> None:
    """A shuffled page table must still produce logical KV order.

    This is the property that makes decode replay comparable with prefill: if
    physical page order leaked through, the same logical sequence would produce
    different arithmetic depending on how the allocator handed out pages.
    """

    core = _RecordingCore()
    runtime = StrictRocmAttentionRuntime(core=core)
    # Logical tokens 0..7 live on physical pages 3 then 1.
    page_table = torch.tensor([[3, 1]], dtype=torch.int32)
    _result, k_cache, _v_cache = _paged_call(
        runtime,
        page_table=page_table,
        seqused_k=torch.tensor([8], dtype=torch.int32),
    )

    assert len(core.calls) == 1
    gathered_k = core.calls[0]["k"]
    assert gathered_k.shape == (1, 1, 8, _HEAD_DIM)

    expected = torch.cat((k_cache[3], k_cache[1]), dim=0)  # [8, H, D] logical order
    assert torch.equal(gathered_k[0].permute(1, 0, 2), expected)


def test_paged_decode_truncates_to_the_cached_length() -> None:
    core = _RecordingCore()
    runtime = StrictRocmAttentionRuntime(core=core)
    page_table = torch.tensor([[0, 1]], dtype=torch.int32)

    _paged_call(
        runtime,
        page_table=page_table,
        seqused_k=torch.tensor([5], dtype=torch.int32),
    )

    # Five cached tokens span two pages but must not expose the page tail.
    assert core.calls[0]["k"].shape == (1, 1, 5, _HEAD_DIM)
    assert core.calls[0]["v"].shape == (1, 1, 5, _HEAD_DIM)
    # The launch is non-causal, so the core is handed no position ids; the
    # truncation above is what bounds the launch to the cached prefix.
    assert core.calls[0]["key_position_ids"] is None


def test_paged_decode_is_not_causal_within_a_launch() -> None:
    """Decode attends over the whole cached prefix, so the launch is not causal."""

    core = _RecordingCore()
    runtime = StrictRocmAttentionRuntime(core=core)

    _paged_call(
        runtime,
        page_table=torch.tensor([[0]], dtype=torch.int32),
        seqused_k=torch.tensor([4], dtype=torch.int32),
    )

    assert core.calls[0]["causal"] is False


def test_paged_decode_keeps_one_kv_group_per_launch() -> None:
    """The TP-degree invariance mechanism must survive into the paged path."""

    core = _RecordingCore()
    runtime = StrictRocmAttentionRuntime(core=core)

    result, _k, _v = _paged_call(
        runtime,
        page_table=torch.tensor([[0], [1]], dtype=torch.int32),
        seqused_k=torch.tensor([4, 4], dtype=torch.int32),
        q_heads=4,
        kv_heads=2,
    )

    # Two rows x two KV groups.
    assert len(core.calls) == 4
    assert result.provenance["core_launch_count"] == 4
    for call in core.calls:
        assert call["k"].size(1) == 1  # exactly one KV group per launch
        assert call["q"].size(1) == 2  # its two Q heads
    assert result.provenance["launch_granularity"] == "one_batch_row_one_kv_group"
    assert result.provenance["tp_degree_invariant"] is True


def test_paged_decode_provenance_does_not_claim_a_paged_kernel() -> None:
    """The gather is the implementation; the provenance must say so."""

    runtime = _runtime()
    result, _k, _v = _paged_call(
        runtime,
        page_table=torch.tensor([[0]], dtype=torch.int32),
        seqused_k=torch.tensor([4], dtype=torch.int32),
    )

    assert result.provenance["paged_kernel"] == "none"
    assert result.provenance["paged_execution"] == "logical_kv_gather_then_dense_core"
    assert result.provenance["split_kv"] == "disabled"
    assert result.provenance["strict_schedule"] == STRICT_ATTENTION_ROCM_SCHEDULE_ID
    assert result.provenance["communication_executed"] is False
    assert result.provenance["query_schedule"] == "paged_single_query_batch"


@pytest.mark.parametrize(
    ("page_table", "seqused_k", "match"),
    [
        (torch.tensor([[9]], dtype=torch.int32), torch.tensor([4], dtype=torch.int32), "outside"),
        (torch.tensor([[0]], dtype=torch.int32), torch.tensor([0], dtype=torch.int32), "positive"),
        (
            torch.tensor([[0]], dtype=torch.int32),
            torch.tensor([9], dtype=torch.int32),
            "within max_seqlen_k",
        ),
    ],
)
def test_paged_decode_fails_closed_on_bad_metadata(page_table, seqused_k, match) -> None:
    runtime = _runtime()
    with pytest.raises(ValueError, match=match):
        _paged_call(runtime, page_table=page_table, seqused_k=seqused_k)


def test_paged_decode_rejects_a_mismatched_out_buffer() -> None:
    runtime = _runtime()
    k_cache = _cache(2)
    q = torch.zeros(1, 1, 1, _HEAD_DIM, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="same shape as q"):
        runtime.forward_paged_with_lse(
            q,
            k_cache,
            k_cache + 1,
            page_table=torch.tensor([[0]], dtype=torch.int32),
            seqused_k=torch.tensor([4], dtype=torch.int32),
            max_seqlen_k=_PAGE_SIZE,
            scale=None,
            out=torch.zeros(2, 1, 1, _HEAD_DIM, dtype=torch.bfloat16),
        )


def test_rocm_registry_does_not_claim_decode_before_a_caller_routes_to_it() -> None:
    """The paged entry point exists, but nothing dispatches to it yet.

    The Vime provider always calls ``forward_with_lse`` and builds its contract
    with ``kv_cache=None``, so no decode request can reach the paged path.
    Claiming the mode here would let the cross-config binding accept a decode
    path that never runs. Flip this together with the dispatch wiring.
    """

    from rl_engine.kernels.attention_contract import AttentionMode
    from rl_engine.kernels.registry import KernelRegistry, OpBackend

    capabilities = KernelRegistry()._attention_capabilities
    capability = capabilities.get(OpBackend.ROCM_STRICT_ATTENTION)
    if capability is None:
        pytest.skip("AITER is unavailable, so the strict ROCm backend is not registered")

    assert AttentionMode.DECODE not in capability.modes
    assert capability.supports_kv_cache is False
    # Whatever the modes, the gather must keep the Split-KV claims intact.
    assert capability.supports_split_kv_disabled is True
    assert capability.supports_split_kv_fixed is False
    assert capability.supports_split_kv_auto is False
