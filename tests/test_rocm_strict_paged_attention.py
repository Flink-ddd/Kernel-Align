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
        requested_out = kwargs.get("out")
        self.calls.append(
            {
                "q": q,
                "k": k.clone(),
                "v": v.clone(),
                "out": requested_out,
                "causal": kwargs.get("causal"),
                "query_position_ids": kwargs.get("query_position_ids"),
                "key_position_ids": kwargs.get("key_position_ids"),
            }
        )

        result_out = torch.zeros(
            q.size(0),
            q.size(1),
            q.size(2),
            _HEAD_DIM,
            dtype=q.dtype,
        )
        if requested_out is not None:
            requested_out.copy_(result_out)
            result_out = requested_out

        class _Result:
            lse = torch.zeros(q.size(0), q.size(1), q.size(2), dtype=torch.float32)
            provenance = {"attention_backend": "aiter.rocm.ck_dense_mha"}

        result = _Result()
        result.out = result_out
        return result

    def forward_decode_with_lse_into(self, q, k, v, *, out, **kwargs) -> Any:
        return self.forward_with_lse(q, k, v, out=out, causal=False, **kwargs)


def _runtime() -> StrictRocmAttentionRuntime:
    return StrictRocmAttentionRuntime(core=_RecordingCore())


def _cache(pages: int, kv_heads: int = 1) -> torch.Tensor:
    total = pages * _PAGE_SIZE * kv_heads * _HEAD_DIM
    return (
        torch.arange(total, dtype=torch.float32)
        .reshape(pages, _PAGE_SIZE, kv_heads, _HEAD_DIM)
        .to(torch.bfloat16)
    )


def _paged_call(
    runtime,
    *,
    page_table,
    seqused_k,
    q_heads=1,
    kv_heads=1,
    pages=4,
    cached_lengths=None,
    page_bounds_epoch=None,
):
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
            cached_lengths=cached_lengths,
            page_bounds_epoch=page_bounds_epoch,
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


def test_paged_decode_reuses_only_runtime_scoped_page_bounds_validation(monkeypatch) -> None:
    runtime = _runtime()
    page_table = torch.tensor([[0]], dtype=torch.int32)
    seqused_k = torch.tensor([4], dtype=torch.int32)
    validation_flags = []
    original = StrictRocmAttentionRuntime._gather_paged_row

    def recording_gather(*args, validate_bounds=True, **kwargs):
        validation_flags.append(validate_bounds)
        return original(*args, validate_bounds=validate_bounds, **kwargs)

    monkeypatch.setattr(
        StrictRocmAttentionRuntime,
        "_gather_paged_row",
        staticmethod(recording_gather),
    )
    epoch = runtime.new_page_bounds_epoch()
    first, _k, _v = _paged_call(
        runtime,
        page_table=page_table,
        seqused_k=seqused_k,
        cached_lengths=(4,),
        page_bounds_epoch=epoch,
    )
    second, _k, _v = _paged_call(
        runtime,
        page_table=page_table,
        seqused_k=seqused_k,
        cached_lengths=(4,),
        page_bounds_epoch=epoch,
    )
    next_epoch, _k, _v = _paged_call(
        runtime,
        page_table=page_table,
        seqused_k=seqused_k,
        cached_lengths=(4,),
        page_bounds_epoch=runtime.new_page_bounds_epoch(),
    )
    unscoped, _k, _v = _paged_call(
        runtime,
        page_table=page_table,
        seqused_k=seqused_k,
        cached_lengths=(4,),
    )

    assert validation_flags == [True, False, True, True]
    assert first.provenance["page_bounds_validation_reused"] is False
    assert second.provenance["page_bounds_validation_reused"] is True
    assert next_epoch.provenance["page_bounds_validation_reused"] is False
    assert unscoped.provenance["page_bounds_validation_reused"] is False


def test_paged_decode_page_bounds_proof_fails_closed_on_metadata_mutation() -> None:
    runtime = _runtime()
    page_table = torch.tensor([[0]], dtype=torch.int32)
    seqused_k = torch.tensor([4], dtype=torch.int32)
    epoch = runtime.new_page_bounds_epoch()
    _paged_call(
        runtime,
        page_table=page_table,
        seqused_k=seqused_k,
        cached_lengths=(4,),
        page_bounds_epoch=epoch,
    )

    page_table.fill_(9)
    with pytest.raises(ValueError, match="outside"):
        _paged_call(
            runtime,
            page_table=page_table,
            seqused_k=seqused_k,
            cached_lengths=(4,),
            page_bounds_epoch=epoch,
        )


def test_paged_decode_revalidates_an_equivalent_metadata_tensor(monkeypatch) -> None:
    runtime = _runtime()
    page_table = torch.tensor([[0]], dtype=torch.int32)
    seqused_k = torch.tensor([4], dtype=torch.int32)
    validation_flags = []
    original = StrictRocmAttentionRuntime._gather_paged_row

    def recording_gather(*args, validate_bounds=True, **kwargs):
        validation_flags.append(validate_bounds)
        return original(*args, validate_bounds=validate_bounds, **kwargs)

    monkeypatch.setattr(
        StrictRocmAttentionRuntime,
        "_gather_paged_row",
        staticmethod(recording_gather),
    )
    epoch = runtime.new_page_bounds_epoch()
    _paged_call(
        runtime,
        page_table=page_table,
        seqused_k=seqused_k,
        cached_lengths=(4,),
        page_bounds_epoch=epoch,
    )
    _paged_call(
        runtime,
        page_table=page_table.clone(),
        seqused_k=seqused_k,
        cached_lengths=(4,),
        page_bounds_epoch=epoch,
    )

    assert validation_flags == [True, True]


def test_paged_decode_validates_every_batch_row_only_once_per_epoch(monkeypatch) -> None:
    runtime = _runtime()
    page_table = torch.tensor([[0], [1]], dtype=torch.int32)
    seqused_k = torch.tensor([4, 4], dtype=torch.int32)
    validation_flags = []
    original = StrictRocmAttentionRuntime._gather_paged_row

    def recording_gather(*args, validate_bounds=True, **kwargs):
        validation_flags.append(validate_bounds)
        return original(*args, validate_bounds=validate_bounds, **kwargs)

    monkeypatch.setattr(
        StrictRocmAttentionRuntime,
        "_gather_paged_row",
        staticmethod(recording_gather),
    )
    epoch = runtime.new_page_bounds_epoch()
    for _ in range(2):
        _paged_call(
            runtime,
            page_table=page_table,
            seqused_k=seqused_k,
            cached_lengths=(4, 4),
            page_bounds_epoch=epoch,
        )

    assert validation_flags == [True, True, False, False]


def test_paged_decode_does_not_cache_a_failed_page_bounds_validation() -> None:
    runtime = _runtime()
    page_table = torch.tensor([[9]], dtype=torch.int32)
    seqused_k = torch.tensor([4], dtype=torch.int32)
    epoch = runtime.new_page_bounds_epoch()

    with pytest.raises(ValueError, match="outside"):
        _paged_call(
            runtime,
            page_table=page_table,
            seqused_k=seqused_k,
            cached_lengths=(4,),
            page_bounds_epoch=epoch,
        )

    page_table.zero_()
    first_valid, _k, _v = _paged_call(
        runtime,
        page_table=page_table,
        seqused_k=seqused_k,
        cached_lengths=(4,),
        page_bounds_epoch=epoch,
    )
    reused, _k, _v = _paged_call(
        runtime,
        page_table=page_table,
        seqused_k=seqused_k,
        cached_lengths=(4,),
        page_bounds_epoch=epoch,
    )

    assert first_valid.provenance["page_bounds_validation_reused"] is False
    assert reused.provenance["page_bounds_validation_reused"] is True


def test_paged_decode_revalidates_inference_metadata_in_a_new_epoch() -> None:
    runtime = _runtime()
    with torch.inference_mode():
        page_table = torch.tensor([[0]], dtype=torch.int32)
        seqused_k = torch.tensor([4], dtype=torch.int32)
        _paged_call(
            runtime,
            page_table=page_table,
            seqused_k=seqused_k,
            cached_lengths=(4,),
            page_bounds_epoch=runtime.new_page_bounds_epoch(),
        )

        # Inference tensors have no version counter. The adapter's fresh epoch
        # is therefore the fail-closed boundary between model forwards.
        page_table.fill_(9)
        with pytest.raises(ValueError, match="outside"):
            _paged_call(
                runtime,
                page_table=page_table,
                seqused_k=seqused_k,
                cached_lengths=(4,),
                page_bounds_epoch=runtime.new_page_bounds_epoch(),
            )


def test_paged_decode_rejects_page_bounds_epoch_from_another_runtime() -> None:
    runtime = _runtime()
    foreign_epoch = _runtime().new_page_bounds_epoch()

    with pytest.raises(ValueError, match="not issued by this ROCm runtime"):
        _paged_call(
            runtime,
            page_table=torch.tensor([[0]], dtype=torch.int32),
            seqused_k=torch.tensor([4], dtype=torch.int32),
            cached_lengths=(4,),
            page_bounds_epoch=foreign_epoch,
        )


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


def test_paged_decode_writes_into_the_callers_output_buffer() -> None:
    core = _RecordingCore()
    runtime = StrictRocmAttentionRuntime(core=core)
    k_cache = _cache(2)
    q = torch.zeros(1, 1, 1, _HEAD_DIM, dtype=torch.bfloat16)
    out = torch.full_like(q, 7)

    with torch.no_grad():
        result = runtime.forward_paged_with_lse(
            q,
            k_cache,
            k_cache + 1,
            page_table=torch.tensor([[0]], dtype=torch.int32),
            seqused_k=torch.tensor([4], dtype=torch.int32),
            max_seqlen_k=_PAGE_SIZE,
            scale=None,
            out=out,
            cached_lengths=(4,),
        )

    assert result.out is out
    assert torch.equal(out, torch.zeros_like(out))
    assert core.calls[0]["out"].data_ptr() == out.data_ptr()
    assert result.provenance["core_output_staging"] == "aiter_direct_caller_group"


@pytest.mark.parametrize(
    ("q_heads", "kv_heads", "expected_lse_cat_parts"),
    [(1, 1, []), (4, 2, [2])],
)
def test_paged_direct_decode_skips_only_singleton_lse_cats(
    monkeypatch,
    q_heads,
    kv_heads,
    expected_lse_cat_parts,
) -> None:
    original_cat = torch.cat
    lse_cat_parts = []

    def recording_cat(tensors, *args, **kwargs):
        tensors = tuple(tensors)
        if tensors and tensors[0].dtype == torch.float32:
            lse_cat_parts.append(len(tensors))
        return original_cat(tensors, *args, **kwargs)

    monkeypatch.setattr(torch, "cat", recording_cat)
    runtime = _runtime()
    k_cache = _cache(2, kv_heads=kv_heads)
    q = torch.zeros(1, q_heads, 1, _HEAD_DIM, dtype=torch.bfloat16)

    with torch.no_grad():
        result = runtime.forward_paged_with_lse(
            q,
            k_cache,
            k_cache + 1,
            page_table=torch.tensor([[0]], dtype=torch.int32),
            seqused_k=torch.tensor([4], dtype=torch.int32),
            max_seqlen_k=_PAGE_SIZE,
            scale=None,
            out=torch.empty_like(q),
            cached_lengths=(4,),
        )

    assert lse_cat_parts == expected_lse_cat_parts
    assert result.lse.shape == (1, q_heads, 1)
    assert result.lse.is_contiguous()
    assert torch.equal(result.lse, torch.zeros_like(result.lse))


def test_paged_decode_writes_each_kv_group_directly_to_its_output_slice() -> None:
    core = _RecordingCore()
    runtime = StrictRocmAttentionRuntime(core=core)
    k_cache = _cache(2, kv_heads=2)
    q = torch.zeros(1, 4, 1, _HEAD_DIM, dtype=torch.bfloat16)
    out = torch.full_like(q, 7)

    with torch.no_grad():
        result = runtime.forward_paged_with_lse(
            q,
            k_cache,
            k_cache + 1,
            page_table=torch.tensor([[0]], dtype=torch.int32),
            seqused_k=torch.tensor([4], dtype=torch.int32),
            max_seqlen_k=_PAGE_SIZE,
            scale=None,
            out=out,
            cached_lengths=(4,),
        )

    assert result.out is out
    assert len(core.calls) == 2
    assert core.calls[0]["out"].data_ptr() == out[:, :2].data_ptr()
    assert core.calls[1]["out"].data_ptr() == out[:, 2:].data_ptr()
    assert torch.equal(out, torch.zeros_like(out))
    assert result.provenance["core_output_staging"] == "aiter_direct_caller_group"


def test_paged_decode_keeps_staging_when_gradient_mode_is_enabled() -> None:
    core = _RecordingCore()
    runtime = StrictRocmAttentionRuntime(core=core)
    k_cache = _cache(2)
    q = torch.zeros(1, 1, 1, _HEAD_DIM, dtype=torch.bfloat16)
    out = torch.full_like(q, 7)

    result = runtime.forward_paged_with_lse(
        q,
        k_cache,
        k_cache + 1,
        page_table=torch.tensor([[0]], dtype=torch.int32),
        seqused_k=torch.tensor([4], dtype=torch.int32),
        max_seqlen_k=_PAGE_SIZE,
        scale=None,
        out=out,
        cached_lengths=(4,),
    )

    assert result.out is out
    assert core.calls[0]["out"] is None
    assert result.provenance["core_output_staging"] == "runtime_group_cat"


def test_paged_staged_decode_keeps_singleton_lse_cats(monkeypatch) -> None:
    original_cat = torch.cat
    lse_cat_parts = []

    def recording_cat(tensors, *args, **kwargs):
        tensors = tuple(tensors)
        if tensors and tensors[0].dtype == torch.float32:
            lse_cat_parts.append(len(tensors))
        return original_cat(tensors, *args, **kwargs)

    monkeypatch.setattr(torch, "cat", recording_cat)
    runtime = _runtime()
    k_cache = _cache(2)
    q = torch.zeros(1, 1, 1, _HEAD_DIM, dtype=torch.bfloat16)
    result = runtime.forward_paged_with_lse(
        q,
        k_cache,
        k_cache + 1,
        page_table=torch.tensor([[0]], dtype=torch.int32),
        seqused_k=torch.tensor([4], dtype=torch.int32),
        max_seqlen_k=_PAGE_SIZE,
        scale=None,
        out=torch.empty_like(q),
        cached_lengths=(4,),
    )

    assert lse_cat_parts == [1, 1, 1]
    assert result.lse.shape == (1, 1, 1)


def test_paged_decode_keeps_staging_when_output_aliases_an_input() -> None:
    core = _RecordingCore()
    runtime = StrictRocmAttentionRuntime(core=core)
    k_cache = _cache(2)
    q = torch.zeros(1, 1, 1, _HEAD_DIM, dtype=torch.bfloat16)

    with torch.no_grad():
        result = runtime.forward_paged_with_lse(
            q,
            k_cache,
            k_cache + 1,
            page_table=torch.tensor([[0]], dtype=torch.int32),
            seqused_k=torch.tensor([4], dtype=torch.int32),
            max_seqlen_k=_PAGE_SIZE,
            scale=None,
            out=q,
            cached_lengths=(4,),
        )

    assert result.out is q
    assert core.calls[0]["out"] is None
    assert result.provenance["core_output_staging"] == "runtime_group_cat"


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
