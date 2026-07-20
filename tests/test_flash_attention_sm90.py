# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Tests for the SM90 (Hopper) TMA + `mma.sync` packed varlen FlashAttention
forward kernel: causal masking, `cu_seqlens` packing, attention-domain LSE
export.

Forward-only this milestone -- no backward/autograd (see
docs/operators/attention-varlen.md's Known Limitations). Validated against two
independent references: (a) an fp32 masked-softmax + logsumexp closed form
(the same reference tests/test_triton_attention_varlen.py uses), and (b)
`triton_flash_attention_varlen`, the cross-platform semantic baseline this
kernel is checked against.

The exported LSE is attention-domain (per query row, over the key dimension),
not the vocab-domain LSE produced by the logp/linear_logp kernels.
"""

import math

import pytest
import torch

from rl_engine.kernels.ops.cuda.attention.flash_attn_sm90 import flash_attention_sm90_varlen
from rl_engine.kernels.ops.triton.triton_attn import triton_flash_attention_varlen


def _sm90_available():
    """SM90 forward needs a Hopper GPU and the kernel compiled into the extension."""
    if not torch.cuda.is_available():
        return False
    try:
        from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE

        if not (_EXT_AVAILABLE and hasattr(_C, "flash_attention_varlen_sm90")):
            return False
    except Exception:  # pragma: no cover
        return False
    return torch.cuda.get_device_capability()[0] == 9


requires_sm90 = pytest.mark.skipif(
    not _sm90_available(),
    reason="flash_attention_varlen_sm90 requires a Hopper (sm_90) GPU with the extension "
    "built KERNEL_ALIGN_FORCE_SM90=1.",
)

_HEAD_DIM = 128  # only head_dim supported by the SM90 kernel this milestone

_ATOL_OUT_REF = 2e-2  # bf16-mma kernel vs. independent fp32 reference
_ATOL_LSE_REF = 2e-2
_ATOL_OUT_TRITON = 3e-2  # two independent bf16-precision compute paths, different tile schedules
_ATOL_LSE_TRITON = 3e-2


def _ref_attn(q, k, v, causal, sm_scale):
    """[H, Sq, D] / [H, Skv, D] fp32 masked-softmax reference with LSE."""
    H, Sq, D = q.shape
    Skv = k.shape[1]
    scores = torch.einsum("hqd,hkd->hqk", q, k) * sm_scale
    if causal:
        mask = torch.triu(
            torch.ones(Sq, Skv, dtype=torch.bool, device=q.device), diagonal=Skv - Sq + 1
        )
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hqk,hkd->hqd", probs, v)
    lse = torch.logsumexp(scores, dim=-1)
    return out, lse


def _cu_seqlens(seqlens, device):
    return torch.tensor(
        [0, *torch.tensor(seqlens).cumsum(0).tolist()], dtype=torch.int32, device=device
    )


def _run_case(seqlens_q, seqlens_k, heads, causal):
    device = "cuda"
    batch = len(seqlens_q)
    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)
    cu_q = _cu_seqlens(seqlens_q, device)
    cu_k = _cu_seqlens(seqlens_k, device)

    gen = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(total_q, heads, _HEAD_DIM, device=device, dtype=torch.bfloat16, generator=gen)
    k = torch.randn(total_k, heads, _HEAD_DIM, device=device, dtype=torch.bfloat16, generator=gen)
    v = torch.randn(total_k, heads, _HEAD_DIM, device=device, dtype=torch.bfloat16, generator=gen)

    sm_scale = 1.0 / math.sqrt(_HEAD_DIM)
    out, lse = flash_attention_sm90_varlen(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max(seqlens_q),
        max(seqlens_k),
        causal=causal,
        sm_scale=sm_scale,
        return_lse=True,
    )
    assert out.shape == (total_q, heads, _HEAD_DIM)
    assert out.dtype == torch.bfloat16
    assert lse.shape == (total_q, heads)
    assert lse.dtype == torch.float32

    out_triton, lse_triton = triton_flash_attention_varlen(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max(seqlens_q),
        max(seqlens_k),
        causal=causal,
        sm_scale=sm_scale,
        return_lse=True,
    )

    out_ref = torch.empty(total_q, heads, _HEAD_DIM, device=device, dtype=torch.float32)
    lse_ref = torch.empty(total_q, heads, device=device, dtype=torch.float32)
    qs = ks = 0
    for b in range(batch):
        sq, sk = seqlens_q[b], seqlens_k[b]
        if sq > 0:
            o, lval = _ref_attn(
                q[qs : qs + sq].float().transpose(0, 1),
                k[ks : ks + sk].float().transpose(0, 1),
                v[ks : ks + sk].float().transpose(0, 1),
                causal,
                sm_scale,
            )
            out_ref[qs : qs + sq] = o.transpose(0, 1)
            lse_ref[qs : qs + sq] = lval.transpose(0, 1)
        qs += sq
        ks += sk

    torch.testing.assert_close(out.float(), out_ref, atol=_ATOL_OUT_REF, rtol=0.0)
    torch.testing.assert_close(lse, lse_ref, atol=_ATOL_LSE_REF, rtol=0.0)
    torch.testing.assert_close(out.float(), out_triton.float(), atol=_ATOL_OUT_TRITON, rtol=0.0)
    torch.testing.assert_close(lse, lse_triton, atol=_ATOL_LSE_TRITON, rtol=0.0)


@requires_sm90
class TestFlashAttentionSM90Varlen:
    def test_prefill_uneven_seqlens_not_block_aligned(self):
        # BLOCK_Q/BLOCK_KV are 64/64; deliberately not multiples of either.
        _run_case([37, 128, 200, 5], [37, 128, 200, 5], heads=4, causal=True)

    def test_non_causal(self):
        _run_case([37, 130, 61], [37, 130, 61], heads=2, causal=False)

    def test_decode_style_sq_less_than_skv(self):
        # Small new-query chunk (e.g. rollout decode step) against a longer KV cache,
        # varying independently per sequence in the batch.
        _run_case([1, 3, 1], [50, 91, 17], heads=4, causal=True)

    def test_head_dim_128(self):
        # head_dim=128 is the only dim this kernel supports -- kept as its own
        # named case for documentation value even though every other case
        # above already uses it.
        _run_case([100, 260], [100, 260], heads=2, causal=True)

    def test_zero_length_sequence_in_batch(self):
        # A fully-masked / empty response is a real occurrence in packed RL
        # batches; exercises the per-CTA early-return-on-seqlen_q==0 path.
        _run_case([0, 128, 0, 37], [50, 128, 0, 37], heads=2, causal=True)

    def test_dense_vs_varlen_consistency(self):
        # batch>1 packed in one call must match calling the same kernel once
        # per sequence (batch=1 each time) -- confirms the packed (b,h) grid
        # decomposition doesn't leak state across CTAs.
        device = "cuda"
        heads = 2
        seqlens = [40, 91, 17]
        sm_scale = 1.0 / math.sqrt(_HEAD_DIM)
        total = sum(seqlens)
        gen = torch.Generator(device=device).manual_seed(1)
        q = torch.randn(total, heads, _HEAD_DIM, device=device, dtype=torch.bfloat16, generator=gen)
        k = torch.randn(total, heads, _HEAD_DIM, device=device, dtype=torch.bfloat16, generator=gen)
        v = torch.randn(total, heads, _HEAD_DIM, device=device, dtype=torch.bfloat16, generator=gen)
        cu = _cu_seqlens(seqlens, device)

        out_packed, lse_packed = flash_attention_sm90_varlen(
            q, k, v, cu, cu, max(seqlens), max(seqlens), causal=True, sm_scale=sm_scale,
            return_lse=True,
        )

        qs = 0
        for s in seqlens:
            q_i = q[qs : qs + s].contiguous()
            k_i = k[qs : qs + s].contiguous()
            v_i = v[qs : qs + s].contiguous()
            cu_i = torch.tensor([0, s], dtype=torch.int32, device=device)
            out_i, lse_i = flash_attention_sm90_varlen(
                q_i, k_i, v_i, cu_i, cu_i, s, s, causal=True, sm_scale=sm_scale, return_lse=True
            )
            torch.testing.assert_close(
                out_packed[qs : qs + s].float(), out_i.float(), atol=1e-3, rtol=0.0
            )
            torch.testing.assert_close(lse_packed[qs : qs + s], lse_i, atol=1e-3, rtol=0.0)
            qs += s

    def test_non_contiguous_rejected(self):
        # Deliberate divergence from the Triton path (which tolerates
        # non-contiguous q/k/v, see test_non_contiguous_q_k_v_backward in
        # tests/test_triton_attention_varlen.py): the compiled SM90 kernel's
        # TMA descriptors assume a fixed contiguous row stride, so the C++
        # wrapper TORCH_CHECKs contiguity and raises rather than silently
        # calling .contiguous() to hide a caller bug.
        device = "cuda"
        heads = 2
        seqlens = [37, 61]
        total = sum(seqlens)
        cu = _cu_seqlens(seqlens, device)
        base = torch.randn(heads, total, _HEAD_DIM, device=device, dtype=torch.bfloat16)
        q = base.transpose(0, 1)
        assert not q.is_contiguous()

        from rl_engine.kernels.ops.base import _C

        with pytest.raises(RuntimeError):
            _C.flash_attention_varlen_sm90(
                q, q, q, cu, cu, max(seqlens), max(seqlens), True, 1.0 / math.sqrt(_HEAD_DIM)
            )

    def test_block_aligned_seqlens(self):
        # BLOCK_Q/BLOCK_KV are 64/64; every other case deliberately avoids
        # exact multiples. An off-by-one in the `row < seqlen_q` boundary
        # guard (e.g. `<=` instead of `<`) would only surface when a Q-tile
        # is exactly full -- no partial tail row to mask -- so this needs its
        # own case distinct from the non-aligned ones.
        _run_case([64, 128, 192], [64, 128, 192], heads=2, causal=True)
        _run_case([64, 128, 192], [64, 128, 192], heads=2, causal=False)

    def test_long_sequence_many_kv_iterations(self):
        # STAGES=2 double-buffering is only exercised across a handful of
        # wraparounds by the other cases (seqlen_k up to 260 -> ~5 KV tiles).
        # Push into the dozens of iterations to stress the mbarrier
        # prefetch/consume pipeline for a buffer-reuse bug that only shows up
        # after many wraparounds.
        _run_case([1536, 777], [1536, 777], heads=2, causal=True)

    def test_single_head(self):
        # H=1 collapses the blockIdx.y = b*H+h decomposition to pure `b`;
        # make sure that degenerate case (division/modulo by 1) isn't broken.
        _run_case([37, 128, 61], [37, 128, 61], heads=1, causal=True)

    def test_many_batch_entries(self):
        # Stress cu_seqlens indexing / grid.y decomposition across more
        # batch entries than any other case (8, mixed short/long/zero).
        _run_case(
            [3, 0, 64, 129, 1, 200, 0, 17],
            [3, 0, 64, 129, 1, 200, 0, 17],
            heads=2,
            causal=True,
        )

    def test_invalid_head_dim_rejected(self):
        # D=128 is the only head_dim this milestone; the C++ wrapper should
        # reject anything else rather than silently misinterpreting the
        # tensor map layout.
        device = "cuda"
        seqlens = [16, 20]
        cu = _cu_seqlens(seqlens, device)
        total = sum(seqlens)
        q = torch.randn(total, 2, 64, device=device, dtype=torch.bfloat16)

        from rl_engine.kernels.ops.base import _C

        with pytest.raises(RuntimeError):
            _C.flash_attention_varlen_sm90(
                q, q, q, cu, cu, max(seqlens), max(seqlens), True, 1.0 / math.sqrt(64)
            )

    def test_invalid_dtype_rejected(self):
        device = "cuda"
        seqlens = [16, 20]
        cu = _cu_seqlens(seqlens, device)
        total = sum(seqlens)
        q = torch.randn(total, 2, _HEAD_DIM, device=device, dtype=torch.float16)

        from rl_engine.kernels.ops.base import _C

        with pytest.raises(RuntimeError):
            _C.flash_attention_varlen_sm90(
                q, q, q, cu, cu, max(seqlens), max(seqlens), True, 1.0 / math.sqrt(_HEAD_DIM)
            )

    def test_gqa_mismatch_rejected(self):
        # k/v with fewer heads than q (GQA) is unsupported; the wrapper
        # should reject rather than reading past k/v's head dimension.
        device = "cuda"
        seqlens = [16, 20]
        cu = _cu_seqlens(seqlens, device)
        total = sum(seqlens)
        q = torch.randn(total, 4, _HEAD_DIM, device=device, dtype=torch.bfloat16)
        kv = torch.randn(total, 2, _HEAD_DIM, device=device, dtype=torch.bfloat16)

        from rl_engine.kernels.ops.base import _C

        with pytest.raises(RuntimeError):
            _C.flash_attention_varlen_sm90(
                q, kv, kv, cu, cu, max(seqlens), max(seqlens), True, 1.0 / math.sqrt(_HEAD_DIM)
            )

    def test_cu_seqlens_batch_mismatch_rejected(self):
        device = "cuda"
        q = torch.randn(36, 2, _HEAD_DIM, device=device, dtype=torch.bfloat16)
        cu_q = _cu_seqlens([16, 20], device)  # batch=2
        cu_k = _cu_seqlens([12, 12, 12], device)  # batch=3

        from rl_engine.kernels.ops.base import _C

        with pytest.raises(RuntimeError):
            _C.flash_attention_varlen_sm90(
                q, q, q, cu_q, cu_k, 20, 12, True, 1.0 / math.sqrt(_HEAD_DIM)
            )

    def test_wrapper_falls_back_for_unsupported_dtype(self):
        # End-to-end check of the Python-level dispatch gate (`_supported`),
        # not just that the raw C++ op rejects bad input: an fp16 tensor
        # should silently route through `flash_attention_sm90_varlen` to the
        # Triton fallback and still produce a correct result, not raise.
        device = "cuda"
        heads = 2
        seqlens = [37, 61]
        total = sum(seqlens)
        cu = _cu_seqlens(seqlens, device)
        sm_scale = 1.0 / math.sqrt(_HEAD_DIM)
        gen = torch.Generator(device=device).manual_seed(2)
        q = torch.randn(total, heads, _HEAD_DIM, device=device, dtype=torch.float16, generator=gen)
        k = torch.randn(total, heads, _HEAD_DIM, device=device, dtype=torch.float16, generator=gen)
        v = torch.randn(total, heads, _HEAD_DIM, device=device, dtype=torch.float16, generator=gen)

        out, lse = flash_attention_sm90_varlen(
            q, k, v, cu, cu, max(seqlens), max(seqlens), causal=True, sm_scale=sm_scale,
            return_lse=True,
        )

        out_triton, lse_triton = triton_flash_attention_varlen(
            q, k, v, cu, cu, max(seqlens), max(seqlens), causal=True, sm_scale=sm_scale,
            return_lse=True,
        )
        # Should be the *same* Triton call under the hood (fp16 is unsupported
        # by the fused kernel), so this should match tightly, not just within
        # the cross-implementation tolerance used elsewhere in this file.
        torch.testing.assert_close(out, out_triton)
        torch.testing.assert_close(lse, lse_triton)
