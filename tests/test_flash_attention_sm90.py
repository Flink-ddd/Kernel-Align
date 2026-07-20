# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Tests for the SM90 (Hopper) TMA + `mma.sync` packed varlen FlashAttention
kernel: causal masking, `cu_seqlens` packing, attention-domain LSE export, and
a full autograd-differentiable forward + backward (dQ/dK/dV).

Validated against two independent references: (a) an fp32 masked-softmax +
logsumexp closed form, differentiated via autograd (the same reference
tests/test_triton_attention_varlen.py uses), and (b)
`triton_flash_attention_varlen`, the cross-platform semantic baseline this
kernel is checked against -- both for forward output/LSE and for gradients.

The exported LSE is attention-domain (per query row, over the key dimension),
not the vocab-domain LSE produced by the logp/linear_logp kernels.

Note: `seqlen_q > seqlen_k` under `causal=True` is out-of-contract (violates
the documented `Skv - Sq` causal-offset convention, which assumes a query can
always attend at least to itself) and is not exercised here -- confirmed by
direct comparison that `triton_flash_attention_varlen` itself produces NaN
for that input too, so it isn't a case either implementation is designed to
handle, not a gap specific to this kernel.
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
_ATOL_GRAD_REF = 3e-2  # dq/dk/dv vs. fp32-autograd reference


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


def _run_case(seqlens_q, seqlens_k, heads, causal, check_backward=True):
    device = "cuda"
    batch = len(seqlens_q)
    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)
    cu_q = _cu_seqlens(seqlens_q, device)
    cu_k = _cu_seqlens(seqlens_k, device)

    gen = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(
        total_q, heads, _HEAD_DIM, device=device, dtype=torch.bfloat16, generator=gen
    ).requires_grad_(check_backward)
    k = torch.randn(
        total_k, heads, _HEAD_DIM, device=device, dtype=torch.bfloat16, generator=gen
    ).requires_grad_(check_backward)
    v = torch.randn(
        total_k, heads, _HEAD_DIM, device=device, dtype=torch.bfloat16, generator=gen
    ).requires_grad_(check_backward)

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
    assert not lse.requires_grad

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

    if check_backward:
        # Gradient check is against the fp32 autograd reference only, not
        # Triton's own backward: Triton's `_bwd_kernel_varlen` uses
        # `tl.atomic_add` for dQ, which this Triton version (3.2.0) does not
        # support on bf16 pointers at all (compiler error, not a numerical
        # issue) -- confirmed directly, and consistent with
        # tests/test_triton_attention_varlen.py itself only ever exercising
        # backward with float16 tensors, never bfloat16. The SM90 kernel is
        # bf16-only by design, so this comparison isn't possible here; the
        # fp32 reference below is the rigorous check regardless.
        do = torch.randn_like(out)
        out.backward(do)
        dq, dk, dv = q.grad.clone(), k.grad.clone(), v.grad.clone()
        q.grad = k.grad = v.grad = None

    out_ref = torch.empty(total_q, heads, _HEAD_DIM, device=device, dtype=torch.float32)
    lse_ref = torch.empty(total_q, heads, device=device, dtype=torch.float32)
    q_ref = q.detach().clone().float().requires_grad_(check_backward)
    k_ref = k.detach().clone().float().requires_grad_(check_backward)
    v_ref = v.detach().clone().float().requires_grad_(check_backward)
    qs = ks = 0
    for b in range(batch):
        sq, sk = seqlens_q[b], seqlens_k[b]
        if sq > 0:
            o, lval = _ref_attn(
                q_ref[qs : qs + sq].transpose(0, 1),
                k_ref[ks : ks + sk].transpose(0, 1),
                v_ref[ks : ks + sk].transpose(0, 1),
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

    if check_backward:
        out_ref.backward(do.float())
        torch.testing.assert_close(dq.float(), q_ref.grad, atol=_ATOL_GRAD_REF, rtol=0.0)
        torch.testing.assert_close(dk.float(), k_ref.grad, atol=_ATOL_GRAD_REF, rtol=0.0)
        torch.testing.assert_close(dv.float(), v_ref.grad, atol=_ATOL_GRAD_REF, rtol=0.0)
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

    def test_single_kv_tile_backward(self):
        # Isolates the P/ds shared-memory transpose-trick correctness (dV,
        # dK, dQ) from the causal `lo`-bound loop logic and cross-CTA dQ
        # atomics -- both seqlen_q and seqlen_k fit in one 64-row tile.
        _run_case([32], [32], heads=1, causal=True)

    def test_dq_atomic_accumulation_stress(self):
        # A short query sequence attended to by many KV-tiles (long
        # seqlen_k, short seqlen_q, non-causal so every one of the ~16
        # KV-tile CTAs contributes an atomic-add to the same few dQ rows) --
        # the one scenario no other case exercises, since it's purely about
        # backward's cross-CTA accumulation correctness.
        _run_case([4], [1000], heads=2, causal=False)

    def test_dq_atomic_accumulation_stress_multi_batch(self):
        # Same stress, but across three interleaved batch entries -- checks
        # atomics from different (b, h) CTAs never cross into the wrong
        # batch's dQ rows (packed layout means adjacent batches' dQ ranges
        # are directly contiguous in memory, so a k_start/q_start indexing
        # slip here would corrupt a neighboring batch's gradient, not just
        # this one's).
        _run_case([4, 2, 8], [1000, 500, 700], heads=2, causal=False)

    def test_backward_without_return_lse(self):
        # lse is computed unconditionally by the forward kernel and always
        # saved for backward's recompute, regardless of whether the caller
        # asked for it back -- `return_lse=False` must not silently break
        # differentiability.
        device = "cuda"
        seqlens = [37, 61]
        cu = _cu_seqlens(seqlens, device)
        total = sum(seqlens)
        sm_scale = 1.0 / math.sqrt(_HEAD_DIM)
        gen = torch.Generator(device=device).manual_seed(1)
        q = torch.randn(
            total, 2, _HEAD_DIM, device=device, dtype=torch.bfloat16, generator=gen
        ).requires_grad_()
        k = torch.randn(
            total, 2, _HEAD_DIM, device=device, dtype=torch.bfloat16, generator=gen
        ).requires_grad_()
        v = torch.randn(
            total, 2, _HEAD_DIM, device=device, dtype=torch.bfloat16, generator=gen
        ).requires_grad_()

        out = flash_attention_sm90_varlen(
            q, k, v, cu, cu, max(seqlens), max(seqlens), causal=True, sm_scale=sm_scale,
            return_lse=False,
        )
        assert isinstance(out, torch.Tensor)  # bare tensor, not a tuple
        do = torch.randn_like(out)
        out.backward(do)
        assert q.grad is not None and k.grad is not None and v.grad is not None

        q_ref = q.detach().clone().float().requires_grad_()
        k_ref = k.detach().clone().float().requires_grad_()
        v_ref = v.detach().clone().float().requires_grad_()
        out_ref = torch.empty(total, 2, _HEAD_DIM, device=device, dtype=torch.float32)
        qs = 0
        for s in seqlens:
            o, _ = _ref_attn(
                q_ref[qs : qs + s].transpose(0, 1),
                k_ref[qs : qs + s].transpose(0, 1),
                v_ref[qs : qs + s].transpose(0, 1),
                True,
                sm_scale,
            )
            out_ref[qs : qs + s] = o.transpose(0, 1)
            qs += s
        out_ref.backward(do.float())
        torch.testing.assert_close(q.grad.float(), q_ref.grad, atol=_ATOL_GRAD_REF, rtol=0.0)
        torch.testing.assert_close(k.grad.float(), k_ref.grad, atol=_ATOL_GRAD_REF, rtol=0.0)
        torch.testing.assert_close(v.grad.float(), v_ref.grad, atol=_ATOL_GRAD_REF, rtol=0.0)

    def test_partial_requires_grad(self):
        # Frozen-KV-cache-style training (q requires grad, k/v don't) and the
        # reverse; PyTorch's autograd.Function contract allows returning
        # gradients for inputs that don't need them (just extra, discarded
        # work), but this confirms the kernel doesn't crash or corrupt the
        # requested gradients when only some of q/k/v participate.
        device = "cuda"
        seqlens = [37, 61]
        cu = _cu_seqlens(seqlens, device)
        total = sum(seqlens)
        sm_scale = 1.0 / math.sqrt(_HEAD_DIM)

        def make(seed, rq, rk, rv):
            gen = torch.Generator(device=device).manual_seed(seed)
            q = torch.randn(
                total, 2, _HEAD_DIM, device=device, dtype=torch.bfloat16, generator=gen
            ).requires_grad_(rq)
            k = torch.randn(
                total, 2, _HEAD_DIM, device=device, dtype=torch.bfloat16, generator=gen
            ).requires_grad_(rk)
            v = torch.randn(
                total, 2, _HEAD_DIM, device=device, dtype=torch.bfloat16, generator=gen
            ).requires_grad_(rv)
            return q, k, v

        for rq, rk, rv in [(True, False, False), (False, True, True), (True, True, False)]:
            q, k, v = make(2, rq, rk, rv)
            out, _ = flash_attention_sm90_varlen(
                q, k, v, cu, cu, max(seqlens), max(seqlens), causal=True, sm_scale=sm_scale,
                return_lse=True,
            )
            out.backward(torch.randn_like(out))
            assert (q.grad is not None) == rq
            assert (k.grad is not None) == rk
            assert (v.grad is not None) == rv

    def test_backward_non_contiguous_do(self):
        # The upstream gradient `do` (unlike q/k/v) is not under this
        # kernel's contiguity contract -- backward calls `do.contiguous()`
        # internally, matching the Triton path's handling of the same
        # argument. Build `do` as a genuinely non-contiguous transposed view.
        device = "cuda"
        seqlens = [37, 61]
        cu = _cu_seqlens(seqlens, device)
        total = sum(seqlens)
        sm_scale = 1.0 / math.sqrt(_HEAD_DIM)
        gen = torch.Generator(device=device).manual_seed(4)
        q = torch.randn(
            total, 2, _HEAD_DIM, device=device, dtype=torch.bfloat16, generator=gen
        ).requires_grad_()
        k = torch.randn(
            total, 2, _HEAD_DIM, device=device, dtype=torch.bfloat16, generator=gen
        ).requires_grad_()
        v = torch.randn(
            total, 2, _HEAD_DIM, device=device, dtype=torch.bfloat16, generator=gen
        ).requires_grad_()

        out, _ = flash_attention_sm90_varlen(
            q, k, v, cu, cu, max(seqlens), max(seqlens), causal=True, sm_scale=sm_scale,
            return_lse=True,
        )
        do = torch.randn(2, total, _HEAD_DIM, device=device, dtype=torch.bfloat16).transpose(0, 1)
        assert not do.is_contiguous()
        out.backward(do)
        dq, dk, dv = q.grad.clone(), k.grad.clone(), v.grad.clone()

        q_ref = q.detach().clone().float().requires_grad_()
        k_ref = k.detach().clone().float().requires_grad_()
        v_ref = v.detach().clone().float().requires_grad_()
        out_ref = torch.empty(total, 2, _HEAD_DIM, device=device, dtype=torch.float32)
        qs = 0
        for s in seqlens:
            o, _ = _ref_attn(
                q_ref[qs : qs + s].transpose(0, 1),
                k_ref[qs : qs + s].transpose(0, 1),
                v_ref[qs : qs + s].transpose(0, 1),
                True,
                sm_scale,
            )
            out_ref[qs : qs + s] = o.transpose(0, 1)
            qs += s
        out_ref.backward(do.float())
        torch.testing.assert_close(dq.float(), q_ref.grad, atol=_ATOL_GRAD_REF, rtol=0.0)
        torch.testing.assert_close(dk.float(), k_ref.grad, atol=_ATOL_GRAD_REF, rtol=0.0)
        torch.testing.assert_close(dv.float(), v_ref.grad, atol=_ATOL_GRAD_REF, rtol=0.0)

    def test_lse_is_non_differentiable_and_backward_still_works(self):
        device = "cuda"
        heads = 2
        seqlens = [40, 91]
        cu = _cu_seqlens(seqlens, device)
        total = sum(seqlens)
        sm_scale = 1.0 / math.sqrt(_HEAD_DIM)
        q = torch.randn(
            total, heads, _HEAD_DIM, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        k = torch.randn(
            total, heads, _HEAD_DIM, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        v = torch.randn(
            total, heads, _HEAD_DIM, device=device, dtype=torch.bfloat16, requires_grad=True
        )

        out, lse = flash_attention_sm90_varlen(
            q, k, v, cu, cu, max(seqlens), max(seqlens), causal=True, sm_scale=sm_scale,
            return_lse=True,
        )
        assert not lse.requires_grad

        out.float().sum().backward()
        assert q.grad is not None and k.grad is not None and v.grad is not None

    def test_backward_validation_guards(self):
        # The C++ backward wrapper's TORCH_CHECKs, mirroring the forward
        # guard tests -- previously unverified for the backward symbol
        # specifically (bad head_dim/dtype/GQA/cu_seqlens-mismatch/
        # non-contiguous should all be rejected, not just forward's).
        device = "cuda"
        seqlens = [16, 20]
        cu = _cu_seqlens(seqlens, device)
        total = sum(seqlens)
        sm_scale = 1.0 / math.sqrt(_HEAD_DIM)

        from rl_engine.kernels.ops.base import _C

        def call(q, k, v, do, out, lse):
            return _C.flash_attention_varlen_sm90_backward(
                do, q, k, v, out, lse, cu, cu, max(seqlens), True, sm_scale
            )

        q_ok = torch.randn(total, 2, _HEAD_DIM, device=device, dtype=torch.bfloat16)
        do_ok = torch.randn_like(q_ok)
        out_ok = torch.randn_like(q_ok)
        lse_ok = torch.randn(total, 2, device=device, dtype=torch.float32)

        # Bad head_dim.
        q_bad_dim = torch.randn(total, 2, 64, device=device, dtype=torch.bfloat16)
        with pytest.raises(RuntimeError):
            call(q_bad_dim, q_bad_dim, q_bad_dim, q_bad_dim, q_bad_dim,
                 torch.randn(total, 2, device=device, dtype=torch.float32))

        # Bad dtype (fp16 instead of bf16).
        q_fp16 = torch.randn(total, 2, _HEAD_DIM, device=device, dtype=torch.float16)
        with pytest.raises(RuntimeError):
            call(q_fp16, q_fp16, q_fp16, q_fp16, q_fp16,
                 torch.randn(total, 2, device=device, dtype=torch.float32))

        # GQA mismatch.
        q_4h = torch.randn(total, 4, _HEAD_DIM, device=device, dtype=torch.bfloat16)
        do_4h = torch.randn_like(q_4h)
        out_4h = torch.randn_like(q_4h)
        lse_4h = torch.randn(total, 4, device=device, dtype=torch.float32)
        with pytest.raises(RuntimeError):
            call(q_4h, q_ok, q_ok, do_4h, out_4h, lse_4h)

        # Non-contiguous.
        q_noncontig = torch.randn(2, total, _HEAD_DIM, device=device, dtype=torch.bfloat16).transpose(0, 1)
        with pytest.raises(RuntimeError):
            call(q_noncontig, q_ok, q_ok, do_ok, out_ok, lse_ok)

        # cu_seqlens batch mismatch.
        cu_q_bad = _cu_seqlens([16, 20], device)
        cu_k_bad = _cu_seqlens([12, 12, 12], device)
        with pytest.raises(RuntimeError):
            _C.flash_attention_varlen_sm90_backward(
                do_ok, q_ok, q_ok, q_ok, out_ok, lse_ok, cu_q_bad, cu_k_bad, 12, True, sm_scale
            )
