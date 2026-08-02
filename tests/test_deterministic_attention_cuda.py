# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Deterministic standard-softmax attention CUDA tests (issue #147).

Covers (per §7 and §8 of the implementation plan):
- Forward correctness via #108 harness (run_operator_suite)
- Backward correctness via #108 harness (check_grad=True, grad_mode="random")
- LSE correctness
- Batch invariance — Axis-A bitwise (slice, position permutation, batch-chunk)
- Sequence-dim chunked-prefill invariance (§7.4)
- Prefill/decode handoff (§7.5)
- KV-cache cat handoff (§7.5.2)
- Scale: None / 0.0 / custom
- GQA dK/dV order validation
- FP64 high-precision gradient comparison
- Gradient batch invariance
- Valid-only vs padded accuracy (not bitwise)
"""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

try:
    from rl_engine.kernels.gtest.op_checks import CandidateSpec, OperatorCase, run_operator_suite
    from rl_engine.kernels.ops.cuda.attention.deterministic_attn import DeterministicAttentionOp
    from rl_engine.kernels.ops.pytorch.attention.standard_attn import NativeAttentionOp

    _OP_AVAILABLE = True
except (ImportError, RuntimeError):
    _OP_AVAILABLE = False

pytestmark = [
    pytestmark,
    pytest.mark.skipif(not _OP_AVAILABLE, reason="CUDA attention op not built"),
]

DEVICE = "cuda"
D = 128


@pytest.fixture
def cuda_op():
    return DeterministicAttentionOp()


@pytest.fixture
def gold_op():
    return NativeAttentionOp()


def _tol(dtype):
    if dtype == torch.bfloat16:
        return 5e-2, 2e-2
    return 1e-3, 1e-3


# =============================================================================
# §8.4 — #108 Harness integration: forward + backward via run_operator_suite
# =============================================================================


def _make_case(name, dtype, hq, hkv, sq, skv, causal, scale=None, padding=False):
    torch.manual_seed(42)
    B = 2
    inputs = {
        "q": torch.randn(B, hq, sq, D, device=DEVICE, dtype=dtype),
        "k": torch.randn(B, hkv, skv, D, device=DEVICE, dtype=dtype),
        "v": torch.randn(B, hkv, skv, D, device=DEVICE, dtype=dtype),
        "causal": causal,
    }
    if scale is not None:
        inputs["scale"] = scale
    if padding:
        mask = torch.ones(B, skv, device=DEVICE, dtype=torch.bool)
        mask[0, skv // 2 :] = False
        mask[1, skv * 3 // 4 :] = False
        inputs["key_padding_mask"] = mask
    gold = NativeAttentionOp()
    return OperatorCase(
        name=name,
        op_class="attention",
        dtype=dtype,
        inputs=inputs,
        gold_fn=gold.forward_fp32,
        grad_input_names=("q", "k", "v"),
    )


def _build_harness_cases():
    return [
        _make_case("bf16-gqa4x1-16x32-causal", torch.bfloat16, 4, 1, 16, 32, True),
        _make_case("bf16-gqa32x8-16x32-causal", torch.bfloat16, 32, 8, 16, 32, True),
        _make_case("bf16-gqa1x1-3x31-nocausal", torch.bfloat16, 1, 1, 3, 31, False),
        _make_case("fp16-gqa4x2-17x33-causal", torch.float16, 4, 2, 17, 33, True),
        _make_case("fp16-gqa32x8-1x64-decode", torch.float16, 32, 8, 1, 64, True),
        _make_case("bf16-gqa4x1-16x16-nocausal", torch.bfloat16, 4, 1, 16, 16, False),
        _make_case("bf16-gqa32x8-64x65-causal", torch.bfloat16, 32, 8, 64, 65, True),
        _make_case("fp16-gqa4x1-65x127-causal", torch.float16, 4, 1, 65, 127, True),
        _make_case("bf16-pad-gqa4x1-8x16", torch.bfloat16, 4, 1, 8, 16, True, padding=True),
        _make_case("bf16-scale0-4x1-8x16", torch.bfloat16, 4, 1, 8, 16, True, scale=0.0),
        _make_case("fp16-scale-custom-4x1-8x16", torch.float16, 4, 1, 8, 16, True, scale=0.05),
    ]


def test_harness_forward():
    """§8.4: run_operator_suite forward — candidate vs gold (accuracy tolerance)."""
    cuda = DeterministicAttentionOp()
    candidate = CandidateSpec(name="cuda-attention", fn=cuda, backend="cuda")
    report = run_operator_suite("attention", candidates=[candidate], cases=_build_harness_cases())
    for cr in report.candidates:
        for case in cr.cases:
            if not case.passed:
                msgs = [
                    o.message or f"idx={o.output_index} max_abs={o.max_abs_error}"
                    for o in case.outputs
                    if not o.passed
                ]
                pytest.fail(f"Forward failed {case.case_name}: {msgs}")
    assert report.passed


def test_harness_backward():
    """§8.4: run_operator_suite backward — grad comparison vs gold fp32 autograd."""
    cuda = DeterministicAttentionOp()
    candidate = CandidateSpec(name="cuda-attention", fn=cuda, backend="cuda")
    report = run_operator_suite(
        "attention",
        candidates=[candidate],
        cases=_build_harness_cases(),
        check_grad=True,
        grad_mode="random",
    )
    for cr in report.candidates:
        for case in cr.cases:
            if not case.passed:
                msgs = [
                    o.message or f"idx={o.output_index} max_abs={o.max_abs_error}"
                    for o in case.outputs
                    if not o.passed
                ]
                pytest.fail(f"Backward failed {case.case_name}: {msgs}")
    assert report.passed


# =============================================================================
# §7.1 — Full correctness sweep (covers Sq=65, Skv=65/127, non-causal sq>skv)
# =============================================================================

SWEEP_CONFIGS = []
for dtype in [torch.bfloat16, torch.float16]:
    for hq, hkv in [(1, 1), (4, 1), (4, 2), (32, 8)]:
        for sq in [1, 3, 16, 17, 64, 65]:
            for skv in [1, 31, 32, 33, 64, 65, 127]:
                for causal in [True, False]:
                    if causal and sq > skv:
                        continue
                    SWEEP_CONFIGS.append((dtype, hq, hkv, sq, skv, causal))


@pytest.mark.parametrize("dtype,hq,hkv,sq,skv,causal", SWEEP_CONFIGS)
def test_forward_sweep(cuda_op, gold_op, dtype, hq, hkv, sq, skv, causal):
    B = 2
    torch.manual_seed(42)
    q = torch.randn(B, hq, sq, D, device=DEVICE, dtype=dtype)
    k = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=dtype)
    v = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=dtype)

    out_cuda = cuda_op.forward(q, k, v, causal=causal)
    out_gold = gold_op.forward_fp32(q, k, v, causal=causal)

    atol, rtol = _tol(dtype)
    torch.testing.assert_close(out_cuda.float(), out_gold.float(), atol=atol, rtol=rtol)


# =============================================================================
# §7.1 — Scale tests (None, 0.0, custom)
# =============================================================================


@pytest.mark.parametrize("scale", [None, 0.0, 0.05])
def test_scale(cuda_op, gold_op, scale):
    B, hq, hkv, sq, skv = 2, 4, 1, 8, 16
    torch.manual_seed(42)
    q = torch.randn(B, hq, sq, D, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)

    out_cuda = cuda_op.forward(q, k, v, causal=True, scale=scale)
    out_gold = gold_op.forward_fp32(q, k, v, causal=True, scale=scale)

    atol, rtol = _tol(torch.bfloat16)
    torch.testing.assert_close(out_cuda.float(), out_gold.float(), atol=atol, rtol=rtol)


# =============================================================================
# Padding and fully-masked row
# =============================================================================


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_forward_with_padding(cuda_op, gold_op, dtype):
    B, hq, hkv, sq, skv = 2, 4, 1, 8, 16
    torch.manual_seed(7)
    q = torch.randn(B, hq, sq, D, device=DEVICE, dtype=dtype)
    k = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=dtype)
    v = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=dtype)
    mask = torch.ones(B, skv, device=DEVICE, dtype=torch.bool)
    mask[0, 10:] = False
    mask[1, 12:] = False

    out_cuda = cuda_op.forward(q, k, v, causal=True, key_padding_mask=mask)
    out_gold = gold_op.forward_fp32(q, k, v, causal=True, key_padding_mask=mask)

    atol, rtol = _tol(dtype)
    torch.testing.assert_close(out_cuda.float(), out_gold.float(), atol=atol, rtol=rtol)


def test_fully_masked_row(cuda_op):
    B, hq, hkv, sq, skv = 1, 1, 1, 2, 4
    q = torch.randn(B, hq, sq, D, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)
    mask = torch.zeros(B, skv, device=DEVICE, dtype=torch.bool)

    out, lse = cuda_op.forward_with_lse(q, k, v, causal=False, key_padding_mask=mask)
    assert (out == 0).all()
    assert (lse == float("-inf")).all()


# =============================================================================
# §7.2/8.5 — LSE correctness
# =============================================================================


def test_lse_correctness(cuda_op):
    B, hq, hkv, sq, skv = 2, 4, 1, 8, 16
    torch.manual_seed(99)
    q = torch.randn(B, hq, sq, D, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)

    _, lse_cuda = cuda_op.forward_with_lse(q, k, v, causal=True)

    scale = 1.0 / math.sqrt(D)
    g = hq // hkv
    k_exp = k.float().repeat_interleave(g, dim=1)
    scores = scale * torch.einsum("bhqd,bhkd->bhqk", q.float(), k_exp)
    sq_idx = torch.arange(sq, device=DEVICE).unsqueeze(1)
    kv_idx = torch.arange(skv, device=DEVICE).unsqueeze(0)
    causal_mask = kv_idx <= (skv - sq + sq_idx)
    scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    lse_gold = torch.logsumexp(scores, dim=-1)

    torch.testing.assert_close(lse_cuda, lse_gold, atol=5e-2, rtol=2e-2)


# =============================================================================
# §7.2 — Valid-only vs padded: near-equal, NOT bitwise
# =============================================================================


def test_valid_only_vs_padded_accuracy(cuda_op):
    """Padding changes reduction width; result is near-equal, not bitwise.

    We use causal=False here so that all valid keys are equally visible
    regardless of Skv. The padded path has extra columns masked to -inf,
    which changes the sum-exp denominator and can shift results slightly.
    """
    B, hq, hkv, sq, skv_valid = 1, 4, 1, 4, 8
    skv_padded = 12
    torch.manual_seed(77)
    q = torch.randn(B, hq, sq, D, device=DEVICE, dtype=torch.bfloat16)
    k_valid = torch.randn(B, hkv, skv_valid, D, device=DEVICE, dtype=torch.bfloat16)
    v_valid = torch.randn(B, hkv, skv_valid, D, device=DEVICE, dtype=torch.bfloat16)

    k_padded = torch.cat(
        [
            k_valid,
            torch.zeros(B, hkv, skv_padded - skv_valid, D, device=DEVICE, dtype=torch.bfloat16),
        ],
        dim=2,
    )
    v_padded = torch.cat(
        [
            v_valid,
            torch.zeros(B, hkv, skv_padded - skv_valid, D, device=DEVICE, dtype=torch.bfloat16),
        ],
        dim=2,
    )
    mask = torch.ones(B, skv_padded, device=DEVICE, dtype=torch.bool)
    mask[:, skv_valid:] = False

    out_valid = cuda_op.forward(q, k_valid, v_valid, causal=False)
    out_padded = cuda_op.forward(q, k_padded, v_padded, causal=False, key_padding_mask=mask)

    atol, rtol = _tol(torch.bfloat16)
    torch.testing.assert_close(out_valid.float(), out_padded.float(), atol=atol, rtol=rtol)


# =============================================================================
# §7.3 — Batch invariance (Axis-A bitwise)
# =============================================================================


def test_batch_invariance_single(cuda_op):
    """Same sample in full batch vs extracted single — bitwise."""
    B, hq, hkv, sq, skv = 4, 4, 1, 8, 16
    torch.manual_seed(11)
    q = torch.randn(B, hq, sq, D, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)

    out_full, lse_full = cuda_op.forward_with_lse(q, k, v, causal=True)

    for i in range(B):
        out_single, lse_single = cuda_op.forward_with_lse(
            q[i : i + 1], k[i : i + 1], v[i : i + 1], causal=True
        )
        assert torch.equal(out_full[i : i + 1], out_single), f"Output batch invariance failed i={i}"
        assert torch.equal(lse_full[i : i + 1], lse_single), f"LSE batch invariance failed i={i}"


def test_batch_invariance_position_permutation(cuda_op):
    """Same sample at different batch positions — bitwise."""
    B, hq, hkv, sq, skv = 4, 32, 8, 8, 16
    torch.manual_seed(12)
    q = torch.randn(B, hq, sq, D, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)

    out_full = cuda_op.forward(q, k, v, causal=True)

    perm = [2, 0, 3, 1]
    q_perm = q[perm]
    k_perm = k[perm]
    v_perm = v[perm]
    out_perm = cuda_op.forward(q_perm, k_perm, v_perm, causal=True)

    for new_pos, orig_pos in enumerate(perm):
        assert torch.equal(
            out_full[orig_pos], out_perm[new_pos]
        ), f"Position permutation invariance failed: orig={orig_pos} new={new_pos}"


def test_batch_invariance_chunk(cuda_op):
    """Batch-dim chunking — bitwise."""
    B, hq, hkv, sq, skv = 4, 4, 1, 8, 16
    torch.manual_seed(13)
    q = torch.randn(B, hq, sq, D, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)

    out_full = cuda_op.forward(q, k, v, causal=True)
    out_chunk = torch.cat(
        [
            cuda_op.forward(q[:2], k[:2], v[:2], causal=True),
            cuda_op.forward(q[2:], k[2:], v[2:], causal=True),
        ],
        dim=0,
    )
    assert torch.equal(out_full, out_chunk), "Batch-chunk invariance failed"


# =============================================================================
# §7.4 — Sequence-dim chunked-prefill invariance
# =============================================================================


@pytest.mark.parametrize(
    "chunk_size,hq,hkv",
    [
        (1, 4, 1),
        (3, 4, 1),
        (8, 4, 1),
        (1, 32, 8),
        (3, 32, 8),
    ],
)
def test_chunked_prefill(cuda_op, chunk_size, hq, hkv):
    B, T = 1, 16
    torch.manual_seed(22)
    q = torch.randn(B, hq, T, D, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(B, hkv, T, D, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, hkv, T, D, device=DEVICE, dtype=torch.bfloat16)

    out_full = cuda_op.forward(q, k, v, causal=True)

    outs = []
    for t in range(0, T, chunk_size):
        c = min(chunk_size, T - t)
        outs.append(
            cuda_op.forward(q[:, :, t : t + c], k[:, :, : t + c], v[:, :, : t + c], causal=True)
        )
    out_chunked = torch.cat(outs, dim=2)
    assert torch.equal(
        out_full, out_chunked
    ), f"Chunked-prefill invariance failed chunk={chunk_size} hq={hq} hkv={hkv}"


@pytest.mark.parametrize("chunk_size", [1, 3, 8])
def test_chunked_prefill_with_padding(cuda_op, chunk_size):
    """§7.4: chunked-prefill with key_padding_mask (mask sliced with Skv)."""
    B, hq, hkv, T = 1, 4, 1, 16
    torch.manual_seed(23)
    q = torch.randn(B, hq, T, D, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(B, hkv, T, D, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, hkv, T, D, device=DEVICE, dtype=torch.bfloat16)
    mask = torch.ones(B, T, device=DEVICE, dtype=torch.bool)
    mask[0, 12:] = False

    out_full = cuda_op.forward(q, k, v, causal=True, key_padding_mask=mask)

    outs = []
    for t in range(0, T, chunk_size):
        c = min(chunk_size, T - t)
        outs.append(
            cuda_op.forward(
                q[:, :, t : t + c],
                k[:, :, : t + c],
                v[:, :, : t + c],
                causal=True,
                key_padding_mask=mask[:, : t + c],
            )
        )
    out_chunked = torch.cat(outs, dim=2)
    assert torch.equal(
        out_full, out_chunked
    ), f"Chunked-prefill with padding failed chunk={chunk_size}"


@pytest.mark.parametrize("chunk_size", [1, 3])
def test_chunked_prefill_lse(cuda_op, chunk_size):
    """§7.4: LSE chunked-prefill invariance."""
    B, hq, hkv, T = 1, 4, 1, 12
    torch.manual_seed(24)
    q = torch.randn(B, hq, T, D, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(B, hkv, T, D, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, hkv, T, D, device=DEVICE, dtype=torch.bfloat16)

    _, lse_full = cuda_op.forward_with_lse(q, k, v, causal=True)

    lses = []
    for t in range(0, T, chunk_size):
        c = min(chunk_size, T - t)
        _, lse_chunk = cuda_op.forward_with_lse(
            q[:, :, t : t + c], k[:, :, : t + c], v[:, :, : t + c], causal=True
        )
        lses.append(lse_chunk)
    lse_chunked = torch.cat(lses, dim=2)
    assert torch.equal(lse_full, lse_chunked), "LSE chunked-prefill invariance failed"


# =============================================================================
# §7.5 — Prefill/decode handoff
# =============================================================================


def test_prefill_decode_slice(cuda_op):
    """§7.5.1: prefill[:, :, -1:] == decode(q[-1:], k_full, v_full)."""
    B, hq, hkv, sq, skv = 1, 4, 1, 8, 16
    torch.manual_seed(33)
    q = torch.randn(B, hq, sq, D, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)

    prefill = cuda_op.forward(q, k, v, causal=True)
    decode = cuda_op.forward(q[:, :, -1:], k, v, causal=True)
    assert torch.equal(prefill[:, :, -1:], decode)


@pytest.mark.parametrize(
    "S_new,hq,hkv",
    [
        (1, 4, 1),
        (1, 32, 8),
        (3, 4, 1),
        (3, 32, 8),
    ],
)
def test_kv_cache_handoff(cuda_op, S_new, hq, hkv):
    """§7.5.2: cat(k_cache, k_new) handoff == prefill tail."""
    B, S_past = 1, 12
    torch.manual_seed(44)
    q_full = torch.randn(B, hq, S_past + S_new, D, device=DEVICE, dtype=torch.bfloat16)
    k_full = torch.randn(B, hkv, S_past + S_new, D, device=DEVICE, dtype=torch.bfloat16)
    v_full = torch.randn(B, hkv, S_past + S_new, D, device=DEVICE, dtype=torch.bfloat16)

    q_new = q_full[:, :, -S_new:]
    prefill_tail = cuda_op.forward(q_full, k_full, v_full, causal=True)[:, :, -S_new:]
    decode_path = cuda_op.forward(q_new, k_full, v_full, causal=True)
    assert torch.equal(decode_path, prefill_tail)


def test_kv_cache_handoff_with_padding(cuda_op):
    """§7.5.2: cat handoff with padding mask."""
    B, hq, hkv, S_past, S_new = 1, 4, 1, 12, 1
    Skv = S_past + S_new
    torch.manual_seed(45)
    q_full = torch.randn(B, hq, Skv, D, device=DEVICE, dtype=torch.bfloat16)
    k_full = torch.randn(B, hkv, Skv, D, device=DEVICE, dtype=torch.bfloat16)
    v_full = torch.randn(B, hkv, Skv, D, device=DEVICE, dtype=torch.bfloat16)
    mask = torch.ones(B, Skv, device=DEVICE, dtype=torch.bool)
    mask[0, 8:10] = False

    q_new = q_full[:, :, -S_new:]
    prefill_tail = cuda_op.forward(q_full, k_full, v_full, causal=True, key_padding_mask=mask)[
        :, :, -S_new:
    ]
    decode_path = cuda_op.forward(q_new, k_full, v_full, causal=True, key_padding_mask=mask)
    assert torch.equal(decode_path, prefill_tail)


# =============================================================================
# §7.6 — Backward
# =============================================================================


def test_backward_smoke(cuda_op):
    """Backward runs and produces gradients with correct shapes."""
    B, hq, hkv, sq, skv = 1, 4, 1, 4, 8
    torch.manual_seed(55)
    q = torch.randn(B, hq, sq, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)

    out = cuda_op.forward(q, k, v, causal=True)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None and q.grad.shape == q.shape
    assert k.grad is not None and k.grad.shape == k.shape
    assert v.grad is not None and v.grad.shape == v.shape


def test_backward_fp64_reference(cuda_op):
    """§7.6: FP64 high-precision gradient comparison."""
    B, hq, hkv, sq, skv = 1, 4, 1, 4, 8
    torch.manual_seed(56)
    q_bf16 = torch.randn(B, hq, sq, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    k_bf16 = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    v_bf16 = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)

    grad_out = torch.randn(B, hq, sq, D, device=DEVICE, dtype=torch.float32)

    out_cuda = cuda_op.forward(q_bf16, k_bf16, v_bf16, causal=True)
    out_cuda.backward(grad_out.to(out_cuda.dtype))
    dq_cuda = q_bf16.grad.float()
    dk_cuda = k_bf16.grad.float()
    dv_cuda = v_bf16.grad.float()

    scale = 1.0 / math.sqrt(D)
    g = hq // hkv
    q64 = q_bf16.detach().double().requires_grad_(True)
    k64 = k_bf16.detach().double().requires_grad_(True)
    v64 = v_bf16.detach().double().requires_grad_(True)
    k64_exp = k64.repeat_interleave(g, dim=1)
    v64_exp = v64.repeat_interleave(g, dim=1)
    scores64 = scale * torch.einsum("bhqd,bhkd->bhqk", q64, k64_exp)
    sq_idx = torch.arange(sq, device=DEVICE).unsqueeze(1)
    kv_idx = torch.arange(skv, device=DEVICE).unsqueeze(0)
    causal_mask = kv_idx <= (skv - sq + sq_idx)
    scores64 = scores64.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    P64 = torch.softmax(scores64, dim=-1)
    out64 = torch.einsum("bhqk,bhkd->bhqd", P64, v64_exp)
    out64.backward(grad_out.double())
    dq_gold = q64.grad.float()
    dk_gold = k64.grad.float()
    dv_gold = v64.grad.float()

    torch.testing.assert_close(dq_cuda, dq_gold, atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(dk_cuda, dk_gold, atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(dv_cuda, dv_gold, atol=5e-2, rtol=2e-2)


def test_gradient_batch_invariance(cuda_op):
    """§7.6: dQ/dK/dV bitwise identical for same sample at different batch positions."""
    B, hq, hkv, sq, skv = 3, 4, 1, 4, 8
    torch.manual_seed(57)
    q = torch.randn(B, hq, sq, D, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)
    grad_out = torch.randn(B, hq, sq, D, device=DEVICE, dtype=torch.bfloat16)

    q_full = q.clone().requires_grad_(True)
    k_full = k.clone().requires_grad_(True)
    v_full = v.clone().requires_grad_(True)
    out_full = cuda_op.forward(q_full, k_full, v_full, causal=True)
    out_full.backward(grad_out)

    for i in range(B):
        qi = q[i : i + 1].clone().requires_grad_(True)
        ki = k[i : i + 1].clone().requires_grad_(True)
        vi = v[i : i + 1].clone().requires_grad_(True)
        out_i = cuda_op.forward(qi, ki, vi, causal=True)
        out_i.backward(grad_out[i : i + 1])
        assert torch.equal(q_full.grad[i : i + 1], qi.grad), f"dQ batch invariance failed i={i}"
        assert torch.equal(k_full.grad[i : i + 1], ki.grad), f"dK batch invariance failed i={i}"
        assert torch.equal(v_full.grad[i : i + 1], vi.grad), f"dV batch invariance failed i={i}"


def test_gqa_dk_dv_order(cuda_op):
    """§7.6/§4.1: GQA dK/dV must follow fixed (hq_local, query_index) order.

    Two checks:
    1. Batch-size invariance: same sample at B=1 vs B=4 gives bitwise-identical dK/dV
       (catches unordered atomics or grid-shape-dependent accumulation).
    2. Correctness vs FP64 gold: with asymmetric per-head Q, a reversed local order
       would produce different numerical results. This catches "deterministic but wrong".
    """
    hq, hkv, sq, skv = 32, 8, 4, 8
    torch.manual_seed(58)
    q_data = torch.randn(1, hq, sq, D, device=DEVICE, dtype=torch.bfloat16)
    k_data = torch.randn(1, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)
    v_data = torch.randn(1, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)
    grad_data = torch.randn(1, hq, sq, D, device=DEVICE, dtype=torch.bfloat16)

    # Check 1: batch-size invariance
    q1 = q_data.clone().requires_grad_(True)
    k1 = k_data.clone().requires_grad_(True)
    v1 = v_data.clone().requires_grad_(True)
    cuda_op.forward(q1, k1, v1, causal=True).backward(grad_data)

    q_batch = q_data.expand(4, -1, -1, -1).contiguous().clone().requires_grad_(True)
    k_batch = k_data.expand(4, -1, -1, -1).contiguous().clone().requires_grad_(True)
    v_batch = v_data.expand(4, -1, -1, -1).contiguous().clone().requires_grad_(True)
    grad_batch = grad_data.expand(4, -1, -1, -1).contiguous()
    cuda_op.forward(q_batch, k_batch, v_batch, causal=True).backward(grad_batch)

    assert torch.equal(k1.grad, k_batch.grad[0:1]), "dK GQA order depends on batch size"
    assert torch.equal(v1.grad, v_batch.grad[0:1]), "dV GQA order depends on batch size"

    # Check 2: correctness vs FP64 reference (catches reversed hq_local order)
    scale = 1.0 / math.sqrt(D)
    g = hq // hkv
    q64 = q_data.detach().double().requires_grad_(True)
    k64 = k_data.detach().double().requires_grad_(True)
    v64 = v_data.detach().double().requires_grad_(True)
    k64_exp = k64.repeat_interleave(g, dim=1)
    v64_exp = v64.repeat_interleave(g, dim=1)
    scores64 = scale * torch.einsum("bhqd,bhkd->bhqk", q64, k64_exp)
    sq_idx = torch.arange(sq, device=DEVICE).unsqueeze(1)
    kv_idx = torch.arange(skv, device=DEVICE).unsqueeze(0)
    causal_mask = kv_idx <= (skv - sq + sq_idx)
    scores64 = scores64.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    P64 = torch.softmax(scores64, dim=-1)
    out64 = torch.einsum("bhqk,bhkd->bhqd", P64, v64_exp)
    out64.backward(grad_data.double())
    dk_gold = k64.grad.float()
    dv_gold = v64.grad.float()

    torch.testing.assert_close(k1.grad.float(), dk_gold, atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(v1.grad.float(), dv_gold, atol=5e-2, rtol=2e-2)
