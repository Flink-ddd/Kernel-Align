# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""ROCm strict Attention acceptance tests that never use a native fallback."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="ROCm GPU is unavailable",
)

from rl_engine.kernels.attention_contract import SplitKVSpec  # noqa: E402
from rl_engine.kernels.ops.rocm.attention.deterministic_attn import (  # noqa: E402
    RLKernelDeterministicAttentionCore,
)
from rl_engine.kernels.ops.rocm.rotary_embedding.rope import RocmDeterministicRoPEOp  # noqa: E402


def _qkv(*, batch: int = 2, sequence: int = 7):
    generator = torch.Generator(device="cpu").manual_seed(942)
    q = torch.randn(batch, 4, sequence, 128, dtype=torch.bfloat16, generator=generator).cuda()
    k = torch.randn(batch, 1, sequence, 128, dtype=torch.bfloat16, generator=generator).cuda()
    v = torch.randn(batch, 1, sequence, 128, dtype=torch.bfloat16, generator=generator).cuda()
    return q, k, v


def test_rocm_attention_core_is_repeat_bitwise_and_no_fallback():
    q, k, v = _qkv()
    positions = torch.arange(q.size(2), device=q.device).expand(q.size(0), -1)
    core = RLKernelDeterministicAttentionCore()
    first = core.forward_with_lse(
        q,
        k,
        v,
        query_position_ids=positions,
        key_position_ids=positions,
    )
    second = core.forward_with_lse(
        q,
        k,
        v,
        query_position_ids=positions,
        key_position_ids=positions,
    )
    assert torch.equal(first.out, second.out)
    assert torch.equal(first.lse, second.lse)
    assert first.provenance["attention_backend"] == "rlkernel.rocm.deterministic_attention"
    assert first.provenance["fallback"] is False
    assert first.provenance["split_kv"]["actual_split_kv_policy"] == "disabled"


def test_rocm_attention_forward_backward_train_rollout_bitwise():
    q, k, v = (tensor.requires_grad_() for tensor in _qkv(batch=1, sequence=5))
    positions = torch.arange(q.size(2), device=q.device).expand(q.size(0), -1)
    core = RLKernelDeterministicAttentionCore()
    train = core.forward_with_lse(
        q,
        k,
        v,
        query_position_ids=positions,
        key_position_ids=positions,
    )
    grad = torch.randn(train.out.shape, dtype=train.out.dtype, device="cpu").cuda()
    (train.out.float() * grad.float()).sum().backward()
    train_grads = tuple(tensor.grad.detach().clone() for tensor in (q, k, v))

    q2, k2, v2 = (tensor.detach().clone().requires_grad_() for tensor in (q, k, v))
    rollout = core.forward_with_lse(
        q2,
        k2,
        v2,
        query_position_ids=positions,
        key_position_ids=positions,
    )
    (rollout.out.float() * grad.float()).sum().backward()
    assert torch.equal(train.out, rollout.out)
    assert torch.equal(train.lse, rollout.lse)
    assert all(
        torch.equal(expected, actual.grad)
        for expected, actual in zip(train_grads, (q2, k2, v2), strict=True)
    )


def test_rocm_rope_is_batch_invariant_and_backward_repeat_bitwise():
    generator = torch.Generator(device="cpu").manual_seed(714)
    x = torch.randn(3, 4, 6, 128, dtype=torch.bfloat16, generator=generator).cuda()
    positions = torch.arange(6, device=x.device).expand(3, -1)
    rope = RocmDeterministicRoPEOp()
    together = rope(x, positions)
    separate = torch.cat([rope(x[index : index + 1], positions[index]) for index in range(3)])
    assert torch.equal(together, separate)
    assert torch.equal(together, rope(x, positions))

    grad = torch.randn(together.shape, dtype=together.dtype, device="cpu").cuda()
    x1 = x.detach().clone().requires_grad_()
    x2 = x.detach().clone().requires_grad_()
    (rope(x1, positions).float() * grad.float()).sum().backward()
    (rope(x2, positions).float() * grad.float()).sum().backward()
    assert torch.equal(x1.grad, x2.grad)


def test_rocm_rope_matches_fp32_rotate_half_reference():
    x = torch.randn(1, 2, 4, 128, dtype=torch.bfloat16, device="cuda")
    positions = torch.arange(4, device=x.device).expand(1, -1)
    actual = RocmDeterministicRoPEOp()(x, positions)
    half = x.size(-1) // 2
    inv_freq = 1.0 / (
        1_000_000.0 ** (torch.arange(half, dtype=torch.float32, device=x.device) / half)
    )
    frequency = positions.float().unsqueeze(-1) * inv_freq
    cos = frequency.cos().unsqueeze(1)
    sin = frequency.sin().unsqueeze(1)
    reference = torch.cat(
        (
            x[..., :half].float() * cos - x[..., half:].float() * sin,
            x[..., half:].float() * cos + x[..., :half].float() * sin,
        ),
        dim=-1,
    ).to(x.dtype)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)


def test_rocm_strict_core_rejects_split_k():
    with pytest.raises(ValueError, match="Split-KV"):
        RLKernelDeterministicAttentionCore(split_kv=SplitKVSpec.fixed(32))
