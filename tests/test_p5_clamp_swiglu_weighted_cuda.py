# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA tests for P5-2 clamp_swiglu_weighted."""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.moe import fixtures, oracle
from rl_engine.moe.cuda_provider import (
    ClampSwiGLUWeightedCudaProvider,
)

_HAS_P5_CUDA = bool(
    torch.cuda.is_available()
    and _EXT_AVAILABLE
    and _C is not None
    and hasattr(_C, "clamp_swiglu_weighted_forward")
    and hasattr(_C, "clamp_swiglu_weighted_backward")
)

requires_p5_cuda = pytest.mark.skipif(
    not _HAS_P5_CUDA,
    reason="P5-2 CUDA extension unavailable",
)


def _assert_exact(
    got: torch.Tensor,
    want: torch.Tensor,
) -> None:
    assert got.dtype == want.dtype
    assert got.shape == want.shape
    assert torch.equal(got, want)


@requires_p5_cuda
def test_boundary_fixture_is_byte_exact() -> None:
    provider = ClampSwiGLUWeightedCudaProvider()

    gate, up, p_s = (tensor.cuda() for tensor in fixtures.make_swiglu_boundary_inputs())

    dh = fixtures.make_grad_output(
        "swiglu_boundary",
        tuple(gate.shape),
    ).cuda()

    h_ref, saved_ref = oracle.clamp_swiglu_weighted_fwd(
        gate,
        up,
        p_s,
    )

    grads_ref = oracle.clamp_swiglu_weighted_bwd(
        dh,
        saved_ref,
    )

    h_got, saved_got = provider.clamp_swiglu_weighted_fwd(
        gate,
        up,
        p_s,
    )

    grads_got = provider.clamp_swiglu_weighted_bwd(
        dh,
        saved_got,
    )

    _assert_exact(h_got, h_ref)

    for key in ("g", "u", "sig", "silu"):
        _assert_exact(
            saved_got[key],
            saved_ref[key],
        )

    for got, want in zip(
        grads_got,
        grads_ref,
        strict=True,
    ):
        assert got is not None
        assert want is not None
        _assert_exact(got, want)


@requires_p5_cuda
def test_shared_variant_has_no_weight_no_clamp_and_no_dp_s() -> None:
    provider = ClampSwiGLUWeightedCudaProvider()

    gate = torch.tensor(
        [[12.0, -12.0, 10.0, -10.0]],
        device="cuda",
    )

    up = torch.tensor(
        [[12.0, -12.0, 10.0, -10.0]],
        device="cuda",
    )

    dh = torch.tensor(
        [[1.0, -0.5, 0.25, -2.0]],
        device="cuda",
        dtype=torch.bfloat16,
    )

    h_ref, saved_ref = oracle.clamp_swiglu_weighted_fwd(
        gate,
        up,
        None,
    )

    dgate_ref, dup_ref, dp_ref = oracle.clamp_swiglu_weighted_bwd(
        dh,
        saved_ref,
    )

    h_got, saved_got = provider.clamp_swiglu_weighted_fwd(
        gate,
        up,
        None,
    )

    dgate_got, dup_got, dp_got = provider.clamp_swiglu_weighted_bwd(
        dh,
        saved_got,
    )

    _assert_exact(h_got, h_ref)
    _assert_exact(saved_got["g"], gate)
    _assert_exact(saved_got["u"], up)
    _assert_exact(dgate_got, dgate_ref)
    _assert_exact(dup_got, dup_ref)

    assert dp_got is None
    assert dp_ref is None


@requires_p5_cuda
def test_repeated_runs_are_byte_exact() -> None:
    provider = ClampSwiGLUWeightedCudaProvider()

    generator = torch.Generator().manual_seed(63)

    gate = (
        torch.randn(
            7,
            257,
            generator=generator,
        )
        * 12.0
    ).cuda()

    up = (
        torch.randn(
            7,
            257,
            generator=generator,
        )
        * 12.0
    ).cuda()

    p_s = torch.rand(
        7,
        generator=generator,
    ).cuda()

    dh = (
        torch.randn(
            7,
            257,
            generator=generator,
        )
        .to(torch.bfloat16)
        .cuda()
    )

    h_first, saved_first = provider.clamp_swiglu_weighted_fwd(
        gate,
        up,
        p_s,
    )

    grads_first = provider.clamp_swiglu_weighted_bwd(
        dh,
        saved_first,
    )

    for _ in range(4):
        h_repeat, saved_repeat = provider.clamp_swiglu_weighted_fwd(
            gate,
            up,
            p_s,
        )

        grads_repeat = provider.clamp_swiglu_weighted_bwd(
            dh,
            saved_repeat,
        )

        _assert_exact(h_repeat, h_first)

        for got, want in zip(
            grads_repeat,
            grads_first,
            strict=True,
        ):
            assert got is not None
            assert want is not None
            _assert_exact(got, want)


@requires_p5_cuda
def test_empty_batch_and_fail_closed_validation() -> None:
    provider = ClampSwiGLUWeightedCudaProvider()

    empty = torch.empty(
        0,
        32,
        device="cuda",
    )

    p_s = torch.empty(
        0,
        device="cuda",
    )

    dh = torch.empty(
        0,
        32,
        device="cuda",
        dtype=torch.bfloat16,
    )

    h, saved = provider.clamp_swiglu_weighted_fwd(
        empty,
        empty,
        p_s,
    )

    dgate, dup, dp_s = provider.clamp_swiglu_weighted_bwd(
        dh,
        saved,
    )

    assert h.shape == (0, 32)
    assert h.dtype == torch.bfloat16
    assert dgate.shape == (0, 32)
    assert dup.shape == (0, 32)
    assert dp_s is not None
    assert dp_s.shape == (0,)

    with pytest.raises(
        RuntimeError,
        match="CUDA tensor",
    ):
        provider.clamp_swiglu_weighted_fwd(
            empty.cpu(),
            empty.cpu(),
            p_s.cpu(),
        )

    with pytest.raises(
        ValueError,
        match="share shape",
    ):
        provider.clamp_swiglu_weighted_fwd(
            empty,
            torch.empty(
                0,
                16,
                device="cuda",
            ),
            p_s,
        )

    with pytest.raises(
        TypeError,
        match="p_s must have dtype fp32",
    ):
        provider.clamp_swiglu_weighted_fwd(
            empty,
            empty,
            p_s.to(torch.bfloat16),
        )
