# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Ascend C SwiGLU coverage: contracts on CPU and kernels on available NPUs."""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.gtest.operator_specs import OP_SPECS
from rl_engine.kernels.ops.pytorch.activation.swiglu import NativeSwiGLUOp
from rl_engine.platforms.device import _npu_available


def _ascend_kernel_available() -> bool:
    if not _npu_available():
        return False
    try:
        from rl_engine.kernels.ops.ascend.activation.swiglu import (
            _C_npu,
            _NPU_EXT_AVAILABLE,
        )
    except Exception:
        return False
    return _NPU_EXT_AVAILABLE and all(
        hasattr(_C_npu, name) for name in ("swiglu_ascend_forward", "swiglu_ascend_backward")
    )


requires_ascend = pytest.mark.skipif(
    not _ascend_kernel_available(),
    reason="Ascend C SwiGLU kernels are not compiled "
    "(needs KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host).",
)


def _op():
    from rl_engine.kernels.ops.ascend.activation.swiglu import SwiGLUAscendOp

    return SwiGLUAscendOp()


def _rand(shape, *, seed: int, dtype: torch.dtype, device: str = "npu") -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator).to(device=device, dtype=dtype)


def _tolerance(dtype: torch.dtype) -> tuple[float, float]:
    if dtype is torch.float32:
        return 1e-5, 1e-5
    if dtype is torch.float16:
        return 1e-3, 1e-3
    if dtype is torch.bfloat16:
        return 2e-2, 1.6e-2
    raise ValueError(f"unsupported dtype: {dtype}")


def test_swiglu_ascend_candidate_is_registered():
    path = OP_SPECS["swiglu"].candidate_paths["ascend"]
    assert path.endswith(".SwiGLUAscendOp")


def test_swiglu_ascend_validation_is_available_without_an_npu():
    from rl_engine.kernels.ops.ascend.activation.swiglu import _validate_inputs

    with pytest.raises(ValueError, match="share shape"):
        _validate_inputs(torch.ones(2, 3), torch.ones(2, 4))
    with pytest.raises(TypeError, match="share dtype"):
        _validate_inputs(torch.ones(8, dtype=torch.float16), torch.ones(8, dtype=torch.bfloat16))
    with pytest.raises(TypeError, match="fp16, bf16, or fp32"):
        _validate_inputs(torch.ones(8, dtype=torch.int32), torch.ones(8, dtype=torch.int32))


@requires_ascend
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(63,), (2, 3, 257), (2, 12288)])
def test_swiglu_ascend_forward_matches_native(dtype: torch.dtype, shape: tuple[int, ...]):
    gate = _rand(shape, seed=1, dtype=dtype)
    up = _rand(shape, seed=2, dtype=dtype)

    actual = _op().forward(gate, up)
    expected = NativeSwiGLUOp().forward_fp32(gate.cpu(), up.cpu())

    atol, rtol = _tolerance(dtype)
    assert actual.dtype is dtype
    torch.testing.assert_close(actual.cpu().float(), expected, atol=atol, rtol=rtol)


@requires_ascend
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_swiglu_ascend_backward_matches_native(dtype: torch.dtype):
    shape = (2, 3, 257)
    gate_cpu = _rand(shape, seed=3, dtype=dtype, device="cpu")
    up_cpu = _rand(shape, seed=4, dtype=dtype, device="cpu")
    dy_cpu = _rand(shape, seed=5, dtype=dtype, device="cpu")

    gate_ref = gate_cpu.float().requires_grad_(True)
    up_ref = up_cpu.float().requires_grad_(True)
    NativeSwiGLUOp().forward_fp32(gate_ref, up_ref).backward(dy_cpu.float())

    gate = gate_cpu.to(device="npu").requires_grad_(True)
    up = up_cpu.to(device="npu").requires_grad_(True)
    _op().forward(gate, up).backward(dy_cpu.to(device="npu"))

    atol, rtol = _tolerance(dtype)
    torch.testing.assert_close(gate.grad.cpu().float(), gate_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(up.grad.cpu().float(), up_ref.grad, atol=atol, rtol=rtol)


@requires_ascend
def test_swiglu_ascend_extreme_finite_gates_do_not_produce_nan():
    gate = torch.tensor([-100.0, -90.0, 0.0, 90.0, 100.0], device="npu")
    up = torch.ones_like(gate)

    actual = _op().forward(gate, up)
    expected = NativeSwiGLUOp().forward_fp32(gate.cpu(), up.cpu())

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=1e-5)


@requires_ascend
def test_swiglu_ascend_batch_and_padding_invariance_forward_backward():
    dtype = torch.bfloat16
    gate = _rand((7, 4, 64), seed=6, dtype=dtype)
    up = _rand((7, 4, 64), seed=7, dtype=dtype)
    dy = _rand(gate.shape, seed=8, dtype=dtype)

    gate_full = gate.detach().clone().requires_grad_(True)
    up_full = up.detach().clone().requires_grad_(True)
    output_full = _op().forward(gate_full, up_full)
    output_full.backward(dy)

    gate_real = gate[:4].detach().clone().requires_grad_(True)
    up_real = up[:4].detach().clone().requires_grad_(True)
    output_real = _op().forward(gate_real, up_real)
    output_real.backward(dy[:4])

    assert torch.equal(output_full[:4], output_real)
    assert torch.equal(gate_full.grad[:4], gate_real.grad)
    assert torch.equal(up_full.grad[:4], up_real.grad)


@requires_ascend
def test_swiglu_ascend_handles_noncontiguous_and_empty_inputs():
    gate = torch.randn(5, 3, device="npu", dtype=torch.bfloat16).T.requires_grad_(True)
    up = torch.randn(5, 3, device="npu", dtype=torch.bfloat16).T.requires_grad_(True)
    assert not gate.is_contiguous() and not up.is_contiguous()
    _op().forward(gate, up).sum().backward()
    assert gate.grad is not None and up.grad is not None

    empty_gate = torch.empty((0, 64), device="npu", dtype=torch.bfloat16, requires_grad=True)
    empty_up = torch.empty_like(empty_gate, requires_grad=True)
    output = _op().forward(empty_gate, empty_up)
    output.sum().backward()
    assert output.shape == empty_gate.shape
    assert empty_gate.grad.shape == empty_gate.shape
    assert empty_up.grad.shape == empty_up.shape
