# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.kernels.ops.pytorch.activation.swiglu import NativeSwiGLUOp

try:
    from rl_engine.kernels.ops.triton.activation.swiglu import TritonSwiGLUOp

    _HAS_TRITON = True
except ImportError:
    TritonSwiGLUOp = None
    _HAS_TRITON = False

try:
    from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
    from rl_engine.kernels.ops.cuda.activation.swiglu import SwiGLUSM90Op

    _HAS_CUDA_SM90_OP = _EXT_AVAILABLE and hasattr(_C, "swiglu_forward_sm90")
except ImportError:
    SwiGLUSM90Op = None
    _HAS_CUDA_SM90_OP = False

_IS_SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
_HAS_CUDA_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
requires_cuda_sm90 = pytest.mark.skipif(
    not (_IS_SM90 and _HAS_CUDA_SM90_OP),
    reason="requires Hopper and the compiled SM90 SwiGLU forward kernel",
)
requires_triton_cuda = pytest.mark.skipif(
    not (_HAS_CUDA_BF16 and _HAS_TRITON), reason="requires CUDA with BF16 support and Triton"
)

_ELEMENTWISE_ATOL = 2e-2
_ELEMENTWISE_RTOL = 1.6e-2
_TP_LOCAL_INTERMEDIATE = 6144


def _inputs(shape, seed=239):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    gate = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    up = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    return gate, up


def _assert_matches_reference(op, gate, up):
    output = op(gate, up)
    reference = NativeSwiGLUOp().forward_fp32(gate, up)
    assert output.shape == gate.shape
    assert output.dtype is torch.bfloat16
    assert output.is_contiguous()
    torch.testing.assert_close(
        output.float(), reference, atol=_ELEMENTWISE_ATOL, rtol=_ELEMENTWISE_RTOL
    )


@requires_cuda_sm90
@pytest.mark.parametrize("shape", [(3, 257), (2, _TP_LOCAL_INTERMEDIATE)])
def test_cuda_sm90_forward_matches_fp32_reference(shape):
    _assert_matches_reference(SwiGLUSM90Op(), *_inputs(shape))


@requires_triton_cuda
@pytest.mark.parametrize("shape", [(3, 257), (2, _TP_LOCAL_INTERMEDIATE)])
def test_triton_forward_matches_fp32_reference(shape):
    _assert_matches_reference(TritonSwiGLUOp(), *_inputs(shape))


@pytest.mark.parametrize("backend", ["cuda", "triton"])
def test_forward_is_batch_and_padding_invariant(backend):
    if backend == "cuda" and not (_IS_SM90 and _HAS_CUDA_SM90_OP):
        pytest.skip("requires Hopper and the compiled SM90 SwiGLU forward kernel")
    if backend == "triton" and not (_HAS_CUDA_BF16 and _HAS_TRITON):
        pytest.skip("requires CUDA with BF16 support and Triton")

    op = SwiGLUSM90Op() if backend == "cuda" else TritonSwiGLUOp()
    gate, up = _inputs((8, 257), seed=240)
    gate_before, up_before = gate.clone(), up.clone()
    full = op(gate, up)
    assert torch.equal(op(gate[3:5], up[3:5]), full[3:5])

    pad_gate, pad_up = _inputs((4, 257), seed=241)
    padded = op(torch.cat((gate, pad_gate)), torch.cat((up, pad_up)))
    assert torch.equal(padded[:8], full)
    assert torch.equal(gate, gate_before)
    assert torch.equal(up, up_before)


@pytest.mark.parametrize("backend", ["cuda", "triton"])
def test_forward_contract_guards(backend):
    if backend == "cuda" and not (_IS_SM90 and _HAS_CUDA_SM90_OP):
        pytest.skip("requires Hopper and the compiled SM90 SwiGLU forward kernel")
    if backend == "triton" and not (_HAS_CUDA_BF16 and _HAS_TRITON):
        pytest.skip("requires CUDA with BF16 support and Triton")

    op = SwiGLUSM90Op() if backend == "cuda" else TritonSwiGLUOp()
    gate, up = _inputs((2, 8), seed=242)
    with pytest.raises(ValueError, match="share shape"):
        op(gate, up[:, :-1])
    with pytest.raises(TypeError, match="bfloat16"):
        op(gate.float(), up.float())

    empty = torch.empty((0, _TP_LOCAL_INTERMEDIATE), device="cuda", dtype=torch.bfloat16)
    assert op(empty, empty).shape == empty.shape

    noncontiguous_gate = gate.t()
    noncontiguous_up = up.t()
    assert not noncontiguous_gate.is_contiguous()
    _assert_matches_reference(op, noncontiguous_gate, noncontiguous_up)


@requires_cuda_sm90
def test_cuda_and_triton_agree_within_elementwise_contract():
    if not _HAS_TRITON:
        pytest.skip("requires Triton")
    gate, up = _inputs((4, 513), seed=243)
    cuda_output = SwiGLUSM90Op()(gate, up)
    triton_output = TritonSwiGLUOp()(gate, up)
    torch.testing.assert_close(
        cuda_output.float(),
        triton_output.float(),
        atol=_ELEMENTWISE_ATOL,
        rtol=_ELEMENTWISE_RTOL,
    )
