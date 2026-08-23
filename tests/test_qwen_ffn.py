# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""First-stage tests for the CUDA/ROCm deterministic Qwen3 FFN.

The CPU tests replace the native extension with a small PyTorch stub.  This
keeps the public API, HuggingFace weight layout, autograd wiring, and BF16
stage-boundary contract testable on build hosts without a GPU.  GPU checks are
single-device only; distributed topology equivalence belongs to the later
collective acceptance phase.
"""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F

import rl_engine.kernels.ops.pytorch.ffn.ffn as ffn_module
from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.pytorch.ffn import qwen3_ffn as public_qwen3_ffn
from rl_engine.kernels.ops.pytorch.ffn.ffn import (
    QWEN3_8B_HIDDEN_SIZE,
    QWEN3_8B_INTERMEDIATE_SIZE,
    qwen3_ffn,
)
from rl_engine.kernels.ops.triton.ffn import qwen3_ffn_triton
from rl_engine.platforms.device import device_ctx

_REQUIRED_SYMBOLS = (
    "det_gemm_fwd",
    "det_gemm_db",
    "swiglu_forward",
    "swiglu_backward",
)
_IS_ROCM = getattr(torch.version, "hip", None) is not None
_IS_GPU = torch.cuda.is_available() and device_ctx.device.type == "cuda"
_HAS_GPU_FFN = (
    _IS_GPU
    and _EXT_AVAILABLE
    and _C is not None
    and all(hasattr(_C, name) for name in _REQUIRED_SYMBOLS)
)
_GPU_NAME = "ROCm" if _IS_ROCM else "CUDA"

requires_gpu_ffn = pytest.mark.skipif(
    not _HAS_GPU_FFN,
    reason=f"{_GPU_NAME} FFN requires a GPU and the GEMM/SwiGLU extension symbols",
)


def _randn(shape, *, seed: int, dtype=torch.float32, device="cpu"):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn(*shape, generator=generator, dtype=torch.float32) * 0.1
    return value.to(device=device, dtype=dtype)


def _reference(hidden_states, gate_weight, up_weight, down_weight):
    gate = hidden_states @ gate_weight.t()
    up = hidden_states @ up_weight.t()
    return (F.silu(gate) * up) @ down_weight.t()


class _TorchKernelStub:
    """Extension-shaped FP32 stub used to check custom autograd algebra."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def det_gemm_fwd(self, a, b):
        self.calls.append("det_gemm_fwd")
        return a @ b

    def det_gemm_db(self, a, grad_output):
        self.calls.append("det_gemm_db")
        return a.t().contiguous() @ grad_output

    def swiglu_forward(self, gate, up):
        self.calls.append("swiglu_forward")
        return F.silu(gate) * up

    def swiglu_backward(self, grad_output, gate, up):
        self.calls.append("swiglu_backward")
        sigmoid = torch.sigmoid(gate)
        grad_gate = grad_output * up * sigmoid * (1.0 + gate * (1.0 - sigmoid))
        grad_up = grad_output * F.silu(gate)
        return grad_gate, grad_up


class _BF16BoundaryStub(_TorchKernelStub):
    """Model native ops that accumulate in FP32 and write BF16 per stage."""

    def __init__(self) -> None:
        super().__init__()
        self.input_dtypes: list[tuple[str, tuple[torch.dtype, ...]]] = []

    def _record(self, name, *values) -> None:
        self.calls.append(name)
        self.input_dtypes.append((name, tuple(value.dtype for value in values)))

    def det_gemm_fwd(self, a, b):
        self._record("det_gemm_fwd", a, b)
        return (a.float() @ b.float()).to(torch.bfloat16)

    def det_gemm_db(self, a, grad_output):
        self._record("det_gemm_db", a, grad_output)
        return (a.float().t().contiguous() @ grad_output.float()).to(torch.bfloat16)

    def swiglu_forward(self, gate, up):
        self._record("swiglu_forward", gate, up)
        return (F.silu(gate.float()) * up.float()).to(torch.bfloat16)

    def swiglu_backward(self, grad_output, gate, up):
        self._record("swiglu_backward", grad_output, gate, up)
        gate_fp32 = gate.float()
        sigmoid = torch.sigmoid(gate_fp32)
        grad_gate = grad_output.float() * up.float() * sigmoid * (1.0 + gate_fp32 * (1.0 - sigmoid))
        grad_up = grad_output.float() * F.silu(gate_fp32)
        return grad_gate.to(torch.bfloat16), grad_up.to(torch.bfloat16)


class _ValidationTensor:
    """Tensor-shaped object for device-independent validation-order tests."""

    def __init__(self, shape, dtype=torch.bfloat16) -> None:
        self.shape = torch.Size(shape)
        self.dtype = dtype
        self.device = torch.device("cuda")
        self.is_cuda = True

    def dim(self):
        return len(self.shape)

    def numel(self):
        result = 1
        for size in self.shape:
            result *= size
        return result

    def size(self, dim):
        return self.shape[dim]


def _install_stub(monkeypatch, stub) -> None:
    monkeypatch.setattr(ffn_module, "_C", stub)
    monkeypatch.setattr(ffn_module, "_EXT_AVAILABLE", True)
    # Native input validation intentionally requires GPU tensors.  Stub tests
    # exercise the public function on CPU after validation itself is tested.
    monkeypatch.setattr(ffn_module, "_validate_ffn_inputs", lambda *args: None)


def test_qwen3_ffn_public_api_and_model_dimensions_are_pinned():
    assert public_qwen3_ffn is qwen3_ffn
    assert QWEN3_8B_HIDDEN_SIZE == 4096
    assert QWEN3_8B_INTERMEDIATE_SIZE == 12288

    signature = inspect.signature(qwen3_ffn)
    assert tuple(signature.parameters) == (
        "rmsnorm_output",
        "gate_weight",
        "up_weight",
        "down_weight",
        "tp_group",
        "cp_group",
        "sequence_parallel",
    )
    assert signature.parameters["tp_group"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["cp_group"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["sequence_parallel"].default is False


def test_qwen3_ffn_stub_shape_layout_and_backward_match_autograd(monkeypatch):
    stub = _TorchKernelStub()
    _install_stub(monkeypatch, stub)

    hidden = _randn((2, 3, 8), seed=0)
    gate_weight = _randn((12, 8), seed=1)
    up_weight = _randn((12, 8), seed=2)
    down_weight = _randn((8, 12), seed=3)
    grad_output = _randn(hidden.shape, seed=4)

    reference_inputs = [
        value.detach().clone().requires_grad_(True)
        for value in (hidden, gate_weight, up_weight, down_weight)
    ]
    expected = _reference(*reference_inputs)
    expected.backward(grad_output)

    actual_inputs = [
        value.detach().clone().requires_grad_(True)
        for value in (hidden, gate_weight, up_weight, down_weight)
    ]
    actual = qwen3_ffn(*actual_inputs)
    actual.backward(grad_output)

    assert actual.shape == hidden.shape
    torch.testing.assert_close(actual, expected.detach())
    for actual_input, reference_input in zip(actual_inputs, reference_inputs, strict=True):
        torch.testing.assert_close(actual_input.grad, reference_input.grad)

    assert stub.calls.count("det_gemm_fwd") == 6
    assert stub.calls.count("det_gemm_db") == 3
    assert stub.calls.count("swiglu_forward") == 1
    assert stub.calls.count("swiglu_backward") == 1


def test_qwen3_ffn_preserves_bf16_at_every_native_stage(monkeypatch):
    stub = _BF16BoundaryStub()
    _install_stub(monkeypatch, stub)

    hidden = _randn((2, 3, 8), seed=10, dtype=torch.bfloat16).requires_grad_(True)
    gate_weight = _randn((12, 8), seed=11, dtype=torch.bfloat16).requires_grad_(True)
    up_weight = _randn((12, 8), seed=12, dtype=torch.bfloat16).requires_grad_(True)
    down_weight = _randn((8, 12), seed=13, dtype=torch.bfloat16).requires_grad_(True)
    grad_output = _randn(hidden.shape, seed=14, dtype=torch.bfloat16)

    actual = qwen3_ffn(hidden, gate_weight, up_weight, down_weight)
    actual.backward(grad_output)

    gate = (hidden.detach().float() @ gate_weight.detach().float().t()).to(torch.bfloat16)
    up = (hidden.detach().float() @ up_weight.detach().float().t()).to(torch.bfloat16)
    activated = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    expected = (activated.float() @ down_weight.detach().float().t()).to(torch.bfloat16)

    assert actual.dtype is torch.bfloat16
    assert torch.equal(actual.detach(), expected)
    assert all(
        dtype is torch.bfloat16 for _, input_dtypes in stub.input_dtypes for dtype in input_dtypes
    )
    for tensor in (hidden, gate_weight, up_weight, down_weight):
        assert tensor.grad is not None
        assert tensor.grad.dtype is torch.bfloat16


@pytest.mark.parametrize("weight_name", ["gate_weight", "up_weight", "down_weight"])
def test_qwen3_ffn_rejects_non_huggingface_weight_layout(weight_name):
    tensors = {
        "rmsnorm_output": torch.empty((2, 8), dtype=torch.bfloat16),
        "gate_weight": torch.empty((12, 8), dtype=torch.bfloat16),
        "up_weight": torch.empty((12, 8), dtype=torch.bfloat16),
        "down_weight": torch.empty((8, 12), dtype=torch.bfloat16),
    }
    tensors[weight_name] = tensors[weight_name].t().contiguous()

    with pytest.raises(ValueError, match=rf"{weight_name} must have shape"):
        qwen3_ffn(**tensors)


def test_qwen3_ffn_rejects_zero_intermediate_size_before_device_dispatch():
    tensors = {
        "rmsnorm_output": torch.empty((2, 8), dtype=torch.bfloat16),
        "gate_weight": torch.empty((0, 8), dtype=torch.bfloat16),
        "up_weight": torch.empty((0, 8), dtype=torch.bfloat16),
        "down_weight": torch.empty((8, 0), dtype=torch.bfloat16),
    }

    with pytest.raises(ValueError, match="intermediate size must be positive"):
        qwen3_ffn(**tensors)


@pytest.mark.parametrize(
    "input_name",
    ("rmsnorm_output", "gate_weight", "up_weight", "down_weight"),
)
def test_qwen3_ffn_rejects_non_bf16_inputs_before_device_dispatch(input_name, monkeypatch):
    monkeypatch.setattr(ffn_module, "Tensor", _ValidationTensor)
    tensors = {
        "rmsnorm_output": _ValidationTensor((2, 8)),
        "gate_weight": _ValidationTensor((12, 8)),
        "up_weight": _ValidationTensor((12, 8)),
        "down_weight": _ValidationTensor((8, 12)),
    }
    tensors[input_name].dtype = torch.float32

    with pytest.raises(TypeError, match=rf"{input_name} must have dtype bfloat16"):
        ffn_module._validate_ffn_inputs(**tensors)


def test_qwen3_ffn_sequence_parallel_flag_must_be_bool(monkeypatch):
    _install_stub(monkeypatch, _TorchKernelStub())
    tensors = (
        torch.empty((2, 8)),
        torch.empty((12, 8)),
        torch.empty((12, 8)),
        torch.empty((8, 12)),
    )

    with pytest.raises(TypeError, match="sequence_parallel must be a bool"):
        qwen3_ffn(*tensors, sequence_parallel=1)


@requires_gpu_ffn
def test_qwen3_ffn_gpu_forward_backward_matches_fp32_reference():
    device = device_ctx.device
    hidden = _randn((2, 3, 32), seed=20, dtype=torch.bfloat16, device=device)
    gate_weight = _randn((64, 32), seed=21, dtype=torch.bfloat16, device=device)
    up_weight = _randn((64, 32), seed=22, dtype=torch.bfloat16, device=device)
    down_weight = _randn((32, 64), seed=23, dtype=torch.bfloat16, device=device)
    grad_output = _randn(hidden.shape, seed=24, dtype=torch.bfloat16, device=device)

    reference_inputs = [
        value.detach().cpu().float().requires_grad_(True)
        for value in (hidden, gate_weight, up_weight, down_weight)
    ]
    expected = _reference(*reference_inputs)
    expected.backward(grad_output.cpu().float())

    actual_inputs = [
        value.detach().clone().requires_grad_(True)
        for value in (hidden, gate_weight, up_weight, down_weight)
    ]
    actual = qwen3_ffn(*actual_inputs)
    actual.backward(grad_output)

    assert actual.shape == hidden.shape
    assert actual.dtype is torch.bfloat16
    torch.testing.assert_close(actual.cpu().float(), expected.detach(), atol=5e-2, rtol=2e-2)
    for actual_input, reference_input in zip(actual_inputs, reference_inputs, strict=True):
        assert actual_input.grad.dtype is torch.bfloat16
        torch.testing.assert_close(
            actual_input.grad.cpu().float(),
            reference_input.grad,
            atol=5e-2,
            rtol=2e-2,
        )


@requires_gpu_ffn
def test_qwen3_ffn_gpu_repeat_is_bitwise_and_train_infer_match():
    device = device_ctx.device
    hidden = _randn((8, 32), seed=30, dtype=torch.bfloat16, device=device)
    gate_weight = _randn((64, 32), seed=31, dtype=torch.bfloat16, device=device)
    up_weight = _randn((64, 32), seed=32, dtype=torch.bfloat16, device=device)
    down_weight = _randn((32, 64), seed=33, dtype=torch.bfloat16, device=device)

    with torch.no_grad():
        first = qwen3_ffn(hidden, gate_weight, up_weight, down_weight)
        second = qwen3_ffn(hidden, gate_weight, up_weight, down_weight)
    train_hidden = hidden.detach().clone().requires_grad_(True)
    train = qwen3_ffn(train_hidden, gate_weight, up_weight, down_weight)

    assert torch.equal(first, second)
    assert torch.equal(first, train.detach())


@requires_gpu_ffn
def test_qwen3_ffn_gpu_output_and_hidden_gradient_are_token_slice_invariant():
    device = device_ctx.device
    hidden = _randn((6, 32), seed=40, dtype=torch.bfloat16, device=device)
    gate_weight = _randn((64, 32), seed=41, dtype=torch.bfloat16, device=device)
    up_weight = _randn((64, 32), seed=42, dtype=torch.bfloat16, device=device)
    down_weight = _randn((32, 64), seed=43, dtype=torch.bfloat16, device=device)
    grad_output = _randn(hidden.shape, seed=44, dtype=torch.bfloat16, device=device)

    full_hidden = hidden.detach().clone().requires_grad_(True)
    full_output = qwen3_ffn(full_hidden, gate_weight, up_weight, down_weight)
    full_output.backward(grad_output)

    slice_hidden = hidden[2:4].detach().clone().requires_grad_(True)
    slice_output = qwen3_ffn(slice_hidden, gate_weight, up_weight, down_weight)
    slice_output.backward(grad_output[2:4])

    assert torch.equal(slice_output, full_output[2:4])
    assert torch.equal(slice_hidden.grad, full_hidden.grad[2:4])


@requires_gpu_ffn
@pytest.mark.parametrize("shape", ((7, 64, 128), (32, 64, 512)))
def test_qwen3_ffn_triton_matches_native_forward_and_backward_bitwise(shape):
    device = device_ctx.device
    tokens, hidden_size, intermediate_size = shape
    values = (
        _randn((tokens, hidden_size), seed=50, dtype=torch.bfloat16, device=device),
        _randn(
            (intermediate_size, hidden_size),
            seed=51,
            dtype=torch.bfloat16,
            device=device,
        ),
        _randn(
            (intermediate_size, hidden_size),
            seed=52,
            dtype=torch.bfloat16,
            device=device,
        ),
        _randn(
            (hidden_size, intermediate_size),
            seed=53,
            dtype=torch.bfloat16,
            device=device,
        ),
    )
    grad_output = _randn(
        (tokens, hidden_size),
        seed=54,
        dtype=torch.bfloat16,
        device=device,
    )
    native_inputs = [value.detach().clone().requires_grad_(True) for value in values]
    triton_inputs = [value.detach().clone().requires_grad_(True) for value in values]

    native_output = qwen3_ffn(*native_inputs)
    triton_output = qwen3_ffn_triton(*triton_inputs)
    native_output.backward(grad_output)
    triton_output.backward(grad_output)

    assert torch.equal(native_output, triton_output)
    for native, triton_input in zip(native_inputs, triton_inputs, strict=True):
        assert torch.equal(native.grad, triton_input.grad)


@requires_gpu_ffn
def test_qwen3_ffn_triton_repeat_and_train_infer_are_bitwise():
    device = device_ctx.device
    values = (
        _randn((8, 64), seed=60, dtype=torch.bfloat16, device=device),
        _randn((128, 64), seed=61, dtype=torch.bfloat16, device=device),
        _randn((128, 64), seed=62, dtype=torch.bfloat16, device=device),
        _randn((64, 128), seed=63, dtype=torch.bfloat16, device=device),
    )
    with torch.no_grad():
        first = qwen3_ffn_triton(*values)
        second = qwen3_ffn_triton(*values)
    training_values = [value.detach().clone().requires_grad_(True) for value in values]
    training = qwen3_ffn_triton(*training_values)

    assert torch.equal(first, second)
    assert torch.equal(first, training.detach())
