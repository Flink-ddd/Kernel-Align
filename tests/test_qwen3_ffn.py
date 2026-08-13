# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""WS2 #239 PR2: complete Qwen3 FFN tolerance and batch-invariance checks."""

from __future__ import annotations

import pytest
import torch

from benchmarks.benchmark_qwen3_ffn import FFNBenchmarkResult, parse_args, render_results
from rl_engine.kernels.ffn import (
    QWEN3_8B_HIDDEN_SIZE,
    QWEN3_8B_INTERMEDIATE_SIZE,
    QWEN3_8B_TP2_INTERMEDIATE_SIZE,
    Qwen3FFN,
    Qwen3FFNProvenance,
    build_qwen3_ffn,
    qwen3_ffn_fp32_reference,
)
from rl_engine.kernels.gtest.tolerance import load_contract
from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE

_HAS_CUDA_CONSISTENT = bool(
    torch.cuda.is_available()
    and _EXT_AVAILABLE
    and _C is not None
    and all(
        hasattr(_C, symbol)
        for symbol in (
            "det_gemm_fwd",
            "det_gemm_da",
            "det_gemm_db",
            "swiglu_forward",
            "swiglu_backward",
        )
    )
)

try:
    import triton  # noqa: F401

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

_IS_SM90 = bool(torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9)


def _consistent_backends():
    return [
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not (_IS_SM90 and _HAS_CUDA_CONSISTENT),
                reason=("CUDA consistent FFN tests require SM90 and compiled " "GEMM/SwiGLU ops"),
            ),
        ),
        pytest.param(
            "triton",
            marks=pytest.mark.skipif(
                not (_IS_SM90 and _HAS_TRITON),
                reason="Triton consistent FFN tests require SM90 and Triton",
            ),
        ),
    ]


def _randn(shape, *, seed: int, dtype=torch.float32, device="cpu", scale=1.0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn(shape, generator=generator) * scale
    return value.to(device=device, dtype=dtype)


def _weights(
    hidden: int,
    intermediate: int,
    *,
    dtype=torch.float32,
    device="cpu",
):
    return (
        _randn(
            (hidden, intermediate),
            seed=11,
            dtype=dtype,
            device=device,
            scale=0.02,
        ),
        _randn(
            (hidden, intermediate),
            seed=12,
            dtype=dtype,
            device=device,
            scale=0.02,
        ),
        _randn(
            (intermediate, hidden),
            seed=13,
            dtype=dtype,
            device=device,
            scale=0.02,
        ),
    )


def _reduction_tolerance(dtype: torch.dtype) -> tuple[float, float]:
    contract = load_contract()
    key = {
        torch.float32: "float32",
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
    }[dtype]
    values = contract["accuracy"]["default"]["reduction"][key]
    return float(values["atol"]), float(values["rtol"])


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype) -> None:
    atol, rtol = _reduction_tolerance(dtype)
    actual_fp32, expected_fp32 = actual.float(), expected.float()
    assert_close = torch.testing.assert_close
    assert_close(actual_fp32, expected_fp32, atol=atol, rtol=rtol)


def test_fast_ffn_constructs_complete_unsharded_qwen3_flow():
    assert QWEN3_8B_HIDDEN_SIZE == 4096
    assert QWEN3_8B_INTERMEDIATE_SIZE == 12288
    assert QWEN3_8B_TP2_INTERMEDIATE_SIZE == 6144

    hidden, intermediate = 16, 48
    module = build_qwen3_ffn(*_weights(hidden, intermediate), path="fast", backend="pytorch")
    x = _randn((2, 3, hidden), seed=20)
    stages = module.forward_with_stages(x)

    assert stages.gate.shape == (2, 3, intermediate)
    assert stages.up.shape == (2, 3, intermediate)
    assert stages.hidden.shape == (2, 3, intermediate)
    assert stages.output.shape == x.shape
    assert module.hidden_size == hidden
    assert module.intermediate_size == intermediate
    assert module.provenance.path == "fast"


def test_ffn_benchmark_defaults_to_qwen3_8b_tp2_and_renders_comparison():
    args = parse_args([])
    assert args.hidden_size == QWEN3_8B_HIDDEN_SIZE
    assert args.intermediate_size == QWEN3_8B_TP2_INTERMEDIATE_SIZE
    assert args.seed == 239

    fast = FFNBenchmarkResult(
        path="fast",
        gemm_backend="pytorch.matmul",
        activation_backend="torch.nn.functional.silu",
        forward_ms=1.0,
        forward_backward_ms=3.0,
        max_abs_error=0.01,
        mean_abs_error=0.001,
        peak_memory_mb=100.0,
        stage_ms={},
    )
    consistent = FFNBenchmarkResult(
        path="consistent",
        gemm_backend="cuda.det_gemm",
        activation_backend="cuda.swiglu",
        forward_ms=2.0,
        forward_backward_ms=6.0,
        max_abs_error=0.0,
        mean_abs_error=0.0,
        peak_memory_mb=120.0,
        stage_ms={},
    )
    table = render_results([consistent, fast])
    assert "consistent" in table and "fast" in table
    assert "2.00x" in table


def test_same_ffn_contract_accepts_tp_local_intermediate_shard():
    # Qwen3-8B uses I=12288 globally and I_local=6144 under TP=2. Small values
    # exercise the same ownership contract without model-scale allocations.
    hidden, intermediate_local = 8, 6
    module = build_qwen3_ffn(*_weights(hidden, intermediate_local), path="fast", backend="pytorch")
    stages = module.forward_with_stages(_randn((4, hidden), seed=21))
    assert stages.hidden.shape == (4, intermediate_local)
    # This is a Down partial; the outer distributed wrapper reduces it.
    assert stages.output.shape == (4, hidden)


def test_fast_ffn_forward_and_all_boundaries_match_fp32_reference():
    hidden, intermediate = 12, 28
    weights = _weights(hidden, intermediate)
    x = _randn((2, 5, hidden), seed=22)
    module = build_qwen3_ffn(*weights, path="fast", backend="pytorch")

    actual = module.forward_with_stages(x)
    expected = qwen3_ffn_fp32_reference(x, *weights)
    for actual_tensor, expected_tensor in zip(
        (actual.gate, actual.up, actual.hidden, actual.output),
        (expected.gate, expected.up, expected.hidden, expected.output),
        strict=True,
    ):
        _assert_close(actual_tensor, expected_tensor, torch.float32)


def test_fast_ffn_complete_backward_matches_fp32_reference():
    hidden, intermediate = 10, 24
    weights = _weights(hidden, intermediate)
    x = _randn((2, 4, hidden), seed=23).requires_grad_(True)
    dy = _randn((2, 4, hidden), seed=24)
    module = build_qwen3_ffn(*weights, path="fast", backend="pytorch")
    actual = module.forward_with_stages(x)
    for tensor in (actual.gate, actual.up, actual.hidden):
        tensor.retain_grad()
    actual.output.backward(dy)

    x_ref = x.detach().clone().requires_grad_(True)
    weight_refs = tuple(weight.detach().clone().requires_grad_(True) for weight in weights)
    expected = qwen3_ffn_fp32_reference(x_ref, *weight_refs)
    for tensor in (expected.gate, expected.up, expected.hidden):
        tensor.retain_grad()
    expected.output.backward(dy)

    for actual_tensor, expected_tensor in zip(
        (
            actual.output,
            actual.gate.grad,
            actual.up.grad,
            actual.hidden.grad,
            x.grad,
            module.gate_weight.grad,
            module.up_weight.grad,
            module.down_weight.grad,
        ),
        (
            expected.output,
            expected.gate.grad,
            expected.up.grad,
            expected.hidden.grad,
            x_ref.grad,
            weight_refs[0].grad,
            weight_refs[1].grad,
            weight_refs[2].grad,
        ),
        strict=True,
    ):
        assert actual_tensor is not None and expected_tensor is not None
        _assert_close(actual_tensor, expected_tensor, torch.float32)


def test_ffn_rejects_invalid_shapes_dtype_device_and_backend():
    hidden, intermediate = 8, 12
    gate, up, down = _weights(hidden, intermediate)
    provenance = Qwen3FFNProvenance("test", "test", "test")

    with pytest.raises(ValueError, match="gate and up weights"):
        Qwen3FFN(
            gate,
            up[:, :-1],
            down,
            gemm_op=torch.matmul,
            swiglu_op=lambda g, u: g * u,
            provenance=provenance,
        )
    with pytest.raises(ValueError, match="down weight"):
        Qwen3FFN(
            gate,
            up,
            down[:-1],
            gemm_op=torch.matmul,
            swiglu_op=lambda g, u: g * u,
            provenance=provenance,
        )
    backend_error = "requires backend='cuda' or backend='triton'"
    with pytest.raises(ValueError, match=backend_error):
        build_qwen3_ffn(gate, up, down, path="consistent", backend="pytorch")
    with pytest.raises(ValueError, match="requires backend='pytorch'"):
        build_qwen3_ffn(gate, up, down, path="fast", backend="cuda")
    with pytest.raises(TypeError, match="requires BF16 weights"):
        build_qwen3_ffn(gate, up, down, path="consistent", backend="cuda")
    bf16_weights = tuple(weight.bfloat16() for weight in (gate, up, down))
    with pytest.raises(RuntimeError, match="requires CUDA SM90 weights"):
        build_qwen3_ffn(*bf16_weights, path="consistent", backend="cuda")

    module = build_qwen3_ffn(gate, up, down, path="fast", backend="pytorch")
    with pytest.raises(ValueError, match="last dimension"):
        module(torch.ones(2, hidden + 1))
    with pytest.raises(TypeError, match="share dtype"):
        module(torch.ones(2, hidden, dtype=torch.float64))
    with pytest.raises(ValueError, match="empty token"):
        module(torch.empty(0, hidden))


@pytest.mark.parametrize("backend", _consistent_backends())
def test_consistent_ffn_qwen3_tp2_local_shape_forward_backward(backend):
    hidden = QWEN3_8B_HIDDEN_SIZE
    intermediate_local = QWEN3_8B_TP2_INTERMEDIATE_SIZE
    dtype, device = torch.bfloat16, "cuda"
    module = build_qwen3_ffn(
        *_weights(hidden, intermediate_local, dtype=dtype, device=device),
        path="consistent",
        backend=backend,
    )
    # M=32 keeps dW on the aligned SM90 path while limiting test memory/time.
    x = _randn((32, hidden), seed=28, dtype=dtype, device=device)
    x.requires_grad_(True)
    dy = _randn((32, hidden), seed=29, dtype=dtype, device=device)

    stages = module.forward_with_stages(x)
    assert stages.gate.shape == (32, intermediate_local)
    assert stages.up.shape == (32, intermediate_local)
    assert stages.hidden.shape == (32, intermediate_local)
    assert stages.output.shape == (32, hidden)

    stages.output.backward(dy)
    expected_grads = (
        (x.grad, x.shape),
        (module.gate_weight.grad, (hidden, intermediate_local)),
        (module.up_weight.grad, (hidden, intermediate_local)),
        (module.down_weight.grad, (intermediate_local, hidden)),
    )
    for gradient, expected_shape in expected_grads:
        assert gradient is not None
        assert gradient.shape == expected_shape
        assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("backend", _consistent_backends())
def test_consistent_ffn_forward_backward_tolerance_against_fp32(backend):
    hidden, intermediate = 128, 256
    dtype, device = torch.bfloat16, "cuda"
    weights = _weights(hidden, intermediate, dtype=dtype, device=device)
    module = build_qwen3_ffn(*weights, path="consistent", backend=backend)
    # M=64 exercises the aligned det_gemm_db reduction path as well as dA.
    x = _randn((2, 32, hidden), seed=30, dtype=dtype, device=device)
    x.requires_grad_(True)
    dy = _randn((2, 32, hidden), seed=31, dtype=dtype, device=device)

    actual = module.forward_with_stages(x)
    for tensor in (actual.gate, actual.up, actual.hidden):
        tensor.retain_grad()
    actual.output.backward(dy)

    x_ref = x.detach().float().requires_grad_(True)
    weight_refs = tuple(weight.detach().float().requires_grad_(True) for weight in weights)
    expected = qwen3_ffn_fp32_reference(x_ref, *weight_refs)
    for tensor in (expected.gate, expected.up, expected.hidden):
        tensor.retain_grad()
    expected.output.backward(dy.float())

    for actual_tensor, expected_tensor in zip(
        (
            actual.gate,
            actual.up,
            actual.hidden,
            actual.output,
            actual.gate.grad,
            actual.up.grad,
            actual.hidden.grad,
            x.grad,
            module.gate_weight.grad,
            module.up_weight.grad,
            module.down_weight.grad,
        ),
        (
            expected.gate,
            expected.up,
            expected.hidden,
            expected.output,
            expected.gate.grad,
            expected.up.grad,
            expected.hidden.grad,
            x_ref.grad,
            weight_refs[0].grad,
            weight_refs[1].grad,
            weight_refs[2].grad,
        ),
        strict=True,
    ):
        assert actual_tensor is not None and expected_tensor is not None
        _assert_close(actual_tensor, expected_tensor, dtype)


def _run_activation_backward(module, x, dy):
    x_run = x.detach().clone().requires_grad_(True)
    stages = module.forward_with_stages(x_run)
    for tensor in (stages.gate, stages.up, stages.hidden):
        tensor.retain_grad()
    stages.output.backward(dy)
    return (
        stages.output.detach(),
        stages.gate.grad.detach(),
        stages.up.grad.detach(),
        stages.hidden.grad.detach(),
        x_run.grad.detach(),
    )


@pytest.mark.parametrize("backend", _consistent_backends())
def test_consistent_ffn_batch_chunk_padding_invariance(backend):
    hidden, intermediate = 128, 256
    dtype, device = torch.bfloat16, "cuda"
    module = build_qwen3_ffn(
        *_weights(hidden, intermediate, dtype=dtype, device=device),
        path="consistent",
        backend=backend,
        trainable=False,
    )
    x = _randn((8, hidden), seed=40, dtype=dtype, device=device)
    dy = _randn((8, hidden), seed=41, dtype=dtype, device=device)

    full = _run_activation_backward(module, x, dy)
    singleton = _run_activation_backward(module, x[:1], dy[:1])
    chunked_parts = [
        _run_activation_backward(module, x[:3], dy[:3]),
        _run_activation_backward(module, x[3:6], dy[3:6]),
        _run_activation_backward(module, x[6:], dy[6:]),
    ]
    chunked = tuple(
        torch.cat([part[index] for part in chunked_parts], dim=0) for index in range(len(full))
    )

    padding_x = _randn((3, hidden), seed=42, dtype=dtype, device=device)
    padding_dy = _randn((3, hidden), seed=43, dtype=dtype, device=device)
    padded = _run_activation_backward(
        module, torch.cat((x, padding_x)), torch.cat((dy, padding_dy))
    )

    for full_tensor, singleton_tensor, chunked_tensor, padded_tensor in zip(
        full, singleton, chunked, padded, strict=True
    ):
        assert torch.equal(full_tensor[:1], singleton_tensor)
        assert torch.equal(full_tensor, chunked_tensor)
        assert torch.equal(full_tensor, padded_tensor[: x.shape[0]])


@pytest.mark.parametrize("backend", _consistent_backends())
def test_consistent_ffn_weight_gradients_repeat_bitwise(backend):
    # dW reduces over tokens. PR2 claims repeat determinism and FP32 tolerance,
    # not bitwise equality after changing token partition/reduction boundaries.
    hidden, intermediate = 128, 256
    dtype, device = torch.bfloat16, "cuda"
    module = build_qwen3_ffn(
        *_weights(hidden, intermediate, dtype=dtype, device=device),
        path="consistent",
        backend=backend,
    )
    x = _randn((64, hidden), seed=50, dtype=dtype, device=device)
    dy = _randn((64, hidden), seed=51, dtype=dtype, device=device)

    def run():
        module.zero_grad(set_to_none=True)
        module(x).backward(dy)
        grads = (parameter.grad for parameter in module.parameters())
        return tuple(grad.detach().clone() for grad in grads)

    expected = run()
    for _ in range(3):
        actual = run()
        pairs = zip(expected, actual, strict=True)
        assert all(torch.equal(expected_grad, actual_grad) for expected_grad, actual_grad in pairs)
