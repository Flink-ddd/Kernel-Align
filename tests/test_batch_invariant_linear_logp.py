# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rl_engine.kernels.gtest.op_checks import run_operator_suite
from rl_engine.kernels.gtest.operator_specs import OP_SPECS, make_candidate, make_operator_case
from rl_engine.kernels.ops.cuda.loss import batch_invariant_linear_logp as op_module
from rl_engine.kernels.ops.pytorch.loss.linear_logp import NativeLinearLogpOp


def _sm90_op_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        if torch.cuda.get_device_capability()[0] != 9:
            return False
        # GPU CI and explicit feature verification must not turn a missing
        # symbol into a skip. Let the op constructor fail with its build hint.
        if os.environ.get("RL_KERNEL_REQUIRE_EXT") == "1":
            return True

        from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE

        return bool(_EXT_AVAILABLE and hasattr(_C, "batch_invariant_linear_logp_sm90"))
    except Exception:
        return False


requires_sm90_op = pytest.mark.skipif(
    not _sm90_op_available(),
    reason="batch-invariant linear_logp requires Hopper and the compiled SM90 symbol",
)


def test_sm90_availability_hard_mode_does_not_skip_a_missing_symbol(monkeypatch):
    from rl_engine.kernels.ops import base as base_module

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(base_module, "_EXT_AVAILABLE", False)
    monkeypatch.setattr(base_module, "_C", object())

    monkeypatch.delenv("RL_KERNEL_REQUIRE_EXT", raising=False)
    assert not _sm90_op_available()

    monkeypatch.setenv("RL_KERNEL_REQUIRE_EXT", "1")
    assert _sm90_op_available()


def _make_inputs(
    seed: int,
    *,
    num_tokens: int,
    hidden_dim: int,
    vocab_size: int,
    bias: bool,
):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.randn(
        num_tokens,
        hidden_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    weight = torch.randn(
        vocab_size,
        hidden_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    target_ids = torch.randint(
        0,
        vocab_size,
        (num_tokens,),
        device="cuda",
        generator=generator,
    )
    bias_tensor = (
        torch.randn(
            vocab_size,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        if bias
        else None
    )
    return hidden, weight, target_ids, bias_tensor


def _reference(hidden, weight, target_ids, bias):
    return NativeLinearLogpOp().forward_fp32(
        hidden,
        weight,
        target_ids,
        bias,
    )


def test_cuda_source_pins_batch_invariant_split_to_vocab_only():
    source_path = (
        Path(__file__).resolve().parents[1] / "csrc" / "cuda" / "fused_linear_logp_sm90.cu"
    )
    source = source_path.read_text(encoding="utf-8")

    match = re.search(
        r"int select_batch_invariant_vocab_splits\(int total_vtiles\) " r"\{(?P<body>.*?)\n\}",
        source,
        flags=re.DOTALL,
    )
    assert match is not None
    invariant_branch = match.group("body")
    assert "total_vtiles" in invariant_branch
    assert "MAX_BATCH_INVARIANT_VOCAB_SPLITS" in invariant_branch
    assert "row_blocks" not in invariant_branch
    assert "multiProcessorCount" not in invariant_branch
    assert "MAX_BATCH_INVARIANT_VOCAB_SPLITS = 32" in source

    entry_start = source.index(
        "std::vector<torch::Tensor> batch_invariant_linear_logp_sm90_forward("
    )
    entry_end = source.index("// Vocab-parallel local-shard forward", entry_start)
    invariant_entry = source[entry_start:entry_end]
    assert "VocabSplitPolicy::kBatchInvariant" in invariant_entry
    assert "VocabSplitPolicy::kThroughput" not in invariant_entry


def _operator_checker_args(candidate: str, *, seq: int = 16) -> SimpleNamespace:
    return SimpleNamespace(
        op="batch_invariant_linear_logp",
        candidate=candidate,
        arch_key="sm90" if candidate == "cuda-sm90" else None,
        batch=2,
        seq=seq,
        vocab=257,
        seed=123,
        input_mode="random",
        normalized_dim=128,
    )


def test_operator_checker_uses_strict_fp32_gold_and_supports_default_candidate():
    invariant_spec = OP_SPECS["batch_invariant_linear_logp"]
    args = _operator_checker_args("pytorch")
    case = make_operator_case(args, torch.float32, torch.device("cpu"))

    assert invariant_spec.gold_method == "forward_fp32"
    assert OP_SPECS["linear_logp"].gold_method == "apply"
    for candidate_name in ("pytorch", "native"):
        args.candidate = candidate_name
        candidate = make_candidate(args)
        report = run_operator_suite(
            "batch_invariant_linear_logp",
            candidates=[candidate],
            cases=[case],
        )

        assert candidate.backend == "pytorch"
        assert getattr(candidate.fn, "__name__", None) == "forward_fp32"
        assert report.passed


@requires_sm90_op
def test_operator_checker_runs_cuda_sm90_candidate_end_to_end():
    args = _operator_checker_args("cuda-sm90", seq=4)
    case = make_operator_case(args, torch.bfloat16, torch.device("cuda"))
    candidate = make_candidate(args)

    with torch.no_grad():
        report = run_operator_suite(
            "batch_invariant_linear_logp",
            candidates=[candidate],
            cases=[case],
        )

    assert candidate.backend == "cuda-sm90"
    assert report.passed


@pytest.mark.parametrize(
    ("config", "error"),
    [
        ("1,31,64", "D must be divisible by 32"),
        ("1,32,65", "V must be divisible by 4"),
        ("1,32", "must contain N,D,V"),
    ],
)
def test_benchmark_rejects_configs_that_leave_the_sm90_comparison_path(config, error):
    from benchmarks.benchmark_batch_invariant_linear_logp import _parse_configs

    with pytest.raises(ValueError, match=error):
        _parse_configs(config)


@pytest.mark.parametrize(
    ("warmup", "iterations", "error"),
    [
        (-1, 1, "warmup must be non-negative"),
        (0, 0, "iterations must be positive"),
        (0, -1, "iterations must be positive"),
    ],
)
def test_benchmark_rejects_invalid_run_counts(warmup, iterations, error):
    from benchmarks.benchmark_batch_invariant_linear_logp import _validate_run_counts

    with pytest.raises(ValueError, match=error):
        _validate_run_counts(warmup, iterations)


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        (["--configs", "1,32"], "must contain N,D,V"),
        (["--configs", "one,32,64"], "integer N,D,V triples"),
        (["--warmup", "-1"], "warmup must be non-negative"),
        (["--iterations", "0"], "iterations must be positive"),
    ],
)
def test_benchmark_cli_reports_invalid_arguments(monkeypatch, capsys, arguments, error):
    from benchmarks.benchmark_batch_invariant_linear_logp import parse_args

    monkeypatch.setattr(sys, "argv", ["benchmark_batch_invariant_linear_logp.py", *arguments])
    with pytest.raises(SystemExit) as caught:
        parse_args()

    assert caught.value.code == 2
    assert error in capsys.readouterr().err


def test_wrapper_uses_dedicated_extension_symbol_and_preserves_leading_shape(
    monkeypatch,
):
    calls = []
    validation_options = []

    class FakeExtension:
        @staticmethod
        def batch_invariant_linear_logp_sm90(hidden, weight, target_ids, bias):
            calls.append((hidden, weight, target_ids, bias))
            num_tokens = hidden.size(0)
            return torch.arange(num_tokens, dtype=torch.float32), torch.zeros(num_tokens)

    monkeypatch.setattr(op_module, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(op_module, "_C", FakeExtension())

    def fake_validate(*args, **kwargs):
        validation_options.append(kwargs["validate_targets"])

    monkeypatch.setattr(op_module, "_validate_inputs", fake_validate)

    hidden = torch.randn(2, 3, 32)
    weight = torch.randn(17, 32)
    target_ids = torch.randint(0, 17, (2, 3))
    op = op_module.BatchInvariantLinearLogpSM90Op()
    output = op(hidden, weight, target_ids, validate=True)

    assert output.shape == (2, 3)
    assert output.dtype == torch.float32
    assert validation_options == [True]
    assert len(calls) == 1
    called_hidden, called_weight, called_targets, called_bias = calls[0]
    assert called_hidden.shape == (6, 32) and called_hidden.is_contiguous()
    assert called_weight.shape == (17, 32) and called_weight.is_contiguous()
    assert called_targets.shape == (6,) and called_targets.dtype == target_ids.dtype
    assert called_bias is None


def test_wrapper_fails_closed_when_extension_symbol_is_missing(monkeypatch):
    monkeypatch.setattr(op_module, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(op_module, "_C", object())

    with pytest.raises(RuntimeError, match="not compiled"):
        op_module.BatchInvariantLinearLogpSM90Op()


def test_validation_rejects_shape_mismatch_before_device_probe():
    hidden = torch.randn(2, 3, 32)
    weight = torch.randn(17, 32)
    target_ids = torch.randint(0, 17, (2, 4))

    with pytest.raises(ValueError, match="leading shape"):
        op_module._validate_inputs(hidden, weight, target_ids, None)


@pytest.mark.parametrize(
    "target_ids",
    [
        torch.tensor([0, 100], dtype=torch.int8),
        torch.tensor([0, 250], dtype=torch.uint8),
        torch.tensor([0, 499], dtype=torch.int16),
    ],
)
def test_target_range_validation_avoids_narrow_integer_overflow(target_ids):
    op_module._validate_target_range(target_ids, vocab_size=500)


@requires_sm90_op
@pytest.mark.parametrize("bias_dtype", [None, torch.bfloat16, torch.float16, torch.float32])
def test_sm90_matches_fp32_reference(bias_dtype):
    op = op_module.BatchInvariantLinearLogpSM90Op()
    hidden, weight, target_ids, bias = _make_inputs(
        310,
        num_tokens=97,
        hidden_dim=128,
        vocab_size=500,
        bias=bias_dtype is not None,
    )
    if bias is not None:
        bias = bias.to(dtype=bias_dtype)

    output = op(hidden, weight, target_ids, bias)
    expected = _reference(hidden, weight, target_ids, bias)

    assert output.dtype == torch.float32
    assert torch.allclose(output, expected, atol=2e-2, rtol=0.0)


@requires_sm90_op
@pytest.mark.parametrize("bias_dtype", [torch.float16, torch.float32])
def test_sm90_preserves_bias_precision_above_bfloat16(bias_dtype):
    op = op_module.BatchInvariantLinearLogpSM90Op()
    hidden = torch.zeros(1, 32, device="cuda", dtype=torch.bfloat16)
    weight = torch.zeros(64, 32, device="cuda", dtype=torch.bfloat16)
    target_ids = torch.zeros(1, device="cuda", dtype=torch.int64)
    bias = torch.zeros(64, device="cuda", dtype=bias_dtype)
    bias[0] = -8.03

    actual = op(hidden, weight, target_ids, bias)
    expected = _reference(hidden, weight, target_ids, bias)
    rounded_bf16 = _reference(hidden, weight, target_ids, bias.to(torch.bfloat16))

    assert torch.allclose(actual, expected, atol=2e-3, rtol=0.0)
    assert (expected - rounded_bf16).abs().item() > 2e-2


@requires_sm90_op
@pytest.mark.parametrize(
    "hidden_dim,vocab_size",
    [
        (32, 63),
        (64, 64),
        (96, 65),
        (128, 2048),
        (160, 2049),
        (4096, 4097),
    ],
)
def test_sm90_matches_fp32_reference_across_hidden_and_vocab_tiles(
    hidden_dim,
    vocab_size,
):
    op = op_module.BatchInvariantLinearLogpSM90Op()
    num_tokens = 3 if hidden_dim >= 4096 else 17
    hidden, weight, target_ids, _ = _make_inputs(
        322 + hidden_dim + vocab_size,
        num_tokens=num_tokens,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        bias=False,
    )

    output = op(hidden, weight, target_ids)
    expected = _reference(hidden, weight, target_ids, None)
    probe = op(hidden[:1], weight, target_ids[:1])

    assert torch.allclose(output, expected, atol=2e-2, rtol=0.0)
    assert torch.equal(probe[0], output[0])


@requires_sm90_op
def test_sm90_is_bitwise_invariant_across_batch_size_position_and_noise():
    op = op_module.BatchInvariantLinearLogpSM90Op()
    hidden, weight, target_ids, _ = _make_inputs(
        311,
        num_tokens=4096,
        hidden_dim=128,
        vocab_size=4096,
        bias=False,
    )
    probe_hidden = hidden[137].clone()
    probe_target = target_ids[137].clone()
    baseline = op(probe_hidden.unsqueeze(0), weight, probe_target.unsqueeze(0))[0]

    probe_positions = (0, 127, 128, 255, 256, 2048, 4095)
    for position in probe_positions:
        hidden[position].copy_(probe_hidden)
        target_ids[position].copy_(probe_target)

    packed = op(hidden, weight, target_ids)
    for position in probe_positions:
        assert torch.equal(packed[position], baseline)

    repeated = op(hidden, weight, target_ids)
    assert torch.equal(repeated, packed)


@requires_sm90_op
@pytest.mark.parametrize(
    "num_tokens,vocab_size,chunk_sizes",
    [
        (33, 500, (1, 7)),
        (4096, 4096, (255, 256, 257, 1024)),
    ],
)
def test_sm90_full_batch_matches_batch_dimension_chunks_bitwise(
    num_tokens, vocab_size, chunk_sizes
):
    op = op_module.BatchInvariantLinearLogpSM90Op()
    hidden, weight, target_ids, bias = _make_inputs(
        312,
        num_tokens=num_tokens,
        hidden_dim=128,
        vocab_size=vocab_size,
        bias=True,
    )

    full = op(hidden, weight, target_ids, bias)
    for chunk_size in chunk_sizes:
        chunked = torch.cat(
            [
                op(
                    hidden[start : start + chunk_size],
                    weight,
                    target_ids[start : start + chunk_size],
                    bias,
                )
                for start in range(0, hidden.size(0), chunk_size)
            ]
        )
        assert torch.equal(chunked, full), f"batch chunk size {chunk_size} changed output bits"


@requires_sm90_op
def test_sm90_production_vocab_matches_reference_and_is_batch_invariant():
    op = op_module.BatchInvariantLinearLogpSM90Op()
    hidden, weight, target_ids, _ = _make_inputs(
        315,
        num_tokens=17,
        hidden_dim=128,
        vocab_size=50257,
        bias=False,
    )
    target_ids[:8] = torch.tensor(
        [0, 63, 64, 2047, 2048, 4095, 50255, 50256],
        device="cuda",
    )

    full = op(hidden, weight, target_ids)
    expected = _reference(hidden, weight, target_ids, None)
    repeated = op(hidden, weight, target_ids)
    chunked = torch.cat(
        [
            op(hidden[start : start + 7], weight, target_ids[start : start + 7])
            for start in range(0, 17, 7)
        ]
    )
    probe = op(hidden[8:9], weight, target_ids[8:9])

    assert torch.allclose(full, expected, atol=2e-2, rtol=0.0)
    assert torch.equal(repeated, full)
    assert torch.equal(chunked, full)
    assert torch.equal(probe[0], full[8])


@requires_sm90_op
def test_sm90_accepts_contiguous_tma_inputs_with_misaligned_storage_offsets():
    op = op_module.BatchInvariantLinearLogpSM90Op()
    num_tokens, hidden_dim, vocab_size = 9, 128, 500
    hidden_storage = torch.randn(
        num_tokens * hidden_dim + 1,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight_storage = torch.randn(
        vocab_size * hidden_dim + 1,
        device="cuda",
        dtype=torch.bfloat16,
    )
    hidden = hidden_storage[1:].view(num_tokens, hidden_dim)
    weight = weight_storage[1:].view(vocab_size, hidden_dim)
    target_ids = torch.arange(num_tokens, device="cuda") % vocab_size

    assert hidden.is_contiguous() and hidden.data_ptr() % 16 != 0
    assert weight.is_contiguous() and weight.data_ptr() % 16 != 0

    actual = op(hidden, weight, target_ids)
    aligned = op(hidden.clone(), weight.clone(), target_ids)
    expected = _reference(hidden, weight, target_ids, None)

    assert torch.equal(actual, aligned)
    assert torch.allclose(actual, expected, atol=2e-2, rtol=0.0)


@requires_sm90_op
def test_sm90_wrapper_copies_strided_inputs_and_preserves_leading_shape():
    op = op_module.BatchInvariantLinearLogpSM90Op()
    generator = torch.Generator(device="cuda").manual_seed(324)
    hidden = torch.randn(
        2,
        3,
        192,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )[..., ::2]
    weight = torch.randn(
        96,
        257,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).t()
    target_ids = torch.randint(
        0,
        257,
        (2, 3, 2),
        device="cuda",
        generator=generator,
    )[..., 0]
    bias = torch.randn(
        257,
        2,
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )[:, 0]

    assert not hidden.is_contiguous()
    assert not weight.is_contiguous()
    assert not target_ids.is_contiguous()
    assert not bias.is_contiguous()

    actual = op(hidden, weight, target_ids, bias)
    expected = _reference(hidden, weight, target_ids, bias)

    assert actual.shape == (2, 3)
    assert torch.allclose(actual, expected, atol=2e-2, rtol=0.0)


@requires_sm90_op
def test_sm90_uses_the_current_non_default_stream():
    op = op_module.BatchInvariantLinearLogpSM90Op()
    desired_hidden, desired_weight, target_ids, desired_bias = _make_inputs(
        316,
        num_tokens=33,
        hidden_dim=128,
        vocab_size=4096,
        bias=True,
    )
    hidden_storage = torch.zeros(
        desired_hidden.numel() + 1,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight_storage = torch.zeros(
        desired_weight.numel() + 1,
        device="cuda",
        dtype=torch.bfloat16,
    )
    bias_storage = torch.zeros(
        desired_bias.numel() + 1,
        device="cuda",
        dtype=torch.bfloat16,
    )
    hidden = hidden_storage[1:].view_as(desired_hidden)
    weight = weight_storage[1:].view_as(desired_weight)
    bias = bias_storage[1:]

    assert hidden.is_contiguous() and hidden.data_ptr() % 16 != 0
    assert weight.is_contiguous() and weight.data_ptr() % 16 != 0
    assert bias.is_contiguous() and bias.data_ptr() % 16 != 0

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(stream):
        # Delay the mutations on this stream. A kernel accidentally launched on
        # the default stream would race ahead and consume the initial zeros.
        torch.cuda._sleep(50_000_000)
        hidden.copy_(desired_hidden)
        weight.copy_(desired_weight)
        bias.copy_(desired_bias)
        output = op(hidden, weight, target_ids, bias, validate=False)
        dependent = output.clone()
        done = torch.cuda.Event()
        done.record()

    done.synchronize()
    expected = _reference(desired_hidden, desired_weight, target_ids, desired_bias)
    assert torch.allclose(dependent, expected, atol=2e-2, rtol=0.0)


@requires_sm90_op
@pytest.mark.parametrize("requires_grad_input", ["hidden", "weight", "bias"])
def test_sm90_forward_only_contract_rejects_autograd(requires_grad_input):
    op = op_module.BatchInvariantLinearLogpSM90Op()
    hidden, weight, target_ids, bias = _make_inputs(
        313,
        num_tokens=8,
        hidden_dim=128,
        vocab_size=500,
        bias=True,
    )
    assert bias is not None
    tensors = {"hidden": hidden, "weight": weight, "bias": bias}
    tensors[requires_grad_input].requires_grad_(True)

    with pytest.raises(RuntimeError, match="forward-only"):
        op(hidden, weight, target_ids, bias)

    with torch.no_grad():
        output = op(hidden, weight, target_ids, bias)
    assert output.shape == (8,)


@requires_sm90_op
def test_sm90_invalid_targets_are_nan_or_raise_when_validation_is_requested():
    op = op_module.BatchInvariantLinearLogpSM90Op()
    hidden, weight, target_ids, _ = _make_inputs(
        314,
        num_tokens=8,
        hidden_dim=128,
        vocab_size=500,
        bias=False,
    )
    target_ids[3] = -100
    target_ids[4] = 2**40

    unchecked = op(hidden, weight, target_ids)
    assert torch.isnan(unchecked[3])
    assert torch.isnan(unchecked[4])
    assert torch.isfinite(torch.cat((unchecked[:3], unchecked[5:]))).all()

    with pytest.raises(ValueError, match="must be in"):
        op(hidden, weight, target_ids, validate=True)


@requires_sm90_op
@pytest.mark.parametrize(
    "target_dtype,invalid_value",
    [
        (torch.uint8, 250),
        (torch.int8, -1),
        (torch.int16, 500),
        (torch.int32, 97),
        (torch.int64, 2**40),
    ],
)
def test_sm90_supported_target_dtypes_preserve_valid_rows_and_mark_invalid(
    target_dtype,
    invalid_value,
):
    op = op_module.BatchInvariantLinearLogpSM90Op()
    hidden, weight, valid_targets, _ = _make_inputs(
        323,
        num_tokens=8,
        hidden_dim=128,
        vocab_size=97,
        bias=False,
    )
    expected = _reference(hidden, weight, valid_targets, None)
    target_ids = valid_targets.to(dtype=target_dtype)
    target_ids[0] = invalid_value

    unchecked = op(hidden, weight, target_ids)

    assert torch.isnan(unchecked[0])
    assert torch.allclose(unchecked[1:], expected[1:], atol=2e-2, rtol=0.0)
    with pytest.raises(ValueError, match="must be in"):
        op(hidden, weight, target_ids, validate=True)


@requires_sm90_op
@pytest.mark.parametrize(
    "target_dtype,target_value",
    [
        (torch.uint8, 250),
        (torch.int16, 499),
    ],
)
def test_sm90_preserves_valid_high_narrow_integer_targets(target_dtype, target_value):
    op = op_module.BatchInvariantLinearLogpSM90Op()
    hidden, weight, target_ids, _ = _make_inputs(
        325,
        num_tokens=4,
        hidden_dim=128,
        vocab_size=500,
        bias=False,
    )
    target_ids = target_ids.to(dtype=target_dtype)
    target_ids[0] = target_value

    actual = op(hidden, weight, target_ids)
    expected = _reference(hidden, weight, target_ids, None)

    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, expected, atol=2e-2, rtol=0.0)


@requires_sm90_op
def test_sm90_rejects_unsupported_bias_dtype():
    op = op_module.BatchInvariantLinearLogpSM90Op()
    hidden, weight, target_ids, _ = _make_inputs(
        317,
        num_tokens=8,
        hidden_dim=128,
        vocab_size=500,
        bias=False,
    )
    bias = torch.randn(500, device="cuda", dtype=torch.float64)

    with pytest.raises(TypeError, match="bias must have dtype"):
        op(hidden, weight, target_ids, bias)


@requires_sm90_op
@pytest.mark.parametrize("target_dtype", [torch.float32, torch.bool])
def test_sm90_raw_symbol_rejects_non_integer_targets(target_dtype):
    from rl_engine.kernels.ops.base import _C

    hidden, weight, target_ids, _ = _make_inputs(
        318,
        num_tokens=8,
        hidden_dim=128,
        vocab_size=500,
        bias=False,
    )
    invalid_dtype_targets = target_ids.to(dtype=target_dtype)

    with pytest.raises(RuntimeError, match="target must have dtype"):
        _C.batch_invariant_linear_logp_sm90(
            hidden,
            weight,
            invalid_dtype_targets,
            None,
        )


@requires_sm90_op
def test_sm90_raw_symbol_rejects_long_target_metadata_before_conversion():
    from rl_engine.kernels.ops.base import _C

    hidden, weight, _target_ids, _ = _make_inputs(
        320,
        num_tokens=8,
        hidden_dim=128,
        vocab_size=500,
        bias=False,
    )
    cpu_target = torch.zeros(8, dtype=torch.int64)
    expanded_target = torch.zeros(1, device="cuda", dtype=torch.int64).expand(9)

    with pytest.raises(RuntimeError, match="must be CUDA tensors"):
        _C.batch_invariant_linear_logp_sm90(hidden, weight, cpu_target, None)
    with pytest.raises(RuntimeError, match="one id per token"):
        _C.batch_invariant_linear_logp_sm90(hidden, weight, expanded_target, None)
