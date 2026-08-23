# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Invariance + correctness tests for det_gemm (WS1).

Runs the ROCm-native Triton path directly. CUDA continues to exercise both its
existing native kernel and Triton, each independently satisfying the contract.
The PyTorch path (torch.matmul) is intentionally NOT tested here: it is the
non-deterministic reference baseline and would fail batch-invariance by design.
"""

import pytest
import torch

from rl_engine.kernels.gtest.tolerance import load_contract
from rl_engine.kernels.ops.cuda.matmul import deterministic_gemm

try:
    from rl_engine.kernels.ops.triton.matmul import deterministic_gemm_triton

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

torch.backends.cuda.matmul.allow_tf32 = False
DEV = "cuda"
IS_ROCM = getattr(torch.version, "hip", None) is not None
HAS_SUPPORTED_GPU = torch.cuda.is_available() and (
    IS_ROCM or torch.cuda.get_device_capability()[0] >= 8
)

pytestmark = pytest.mark.skipif(
    not HAS_SUPPORTED_GPU,
    reason="det_gemm requires a ROCm GPU or CUDA SM80+",
)

# ROCm acceptance intentionally depends only on the Triton implementation.
_BACKENDS = [] if IS_ROCM else [("cuda", deterministic_gemm)]
if _HAS_TRITON:
    _BACKENDS.append(("triton", deterministic_gemm_triton))


def _rand(*shape):
    return torch.randn(*shape, device=DEV, dtype=torch.bfloat16)


_K_TREE_LEAF = 32


def _k_tree_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Canonical FP32-leaf/BF16-node midpoint tree used by Triton."""

    a = a.detach().contiguous()
    b = b.detach().contiguous()

    def reduce_range(lo: int, hi: int) -> torch.Tensor:
        if hi - lo <= _K_TREE_LEAF:
            return (a[:, lo:hi].float() @ b[lo:hi, :].float()).to(torch.bfloat16)
        midpoint = lo + (hi - lo) // 2
        return reduce_range(lo, midpoint) + reduce_range(midpoint, hi)

    return reduce_range(0, a.size(1))


def _balanced_tree_sum(parts: list[torch.Tensor]) -> torch.Tensor:
    level = parts
    while len(level) > 1:
        level = [level[index] + level[index + 1] for index in range(0, len(level), 2)]
    return level[0]


@pytest.mark.parametrize("tp_size", (2, 4, 8))
@pytest.mark.skipif(not _HAS_TRITON, reason="Triton is unavailable")
def test_forward_matches_balanced_contiguous_k_shards_bitwise(tp_size):
    """The GEMM K-tree and the TP collective rank tree must be the same graph."""

    torch.manual_seed(8)
    # Qwen3-8B's down projection uses K=12288.  Its 384 32-wide leaves also
    # exercise the non-power-of-two midpoint tree used by the SM90 kernel.
    m, k, n = 4, 12288, 64
    a, b = _rand(m, k), _rand(k, n)
    width = k // tp_size
    parts = [
        deterministic_gemm_triton(
            a[:, rank * width : (rank + 1) * width].contiguous(),
            b[rank * width : (rank + 1) * width].contiguous(),
        )
        for rank in range(tp_size)
    ]

    full = deterministic_gemm_triton(a, b)
    sharded = _balanced_tree_sum(parts)
    assert torch.equal(full, sharded), (
        f"full GEMM differed from balanced TP={tp_size} shards at "
        f"{int((full != sharded).sum().item())} elements"
    )


@pytest.mark.parametrize("name,gemm", _BACKENDS)
def test_forward_batch_invariance(name, gemm):
    # A row's output must not change when other rows join the batch.
    torch.manual_seed(0)
    K, N = 4096, 4096
    b = _rand(K, N)
    row = _rand(1, K)
    out1 = gemm(row, b)
    big = _rand(512, K)
    big[0] = row[0]
    outN = gemm(big, b)
    assert torch.equal(out1[0], outN[0]), f"{name}: forward batch-invariance broken"


@pytest.mark.parametrize("name,gemm", _BACKENDS)
def test_forward_chunked_prefill(name, gemm):
    # Splitting M then concatenating must match the full GEMM bitwise.
    torch.manual_seed(1)
    M, K, N = 256, 4096, 4096
    a, b = _rand(M, K), _rand(K, N)
    full = gemm(a, b)
    chunked = torch.cat([gemm(a[:100], b), gemm(a[100:], b)], dim=0)
    assert torch.equal(full, chunked), f"{name}: chunked-prefill broke invariance"


@pytest.mark.parametrize("name,gemm", _BACKENDS)
def test_forward_padding_invariance(name, gemm):
    # Padding rows must not affect valid rows' output.
    torch.manual_seed(2)
    M, K, N = 100, 4096, 4096
    a, b = _rand(M, K), _rand(K, N)
    base = gemm(a, b)
    a_pad = torch.cat([a, _rand(28, K)], dim=0)
    padded = gemm(a_pad, b)
    assert torch.equal(base, padded[:M]), f"{name}: padding changed valid-row output"


@pytest.mark.parametrize("name,gemm", _BACKENDS)
def test_forward_correctness(name, gemm):
    torch.manual_seed(3)
    M, K, N = 128, 2048, 2048
    a, b = _rand(M, K), _rand(K, N)
    out = gemm(a, b).float()
    ref = _k_tree_gemm(a, b).float()
    contract = load_contract()
    thresholds = contract["accuracy"]["default"]["reduction"]["bfloat16"]
    torch.testing.assert_close(out, ref, atol=thresholds["atol"], rtol=thresholds["rtol"])


@pytest.mark.parametrize("name,gemm", _BACKENDS)
def test_backward_batch_invariance(name, gemm):
    # dA for a row must be invariant to the surrounding batch.
    torch.manual_seed(4)
    K, N = 2048, 2048
    b = _rand(K, N)
    row = _rand(1, K).requires_grad_(True)
    gemm(row, b).sum().backward()
    g1 = row.grad.clone()
    big = _rand(256, K)
    big[0] = row.detach()[0]
    big.requires_grad_(True)
    gemm(big, b).sum().backward()
    assert torch.equal(g1[0], big.grad[0]), f"{name}: backward dA batch-invariance broken"


@pytest.mark.parametrize("name,gemm", _BACKENDS)
def test_backward_correctness(name, gemm):
    torch.manual_seed(5)
    M, K, N = 64, 1024, 1024
    a = _rand(M, K).requires_grad_(True)
    b = _rand(K, N).requires_grad_(True)
    g = _rand(M, N)
    gemm(a, b).backward(g)
    expected_da = _k_tree_gemm(g, b.detach().t().contiguous())
    expected_db = _k_tree_gemm(a.detach().t().contiguous(), g)
    contract = load_contract()
    thresholds = contract["accuracy"]["default"]["reduction"]["bfloat16"]
    torch.testing.assert_close(
        a.grad.float(),
        expected_da.float(),
        atol=thresholds["atol"],
        rtol=thresholds["rtol"],
    )
    torch.testing.assert_close(
        b.grad.float(),
        expected_db.float(),
        atol=thresholds["atol"],
        rtol=thresholds["rtol"],
    )


@pytest.mark.parametrize("name,gemm", _BACKENDS)
@pytest.mark.parametrize(
    "shape",
    [
        (4096, 4096, 12288),  # qkv
        (4096, 4096, 4096),  # o_proj
        (4096, 4096, 14336),  # mlp_up
        (4096, 14336, 4096),  # mlp_dn
        (4096, 4096, 32000),  # lm_head
    ],
)
def test_target_shapes_invariance(name, gemm, shape):
    # Standard-Transformer projection shapes stay batch-invariant.
    torch.manual_seed(6)
    M, K, N = shape
    b = _rand(K, N)
    row = _rand(1, K)
    big = _rand(64, K)
    big[0] = row[0]
    assert torch.equal(gemm(row, b)[0], gemm(big, b)[0]), (
        f"{name}: batch-invariance broken at shape {shape}"
    )


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton is unavailable")
def test_triton_ragged_tiles_mask_all_axes():
    """Non-multiple M/N/K tiles must not read or write past tensor bounds."""
    torch.manual_seed(7)
    a = _rand(80, 130).requires_grad_(True)
    b = _rand(130, 129).requires_grad_(True)
    out = deterministic_gemm_triton(a, b)
    torch.cuda.synchronize()
    assert tuple(out.shape) == (80, 129)
    assert torch.equal(out, _k_tree_gemm(a, b))
    out.backward(_rand(80, 129))
    torch.cuda.synchronize()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(b.grad).all()


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton is unavailable")
@pytest.mark.parametrize("shape", ((4, 65, 129), (8, 4096, 128), (4, 12288, 64)))
def test_triton_tree_matches_python_reference_forward_bitwise(shape):
    """Triton evaluates the canonical FP32-leaf/BF16-node tree exactly."""

    torch.manual_seed(47)
    m_size, k_size, n_size = shape
    a = _rand(m_size, k_size)
    b = _rand(k_size, n_size)

    expected = _k_tree_gemm(a, b)
    triton_output = deterministic_gemm_triton(a, b)

    assert torch.equal(expected, triton_output), (
        f"Triton/reference mismatch at {shape}: "
        f"{int((expected != triton_output).sum().item())} elements"
    )


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton is unavailable")
def test_triton_tree_matches_python_reference_backward_bitwise():
    torch.manual_seed(48)
    a = _rand(32, 512)
    b = _rand(512, 128)
    grad_output = _rand(32, 128)
    triton_inputs = [value.detach().clone().requires_grad_(True) for value in (a, b)]

    deterministic_gemm_triton(*triton_inputs).backward(grad_output)

    expected = (
        _k_tree_gemm(grad_output, b.t().contiguous()),
        _k_tree_gemm(a.t().contiguous(), grad_output),
    )
    for expected_grad, triton_input in zip(expected, triton_inputs, strict=True):
        assert torch.equal(expected_grad, triton_input.grad)
