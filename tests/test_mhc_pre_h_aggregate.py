# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch


def _kernel_available() -> bool:
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] < 8:
        return False
    try:
        from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
    except Exception:
        return False
    return _EXT_AVAILABLE and hasattr(_C, "mhc_pre_h_aggregate")


requires_mhc_kernel = pytest.mark.skipif(
    not _kernel_available(),
    reason="mhc_pre_h_aggregate requires the CUDA extension on SM80 or newer",
)


def _same_bytes(left: torch.Tensor, right: torch.Tensor) -> bool:
    return torch.equal(left.view(torch.uint8), right.view(torch.uint8))


@requires_mhc_kernel
def test_mhc_pre_h_aggregate_is_batch_invariant_and_matches_pytorch():
    from rl_engine.kernels.ops.base import _C

    num_tokens = 129
    hidden_size = 4096
    num_runs = 100
    generator = torch.Generator(device="cuda").manual_seed(0)
    residual = torch.randn(
        num_tokens,
        4,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    pre = torch.rand(
        num_tokens,
        4,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )

    first = _C.mhc_pre_h_aggregate(residual, pre)
    for _ in range(num_runs - 1):
        repeated = _C.mhc_pre_h_aggregate(residual, pre)
        assert _same_bytes(first, repeated)

    token_by_token = torch.cat(
        [
            _C.mhc_pre_h_aggregate(residual[token : token + 1], pre[token : token + 1])
            for token in range(num_tokens)
        ]
    )
    assert _same_bytes(first, token_by_token)

    reference = torch.sum(pre.unsqueeze(-1) * residual.to(torch.float32), dim=1).to(torch.bfloat16)
    torch.testing.assert_close(first, reference, atol=5e-2, rtol=1e-2)


@requires_mhc_kernel
def test_mhc_pre_h_aggregate_backward_is_batch_invariant_and_matches_pytorch():
    from rl_engine.kernels.ops.base import _C
    from rl_engine.kernels.ops.cuda.mhc import MHCPreHAggregateCudaOp
    from rl_engine.kernels.ops.pytorch.mhc import NativeMHCPreHAggregateOp

    num_tokens = 129
    hidden_size = 4096
    generator = torch.Generator(device="cuda").manual_seed(1)
    residual = torch.randn(
        num_tokens,
        4,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    pre = torch.rand(
        num_tokens,
        4,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    grad_output = torch.randn(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )

    first = _C.mhc_pre_h_aggregate_backward(grad_output, residual, pre)
    for _ in range(99):
        repeated = _C.mhc_pre_h_aggregate_backward(grad_output, residual, pre)
        assert _same_bytes(first[0], repeated[0])
        assert _same_bytes(first[1], repeated[1])

    split = [
        _C.mhc_pre_h_aggregate_backward(
            grad_output[token : token + 1],
            residual[token : token + 1],
            pre[token : token + 1],
        )
        for token in range(num_tokens)
    ]
    split_grad_residual = torch.cat([grads[0] for grads in split])
    split_grad_pre = torch.cat([grads[1] for grads in split])
    assert _same_bytes(first[0], split_grad_residual)
    assert _same_bytes(first[1], split_grad_pre)

    candidate_residual = residual.detach().requires_grad_(True)
    candidate_pre = pre.detach().requires_grad_(True)
    candidate_output = MHCPreHAggregateCudaOp()(candidate_residual, candidate_pre)
    candidate_grads = torch.autograd.grad(
        candidate_output,
        (candidate_residual, candidate_pre),
        grad_outputs=grad_output,
    )
    assert _same_bytes(first[0], candidate_grads[0])
    assert _same_bytes(first[1], candidate_grads[1])

    reference_residual = residual.detach().requires_grad_(True)
    reference_pre = pre.detach().requires_grad_(True)
    reference_output = NativeMHCPreHAggregateOp()(reference_residual, reference_pre)
    reference_grads = torch.autograd.grad(
        reference_output,
        (reference_residual, reference_pre),
        grad_outputs=grad_output,
    )
    torch.testing.assert_close(candidate_grads[0], reference_grads[0], atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(candidate_grads[1], reference_grads[1], atol=5e-2, rtol=2e-2)
