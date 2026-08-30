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
            _C.mhc_pre_h_aggregate(
                residual[token : token + 1], pre[token : token + 1]
            )
            for token in range(num_tokens)
        ]
    )
    assert _same_bytes(first, token_by_token)

    reference = torch.sum(
        pre.unsqueeze(-1) * residual.to(torch.float32), dim=1
    ).to(torch.bfloat16)
    torch.testing.assert_close(first, reference, atol=5e-2, rtol=1e-2)
