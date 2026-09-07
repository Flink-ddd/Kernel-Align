import pytest
import torch

from rl_engine.kernels.ops.cuda.loss.logp import FusedLogpGenericOp
from rl_engine.kernels.registry import KernelRegistry


def _musa_available() -> bool:
    return hasattr(torch, "musa") and torch.musa.is_available()


@pytest.mark.skipif(not _musa_available(), reason="requires a MUSA device")
def test_musa_fused_logp_matches_reference_and_supports_backward():
    from rl_engine import _C

    assert hasattr(_C, "fused_logp")
    logits = torch.randn(4, 257, device="musa", dtype=torch.float32, requires_grad=True)
    token_ids = torch.tensor([0, 17, 128, 256], device="musa", dtype=torch.long)

    output = FusedLogpGenericOp()(logits, token_ids)
    reference = torch.log_softmax(logits.float(), dim=-1).gather(
        1, token_ids[:, None]
    ).squeeze(1)
    assert torch.allclose(output, reference, atol=1e-5, rtol=1e-5)

    output.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.mark.skipif(not _musa_available(), reason="requires a MUSA device")
def test_musa_registry_selects_fused_logp_backend():
    backend = KernelRegistry().get_op("logp", device="musa")
    assert backend.__class__.__name__ == "FusedLogpGenericOp"
