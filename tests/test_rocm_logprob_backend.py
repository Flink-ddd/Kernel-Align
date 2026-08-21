# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from rl_engine.kernels.logprob_contract import (
    LogprobContract,
    MaskSpec,
    ReductionSpec,
    ShardingSpec,
)
from rl_engine.kernels.ops.pytorch.loss.vocab_parallel_logp import VocabParallelLogprobOp
from rl_engine.kernels.ops.rocm.loss.vocab_parallel_logp import RocmVocabParallelLogprobOp
from rl_engine.kernels.registry import KernelRegistry, OpBackend


def test_rocm_backend_preserves_ws2_operator_surface():
    assert issubclass(RocmVocabParallelLogprobOp, VocabParallelLogprobOp)
    assert RocmVocabParallelLogprobOp.op_class == "logprob"
    assert RocmVocabParallelLogprobOp.is_batch_invariant


def test_rocm_backend_is_gated_by_native_extension(monkeypatch):
    registry = KernelRegistry()
    registry._platform = lambda: "rocm"
    candidates = registry._logprob_candidates["rocm"]
    if OpBackend.ROCM_VOCAB_PARALLEL_LOGP in candidates:
        assert candidates[0] is OpBackend.ROCM_VOCAB_PARALLEL_LOGP
        capability = registry._logprob_capabilities["rocm"][OpBackend.ROCM_VOCAB_PARALLEL_LOGP]
        assert capability.backend_id == "rocm-vocab-parallel-logp-ws2"
        assert capability.implementation_kind == "production"
    else:
        assert candidates[0] is OpBackend.PYTORCH_VOCAB_PARALLEL_LOGP


def test_rocm_native_tile_kernel_is_hip_guarded_and_registered():
    source = (Path(__file__).resolve().parents[1] / "csrc" / "ops.cpp").read_text(encoding="utf-8")
    kernel = (
        Path(__file__).resolve().parents[1] / "csrc" / "deterministic_logp_kernel.cu"
    ).read_text(encoding="utf-8")
    assert "deterministic_logp_tile_stats" in source
    assert "deterministic_logp_tile_stats_kernel" in kernel
    assert "atomic" not in kernel.lower()
    assert "__HIPCC__" in source
    assert "deterministic_collective_all_gather" in source


def test_rocm_backend_import_does_not_require_native_extension():
    # Importing the wrapper must remain possible in CPU-only CI; capability
    # loading, not module import, decides whether the native fast path exists.
    op = RocmVocabParallelLogprobOp()
    assert isinstance(op, VocabParallelLogprobOp)


def test_explicit_native_backend_fails_closed_when_extension_is_missing(monkeypatch):
    import rl_engine.kernels.registry as registry_module

    monkeypatch.setattr(registry_module, "_rocm_vocab_logprob_native_available", lambda: False)
    registry = KernelRegistry()
    registry._platform = lambda: "rocm"
    contract = LogprobContract(
        role="train",
        dtype="fp32",
        mask=MaskSpec(num_tokens=1, active_mask=(True,)),
        sharding=ShardingSpec(
            tp_rank=0,
            tp_world_size=1,
            vocab_shard_bounds=((0, 8),),
            real_vocab_size=8,
            padded_vocab_size=8,
        ),
        reduction=ReductionSpec(),
    )

    with pytest.raises(RuntimeError, match="rocm-vocab-parallel-logp-ws2"):
        registry.get_logprob_op(
            contract,
            requested_backend="rocm-vocab-parallel-logp-ws2",
        )
