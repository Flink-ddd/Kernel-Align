"""ROCm WS2 vocab-parallel logprob backend.

The native HIP kernel computes only local FP32 tile partials. The reference
operator owns TP transport, fixed tile-order merging, target ownership,
entropy, and autograd so ROCm and issue #241 share one contract.
"""

from rl_engine.kernels.ops.pytorch.loss.vocab_parallel_logp import VocabParallelLogprobOp


class RocmVocabParallelLogprobOp(VocabParallelLogprobOp):
    """Contract-preserving ROCm implementation with a HIP local reduction."""

    op_class = "logprob"
    is_batch_invariant = True
    use_native_tile_stats = True
