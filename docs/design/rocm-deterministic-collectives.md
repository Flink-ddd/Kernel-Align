# ROCm deterministic collectives

This document defines the ROCm communication boundary used by deterministic
TP/CP kernels.  The implementation is intentionally separate from the CUDA IPC
collective in `csrc/cuda/distributed/deterministic_collective.cu`.

## Contract

`RCCLDeterministicCollective` implements `all_reduce`, `all_gather`, and
`reduce_scatter` with the same Python call shape as the CUDA collective.  It
supports process-group sizes 1, 2, 4, and 8 and FP32/FP16/BF16 reductions.
Inputs and outputs must be contiguous and all ranks must call operations in the
same order with matching shapes, dtypes, and capacity.

RCCL is used only for rank-ordered tensor transport:

1. `all_gather_into_tensor` gathers every rank's bit patterns.
2. Each rank evaluates the same balanced tree locally:
   `((rank0 + rank1) + (rank2 + rank3)) + ...`.
3. `reduce_scatter` slices the rank-owned rows after that fixed reduction.

RCCL's `all_reduce` and `reduce_scatter` are not used for strict reductions.
They guarantee a mathematical reduction but do not expose a stable
floating-point operand order.  Delegating the arithmetic to them would weaken
the cross-TP bitwise contract.

`create_deterministic_collective` is the platform boundary.  It selects the
existing CUDA IPC implementation on NVIDIA and the RCCL transport
implementation when `torch.version.hip` is set.  The FFN path calls this factory
instead of importing the CUDA class directly.

The ROCm extension build also excludes the CUDA IPC source and does not link
`libcuda`; the Python transport has no CUDA-driver dependency.

## Relationship to vLLM

The backend split follows the useful parts of vLLM's device communicator
design:

- PyTorch exposes RCCL through the `nccl` process-group API.
- ROCm AllGather uses `torch.distributed.all_gather_into_tensor` rather than a
  manually allocated PyNccl path.
- Backend, topology, world-size, dtype, layout, and capacity checks fail closed
  before entering an optimized path.

vLLM QuickReduce, custom all-reduce, and AITER all-reduce are not used in the
strict path.  Those are valuable performance implementations, but their
reduction order is not the fixed balanced tree required here.  They can be
added later as an explicitly non-strict performance mode with separate
provenance and toleranced correctness tests.

## Compute/communication fusion

The current ROCm collective is synchronous and reports
`supports_async_overlap = False` and
`supports_compute_communication_fusion = False`.  FFN and Attention keep the
dependency boundaries explicit:

```text
sequence-parallel FFN:  AllGather(input) -> GEMM -> ReduceScatter(output)
strict CP Attention:    AllGather(Q/K/V/positions) -> Attention -> Scatter(output/LSE)
```

This is neither a fused GEMM+collective kernel nor two-stream overlap.  vLLM's
generic AsyncTP GEMM/communication fusion is currently a CUDA path; ROCm AITER
has narrower fusions such as all-reduce plus RMSNorm, which do not replace this
contract.

Future overlap must preserve collective issue order, the local reduction tree,
and stage boundaries.  It also needs repeat-bitwise tests before being marked
strict.  A useful first candidate is backward work that is independent of a
pending reduction; the forward SP AllGather and final row-parallel reduction
are data dependencies and cannot simply be overlapped with their adjacent
GEMMs.

The “fill rank slots as messages arrive and merge ready contiguous blocks”
scheme needs a lower-level P2P or HIP/XGMI transport. PyTorch/RCCL
`all_gather_into_tensor` exposes completion of the whole collective, not
per-rank arrival events. Such a pipeline may merge only canonical sibling
subtrees when both are ready; merging arbitrary contiguous arrivals would
change floating-point parenthesization. It is therefore a follow-up transport,
not an optimization silently hidden inside this baseline.

## Performance acceptance

The transport-only baseline favors correctness and portability. AllGather
writes directly to its final output; reductions reuse one lazily grown
`world_size * input_bytes` byte workspace until `close()`. It still moves more
data than a native RCCL AllReduce. A ROCm GPU PR should therefore report, for
world sizes 2/4/8 and representative FFN tensors:

- latency and effective bandwidth for all three collectives;
- peak temporary memory;
- comparison with RCCL and vLLM's available ROCm communicator;
- repeat-bitwise and cross-TP results;
- end-to-end TP/CP/SP FFN timing, not only isolated transport timing.

A later fixed-tree HIP/XGMI implementation may replace the transport behind the
same factory after it satisfies those checks.

Run the included native-RCCL comparison on a single node, for example:

```bash
torchrun --standalone --nproc-per-node=8 \
  benchmarks/benchmark_rocm_collectives.py \
  --size-bytes 4096 65536 1048576 16777216 \
  --output benchmarks/results/rocm_collectives_mi300x.json
```

The benchmark records slowest-rank latency, temporary allocation, repeat
bitwise status, and the ratio to native RCCL. Native RCCL remains a performance
reference only, not the strict arithmetic reference.
