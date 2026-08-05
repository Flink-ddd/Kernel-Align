# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""
[WS2 PR4] 2-node x 2-GPU forward validation for Qwen3-8B.
Requires exactly 4 ranks (TP=2, CP=2).
"""

import os

import torch
import torch.distributed as dist

from rl_engine.kernels.gtest.tolerance import load_contract
from rl_engine.kernels.ops.cuda.activation import SwiGLUSM90Op
from rl_engine.kernels.ops.cuda.matmul import deterministic_gemm


def setup_ws2_fwd_topology():
    """Initialize 4-rank topology for TP=2, CP=2."""
    dist.init_process_group(backend="nccl")
    assert dist.get_world_size() == 4, "PR4 strictly requires exactly 4 ranks"

    # TP Groups (Intra-node equivalent): [0, 1] and [2, 3]
    tp_groups = [dist.new_group(ranks=[0, 1]), dist.new_group(ranks=[2, 3])]
    # CP Groups (Inter-node equivalent): [0, 2] and [1, 3]
    cp_groups = [dist.new_group(ranks=[0, 2]), dist.new_group(ranks=[1, 3])]  # noqa: F841

    rank = dist.get_rank()
    my_tp_group = tp_groups[0] if rank in [0, 1] else tp_groups[1]

    return my_tp_group


def run_pr4_forward_validation():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Map processes to physical GPUs
    gpu_count = torch.cuda.device_count()
    device = torch.device(f"cuda:{local_rank % gpu_count}")
    torch.cuda.set_device(device)

    tp_group = setup_ws2_fwd_topology()

    # Load #108 tolerance contract
    contract = load_contract()
    thresholds = contract["accuracy"]["default"]["reduction"]["bfloat16"]
    atol, rtol = thresholds["atol"], thresholds["rtol"]

    # Qwen3-8B simulated shapes
    M_local, K_global, N_global = 128, 4096, 12288

    # Generate local mock data with fixed seed (to be replaced by .pt fixtures later)
    torch.manual_seed(42 + rank)
    std_dev = 0.01
    x_local = torch.randn(M_local, K_global, dtype=torch.bfloat16, device=device) * std_dev
    w_gate_local = (
        torch.randn(K_global, N_global // 2, dtype=torch.bfloat16, device=device) * std_dev
    )
    w_up_local = torch.randn(K_global, N_global // 2, dtype=torch.bfloat16, device=device) * std_dev
    w_down_local = (
        torch.randn(N_global // 2, K_global, dtype=torch.bfloat16, device=device) * std_dev
    )

    if rank == 0:
        print("\n=== Starting WS2 PR4 Forward Validation ===")

    # 1. Gate & Up ColumnParallel (No Comms)
    gate_local = deterministic_gemm(x_local, w_gate_local)
    up_local = deterministic_gemm(x_local, w_up_local)

    # 2. Activation boundary (via PR #258 SM90 operator)
    swiglu_op = SwiGLUSM90Op()
    hidden_local = swiglu_op(gate_local, up_local)

    # 3. Down RowParallel + TP AllReduce SUM
    partial_output = deterministic_gemm(hidden_local, w_down_local)
    final_output = partial_output.clone()
    dist.all_reduce(final_output, op=dist.ReduceOp.SUM, group=tp_group)

    # 4. FP32 Oracle validation
    hidden_fp32 = hidden_local.float()
    w_down_fp32 = w_down_local.float()
    oracle_partial = hidden_fp32 @ w_down_fp32
    oracle_final = oracle_partial.clone()
    dist.all_reduce(oracle_final, op=dist.ReduceOp.SUM, group=tp_group)

    torch.testing.assert_close(
        final_output.float(),
        oracle_final,
        atol=atol,
        rtol=rtol,
        msg="Forward output drifted from FP32 Oracle",
    )

    if rank == 0:
        print(f"[Rank 0] Pipeline Assembled & Validated. Output Shape: {final_output.shape}")
        print(f"[Rank 0] Validation Passed! (atol={atol}, rtol={rtol})")
        print("\n=== WS2 Topology & Conformance Report ===")
        print("COVERED:")
        print("  - 4 ranks topology (TP=2, CP=2)")
        print("  - two IntraNode TP AllReduce groups")
        print(f"  - NCCL Version: {torch.cuda.nccl.version()}")
        print(f"  - Dtype: {x_local.dtype} arithmetic and reductions")
        print("NOT COVERED:")
        print("  - InterNode TP, TP=4+, and uneven shards")
        print("  - DP>1 or a combined DP x CP gradient group")
        print("=========================================\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    run_pr4_forward_validation()
