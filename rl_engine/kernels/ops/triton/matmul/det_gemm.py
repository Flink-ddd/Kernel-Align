# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Batch- and TP-invariant deterministic GEMM, Triton path.

BF16 outputs use the same arithmetic graph as the native deterministic kernel:
each canonical K-tree leaf contains at most 32 values, accumulates in FP32, and
rounds to BF16.  Separate Triton kernels then evaluate the canonical midpoint
tree with BF16 nodes.  The strict ROCm leaf keeps the native gfx942 scalar FMA
order so both implementations have zero mismatch; a future MFMA leaf requires
a separately versioned arithmetic contract because its internal accumulation
does not match the scalar reference at every BF16 rounding boundary.

Autotuning and split-K are intentionally disabled.  FP32-output calls preserve
the earlier fixed-order, no-split-K contract and do not use BF16 tree nodes.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

from rl_engine.kernels.ops.backward_runtime import record_backward
from rl_engine.utils.logger import logger

# Pinned. NOT autotuned (autotune picks per-shape configs -> breaks invariance).
_BLOCK_M, _BLOCK_N, _BLOCK_K = 64, 64, 32
_TREE_PLANS: dict[tuple[int, int], "_DeviceTreePlan"] = {}
_TREE_PLAN_LOCK = threading.Lock()


@dataclass(frozen=True)
class _TreePlan:
    leaf_starts: tuple[int, ...]
    leaf_lengths: tuple[int, ...]
    leaf_nodes: tuple[int, ...]
    reduction_levels: tuple[tuple[tuple[int, int, int], ...], ...]
    root: int
    node_count: int


@dataclass(frozen=True)
class _DeviceTreePlan:
    host: _TreePlan
    leaf_starts: torch.Tensor
    leaf_lengths: torch.Tensor
    leaf_nodes: torch.Tensor
    reduction_levels: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]


def _build_tree_plan(k_size: int) -> _TreePlan:
    if k_size <= 0:
        raise ValueError(f"deterministic GEMM K must be positive, got {k_size}")

    leaf_starts: list[int] = []
    leaf_lengths: list[int] = []
    leaf_nodes: list[int] = []
    reductions_by_height: dict[int, list[tuple[int, int, int]]] = {}
    next_node = 0

    def visit(begin: int, end: int) -> tuple[int, int]:
        nonlocal next_node
        if end - begin <= _BLOCK_K:
            node = next_node
            next_node += 1
            leaf_starts.append(begin)
            leaf_lengths.append(end - begin)
            leaf_nodes.append(node)
            return node, 0

        midpoint = begin + (end - begin) // 2
        lower, lower_height = visit(begin, midpoint)
        upper, upper_height = visit(midpoint, end)
        node = next_node
        next_node += 1
        height = max(lower_height, upper_height) + 1
        reductions_by_height.setdefault(height, []).append((lower, upper, node))
        return node, height

    root, max_height = visit(0, k_size)
    reduction_levels = tuple(
        tuple(reductions_by_height.get(height, ()))
        for height in range(1, max_height + 1)
    )
    return _TreePlan(
        leaf_starts=tuple(leaf_starts),
        leaf_lengths=tuple(leaf_lengths),
        leaf_nodes=tuple(leaf_nodes),
        reduction_levels=reduction_levels,
        root=root,
        node_count=next_node,
    )


def _device_tree_plan(k_size: int, device: torch.device) -> _DeviceTreePlan:
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    key = (device_index, k_size)
    with _TREE_PLAN_LOCK:
        cached = _TREE_PLANS.get(key)
        if cached is not None:
            return cached

        host = _build_tree_plan(k_size)

        def indices(values: tuple[int, ...]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int32, device=device)

        levels = []
        for operations in host.reduction_levels:
            lower, upper, output = zip(*operations, strict=True)
            levels.append((indices(lower), indices(upper), indices(output)))
        result = _DeviceTreePlan(
            host=host,
            leaf_starts=indices(host.leaf_starts),
            leaf_lengths=indices(host.leaf_lengths),
            leaf_nodes=indices(host.leaf_nodes),
            reduction_levels=tuple(levels),
        )
        _TREE_PLANS[key] = result
        return result


if _TRITON_AVAILABLE:

    @triton.jit
    def _det_gemm_tree_leaf_kernel(
        a_ptr,
        b_ptr,
        workspace_ptr,
        leaf_starts_ptr,
        leaf_lengths_ptr,
        leaf_nodes_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        leaf = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        leaf_start = tl.load(leaf_starts_ptr + leaf)
        leaf_length = tl.load(leaf_lengths_ptr + leaf)
        leaf_node = tl.load(leaf_nodes_ptr + leaf)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        # Keep the leaf's ascending scalar FMA order identical to the native
        # gfx942 correctness kernel.  A tl.dot/MFMA leaf is topology-stable but
        # differs from the scalar reference at rare BF16 rounding boundaries.
        for offset in tl.static_range(0, BLOCK_K):
            k_offset = leaf_start + offset
            active = offset < leaf_length
            a = tl.load(
                a_ptr + offs_m * stride_am + k_offset * stride_ak,
                mask=(offs_m < M) & active,
                other=0.0,
            ).to(tl.float32)
            b = tl.load(
                b_ptr + k_offset * stride_bk + offs_n * stride_bn,
                mask=(offs_n < N) & active,
                other=0.0,
            ).to(tl.float32)
            acc += a[:, None] * b[None, :]
        output_offsets = leaf_node * M * N + offs_m[:, None] * N + offs_n[None, :]
        output_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(
            workspace_ptr + output_offsets,
            acc.to(workspace_ptr.dtype.element_ty),
            mask=output_mask,
        )

    @triton.jit
    def _det_gemm_tree_reduce_kernel(
        workspace_ptr,
        lower_nodes_ptr,
        upper_nodes_ptr,
        output_nodes_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        operation = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * BLOCK + tl.arange(0, BLOCK)
        elements = M * N
        mask = offsets < elements
        lower_node = tl.load(lower_nodes_ptr + operation)
        upper_node = tl.load(upper_nodes_ptr + operation)
        output_node = tl.load(output_nodes_ptr + operation)
        lower = tl.load(
            workspace_ptr + lower_node * elements + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        upper = tl.load(
            workspace_ptr + upper_node * elements + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        result = lower + upper
        tl.store(
            workspace_ptr + output_node * elements + offsets,
            result.to(workspace_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _copy_tree_root_kernel(
        workspace_ptr,
        output_ptr,
        root,
        elements,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < elements
        values = tl.load(workspace_ptr + root * elements + offsets, mask=mask)
        tl.store(output_ptr + offsets, values, mask=mask)

    @triton.jit
    def _det_gemm_fp32_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        PROMOTE_INPUTS: tl.constexpr,
    ):
        # One program = one output tile, walks the whole K in fixed order.
        # No split-K -> K-accumulation order independent of M -> batch-invariant.
        pid_m, pid_n = tl.program_id(0), tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_rem = K - k * BLOCK_K
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_rem)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b_mask = (offs_k[:, None] < k_rem) & (offs_n[None, :] < N)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            if PROMOTE_INPUTS:
                a = a.to(tl.float32)
                b = b.to(tl.float32)
            acc += tl.dot(a, b, allow_tf32=False)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        c = acc.to(c_ptr.dtype.element_ty)
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, c, mask=mask)


def _triton_gemm_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = a.contiguous(), b.contiguous()
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, _BLOCK_M), triton.cdiv(N, _BLOCK_N))
    _det_gemm_fp32_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        PROMOTE_INPUTS=True,
    )
    return c


def _triton_tree_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is unavailable")
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("Triton deterministic GEMM expects two 2-D tensors")
    if a.size(1) != b.size(0):
        raise ValueError(f"Triton deterministic GEMM K mismatch: {a.size(1)} and {b.size(0)}")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError("Triton tree GEMM requires BF16 inputs")
    if not a.is_cuda or not b.is_cuda or a.device != b.device:
        raise RuntimeError("Triton tree GEMM inputs must share one CUDA/ROCm device")

    a = a.contiguous()
    b = b.contiguous()
    m_size, k_size = a.shape
    n_size = b.size(1)
    plan = _device_tree_plan(k_size, a.device)
    workspace = torch.empty(
        (plan.host.node_count, m_size, n_size),
        dtype=torch.bfloat16,
        device=a.device,
    )
    leaf_grid = (
        len(plan.host.leaf_nodes),
        triton.cdiv(m_size, _BLOCK_M),
        triton.cdiv(n_size, _BLOCK_N),
    )
    _det_gemm_tree_leaf_kernel[leaf_grid](
        a,
        b,
        workspace,
        plan.leaf_starts,
        plan.leaf_lengths,
        plan.leaf_nodes,
        M=m_size,
        N=n_size,
        K=k_size,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
    )
    reduction_block = 256
    for operations, (lower, upper, output) in zip(
        plan.host.reduction_levels,
        plan.reduction_levels,
        strict=True,
    ):
        grid = (len(operations), triton.cdiv(m_size * n_size, reduction_block))
        _det_gemm_tree_reduce_kernel[grid](
            workspace,
            lower,
            upper,
            output,
            M=m_size,
            N=n_size,
            BLOCK=reduction_block,
        )

    result = torch.empty((m_size, n_size), dtype=torch.bfloat16, device=a.device)
    copy_block = 256
    _copy_tree_root_kernel[(triton.cdiv(result.numel(), copy_block),)](
        workspace,
        result,
        plan.host.root,
        result.numel(),
        BLOCK=copy_block,
    )
    return result


class _TritonDetGemmFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, output_fp32=False):
        ctx.save_for_backward(a, b)
        ctx.output_fp32 = bool(output_fp32)
        return _triton_gemm_fp32(a, b) if output_fp32 else _triton_tree_gemm(a, b)

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        da = (
            _triton_tree_gemm(grad_out.to(torch.bfloat16), b.t().contiguous()).reshape_as(a)
            if ctx.needs_input_grad[0]
            else None
        )
        db = (
            _triton_tree_gemm(a.t().contiguous(), grad_out.to(torch.bfloat16))
            if ctx.needs_input_grad[1]
            else None
        )
        record_backward(
            "det_gemm",
            kernel_id="rl_engine.kernels.ops.triton.matmul.det_gemm._triton_gemm",
            impl="triton_det_gemm",
            family="triton",
        )
        return da, db, None


class TritonDetGemmOp:
    """Batch-invariant deterministic GEMM, Triton path."""

    def __init__(self):
        if not _TRITON_AVAILABLE:
            raise RuntimeError("Triton not available for TritonDetGemmOp")
        logger.info("TritonDetGemmOp ready (deterministic, autotune disabled).")

    def __call__(self, a, b):
        assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16, "BF16 only"
        assert a.is_cuda and b.is_cuda, "CUDA only"
        return _TritonDetGemmFn.apply(a, b, False)

    def forward_fp32(self, a, b):
        if a.dtype not in (torch.bfloat16, torch.float32) or b.dtype not in (
            torch.bfloat16,
            torch.float32,
        ):
            raise TypeError("FP32-output Triton GEMM requires BF16 or FP32 inputs")
        assert a.is_cuda and b.is_cuda, "CUDA only"
        return _TritonDetGemmFn.apply(a, b, True)

    forward_accum_fp32 = forward_fp32

    def parameter_vjp_contributions_fp32(self, *, a, b, grad_output):
        del b
        rows_a = a.float()
        rows_g = grad_output.float()
        return {"b": rows_a[:, :, None] * rows_g[:, None, :]}


def deterministic_gemm_triton(a, b):
    return _TritonDetGemmFn.apply(a, b, False)
