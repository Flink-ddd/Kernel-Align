// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Hopper (SM90) forward-only FlashAttention: causal masking, packed
// variable-length (cu_seqlens) input, attention-domain LSE export.
//
// Compute path is TMA (global->shared loads) + PTX `mma.sync.aligned.m16n8k16`
// (not literal `wgmma.mma_async`) -- the same combination every other "SM90"
// kernel in this repo already uses (fused_logp_sm90.cu,
// fused_linear_logp_sm90.cu). Softmax/PV math, tiling constants, and the
// mma.sync fragment layout are adapted from
// csrc/cuda/attention/prefix_shared_attention.cu (a working dense mma.sync
// attention kernel already in this repo); the TMA double-buffering pipeline
// is adapted from csrc/cuda/fused_linear_logp_sm90.cu. This is the
// cross-platform-baseline-matching forward kernel described in
// docs/operators/attention-varlen.md; validated against
// rl_engine/kernels/ops/triton/triton_attn.py's varlen path, which is the
// semantic reference. No backward pass this milestone.

#include "../../utils/tma_utils.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <float.h>
#include <math_constants.h>
#include <torch/extension.h>
#include <vector>

namespace {

constexpr int BLOCK_Q = 64;
constexpr int BLOCK_KV = 64;
constexpr int DIM = 128;        // only head_dim supported this milestone
constexpr int NUM_WARPS = 4;
constexpr int WARP_SIZE = 32;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
constexpr int STAGES = 2;       // K/V double buffering

constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;
constexpr int WARP_Q = BLOCK_Q / NUM_WARPS;

static_assert(WARP_Q % MMA_M == 0, "WARP_Q must be a multiple of MMA_M");
static_assert(DIM % MMA_K == 0 && DIM % MMA_N == 0, "DIM must be a multiple of MMA_K/MMA_N");
static_assert(BLOCK_KV % MMA_N == 0 && BLOCK_KV % MMA_K == 0,
              "BLOCK_KV must be a multiple of MMA_N/MMA_K");

__device__ __host__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

// Tensor-core / shared-memory helpers -- same PTX and fragment layout as
// csrc/cuda/attention/prefix_shared_attention.cu, validated on this repo's
// Hopper GPUs. Kept as a local copy (not shared via a header) matching how
// fused_linear_logp_sm90.cu keeps its own copy of the analogous helpers.
__device__ inline void ldmatrix_x4(uint32_t regs[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
                 : "r"(addr));
}

__device__ inline void ldmatrix_x4_trans(uint32_t regs[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
                 : "r"(addr));
}

__device__ inline void mma_m16n8k16(uint32_t A[4], uint32_t B[2], float D[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                 : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
                   "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));
}

// 2D bf16 tensor map, swizzle pinned to NONE: this kernel reads its tiles with
// plain row-major ldmatrix addressing (no manual XOR swizzle), so the TMA
// writes must be unswizzled too. Same rationale/pattern as
// fused_linear_logp_sm90.cu's helper of the same name (kept as a local copy,
// not shared, to match that file's own precedent).
inline void init_tensor_map_noswizzle(CUtensorMap *tmap, const nv_bfloat16 *gmem,
                                      uint64_t gmem_height, uint64_t gmem_width,
                                      uint32_t box_height, uint32_t box_width) {
    uint64_t size[2] = {gmem_width, gmem_height};
    uint64_t stride[1] = {gmem_width * sizeof(nv_bfloat16)};
    uint32_t box[2] = {box_width, box_height};
    uint32_t elem_stride[2] = {1, 1};
    CUresult res = cuTensorMapEncodeTiled(
        tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void *)gmem, size, stride, box, elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(res == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed for flash_attention_varlen_sm90");
}

// Q/K/V are packed [total_tokens, H, DIM] bf16, treated as logical 2D
// [total_tokens, H*DIM] matrices (row stride H*DIM elements -- the tensor's
// natural contiguous stride). Each TMA box spans one head's DIM-wide column
// slice (x = h*DIM) at a given packed-row offset (y = seq_start + tile_off).
// Cross-sequence-boundary reads (a KV tile near a short sequence's tail
// pulling in the next sequence's real K/V rows) are real and are handled by
// explicit seqlen_k/causal masking of the S=QK^T fragments below, not by
// making TMA itself sequence-aware -- the same design choice
// docs/operators/attention-varlen.md documents for the Triton kernel.
__global__ __launch_bounds__(TB_SIZE) void flash_attention_varlen_sm90_kernel(
    const __grid_constant__ CUtensorMap q_tmap,
    const __grid_constant__ CUtensorMap k_tmap,
    const __grid_constant__ CUtensorMap v_tmap,
    const int *__restrict__ cu_seqlens_q,
    const int *__restrict__ cu_seqlens_k,
    nv_bfloat16 *__restrict__ Out, // [total_q, H, DIM]
    float *__restrict__ Lse,       // [total_q, H]
    int H,
    float sm_scale,
    bool causal) {
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int q_block_id = blockIdx.x;
    const int b = blockIdx.y / H;
    const int h = blockIdx.y % H;

    const int q_start = cu_seqlens_q[b];
    const int seqlen_q = cu_seqlens_q[b + 1] - q_start;
    if (q_block_id * BLOCK_Q >= seqlen_q)
        return; // also makes zero-length sequences a no-op for every CTA assigned to them

    const int k_start = cu_seqlens_k[b];
    const int seqlen_k = cu_seqlens_k[b + 1] - k_start;
    const int causal_offset = seqlen_k - seqlen_q;

    const int hi_rows = causal ? min(seqlen_k, (q_block_id + 1) * BLOCK_Q + causal_offset)
                               : seqlen_k;
    const int num_kv_iter = cdiv(max(hi_rows, 0), BLOCK_KV);
    // Row indices computed from warp_id/lane_id below are local to this
    // Q-tile's shared-memory layout; add this to get the row's position
    // within the sequence (what seqlen_q/causal comparisons and the
    // Out/Lse global address need).
    const int q_tile_row_base = q_block_id * BLOCK_Q;

    extern __shared__ __align__(1024) char smem[];
    nv_bfloat16 *sQ = reinterpret_cast<nv_bfloat16 *>(smem);
    nv_bfloat16 *sK = sQ + BLOCK_Q * DIM;
    nv_bfloat16 *sV = sK + STAGES * BLOCK_KV * DIM;
    int *mbar_base = reinterpret_cast<int *>(sV + STAGES * BLOCK_KV * DIM);

    const uint64_t sQ_tma = __cvta_generic_to_shared(sQ);
    const uint64_t sK_tma = __cvta_generic_to_shared(sK);
    const uint64_t sV_tma = __cvta_generic_to_shared(sV);
    const uint32_t sQ_base = static_cast<uint32_t>(sQ_tma);
    const uint32_t sK_base = static_cast<uint32_t>(sK_tma);
    const uint32_t sV_base = static_cast<uint32_t>(sV_tma);

    const uint64_t mbar_q = __cvta_generic_to_shared(mbar_base);
    uint64_t mbar_kv[STAGES];
#pragma unroll
    for (int s = 0; s < STAGES; ++s)
        mbar_kv[s] = __cvta_generic_to_shared(mbar_base + 2 * (1 + s));

    if (tid == 0) {
        mbarrier_init(mbar_q, 1);
#pragma unroll
        for (int s = 0; s < STAGES; ++s)
            mbarrier_init(mbar_kv[s], 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    // Q: single TMA load for the whole CTA lifetime (Q doesn't change across
    // the KV loop).
    const uint32_t q_tile_bytes = BLOCK_Q * DIM * sizeof(nv_bfloat16);
    if (tid == 0) {
        mbarrier_arrive_expect_tx(mbar_q, q_tile_bytes);
        tma_2d_g2s(sQ_tma, &q_tmap, h * DIM, q_start + q_block_id * BLOCK_Q, mbar_q);
    }
    mbarrier_wait(mbar_q, 0);
    __syncthreads();

    uint32_t Q_smem_thread;
    {
        const int row_off = warp_id * WARP_Q + (lane_id % 16);
        const int col_off = (lane_id / 16) * 8;
        Q_smem_thread = sQ_base + (row_off * DIM + col_off) * sizeof(nv_bfloat16);
    }
    uint32_t K_smem_thread;
    {
        const int row_off = lane_id % 8;
        const int col_off = (lane_id / 8) * 8;
        K_smem_thread = sK_base + (row_off * DIM + col_off) * sizeof(nv_bfloat16);
    }
    uint32_t V_smem_thread;
    {
        const int row_off = lane_id % 16;
        const int col_off = (lane_id / 16) * 8;
        V_smem_thread = sV_base + (row_off * DIM + col_off) * sizeof(nv_bfloat16);
    }

    uint32_t Q_rmem[WARP_Q / MMA_M][DIM / MMA_K][4];
#pragma unroll
    for (int mi = 0; mi < WARP_Q / MMA_M; mi++)
#pragma unroll
        for (int kd = 0; kd < DIM / MMA_K; kd++) {
            uint32_t addr =
                Q_smem_thread + mi * MMA_M * DIM * sizeof(nv_bfloat16) + kd * MMA_K * sizeof(nv_bfloat16);
            ldmatrix_x4(Q_rmem[mi][kd], addr);
        }
    __syncthreads();

    if (num_kv_iter == 0) {
        // No valid keys for this Q-tile (e.g. seqlen_k == 0). Write a
        // degenerate all-masked result rather than leaving Out/Lse
        // uninitialized: out=0, lse=-inf-ish sentinel via log(tiny).
#pragma unroll
        for (int mi = 0; mi < WARP_Q / MMA_M; mi++) {
            const int row0 = q_tile_row_base + warp_id * WARP_Q + mi * MMA_M + lane_id / 4;
            if (lane_id % 4 == 0) {
                const float sentinel_lse = -CUDART_INF_F;
                if (row0 < seqlen_q)
                    Lse[(q_start + row0) * H + h] = sentinel_lse;
                if (row0 + 8 < seqlen_q)
                    Lse[(q_start + row0 + 8) * H + h] = sentinel_lse;
            }
#pragma unroll
            for (int d = 0; d < DIM / MMA_N; d++) {
                const int col = d * MMA_N + (lane_id % 4) * 2;
                nv_bfloat162 zero2 = __float22bfloat162_rn({0.0f, 0.0f});
                if (row0 < seqlen_q)
                    *reinterpret_cast<nv_bfloat162 *>(Out + ((q_start + row0) * H + h) * DIM + col) =
                        zero2;
                if (row0 + 8 < seqlen_q)
                    *reinterpret_cast<nv_bfloat162 *>(Out + ((q_start + row0 + 8) * H + h) * DIM +
                                                      col) = zero2;
            }
        }
        return;
    }

    const uint32_t kv_tile_bytes = 2 * BLOCK_KV * DIM * sizeof(nv_bfloat16); // K+V combined
    auto issue_kv = [&](int kv_id) {
        if (kv_id < 0 || kv_id >= num_kv_iter)
            return;
        const int buf = kv_id % STAGES;
        const uint32_t off = buf * BLOCK_KV * DIM * sizeof(nv_bfloat16);
        mbarrier_arrive_expect_tx(mbar_kv[buf], kv_tile_bytes);
        tma_2d_g2s(sK_tma + off, &k_tmap, h * DIM, k_start + kv_id * BLOCK_KV, mbar_kv[buf]);
        tma_2d_g2s(sV_tma + off, &v_tmap, h * DIM, k_start + kv_id * BLOCK_KV, mbar_kv[buf]);
    };

    int phase[STAGES];
#pragma unroll
    for (int s = 0; s < STAGES; ++s)
        phase[s] = 0;
    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES - 1; ++s)
            issue_kv(s);
    }

    uint32_t K_rmem[BLOCK_KV / MMA_N][DIM / MMA_K][2];
    uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];
    uint32_t V_rmem[BLOCK_KV / MMA_K][DIM / MMA_N][2];

    float O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4] = {};
    float rowmax[WARP_Q / MMA_M][2];
    float rowsumexp[WARP_Q / MMA_M][2] = {};
#pragma unroll
    for (int mi = 0; mi < WARP_Q / MMA_M; mi++) {
        rowmax[mi][0] = -FLT_MAX;
        rowmax[mi][1] = -FLT_MAX;
    }

    for (int kv_id = 0; kv_id < num_kv_iter; kv_id++) {
        const int buf = kv_id % STAGES;
        const uint32_t buf_off = buf * BLOCK_KV * DIM * sizeof(nv_bfloat16);

        if (tid == 0)
            issue_kv(kv_id + STAGES - 1);

        mbarrier_wait(mbar_kv[buf], phase[buf]);
        phase[buf] ^= 1;
        __syncthreads();

        float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {};

#pragma unroll
        for (int nkv = 0; nkv < BLOCK_KV / MMA_N; nkv++)
#pragma unroll
            for (int kd = 0; kd < DIM / MMA_K; kd += 2) {
                uint32_t addr = K_smem_thread + buf_off;
                addr += nkv * MMA_N * DIM * sizeof(nv_bfloat16);
                addr += kd * MMA_K * sizeof(nv_bfloat16);
                ldmatrix_x4(K_rmem[nkv][kd], addr);
            }

#pragma unroll
        for (int mi = 0; mi < WARP_Q / MMA_M; mi++)
#pragma unroll
            for (int nkv = 0; nkv < BLOCK_KV / MMA_N; nkv++)
#pragma unroll
                for (int kd = 0; kd < DIM / MMA_K; kd++)
                    mma_m16n8k16(Q_rmem[mi][kd], K_rmem[nkv][kd], S_rmem[mi][nkv]);

#pragma unroll
        for (int mi = 0; mi < WARP_Q / MMA_M; mi++) {
            const int row_a = q_tile_row_base + warp_id * WARP_Q + mi * MMA_M + lane_id / 4;
            const int row_b = row_a + 8;

#pragma unroll
            for (int nkv = 0; nkv < BLOCK_KV / MMA_N; nkv++) {
                float *regs = S_rmem[mi][nkv];
                const int col_a = kv_id * BLOCK_KV + nkv * MMA_N + (lane_id % 4) * 2;
                const int col_b = col_a + 1;
                regs[0] *= sm_scale;
                regs[1] *= sm_scale;
                regs[2] *= sm_scale;
                regs[3] *= sm_scale;
                if (col_a >= seqlen_k || (causal && row_a < col_a - causal_offset))
                    regs[0] = -CUDART_INF_F;
                if (col_b >= seqlen_k || (causal && row_a < col_b - causal_offset))
                    regs[1] = -CUDART_INF_F;
                if (col_a >= seqlen_k || (causal && row_b < col_a - causal_offset))
                    regs[2] = -CUDART_INF_F;
                if (col_b >= seqlen_k || (causal && row_b < col_b - causal_offset))
                    regs[3] = -CUDART_INF_F;
            }

            float this_rowmax[2];
#pragma unroll
            for (int nkv = 0; nkv < BLOCK_KV / MMA_N; nkv++) {
                float *regs = S_rmem[mi][nkv];
                if (nkv == 0) {
                    this_rowmax[0] = max(regs[0], regs[1]);
                    this_rowmax[1] = max(regs[2], regs[3]);
                } else {
                    this_rowmax[0] = max(this_rowmax[0], max(regs[0], regs[1]));
                    this_rowmax[1] = max(this_rowmax[1], max(regs[2], regs[3]));
                }
            }
            this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFFFFFFu, this_rowmax[0], 1));
            this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFFFFFFu, this_rowmax[0], 2));
            this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFFFFFFu, this_rowmax[1], 1));
            this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFFFFFFu, this_rowmax[1], 2));
            this_rowmax[0] = max(this_rowmax[0], rowmax[mi][0]);
            this_rowmax[1] = max(this_rowmax[1], rowmax[mi][1]);

            float rescale[2];
            rescale[0] = __expf(rowmax[mi][0] - this_rowmax[0]);
            rescale[1] = __expf(rowmax[mi][1] - this_rowmax[1]);
#pragma unroll
            for (int d = 0; d < DIM / MMA_N; d++) {
                O_rmem[mi][d][0] *= rescale[0];
                O_rmem[mi][d][1] *= rescale[0];
                O_rmem[mi][d][2] *= rescale[1];
                O_rmem[mi][d][3] *= rescale[1];
            }
            rowmax[mi][0] = this_rowmax[0];
            rowmax[mi][1] = this_rowmax[1];

            float this_rowsumexp[2];
#pragma unroll
            for (int nkv = 0; nkv < BLOCK_KV / MMA_N; nkv++) {
                float *regs = S_rmem[mi][nkv];
                regs[0] = __expf(regs[0] - rowmax[mi][0]);
                regs[1] = __expf(regs[1] - rowmax[mi][0]);
                regs[2] = __expf(regs[2] - rowmax[mi][1]);
                regs[3] = __expf(regs[3] - rowmax[mi][1]);
                if (nkv == 0) {
                    this_rowsumexp[0] = regs[0] + regs[1];
                    this_rowsumexp[1] = regs[2] + regs[3];
                } else {
                    this_rowsumexp[0] += regs[0] + regs[1];
                    this_rowsumexp[1] += regs[2] + regs[3];
                }

                nv_bfloat162 *pfrag = reinterpret_cast<nv_bfloat162 *>(P_rmem[mi][nkv / 2]);
                pfrag[(nkv % 2) * 2] = __float22bfloat162_rn({regs[0], regs[1]});
                pfrag[(nkv % 2) * 2 + 1] = __float22bfloat162_rn({regs[2], regs[3]});
            }
            this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFFu, this_rowsumexp[0], 1);
            this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFFu, this_rowsumexp[0], 2);
            this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFFu, this_rowsumexp[1], 1);
            this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFFu, this_rowsumexp[1], 2);
            rowsumexp[mi][0] = rowsumexp[mi][0] * rescale[0] + this_rowsumexp[0];
            rowsumexp[mi][1] = rowsumexp[mi][1] * rescale[1] + this_rowsumexp[1];
        }

#pragma unroll
        for (int nkv = 0; nkv < BLOCK_KV / MMA_K; nkv++)
#pragma unroll
            for (int d = 0; d < DIM / MMA_N; d += 2) {
                uint32_t addr = V_smem_thread + buf_off;
                addr += nkv * MMA_K * DIM * sizeof(nv_bfloat16);
                addr += d * MMA_N * sizeof(nv_bfloat16);
                ldmatrix_x4_trans(V_rmem[nkv][d], addr);
            }

#pragma unroll
        for (int mi = 0; mi < WARP_Q / MMA_M; mi++)
#pragma unroll
            for (int d = 0; d < DIM / MMA_N; d++)
#pragma unroll
                for (int nkv = 0; nkv < BLOCK_KV / MMA_K; nkv++)
                    mma_m16n8k16(P_rmem[mi][nkv], V_rmem[nkv][d], O_rmem[mi][d]);

        __syncthreads();
    }

#pragma unroll
    for (int mi = 0; mi < WARP_Q / MMA_M; mi++) {
        const int row0 = q_tile_row_base + warp_id * WARP_Q + mi * MMA_M + lane_id / 4;
        const float inv0 = 1.0f / fmaxf(rowsumexp[mi][0], 1e-30f);
        const float inv1 = 1.0f / fmaxf(rowsumexp[mi][1], 1e-30f);

        if (lane_id % 4 == 0) {
            const float lse0 = rowmax[mi][0] + logf(fmaxf(rowsumexp[mi][0], 1e-30f));
            const float lse1 = rowmax[mi][1] + logf(fmaxf(rowsumexp[mi][1], 1e-30f));
            if (row0 < seqlen_q)
                Lse[(q_start + row0) * H + h] = lse0;
            if (row0 + 8 < seqlen_q)
                Lse[(q_start + row0 + 8) * H + h] = lse1;
        }

#pragma unroll
        for (int d = 0; d < DIM / MMA_N; d++) {
            const int col = d * MMA_N + (lane_id % 4) * 2;
            float *regs = O_rmem[mi][d];
            regs[0] *= inv0;
            regs[1] *= inv0;
            regs[2] *= inv1;
            regs[3] *= inv1;

            if (row0 < seqlen_q)
                *reinterpret_cast<nv_bfloat162 *>(Out + ((q_start + row0) * H + h) * DIM + col) =
                    __float22bfloat162_rn({regs[0], regs[1]});
            if (row0 + 8 < seqlen_q)
                *reinterpret_cast<nv_bfloat162 *>(Out + ((q_start + row0 + 8) * H + h) * DIM + col) =
                    __float22bfloat162_rn({regs[2], regs[3]});
        }
    }
}

// Delta[row,h] = sum_d(Out[row,h,d] * DO[row,h,d]), the per-query-row term
// the backward recompute needs (`_bwd_preprocess_varlen` in triton_attn.py).
// One block per (row, head); DIM=128 threads, one per element, block reduce.
__global__ void flash_attention_bwd_preprocess_kernel(
    const nv_bfloat16 *__restrict__ Out,
    const nv_bfloat16 *__restrict__ DO,
    float *__restrict__ Delta,
    int total_q,
    int H) {
    const int row = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    if (row >= total_q)
        return;

    const int base = (row * H + h) * DIM;
    float val = __bfloat162float(Out[base + tid]) * __bfloat162float(DO[base + tid]);

#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFFu, val, off);

    __shared__ float warp_sums[DIM / WARP_SIZE];
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    if (lane == 0)
        warp_sums[warp] = val;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < DIM / WARP_SIZE; w++)
            total += warp_sums[w];
        Delta[row * H + h] = total;
    }
}

// Backward: grid over KV-tiles (the CTA's fixed, exclusively-owned dK/dV
// range), looping over Q-tiles internally -- the transpose of forward's
// grid-over-Q-tiles structure. Mirrors `_bwd_kernel_varlen` in
// triton_attn.py exactly (same lo-bound causal skip, same recompute-based
// algorithm using the saved `Lse` instead of an online rescale).
//
// P/ds only ever need to be read back as an mma.sync "A" operand (never
// "B"), so rather than using `ldmatrix_x4_trans` on their natural
// [query,key] layout (which is the trick this file uses for tensors sourced
// directly from HBM via TMA, e.g. V/dO/Q's second, transposed reads), P/ds
// are written to a small shared-memory scratch tile already
// row/col-swapped (key-major) by the store itself -- 4 independent scalar
// stores per thread, not vectorized, since the transposed target addresses
// aren't adjacent -- then read back with a plain (non-trans) `ldmatrix_x4`
// using the same addressing *shape* as this file's Q-as-A-operand load,
// substituting BLOCK_KV/BLOCK_Q for BLOCK_Q/DIM's roles.
__global__ __launch_bounds__(TB_SIZE) void flash_attention_varlen_sm90_bwd_kernel(
    const __grid_constant__ CUtensorMap q_tmap,
    const __grid_constant__ CUtensorMap k_tmap,
    const __grid_constant__ CUtensorMap v_tmap,
    const __grid_constant__ CUtensorMap do_tmap,
    const int *__restrict__ cu_seqlens_q,
    const int *__restrict__ cu_seqlens_k,
    const float *__restrict__ Lse,   // [total_q, H]
    const float *__restrict__ Delta, // [total_q, H]
    nv_bfloat16 *__restrict__ DQ,    // [total_q, H, DIM] -- atomic-add target
    nv_bfloat16 *__restrict__ DK,    // [total_k, H, DIM]
    nv_bfloat16 *__restrict__ DV,    // [total_k, H, DIM]
    int H,
    float sm_scale,
    bool causal) {
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int kv_block_id = blockIdx.x;
    const int b = blockIdx.y / H;
    const int h = blockIdx.y % H;

    const int k_start = cu_seqlens_k[b];
    const int seqlen_k = cu_seqlens_k[b + 1] - k_start;
    if (kv_block_id * BLOCK_KV >= seqlen_k)
        return;

    const int q_start = cu_seqlens_q[b];
    const int seqlen_q = cu_seqlens_q[b + 1] - q_start;
    const int causal_offset = seqlen_k - seqlen_q;
    const int kv_tile_row_base = kv_block_id * BLOCK_KV;

    // Query rows before `kv_tile_row_base - causal_offset` never attend to
    // this KV tile; the Q-loop below can start there instead of row 0. If
    // this pushes `lo >= seqlen_q` the loop below runs zero iterations and
    // dk/dv (register-accumulated, initialized to zero) are still written
    // unconditionally at the end -- correctly zero, no special case needed.
    int lo = causal ? max(0, kv_tile_row_base - causal_offset) : 0;
    lo = (lo / BLOCK_Q) * BLOCK_Q;

    extern __shared__ __align__(1024) char smem[];
    nv_bfloat16 *sK = reinterpret_cast<nv_bfloat16 *>(smem);
    nv_bfloat16 *sV = sK + BLOCK_KV * DIM;
    nv_bfloat16 *sQ = sV + BLOCK_KV * DIM;
    nv_bfloat16 *sDO = sQ + STAGES * BLOCK_Q * DIM;
    nv_bfloat16 *sPT = sDO + STAGES * BLOCK_Q * DIM; // [BLOCK_KV][BLOCK_Q] transpose scratch
    int *mbar_base = reinterpret_cast<int *>(sPT + BLOCK_KV * BLOCK_Q);

    const uint64_t sK_tma = __cvta_generic_to_shared(sK);
    const uint64_t sV_tma = __cvta_generic_to_shared(sV);
    const uint64_t sQ_tma = __cvta_generic_to_shared(sQ);
    const uint64_t sDO_tma = __cvta_generic_to_shared(sDO);
    const uint32_t sK_base = static_cast<uint32_t>(sK_tma);
    const uint32_t sV_base = static_cast<uint32_t>(sV_tma);
    const uint32_t sQ_base = static_cast<uint32_t>(sQ_tma);
    const uint32_t sDO_base = static_cast<uint32_t>(sDO_tma);
    const uint32_t sPT_base = static_cast<uint32_t>(__cvta_generic_to_shared(sPT));

    const uint64_t mbar_kv = __cvta_generic_to_shared(mbar_base);
    uint64_t mbar_qdo[STAGES];
#pragma unroll
    for (int s = 0; s < STAGES; ++s)
        mbar_qdo[s] = __cvta_generic_to_shared(mbar_base + 2 * (1 + s));

    if (tid == 0) {
        mbarrier_init(mbar_kv, 1);
#pragma unroll
        for (int s = 0; s < STAGES; ++s)
            mbarrier_init(mbar_qdo[s], 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    // K + V: single TMA load for this CTA's fixed KV-tile lifetime.
    const uint32_t kv_tile_bytes = 2 * BLOCK_KV * DIM * sizeof(nv_bfloat16);
    if (tid == 0) {
        mbarrier_arrive_expect_tx(mbar_kv, kv_tile_bytes);
        tma_2d_g2s(sK_tma, &k_tmap, h * DIM, k_start + kv_tile_row_base, mbar_kv);
        tma_2d_g2s(sV_tma, &v_tmap, h * DIM, k_start + kv_tile_row_base, mbar_kv);
    }
    mbarrier_wait(mbar_kv, 0);
    __syncthreads();

    // K/V "natural" addressing (this file's forward-K-in-QK^T role: B
    // operand, contracts over D, plain ldmatrix_x4).
    uint32_t K_smem_thread_nat;
    {
        const int row_off = lane_id % 8;
        const int col_off = (lane_id / 8) * 8;
        K_smem_thread_nat = sK_base + (row_off * DIM + col_off) * sizeof(nv_bfloat16);
    }
    uint32_t V_smem_thread_nat;
    {
        const int row_off = lane_id % 8;
        const int col_off = (lane_id / 8) * 8;
        V_smem_thread_nat = sV_base + (row_off * DIM + col_off) * sizeof(nv_bfloat16);
    }
    // K "transposed" addressing (this file's forward-V-in-P@V role: B
    // operand, contracts over the KV/seq dimension, ldmatrix_x4_trans) --
    // used for dQ = ds @ K.
    uint32_t K_smem_thread_trans;
    {
        const int row_off = lane_id % 16;
        const int col_off = (lane_id / 16) * 8;
        K_smem_thread_trans = sK_base + (row_off * DIM + col_off) * sizeof(nv_bfloat16);
    }
    // Q/DO "natural" addressing (forward-Q-in-QK^T role: A operand).
    uint32_t Q_smem_thread_nat;
    {
        const int row_off = warp_id * WARP_Q + (lane_id % 16);
        const int col_off = (lane_id / 16) * 8;
        Q_smem_thread_nat = sQ_base + (row_off * DIM + col_off) * sizeof(nv_bfloat16);
    }
    uint32_t DO_smem_thread_nat;
    {
        const int row_off = warp_id * WARP_Q + (lane_id % 16);
        const int col_off = (lane_id / 16) * 8;
        DO_smem_thread_nat = sDO_base + (row_off * DIM + col_off) * sizeof(nv_bfloat16);
    }
    // Q/DO "transposed" addressing (forward-V-in-P@V role: B operand,
    // contracts over the query dimension) -- Q for dK = ds^T @ Q, DO for
    // dV = P^T @ dO.
    uint32_t Q_smem_thread_trans;
    {
        const int row_off = lane_id % 16;
        const int col_off = (lane_id / 16) * 8;
        Q_smem_thread_trans = sQ_base + (row_off * DIM + col_off) * sizeof(nv_bfloat16);
    }
    uint32_t DO_smem_thread_trans;
    {
        const int row_off = lane_id % 16;
        const int col_off = (lane_id / 16) * 8;
        DO_smem_thread_trans = sDO_base + (row_off * DIM + col_off) * sizeof(nv_bfloat16);
    }
    // sPT[BLOCK_KV][BLOCK_Q] read-as-A-operand addressing: same shape as
    // this file's Q-as-A-operand read, with BLOCK_Q substituted for DIM as
    // the row stride (sPT's rows only span BLOCK_KV, not the full DIM) and
    // BLOCK_KV substituted for BLOCK_Q as the row extent being tiled across
    // warps (numerically identical here since BLOCK_Q == BLOCK_KV == 64).
    uint32_t PT_smem_thread;
    {
        const int row_off = warp_id * WARP_Q + (lane_id % 16);
        const int col_off = (lane_id / 16) * 8;
        PT_smem_thread = sPT_base + (row_off * BLOCK_Q + col_off) * sizeof(nv_bfloat16);
    }

    uint32_t K_rmem_nat[BLOCK_KV / MMA_N][DIM / MMA_K][2];
#pragma unroll
    for (int nkv = 0; nkv < BLOCK_KV / MMA_N; nkv++)
#pragma unroll
        for (int kd = 0; kd < DIM / MMA_K; kd += 2) {
            uint32_t addr = K_smem_thread_nat + nkv * MMA_N * DIM * sizeof(nv_bfloat16) +
                            kd * MMA_K * sizeof(nv_bfloat16);
            ldmatrix_x4(K_rmem_nat[nkv][kd], addr);
        }
    uint32_t V_rmem_nat[BLOCK_KV / MMA_N][DIM / MMA_K][2];
#pragma unroll
    for (int nkv = 0; nkv < BLOCK_KV / MMA_N; nkv++)
#pragma unroll
        for (int kd = 0; kd < DIM / MMA_K; kd += 2) {
            uint32_t addr = V_smem_thread_nat + nkv * MMA_N * DIM * sizeof(nv_bfloat16) +
                            kd * MMA_K * sizeof(nv_bfloat16);
            ldmatrix_x4(V_rmem_nat[nkv][kd], addr);
        }
    uint32_t K_rmem_trans[BLOCK_KV / MMA_K][DIM / MMA_N][2];
#pragma unroll
    for (int nkv = 0; nkv < BLOCK_KV / MMA_K; nkv++)
#pragma unroll
        for (int d = 0; d < DIM / MMA_N; d += 2) {
            uint32_t addr = K_smem_thread_trans + nkv * MMA_K * DIM * sizeof(nv_bfloat16) +
                            d * MMA_N * sizeof(nv_bfloat16);
            ldmatrix_x4_trans(K_rmem_trans[nkv][d], addr);
        }

    const uint32_t qdo_tile_bytes = 2 * BLOCK_Q * DIM * sizeof(nv_bfloat16);
    auto issue_qdo = [&](int q_id) {
        const int start_m = lo + q_id * BLOCK_Q;
        if (start_m < 0 || start_m >= seqlen_q)
            return;
        const int buf = q_id % STAGES;
        const uint32_t off = buf * BLOCK_Q * DIM * sizeof(nv_bfloat16);
        mbarrier_arrive_expect_tx(mbar_qdo[buf], qdo_tile_bytes);
        tma_2d_g2s(sQ_tma + off, &q_tmap, h * DIM, q_start + start_m, mbar_qdo[buf]);
        tma_2d_g2s(sDO_tma + off, &do_tmap, h * DIM, q_start + start_m, mbar_qdo[buf]);
    };

    int phase[STAGES];
#pragma unroll
    for (int s = 0; s < STAGES; ++s)
        phase[s] = 0;
    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES - 1; ++s)
            issue_qdo(s);
    }

    float dk_out[WARP_Q / MMA_M][DIM / MMA_N][4] = {};
    float dv_out[WARP_Q / MMA_M][DIM / MMA_N][4] = {};

    int q_id = 0;
    for (int start_m = lo; start_m < seqlen_q; start_m += BLOCK_Q, q_id++) {
        const int buf = q_id % STAGES;
        const uint32_t buf_off = buf * BLOCK_Q * DIM * sizeof(nv_bfloat16);

        if (tid == 0)
            issue_qdo(q_id + STAGES - 1);

        mbarrier_wait(mbar_qdo[buf], phase[buf]);
        phase[buf] ^= 1;
        __syncthreads();

        uint32_t Q_rmem_nat[WARP_Q / MMA_M][DIM / MMA_K][4];
#pragma unroll
        for (int mi = 0; mi < WARP_Q / MMA_M; mi++)
#pragma unroll
            for (int kd = 0; kd < DIM / MMA_K; kd++) {
                uint32_t addr = Q_smem_thread_nat + buf_off + mi * MMA_M * DIM * sizeof(nv_bfloat16) +
                                kd * MMA_K * sizeof(nv_bfloat16);
                ldmatrix_x4(Q_rmem_nat[mi][kd], addr);
            }

        // S = Q @ K^T (identical pattern to forward's QK^T).
        float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {};
#pragma unroll
        for (int mi = 0; mi < WARP_Q / MMA_M; mi++)
#pragma unroll
            for (int nkv = 0; nkv < BLOCK_KV / MMA_N; nkv++)
#pragma unroll
                for (int kd = 0; kd < DIM / MMA_K; kd++)
                    mma_m16n8k16(Q_rmem_nat[mi][kd], K_rmem_nat[nkv][kd], S_rmem[mi][nkv]);

        // dP = DO @ V^T (same operand shapes as S = Q @ K^T: DO as A, V-nat as B).
        uint32_t DO_rmem_nat[WARP_Q / MMA_M][DIM / MMA_K][4];
#pragma unroll
        for (int mi = 0; mi < WARP_Q / MMA_M; mi++)
#pragma unroll
            for (int kd = 0; kd < DIM / MMA_K; kd++) {
                uint32_t addr = DO_smem_thread_nat + buf_off +
                                mi * MMA_M * DIM * sizeof(nv_bfloat16) + kd * MMA_K * sizeof(nv_bfloat16);
                ldmatrix_x4(DO_rmem_nat[mi][kd], addr);
            }
        float dP_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {};
#pragma unroll
        for (int mi = 0; mi < WARP_Q / MMA_M; mi++)
#pragma unroll
            for (int nkv = 0; nkv < BLOCK_KV / MMA_N; nkv++)
#pragma unroll
                for (int kd = 0; kd < DIM / MMA_K; kd++)
                    mma_m16n8k16(DO_rmem_nat[mi][kd], V_rmem_nat[nkv][kd], dP_rmem[mi][nkv]);

        uint32_t P_pack[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];
        uint32_t ds_pack[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];

#pragma unroll
        for (int mi = 0; mi < WARP_Q / MMA_M; mi++) {
            const int row_a = start_m + warp_id * WARP_Q + mi * MMA_M + lane_id / 4;
            const int row_b = row_a + 8;
            const float lse_a = (row_a < seqlen_q) ? Lse[(q_start + row_a) * H + h] : 0.0f;
            const float lse_b = (row_b < seqlen_q) ? Lse[(q_start + row_b) * H + h] : 0.0f;
            const float delta_a = (row_a < seqlen_q) ? Delta[(q_start + row_a) * H + h] : 0.0f;
            const float delta_b = (row_b < seqlen_q) ? Delta[(q_start + row_b) * H + h] : 0.0f;

#pragma unroll
            for (int nkv = 0; nkv < BLOCK_KV / MMA_N; nkv++) {
                float *s = S_rmem[mi][nkv];
                float *dp = dP_rmem[mi][nkv];
                // Local (0..BLOCK_KV-1, for sPT scratch addressing) vs.
                // global-within-sequence (for seqlen_k/causal comparisons)
                // column index -- sPT is only BLOCK_KV rows, so it must be
                // indexed local-to-this-CTA's-KV-tile, not by the global
                // column (which can exceed BLOCK_KV once kv_tile_row_base > 0).
                const int col_local_a = nkv * MMA_N + (lane_id % 4) * 2;
                const int col_local_b = col_local_a + 1;
                const int col_a = kv_tile_row_base + col_local_a;
                const int col_b = col_a + 1;

                const bool keep00 = col_a < seqlen_k && row_a < seqlen_q &&
                                    (!causal || row_a >= col_a - causal_offset);
                const bool keep01 = col_b < seqlen_k && row_a < seqlen_q &&
                                    (!causal || row_a >= col_b - causal_offset);
                const bool keep10 = col_a < seqlen_k && row_b < seqlen_q &&
                                    (!causal || row_b >= col_a - causal_offset);
                const bool keep11 = col_b < seqlen_k && row_b < seqlen_q &&
                                    (!causal || row_b >= col_b - causal_offset);

                const float p00 = keep00 ? __expf(s[0] * sm_scale - lse_a) : 0.0f;
                const float p01 = keep01 ? __expf(s[1] * sm_scale - lse_a) : 0.0f;
                const float p10 = keep10 ? __expf(s[2] * sm_scale - lse_b) : 0.0f;
                const float p11 = keep11 ? __expf(s[3] * sm_scale - lse_b) : 0.0f;

                const float ds00 = p00 * (dp[0] - delta_a) * sm_scale;
                const float ds01 = p01 * (dp[1] - delta_a) * sm_scale;
                const float ds10 = p10 * (dp[2] - delta_b) * sm_scale;
                const float ds11 = p11 * (dp[3] - delta_b) * sm_scale;

                // Natural-orientation ds pack (A operand for dQ = ds @ K),
                // same 2-nkv-per-slot packing convention forward uses for P.
                nv_bfloat162 *ds_frag = reinterpret_cast<nv_bfloat162 *>(ds_pack[mi][nkv / 2]);
                ds_frag[(nkv % 2) * 2] = __float22bfloat162_rn({ds00, ds01});
                ds_frag[(nkv % 2) * 2 + 1] = __float22bfloat162_rn({ds10, ds11});

                // Transposed-write scratch for P (this iteration's dV) and,
                // after dV is consumed below, ds (this iteration's dK) --
                // key-major: sPT[key][query], 4 independent scalar stores.
                const int local_row_a = warp_id * WARP_Q + mi * MMA_M + lane_id / 4;
                const int local_row_b = local_row_a + 8;
                sPT[col_local_a * BLOCK_Q + local_row_a] = __float2bfloat16_rn(p00);
                sPT[col_local_b * BLOCK_Q + local_row_a] = __float2bfloat16_rn(p01);
                sPT[col_local_a * BLOCK_Q + local_row_b] = __float2bfloat16_rn(p10);
                sPT[col_local_b * BLOCK_Q + local_row_b] = __float2bfloat16_rn(p11);
            }
        }
        __syncthreads();

        // dV += P^T @ DO (P read back key-major/natural from sPT as A;
        // DO's transposed read as B).
        uint32_t PT_rmem[WARP_Q / MMA_M][BLOCK_Q / MMA_K][4];
#pragma unroll
        for (int mi = 0; mi < WARP_Q / MMA_M; mi++)
#pragma unroll
            for (int kd = 0; kd < BLOCK_Q / MMA_K; kd++) {
                uint32_t addr =
                    PT_smem_thread + mi * MMA_M * BLOCK_Q * sizeof(nv_bfloat16) + kd * MMA_K * sizeof(nv_bfloat16);
                ldmatrix_x4(PT_rmem[mi][kd], addr);
            }
        uint32_t DO_rmem_trans[BLOCK_Q / MMA_K][DIM / MMA_N][2];
#pragma unroll
        for (int nkv = 0; nkv < BLOCK_Q / MMA_K; nkv++)
#pragma unroll
            for (int d = 0; d < DIM / MMA_N; d += 2) {
                uint32_t addr = DO_smem_thread_trans + buf_off +
                                nkv * MMA_K * DIM * sizeof(nv_bfloat16) + d * MMA_N * sizeof(nv_bfloat16);
                ldmatrix_x4_trans(DO_rmem_trans[nkv][d], addr);
            }
#pragma unroll
        for (int mi = 0; mi < WARP_Q / MMA_M; mi++)
#pragma unroll
            for (int d = 0; d < DIM / MMA_N; d++)
#pragma unroll
                for (int kd = 0; kd < BLOCK_Q / MMA_K; kd++)
                    mma_m16n8k16(PT_rmem[mi][kd], DO_rmem_trans[kd][d], dv_out[mi][d]);

        __syncthreads(); // sPT fully consumed by every thread before ds overwrites it

        // Re-issue the ds transposed write into the same sPT scratch.
#pragma unroll
        for (int mi = 0; mi < WARP_Q / MMA_M; mi++) {
            const int local_row_a = warp_id * WARP_Q + mi * MMA_M + lane_id / 4;
            const int local_row_b = local_row_a + 8;
#pragma unroll
            for (int nkv = 0; nkv < BLOCK_KV / MMA_N; nkv++) {
                // Local (0..BLOCK_KV-1) column -- sPT-only, no masking use
                // here, so no global-within-sequence variant is needed.
                const int col_a = nkv * MMA_N + (lane_id % 4) * 2;
                const int col_b = col_a + 1;
                const nv_bfloat162 *ds_frag =
                    reinterpret_cast<const nv_bfloat162 *>(ds_pack[mi][nkv / 2]) + (nkv % 2) * 2;
                const nv_bfloat162 v0 = ds_frag[0];
                const nv_bfloat162 v1 = ds_frag[1];
                sPT[col_a * BLOCK_Q + local_row_a] = v0.x;
                sPT[col_b * BLOCK_Q + local_row_a] = v0.y;
                sPT[col_a * BLOCK_Q + local_row_b] = v1.x;
                sPT[col_b * BLOCK_Q + local_row_b] = v1.y;
            }
        }
        __syncthreads();

        // dK += ds^T @ Q (ds read back key-major/natural from sPT as A;
        // Q's transposed read as B).
        uint32_t dsT_rmem[WARP_Q / MMA_M][BLOCK_Q / MMA_K][4];
#pragma unroll
        for (int mi = 0; mi < WARP_Q / MMA_M; mi++)
#pragma unroll
            for (int kd = 0; kd < BLOCK_Q / MMA_K; kd++) {
                uint32_t addr =
                    PT_smem_thread + mi * MMA_M * BLOCK_Q * sizeof(nv_bfloat16) + kd * MMA_K * sizeof(nv_bfloat16);
                ldmatrix_x4(dsT_rmem[mi][kd], addr);
            }
        uint32_t Q_rmem_trans[BLOCK_Q / MMA_K][DIM / MMA_N][2];
#pragma unroll
        for (int nkv = 0; nkv < BLOCK_Q / MMA_K; nkv++)
#pragma unroll
            for (int d = 0; d < DIM / MMA_N; d += 2) {
                uint32_t addr = Q_smem_thread_trans + buf_off +
                                nkv * MMA_K * DIM * sizeof(nv_bfloat16) + d * MMA_N * sizeof(nv_bfloat16);
                ldmatrix_x4_trans(Q_rmem_trans[nkv][d], addr);
            }
#pragma unroll
        for (int mi = 0; mi < WARP_Q / MMA_M; mi++)
#pragma unroll
            for (int d = 0; d < DIM / MMA_N; d++)
#pragma unroll
                for (int kd = 0; kd < BLOCK_Q / MMA_K; kd++)
                    mma_m16n8k16(dsT_rmem[mi][kd], Q_rmem_trans[kd][d], dk_out[mi][d]);

        // dQ += ds @ K (ds natural/registers as A -- no transpose needed,
        // output row = query matches ds's natural row already; K's
        // transposed read as B). Atomic-add straight into bf16 DQ (native
        // atom.add.noftz.bf16 on SM90).
        float dQ_acc[WARP_Q / MMA_M][DIM / MMA_N][4] = {};
#pragma unroll
        for (int mi = 0; mi < WARP_Q / MMA_M; mi++)
#pragma unroll
            for (int d = 0; d < DIM / MMA_N; d++)
#pragma unroll
                for (int kd = 0; kd < BLOCK_KV / MMA_K; kd++)
                    mma_m16n8k16(ds_pack[mi][kd], K_rmem_trans[kd][d], dQ_acc[mi][d]);

#pragma unroll
        for (int mi = 0; mi < WARP_Q / MMA_M; mi++) {
            const int row_a = start_m + warp_id * WARP_Q + mi * MMA_M + lane_id / 4;
            const int row_b = row_a + 8;
#pragma unroll
            for (int d = 0; d < DIM / MMA_N; d++) {
                const int col = d * MMA_N + (lane_id % 4) * 2;
                float *regs = dQ_acc[mi][d];
                if (row_a < seqlen_q) {
                    atomicAdd(&DQ[((q_start + row_a) * H + h) * DIM + col], __float2bfloat16_rn(regs[0]));
                    atomicAdd(&DQ[((q_start + row_a) * H + h) * DIM + col + 1],
                              __float2bfloat16_rn(regs[1]));
                }
                if (row_b < seqlen_q) {
                    atomicAdd(&DQ[((q_start + row_b) * H + h) * DIM + col], __float2bfloat16_rn(regs[2]));
                    atomicAdd(&DQ[((q_start + row_b) * H + h) * DIM + col + 1],
                              __float2bfloat16_rn(regs[3]));
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int mi = 0; mi < WARP_Q / MMA_M; mi++) {
        const int row = kv_tile_row_base + warp_id * WARP_Q + mi * MMA_M + lane_id / 4;
        const int row8 = row + 8;
#pragma unroll
        for (int d = 0; d < DIM / MMA_N; d++) {
            const int col = d * MMA_N + (lane_id % 4) * 2;
            float *dk_regs = dk_out[mi][d];
            float *dv_regs = dv_out[mi][d];
            if (row < seqlen_k) {
                DK[((k_start + row) * H + h) * DIM + col] = __float2bfloat16_rn(dk_regs[0]);
                DK[((k_start + row) * H + h) * DIM + col + 1] = __float2bfloat16_rn(dk_regs[1]);
                DV[((k_start + row) * H + h) * DIM + col] = __float2bfloat16_rn(dv_regs[0]);
                DV[((k_start + row) * H + h) * DIM + col + 1] = __float2bfloat16_rn(dv_regs[1]);
            }
            if (row8 < seqlen_k) {
                DK[((k_start + row8) * H + h) * DIM + col] = __float2bfloat16_rn(dk_regs[2]);
                DK[((k_start + row8) * H + h) * DIM + col + 1] = __float2bfloat16_rn(dk_regs[3]);
                DV[((k_start + row8) * H + h) * DIM + col] = __float2bfloat16_rn(dv_regs[2]);
                DV[((k_start + row8) * H + h) * DIM + col + 1] = __float2bfloat16_rn(dv_regs[3]);
            }
        }
    }
}

} // namespace

// Forward-only packed varlen FlashAttention (SM90, TMA + mma.sync). q/k/v are
// bf16 [total, H, DIM=128], contiguous, no GQA. cu_seqlens_{q,k} are int32
// [batch+1] cumulative offsets (cu_seqlens[0]==0), same convention as
// rl_engine/kernels/ops/triton/triton_attn.py's varlen path. Returns
// (out [total_q, H, DIM] bf16, lse [total_q, H] f32) -- lse is the
// attention-domain log-sum-exp (M + log(L)), not the vocab-domain LSE used by
// the fused_logp/linear_logp kernels.
std::vector<torch::Tensor> flash_attention_varlen_sm90_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor cu_seqlens_q,
    torch::Tensor cu_seqlens_k,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    bool causal,
    double sm_scale) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA tensors");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
                "q/k/v must be on the same device");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
                    v.scalar_type() == at::kBFloat16,
                "flash_attention_varlen_sm90 requires bfloat16 q/k/v");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q/k/v must be [total, H, D]");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "flash_attention_varlen_sm90 requires contiguous q/k/v (no transposed/strided "
                "views); the Triton varlen path tolerates non-contiguous inputs, this kernel "
                "does not");

    const int total_q = q.size(0);
    const int H = q.size(1);
    const int D = q.size(2);
    TORCH_CHECK(D == DIM, "flash_attention_varlen_sm90 only supports head_dim=", DIM,
                " in this milestone, got ", D);
    TORCH_CHECK(k.size(1) == H && v.size(1) == H,
                "GQA is not supported: k/v head count must equal q's");
    TORCH_CHECK(k.size(2) == D && v.size(2) == D, "k/v head_dim must match q's");
    const int total_k = k.size(0);
    TORCH_CHECK(v.size(0) == total_k, "k and v must have the same total token count");

    TORCH_CHECK(cu_seqlens_q.is_cuda() && cu_seqlens_k.is_cuda(), "cu_seqlens must be on CUDA");
    TORCH_CHECK(cu_seqlens_q.scalar_type() == at::kInt && cu_seqlens_k.scalar_type() == at::kInt,
                "cu_seqlens_q/cu_seqlens_k must be int32");
    TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_k.dim() == 1, "cu_seqlens must be 1-D");
    TORCH_CHECK(cu_seqlens_q.is_contiguous() && cu_seqlens_k.is_contiguous(),
                "cu_seqlens must be contiguous");
    const int batch = cu_seqlens_q.numel() - 1;
    TORCH_CHECK(batch >= 1, "cu_seqlens_q must have at least 2 elements");
    TORCH_CHECK(cu_seqlens_k.numel() - 1 == batch, "cu_seqlens_q/cu_seqlens_k batch mismatch");

    at::cuda::CUDAGuard device_guard(q.device());

    auto out = torch::empty_like(q);
    auto lse = torch::empty({total_q, H}, q.options().dtype(torch::kFloat));

    CUtensorMap q_tmap, k_tmap, v_tmap;
    init_tensor_map_noswizzle(&q_tmap, reinterpret_cast<const nv_bfloat16 *>(q.data_ptr<at::BFloat16>()),
                              total_q, H * D, BLOCK_Q, D);
    init_tensor_map_noswizzle(&k_tmap, reinterpret_cast<const nv_bfloat16 *>(k.data_ptr<at::BFloat16>()),
                              total_k, H * D, BLOCK_KV, D);
    init_tensor_map_noswizzle(&v_tmap, reinterpret_cast<const nv_bfloat16 *>(v.data_ptr<at::BFloat16>()),
                              total_k, H * D, BLOCK_KV, D);

    const int smem = static_cast<int>((BLOCK_Q * DIM + 2 * STAGES * BLOCK_KV * DIM) *
                                          sizeof(nv_bfloat16) +
                                      (1 + STAGES) * 8);
    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(flash_attention_varlen_sm90_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }

    dim3 grid(cdiv(static_cast<int>(max_seqlen_q), BLOCK_Q), batch * H, 1);
    flash_attention_varlen_sm90_kernel<<<grid, TB_SIZE, smem, at::cuda::getCurrentCUDAStream()>>>(
        q_tmap, k_tmap, v_tmap, cu_seqlens_q.data_ptr<int>(), cu_seqlens_k.data_ptr<int>(),
        reinterpret_cast<nv_bfloat16 *>(out.data_ptr<at::BFloat16>()), lse.data_ptr<float>(), H,
        static_cast<float>(sm_scale), causal);

    return {out, lse};
}

// Backward for flash_attention_varlen_sm90_forward: dQ/dK/dV via the
// recompute-based algorithm in triton_attn.py's `_bwd_kernel_varlen`
// (grid over KV-tiles, Q-tiles looped internally), using the forward's
// saved `lse` directly (p = exp(qk*scale - lse), algebraically identical to
// Triton's separate-M/L form -- see the comment on this in the plan/PR).
// `dq` is atomic-accumulated in bf16 (native SM90 `atom.add.noftz.bf16`),
// `dk`/`dv` are written once per KV-tile CTA (no atomics needed there).
std::vector<torch::Tensor> flash_attention_varlen_sm90_backward(
    torch::Tensor do_,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor out,
    torch::Tensor lse,
    torch::Tensor cu_seqlens_q,
    torch::Tensor cu_seqlens_k,
    int64_t max_seqlen_k,
    bool causal,
    double sm_scale) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && do_.is_cuda() && out.is_cuda() &&
                    lse.is_cuda(),
                "flash_attention_varlen_sm90_backward requires CUDA tensors");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
                    v.scalar_type() == at::kBFloat16 && do_.scalar_type() == at::kBFloat16 &&
                    out.scalar_type() == at::kBFloat16,
                "flash_attention_varlen_sm90_backward requires bfloat16 q/k/v/do/out");
    TORCH_CHECK(lse.scalar_type() == at::kFloat, "lse must be float32");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3 && do_.dim() == 3 && out.dim() == 3,
                "q/k/v/do/out must be [total, H, D]");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && do_.is_contiguous() &&
                    out.is_contiguous() && lse.is_contiguous(),
                "flash_attention_varlen_sm90_backward requires contiguous q/k/v/do/out/lse");

    const int total_q = q.size(0);
    const int H = q.size(1);
    const int D = q.size(2);
    TORCH_CHECK(D == DIM, "flash_attention_varlen_sm90_backward only supports head_dim=", DIM,
                " in this milestone, got ", D);
    TORCH_CHECK(k.size(1) == H && v.size(1) == H,
                "GQA is not supported: k/v head count must equal q's");
    TORCH_CHECK(k.size(2) == D && v.size(2) == D, "k/v head_dim must match q's");
    TORCH_CHECK(do_.sizes() == q.sizes() && out.sizes() == q.sizes(),
                "do/out must have the same shape as q");
    TORCH_CHECK(lse.size(0) == total_q && lse.size(1) == H, "lse must be [total_q, H]");
    const int total_k = k.size(0);
    TORCH_CHECK(v.size(0) == total_k, "k and v must have the same total token count");

    TORCH_CHECK(cu_seqlens_q.is_cuda() && cu_seqlens_k.is_cuda(), "cu_seqlens must be on CUDA");
    TORCH_CHECK(cu_seqlens_q.scalar_type() == at::kInt && cu_seqlens_k.scalar_type() == at::kInt,
                "cu_seqlens_q/cu_seqlens_k must be int32");
    TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_k.dim() == 1, "cu_seqlens must be 1-D");
    TORCH_CHECK(cu_seqlens_q.is_contiguous() && cu_seqlens_k.is_contiguous(),
                "cu_seqlens must be contiguous");
    const int batch = cu_seqlens_q.numel() - 1;
    TORCH_CHECK(batch >= 1, "cu_seqlens_q must have at least 2 elements");
    TORCH_CHECK(cu_seqlens_k.numel() - 1 == batch, "cu_seqlens_q/cu_seqlens_k batch mismatch");

    at::cuda::CUDAGuard device_guard(q.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto dq = torch::zeros_like(q); // atomic-accumulated, must start at zero
    auto dk = torch::empty_like(k);
    auto dv = torch::empty_like(v);
    auto delta = torch::empty({total_q, H}, q.options().dtype(torch::kFloat));

    flash_attention_bwd_preprocess_kernel<<<dim3(total_q, H), DIM, 0, stream>>>(
        reinterpret_cast<const nv_bfloat16 *>(out.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16 *>(do_.data_ptr<at::BFloat16>()), delta.data_ptr<float>(),
        total_q, H);

    CUtensorMap q_tmap, k_tmap, v_tmap, do_tmap;
    init_tensor_map_noswizzle(&q_tmap, reinterpret_cast<const nv_bfloat16 *>(q.data_ptr<at::BFloat16>()),
                              total_q, H * D, BLOCK_Q, D);
    init_tensor_map_noswizzle(&k_tmap, reinterpret_cast<const nv_bfloat16 *>(k.data_ptr<at::BFloat16>()),
                              total_k, H * D, BLOCK_KV, D);
    init_tensor_map_noswizzle(&v_tmap, reinterpret_cast<const nv_bfloat16 *>(v.data_ptr<at::BFloat16>()),
                              total_k, H * D, BLOCK_KV, D);
    init_tensor_map_noswizzle(&do_tmap,
                              reinterpret_cast<const nv_bfloat16 *>(do_.data_ptr<at::BFloat16>()), total_q,
                              H * D, BLOCK_Q, D);

    const int smem = static_cast<int>((2 * BLOCK_KV * DIM + 2 * STAGES * BLOCK_Q * DIM +
                                          BLOCK_KV * BLOCK_Q) *
                                          sizeof(nv_bfloat16) +
                                      (1 + STAGES) * 8);
    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(flash_attention_varlen_sm90_bwd_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }

    dim3 grid(cdiv(static_cast<int>(max_seqlen_k), BLOCK_KV), batch * H, 1);
    flash_attention_varlen_sm90_bwd_kernel<<<grid, TB_SIZE, smem, stream>>>(
        q_tmap, k_tmap, v_tmap, do_tmap, cu_seqlens_q.data_ptr<int>(), cu_seqlens_k.data_ptr<int>(),
        lse.data_ptr<float>(), delta.data_ptr<float>(),
        reinterpret_cast<nv_bfloat16 *>(dq.data_ptr<at::BFloat16>()),
        reinterpret_cast<nv_bfloat16 *>(dk.data_ptr<at::BFloat16>()),
        reinterpret_cast<nv_bfloat16 *>(dv.data_ptr<at::BFloat16>()), H, static_cast<float>(sm_scale),
        causal);

    return {dq, dk, dv};
}
