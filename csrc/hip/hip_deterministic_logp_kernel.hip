// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// ROCm-tuned kernels for the WS2 vocab-parallel logprob backend
// (rocm-vocab-parallel-logp-ws2). This file is compiled only for ROCm builds;
// the shared csrc/deterministic_logp_kernel.cu keeps the SM90-tuned CUDA path.
//
// Every per-tile reduction below uses an element-to-thread assignment and an
// accumulation order fixed by (BlockSize, Vec) alone, measured from the start
// of the tile and independent of the storage dtype. A tile therefore produces
// identical bits whichever TP rank owns it, wherever it sits inside that
// rank's local shard, and whether the shard is BF16, FP16, or FP32. The fixed
// global tile-order merge in Python relies on exactly that.
//
// Tuned on MI300X (gfx942): 8-element vector loads straight from the stored
// shard (no FP32 copy), a 128-thread block per (row, tile), and a fused
// elementwise backward. Build-time knobs are injected by setup.py from the
// environment variables of the same name.

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPException.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <torch/extension.h>

#ifndef DETERMINISTIC_LOGP_TILE_BLOCK_SIZE
#define DETERMINISTIC_LOGP_TILE_BLOCK_SIZE 128
#endif
#ifndef DETERMINISTIC_LOGP_TILE_VECTOR_ELEMENTS
#define DETERMINISTIC_LOGP_TILE_VECTOR_ELEMENTS 8
#endif
#ifndef DETERMINISTIC_LOGP_BACKWARD_BLOCK_SIZE
#define DETERMINISTIC_LOGP_BACKWARD_BLOCK_SIZE 256
#endif

namespace {

constexpr int kTileBlockSize = DETERMINISTIC_LOGP_TILE_BLOCK_SIZE;
constexpr int kTileVectorElements = DETERMINISTIC_LOGP_TILE_VECTOR_ELEMENTS;
constexpr int kBackwardBlockSize = DETERMINISTIC_LOGP_BACKWARD_BLOCK_SIZE;
// Logical 32-lane warps on a 64-wide wavefront keep the reduction tree identical
// to the shared CUDA kernel's topology.
constexpr int kLogicalWarpSize = 32;

static_assert(kTileBlockSize % kLogicalWarpSize == 0, "tile block must be warp-aligned");
static_assert(kTileBlockSize <= 1024, "tile block too large");
static_assert(kTileVectorElements >= 1, "vector width must be at least one element");
static_assert(kBackwardBlockSize % kLogicalWarpSize == 0, "backward block must be warp-aligned");

template <typename T>
__device__ __forceinline__ T shfl_down_32(T value, unsigned int delta) {
    return __shfl_down(value, delta, kLogicalWarpSize);
}

// Fixed-tree block max. Threads without a warp value contribute -inf, so a tile
// with no real-vocabulary column reduces to the (-inf, 0) identity partial.
template <int BlockSize>
__device__ __forceinline__ float block_reduce_max(float val) {
    constexpr int WarpCount = BlockSize / kLogicalWarpSize;
    __shared__ float shared[WarpCount];
    const int lane = threadIdx.x & (kLogicalWarpSize - 1);
    const int wid = threadIdx.x / kLogicalWarpSize;
#pragma unroll
    for (int offset = kLogicalWarpSize / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, shfl_down_32(val, offset));
    }
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    const bool has_warp_value = threadIdx.x < WarpCount;
    val = has_warp_value ? shared[threadIdx.x] : -std::numeric_limits<float>::infinity();
    if (wid == 0) {
#pragma unroll
        for (int offset = kLogicalWarpSize / 2; offset > 0; offset >>= 1) {
            val = fmaxf(val, shfl_down_32(val, offset));
        }
    }
    return val;
}

template <int BlockSize>
__device__ __forceinline__ float block_reduce_sum(float val) {
    constexpr int WarpCount = BlockSize / kLogicalWarpSize;
    __shared__ float shared[WarpCount];
    const int lane = threadIdx.x & (kLogicalWarpSize - 1);
    const int wid = threadIdx.x / kLogicalWarpSize;
#pragma unroll
    for (int offset = kLogicalWarpSize / 2; offset > 0; offset >>= 1) {
        val += shfl_down_32(val, offset);
    }
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    const bool has_warp_value = threadIdx.x < WarpCount;
    val = has_warp_value ? shared[threadIdx.x] : 0.0f;
    if (wid == 0) {
#pragma unroll
        for (int offset = kLogicalWarpSize / 2; offset > 0; offset >>= 1) {
            val += shfl_down_32(val, offset);
        }
    }
    return val;
}

template <int Bytes>
struct Packed;
template <>
struct Packed<16> {
    using type = uint4;
};
template <>
struct Packed<8> {
    using type = uint2;
};
template <>
struct Packed<4> {
    using type = unsigned int;
};
template <>
struct Packed<2> {
    using type = unsigned short;
};

// Vector load/store of Vec consecutive elements in 16-byte (or smaller) pieces.
// Alignment only selects the instruction; the values and their order are the
// same either way.
template <typename scalar_t, int Vec>
__device__ __forceinline__ void load_vector(const scalar_t* __restrict__ ptr, scalar_t (&out)[Vec]) {
    constexpr int Bytes = Vec * static_cast<int>(sizeof(scalar_t));
    constexpr int PieceBytes = Bytes < 16 ? Bytes : 16;
    constexpr int Pieces = Bytes / PieceBytes;
    static_assert(Bytes % PieceBytes == 0, "vector width must split into whole pieces");
    using packed_t = typename Packed<PieceBytes>::type;
    if ((reinterpret_cast<uintptr_t>(ptr) % alignof(packed_t)) == 0) {
        packed_t packed[Pieces];
#pragma unroll
        for (int piece = 0; piece < Pieces; ++piece) {
            packed[piece] = reinterpret_cast<const packed_t*>(ptr)[piece];
        }
        __builtin_memcpy(out, packed, Bytes);
    } else {
#pragma unroll
        for (int i = 0; i < Vec; ++i) {
            out[i] = ptr[i];
        }
    }
}

template <typename scalar_t, int Vec>
__device__ __forceinline__ void store_vector(scalar_t* __restrict__ ptr, const scalar_t (&in)[Vec]) {
    constexpr int Bytes = Vec * static_cast<int>(sizeof(scalar_t));
    constexpr int PieceBytes = Bytes < 16 ? Bytes : 16;
    constexpr int Pieces = Bytes / PieceBytes;
    using packed_t = typename Packed<PieceBytes>::type;
    if ((reinterpret_cast<uintptr_t>(ptr) % alignof(packed_t)) == 0) {
        packed_t packed[Pieces];
        __builtin_memcpy(packed, in, Bytes);
#pragma unroll
        for (int piece = 0; piece < Pieces; ++piece) {
            reinterpret_cast<packed_t*>(ptr)[piece] = packed[piece];
        }
    } else {
#pragma unroll
        for (int i = 0; i < Vec; ++i) {
            ptr[i] = in[i];
        }
    }
}

// Per-row, per-tile FP32 (max, sumexp) partials over the real-vocabulary part
// of the tile. Two passes over a tile that stays L2-resident: max, then sumexp.
template <typename scalar_t, int BlockSize, int Vec>
__global__ void __launch_bounds__(BlockSize) hip_tile_stats_kernel(
    const scalar_t* __restrict__ logits,
    float* __restrict__ tile_max,
    float* __restrict__ tile_sum,
    int64_t rows,
    int64_t local_vocab,
    int64_t vocab_start,
    int64_t real_vocab,
    int64_t tile_size,
    int64_t local_tiles) {
    constexpr int Chunk = BlockSize * Vec;
    const int64_t tile_index = static_cast<int64_t>(blockIdx.y);
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows || tile_index >= local_tiles) {
        return;
    }
    const int64_t col_begin = tile_index * tile_size;
    const int64_t col_end = min(col_begin + tile_size, local_vocab);
    // Columns at or beyond the real vocabulary are padding and contribute nothing.
    const int64_t real_end = min(col_end, max(real_vocab - vocab_start, static_cast<int64_t>(0)));
    const scalar_t* __restrict__ row_ptr = logits + row * local_vocab;

    float local_max = -std::numeric_limits<float>::infinity();
    for (int64_t base = col_begin + static_cast<int64_t>(threadIdx.x) * Vec; base < real_end;
         base += Chunk) {
        scalar_t values[Vec];
        if (base + Vec <= real_end) {
            load_vector<scalar_t, Vec>(row_ptr + base, values);
#pragma unroll
            for (int i = 0; i < Vec; ++i) {
                local_max = fmaxf(local_max, static_cast<float>(values[i]));
            }
        } else {
#pragma unroll
            for (int i = 0; i < Vec; ++i) {
                if (base + i < real_end) {
                    local_max = fmaxf(local_max, static_cast<float>(row_ptr[base + i]));
                }
            }
        }
    }
    const float max_value = block_reduce_max<BlockSize>(local_max);
    __shared__ float row_max;
    if (threadIdx.x == 0) {
        row_max = max_value;
    }
    __syncthreads();
    const float tile_max_value = row_max;

    float sum_value = 0.0f;
    if (isfinite(tile_max_value)) {
        float local_sum = 0.0f;
        for (int64_t base = col_begin + static_cast<int64_t>(threadIdx.x) * Vec; base < real_end;
             base += Chunk) {
            scalar_t values[Vec];
            if (base + Vec <= real_end) {
                load_vector<scalar_t, Vec>(row_ptr + base, values);
#pragma unroll
                for (int i = 0; i < Vec; ++i) {
                    local_sum += expf(static_cast<float>(values[i]) - tile_max_value);
                }
            } else {
#pragma unroll
                for (int i = 0; i < Vec; ++i) {
                    if (base + i < real_end) {
                        local_sum += expf(static_cast<float>(row_ptr[base + i]) - tile_max_value);
                    }
                }
            }
        }
        sum_value = block_reduce_sum<BlockSize>(local_sum);
    }
    if (threadIdx.x == 0) {
        const int64_t output_index = row * local_tiles + tile_index;
        tile_max[output_index] = tile_max_value;
        tile_sum[output_index] = sum_value;
    }
}

// grad = g_logp * (onehot - p) + g_lse * p with p = exp(z - lse) on finite rows,
// p = 0 otherwise, and 0 on padding columns. Elementwise, so the result does not
// depend on the launch geometry.
template <typename scalar_t, int BlockSize, int Vec>
__global__ void __launch_bounds__(BlockSize) hip_backward_kernel(
    const scalar_t* __restrict__ logits,
    const float* __restrict__ lse,
    const float* __restrict__ coef_logp,
    const float* __restrict__ coef_lse,
    const int64_t* __restrict__ target_local,
    scalar_t* __restrict__ grad,
    int64_t rows,
    int64_t local_vocab,
    int64_t vocab_start,
    int64_t real_vocab,
    bool has_lse_grad) {
    constexpr int Chunk = BlockSize * Vec;
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows) {
        return;
    }
    const float row_lse = lse[row];
    const bool finite_row = isfinite(row_lse);
    const float lse_safe = finite_row ? row_lse : 0.0f;
    const float g_logp = coef_logp[row];
    const float g_lse = has_lse_grad ? coef_lse[row] : 0.0f;
    const int64_t hit = target_local[row];
    const int64_t real_end = min(local_vocab, max(real_vocab - vocab_start, static_cast<int64_t>(0)));
    const scalar_t* __restrict__ row_in = logits + row * local_vocab;
    scalar_t* __restrict__ row_out = grad + row * local_vocab;
    const int64_t stride = static_cast<int64_t>(gridDim.y) * Chunk;

    for (int64_t base = static_cast<int64_t>(blockIdx.y) * Chunk +
                        static_cast<int64_t>(threadIdx.x) * Vec;
         base < local_vocab; base += stride) {
        scalar_t values[Vec];
        scalar_t outputs[Vec];
        const bool full = base + Vec <= local_vocab;
        if (full) {
            load_vector<scalar_t, Vec>(row_in + base, values);
        } else {
#pragma unroll
            for (int i = 0; i < Vec; ++i) {
                values[i] = (base + i < local_vocab) ? row_in[base + i] : static_cast<scalar_t>(0.0f);
            }
        }
#pragma unroll
        for (int i = 0; i < Vec; ++i) {
            const int64_t col = base + i;
            float value = 0.0f;
            if (col < real_end) {
                const float p = finite_row ? expf(static_cast<float>(values[i]) - lse_safe) : 0.0f;
                const float onehot = (col == hit) ? 1.0f : 0.0f;
                value = g_logp * (onehot - p);
                if (has_lse_grad) {
                    value = value + g_lse * p;
                }
            }
            outputs[i] = static_cast<scalar_t>(value);
        }
        if (full) {
            store_vector<scalar_t, Vec>(row_out + base, outputs);
        } else {
#pragma unroll
            for (int i = 0; i < Vec; ++i) {
                if (base + i < local_vocab) {
                    row_out[base + i] = outputs[i];
                }
            }
        }
    }
}

void check_logits(const torch::Tensor& logits) {
    TORCH_CHECK(logits.is_cuda(), "logits must be a ROCm tensor");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D [tokens, local_vocab]");
    TORCH_CHECK(logits.scalar_type() == at::ScalarType::Half ||
                    logits.scalar_type() == at::ScalarType::BFloat16 ||
                    logits.scalar_type() == at::ScalarType::Float,
                "logits must be float16, bfloat16, or float32");
}

} // namespace

std::vector<torch::Tensor> hip_deterministic_logp_tile_stats(
    torch::Tensor logits,
    int64_t vocab_start,
    int64_t real_vocab,
    int64_t num_tiles) {
    check_logits(logits);
    TORCH_CHECK(vocab_start >= 0 && real_vocab > 0 && num_tiles > 0,
                "invalid vocabulary metadata");
    auto input = logits.contiguous();
    const int64_t rows = input.size(0);
    const int64_t local_vocab = input.size(1);
    TORCH_CHECK(local_vocab > 0 && local_vocab % num_tiles == 0,
                "local_vocab must be divisible by num_tiles");
    TORCH_CHECK(num_tiles <= 65535, "num_tiles must fit the launch grid");
    const int64_t tile_size = local_vocab / num_tiles;
    auto options = input.options().dtype(at::ScalarType::Float);
    auto tile_max = torch::empty({rows, num_tiles}, options);
    auto tile_sum = torch::empty({rows, num_tiles}, options);
    if (rows == 0) {
        return {tile_max, tile_sum};
    }
    const dim3 grid(static_cast<unsigned int>(rows), static_cast<unsigned int>(num_tiles), 1);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
        "hip_deterministic_logp_tile_stats", ([&] {
            hip_tile_stats_kernel<scalar_t, kTileBlockSize, kTileVectorElements>
                <<<grid, kTileBlockSize, 0, stream>>>(
                    input.data_ptr<scalar_t>(), tile_max.data_ptr<float>(),
                    tile_sum.data_ptr<float>(), rows, local_vocab, vocab_start,
                    real_vocab, tile_size, num_tiles);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {tile_max, tile_sum};
}

torch::Tensor hip_deterministic_logp_backward(
    torch::Tensor logits,
    torch::Tensor lse,
    torch::Tensor coef_logp,
    torch::Tensor coef_lse,
    torch::Tensor target_local,
    int64_t vocab_start,
    int64_t real_vocab,
    bool has_lse_grad) {
    check_logits(logits);
    TORCH_CHECK(vocab_start >= 0 && real_vocab > 0, "invalid vocabulary metadata");
    auto input = logits.contiguous();
    const int64_t rows = input.size(0);
    const int64_t local_vocab = input.size(1);
    auto check_row_vector = [&](const torch::Tensor& tensor, at::ScalarType dtype, const char* name) {
        TORCH_CHECK(tensor.is_cuda() && tensor.device() == input.device(), name,
                    " must live on the logits device");
        TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
        TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == rows, name,
                    " must have one entry per token");
        TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    };
    check_row_vector(lse, at::ScalarType::Float, "lse");
    check_row_vector(coef_logp, at::ScalarType::Float, "coef_logp");
    check_row_vector(coef_lse, at::ScalarType::Float, "coef_lse");
    check_row_vector(target_local, at::ScalarType::Long, "target_local");
    auto grad = torch::empty_like(input);
    if (rows == 0 || local_vocab == 0) {
        return grad;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
        "hip_deterministic_logp_backward", ([&] {
            constexpr int Chunk = kBackwardBlockSize * kTileVectorElements;
            const int64_t chunks = (local_vocab + Chunk - 1) / Chunk;
            const dim3 grid(static_cast<unsigned int>(rows),
                            static_cast<unsigned int>(std::min<int64_t>(chunks, 65535)), 1);
            hip_backward_kernel<scalar_t, kBackwardBlockSize, kTileVectorElements>
                <<<grid, kBackwardBlockSize, 0, stream>>>(
                    input.data_ptr<scalar_t>(), lse.data_ptr<float>(),
                    coef_logp.data_ptr<float>(), coef_lse.data_ptr<float>(),
                    target_local.data_ptr<int64_t>(), grad.data_ptr<scalar_t>(), rows,
                    local_vocab, vocab_start, real_vocab, has_lse_grad);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad;
}
