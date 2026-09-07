// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include <musa_runtime.h>
#include <torch/extension.h>
#include <torch_musa/csrc/aten/musa/Exceptions.h>
#include <torch_musa/csrc/aten/musa/MUSAContext.h>

#include <cfloat>

namespace {

constexpr int kBlockSize = 256;

__device__ __forceinline__ float block_reduce_max(float value) {
    __shared__ float partial[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset, 32));
    }
    if (lane == 0) {
        partial[warp] = value;
    }
    __syncthreads();

    value = threadIdx.x < (kBlockSize / 32) ? partial[lane] : -FLT_MAX;
    if (warp == 0) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset, 32));
        }
    }
    if (threadIdx.x == 0) {
        partial[0] = value;
    }
    __syncthreads();
    return partial[0];
}

__device__ __forceinline__ float block_reduce_sum(float value) {
    __shared__ float partial[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset, 32);
    }
    if (lane == 0) {
        partial[warp] = value;
    }
    __syncthreads();

    value = threadIdx.x < (kBlockSize / 32) ? partial[lane] : 0.0f;
    if (warp == 0) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset, 32);
        }
    }
    if (threadIdx.x == 0) {
        partial[0] = value;
    }
    __syncthreads();
    return partial[0];
}

template <typename scalar_t>
__global__ void fused_logp_kernel(
    const scalar_t* __restrict__ logits,
    const int64_t* __restrict__ token_ids,
    scalar_t* __restrict__ output,
    int rows,
    int vocab) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const scalar_t* row_logits = logits + static_cast<size_t>(row) * vocab;
    float row_max = -FLT_MAX;
    for (int col = threadIdx.x; col < vocab; col += blockDim.x) {
        row_max = fmaxf(row_max, static_cast<float>(row_logits[col]));
    }
    row_max = block_reduce_max(row_max);

    float row_sum = 0.0f;
    for (int col = threadIdx.x; col < vocab; col += blockDim.x) {
        row_sum += expf(static_cast<float>(row_logits[col]) - row_max);
    }
    row_sum = block_reduce_sum(row_sum);

    if (threadIdx.x == 0) {
        const int64_t target = token_ids[row];
        const float target_logit = static_cast<float>(row_logits[target]);
        output[row] = static_cast<scalar_t>(target_logit - row_max - logf(row_sum));
    }
}

}  // namespace

torch::Tensor fused_logp_forward_musa(torch::Tensor logits, torch::Tensor token_ids) {
    auto output = torch::empty({logits.size(0)}, logits.options());
    const int rows = static_cast<int>(logits.size(0));
    const int vocab = static_cast<int>(logits.size(1));
    if (rows == 0) {
        return output;
    }
    auto stream = at::musa::getCurrentMUSAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        logits.scalar_type(),
        "musa_fused_logp",
        [&] {
            fused_logp_kernel<scalar_t><<<rows, kBlockSize, 0, stream>>>(
                logits.data_ptr<scalar_t>(),
                token_ids.data_ptr<int64_t>(),
                output.data_ptr<scalar_t>(),
                rows,
                vocab);
        });
    C10_MUSA_KERNEL_LAUNCH_CHECK();
    return output;
}
