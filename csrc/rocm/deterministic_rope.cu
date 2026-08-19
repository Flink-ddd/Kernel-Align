// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

template <typename scalar_t>
__global__ void deterministic_rope_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    scalar_t* __restrict__ out,
    int64_t n_rows,
    int table_rows,
    int half,
    float sin_sign) {
  const int64_t index = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  const int64_t count = n_rows * static_cast<int64_t>(half);
  if (index >= count) {
    return;
  }

  const int64_t row = index / half;
  const int pair = static_cast<int>(index % half);
  const int table_row = static_cast<int>(row % table_rows);
  const float c = cos[table_row * half + pair];
  const float s = sin[table_row * half + pair] * sin_sign;
  const int64_t base = row * (2LL * half);
  const float low = static_cast<float>(x[base + pair]);
  const float high = static_cast<float>(x[base + pair + half]);

  out[base + pair] = static_cast<scalar_t>(low * c - high * s);
  out[base + pair + half] = static_cast<scalar_t>(high * c + low * s);
}

}  // namespace

torch::Tensor deterministic_rope_apply_rocm(
    torch::Tensor x,
    torch::Tensor cos,
    torch::Tensor sin,
    double sin_sign) {
  TORCH_CHECK(x.is_cuda(), "ROCm RoPE: x must be a GPU tensor");
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous(),
              "ROCm RoPE: x must be contiguous [rows, head_dim]");
  TORCH_CHECK(cos.is_cuda() && sin.is_cuda(),
              "ROCm RoPE: cos and sin must be GPU tensors");
  TORCH_CHECK(cos.scalar_type() == torch::kFloat32 &&
                  sin.scalar_type() == torch::kFloat32,
              "ROCm RoPE: cos and sin must be FP32");
  TORCH_CHECK(cos.is_contiguous() && sin.is_contiguous(),
              "ROCm RoPE: cos and sin must be contiguous");
  TORCH_CHECK(cos.dim() == 2 && sin.sizes() == cos.sizes(),
              "ROCm RoPE: cos and sin must have shape [table_rows, head_dim/2]");
  TORCH_CHECK(x.size(1) % 2 == 0, "ROCm RoPE: head_dim must be even");
  TORCH_CHECK(cos.size(0) > 0 && cos.size(1) == x.size(1) / 2,
              "ROCm RoPE: invalid cos/sin table shape");
  TORCH_CHECK(x.size(0) % cos.size(0) == 0,
              "ROCm RoPE: row count must be divisible by the position table size");

  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto out = torch::empty_like(x);
  const int64_t n_rows = x.size(0);
  const int half = static_cast<int>(x.size(1) / 2);
  const int table_rows = static_cast<int>(cos.size(0));
  const int64_t count = n_rows * static_cast<int64_t>(half);
  constexpr int threads = 256;
  const int64_t blocks = (count + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x.scalar_type(),
      "deterministic_rope_apply_rocm",
      [&] {
        deterministic_rope_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            cos.data_ptr<float>(),
            sin.data_ptr<float>(),
            out.data_ptr<scalar_t>(),
            n_rows,
            table_rows,
            half,
            static_cast<float>(sin_sign));
      });
  return out;
}
