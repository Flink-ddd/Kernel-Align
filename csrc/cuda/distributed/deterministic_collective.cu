// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <tuple>
#include <vector>

namespace {

constexpr int kMaxDeterministicWorldSize = 8;
constexpr int kThreads = 256;
constexpr int kMaxBlocks = 4096;

struct PeerPointers {
  const void* values[kMaxDeterministicWorldSize];
};

bool is_supported_world_size(int64_t world_size) {
  return world_size == 1 || world_size == 2 || world_size == 4 || world_size == 8;
}

template <typename T>
__device__ __forceinline__ T ordered_add(T lower, T upper);

template <>
__device__ __forceinline__ float ordered_add(float lower, float upper) {
  float result;
  asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(lower), "f"(upper));
  return result;
}

template <>
__device__ __forceinline__ half ordered_add(half lower, half upper) {
  return __hadd(lower, upper);
}

#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
template <>
__device__ __forceinline__ nv_bfloat16 ordered_add(
    nv_bfloat16 lower,
    nv_bfloat16 upper) {
  return __hadd(lower, upper);
}
#endif

template <typename T, int WorldSize>
__device__ __forceinline__ T fixed_tree_reduce(
    const PeerPointers& peers,
    int64_t index) {
  static_assert(
      WorldSize == 1 || WorldSize == 2 || WorldSize == 4 || WorldSize == 8,
      "deterministic collectives only support TP sizes 1, 2, 4, and 8");
  const auto* rank0 = static_cast<const T*>(peers.values[0]);
  if constexpr (WorldSize == 1) {
    return rank0[index];
  } else {
    const auto* rank1 = static_cast<const T*>(peers.values[1]);
    const T sum01 = ordered_add(rank0[index], rank1[index]);
    if constexpr (WorldSize == 2) {
      return sum01;
    } else {
      const auto* rank2 = static_cast<const T*>(peers.values[2]);
      const auto* rank3 = static_cast<const T*>(peers.values[3]);
      const T sum23 = ordered_add(rank2[index], rank3[index]);
      const T sum03 = ordered_add(sum01, sum23);
      if constexpr (WorldSize == 4) {
        return sum03;
      } else {
        const auto* rank4 = static_cast<const T*>(peers.values[4]);
        const auto* rank5 = static_cast<const T*>(peers.values[5]);
        const auto* rank6 = static_cast<const T*>(peers.values[6]);
        const auto* rank7 = static_cast<const T*>(peers.values[7]);
        const T sum45 = ordered_add(rank4[index], rank5[index]);
        const T sum67 = ordered_add(rank6[index], rank7[index]);
        const T sum47 = ordered_add(sum45, sum67);
        return ordered_add(sum03, sum47);
      }
    }
  }
}

template <typename T, int WorldSize>
__global__ void deterministic_all_reduce_kernel(
    PeerPointers peers,
    T* output,
    int64_t element_count) {
  const int64_t thread_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_index; index < element_count; index += stride) {
    output[index] = fixed_tree_reduce<T, WorldSize>(peers, index);
  }
}

template <typename T>
void launch_all_reduce(
    const PeerPointers& peers,
    T* output,
    int64_t element_count,
    int blocks,
    int64_t world_size,
    cudaStream_t stream) {
  switch (world_size) {
    case 1:
      deterministic_all_reduce_kernel<T, 1><<<blocks, kThreads, 0, stream>>>(
          peers, output, element_count);
      break;
    case 2:
      deterministic_all_reduce_kernel<T, 2><<<blocks, kThreads, 0, stream>>>(
          peers, output, element_count);
      break;
    case 4:
      deterministic_all_reduce_kernel<T, 4><<<blocks, kThreads, 0, stream>>>(
          peers, output, element_count);
      break;
    case 8:
      deterministic_all_reduce_kernel<T, 8><<<blocks, kThreads, 0, stream>>>(
          peers, output, element_count);
      break;
    default:
      TORCH_CHECK(false, "unsupported deterministic collective world size ", world_size);
  }
}

template <typename T, int WorldSize>
__global__ void deterministic_reduce_scatter_kernel(
    PeerPointers peers,
    T* output,
    int64_t output_element_count,
    int rank) {
  const int64_t thread_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t input_offset = static_cast<int64_t>(rank) * output_element_count;
  for (int64_t index = thread_index; index < output_element_count; index += stride) {
    output[index] = fixed_tree_reduce<T, WorldSize>(peers, input_offset + index);
  }
}

template <typename T>
void launch_reduce_scatter(
    const PeerPointers& peers,
    T* output,
    int64_t output_element_count,
    int rank,
    int blocks,
    int64_t world_size,
    cudaStream_t stream) {
  switch (world_size) {
    case 1:
      deterministic_reduce_scatter_kernel<T, 1><<<blocks, kThreads, 0, stream>>>(
          peers, output, output_element_count, rank);
      break;
    case 2:
      deterministic_reduce_scatter_kernel<T, 2><<<blocks, kThreads, 0, stream>>>(
          peers, output, output_element_count, rank);
      break;
    case 4:
      deterministic_reduce_scatter_kernel<T, 4><<<blocks, kThreads, 0, stream>>>(
          peers, output, output_element_count, rank);
      break;
    case 8:
      deterministic_reduce_scatter_kernel<T, 8><<<blocks, kThreads, 0, stream>>>(
          peers, output, output_element_count, rank);
      break;
    default:
      TORCH_CHECK(false, "unsupported deterministic collective world size ", world_size);
  }
}

__global__ void deterministic_all_gather_kernel(
    PeerPointers peers,
    uint8_t* output,
    int64_t input_bytes,
    int64_t world_size) {
  const int64_t output_bytes = input_bytes * world_size;
  const int64_t thread_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_index; index < output_bytes; index += stride) {
    const int peer = static_cast<int>(index / input_bytes);
    const int64_t peer_offset = index - static_cast<int64_t>(peer) * input_bytes;
    output[index] = static_cast<const uint8_t*>(peers.values[peer])[peer_offset];
  }
}

class DeterministicCollectiveState {
 public:
  DeterministicCollectiveState(
      torch::Tensor& staging,
      const std::vector<std::vector<int64_t>>& handles,
      const std::vector<int64_t>& offsets,
      int64_t rank)
      : rank_(rank),
        world_size_(handles.size()),
        device_index_(staging.get_device()),
        capacity_bytes_(staging.numel() * staging.element_size()) {
    TORCH_CHECK(staging.is_cuda(), "collective staging buffer must be CUDA");
    TORCH_CHECK(staging.is_contiguous(), "collective staging buffer must be contiguous");
    TORCH_CHECK(
        staging.scalar_type() == torch::kUInt8,
        "collective staging buffer must have dtype torch.uint8");
    TORCH_CHECK(capacity_bytes_ > 0, "collective staging capacity must be positive");
    TORCH_CHECK(
        is_supported_world_size(world_size_),
        "deterministic collectives require world size 1, 2, 4, or 8; got ",
        world_size_);
    TORCH_CHECK(
        offsets.size() == handles.size(),
        "deterministic collectives require one IPC offset per handle");
    TORCH_CHECK(
        rank_ >= 0 && rank_ < world_size_,
        "deterministic collective rank must be in [0, ",
        world_size_,
        ")");

    for (auto& peer : peers_.values) {
      peer = nullptr;
    }
    imported_bases_.fill(nullptr);
    try {
      for (int peer = 0; peer < world_size_; ++peer) {
        TORCH_CHECK(
            handles[peer].size() == sizeof(cudaIpcMemHandle_t),
            "invalid CUDA IPC handle size for rank ",
            peer);
        TORCH_CHECK(offsets[peer] >= 0, "negative CUDA IPC offset for rank ", peer);

        if (peer == rank_) {
          peers_.values[peer] = staging.data_ptr();
          continue;
        }

        cudaIpcMemHandle_t handle{};
        auto* raw_handle = reinterpret_cast<uint8_t*>(&handle);
        for (size_t byte = 0; byte < sizeof(handle); ++byte) {
          TORCH_CHECK(
              handles[peer][byte] >= 0 && handles[peer][byte] <= 255,
              "invalid CUDA IPC handle byte for rank ",
              peer);
          raw_handle[byte] = static_cast<uint8_t>(handles[peer][byte]);
        }

        void* base = nullptr;
        AT_CUDA_CHECK(cudaIpcOpenMemHandle(
            &base,
            handle,
            cudaIpcMemLazyEnablePeerAccess));
        imported_bases_[peer] = base;
        peers_.values[peer] = static_cast<const char*>(base) + offsets[peer];
      }
    } catch (...) {
      close_imports();
      throw;
    }
  }

  ~DeterministicCollectiveState() {
    int previous_device = -1;
    if (cudaGetDevice(&previous_device) == cudaSuccess && previous_device != device_index_) {
      if (cudaSetDevice(device_index_) != cudaSuccess) {
        return;
      }
    }
    close_imports();
    if (previous_device >= 0 && previous_device != device_index_) {
      cudaSetDevice(previous_device);
    }
  }

  void stage(torch::Tensor& input, cudaStream_t stream) {
    check_tensor(input, "input");
    const int64_t input_bytes = input.numel() * input.element_size();
    TORCH_CHECK(
        input_bytes <= capacity_bytes_,
        "input requires ",
        input_bytes,
        " bytes but staging capacity is ",
        capacity_bytes_);
    if (input_bytes > 0) {
      AT_CUDA_CHECK(cudaMemcpyAsync(
          const_cast<void*>(peers_.values[rank_]),
          input.data_ptr(),
          input_bytes,
          cudaMemcpyDeviceToDevice,
          stream));
    }
    staged_bytes_ = input_bytes;
    staged_scalar_type_ = input.scalar_type();
    has_staged_input_ = true;
  }

  void all_reduce(torch::Tensor& output, cudaStream_t stream) const {
    check_tensor(output, "output");
    TORCH_CHECK(has_staged_input_, "stage() must be called before all_reduce()");
    TORCH_CHECK(
        output.scalar_type() == staged_scalar_type_,
        "all-reduce output dtype must match the staged input dtype");
    TORCH_CHECK(
        output.numel() * output.element_size() == staged_bytes_,
        "all-reduce output size must match the staged input size");

    const int64_t element_count = output.numel();
    if (element_count == 0) {
      return;
    }
    const int blocks = static_cast<int>(std::min<int64_t>(
        kMaxBlocks,
        (element_count + kThreads - 1) / kThreads));

    switch (output.scalar_type()) {
      case at::ScalarType::Float:
        launch_all_reduce<float>(
            peers_,
            static_cast<float*>(output.data_ptr()),
            element_count,
            blocks,
            world_size_,
            stream);
        break;
      case at::ScalarType::Half:
        launch_all_reduce<half>(
            peers_,
            static_cast<half*>(output.data_ptr()),
            element_count,
            blocks,
            world_size_,
            stream);
        break;
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
      case at::ScalarType::BFloat16:
        launch_all_reduce<nv_bfloat16>(
            peers_,
            static_cast<nv_bfloat16*>(output.data_ptr()),
            element_count,
            blocks,
            world_size_,
            stream);
        break;
#endif
      default:
        TORCH_CHECK(
            false,
            "deterministic all-reduce supports float32, float16, and bfloat16; got ",
            output.scalar_type());
    }
    AT_CUDA_CHECK(cudaGetLastError());
  }

  void reduce_scatter(torch::Tensor& output, cudaStream_t stream) const {
    check_tensor(output, "output");
    TORCH_CHECK(has_staged_input_, "stage() must be called before reduce_scatter()");
    TORCH_CHECK(
        output.scalar_type() == staged_scalar_type_,
        "reduce-scatter output dtype must match the staged input dtype");
    TORCH_CHECK(
        output.numel() * output.element_size() * world_size_ == staged_bytes_,
        "reduce-scatter output must contain one world-size fraction of the staged input");

    const int64_t output_element_count = output.numel();
    if (output_element_count == 0) {
      return;
    }
    const int blocks = static_cast<int>(std::min<int64_t>(
        kMaxBlocks,
        (output_element_count + kThreads - 1) / kThreads));

    switch (output.scalar_type()) {
      case at::ScalarType::Float:
        launch_reduce_scatter<float>(
            peers_,
            static_cast<float*>(output.data_ptr()),
            output_element_count,
            rank_,
            blocks,
            world_size_,
            stream);
        break;
      case at::ScalarType::Half:
        launch_reduce_scatter<half>(
            peers_,
            static_cast<half*>(output.data_ptr()),
            output_element_count,
            rank_,
            blocks,
            world_size_,
            stream);
        break;
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
      case at::ScalarType::BFloat16:
        launch_reduce_scatter<nv_bfloat16>(
            peers_,
            static_cast<nv_bfloat16*>(output.data_ptr()),
            output_element_count,
            rank_,
            blocks,
            world_size_,
            stream);
        break;
#endif
      default:
        TORCH_CHECK(
            false,
            "deterministic reduce-scatter supports float32, float16, and bfloat16; got ",
            output.scalar_type());
    }
    AT_CUDA_CHECK(cudaGetLastError());
  }

  void all_gather(torch::Tensor& output, cudaStream_t stream) const {
    check_tensor(output, "output");
    TORCH_CHECK(has_staged_input_, "stage() must be called before all_gather()");
    TORCH_CHECK(
        output.scalar_type() == staged_scalar_type_,
        "all-gather output dtype must match the staged input dtype");
    TORCH_CHECK(
        output.numel() * output.element_size() ==
            staged_bytes_ * world_size_,
        "all-gather output must contain one staged input per rank");

    const int64_t output_bytes = output.numel() * output.element_size();
    if (output_bytes == 0) {
      return;
    }
    const int blocks = static_cast<int>(std::min<int64_t>(
        kMaxBlocks,
        (output_bytes + kThreads - 1) / kThreads));
    deterministic_all_gather_kernel<<<blocks, kThreads, 0, stream>>>(
        peers_,
        static_cast<uint8_t*>(output.data_ptr()),
        staged_bytes_,
        world_size_);
    AT_CUDA_CHECK(cudaGetLastError());
  }

 private:
  void check_tensor(const torch::Tensor& tensor, const char* name) const {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(
        tensor.get_device() == device_index_,
        name,
        " must be on cuda:",
        device_index_,
        ", got ",
        tensor.device());
  }

  void close_imports() noexcept {
    for (int peer = 0; peer < world_size_; ++peer) {
      if (imported_bases_[peer] != nullptr) {
        cudaIpcCloseMemHandle(imported_bases_[peer]);
        imported_bases_[peer] = nullptr;
      }
    }
  }

  int64_t rank_;
  int64_t world_size_;
  int device_index_;
  int64_t capacity_bytes_;
  int64_t staged_bytes_{0};
  at::ScalarType staged_scalar_type_{at::ScalarType::Undefined};
  bool has_staged_input_{false};
  PeerPointers peers_{};
  std::array<void*, kMaxDeterministicWorldSize> imported_bases_{};
};

DeterministicCollectiveState* state_from_handle(int64_t handle) {
  TORCH_CHECK(handle != 0, "deterministic collective handle is closed");
  return reinterpret_cast<DeterministicCollectiveState*>(handle);
}

}  // namespace

std::tuple<std::vector<int64_t>, int64_t> deterministic_collective_ipc_meta(
    torch::Tensor& tensor) {
  const c10::cuda::CUDAGuard device_guard(tensor.device());
  TORCH_CHECK(tensor.is_cuda(), "IPC tensor must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), "IPC tensor must be contiguous");
  TORCH_CHECK(tensor.numel() > 0, "cannot export an empty CUDA allocation");

  CUdeviceptr allocation_base = 0;
  size_t allocation_size = 0;
  const auto pointer = reinterpret_cast<CUdeviceptr>(tensor.data_ptr());
  TORCH_CHECK(
      cuPointerGetAttribute(
          &allocation_base,
          CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
          pointer) == CUDA_SUCCESS,
      "failed to query CUDA allocation base");
  TORCH_CHECK(
      cuPointerGetAttribute(
          &allocation_size,
          CU_POINTER_ATTRIBUTE_RANGE_SIZE,
          pointer) == CUDA_SUCCESS,
      "failed to query CUDA allocation size");

  const int64_t offset = static_cast<int64_t>(pointer - allocation_base);
  const int64_t tensor_bytes = tensor.numel() * tensor.element_size();
  TORCH_CHECK(offset >= 0, "invalid negative CUDA allocation offset");
  TORCH_CHECK(
      static_cast<size_t>(offset + tensor_bytes) <= allocation_size,
      "IPC tensor exceeds its CUDA allocation");

  cudaIpcMemHandle_t handle{};
  AT_CUDA_CHECK(cudaIpcGetMemHandle(
      &handle,
      reinterpret_cast<void*>(allocation_base)));
  const auto* raw_handle = reinterpret_cast<const uint8_t*>(&handle);
  std::vector<int64_t> bytes(sizeof(handle));
  for (size_t byte = 0; byte < sizeof(handle); ++byte) {
    bytes[byte] = raw_handle[byte];
  }
  return std::make_tuple(bytes, offset);
}

int64_t deterministic_collective_create(
    torch::Tensor& staging,
    const std::vector<std::vector<int64_t>>& handles,
    const std::vector<int64_t>& offsets,
    int64_t rank) {
  const c10::cuda::CUDAGuard device_guard(staging.device());
  auto state = std::make_unique<DeterministicCollectiveState>(
      staging,
      handles,
      offsets,
      rank);
  return reinterpret_cast<int64_t>(state.release());
}

void deterministic_collective_destroy(int64_t handle) {
  delete state_from_handle(handle);
}

void deterministic_collective_stage(int64_t handle, torch::Tensor& input) {
  const c10::cuda::CUDAGuard device_guard(input.device());
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  state_from_handle(handle)->stage(input, stream);
}

void deterministic_collective_all_reduce(int64_t handle, torch::Tensor& output) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  state_from_handle(handle)->all_reduce(output, stream);
}

void deterministic_collective_reduce_scatter(int64_t handle, torch::Tensor& output) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  state_from_handle(handle)->reduce_scatter(output, stream);
}

void deterministic_collective_all_gather(int64_t handle, torch::Tensor& output) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  state_from_handle(handle)->all_gather(output, stream);
}
