// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

// NVTX range helpers for the pybind operator bindings.
//
// NVTX v3 is header-only and ships with CUDA >= 10, so no linker change is
// needed. ROCm builds and builds with KERNEL_ALIGN_DISABLE_NVTX take the
// identity pass-through branch, which generates no code.

#pragma once

#include <cstdlib>
#include <cstring>
#include <utility>

#if defined(KERNEL_ALIGN_WITH_CUDA) && !defined(KERNEL_ALIGN_DISABLE_NVTX)
#include <nvtx3/nvToolsExt.h>

namespace rlk {

inline bool nvtx_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("RL_KERNEL_NVTX");
    return value != nullptr &&
           (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
            std::strcmp(value, "yes") == 0 || std::strcmp(value, "on") == 0);
  }();
  return enabled;
}

class NvtxRange {
 public:
  explicit NvtxRange(const char* name) : active_(nvtx_enabled()) {
    if (active_) {
      nvtxRangePushA(name);
    }
  }
  ~NvtxRange() {
    if (active_) {
      nvtxRangePop();
    }
  }
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;

 private:
  bool active_;
};

// Wraps a free function so the pybind-bound call is enclosed in an NVTX range.
template <typename Ret, typename... Args>
auto nvtx_wrap(const char* name, Ret (*fn)(Args...)) {
  return [name, fn](Args... args) -> Ret {
    NvtxRange guard{name};
    return fn(std::forward<Args>(args)...);
  };
}

}  // namespace rlk

#define RLK_NVTX_RANGE(name) ::rlk::NvtxRange _rlk_nvtx_guard{(name)}

#else  // ROCm / NVTX-disabled builds: identity pass-through, zero code generated.

namespace rlk {

template <typename F>
auto nvtx_wrap(const char*, F fn) {
  return fn;
}

}  // namespace rlk

#define RLK_NVTX_RANGE(name)

#endif
