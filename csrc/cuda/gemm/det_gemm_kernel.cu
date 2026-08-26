// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
// csrc/cuda/gemm/det_gemm_kernel.cu
//
// WS1 Batch-invariant deterministic GEMM (hand-written, no CUTLASS).
//
//   SM90 path : TMA load + mma.sync (m16n8k16), FP32 accum, single-CTA-per-tile,
//               fixed K order, NO split-K -> batch-invariant.
//   Fallback  : naive FP32 scalar kernel (also the correctness ground truth).
//
// Both: BF16 in / FP32 accum / BF16 store / no TF32 / no split-K.
// K is reduced with a mid-split tree. A contiguous half-K GEMM is one child,
// so simulated TP=2 (a+b) matches TP=1. TP=8 left-fold does not.
// Leaves stay FP32 (naive: 32-wide MAC; SM90: one BK).
//   fwd: C = A @ B   |   dA = dC @ B^T   |   dB = A^T @ dC
// Backward reuses the forward kernel on transposed operands.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>

#if defined(RL_KERNEL_ENABLE_SM90)
#include "det_gemm_tma.cuh"
#endif

namespace {

using nv_bf16 = __nv_bfloat16;

template <typename output_t>
__device__ __forceinline__ output_t cast_output(float value);

template <>
__device__ __forceinline__ nv_bf16 cast_output<nv_bf16>(float value) {
  return __float2bfloat16(value);
}

template <>
__device__ __forceinline__ float cast_output<float>(float value) {
  return value;
}

__host__ __device__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

// Must match SM90 BK so an aligned-K naive tree equals the SM90 tile tree.
constexpr int K_TREE_LEAF = 32;

__device__ __forceinline__ nv_bf16 bf16_add(nv_bf16 a, nv_bf16 b) {
  return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
}

// True iff [lo, hi) is a node of the mid-split tree over [0, n).
__device__ __forceinline__ bool is_mid_split_node(int lo, int hi, int n) {
  int a = 0, b = n;
  while (b - a > 1) {
    if (a == lo && b == hi) return true;
    const int m = a + (b - a) / 2;
    if (hi <= m)
      b = m;
    else if (lo >= m)
      a = m;
    else
      return false;
  }
  return a == lo && b == hi;
}

__device__ nv_bf16 k_tree_naive(const nv_bf16* __restrict__ A, const nv_bf16* __restrict__ B,
                                int row, int col, int N, int K, int lo, int hi) {
  if (hi - lo <= K_TREE_LEAF) {
    float acc = 0.0f;
    for (int k = lo; k < hi; ++k)
      acc += __bfloat162float(A[row * K + k]) * __bfloat162float(B[k * N + col]);
    return __float2bfloat16(acc);
  }
  const int mid = lo + (hi - lo) / 2;
  return bf16_add(k_tree_naive(A, B, row, col, N, K, lo, mid),
                  k_tree_naive(A, B, row, col, N, K, mid, hi));
}

// Naive FP32 scalar kernel (fallback + ground truth). Batch-invariant by
// construction: one thread = one output element, mid-split K tree.
constexpr int NAIVE_TILE = 16;

template <typename output_t>
__global__ void det_gemm_naive(const nv_bf16* __restrict__ A,
                               const nv_bf16* __restrict__ B,
                               output_t* __restrict__ C,
                               int M, int N, int K) {
  const int row = blockIdx.y * NAIVE_TILE + threadIdx.y;
  const int col = blockIdx.x * NAIVE_TILE + threadIdx.x;
  if (row >= M || col >= N) return;
  C[row * N + col] = cast_output<output_t>(
      __bfloat162float(k_tree_naive(A, B, row, col, N, K, 0, K)));
}

template <typename output_t>
void launch_naive(const nv_bf16* A, const nv_bf16* B, output_t* C,
                  int M, int N, int K, cudaStream_t stream) {
  dim3 block(NAIVE_TILE, NAIVE_TILE);
  dim3 grid(cdiv(N, NAIVE_TILE), cdiv(M, NAIVE_TILE));
  det_gemm_naive<output_t><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

#if defined(RL_KERNEL_ENABLE_SM90)
// SM90 path: TMA load + mma.sync. C[M,N] = A[M,K] @ B[K,N].
// Each CTA owns one [BM,BN] output tile, walks full K in fixed order (no
// split-K). A tile [BM,BK] row-major; B operand col-major [n,k] supplied by
// passing B^T ([N,K] row-major) so the B smem tile is [BN,BK] (row=n,col=k),
// matching the validated logp ldmatrix addressing.
constexpr int BM = 128, DECODE_BM = 16, BN = 64, BK = 32;
static_assert(BK == K_TREE_LEAF, "SM90 tile width must match the naive K-tree leaf");
constexpr int WARPS = 4;
constexpr int STAGES = 2;

constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
constexpr int N_TILES = BN / MMA_N;       // 8
constexpr int K_TILES = BK / MMA_K;       // 2
constexpr int KK_GROUPS = BK / 32;        // 1
constexpr int TREE_DEPTH = 16;

__device__ __forceinline__ void ldmatrix_x4(uint32_t regs[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
               : "r"(addr));
}
__device__ __forceinline__ void mma_m16n8k16(const uint32_t A[4], const uint32_t B[2], float D[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
               : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
               : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
                 "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));
}

template <typename output_t, int TILE_M, int NUM_WARPS>
__global__ void det_gemm_sm90_kernel(const __grid_constant__ CUtensorMap a_tmap,
                                     const __grid_constant__ CUtensorMap bt_tmap,
                                     output_t* __restrict__ C,
                                     int M, int N, int K) {
  static_assert(TILE_M % NUM_WARPS == 0);
  static_assert((TILE_M / NUM_WARPS) % MMA_M == 0);
  constexpr int WARP_M = TILE_M / NUM_WARPS;
  constexpr int M_TILES = WARP_M / MMA_M;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int row_base = blockIdx.y * TILE_M;
  const int col_base = blockIdx.x * BN;
  const int kd = K / BK;

  extern __shared__ __align__(1024) char smem[];
  nv_bf16* sA = reinterpret_cast<nv_bf16*>(smem);
  nv_bf16* sB = reinterpret_cast<nv_bf16*>(sA + STAGES * TILE_M * BK);
  int* mbar_base = reinterpret_cast<int*>(sB + STAGES * BN * BK);

  const uint32_t sA_base = static_cast<uint32_t>(__cvta_generic_to_shared(sA));
  const uint32_t sB_base = static_cast<uint32_t>(__cvta_generic_to_shared(sB));
  uint32_t mbar[STAGES];
#pragma unroll
  for (int s = 0; s < STAGES; ++s)
    mbar[s] = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_base + 2 * s));

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) det_gemm::mbar_init(mbar[s], 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();

  const uint32_t tile_bytes = (TILE_M * BK + BN * BK) * sizeof(nv_bf16);

  auto issue_load = [&](int k) {
    const int buf = k % STAGES;
    const int koff = k * BK;
    det_gemm::tma_2d_g2s(sA_base + buf * TILE_M * BK * sizeof(nv_bf16), &a_tmap, koff, row_base, mbar[buf]);
    det_gemm::tma_2d_g2s(sB_base + buf * BN * BK * sizeof(nv_bf16), &bt_tmap, koff, col_base, mbar[buf]);
    det_gemm::mbar_arrive_expect_tx(mbar[buf], tile_bytes);
  };

  int phase[STAGES];
#pragma unroll
  for (int s = 0; s < STAGES; ++s) phase[s] = 0;

  float tile_acc[M_TILES][N_TILES][4];
  nv_bf16 tree_v[M_TILES][N_TILES][4];
  nv_bf16 tree_stk[TREE_DEPTH][M_TILES][N_TILES][4];
  int tree_lo[TREE_DEPTH], tree_hi[TREE_DEPTH];
  int sp = 0;

  if (tid == 0)
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s)
      if (s < kd) issue_load(s);

  for (int k = 0; k < kd; ++k) {       // fixed ascending tile order, NO split-K
    const int buf = k % STAGES;
    if (tid == 0 && k + (STAGES - 1) < kd) issue_load(k + (STAGES - 1));
    det_gemm::mbar_wait(mbar[buf], phase[buf]);
    phase[buf] ^= 1;
    __syncthreads();

    const uint32_t sA_buf = sA_base + buf * TILE_M * BK * sizeof(nv_bf16);
    const uint32_t sB_buf = sB_base + buf * BN * BK * sizeof(nv_bf16);

#pragma unroll
    for (int mi = 0; mi < M_TILES; ++mi)
#pragma unroll
      for (int n = 0; n < N_TILES; ++n)
        tile_acc[mi][n][0] = tile_acc[mi][n][1] = tile_acc[mi][n][2] = tile_acc[mi][n][3] = 0.0f;

    uint32_t A[M_TILES][K_TILES][4];
#pragma unroll
    for (int mi = 0; mi < M_TILES; ++mi) {
      const int row0 = warp * WARP_M + mi * MMA_M + (lane % 16);
#pragma unroll
      for (int kt = 0; kt < K_TILES; ++kt) {
        const uint32_t a_addr =
            sA_buf + (row0 * BK + (lane / 16) * 8 + kt * MMA_K) * sizeof(nv_bf16);
        ldmatrix_x4(A[mi][kt], a_addr);
      }
    }

#pragma unroll
    for (int n = 0; n < N_TILES; ++n) {
#pragma unroll
      for (int kk = 0; kk < KK_GROUPS; ++kk) {
        uint32_t b4[4];
        const uint32_t b_addr =
            sB_buf + ((n * MMA_N + (lane % 8)) * BK + (lane / 8) * 8 + kk * 32) * sizeof(nv_bf16);
        ldmatrix_x4(b4, b_addr);
        const uint32_t B0[2] = {b4[0], b4[1]};
        const uint32_t B1[2] = {b4[2], b4[3]};
#pragma unroll
        for (int mi = 0; mi < M_TILES; ++mi) {
          mma_m16n8k16(A[mi][2 * kk + 0], B0, tile_acc[mi][n]);
          mma_m16n8k16(A[mi][2 * kk + 1], B1, tile_acc[mi][n]);
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int mi = 0; mi < M_TILES; ++mi)
#pragma unroll
      for (int n = 0; n < N_TILES; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) tree_v[mi][n][i] = __float2bfloat16(tile_acc[mi][n][i]);

    int lo = k, hi = k + 1;
    while (sp > 0 && tree_hi[sp - 1] == lo && is_mid_split_node(tree_lo[sp - 1], hi, kd)) {
#pragma unroll
      for (int mi = 0; mi < M_TILES; ++mi)
#pragma unroll
        for (int n = 0; n < N_TILES; ++n)
#pragma unroll
          for (int i = 0; i < 4; ++i)
            tree_v[mi][n][i] = bf16_add(tree_stk[sp - 1][mi][n][i], tree_v[mi][n][i]);
      lo = tree_lo[sp - 1];
      --sp;
    }
    if (hi < kd) {
#pragma unroll
      for (int mi = 0; mi < M_TILES; ++mi)
#pragma unroll
        for (int n = 0; n < N_TILES; ++n)
#pragma unroll
          for (int i = 0; i < 4; ++i) tree_stk[sp][mi][n][i] = tree_v[mi][n][i];
      tree_lo[sp] = lo;
      tree_hi[sp] = hi;
      ++sp;
    }
  }

#pragma unroll
  for (int mi = 0; mi < M_TILES; ++mi) {
    const int row = row_base + warp * WARP_M + mi * MMA_M + lane / 4;
#pragma unroll
    for (int n = 0; n < N_TILES; ++n) {
      const int col = col_base + n * MMA_N + (lane % 4) * 2;
      if (row < M && col + 1 < N) {
        C[row * N + col + 0] = cast_output<output_t>(__bfloat162float(tree_v[mi][n][0]));
        C[row * N + col + 1] = cast_output<output_t>(__bfloat162float(tree_v[mi][n][1]));
      }
      if (row + 8 < M && col + 1 < N) {
        C[(row + 8) * N + col + 0] = cast_output<output_t>(__bfloat162float(tree_v[mi][n][2]));
        C[(row + 8) * N + col + 1] = cast_output<output_t>(__bfloat162float(tree_v[mi][n][3]));
      }
    }
  }
}

template <typename output_t>
bool launch_sm90(const nv_bf16* A, const nv_bf16* Bt, output_t* C,
                 int M, int N, int K, cudaStream_t stream) {
  if (N % BN != 0 || K % BK != 0) return false;  // fall back

  const bool decode_tile = M <= DECODE_BM;
  const int tile_m = decode_tile ? DECODE_BM : BM;
  const int num_warps = decode_tile ? 1 : WARPS;
  CUtensorMap a_tmap, bt_tmap;
  det_gemm::init_tmap_noswizzle(&a_tmap, A, M, K, tile_m, BK);
  det_gemm::init_tmap_noswizzle(&bt_tmap, Bt, N, K, BN, BK);

  const int smem = STAGES * (tile_m * BK + BN * BK) * sizeof(nv_bf16) + STAGES * 8;
  if (smem > 48 * 1024) {
    static std::once_flag configure_large_smem;
    std::call_once(configure_large_smem, [smem]() {
      const cudaError_t status = cudaFuncSetAttribute(
          det_gemm_sm90_kernel<output_t, BM, WARPS>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
      TORCH_CHECK(status == cudaSuccess,
                  "det_gemm: cudaFuncSetAttribute failed: ",
                  cudaGetErrorString(status));
    });
  }

  dim3 grid(cdiv(N, BN), cdiv(M, tile_m));
  if (decode_tile) {
    det_gemm_sm90_kernel<output_t, DECODE_BM, 1>
        <<<grid, 32, smem, stream>>>(a_tmap, bt_tmap, C, M, N, K);
  } else {
    det_gemm_sm90_kernel<output_t, BM, WARPS>
        <<<grid, num_warps * 32, smem, stream>>>(a_tmap, bt_tmap, C, M, N, K);
  }
  return true;
}
#endif  // RL_KERNEL_ENABLE_SM90

int sm_major() {
  constexpr int kMaxCachedDevices = 64;
  static std::array<std::atomic<int>, kMaxCachedDevices> cached{};
  int dev = 0;
  const cudaError_t device_status = cudaGetDevice(&dev);
  TORCH_CHECK(device_status == cudaSuccess,
              "det_gemm: cudaGetDevice failed: ",
              cudaGetErrorString(device_status));
  if (dev >= 0 && dev < kMaxCachedDevices) {
    const int value = cached[dev].load(std::memory_order_acquire);
    if (value != 0) return value;
  }
  cudaDeviceProp p{};
  const cudaError_t properties_status = cudaGetDeviceProperties(&p, dev);
  TORCH_CHECK(properties_status == cudaSuccess,
              "det_gemm: cudaGetDeviceProperties failed: ",
              cudaGetErrorString(properties_status));
  if (dev >= 0 && dev < kMaxCachedDevices)
    cached[dev].store(p.major, std::memory_order_release);
  return p.major;
}
inline const nv_bf16* bf16(const torch::Tensor& t) {
  return reinterpret_cast<const nv_bf16*>(t.data_ptr<at::BFloat16>());
}
inline nv_bf16* bf16o(torch::Tensor& t) {
  return reinterpret_cast<nv_bf16*>(t.data_ptr<at::BFloat16>());
}
bool require_sm90() {
  const char* value = std::getenv("RL_KERNEL_DET_GEMM_SM90_ONLY");
  return value != nullptr &&
      (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
       std::strcmp(value, "yes") == 0 || std::strcmp(value, "on") == 0);
}
void check_in(const torch::Tensor& t, const char* n) {
  TORCH_CHECK(t.is_cuda(), n, " must be CUDA");
  TORCH_CHECK(t.scalar_type() == torch::kBFloat16, n, " must be bf16");
}

torch::Tensor gemm_dispatch_bt(const torch::Tensor& a, const torch::Tensor& bt,
                               bool output_fp32 = false) {
  const int M = a.size(0), K = a.size(1), N = bt.size(0);
  auto options = a.options().dtype(output_fp32 ? torch::kFloat32 : torch::kBFloat16);
  auto c = torch::empty({M, N}, options);
  if (M == 0 || N == 0) return c;
  auto stream = at::cuda::getCurrentCUDAStream();

#if defined(RL_KERNEL_ENABLE_SM90)
  // TMA zero-fills out-of-bounds rows in the final M tile. Valid rows therefore
  // use the same tensor-core instructions and K tree for every batch size without
  // materializing or computing padded rows.
  if (sm_major() >= 9 && K % BK == 0 && N % BN == 0) {
    torch::Tensor a_use = a;
    torch::Tensor bt_use = bt;
    if (reinterpret_cast<std::uintptr_t>(a_use.data_ptr()) % 16 != 0)
      a_use = a.clone();
    if (reinterpret_cast<std::uintptr_t>(bt_use.data_ptr()) % 16 != 0)
      bt_use = bt.clone();
    const bool launched = output_fp32
        ? launch_sm90<float>(
              bf16(a_use), bf16(bt_use), c.data_ptr<float>(), M, N, K, stream)
        : launch_sm90<nv_bf16>(
              bf16(a_use), bf16(bt_use), bf16o(c), M, N, K, stream);
    if (launched) return c;
  }
#endif
  TORCH_CHECK(
      !require_sm90(),
      "RL_KERNEL_DET_GEMM_SM90_ONLY=1 forbids det_gemm_naive; SM90 dispatch "
      "was unavailable for M=", M, ", N=", N, ", K=", K,
      ". Rebuild with KERNEL_ALIGN_DET_GEMM_SM90=1 and use SM90-aligned N/K.");
  auto b = bt.t().contiguous();
  if (output_fp32)
    launch_naive<float>(bf16(a), bf16(b), c.data_ptr<float>(), M, N, K, stream);
  else
    launch_naive<nv_bf16>(bf16(a), bf16(b), bf16o(c), M, N, K, stream);
  return c;
}

torch::Tensor gemm_dispatch(const torch::Tensor& a, const torch::Tensor& b,
                            bool output_fp32 = false) {
  return gemm_dispatch_bt(a, b.t().contiguous(), output_fp32);
}

}  // anonymous namespace

bool det_gemm_sm90_compiled() {
#if defined(RL_KERNEL_ENABLE_SM90)
  return true;
#else
  return false;
#endif
}

torch::Tensor det_gemm_fwd(torch::Tensor a, torch::Tensor b) {
  check_in(a, "A"); check_in(b, "B");
  a = a.contiguous(); b = b.contiguous();
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "det_gemm_fwd: expect 2D [M,K]@[K,N]");
  TORCH_CHECK(b.size(0) == a.size(1), "det_gemm_fwd: K mismatch");
  return gemm_dispatch(a, b);
}

torch::Tensor det_gemm_fwd_fp32(torch::Tensor a, torch::Tensor b) {
  check_in(a, "A"); check_in(b, "B");
  a = a.contiguous(); b = b.contiguous();
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "det_gemm_fwd_fp32: expect 2D [M,K]@[K,N]");
  TORCH_CHECK(b.size(0) == a.size(1), "det_gemm_fwd_fp32: K mismatch");
  // Keep the FP32 running sum; only the final store is BF16.
  return gemm_dispatch(a, b, true);
}

torch::Tensor det_gemm_fwd_weight(torch::Tensor a, torch::Tensor weight) {
  check_in(a, "A"); check_in(weight, "weight");
  a = a.contiguous(); weight = weight.contiguous();
  TORCH_CHECK(a.dim() == 2 && weight.dim() == 2,
              "det_gemm_fwd_weight: expect A[M,K], weight[N,K]");
  TORCH_CHECK(weight.size(1) == a.size(1), "det_gemm_fwd_weight: K mismatch");
  return gemm_dispatch_bt(a, weight);
}

torch::Tensor det_gemm_fwd_rhs_transposed(torch::Tensor a, torch::Tensor bt) {
  return det_gemm_fwd_weight(a, bt);
}

torch::Tensor det_gemm_da_weight(torch::Tensor dc, torch::Tensor weight) {
  check_in(dc, "dC"); check_in(weight, "weight");
  dc = dc.contiguous(); weight = weight.contiguous();
  TORCH_CHECK(dc.dim() == 2 && weight.dim() == 2,
              "det_gemm_da_weight: expect dC[M,N], weight[N,K]");
  TORCH_CHECK(dc.size(1) == weight.size(0), "det_gemm_da_weight: N mismatch");
  return gemm_dispatch(dc, weight);
}

torch::Tensor det_gemm_dw(torch::Tensor a, torch::Tensor dc) {
  check_in(a, "A"); check_in(dc, "dC");
  a = a.contiguous(); dc = dc.contiguous();
  TORCH_CHECK(a.dim() == 2 && dc.dim() == 2,
              "det_gemm_dw: expect A[M,K], dC[M,N]");
  TORCH_CHECK(a.size(0) == dc.size(0), "det_gemm_dw: M mismatch");
  return gemm_dispatch(dc.t().contiguous(), a);
}

torch::Tensor det_gemm_db_transposed(torch::Tensor a, torch::Tensor dc) {
  return det_gemm_dw(a, dc);
}

torch::Tensor det_gemm_da(torch::Tensor dc, torch::Tensor b) {
  check_in(dc, "dC"); check_in(b, "B");
  dc = dc.contiguous();
  auto bt = b.t().contiguous();
  TORCH_CHECK(bt.size(0) == dc.size(1), "det_gemm_da: N mismatch");
  return gemm_dispatch(dc, bt);
}

torch::Tensor det_gemm_db(torch::Tensor a, torch::Tensor dc) {
  check_in(a, "A"); check_in(dc, "dC");
  dc = dc.contiguous();
  auto at = a.t().contiguous();
  TORCH_CHECK(dc.size(0) == at.size(1), "det_gemm_db: M mismatch");
  return gemm_dispatch(at, dc);
}
