# ROCm Logprob 实现说明

日期：2026-08-22

这次工作是在临时分支 `work/ws2-logprob-rocm` 上完成的。它基于 issue #241 的 PR1-PR5 integration，再合入当前 PR4 的最新更新。

## 一句话概括

新增了一个 ROCm 版 WS2 vocab-parallel logprob 后端：每个 GPU 用 HIP/CUDA 可移植 kernel 计算本地 vocab tile 的 FP32 `(max, sumexp)`，TP 之间仍使用 issue #241 已确定的 all-gather + 固定顺序 merge。这样不会为了追求 ROCm 速度而改变原来的数值契约。

## 改了什么

### 1. 增加本地 tile partial kernel

文件：`csrc/deterministic_logp_kernel.cu`

新增 `deterministic_logp_tile_stats`：

- 输入一个 TP rank 的 local vocab logits；
- 每个 vocab tile 输出 FP32 `max` 和 `sumexp`；
- 过滤真实 vocab 之外的 padding 列；
- 使用固定 block reduction，不使用 atomic 或不确定的全局归约；
- 同一份 `.cu` 源码可由 CUDA 或 HIP 编译。

这个 kernel 只负责 rank-local 计算，不负责 RCCL/NCCL，也不负责全局 LSE 合并。跨 TP 的合并顺序仍由已有 WS2 Python 实现控制。

### 2. 增加 ROCm backend 注册

文件：

- `rl_engine/kernels/ops/rocm/loss/vocab_parallel_logp.py`
- `rl_engine/kernels/ops/rocm/loss/__init__.py`
- `rl_engine/kernels/registry.py`

新增 backend：

```text
rocm-vocab-parallel-logp-ws2
```

它只在 ROCm platform 注册，放在 PyTorch reference 前面。backend 继承现有 `VocabParallelLogprobOp`，所以以下逻辑仍然共用一份实现：

- TP contract 和 preflight；
- tile-aligned shard 检查；
- all-gather transport；
- global tile order merge；
- selected-token owner gather；
- entropy 的显式 rank-order merge；
- backward 和 active-mask 语义。

只有在 ROCm native extension 已经加载、且提供新符号时，registry 才会注册这个 production backend。native 不可用时，registry 不会把它显示成可用的 ROCm production 实现，而是明确使用已有的 PyTorch reference backend。这样 backend provenance 是准确的，不会出现“界面显示用了 ROCm，实际却悄悄跑了另一条路径”的情况。

如果调用方显式要求 native ROCm backend，但扩展或符号缺失，会直接抛出清晰的 `RuntimeError`，不会在 wrapper 内部静默 fallback。PyTorch reference 始终使用纯 PyTorch tile 统计；ROCm 子类只有在 native 能力已由 registry 检查通过时才启用 HIP tile kernel。

### 3. 增加 native Python binding 和类型声明

文件：

- `csrc/ops.cpp`
- `rl_engine/_C.pyi`

注册了 `deterministic_logp_tile_stats` 的 Python binding 和类型签名。

同时修复了一个 ROCm 构建隐患：`deterministic_collective_*` 使用 CUDA IPC，但 ROCm 构建不编译对应源文件。现在这些符号的声明和 pybind 注册在 HIP 编译时都会被排除，避免 ROCm 链接阶段出现 unresolved symbol。

Prefix-Shared Attention 的 NVIDIA PTX 注册也改成 HIP 编译时排除，与 PR4 的 source gating 保持一致。

### 4. 增加测试

文件：`tests/test_rocm_logprob_backend.py`

覆盖：

- ROCm backend 继承并保持 WS2 operator surface；
- backend 只在 ROCm registry 中注册；
- native extension 可用时才注册 ROCm production backend；不可用时首选 PyTorch reference；
- native tile kernel 和 binding 存在；
- CPU-only 环境可以导入 ROCm wrapper，不要求 native extension。
- reference/native 两条执行路径不会互相静默切换。

另外补上了和 PR #319/#325 思路一致的验证入口：

- ROCm native 路径复用 TP2/TP4 的 TP1 对照、重复执行、forward/backward 和 bitwise 检查；
- ROCm 多卡 `TP2 x CP2` CLI 测试要求所有 rank 的实际 backend 都是
  `rocm-vocab-parallel-logp-ws2`，且 provenance 不得标记 fallback；
- 显式请求 native backend 但扩展缺失时，测试要求直接 fail fast。

这些测试在非 ROCm 或 GPU 数量不足的环境会跳过；真正执行需要带 ROCm native extension 的多卡机器。

## 没有改什么

- 没有改 issue #241 的 TP/CP 语义；CP 仍不是 logprob 的 merge axis。
- 没有用 RCCL `all_reduce` 取代固定顺序的 all-gather + local merge。
- 没有把 AITER/Composable Kernel 强行接进 strict path。
- 没有把 SM90/TMA/WGMMA 源文件加入 ROCm build。
- 没有修改 Vime provider 的 contract 或 entropy 语义。
- 没有删除任何仓库文件；只是在 ROCm 编译时排除了不适用的 CUDA IPC/PTX 注册。

## 验证结果

在当前 Windows/CUDA 开发环境中：

```text
70 passed, 11 skipped
```

通过的测试包括：

- ROCm backend dispatch 测试；
- issue #241 logprob contract 测试；
- vocab-parallel logprob reference 测试；
- Vime selected-logprob provider 测试。

CUDA extension build 没有完成，原因是当前机器缺少 Microsoft Visual C++ `cl.exe`；同时本地 `nvcc` 是 CUDA 12.6，而 PyTorch 是 CUDA 12.8。当前机器也没有 `hipcc` 和 ROCm runtime，因此以下项目尚未在本地验证：

- gfx942/gfx950 HIP 编译；
- MI300X RCCL TP2/TP4；
- ROCm native tile kernel 与 PyTorch reference 的真实数值/bitwise 对比。

建议在 ROCm 机器上运行：

```bash
PYTORCH_ROCM_ARCH=gfx942 RL_KERNEL_REQUIRE_EXT=1 MAX_JOBS=16 \
  python setup.py build_ext --inplace

PYTHONHASHSEED=0 pytest -q -ra \
  tests/test_rocm_logprob_backend.py \
  tests/test_logprob_contract.py \
  tests/test_vocab_parallel_logp.py \
  tests/test_distributed_logprob_comparison.py
```

真实多 GPU 验证还需要补充 RCCL TP2/TP4 的运行命令和结果；在拿到 gfx942 结果前，不应把这个 backend 宣称为已完成性能优化，只能称为 strict semantics-preserving ROCm implementation。

## 当前提交

实现尚未推送远程；代码位于当前工作分支 `work/ws2-logprob-rocm`。合并基线提交是 `b811abc`，本次实现文件仍在工作区，待 ROCm 环境验证后再拆分成正式 PR commits。
