<p align="center">
  <img src="docs/assets/logo.png" width="220" alt="RL-Kernel logo">
</p>

<h1 align="center">RL-Kernel</h1>

<p align="center">
  <strong>Building cross-hardware and multi-model RL post-training infrastructure for kernel-level train–inference consistency.</strong>
</p>

<p align="center">
  <a href="https://rl-align.github.io/RL-Kernel/"><img src="https://img.shields.io/badge/Documentation-Docs-2ea44f" alt="Documentation"></a>
  <a href="https://rl-align.slack.com/join/shared_invite/zt-46bxj7uyt-gEK3xzwSJr_lppJsZolR~g#/shared-invite/email"><img src="https://img.shields.io/badge/Slack-Join%20Us-4A154B" alt="Slack"></a>
  <a href="https://www.linkedin.com/company/rl-align"><img src="https://img.shields.io/badge/LinkedIn-Follow-0A66C2?logo=linkedin&logoColor=white" alt="Follow RL-Align on LinkedIn"></a>
  <a href="./docs/community/wechat.md"><img src="https://img.shields.io/badge/WeChat-Join%20Group-07C160?logo=wechat&logoColor=white" alt="WeChat"></a>
  <a href="./docs/assets/whatsapp-group.png"><img src="https://img.shields.io/badge/WhatsApp-Join%20Group-25D366?logo=whatsapp&logoColor=white" alt="WhatsApp"></a>
  <a href="https://deepwiki.com/RL-Align/RL-Kernel"><img src="https://img.shields.io/badge/Ask-DeepWiki-7B3FE4" alt="Ask DeepWiki"></a>
  <a href="#hardware-support"><img src="https://img.shields.io/badge/Supported-CUDA%20%7C%20ROCm-2ea44f" alt="CUDA and ROCm supported"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache 2.0 license"></a>
</p>

<p align="center">
  <a href="#architecture">Architecture</a> ·
  <a href="#current-scope-and-roadmap">Current scope</a> ·
  <a href="#benchmark-highlights">Results</a> ·
  <a href="#hardware-support">Hardware support</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="https://rl-align.github.io/RL-Kernel/">Documentation</a>
</p>

**RL-Kernel** is high-performance **RL post-training infrastructure** designed for
bitwise, operator-level train–inference consistency across heterogeneous rollout and
training engines. It combines deterministic operators, hardware-aware runtime dispatch,
and accelerator-specific kernels to improve execution speed and memory efficiency for
GRPO, PPO, and related RL workloads.

The project is building toward cross-hardware and multi-model coverage. Today, the
validated end-to-end path is **Qwen3-8B Dense with VIME, vLLM, and Megatron-LM**;
DeepSeek-V4 Flash MoE and additional RL orchestration integrations are under active
development or on the roadmap.

## Why RL-Kernel?

Rollout and training engines can produce different log probabilities for the same tokens
and model weights because their kernels, batching, and reduction orders differ. Those
differences enter the policy ratios and KL terms used by RL algorithms.

- **Exact train–inference consistency:** deterministic operator contracts align the strict
  dense-model path. The published experiment verifies exact runtime LogP agreement across
  all 200 training steps.
- **RL-native operator stack:** deterministic attention and dense FFN, fused and
  batch-invariant LogP, GRPO/PPO objectives, and deterministic collectives cover the
  numerical boundaries that matter to RL post-training.
- **Efficient RL execution:** fused log-probability computation, optimized attention and FFN
  paths, and deterministic collectives target rollout, memory, and synchronization costs.
- **Current integration:** VIME is the supported RL orchestration layer, with vLLM for
  rollout and Megatron-LM for training. RL-Kernel supplies the operators beneath them.
- **Current accelerator targets:** NVIDIA SM90 (H100, H200, and GH200) and AMD gfx942
  (MI300A, MI300X, and MI325X) are the supported targets. Ascend dav_c220 has partial
  operator coverage; other hardware models are being adapted.

## Architecture

RL-Kernel sits between framework execution engines and accelerator backends. Runtime
adapters select operators through a hardware-aware registry; strict routes enforce the
required numerical contract and expose execution provenance.

The complete architecture below is a layer map, not a current support matrix. It shows
the broader external scheduling and engine ecosystem, operator-library layers, and
hardware abstraction boundary.

<p align="center">
  <img src="docs/assets/RL-Kernel underlying operator library technical architecture.png" alt="RL-Kernel global architecture" width="800">
</p>

The following diagram is a concise view of the validated runtime path, planned
orchestration integrations, and accelerator coverage. Solid arrows represent the current
VIME path; dashed arrows from Miles and AReaL represent roadmap work.

```mermaid
flowchart TB
    VIME["VIME · integrated"] --> ORCH["RL orchestration integration"]
    MILES["Miles · roadmap"] -.-> ORCH
    AREAL["AReaL · roadmap"] -.-> ORCH
    ORCH --> VLLM["vLLM · rollout"]
    ORCH --> MEGATRON["Megatron-LM · training"]
    VLLM --> RLK["RL-Kernel · deterministic and optimized operators"]
    MEGATRON --> RLK
    RLK --> CUDA["CUDA · SM90<br/>H100 / H200 / GH200"]
    RLK --> ROCM["ROCm · gfx942<br/>MI300A / MI300X / MI325X"]
    RLK -.-> ASCEND["Ascend · dav_c220<br/>partial adaptation"]
    RLK -.-> MUSA["Moore Threads / MUSA · in progress"]
```

The published benchmark validates **VIME + vLLM + Megatron-LM on CUDA**. Framework and
backend coverage are documented independently. See [runtime dispatch](./docs/design/runtime-dispatch.md)
for operator selection and strict execution contracts.

## Current Scope and Roadmap

Current end-to-end support is deliberately scoped to **Qwen3-8B Dense** and **VIME**.
Items in development or on the roadmap are not yet part of the supported path.

| Dimension | Available today | In development / roadmap |
| :--- | :--- | :--- |
| **Model architecture** | Qwen3-8B Dense | [DeepSeek-V4-Flash-0731 MoE](./docs/blog/2026-08-09-dsv4-flash-moe-consistency-roadmap.md) — active development |
| **RL orchestration** | VIME | Miles and AReaL |
| **Execution engines** | vLLM rollout + Megatron-LM training | Additional engine integrations will follow validated operator coverage |

## Benchmark Highlights

### VIME native vs. RL-Kernel + VIME

The completed experiment in [PR #377](https://github.com/RL-Align/RL-Kernel/pull/377)
compares **G10**, VIME's native production operator path, with **optimized G11**, the
strict RL-Kernel path. Both use VIME with vLLM rollout and Megatron-LM training, and both
enable rollout-logp reuse. G11 applies RL-Kernel attention, FFN, and LogP on both paths.

**Setup:** Qwen3-8B BF16 · GRPO · 1 node with 8×H100 80GB · actor TP4/CP2/PP1 ·
two TP4 rollout engines · 8 prompts × 16 samples (batch 128) · 200 steps · seed 1234 ·
maximum response length 7,168 · KL-loss coefficient 0.001.

| Metric | VIME native (G10) | VIME + RL-Kernel (G11) | G11 result |
| :--- | ---: | ---: | :--- |
| Steps with nonzero train–rollout LogP mismatch | 200 / 200 | **0 / 200** | **Exact agreement at every step** |
| Maximum absolute Δlogp across the run | 1.591547 | **0** | **Zero measured difference** |
| Mean rollout time | 130.22 s/step | **82.75 s/step** | **36.5% lower** |
| Mean rollout throughput | 672.39 tok/GPU/s | **1,134.00 tok/GPU/s** | **68.7% higher** |
| Mean reference LogP time | 20.90 s/step | 20.92 s/step | Approximately equal |
| Mean actor training time | **80.51 s/step** | 107.18 s/step | 33.1% higher |
| Mean end-to-end step time | 251.99 s/step | **231.27 s/step** | **8.2% lower** |

G11 saves **47.47 seconds per rollout step**, offsetting the additional actor training
cost for a net saving of **20.72 seconds per end-to-end step**.

![Qwen3-8B performance comparison: stage times, throughput, and relative changes for VIME native G10 and optimized RL-Kernel G11](./docs/assets/qwen3-8b-performance-summary.png)

## Hardware Support

The following matrix lists the current architecture targets and their corresponding
hardware models. It does not imply published end-to-end validation on every listed model,
or extend the current end-to-end model claim beyond Qwen3-8B Dense on H100.

| Vendor | Architecture target | Corresponding hardware models | Software stack | Coverage / progress |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA** | SM90 (compute capability 9.0) | H100, H200, GH200 | CUDA | **Supported**, dense-model strict path. Published end-to-end results use H100. |
| **AMD** | gfx942 (CDNA 3) | Instinct MI300A, MI300X, MI325X | ROCm | **Supported**, native extension loading and backend checks verified on MI300X. |
| **Huawei** | dav_c220 (CANN dav-2201 target) | Ascend dav_c220 | CANN 9.1.0 / Ascend C | **Partially adapted**, selected operators are available. Other Ascend models are being adapted. |
| **Moore Threads** | Under adaptation | Not specified | MUSA | **In progress**, hardware adaptation and dense-model integration are underway. |

Support applies to the listed architecture targets and implemented dense-model paths;
model, dtype, operator, and parallelism coverage varies by backend. The performance
numbers above are CUDA/H100 measurements. **Other hardware models are being adapted.**

## Quick Start

Install Python 3.10 or newer, a PyTorch build matching your accelerator runtime, and the
corresponding CUDA or ROCm compiler toolchain. Then clone RL-Kernel:

```bash
git clone https://github.com/RL-Align/RL-Kernel.git
cd RL-Kernel
```

For the Qwen3-8B train–rollout command and setup for VIME with RL-Kernel, see the
[8×H100 integration runbook](https://github.com/RL-Align/RL-Kernel/issues/342).

### NVIDIA CUDA

Build against a visible NVIDIA GPU. Set TORCH_CUDA_ARCH_LIST when you want to pin the
target architecture instead of relying on device detection.

```bash
# H100 / H200 / GH200 (SM90)
MAX_JOBS=8 \
RL_KERNEL_REQUIRE_EXT=1 \
TORCH_CUDA_ARCH_LIST="9.0+PTX" \
  python3 -m pip install --no-build-isolation --no-deps -e .
```

The supported CUDA target is SM90 (compute capability 9.0), corresponding to NVIDIA H100,
H200, and GH200. This build command and the native extension check below have been
validated on an NVIDIA H100 80GB HBM3. Other CUDA architectures are being adapted.

Verify the loaded extension, GPU, SM capability, and required native symbol:

```bash
python3 -c "import torch, rl_engine._C as C; print('GPU:', torch.cuda.get_device_name(0)); print('Capability:', torch.cuda.get_device_capability(0)); print('Extension:', C.__file__); print('fused_logp:', hasattr(C, 'fused_logp')); assert hasattr(C, 'fused_logp'); print('H100 build: PASS')"
```

### AMD ROCm

Set the target explicitly. The supported gfx942 target corresponds to AMD Instinct MI300A,
MI300X, and MI325X:

```bash
PYTORCH_ROCM_ARCH=gfx942 python3 setup.py develop
```

Verify the ROCm environment and required native symbol:

```bash
python3 scripts/check_rocm_env.py
python3 -c "import torch, rl_engine._C as C; print('GPU:', torch.cuda.get_device_name(0)); print('HIP:', torch.version.hip); print('Extension:', C.__file__); print('fused_logp:', hasattr(C, 'fused_logp')); assert hasattr(C, 'fused_logp'); print('MI300X build: PASS')"
```

The native extension load and environment checks have been validated on an AMD Instinct
MI300X reporting gfx942. Other ROCm architectures are being adapted.

For CPU-only or pure-Python development, use an editable pip installation. Ascend support
is currently limited to partial operator coverage on dav_c220 (CANN dav-2201 target);
other Ascend models are being adapted. Moore Threads support is also under development.
Neither backend has a general-purpose Quick Start build command yet. See the [installation
guide](./docs/getting_started/installation.md) for backend dependencies and troubleshooting.

## Community and Contributions

Join us on [Slack](https://rl-align.slack.com/join/shared_invite/zt-46bxj7uyt-gEK3xzwSJr_lppJsZolR~g#/shared-invite/email)
or [WeChat](./docs/community/wechat.md), and
[open an issue](https://github.com/RL-Align/RL-Kernel/issues) for bugs and feature requests.
Contributions to kernels, framework integrations, hardware adaptation, and benchmarks
are welcome. See the [contributing guide](./docs/contributing/README.md).

## Acknowledgments

RL-Kernel builds on the work of the open-source AI infrastructure community, including
[VIME](https://github.com/RL-Align/vime), [vLLM](https://github.com/vllm-project/vllm),
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM),
[DeepSpeed](https://github.com/deepspeedai/DeepSpeed), and
[FlashInfer](https://github.com/flashinfer-ai/flashinfer).
We thank their contributors and everyone helping bring RL-Kernel to new accelerators.

Licensed under the [Apache License 2.0](./LICENSE).
