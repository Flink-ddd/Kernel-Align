<p align="center">
  <img src="docs/assets/logo.png" width="220" alt="RL-Kernel Logo">
</p>

<h1 align="center">RL-Kernel</h1>

<p align="center">
  <strong>Extreme Infrastructure for GRPO & Large-Scale Reinforcement Learning.</strong>
</p>

<p align="center">
  <a href="https://rl-align.github.io/RL-Kernel/"><img src="https://img.shields.io/badge/Documentation-Docs-2ea44f" alt="Documentation"></a>
  <a href="https://rl-align.slack.com/join/shared_invite/zt-46bxj7uyt-gEK3xzwSJr_lppJsZolR~g#/shared-invite/email"><img src="https://img.shields.io/badge/Slack-Join%20Us-4A154B" alt="Slack"></a>
  <a href="./docs/community/wechat.md"><img src="https://img.shields.io/badge/WeChat-Join%20Group-07C160?logo=wechat&logoColor=white" alt="WeChat"></a>
  <a href="./docs/assets/whatsapp-group.png"><img src="https://img.shields.io/badge/WhatsApp-Join%20Group-25D366?logo=whatsapp&logoColor=white" alt="WhatsApp"></a>
  <a href="https://deepwiki.com/RL-Align/RL-Kernel"><img src="https://img.shields.io/badge/Ask-DeepWiki-7B3FE4" alt="Ask DeepWiki"></a>
  <a href="https://github.com/RL-Align/RL-Kernel"><img src="https://img.shields.io/badge/Hardware-NVIDIA%20CUDA%20%7C%20AMD%20ROCm%20%7C%20HUAWEI%20Ascend-orange" alt="Hardware"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
</p>


**RL-Kernel** is a high-performance, memory-efficient infrastructure for Reinforcement Learning post-training. It eliminates the memory and latency bottlenecks in Large Language Model alignment, This project targets AI infrastructure engineers, algorithm researchers, and enterprise-level large model alignment scenarios, providing specialized kernels for algorithms like **GRPO**, **PPO**, and **DPO**.


---

# Our Core Philosophy

**1. Operator-Level Train-Inference Consistency**
The biggest hidden barrier in large-scale RL is the subtle numerical divergence between rollout engines (e.g., vLLM) and training engines (e.g., Megatron/DeepSpeed). RL-Kernel implements deterministic operators for dense-model workloads to align numerical behavior between rollout and training paths. By pinning computational graphs and reduction orders at the operator level, it helps prevent reward hacking and distribution drift caused by numerical divergence.

**2. Extreme Memory & Compute Efficiency**
We replace naive PyTorch paths—which suffer from $O(G \cdot L \cdot V)$ memory explosion—with specialized industrial-grade kernels (like `prefix_shared_attention` and `fused_logp`). This reduces VRAM consumption and supports larger batch sizes for GRPO workloads.

---

# Global Architecture

RL-Kernel sits strictly at the operator layer, acting as a non-intrusive bridge between high-level alignment orchestration (e.g., vime, slime) and foundational execution engines. We ensure maximum throughput and rigorous numerical parity without modifying upstream framework source code.

<p align="center">
  <img src="docs/assets/RL-Kernel underlying operator library technical architecture.png" alt="RL-Kernel Global Architecture" width="800">
</p>

*Note: RL-Kernel integrates natively into Rollout Engines (vLLM, sglang, LMDeploy) and Training Engines (Megatron, DeepSpeed) via non-intrusive custom operator hooks, powered by CUDA, Triton, and ROCm backends, with an Ascend CANN backend for batch-invariant logprob.*

---

# Key Features

- **Dense-Model Train–Inference Consistency**: Implements operator-level consistency for dense-model workloads, validated end to end on Qwen3-8B Dense.
- **Multi-Platform Accelerator Support**: Supports **NVIDIA CUDA** and **AMD ROCm**, with **HUAWEI Ascend (NPU/CANN)** support for batch-invariant logprob.
- **Hardware Roadmap**: Broader support is planned for **MetaX**, **Cambricon**, **Moore Threads**, and more.

---

# Architecture

RL-Kernel sits between high-level alignment libraries and low-level accelerator kernels, ensuring maximum throughput without sacrificing flexibility.

---

# Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/RL-Align/RL-Kernel.git
cd RL-Kernel

# CPU-only / pure-Python fallback
python -m pip install -e .

# Native CUDA or ROCm extension (install a matching PyTorch build first)
RL_KERNEL_REQUIRE_EXT=1 python -m pip install --no-build-isolation -e .
python -c "import rl_engine._C as _C; assert hasattr(_C, 'fused_logp'); print(_C.__file__)"
```

### Contributions
Inspired by the kernel designs of vLLM and DeepSpeed. As an active contributor to the AI Infrastructure ecosystem, RL-Kernel aims to push the boundaries of RL efficiency.

Target: Building the most efficient RLHF toolchain for the open-source community.

# Support
Don’t hesitate to ask!

Contact the developers and community in [Slack](https://rl-align.slack.com/join/shared_invite/zt-46bxj7uyt-gEK3xzwSJr_lppJsZolR~g#/shared-invite/email) if you need any help.

[Open an issue](https://github.com/RL-Align/RL-Kernel/issues) if you find a bug in **RL-Kernel**.

# Documentation

The documentation of **RL-Kernel** is located on the website: [https://rl-align.github.io/RL-Kernel](https://rl-align.github.io/RL-Kernel)
or in the [docs](./docs) directory of the source code.

Featured docs:

- [Announcing RL-Kernel for vime: Faster and Leaner `linear_logp` for Full RL Rollouts](./docs/blog/2026-07-08-announcing-rl-kernel-linear-logp-for-vime.md)
- [中文版：发布 vime + RL-Kernel](./docs/blog/2026-07-08-announcing-rl-kernel-linear-logp-for-vime-zh.md)

# Acknowledgments

RL-Kernel builds on the shoulders of excellent open-source projects:

- **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)** — We integrate FlashInfer's fused sampling kernels as an NVIDIA backend for sampling workloads.
- **[vLLM](https://github.com/vllm-project/vllm)** — Inspired by vLLM's kernel design philosophy and hardware-aware scheduling approach.
- **[DeepSpeed](https://github.com/microsoft/DeepSpeed)** — Inspired by DeepSpeed's approach to memory-efficient training infrastructure.

We are grateful to these teams for their contributions to the open-source AI infrastructure ecosystem.
