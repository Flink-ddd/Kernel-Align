# RL-Engine

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hardware](https://img.shields.io/badge/Hardware-AMD%20ROCm%20%7C%20NVIDIA%20CUDA-orange)](https://github.com/Flink-ddd/RL-Engine-Lab)

**RL-Engine-Lab** is a transparent, high-performance infrastructure for Reinforcement Learning (RL) post-training. It bridges the gap between high-level alignment algorithms (DPO, GRPO, PPO) and low-level hardware optimizations on both AMD (ROCm) and NVIDIA (CUDA) platforms.

## Key Features

- **Hardware-Aware Design**: Built-in support for AMD/NVIDIA with automatic backend discovery.
- **Inference Optimized**: Native integration with **vLLM** for fast rollout/sampling.
- **Alignment Ready**: Clean implementations of DPO and GRPO (DeepSeek-style).
- **Infra-First**: Designed for learning and extending RLHF toolchains (DeepSpeed, Ray, Triton).

---
*Stay hungry, stay infra.*
