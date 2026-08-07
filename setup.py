# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import importlib.util
import os
from pathlib import Path

from setuptools import find_packages, setup


def _load_envs_module():
    envs_path = Path(__file__).with_name("envs.py")
    spec = importlib.util.spec_from_file_location("_rl_kernel_envs", envs_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load environment helpers from {envs_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


envs = _load_envs_module()


def _load_torch_extension_tools():
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except ImportError:
        return None, None, None, None

    try:
        from torch.utils.cpp_extension import ROCMExtension
    except ImportError:
        ROCMExtension = None

    return torch, BuildExtension, CUDAExtension, ROCMExtension


def _cuda_define_from_env(name: str, macro: str) -> list[str]:
    value = os.environ.get(name)
    if value is None:
        return []
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return [f"-D{macro}={parsed}"]


def get_extensions():
    torch, _, CUDAExtension, ROCMExtension = _load_torch_extension_tools()
    if torch is None:
        return []

    extensions = []
    torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
    torch_rpath = ["-Wl,-rpath,$ORIGIN/../torch/lib"]
    if os.environ.get("KERNEL_ALIGN_DEV_RPATH") == "1":
        torch_rpath.append(f"-Wl,-rpath,{torch_lib_dir}")
    is_rocm = torch.version.hip is not None

    if is_rocm and ROCMExtension is not None:
        extensions.append(
            ROCMExtension(
                name="rl_engine._C",
                sources=[
                    "csrc/ops.cpp",
                    "csrc/fused_logp_kernel.cpp",
                ],
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                    "hipcc": ["-O3", "--use_fast_math", "-Xhipcc", "-compress-all"],
                },
                extra_link_args=list(torch_rpath),
            )
        )
    elif torch.cuda.is_available():
        cuda_sources = [
            "csrc/ops.cpp",
            "csrc/fused_logp_kernel.cu",
            "csrc/deterministic_logp_kernel.cu",
            "csrc/cuda/attention/prefix_shared_attention.cu",
            "csrc/cuda/gemm/det_gemm_kernel.cu",
            "csrc/cuda/rmsnorm.cu",
            "csrc/cuda/attention/deterministic_attention.cu",
        ]

        cc_major, cc_minor = torch.cuda.get_device_capability()
        enable_sm90 = os.environ.get("KERNEL_ALIGN_FORCE_SM90") == "1"
        nvcc_flags = ["-O3", "-Xfatbin", "-compress-all"]
        if envs.env_flag(envs.KERNEL_ALIGN_USE_FAST_MATH):
            nvcc_flags.append("--use_fast_math")
        if not enable_sm90:
            # SM90 build emits 90a below; mixing plain compute_90 breaks TMA ptxas.
            nvcc_flags.append(
                f"-gencode=arch=compute_{cc_major}{cc_minor},code=sm_{cc_major}{cc_minor}"
            )
        nvcc_flags.append("--expt-relaxed-constexpr")
        nvcc_flags.append("--expt-extended-lambda")
        nvcc_flags.extend(
            _cuda_define_from_env(
                "FUSED_LOGP_TWOPASS_BLOCK_SIZE",
                "FUSED_LOGP_TWOPASS_BLOCK_SIZE",
            )
        )
        nvcc_flags.extend(
            _cuda_define_from_env(
                "FUSED_LOGP_ONLINE_BLOCK_SIZE",
                "FUSED_LOGP_ONLINE_BLOCK_SIZE",
            )
        )
        nvcc_flags.extend(
            _cuda_define_from_env(
                "FUSED_LOGP_ONLINE_SPARSE_LARGE_VOCAB_BLOCK_SIZE",
                "FUSED_LOGP_ONLINE_SPARSE_LARGE_VOCAB_BLOCK_SIZE",
            )
        )
        nvcc_flags.extend(
            _cuda_define_from_env(
                "FUSED_LOGP_ONLINE_LARGE_ROW_BYTES_THRESHOLD",
                "FUSED_LOGP_ONLINE_LARGE_ROW_BYTES_THRESHOLD",
            )
        )
        nvcc_flags.extend(
            _cuda_define_from_env(
                "FUSED_LOGP_ONLINE_SPARSE_DENSITY_NUMERATOR",
                "FUSED_LOGP_ONLINE_SPARSE_DENSITY_NUMERATOR",
            )
        )
        nvcc_flags.extend(
            _cuda_define_from_env(
                "FUSED_LOGP_ONLINE_SPARSE_DENSITY_DENOMINATOR",
                "FUSED_LOGP_ONLINE_SPARSE_DENSITY_DENOMINATOR",
            )
        )
        nvcc_flags.extend(
            _cuda_define_from_env(
                "FUSED_LOGP_ONLINE_MIN_BLOCKS_PER_SM",
                "FUSED_LOGP_ONLINE_MIN_BLOCKS_PER_SM",
            )
        )
        if envs.env_flag(envs.KERNEL_ALIGN_NCU_LINEINFO):
            nvcc_flags.append("-lineinfo")
        if os.name == "nt" and envs.env_flag(envs.KERNEL_ALIGN_ALLOW_UNSUPPORTED_MSVC):
            nvcc_flags.append("-allow-unsupported-compiler")
            nvcc_flags.append("-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")

        cxx_flags = ["-O3", "-std=c++17", "-DKERNEL_ALIGN_WITH_CUDA"]
        extra_link_args = list(torch_rpath)

        sm90_srcs = [
            "csrc/cuda/fused_logp_sm90.cu",
            "csrc/cuda/fused_linear_logp_sm90.cu",  # TMA + WGMMA fused linear log-prob
            "csrc/cuda/batch_invariant_logp_kernel_sm90.cu",  # TMA batch-invariant logp
            "csrc/cuda/rope_sm90.cu",  # RoPE rotate-half apply, gated to SM90 build
            "csrc/cuda/embedding_lm_head_sm90.cu",  # single-card batch-invariant embedding/lm-head
        ]
        enable_sm90 = envs.env_flag(envs.KERNEL_ALIGN_FORCE_SM90)
        present_sm90 = [s for s in sm90_srcs if os.path.exists(s)]
        if enable_sm90 and present_sm90:
            tma_arch = f"{cc_major}{cc_minor}a"  # WGMMA/TMA require the arch-native 'a' variant
            cuda_sources.extend(present_sm90)
            nvcc_flags.append(f"-gencode=arch=compute_{tma_arch},code=sm_{tma_arch}")
            cxx_flags.append("-DKERNEL_ALIGN_WITH_SM90")
            extra_link_args.append("-lcuda")

        # det_gemm SM90 (mma.sync + TMA) path: independent of the fused_logp
        # SM90 sources, which currently fail ptxas on CUDA 12.4 (shared::cta in
        # the shared tma_utils.cuh). det_gemm uses its own gemm/det_gemm_tma.cuh.
        enable_det_gemm_sm90 = os.environ.get("KERNEL_ALIGN_DET_GEMM_SM90") == "1"
        if enable_det_gemm_sm90:
            tma_arch = f"{cc_major}{cc_minor}a"
            arch_flag = f"-gencode=arch=compute_{tma_arch},code=sm_{tma_arch}"
            if arch_flag not in nvcc_flags:
                nvcc_flags.append(arch_flag)
            if "-lcuda" not in extra_link_args:
                extra_link_args.append("-lcuda")
            nvcc_flags.append("-DRL_KERNEL_ENABLE_SM90")
            cxx_flags.append("-DRL_KERNEL_ENABLE_SM90")

        # Qwen3 SiLU/SwiGLU is independently buildable, following det_gemm's
        # fail-closed pattern. This avoids forcing unrelated TMA/logp sources
        # into an activation-only build. KERNEL_ALIGN_FORCE_SM90 remains an
        # umbrella switch for users that intentionally build every SM90 op.
        enable_activation_sm90 = enable_sm90 or envs.env_flag(envs.KERNEL_ALIGN_ACTIVATION_SM90)
        activation_sm90_source = "csrc/cuda/activation/swiglu_sm90.cu"
        if enable_activation_sm90 and not os.path.exists(activation_sm90_source):
            raise FileNotFoundError(f"SM90 activation source is missing: {activation_sm90_source}")
        if enable_activation_sm90:
            cuda_sources.append(activation_sm90_source)
            nvcc_flags.append("-DRL_KERNEL_ENABLE_ACTIVATION_SM90")
            cxx_flags.append("-DRL_KERNEL_ENABLE_ACTIVATION_SM90")

        extensions.append(
            CUDAExtension(
                name="rl_engine._C",
                sources=cuda_sources,
                include_dirs=[],
                extra_compile_args={
                    "cxx": cxx_flags,
                    "nvcc": nvcc_flags,
                },
                extra_link_args=extra_link_args,
            )
        )
    return extensions


def get_cmdclass():
    _, BuildExtension, _, _ = _load_torch_extension_tools()
    if BuildExtension is None:
        return {}
    return {"build_ext": BuildExtension}


setup(
    name="rl-engine",
    version="0.1.0",
    packages=find_packages(include=["rl_engine", "rl_engine.*"]),
    install_requires=[
        "torch>=2.4.1",
        "tabulate",
        "numpy",
        "accelerate",
        "transformers==5.13.1",
    ],
    ext_modules=get_extensions(),
    cmdclass=get_cmdclass(),
    extras_require={
        "cuda": ["flashinfer"],
        "rocm": ["aiter"],
        "vllm": ["vllm>=0.6.0"],
    },
    python_requires=">=3.10",
    include_package_data=True,
    zip_safe=False,
)
