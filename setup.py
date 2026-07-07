# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import os

from setuptools import find_packages, setup


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


def _sm90_arch_from_env_or_device(cc_major: int, cc_minor: int) -> str:
    override = os.environ.get("KERNEL_ALIGN_SM90_ARCH")
    if override:
        return (
            override.strip().lower().removeprefix("sm_").removeprefix("compute_").replace(".", "")
        )
    if cc_major == 9:
        return f"{cc_major}{cc_minor}a"
    return "90a"


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

    if is_rocm:
        rocm_extension = ROCMExtension or CUDAExtension
        if rocm_extension is None:
            return []
        extensions.append(
            rocm_extension(
                name="rl_engine._C",
                sources=[
                    "csrc/ops.cpp",
                    "csrc/fused_logp_kernel.cu",
                ],
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                    # PyTorch ROCm builds may expose CUDAExtension but not
                    # ROCMExtension; BuildExtension still routes the "nvcc"
                    # bucket to hipcc in that case. Keep these flags HIP-safe.
                    "nvcc": ["-O3"],
                },
                extra_link_args=list(torch_rpath),
            )
        )
    elif torch.cuda.is_available() or os.environ.get("FORCE_CUDA") == "1":
        cuda_sources = [
            "csrc/ops.cpp",
            "csrc/fused_logp_kernel.cu",
            "csrc/cuda/attention/prefix_shared_attention.cu",
        ]

        cc_major, cc_minor = torch.cuda.get_device_capability()
        nvcc_flags = ["-O3", "--use_fast_math", "-Xfatbin", "-compress-all"]
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
        if os.environ.get("KERNEL_ALIGN_NCU_LINEINFO") == "1":
            nvcc_flags.append("-lineinfo")

        cxx_flags = ["-O3", "-std=c++17", "-DKERNEL_ALIGN_WITH_CUDA"]
        extra_link_args = list(torch_rpath)

        sm90_srcs = [
            "csrc/cuda/fused_logp_sm90.cu",
            "csrc/cuda/fused_linear_logp_sm90.cu",  # TMA + WGMMA fused linear log-prob
        ]
        enable_sm90 = os.environ.get("KERNEL_ALIGN_FORCE_SM90") == "1"
        present_sm90 = [s for s in sm90_srcs if os.path.exists(s)]
        if enable_sm90 and present_sm90:
            # WGMMA/TMA require an arch-native 'a' target.  Keep forced SM90
            # builds probeable from non-Hopper CUDA hosts by defaulting to 90a.
            tma_arch = _sm90_arch_from_env_or_device(cc_major, cc_minor)
            cuda_sources.extend(present_sm90)
            nvcc_flags.append(f"-gencode=arch=compute_{tma_arch},code=sm_{tma_arch}")
            cxx_flags.append("-DKERNEL_ALIGN_WITH_SM90")
            extra_link_args.append("-lcuda")

        extensions.append(
            CUDAExtension(
                name="rl_engine._C",
                sources=cuda_sources,
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


TEST_REQUIRES = ["pytest", "tabulate"]
BENCH_REQUIRES = ["tabulate"]
HF_REQUIRES = ["accelerate", "transformers"]
DEV_REQUIRES = [
    *TEST_REQUIRES,
    "black",
    "isort",
    "ruff",
    "mypy",
    "pre-commit",
    *HF_REQUIRES,
]


setup(
    name="rl-engine",
    version="0.1.0",
    packages=find_packages(include=["rl_engine", "rl_engine.*"]),
    install_requires=[
        "torch>=2.4.1",
        "numpy",
    ],
    ext_modules=get_extensions(),
    cmdclass=get_cmdclass(),
    extras_require={
        "cuda": ["flashinfer-python", "nvidia-ml-py"],
        # amd-aiter must be installed from source; see docs/getting_started/installation.md
        "rocm": [],
        "vllm": ["vllm>=0.6.0"],
        "hf": HF_REQUIRES,
        "bench": BENCH_REQUIRES,
        "test": TEST_REQUIRES,
        "dev": DEV_REQUIRES,
    },
    python_requires=">=3.10",
    include_package_data=True,
    zip_safe=False,
)
