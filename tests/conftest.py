# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import os
import pathlib
import sys

# Accuracy tests compare fused kernels against native fp32 PyTorch oracles.
# NVIDIA_TF32_OVERRIDE=1 (set by some GPU cloud images) forces cuBLAS into
# TF32 regardless of torch.backends settings, degrading the oracle itself to
# ~1e-2 absolute error. Pin it off before the first cuBLAS handle is created
# so fp32 comparisons stay meaningful on every CI machine.
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"


def _add_windows_dll_dirs():
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return

    try:
        import torch
    except ImportError:
        return

    candidate_dirs = [pathlib.Path(torch.__file__).parent / "lib"]

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidate_dirs.append(pathlib.Path(cuda_path) / "bin")

    for path in candidate_dirs:
        if path.exists():
            os.add_dll_directory(str(path))


_add_windows_dll_dirs()
