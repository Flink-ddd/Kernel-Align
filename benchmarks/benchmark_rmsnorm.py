import argparse
import time

import torch

from rl_engine.kernels.ops.pytorch.norm.rms_norm import NativeRMSNormOp
from rl_engine.platforms.device import device_ctx

if device_ctx.device_type != "npu":
    from rl_engine.kernels.ops.triton.rmsnorm_triton import rmsnorm_triton

    try:
        from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
        from rl_engine.kernels.ops.cuda.norm.rmsnorm import rmsnorm_cuda

        HAS_CUDA_EXT = _EXT_AVAILABLE and hasattr(_C, "rmsnorm_forward")
    except ImportError:
        HAS_CUDA_EXT = False


def _accel():
    """The active accelerator module (torch.npu on Ascend, torch.cuda otherwise)."""
    if device_ctx.device_type == "npu":
        return torch.npu
    return torch.cuda


def _maybe_ascend_op():
    """The Ascend C op, or None when unavailable (no NPU / not built)."""
    if device_ctx.device_type != "npu":
        return None
    try:
        from rl_engine.kernels.ops.ascend.norm.rmsnorm import RMSNormAscendOp

        return RMSNormAscendOp()
    except (ImportError, OSError, RuntimeError):
        return None


def bench(fn, x, w, dy, warmup=20, iters=100):
    accel = _accel()
    for _ in range(warmup):
        x.grad = None
        w.grad = None
        y = fn(x, w)
        y.backward(dy)
    accel.synchronize()

    start = time.time()
    for _ in range(iters):
        x.grad = None
        w.grad = None
        y = fn(x, w)
        y.backward(dy)
    accel.synchronize()
    return (time.time() - start) * 1000.0 / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=1024)
    parser.add_argument("--H", type=int, default=4096)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    args = parser.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    device = "npu" if device_ctx.device_type == "npu" else "cuda"
    T, H = args.T, args.H

    torch.manual_seed(0)
    x_base = torch.randn((T, H), device=device, dtype=dtype) * 0.2
    w_base = torch.randn((H,), device=device, dtype=dtype) * 0.2
    dy = torch.randn((T, H), device=device, dtype=dtype) * 0.2

    def make_inputs():
        return (
            x_base.detach().clone().requires_grad_(True),
            w_base.detach().clone().requires_grad_(True),
        )

    native = NativeRMSNormOp()

    x, w = make_inputs()
    t_ref = bench(lambda a, b: native.forward(a, b), x, w, dy)
    print(f"pytorch ref : {t_ref:.4f} ms")

    if device == "cuda":
        x, w = make_inputs()
        t_tri = bench(lambda a, b: rmsnorm_triton(a, b), x, w, dy)
        print(f"triton      : {t_tri:.4f} ms | speedup vs ref: {t_ref / t_tri:.2f}x")

        if HAS_CUDA_EXT:
            x, w = make_inputs()
            t_cuda = bench(lambda a, b: rmsnorm_cuda(a, b), x, w, dy)
            print(f"cuda        : {t_cuda:.4f} ms | speedup vs ref: {t_ref / t_cuda:.2f}x")
        else:
            print("cuda        : skipped, extension is not built")
    else:
        ascend_op = _maybe_ascend_op()
        if ascend_op is not None:
            x, w = make_inputs()
            t_asc = bench(lambda a, b: ascend_op(a, b), x, w, dy)
            print(f"ascend      : {t_asc:.4f} ms | speedup vs ref: {t_ref / t_asc:.2f}x")
        else:
            print("ascend      : skipped, extension is not built")


if __name__ == "__main__":
    main()
