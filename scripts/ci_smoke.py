# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CI fail-fast smoke check for the compiled CUDA extension (``rl_engine._C``).

Exits non-zero with a clear message when either failure mode from issue #191
occurs, so GPU CI cannot pass while silently running only native fallbacks:

  * Bug A - the extension did not build at all (e.g. PEP 517 build isolation hid
    torch from ``setup.py``), so ``from rl_engine import _C`` raises ImportError.
  * Bug B - the extension built for the wrong GPU architecture; the import
    succeeds (``dlopen`` does not check arch) but the first kernel launch raises
    ``cudaErrorNoKernelImageForDevice`` once the stream is synchronized.

The baseline launch uses ``fused_logp``, which is registered unconditionally. When an SM90
activation build is requested on Hopper, the smoke check additionally requires and launches
the fused SwiGLU forward symbol.
"""
import os
import sys

import torch


def main() -> int:
    if not torch.cuda.is_available():
        print("[smoke] FATAL: CUDA is not available in this CI environment", file=sys.stderr)
        return 2

    print(f"[smoke] torch: {torch.__version__} (cuda {torch.version.cuda})")
    print(f"[smoke] device: {torch.cuda.get_device_name()}")
    cc = torch.cuda.get_device_capability()
    print(f"[smoke] capability: sm_{cc[0]}{cc[1]}")

    # (1) Import check -> catches Bug A (no .so was compiled at all).
    try:
        from rl_engine import _C
    except ImportError as exc:
        print(
            "[smoke] FATAL: compiled extension rl_engine._C is missing - the CUDA "
            "kernels were not built.\n"
            "        Likely PEP 517 build isolation hid torch from setup.py; install "
            "with `pip install --no-build-isolation -e .`.\n"
            f"        Underlying error: {exc}",
            file=sys.stderr,
        )
        return 1
    print(f"[smoke] _C file: {getattr(_C, '__file__', None)}")

    # (2) Real launch + synchronize -> catches Bug B (arch mismatch only surfaces
    #     on launch, asynchronously, so the sync is required to raise it here).
    try:
        logits = torch.randn(4, 32, device="cuda", dtype=torch.float32)
        token_ids = torch.randint(0, 32, (4,), device="cuda", dtype=torch.long)
        out = _C.fused_logp(logits, token_ids)
        torch.cuda.synchronize()
    # Broad on purpose: an arch mismatch raises a CUDA RuntimeError (not ImportError),
    # and any launch failure whatsoever must fail the smoke check loudly.
    except Exception as exc:
        print(
            "[smoke] FATAL: rl_engine._C built but fused_logp failed to launch on "
            f"sm_{cc[0]}{cc[1]}.\n"
            "        The extension was likely compiled for a different architecture; "
            "set TORCH_CUDA_ARCH_LIST / TARGET_SM to match this GPU.\n"
            f"        Underlying error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    if tuple(out.shape) != (4,):
        print(
            f"[smoke] FATAL: unexpected fused_logp output shape {tuple(out.shape)}", file=sys.stderr
        )
        return 1

    activation_requested = (
        os.environ.get("KERNEL_ALIGN_FORCE_SM90") == "1"
        or os.environ.get("KERNEL_ALIGN_ACTIVATION_SM90") == "1"
    )
    if cc[0] == 9 and activation_requested:
        activation_symbols = ("swiglu_forward_sm90",)
        missing = [symbol for symbol in activation_symbols if not hasattr(_C, symbol)]
        if missing:
            print(
                "[smoke] FATAL: SM90 activation build is missing symbols: " + ", ".join(missing),
                file=sys.stderr,
            )
            return 1
        try:
            gate = torch.randn(3, 17, device="cuda", dtype=torch.bfloat16)
            up = torch.randn_like(gate)
            activation_out = _C.swiglu_forward_sm90(gate, up)
            torch.cuda.synchronize()
        except Exception as exc:
            print(
                "[smoke] FATAL: SM90 SwiGLU forward launch failed: " f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 1
        expected_shape = tuple(gate.shape)
        if tuple(activation_out.shape) != expected_shape:
            print("[smoke] FATAL: SM90 SwiGLU returned an unexpected shape", file=sys.stderr)
            return 1

    print(f"[smoke] OK: rl_engine._C built and fused_logp ran on sm_{cc[0]}{cc[1]}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
