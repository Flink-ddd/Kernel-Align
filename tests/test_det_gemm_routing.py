from types import SimpleNamespace

import pytest

import rl_engine.kernels.ops.cuda.matmul.det_gemm as det_gemm_module
from rl_engine.kernels.ops.cuda.matmul.det_gemm import DetGemmOp


def test_strict_route_report_is_explicit_and_emitted_once(monkeypatch, capsys):
    monkeypatch.setenv("RL_KERNEL_MODE", "strict")
    monkeypatch.setenv("RL_KERNEL_DET_GEMM_BACKEND", "sm90")
    monkeypatch.setattr(det_gemm_module, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(
        det_gemm_module,
        "_C",
        SimpleNamespace(det_gemm_fwd=lambda a, b: None),
    )
    monkeypatch.setattr(det_gemm_module, "_ROUTE_REPORTED", False)

    DetGemmOp()
    DetGemmOp()

    routes = [
        line for line in capsys.readouterr().out.splitlines() if "[route]" in line
    ]
    assert routes == [
        "[RL-Kernel][route] mode=strict module=gemm requested=strict "
        "actual=rlkernel.det_gemm.sm90.v1 fallback=false"
    ]


def test_missing_extension_fails_instead_of_falling_back(monkeypatch, capsys):
    monkeypatch.setenv("RL_KERNEL_DET_GEMM_BACKEND", "sm90")
    monkeypatch.setattr(det_gemm_module, "_EXT_AVAILABLE", False)
    monkeypatch.setattr(det_gemm_module, "_ROUTE_REPORTED", False)

    with pytest.raises(RuntimeError, match="refusing non-strict fallback"):
        DetGemmOp()
    assert "module=gemm" not in capsys.readouterr().out
