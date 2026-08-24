# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Fail-closed operator routing shared by framework integrations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import Lock
from typing import Any

from rl_engine.integrations.ablation import Implementation, IntegrationPlan


@dataclass(frozen=True)
class OperatorReadback:
    framework: str
    target: str
    module: str
    case_id: str
    implementation: str
    backend_id: str
    call_count: int

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class FrameworkOperatorIntegration:
    """Route framework calls without importing or modifying framework packages."""

    def __init__(
        self,
        *,
        framework: str,
        target: str,
        plan: IntegrationPlan,
        rl_kernel_operators: Mapping[str, Callable[..., Any]],
    ) -> None:
        self.framework = framework
        self.target = target
        self.plan = plan
        self._rl_kernel_operators = dict(rl_kernel_operators)
        self._counts: Counter[str] = Counter()
        self._readbacks: dict[str, OperatorReadback] = {}
        self._lock = Lock()

    def execute(
        self,
        module: str,
        native: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        normalized = module.strip().lower()
        if not callable(native):
            raise TypeError("native operator must be callable")
        implementation = self.plan.implementation_for(normalized, self.target)
        selected: Callable[..., Any]
        if implementation is Implementation.PRODUCTION:
            selected = native
        else:
            rl_kernel_operator = self._rl_kernel_operators.get(normalized)
            if rl_kernel_operator is None:
                raise RuntimeError(
                    f"{self.framework} {normalized} selected RL-Kernel "
                    "but no operator was installed"
                )
            selected = rl_kernel_operator
        result = selected(*args, **kwargs)
        backend_id = getattr(selected, "backend_id", None)
        if not isinstance(backend_id, str) or not backend_id.strip():
            backend_id = (
                f"{self.framework}.production.{normalized}"
                if implementation is Implementation.PRODUCTION
                else f"rlkernel.{normalized}.unidentified"
            )
        with self._lock:
            self._counts[normalized] += 1
            case = self.plan.cases[normalized]
            self._readbacks[normalized] = OperatorReadback(
                framework=self.framework,
                target=self.target,
                module=normalized,
                case_id=case.case_id,
                implementation=implementation.value,
                backend_id=backend_id,
                call_count=self._counts[normalized],
            )
        return result

    def readback(self) -> dict[str, Any]:
        with self._lock:
            return {
                "framework": self.framework,
                "target": self.target,
                "plan": self.plan.to_dict(),
                "operators": {
                    module: readback.to_dict() for module, readback in self._readbacks.items()
                },
            }


__all__ = ["FrameworkOperatorIntegration", "OperatorReadback"]
