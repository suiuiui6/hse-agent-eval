from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvaluationData:
    """处理 EvaluationData 相关逻辑。"""
    input: Any
    actual_output: Any


@dataclass(frozen=True)
class EvaluationOutput:
    """处理 EvaluationOutput 相关逻辑。"""
    score: float
    test_pass: bool
    label: str
    reason: str