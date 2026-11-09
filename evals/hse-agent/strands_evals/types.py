from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvaluationData:
    input: Any
    actual_output: Any


@dataclass(frozen=True)
class EvaluationOutput:
    score: float
    test_pass: bool
    label: str
    reason: str