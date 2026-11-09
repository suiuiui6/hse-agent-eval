from __future__ import annotations

from typing import Generic, TypeVar

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class Evaluator(Generic[TInput, TOutput]):
    def evaluate(self, evaluation_case):
        raise NotImplementedError