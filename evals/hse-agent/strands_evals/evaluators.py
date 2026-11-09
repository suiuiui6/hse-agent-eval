from __future__ import annotations

from typing import Generic, TypeVar

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class Evaluator(Generic[TInput, TOutput]):
    """处理 Evaluator 相关逻辑。"""
    def evaluate(self, evaluation_case):
        """处理 evaluate 相关逻辑。"""
        raise NotImplementedError