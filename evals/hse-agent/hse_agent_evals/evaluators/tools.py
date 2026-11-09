from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from strands_evals.evaluators import Evaluator
from strands_evals.types import EvaluationData, EvaluationOutput

from hse_agent_evals.models import EvalCase, RunRecord, ToolCall


@dataclass(frozen=True)
class ToolScore:
    precision: float
    recall: float
    argument_accuracy: float
    sequence_accuracy: float
    forbidden_calls: list[str]
    actual_tools: list[str]
    allowed_selection_tools: list[str]
    argument_field_results: list[dict[str, Any]]
    matched_argument_field_count: int
    expected_argument_field_count: int
    passed: bool


def score_tools(
    required: Iterable[str],
    forbidden: Iterable[str],
    expected_arguments: dict[str, dict[str, Any]],
    actual: Iterable[dict[str, Any] | ToolCall],
    allowed_sequences: Iterable[Iterable[str]] | None = None,
) -> ToolScore:
    required_list = list(required)
    required_set = set(required_list)
    forbidden_set = set(forbidden)
    actual_calls = [_call_parts(call) for call in actual]
    actual_tools = [tool for tool, _ in actual_calls]

    sequences = None
    allowed_selection_set = set(required_set)
    if allowed_sequences is not None:
        sequences = [list(sequence) for sequence in allowed_sequences]
        for sequence in sequences:
            allowed_selection_set.update(sequence)

    if actual_tools:
        precision = sum(tool in allowed_selection_set for tool in actual_tools) / len(actual_tools)
    else:
        precision = 1.0 if not required_list else 0.0

    if required_list:
        recall = sum(tool in set(actual_tools) for tool in set(required_list)) / len(set(required_list))
    else:
        recall = 1.0

    expected_fields = 0
    matched_fields = 0
    argument_field_results: list[dict[str, Any]] = []
    first_args: dict[str, dict[str, Any]] = {}
    for tool, args in actual_calls:
        first_args.setdefault(tool, args)
    for tool in sorted(expected_arguments):
        expected = expected_arguments[tool]
        actual_args = first_args.get(tool, {})
        for path, expected_value in _leaf_values(expected):
            expected_fields += 1
            found, actual_value = _path_value(actual_args, path)
            matched = found and actual_value == expected_value
            matched_fields += matched
            argument_field_results.append(
                {
                    "matched": matched,
                    "path": ".".join(path),
                    "tool": tool,
                }
            )
    argument_accuracy = matched_fields / expected_fields if expected_fields else 1.0

    if allowed_sequences is None:
        sequence_accuracy = 1.0
    else:
        assert sequences is not None
        sequence_accuracy = 1.0 if actual_tools in sequences else 0.0

    forbidden_calls = [tool for tool in actual_tools if tool in forbidden_set]
    passed = (
        precision == 1.0
        and recall == 1.0
        and argument_accuracy == 1.0
        and sequence_accuracy == 1.0
        and not forbidden_calls
    )
    return ToolScore(
        precision=precision,
        recall=recall,
        argument_accuracy=argument_accuracy,
        sequence_accuracy=sequence_accuracy,
        forbidden_calls=forbidden_calls,
        actual_tools=actual_tools,
        allowed_selection_tools=sorted(allowed_selection_set),
        argument_field_results=argument_field_results,
        matched_argument_field_count=matched_fields,
        expected_argument_field_count=expected_fields,
        passed=passed,
    )


class ToolAccuracyEvaluator(Evaluator[EvalCase, RunRecord]):
    def evaluate(self, evaluation_case: EvaluationData) -> list[EvaluationOutput]:
        inputs = evaluation_inputs(evaluation_case)
        if inputs is None:
            return invalid_evaluation_output("tool_accuracy")
        case, run = inputs
        score = score_tools(
            required=case.required_tools,
            forbidden=case.forbidden_tools,
            expected_arguments=case.expected_arguments,
            actual=run.tool_calls,
            allowed_sequences=case.allowed_tool_sequences,
        )
        details = tool_evidence(case, run, score)
        numeric_score = (
            score.precision + score.recall + score.argument_accuracy + score.sequence_accuracy
        ) / 4
        return [
            EvaluationOutput(
                score=numeric_score,
                test_pass=score.passed,
                label="tool_accuracy",
                reason=stable_reason(details),
            )
        ]


def tool_evidence(case: EvalCase, run: RunRecord, score: ToolScore) -> dict[str, Any]:
    return {
        "actualTools": score.actual_tools,
        "allowedSelectionTools": score.allowed_selection_tools,
        "allowedToolSequences": case.allowed_tool_sequences,
        "argumentAccuracy": score.argument_accuracy,
        "argumentFieldResults": score.argument_field_results,
        "expectedArgumentFieldCount": score.expected_argument_field_count,
        "forbiddenCalls": score.forbidden_calls,
        "forbiddenTools": case.forbidden_tools,
        "matchedArgumentFieldCount": score.matched_argument_field_count,
        "precision": score.precision,
        "recall": score.recall,
        "requiredTools": case.required_tools,
        "sequenceAccuracy": score.sequence_accuracy,
        "status": run.status,
        "trajectoryIncomplete": run.trajectory_incomplete,
    }


def evaluation_inputs(evaluation_case: Any) -> tuple[EvalCase, RunRecord] | None:
    if not isinstance(evaluation_case, EvaluationData):
        return None
    if not isinstance(evaluation_case.input, EvalCase):
        return None
    if not isinstance(evaluation_case.actual_output, RunRecord):
        return None
    return evaluation_case.input, evaluation_case.actual_output


def invalid_evaluation_output(label: str) -> list[EvaluationOutput]:
    return [
        EvaluationOutput(
            score=0.0,
            test_pass=False,
            label=label,
            reason=stable_reason({"validationError": "invalid_evaluation_data"}),
        )
    ]


def stable_reason(details: dict[str, Any]) -> str:
    return json.dumps(details, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def status_eligible(
    status: str,
    confirmation_required: bool = False,
    expected_cancellation: bool = False,
) -> bool:
    normalized = status.strip().lower()
    if normalized in {
        "success",
        "completed",
        "complete",
    }:
        return True
    if expected_cancellation and normalized == "cancelled":
        return True
    return confirmation_required and normalized in {
        "waiting_confirmation",
        "confirmation_required",
        "pending_confirmation",
    }


def _call_parts(call: dict[str, Any] | ToolCall) -> tuple[str, dict[str, Any]]:
    if isinstance(call, ToolCall):
        return call.tool, call.args
    if isinstance(call, dict):
        tool = call.get("tool", "")
        args = call.get("args", {})
        return (tool if isinstance(tool, str) else "", args if isinstance(args, dict) else {})
    return "", {}


def _leaf_values(value: dict[str, Any], prefix: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    leaves: list[tuple[tuple[str, ...], Any]] = []
    for key in sorted(value):
        item = value[key]
        path = prefix + (key,)
        if isinstance(item, dict) and item:
            leaves.extend(_leaf_values(item, path))
        else:
            leaves.append((path, item))
    return leaves


def _path_value(value: dict[str, Any], path: tuple[str, ...]) -> tuple[bool, Any]:
    current: Any = value
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current
