from __future__ import annotations

from typing import Any

from strands_evals.evaluators import Evaluator
from strands_evals.types import EvaluationData, EvaluationOutput

from hse_agent_evals.evaluators.tools import (
    evaluation_inputs,
    invalid_evaluation_output,
    stable_reason,
    status_eligible,
)
from hse_agent_evals.models import EvalCase, RunRecord


class EfficiencyEvaluator(Evaluator[EvalCase, RunRecord]):
    def evaluate(self, evaluation_case: EvaluationData) -> list[EvaluationOutput]:
        inputs = evaluation_inputs(evaluation_case)
        if inputs is None:
            return invalid_evaluation_output("efficiency")
        case, run = inputs
        rounds_within_limit = run.round_count <= case.max_rounds
        task_status_eligible = status_eligible(
            run.status,
            case.confirmation_required,
            expected_cancellation=case.confirmation_action == "cancel",
        )
        passed = rounds_within_limit and task_status_eligible
        details = efficiency_evidence(case, run, rounds_within_limit, task_status_eligible)
        return [
            EvaluationOutput(
                score=1.0 if passed else 0.0,
                test_pass=passed,
                label="efficiency",
                reason=stable_reason(details),
            )
        ]


def efficiency_evidence(
    case: EvalCase,
    run: RunRecord,
    rounds_within_limit: bool | None = None,
    task_status_eligible: bool | None = None,
) -> dict[str, Any]:
    if rounds_within_limit is None:
        rounds_within_limit = run.round_count <= case.max_rounds
    if task_status_eligible is None:
        task_status_eligible = status_eligible(
            run.status,
            case.confirmation_required,
            expected_cancellation=case.confirmation_action == "cancel",
        )
    return {
        "durations": {
            "modelMs": run.model_duration_ms,
            "toolMs": run.tool_duration_ms,
            "totalMs": run.duration_ms,
        },
        "maxRounds": case.max_rounds,
        "roundCount": run.round_count,
        "roundsWithinLimit": rounds_within_limit,
        "status": run.status,
        "taskStatusEligible": task_status_eligible,
        "tokens": {
            "completion": run.completion_tokens,
            "prompt": run.prompt_tokens,
            "total": run.total_tokens,
        },
        "trajectoryIncomplete": run.trajectory_incomplete,
    }
