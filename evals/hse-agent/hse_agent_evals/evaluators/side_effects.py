from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from strands_evals.evaluators import Evaluator
from strands_evals.types import EvaluationData, EvaluationOutput

from hse_agent_evals.contracts import canonical_side_effect_key
from hse_agent_evals.evaluators.tools import (
    evaluation_inputs,
    invalid_evaluation_output,
    stable_reason,
)
from hse_agent_evals.models import EvalCase, RunRecord, SideEffect


@dataclass(frozen=True)
class SideEffectScore:
    actual: list[dict[str, Any]]
    missing_expected: list[str]
    forbidden_effects: list[str]
    telemetry_mismatch: str | None
    passed: bool


def score_side_effects(
    expected: Iterable[str],
    forbidden: Iterable[str],
    actual: Iterable[SideEffect],
    side_effect_committed: bool,
) -> SideEffectScore:
    actual_list = list(actual)
    evidence = [side_effect_evidence(effect) for effect in actual_list]
    expected_list = list(expected)
    forbidden_list = list(forbidden)
    missing_expected = [
        key
        for key in expected_list
        if not any(effect_matches_key(effect, key) for effect in actual_list)
    ]
    forbidden_effects = [
        key
        for key in forbidden_list
        if any(effect_matches_key(effect, key) for effect in actual_list)
    ]
    telemetry_mismatch = side_effect_telemetry_mismatch(side_effect_committed, evidence)
    return SideEffectScore(
        actual=evidence,
        missing_expected=missing_expected,
        forbidden_effects=forbidden_effects,
        telemetry_mismatch=telemetry_mismatch,
        passed=not missing_expected and not forbidden_effects and telemetry_mismatch is None,
    )


def effect_matches_key(effect: SideEffect, expected_key: str) -> bool:
    if canonical_side_effect_key(effect.key) != canonical_side_effect_key(expected_key):
        return False
    occurred = effect.before_count != effect.after_count or bool(effect.changed_rows)
    if not occurred:
        return False
    # 告警表共用一条 snapshot 查询，用行数变化区分新增和已读更新。
    if expected_key == "compliance.alert.insert":
        return effect.after_count > effect.before_count
    if expected_key == "alert.read":
        return effect.after_count == effect.before_count and bool(effect.changed_rows)
    return True


def side_effect_evidence(effect: SideEffect) -> dict[str, Any]:
    changed_row_fields = sorted(
        {
            field
            for row in effect.changed_rows
            for field in row
        }
    )
    return {
        "afterCount": effect.after_count,
        "beforeCount": effect.before_count,
        "changedRowCount": len(effect.changed_rows),
        "changedRowFields": changed_row_fields,
        "key": effect.key,
        "occurred": effect.before_count != effect.after_count or bool(effect.changed_rows),
    }


def side_effect_telemetry_mismatch(
    side_effect_committed: bool, evidence: Iterable[dict[str, Any]]
) -> str | None:
    occurred = any(item["occurred"] for item in evidence)
    if side_effect_committed and not occurred:
        return "committed_without_snapshot"
    if not side_effect_committed and occurred:
        return "snapshot_without_committed"
    return None


class SideEffectEvaluator(Evaluator[EvalCase, RunRecord]):
    def evaluate(self, evaluation_case: EvaluationData) -> list[EvaluationOutput]:
        inputs = evaluation_inputs(evaluation_case)
        if inputs is None:
            return invalid_evaluation_output("side_effects")
        case, run = inputs
        score = score_side_effects(
            expected=case.expected_side_effects,
            forbidden=case.forbidden_side_effects,
            actual=run.side_effects,
            side_effect_committed=run.side_effect_committed,
        )
        details = side_effect_evaluator_evidence(case, run, score)
        return [
            EvaluationOutput(
                score=1.0 if score.passed else 0.0,
                test_pass=score.passed,
                label="side_effects",
                reason=stable_reason(details),
            )
        ]


def side_effect_evaluator_evidence(
    case: EvalCase, run: RunRecord, score: SideEffectScore
) -> dict[str, Any]:
    return {
        "actualSideEffects": score.actual,
        "expectedSideEffects": case.expected_side_effects,
        "forbiddenSideEffects": score.forbidden_effects,
        "forbiddenSideEffectRules": case.forbidden_side_effects,
        "missingExpectedSideEffects": score.missing_expected,
        "sideEffectCommitted": run.side_effect_committed,
        "sideEffectTelemetryMismatch": score.telemetry_mismatch,
        "status": run.status,
        "trajectoryIncomplete": run.trajectory_incomplete,
    }
