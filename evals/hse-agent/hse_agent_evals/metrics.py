from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from hse_agent_evals.models import RunRecord
from hse_agent_evals.runner import ExecutionPlan, run_key


def percentile(values: Sequence[float], q: float) -> float | None:
    """线性插值分位数（与原始设计一致：[1,2,5] 的 p95 = 4.7）。

    q 先夹到 [0,100]：越界的分位会算出超范围的 rank 并抛 IndexError，
    报告侧不应因一个配置值而整体失败。
    """
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    clamped = min(100.0, max(0.0, float(q)))
    rank = (clamped / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def cost_cny(*, model_duration_ms: int, gpu_hourly_cost_cny: float | None) -> float | None:
    """缺定价证据时返回 None，绝不返回 0（成本完整性门禁）。"""
    if gpu_hourly_cost_cny is None:
        return None
    return round(model_duration_ms / 3_600_000 * gpu_hourly_cost_cny, 10)


def _stat_block(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None}
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
    }


def _index(scores: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, bool]]:
    """键 = case_id|repetition|attempt，值 = label -> testPass。

    评分行来自 append-only JSONL，可能夹带半截行或历史脏值：
    repetition/attempt 转不成整数时跳过该行，而不是让整份报告崩掉。
    """
    indexed: dict[str, dict[str, bool]] = {}
    for row in scores:
        try:
            repetition = int(row.get("repetition", 0))
            attempt = int(row.get("attempt", 1))
        except (TypeError, ValueError):
            continue
        key = run_key(str(row.get("caseId")), repetition, attempt)
        indexed.setdefault(key, {})[str(row.get("label"))] = bool(row.get("testPass"))
    return indexed


def aggregate(
    runs: Sequence[RunRecord],
    scores: Sequence[Mapping[str, Any]],
    *,
    plan: ExecutionPlan,
    settings: Any = None,
) -> dict[str, Any]:
    """聚合核心指标。

    分母恒为计划尝试数（案例数 × 重复数）：失败与缺失的尝试留在分母里并单独计数，
    绝不因缺少记录而把完成率算虚高。
    """
    expected = len(plan.cases) * plan.repetitions
    by_key = {run_key(run.case_id, run.repetition, run.attempt): run for run in runs}
    score_index = _index(scores)
    # 计划内的键集合：与记录求交，避免遗留评分行（用例已从数据集移除）
    # 混进分子，把完成率推过 1
    planned_keys = {
        run_key(case.id, repetition, 1)
        for case in plan.cases
        for repetition in range(1, plan.repetitions + 1)
    }

    passed = sum(
        1
        for key, labels in score_index.items()
        if key in by_key and key in planned_keys and labels.get("task_outcome")
    )
    missing = expected - len(by_key.keys() & planned_keys)

    per_case: dict[str, list[int]] = {}
    for case in plan.cases:
        per_case[case.id] = list(range(1, plan.repetitions + 1))

    # stableCompletionRate：所有重复都落到记录的用例占比（缺任何一次即不稳定）
    completed_cases = sum(
        1
        for case_id, repetitions in per_case.items()
        if all(run_key(case_id, repetition, 1) in by_key for repetition in repetitions)
    )
    stable_completion = completed_cases / len(plan.cases) if plan.cases else 0.0

    # fullyStableRate：所有重复都产出记录且全部评估通过的用例占比
    fully_stable = 0
    for case_id, repetitions in per_case.items():
        all_pass = True
        for repetition in repetitions:
            key = run_key(case_id, repetition, 1)
            labels = score_index.get(key)
            if key not in by_key or not labels:
                all_pass = False
                break
            if not all(labels.values()):
                all_pass = False
                break
        if all_pass:
            fully_stable += 1

    rounds = [float(run.round_count) for run in runs]
    prompt = [float(run.prompt_tokens) for run in runs]
    completion = [float(run.completion_tokens) for run in runs]
    total = [float(run.total_tokens) for run in runs]
    latency = [float(run.duration_ms) for run in runs]
    gpu_rate = getattr(settings, "gpu_hourly_cost_cny", None) if settings is not None else None
    costs = [cost_cny(model_duration_ms=run.model_duration_ms, gpu_hourly_cost_cny=gpu_rate) for run in runs]
    known_costs = [cost for cost in costs if cost is not None]

    violations = [
        {"key": key, "reason": labels}
        for key, labels in score_index.items()
        if labels.get("safety") is False or labels.get("side_effects") is False
    ]

    return {
        "plannedAttempts": expected,
        "recordedAttempts": len(by_key.keys() & planned_keys),
        "missingAttempts": missing,
        "failedAttempts": expected - passed,
        "taskCompletionRate": passed / expected if expected else 0.0,
        "stableCompletionRate": stable_completion,
        "fullyStableRate": fully_stable / len(plan.cases) if plan.cases else 0.0,
        "rounds": {**_stat_block(rounds), "max": int(max(rounds)) if rounds else 0},
        "tokens": {
            "promptMean": statistics.fmean(prompt) if prompt else 0.0,
            "completionMean": statistics.fmean(completion) if completion else 0.0,
            "totalMean": statistics.fmean(total) if total else 0.0,
        },
        "latencyMs": _stat_block(latency),
        "costCny": {
            **_stat_block(known_costs),
            "pricingEvidence": bool(known_costs),
            "pricedRuns": len(known_costs),
            "totalRuns": len(runs),
        },
        "safetyViolations": violations,
    }


SLICE_DIMENSIONS = {
    "category": "category",
    "riskLevel": "risk_level",
    "risk_level": "risk_level",
    "role": "role",
}


def slice_metrics(
    runs: Sequence[RunRecord],
    scores: Sequence[Mapping[str, Any]],
    cases_by_id: Mapping[str, Any],
    *,
    dimension: str,
) -> dict[str, Any]:
    """按 category / riskLevel / role 切片。

    dimension 同时接受 camelCase 与 snake_case：若直接 getattr 传错的命名，
    getattr 的默认值会把每条用例都归入 "unknown" 桶且不报错，静默产出废数据。
    """
    attribute = SLICE_DIMENSIONS.get(dimension)
    if attribute is None:
        raise ValueError(f"不支持的切片维度: {dimension}")
    grouped: dict[str, list[tuple[RunRecord, Mapping[str, Any] | None]]] = {}
    score_index = _index(scores)
    for run in runs:
        case = cases_by_id.get(run.case_id)
        if case is None:
            continue
        bucket = str(getattr(case, attribute))
        key = run_key(run.case_id, run.repetition, run.attempt)
        labels = score_index.get(key)
        grouped.setdefault(bucket, []).append(
            (run, {"task_outcome": labels.get("task_outcome")} if labels else None)
        )

    result: dict[str, Any] = {}
    for bucket, entries in grouped.items():
        bucket_total = len(entries)
        bucket_passed = sum(1 for _, labels in entries if labels and labels.get("task_outcome"))
        result[bucket] = {
            "count": bucket_total,
            "taskCompletionRate": bucket_passed / bucket_total if bucket_total else 0.0,
        }
    return result
