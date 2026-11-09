from __future__ import annotations

import pytest

from hse_agent_evals.metrics import _index, aggregate, cost_cny, percentile, slice_metrics
from hse_agent_evals.runner import ExecutionPlan, build_run_record, run_key

from _factories import make_case, make_response


def test_percentile_uses_linear_interpolation():
    assert percentile([1, 2, 5], 50) == 2
    assert percentile([1, 2, 5], 95) == pytest.approx(4.7)
    assert percentile([], 50) is None
    assert percentile([3], 95) == 3


def test_cost_is_null_without_pricing_evidence():
    assert cost_cny(model_duration_ms=60000, gpu_hourly_cost_cny=None) is None


def test_cost_uses_gpu_hourly_rate():
    assert cost_cny(model_duration_ms=60000, gpu_hourly_cost_cny=6) == pytest.approx(0.1)


def _run(case, repetition, *, status="success", rounds=1, tokens=110):
    return build_run_record(
        run_id="run-001",
        case=case,
        repetition=repetition,
        attempt=1,
        response=make_response(status=status, roundCount=rounds, tokens=tokens),
        side_effects=[],
        session_id=f"s-{repetition}",
        owner_id="980001",
        tenant_id="dept:210",
        model="qwen2.5:7b",
        provider="local",
        duration_ms=900,
        confirmation_before_side_effects=None,
        side_effect_committed=False,
    )


def _scores(case_id, repetition, passed):
    return [
        {
            "caseId": case_id,
            "repetition": repetition,
            "attempt": 1,
            "label": "task_outcome",
            "score": 1.0 if passed else 0.0,
            "testPass": passed,
            "reason": "{}",
        }
    ]


def test_aggregate_counts_failures_in_denominator():
    case = make_case()
    runs = [_run(case, 1), _run(case, 2), _run(case, 3, status="failed")]
    scores = _scores(case.id, 1, True) + _scores(case.id, 2, True) + _scores(case.id, 3, False)
    plan = ExecutionPlan(cases=[case], repetitions=3, mode="baseline", dataset="all", rerun_failures=False)

    metrics = aggregate(runs, scores, plan=plan, settings=None)

    assert metrics["taskCompletionRate"] == pytest.approx(2 / 3)
    assert metrics["stableCompletionRate"] == 1.0  # 三次都已执行
    assert metrics["fullyStableRate"] == 0.0  # 非每次都通过
    assert metrics["rounds"]["mean"] == pytest.approx(1.0)
    assert metrics["costCny"]["mean"] is None
    assert metrics["costCny"]["pricingEvidence"] is False


def test_aggregate_keeps_missing_records_in_denominator():
    case = make_case()
    plan = ExecutionPlan(cases=[case], repetitions=3, mode="baseline", dataset="all", rerun_failures=False)

    metrics = aggregate([_run(case, 1)], _scores(case.id, 1, True), plan=plan, settings=None)

    assert metrics["taskCompletionRate"] == pytest.approx(1 / 3)
    assert metrics["stableCompletionRate"] == 0.0
    assert metrics["rounds"]["max"] == 1


def test_slice_metrics_groups_by_case_dimension():
    """切片按用例维度分组，供报告展示 category/riskLevel/role 三维。"""
    l1 = make_case()
    l2 = make_case(id="hse-l2-urgent-002", category="l2", riskLevel="high")
    runs = [_run(l1, 1), _run(l2, 1)]
    scores = _scores(l1.id, 1, True) + _scores(l2.id, 1, False)
    cases_by_id = {case.id: case for case in (l1, l2)}

    by_category = slice_metrics(runs, scores, cases_by_id, dimension="category")
    assert by_category["l1"] == {"count": 1, "taskCompletionRate": 1.0}
    assert by_category["l2"] == {"count": 1, "taskCompletionRate": 0.0}

    by_risk = slice_metrics(runs, scores, cases_by_id, dimension="risk_level")
    assert by_risk["high"] == {"count": 1, "taskCompletionRate": 0.0}

    # 未登记的用例被忽略，不产生 unknown 桶
    only_l1 = slice_metrics(runs, scores, {l1.id: l1}, dimension="category")
    assert set(only_l1) == {"l1"}


def test_percentile_clamps_out_of_range_q():
    """越界分位不应抛 IndexError —— 报告侧不该因一个配置值整体失败。"""
    assert percentile([1, 2, 5], 150) == 5
    assert percentile([1, 2, 5], -10) == 1


def test_index_skips_malformed_rows():
    """append-only JSONL 里的半截/脏行应被跳过，而不是让整份报告崩掉。"""
    rows = [
        {"caseId": "a", "repetition": None, "attempt": 1, "label": "task_outcome", "testPass": True},
        {"caseId": "a", "repetition": "x", "attempt": 1, "label": "task_outcome", "testPass": True},
        {"caseId": "a", "repetition": 1, "attempt": 1, "label": "task_outcome", "testPass": True},
    ]
    assert _index(rows) == {"a|1|1": {"task_outcome": True}}


def test_slice_metrics_rejects_unknown_dimension():
    """传错维度名必须报错，不能静默把所有用例归入 unknown 桶。"""
    case = make_case()
    with pytest.raises(ValueError):
        slice_metrics([_run(case, 1)], _scores(case.id, 1, True), {case.id: case}, dimension="bogus")


def test_slice_metrics_accepts_camel_and_snake_dimension():
    case = make_case(riskLevel="high")
    runs = [_run(case, 1)]
    scores = _scores(case.id, 1, True)
    by_id = {case.id: case}
    assert slice_metrics(runs, scores, by_id, dimension="riskLevel") == {
        "high": {"count": 1, "taskCompletionRate": 1.0}
    }
    assert slice_metrics(runs, scores, by_id, dimension="risk_level") == {
        "high": {"count": 1, "taskCompletionRate": 1.0}
    }


def test_aggregate_ignores_orphan_rows_outside_plan():
    """遗留评分行（用例已从数据集移除）不得把完成率推过 1。"""
    planned = make_case()
    removed = make_case(id="hse-l1-search-009")
    runs = [_run(planned, 1), _run(removed, 1)]
    scores = _scores(planned.id, 1, True) + _scores(removed.id, 1, True)
    plan = ExecutionPlan(cases=[planned], repetitions=1, mode="baseline", dataset="all", rerun_failures=False)

    metrics = aggregate(runs, scores, plan=plan, settings=None)

    assert metrics["taskCompletionRate"] == 1.0
    assert metrics["recordedAttempts"] == 1
    assert metrics["missingAttempts"] == 0
