from __future__ import annotations

import json

from hse_agent_evals.reporting import write_artifacts
from hse_agent_evals.runner import ExecutionPlan, build_run_record

from _factories import make_case, make_response


def _run(case, repetition):
    return build_run_record(
        run_id="run-001",
        case=case,
        repetition=repetition,
        attempt=1,
        response=make_response(),
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


def test_write_artifacts_emits_all_expected_files(tmp_path):
    cases = [make_case(id=f"hse-l1-search-{index:03d}") for index in range(1, 26)]
    runs = [_run(case, repetition) for case in cases for repetition in (1, 2)]
    scores = [
        {
            "caseId": run.case_id,
            "repetition": run.repetition,
            "attempt": 1,
            "label": "task_outcome",
            "score": 1.0,
            "testPass": True,
            "reason": "{}",
        }
        for run in runs
    ]
    plan = ExecutionPlan(cases=cases, repetitions=2, mode="baseline", dataset="all", rerun_failures=False)

    bundle = write_artifacts(
        tmp_path,
        plan=plan,
        runs=runs,
        scores=scores,
        settings=None,
        dataset_checksums={"development": "abc"},
        mode="baseline",
    )

    for name in ("manifest.json", "metrics.json", "failures.json", "report.md", "human-review-sample.jsonl"):
        assert (tmp_path / name).exists(), name
    assert (tmp_path / "case-scores.jsonl").exists()

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "baseline"
    assert manifest["recordedAttempts"] == 50

    sample_lines = (tmp_path / "human-review-sample.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(sample_lines) >= 20

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["taskCompletionRate"] == 1.0
    # 切片必须落到产物里（产品契约要求 category/risk/role 三维）
    assert set(metrics["slices"]) == {"category", "riskLevel", "role"}
    assert metrics["slices"]["category"]["l1"]["count"] == 50


def test_write_artifacts_survives_an_empty_run(tmp_path):
    """空运行（零记录）时统计量为 None，报告必须仍能生成而不是崩掉。"""
    case = make_case()
    plan = ExecutionPlan(cases=[case], repetitions=1, mode="baseline", dataset="all", rerun_failures=False)

    write_artifacts(tmp_path, plan=plan, runs=[], scores=[], settings=None, mode="baseline")

    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "n/a" in report
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["taskCompletionRate"] == 0.0
    assert metrics["rounds"]["mean"] is None

    # 有记录但无定价时 cost.mean 为 None，报告里必须是 n/a 而不是字面量 None
    run = _run(case, 1)
    score = {"caseId": case.id, "repetition": 1, "attempt": 1, "label": "task_outcome",
             "score": 1.0, "testPass": True, "reason": "{}"}
    write_artifacts(tmp_path / "priced", plan=plan, runs=[run], scores=[score], settings=None, mode="baseline")
    priced_report = (tmp_path / "priced" / "report.md").read_text(encoding="utf-8")
    assert "成本(CNY) 均值: n/a" in priced_report
    assert "None" not in priced_report


def test_review_sample_covers_every_category(tmp_path):
    """人工抽检必须覆盖全部类别。

    回归：仅按 case_id 排序截断时，前 20 条全部落在 boundary/failure/l1，
    l2/multi_tool/security 等一条都抽不到，复核样本失去代表性。
    """
    categories = ["no_tool", "l1", "l2", "multi_tool", "security"]
    cases = [
        make_case(id=f"hse-l1-search-{index:03d}", category=categories[index % len(categories)])
        for index in range(1, 41)
    ]
    runs = [_run(case, 1) for case in cases]
    scores = [
        {
            "caseId": run.case_id,
            "repetition": 1,
            "attempt": 1,
            "label": "task_outcome",
            "score": 1.0,
            "testPass": True,
            "reason": "{}",
        }
        for run in runs
    ]
    plan = ExecutionPlan(cases=cases, repetitions=1, mode="baseline", dataset="all", rerun_failures=False)

    write_artifacts(tmp_path, plan=plan, runs=runs, scores=scores, settings=None, mode="baseline")

    sample = [
        json.loads(line)
        for line in (tmp_path / "human-review-sample.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ]
    sampled_ids = {row["caseId"] for row in sample}
    sampled_categories = {case.category for case in cases if case.id in sampled_ids}
    assert sampled_categories == set(categories)
    assert len(sample) >= 20


def test_review_sample_stratifies_real_id_scheme(tmp_path):
    """回归钉：真实数据集按「类别前缀」命名，排序截断会漏掉一半类别。

    真实金标 100 例的前 20 条（按 case_id 排序）全部落在 boundary/failure/l1，
    l2/l3/multi_tool/no_tool/security 一条都抽不到。上面的用例因 id 前缀统一
    且类别均匀轮转，碰巧掩盖了这个缺陷，故此处复刻真实命名与分布。
    """
    distribution = {
        "no-tool": 10,
        "l1": 25,
        "l2": 15,
        "l3": 15,
        "multi-tool": 15,
        "boundary": 10,
        "security": 5,
        "failure": 5,
    }
    cases = [
        make_case(id=f"hse-{prefix}-{index:03d}", category=category)
        for prefix, count in distribution.items()
        for category, index in [(prefix.replace("-", "_"), position) for position in range(1, count + 1)]
    ]
    # 断言夹具确实复刻了偏斜：按排序截断只能覆盖 3 个类别
    truncated = {case.category for case in sorted(cases, key=lambda c: c.id)[:20]}
    assert len(truncated) < len(distribution), "夹具未复刻出偏斜，回归钉失效"

    runs = [_run(case, 1) for case in cases]
    scores = [
        {
            "caseId": run.case_id,
            "repetition": 1,
            "attempt": 1,
            "label": "task_outcome",
            "score": 1.0,
            "testPass": True,
            "reason": "{}",
        }
        for run in runs
    ]
    plan = ExecutionPlan(cases=cases, repetitions=1, mode="baseline", dataset="all", rerun_failures=False)

    write_artifacts(tmp_path, plan=plan, runs=runs, scores=scores, settings=None, mode="baseline")

    sample = [
        json.loads(line)
        for line in (tmp_path / "human-review-sample.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ]
    sampled_ids = {row["caseId"] for row in sample}
    sampled_categories = {case.category for case in cases if case.id in sampled_ids}
    assert sampled_categories == {case.category for case in cases}
    assert len(sample) >= 20


def test_review_sample_is_not_empty_when_everything_fails(tmp_path):
    """模型成片失败时人工抽检样本不能为空。

    回归：抽样原先只在「通过」的行里取，一旦全部失败样本就是 0 条，
    复核恰恰在最需要它的场景失效。
    """
    cases = [make_case(id=f"hse-l1-search-{index:03d}") for index in range(1, 26)]
    runs = [_run(case, 1) for case in cases]
    failing = [
        {
            "caseId": run.case_id,
            "repetition": 1,
            "attempt": 1,
            "label": "task_outcome",
            "score": 0.0,
            "testPass": False,
            "reason": "{}",
        }
        for run in runs
    ]
    plan = ExecutionPlan(cases=cases, repetitions=1, mode="baseline", dataset="all", rerun_failures=False)

    write_artifacts(tmp_path, plan=plan, runs=runs, scores=failing, settings=None, mode="baseline")

    sample = (tmp_path / "human-review-sample.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(sample) >= 20
