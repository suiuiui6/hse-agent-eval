from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from hse_agent_evals.metrics import aggregate, slice_metrics
from hse_agent_evals.models import RunRecord
from hse_agent_evals.runner import ExecutionPlan

REVIEW_MINIMUM = 20


def _write_atomic(path: Path, content: str) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temp, path)


def _dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _stable_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )


def _num(value: Any, digits: int = 2) -> str:
    """空运行会把统计量算成 None，格式化前必须兜住，否则报告直接崩。"""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def select_review_sample(
    plan: ExecutionPlan,
    scores: Sequence[Mapping[str, Any]],
    cases_by_id: Mapping[str, Any],
    *,
    minimum: int = REVIEW_MINIMUM,
) -> list[dict[str, Any]]:
    """按类别覆盖抽样，再补足到 minimum。

    仅按 case_id 排序截断会严重偏斜：实测前 20 条全部落在
    boundary/failure/l1，l2、l3、multi_tool、no_tool、security 一个都抽不到，
    人工复核的价值随之失效。故按类别轮转取用，首轮即覆盖全部类别。

    抽样面向**全部输出**而非仅通过的输出：人工复核要看的恰恰是模型的真实表现，
    若只抽通过项，一旦模型成片失败就会得到空样本（实测 300 条全失败时样本为 0），
    复核反而在最需要它的场景下失效。
    """
    outcomes = [row for row in scores if row.get("label") == "task_outcome"]

    def _order(row: Mapping[str, Any]) -> tuple[str, int]:
        try:
            repetition = int(row.get("repetition", 0))
        except (TypeError, ValueError):
            repetition = 0
        return (str(row.get("caseId")), repetition)

    ordered = sorted(outcomes, key=_order)

    # 每个 case 只取一次，保留其最早的一次重复
    by_case: dict[str, dict[str, Any]] = {}
    for row in ordered:
        by_case.setdefault(str(row.get("caseId")), {**dict(row), "rubricRequired": True})

    # 按类别分组，组内按 case_id 稳定排序
    by_category: dict[str, list[dict[str, Any]]] = {}
    for case_id, row in sorted(by_case.items()):
        case = cases_by_id.get(case_id)
        category = str(getattr(case, "category", "unknown")) if case is not None else "unknown"
        by_category.setdefault(category, []).append(row)

    # 轮转取用：每类每轮最多取一条，直到达到 minimum 或无候选
    # —— 首轮即覆盖全部类别，后续轮保持各类别数量均衡，避免主导类别挤掉弱类别
    sample: list[dict[str, Any]] = []
    depth = 0
    while len(sample) < minimum:
        progressed = False
        for category in sorted(by_category):
            candidates = by_category[category]
            if depth < len(candidates):
                sample.append(candidates[depth])
                progressed = True
                if len(sample) >= minimum:
                    break
        if not progressed:
            break
        depth += 1
    return sample


def write_artifacts(
    run_dir: str | Path,
    *,
    plan: ExecutionPlan,
    runs: Sequence[RunRecord],
    scores: Sequence[Mapping[str, Any]],
    settings: Any,
    dataset_checksums: Mapping[str, str] | None = None,
    mode: str = "baseline",
) -> dict[str, Path]:
    target = Path(run_dir)
    target.mkdir(parents=True, exist_ok=True)
    metrics = aggregate(runs, scores, plan=plan, settings=settings)

    # 产品契约要求按 类别/风险/角色 切片，否则切片指标无人消费
    cases_by_id = {case.id: case for case in plan.cases}
    metrics["slices"] = {
        "category": slice_metrics(runs, scores, cases_by_id, dimension="category"),
        "riskLevel": slice_metrics(runs, scores, cases_by_id, dimension="risk_level"),
        "role": slice_metrics(runs, scores, cases_by_id, dimension="role"),
    }

    failures = [dict(row) for row in scores if row.get("testPass") is False]
    manifest = {
        "runId": runs[0].run_id if runs else "",
        "mode": mode,
        "dataset": plan.dataset,
        "repetitions": plan.repetitions,
        "plannedAttempts": len(plan.cases) * plan.repetitions,
        "recordedAttempts": metrics["recordedAttempts"],
        "datasetChecksums": dict(dataset_checksums or {}),
        "evaluators": sorted({str(row.get("label")) for row in scores}),
    }
    sample = select_review_sample(plan, scores, cases_by_id)

    written = {
        "manifest": target / "manifest.json",
        "metrics": target / "metrics.json",
        "failures": target / "failures.json",
        "case_scores": target / "case-scores.jsonl",
        "review_sample": target / "human-review-sample.jsonl",
        "report": target / "report.md",
    }
    _write_atomic(written["manifest"], _dump(manifest))
    _write_atomic(written["metrics"], _dump(metrics))
    _write_atomic(written["failures"], _dump(failures))
    _write_atomic(written["case_scores"], _stable_jsonl(scores))
    _write_atomic(written["review_sample"], _stable_jsonl(sample))
    _write_atomic(written["report"], render_report(manifest, metrics, failures))
    return written


def render_report(
    manifest: Mapping[str, Any], metrics: Mapping[str, Any], failures: Sequence[Mapping[str, Any]]
) -> str:
    rounds = metrics.get("rounds", {})
    cost = metrics.get("costCny", {})
    lines = [
        "# HSE Agent 评测报告",
        "",
        f"- 运行标识: {manifest.get('runId')}",
        f"- 模式: {manifest.get('mode')}",
        f"- 数据集: {manifest.get('dataset')}",
        f"- 计划尝试: {manifest.get('plannedAttempts')} / 已记录: {manifest.get('recordedAttempts')}",
        "",
        "## 核心指标",
        "",
        f"- 任务完成率: {_num(metrics.get('taskCompletionRate'), 4)}",
        f"- 稳定完成率: {_num(metrics.get('stableCompletionRate'), 4)}",
        f"- 全稳定率: {_num(metrics.get('fullyStableRate'), 4)}",
        f"- 轮数 均值/P50/P95/最大: "
        f"{_num(rounds.get('mean'))} / {_num(rounds.get('p50'))} / "
        f"{_num(rounds.get('p95'))} / {rounds.get('max')}",
        f"- 成本(CNY) 均值: {_num(cost.get('mean'))}",
        f"- 成本定价证据: {cost.get('pricingEvidence')}（{cost.get('pricedRuns')}/{cost.get('totalRuns')} 次有定价）",
        "",
        "## 零容忍",
        "",
        f"- 安全/副作用违规数: {len(metrics.get('safetyViolations', []))}",
        f"- 失败尝试数: {metrics.get('failedAttempts')}",
    ]
    if failures:
        lines.extend(["", "## 失败样例", ""])
        for row in list(failures)[:20]:
            lines.append(f"- {row.get('caseId')} r{row.get('repetition')} [{row.get('label')}]")
    return "\n".join(lines) + "\n"
