from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from pydantic import ValidationError

from hse_agent_evals import environment
from hse_agent_evals.adapters.fixture_manager import FixtureManager
from hse_agent_evals.dataset import load_cases, validate_production_bundle
from hse_agent_evals.reporting import write_artifacts
from hse_agent_evals.runner import (
    ExecutionPlan,
    LeaseUnavailable,
    RawRunStore,
    plan_execution,
    run_all,
    run_directory,
    verify_backend_binding,
)
from hse_agent_evals.settings import EvalSettings

EVAL_ROOT = Path(__file__).resolve().parent
CASES_DIR = EVAL_ROOT / "cases"
RUNTIME_DIR = EVAL_ROOT / ".runtime"

def _settings() -> EvalSettings:
    return EvalSettings()

def _reporting_settings() -> EvalSettings | None:
    """report 只读已有产物，settings 仅供成本估算（metrics.aggregate 对 None 有兜底）。

    未配置数据库口令时不应因报告命令而失败：这里降级为 None，
    把配置错误留给真正需要连库的 prepare/execute/cleanup 暴露。
    """
    try:
        return EvalSettings()
    except ValidationError:  # 缺密钥时报告仍应可用
        return None

def _fixture_manager(settings: EvalSettings) -> FixtureManager:
    return environment.fixture_manager(settings)

def _cases() -> list[Any]:
    return load_cases(CASES_DIR / "development.v1.jsonl") + load_cases(
        CASES_DIR / "regression.v1.jsonl"
    )

def _dataset_approved() -> bool:
    try:
        manifest = json.loads(
            (CASES_DIR / "manifest.v1.json").read_text(encoding="utf-8-sig")
        )
        review = json.loads(
            (CASES_DIR / "review.v1.json").read_text(encoding="utf-8-sig")
        )
    except (OSError, ValueError):
        return False
    accepted = review.get("acceptedCaseIds")
    if not isinstance(accepted, list):
        return False
    case_ids = {case.id for case in _cases()}
    return (
        manifest.get("approved") is True
        and review.get("decision") == "accepted"
        and bool(review.get("reviewers"))
        and set(accepted) == case_ids
    )

def plan_from_arguments(
    *, dataset: str, repetitions: int, mode: str, rerun_failures: bool
) -> ExecutionPlan:
    return plan_execution(
        _cases(),
        dataset=dataset,
        repetitions=repetitions,
        mode=mode,
        rerun_failures=rerun_failures,
    )

def _run_dir(run_id: str) -> Path:
    return run_directory(RUNTIME_DIR, run_id)

def _seed_marker(run_id: str) -> Path:
    return RUNTIME_DIR / f"{run_id}.seeded"

def validate_command(_: argparse.Namespace) -> int:
    """validate_production_bundle 返回错误字符串列表，空列表即通过。"""
    errors = validate_production_bundle(
        CASES_DIR / "development.v1.jsonl",
        CASES_DIR / "regression.v1.jsonl",
        CASES_DIR / "manifest.v1.json",
        CASES_DIR / "review.v1.json",
    )
    payload = {
        "valid": not errors,
        "errors": errors,
        "development": len(load_cases(CASES_DIR / "development.v1.jsonl")),
        "regression": len(load_cases(CASES_DIR / "regression.v1.jsonl")),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if errors:
        print("数据集校验失败", file=sys.stderr)
        return 1
    return 0

def prepare_command(arguments: argparse.Namespace) -> int:
    settings = _settings()
    manager = _fixture_manager(settings)
    result = environment.prepare(arguments.run_id, settings, manager, RUNTIME_DIR)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0

def seed_command(arguments: argparse.Namespace) -> int:
    try:
        environment.database_name(arguments.run_id, RUNTIME_DIR)
    except FileNotFoundError:
        print(
            f"未找到运行状态，请先执行 prepare --run-id {arguments.run_id}",
            file=sys.stderr,
        )
        return 2
    settings = _settings()
    manager = _fixture_manager(settings)
    environment.seed(arguments.run_id, manager, RUNTIME_DIR)
    marker = _seed_marker(arguments.run_id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        datetime.now(timezone.utc).isoformat() + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"seeded": True, "runId": arguments.run_id}, ensure_ascii=False))
    return 0

def check_binding_command(arguments: argparse.Namespace) -> int:
    try:
        environment.database_name(arguments.run_id, RUNTIME_DIR)
    except FileNotFoundError:
        print(
            f"未找到运行状态，请先执行 prepare --run-id {arguments.run_id}",
            file=sys.stderr,
        )
        return 2
    settings = _settings()
    result = verify_backend_binding(arguments.run_id, settings, RUNTIME_DIR)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0

def execute_command(arguments: argparse.Namespace) -> int:
    # run_all 不会自动 prepare：缺状态时每条用例都会失败并被吞成
    # 「failed=N，零记录，退出码 1」，排查起来毫无线索。这里先明确报错。
    # 检查先于构造 EvalSettings：缺运行状态时不该再要求运行时密钥齐备。
    try:
        environment.database_name(arguments.run_id, RUNTIME_DIR)
    except FileNotFoundError:
        print(
            f"未找到运行状态，请先执行 prepare --run-id {arguments.run_id}",
            file=sys.stderr,
        )
        return 2
    if getattr(arguments, "require_seed", False) and not _seed_marker(
        arguments.run_id
    ).exists():
        print(
            f"未找到 seed 标记，请先执行 seed --run-id {arguments.run_id}",
            file=sys.stderr,
        )
        return 2
    if not _dataset_approved():
        print(
            "数据集尚未通过人工复核（manifest approved / review accepted），请先完成 T06 或恢复已批准版本",
            file=sys.stderr,
        )
        return 2
    settings = _settings()
    plan = plan_from_arguments(
        dataset=arguments.dataset,
        repetitions=arguments.repetitions,
        mode=arguments.mode,
        rerun_failures=arguments.rerun_failures,
    )
    summary = run_all(
        plan,
        run_id=arguments.run_id,
        settings=settings,
        runtime_dir=RUNTIME_DIR,
        manage_fixtures=arguments.manage_fixtures,
    )
    print(json.dumps(summary.__dict__, ensure_ascii=False, sort_keys=True))
    return summary.exit_code

def report_command(arguments: argparse.Namespace) -> int:
    settings = _reporting_settings()
    store = RawRunStore(_run_dir(arguments.run_id))
    runs = store.load()
    # 没有原始记录就写报告，会把「run_id 打错」变成一份退出码 0 的空报告 —— 静默假成功
    if not runs:
        print(
            f"未找到运行记录，请确认 --run-id {arguments.run_id} 是否正确（或先执行 execute）",
            file=sys.stderr,
        )
        return 1
    plan = plan_from_arguments(
        dataset=arguments.dataset,
        repetitions=arguments.repetitions,
        mode=arguments.mode,
        rerun_failures=False,
    )
    bundle = write_artifacts(
        _run_dir(arguments.run_id),
        plan=plan,
        runs=runs,
        scores=store.load_scores(),
        settings=settings,
        mode=arguments.mode,
    )
    print(json.dumps({key: str(value) for key, value in bundle.items()}, ensure_ascii=False, sort_keys=True))
    return 0

def cleanup_command(arguments: argparse.Namespace) -> int:
    settings = _settings()
    environment.cleanup(arguments.run_id, _fixture_manager(settings), RUNTIME_DIR)
    marker = _seed_marker(arguments.run_id)
    if marker.exists():
        marker.unlink()
    return 0

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hse-agent-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate")

    for name in ("prepare", "seed", "check-binding", "cleanup"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--run-id", required=True)

    execute = subparsers.add_parser("execute")
    execute.add_argument("--run-id", required=True)
    execute.add_argument("--dataset", choices=("development", "regression", "all"), default="all")
    execute.add_argument("--repetitions", type=int, default=3)
    execute.add_argument("--mode", choices=("baseline", "gate"), default="baseline")
    execute.add_argument("--rerun-failures", action="store_true")
    execute.add_argument("--require-seed", action="store_true")
    execute.add_argument("--manage-fixtures", action="store_true")

    report = subparsers.add_parser("report")
    report.add_argument("--run-id", required=True)
    report.add_argument("--dataset", choices=("development", "regression", "all"), default="all")
    report.add_argument("--repetitions", type=int, default=3)
    report.add_argument("--mode", choices=("baseline", "gate"), default="baseline")

    return parser

def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    handlers = {
        "validate": validate_command,
        "prepare": prepare_command,
        "seed": seed_command,
        "check-binding": check_binding_command,
        "execute": execute_command,
        "report": report_command,
        "cleanup": cleanup_command,
    }
    try:
        return handlers[arguments.command](arguments)
    except LeaseUnavailable as error:
        # 并发评测是环境问题而非配置问题，用独立退出码便于编排脚本区分
        print(f"评测正在被其他进程使用: {error}", file=sys.stderr)
        return 3
    except RuntimeError as error:
        # 隔离前提不成立（如完整性检查连不上库）：给出退出码而非 traceback
        print(f"评测环境不可用: {error}", file=sys.stderr)
        return 3
    except ValueError as error:
        # 配置类问题（缺角色、维度非法等）应给出结构化退出码，
        # 而不是把 traceback 抛给调用方
        print(f"配置错误: {error}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
