from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import run as cli
from hse_agent_evals.runner import ExecutionPlan

from _factories import make_response

CASES_DIR = Path(__file__).resolve().parents[1] / "cases"


def test_validate_returns_zero_for_approved_dataset(capsys):
    """验证 validate returns zero for approved dataset 场景。"""
    exit_code = cli.main(["validate"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["development"] == 70
    assert payload["regression"] == 30


def test_unknown_dataset_is_rejected():
    """验证 unknown dataset is rejected 场景。"""
    with pytest.raises(SystemExit):
        cli.main(["execute", "--run-id", "run-001", "--dataset", "bogus"])


def test_plan_from_arguments_selects_split():
    """验证 plan from arguments selects split 场景。"""
    plan = cli.plan_from_arguments(
        dataset="regression", repetitions=2, mode="gate", rerun_failures=True
    )
    assert isinstance(plan, ExecutionPlan)
    assert plan.mode == "gate"
    assert plan.rerun_failures is True


def test_execute_without_prepare_reports_clear_error(tmp_path, monkeypatch, capsys):
    """未 prepare 就 execute 时必须给出明确提示，而不是 100 条用例静默失败。"""
    monkeypatch.setattr(cli, "RUNTIME_DIR", tmp_path)
    exit_code = cli.main(["execute", "--run-id", "run-missing", "--repetitions", "1"])
    assert exit_code == 2
    assert "prepare" in capsys.readouterr().err


def test_execute_requires_seed_marker_when_requested(tmp_path, monkeypatch, capsys):
    """验证 execute requires seed marker when requested 场景。"""
    monkeypatch.setattr(cli, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(
        cli.environment,
        "database_name",
        lambda run_id, runtime_dir: "engine_agent_eval_x",
    )

    exit_code = cli.main(
        ["execute", "--run-id", "run-001", "--repetitions", "1", "--require-seed"]
    )

    assert exit_code == 2
    assert "seed" in capsys.readouterr().err


def test_execute_rejects_unapproved_dataset(tmp_path, monkeypatch, capsys):
    """验证 execute rejects unapproved dataset 场景。"""
    monkeypatch.setattr(cli, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(
        cli.environment,
        "database_name",
        lambda run_id, runtime_dir: "engine_agent_eval_x",
    )
    monkeypatch.setattr(cli, "_dataset_approved", lambda: False)
    (tmp_path / "run-001.seeded").write_text("seeded\n", encoding="utf-8")

    exit_code = cli.main(
        ["execute", "--run-id", "run-001", "--repetitions", "1", "--require-seed"]
    )

    assert exit_code == 2
    assert "人工复核" in capsys.readouterr().err


def test_seed_command_writes_marker(tmp_path, monkeypatch):
    """验证 seed command writes marker 场景。"""
    monkeypatch.setattr(cli, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(cli, "_settings", lambda: object())
    monkeypatch.setattr(cli, "_fixture_manager", lambda settings: object())
    monkeypatch.setattr(cli.environment, "seed", lambda *args: None)
    monkeypatch.setattr(
        cli.environment,
        "database_name",
        lambda run_id, runtime_dir: "engine_agent_eval_x",
    )

    exit_code = cli.main(["seed", "--run-id", "run-001"])

    assert exit_code == 0
    assert (tmp_path / "run-001.seeded").exists()


def test_check_binding_command_prints_result(tmp_path, monkeypatch, capsys):
    """验证 check binding command prints result 场景。"""
    monkeypatch.setattr(cli, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(cli, "_settings", lambda: object())
    monkeypatch.setattr(
        cli.environment,
        "database_name",
        lambda run_id, runtime_dir: "engine_agent_eval_x",
    )
    monkeypatch.setattr(
        cli,
        "verify_backend_binding",
        lambda run_id, settings, runtime_dir: {
            "runId": run_id,
            "database": "engine_agent_eval_x",
            "profile": "agent-eval",
        },
    )

    exit_code = cli.main(["check-binding", "--run-id", "run-001"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["database"] == "engine_agent_eval_x"


def test_main_maps_value_error_to_exit_two(tmp_path, monkeypatch, capsys):
    """配置类错误（如缺角色）应返回结构化退出码 2，而不是抛 traceback。"""
    monkeypatch.setattr(cli, "RUNTIME_DIR", tmp_path)
    def boom(run_id, runtime_dir):
        """处理 boom 相关逻辑。"""
        return "engine_agent_eval_x"
    monkeypatch.setattr(cli.environment, "database_name", boom)
    monkeypatch.setattr(cli, "_dataset_approved", lambda: True)

    def raising_runner(plan, *, run_id, settings, runtime_dir, manage_fixtures):
        """处理 raising runner 相关逻辑。"""
        raise ValueError("评测身份夹具缺少角色: ghost")

    exit_code = cli.main(["execute", "--run-id", "run-001", "--repetitions", "1"])
    assert exit_code == 2
    assert "配置错误" in capsys.readouterr().err


def test_execute_passes_manage_fixtures(tmp_path, monkeypatch):
    """验证 execute passes manage fixtures 场景。"""
    monkeypatch.setattr(cli, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(
        cli.environment,
        "database_name",
        lambda run_id, runtime_dir: "engine_agent_eval_x",
    )
    monkeypatch.setattr(cli, "_settings", lambda: object())
    monkeypatch.setattr(cli, "_dataset_approved", lambda: True)
    (tmp_path / "run-001.seeded").write_text("seeded\n", encoding="utf-8")
    seen = {}

    def fake_run_all(plan, *, run_id, settings, runtime_dir, manage_fixtures):
        """处理 fake run all 相关逻辑。"""
        seen["manage_fixtures"] = manage_fixtures
        return SimpleNamespace(exit_code=0)

    monkeypatch.setattr(cli, "run_all", fake_run_all)

    exit_code = cli.main(
        [
            "execute",
            "--run-id",
            "run-001",
            "--repetitions",
            "1",
            "--require-seed",
            "--manage-fixtures",
        ]
    )

    assert exit_code == 0
    assert seen["manage_fixtures"] is True


def _seed_run(run_id, tmp_path, cases=None):
    """写一条真实原始记录，供 report 用例使用。"""
    from hse_agent_evals.runner import build_run_record

    case = (cases or [cli.load_cases(CASES_DIR / "development.v1.jsonl")[0]])[0]
    store = cli.RawRunStore(cli.run_directory(tmp_path, run_id))
    store.append(
        build_run_record(
            run_id=run_id,
            case=case,
            repetition=1,
            attempt=1,
            response=make_response(),
            side_effects=[],
            session_id="s-1",
            owner_id="980001",
            tenant_id="dept:210",
            model="qwen2.5:7b",
            provider="local",
            duration_ms=900,
            confirmation_before_side_effects=None,
            side_effect_committed=False,
        )
    )
    store.append_scores(
        [
            {
                "caseId": case.id,
                "repetition": 1,
                "attempt": 1,
                "label": "task_outcome",
                "score": 1.0,
                "testPass": True,
                "reason": "{}",
            }
        ]
    )
    return case


def test_report_command_writes_artifacts_for_existing_run(tmp_path, monkeypatch):
    """report 应能从已有原始结果重建产物。"""
    monkeypatch.setattr(cli, "RUNTIME_DIR", tmp_path)
    case = _seed_run("run-001", tmp_path)
    monkeypatch.setattr(cli, "_cases", lambda: [case])

    exit_code = cli.main(["report", "--run-id", "run-001", "--repetitions", "1"])

    assert exit_code == 0
    assert (cli.run_directory(tmp_path, "run-001") / "report.md").exists()


def test_report_without_records_fails_loudly(tmp_path, monkeypatch, capsys):
    """run_id 打错时不得静默产出一份退出码 0 的空报告。"""
    monkeypatch.setattr(cli, "RUNTIME_DIR", tmp_path)

    exit_code = cli.main(["report", "--run-id", "typo-run", "--repetitions", "1"])

    assert exit_code == 1
    assert "未找到运行记录" in capsys.readouterr().err
    assert not (cli.run_directory(tmp_path, "typo-run") / "report.md").exists()
