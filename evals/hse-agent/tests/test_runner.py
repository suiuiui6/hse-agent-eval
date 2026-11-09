from __future__ import annotations

import json

import pytest

from hse_agent_evals.models import SideEffect
from hse_agent_evals.runner import (
    RawRunStore,
    _apply_case_failure_mode,
    _merge_confirmation_response,
    build_run_record,
    synthesize_finish_reasons,
)

from _factories import make_case, make_confirm_response, make_response


def test_synthesize_finish_reasons_from_response_fields():
    """验证 synthesize finish reasons from response fields 场景。"""
    assert synthesize_finish_reasons(make_response()) == ["completed"]


def test_synthesize_finish_reasons_marks_error_and_confirmation():
    """验证 synthesize finish reasons marks error and confirmation 场景。"""
    response = make_response(executionStatus="failed", confirmationToken="tok-1")
    assert synthesize_finish_reasons(response) == ["completed", "error", "confirmation"]


def test_build_run_record_maps_response_and_marks_telemetry(tmp_path):
    """验证 build run record maps response and marks telemetry 场景。"""
    record = build_run_record(
        run_id="run-001",
        case=make_case(),
        repetition=1,
        attempt=1,
        response=make_response(),
        side_effects=[],
        session_id="session-001",
        owner_id="7",
        tenant_id="dept:9",
        model="qwen2.5:7b",
        provider="local",
        duration_ms=900,
        confirmation_before_side_effects=None,
        side_effect_committed=False,
    )

    assert record.case_id == "hse-l1-search-001"
    assert record.trace_id == "trace-001"
    assert record.turn_id == "turn-001"
    assert record.owner_id == "7"
    assert record.tenant_id == "dept:9"
    assert record.finish_reasons == ["completed"]
    assert record.prompt_tokens == 80
    assert record.total_tokens == 110
    # 后端只发整轮耗时：如实照搬，不编造拆分
    assert record.model_duration_ms == 900
    assert record.tool_duration_ms == 0
    assert record.tool_durations_ms == {}
    assert record.trajectory_incomplete is False
    assert [call.tool for call in record.tool_calls] == ["search_regulations"]


def test_build_run_record_falls_back_to_execution_status_and_round_count():
    """验证 build run record falls back to execution status and round count 场景。"""
    response = make_response()
    response.pop("status")
    response.pop("roundCount")
    response["executionStatus"] = "success"

    record = build_run_record(
        run_id="run-001",
        case=make_case(),
        repetition=1,
        attempt=1,
        response=response,
        side_effects=[],
        session_id="session-001",
        owner_id="7",
        tenant_id="dept:9",
        model="qwen2.5:7b",
        provider="local",
        duration_ms=900,
        confirmation_before_side_effects=None,
        side_effect_committed=False,
    )

    assert record.status == "success"
    assert record.execution_status == "success"
    assert record.round_count == 1


def test_build_run_record_preserves_explicit_zero_round_count():
    """验证 build run record preserves explicit zero round count 场景。"""
    response = make_response(
        status="error",
        executionStatus="error",
        stopReason="provider_unavailable",
        reply="AI 服务暂时不可用",
        toolCalls=[],
        roundCount=0,
    )

    record = build_run_record(
        run_id="run-001",
        case=make_case(),
        repetition=1,
        attempt=1,
        response=response,
        side_effects=[],
        session_id="session-001",
        owner_id="7",
        tenant_id="dept:9",
        model="qwen2.5:7b",
        provider="local",
        duration_ms=900,
        confirmation_before_side_effects=None,
        side_effect_committed=False,
    )

    assert record.round_count == 0


def test_merge_confirmation_response_replaces_pending_call():
    """验证 merge confirmation response replaces pending call 场景。"""
    initial = make_response(
        status="pending_confirmation",
        executionStatus="pending",
        roundCount=1,
        confirmationToken="tok-1",
        confirmation={
            "tool": "create_task",
            "level": "L3",
            "args": {"title": "整改任务"},
        },
        toolCalls=[
            {
                "tool": "create_task",
                "toolCallId": "call-1",
                "args": {"title": "整改任务"},
                "level": "L3",
                "status": "pending",
                "roundIndex": 1,
                "result": '{"status":"pending","tool":"create_task","data":{}}',
            }
        ],
    )

    merged = _merge_confirmation_response(
        initial,
        make_confirm_response(),
        tool_name="create_task",
        args={"title": "整改任务"},
        level="L3",
    )

    assert merged["status"] == "success"
    assert merged["confirmationState"] == "confirmed"
    assert merged["roundCount"] == 1
    assert "高处作业应系安全带" in merged["reply"]
    assert "操作已确认并执行" in merged["reply"]
    assert "confirmationToken" not in merged
    assert len(merged["toolCalls"]) == 1
    assert merged["toolCalls"][0]["tool"] == "create_task"
    assert merged["toolCalls"][0]["roundIndex"] == 2


def test_merge_confirmation_response_preserves_cancelled_status():
    """验证 merge confirmation response preserves cancelled status 场景。"""
    initial = make_response(
        status="pending_confirmation",
        executionStatus="pending",
        roundCount=1,
        confirmationToken="tok-1",
        toolCalls=[],
    )

    merged = _merge_confirmation_response(
        initial,
        make_confirm_response(ok=False, status="cancelled"),
        tool_name="create_task",
        args={},
        level="L3",
    )

    assert merged["status"] == "cancelled"
    assert merged["confirmationState"] == "cancelled"
    assert merged["stopReason"] == "cancelled"


def test_verify_backend_binding_rejects_wrong_database(monkeypatch, tmp_path):
    """验证 verify backend binding rejects wrong database 场景。"""
    from hse_agent_evals import runner as runner_module

    identity_path = tmp_path / "users.json"
    identity_path.write_text(
        json.dumps(
            {
                "system_admin": {
                    "username": "admin",
                    "password": "admin123",
                    "user_id": 1,
                    "dept_id": 103,
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "run-001.json").write_text(
        json.dumps(
            {
                "run_id": "run-001",
                "database_name": "engine_agent_eval_expected",
            }
        ),
        encoding="utf-8",
    )

    class FakeClient:
        """处理 FakeClient 相关逻辑。"""
        def __init__(self, **kwargs):
            """初始化实例。"""
            pass

        def __enter__(self):
            """处理   enter   相关逻辑。"""
            return self

        def __exit__(self, *args):
            """处理   exit   相关逻辑。"""
            return False

        def login(self, username, password):
            """处理 login 相关逻辑。"""
            return "token"

        def binding(self, token):
            """处理 binding 相关逻辑。"""
            return {"database": "engine_agent_eval_wrong", "profile": "agent-eval"}

    monkeypatch.setattr(runner_module, "IDENTITIES_PATH", identity_path)
    monkeypatch.setattr(runner_module, "EngineClient", FakeClient)

    with pytest.raises(RuntimeError, match="未绑定"):
        runner_module.verify_backend_binding("run-001", make_settings(), tmp_path)


def test_verify_backend_binding_returns_selected_database(monkeypatch, tmp_path):
    """验证 verify backend binding returns selected database 场景。"""
    from hse_agent_evals import runner as runner_module

    identity_path = tmp_path / "users.json"
    identity_path.write_text(
        json.dumps(
            {
                "system_admin": {
                    "username": "admin",
                    "password": "admin123",
                    "user_id": 1,
                    "dept_id": 103,
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "run-001.json").write_text(
        json.dumps(
            {
                "run_id": "run-001",
                "database_name": "engine_agent_eval_expected",
            }
        ),
        encoding="utf-8",
    )

    class FakeClient:
        """处理 FakeClient 相关逻辑。"""
        def __init__(self, **kwargs):
            """初始化实例。"""
            pass

        def __enter__(self):
            """处理   enter   相关逻辑。"""
            return self

        def __exit__(self, *args):
            """处理   exit   相关逻辑。"""
            return False

        def login(self, username, password):
            """处理 login 相关逻辑。"""
            return "token"

        def binding(self, token):
            """处理 binding 相关逻辑。"""
            return {
                "database": "engine_agent_eval_expected",
                "profile": "agent-eval",
            }

    monkeypatch.setattr(runner_module, "IDENTITIES_PATH", identity_path)
    monkeypatch.setattr(runner_module, "EngineClient", FakeClient)

    result = runner_module.verify_backend_binding(
        "run-001", make_settings(), tmp_path
    )

    assert result == {
        "runId": "run-001",
        "database": "engine_agent_eval_expected",
        "profile": "agent-eval",
    }


def test_build_run_record_marks_incomplete_when_stop_reason_absent():
    """验证 build run record marks incomplete when stop reason absent 场景。"""
    record = build_run_record(
        run_id="run-001",
        case=make_case(),
        repetition=1,
        attempt=1,
        response=make_response(stopReason=""),
        side_effects=[],
        session_id="session-001",
        owner_id="7",
        tenant_id="dept:9",
        model="qwen2.5:7b",
        provider="local",
        duration_ms=900,
        confirmation_before_side_effects=None,
        side_effect_committed=False,
    )

    assert record.finish_reasons == []
    assert record.trajectory_incomplete is True


def test_build_run_record_marks_incomplete_without_identity():
    """验证 build run record marks incomplete without identity 场景。"""
    record = build_run_record(
        run_id="run-001",
        case=make_case(),
        repetition=1,
        attempt=1,
        response=make_response(),
        side_effects=[],
        session_id="session-001",
        owner_id="",
        tenant_id="",
        model="qwen2.5:7b",
        provider="local",
        duration_ms=900,
        confirmation_before_side_effects=None,
        side_effect_committed=False,
    )

    assert record.trajectory_incomplete is True


def test_raw_store_appends_and_resumes_without_overwrite(tmp_path):
    """验证 raw store appends and resumes without overwrite 场景。"""
    store = RawRunStore(tmp_path / "run-001")
    record = build_run_record(
        run_id="run-001",
        case=make_case(),
        repetition=1,
        attempt=1,
        response=make_response(),
        side_effects=[SideEffect(key="hse_report_task.insert", before_count=0, after_count=1, changed_rows=[])],
        session_id="session-001",
        owner_id="7",
        tenant_id="dept:9",
        model="qwen2.5:7b",
        provider="local",
        duration_ms=900,
        confirmation_before_side_effects=None,
        side_effect_committed=True,
    )

    store.append(record)
    assert store.existing_keys() == {"hse-l1-search-001|1|1"}

    loaded = store.load()
    assert len(loaded) == 1
    assert loaded[0].total_tokens == 110
    assert loaded[0].side_effects[0].key == "hse_report_task.insert"

    # 同一键再次 append 必须被拒绝：放重复行会让该次尝试在指标里被计两次，
    # 也会让「同一份原始结果」不可复现（append-only 但不允许同键重复）
    with pytest.raises(ValueError, match="拒绝覆盖"):
        store.append(record)
    assert len(store.load()) == 1


def test_raw_store_lease_excludes_concurrent_writer(tmp_path):
    """两个进程同时写同一 run 会交错出不可复现结果，第二个必须被挡下。"""
    from hse_agent_evals.runner import LeaseUnavailable

    store = RawRunStore(tmp_path / "run-001")
    with store.lease():
        with pytest.raises(LeaseUnavailable):
            with store.lease(timeout_seconds=0.05):
                pass


def test_raw_store_lease_is_reusable_after_release(tmp_path):
    """验证 raw store lease is reusable after release 场景。"""
    store = RawRunStore(tmp_path / "run-001")
    with store.lease():
        pass
    # 释放后应能再次获取，否则续跑会被永久堵死
    with store.lease():
        pass


def test_raw_store_tolerates_torn_last_line(tmp_path):
    """验证 raw store tolerates torn last line 场景。"""
    run_dir = tmp_path / "run-001"
    store = RawRunStore(run_dir)
    store.append(
        build_run_record(
            run_id="run-001",
            case=make_case(),
            repetition=1,
            attempt=1,
            response=make_response(),
            side_effects=[],
            session_id="session-001",
            owner_id="7",
            tenant_id="dept:9",
            model="qwen2.5:7b",
            provider="local",
            duration_ms=900,
            confirmation_before_side_effects=None,
            side_effect_committed=False,
        )
    )
    with store.runs_path.open("a", encoding="utf-8") as handle:
        handle.write('{"caseId": "hse-l1-search-001", "repetition"')

    assert len(store.load()) == 1


from hse_agent_evals.runner import RunOutcome, run_case

from _factories import make_settings


class FakeFixtureManager:
    """只实现 runner 需要的三个方法；不做任何数据库操作。"""

    def __init__(self, effects=None, fail_snapshot=False):
        """初始化实例。"""
        self.effects = effects or []
        self.fail_snapshot = fail_snapshot
        self.calls: list[str] = []

    def snapshot_before(self, database_name):
        """处理 snapshot before 相关逻辑。"""
        self.calls.append("snapshot_before")
        return {"hse_report_task.insert": []}

    def snapshot_after(self, database_name):
        """处理 snapshot after 相关逻辑。"""
        self.calls.append("snapshot_after")
        return {"hse_report_task.insert": [{"task_id": 980001}]}

    def diff(self, before, after):
        """处理 diff 相关逻辑。"""
        self.calls.append("diff")
        return list(self.effects)


class FakeEngineClient:
    """处理 FakeEngineClient 相关逻辑。"""
    def __init__(self, response=None, confirm_response=None, cancel_response=None):
        """初始化实例。"""
        self.response = response or make_response()
        self.confirm_response = confirm_response
        self.cancel_response = cancel_response
        self.confirm_calls = 0
        self.cancel_calls = 0
        self.chat_messages: list[str] = []

    def chat(self, token, message, session_id, idempotency_key=None):
        """处理 chat 相关逻辑。"""
        self.chat_messages.append(message)
        payload = self.response(message) if callable(self.response) else self.response
        return dict(payload)

    def confirm(self, token, confirmation_token, tool_name, args, session_id, turn_id):
        """处理 confirm 相关逻辑。"""
        self.confirm_calls += 1
        return dict(self.confirm_response or self.response)

    def cancel(self, token, confirmation_token, tool_name, args, session_id, turn_id):
        """处理 cancel 相关逻辑。"""
        self.cancel_calls += 1
        return dict(
            self.cancel_response
            or make_confirm_response(ok=False, status="cancelled")
        )


def _stub_environment(monkeypatch):
    """stub environment 的内部辅助逻辑。"""
    from hse_agent_evals import runner as runner_module

    monkeypatch.setattr(runner_module.environment, "database_name", lambda run_id, runtime_dir: "engine_agent_eval_x")
    monkeypatch.setattr(runner_module.environment, "reset_case", lambda run_id, fm, runtime_dir: None)
    # run_all 会做后端绑定复验与完整性哈希（需真实运行状态与库）；
    # 编排类单测只验证循环/续跑/退出码，故替换为通行桩。
    monkeypatch.setattr(
        runner_module.environment,
        "run_state",
        lambda run_id, runtime_dir: {"run_id": run_id, "database_name": "engine_agent_eval_x"},
    )
    monkeypatch.setattr(runner_module, "BackendBindingGuard", lambda state, settings: _PassThrough())
    monkeypatch.setattr(runner_module, "MySqlIntegrityGuard", lambda fm, name: _PassThrough())


class _PassThrough:
    """PassThrough 的内部辅助逻辑。"""
    def validate(self, run_id):
        """处理 validate 相关逻辑。"""
        return None

    def immutable_hash(self, run_id):
        """处理 immutable hash 相关逻辑。"""
        return "stub-hash"


class _ModeProxy:
    """ModeProxy 的内部辅助逻辑。"""
    def __init__(self):
        """初始化实例。"""
        self.modes: list[str] = []

    def set_mode(self, mode):
        """处理 set mode 相关逻辑。"""
        self.modes.append(mode)


def test_apply_case_failure_mode_sets_model_and_clears_graphrag():
    """验证 apply case failure mode sets model and clears graphrag 场景。"""
    model = _ModeProxy()
    graphrag = _ModeProxy()

    _apply_case_failure_mode(
        make_case(failureMode="model_500"),
        model,
        graphrag,
    )

    assert model.modes == ["model_500"]
    assert graphrag.modes == ["none"]


def test_apply_case_failure_mode_sets_graphrag_timeout():
    """验证 apply case failure mode sets graphrag timeout 场景。"""
    model = _ModeProxy()
    graphrag = _ModeProxy()

    _apply_case_failure_mode(
        make_case(failureMode="graphrag_timeout"),
        model,
        graphrag,
    )

    assert model.modes == ["none"]
    assert graphrag.modes == ["graphrag_timeout"]


def test_apply_case_failure_mode_requires_managed_proxy():
    """验证 apply case failure mode requires managed proxy 场景。"""
    with pytest.raises(RuntimeError, match="模型代理"):
        _apply_case_failure_mode(
            make_case(failureMode="model_empty"),
            None,
            None,
        )


def test_run_case_plays_all_messages_and_returns_record(monkeypatch, tmp_path):
    """验证 run case plays all messages and returns record 场景。"""
    _stub_environment(monkeypatch)
    case = make_case(messages=["消息A", "消息B"])
    # 每条消息返回不同响应，证明记录的是【最后一条】的结果而非其中任意一条
    client = FakeEngineClient(
        response=lambda message: make_response(reply=f"答复-{message}", roundCount=len(message))
    )

    outcome = run_case(
        case=case,
        run_id="run-001",
        repetition=1,
        attempt=1,
        settings=make_settings(),
        client=client,
        fixture_manager=FakeFixtureManager(),
        token="eval-token",
        owner_id="980001",
        tenant_id="dept:210",
        runtime_dir=tmp_path,
    )

    assert isinstance(outcome, RunOutcome)
    assert outcome.error is None
    assert outcome.fatal is False
    assert client.chat_messages == ["消息A", "消息B"]
    assert outcome.record is not None
    assert outcome.record.case_id == "hse-l1-search-001"
    assert outcome.record.repetition == 1
    assert outcome.record.attempt == 1
    # 回复采用最后一条消息；工具调用和轮次跨轮累计
    assert outcome.record.reply == "答复-消息B"
    assert outcome.record.round_count == len("消息A") + len("消息B")


def test_run_case_applies_and_resets_failure_mode(monkeypatch, tmp_path):
    """验证 run case applies and resets failure mode 场景。"""
    _stub_environment(monkeypatch)
    model = _ModeProxy()
    graphrag = _ModeProxy()

    outcome = run_case(
        case=make_case(failureMode="model_empty"),
        run_id="run-001",
        repetition=1,
        attempt=1,
        settings=make_settings(),
        client=FakeEngineClient(),
        fixture_manager=FakeFixtureManager(),
        token="eval-token",
        owner_id="980001",
        tenant_id="dept:210",
        runtime_dir=tmp_path,
        model_proxy=model,
        graphrag_fixture=graphrag,
    )

    assert outcome.record is not None
    assert model.modes == ["model_empty", "none"]
    assert graphrag.modes == ["none", "none"]


def test_run_case_accumulates_tool_calls_across_turns(monkeypatch, tmp_path):
    """验证 run case accumulates tool calls across turns 场景。"""
    _stub_environment(monkeypatch)

    def response(message):
        """处理 response 相关逻辑。"""
        tool = "get_my_tasks" if message == "消息A" else "get_alerts"
        return make_response(
            toolCalls=[
                {
                    "tool": tool,
                    "toolCallId": f"call-{message}",
                    "args": {},
                    "level": "L1",
                    "status": "success",
                    "roundIndex": 1,
                    "result": '{"status":"success","tool":"get_my_tasks","data":[]}',
                }
            ],
            roundCount=1,
        )

    outcome = run_case(
        case=make_case(messages=["消息A", "消息B"]),
        run_id="run-001",
        repetition=1,
        attempt=1,
        settings=make_settings(),
        client=FakeEngineClient(response=response),
        fixture_manager=FakeFixtureManager(),
        token="eval-token",
        owner_id="980001",
        tenant_id="dept:210",
        runtime_dir=tmp_path,
    )

    assert outcome.record is not None
    assert [call.tool for call in outcome.record.tool_calls] == [
        "get_my_tasks",
        "get_alerts",
    ]
    assert [call.round_index for call in outcome.record.tool_calls] == [1, 2]
    assert outcome.record.round_count == 2


def test_run_case_cancels_confirmation(monkeypatch, tmp_path):
    """验证 run case cancels confirmation 场景。"""
    _stub_environment(monkeypatch)
    client = FakeEngineClient(
        response=make_response(
            status="pending_confirmation",
            executionStatus="pending",
            roundCount=1,
            confirmationToken="tok-1",
            confirmation={
                "tool": "create_task",
                "level": "L3",
                "args": {"title": "整改任务"},
            },
            toolCalls=[
                {
                    "tool": "create_task",
                    "toolCallId": "call-1",
                    "args": {"title": "整改任务"},
                    "level": "L3",
                    "status": "pending",
                    "roundIndex": 1,
                    "result": '{"status":"pending","tool":"create_task","data":{}}',
                }
            ],
        ),
        cancel_response=make_confirm_response(ok=False, status="cancelled"),
    )

    outcome = run_case(
        case=make_case(
            confirmationRequired=True,
            confirmationAction="cancel",
            requiredTools=["create_task"],
            forbiddenTools=[],
            allowedToolSequences=[["create_task"]],
            expectedArguments={},
        ),
        run_id="run-001",
        repetition=1,
        attempt=1,
        settings=make_settings(),
        client=client,
        fixture_manager=FakeFixtureManager(),
        token="eval-token",
        owner_id="980001",
        tenant_id="dept:210",
        runtime_dir=tmp_path,
    )

    assert outcome.record is not None
    assert client.cancel_calls == 1
    assert client.confirm_calls == 0
    assert outcome.record.status == "cancelled"
    assert outcome.record.confirmation_state == "cancelled"
    assert outcome.record.tool_calls[0].status == "cancelled"


def test_run_case_repeats_confirmation_when_requested(monkeypatch, tmp_path):
    """验证 run case repeats confirmation when requested 场景。"""
    _stub_environment(monkeypatch)
    client = FakeEngineClient(
        response=make_response(
            status="pending_confirmation",
            executionStatus="pending",
            roundCount=1,
            confirmationToken="tok-1",
            confirmation={
                "tool": "create_task",
                "level": "L3",
                "args": {"title": "整改任务"},
            },
            toolCalls=[
                {
                    "tool": "create_task",
                    "toolCallId": "call-1",
                    "args": {"title": "整改任务"},
                    "level": "L3",
                    "status": "pending",
                    "roundIndex": 1,
                    "result": '{"status":"pending","tool":"create_task","data":{}}',
                }
            ],
        ),
        confirm_response=make_confirm_response(),
    )

    outcome = run_case(
        case=make_case(
            confirmationRequired=True,
            confirmationAttempts=2,
            requiredTools=["create_task"],
            forbiddenTools=[],
            allowedToolSequences=[["create_task"]],
            expectedArguments={},
        ),
        run_id="run-001",
        repetition=1,
        attempt=1,
        settings=make_settings(),
        client=client,
        fixture_manager=FakeFixtureManager(),
        token="eval-token",
        owner_id="980001",
        tenant_id="dept:210",
        runtime_dir=tmp_path,
    )

    assert outcome.record is not None
    assert client.confirm_calls == 2
    assert len(outcome.record.tool_calls) == 1
    assert outcome.record.status == "success"


def test_run_case_treats_infrastructure_failure_as_non_fatal(monkeypatch, tmp_path):
    """验证 run case treats infrastructure failure as non fatal 场景。"""
    from hse_agent_evals import runner as runner_module

    def explode(run_id, fixture_manager, runtime_dir):
        """处理 explode 相关逻辑。"""
        raise RuntimeError("重置失败")

    monkeypatch.setattr(runner_module.environment, "database_name", lambda run_id, runtime_dir: "engine_agent_eval_x")
    monkeypatch.setattr(runner_module.environment, "reset_case", explode)

    outcome = run_case(
        case=make_case(),
        run_id="run-001",
        repetition=1,
        attempt=1,
        settings=make_settings(),
        client=FakeEngineClient(),
        fixture_manager=FakeFixtureManager(),
        token="eval-token",
        owner_id="980001",
        tenant_id="dept:210",
        runtime_dir=tmp_path,
    )

    assert outcome.record is None
    assert outcome.fatal is False
    assert "重置失败" in (outcome.error or "")


def test_run_case_records_confirmation_ordering_evidence(monkeypatch, tmp_path):
    """验证 run case records confirmation ordering evidence 场景。"""
    _stub_environment(monkeypatch)
    case = make_case(confirmationRequired=True)
    client = FakeEngineClient(
        response=make_response(
            status="pending_confirmation",
            confirmationToken="tok-1",
            executionStatus="pending",
            roundCount=1,
            confirmation={
                "tool": "create_task",
                "level": "L3",
                "args": {"title": "整改任务"},
            },
            toolCalls=[
                {
                    "tool": "create_task",
                    "toolCallId": "call-1",
                    "args": {"title": "整改任务"},
                    "level": "L3",
                    "status": "pending",
                    "roundIndex": 1,
                    "result": '{"status":"pending","tool":"create_task","data":{}}',
                }
            ],
        ),
        confirm_response=make_confirm_response(),
    )

    outcome = run_case(
        case=case,
        run_id="run-001",
        repetition=1,
        attempt=1,
        settings=make_settings(),
        client=client,
        fixture_manager=FakeFixtureManager(),
        token="eval-token",
        owner_id="980001",
        tenant_id="dept:210",
        runtime_dir=tmp_path,
    )

    assert outcome.record is not None
    # 确认前快照无副作用 → 顺序合规
    assert outcome.record.confirmation_before_side_effects is True
    assert outcome.record.status == "success"
    assert outcome.record.confirmation_state == "confirmed"
    assert outcome.record.round_count == 1
    assert outcome.record.trajectory_incomplete is False
    assert len(outcome.record.tool_calls) == 1
    assert outcome.record.tool_calls[0].tool == "create_task"
    assert outcome.record.tool_calls[0].round_index == 2


from hse_agent_evals.runner import (
    ExecutionPlan,
    build_run_record,
    plan_execution,
    run_all,
    run_key,
)

from _factories import make_settings


def _fake_factories(monkeypatch, outcomes_by_case):
    """让 run_all 不碰真实网络/数据库。"""
    from hse_agent_evals import runner as runner_module

    _stub_environment(monkeypatch)
    monkeypatch.setattr(
        runner_module,
        "_load_identities",
        lambda settings: {
            role: {"token": "t", "ownerId": "980001", "tenantId": "dept:210"}
            for role in ("admin", "hse_user", "other_dept_user")
        },
    )
    calls: list[str] = []

    def fake_run_case_factory():
        """处理 fake run case factory 相关逻辑。"""
        def fake_run_case(**kwargs):
            """处理 fake run case 相关逻辑。"""
            case = kwargs["case"]
            calls.append(run_key(case.id, kwargs["repetition"], kwargs["attempt"]))
            return outcomes_by_case(case, kwargs["repetition"], kwargs["attempt"])
        return fake_run_case

    monkeypatch.setattr(runner_module, "run_case", fake_run_case_factory())
    return calls


def test_run_all_rejects_failure_modes_without_managed_fixtures(tmp_path):
    """验证 run all rejects failure modes without managed fixtures 场景。"""
    plan = ExecutionPlan(
        cases=[make_case(failureMode="model_500")],
        repetitions=1,
        mode="baseline",
        dataset="all",
        rerun_failures=False,
    )

    with pytest.raises(ValueError, match="manage-fixtures"):
        run_all(
            plan,
            run_id="run-001",
            settings=make_settings(),
            runtime_dir=tmp_path,
        )


def test_run_all_produces_expected_number_of_records(monkeypatch, tmp_path):
    """验证 run all produces expected number of records 场景。"""
    cases = [make_case(), make_case(id="hse-l1-search-002")]

    def outcomes(case, repetition, attempt):
        """处理 outcomes 相关逻辑。"""
        return RunOutcome(
            record=build_run_record(
                run_id="run-001",
                case=case,
                repetition=repetition,
                attempt=attempt,
                response=make_response(),
                side_effects=[],
                session_id=f"s-{repetition}-{attempt}",
                owner_id="980001",
                tenant_id="dept:210",
                model="qwen2.5:7b",
                provider="local",
                duration_ms=900,
                confirmation_before_side_effects=None,
                side_effect_committed=False,
            ),
            error=None,
            fatal=False,
        )

    calls = _fake_factories(monkeypatch, outcomes)
    plan = plan_execution(cases, dataset="all", repetitions=3, mode="baseline", rerun_failures=False)

    summary = run_all(plan, run_id="run-001", settings=make_settings(), runtime_dir=tmp_path)

    assert len(calls) == 6  # 2 案例 × 3 次
    assert summary.executed == 6
    assert summary.exit_code == 0
    assert len(RawRunStore(tmp_path / "runs" / "run-001").load()) == 6


def test_run_all_resumes_without_re_executing(monkeypatch, tmp_path):
    """验证 run all resumes without re executing 场景。"""
    cases = [make_case()]

    def outcomes(case, repetition, attempt):
        """处理 outcomes 相关逻辑。"""
        return RunOutcome(
            record=build_run_record(
                run_id="run-001",
                case=case,
                repetition=repetition,
                attempt=attempt,
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
            ),
            error=None,
            fatal=False,
        )

    calls = _fake_factories(monkeypatch, outcomes)
    plan = plan_execution(cases, dataset="all", repetitions=3, mode="baseline", rerun_failures=False)

    run_all(plan, run_id="run-001", settings=make_settings(), runtime_dir=tmp_path)
    first_round = len(calls)

    summary = run_all(plan, run_id="run-001", settings=make_settings(), runtime_dir=tmp_path)

    assert len(calls) == first_round  # 第二次不再执行
    assert summary.skipped == 3
    assert summary.executed == 0


def test_plan_execution_filters_dataset_split():
    """验证 plan execution filters dataset split 场景。"""
    cases = [make_case(), make_case(id="hse-l1-search-002", datasetSplit="regression")]
    plan = plan_execution(cases, dataset="regression", repetitions=1, mode="baseline", rerun_failures=False)
    assert [case.id for case in plan.cases] == ["hse-l1-search-002"]


def test_run_all_gate_mode_fails_on_zero_tolerance_violation(monkeypatch, tmp_path):
    """gate 模式：安全类评估未通过必须非零退出。"""
    cases = [make_case()]

    def outcomes(case, repetition, attempt):
        """处理 outcomes 相关逻辑。"""
        return RunOutcome(
            record=build_run_record(
                run_id="run-001",
                case=case,
                repetition=repetition,
                attempt=attempt,
                response=make_response(toolCalls=[], status="failed"),
                side_effects=[],
                session_id=f"s-{repetition}",
                owner_id="980001",
                tenant_id="dept:210",
                model="qwen2.5:7b",
                provider="local",
                duration_ms=900,
                confirmation_before_side_effects=None,
                side_effect_committed=False,
            ),
            error=None,
            fatal=False,
        )

    _fake_factories(monkeypatch, outcomes)
    baseline = plan_execution(cases, dataset="all", repetitions=1, mode="baseline", rerun_failures=False)
    assert run_all(baseline, run_id="run-001", settings=make_settings(), runtime_dir=tmp_path).exit_code == 0

    gate = plan_execution(cases, dataset="all", repetitions=1, mode="gate", rerun_failures=False)
    summary = run_all(gate, run_id="run-002", settings=make_settings(), runtime_dir=tmp_path)
    assert summary.exit_code == 1


def test_run_all_propagates_fatal_outcome(monkeypatch, tmp_path):
    """污染/清理失败时，即便产出了记录也必须非零退出。"""
    cases = [make_case()]

    def outcomes(case, repetition, attempt):
        """处理 outcomes 相关逻辑。"""
        return RunOutcome(
            record=build_run_record(
                run_id="run-001",
                case=case,
                repetition=repetition,
                attempt=attempt,
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
            ),
            error=None,
            fatal=True,
            cleanup_error="清理失败",
        )

    _fake_factories(monkeypatch, outcomes)
    plan = plan_execution(cases, dataset="all", repetitions=1, mode="baseline", rerun_failures=False)

    summary = run_all(plan, run_id="run-001", settings=make_settings(), runtime_dir=tmp_path)

    assert summary.fatal is True
    assert summary.exit_code == 1


def test_run_all_rerun_failures_retries_as_second_attempt(monkeypatch, tmp_path):
    """失败尝试在 rerun_failures 下以 attempt=2 重跑，且 attempt=1 保持不变。"""
    cases = [make_case()]
    attempts: list[int] = []

    def outcomes(case, repetition, attempt):
        """处理 outcomes 相关逻辑。"""
        attempts.append(attempt)
        # 首次失败（无工具调用），重跑成功
        failed = attempt == 1
        return RunOutcome(
            record=build_run_record(
                run_id="run-001",
                case=case,
                repetition=repetition,
                attempt=attempt,
                response=make_response(toolCalls=[], status="failed") if failed else make_response(),
                side_effects=[],
                session_id=f"s-{repetition}-{attempt}",
                owner_id="980001",
                tenant_id="dept:210",
                model="qwen2.5:7b",
                provider="local",
                duration_ms=900,
                confirmation_before_side_effects=None,
                side_effect_committed=False,
            ),
            error=None,
            fatal=False,
        )

    _fake_factories(monkeypatch, outcomes)
    plan = plan_execution(cases, dataset="all", repetitions=1, mode="baseline", rerun_failures=False)
    run_all(plan, run_id="run-001", settings=make_settings(), runtime_dir=tmp_path)

    store = RawRunStore(tmp_path / "runs" / "run-001")
    assert store.existing_keys() == {"hse-l1-search-001|1|1"}

    retry_plan = plan_execution(cases, dataset="all", repetitions=1, mode="baseline", rerun_failures=True)
    summary = run_all(retry_plan, run_id="run-001", settings=make_settings(), runtime_dir=tmp_path)

    assert attempts == [1, 2]
    assert summary.executed == 1
    # attempt=1 的原记录未被覆盖，两条并存
    assert store.existing_keys() == {"hse-l1-search-001|1|1", "hse-l1-search-001|1|2"}


def test_run_all_fails_when_no_records_landed(monkeypatch, tmp_path):
    """环境整体不可用（零记录）时 baseline 也必须非零退出。"""
    cases = [make_case()]

    def outcomes(case, repetition, attempt):
        """处理 outcomes 相关逻辑。"""
        return RunOutcome(None, "执行用例失败: 后端不可用", False)

    _fake_factories(monkeypatch, outcomes)
    plan = plan_execution(cases, dataset="all", repetitions=1, mode="baseline", rerun_failures=False)

    summary = run_all(plan, run_id="run-001", settings=make_settings(), runtime_dir=tmp_path)

    assert summary.executed == 0
    assert summary.failed == 1
    assert summary.exit_code == 1


def test_build_run_record_prefers_explicit_stages_over_whole_turn():
    """后端补发分阶段耗时后，应优先采用拆分，而不是整轮计入模型。"""
    record = build_run_record(
        run_id="run-001",
        case=make_case(),
        repetition=1,
        attempt=1,
        response=make_response(durationMs=900, modelDurationMs=700, toolDurationMs=150),
        side_effects=[],
        session_id="session-001",
        owner_id="7",
        tenant_id="dept:9",
        model="qwen2.5:7b",
        provider="local",
        duration_ms=900,
        confirmation_before_side_effects=None,
        side_effect_committed=False,
    )

    assert record.model_duration_ms == 700
    assert record.tool_duration_ms == 150
    # 整轮耗时仍保留原值（含编排开销，不等于两段之和）
    assert record.duration_ms == 900


def test_build_run_record_prefers_explicit_finish_reasons():
    """后端补发 finishReasons 时直接用，不再从 stopReason 推导。"""
    record = build_run_record(
        run_id="run-001",
        case=make_case(),
        repetition=1,
        attempt=1,
        response=make_response(finishReasons=["tool_calls", "stop"]),
        side_effects=[],
        session_id="session-001",
        owner_id="7",
        tenant_id="dept:9",
        model="qwen2.5:7b",
        provider="local",
        duration_ms=900,
        confirmation_before_side_effects=None,
        side_effect_committed=False,
    )

    assert record.finish_reasons == ["tool_calls", "stop"]


def test_score_rows_take_staged_flag_from_caller_not_record_shape():
    """来源标记必须由知道 response 的一侧给出。

    回归：曾按数值形状反推（tool==0 且 model==duration 即视为未拆分为），
    但「单轮无工具的 stop 用例」其形状与回退口径完全一致，
    会把真正来自后端显式字段的记录误标为未拆分。
    """
    from hse_agent_evals.runner import _score_record

    case = make_case()
    record = build_run_record(
        run_id="r", case=case, repetition=1, attempt=1,
        response=make_response(durationMs=900, modelDurationMs=900, toolDurationMs=0),
        side_effects=[], session_id="s", owner_id="7", tenant_id="dept:9",
        model="m", provider="p", duration_ms=900,
        confirmation_before_side_effects=None, side_effect_committed=False,
    )
    # 单轮无工具：形状与回退口径相同，但来源确是显式字段
    assert record.model_duration_ms == 900 and record.tool_duration_ms == 0

    marked = _score_record(case, record, staged_durations=True)
    legacy = _score_record(case, record, staged_durations=False)

    assert {row["telemetry"]["stagedDurations"] for row in marked} == {True}
    assert {row["telemetry"]["stagedDurations"] for row in legacy} == {False}
