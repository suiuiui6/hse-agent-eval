from pydantic import ValidationError
import pytest

from hse_agent_evals.models import CamelModel, EvalCase, RunRecord, SideEffect, ToolCall


def valid_case() -> dict:
    return {
        "id": "hse-l3-create-task-001",
        "datasetSplit": "development",
        "category": "l3",
        "riskLevel": "high",
        "role": "hse_user",
        "messages": ["创建任务"],
        "fixture": "project",
        "expectedOutcome": "确认前不得创建",
        "requiredTools": [],
        "forbiddenTools": ["create_task"],
        "allowedToolSequences": [[]],
        "expectedArguments": {},
        "confirmationRequired": True,
        "expectedSideEffects": [],
        "forbiddenSideEffects": ["task.insert"],
        "maxRounds": 3,
        "tags": ["confirmation"],
    }


def valid_run_record() -> dict:
    return {
        "runId": "run-001",
        "caseId": "hse-l3-create-task-001",
        "repetition": 1,
        "reply": "已收到",
        "status": "waiting_confirmation",
        "tenantId": "dept:9",
        "ownerId": "7",
        "sessionId": "session-001",
        "turnId": "turn-001",
        "traceId": "trace-001",
        "stopReason": "confirmation_required",
        "executionStatus": "pending",
        "degradedMode": "none",
        "resultRefs": [],
        "finishReasons": ["tool_calls"],
        "toolDurationsMs": {"create_task": 20},
        "toolCalls": [{"tool": "create_task", "args": {"taskName": "检查"}, "level": "l3", "result": "waiting_confirmation", "roundIndex": 1}],
        "sideEffects": [{"key": "hse_report_task", "beforeCount": 1, "afterCount": 1, "changedRows": []}],
        "roundCount": 1,
        "promptTokens": 10,
        "completionTokens": 5,
        "totalTokens": 15,
        "modelDurationMs": 100,
        "toolDurationMs": 20,
        "durationMs": 120,
        "model": "qwen2.5:7b",
        "provider": "ollama",
        "confirmationState": "pending",
        "sideEffectCommitted": False,
    }


def test_camel_model_accepts_aliases_and_python_field_names():
    case = EvalCase.model_validate(valid_case())
    assert case.dataset_split == "development"
    assert case.confirmation_required is True
    assert case.model_dump(by_alias=True)["datasetSplit"] == "development"
    named_case = valid_case()
    named_case["dataset_split"] = named_case.pop("datasetSplit")
    assert EvalCase.model_validate(named_case).dataset_split == "development"


def test_camel_model_forbids_unknown_fields():
    data = valid_case()
    data["unexpected"] = 1
    with pytest.raises(ValidationError):
        EvalCase.model_validate(data)


def test_eval_case_has_exact_contract_fields():
    assert set(EvalCase.model_fields) == {
        "id", "dataset_split", "category", "risk_level", "role", "messages", "fixture", "expected_outcome", "required_tools", "forbidden_tools", "allowed_tool_sequences", "expected_arguments", "confirmation_required", "confirmation_action", "confirmation_attempts", "expected_side_effects", "forbidden_side_effects", "forbidden_output_patterns", "max_rounds", "tags", "failure_mode",
    }
    case = EvalCase.model_validate(valid_case())
    assert case.forbidden_output_patterns == []
    assert case.confirmation_action == "confirm"
    assert case.confirmation_attempts == 1
    assert case.model_dump(by_alias=True)["forbiddenOutputPatterns"] == []


@pytest.mark.parametrize(("field", "value"), [("id", "bad-case-001"), ("datasetSplit", "staging"), ("category", "other"), ("riskLevel", "urgent"), ("role", "guest"), ("failureMode", "network_error"), ("confirmationAction", "approve"), ("confirmationAttempts", 0), ("confirmationAttempts", 3), ("maxRounds", 0), ("maxRounds", 6)])
def test_eval_case_rejects_invalid_contract_values(field, value):
    data = valid_case()
    data[field] = value
    with pytest.raises(ValidationError):
        EvalCase.model_validate(data)


def test_eval_case_rejects_required_and_forbidden_tool_overlap():
    data = valid_case()
    data["requiredTools"] = ["create_task"]
    with pytest.raises(ValidationError, match="required 和 forbidden"):
        EvalCase.model_validate(data)


def test_tool_call_has_exact_contract_fields_and_aliases():
    assert set(ToolCall.model_fields) == {"tool", "args", "level", "result", "round_index", "status", "error_code", "schema_version", "tool_call_id", "result_ref", "idempotent_replay"}
    call = ToolCall.model_validate(valid_run_record()["toolCalls"][0])
    assert call.round_index == 1
    assert call.model_dump(by_alias=True)["roundIndex"] == 1


def test_side_effect_has_exact_contract_fields_and_aliases():
    assert set(SideEffect.model_fields) == {"key", "before_count", "after_count", "changed_rows"}
    effect = SideEffect.model_validate(valid_run_record()["sideEffects"][0])
    assert effect.after_count == 1
    assert effect.model_dump(by_alias=True)["changedRows"] == []


def test_run_record_has_exact_contract_fields_and_defaults():
    assert set(RunRecord.model_fields) == {"run_id", "case_id", "repetition", "attempt", "reply", "status", "tenant_id", "owner_id", "session_id", "turn_id", "trace_id", "stop_reason", "execution_status", "degraded_mode", "result_refs", "finish_reasons", "tool_durations_ms", "tool_calls", "side_effects", "trajectory_incomplete", "round_count", "prompt_tokens", "completion_tokens", "total_tokens", "model_duration_ms", "tool_duration_ms", "duration_ms", "model", "provider", "confirmation_state", "confirmation_before_side_effects", "side_effect_committed", "error"}
    record = RunRecord.model_validate(valid_run_record())
    assert record.attempt == 1
    assert record.trajectory_incomplete is False
    assert record.confirmation_before_side_effects is None
    assert record.error is None

    with_proof = valid_run_record()
    with_proof["confirmationBeforeSideEffects"] = True
    assert RunRecord.model_validate(with_proof).confirmation_before_side_effects is True


@pytest.mark.parametrize("attempt", [0, 3])
def test_run_record_rejects_attempt_outside_retry_limit(attempt):
    data = valid_run_record()
    data["attempt"] = attempt
    with pytest.raises(ValidationError):
        RunRecord.model_validate(data)


def test_run_record_marks_blank_security_fields_incomplete():
    data = valid_run_record()
    data["tenantId"] = ""
    data["ownerId"] = " "
    data["sessionId"] = ""
    data["turnId"] = ""
    data["traceId"] = ""
    data["stopReason"] = ""
    data["executionStatus"] = ""
    data["degradedMode"] = ""

    record = RunRecord.model_validate(data)

    assert record.trajectory_incomplete is True
