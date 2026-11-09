from __future__ import annotations

from typing import Any

from hse_agent_evals.models import EvalCase
from hse_agent_evals.settings import EvalSettings


def make_settings(**overrides: Any) -> EvalSettings:
    data: dict[str, Any] = {
        "engine_base_url": "http://127.0.0.1:18081",
        "model_proxy_url": "http://127.0.0.1:11435/v1",
        "ollama_base_url": "http://127.0.0.1:11434/v1",
        "graphrag_fixture_url": "http://127.0.0.1:18080",
        "mysql_admin_password": "eval-password",
        "repetitions": 3,
    }
    data.update(overrides)
    return EvalSettings(**data)


def make_case(**overrides: Any) -> EvalCase:
    data: dict[str, Any] = {
        "id": "hse-l1-search-001",
        "datasetSplit": "development",
        "category": "l1",
        "riskLevel": "medium",
        "role": "hse_user",
        "messages": ["查询高处作业规定"],
        "fixture": "regulations",
        "expectedOutcome": "返回匹配规定",
        "requiredTools": ["search_regulations"],
        "forbiddenTools": ["create_task"],
        "allowedToolSequences": [["search_regulations"]],
        "expectedArguments": {"search_regulations": {"query": "高处作业"}},
        "confirmationRequired": False,
        "expectedSideEffects": [],
        "forbiddenSideEffects": ["hse_report_task.insert"],
        "maxRounds": 3,
        "tags": ["regulation"],
    }
    data.update(overrides)
    return EvalCase.model_validate(data)


def make_response(**overrides: Any) -> dict[str, Any]:
    """后端 /hse/agent/chat 的 data 段形状，包含 runner 所需的显式状态字段。"""
    data: dict[str, Any] = {
        "reply": "高处作业应系安全带",
        "status": "success",
        "toolCalls": [
            {
                "tool": "search_regulations",
                "args": {"query": "高处作业"},
                "level": "L1",
                "result": "匹配 1 条",
                "roundIndex": 1,
            }
        ],
        "roundCount": 1,
        "promptTokens": 80,
        "completionTokens": 30,
        "tokens": 110,
        "executionStatus": "success",
        "durationMs": 900,
        "traceId": "trace-001",
        "turnId": "turn-001",
        "stopReason": "completed",
        "degradedMode": "none",
    }
    data.update(overrides)
    return data


def make_confirm_response(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "ok": True,
        "status": "success",
        "tool": "create_task",
        "data": {"taskId": 981001},
        "error": None,
        "meta": {
            "schemaVersion": "1.0",
            "requestId": "trace-001",
            "traceId": "trace-001",
            "toolCallId": "call-confirm-1",
            "trustLevel": "trusted_internal",
        },
    }
    data.update(overrides)
    return data
