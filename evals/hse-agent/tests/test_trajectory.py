import json

import pytest

from hse_agent_evals.adapters.trajectory import map_tool_calls


def test_prefers_database_tool_calls_and_parses_json_arguments():
    """验证 prefers database tool calls and parses json arguments 场景。"""
    mapped = map_tool_calls(
        database_log={
            "tool_calls": json.dumps(
                [
                    {
                        "tool": "search_regulations",
                        "args": '{"query": "高处作业"}',
                        "level": "L1",
                        "result": {"count": 2},
                        "roundIndex": 2,
                    }
                ],
                ensure_ascii=False,
            )
        },
        response={
            "toolCalls": [
                {
                    "tool": "fallback_tool",
                    "args": {},
                    "level": "L1",
                    "result": "unused",
                    "roundIndex": 1,
                }
            ]
        },
    )

    assert mapped.source == "database"
    assert [call.tool for call in mapped.tool_calls] == ["search_regulations"]
    assert mapped.tool_calls[0].args == {"query": "高处作业"}
    assert mapped.tool_calls[0].result == '{"count":2}'
    assert mapped.tool_calls[0].round_index == 2
    assert mapped.trajectory_incomplete is False


def test_falls_back_to_response_and_marks_missing_round_incomplete():
    """验证 falls back to response and marks missing round incomplete 场景。"""
    mapped = map_tool_calls(
        database_log=None,
        response={
            "toolCalls": [
                {"tool": "first", "args": {}, "level": "L1", "result": "ok"},
                {"tool": "second", "args": {}, "level": "L2", "result": "ok"},
            ]
        },
    )

    assert mapped.source == "response"
    assert [call.round_index for call in mapped.tool_calls] == [1, 1]
    assert mapped.trajectory_incomplete is True


def test_malformed_database_log_uses_response_without_hiding_incompleteness():
    """验证 malformed database log uses response without hiding incompleteness 场景。"""
    mapped = map_tool_calls(
        database_log={"tool_calls": "not-json"},
        response={
            "toolCalls": [
                {
                    "tool": "get_my_tasks",
                    "args": {},
                    "level": "L1",
                    "result": "[]",
                    "roundIndex": 1,
                }
            ]
        },
    )

    assert mapped.source == "response"
    assert [call.tool for call in mapped.tool_calls] == ["get_my_tasks"]
    assert mapped.trajectory_incomplete is True


def test_malformed_arguments_are_controlled_and_marked_incomplete():
    """验证 malformed arguments are controlled and marked incomplete 场景。"""
    mapped = map_tool_calls(
        database_log={
            "tool_calls": [
                {
                    "tool": "search_regulations",
                    "args": "{bad-json",
                    "level": "L1",
                    "result": "error",
                    "roundIndex": 1,
                }
            ]
        },
        response={},
    )

    assert mapped.tool_calls[0].args == {}
    assert mapped.trajectory_incomplete is True
    assert mapped.to_evidence() == {
        "source": "database",
        "toolCalls": [
            {
                "args": {},
                "level": "L1",
                "result": "error",
                "roundIndex": 1,
                "tool": "search_regulations",
            }
        ],
        "trajectoryIncomplete": True,
    }


def test_partially_invalid_database_top_level_list_falls_back_to_complete_response():
    """验证 partially invalid database top level list falls back to complete response 场景。"""
    mapped = map_tool_calls(
        database_log=[
            {
                "tool": "search_regulations",
                "args": "{bad-json",
                "level": "L1",
                "result": "partial",
                "roundIndex": 1,
            }
        ],
        response={
            "toolCalls": [
                {
                    "tool": "search_regulations",
                    "args": {"query": "高处作业"},
                    "level": "L1",
                    "result": "ok",
                    "roundIndex": 1,
                },
                {
                    "tool": "create_task",
                    "args": {},
                    "level": "L3",
                    "result": "blocked",
                    "roundIndex": 2,
                },
            ]
        },
    )

    assert mapped.source == "response"
    assert [call.tool for call in mapped.tool_calls] == [
        "search_regulations",
        "create_task",
    ]
    assert mapped.trajectory_incomplete is True


def test_non_list_sources_are_controlled_as_empty_incomplete_trajectory():
    """验证 non list sources are controlled as empty incomplete trajectory 场景。"""
    mapped = map_tool_calls(
        database_log={"tool_calls": {"tool": "not-a-list"}},
        response={"toolCalls": "not-a-list"},
    )

    assert mapped.tool_calls == []
    assert mapped.source == "none"
    assert mapped.trajectory_incomplete is True


@pytest.mark.parametrize(
    ("status", "ok", "data", "error", "expected_code"),
    [
        ("success", True, {"taskId": 12}, None, None),
        ("empty", True, {}, None, None),
        ("error", False, {}, {"code": "UNKNOWN_TOOL", "retryable": False}, "UNKNOWN_TOOL"),
        ("pending", False, {"state": "executing"}, None, None),
        ("cancelled", False, {}, None, None),
        ("result_unknown", False, {}, {"code": "RESULT_UNKNOWN", "retryable": False}, "RESULT_UNKNOWN"),
    ],
)
def test_unified_tool_envelope_retains_status_and_metadata(
    status, ok, data, error, expected_code
):
    """验证 unified tool envelope retains status and metadata 场景。"""
    mapped = map_tool_calls(
        database_log=None,
        response={
            "toolCalls": [
                {
                    "tool": "create_task",
                    "args": {"deptId": 9},
                    "level": "L3",
                    "toolCallId": "call-1",
                    "result": json.dumps(
                        {
                            "ok": ok,
                            "status": status,
                            "tool": "create_task",
                            "data": data,
                            "error": error,
                            "meta": {
                                "schemaVersion": "1.0",
                                "toolCallId": "call-1",
                                "idempotentReplay": status == "success",
                            },
                        },
                        ensure_ascii=False,
                    ),
                    "roundIndex": 2,
                }
            ]
        },
    )

    call = mapped.tool_calls[0]
    assert mapped.source == "response"
    assert mapped.trajectory_incomplete is False
    assert call.status == status
    assert call.error_code == expected_code
    assert call.schema_version == "1.0"
    assert call.tool_call_id == "call-1"
    assert call.idempotent_replay is (status == "success")


def test_unified_tool_envelope_retains_result_ref_from_projected_data():
    """验证 unified tool envelope retains result ref from projected data 场景。"""
    mapped = map_tool_calls(
        database_log=None,
        response={
            "toolCalls": [
                {
                    "tool": "read_tool_result_page",
                    "args": {"resultRef": "tool-result://stored-1", "offset": 0},
                    "level": "L1",
                    "result": json.dumps(
                        {
                            "ok": True,
                            "status": "success",
                            "tool": "read_tool_result_page",
                            "data": {
                                "resultRef": "tool-result://stored-1",
                                "content": "第 1 页",
                                "truncated": False,
                            },
                            "error": None,
                            "meta": {"schemaVersion": "1.0", "toolCallId": "call-2"},
                        },
                        ensure_ascii=False,
                    ),
                    "roundIndex": 1,
                }
            ]
        },
    )

    call = mapped.tool_calls[0]
    evidence = mapped.to_evidence()

    assert call.result_ref == "tool-result://stored-1"
    assert evidence["toolCalls"][0]["resultRef"] == "tool-result://stored-1"
    assert evidence["toolCalls"][0]["schemaVersion"] == "1.0"
