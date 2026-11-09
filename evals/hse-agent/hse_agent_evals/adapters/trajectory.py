from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from hse_agent_evals.models import ToolCall


ALLOWED_TOOL_STATUSES = {
    "success",
    "empty",
    "error",
    "pending",
    "cancelled",
    "result_unknown",
}


@dataclass(frozen=True)
class TrajectoryMapping:
    """处理 TrajectoryMapping 相关逻辑。"""
    tool_calls: list[ToolCall]
    trajectory_incomplete: bool
    source: str

    def to_evidence(self) -> dict[str, Any]:
        """处理 to evidence 相关逻辑。"""
        return {
            "source": self.source,
            "toolCalls": [
                call.model_dump(by_alias=True, mode="json", exclude_none=True)
                for call in self.tool_calls
            ],
            "trajectoryIncomplete": self.trajectory_incomplete,
        }


def map_tool_calls(
    database_log: dict[str, Any] | list[Any] | None,
    response: dict[str, Any] | None,
) -> TrajectoryMapping:
    """处理 map tool calls 相关逻辑。"""
    database_value, database_present = _database_calls(database_log)
    incomplete = False
    database_partial: list[ToolCall] | None = None

    if database_present:
        raw_calls, valid = _as_list(database_value)
        if valid:
            calls, parse_incomplete = _parse_calls(raw_calls)
            if not parse_incomplete:
                return TrajectoryMapping(calls, False, "database")
            database_partial = calls
        incomplete = True

    response_value, response_present = _response_calls(response)
    if response_present:
        raw_calls, valid = _as_list(response_value)
        if valid:
            calls, parse_incomplete = _parse_calls(raw_calls)
            if not parse_incomplete:
                return TrajectoryMapping(calls, incomplete, "response")
            if database_partial is None:
                return TrajectoryMapping(calls, True, "response")
        incomplete = True

    if database_partial is not None:
        return TrajectoryMapping(database_partial, True, "database")
    return TrajectoryMapping([], incomplete, "none")


map_trajectory = map_tool_calls


def _database_calls(database_log: dict[str, Any] | list[Any] | None) -> tuple[Any, bool]:
    """database calls 的内部辅助逻辑。"""
    if isinstance(database_log, list):
        return database_log, True
    if not isinstance(database_log, dict):
        return None, False
    for key in ("tool_calls", "toolCalls"):
        if key in database_log:
            return database_log[key], True
    return None, False


def _response_calls(response: dict[str, Any] | None) -> tuple[Any, bool]:
    """response calls 的内部辅助逻辑。"""
    if not isinstance(response, dict):
        return None, False
    for key in ("toolCalls", "tool_calls"):
        if key in response:
            return response[key], True
    return None, False


def _as_list(value: Any) -> tuple[list[Any], bool]:
    """as list 的内部辅助逻辑。"""
    if isinstance(value, list):
        return value, True
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return [], False
        if isinstance(decoded, list):
            return decoded, True
    return [], False


def _parse_calls(raw_calls: list[Any]) -> tuple[list[ToolCall], bool]:
    """parse calls 的内部辅助逻辑。"""
    calls: list[ToolCall] = []
    incomplete = False
    for raw in raw_calls:
        if not isinstance(raw, dict):
            incomplete = True
            continue
        tool = raw.get("tool", raw.get("name", raw.get("toolName")))
        if not isinstance(tool, str) or not tool.strip():
            incomplete = True
            continue

        args, args_valid = _parse_args(raw.get("args", raw.get("arguments", {})))
        incomplete = incomplete or not args_valid

        round_value = raw.get("roundIndex", raw.get("round_index"))
        if isinstance(round_value, bool) or not isinstance(round_value, int) or round_value < 1:
            round_value = 1
            incomplete = True

        level = raw.get("level", "")
        if not isinstance(level, str):
            level = _stable_text(level)
            incomplete = True

        result_text, status, error_code, schema_version, tool_call_id_from_result, result_ref, idempotent_replay, result_incomplete = _parse_result(
            raw.get("result", ""), raw.get("status")
        )
        incomplete = incomplete or result_incomplete

        tool_call_id = _text(raw.get("toolCallId")) or _text(raw.get("tool_call_id")) or tool_call_id_from_result

        calls.append(
            ToolCall(
                tool=tool,
                args=args,
                level=level,
                result=result_text,
                round_index=round_value,
                status=status,
                error_code=error_code,
                schema_version=schema_version,
                tool_call_id=tool_call_id,
                result_ref=_text(raw.get("resultRef")) or result_ref,
                idempotent_replay=idempotent_replay,
            )
        )
    return calls, incomplete


def _parse_args(value: Any) -> tuple[dict[str, Any], bool]:
    """parse args 的内部辅助逻辑。"""
    if isinstance(value, dict):
        return value, True
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return {}, False
        if isinstance(decoded, dict):
            return decoded, True
    return {}, False


def _parse_result(
    value: Any,
    fallback_status: Any,
) -> tuple[str, str | None, str | None, str | None, str | None, str | None, bool | None, bool]:
    """parse result 的内部辅助逻辑。"""
    status = _normalized_status(fallback_status)
    incomplete = isinstance(fallback_status, str) and status is None
    envelope = _maybe_envelope(value)
    if envelope is None:
        return _stable_text(value), status, None, None, None, _extract_result_ref(value), None, incomplete

    envelope_status = _normalized_status(envelope.get("status"))
    if envelope_status is None:
        incomplete = True
    else:
        status = envelope_status

    meta = envelope.get("meta") if isinstance(envelope.get("meta"), dict) else {}
    error = envelope.get("error") if isinstance(envelope.get("error"), dict) else {}
    data = envelope.get("data")

    return (
        _stable_result_value(data),
        status,
        _text(error.get("code")),
        _text(meta.get("schemaVersion")),
        _text(meta.get("toolCallId")),
        _extract_result_ref(envelope),
        meta.get("idempotentReplay") if isinstance(meta.get("idempotentReplay"), bool) else None,
        incomplete,
    )


def _maybe_envelope(value: Any) -> dict[str, Any] | None:
    """maybe envelope 的内部辅助逻辑。"""
    candidate = value
    if isinstance(value, str):
        try:
            candidate = json.loads(value)
        except (TypeError, ValueError):
            return None
    if not isinstance(candidate, dict):
        return None
    if "status" not in candidate or "tool" not in candidate:
        return None
    if not any(key in candidate for key in ("data", "error", "meta")):
        return None
    return candidate


def _normalized_status(value: Any) -> str | None:
    """normalized status 的内部辅助逻辑。"""
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if normalized in ALLOWED_TOOL_STATUSES else None


def _extract_result_ref(value: Any) -> str | None:
    """extract result ref 的内部辅助逻辑。"""
    if isinstance(value, dict):
        direct = _text(value.get("resultRef"))
        if direct:
            return direct
        nested = value.get("data")
        if isinstance(nested, dict):
            return _text(nested.get("resultRef"))
    return None


def _stable_result_value(value: Any) -> str:
    """stable result value 的内部辅助逻辑。"""
    if value is None:
        return ""
    if isinstance(value, dict) and not value:
        return ""
    if isinstance(value, list) and not value:
        return ""
    return _stable_text(value)


def _stable_text(value: Any) -> str:
    """stable text 的内部辅助逻辑。"""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    return str(value)


def _text(value: Any) -> str | None:
    """text 的内部辅助逻辑。"""
    if not isinstance(value, str):
        return None
    return value if value.strip() else None