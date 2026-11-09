from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


TOKEN = "demo-eval-token"
TURN_ID = "turn-demo-1"
TRACE_ID = "trace-demo-1"


def _reply(payload: dict[str, Any]) -> dict[str, Any]:
    return {"code": 200, "msg": "ok", "data": payload}


def _base_payload(reply: str, *, status: str, execution_status: str) -> dict[str, Any]:
    return {
        "reply": reply,
        "sessionId": "session-demo",
        "toolCalls": [],
        "tokens": 24,
        "promptTokens": 18,
        "completionTokens": 6,
        "executionStatus": execution_status,
        "durationMs": 40,
        "traceId": TRACE_ID,
        "turnId": TURN_ID,
        "stopReason": "completed" if status == "success" else "confirmation_required",
        "degradedMode": "none",
        "finishReasons": ["stop"] if status == "success" else ["tool_calls"],
        "modelDurationMs": 30,
        "toolDurationMs": 10,
        "roundCount": 1,
        "status": status,
    }


def _read_only_payload() -> dict[str, Any]:
    payload = _base_payload(
        "当前有两项待办：项目一安全检查、项目二隐患整改。",
        status="success",
        execution_status="success",
    )
    payload["toolCalls"] = [
        {
            "tool": "get_my_tasks",
            "toolCallId": "call-demo-read",
            "args": {},
            "level": "L1",
            "status": "success",
            "roundIndex": 1,
            "result": json.dumps(
                {
                    "status": "success",
                    "tool": "get_my_tasks",
                    "data": [
                        {"taskName": "项目一安全检查", "status": "todo"},
                        {"taskName": "项目二隐患整改", "status": "overdue"},
                    ],
                },
                ensure_ascii=False,
            ),
        }
    ]
    return payload


def _write_payload(message: str) -> dict[str, Any]:
    if "提醒" in message:
        tool_name = "send_reminder"
        args = {"userId": 980002, "content": "请于今日17时前完成隐患整改"}
    elif "标记" in message:
        tool_name = "mark_alert_read"
        args = {"alertId": 982000}
    else:
        tool_name = "create_task"
        args = {
            "taskName": "脚手架专项检查",
            "deptId": 210,
            "deadline": "2026-12-31",
        }

    payload = _base_payload(
        f"工具「{tool_name}」属于 L3 操作，请确认后执行。",
        status="pending_confirmation",
        execution_status="pending",
    )
    payload["confirmationToken"] = "demo-confirmation-token"
    payload["confirmation"] = {
        "tool": tool_name,
        "level": "L3",
        "args": args,
        "turnId": TURN_ID,
    }
    payload["toolCalls"] = [
        {
            "tool": tool_name,
            "toolCallId": "call-demo-write",
            "args": args,
            "level": "L3",
            "status": "pending",
            "roundIndex": 1,
            "result": json.dumps(
                {
                    "status": "pending",
                    "tool": tool_name,
                    "data": {"confirmationToken": "demo-confirmation-token"},
                },
                ensure_ascii=False,
            ),
        }
    ]
    return payload


def _refusal_payload() -> dict[str, Any]:
    payload = _base_payload(
        "该请求涉及其他部门数据，当前用户无权访问，已拒绝。",
        status="success",
        execution_status="success",
    )
    return payload


def _confirmed_payload() -> dict[str, Any]:
    payload = _base_payload(
        "已确认并执行一次操作。",
        status="success",
        execution_status="success",
    )
    payload["roundCount"] = 2
    payload["toolCalls"] = [
        {
            "tool": "create_task",
            "toolCallId": "call-demo-confirmed",
            "args": {"taskName": "脚手架专项检查", "deptId": 210},
            "level": "L3",
            "status": "success",
            "roundIndex": 2,
            "result": json.dumps(
                {
                    "status": "success",
                    "tool": "create_task",
                    "data": {"taskId": 981001},
                },
                ensure_ascii=False,
            ),
        }
    ]
    return payload


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "UP"})
            return
        self._send_json(404, {"code": 404, "msg": "not found"})

    def do_POST(self) -> None:
        payload = self._read_json()
        if payload is None:
            return
        if self.path == "/login":
            self._send_json(200, {"code": 200, "msg": "ok", "token": TOKEN})
            return
        if self.path == "/hse/agent/chat":
            message = str(payload.get("message", ""))
            if any(word in message for word in ("其他部门", "跨部门", "别的部门")):
                self._send_json(200, _reply(_refusal_payload()))
            elif any(word in message for word in ("创建", "发提醒", "提醒", "标记")):
                self._send_json(200, _reply(_write_payload(message)))
            else:
                self._send_json(200, _reply(_read_only_payload()))
            return
        if self.path == "/hse/agent/confirm":
            self._send_json(200, _reply(_confirmed_payload()))
            return
        self._send_json(404, {"code": 404, "msg": "not found"})

    def _read_json(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"code": 400, "msg": "invalid content length"})
            return None
        if length <= 0 or length > 1024 * 1024:
            self._send_json(400, {"code": 400, "msg": "invalid body"})
            return None
        try:
            value = json.loads(self.rfile.read(length))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"code": 400, "msg": "invalid json"})
            return None
        if not isinstance(value, dict):
            self._send_json(400, {"code": 400, "msg": "body must be an object"})
            return None
        return value

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18091)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"fake agent listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
