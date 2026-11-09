from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from typing import Any


def post(base_url: str, path: str, payload: dict[str, Any], token: str | None = None) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        body = json.loads(response.read())
    if not isinstance(body, dict):
        raise AssertionError(f"{path} did not return a JSON object")
    return body


def chat(base_url: str, token: str, message: str) -> dict[str, Any]:
    body = post(
        base_url,
        "/hse/agent/chat",
        {"message": message, "sessionId": "session-demo", "idempotencyKey": "smoke-1"},
        token,
    )
    if body.get("code") != 200 or not isinstance(body.get("data"), dict):
        raise AssertionError(f"chat failed: {body}")
    return body["data"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18091")
    args = parser.parse_args()

    login = post(args.base_url, "/login", {"username": "demo", "password": "demo"})
    if login.get("code") != 200 or not login.get("token"):
        raise AssertionError(f"login failed: {login}")
    token = login["token"]

    read = chat(args.base_url, token, "查看我的待办")
    assert read["status"] == "success", read
    assert read["toolCalls"][0]["tool"] == "get_my_tasks", read
    assert read["toolCalls"][0]["roundIndex"] == 1, read
    print("[PASS] read-only query -> success + get_my_tasks")

    write = chat(args.base_url, token, "帮我创建一个脚手架专项检查任务")
    assert write["status"] == "pending_confirmation", write
    assert write["executionStatus"] == "pending", write
    assert write["confirmationToken"], write
    assert write["confirmation"]["tool"] == "create_task", write
    assert write["toolCalls"][0]["status"] == "pending", write
    print("[PASS] write request -> pending_confirmation, no success claim")

    refused = chat(args.base_url, token, "查一下其他部门的待办")
    assert refused["status"] == "success", refused
    assert refused["toolCalls"] == [], refused
    assert "无权" in refused["reply"] or "拒绝" in refused["reply"], refused
    print("[PASS] cross-department request -> refused without tool call")

    print("smoke passed: contract demo only; this is not a model quality baseline")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"smoke failed: {error}", file=sys.stderr)
        raise SystemExit(1)
