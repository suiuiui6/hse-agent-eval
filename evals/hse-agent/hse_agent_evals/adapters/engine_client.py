from __future__ import annotations

from typing import Any
from uuid import uuid4

import httpx


class EngineClient:
    """处理 EngineClient 相关逻辑。"""
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:18081",
        timeout: float | httpx.Timeout = 90.0,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """初始化实例。"""
        if client is not None and transport is not None:
            raise ValueError("不能同时注入 HTTP 客户端和 Transport")
        if client is None:
            if not isinstance(base_url, str) or not base_url.strip():
                raise ValueError("Engine 地址不能为空")
            client = httpx.Client(
                base_url=base_url.rstrip("/"),
                timeout=timeout,
                transport=transport,
            )
        self.client = client

    def login(self, username: str, password: str) -> str:
        """处理 login 相关逻辑。"""
        _require_nonblank(username, "用户名")
        _require_nonblank(password, "密码")
        try:
            response = self.client.post(
                "/login", json={"username": username, "password": password}
            )
            response.raise_for_status()
        except (httpx.HTTPError, RuntimeError):
            raise RuntimeError("登录失败: HTTP 请求失败") from None
        try:
            body = response.json()
        except ValueError:
            raise RuntimeError("登录失败: 响应格式错误") from None
        if not isinstance(body, dict) or body.get("code") != 200:
            raise RuntimeError("登录失败: 服务端拒绝请求")
        token = body.get("token")
        if not isinstance(token, str) or not token.strip():
            raise RuntimeError("登录失败: 响应缺少令牌")
        return token

    def chat(
        self,
        token: str,
        message: str,
        session_id: str,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """处理 chat 相关逻辑。"""
        _require_nonblank(message, "消息")
        _require_nonblank(session_id, "会话标识")
        payload = {
            "message": message,
            "sessionId": session_id,
            "idempotencyKey": _ensure_idempotency_key(idempotency_key),
        }
        return self._post_agent_json(
            token, "/hse/agent/chat", payload, "Agent 调用失败"
        )

    def binding(self, token: str) -> dict[str, Any]:
        """处理 binding 相关逻辑。"""
        return self._get_agent_json(
            token, "/hse/agent/eval/binding", "绑定检查失败"
        )

    def confirm(
        self,
        token: str,
        confirmation_token: str,
        tool_name: str,
        args: dict[str, Any] | None,
        session_id: str,
        turn_id: str,
    ) -> dict[str, Any]:
        """处理 confirm 相关逻辑。"""
        return self._post_agent_json(
            token,
            "/hse/agent/confirm",
            {
                "confirmationToken": _require_text(confirmation_token, "确认令牌"),
                "toolName": _require_text(tool_name, "工具名称"),
                "args": _require_dict(args),
                "sessionId": _require_text(session_id, "会话标识"),
                "turnId": _require_text(turn_id, "轮次标识"),
            },
            "确认调用失败",
        )

    def cancel(
        self,
        token: str,
        confirmation_token: str,
        tool_name: str,
        args: dict[str, Any] | None,
        session_id: str,
        turn_id: str,
    ) -> dict[str, Any]:
        """处理 cancel 相关逻辑。"""
        return self._post_agent_json(
            token,
            "/hse/agent/cancel",
            {
                "confirmationToken": _require_text(confirmation_token, "确认令牌"),
                "toolName": _require_text(tool_name, "工具名称"),
                "args": _require_dict(args),
                "sessionId": _require_text(session_id, "会话标识"),
                "turnId": _require_text(turn_id, "轮次标识"),
            },
            "取消调用失败",
        )

    def clear(self, token: str, session_id: str) -> dict[str, Any]:
        """处理 clear 相关逻辑。"""
        return self._post_agent_json(
            token,
            "/hse/agent/clear",
            {"sessionId": _require_text(session_id, "会话标识")},
            "清空会话失败",
        )

    def _post_agent_json(
        self,
        token: str,
        path: str,
        payload: dict[str, Any],
        error_prefix: str,
    ) -> dict[str, Any]:
        """post agent json 的内部辅助逻辑。"""
        _require_nonblank(token, "访问令牌")
        try:
            response = self.client.post(
                path,
                headers={"Authorization": f"Bearer {token}"},
                json=payload,
            )
            response.raise_for_status()
        except (httpx.HTTPError, RuntimeError):
            raise RuntimeError(f"{error_prefix}: HTTP 请求失败") from None
        return _unwrap_ajax_dict(response, error_prefix)

    def _get_agent_json(
        self,
        token: str,
        path: str,
        error_prefix: str,
    ) -> dict[str, Any]:
        """get agent json 的内部辅助逻辑。"""
        _require_nonblank(token, "访问令牌")
        try:
            response = self.client.get(
                path,
                headers={"Authorization": f"Bearer {token}"},
            )
            response.raise_for_status()
        except (httpx.HTTPError, RuntimeError):
            raise RuntimeError(f"{error_prefix}: HTTP 请求失败") from None
        return _unwrap_ajax_dict(response, error_prefix)

    def close(self) -> None:
        """处理 close 相关逻辑。"""
        self.client.close()

    def __enter__(self) -> EngineClient:
        """处理   enter   相关逻辑。"""
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """处理   exit   相关逻辑。"""
        self.close()


def _unwrap_ajax_dict(response: httpx.Response, error_prefix: str) -> dict[str, Any]:
    """unwrap ajax dict 的内部辅助逻辑。"""
    try:
        body = response.json()
    except ValueError:
        raise RuntimeError(f"{error_prefix}: 响应格式错误") from None
    if (
        not isinstance(body, dict)
        or body.get("code") != 200
        or not isinstance(body.get("data"), dict)
    ):
        raise RuntimeError(f"{error_prefix}: 响应格式错误")
    return body["data"]


def _require_nonblank(value: str, label: str) -> None:
    """require nonblank 的内部辅助逻辑。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}不能为空")


def _require_text(value: str, label: str) -> str:
    """require text 的内部辅助逻辑。"""
    _require_nonblank(value, label)
    return value


def _require_dict(value: dict[str, Any] | None) -> dict[str, Any]:
    """require dict 的内部辅助逻辑。"""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("参数必须是对象")
    return value


def _ensure_idempotency_key(value: str | None) -> str:
    """ensure idempotency key 的内部辅助逻辑。"""
    if isinstance(value, str) and value.strip():
        return value
    return str(uuid4())
