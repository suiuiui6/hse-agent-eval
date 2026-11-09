import json

import httpx
import pytest

from hse_agent_evals.adapters.engine_client import EngineClient


CHAT_DATA = {
    "reply": "共有 2 项待办",
    "status": "success",
    "toolCalls": [
        {
            "tool": "get_my_tasks",
            "args": {},
            "level": "L1",
            "result": "[]",
            "roundIndex": 1,
        }
    ],
    "roundCount": 2,
    "promptTokens": 80,
    "completionTokens": 30,
    "tokens": 110,
    "durationMs": 900,
}


def test_chat_unwraps_ajax_result_data_and_sends_bearer_token():
    seen_requests = []

    def handler(request):
        seen_requests.append(request)
        return httpx.Response(
            200,
            json={"code": 200, "msg": "操作成功", "data": CHAT_DATA},
        )

    transport = httpx.MockTransport(handler)
    with EngineClient(
        base_url="http://127.0.0.1:18081",
        transport=transport,
        timeout=2.0,
    ) as client:
        result = client.chat("eval-jwt-token", "查询我的待办", "session-001", "idem-001")

    assert result == CHAT_DATA
    request = seen_requests[0]
    assert request.url.path == "/hse/agent/chat"
    assert request.headers["Authorization"] == "Bearer eval-jwt-token"
    assert json.loads(request.content) == {
        "message": "查询我的待办",
        "sessionId": "session-001",
        "idempotencyKey": "idem-001",
    }


def test_binding_reads_backend_database_metadata():
    def handler(request):
        assert request.method == "GET"
        assert request.url.path == "/hse/agent/eval/binding"
        assert request.headers["Authorization"] == "Bearer eval-jwt-token"
        return httpx.Response(
            200,
            json={
                "code": 200,
                "msg": "操作成功",
                "data": {
                    "database": "engine_agent_eval_test",
                    "profile": "agent-eval",
                },
            },
        )

    with EngineClient(
        base_url="http://127.0.0.1:18081",
        transport=httpx.MockTransport(handler),
        timeout=2.0,
    ) as client:
        result = client.binding("eval-jwt-token")

    assert result == {
        "database": "engine_agent_eval_test",
        "profile": "agent-eval",
    }


def test_login_returns_token_and_supports_injected_httpx_client():
    def handler(request):
        assert request.url.path == "/login"
        assert json.loads(request.content) == {
            "username": "hse_user",
            "password": "eval-password",
        }
        return httpx.Response(
            200,
            json={"code": 200, "msg": "操作成功", "token": "signed-token"},
        )

    http_client = httpx.Client(
        base_url="http://127.0.0.1:18081",
        transport=httpx.MockTransport(handler),
    )
    client = EngineClient(client=http_client)

    assert client.login("hse_user", "eval-password") == "signed-token"
    client.close()
    assert http_client.is_closed


@pytest.mark.parametrize("token", [0, False, [], {}])
def test_login_rejects_non_string_token(token):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={"code": 200, "msg": "操作成功", "token": token},
        )
    )
    client = EngineClient(
        base_url="http://127.0.0.1:18081",
        transport=transport,
    )

    with pytest.raises(RuntimeError, match="登录失败"):
        client.login("hse_user", "eval-password")

    client.close()


@pytest.mark.parametrize(
    ("username", "password"),
    [("", "password"), (" ", "password"), ("user", ""), ("user", " ")],
)
def test_login_rejects_blank_credentials_before_request(username, password):
    calls = []
    transport = httpx.MockTransport(lambda request: calls.append(request))
    client = EngineClient(
        base_url="http://127.0.0.1:18081",
        transport=transport,
    )

    with pytest.raises(ValueError, match="用户名|密码"):
        client.login(username, password)

    client.close()
    assert calls == []


@pytest.mark.parametrize(
    ("token", "message", "session_id", "error_text"),
    [
        ("", "message", "session", "令牌"),
        ("token", " ", "session", "消息"),
        ("token", "message", "", "会话"),
    ],
)
def test_chat_rejects_blank_security_inputs_before_request(
    token, message, session_id, error_text
):
    calls = []
    transport = httpx.MockTransport(lambda request: calls.append(request))
    client = EngineClient(
        base_url="http://127.0.0.1:18081",
        transport=transport,
    )

    with pytest.raises(ValueError, match=error_text):
        client.chat(token, message, session_id)

    client.close()
    assert calls == []


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, json={"code": 401, "msg": "bad-password-secret"}),
        httpx.Response(200, json={"code": 500, "msg": "bad-password-secret"}),
        httpx.Response(200, json={"code": 200, "msg": "ok"}),
        httpx.Response(200, content=b"not-json"),
        httpx.Response(200, json=[]),
    ],
)
def test_login_rejects_http_and_malformed_ajax_results_without_exposing_password(
    response,
):
    password = "bad-password-secret"
    transport = httpx.MockTransport(lambda request: response)
    client = EngineClient(
        base_url="http://127.0.0.1:18081",
        transport=transport,
    )

    with pytest.raises(RuntimeError, match="登录失败") as exc_info:
        client.login("hse_user", password)

    client.close()
    assert password not in str(exc_info.value)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, json={"code": 500, "msg": "eval-jwt-secret"}),
        httpx.Response(200, json={"code": 500, "msg": "eval-jwt-secret"}),
        httpx.Response(200, json={"code": 200, "data": []}),
        httpx.Response(200, content=b"not-json"),
        httpx.Response(200, json=[]),
    ],
)
def test_chat_rejects_http_and_malformed_ajax_results_without_exposing_token(
    response,
):
    token = "eval-jwt-secret"
    transport = httpx.MockTransport(lambda request: response)
    client = EngineClient(
        base_url="http://127.0.0.1:18081",
        transport=transport,
    )

    with pytest.raises(RuntimeError, match="Agent 调用失败") as exc_info:
        client.chat(token, "查询待办", "session-001")

    client.close()
    assert token not in str(exc_info.value)


def test_engine_client_never_echoes_untrusted_ajax_result_messages():
    untrusted_message = "password=remote-password-secret jwt=remote-jwt-secret"

    with (
        EngineClient(
            base_url="http://127.0.0.1:18081",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json={"code": 500, "msg": untrusted_message}
                )
            ),
        ) as login_client,
        EngineClient(
            base_url="http://127.0.0.1:18081",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json={"code": 500, "msg": untrusted_message}
                )
            ),
        ) as chat_client,
    ):
        with pytest.raises(RuntimeError) as login_error:
            login_client.login("hse_user", "local-password-secret")
        with pytest.raises(RuntimeError) as chat_error:
            chat_client.chat("local-jwt-secret", "查询待办", "session-001")

    assert str(login_error.value) == "登录失败: 服务端拒绝请求"
    assert str(chat_error.value) == "Agent 调用失败: 响应格式错误"
    assert untrusted_message not in str(login_error.value)
    assert untrusted_message not in str(chat_error.value)


def test_transport_failure_is_wrapped_without_exposing_secrets():
    password = "transport-password-secret"
    token = "transport-token-secret"

    def handler(request):
        raise httpx.ConnectError("dependency unavailable", request=request)

    client = EngineClient(
        base_url="http://127.0.0.1:18081",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(RuntimeError, match="登录失败") as login_error:
        client.login("hse_user", password)
    with pytest.raises(RuntimeError, match="Agent 调用失败") as chat_error:
        client.chat(token, "查询待办", "session-001")

    client.close()
    assert password not in str(login_error.value)
    assert token not in str(chat_error.value)


def test_context_manager_closes_owned_client():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, json={"code": 200, "msg": "ok", "token": "token"}
        )
    )
    client = EngineClient(
        base_url="http://127.0.0.1:18081",
        transport=transport,
    )

    with client as active:
        assert active is client
        assert active.login("user", "password") == "token"

    assert client.client.is_closed


def test_confirm_preserves_all_server_binding_fields():
    seen_requests = []
    data = {"status": "success", "tool": "create_task", "data": {"taskId": 12}}

    def handler(request):
        seen_requests.append(request)
        return httpx.Response(
            200,
            json={"code": 200, "msg": "操作成功", "data": data},
        )

    with EngineClient(
        base_url="http://127.0.0.1:18081",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = client.confirm(
            "eval-jwt-token",
            "confirm-001",
            "create_task",
            {"deptId": 9, "taskName": "高处作业检查"},
            "session-001",
            "turn-001",
        )

    assert result == data
    request = seen_requests[0]
    assert request.url.path == "/hse/agent/confirm"
    assert request.headers["Authorization"] == "Bearer eval-jwt-token"
    assert json.loads(request.content) == {
        "confirmationToken": "confirm-001",
        "toolName": "create_task",
        "args": {"deptId": 9, "taskName": "高处作业检查"},
        "sessionId": "session-001",
        "turnId": "turn-001",
    }


def test_cancel_preserves_all_server_binding_fields():
    seen_requests = []
    data = {"status": "cancelled", "tool": "create_task", "meta": {"toolCallId": "call-1"}}

    def handler(request):
        seen_requests.append(request)
        return httpx.Response(
            200,
            json={"code": 200, "msg": "操作成功", "data": data},
        )

    with EngineClient(
        base_url="http://127.0.0.1:18081",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = client.cancel(
            "eval-jwt-token",
            "confirm-001",
            "create_task",
            {"deptId": 9, "taskName": "高处作业检查"},
            "session-001",
            "turn-001",
        )

    assert result == data
    request = seen_requests[0]
    assert request.url.path == "/hse/agent/cancel"
    assert request.headers["Authorization"] == "Bearer eval-jwt-token"
    assert json.loads(request.content) == {
        "confirmationToken": "confirm-001",
        "toolName": "create_task",
        "args": {"deptId": 9, "taskName": "高处作业检查"},
        "sessionId": "session-001",
        "turnId": "turn-001",
    }


def test_clear_posts_session_binding_and_unwraps_data():
    seen_requests = []
    data = {"status": "deleted", "sessionId": "session-001"}

    def handler(request):
        seen_requests.append(request)
        return httpx.Response(
            200,
            json={"code": 200, "msg": "操作成功", "data": data},
        )

    with EngineClient(
        base_url="http://127.0.0.1:18081",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = client.clear("eval-jwt-token", "session-001")

    assert result == data
    request = seen_requests[0]
    assert request.url.path == "/hse/agent/clear"
    assert request.headers["Authorization"] == "Bearer eval-jwt-token"
    assert json.loads(request.content) == {"sessionId": "session-001"}
