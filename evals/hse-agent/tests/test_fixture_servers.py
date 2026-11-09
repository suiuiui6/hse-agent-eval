import importlib
import json
import socket
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError

import httpx
import pytest

from hse_agent_evals.fixtures.graphrag_server import GraphRAGFixtureServer
from hse_agent_evals.fixtures.model_proxy import ModelProxyServer
import hse_agent_evals.fixtures.model_proxy as model_proxy_module


EVAL_ROOT = Path(__file__).resolve().parents[1]
RESPONSES_PATH = EVAL_ROOT / "fixtures" / "graphrag" / "responses.json"


class UpstreamResponse:
    """处理 UpstreamResponse 相关逻辑。"""
    def __init__(
        self,
        body=b'{"id":"chatcmpl-upstream","choices":[{"message":{"content":"ok"}}]}',
        status=201,
        content_type="application/json; charset=utf-8",
    ):
        """初始化实例。"""
        self.body = body
        self.status = status
        self.headers = {"Content-Type": content_type}

    def read(self):
        """处理 read 相关逻辑。"""
        return self.body

    def __enter__(self):
        """处理   enter   相关逻辑。"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """处理   exit   相关逻辑。"""
        return False


@contextmanager
def serve_local_http(handler_type):
    """处理 serve local http 相关逻辑。"""
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_type)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def wait_until(condition, description, timeout_seconds=3.0):
    """处理 wait until 相关逻辑。"""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError(f"等待{description}超时")


def test_graphrag_serves_stable_health_capabilities_and_keyword_evidence():
    """验证 graphrag serves stable health capabilities and keyword evidence 场景。"""
    with GraphRAGFixtureServer(responses_path=RESPONSES_PATH, port=0) as server:
        with httpx.Client(base_url=server.base_url) as client:
            health = client.get("/api/v1/health")
            capabilities = client.get("/api/v1/capabilities")
            result = client.post(
                "/api/v1/rag/query",
                json={
                    "question": "动火作业应当采取哪些安全措施？",
                    "top_k": 3,
                    "mode": "evidence",
                    "include_sources": True,
                },
            )

    assert health.status_code == 200
    assert health.json() == {
        "status": "healthy",
        "provider": "hse-agent-eval-graphrag",
        "version": "1.0",
    }
    assert capabilities.status_code == 200
    assert capabilities.json() == {
        "capabilities": ["rag.query", "evidence.sources"],
        "query_modes": ["evidence"],
    }
    assert result.status_code == 200
    body = result.json()
    assert body["metadata"]["query_id"] == "fixture-hot-work"
    assert "动火" in body["answer"]
    assert body["sources"]
    assert body["sources"][0]["snippet"]
    assert body["sources"][0]["source_file"]
    assert body["sources"][0]["relevance_score"] > 0


def test_graphrag_accepts_chunked_utf8_json_from_java_style_clients():
    """验证 graphrag accepts chunked utf8 json from java style clients 场景。"""
    payload = json.dumps(
        {"question": "动火作业应当采取哪些安全措施？", "top_k": 3},
        ensure_ascii=False,
    ).encode("utf-8")
    with GraphRAGFixtureServer(responses_path=RESPONSES_PATH, port=0) as server:
        response = httpx.post(
            server.base_url + "/api/v1/rag/query",
            content=iter([payload[:17], payload[17:]]),
            headers={"Content-Type": "application/json; charset=utf-8"},
        )

    assert response.status_code == 200
    assert response.json()["metadata"]["query_id"] == "fixture-hot-work"


def test_graphrag_matches_query_alias_and_returns_deterministic_no_match():
    """验证 graphrag matches query alias and returns deterministic no match 场景。"""
    with GraphRAGFixtureServer(responses_path=RESPONSES_PATH, port=0) as server:
        with httpx.Client(base_url=server.base_url) as client:
            matched = client.post(
                "/api/v1/rag/query",
                json={"query": "高处作业的防护要求"},
            ).json()
            unmatched = client.post(
                "/api/v1/rag/query",
                json={"question": "完全没有对应条目的问题"},
            ).json()

    assert matched["metadata"]["query_id"] == "fixture-working-at-height"
    assert matched["sources"][0]["source_file"] == "demo-height-work.md"
    assert unmatched == {
        "answer": "未检索到匹配的演示条目。",
        "sources": [],
        "metadata": {"query_id": "fixture-no-match"},
    }


def test_graphrag_rejects_unknown_paths_methods_and_malformed_json():
    """验证 graphrag rejects unknown paths methods and malformed json 场景。"""
    with GraphRAGFixtureServer(responses_path=RESPONSES_PATH, port=0) as server:
        with httpx.Client(base_url=server.base_url) as client:
            assert client.get("/unknown").status_code == 404
            assert client.post("/api/v1/health", json={}).status_code == 405
            assert client.get("/api/v1/rag/query").status_code == 405
            malformed = client.post(
                "/api/v1/rag/query",
                content=b"{",
                headers={"Content-Type": "application/json"},
            )
            missing_query = client.post("/api/v1/rag/query", json={})

    assert malformed.status_code == 400
    assert missing_query.status_code == 400


@pytest.mark.parametrize("method", ["PUT", "DELETE", "PATCH", "OPTIONS", "TRACE", "HEAD"])
def test_graphrag_returns_json_405_for_every_unsupported_method(method):
    """验证 graphrag returns json 405 for every unsupported method 场景。"""
    with GraphRAGFixtureServer(responses_path=RESPONSES_PATH, port=0) as server:
        response = httpx.request(
            method,
            server.base_url + "/api/v1/rag/query",
            content=b"{}",
        )

    assert response.status_code == 405
    assert response.headers["content-type"] == "application/json; charset=utf-8"
    assert response.headers["connection"] == "close"
    if method == "HEAD":
        assert response.content == b""
        assert int(response.headers["content-length"]) > 0
    else:
        assert response.json() == {"error": "method_not_allowed"}
        assert int(response.headers["content-length"]) == len(response.content)


def test_graphrag_timeout_waits_past_read_timeout_then_closes_connection():
    """验证 graphrag timeout waits past read timeout then closes connection 场景。"""
    server = GraphRAGFixtureServer(
        responses_path=RESPONSES_PATH,
        port=0,
        java_read_timeout_seconds=0.04,
        timeout_margin_seconds=0.02,
    )
    with server:
        server.set_mode("graphrag_timeout")
        started = time.monotonic()
        with httpx.Client(base_url=server.base_url, timeout=0.5) as client:
            with pytest.raises(httpx.RemoteProtocolError):
                client.post("/api/v1/rag/query", json={"question": "动火"})
        elapsed = time.monotonic() - started

    assert elapsed >= 0.04


def test_graphrag_lifecycle_is_reusable_and_mode_can_reset():
    """验证 graphrag lifecycle is reusable and mode can reset 场景。"""
    server = GraphRAGFixtureServer(responses_path=RESPONSES_PATH, port=0)
    server.start()
    first_url = server.base_url
    server.set_mode("graphrag_timeout")
    server.set_mode("none")
    assert httpx.get(first_url + "/api/v1/health").status_code == 200
    server.stop()
    assert not server.is_running


def test_graphrag_stop_interrupts_inflight_timeout_request():
    """验证 graphrag stop interrupts inflight timeout request 场景。"""
    server = GraphRAGFixtureServer(
        responses_path=RESPONSES_PATH,
        port=0,
        java_read_timeout_seconds=5.0,
        timeout_margin_seconds=0.1,
    )
    handler_entered = threading.Event()
    request_finished = threading.Event()
    request_errors = []
    original_current_mode = server._current_mode

    def observed_current_mode():
        """处理 observed current mode 相关逻辑。"""
        mode = original_current_mode()
        if mode == "graphrag_timeout":
            handler_entered.set()
        return mode

    def request_timeout(url):
        """处理 request timeout 相关逻辑。"""
        try:
            httpx.post(
                url,
                json={"question": "动火"},
                timeout=10.0,
            )
        except httpx.HTTPError:
            pass
        except BaseException as error:
            request_errors.append(error)
        finally:
            request_finished.set()

    server._current_mode = observed_current_mode
    server.start()
    server.set_mode("graphrag_timeout")
    requester = threading.Thread(
        target=request_timeout,
        args=(server.base_url + "/api/v1/rag/query",),
    )
    try:
        requester.start()
        assert handler_entered.wait(10.0)
        server.stop()
        assert request_finished.wait(3.0)
    finally:
        server.stop()
        requester.join(timeout=12.0)

    assert not requester.is_alive()
    assert request_errors == []

    server.start()
    try:
        assert server.is_running
        assert httpx.get(server.base_url + "/api/v1/health").status_code == 200
    finally:
        server.stop()
        server.stop()
    assert not server.is_running


def test_graphrag_stop_waits_for_normal_query_handler_and_remains_reusable():
    """验证 graphrag stop waits for normal query handler and remains reusable 场景。"""
    server = GraphRAGFixtureServer(responses_path=RESPONSES_PATH, port=0)
    entered = threading.Event()
    release = threading.Event()
    request_finished = threading.Event()
    stop_finished = threading.Event()
    responses = []
    errors = []

    def blocking_match(query):
        """处理 blocking match 相关逻辑。"""
        entered.set()
        release.wait(timeout=5.0)
        return {
            "answer": "固定回答",
            "sources": [],
            "metadata": {"query_id": "blocking-query"},
        }

    def request_query(url):
        """处理 request query 相关逻辑。"""
        try:
            responses.append(
                httpx.post(url, json={"question": "动火"}, timeout=5.0)
            )
        except BaseException as error:
            errors.append(error)
        finally:
            request_finished.set()

    def stop_server():
        """处理 stop server 相关逻辑。"""
        try:
            server.stop()
        finally:
            stop_finished.set()

    server._match_response = blocking_match
    server.start()
    requester = threading.Thread(
        target=request_query,
        args=(server.base_url + "/api/v1/rag/query",),
    )
    stopper = threading.Thread(target=stop_server)
    requester.start()
    stopper_started = False
    try:
        assert entered.wait(3.0)
        stopper.start()
        stopper_started = True
        stopped_while_handler_active = stop_finished.wait(1.0)
        release.set()
        assert stop_finished.wait(3.0)
        assert request_finished.wait(3.0)
    finally:
        release.set()
        server.stop()
        requester.join(timeout=3.0)
        if stopper_started:
            stopper.join(timeout=3.0)

    assert not stopped_while_handler_active
    assert not requester.is_alive()
    assert not stopper.is_alive()
    assert errors == []
    assert responses[0].status_code == 200
    server.start()
    server.stop()
    server.stop()
    assert not server.is_running


@pytest.mark.parametrize("server_type", [GraphRAGFixtureServer, ModelProxyServer])
def test_fixture_servers_reject_non_loopback_bind(server_type):
    """验证 fixture servers reject non loopback bind 场景。"""
    kwargs = {"responses_path": RESPONSES_PATH} if server_type is GraphRAGFixtureServer else {}
    with pytest.raises(ValueError, match="回环"):
        server_type(host="0.0.0.0", port=0, **kwargs)


def test_model_proxy_forwards_to_loopback_and_forces_model():
    """验证 model proxy forwards to loopback and forces model 场景。"""
    captured = []

    def opener(request, timeout):
        """处理 opener 相关逻辑。"""
        captured.append((request, timeout))
        return UpstreamResponse()

    with ModelProxyServer(port=0, upstream_opener=opener) as server:
        response = httpx.post(
            server.base_url + "/chat/completions",
            json={"model": "unsafe-model", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer proxy-client-secret"},
        )

    assert response.status_code == 201
    assert response.headers["content-type"] == "application/json; charset=utf-8"
    assert response.json()["id"] == "chatcmpl-upstream"
    request, timeout = captured[0]
    assert request.full_url == "http://127.0.0.1:11434/v1/chat/completions"
    assert json.loads(request.data)["model"] == "qwen2.5:7b"
    assert request.get_header("Authorization") == "Bearer proxy-client-secret"
    assert timeout > 0


def test_model_proxy_default_opener_does_not_follow_upstream_redirect():
    """验证 model proxy default opener does not follow upstream redirect 场景。"""
    target_requests = []
    redirect_body = b"redirect upstream body"

    class TargetHandler(BaseHTTPRequestHandler):
        """处理 TargetHandler 相关逻辑。"""
        def do_GET(self):
            """处理 do GET 相关逻辑。"""
            target_requests.append(("GET", self.path))
            body = b"target response"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            """处理 log message 相关逻辑。"""
            return

    with serve_local_http(TargetHandler) as target_url:

        class RedirectHandler(BaseHTTPRequestHandler):
            """处理 RedirectHandler 相关逻辑。"""
            def do_POST(self):
                """处理 do POST 相关逻辑。"""
                self.send_response(302)
                self.send_header("Location", target_url + "/redirect-target")
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(redirect_body)))
                self.end_headers()
                self.wfile.write(redirect_body)

            def log_message(self, format, *args):
                """处理 log message 相关逻辑。"""
                return

        with serve_local_http(RedirectHandler) as upstream_url:
            with ModelProxyServer(
                port=0,
                upstream_url=upstream_url + "/v1/chat/completions",
            ) as server:
                response = httpx.post(
                    server.base_url + "/chat/completions",
                    json={"messages": []},
                )

    assert (response.status_code, target_requests) == (302, [])
    assert response.headers["location"] == target_url + "/redirect-target"
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    assert response.content == redirect_body


def test_model_proxy_default_opener_bypasses_environment_proxy(monkeypatch):
    """验证 model proxy default opener bypasses environment proxy 场景。"""
    upstream_requests = []
    trap_requests = []

    class UpstreamHandler(BaseHTTPRequestHandler):
        """处理 UpstreamHandler 相关逻辑。"""
        def do_POST(self):
            """处理 do POST 相关逻辑。"""
            upstream_requests.append(self.path)
            self.rfile.read(int(self.headers["Content-Length"]))
            body = b'{"id":"direct-upstream","choices":[]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            """处理 log message 相关逻辑。"""
            return

    class TrapProxyHandler(BaseHTTPRequestHandler):
        """处理 TrapProxyHandler 相关逻辑。"""
        def do_POST(self):
            """处理 do POST 相关逻辑。"""
            trap_requests.append(self.path)
            body = b"proxy trap"
            self.send_response(502)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            """处理 log message 相关逻辑。"""
            return

    try:
        with serve_local_http(UpstreamHandler) as upstream_url:
            with serve_local_http(TrapProxyHandler) as trap_proxy_url:
                with monkeypatch.context() as environment:
                    environment.setenv("HTTP_PROXY", trap_proxy_url)
                    environment.setenv("HTTPS_PROXY", trap_proxy_url)
                    environment.setenv("NO_PROXY", "")
                    environment.setenv("no_proxy", "")
                    environment.delenv("ALL_PROXY", raising=False)
                    environment.delenv("all_proxy", raising=False)
                    module = importlib.reload(model_proxy_module)
                    with module.ModelProxyServer(
                        port=0,
                        upstream_url=upstream_url + "/v1/chat/completions",
                    ) as server:
                        with httpx.Client(trust_env=False) as client:
                            response = client.post(
                                server.base_url + "/chat/completions",
                                json={"messages": []},
                            )
    finally:
        importlib.reload(model_proxy_module)

    assert response.status_code == 200
    assert response.json()["id"] == "direct-upstream"
    assert upstream_requests == ["/v1/chat/completions"]
    assert trap_requests == []


def test_model_proxy_passes_timeout_as_opener_timeout_keyword():
    """验证 model proxy passes timeout as opener timeout keyword 场景。"""
    captured = []

    def opener(request, *, timeout):
        """处理 opener 相关逻辑。"""
        captured.append(timeout)
        return UpstreamResponse(status=200)

    with ModelProxyServer(
        port=0,
        upstream_opener=opener,
        upstream_timeout_seconds=1.25,
    ) as server:
        response = httpx.post(
            server.base_url + "/chat/completions",
            json={"messages": []},
        )

    assert response.status_code == 200
    assert captured == [1.25]


@pytest.mark.parametrize(
    "upstream_url",
    [
        "http://192.168.1.8:11434/v1/chat/completions",
        "https://example.com/v1/chat/completions",
        "ftp://127.0.0.1:11434/v1/chat/completions",
    ],
)
def test_model_proxy_rejects_non_loopback_or_non_http_upstream(upstream_url):
    """验证 model proxy rejects non loopback or non http upstream 场景。"""
    with pytest.raises(ValueError, match="回环"):
        ModelProxyServer(port=0, upstream_url=upstream_url)


@pytest.mark.parametrize(
    ("mode", "status_code", "expected"),
    [
        ("model_500", 500, ("error", "type", "server_error")),
        ("model_empty", 200, ("choices", None, [])),
        ("model_unknown_tool", 200, ("tool", "name", "unknown_hse_tool")),
        ("model_bad_arguments", 200, ("tool", "name", "create_task")),
    ],
)
def test_model_proxy_failure_modes_are_deterministic(mode, status_code, expected):
    """验证 model proxy failure modes are deterministic 场景。"""
    upstream_calls = []

    def opener(request, timeout):
        """处理 opener 相关逻辑。"""
        upstream_calls.append(request)
        return UpstreamResponse()

    with ModelProxyServer(port=0, upstream_opener=opener) as server:
        server.set_mode(mode)
        response = httpx.post(
            server.base_url + "/chat/completions",
            json={"model": "anything", "messages": []},
        )

    assert response.status_code == status_code
    body = response.json()
    kind, key, value = expected
    if kind == "error":
        assert body["error"][key] == value
    elif kind == "choices":
        assert body[kind] == value
    else:
        function = body["choices"][0]["message"]["tool_calls"][0]["function"]
        assert function[key] == value
        if mode == "model_bad_arguments":
            with pytest.raises(json.JSONDecodeError):
                json.loads(function["arguments"])
    assert upstream_calls == []


def test_model_proxy_mode_reset_restores_real_forwarding():
    """验证 model proxy mode reset restores real forwarding 场景。"""
    calls = []

    def opener(request, timeout):
        """处理 opener 相关逻辑。"""
        calls.append(request)
        return UpstreamResponse(status=200)

    with ModelProxyServer(port=0, upstream_opener=opener) as server:
        server.set_mode("model_empty")
        empty = httpx.post(
            server.base_url + "/chat/completions", json={"messages": []}
        )
        server.set_mode("none")
        forwarded = httpx.post(
            server.base_url + "/chat/completions", json={"messages": []}
        )

    assert empty.json()["choices"] == []
    assert forwarded.json()["id"] == "chatcmpl-upstream"
    assert len(calls) == 1


def test_model_proxy_rejects_unknown_paths_methods_and_malformed_json():
    """验证 model proxy rejects unknown paths methods and malformed json 场景。"""
    with ModelProxyServer(port=0, upstream_opener=lambda request, timeout: UpstreamResponse()) as server:
        with httpx.Client(base_url=server.base_url) as client:
            assert client.post("/unknown", json={}).status_code == 404
            assert client.get("/chat/completions").status_code == 405
            malformed = client.post(
                "/chat/completions",
                content=b"{",
                headers={"Content-Type": "application/json"},
            )

    assert malformed.status_code == 400


@pytest.mark.parametrize("method", ["PUT", "DELETE", "PATCH", "OPTIONS", "TRACE", "HEAD"])
def test_model_proxy_returns_json_405_for_every_unsupported_method(method):
    """验证 model proxy returns json 405 for every unsupported method 场景。"""
    with ModelProxyServer(port=0) as server:
        response = httpx.request(
            method,
            server.base_url + "/chat/completions",
            content=b"{}",
        )

    assert response.status_code == 405
    assert response.headers["content-type"] == "application/json; charset=utf-8"
    assert response.headers["connection"] == "close"
    if method == "HEAD":
        assert response.content == b""
        assert int(response.headers["content-length"]) > 0
    else:
        assert response.json() == {"error": "method_not_allowed"}
        assert int(response.headers["content-length"]) == len(response.content)


@pytest.mark.parametrize(
    ("server_factory", "unknown_path"),
    [
        (
            lambda: GraphRAGFixtureServer(responses_path=RESPONSES_PATH, port=0),
            "/unknown",
        ),
        (lambda: ModelProxyServer(port=0), "/unknown"),
    ],
    ids=["graphrag", "model-proxy"],
)
@pytest.mark.parametrize("method", ["PUT", "DELETE", "PATCH"])
def test_fixture_servers_return_json_404_for_unknown_paths_with_any_method(
    server_factory, unknown_path, method
):
    """验证 fixture servers return json 404 for unknown paths with any method 场景。"""
    with server_factory() as server:
        response = httpx.request(method, server.base_url + unknown_path)

    assert response.status_code == 404
    assert response.headers["content-type"] == "application/json; charset=utf-8"
    assert response.headers["connection"] == "close"
    assert response.json() == {"error": "not_found"}
    assert int(response.headers["content-length"]) == len(response.content)


@pytest.mark.parametrize(
    ("server_factory", "path", "status_code", "body"),
    [
        (
            lambda: GraphRAGFixtureServer(responses_path=RESPONSES_PATH, port=0),
            "/api/v1/health",
            405,
            {"error": "method_not_allowed"},
        ),
        (
            lambda: GraphRAGFixtureServer(responses_path=RESPONSES_PATH, port=0),
            "/api/v1/capabilities",
            405,
            {"error": "method_not_allowed"},
        ),
        (
            lambda: GraphRAGFixtureServer(responses_path=RESPONSES_PATH, port=0),
            "/unknown",
            404,
            {"error": "not_found"},
        ),
        (lambda: ModelProxyServer(port=0), "/unknown", 404, {"error": "not_found"}),
    ],
    ids=["graphrag-health", "graphrag-capabilities", "graphrag-unknown", "model-proxy-unknown"],
)
def test_fixture_servers_discard_body_before_post_rejection(
    server_factory, path, status_code, body
):
    """验证 fixture servers discard body before post rejection 场景。"""
    with server_factory() as server:
        response = httpx.post(
            server.base_url + path,
            content=b"x" * (256 * 1024),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == status_code
    assert response.headers["content-type"] == "application/json; charset=utf-8"
    assert response.headers["connection"] == "close"
    assert response.json() == body
    assert int(response.headers["content-length"]) == len(response.content)


def test_model_proxy_stop_waits_for_upstream_handler_and_remains_reusable():
    """验证 model proxy stop waits for upstream handler and remains reusable 场景。"""
    entered = threading.Event()
    release = threading.Event()
    requester_ready = threading.Event()
    request_allowed = threading.Event()
    request_finished = threading.Event()
    stop_started = threading.Event()
    stop_finished = threading.Event()
    responses = []
    errors = []

    def blocking_opener(request, *, timeout):
        """处理 blocking opener 相关逻辑。"""
        entered.set()
        release.wait(timeout=5.0)
        return UpstreamResponse(status=200)

    server = ModelProxyServer(port=0, upstream_opener=blocking_opener)

    def request_completion(url):
        """处理 request completion 相关逻辑。"""
        try:
            requester_ready.set()
            if not request_allowed.wait(3.0):
                raise RuntimeError("请求线程未获准发起请求")
            responses.append(
                httpx.post(url, json={"messages": []}, timeout=5.0)
            )
        except BaseException as error:
            errors.append(error)
        finally:
            request_finished.set()

    def stop_server():
        """处理 stop server 相关逻辑。"""
        try:
            stop_started.set()
            server.stop()
        finally:
            stop_finished.set()

    server.start()
    serve_thread = server._thread
    assert serve_thread is not None
    requester = threading.Thread(
        target=request_completion,
        args=(server.base_url + "/chat/completions",),
    )
    stopper = threading.Thread(target=stop_server)
    requester_started = False
    stopper_started = False
    try:
        readiness = httpx.get(server.base_url + "/chat/completions", timeout=3.0)
        assert readiness.status_code == 405
        requester.start()
        requester_started = True
        assert requester_ready.wait(3.0)
        request_allowed.set()
        assert entered.wait(3.0)
        stopper.start()
        stopper_started = True
        assert stop_started.wait(3.0)
        wait_until(lambda: not serve_thread.is_alive(), "服务线程停止")
        assert not stop_finished.is_set()
        release.set()
        assert stop_finished.wait(3.0)
        assert request_finished.wait(3.0)
    finally:
        request_allowed.set()
        release.set()
        server.stop()
        if requester_started:
            requester.join(timeout=3.0)
        if stopper_started:
            stopper.join(timeout=3.0)

    assert not requester.is_alive()
    assert not stopper.is_alive()
    assert errors == []
    assert responses[0].status_code == 200
    server.start()
    server.stop()
    server.stop()
    assert not server.is_running


@pytest.mark.parametrize(
    ("server_factory", "path"),
    [
        (
            lambda: GraphRAGFixtureServer(responses_path=RESPONSES_PATH, port=0),
            "/api/v1/rag/query",
        ),
        (lambda: ModelProxyServer(port=0), "/v1/chat/completions"),
    ],
    ids=["graphrag", "model-proxy"],
)
def test_fixture_stop_bounds_partial_request_body_and_closes_connection(
    server_factory, path
):
    """验证 fixture stop bounds partial request body and closes connection 场景。"""
    server = server_factory()
    server.start()
    port = httpx.URL(server.base_url).port
    connection = socket.create_connection(("127.0.0.1", port), timeout=2.0)
    connection.sendall(
        f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 100\r\n\r\n{{".encode()
    )
    time.sleep(1.0)
    server.stop()
    connection.settimeout(2.0)
    try:
        remaining = connection.recv(1)
    except TimeoutError:
        remaining = None
    finally:
        connection.close()
        server.stop()

    assert remaining == b""


def test_model_proxy_propagates_upstream_http_error_status_and_content_type():
    """验证 model proxy propagates upstream http error status and content type 场景。"""
    def opener(request, timeout):
        """处理 opener 相关逻辑。"""
        raise HTTPError(
            request.full_url,
            429,
            "Too Many Requests",
            {"Content-Type": "application/problem+json"},
            None,
        )

    with ModelProxyServer(port=0, upstream_opener=opener) as server:
        response = httpx.post(
            server.base_url + "/chat/completions", json={"messages": []}
        )

    assert response.status_code == 429
    assert response.headers["content-type"] == "application/problem+json"


def test_model_proxy_rejects_unknown_mode():
    """验证 model proxy rejects unknown mode 场景。"""
    server = ModelProxyServer(port=0)
    with pytest.raises(ValueError, match="模式"):
        server.set_mode("surprise")
