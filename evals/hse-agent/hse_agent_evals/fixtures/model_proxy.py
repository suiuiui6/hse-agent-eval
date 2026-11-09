from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


MODEL = "qwen2.5:7b"
DEFAULT_UPSTREAM_URL = "http://127.0.0.1:11434/v1/chat/completions"
VALID_MODES = {
    "none",
    "model_500",
    "model_empty",
    "model_unknown_tool",
    "model_bad_arguments",
}


class _NoRedirectHandler(HTTPRedirectHandler):
    """NoRedirectHandler 的内部辅助逻辑。"""
    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        """处理 redirect request 相关逻辑。"""
        return None


def _build_direct_upstream_opener() -> Callable[..., Any]:
    """build direct upstream opener 的内部辅助逻辑。"""
    return build_opener(ProxyHandler({}), _NoRedirectHandler()).open


_DEFAULT_UPSTREAM_OPENER = _build_direct_upstream_opener()


class ModelProxyServer:
    """处理 ModelProxyServer 相关逻辑。"""
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        upstream_url: str = DEFAULT_UPSTREAM_URL,
        upstream_opener: Callable[..., Any] = _DEFAULT_UPSTREAM_OPENER,
        upstream_timeout_seconds: float = 90.0,
        accepted_socket_timeout_seconds: float = 0.5,
    ) -> None:
        """初始化实例。"""
        _require_loopback_host(host)
        _require_loopback_upstream(upstream_url)
        if not 0 <= port <= 65535:
            raise ValueError("模型代理端口无效")
        if upstream_timeout_seconds <= 0:
            raise ValueError("模型代理超时必须为正数")
        if accepted_socket_timeout_seconds <= 0:
            raise ValueError("模型代理请求读取超时必须为正数")
        self.host = host
        self._requested_port = port
        self.upstream_url = upstream_url
        self.upstream_opener = upstream_opener
        self.upstream_timeout_seconds = upstream_timeout_seconds
        self.accepted_socket_timeout_seconds = accepted_socket_timeout_seconds
        self._mode = "none"
        self._mode_lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self._http_server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        """处理 is running 相关逻辑。"""
        thread = self._thread
        return self._http_server is not None and thread is not None and thread.is_alive()

    @property
    def base_url(self) -> str:
        """处理 base url 相关逻辑。"""
        server = self._http_server
        if server is None:
            raise RuntimeError("模型代理尚未启动")
        return f"http://{self.host}:{server.server_address[1]}/v1"

    def set_mode(self, mode: str) -> None:
        """处理 set mode 相关逻辑。"""
        if mode not in VALID_MODES:
            raise ValueError(f"不支持的模型代理模式: {mode}")
        with self._mode_lock:
            self._mode = mode

    def start(self) -> ModelProxyServer:
        """处理 start 相关逻辑。"""
        with self._lifecycle_lock:
            if self.is_running:
                return self
            server = ThreadingHTTPServer(
                (self.host, self._requested_port), self._handler_type()
            )
            server.daemon_threads = False
            server.block_on_close = True
            thread = threading.Thread(
                target=server.serve_forever,
                name="hse-eval-model-proxy",
                daemon=True,
            )
            self._http_server = server
            self._thread = thread
            thread.start()
            return self

    def stop(self) -> None:
        """处理 stop 相关逻辑。"""
        with self._lifecycle_lock:
            server = self._http_server
            thread = self._thread
            if server is None:
                return
            server.shutdown()
            server.server_close()
            if thread is not None:
                thread.join(timeout=2.0)
                if thread.is_alive():
                    raise RuntimeError("模型代理未能停止")
            self._http_server = None
            self._thread = None

    def __enter__(self) -> ModelProxyServer:
        """处理   enter   相关逻辑。"""
        return self.start()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """处理   exit   相关逻辑。"""
        self.stop()

    def _current_mode(self) -> str:
        """current mode 的内部辅助逻辑。"""
        with self._mode_lock:
            return self._mode

    def _handler_type(self) -> type[BaseHTTPRequestHandler]:
        """handler type 的内部辅助逻辑。"""
        owner = self

        class Handler(BaseHTTPRequestHandler):
            """处理 Handler 相关逻辑。"""
            protocol_version = "HTTP/1.1"

            def setup(self) -> None:
                """处理 setup 相关逻辑。"""
                super().setup()
                self.connection.settimeout(owner.accepted_socket_timeout_seconds)

            def __getattr__(self, name: str) -> Any:
                """处理   getattr   相关逻辑。"""
                if name.startswith("do_"):
                    return self._method_not_allowed
                raise AttributeError(name)

            def _discard_request_body(self) -> None:
                """discard request body 的内部辅助逻辑。"""
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    return
                if 0 < content_length <= 4 * 1024 * 1024:
                    try:
                        self.rfile.read(content_length)
                    except OSError:
                        return

            def _method_not_allowed(self) -> None:
                """method not allowed 的内部辅助逻辑。"""
                self._discard_request_body()
                path = urlsplit(self.path).path
                if path == "/v1/chat/completions":
                    self._send_json(405, {"error": "method_not_allowed"})
                else:
                    self._send_json(404, {"error": "not_found"})

            def do_GET(self) -> None:
                """处理 do GET 相关逻辑。"""
                path = urlsplit(self.path).path
                if path == "/v1/chat/completions":
                    self._send_json(405, {"error": "method_not_allowed"})
                else:
                    self._send_json(404, {"error": "not_found"})

            def do_POST(self) -> None:
                """处理 do POST 相关逻辑。"""
                path = urlsplit(self.path).path
                if path != "/v1/chat/completions":
                    self._discard_request_body()
                    self._send_json(404, {"error": "not_found"})
                    return
                payload = self._read_json()
                if payload is None:
                    return
                mode = owner._current_mode()
                if mode != "none":
                    status, response = _failure_response(mode)
                    self._send_json(status, response)
                    return
                payload["model"] = MODEL
                request_headers = {"Content-Type": "application/json"}
                authorization = self.headers.get("Authorization")
                if authorization:
                    request_headers["Authorization"] = authorization
                accept = self.headers.get("Accept")
                if accept:
                    request_headers["Accept"] = accept
                request = Request(
                    owner.upstream_url,
                    data=json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8"),
                    headers=request_headers,
                    method="POST",
                )
                try:
                    with owner.upstream_opener(
                        request, timeout=owner.upstream_timeout_seconds
                    ) as upstream:
                        self._send_bytes(
                            int(upstream.status),
                            upstream.read(),
                            upstream.headers.get(
                                "Content-Type", "application/octet-stream"
                            ),
                            upstream.headers.get("Location"),
                        )
                except HTTPError as error:
                    try:
                        body = error.read()
                    except (AttributeError, OSError):
                        body = b""
                    content_type = (
                        error.headers.get("Content-Type")
                        if error.headers is not None
                        else None
                    )
                    self._send_bytes(
                        error.code,
                        body,
                        content_type or "application/octet-stream",
                        error.headers.get("Location")
                        if error.headers is not None
                        else None,
                    )
                except (URLError, TimeoutError, OSError):
                    self._send_json(
                        502,
                        {
                            "error": {
                                "message": "模型上游不可用",
                                "type": "upstream_error",
                            }
                        },
                    )

            def _read_json(self) -> dict[str, Any] | None:
                """read json 的内部辅助逻辑。"""
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self._send_json(400, {"error": "invalid_content_length"})
                    return None
                if content_length <= 0 or content_length > 4 * 1024 * 1024:
                    self._send_json(400, {"error": "invalid_json"})
                    return None
                try:
                    payload = json.loads(self.rfile.read(content_length))
                except OSError:
                    self.close_connection = True
                    return None
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._send_json(400, {"error": "invalid_json"})
                    return None
                if not isinstance(payload, dict):
                    self._send_json(400, {"error": "invalid_json"})
                    return None
                return payload

            def _send_json(self, status: int, payload: dict[str, Any]) -> None:
                """send json 的内部辅助逻辑。"""
                self._send_bytes(
                    status,
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8"),
                    "application/json; charset=utf-8",
                )

            def _send_bytes(
                self,
                status: int,
                body: bytes,
                content_type: str,
                location: str | None = None,
            ) -> None:
                """send bytes 的内部辅助逻辑。"""
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                if location is not None:
                    self.send_header("Location", location)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                self.close_connection = True

            def log_message(self, format: str, *args: Any) -> None:
                """处理 log message 相关逻辑。"""
                return

        return Handler


def _failure_response(mode: str) -> tuple[int, dict[str, Any]]:
    """failure response 的内部辅助逻辑。"""
    if mode == "model_500":
        return 500, {
            "error": {
                "message": "固定模型故障",
                "type": "server_error",
            }
        }
    if mode == "model_empty":
        return 200, {
            "id": "chatcmpl-fixture-empty",
            "object": "chat.completion",
            "created": 0,
            "model": MODEL,
            "choices": [],
            "usage": _empty_usage(),
        }
    function = {
        "name": "unknown_hse_tool" if mode == "model_unknown_tool" else "create_task",
        "arguments": "{}" if mode == "model_unknown_tool" else "{invalid-json",
    }
    return 200, {
        "id": f"chatcmpl-fixture-{mode}",
        "object": "chat.completion",
        "created": 0,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call-fixture-{mode}",
                            "type": "function",
                            "function": function,
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": _empty_usage(),
    }


def _empty_usage() -> dict[str, int]:
    """empty usage 的内部辅助逻辑。"""
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _require_loopback_host(host: str) -> None:
    """require loopback host 的内部辅助逻辑。"""
    try:
        is_loopback = host.casefold() == "localhost" or ip_address(host).is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise ValueError("Fixture 服务必须绑定回环地址")


def _require_loopback_upstream(upstream_url: str) -> None:
    """require loopback upstream 的内部辅助逻辑。"""
    parsed = urlsplit(upstream_url)
    try:
        port = parsed.port
        hostname = parsed.hostname
        is_loopback = hostname is not None and (
            hostname.casefold() == "localhost" or ip_address(hostname).is_loopback
        )
    except ValueError:
        port = None
        is_loopback = False
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or not is_loopback
        or parsed.username is not None
        or parsed.password is not None
        or (port is not None and not 1 <= port <= 65535)
        or parsed.path != "/v1/chat/completions"
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("模型上游必须是回环 HTTP chat completions 地址")
