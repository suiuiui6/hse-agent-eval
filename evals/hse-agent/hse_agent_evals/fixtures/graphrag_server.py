from __future__ import annotations

import copy
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


DEFAULT_RESPONSES_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures" / "graphrag" / "responses.json"
)
VALID_MODES = {"none", "graphrag_timeout"}
HEALTH_RESPONSE = {
    "status": "healthy",
    "provider": "hse-agent-eval-graphrag",
    "version": "1.0",
}
CAPABILITIES_RESPONSE = {
    "capabilities": ["rag.query", "evidence.sources"],
    "query_modes": ["evidence"],
}


class GraphRAGFixtureServer:
    def __init__(
        self,
        responses_path: str | Path = DEFAULT_RESPONSES_PATH,
        host: str = "127.0.0.1",
        port: int = 0,
        java_read_timeout_seconds: float = 30.0,
        timeout_margin_seconds: float = 1.0,
        accepted_socket_timeout_seconds: float = 0.5,
    ) -> None:
        _require_loopback_host(host)
        if not 0 <= port <= 65535:
            raise ValueError("GraphRAG fixture 端口无效")
        if java_read_timeout_seconds <= 0 or timeout_margin_seconds <= 0:
            raise ValueError("GraphRAG fixture 超时必须为正数")
        if accepted_socket_timeout_seconds <= 0:
            raise ValueError("GraphRAG fixture 请求读取超时必须为正数")
        self.host = host
        self._requested_port = port
        self.java_read_timeout_seconds = java_read_timeout_seconds
        self.timeout_margin_seconds = timeout_margin_seconds
        self.accepted_socket_timeout_seconds = accepted_socket_timeout_seconds
        self._responses, self._default_response = _load_responses(responses_path)
        self._mode = "none"
        self._mode_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._lifecycle_lock = threading.RLock()
        self._http_server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return self._http_server is not None and thread is not None and thread.is_alive()

    @property
    def base_url(self) -> str:
        server = self._http_server
        if server is None:
            raise RuntimeError("GraphRAG fixture 尚未启动")
        return f"http://{self.host}:{server.server_address[1]}"

    def set_mode(self, mode: str) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"不支持的 GraphRAG fixture 模式: {mode}")
        with self._mode_lock:
            self._mode = mode

    def start(self) -> GraphRAGFixtureServer:
        with self._lifecycle_lock:
            if self.is_running:
                return self
            self._stop_event.clear()
            handler = self._handler_type()
            server = ThreadingHTTPServer((self.host, self._requested_port), handler)
            server.daemon_threads = False
            server.block_on_close = True
            thread = threading.Thread(
                target=server.serve_forever,
                name="hse-eval-graphrag-fixture",
                daemon=True,
            )
            self._http_server = server
            self._thread = thread
            thread.start()
            return self

    def stop(self) -> None:
        with self._lifecycle_lock:
            server = self._http_server
            thread = self._thread
            if server is None:
                return
            self._stop_event.set()
            server.shutdown()
            server.server_close()
            if thread is not None:
                thread.join(timeout=2.0)
                if thread.is_alive():
                    raise RuntimeError("GraphRAG fixture 未能停止")
            self._http_server = None
            self._thread = None

    def __enter__(self) -> GraphRAGFixtureServer:
        return self.start()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.stop()

    def _current_mode(self) -> str:
        with self._mode_lock:
            return self._mode

    def _match_response(self, query: str) -> dict[str, Any]:
        normalized_query = query.casefold()
        for entry in self._responses:
            if any(keyword.casefold() in normalized_query for keyword in entry["keywords"]):
                return copy.deepcopy(entry["response"])
        return copy.deepcopy(self._default_response)

    def _handler_type(self) -> type[BaseHTTPRequestHandler]:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self) -> None:
                super().setup()
                self.connection.settimeout(owner.accepted_socket_timeout_seconds)

            def __getattr__(self, name: str) -> Any:
                if name.startswith("do_"):
                    return self._method_not_allowed
                raise AttributeError(name)

            def _discard_request_body(self) -> None:
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
                self._discard_request_body()
                path = urlsplit(self.path).path
                if path in {
                    "/api/v1/health",
                    "/api/v1/capabilities",
                    "/api/v1/rag/query",
                }:
                    self._send_json(405, {"error": "method_not_allowed"})
                else:
                    self._send_json(404, {"error": "not_found"})

            def do_GET(self) -> None:
                path = urlsplit(self.path).path
                if path == "/api/v1/health":
                    self._send_json(200, HEALTH_RESPONSE)
                elif path == "/api/v1/capabilities":
                    self._send_json(200, CAPABILITIES_RESPONSE)
                elif path == "/api/v1/rag/query":
                    self._send_json(405, {"error": "method_not_allowed"})
                else:
                    self._send_json(404, {"error": "not_found"})

            def do_POST(self) -> None:
                path = urlsplit(self.path).path
                if path in {"/api/v1/health", "/api/v1/capabilities"}:
                    self._discard_request_body()
                    self._send_json(405, {"error": "method_not_allowed"})
                    return
                if path != "/api/v1/rag/query":
                    self._discard_request_body()
                    self._send_json(404, {"error": "not_found"})
                    return
                payload = self._read_json()
                if payload is None:
                    return
                question = payload.get("question")
                if not isinstance(question, str) or not question.strip():
                    question = payload.get("query")
                if not isinstance(question, str) or not question.strip():
                    self._send_json(400, {"error": "question_required"})
                    return
                if owner._current_mode() == "graphrag_timeout":
                    owner._stop_event.wait(
                        owner.java_read_timeout_seconds + owner.timeout_margin_seconds
                    )
                    self.close_connection = True
                    try:
                        self.connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    return
                self._send_json(200, owner._match_response(question.strip()))

            def _read_json(self) -> dict[str, Any] | None:
                transfer_encoding = self.headers.get("Transfer-Encoding", "").casefold()
                if "chunked" in transfer_encoding:
                    raw_payload = self._read_chunked_body()
                    if raw_payload is None:
                        self._send_json(400, {"error": "invalid_json"})
                        return None
                else:
                    raw_payload = self._read_content_length_body()
                    if raw_payload is None:
                        return None
                try:
                    payload = json.loads(raw_payload)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._send_json(400, {"error": "invalid_json"})
                    return None
                if not isinstance(payload, dict):
                    self._send_json(400, {"error": "invalid_json"})
                    return None
                return payload

            def _read_content_length_body(self) -> bytes | None:
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self._send_json(400, {"error": "invalid_content_length"})
                    return None
                if content_length <= 0 or content_length > 1024 * 1024:
                    self._send_json(400, {"error": "invalid_json"})
                    return None
                try:
                    return self.rfile.read(content_length)
                except OSError:
                    self.close_connection = True
                    return None

            def _read_chunked_body(self) -> bytes | None:
                body = bytearray()
                try:
                    while True:
                        size_line = self.rfile.readline(128)
                        if not size_line or len(size_line) >= 128:
                            return None
                        size = int(size_line.split(b";", 1)[0].strip(), 16)
                        if size < 0 or len(body) + size > 1024 * 1024:
                            return None
                        if size == 0:
                            while True:
                                trailer = self.rfile.readline(8192)
                                if trailer in {b"\r\n", b"\n"}:
                                    return bytes(body)
                                if not trailer or len(trailer) >= 8192:
                                    return None
                        chunk = self.rfile.read(size)
                        if len(chunk) != size or self.rfile.read(2) != b"\r\n":
                            return None
                        body.extend(chunk)
                except (OSError, ValueError):
                    return None

            def _send_json(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                self.close_connection = True

            def log_message(self, format: str, *args: Any) -> None:
                return

        return Handler


def _require_loopback_host(host: str) -> None:
    try:
        is_loopback = host.casefold() == "localhost" or ip_address(host).is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise ValueError("Fixture 服务必须绑定回环地址")


def _load_responses(
    responses_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(Path(responses_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("GraphRAG fixture 响应文件格式无效")
    responses = payload.get("responses")
    default_response = payload.get("default")
    if not isinstance(responses, list) or not isinstance(default_response, dict):
        raise ValueError("GraphRAG fixture 响应文件缺少必要字段")
    for entry in responses:
        if not isinstance(entry, dict) or not isinstance(entry.get("keywords"), list):
            raise ValueError("GraphRAG fixture 关键词格式无效")
        _validate_response(entry.get("response"))
    _validate_response(default_response)
    return responses, default_response


def _validate_response(response: Any) -> None:
    if (
        not isinstance(response, dict)
        or not isinstance(response.get("answer"), str)
        or not isinstance(response.get("sources"), list)
        or not isinstance(response.get("metadata"), dict)
        or not isinstance(response["metadata"].get("query_id"), str)
    ):
        raise ValueError("GraphRAG fixture 响应格式无效")
