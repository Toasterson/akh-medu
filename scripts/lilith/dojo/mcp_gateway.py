"""Loopback MCP gateway over the in-process replay servers (Anima WI 01a0e270-475e, plan slice C4).

The teacher path (`acp_teacher.py`) runs the model inside an akh, which calls tools itself over MCP
instead of handing tool calls back to the runner. This gateway lets it do that WITHOUT a second copy
of the replay servers: it serves the SAME `ReplayServer` / `DojoGitServer` objects the in-process
broker uses, in this process, over MCP streamable HTTP, so

- the receipts land in the episode's one `ReceiptLog` (the `Episode` carries them unmodified);
- every result goes through `runner.Broker.call`, so the akh's model reads the exact bytes a
  model driven by the in-process loop reads (`{"isError", "_meta", "result"}`, receipts in `_meta`);
- `cos_declaration` is stripped by the same code before the server sees the call.

One gateway serves many episodes. Each episode gets its own unguessable token; its servers are
reachable only at `http://127.0.0.1:<port>/dojo/<token>/<server>` with `Authorization: Bearer
<token>`, and only until the episode closes (a closed or unknown token answers 404). The gateway
binds loopback only.

Wire: one JSON-RPC request per POST, answered as `application/json` (the akh's MCP client accepts
JSON or SSE). Notifications answer 202 with no body. `Mcp-Session-Id` is not required.
"""
from __future__ import annotations

import dataclasses
import hmac
import http.server
import json
import secrets
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dojo.errors import DojoError
from dojo.runner import Broker, Dispatched

LOOPBACK = "127.0.0.1"
PATH_PREFIX = "/dojo/"
MAX_BODY = 4 * 1024 * 1024
MCP_PROTOCOL_VERSION = "2025-06-18"


class GatewayError(DojoError):
    code = "lilith::dojo::mcp_gateway"
    help = "the gateway binds 127.0.0.1 only; check that no other process holds the port and that the episode is open"


@dataclasses.dataclass(frozen=True)
class GatewayCall:
    """One `tools/call` the gateway routed (in arrival order)."""

    server: str
    tool: str
    dispatched: Dispatched
    at: float  # the gateway clock when the call arrived


class EpisodeTools:
    """The tool surface one episode exposes through the gateway."""

    def __init__(self, gateway: "ReplayMcpGateway", token: str, broker: Broker):
        self.gateway = gateway
        self.token = token
        self.broker = broker
        self.calls: list[GatewayCall] = []
        self.rejected: list[str] = []  # requests that reached the episode but named no replay tool
        self._lock = threading.Lock()
        self.closed = False

    @property
    def servers(self) -> tuple[str, ...]:
        return tuple(self.broker.servers)

    def url(self, server: str) -> str:
        return f"http://{LOOPBACK}:{self.gateway.port}{PATH_PREFIX}{self.token}/{server}"

    def acp_mcp_servers(self) -> list[dict]:
        """ACP `NewSessionRequest.mcpServers`: one streamable-HTTP entry per replay server. The akh
        names its proxy tools `mcp__<name>__<tool>`, the names the in-process broker uses."""
        return [{"type": "http", "name": server, "url": self.url(server),
                 "headers": [{"name": "Authorization", "value": f"Bearer {self.token}"}]}
                for server in self.servers]

    def close(self) -> None:
        self.gateway._close(self.token)

    def __enter__(self) -> "EpisodeTools":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- JSON-RPC ------------------------------------------------------------------------------

    def handle(self, server: str, request: Mapping[str, Any]) -> dict | None:
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}
        if request_id is None:
            return None  # notifications (notifications/initialized, cancelled, ...)
        if method == "initialize":
            return _ok(request_id, {"protocolVersion": str(params.get("protocolVersion") or MCP_PROTOCOL_VERSION),
                                    "capabilities": {"tools": {"listChanged": False}},
                                    "serverInfo": {"name": f"lilith-dojo-{server}", "version": "1"}})
        if method == "ping":
            return _ok(request_id, {})
        if method == "tools/list":
            return _ok(request_id, {"tools": self.broker.server_tools(server)})
        if method == "tools/call":
            name, arguments = params.get("name"), params.get("arguments")
            if arguments is None:
                arguments = {}
            if not isinstance(name, str) or not isinstance(arguments, dict):
                return _err(request_id, -32602, "tools/call needs a name and an arguments object")
            dispatched = self.broker.call(server, name, arguments)
            with self._lock:
                if dispatched.receipt_id is None:
                    self.rejected.append(f"{server}/{name}: {dispatched.problem}")
                self.calls.append(GatewayCall(server, name, dispatched, self.gateway.clock()))
            meta = json.loads(dispatched.text).get("_meta") or {}
            result: dict[str, Any] = {"content": [{"type": "text", "text": dispatched.text}],
                                      "isError": dispatched.is_error}
            if meta:
                result["_meta"] = meta
            return _ok(request_id, result)
        return _err(request_id, -32601, f"method {method!r} not supported by the dojo gateway")


def _ok(request_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _err(request_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


class ReplayMcpGateway:
    """A loopback HTTP server that routes `/dojo/<token>/<server>` to one open episode."""

    def __init__(self, port: int = 0, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self._episodes: dict[str, EpisodeTools] = {}
        self._lock = threading.Lock()
        gateway = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args) -> None:  # the token is in the path; never log it
                return

            def _send(self, status: int, body: dict | None) -> None:
                raw = b"" if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                if body is not None:
                    self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                if raw:
                    self.wfile.write(raw)

            def do_GET(self) -> None:  # no server-initiated stream
                self._send(405, {"error": "the dojo gateway answers POST only"})

            def do_DELETE(self) -> None:
                self._send(405, {"error": "the dojo gateway has no sessions to delete"})

            def do_POST(self) -> None:
                found = gateway._route(self.path, self.headers.get("Authorization"))
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_BODY:
                    self._send(413, {"error": "request too large"})
                    return
                raw = self.rfile.read(length) if length else b""
                if found is None:
                    self._send(404, {"error": "no open dojo episode at this path"})
                    return
                episode, server = found
                try:
                    request = json.loads(raw or b"null")
                except ValueError:
                    self._send(400, _err(None, -32700, "parse error"))
                    return
                if not isinstance(request, dict):
                    self._send(400, _err(None, -32600, "batch requests are not supported"))
                    return
                response = episode.handle(server, request)
                self._send(202, None) if response is None else self._send(200, response)

        self._server = http.server.ThreadingHTTPServer((LOOPBACK, port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="dojo-mcp-gateway", daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def open(self, broker: Broker) -> EpisodeTools:
        token = secrets.token_urlsafe(32)
        episode = EpisodeTools(self, token, broker)
        with self._lock:
            self._episodes[token] = episode
        return episode

    def _close(self, token: str) -> None:
        with self._lock:
            episode = self._episodes.pop(token, None)
        if episode is not None:
            episode.closed = True

    def _route(self, path: str, authorization: str | None) -> tuple[EpisodeTools, str] | None:
        if not path.startswith(PATH_PREFIX):
            return None
        token, _, server = path[len(PATH_PREFIX):].partition("/")
        server = server.split("?", 1)[0].rstrip("/")
        with self._lock:
            episode = self._episodes.get(token)
        if episode is None or not episode.broker.has_server(server):
            return None
        if not hmac.compare_digest(authorization or "", f"Bearer {token}"):
            return None
        return episode, server

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self) -> "ReplayMcpGateway":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()
