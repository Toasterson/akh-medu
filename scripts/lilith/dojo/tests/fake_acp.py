"""A fake akhomed ACP peer for the teacher-path tests. No network beyond 127.0.0.1.

It speaks the subset of akhomed's ACP the dojo client uses (newline-delimited JSON-RPC over TCP):
`initialize`, `_akh/tools`, `_akh/health`, `_akh/usage`, `session/new`, `session/prompt`,
`session/cancel`. During `session/prompt` it runs a *script* that plays the akh's agentic loop and
reaches the client-pushed MCP servers the way akhomed's session bridge does
(`src/acp/mcp_bridge.rs` + `src/upstream_client.rs`): `initialize`, `notifications/initialized`,
then `tools/list` / `tools/call` over HTTP with the pushed `Authorization` header, and the model reads
only the joined `content[].text` (akhomed drops `_meta` and `isError`).
"""
from __future__ import annotations

import dataclasses
import itertools
import json
import queue
import socket
import threading
import urllib.error
import urllib.request
from typing import Any, Callable

TOKEN = "dojo-teacher-token-0123456789abcdef"


@dataclasses.dataclass
class PeerState:
    own_tools: list[str] = dataclasses.field(default_factory=list)
    stale: bool = False
    mcp_http: bool = True
    symbols: int = 10
    triples: int = 20
    prompt_tokens: int = 0
    completion_tokens: int = 0
    token: str = TOKEN


class ScriptContext:
    """What a script (the fake akh's "model") can do inside one turn."""

    _ids = itertools.count(1)

    def __init__(self, peer: "FakeAkhomed", conn: "_Conn", session: dict, prompt: list[dict]):
        self.peer = peer
        self.conn = conn
        self.session = session
        self.prompt_blocks = prompt
        self.seen: list[str] = []  # every tool result text the "model" read

    @property
    def prompt_text(self) -> str:
        return "\n".join(b.get("text", "") for b in self.prompt_blocks)

    @property
    def situation(self) -> dict:
        text = self.prompt_text
        return json.loads(text.split("Situation (JSON):\n", 1)[1].split("\n", 1)[0])

    def _server(self, name: str) -> dict:
        return next(s for s in self.session["mcpServers"] if s["name"] == name)

    def _post(self, server: dict, body: dict) -> tuple[int, Any]:
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        headers.update({h["name"]: h["value"] for h in server.get("headers", [])})
        request = urllib.request.Request(server["url"], data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                raw = response.read()
                return response.status, (json.loads(raw) if raw else None)
        except urllib.error.HTTPError as error:
            return error.code, None

    def list_tools(self, server_name: str) -> list[dict]:
        server = self._server(server_name)
        self._post(server, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}})
        self._post(server, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        _, body = self._post(server, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        return body["result"]["tools"]

    def call(self, name: str, arguments: dict, *, text: str = "") -> str:
        """One proxied tool call: narrate, tool_call notification, HTTP call, tool_call_update."""
        if text:
            self.say(text)
        _, server_name, tool = name.split("__", 2)
        server = self._server(server_name)
        call_id = f"{name}_{next(self._ids)}"
        self.update({"sessionUpdate": "tool_call", "toolCallId": call_id, "title": tool, "kind": "execute",
                     "status": "in_progress"})
        self._post(server, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        self._post(server, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        status, body = self._post(server, {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                           "params": {"name": tool, "arguments": arguments}})
        if status != 200 or body is None or "result" not in body:
            text_out = f"HTTP {status}"
        else:
            text_out = "\n".join(c.get("text", "") for c in body["result"].get("content", []))
        self.seen.append(text_out)
        self.update({"sessionUpdate": "tool_call_update", "toolCallId": call_id, "status": "completed",
                     "content": [{"type": "content", "content": {"type": "text", "text": text_out[:200]}}]})
        self.update({"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": f"[{name}:ok] ...\n"}})
        return text_out

    def start_foreign_tool(self, name: str) -> None:
        """The akh starts a tool that is not a replay tool (what a full akh could do)."""
        self.update({"sessionUpdate": "tool_call", "toolCallId": f"{name}_{next(self._ids)}", "title": name,
                     "kind": "fetch", "status": "in_progress"})

    def ask_permission(self, tool_name: str) -> dict:
        return self.conn.request("session/request_permission", {
            "sessionId": self.session["id"],
            "toolCall": {"toolCallId": tool_name, "title": f"Run tool \"{tool_name}\""},
            "options": [{"optionId": "allow", "name": "Allow", "kind": "allow_once"},
                        {"optionId": "reject", "name": "Reject", "kind": "reject_once"}]})

    def say(self, text: str) -> None:
        self.update({"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": text}})

    def update(self, update: dict) -> None:
        self.conn.send({"jsonrpc": "2.0", "method": "session/update",
                        "params": {"sessionId": self.session["id"], "update": update}})


Script = Callable[[ScriptContext], Any]


class _Conn:
    def __init__(self, sock: socket.socket, peer: "FakeAkhomed"):
        self.sock = sock
        self.peer = peer
        self.inbox: "queue.Queue[dict | None]" = queue.Queue()
        self._ids = itertools.count(1000)
        self.lock = threading.Lock()

    def send(self, message: dict) -> None:
        with self.lock:
            self.sock.sendall((json.dumps(message) + "\n").encode())

    def request(self, method: str, params: dict) -> dict:
        request_id = next(self._ids)
        self.send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        while True:
            message = self.inbox.get(timeout=10)
            if message is None:
                raise ConnectionError("client went away")
            if message.get("id") == request_id and "method" not in message:
                return message
            self.peer.record(message)


class FakeAkhomed:
    """`with FakeAkhomed(script) as peer:` then point the client at `peer.endpoint`."""

    def __init__(self, script: Script | None = None, state: PeerState | None = None):
        self.script = script or (lambda ctx: None)
        self.state = state or PeerState()
        self.received: list[dict] = []  # every client message
        self.sessions: list[dict] = []
        self.cancelled: list[str] = []
        self.contexts: list[ScriptContext] = []
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(8)
        self._stop = False
        self._lock = threading.Lock()

    @property
    def endpoint(self) -> str:
        return f"tcp://127.0.0.1:{self._server.getsockname()[1]}"

    def record(self, message: dict) -> None:
        with self._lock:
            self.received.append(message)
            if message.get("method") == "session/cancel":
                self.cancelled.append(message["params"]["sessionId"])

    def __enter__(self) -> "FakeAkhomed":
        threading.Thread(target=self._accept, daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop = True
        self._server.close()

    def _accept(self) -> None:
        while not self._stop:
            try:
                sock, _ = self._server.accept()
            except OSError:
                return
            conn = _Conn(sock, self)
            threading.Thread(target=self._read, args=(conn,), daemon=True).start()
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _read(self, conn: _Conn) -> None:
        buffer = b""
        try:
            while True:
                chunk = conn.sock.recv(65536)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if line.strip():
                        conn.inbox.put(json.loads(line))
        except OSError:
            pass
        conn.inbox.put(None)

    def _serve(self, conn: _Conn) -> None:
        authorized = False
        while True:
            try:
                message = conn.inbox.get(timeout=30)
            except queue.Empty:
                return
            if message is None:
                return
            self.record(message)
            method, params, rid = message.get("method"), message.get("params") or {}, message.get("id")
            if rid is None:
                continue
            if method == "initialize":
                authorized = params.get("_meta", {}).get("authorization") == f"Bearer {self.state.token}"
                if not authorized:
                    conn.send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32000, "message": "unauthenticated"}})
                    continue
                conn.send({"jsonrpc": "2.0", "id": rid, "result": {
                    "protocolVersion": 1, "agentInfo": {"name": "akhomed", "version": "test"},
                    "agentCapabilities": {"loadSession": True, "mcpCapabilities": {"http": self.state.mcp_http, "sse": False}}}})
                continue
            if not authorized:
                conn.send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32000, "message": "initialize first"}})
                continue
            if method == "_akh/tools":
                conn.send({"jsonrpc": "2.0", "id": rid, "result": {
                    "workspace": params.get("workspace"), "stale": self.state.stale,
                    "tools": [{"name": n, "description": n} for n in self.state.own_tools]}})
            elif method == "_akh/health":
                conn.send({"jsonrpc": "2.0", "id": rid, "result": {
                    "status": "ok", "workspace": params.get("workspace"),
                    "symbols": self.state.symbols, "triples": self.state.triples}})
            elif method == "_akh/usage":
                conn.send({"jsonrpc": "2.0", "id": rid, "result": {"workspace": params.get("workspace"), "stale": False,
                    "usage": {"prompt_tokens": self.state.prompt_tokens,
                              "completion_tokens": self.state.completion_tokens}}})
            elif method == "session/new":
                session = {"id": f"s{len(self.sessions) + 1}", **params}
                self.sessions.append(session)
                conn.send({"jsonrpc": "2.0", "id": rid, "result": {"sessionId": session["id"]}})
            elif method == "session/prompt":
                session = next(s for s in self.sessions if s["id"] == params["sessionId"])
                ctx = ScriptContext(self, conn, session, params["prompt"])
                self.contexts.append(ctx)
                outcome = self.script(ctx)
                self.state.prompt_tokens += 1000
                self.state.completion_tokens += 250
                stop = outcome if isinstance(outcome, str) else "end_turn"
                conn.send({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": stop}})
            else:
                conn.send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"{method} not found"}})
