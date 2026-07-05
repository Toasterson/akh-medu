#!/usr/bin/env python3
"""Streaming ACP client for akhomed — drive an akh live over tcp:8201.

The akhomed daemon speaks line-delimited JSON-RPC (one JSON object per '\\n')
on its ACP TCP port. A turn streams `session/update` notifications: akhomed maps
the LLM's reasoning *and* tool steps to `agent_thought_chunk` (tool steps look
like "[tool:status] output"), and the final answer to `agent_message_chunk`.
The turn ends with the `session/prompt` result `{stopReason: ...}`.

Workspace selection: the target akh is encoded in the `session/new` `cwd` as
`akh-ws:<name>`; one session can carry many prompts (multi-turn memory).

Usage (interactive):  python3 acp_adapter.py --workspace tecton
Usage (one-shot):     python3 acp_adapter.py -w tecton -p "Reply with one word: hi"
As a library:         from acp_adapter import AkhClient
"""
from __future__ import annotations
import argparse, json, re, socket, sys, time
from dataclasses import dataclass, field

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8201
TOOL_STEP_RE = re.compile(r"\[([a-z0-9_]+):([a-z0-9_]+)\]", re.I)


@dataclass
class Turn:
    """Result of one prompt turn."""
    answer: str = ""                       # concatenated agent_message_chunk text
    reasoning: str = ""                    # concatenated agent_thought_chunk text
    tool_steps: list[str] = field(default_factory=list)  # parsed "[tool:status]" markers
    stop_reason: str | None = None
    elapsed_s: float = 0.0
    timed_out: bool = False

    @property
    def tools_used(self) -> list[str]:
        return [s.split(":", 1)[0] for s in self.tool_steps]


class AkhClient:
    """One live ACP session against a single akh workspace."""

    def __init__(self, workspace: str, host: str = DEFAULT_HOST,
                 port: int = DEFAULT_PORT, timeout: float = 180.0):
        self.workspace = workspace
        self.host, self.port, self.timeout = host, port, timeout
        self._sock: socket.socket | None = None
        self._f = None
        self._id = 0
        self.session_id: str | None = None
        self.agent_info: dict = {}

    # ── connection lifecycle ────────────────────────────────────────────
    def __enter__(self) -> "AkhClient":
        self.connect(); return self

    def __exit__(self, *exc):
        self.close()

    def connect(self) -> None:
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._f = self._sock.makefile("rwb", buffering=0)
        init = self._rpc("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
        self.agent_info = init.get("agentInfo", {})
        new = self._rpc("session/new", {"cwd": f"akh-ws:{self.workspace}", "mcpServers": []})
        self.session_id = new["sessionId"]

    def close(self) -> None:
        try:
            if self._f: self._f.close()
            if self._sock: self._sock.close()
        except OSError:
            pass

    # ── low-level JSON-RPC over newline-delimited frames ────────────────
    def _send(self, obj: dict) -> None:
        self._f.write((json.dumps(obj) + "\n").encode())

    def _readline(self) -> str:
        line = self._f.readline()
        if not line:
            raise ConnectionError("ACP connection closed by daemon")
        return line.decode().strip()

    def _rpc(self, method: str, params: dict) -> dict:
        """Send a request and read until its matching response (no streaming)."""
        self._id += 1
        rid = self._id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        while True:
            msg = json.loads(self._readline())
            if msg.get("id") == rid:
                if "error" in msg:
                    raise RuntimeError(f"{method} failed: {msg['error']}")
                return msg.get("result", {})
            self._handle_incoming(msg)  # ignore/notify anything arriving early

    def _handle_incoming(self, msg: dict) -> None:
        """Defensive: a server->client request (e.g. permission) gets a safe
        reply so the turn can't deadlock. With bypass permission mode none are
        expected, but we never want to hang."""
        m = msg.get("method")
        if m and "id" in msg:  # it's a request expecting a response
            self._send({"jsonrpc": "2.0", "id": msg["id"],
                        "result": {"outcome": {"outcome": "cancelled"}}})

    # ── prompt turn (streamed) ──────────────────────────────────────────
    def prompt(self, text: str, *, on_thought=None, on_message=None,
               echo: bool = False) -> Turn:
        """Run one turn. Streams update chunks through callbacks (or stdout when
        echo=True) and returns the accumulated Turn."""
        self._id += 1
        rid = self._id
        self._send({"jsonrpc": "2.0", "id": rid, "method": "session/prompt",
                    "params": {"sessionId": self.session_id,
                               "prompt": [{"type": "text", "text": text}]}})
        turn = Turn()
        start = time.monotonic()
        deadline = start + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                turn.timed_out = True
                break
            self._sock.settimeout(remaining)
            try:
                line = self._readline()
            except (socket.timeout, TimeoutError):
                turn.timed_out = True
                break
            msg = json.loads(line)
            if msg.get("method") == "session/update":
                u = msg["params"]["update"]
                kind = u.get("sessionUpdate")
                txt = (u.get("content") or {}).get("text", "")
                if kind == "agent_message_chunk":
                    turn.answer += txt
                    if echo: sys.stdout.write(txt); sys.stdout.flush()
                    if on_message: on_message(txt)
                elif kind == "agent_thought_chunk":
                    turn.reasoning += txt
                    for mt in TOOL_STEP_RE.finditer(txt):
                        turn.tool_steps.append(f"{mt.group(1)}:{mt.group(2)}")
                    if echo: sys.stderr.write(f"\033[2m{txt}\033[0m"); sys.stderr.flush()
                    if on_thought: on_thought(txt)
            elif msg.get("id") == rid:
                turn.stop_reason = (msg.get("result") or {}).get("stopReason")
                if "error" in msg:
                    turn.answer += f"[error] {msg['error']}"
                break
            else:
                self._handle_incoming(msg)
        turn.elapsed_s = time.monotonic() - start
        return turn


def _interactive(client: AkhClient) -> None:
    info = client.agent_info
    print(f"Connected to akh '{client.workspace}' (agent {info.get('name')} "
          f"v{info.get('version')}), session {client.session_id}.")
    print("Type a message; Ctrl-D or /quit to exit. Reasoning is dimmed on stderr.\n")
    while True:
        try:
            line = input("> ")
        except EOFError:
            print(); break
        if line.strip() in ("/quit", "/exit"):
            break
        if not line.strip():
            continue
        turn = client.prompt(line, echo=True)
        print()  # newline after streamed answer
        meta = f"[{turn.stop_reason or 'timeout'} · {turn.elapsed_s:.1f}s"
        if turn.tool_steps:
            meta += f" · tools: {', '.join(sorted(set(turn.tools_used)))}"
        print(f"\033[2m{meta}]\033[0m")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Streaming ACP client for akhomed")
    ap.add_argument("-w", "--workspace", default="tecton")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("-p", "--prompt", help="one-shot prompt then exit")
    args = ap.parse_args(argv)
    with AkhClient(args.workspace, args.host, args.port, args.timeout) as c:
        if args.prompt:
            turn = c.prompt(args.prompt, echo=True)
            print()
            print(f"\033[2m[{turn.stop_reason or 'timeout'} · {turn.elapsed_s:.1f}s · "
                  f"tools: {', '.join(sorted(set(turn.tools_used))) or 'none'}]\033[0m")
        else:
            _interactive(c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
