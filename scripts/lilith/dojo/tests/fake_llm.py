"""Local fake OpenAI-compatible and Anthropic servers for the runner tests.

Each fake takes a *script*: a callable `(request_json, call_index) -> turn`, where a turn is
`Say(text)`, `Call(name, args)` (one or more), or `Fail(status)`. The fakes record every request
(headers and body) so tests can assert what the adapters sent. No network beyond 127.0.0.1."""
from __future__ import annotations

import dataclasses
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable


@dataclasses.dataclass
class Say:
    text: str
    reasoning: str | None = None


@dataclasses.dataclass
class Call:
    name: str
    args: dict | str  # a str is sent verbatim (to test malformed arguments)
    text: str = ""


@dataclasses.dataclass
class Calls:
    calls: list[Call]
    text: str = ""


@dataclasses.dataclass
class Fail:
    status: int = 500
    body: str = "boom"


Script = Callable[[dict, int], Any]


class FakeLLM:
    """kind: 'openai' or 'anthropic'."""

    def __init__(self, kind: str, script: Script):
        self.kind = kind
        self.script = script
        self.requests: list[dict] = []  # {"path", "headers", "body"}
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                body = json.loads(raw)
                with outer._lock:
                    index = len(outer.requests)
                    outer.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                                           "body": body})
                if self.headers.get("User-Agent", "").startswith("Python-urllib"):
                    return self._send(403, {"error": "python UA blocked"})
                turn = outer.script(body, index)
                if isinstance(turn, Fail):
                    return self._send(turn.status, turn.body)
                self._send(200, outer.render(turn, body))

            def _send(self, status: int, payload: Any):
                data = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.httpd.server_address
        return f"http://{host}:{port}" + ("/v1" if self.kind == "openai" else "")

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()

    # -- wire formats ------------------------------------------------------------------------

    def render(self, turn: Any, body: dict) -> dict:
        calls = turn.calls if isinstance(turn, Calls) else ([turn] if isinstance(turn, Call) else [])
        text = turn.text if isinstance(turn, (Calls, Call)) else turn.text
        if self.kind == "openai":
            message: dict[str, Any] = {"role": "assistant", "content": text or None}
            if isinstance(turn, Say) and turn.reasoning:
                message["reasoning_content"] = turn.reasoning
            if calls:
                message["tool_calls"] = [{
                    "id": f"call_{n}", "type": "function",
                    "function": {"name": c.name, "arguments": c.args if isinstance(c.args, str) else json.dumps(c.args)}}
                    for n, c in enumerate(calls)]
            return {"choices": [{"message": message, "finish_reason": "tool_calls" if calls else "stop"}],
                    "usage": {"prompt_tokens": 1000, "completion_tokens": 50,
                              "prompt_tokens_details": {"cached_tokens": 800}}}
        blocks: list[dict] = []
        if isinstance(turn, Say) and turn.reasoning:
            blocks.append({"type": "thinking", "thinking": turn.reasoning})
        if text:
            blocks.append({"type": "text", "text": text})
        for n, c in enumerate(calls):
            blocks.append({"type": "tool_use", "id": f"toolu_{n}", "name": c.name,
                           "input": c.args if isinstance(c.args, dict) else {}})
        return {"content": blocks, "stop_reason": "tool_use" if calls else "end_turn",
                "usage": {"input_tokens": 200, "output_tokens": 50, "cache_read_input_tokens": 800,
                          "cache_creation_input_tokens": 0}}
