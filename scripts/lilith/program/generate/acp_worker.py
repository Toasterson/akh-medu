"""The sealed teacher akh over ACP, one connection per worker (no batch or direct model-API calls).

`AcpTeacher` spawns `akh-acp --akhomed` (stdio JSON-RPC bridge to the Roman core's ACP listener, the same
transport as `~/lilith-private/acp_turn.py`) with the `dojo-teacher` principal token, opens a fresh session per
task (tasks must not see each other), sends one prompt and collects the message text. Connection refusal (a
core restart) is retried with backoff; anything else surfaces as `TeacherError`.

`Teacher` is the seam the generators depend on; tests substitute `FakeTeacher` (no network, no subprocess).
Secrets are never printed: the token is read by `akh-acp` from its file, this module only sees the path.
"""
from __future__ import annotations

import dataclasses
import json
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Protocol, Sequence

TEACHER_ID = "dojo-teacher/glm-5.3"
DEFAULT_ROOT = Path.home() / "Library" / "Application Support" / "roman-akhs"
DEFAULT_ENDPOINT = "tcp://127.0.0.1:18444"


class TeacherError(Exception):
    """The teacher could not answer. `retryable` marks transport trouble (refused connection, EOF, timeout)."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


@dataclasses.dataclass(frozen=True)
class TeacherReply:
    text: str
    elapsed_s: float
    stop_reason: str | None = None


class Teacher(Protocol):
    def ask(self, prompt: str) -> TeacherReply: ...

    def close(self) -> None: ...


@dataclasses.dataclass(frozen=True)
class AcpConfig:
    akh: str = "dojo-teacher"
    binary: Path = DEFAULT_ROOT / "bin" / "akh-acp-main-ade6e0f7f746"
    endpoint: str = DEFAULT_ENDPOINT
    token_file: Path = DEFAULT_ROOT / "secrets" / "dojo-teacher.key"
    turn_timeout_s: float = 420.0
    rpc_timeout_s: float = 90.0
    cwd: str = "/tmp"


@dataclasses.dataclass(frozen=True)
class Backoff:
    attempts: int = 6
    initial_s: float = 5.0
    factor: float = 2.0
    max_s: float = 120.0

    def delays(self) -> list[float]:
        out, d = [], self.initial_s
        for _ in range(self.attempts):
            out.append(min(d, self.max_s))
            d *= self.factor
        return out


class _Connection:
    """One `akh-acp` subprocess speaking newline-delimited JSON-RPC."""

    def __init__(self, config: AcpConfig):
        self.config = config
        self._next = 0
        self._inbox: "queue.Queue[dict | None]" = queue.Queue()
        self._proc = subprocess.Popen(
            [str(config.binary), "--akhomed", "--endpoint", config.endpoint, "--token-file", str(config.token_file),
             "--akh", config.akh],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if isinstance(message, dict):
                self._inbox.put(message)
        self._inbox.put(None)

    def _send(self, message: dict) -> None:
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        except (OSError, ValueError) as error:
            raise TeacherError(f"akh-acp pipe closed: {error}", retryable=True) from None

    def request(self, method: str, params: dict, timeout_s: float, chunks: list[str] | None = None) -> dict:
        self._next += 1
        rid = self._next
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        deadline = time.monotonic() + timeout_s
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TeacherError(f"{method}: no answer within {timeout_s:.0f}s", retryable=True)
            try:
                message = self._inbox.get(timeout=left)
            except queue.Empty:
                continue
            if message is None:
                raise TeacherError(f"{method}: akh-acp closed the connection", retryable=True)
            if message.get("id") == rid and ("result" in message or "error" in message):
                if "error" in message:
                    error = message["error"] or {}
                    raise TeacherError(f"{method}: JSON-RPC error {error.get('code')}: {str(error.get('message'))[:200]}")
                return message.get("result") or {}
            if message.get("method") == "session/update" and chunks is not None:
                update = (message.get("params") or {}).get("update") or {}
                if update.get("sessionUpdate") == "agent_message_chunk":
                    text = (update.get("content") or {}).get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            elif "method" in message and "id" in message:  # a client request (permission, fs): the sealed teacher gets none
                self._send({"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32601, "message": "not supported"}})

    def close(self) -> None:
        for step in (lambda: self._proc.stdin and self._proc.stdin.close(), self._proc.terminate):
            try:
                step()
            except OSError:
                pass
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()


class AcpTeacher:
    """A teacher bound to one worker. Not thread-safe: one instance per worker thread."""

    def __init__(self, config: AcpConfig = AcpConfig(), backoff: Backoff = Backoff(),
                 sleep: Callable[[float], None] = time.sleep):
        self.config = config
        self.backoff = backoff
        self._sleep = sleep
        self._conn: _Connection | None = None

    def _connect(self) -> _Connection:
        if self._conn is None:
            conn = _Connection(self.config)
            try:
                conn.request("initialize", {"protocolVersion": 1, "clientCapabilities": {}}, self.config.rpc_timeout_s)
            except TeacherError:
                conn.close()
                raise
            self._conn = conn
        return self._conn

    def _drop(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _once(self, prompt: str) -> TeacherReply:
        t0 = time.monotonic()
        conn = self._connect()
        session = conn.request("session/new", {"cwd": self.config.cwd, "mcpServers": []}, self.config.rpc_timeout_s)
        sid = session.get("sessionId")
        if not sid:
            raise TeacherError("session/new returned no sessionId")
        chunks: list[str] = []
        result = conn.request("session/prompt", {"sessionId": sid, "prompt": [{"type": "text", "text": prompt}]},
                              self.config.turn_timeout_s, chunks)
        return TeacherReply("".join(chunks), time.monotonic() - t0, result.get("stopReason"))

    def ask(self, prompt: str) -> TeacherReply:
        delays = self.backoff.delays()
        for attempt in range(len(delays) + 1):
            try:
                return self._once(prompt)
            except TeacherError as error:
                self._drop()
                if not error.retryable or attempt == len(delays):
                    raise
                self._sleep(delays[attempt])
        raise TeacherError("unreachable")  # pragma: no cover

    def close(self) -> None:
        self._drop()


class FakeTeacher:
    """Deterministic in-process teacher for tests: `responder(prompt) -> text` (or raises TeacherError)."""

    def __init__(self, responder: Callable[[str], str]):
        self.responder = responder
        self.prompts: list[str] = []

    def ask(self, prompt: str) -> TeacherReply:
        self.prompts.append(prompt)
        return TeacherReply(self.responder(prompt), 0.0, "end_turn")

    def close(self) -> None:
        pass


def run_pool(items: Sequence, make_teacher: Callable[[], Teacher], work: Callable[[Teacher, object], None],
             workers: int, on_error: Callable[[object, Exception], None] | None = None) -> None:
    """Run `work(teacher, item)` over `items` with at most `workers` threads, one teacher each. A worker's
    failure on one item is reported through `on_error` and does not stop the others."""
    todo: "queue.Queue" = queue.Queue()
    for item in items:
        todo.put(item)

    def loop() -> None:
        teacher = make_teacher()
        try:
            while True:
                try:
                    item = todo.get_nowait()
                except queue.Empty:
                    return
                try:
                    work(teacher, item)
                except Exception as error:  # noqa: BLE001 - one bad item must not kill the worker
                    if on_error is not None:
                        on_error(item, error)
        finally:
            teacher.close()

    threads = [threading.Thread(target=loop, daemon=False) for _ in range(max(1, min(workers, len(items) or 1)))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
