"""A minimal ACP/TCP client for one read-only call, `_akh/transcripts/list`.

Wire (agent-client-protocol over TCP): newline-delimited JSON-RPC 2.0. The daemon
authenticates `initialize` by `_meta.authorization = "Bearer <token>"`; the ext
request then names the akh in `params.workspace`. The bearer token is only ever
put on the wire: it never appears in an exception, a log line or a file.
"""
from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any, Callable, Iterator

from c2.errors import AcpUnavailableError, TranscriptsUnavailableError

DEFAULT_ENDPOINT = "127.0.0.1:18444"  # deploy/roman-mini/full_akh.py ACP_TCP_PORT
CONNECT_TIMEOUT = 5.0
REQUEST_TIMEOUT = 60.0
PAGE_LIMIT = 500
UNAVAILABLE_CODE = "akh::acp::transcripts_unavailable"
LIST_METHOD = "_akh/transcripts/list"
METHOD_NOT_FOUND = -32601

Opener = Callable[[tuple[str, int], float], socket.socket]


def _connect(address: tuple[str, int], timeout: float) -> socket.socket:
    return socket.create_connection(address, timeout=timeout)


def read_token(path: Path) -> str:
    try:
        token = next((line.strip() for line in path.read_text().splitlines() if line.strip()), "")
    except OSError as error:
        raise AcpUnavailableError(f"cannot read the operator token file {path}: {error.strerror}") from None
    if not token:
        raise AcpUnavailableError(f"the operator token file {path} is empty")
    return token


def split_endpoint(endpoint: str) -> tuple[str, int]:
    host, _, port = endpoint.removeprefix("tcp://").rpartition(":")
    if not host or not port.isdigit():
        raise AcpUnavailableError(f"endpoint {endpoint!r} is not host:port")
    return host, int(port)


class _Wire:
    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.reader = sock.makefile("rb")
        self.next_id = 1

    def call(self, method: str, params: dict[str, Any]) -> Any:
        request_id = self.next_id
        self.next_id += 1
        line = json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        self.sock.sendall(line.encode() + b"\n")
        while True:
            raw = self.reader.readline()
            if not raw:
                raise AcpUnavailableError(f"the daemon closed the connection during {method}")
            message = json.loads(raw)
            if message.get("id") != request_id or "method" in message:
                continue  # a notification or a request from the server: not our answer
            if "error" in message:
                raise _RpcError(method, message["error"])
            return message.get("result")


class _RpcError(Exception):
    def __init__(self, method: str, error: dict[str, Any]):
        super().__init__(f"{method}: {error.get('message', 'error')}")
        self.error = error


def ext_request(endpoint: str, token: str, method: str, params: dict[str, Any], *,
                opener: Opener = _connect, timeout: float = REQUEST_TIMEOUT) -> Any:
    """Connect, `initialize`, send one `_akh/*` request, disconnect; returns its JSON result."""
    address = split_endpoint(endpoint)
    try:
        sock = opener(address, CONNECT_TIMEOUT)
    except OSError as error:
        raise AcpUnavailableError(f"cannot reach the core ACP listener at {endpoint}: {error.strerror or error}") from None
    try:
        sock.settimeout(timeout)
        wire = _Wire(sock)
        try:
            wire.call("initialize", {"protocolVersion": 1, "clientCapabilities": {},
                                     "_meta": {"authorization": f"Bearer {token}"}})
        except _RpcError as error:
            status = (error.error.get("data") or {}).get("status", error.error.get("code"))
            raise AcpUnavailableError(f"the core refused the operator token at {endpoint} (status {status})") from None
        try:
            return wire.call(method, params)
        except _RpcError as error:
            detail = error.error.get("data") or {}
            if error.error.get("code") == METHOD_NOT_FOUND and method == LIST_METHOD and "code" not in detail:
                raise TranscriptsUnavailableError(
                    f"the core at {endpoint} does not know {LIST_METHOD}: it runs a build older than ADR 116 C2") from None
            if isinstance(detail, dict) and detail.get("code") == UNAVAILABLE_CODE:
                raise TranscriptsUnavailableError(str(detail.get("message"))) from None
            raise AcpUnavailableError(f"{method} failed: {error.error.get('message')} {detail}".strip()) from None
    except (OSError, ValueError) as error:  # socket timeout/reset, or a line that is not JSON
        raise AcpUnavailableError(f"ACP exchange with {endpoint} failed: {type(error).__name__}") from None
    finally:
        sock.close()


def list_turns(endpoint: str, token: str, workspace: str, *, after_seq: int = 0, page_limit: int = PAGE_LIMIT,
               opener: Opener = _connect) -> Iterator[dict[str, Any]]:
    """Every captured turn of `workspace` after `after_seq`, oldest first, paged by watermark."""
    watermark = after_seq
    while True:
        page = ext_request(endpoint, token, LIST_METHOD,
                           {"workspace": workspace, "after_seq": watermark, "limit": page_limit}, opener=opener)
        turns = page.get("turns") or []
        yield from turns
        if not turns or page.get("next_after_seq", watermark) <= watermark:
            return
        watermark = page["next_after_seq"]
