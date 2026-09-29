#!/usr/bin/env python3
"""Lilith Chief-of-Staff read-only tracker recorder (Anima WI 01a0e270-137c,
plan slice C3, ADR 116 Decision 3).

Every run captures a content-addressed snapshot of what Lilith would read:
Anima (plans, work items, live-item detail/loop-state, changed-item threads,
sessions, changes, CI status, scheduling state) for one or more projects, the
akh-medu git refs, the Agent Board (REST, GET-only) and fleet status (GET-only
endpoints from a config file). Both are `not_configured` until a token file /
config is supplied; neither ever carries credentials in the repo.

Read-only guarantee: every MCP tool call is checked against an explicit
allowlist *before* a request is built (`ReadOnlyToolRefused`), and every
Board/fleet HTTP request goes through `ReadOnlyHttpClient`, which refuses any
method other than GET before the transport is touched (`ReadOnlyHttpRefused`).
Never call a write tool.

Multiple Anima projects: one `AnimaCollector` per project id. The Akh-Medu
project keeps the collector name `anima` (so single-project snapshots and the
dojo loader are unchanged); every other project is `anima:<project_id>`. Each
response record also carries an additive `project_id` field. The schema stays
`lilith/cos-snapshot@1`: readers ignore unknown record fields, and old
snapshots simply have no `project_id`.

Fail-closed per collector: one collector's exception is caught by the
orchestrator and recorded as `{"status": "error", "errors": [...]}` for that
collector only; the rest of the snapshot still runs.

Output layout under the snapshot root (env `LILITH_SNAPSHOT_ROOT`, default
`/Volumes/akh-medu-runs/lilith/cos-snapshots`):

    <root>/<UTC-timestamp>/manifest.json   schema "lilith/cos-snapshot@1"
    <root>/objects/<sha256>.json           deduplicated response bodies

All writes are tmp-file-then-rename (atomic). Secrets (Authorization
headers, bearer tokens) are redacted from every error string before it is
stored or printed.

Usage:
    cos_recorder.py run [--project-id ID ...] [--board-secrets-file F] [--fleet-config F]
    cos_recorder.py gap-report --day 2026-09-28
    cos_recorder.py gap-report --since 2026-09-15 [--format json|markdown]
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import urllib.error
import urllib.parse
import uuid
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

SCHEMA = "lilith/cos-snapshot@1"
DEFAULT_ROOT = "/Volumes/akh-medu-runs/lilith/cos-snapshots"
ANIMA_MCP_URL = "https://anima.wegmueller.it/mcp"
ANIMA_PROJECT_ID = "019d9844-ce74-7993-a9f0-c953b8f04a4e"
GIT_REMOTE = "https://code.aopc.cloud/toasterson/akh-medu.git"
SLOT_HOURS = 2
GAP_THRESHOLD_HOURS = 2.5
DEFAULT_BOARD_URL = "http://127.0.0.1:18089"

# API contract, observed live 2026-09-29: Anima's `get_thread` answers a work
# item that has no thread yet with a JSON-RPC error (not an empty result).
# Matched on tool name + numeric code + exact message; this is a recorded
# server contract, not natural-language classification.
THREAD_ABSENT_CONTRACT = ("get_thread", -32603, "Thread not found")

# --- Read-only tool allowlist -----------------------------------------------
#
# Anything not in this set is refused in `call_read_tool` before a request is
# ever built. This is the whole read-only guarantee; keep it explicit and
# never extend it with a write tool.
ANIMA_READ_TOOLS: frozenset[str] = frozenset({
    "list_plans",
    "list_work_items",
    "get_work_item_detail",
    "get_wi_loop_state",
    "get_thread",
    "list_sessions",
    "list_changes",
    "get_project_ci_status",
    "get_scheduling_state",
})

LIVE_PHASES_EXCLUDED = {"done"}
LIVE_STATUS_EXCLUDED = {"cancelled"}


class McpToolError(RuntimeError):
    """A tool call the server answered with an error (JSON-RPC error object or
    `isError` result). Carries the structured parts so callers can classify
    specific, documented contracts instead of matching free text."""

    def __init__(self, tool: str, code: int | None, message: str, raw: str):
        super().__init__(raw)
        self.tool = tool
        self.code = code
        self.message = message

    def is_thread_absent(self) -> bool:
        return (self.tool, self.code, self.message) == THREAD_ABSENT_CONTRACT


class ReadOnlyToolRefused(ValueError):
    """Raised when code asks to call an MCP tool outside the read allowlist."""

    def __init__(self, name: str, allowlist: frozenset[str]):
        super().__init__(f"refusing non-allowlisted tool call: {name!r} not in {sorted(allowlist)}")
        self.name = name


# --- Secret redaction --------------------------------------------------------

_BEARER_RE = re.compile(r"Bearer\s+\S+", re.IGNORECASE)
_AUTH_HEADER_RE = re.compile(r"(['\"]?Authorization['\"]?\s*[:=]\s*)['\"]?[^'\",}\s]+", re.IGNORECASE)


def redact(text: str) -> str:
    """Strip bearer tokens / Authorization header values from an error string."""
    text = _BEARER_RE.sub("Bearer [redacted]", text)
    text = _AUTH_HEADER_RE.sub(r"\1[redacted]", text)
    return text


# --- Canonical JSON + content addressing ------------------------------------

def canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    body = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    atomic_write_bytes(path, body)


def store_object(root: Path, payload: Any) -> tuple[str, bool]:
    """Write `payload` content-addressed under `<root>/objects/`, deduplicated.

    Returns (sha256_hex, stored_new). Existing objects are never rewritten.
    """
    data = canonical_json(payload)
    digest = sha256_hex(data)
    path = root / "objects" / f"{digest}.json"
    if path.exists():
        return digest, False
    atomic_write_bytes(path, data)
    return digest, True


# --- Minimal MCP JSON-RPC client --------------------------------------------

class Transport(Protocol):
    def post(self, url: str, headers: dict[str, str], body: bytes) -> tuple[dict[str, str], bytes]:
        ...


class UrllibTransport:
    """Default transport: stdlib `urllib`, no extra dependency."""

    def post(self, url: str, headers: dict[str, str], body: bytes) -> tuple[dict[str, str], bytes]:
        import urllib.request

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=60) as response:
            return dict(response.headers.items()), response.read()


def _parse_jsonrpc_body(body: str) -> dict:
    body = body.strip()
    if not body:
        raise RuntimeError("empty MCP response body")
    if body.startswith("{"):
        return json.loads(body)
    for line in body.splitlines():
        if line.startswith("data:"):
            chunk = line[5:].strip()
            if chunk:
                return json.loads(chunk)
    raise RuntimeError("MCP response contained no JSON-RPC payload")


@dataclasses.dataclass
class McpClient:
    url: str
    token: str
    client_name: str
    transport: Transport = dataclasses.field(default_factory=UrllibTransport)
    session_id: str | None = dataclasses.field(default=None, init=False)
    _next_id: int = dataclasses.field(default=1, init=False)

    def _request(self, method: str, params: dict | None = None, notification: bool = False) -> dict | None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        if not notification:
            payload["id"] = self._next_id
            self._next_id += 1
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        resp_headers, body = self.transport.post(self.url, headers, json.dumps(payload).encode("utf-8"))
        self.session_id = resp_headers.get("Mcp-Session-Id", self.session_id)
        if notification:
            return None
        return _parse_jsonrpc_body(body.decode("utf-8"))

    def initialize(self) -> None:
        self._request(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": self.client_name, "version": "1"},
            },
        )
        self._request("notifications/initialized", notification=True)

    def list_tools(self) -> list[dict]:
        result = self._request("tools/list")
        assert result is not None
        if "error" in result:
            raise RuntimeError(f"tools/list failed: {result['error']}")
        return result["result"]["tools"]

    def call_tool(self, name: str, arguments: dict) -> Any:
        result = self._request("tools/call", {"name": name, "arguments": arguments})
        assert result is not None
        if "error" in result:
            err = result["error"] if isinstance(result["error"], dict) else {}
            raise McpToolError(
                name, err.get("code"), str(err.get("message", "")), f"tool call {name!r} failed: {result['error']}"
            )
        payload = result["result"]
        content = payload.get("content") or []
        text = content[0].get("text", "") if content else ""
        if payload.get("isError"):
            raise McpToolError(name, None, text, f"tool call {name!r} returned an error: {text or '(no message)'}")
        if not text:
            raise RuntimeError(f"tool call {name!r} returned no content")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Some tools (observed live on `get_thread`) return a
            # human-readable transcript instead of a JSON envelope. Capture
            # it verbatim rather than failing the collector on a real,
            # successful response that just isn't JSON-shaped.
            return text


def call_read_tool(client: McpClient, allowlist: frozenset[str], name: str, arguments: dict) -> Any:
    """The read-only guarantee: refuse before any request is sent."""
    if name not in allowlist:
        raise ReadOnlyToolRefused(name, allowlist)
    return client.call_tool(name, arguments)


# --- Anima PAT resolution -----------------------------------------------------

def resolve_anima_pat() -> str:
    token = os.environ.get("ANIMA_PAT")
    if token and token.strip():
        return token.strip()
    binary = os.path.expanduser("~/.local/bin/anima-pat")
    try:
        proc = subprocess.run([binary, "get"], capture_output=True, text=True, timeout=20, check=True)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"could not resolve ANIMA_PAT: {redact(str(error))}") from None
    token = proc.stdout.strip()
    if not token:
        raise RuntimeError("anima-pat get returned an empty token")
    return token


# --- Collector protocol -------------------------------------------------------

@dataclasses.dataclass
class ResponseRecord:
    call: str
    args: dict
    sha256: str
    stored_new: bool
    project_id: str | None = None
    # Only set for get_thread: "none" = explicit absent result (no thread / no messages).
    thread_state: str | None = None

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"call": self.call, "args": self.args, "sha256": self.sha256}
        if self.project_id:
            out["project_id"] = self.project_id
        if self.thread_state:
            out["thread_state"] = self.thread_state
        return out


@dataclasses.dataclass
class CollectorResult:
    status: str  # "ok" | "partial" | "skipped" | "error" | "not_configured"
    responses: list[ResponseRecord] = dataclasses.field(default_factory=list)
    errors: list[str] = dataclasses.field(default_factory=list)
    tool_contract_sha256: str | None = None
    extra: dict = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict:
        out: dict[str, Any] = {
            "status": self.status,
            "responses": [r.to_dict() for r in self.responses],
            "errors": self.errors,
        }
        if self.tool_contract_sha256:
            out["tool_contract_sha256"] = self.tool_contract_sha256
        if self.extra:
            out["extra"] = self.extra
        return out


@dataclasses.dataclass
class SnapshotContext:
    root: Path
    timestamp: str
    previous_work_items: dict[str, dict]
    current_work_items: dict[str, dict] = dataclasses.field(default_factory=dict)


class Collector(Protocol):
    name: str

    def collect(self, ctx: SnapshotContext) -> CollectorResult:
        ...


# --- Pagination helper --------------------------------------------------------

_CURSOR_KEYS = ("next_page_token", "next_cursor", "cursor")


def _extract_page(response: dict) -> tuple[list, str | None]:
    """Heuristic page reader: the first list-valued field is the items page,
    the first populated key from `_CURSOR_KEYS` is the next cursor."""
    items: list = []
    for value in response.values():
        if isinstance(value, list):
            items = value
            break
    cursor = None
    for key in _CURSOR_KEYS:
        value = response.get(key)
        if value:
            cursor = value
            break
    return items, cursor


def _is_live(work_item: dict) -> bool:
    phase = str(work_item.get("current_phase", "")).lower()
    status = str(work_item.get("status", "")).lower()
    return phase not in LIVE_PHASES_EXCLUDED and status not in LIVE_STATUS_EXCLUDED


def _item_fingerprint(work_item: dict) -> str:
    """Digest of the whole list entry. Anima's list has no ``updated_at``, so comparing
    that field alone (None == None) never detected a change (C4 review, 2026-09-28)."""
    return hashlib.sha256(json.dumps(work_item, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _changed_since(current_items: list[dict], previous_map: dict[str, dict]) -> list[str]:
    changed = []
    for wi in current_items:
        wid = wi.get("id")
        if not wid:
            continue
        prev = previous_map.get(wid)
        if prev is None or _item_fingerprint(prev) != _item_fingerprint(wi):
            changed.append(wid)
    return changed


def _thread_targets(current_items: list[dict], previous_map: dict[str, dict]) -> list[str]:
    """Threads to capture: every live item (thread messages don't change the list entry,
    so only a fetch sees them; the content-addressed store dedups unchanged responses)
    plus any non-live item whose list entry changed since the previous snapshot."""
    live = [wi["id"] for wi in current_items if wi.get("id") and _is_live(wi)]
    changed = _changed_since(current_items, previous_map)
    return list(dict.fromkeys(live + changed))


# --- Anima collector -----------------------------------------------------------

def anima_collector_name(project_id: str) -> str:
    """The Akh-Medu project keeps the historical name `anima`; others are `anima:<id>`."""
    return "anima" if project_id == ANIMA_PROJECT_ID else f"anima:{project_id}"


def _detail_thread_signal(detail: Any) -> str:
    """Structural thread signal from an already-fetched `get_work_item_detail`.

    Returns "none" when the detail lists zero thread messages and none were elided
    (the item has no thread, or an empty one), "present" when the detail carries the
    messages, "elided" when it says some were cut off (the full thread needs
    `get_thread`), "unknown" when the detail is missing or not shaped as expected.
    """
    if not isinstance(detail, dict):
        return "unknown"
    messages = detail.get("thread_messages")
    if not isinstance(messages, list):
        return "unknown"
    if detail.get("thread_messages_elided"):
        return "elided"
    return "present" if messages else "none"


ANIMA_PROFILES = frozenset({"full", "detail"})
ABSENT_THREAD_BODY = {"thread": None, "thread_state": "none"}


@dataclasses.dataclass
class AnimaCollector:
    project_id: str = ANIMA_PROJECT_ID
    mcp_url: str = ANIMA_MCP_URL
    pat_resolver: Callable[[], str] | None = None
    transport: Transport = dataclasses.field(default_factory=UrllibTransport)
    # "full": detail + loop state + sessions + changes + scheduling state + thread per live item
    # (about 5 calls each). "detail": only get_work_item_detail per live item (which already
    # carries the thread messages); get_thread is called only when the detail says they were
    # elided. Cuts a large project's per-run reads by roughly 5x.
    profile: str = "full"
    name: str = dataclasses.field(default="anima", init=False)

    def __post_init__(self) -> None:
        if self.profile not in ANIMA_PROFILES:
            raise ValueError(f"unknown Anima profile {self.profile!r}; expected one of {sorted(ANIMA_PROFILES)}")
        self.name = anima_collector_name(self.project_id)

    def collect(self, ctx: SnapshotContext) -> CollectorResult:
        result = CollectorResult(status="ok")
        result.extra["project_id"] = self.project_id
        result.extra["profile"] = self.profile
        try:
            token = (self.pat_resolver or resolve_anima_pat)()
        except Exception as error:
            result.status = "error"
            result.errors.append(redact(str(error)))
            return result

        client = McpClient(self.mcp_url, token, "lilith-cos-recorder", transport=self.transport)
        try:
            client.initialize()
            tools = client.list_tools()
        except Exception as error:
            result.status = "error"
            result.errors.append(redact(str(error)))
            return result
        result.tool_contract_sha256, _ = store_object(ctx.root, {"tools": tools})

        def record(cname: str, cargs: dict, response: Any, thread_state: str | None = None) -> None:
            digest, stored_new = store_object(ctx.root, response)
            result.responses.append(
                ResponseRecord(cname, cargs, digest, stored_new, project_id=self.project_id, thread_state=thread_state)
            )

        def call(cname: str, cargs: dict) -> Any:
            try:
                response = call_read_tool(client, ANIMA_READ_TOOLS, cname, cargs)
            except Exception as error:
                result.errors.append(f"{cname}({cargs}): {redact(str(error))}")
                result.status = "partial"
                return None
            record(cname, cargs, response)
            return response

        plans_items = self._paginate(call, "list_plans", {"project_id": self.project_id})
        work_items = self._paginate(call, "list_work_items", {"project_id": self.project_id})

        result.extra["plan_count"] = len(plans_items)
        result.extra["work_item_count"] = len(work_items)

        live_items = [wi for wi in work_items if _is_live(wi)]
        result.extra["live_work_item_count"] = len(live_items)

        details: dict[str, Any] = {}
        for wi in live_items:
            wid = wi.get("id")
            if not wid:
                continue
            details[wid] = call("get_work_item_detail", {"id": wid})
            if self.profile == "full":
                call("get_wi_loop_state", {"work_item_id": wid})
                call("list_sessions", {"work_item_id": wid})
                call("list_changes", {"work_item_id": wid})
                call("get_scheduling_state", {"work_item_id": wid})

        changed_items = _changed_since(work_items, ctx.previous_work_items)
        result.extra["changed_item_count"] = len(changed_items)
        thread_targets = _thread_targets(work_items, ctx.previous_work_items)
        if self.profile == "detail":
            # The detail already holds the thread; only elided ones need the full transcript.
            thread_targets = [w for w in thread_targets if _detail_thread_signal(details.get(w)) == "elided"]
        result.extra["thread_target_count"] = len(thread_targets)
        thread_states: Counter[str] = Counter()
        for wid in thread_targets:
            args = {"work_item_id": wid}
            # Structural first: the detail we already hold says whether the item has
            # thread messages, so an item without one costs no get_thread call at all.
            if _detail_thread_signal(details.get(wid)) == "none":
                record("get_thread", args, dict(ABSENT_THREAD_BODY), thread_state="none")
                thread_states["none"] += 1
                continue
            try:
                response = call_read_tool(client, ANIMA_READ_TOOLS, "get_thread", args)
            except McpToolError as error:
                if error.is_thread_absent():
                    # Documented contract (see THREAD_ABSENT_CONTRACT): explicit absence, not an error.
                    record("get_thread", args, dict(ABSENT_THREAD_BODY), thread_state="none")
                    thread_states["none"] += 1
                else:
                    result.errors.append(f"get_thread({args}): {redact(str(error))}")
                    result.status = "partial"
                continue
            except Exception as error:
                result.errors.append(f"get_thread({args}): {redact(str(error))}")
                result.status = "partial"
                continue
            record("get_thread", args, response)
            thread_states["present"] += 1
        result.extra["thread_states"] = dict(sorted(thread_states.items()))

        call("get_project_ci_status", {"project_id": self.project_id})

        ctx.current_work_items.update({wi["id"]: wi for wi in work_items if wi.get("id")})

        if result.status == "ok" and not result.responses:
            result.status = "error"
            result.errors.append("no responses captured")
        return result

    @staticmethod
    def _paginate(call: Callable[[str, dict], Any], name: str, base_args: dict) -> list[dict]:
        items: list[dict] = []
        cursor: str | None = None
        while True:
            args = dict(base_args)
            if cursor:
                args["page_token"] = cursor
            response = call(name, args)
            if response is None:
                break
            page_items, cursor = _extract_page(response)
            items.extend(page_items)
            if not cursor:
                break
        return items


# --- Git collector ---------------------------------------------------------

def _parse_ls_remote(output: str) -> list[dict]:
    refs = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        refs.append({"sha": parts[0], "ref": parts[1]})
    return refs


@dataclasses.dataclass
class GitCollector:
    remote: str = GIT_REMOTE
    runner: Callable[[list[str]], subprocess.CompletedProcess] | None = None
    name: str = dataclasses.field(default="git", init=False)

    def collect(self, ctx: SnapshotContext) -> CollectorResult:
        result = CollectorResult(status="ok")
        runner = self.runner or (
            lambda cmd: subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
        )
        try:
            proc = runner(["git", "ls-remote", self.remote])
        except (OSError, subprocess.SubprocessError) as error:
            result.status = "error"
            result.errors.append(redact(str(error)))
            return result
        refs = _parse_ls_remote(proc.stdout)
        digest, stored_new = store_object(ctx.root, {"remote": self.remote, "refs": refs})
        result.responses.append(ResponseRecord("git_ls_remote", {"remote": self.remote}, digest, stored_new))
        result.extra["ref_count"] = len(refs)
        return result


# --- Read-only HTTP client (Board + fleet) -----------------------------------

class ReadOnlyHttpRefused(ValueError):
    """Raised when code asks the read-only HTTP client for anything but a GET."""

    def __init__(self, method: str, path: str):
        super().__init__(f"refusing non-GET request: {method.upper()} {path}")
        self.method = method


class HttpTransport(Protocol):
    def request(self, method: str, url: str, headers: dict[str, str]) -> tuple[int, dict[str, str], bytes]:
        ...


class UrllibHttpTransport:
    """Default HTTP transport: stdlib `urllib`. Returns (status, lower-cased headers, body)."""

    def request(self, method: str, url: str, headers: dict[str, str]) -> tuple[int, dict[str, str], bytes]:
        import urllib.request

        req = urllib.request.Request(url, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.status, {k.lower(): v for k, v in response.headers.items()}, response.read()
        except urllib.error.HTTPError as error:
            return error.code, {k.lower(): v for k, v in error.headers.items()}, error.read()


class HttpReadError(RuntimeError):
    def __init__(self, status: int | None, message: str):
        super().__init__(redact(message))
        self.status = status


@dataclasses.dataclass
class ReadOnlyHttpClient:
    """The Board/fleet read-only guarantee: the only method that can leave this
    class is GET. `send` refuses everything else *before* the transport is
    touched, so a coding mistake elsewhere cannot turn into a tracker write.
    """

    base_url: str
    token: str
    transport: HttpTransport = dataclasses.field(default_factory=UrllibHttpTransport)

    def send(self, method: str, path: str) -> tuple[int, dict[str, str], bytes]:
        if method.upper() != "GET":
            raise ReadOnlyHttpRefused(method, path)
        if not path.startswith("/") or "://" in path or path.startswith("//"):
            raise ReadOnlyHttpRefused(method, f"{path} (must be a server-relative path)")
        headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/json"}
        try:
            return self.transport.request("GET", self.base_url.rstrip("/") + path, headers)
        except (OSError, urllib.error.URLError) as error:
            raise HttpReadError(None, f"GET {path} transport error: {error}") from None

    def get_json(self, path: str) -> tuple[Any, dict[str, str]]:
        status, headers, body = self.send("GET", path)
        if not 200 <= status < 300:
            raise HttpReadError(status, f"GET {path} failed (HTTP {status}): {body[:200]!r}")
        try:
            return json.loads(body.decode("utf-8")), headers
        except ValueError:
            raise HttpReadError(status, f"GET {path} returned non-JSON body") from None


def read_bearer_token(path: Path) -> str:
    """Token from a `{"bearer_token": ...}` JSON secrets file or a plain one-line file."""
    try:
        raw = path.read_text().strip()
    except OSError as error:
        raise RuntimeError(f"cannot read secrets file {path}: {redact(str(error))}") from None
    token: Any = raw
    if raw.startswith("{"):
        try:
            token = json.loads(raw).get("bearer_token")
        except ValueError:
            token = None
    if not token or not isinstance(token, str):
        raise RuntimeError(f"{path} has no non-empty bearer token")
    return token


# --- Agent Board collector (REST, GET-only) ---------------------------------

@dataclasses.dataclass
class BoardCollector:
    """Reads the Agent Board's REST API with GETs only.

    Captures the agent roster, the message-type registry, the thread list and,
    for every *open* thread, its messages (forward `next-cursor` pagination,
    bounded by `max_pages`). It never reads the inbox: acknowledging is an
    explicit POST on the board, but the recorder keeps clear of per-agent state
    anyway. The token file is supplied by the caller (`--board-secrets-file`);
    none is stored in the repo. Without one the collector is `not_configured`.
    """

    base_url: str = DEFAULT_BOARD_URL
    secrets_file: Path | None = None
    transport: HttpTransport = dataclasses.field(default_factory=UrllibHttpTransport)
    max_pages: int = 50
    page_limit: int = 200
    name: str = dataclasses.field(default="board", init=False)

    def collect(self, ctx: SnapshotContext) -> CollectorResult:
        if self.secrets_file is None:
            return CollectorResult(
                status="not_configured",
                extra={"reason": "no board secrets file (--board-secrets-file / LILITH_BOARD_SECRETS)"},
            )
        result = CollectorResult(status="ok")
        try:
            client = ReadOnlyHttpClient(self.base_url, read_bearer_token(self.secrets_file), self.transport)
        except RuntimeError as error:
            result.status = "error"
            result.errors.append(redact(str(error)))
            return result

        def fetch(call: str, args: dict, path: str) -> tuple[Any, dict[str, str]] | None:
            try:
                body, headers = client.get_json(path)
            except HttpReadError as error:
                result.errors.append(f"{call}({args}): {redact(str(error))}")
                result.status = "partial"
                return None
            digest, stored_new = store_object(ctx.root, body)
            result.responses.append(ResponseRecord(call, args, digest, stored_new))
            return body, headers

        fetch("board_agents", {}, "/v1/agents")
        fetch("board_message_types", {}, "/v1/message-types")
        listed = fetch("board_threads", {}, "/v1/threads")
        threads = listed[0] if listed and isinstance(listed[0], list) else []
        open_threads = [t for t in threads if isinstance(t, dict) and t.get("id") and not t.get("closed_at")]
        result.extra["thread_count"] = len(threads)
        result.extra["open_thread_count"] = len(open_threads)

        for thread in open_threads:
            thread_id = thread["id"]
            after: str | None = None
            for _ in range(self.max_pages):
                args: dict[str, Any] = {"thread_id": thread_id}
                query = f"limit={self.page_limit}"
                if after:
                    args["after_message_id"] = after
                    query += f"&after_message_id={urllib.parse.quote(after)}"
                page = fetch("board_thread_messages", args, f"/v1/threads/{urllib.parse.quote(str(thread_id))}/messages?{query}")
                if page is None:
                    break
                after = page[1].get("next-cursor")
                if not after:
                    break
            else:
                result.errors.append(f"board_thread_messages({thread_id}): stopped after {self.max_pages} pages")
                result.status = "partial"
        return result


# --- Fleet status collector (config-driven, GET-only) -----------------------

@dataclasses.dataclass
class FleetCollector:
    """GETs the fleet's status endpoints listed in a JSON config file.

    Config shape (no secrets; token files are referenced by path):

        {"endpoints": [
          {"name": "sessions", "url": "http://127.0.0.1:18431/v1/sessions?compact=true",
           "secrets_file": "/path/to/operator.key"}
        ]}

    Each endpoint is fetched through `ReadOnlyHttpClient`; the response JSON is
    stored content-addressed under the call `fleet_get` with `{"endpoint": name}`.
    Without a config the collector is `not_configured`.
    """

    config_path: Path | None = None
    transport: HttpTransport = dataclasses.field(default_factory=UrllibHttpTransport)
    name: str = dataclasses.field(default="fleet", init=False)

    def collect(self, ctx: SnapshotContext) -> CollectorResult:
        if self.config_path is None:
            return CollectorResult(
                status="not_configured",
                extra={"reason": "no fleet config (--fleet-config / LILITH_FLEET_CONFIG)"},
            )
        result = CollectorResult(status="ok")
        try:
            endpoints = json.loads(self.config_path.read_text())["endpoints"]
            if not isinstance(endpoints, list):
                raise TypeError("endpoints must be a list")
        except (OSError, ValueError, KeyError, TypeError) as error:
            result.status = "error"
            result.errors.append(f"bad fleet config {self.config_path}: {redact(str(error))}")
            return result
        for endpoint in endpoints:
            ename = str(endpoint.get("name", "")) if isinstance(endpoint, dict) else ""
            try:
                parts = urllib.parse.urlsplit(endpoint["url"])
                path = parts.path + (f"?{parts.query}" if parts.query else "")
                base = f"{parts.scheme}://{parts.netloc}"
                client = ReadOnlyHttpClient(base, read_bearer_token(Path(endpoint["secrets_file"])), self.transport)
                body, _ = client.get_json(path)
            except (RuntimeError, KeyError, TypeError, ValueError) as error:
                result.errors.append(f"fleet_get({ename!r}): {redact(str(error))}")
                result.status = "partial"
                continue
            digest, stored_new = store_object(ctx.root, body)
            result.responses.append(ResponseRecord("fleet_get", {"endpoint": ename}, digest, stored_new))
        result.extra["endpoint_count"] = len(endpoints)
        if result.status == "partial" and not result.responses:
            result.status = "error"
        return result


DEFAULT_COLLECTORS: tuple[Collector, ...] = (AnimaCollector(), GitCollector(), BoardCollector(), FleetCollector())


def parse_id_list(values: Iterable[str] | None, env: str | None) -> list[str]:
    """Explicit CLI values win over the env list; entries may be comma-separated."""
    raw: list[str] = []
    for value in values or ():
        raw.extend(value.split(","))
    if not raw and env:
        raw = env.split(",")
    return list(dict.fromkeys(x.strip() for x in raw if x.strip()))


def parse_project_ids(values: Iterable[str] | None, env: str | None) -> list[str]:
    """Default is the Akh-Medu project alone."""
    return parse_id_list(values, env) or [ANIMA_PROJECT_ID]


def build_collectors(
    project_ids: list[str],
    detail_project_ids: Iterable[str] = (),
    board_secrets_file: Path | None = None,
    board_url: str = DEFAULT_BOARD_URL,
    fleet_config: Path | None = None,
) -> tuple[Collector, ...]:
    detail = set(detail_project_ids)
    anima = tuple(AnimaCollector(project_id=pid, profile="detail" if pid in detail else "full") for pid in project_ids)
    return (
        *anima,
        GitCollector(),
        BoardCollector(base_url=board_url, secrets_file=board_secrets_file),
        FleetCollector(config_path=fleet_config),
    )


# --- Snapshot orchestration ---------------------------------------------------

def find_previous_snapshot(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates = sorted(
        (p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").exists()),
        key=lambda p: p.name,
    )
    return candidates[-1] if candidates else None


def load_previous_work_items(root: Path, previous_dir: Path | None) -> dict[str, dict]:
    if previous_dir is None:
        return {}
    try:
        manifest = json.loads((previous_dir / "manifest.json").read_text())
    except (OSError, ValueError):
        return {}
    items: dict[str, dict] = {}
    for cname, anima in manifest.get("collectors", {}).items():
        if cname != "anima" and not cname.startswith("anima:"):
            continue
        for response in anima.get("responses", []):
            if response.get("call") != "list_work_items":
                continue
            obj_path = root / "objects" / f"{response.get('sha256')}.json"
            try:
                payload = json.loads(obj_path.read_text())
            except (OSError, ValueError):
                continue
            page_items, _ = _extract_page(payload)
            for wi in page_items:
                wid = wi.get("id")
                if wid:
                    items[wid] = wi
    return items


def run_snapshot(
    root: Path,
    collectors: Iterable[Collector] | None = None,
    now: datetime | None = None,
) -> dict:
    # Resolved as a global lookup (not a bound default) so tests can swap
    # `DEFAULT_COLLECTORS` at the module level without touching call sites.
    if collectors is None:
        collectors = DEFAULT_COLLECTORS
    now = now or datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    previous_dir = find_previous_snapshot(root)
    previous_work_items = load_previous_work_items(root, previous_dir)
    ctx = SnapshotContext(root=root, timestamp=timestamp, previous_work_items=previous_work_items)

    collector_results: dict[str, dict] = {}
    for collector in collectors:
        try:
            outcome = collector.collect(ctx)
        except Exception as error:  # fail-closed: one collector never aborts the run
            outcome = CollectorResult(status="error", errors=[redact(str(error))])
        collector_results[collector.name] = outcome.to_dict()

    manifest = {
        "schema": SCHEMA,
        "timestamp": now.isoformat(),
        "collectors": collector_results,
    }
    atomic_write_json(root / timestamp / "manifest.json", manifest)
    return manifest


# --- Gap report ----------------------------------------------------------------

_CALL_RE = re.compile(r"^([A-Za-z_][\w:.\-]*)\(")
_CODE_RE = re.compile(r"'code': (-?\d+)")
_HTTP_RE = re.compile(r"HTTP (\d{3})")


def error_kind(text: str) -> str:
    """Bucket a recorded error string as `<call>/<code|http-status|other>`.

    Recorded errors look like `call({args}): ...`; the bucket keeps the call name
    and the structured code (JSON-RPC `'code': N` or `HTTP nnn`) so the daily
    report groups by cause without matching message prose.
    """
    call_match = _CALL_RE.match(text)
    call = call_match.group(1) if call_match else "collector"
    code = _CODE_RE.search(text)
    http = _HTTP_RE.search(text)
    qualifier = f"code {code.group(1)}" if code else f"http {http.group(1)}" if http else "other"
    return f"{call}/{qualifier}"


def _load_snapshots(root: Path) -> list[tuple[str, datetime, dict | None]]:
    out: list[tuple[str, datetime, dict | None]] = []
    if not root.exists():
        return out
    for p in sorted(root.iterdir()):
        if not p.is_dir() or not (p / "manifest.json").exists():
            continue
        try:
            ts = datetime.strptime(p.name, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        try:
            manifest = json.loads((p / "manifest.json").read_text())
        except (OSError, ValueError):
            manifest = None
        out.append((p.name, ts, manifest))
    return out


def _day_bounds(day: str) -> tuple[datetime, datetime]:
    d = date.fromisoformat(day)
    start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    return start, start + timedelta(days=1)


def _range_report(root: Path, start: datetime, end: datetime, label: dict) -> dict:
    expected = []
    t = start
    while t < end:
        expected.append(t)
        t += timedelta(hours=SLOT_HOURS)

    everything = _load_snapshots(root)
    in_range = [x for x in everything if start <= x[1] < end]

    captured_slots: set[datetime] = set()
    failures: list[dict] = []
    status_counts: dict[str, Counter] = {}
    kinds: Counter = Counter()
    for name, ts, manifest in in_range:
        slot_index = int((ts - start).total_seconds() // (SLOT_HOURS * 3600))
        captured_slots.add(start + timedelta(hours=SLOT_HOURS * slot_index))
        if manifest is None:
            failures.append({"snapshot": name, "collector": None, "status": "unreadable_manifest", "errors": []})
            kinds["manifest/unreadable"] += 1
            continue
        for cname, cresult in manifest.get("collectors", {}).items():
            status = cresult.get("status", "unknown")
            status_counts.setdefault(cname, Counter())[status] += 1
            errors = cresult.get("errors", [])
            for err in errors:
                kinds[error_kind(str(err))] += 1
            if status in ("error", "partial") or errors:
                failures.append({"snapshot": name, "collector": cname, "status": status, "errors": errors})

    # Time gaps: anchor on the last snapshot before the range and the first after,
    # so a gap that straddles a boundary is still reported.
    anchors = [x[1] for x in everything if x[1] < start][-1:] + [x[1] for x in in_range]
    later = [x[1] for x in everything if x[1] >= end][:1]
    # No later snapshot: measure to the end of the range, or to now for a range still open.
    anchors += later or ([min(end, datetime.now(timezone.utc))] if anchors else [])
    threshold = timedelta(hours=GAP_THRESHOLD_HOURS)
    time_gaps = [
        {"from": a.isoformat(), "to": b.isoformat(), "hours": round((b - a).total_seconds() / 3600, 2)}
        for a, b in zip(anchors, anchors[1:])
        if b - a > threshold and a < end and b > start
    ]

    missing = [t.isoformat() for t in expected if t not in captured_slots]
    return {
        **label,
        "expected_slots": [t.isoformat() for t in expected],
        "captured_slots": sorted(t.isoformat() for t in captured_slots),
        "missing_slots": missing,
        "actual_snapshots": [x[0] for x in in_range],
        "snapshot_count": len(in_range),
        "gap_threshold_hours": GAP_THRESHOLD_HOURS,
        "time_gaps": time_gaps,
        "collector_status_counts": {c: dict(sorted(n.items())) for c, n in sorted(status_counts.items())},
        "error_kinds": dict(sorted(kinds.items())),
        "failures": failures,
    }


def gap_report(root: Path, day: str) -> dict:
    start, end = _day_bounds(day)
    return _range_report(root, start, end, {"day": day})


def gap_report_range(root: Path, since: str, until: str | None = None) -> dict:
    """`since` inclusive, `until` inclusive (UTC days); `until` defaults to today."""
    start, _ = _day_bounds(since)
    last = until or datetime.now(timezone.utc).date().isoformat()
    _, end = _day_bounds(last)
    return _range_report(root, start, end, {"since": since, "until": last})


def render_markdown(report: dict) -> str:
    title = report.get("day") or f"{report.get('since')} .. {report.get('until')}"
    lines = [
        f"# Lilith CoS recorder gap report, {title} (UTC)",
        "",
        f"- Snapshots: {report['snapshot_count']} captured, {len(report['missing_slots'])} of "
        f"{len(report['expected_slots'])} {SLOT_HOURS}-hour slots missing",
        f"- Gaps over {report['gap_threshold_hours']} h: {len(report['time_gaps'])}",
        "",
    ]
    if report["time_gaps"]:
        lines += ["## Time gaps", ""]
        lines += [f"- {g['from']} to {g['to']}: {g['hours']} h" for g in report["time_gaps"]]
        lines.append("")
    lines += ["## Collector status", ""]
    if report["collector_status_counts"]:
        lines += ["| collector | " + " | ".join(sorted({s for c in report["collector_status_counts"].values() for s in c})) + " |"]
        statuses = sorted({s for c in report["collector_status_counts"].values() for s in c})
        lines += ["|---|" + "---|" * len(statuses)]
        for cname, counts in report["collector_status_counts"].items():
            lines.append(f"| {cname} | " + " | ".join(str(counts.get(s, 0)) for s in statuses) + " |")
    else:
        lines.append("No snapshots in range.")
    lines += ["", "## Errors by kind", ""]
    if report["error_kinds"]:
        lines += [f"- `{k}`: {v}" for k, v in report["error_kinds"].items()]
    else:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)


# --- CLI -------------------------------------------------------------------

def resolve_root(cli_root: Path | None) -> Path:
    if cli_root is not None:
        return cli_root
    return Path(os.environ.get("LILITH_SNAPSHOT_ROOT", DEFAULT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lilith Chief-of-Staff read-only tracker recorder")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="capture one snapshot")
    run_p.add_argument("--root", type=Path, default=None, help="overrides LILITH_SNAPSHOT_ROOT")
    run_p.add_argument("--project-id", action="append", default=None,
                       help="Anima project id (repeatable or comma-separated; env LILITH_ANIMA_PROJECT_IDS); "
                            "default: the Akh-Medu project alone")
    run_p.add_argument("--detail-project-id", action="append", default=None,
                       help="projects (a subset of the above) captured with the cheaper `detail` profile: "
                            "work-item detail only, no loop state / sessions / changes / scheduling "
                            "(repeatable or comma-separated; env LILITH_ANIMA_DETAIL_PROJECT_IDS)")
    run_p.add_argument("--board-secrets-file", type=Path, default=None,
                       help="Agent Board token file, JSON {\"bearer_token\": ...} or plain (env LILITH_BOARD_SECRETS)")
    run_p.add_argument("--board-url", default=None, help=f"Agent Board base URL (env LILITH_BOARD_URL, default {DEFAULT_BOARD_URL})")
    run_p.add_argument("--fleet-config", type=Path, default=None,
                       help="fleet status endpoints JSON (env LILITH_FLEET_CONFIG)")

    gap_p = sub.add_parser("gap-report", help="expected-vs-actual captures, time gaps, statuses and errors")
    when = gap_p.add_mutually_exclusive_group(required=True)
    when.add_argument("--day", help="UTC day, YYYY-MM-DD")
    when.add_argument("--since", help="UTC start day, YYYY-MM-DD (range through --until or today)")
    gap_p.add_argument("--until", default=None, help="UTC end day for --since (inclusive)")
    gap_p.add_argument("--root", type=Path, default=None, help="overrides LILITH_SNAPSHOT_ROOT")
    gap_p.add_argument("--format", choices=("json", "markdown"), default="json")
    gap_p.add_argument("--markdown-out", type=Path, default=None, help="also write the markdown summary here")

    return parser


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = resolve_root(args.root)

    if args.command == "run":
        collectors = build_collectors(
            parse_project_ids(args.project_id, os.environ.get("LILITH_ANIMA_PROJECT_IDS")),
            detail_project_ids=parse_id_list(args.detail_project_id, os.environ.get("LILITH_ANIMA_DETAIL_PROJECT_IDS")),
            board_secrets_file=args.board_secrets_file or _env_path("LILITH_BOARD_SECRETS"),
            board_url=args.board_url or os.environ.get("LILITH_BOARD_URL") or DEFAULT_BOARD_URL,
            fleet_config=args.fleet_config or _env_path("LILITH_FLEET_CONFIG"),
        )
        manifest = run_snapshot(root, collectors)
        summary = {
            "schema": manifest["schema"],
            "timestamp": manifest["timestamp"],
            "collectors": {
                name: {
                    "status": c["status"],
                    "response_count": len(c["responses"]),
                    "error_count": len(c["errors"]),
                }
                for name, c in manifest["collectors"].items()
            },
        }
        print(json.dumps(summary, sort_keys=True))
        return 0

    if args.command == "gap-report":
        report = gap_report(root, args.day) if args.day else gap_report_range(root, args.since, args.until)
        markdown = render_markdown(report)
        if args.markdown_out:
            atomic_write_bytes(args.markdown_out, markdown.encode("utf-8"))
        print(markdown if args.format == "markdown" else json.dumps(report, indent=2, sort_keys=True))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
