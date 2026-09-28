#!/usr/bin/env python3
"""Lilith Chief-of-Staff read-only tracker recorder (Anima WI 01a0e270-137c,
plan slice C3, ADR 116 Decision 3).

Every run captures a content-addressed snapshot of what Lilith would read:
Anima (plans, work items, live-item detail/loop-state, changed-item threads,
sessions, changes, CI status, scheduling state), the akh-medu git refs, the
Agent Board (when configured) and fleet status (stub extension point).

Read-only guarantee: every MCP tool call is checked against an explicit
allowlist *before* a request is built. A disallowed tool name raises
`ReadOnlyToolRefused` and nothing is sent over the wire. Never call a write
tool.

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
    cos_recorder.py run
    cos_recorder.py gap-report --day 2026-09-28
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

SCHEMA = "lilith/cos-snapshot@1"
DEFAULT_ROOT = "/Volumes/akh-medu-runs/lilith/cos-snapshots"
ANIMA_MCP_URL = "https://anima.wegmueller.it/mcp"
ANIMA_PROJECT_ID = "019d9844-ce74-7993-a9f0-c953b8f04a4e"
GIT_REMOTE = "https://code.aopc.cloud/toasterson/akh-medu.git"
SLOT_HOURS = 2

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
            raise RuntimeError(f"tool call {name!r} failed: {result['error']}")
        payload = result["result"]
        content = payload.get("content") or []
        text = content[0].get("text", "") if content else ""
        if payload.get("isError"):
            raise RuntimeError(f"tool call {name!r} returned an error: {text or '(no message)'}")
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

    def to_dict(self) -> dict:
        return {"call": self.call, "args": self.args, "sha256": self.sha256}


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

@dataclasses.dataclass
class AnimaCollector:
    project_id: str = ANIMA_PROJECT_ID
    mcp_url: str = ANIMA_MCP_URL
    pat_resolver: Callable[[], str] | None = None
    transport: Transport = dataclasses.field(default_factory=UrllibTransport)
    name: str = dataclasses.field(default="anima", init=False)

    def collect(self, ctx: SnapshotContext) -> CollectorResult:
        result = CollectorResult(status="ok")
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

        def call(cname: str, cargs: dict) -> Any:
            try:
                response = call_read_tool(client, ANIMA_READ_TOOLS, cname, cargs)
            except Exception as error:
                result.errors.append(f"{cname}({cargs}): {redact(str(error))}")
                result.status = "partial"
                return None
            digest, stored_new = store_object(ctx.root, response)
            result.responses.append(ResponseRecord(cname, cargs, digest, stored_new))
            return response

        plans_items = self._paginate(call, "list_plans", {"project_id": self.project_id})
        work_items = self._paginate(call, "list_work_items", {"project_id": self.project_id})

        result.extra["plan_count"] = len(plans_items)
        result.extra["work_item_count"] = len(work_items)

        live_items = [wi for wi in work_items if _is_live(wi)]
        result.extra["live_work_item_count"] = len(live_items)

        for wi in live_items:
            wid = wi.get("id")
            if not wid:
                continue
            call("get_work_item_detail", {"id": wid})
            call("get_wi_loop_state", {"work_item_id": wid})
            call("list_sessions", {"work_item_id": wid})
            call("list_changes", {"work_item_id": wid})
            call("get_scheduling_state", {"work_item_id": wid})

        changed_items = _changed_since(work_items, ctx.previous_work_items)
        result.extra["changed_item_count"] = len(changed_items)
        thread_targets = _thread_targets(work_items, ctx.previous_work_items)
        result.extra["thread_target_count"] = len(thread_targets)
        for wid in thread_targets:
            call("get_thread", {"work_item_id": wid})

        call("get_project_ci_status", {"project_id": self.project_id})

        ctx.current_work_items = {wi["id"]: wi for wi in work_items if wi.get("id")}

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


# --- Agent Board collector (pluggable, env-gated) ---------------------------

@dataclasses.dataclass
class BoardCollector:
    """Reads the Agent Board over its own MCP endpoint when configured.

    `BOARD_MCP_URL` / `BOARD_TOKEN` select the server. `BOARD_READ_TOOLS`
    (comma-separated) is the read allowlist for that server's tool names;
    the board's tool contract is unknown ahead of time, so without an
    explicit allowlist the collector records the contract only and calls
    nothing — it never guesses at a write tool.
    """

    transport: Transport = dataclasses.field(default_factory=UrllibTransport)
    name: str = dataclasses.field(default="board", init=False)

    def collect(self, ctx: SnapshotContext) -> CollectorResult:
        url = os.environ.get("BOARD_MCP_URL")
        token = os.environ.get("BOARD_TOKEN")
        if not url or not token:
            return CollectorResult(status="skipped", extra={"reason": "BOARD_MCP_URL/BOARD_TOKEN not set"})

        result = CollectorResult(status="ok")
        client = McpClient(url, token, "lilith-cos-recorder", transport=self.transport)
        try:
            client.initialize()
            tools = client.list_tools()
        except Exception as error:
            result.status = "error"
            result.errors.append(redact(str(error)))
            return result
        result.tool_contract_sha256, _ = store_object(ctx.root, {"tools": tools})

        tool_names = {t.get("name") for t in tools if isinstance(t, dict)}
        allow_env = os.environ.get("BOARD_READ_TOOLS", "")
        read_tools = frozenset(x.strip() for x in allow_env.split(",") if x.strip())
        called = 0
        for tname in sorted(read_tools & tool_names):
            try:
                response = call_read_tool(client, read_tools, tname, {})
            except Exception as error:
                result.errors.append(f"{tname}: {redact(str(error))}")
                result.status = "partial"
                continue
            digest, stored_new = store_object(ctx.root, response)
            result.responses.append(ResponseRecord(tname, {}, digest, stored_new))
            called += 1
        result.extra["tool_count"] = len(tool_names)
        result.extra["called_tools"] = called
        if not read_tools:
            result.extra["note"] = "BOARD_READ_TOOLS not set; recorded the tool contract only"
        return result


# --- Fleet status collector (stub extension point) --------------------------

@dataclasses.dataclass
class FleetCollector:
    """Stub. Fleet/run status has no read-only MCP surface wired yet; when one
    lands, replace this with a `collect()` following the Board pattern."""

    name: str = dataclasses.field(default="fleet", init=False)

    def collect(self, ctx: SnapshotContext) -> CollectorResult:
        return CollectorResult(
            status="not_configured",
            extra={"reason": "fleet/run status source not wired yet; extension point"},
        )


DEFAULT_COLLECTORS: tuple[Collector, ...] = (AnimaCollector(), GitCollector(), BoardCollector(), FleetCollector())


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
    anima = manifest.get("collectors", {}).get("anima", {})
    items: dict[str, dict] = {}
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

def _day_bounds(day: str) -> tuple[datetime, datetime]:
    d = date.fromisoformat(day)
    start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    return start, start + timedelta(days=1)


def gap_report(root: Path, day: str) -> dict:
    start, end = _day_bounds(day)
    expected = []
    t = start
    while t < end:
        expected.append(t)
        t += timedelta(hours=SLOT_HOURS)

    captured_slots: set[datetime] = set()
    actual_snapshots: list[str] = []
    failures: list[dict] = []

    if root.exists():
        for p in sorted(root.iterdir()):
            if not p.is_dir() or not (p / "manifest.json").exists():
                continue
            try:
                ts = datetime.strptime(p.name, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if not (start <= ts < end):
                continue
            actual_snapshots.append(p.name)
            slot_index = int((ts - start).total_seconds() // (SLOT_HOURS * 3600))
            captured_slots.add(start + timedelta(hours=SLOT_HOURS * slot_index))
            try:
                manifest = json.loads((p / "manifest.json").read_text())
            except (OSError, ValueError):
                failures.append({"snapshot": p.name, "collector": None, "status": "unreadable_manifest", "errors": []})
                continue
            for cname, cresult in manifest.get("collectors", {}).items():
                if cresult.get("status") in ("error", "partial") or cresult.get("errors"):
                    failures.append(
                        {
                            "snapshot": p.name,
                            "collector": cname,
                            "status": cresult.get("status"),
                            "errors": cresult.get("errors", []),
                        }
                    )

    missing = [t.isoformat() for t in expected if t not in captured_slots]
    return {
        "day": day,
        "expected_slots": [t.isoformat() for t in expected],
        "captured_slots": sorted(t.isoformat() for t in captured_slots),
        "missing_slots": missing,
        "actual_snapshots": actual_snapshots,
        "failures": failures,
    }


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

    gap_p = sub.add_parser("gap-report", help="expected-vs-actual captures and failures for a day")
    gap_p.add_argument("--day", required=True, help="UTC day, YYYY-MM-DD")
    gap_p.add_argument("--root", type=Path, default=None, help="overrides LILITH_SNAPSHOT_ROOT")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = resolve_root(args.root)

    if args.command == "run":
        manifest = run_snapshot(root)
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
        report = gap_report(root, args.day)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
