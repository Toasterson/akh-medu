import importlib.util
import json
import sys
from pathlib import Path

import pytest

P = Path(__file__).with_name("cos_recorder.py")
spec = importlib.util.spec_from_file_location("cos_recorder", P)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


# --- fake MCP transport ------------------------------------------------------

class ScriptedTransport:
    """Fake JSON-RPC transport: dispatches through a handler(method, params)."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def post(self, url, headers, body):
        payload = json.loads(body)
        self.calls.append({"method": payload["method"], "params": payload.get("params"), "headers": headers})
        method = payload["method"]
        if method == "notifications/initialized":
            return {"Mcp-Session-Id": "sess-1"}, b""
        result = self.handler(method, payload.get("params") or {})
        envelope = {"jsonrpc": "2.0", "id": payload.get("id"), "result": result}
        return {"Mcp-Session-Id": "sess-1"}, json.dumps(envelope).encode()


def tool_call_envelope(payload):
    return {"content": [{"type": "text", "text": json.dumps(payload)}], "isError": False}


# --- read-only allowlist -----------------------------------------------------

def test_read_only_allowlist_refuses_unknown_tool_before_sending():
    transport = ScriptedTransport(lambda method, params: {})
    client = m.McpClient(url="http://fake", token="t", client_name="test", transport=transport)
    with pytest.raises(m.ReadOnlyToolRefused):
        m.call_read_tool(client, m.ANIMA_READ_TOOLS, "memory_write", {})
    assert transport.calls == []  # refused before any request was built


def test_read_only_allowlist_permits_listed_tool():
    def handler(method, params):
        if method == "tools/call":
            assert params["name"] == "list_plans"
            return tool_call_envelope({"plans": []})
        raise AssertionError(method)

    transport = ScriptedTransport(handler)
    client = m.McpClient(url="http://fake", token="t", client_name="test", transport=transport)
    response = m.call_read_tool(client, m.ANIMA_READ_TOOLS, "list_plans", {"project_id": "p"})
    assert response == {"plans": []}


def test_call_tool_captures_non_json_text_content_verbatim():
    # Observed live: get_thread sometimes returns a human-readable transcript
    # instead of a JSON envelope. That's a real, successful response and
    # should be captured as-is, not treated as a parse failure.
    def handler(method, params):
        if method == "tools/call":
            return {"content": [{"type": "text", "text": "Thread ID: abc\n[t] user: hello"}], "isError": False}
        raise AssertionError(method)

    client = m.McpClient(url="http://fake", token="t", client_name="test", transport=ScriptedTransport(handler))
    response = m.call_read_tool(client, m.ANIMA_READ_TOOLS, "get_thread", {"work_item_id": "w1"})
    assert response == "Thread ID: abc\n[t] user: hello"


def test_write_tool_names_are_not_in_the_allowlist():
    for suspicious in ("update_work_item", "create_work_item", "post_message", "hold_work_item", "cancel_session"):
        assert suspicious not in m.ANIMA_READ_TOOLS


# --- redaction ----------------------------------------------------------------

def test_redact_strips_bearer_and_authorization_header():
    assert "sekret" not in m.redact("failed: Authorization: Bearer sekrettoken123")
    assert "sekret" not in m.redact('{"Authorization": "Bearer sekrettoken123"}')
    assert "sekret" not in m.redact("request with Bearer sekrettoken123 rejected")


# --- content addressing / dedup / atomicity -----------------------------------

def test_store_object_dedup(tmp_path):
    d1, new1 = m.store_object(tmp_path, {"a": 1})
    d2, new2 = m.store_object(tmp_path, {"a": 1})
    assert d1 == d2
    assert new1 is True and new2 is False
    assert len(list((tmp_path / "objects").glob("*.json"))) == 1


def test_store_object_distinct_payloads_distinct_hashes(tmp_path):
    d1, _ = m.store_object(tmp_path, {"a": 1})
    d2, _ = m.store_object(tmp_path, {"a": 2})
    assert d1 != d2


def test_atomic_write_leaves_no_tmp_files(tmp_path):
    target = tmp_path / "sub" / "manifest.json"
    m.atomic_write_json(target, {"x": 1})
    assert json.loads(target.read_text())["x"] == 1
    leftovers = [p for p in target.parent.iterdir() if p.name.startswith(".")]
    assert leftovers == []


# --- pagination helper ---------------------------------------------------------

def test_extract_page_finds_items_and_cursor():
    items, cursor = m._extract_page({"work_items": [{"id": "1"}], "next_page_token": "p2"})
    assert items == [{"id": "1"}] and cursor == "p2"
    items, cursor = m._extract_page({"work_items": [{"id": "1"}], "next_page_token": None})
    assert items == [{"id": "1"}] and cursor is None


# --- live-item / changed-item filters -----------------------------------------

def test_is_live_excludes_done_and_cancelled():
    assert m._is_live({"current_phase": "working", "status": "open"}) is True
    assert m._is_live({"current_phase": "done", "status": "open"}) is False
    assert m._is_live({"current_phase": "testing", "status": "cancelled"}) is False


def test_changed_since_detects_new_and_updated():
    previous = {"w1": {"updated_at": "t0"}}
    current = [{"id": "w1", "updated_at": "t1"}, {"id": "w2", "updated_at": "t0"}]
    changed = m._changed_since(current, previous)
    assert set(changed) == {"w1", "w2"}  # w1 updated, w2 new


def test_changed_since_empty_when_unchanged():
    previous = {"w1": {"updated_at": "t0"}}
    current = [{"id": "w1", "updated_at": "t0"}]
    assert m._changed_since(current, previous) == []


# --- Anima collector: full capture through a fake MCP server -----------------

WI1 = {"id": "w1", "current_phase": "done", "status": "open", "updated_at": "t0"}
WI2 = {"id": "w2", "current_phase": "working", "status": "open", "updated_at": "t0"}
WI3 = {"id": "w3", "current_phase": "testing", "status": "cancelled", "updated_at": "t0"}
TOOLS_FIXTURE = [{"name": name, "inputSchema": {}} for name in sorted(m.ANIMA_READ_TOOLS)]


def make_anima_handler(call_log):
    def router(name, args):
        call_log.append((name, args))
        if name == "list_plans":
            return {"plans": [], "next_page_token": None}
        if name == "list_work_items":
            if args.get("page_token") == "p2":
                return {"work_items": [WI3], "next_page_token": None}
            return {"work_items": [WI1, WI2], "next_page_token": "p2"}
        if name == "get_work_item_detail":
            return {"work_item": {"id": args["id"]}}
        if name == "get_wi_loop_state":
            return {"work_item_id": args["work_item_id"]}
        if name == "list_sessions":
            return {"sessions": []}
        if name == "list_changes":
            return {"changes": []}
        if name == "get_scheduling_state":
            return {"work_item_id": args["work_item_id"]}
        if name == "get_thread":
            return {"messages": []}
        if name == "get_project_ci_status":
            return {"resolvable": True}
        raise AssertionError(f"unexpected tool {name}")

    def handler(method, params):
        if method == "initialize":
            return {"protocolVersion": "2025-03-26", "capabilities": {}}
        if method == "tools/list":
            return {"tools": TOOLS_FIXTURE}
        if method == "tools/call":
            payload = router(params["name"], params.get("arguments", {}))
            return tool_call_envelope(payload)
        raise AssertionError(f"unexpected method {method}")

    return handler


def test_anima_collector_full_capture_first_run(tmp_path):
    call_log = []
    transport = ScriptedTransport(make_anima_handler(call_log))
    collector = m.AnimaCollector(
        project_id="proj", mcp_url="http://fake", pat_resolver=lambda: "secret-token-abc", transport=transport
    )
    ctx = m.SnapshotContext(root=tmp_path, timestamp="20260101T000000Z", previous_work_items={})
    result = collector.collect(ctx)

    assert result.status == "ok"
    names = [n for n, _ in call_log]
    assert names.count("list_work_items") == 2  # paginated
    assert names.count("get_work_item_detail") == 1  # only the one live item (w2)
    assert names.count("get_wi_loop_state") == 1
    assert names.count("get_thread") == 3  # first run: every item is "changed"
    assert result.extra["work_item_count"] == 3
    assert result.extra["live_work_item_count"] == 1
    assert result.extra["changed_item_count"] == 3

    # tool contract recorded with its own sha256
    assert result.tool_contract_sha256 is not None
    contract_path = tmp_path / "objects" / f"{result.tool_contract_sha256}.json"
    assert contract_path.exists()
    assert json.loads(contract_path.read_text())["tools"] == TOOLS_FIXTURE

    # no secret leaks into any stored object
    for obj in (tmp_path / "objects").glob("*.json"):
        assert "secret-token-abc" not in obj.read_text()


def test_anima_collector_second_run_only_captures_changed_thread(tmp_path):
    call_log = []
    transport = ScriptedTransport(make_anima_handler(call_log))
    collector = m.AnimaCollector(
        project_id="proj", mcp_url="http://fake", pat_resolver=lambda: "t", transport=transport
    )
    previous = {"w1": WI1, "w2": WI2, "w3": WI3}  # identical updated_at -> nothing changed
    ctx = m.SnapshotContext(root=tmp_path, timestamp="20260101T020000Z", previous_work_items=previous)
    result = collector.collect(ctx)
    assert result.extra["changed_item_count"] == 0
    assert [n for n, _ in call_log].count("get_thread") == 0


def test_anima_collector_pat_failure_is_fail_closed(tmp_path):
    def boom():
        raise RuntimeError("no keychain entry, Authorization: Bearer topsecret")

    collector = m.AnimaCollector(pat_resolver=boom, transport=ScriptedTransport(lambda *_: {}))
    ctx = m.SnapshotContext(root=tmp_path, timestamp="t", previous_work_items={})
    result = collector.collect(ctx)
    assert result.status == "error"
    assert "topsecret" not in result.errors[0]
    assert result.responses == []


def test_anima_collector_partial_on_per_item_tool_error(tmp_path):
    call_log = []

    def handler(method, params):
        if method == "initialize":
            return {}
        if method == "tools/list":
            return {"tools": TOOLS_FIXTURE}
        if method == "tools/call":
            name = params["name"]
            call_log.append(name)
            if name == "get_wi_loop_state":
                raise RuntimeError("boom")  # simulated by handler raising -> caught below
            return make_anima_handler([])(method, params)
        raise AssertionError(method)

    # ScriptedTransport doesn't catch handler exceptions; call_tool propagates them
    transport = ScriptedTransport(handler)
    collector = m.AnimaCollector(project_id="proj", mcp_url="http://fake", pat_resolver=lambda: "t", transport=transport)
    ctx = m.SnapshotContext(root=tmp_path, timestamp="t", previous_work_items={})
    result = collector.collect(ctx)
    assert result.status == "partial"
    assert any("get_wi_loop_state" in e for e in result.errors)
    # other calls still happened
    assert "get_work_item_detail" in call_log


# --- Git collector -------------------------------------------------------------

class _Proc:
    def __init__(self, stdout):
        self.stdout = stdout


def test_git_collector_parses_refs(tmp_path):
    def fake_runner(cmd):
        assert cmd[:2] == ["git", "ls-remote"]
        return _Proc("abc123\trefs/heads/main\ndef456\trefs/tags/v1\n")

    ctx = m.SnapshotContext(root=tmp_path, timestamp="t", previous_work_items={})
    result = m.GitCollector(runner=fake_runner).collect(ctx)
    assert result.status == "ok"
    assert result.extra["ref_count"] == 2
    obj = json.loads((tmp_path / "objects" / f"{result.responses[0].sha256}.json").read_text())
    assert obj["refs"] == [{"sha": "abc123", "ref": "refs/heads/main"}, {"sha": "def456", "ref": "refs/tags/v1"}]


def test_git_collector_fails_closed_on_subprocess_error(tmp_path):
    def fake_runner(cmd):
        raise OSError("git not found")

    ctx = m.SnapshotContext(root=tmp_path, timestamp="t", previous_work_items={})
    result = m.GitCollector(runner=fake_runner).collect(ctx)
    assert result.status == "error"
    assert result.responses == []


# --- Board collector -------------------------------------------------------------

def test_board_collector_skips_without_env(monkeypatch, tmp_path):
    monkeypatch.delenv("BOARD_MCP_URL", raising=False)
    monkeypatch.delenv("BOARD_TOKEN", raising=False)
    ctx = m.SnapshotContext(root=tmp_path, timestamp="t", previous_work_items={})
    result = m.BoardCollector().collect(ctx)
    assert result.status == "skipped"
    assert "BOARD_MCP_URL" in result.extra["reason"]


def test_board_collector_records_contract_when_configured(monkeypatch, tmp_path):
    monkeypatch.setenv("BOARD_MCP_URL", "http://board.fake/mcp")
    monkeypatch.setenv("BOARD_TOKEN", "board-token")
    monkeypatch.delenv("BOARD_READ_TOOLS", raising=False)

    def handler(method, params):
        if method == "initialize":
            return {}
        if method == "tools/list":
            return {"tools": [{"name": "list_threads"}]}
        raise AssertionError("no tool calls expected without BOARD_READ_TOOLS")

    result = m.BoardCollector(transport=ScriptedTransport(handler)).collect(
        m.SnapshotContext(root=tmp_path, timestamp="t", previous_work_items={})
    )
    assert result.status == "ok"
    assert result.tool_contract_sha256 is not None
    assert result.extra["called_tools"] == 0


def test_board_collector_calls_allowlisted_tools(monkeypatch, tmp_path):
    monkeypatch.setenv("BOARD_MCP_URL", "http://board.fake/mcp")
    monkeypatch.setenv("BOARD_TOKEN", "board-token")
    monkeypatch.setenv("BOARD_READ_TOOLS", "list_threads")

    def handler(method, params):
        if method == "initialize":
            return {}
        if method == "tools/list":
            return {"tools": [{"name": "list_threads"}, {"name": "post_message"}]}
        if method == "tools/call":
            assert params["name"] == "list_threads"
            return tool_call_envelope({"threads": []})
        raise AssertionError(method)

    result = m.BoardCollector(transport=ScriptedTransport(handler)).collect(
        m.SnapshotContext(root=tmp_path, timestamp="t", previous_work_items={})
    )
    assert result.status == "ok"
    assert result.extra["called_tools"] == 1
    assert result.responses[0].call == "list_threads"


# --- Fleet collector -------------------------------------------------------------

def test_fleet_collector_is_not_configured(tmp_path):
    result = m.FleetCollector().collect(m.SnapshotContext(root=tmp_path, timestamp="t", previous_work_items={}))
    assert result.status == "not_configured"


# --- Orchestration: fail-closed, manifest schema, atomicity -----------------

class _OkCollector:
    name = "ok"

    def collect(self, ctx):
        return m.CollectorResult(status="ok", responses=[m.ResponseRecord("x", {}, "deadbeef", True)])


class _BoomCollector:
    name = "boom"

    def collect(self, ctx):
        raise RuntimeError("kaboom with Authorization: Bearer topsecretvalue")


def test_run_snapshot_fail_closed_per_collector_and_manifest_schema(tmp_path):
    manifest = m.run_snapshot(tmp_path, collectors=[_OkCollector(), _BoomCollector()])
    assert manifest["schema"] == "lilith/cos-snapshot@1"
    assert "timestamp" in manifest
    assert manifest["collectors"]["ok"]["status"] == "ok"
    assert manifest["collectors"]["boom"]["status"] == "error"
    dumped = json.dumps(manifest)
    assert "topsecretvalue" not in dumped
    assert "[redacted]" in manifest["collectors"]["boom"]["errors"][0]

    snap_dirs = [p for p in tmp_path.iterdir() if p.is_dir() and p.name != "objects"]
    assert len(snap_dirs) == 1
    assert (snap_dirs[0] / "manifest.json").exists()
    on_disk = json.loads((snap_dirs[0] / "manifest.json").read_text())
    assert on_disk == manifest


def test_run_snapshot_uses_atomic_rename(tmp_path, monkeypatch):
    calls = []
    real_replace = m.os.replace

    def spy_replace(src, dst):
        calls.append((str(src), str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(m.os, "replace", spy_replace)
    m.run_snapshot(tmp_path, collectors=[_OkCollector()])
    assert calls  # at least the manifest write went through tmp+rename
    for src, _dst in calls:
        assert ".tmp-" in src


# --- Second-run dedup across snapshots ---------------------------------------

def test_second_snapshot_dedups_identical_git_object(tmp_path):
    def fake_runner(cmd):
        return _Proc("abc123\trefs/heads/main\n")

    collectors = [m.GitCollector(runner=fake_runner)]
    first = m.run_snapshot(tmp_path, collectors=collectors, now=__import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").timezone.utc))
    second = m.run_snapshot(tmp_path, collectors=collectors, now=__import__("datetime").datetime(2026, 1, 1, 2, tzinfo=__import__("datetime").timezone.utc))
    sha1 = first["collectors"]["git"]["responses"][0]["sha256"]
    sha2 = second["collectors"]["git"]["responses"][0]["sha256"]
    assert sha1 == sha2
    assert len(list((tmp_path / "objects").glob("*.json"))) == 1


# --- find_previous_snapshot / load_previous_work_items -----------------------

def test_load_previous_work_items_round_trips_through_object_store(tmp_path):
    call_log = []
    transport = ScriptedTransport(make_anima_handler(call_log))
    collector = m.AnimaCollector(project_id="proj", mcp_url="http://fake", pat_resolver=lambda: "t", transport=transport)
    manifest = m.run_snapshot(tmp_path, collectors=[collector], now=__import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").timezone.utc))
    previous_dir = m.find_previous_snapshot(tmp_path)
    assert previous_dir is not None
    loaded = m.load_previous_work_items(tmp_path, previous_dir)
    assert set(loaded) == {"w1", "w2", "w3"}


# --- gap report ------------------------------------------------------------

def _write_manifest(root, name, collectors):
    d = root / name
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"schema": m.SCHEMA, "timestamp": name, "collectors": collectors}))


def test_gap_report_missing_slots_and_failures(tmp_path):
    _write_manifest(tmp_path, "20260928T000512Z", {"anima": {"status": "ok", "responses": [], "errors": []}})
    _write_manifest(tmp_path, "20260928T041000Z", {"anima": {"status": "error", "responses": [], "errors": ["boom"]}})
    # outside the requested day: ignored
    _write_manifest(tmp_path, "20260929T000000Z", {"anima": {"status": "ok", "responses": [], "errors": []}})

    report = m.gap_report(tmp_path, "2026-09-28")
    assert report["day"] == "2026-09-28"
    assert len(report["expected_slots"]) == 12  # every 2h across a day
    assert "2026-09-28T02:00:00+00:00" in report["missing_slots"]
    assert "2026-09-28T00:00:00+00:00" not in report["missing_slots"]
    assert len(report["actual_snapshots"]) == 2
    assert any(f["collector"] == "anima" and f["status"] == "error" for f in report["failures"])


def test_gap_report_empty_root(tmp_path):
    report = m.gap_report(tmp_path / "does-not-exist", "2026-09-28")
    assert report["actual_snapshots"] == []
    assert len(report["missing_slots"]) == 12


# --- CLI wiring ---------------------------------------------------------------

def test_resolve_root_prefers_cli_then_env(monkeypatch):
    monkeypatch.setenv("LILITH_SNAPSHOT_ROOT", "/env/root")
    assert m.resolve_root(Path("/cli/root")) == Path("/cli/root")
    assert m.resolve_root(None) == Path("/env/root")
    monkeypatch.delenv("LILITH_SNAPSHOT_ROOT")
    assert m.resolve_root(None) == Path(m.DEFAULT_ROOT)


def test_main_run_writes_snapshot_and_prints_summary(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(m, "DEFAULT_COLLECTORS", (_OkCollector(),))
    rc = m.main(["run", "--root", str(tmp_path)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["schema"] == "lilith/cos-snapshot@1"
    assert out["collectors"]["ok"]["status"] == "ok"
