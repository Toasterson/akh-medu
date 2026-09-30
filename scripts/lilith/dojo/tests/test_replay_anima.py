"""Anima reads the recorder holds as *data* are served whatever their argument spelling (`dojo.replay_anima`).

The fixture mirrors the real `lilith/cos-snapshot@1` layout the C3 recorder writes: one primary `anima`
collector plus an `anima:<project_id>` collector per extra project, each with a cursor-linked
`list_work_items` walk (opaque `page_token`s, no `page_size` recorded), `list_plans`, and detail records for
live items only. Synthetic ids; nothing private.
"""
import json
from datetime import datetime, timezone

import cos_recorder as rec
import pytest

from dojo import replay_anima
from dojo.replay_coverage import coverage_of, main as coverage_main
from dojo.replay_server import DojoLedger, ReplayServer
from dojo.snapshot import SnapshotStore, canonical_json, sha256_hex

P1 = "11111111-1111-7111-8111-111111111111"
P2 = "22222222-2222-7222-8222-222222222222"
PLAN_A, PLAN_B, PLAN_C = "aaaaaaaa-0000-7000-8000-000000000001", "aaaaaaaa-0000-7000-8000-000000000002", "cccccccc-0000-7000-8000-000000000003"
WHEN = datetime(2026, 9, 29, 22, 48, 15, tzinfo=timezone.utc)
SID = WHEN.strftime("%Y%m%dT%H%M%SZ")

LIST_SCHEMA = {"type": "object", "properties": {
    "current_phase": {"type": "string"}, "page_size": {"type": "integer"}, "page_token": {"type": "string"},
    "plan_id": {"type": "string"}, "project_id": {"type": "string"}}, "required": []}
ID_SCHEMA = {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}
DETAIL_SCHEMA = {"type": "object", "properties": {
    "id": {"type": "string"}, "change_page_size": {"type": "integer"}, "message_page_size": {"type": "integer"}},
    "required": ["id"]}
TOOLS = [
    {"name": "list_work_items", "inputSchema": LIST_SCHEMA},
    {"name": "list_projects", "inputSchema": {"type": "object", "properties": {"page_size": {"type": "integer"}}}},
    {"name": "get_work_item_by_id", "inputSchema": ID_SCHEMA},
    {"name": "get_work_item_detail", "inputSchema": DETAIL_SCHEMA},
    {"name": "get_plan", "inputSchema": ID_SCHEMA},
    {"name": "list_plans", "inputSchema": {"type": "object", "properties": {
        "project_id": {"type": "string"}, "page_token": {"type": "string"}}, "required": ["project_id"]}},
    {"name": "lane_occupancy", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "stranded_work_items", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "seshat_status", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "find", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}}},
]


def item(wid, code, phase, plan, project):
    return {"id": wid, "code": code, "current_phase": phase, "status": "open", "plan_id": plan, "priority": "high",
            "project_id": project, "title": f"title {code}"}


P1_ITEMS = [item("w1", "WI-1", "working", PLAN_A, P1), item("w2", "WI-2", "parked", PLAN_A, P1),
            item("w3", "WI-3", "done", PLAN_B, P1), item("w4", "WI-4", "ready", PLAN_B, P1),
            item("w5", "WI-5", "draft", None, P1)]
P2_ITEMS = [item("x1", "WI-1", "working", PLAN_C, P2)]
LIVE = {"w1", "w2", "w4", "w5", "x1"}


def detail(wi):
    return {"work_item": {k: wi[k] for k in ("id", "code", "current_phase", "plan_id", "project_id")},
            "prs": [], "change_history": [], "thread_messages": [], "thread_messages_elided": False}


def build_root(root, truncate_walk=False, when=WHEN, projects=None):
    responses = {"anima": [], f"anima:{P2}": []}

    def put(collector, call, args, payload):
        digest, _ = rec.store_object(root, payload)
        responses[collector].append({"call": call, "args": args, "sha256": digest})

    def walk(collector, project, items, per_page):
        pages = [items[i:i + per_page] for i in range(0, len(items), per_page)]
        tokens = [None] + [f"tok{n}" for n in range(1, len(pages))]
        for n, page in enumerate(pages):
            last = n == len(pages) - 1
            args = {"project_id": project} | ({"page_token": tokens[n]} if tokens[n] else {})
            nxt = f"tok{n + 1}" if (not last or truncate_walk) else None
            put(collector, "list_work_items", args, {"work_items": page, "total": len(items), "next_page_token": nxt})
            if truncate_walk and last:
                break

    put("anima", "list_plans", {"project_id": P1}, {"plans": [{"id": PLAN_A, "title": "A"}, {"id": PLAN_B, "title": "B"}],
                                                   "next_page_token": None, "total": 2})
    walk("anima", P1, P1_ITEMS, 2)
    put(f"anima:{P2}", "list_plans", {"project_id": P2}, {"plans": [{"id": PLAN_C, "title": "C"}],
                                                         "next_page_token": None, "total": 1})
    walk(f"anima:{P2}", P2, P2_ITEMS, 50)
    for wi in P1_ITEMS + P2_ITEMS:
        if wi["id"] in LIVE:
            put("anima" if wi["project_id"] == P1 else f"anima:{P2}", "get_work_item_detail", {"id": wi["id"]}, detail(wi))
    for collector, project in (("anima", P1), (f"anima:{P2}", P2)):
        put(collector, "get_project_ci_status", {"project_id": project}, {"resolvable": True})
    if projects is not None:  # a later recorder that captured list_projects itself
        put("anima", "list_projects", {}, {"projects": projects, "total": len(projects)})
    contract_sha, _ = rec.store_object(root, {"tools": TOOLS})
    refs_sha, _ = rec.store_object(root, {"remote": rec.GIT_REMOTE, "refs": [{"ref": "refs/heads/main", "sha": "1" * 40}]})
    collectors = {
        "anima": {"status": "ok", "errors": [], "responses": responses["anima"], "tool_contract_sha256": contract_sha,
                  "extra": {"project_id": P1, "profile": "full", "work_item_count": 5, "plan_count": 2}},
        f"anima:{P2}": {"status": "ok", "errors": [], "responses": responses[f"anima:{P2}"],
                        "tool_contract_sha256": contract_sha,
                        "extra": {"project_id": P2, "profile": "detail", "work_item_count": 1, "plan_count": 1}},
        "git": {"status": "ok", "errors": [], "responses": [
            {"call": "git_ls_remote", "args": {"remote": rec.GIT_REMOTE}, "sha256": refs_sha}]},
        "board": {"status": "ok", "errors": [], "responses": []},
        "fleet": {"status": "ok", "errors": [], "responses": []},
    }
    sid = when.strftime("%Y%m%dT%H%M%SZ")
    rec.atomic_write_json(root / sid / "manifest.json", {"schema": rec.SCHEMA, "timestamp": when.isoformat(),
                                                         "collectors": collectors})
    return SnapshotStore(root).get(sid)


@pytest.fixture
def snap(tmp_path):
    return build_root(tmp_path / "snapshots")


@pytest.fixture
def server(snap, tmp_path):
    return ReplayServer(snap, DojoLedger(tmp_path / "ledger.jsonl"))


def call(server, tool, arguments):
    response = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                              "params": {"name": tool, "arguments": arguments}})["result"]
    return response["isError"], json.loads(response["content"][0]["text"])


def ids(payload):
    return [wi["id"] for wi in payload["work_items"]]


def test_list_work_items_without_arguments_is_the_union_of_the_recorded_projects(server):
    error, payload = call(server, "list_work_items", {})
    assert not error
    assert ids(payload) == ["w1", "w2", "w3", "w4", "w5", "x1"] and payload["total"] == 6
    assert payload["next_page_token"] is None
    receipt = server.receipts.receipts[-1]
    assert "derive:list_work_items" in receipt.match_rules and "scope:all-recorded-projects" in receipt.match_rules
    assert receipt.response_sha256 == sha256_hex(canonical_json(payload)) and receipt.source.value == "snapshot"


@pytest.mark.parametrize("size", [1, 2, 3, 100])
def test_any_page_size_walks_the_recorded_items_in_order(server, size):
    collected, token = [], None
    for _ in range(10):
        args = {"project_id": P1, "page_size": size} | ({"page_token": token} if token else {})
        error, payload = call(server, "list_work_items", args)
        assert not error and payload["total"] == 5
        collected += ids(payload)
        token = payload["next_page_token"]
        if token is None:
            break
    assert collected == ["w1", "w2", "w3", "w4", "w5"]


def test_a_cursor_the_recorder_saw_is_honoured_with_a_new_page_size(server):
    error, recorded = call(server, "list_work_items", {"project_id": P1, "page_token": "tok1"})
    assert not error and ids(recorded) == ["w3", "w4"]  # the recorded page itself, exact
    assert server.receipts.receipts[-1].match_rules == ()
    error, paged = call(server, "list_work_items", {"project_id": P1, "page_token": "tok1", "page_size": 1})
    assert not error and ids(paged) == ["w3"] and paged["next_page_token"] is not None


def test_filters_narrow_the_recorded_items(server):
    assert ids(call(server, "list_work_items", {"current_phase": "parked"})[1]) == ["w2"]
    assert ids(call(server, "list_work_items", {"project_id": P1, "plan_id": PLAN_B})[1]) == ["w3", "w4"]
    assert ids(call(server, "list_work_items", {"plan_id": PLAN_C, "page_size": 100})[1]) == ["x1"]
    error, none = call(server, "list_work_items", {"current_phase": "testing"})
    assert not error and none["work_items"] == [] and none["total"] == 0


def test_a_project_that_was_not_recorded_has_its_own_reason(server):
    error, payload = call(server, "list_work_items", {"project_id": "99999999-0000-7000-8000-000000000000"})
    assert error and payload["reason"] == "project-not-recorded"
    assert payload["recorded_projects"] == [P1, P2]
    error, payload = call(server, "list_plans", {"project_id": "99999999-0000-7000-8000-000000000000"})
    assert error and payload["reason"] == "project-not-recorded"
    # a plan id passed as a project id (a model mistake seen in the teacher run) says so too
    error, payload = call(server, "list_plans", {"project_id": PLAN_A})
    assert error and payload["reason"] == "project-not-recorded"


def test_an_unknown_cursor_or_argument_is_an_error_not_a_guess(server):
    error, payload = call(server, "list_work_items", {"project_id": P1, "page_token": "not-a-cursor"})
    assert error and payload["reason"] == "page-token-unknown"
    error, payload = call(server, "list_work_items", {"project_id": P1, "assignee": "x"})
    assert error and payload["reason"] == "arguments-not-recorded"


def test_list_projects_answers_from_the_recorded_project_set(server):
    error, payload = call(server, "list_projects", {})
    assert not error and [p["id"] for p in payload["projects"]] == [P1, P2] and payload["total"] == 2
    error, first = call(server, "list_projects", {"page_size": 1})
    assert not error and [p["id"] for p in first["projects"]] == [P1]


def test_derived_list_projects_takes_names_from_the_earliest_snapshot_that_recorded_them(tmp_path):
    root = tmp_path / "snapshots"
    old = build_root(root)  # recorded before the recorder captured list_projects: ids only
    meta = {"description": "d", "git_remote": "ssh://git/x.git", "status": "active"}
    later_when = datetime(2026, 9, 30, 8, 0, tzinfo=timezone.utc)
    later = build_root(root, when=later_when, projects=[
        {"id": P1, "name": "Alpha", **meta}, {"id": P2, "name": "Beta", **meta},
        {"id": "33333333-3333-7333-8333-333333333333", "name": "Later project", **meta}])
    newest = build_root(root, when=datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc), projects=[
        {"id": P1, "name": "Alpha renamed", **meta}])
    server = ReplayServer(old, DojoLedger(tmp_path / "l.jsonl"))
    error, payload = call(server, "list_projects", {})
    assert not error and payload["total"] == 2  # the later project is not in this snapshot's collectors
    assert [(p["id"], p["name"]) for p in payload["projects"]] == [(P1, "Alpha"), (P2, "Beta")]  # earliest wins
    assert all(p["git_remote"] == meta["git_remote"] and p["status"] == "active" for p in payload["projects"])
    assert all(p["work_item_count"] >= 1 for p in payload["projects"])
    assert server.receipts.receipts[-1].match_rules == (f"derive:list_projects<-collectors+names@{later.id}",)
    # a snapshot that recorded the call itself keeps its own bytes
    own = ReplayServer(newest, DojoLedger(tmp_path / "l2.jsonl"))
    error, payload = call(own, "list_projects", {})
    assert not error and payload["projects"][0]["name"] == "Alpha renamed" and own.receipts.receipts[-1].match_rules == ()


def test_derived_list_projects_keeps_ids_only_when_no_snapshot_recorded_names(server):
    error, payload = call(server, "list_projects", {})
    assert not error and all("name" not in p for p in payload["projects"])
    assert server.receipts.receipts[-1].match_rules == ("derive:list_projects<-collectors",)


def test_work_item_lookups_answer_for_every_recorded_item(server):
    for wid, project in (("w1", P1), ("w3", P1), ("x1", P2)):  # live, done (no detail recorded), other project
        error, payload = call(server, "get_work_item_by_id", {"id": wid})
        assert not error and payload["work_item"]["id"] == wid and payload["work_item"]["project_id"] == project
    error, payload = call(server, "get_work_item_by_id", {"id": "nowhere"})
    assert error and payload["reason"] == "work-item-not-recorded"
    error, payload = call(server, "get_work_item_detail", {"id": "x1"})
    assert not error and payload["work_item"]["id"] == "x1"
    error, payload = call(server, "get_work_item_detail", {"id": "w3"})  # listed, detail never recorded (terminal)
    assert error and payload["reason"] == "detail-not-recorded" and payload["listed"]["id"] == "w3"
    error, payload = call(server, "get_work_item_detail", {"id": "nowhere"})
    assert error and payload["reason"] == "work-item-not-recorded"


def test_detail_history_page_sizes_are_covered_by_a_default_sized_untruncated_recording(server):
    error, payload = call(server, "get_work_item_detail", {"id": "w1", "message_page_size": 100, "change_page_size": 10})
    assert not error and payload["work_item"]["id"] == "w1"
    assert "derive:get_work_item_detail:history-page-sizes" in server.receipts.receipts[-1].match_rules


def test_get_plan_answers_from_the_recorded_plans(server):
    error, payload = call(server, "get_plan", {"id": PLAN_A})
    assert not error and payload["plan"]["id"] == PLAN_A
    error, payload = call(server, "get_plan", {"id": PLAN_C})  # another project's plan
    assert not error and payload["plan"]["title"] == "C"
    error, payload = call(server, "get_plan", {"id": "nowhere"})
    assert error and payload["reason"] == "plan-not-recorded"


def test_tools_the_snapshot_holds_nothing_for_stay_not_recorded(server):
    for tool in ("lane_occupancy", "stranded_work_items", "seshat_status"):
        error, payload = call(server, tool, {})
        assert error and payload["reason"] == "tool-not-recorded"
    error, payload = call(server, "find", {"query": "anything"})
    assert error and payload["reason"] == "not-replayable"


def test_a_walk_that_does_not_end_in_an_empty_cursor_cannot_back_a_filter(tmp_path):
    truncated = build_root(tmp_path / "truncated", truncate_walk=True)
    server = ReplayServer(truncated, DojoLedger(tmp_path / "l.jsonl"))
    error, payload = call(server, "list_work_items", {"project_id": P1, "current_phase": "parked"})
    assert error and payload["reason"] == "listing-incomplete" and payload["incomplete_projects"] == [P1]
    error, payload = call(server, "list_work_items", {"project_id": P1, "page_token": "tok1"})
    assert not error  # the recorded page itself is still answered exactly


def test_derivation_needs_no_tool_name_prefix_and_reads_the_recorded_layout(snap):
    listings = replay_anima.project_listings(snap)
    assert list(listings) == [P1, P2]  # primary first
    assert listings[P1].complete and [w["id"] for w in listings[P1].items] == ["w1", "w2", "w3", "w4", "w5"]
    assert listings[P1].offsets == {"tok1": 2, "tok2": 4}


def test_coverage_reports_what_is_servable_per_snapshot(snap, tmp_path, capsys):
    cov = coverage_of(snap)
    status = {t.tool: t.status for t in cov.tools}
    assert status["list_work_items"] == "recorded+derived"
    assert status["list_projects"] == "derived" and status["get_plan"] == "derived"
    assert status["get_work_item_by_id"] == "derived"
    assert status["lane_occupancy"] == "absent" and status["find"] == "not-replayable"
    assert cov.projects == 2 and cov.work_items == 6
    assert coverage_main(["--root", str(snap.root), "--snapshot", SID]) == 0
    detail = capsys.readouterr().out
    assert "SERVED list_work_items {}" in detail and "| lane_occupancy | absent |" in detail
    assert coverage_main(["--root", str(snap.root), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["snapshot"] == SID
    assert coverage_main(["--root", str(snap.root)]) == 0
    assert f"{SID}  projects=2 items=6" in capsys.readouterr().out
    assert coverage_main(["--root", str(snap.root), "--snapshot", "20200101T000000Z"]) == 2
