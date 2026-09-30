"""Schema-driven replay equivalence (`dojo.replay_match`) and its audit trail on receipts."""
import json
from pathlib import Path

import cos_recorder as rec
import pytest

from dojo.replay_audit import CLASS_EQUIVALENT, CLASS_OTHER, CLASS_UNRECORDED, classify
from dojo.replay_match import normalize_arguments, page_size_satisfied, resolve_recorded_read, resolve_ref
from dojo.replay_server import DojoGitServer, DojoLedger, ReceiptLog, ReplayServer, StaticAncestryOracle
from fixtures import MAIN_1, T0

PROJECT = "proj-0001"
FORENSICS = {"type": "object", "properties": {
    "work_item_id": {"type": "string"},
    "max_events": {"type": "integer", "default": 200, "minimum": 1, "maximum": 500}}}
TOOLS = [
    {"name": "list_work_items", "inputSchema": {"type": "object", "properties": {
        "project_id": {"type": "string"}, "current_phase": {"type": "string"},
        "page_size": {"type": "integer"}, "page_token": {"type": "string"}}}},
    {"name": "list_plans", "inputSchema": {"type": "object", "properties": {
        "project_id": {"type": "string"}, "page_token": {"type": "string"}, "limit": {"type": "integer"}}}},
    {"name": "get_wi_loop_state", "inputSchema": FORENSICS},
    {"name": "get_thread", "inputSchema": {"type": "object", "properties": {
        "work_item_id": {"type": "string"}, "page_size": {"type": "integer"}}}},
    {"name": "get_work_item_detail", "inputSchema": {"type": "object", "properties": {
        "id": {"type": "string"}, "message_page_size": {"type": "integer"}, "change_page_size": {"type": "integer"},
        "archived": {"type": "boolean"}}}},
    {"name": "get_work_item_forensics", "inputSchema": FORENSICS},
    {"name": "get_plan", "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}}},
]
PAGE_1 = {"work_items": [{"id": f"w{i}"} for i in range(3)], "next_page_token": "tok", "total": 5}
PAGE_2 = {"work_items": [{"id": "w3"}, {"id": "w4"}], "next_page_token": "", "total": 5}
PLANS = {"plans": [{"id": "p1"}, {"id": "p2"}], "next_page_token": None, "total": 2}
DETAIL = {"work_item": {"id": "w0"}, "thread_messages": [], "change_history": []}


def build(root: Path):
    responses = []

    def put(call, args, payload):
        digest, _ = rec.store_object(root, payload)
        responses.append({"call": call, "args": args, "sha256": digest})

    put("list_work_items", {"project_id": PROJECT}, PAGE_1)
    put("list_work_items", {"project_id": PROJECT, "page_token": "tok"}, PAGE_2)
    put("list_plans", {"project_id": PROJECT}, PLANS)
    put("get_wi_loop_state", {"work_item_id": "w0"}, {"work_item_id": "w0"})
    put("get_thread", {"work_item_id": "w0"}, "plain text thread")
    put("get_work_item_detail", {"id": "w0"}, DETAIL)
    contract_sha, _ = rec.store_object(root, {"tools": TOOLS})
    refs_sha, _ = rec.store_object(root, {"remote": rec.GIT_REMOTE, "refs": [
        {"ref": "HEAD", "sha": MAIN_1}, {"ref": "refs/heads/main", "sha": MAIN_1},
        {"ref": "refs/heads/feature/x", "sha": "3" * 40}, {"ref": "refs/tags/v1", "sha": "4" * 40}]})
    sid = T0.strftime("%Y%m%dT%H%M%SZ")
    rec.atomic_write_json(root / sid / "manifest.json", {
        "schema": rec.SCHEMA, "timestamp": T0.isoformat(),
        "collectors": {
            "anima": {"status": "ok", "responses": responses, "errors": [], "tool_contract_sha256": contract_sha},
            "git": {"status": "ok", "errors": [], "responses": [
                {"call": "git_ls_remote", "args": {"remote": rec.GIT_REMOTE}, "sha256": refs_sha}]}}})
    from dojo.snapshot import SnapshotStore
    return SnapshotStore(root).get(sid)


@pytest.fixture
def snap(tmp_path):
    return build(tmp_path / "snapshots")


def schema(snap, name):
    return next(t["inputSchema"] for t in snap.tool_contract()[0] if t["name"] == name)


def resolve(snap, name, args):
    return resolve_recorded_read(snap, name, schema(snap, name), args)


def test_exact_call_has_no_rules(snap):
    found = resolve(snap, "list_work_items", {"project_id": PROJECT})
    assert found.payload == PAGE_1 and found.match.exact
    assert found.match.matched == {"project_id": PROJECT}


def test_page_size_equal_to_the_recorded_full_page_is_the_same_call(snap):
    found = resolve(snap, "list_work_items", {"project_id": PROJECT, "page_size": 3})
    assert found.payload == PAGE_1
    assert found.match.rules == ("page-size-satisfied:page_size=3",)
    assert found.match.requested == {"project_id": PROJECT, "page_size": 3}
    assert found.match.matched == {"project_id": PROJECT}
    assert found.ref.args == json.dumps({"project_id": PROJECT}, sort_keys=True, separators=(",", ":"))


@pytest.mark.parametrize("size", [1, 2, 4, 100])
def test_page_size_that_the_recorded_page_does_not_match_is_derived_from_the_recorded_items(snap, size):
    """Not the recorded page, so never passed off as it: re-paged over the recorded walk, rule named."""
    found = resolve(snap, "list_work_items", {"project_id": PROJECT, "page_size": size})
    expected = [{"id": f"w{i}"} for i in range(5)][:size]
    assert found.payload["work_items"] == expected and found.payload["total"] == 5
    assert ("derive:list_work_items" in found.match.rules) and not found.match.exact
    assert (found.payload["next_page_token"] is None) == (size >= 5)


@pytest.mark.parametrize("size", [2, 3, 50])
def test_page_size_over_a_complete_listing_is_the_same_call(snap, size):
    found = resolve(snap, "list_plans", {"project_id": PROJECT, "limit": size})
    assert found.payload == PLANS and found.match.rules == (f"page-size-satisfied:limit={size}",)


def test_numeric_string_page_size_is_coerced_then_checked(snap):
    found = resolve(snap, "list_plans", {"project_id": PROJECT, "limit": "50"})
    assert found.match.rules == ("coerce:limit:integer", "page-size-satisfied:limit=50")
    assert resolve(snap, "list_plans", {"project_id": PROJECT, "limit": 1}) is None


def test_recorded_page_with_token_still_matches_exactly(snap):
    found = resolve(snap, "list_work_items", {"project_id": PROJECT, "page_token": "tok"})
    assert found.payload == PAGE_2 and found.match.exact
    again = resolve(snap, "list_work_items", {"project_id": PROJECT, "page_token": "tok", "page_size": 2})
    assert again.payload == PAGE_2 and again.match.rules == ("page-size-satisfied:page_size=2",)


def test_empty_page_token_is_the_first_page(snap):
    found = resolve(snap, "list_work_items", {"project_id": PROJECT, "page_token": ""})
    assert found.payload == PAGE_1 and found.match.rules == ("empty-token:page_token",)


def test_unrecorded_scopes_and_identifiers_are_never_invented(snap):
    assert resolve(snap, "list_work_items", {"project_id": "elsewhere"}) is None
    assert resolve(snap, "list_work_items", {"project_id": PROJECT, "bogus": 1}) is None
    assert resolve(snap, "get_work_item_detail", {"id": "unrecorded"}) is None
    assert resolve(snap, "get_plan", {"id": "unrecorded"}) is None
    assert resolve(snap, "get_work_item_forensics", {"work_item_id": "w0"}) is None  # tool never recorded


def test_unlistable_or_multi_list_responses_have_no_completeness_proof(snap):
    assert resolve(snap, "get_thread", {"work_item_id": "w0", "page_size": 50}) is None
    assert resolve(snap, "get_work_item_detail", {"id": "w0", "message_page_size": 50}) is None


def test_types_are_coerced_only_where_the_schema_says_so(snap):
    args, rules = normalize_arguments(schema(snap, "get_work_item_detail"),
                                      {"id": 7, "archived": "true", "message_page_size": "50.0", "unknown": "1"})
    assert args == {"id": "7", "archived": True, "message_page_size": "50.0", "unknown": "1"}
    assert rules == ["coerce:archived:boolean", "coerce:id:string"]
    # argument key order is irrelevant: identity is canonical JSON
    assert resolve(snap, "list_work_items", {"page_token": "tok", "project_id": PROJECT}).payload == PAGE_2


def test_schema_default_is_the_omitted_argument(snap):
    args, rules = normalize_arguments(FORENSICS, {"work_item_id": "w0", "max_events": 200})
    assert args == {"work_item_id": "w0"} and rules == ["default:max_events"]
    args, rules = normalize_arguments(FORENSICS, {"work_item_id": "w0", "max_events": "200"})
    assert args == {"work_item_id": "w0"} and rules == ["coerce:max_events:integer", "default:max_events"]
    args, _ = normalize_arguments(FORENSICS, {"work_item_id": "w0", "max_events": 5})
    assert args["max_events"] == 5
    found = resolve(snap, "get_wi_loop_state", {"work_item_id": "w0", "max_events": 200})
    assert found.match.rules == ("default:max_events",)


def test_page_size_satisfied_needs_one_list_and_a_next_page_key():
    assert page_size_satisfied({"items": [1, 2], "nextPageToken": ""}, 5)
    assert page_size_satisfied({"items": [1, 2], "next_cursor": "c"}, 2)
    assert not page_size_satisfied({"items": [1, 2], "next_cursor": "c"}, 3)
    assert not page_size_satisfied({"items": [1, 2]}, 5)
    assert not page_size_satisfied({"a": [1], "b": [2], "next_page_token": ""}, 5)
    assert not page_size_satisfied({"items": [1], "next_page_token": ""}, 0)
    assert not page_size_satisfied([1, 2], 5)


def test_receipt_records_requested_and_matched_arguments(snap, tmp_path):
    server = ReplayServer(snap, DojoLedger(tmp_path / "l.jsonl"))
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": "list_work_items", "arguments": {"project_id": PROJECT, "page_size": 3}}}
    result = server.handle(request)["result"]
    assert result["isError"] is False
    (receipt,) = server.receipts.receipts
    assert receipt.arguments == {"project_id": PROJECT, "page_size": 3}
    assert receipt.matched_arguments == {"project_id": PROJECT}
    assert receipt.match_rules == ("page-size-satisfied:page_size=3",)
    assert receipt.to_dict()["matched_arguments"] == {"project_id": PROJECT}
    assert receipt.response_sha256 == snap.find("list_work_items", {"project_id": PROJECT}).sha256
    exact = server.handle({**request, "id": 2, "params": {"name": "list_work_items",
                                                          "arguments": {"project_id": PROJECT}}})["result"]
    assert exact["isError"] is False
    assert "matched_arguments" not in server.receipts.receipts[1].to_dict()


def test_error_receipts_carry_a_machine_readable_reason(snap, tmp_path):
    server = ReplayServer(snap, DojoLedger(tmp_path / "l.jsonl"))
    for name, args, reason in (("get_work_item_detail", {"id": "nope"}, "work-item-not-recorded"),
                               ("get_work_item_forensics", {"work_item_id": "w0"}, "tool-not-recorded")):
        result = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                "params": {"name": name, "arguments": args}})["result"]
        assert result["isError"] is True
        assert json.loads(result["content"][0]["text"])["reason"] == reason
    assert [r.error_reason for r in server.receipts.receipts] == ["work-item-not-recorded", "tool-not-recorded"]


def test_resolve_ref_follows_git_spellings_against_recorded_refs():
    refs = {"HEAD": "a", "refs/heads/main": "a", "refs/heads/feature/x": "b", "refs/tags/v1": "c"}
    assert resolve_ref(refs, None) == ("refs/heads/main", ())
    assert resolve_ref(refs, "refs/heads/main") == ("refs/heads/main", ())
    assert resolve_ref(refs, "origin/main") == ("refs/heads/main", ("ref-alias:origin/main->refs/heads/main",))
    assert resolve_ref(refs, "refs/remotes/origin/main")[0] == "refs/heads/main"
    assert resolve_ref(refs, "main")[0] == "refs/heads/main"
    assert resolve_ref(refs, "heads/feature/x")[0] == "refs/heads/feature/x"
    assert resolve_ref(refs, "origin/feature/x")[0] == "refs/heads/feature/x"
    assert resolve_ref(refs, "v1")[0] == "refs/tags/v1"
    assert resolve_ref(refs, "origin/HEAD")[0] == "HEAD"
    assert resolve_ref(refs, "upstream/main") == (None, ())
    assert resolve_ref(refs, "origin/nope") == (None, ())
    assert resolve_ref({"refs/heads/main": "a", "refs/heads/origin/main": "b"}, "origin/main")[0] == "refs/heads/origin/main"


def test_git_ancestry_accepts_origin_main_and_keeps_both_spellings(snap):
    log = ReceiptLog()
    git = DojoGitServer(snap, StaticAncestryOracle({("c" * 40, MAIN_1): True}), log)
    payload = git.call("git_is_ancestor", {"commit": "c" * 40, "ref": "origin/main"})[1]
    assert payload == {"commit": "c" * 40, "ref": "refs/heads/main", "ref_sha": MAIN_1, "is_ancestor": True,
                       "requested_ref": "origin/main"}
    (receipt,) = log.receipts
    assert receipt.source.value == "oracle"
    assert receipt.arguments["ref"] == "origin/main"
    assert receipt.matched_arguments == {"commit": "c" * 40, "ref": "refs/heads/main"}
    assert receipt.match_rules == ("ref-alias:origin/main->refs/heads/main",)


def test_git_ancestry_errors_name_their_reason(snap):
    log = ReceiptLog()
    git = DojoGitServer(snap, StaticAncestryOracle({}), log)
    assert git.call("git_is_ancestor", {"commit": "c" * 40, "ref": "elsewhere/main"})[1]["reason"] == "ref-not-recorded"
    assert git.call("git_is_ancestor", {"commit": "c" * 40})[1]["reason"] == "commit-unknown-to-oracle"
    assert git.call("git_is_ancestor", {"ref": "origin/main"})[1]["reason"] == "commit-missing"
    assert [r.error_reason for r in log.receipts] == ["ref-not-recorded", "commit-unknown-to-oracle", "commit-missing"]


def test_audit_classifies_misses(snap):
    def klass(server, tool, args):
        return classify(snap, server, tool, args).klass

    assert klass("anima", "list_work_items", {"project_id": PROJECT, "page_size": 3}) == CLASS_EQUIVALENT
    assert klass("anima", "get_plan", {"id": "p1"}) == CLASS_EQUIVALENT  # derived from the recorded list_plans
    assert klass("anima", "get_work_item_forensics", {"work_item_id": "w0"}) == CLASS_UNRECORDED
    assert klass("anima", "get_work_item_detail", {"id": "nope"}) == CLASS_UNRECORDED
    assert klass("anima", "list_work_items", {"current_phase": "working"}) == CLASS_EQUIVALENT
    assert klass("anima", "list_work_items", {"bogus": 1}) == CLASS_OTHER
    assert klass("git", "git_is_ancestor", {"commit": "c" * 40, "ref": "origin/main"}) == CLASS_EQUIVALENT
    assert klass("git", "git_is_ancestor", {"commit": "c" * 40, "ref": "elsewhere/main"}) == CLASS_UNRECORDED
