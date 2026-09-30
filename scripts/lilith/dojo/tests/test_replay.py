import io
import json

import cos_recorder as rec
import pytest

from dojo.errors import LedgerError, ObjectIntegrityError
from dojo.replay_server import (
    ANIMA_READ_TOOLS,
    DojoGitServer,
    DojoLedger,
    ReceiptLog,
    ReplayServer,
    StaticAncestryOracle,
    serve,
)
from dojo.snapshot import SnapshotStore, canonical_json
from fixtures import MAIN_1, T0, TOOLS, WI


def rpc(server, method, params=None, rid=1):
    return server.handle({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})


def one_snapshot(world):
    sid = world.snapshot(T0, [WI("wi-1", "working", ci_green=True), WI("wi-2", "done")])
    return world.store().get(sid)


def test_canonical_json_and_allowlist_match_the_recorder():
    payload = {"b": [1, {"z": "ü"}], "a": None}
    assert canonical_json(payload) == rec.canonical_json(payload)
    assert ANIMA_READ_TOOLS == rec.ANIMA_READ_TOOLS


def test_tools_list_serves_recorded_contract_and_its_hash(world, tmp_path):
    snap = one_snapshot(world)
    server = ReplayServer(snap, DojoLedger(tmp_path / "ledger.jsonl"))
    result = rpc(server, "tools/list")["result"]
    assert result["tools"] == TOOLS
    assert result["_meta"]["lilith/contract_sha256"] == rec.sha256_hex(rec.canonical_json({"tools": TOOLS}))
    assert result["_meta"]["lilith/snapshot"] == snap.id


def test_recorded_read_answers_with_recorded_bytes_and_issues_receipt(world, tmp_path):
    snap = one_snapshot(world)
    server = ReplayServer(snap, DojoLedger(tmp_path / "ledger.jsonl"))
    result = rpc(server, "tools/call", {"name": "get_work_item_detail", "arguments": {"id": "wi-1"}})["result"]
    recorded, ref = snap.detail("wi-1")
    assert json.loads(result["content"][0]["text"]) == recorded
    assert result["isError"] is False
    assert result["_meta"]["lilith/digest"] == f"sha256:{ref.sha256}"
    (receipt,) = server.receipts.receipts
    assert receipt.response_sha256 == ref.sha256 and receipt.observed_at == snap.unix_time
    assert server.ledger.entries == []


def test_unrecorded_read_is_an_error_and_never_reaches_the_ledger(world, tmp_path):
    snap = one_snapshot(world)
    server = ReplayServer(snap, DojoLedger(tmp_path / "ledger.jsonl"))
    for name, args in (("get_work_item_detail", {"id": "nope"}), ("get_work_item_forensics", {"work_item_id": "wi-1"})):
        result = rpc(server, "tools/call", {"name": name, "arguments": args})["result"]
        assert result["isError"] is True
        assert result["_meta"]["lilith/source"] == "error"
    assert server.ledger.entries == []
    assert not (tmp_path / "ledger.jsonl").exists()


def test_write_tool_goes_to_the_hash_chained_dojo_ledger(world, tmp_path):
    snap = one_snapshot(world)
    path = tmp_path / "ledger.jsonl"
    server = ReplayServer(snap, DojoLedger(path))
    r1 = rpc(server, "tools/call", {"name": "approve_and_merge", "arguments": {"work_item_id": "wi-1"}})["result"]
    r2 = rpc(server, "tools/call", {"name": "post_message", "arguments": {"content": "x", "work_item_id": "wi-1"}}, 2)["result"]
    assert json.loads(r1["content"][0]["text"])["ledger_seq"] == 1
    assert json.loads(r2["content"][0]["text"])["ledger_seq"] == 2
    reloaded = DojoLedger(path)
    assert [e.tool for e in reloaded.entries] == ["approve_and_merge", "post_message"]
    assert reloaded.entries[1].prev_sha256 == reloaded.entries[0].entry_sha256
    lines = path.read_text().splitlines()
    tampered = json.loads(lines[0])
    tampered["arguments"]["work_item_id"] = "wi-2"
    path.write_text(json.dumps(tampered) + "\n" + lines[1] + "\n")
    with pytest.raises(LedgerError):
        DojoLedger(path)


def test_unknown_tool_is_refused_without_receipt_or_ledger(world, tmp_path):
    snap = one_snapshot(world)
    server = ReplayServer(snap, DojoLedger(tmp_path / "ledger.jsonl"))
    response = rpc(server, "tools/call", {"name": "drop_database", "arguments": {}})
    assert response["error"]["code"] == -32602
    assert server.receipts.receipts == [] and server.ledger.entries == []
    assert rpc(server, "resources/list")["error"]["code"] == -32601


def test_ledger_inside_snapshot_root_is_refused(world):
    one_snapshot(world)
    with pytest.raises(LedgerError):
        DojoLedger(world.root / "ledger.jsonl", forbidden_roots=(world.root,))


def test_replay_is_deterministic(world, tmp_path):
    snap = one_snapshot(world)
    script = [("tools/list", {}), ("tools/call", {"name": "list_work_items", "arguments": {"project_id": "proj-0001"}}),
              ("tools/call", {"name": "approve_and_merge", "arguments": {"work_item_id": "wi-1"}}),
              ("tools/call", {"name": "get_wi_loop_state", "arguments": {"work_item_id": "wi-1"}})]
    outputs = []
    for run in ("a", "b"):
        server = ReplayServer(SnapshotStore(world.root).get(snap.id), DojoLedger(tmp_path / f"{run}.jsonl"))
        outputs.append([rpc(server, m, p, i) for i, (m, p) in enumerate(script)])
    assert outputs[0] == outputs[1]
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()


def test_git_server_answers_refs_and_ancestry_from_the_snapshot(world):
    snap = one_snapshot(world)
    log = ReceiptLog()
    git = DojoGitServer(snap, StaticAncestryOracle({("c" * 40, MAIN_1): True}), log)
    refs = rpc(git, "tools/call", {"name": "git_ls_remote", "arguments": {}})["result"]
    assert json.loads(refs["content"][0]["text"])["refs"][1]["sha"] == MAIN_1
    yes = rpc(git, "tools/call", {"name": "git_is_ancestor", "arguments": {"commit": "c" * 40}}, 2)["result"]
    assert json.loads(yes["content"][0]["text"])["is_ancestor"] is True
    unknown = rpc(git, "tools/call", {"name": "git_is_ancestor", "arguments": {"commit": "d" * 40}}, 3)["result"]
    assert unknown["isError"] is True
    assert [r.source.value for r in log.receipts] == ["snapshot", "oracle", "error"]


def test_tampered_object_is_detected(world):
    snap = one_snapshot(world)
    _, ref = snap.detail("wi-1")
    path = world.root / "objects" / f"{ref.sha256}.json"
    path.write_bytes(path.read_bytes().replace(b"working", b"done!!!"))
    fresh = SnapshotStore(world.root).get(snap.id)
    with pytest.raises(ObjectIntegrityError):
        fresh.detail("wi-1")


def test_stdio_serve_round_trip(world, tmp_path):
    snap = one_snapshot(world)
    server = ReplayServer(snap, DojoLedger(tmp_path / "ledger.jsonl"))
    stdin = io.StringIO("\n".join([
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        "not json",
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    ]) + "\n")
    stdout = io.StringIO()
    serve(server, stdin, stdout)
    lines = [json.loads(l) for l in stdout.getvalue().splitlines()]
    assert [l.get("id") for l in lines] == [1, None, 2]
    assert lines[1]["error"]["code"] == -32700
