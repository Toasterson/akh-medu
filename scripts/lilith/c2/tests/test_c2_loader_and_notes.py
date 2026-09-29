import json
from datetime import timedelta
from pathlib import Path

import pytest

from c2 import c2_notes as notes
from c2 import harvest_sessions as hs
from dojo import baseline as B
from dojo.dojo_report import build_report, render
from dojo.episode import Family
from dojo.errors import DojoError
from dojo.families import CorrectionRecord, load_corrections
from dojo.replay_server import StaticAncestryOracle
from fixtures import WI
from support import T0, WI_A, WI_B, WI_C, WI_GHOST, call, row


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_load_corrections_reads_the_five_fields_and_ignores_extras(tmp_path):
    path = tmp_path / "corrections.jsonl"
    write_jsonl(path, [{"episode_ref": "s#1", "snapshot_id": "20260930T060000Z", "corrected_signature": "abc",
                        "expected_tool": None, "subject_ids": ["b", "a"], "session_id": "s", "correction_turn_seq": 4}])
    (record,) = load_corrections(path)
    assert record == CorrectionRecord("s#1", "20260930T060000Z", "abc", None, ("a", "b"))


@pytest.mark.parametrize("line,needle", [
    ("not json", "not JSON"),
    (json.dumps({"episode_ref": "s#1", "corrected_signature": "abc"}), "snapshot_id"),
    (json.dumps({"episode_ref": "s#1", "snapshot_id": None, "corrected_signature": "abc"}), "snapshot_id"),
])
def test_load_corrections_refuses_records_that_cannot_be_episodes(tmp_path, line, needle):
    path = tmp_path / "c.jsonl"
    path.write_text(line + "\n")
    with pytest.raises(DojoError, match=needle) as raised:
        load_corrections(path)
    assert ":1:" in str(raised.value) and raised.value.help


def corrections_file(tmp_path, world, sid):
    path = tmp_path / "corrections.jsonl"
    write_jsonl(path, [{"episode_ref": "s#0", "snapshot_id": sid, "corrected_signature": "abc",
                        "expected_tool": "get_work_item_detail", "subject_ids": [WI_A]}])
    return path


def test_dojo_report_counts_corrections(tmp_path, world):
    sid = world.snapshot(T0, [WI(WI_A, "parked", parked_reason="retry-budget")])
    without = build_report(world.root, None, None)
    assert without["families"]["correction"]["episodes"] == 0 and "none are recorded" in without["families"]["correction"]["note"]
    report = build_report(world.root, None, None, corrections_file(tmp_path, world, sid))
    assert report["families"]["correction"]["episodes"] == 1
    assert "1 recorded corrections" in render(report)


def test_dojo_report_flags_corrections_naming_unknown_snapshots(tmp_path, world):
    world.snapshot(T0, [WI(WI_A, "parked")])
    report = build_report(world.root, None, None, corrections_file(tmp_path, world, "20200101T000000Z"))
    assert report["families"]["correction"]["episodes"] == 0
    assert "1 name a snapshot this root does not hold" in report["families"]["correction"]["note"]


def test_runner_selects_correction_episodes_when_given_the_file(tmp_path, world):
    sid = world.snapshot(T0, [WI(WI_A, "parked", parked_reason="retry-budget")])
    records = load_corrections(corrections_file(tmp_path, world, sid))
    chosen, selection = B.select_episodes(world.store(), StaticAncestryOracle({}), [Family.CORRECTION], "all", None, 0,
                                          corrections=records)
    assert selection["selected"] == {"correction": 1}
    assert chosen[0].spec.label["expected_tool"] == "get_work_item_detail"
    none, _ = B.select_episodes(world.store(), StaticAncestryOracle({}), [Family.CORRECTION], "all", None, 0)
    assert none == []


def test_runner_cli_dry_run_accepts_the_corrections_flag(tmp_path, world, capsys):
    sid = world.snapshot(T0, [WI(WI_A, "parked", parked_reason="retry-budget")])
    path = corrections_file(tmp_path, world, sid)
    code = B.main(["run", "--root", str(world.root), "--family", "correction", "--split", "all", "--corrections", str(path),
                   "--model-kind", "openai", "--model", "lilith", "--key-env", "NOT_SET", "--allow-dirty", "--dry-run"])
    assert code == 0
    assert '"correction": 1' in capsys.readouterr().out


# --- baseline notes ---------------------------------------------------------------------------------------------------------------------------------


def session_of(rows):
    return hs.build_session([hs.normalise_turn(r) for r in rows], "lilith", "127.0.0.1:18444")


def test_fabrication_is_an_id_no_receipt_or_operator_message_backs():
    session = session_of([
        row(1, 0, "what is parked?", f"{WI_A} is parked; {WI_GHOST} is next",
            [call("mcp__anima__list_work_items", output=f"[{WI_A}]")]),
        row(2, 2, f"and {WI_B[:13]}?", f"{WI_B} is fine"),  # the operator named it first
    ])
    first, second = notes.check_fabrications(session)
    assert first["claims"] == 2 and first["backed"] == 1 and first["fabricated"] == [WI_GHOST]
    assert not first["unverifiable"]
    assert second["fabricated"] == []


def test_a_cut_or_missing_receipt_makes_a_miss_unverifiable_not_fabricated():
    cut = call("mcp__anima__list_work_items", output=f"[{WI_A}, ...")
    session = session_of([row(1, 0, "q", f"{WI_GHOST}", [cut])])
    (result,) = notes.check_fabrications(session)
    assert result["fabricated"] == [WI_GHOST] and result["unverifiable"]
    legacy = session_of([row(1, 0, "q", f"{WI_GHOST}", [{"tool": "kg_query", "ok": True}])])
    assert notes.check_fabrications(legacy)[0]["unverifiable"]


def test_missed_items_against_the_snapshot_briefing_buckets(tmp_path, world):
    world.snapshot(T0 - timedelta(hours=1), [
        WI(WI_A, "parked", parked_reason="rebase-conflict"), WI(WI_B, "working"), WI(WI_C, "ready"),
    ])
    session = session_of([
        row(1, 0, "morning", f"In flight: {WI_B}. Parked: {WI_A}."),  # names 2 of 3, omits the ready item
        row(2, 3, "thanks", "welcome"),
    ])
    index = hs.SnapshotIndex(world.root)
    (briefing,) = notes.check_missed(session, world.store(), index, min_ids=2, forced=frozenset())
    assert briefing["turn"] == 0 and briefing["missed"] == [WI_C] and briefing["required"] == 3
    assert briefing["recall"] == pytest.approx(0.667, abs=1e-3)
    # below the id threshold it is not a briefing, unless the supervisor names it
    assert notes.check_missed(session, world.store(), index, min_ids=3, forced=frozenset()) == []
    forced = frozenset({f"{session['session_id']}#0"})
    assert len(notes.check_missed(session, world.store(), index, min_ids=3, forced=forced)) == 1
    assert notes.check_missed(session, None, index, 2, frozenset()) == []


def test_short_ids_count_as_mentions(world):
    world.snapshot(T0 - timedelta(hours=1), [WI(WI_A, "working"), WI(WI_B, "working")])
    session = session_of([row(1, 0, "q", f"{WI_A[:13]} and {WI_B[:13]}")])
    (result,) = notes.check_missed(session, world.store(), hs.SnapshotIndex(world.root), 2, frozenset())
    assert result["missed"] == []


def test_autonomy_noise_lists_calls_beyond_reading_and_own_state():
    session = session_of([row(1, 0, "look", "done", [
        call("mcp__anima__get_work_item_detail"),
        call("mcp__anima__approve_and_merge", wrote=True),
        call("some_unlisted_tool"),
    ])])
    result = notes.check_autonomy(session)
    tools = {c["tool"]: c["tier"] for c in result["beyond_own_state_calls"]}
    assert tools["mcp__anima__approve_and_merge"] == "t3-operator-class" and "some_unlisted_tool" in tools
    assert "mcp__anima__get_work_item_detail" not in tools
    assert result["operator_class_actions"] == 1 and result["tool_calls"] == 3


def test_notes_end_to_end_write_md_and_json_and_judge_the_exit_condition(tmp_path, world):
    world.snapshot(T0 - timedelta(hours=1), [WI(WI_A, "working"), WI(WI_B, "ready")])
    turns = []
    for n in range(6):  # six sessions of two turns, an hour apart
        base = n * 120
        turns += [row(10 * n + 1, base, "q", f"{WI_A} {WI_B}", [call("kg_query", output=f"{WI_A} {WI_B}")]),
                  row(10 * n + 2, base + 2, "✎ ok", "again")]
    out = tmp_path / "out"
    hs.harvest(hs.HarvestConfig(out, "127.0.0.1:18444", "x" * 30, snapshot_root=world.root), lambda *a: iter(turns))
    code = notes.main(["--out", str(out), "--snapshot-root", str(world.root), "--tok-per-s", "31.5", "--spend-usd", "4.2",
                       "--briefing-min-ids", "2"])
    assert code == 0
    data = json.loads((out / "c2-notes.json").read_text())
    assert data["exit_condition"] == {"sessions_counted": 6, "required": 5, "min_turns": 2, "met": True}
    assert data["corrections"]["matched"] == 6
    assert data["throughput"] == {"tok_per_s": 31.5, "spend_usd": 4.2}
    md = (out / "c2-notes.md").read_text()
    assert "**met**" in md and "tok/s: 31.5" in md and "spend: $4.20" in md
    before = (out / "c2-notes.md").stat().st_mtime_ns
    notes.main(["--out", str(out), "--snapshot-root", str(world.root), "--tok-per-s", "31.5", "--spend-usd", "4.2",
                "--briefing-min-ids", "2"])
    assert (out / "c2-notes.md").stat().st_mtime_ns == before  # idempotent


def test_notes_say_not_met_with_too_few_sessions(tmp_path):
    out = tmp_path / "out"
    hs.harvest(hs.HarvestConfig(out, "127.0.0.1:18444", "x" * 30), lambda *a: iter([row(1, 0, "q", "a"), row(2, 1, "q", "a")]))
    assert notes.main(["--out", str(out), "--snapshot-root", str(tmp_path / "none")]) == 0
    data = json.loads((out / "c2-notes.json").read_text())
    assert not data["exit_condition"]["met"] and not data["missed_items_computed"]
    assert "NOT met" in (out / "c2-notes.md").read_text()


def test_notes_exit_code_3_when_the_share_is_not_mounted(capsys):
    assert notes.main(["--out", "/Volumes/akh-medu-runs-not-mounted/c2"]) == 3
