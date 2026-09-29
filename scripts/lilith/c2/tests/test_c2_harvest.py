import json
from datetime import timedelta

import pytest

from c2 import harvest_sessions as hs
from c2.errors import NfsUnmountedError
from dojo.families import BuildContext, Family, build_all, load_corrections
from fixtures import WI
from support import T0, WI_A, WI_B, WI_GHOST, call, row

CORE = "127.0.0.1:18444"


def turns_two_sessions():
    return [
        row(10, 0, "morning?", f"parked: {WI_A}", [call("mcp__anima__list_work_items", output=f"[{WI_A}]")]),
        row(11, 5, "✎ `mcp__anima__get_work_item_detail` you skipped detail", "retrying",
            [call("mcp__anima__get_work_item_detail", output=WI_A)]),
        row(12, 6, "", "auto", [call("kg_query")], dispatch=True),
        row(13, 300, "next thing", "done"),  # 294 min idle: a second session
    ]


def config(tmp_path, snapshot_root=None):
    return hs.HarvestConfig(out=tmp_path / "out", endpoint=CORE, token="x" * 30, snapshot_root=snapshot_root)


def fetch_of(rows):
    return lambda endpoint, token, workspace: iter(rows)


def test_sessions_split_on_idle_gap_and_autonomous_turns_stay_out(tmp_path):
    report = hs.harvest(config(tmp_path), fetch_of(turns_two_sessions()))
    assert report["sessions"] == 2 and report["autonomous_turns"] == 1
    files = sorted((tmp_path / "out" / "sessions").glob("*.json"))
    assert len(files) == 2
    first = json.loads(files[0].read_text())
    assert first["schema"] == "lilith/c2-session@1" and first["turn_count"] == 2
    assert first["provenance"]["models"] == ["gemma-4-12b"]
    assert first["provenance"]["source"]["method"] == "_akh/transcripts/list"
    call0 = first["turns"][0]["tool_calls"][0]
    assert call0["tool"] == "mcp__anima__list_work_items" and call0["output"] == f"[{WI_A}]" and call0["receipt_captured"]
    assert first["turns"][0]["timings"]["tool_ms"] == 120
    assert first["turns"][1]["timings"]["since_previous_turn_ms"] == 5 * 60 * 1000
    auto = (tmp_path / "out" / "autonomous.jsonl").read_text().splitlines()
    assert json.loads(auto[0])["seq"] == 12


def test_correction_records_point_at_the_corrected_turn_and_nearest_snapshot(tmp_path, world):
    before = world.snapshot(T0 - timedelta(hours=2), [WI(WI_A, "parked", parked_reason="rebase-conflict")])
    world.snapshot(T0 + timedelta(hours=1), [WI(WI_A, "parked", parked_reason="rebase-conflict")])
    hs.harvest(config(tmp_path, world.root), fetch_of(turns_two_sessions()))
    lines = (tmp_path / "out" / "corrections.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    session_id = hs.session_id(hs.normalise_turn(turns_two_sessions()[0]))
    assert record["episode_ref"] == f"{session_id}#0"
    assert record["snapshot_id"] == before  # the later snapshot postdates the corrected turn
    assert record["expected_tool"] == "mcp__anima__get_work_item_detail"
    assert record["subject_ids"] == [WI_A]
    assert record["corrected_signature"] == hs.tool_sequence_signature([{"tool": "mcp__anima__list_work_items"}])
    assert record["correction_turn_seq"] == 11 and record["corrected_turn_seq"] == 10


def test_a_correction_without_a_snapshot_is_kept_apart_not_dropped(tmp_path):
    hs.harvest(config(tmp_path, tmp_path / "no-snapshots"), fetch_of(turns_two_sessions()))
    assert not (tmp_path / "out" / "corrections.jsonl").exists()
    unmatched = (tmp_path / "out" / "corrections-unmatched.jsonl").read_text().splitlines()
    assert json.loads(unmatched[0])["snapshot_id"] is None


def test_a_correction_as_the_first_message_has_nothing_to_correct(tmp_path, world):
    world.snapshot(T0 - timedelta(hours=1), [WI(WI_A, "parked")])
    rows = [row(1, 0, "✎ wrong", "ok")]
    report = hs.harvest(config(tmp_path, world.root), fetch_of(rows))
    assert report["corrections"] == 0 and report["corrections_unmatched"] == 0


def test_harvest_is_idempotent_and_never_deletes(tmp_path, world):
    world.snapshot(T0 - timedelta(hours=2), [WI(WI_A, "parked")])
    cfg = config(tmp_path, world.root)
    hs.harvest(cfg, fetch_of(turns_two_sessions()))
    out = tmp_path / "out"
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.rglob("*") if p.is_file()}
    second = hs.harvest(cfg, fetch_of(turns_two_sessions()))
    assert set(second["session_files"].values()) == {"unchanged"}
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.rglob("*") if p.is_file()} == before
    # the transcript loses the second session (e.g. --since moved): nothing on disk goes away
    hs.harvest(cfg, fetch_of(turns_two_sessions()[:2]))
    assert {p for p in before} <= {p for p in out.rglob("*") if p.is_file()}


def test_a_session_file_never_shrinks(tmp_path):
    cfg = config(tmp_path)
    hs.harvest(cfg, fetch_of(turns_two_sessions()))
    shorter = turns_two_sessions()[:1]
    report = hs.harvest(cfg, fetch_of(shorter))
    assert list(report["session_files"].values()) == ["kept"]
    kept = json.loads(next((tmp_path / "out" / "sessions").glob("*s10.json")).read_text())
    assert kept["turn_count"] == 2


def test_a_growing_session_updates_its_file_in_place(tmp_path):
    cfg = config(tmp_path)
    hs.harvest(cfg, fetch_of(turns_two_sessions()[:1]))
    report = hs.harvest(cfg, fetch_of(turns_two_sessions()[:2]))
    assert list(report["session_files"].values()) == ["written"]
    assert len(list((tmp_path / "out" / "sessions").glob("*.json"))) == 1


def test_since_ignores_older_dev_chats(tmp_path):
    cfg = hs.HarvestConfig(tmp_path / "out", CORE, "x" * 30, since=T0 + timedelta(minutes=200))
    report = hs.harvest(cfg, fetch_of(turns_two_sessions()))
    assert report["turns"] == 1 and report["sessions"] == 1


def test_older_captures_without_receipts_are_marked_so(tmp_path):
    legacy = [row(1, 0, "q", "a", [{"tool": "kg_query", "ok": True, "wrote": False}])]
    hs.harvest(config(tmp_path), fetch_of(legacy))
    session = json.loads(next((tmp_path / "out" / "sessions").glob("*.json")).read_text())
    assert session["provenance"]["receipts_captured"] is False
    assert session["turns"][0]["tool_calls"][0]["receipt_captured"] is False


def test_unmounted_share_stops_before_any_fetch_or_write(tmp_path):
    fetched = []
    out = hs.Path("/Volumes/akh-medu-runs-not-mounted/lilith/c2")
    cfg = hs.HarvestConfig(out, CORE, "x" * 30)
    with pytest.raises(NfsUnmountedError) as raised:
        hs.harvest(cfg, lambda *a: fetched.append(a) or iter(()), ismount=lambda p: False)
    assert raised.value.exit_code == 3 and not fetched and not out.exists()


def test_main_exit_codes(tmp_path, capsys):
    token = tmp_path / "key"
    token.write_text("y" * 30)
    unmounted = ["--out", "/Volumes/akh-medu-runs-not-mounted/c2", "--token-file", str(token)]
    assert hs.main(unmounted) == 3
    assert "not mounted" in capsys.readouterr().err
    refused = ["--out", str(tmp_path / "o"), "--token-file", str(token), "--endpoint", "127.0.0.1:1"]
    assert hs.main(refused) == 4
    assert "cannot reach" in capsys.readouterr().err
    assert hs.main(["--out", str(tmp_path / "o"), "--since", "yesterday-ish"]) == 2


def test_ids_are_extracted_structurally_and_longest_form_wins():
    text = f"see {WI_A[:13]} and {WI_A}, also {WI_B[:13]} but not 12345678-12"
    assert hs.extract_ids(text) == sorted([WI_A, WI_B[:13]])


def test_harvested_corrections_become_dojo_episodes(tmp_path, world):
    world.snapshot(T0 - timedelta(hours=2), [WI(WI_A, "parked", parked_reason="rebase-conflict")])
    hs.harvest(config(tmp_path, world.root), fetch_of(turns_two_sessions()))
    records = load_corrections(tmp_path / "out" / "corrections.jsonl")
    assert len(records) == 1 and records[0].subject_ids == (WI_A,)
    built = build_all(world.store(), BuildContext(corrections=records), [Family.CORRECTION])
    specs = built[Family.CORRECTION].specs
    assert len(specs) == 1
    assert specs[0].label["expected_tool"] == "mcp__anima__get_work_item_detail"
    assert WI_GHOST not in specs[0].subject_ids
