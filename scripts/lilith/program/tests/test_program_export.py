"""Source readers and the export end to end, on synthetic private inputs (the repo's own public kata and
sparring corpus is real data; everything private here is fabricated)."""
import hashlib
import json
from pathlib import Path

import pytest

from program import coverage as COV
from program import schema as S
from program import sources as SRC
from program.export import Config, export_corpus, parse_args

WI_TRAIN = "01a00000-0000-7000-8000-00000000000a"
WI_DEV = "01a00000-0000-7000-8000-00000000000b"
WI_COS_HELD = "01a00000-0000-7000-8000-00000000000c"
HELD_RESULT = "held-out tool result text"


def jl(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def lesson(rid, session, wi, receipt_id, family="status"):
    return {"record_id": rid, "session": session, "work_item_id": wi, "family": family, "source": "historical_tracker_receipt",
            "target_source": "deterministic_receipt_projection", "memory_seed": None, "weight": 1.0,
            "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": f"report {wi}"}],
            "answer": json.dumps({"work_item_id": wi}), "expected": {"receipt_id": receipt_id}}


def cos_row(rid, session, split, result, wi=None):
    args = json.dumps({"command": "x " + (wi or "")})
    steps = [{"kind": "tool_call", "name": "Bash", "args": args, "result_text": None, "text": None, "train": True, "truncated": False},
             {"kind": "tool_result", "name": "Bash", "args": None, "result_text": result, "text": None, "train": False, "truncated": False},
             {"kind": "assistant_text", "name": None, "args": None, "result_text": None, "text": "final " + rid, "train": True, "truncated": False}]
    return {"record_id": rid, "schema": "akh/lilith-transcript-trajectory@2", "split": split, "input": "operator asks " + rid,
            "labels": {"cos_class": "briefing", "operator_correction": False},
            "provenance": {"session_sha256": session}, "trajectory": {"steps": steps}, "turn_final_reply": {"text": "final " + rid}}


@pytest.fixture
def world(tmp_path):
    hist = tmp_path / "hist"
    hist.mkdir()
    receipts = [{"receipt_id": f"r{i}", "facts": {"id": w, "n": i}} for i, w in enumerate([WI_TRAIN, WI_DEV, WI_TRAIN])]
    jl(hist / "receipts.jsonl", receipts)
    train = [lesson("L1", "s-train", WI_TRAIN, "r0"), lesson("L2", "s-train2", WI_DEV, "r2", "memory_revision")]
    jl(hist / "train.jsonl", train)
    jl(hist / "train.tokens.jsonl", [{"record_id": r["record_id"], "input_ids": [5, 6, 7, 8], "target_mask": [False, False, True, True],
                                      "weight": 1.0, "source": r["family"]} for r in train])
    jl(hist / "dev.jsonl", [lesson("D1", "s-dev", WI_DEV, "r1")])
    jl(hist / "historical_eval.jsonl", [])
    (hist / "tokenization.json").write_text(json.dumps({"files": {"train.tokens.jsonl": "x"}, "tokenizer_pins": {}}))
    cur = tmp_path / "cur"
    cur.mkdir()
    jl(cur / "cos-style-train.jsonl", [cos_row("T1", "sess-a", "train", "ok result"), cos_row("T2", "sess-b", "train", HELD_RESULT)])
    jl(cur / "cos-style-validation.jsonl", [cos_row("V1", "sess-c", "validation", "val result")])
    (cur / "recipe.json").write_text(json.dumps({"style_sessions_excluded_for_historical_holdout_overlap": ["sess-held"]}))
    traj = tmp_path / "traj.jsonl"
    jl(traj, [cos_row("H1", "sess-held", "train", HELD_RESULT, WI_COS_HELD)])
    spots = tmp_path / "spots.json"
    spots.write_text(json.dumps([{"record_id": "spot", "prompt": "resolve project Delta"}]))
    gen = tmp_path / "gen"
    gen.mkdir()
    good = SRC.make_record(source="variation", source_ref="gen", stage="yellow", families=["silence"], language="de",
                           messages=[S.user("variation q"), S.assistant("variation a")])
    (gen / "yellow.jsonl").write_text(json.dumps(good) + "\n" + json.dumps({**good, "weight": 7}) + "\nnot json\n")
    return Config(out=tmp_path / "out", history=hist, curriculum=cur, behavior_cases=spots, cos_trajectories=traj,
                  c2_sessions=tmp_path / "c2", generated_dir=gen, snapshot_root=None, skip_dojo=True,
                  sentinel_exclusions=tmp_path / "no-sentinels.jsonl")


def digest_tree(root: Path):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def test_export_is_deterministic_and_idempotent(world, codec, tmp_path):
    first = export_corpus(world, codec)
    one = digest_tree(world.out)
    again = export_corpus(world, codec)
    assert digest_tree(world.out) == one
    other = Config(**{**world.__dict__, "out": tmp_path / "out2"})
    export_corpus(other, codec)
    assert digest_tree(other.out) == one
    assert first == again


def test_manifest_contract_and_archives(world, codec):
    m = export_corpus(world, codec)
    assert m["schema"] == "lilith/program-manifest@1" and m["run_id"] == "lilith-program-r1-20261003"
    assert m["parent"]["revision"] == "fake" and "tokenizer_pins" in m["parent"] and m["parent_revision"] == "fake"
    plan = {s["name"]: s for s in m["stages"]}
    assert [s["name"] for s in m["stages"]] == list(S.STAGES)
    assert (plan["white"]["sft_passes"], plan["white"]["preference_passes"], plan["white"]["lr_hint"]) == (2, 0, 1e-4)
    assert (plan["yellow"]["sft_passes"], plan["green"]["sft_passes"], plan["green"]["preference_passes"]) == (2, 1, 1)
    assert plan["green"]["onpolicy"] == {"families": ["memory-epistemics", "read-failure", "scope", "correction", "park-triage"],
                                         "quota": 120, "snapshot_until": "20261001T235959Z"}
    assert plan["brown"]["onpolicy"]["quota"] == 150 and plan["black"]["onpolicy"]["quota"] == 60
    assert plan["white"]["onpolicy"] is None and plan["yellow"]["onpolicy"] is None
    for s in m["stages"]:
        assert set(s["counts"]) == {"sft", "preference", "train", "dev"}
        archive = world.out / s["archive"]
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == s["archive_sha256"] == m["files_sha256"][s["archive"]]
        rows = [json.loads(line) for line in archive.read_text().splitlines()]
        assert len(rows) == s["counts"]["sft"] + s["counts"]["preference"] == s["counts"]["train"] + s["counts"]["dev"]
        for r in rows:
            assert S.validate_tokens(r) == []
            assert r["split"] in ("train", "dev") and r["weight"] > 0 and r["stage"] == s["name"]
    assert (world.out / "coverage.json").is_file() and (world.out / "exclusions.jsonl").is_file() and (world.out / "unfit.jsonl").is_file()
    assert m["exclusions"]["exposure_index"]["response_hashes"] > 0


def test_stage_sources_follow_the_frozen_plan(world, codec):
    m = export_corpus(world, codec)
    by_stage = m["counts"]["by_stage"]
    assert set(by_stage["white"]["by_source"]) >= {"kata_drill", "historical_receipt"}
    assert set(by_stage["yellow"]["by_source"]) >= {"kata_drill", "cos_transcript", "variation"}
    assert set(by_stage["green"]["by_source"]) >= {"sparring"}
    assert set(by_stage["black"]["by_source"]) >= {"lineage"}
    belts = {r["belt"] for r in SRC.read_jsonl(SRC.KATA_PATH)}
    assert belts == {"white", "yellow"}
    assert m["counts"]["by_source"]["kata_drill"] == 240


def test_held_out_exposure_is_excluded_with_counts(world, codec):
    m = export_corpus(world, codec)
    ex = [json.loads(line) for line in (world.out / "exclusions.jsonl").read_text().splitlines()]
    reasons = {(e["reason"], e["source"]) for e in ex}
    assert ("exposure_tool_response", "cos_transcript") in reasons  # T2 carries the held-out tool result text
    assert ("exposure_work_item", "historical_receipt") in reasons  # L2 is about the dev lesson's work item
    kept_ids = {json.loads(line)["source_ref"]["original_id"] for line in (world.out / "records" / "yellow.jsonl").read_text().splitlines()
                if json.loads(line)["source"] == "cos_transcript" and json.loads(line)["split"] != "excluded"}
    assert "T2" not in kept_ids and "T1" in kept_ids
    assert m["exclusions"]["by_reason"]["exposure_tool_response"] >= 1


def test_cos_validation_stays_dev_and_generated_records_are_validated(world, codec):
    m = export_corpus(world, codec)
    rows = [json.loads(line) for line in (world.out / "records" / "yellow.jsonl").read_text().splitlines()]
    val = [r for r in rows if r["source"] == "cos_transcript" and r["source_ref"]["original_id"] == "V1"]
    assert val and val[0]["split"] == "dev"
    assert m["reports"]["generated"] == {"accepted": 1, "rejected": 2}
    assert any(r["source"] == "variation" for r in rows)


def test_historical_tokens_are_reused_verbatim(world, codec):
    export_corpus(world, codec)
    rows = [json.loads(line) for line in (world.out / "stage-1-white.tokens.jsonl").read_text().splitlines()]
    hist = [r for r in rows if r["source"] == "historical_receipt"]
    assert hist and all(r["input_ids"] == [5, 6, 7, 8] and r["target_mask"] == [0, 0, 1, 1] for r in hist)


def test_rehearsal_is_capped_per_stage(world, codec):
    export_corpus(world, codec)
    for i, st in enumerate(S.STAGES, 1):
        rows = [json.loads(line) for line in (world.out / f"stage-{i}-{st}.tokens.jsonl").read_text().splitlines()]
        if rows:
            share = sum(r["source"] == "rehearsal" for r in rows) / len(rows)
            assert share <= 0.10 + 1e-9


def test_unfit_records_are_listed_not_archived(world, codec, monkeypatch):
    import program.export as X

    real = X.tokenize_records

    def small(records, codec, limit=S.CONTEXT_LIMIT):
        return real(records, codec, 120)

    monkeypatch.setattr(X, "tokenize_records", small)
    m = export_corpus(world, codec)
    unfit = [json.loads(line) for line in (world.out / "unfit.jsonl").read_text().splitlines()]
    assert m["unfit"]["count"] == len(unfit) > 0
    archived = {json.loads(line)["record_id"] for i, st in enumerate(S.STAGES, 1)
                for line in (world.out / f"stage-{i}-{st}.tokens.jsonl").read_text().splitlines()}
    assert not archived & {u["record_id"] for u in unfit}


# -- source readers ----------------------------------------------------------------------------------------------


def test_kata_system_prompt_uses_the_mapped_voice_and_fields_only():
    rows = SRC.kata_records()
    assert len(rows) == 240
    first = rows[0]
    assert first["voice"] in S.VOICES
    assert f"voice: {first['voice']}." in first["messages"][0]["content"]
    source = SRC.read_jsonl(SRC.KATA_PATH)[0]
    assert first["messages"][1]["content"] == source["situation"] and first["messages"][2]["content"] == source["move_chosen"]
    assert {r["voice"] for r in rows} == set(S.VOICES)


def test_sparring_windows_and_lineage_are_split_by_role():
    windows = SRC.sparring_records()
    lineage = SRC.lineage_records()
    sparring_src = SRC.read_jsonl(SRC.SPARRING_PATH)
    assert len(windows) == sum(1 for r in sparring_src for t in r["dialogue"] if t["role"] == "assistant")
    for w in windows:
        assert w["messages"][-1]["role"] == "assistant"
        assert all(S.trained(m) for m in w["messages"] if m["role"] == "assistant")
    for text in ("shadow_avoided", "why_this_move"):
        assert not any(text in m["content"] for w in windows for m in w["messages"])
    black = [r for r in lineage if r["provenance"]["lineage"] == "sparring"]
    assert black and "shadow_avoided" in black[0]["messages"][-1]["content"]
    assert all(r["stage"] == "black" for r in lineage) and len(lineage) == 240 + 90


def test_cos_messages_pair_results_fifo_and_drop_unanswered_calls():
    row = cos_row("X", "s", "train", "result")
    row["trajectory"]["steps"].insert(0, {"kind": "tool_call", "name": "Orphan", "args": "{}", "result_text": None, "text": None,
                                          "train": True, "truncated": False})
    msgs, notes = SRC.cos_messages(row, "SYSTEM")
    roles = [m["role"] for m in msgs]
    assert roles[0:2] == ["system", "user"] and roles[-1] == "assistant"
    assert roles.count("tool") == 1
    calls = [m for m in msgs if m.get("tool_calls")]
    assert len(calls) == 1 and calls[0]["tool_calls"][0]["function"]["name"] == "Orphan"
    assert notes["calls_dropped_without_observation"] == 1


def test_c2_positive_turn_weight_follows_the_round(tmp_path):
    verdicts = {"analysis_file": "a", "round_of_session": {"S1": 2}, "round_weight": {"1": 1.0, "2": 0.7},
                "weight_basis": {"2": "supervisor"}, "turns": [{"session": "S1", "seq": 1, "verdict": "correct", "reason": "ok"},
                                                              {"session": "S1", "seq": 2, "verdict": "wrong", "reason": "bad"}]}
    session = {"turns": [{"seq": 1, "index": 0, "user_prompt": "ask <x>client</x>", "agent_answer": "answer",
                          "tool_calls": [{"tool": "t", "args": "{\"id\": \"%s\"}" % WI_TRAIN, "output": "out"}]},
                         {"seq": 2, "index": 1, "user_prompt": "q2", "agent_answer": "wrong", "tool_calls": []}]}
    (tmp_path / "S1.json").write_text(json.dumps(session))
    recs, excluded = SRC.c2_records(tmp_path, verdicts)
    assert len(recs) == 1 and recs[0]["weight"] == 0.7 and recs[0]["messages"][1]["content"] == "ask"
    assert [e["reason"] for e in excluded] == ["verdict_wrong"]


def test_coverage_reports_gaps_and_optional_forms(make):
    doc = json.dumps({"contract": "x", "diamond": {"discover": []}, "memory_review": []})
    full = [make(voice=v, kata="double-diamond", families=["double-diamond"]) for v in S.VOICES]
    full.append(make(messages=[S.user("q"), S.assistant(doc)], families=["memory-epistemics"], voice="seeker", source="dojo_sibling"))
    cov = COV.compute(full, {"green": ["scope"]})
    assert cov["double_diamond"]["white/source-labelled-kata"] == 8
    assert cov["errc"] == {"white/answer-document": 1}
    gaps = {(u["cell"], u.get("stage"), u.get("voice") or u.get("family") or u.get("capability")) for u in cov["uncovered"]}
    assert ("stage x voice", "green", "maker") in gaps
    assert ("stage x family", "green", "scope") in gaps
    assert ("rehearsal capability", None, "tool_native_calls") in gaps


def test_onpolicy_needs_a_base_export(tmp_path, codec):
    from program.export import onpolicy_export

    with pytest.raises(S.ProgramError):
        onpolicy_export(Config(out=tmp_path / "nowhere"), codec, tmp_path, "green")


def test_cli_defaults():
    a = parse_args(["--out", "x"])
    assert a.snapshot_until == "20261001T235959Z" and a.onpolicy is None and a.dojo_prompt == "compact"


def test_sentinels_absent_is_null_and_present_excludes_from_every_stage(world, codec, tmp_path):
    m = export_corpus(world, codec)
    assert m["exclusions"]["sentinels"] is None and "absent" in m["exclusions"]["sentinel_file"]["status"]
    drill = SRC.read_jsonl(SRC.KATA_PATH)[0]
    pick = SRC.read_jsonl(SRC.SPARRING_PATH)[0]
    sentinel = tmp_path / "sentinels.jsonl"
    sentinel.write_text(json.dumps({"kind": "kata-drill", "content_sha256": S.sha256_hex(drill["situation"])}) + "\n"
                        + json.dumps({"id": S.sha256_hex(pick)}) + "\n")
    held = Config(**{**world.__dict__, "sentinel_exclusions": sentinel})
    m2 = export_corpus(held, codec)
    assert m2["exclusions"]["sentinels"] >= 3  # the drill, its black-belt lineage record and the sparring row's records
    assert m2["exclusions"]["sentinel_file"]["keys"] >= 2
    ex = [json.loads(line) for line in (held.out / "exclusions.jsonl").read_text().splitlines() if '"sentinel"' in line]
    assert {e["stage"] for e in ex} >= {"black"} and any(e["source"] == "kata_drill" for e in ex)
    for i, st in enumerate(S.STAGES, 1):
        for line in (held.out / f"stage-{i}-{st}.tokens.jsonl").read_text().splitlines():
            assert json.loads(line)["record_id"] not in {e["record_id"] for e in ex}
