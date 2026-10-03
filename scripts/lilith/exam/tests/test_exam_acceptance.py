"""The freeze and the acceptance driver, over synthetic worlds and fake endpoints. No private data, no network
beyond 127.0.0.1."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from dojo.errors import RunProvenanceError  # noqa: F401  (conftest puts the dojo tests on sys.path)
from dojo.snapshot import SnapshotStore
from exam import belts, freeze as FZ, run_acceptance as RA
from exam.certify import GEMMA4_12B_PARENT, ModelManifest
from exam.errors import ExamError, ManifestIntegrityError, UnpinnedManifestError
from exam.manifest import ExamManifest
from exam.panel import Rubric
from exam.result_doc import render
from fake_llm import FakeLLM, Say
from fixtures import T0, WI, hours
from program import splits as SPL
from test_runner import park_script

REPO = FZ.REPO
PARKED = dict(parked_reason="rebase-conflict")


def stamp(when):
    return when.strftime("%Y%m%dT%H%M%SZ")


# --- synthetic inputs --------------------------------------------------------------------------------------------------------------


def write_jsonl(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))
    return path


RECEIPT_SYSTEM = "You are Lilith, the Chief of Staff. Report only what the dated tracker receipt establishes."


def receipt_corpus(tmp_path: Path) -> Path:
    directory = tmp_path / "receipts"
    directory.mkdir()

    def record(n: int, family: str, split: str) -> dict:
        expected = {"as_of": f"2026-09-22T08:0{n}:00+00:00", "current_phase": "parked", "parked_reason": "too-big",
                    "receipt_id": f"{n:064x}", "work_item_id": f"01a0000{n}-0000-0000-0000-000000000000"}
        return {"record_id": f"{n + 100:064x}", "family": family, "work_item_id": expected["work_item_id"], "expected": expected,
                "messages": [{"role": "system", "content": RECEIPT_SYSTEM},
                             {"role": "user", "content": f"Report the phase of item {n}.\nReturn JSON with keys: as_of, current_phase"
                                                         f", parked_reason, receipt_id, work_item_id\nreceipt {n:064x}"}]}

    write_jsonl(directory / "dev.jsonl", [record(1, "status", "dev"), record(2, "memory_revision", "dev")])
    write_jsonl(directory / "historical_eval.jsonl", [record(3, "status", "eval")])
    write_jsonl(directory / "train.jsonl", [record(4, "status", "train")])
    return directory


SPOTS = [
    {"record_id": "spot-scope", "family": "scope", "prompt": "Resolve project Delta. Return JSON with project_id.",
     "expected": {"project_id": "3f0752ba-56f5-466e-9c03-83106eb5b914"}},
    {"record_id": "spot-read-failure", "family": "read-failure", "prompt": "The read failed. Return JSON with status.",
     "expected": {"status": "could not read", "park_reason": None, "verified": False}},
]


def freeze_inputs(tmp_path: Path, world, name: str = "frozen") -> FZ.Inputs:
    world.snapshot(T0, [WI("p", "parked", **PARKED), WI("q", "parked", **PARKED)])
    world.snapshot(T0 + hours(2), [WI("p", "parked", **PARKED), WI("q", "parked", **PARKED)])
    return FZ.Inputs(
        run_id="lilith-test-r1", out=tmp_path / name, snapshot_root=world.root, since=stamp(T0 + hours(2)), until=stamp(T0 + hours(2)),
        repo=REPO, memory_sessions=None, illumos_pack=None, drills=REPO / "data/phase51/character/kata-drills.jsonl",
        sparring=REPO / "data/phase51/character/sparring-green.jsonl", receipt_corpus=receipt_corpus(tmp_path),
        spots=_spots_file(tmp_path),
        teacher_provenance=None, session_platform=REPO / "scripts/session-platform")


def _spots_file(tmp_path: Path) -> Path:
    path = tmp_path / "spots.json"
    path.write_text(json.dumps(SPOTS))
    return path


@pytest.fixture
def frozen(tmp_path, world):
    inp = freeze_inputs(tmp_path, world)
    report = FZ.freeze(inp)
    return inp, report


# --- freeze ----------------------------------------------------------------------------------------------------------------------


def test_freeze_writes_a_scorable_manifest_with_every_pin(frozen):
    inp, report = frozen
    manifest = FZ.load_frozen(inp.out)
    manifest.require_scorable()  # also run inside load()
    assert manifest.manifest_sha256 == report["manifest_sha256"]
    assert manifest.run_id == "lilith-test-r1" and manifest.floors_sha256
    families = set(manifest.sentinels)
    assert "park-triage" in families and "coding:v2" in families and belts.SPOT_FAMILY in families
    assert {"historical-receipt:status", "historical-receipt:memory-revision"} <= families
    assert any(f.startswith("kata:white:") for f in families) and any(f.startswith("kata:yellow:") for f in families)
    assert len([f for f in families if f.startswith("sparring:")]) == 6
    assert FZ.verify(inp.out)["families"] == len(families)
    # every family is frozen non-core and the shortfall is on the record
    assert all(not manifest.family_gate(f).core for f in families)
    assert manifest.notes["sealed_shortfall"]["families"]["park-triage"]["sentinel_episodes"] == 2
    assert "tool:*" in manifest.notes["not_frozen"] and "illumos-guidance" in manifest.notes["not_frozen"]
    assert manifest.notes["window"]["snapshot_since"] == stamp(T0 + hours(2))


def test_freeze_is_reproducible_and_write_once(tmp_path, world):
    inp = freeze_inputs(tmp_path, world)
    first = FZ.freeze(inp)
    again = FZ.freeze(FZ.Inputs(**{**inp.__dict__, "out": tmp_path / "second"}))
    assert first["manifest_sha256"] == again["manifest_sha256"]
    assert {k: v for k, v in first["files"].items()} == {k: v for k, v in again["files"].items()}
    with pytest.raises(ExamError, match="exists"):
        FZ.freeze(inp)


def test_sealed_specs_come_only_from_the_window(tmp_path, world):
    inp = freeze_inputs(tmp_path, world)
    FZ.freeze(inp)
    rows = belts.load_jsonl(FZ.sentinel_path(inp.out, "park-triage"))
    assert rows and all(r["snapshot"] >= inp.since for r in rows)
    early = FZ.Inputs(**{**inp.__dict__, "out": tmp_path / "early", "since": stamp(T0)})
    FZ.freeze(early)
    earlier = belts.load_jsonl(FZ.sentinel_path(early.out, "park-triage"))
    assert {r["snapshot"] for r in earlier} == {stamp(T0)} and {r["snapshot"] for r in rows} == {inp.since}


def test_held_out_drills_are_a_deterministic_ten_percent_per_kata_and_belt(frozen):
    inp, _ = frozen
    drills = belts.load_jsonl(inp.drills)
    rows, picked = belts.kata_rows(drills)
    again, _ = belts.kata_rows(list(reversed(drills)))
    assert {r["case_id"] for r in rows} == {r["case_id"] for r in again}  # file order does not matter
    per_group: dict[tuple[str, str], int] = {}
    for r in rows:
        per_group[(r["kata"], r["belt"])] = per_group.get((r["kata"], r["belt"]), 0) + 1
    assert set(per_group.values()) == {2} and len(per_group) == 16  # ceil(10% of 15) per (kata, belt)
    assert len(picked) == len(rows) == 32


def test_the_exclusion_file_is_what_the_exporter_reads(frozen):
    inp, report = frozen
    keys = SPL.load_sentinels(inp.out / FZ.EXCLUSIONS_FILE)
    drills = belts.load_jsonl(inp.drills)
    rows, _ = belts.kata_rows(drills)
    sample = rows[0]
    # a variation record that names a held-out seed in its provenance is matched through the exporter's own function
    variation = {"record_id": "r" * 64, "provenance": {"seed_record_id": sample["seed_record_id"]},
                 "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "unrelated words here"}]}
    assert SPL.record_keys_for_sentinels(variation) & keys
    # a record that merely contains the held-out situation as a message is matched through the text hash
    quoting = {"record_id": "q" * 64, "messages": [{"role": "user", "content": sample["prompt"]}]}
    assert SPL.record_keys_for_sentinels(quoting) & keys
    # an unrelated record is not
    other = {"record_id": "o" * 64, "messages": [{"role": "user", "content": "something else entirely"}]}
    assert not SPL.record_keys_for_sentinels(other) & keys
    # every exporter key is an id or a sha256: no descriptive value that would exclude unrelated records
    for key in keys:
        assert len(key) == 64 or ":" in key or key.startswith("spot-"), key
    assert report["exclusions"]["by_tag"]["kata"] == 32
    detail = belts.load_jsonl(inp.out / FZ.DETAIL_FILE)
    assert any(r.get("kata") == sample["kata"] for r in detail)  # the description lives in the sidecar


def test_decontam_runs_later_and_reports_hits(frozen, tmp_path):
    inp, _ = frozen
    empty = tmp_path / "corpus"
    assert FZ.decontaminate(inp.out, empty)["status"] == "corpus-directory-missing"
    empty.mkdir()
    assert FZ.decontaminate(inp.out, empty)["status"] == "no-archives"
    rows, _ = belts.kata_rows(belts.load_jsonl(inp.drills))
    leaked = {"record_id": "a" * 64, "messages": [{"role": "user", "content": rows[0]["prompt"]},
                                                  {"role": "assistant", "content": "x"}]}
    clean = {"record_id": "b" * 64, "messages": [{"role": "user", "content": "nothing alike"}, {"role": "assistant", "content": "y"}]}
    write_jsonl(empty / "stage.records.jsonl", [leaked, clean])
    token_only = {"schema": "lilith/program-tokens@1", "record_id": rows[1]["seed_record_id"], "input_ids": [1, 2], "target_mask": [False, True]}
    write_jsonl(empty / "stage.tokens.jsonl", [token_only])
    result = FZ.decontaminate(inp.out, empty, tmp_path / "report.json")
    assert result["status"] == "hits" and result["id_hits"] >= 2 and result["exact_hits"] >= 1
    assert (tmp_path / "report.json").is_file()


def test_a_tampered_sentinel_or_manifest_is_refused(frozen):
    inp, report = frozen
    path = FZ.sentinel_path(inp.out, "park-triage")
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ManifestIntegrityError):
        FZ.load_frozen(inp.out)
    with pytest.raises(ManifestIntegrityError):
        FZ.load_frozen(inp.out, "0" * 64)


def test_the_rubric_is_anchored_and_pinned(frozen):
    inp, _ = frozen
    rubric = Rubric.from_dict(json.loads((inp.out / FZ.RUBRIC_FILE).read_text()))
    assert rubric.rubric_sha256 == FZ.cos_rubric().rubric_sha256
    assert all(set(d.anchors) == {0, 1, 2, 3} for d in rubric.dimensions)
    assert {"effect", "guidance", "grounding", "fit"} == set(rubric.dimension_ids)


# --- belts: scoring -----------------------------------------------------------------------------------------------------------------


def test_receipt_scoring_classifies_status_and_identifier_errors():
    expected = {"as_of": "t", "current_phase": "parked", "parked_reason": None, "receipt_id": "r" * 64, "work_item_id": "w"}
    base = {"case_id": "c", "family": "historical-receipt:status", "kind": belts.KIND_RECEIPT, "expected": expected,
            "catastrophic_fields": list(belts.RECEIPT_STATUS_FIELDS), "precursor_fields": list(belts.RECEIPT_IDENTIFIER_FIELDS)}
    ok = belts.score_structured(base, json.dumps(expected))
    assert ok.passed and not ok.catastrophic and not ok.precursor
    wrong_phase = belts.score_structured(base, json.dumps({**expected, "current_phase": "done"}))
    assert not wrong_phase.passed and wrong_phase.catastrophic
    wrong_id = belts.score_structured(base, json.dumps({**expected, "receipt_id": "x" * 64}))
    assert not wrong_id.passed and wrong_id.precursor and not wrong_id.catastrophic
    extra = belts.score_structured(base, json.dumps({**expected, "note": "more"}))
    assert not extra.passed and not extra.catastrophic
    junk = belts.score_structured(base, "I think it is parked")
    assert not junk.passed
    fenced = belts.score_structured({**base, "kind": belts.KIND_SPOT}, "```json\n" + json.dumps(expected) + "\n```")
    assert fenced.passed  # spots accept a complete fence, receipts keep the bare-JSON contract
    assert not belts.score_structured(base, "```json\n" + json.dumps(expected) + "\n```").passed


def test_reference_moves_pass_their_own_structure_checks_mostly():
    drills = belts.load_jsonl(REPO / "data/phase51/character/kata-drills.jsonl")
    profiles = belts.build_language_profiles(drills)
    rows, _ = belts.kata_rows(drills, fraction=1.0)
    cal = belts.calibration(rows, profiles)
    total = sum(g["n"] for g in cal.values())
    assert sum(g["pass"] for g in cal.values()) / total >= 0.9


def test_structure_checks_use_no_word_lists_and_reject_empty_or_wrong_language():
    drills = belts.load_jsonl(REPO / "data/phase51/character/kata-drills.jsonl")
    profiles = belts.build_language_profiles(drills)
    rows, _ = belts.kata_rows(drills, fraction=1.0)
    de = next(r for r in rows if r["lang"] == "de" and r["kata"] == "clarify-before-answer")
    assert not belts.score_belt(de, "", profiles)[0].passed
    assert belts.score_belt(de, de["target"], profiles)[0].passed
    english = "Which of the two would you like me to do first, the plumber or the electrician?"
    outcome, checks = belts.score_belt(de, english, profiles)
    assert any(c.name.startswith("language") and not c.passed for c in checks) or not outcome.passed


# --- the driver ------------------------------------------------------------------------------------------------------------------


KATA_REPLY_PARENT = "ok"


def make_script(inp: FZ.Inputs, task_module, candidate_wrong_coding: bool = False):
    """One fake server for both arms: the dojo (with tools) is answered by `park_script`; chat items by the frozen
    sentinel answers for `lilith` and by a poor answer for `lilith-parent`."""
    by_prompt: dict[str, str] = {}
    for family_file in (inp.out / "sentinels").glob("*.jsonl"):
        for row in belts.load_jsonl(family_file):
            if row["kind"] == belts.KIND_DOJO:
                continue
            by_prompt[belts.chat_messages(row)[-1]["content"]] = row["target"]
    dojo = park_script("openai")

    def script(body, index):
        if body.get("tools"):
            return dojo(body, index)
        prompt = body["messages"][-1]["content"]
        if prompt.startswith("Reply with the single word"):
            return Say("ready")
        if body["model"] == "lilith":
            return Say(by_prompt.get(prompt, "no answer"))
        return Say("ok")

    return script


@pytest.fixture
def run_dir(tmp_path, frozen):
    inp, report = frozen
    key = tmp_path / "key"
    key.write_text("sk-test-not-a-real-key")
    key.chmod(0o600)
    adapter_dir = tmp_path / "candidate-adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"adapter-bytes")
    import hashlib

    adapter_sha = hashlib.sha256(b"adapter-bytes").hexdigest()
    candidate = {
        "schema": "lilith/program-candidate@1", "run_id": "lilith-test-r1", "status": "candidate_inactive", "activated": False,
        "adapter": {"adapter_model_sha256": adapter_sha, "tensor_sha256": "t" * 64},
        "parent": {"repo": GEMMA4_12B_PARENT.repo, "revision": GEMMA4_12B_PARENT.revision},
        "program_manifest_sha256": "a" * 64, "archives_sha256": {"white": "b" * 64, "black": "c" * 64},
        "code": {"main_sha": "d" * 40}, "init_adapter": None, "seed": 7, "stage_steps": {"white": {"main": 3}}, "total_steps": 12,
        "cold_reload_probe": "written", "created_at": "2026-10-04T00:00:00+00:00"}
    run = tmp_path / "program-run"
    run.mkdir()
    (run / "candidate-manifest.json").write_text(json.dumps(candidate))
    (run / "cold-reload.probe.json").write_text("{}")
    return inp, report, key, adapter_dir, run


def args_for(inp, report, key, adapter_dir, run, fake_url, out, **extra):
    argv = ["run", "--exam-dir", str(inp.out), "--exam-manifest-sha256", report["manifest_sha256"],
            "--candidate-manifest", str(run / "candidate-manifest.json"), "--adapter-dir", str(adapter_dir),
            "--candidate-base-url", fake_url, "--candidate-key-file", str(key), "--snapshot-root", str(inp.snapshot_root),
            "--repo", str(REPO), "--memory-sessions", str(inp.out / "no-sessions"), "--out", str(out), "--no-main-check",
            "--allow-dirty", "--concurrency", "2", "--dojo-concurrency", "1"]
    for k, v in extra.items():
        argv += [f"--{k.replace('_', '-')}"] + ([] if v is True else [str(v)])
    return RA.build_parser().parse_args(argv)


def run_driver(inp, report, key, adapter_dir, run, out, fake, **extra):
    args = args_for(inp, report, key, adapter_dir, run, fake.base_url, out, **extra)
    args.candidate = RA._endpoint(args, "candidate", RA.ARM_CANDIDATE, None)
    args.parent = RA._endpoint(args, "parent", RA.ARM_PARENT, args.candidate)
    return RA.run(args, progress=lambda line: None)


def test_driver_end_to_end_with_fakes_then_panel_round_trip(run_dir, tmp_path):
    inp, report, key, adapter_dir, run = run_dir
    task_module = belts.load_task_module(REPO / "scripts/session-platform")
    # the candidate answers coding items with the reference solutions: route them by prompt
    out = tmp_path / "acceptance"
    result_md = tmp_path / "RESULT.md"
    with FakeLLM("openai", make_script(inp, task_module)) as fake:
        first = run_driver(inp, report, key, adapter_dir, run, out, fake, result_doc=result_md)
        served = [r for r in fake.requests if r["path"].endswith("/models")]
        assert served and served[0]["headers"]["user-agent"] == "curl/8.7.1"  # Runpod's proxy refuses Python's UA
    assert first["release_ready"] is False
    gate = json.loads((out / "gate-report.json").read_text())
    assert gate["exam_manifest_sha256"] == report["manifest_sha256"]
    blockers = "\n".join(gate["blockers"])
    # every thing that could not be evaluated is on the record, none skipped silently
    for needle in ("not evaluated: tool:*", "not evaluated: knowledge:*", "not evaluated: restart-recall", "underpowered: park-triage",
                   "panel: no judge results", "kata/sparring panel verdicts pending"):
        assert needle in blockers, needle
    evidence = json.loads((out / "acceptance-report.json").read_text())
    cand = gate["matrices"][0]["families"]
    assert gate["matrices"][0]["arm"] == "candidate" and cand["park-triage"]["n"] == 2
    assert cand["park-triage"]["verdict"] in ("pass", "insufficient-evidence")
    assert cand["coding:v2"]["n"] == 8 and cand["historical-receipt:status"]["n"] == 2
    assert evidence["reference_matrices"]["frozen-parent"]["families"]["park-triage"]["n"] == 2
    assert "park-triage" in evidence["comparisons"]["frozen-parent"]
    # candidate answers were the frozen targets: receipts and spots all correct, the parent's "ok" fails them
    assert cand["historical-receipt:status"]["passes"] == 2 and cand["behavior-spots"]["passes"] == 2
    parent = evidence["reference_matrices"]["frozen-parent"]["families"]
    assert parent["historical-receipt:status"]["passes"] == 0
    assert evidence["comparisons"]["frozen-parent"]["historical-receipt:status"]["delta"] == 1.0
    # certify: a validated candidate manifest bound to the exam manifest, written to the registry
    model = ModelManifest.from_dict(json.loads((out / "candidate-model-manifest.json").read_text()))
    assert model.exam_manifest_sha256 == report["manifest_sha256"] and model.tool_contract_sha256
    assert (out / "registry" / "candidates" / f"{model.manifest_sha256}.json").is_file()
    assert evidence["serving"]["ok"] and evidence["cold_reload"]["probe_present"]
    assert (out / "commands" / "teacher.txt").is_file()
    # the RESULT skeleton renders every section, with explicit "not recorded" where nothing exists
    text = result_md.read_text()
    assert "## Verdict" in text and "NOT CERTIFIED" in text and "_Not recorded yet:" in text
    assert render(json.loads((out / "run-result.json").read_text())) == text

    # panel work orders: blinded, one per (batch, judge); then results come back through --panel-results
    orders = sorted((out / "panel" / "orders").glob("primary-*.jsonl"))
    assert orders
    results = tmp_path / "judge-results"
    results.mkdir()
    rubric_sha = FZ.cos_rubric().rubric_sha256
    for order in orders:
        judge = order.stem.rsplit("-", 1)[1]
        lines = []
        for raw in order.read_text().splitlines():
            item = json.loads(raw)
            assert '"arm"' not in json.dumps(item) and "frozen-parent" not in json.dumps(item)
            lines.append({"schema": "lilith/panel-result@1", "blind_id": item["blind_id"], "judge": judge, "rubric_sha256": rubric_sha,
                          "verdict": "pass", "scores": {d: 2 for d in ("effect", "guidance", "grounding", "fit")},
                          "cited_turns": ["t1"], "note": "synthetic"})
        write_jsonl(results / f"{order.stem.replace('primary', 'result')}.jsonl", lines)
    with FakeLLM("openai", make_script(inp, task_module)) as fake:
        second = run_driver(inp, report, key, adapter_dir, run, tmp_path / "acceptance", fake, offline=True, panel_results=results)
    assert second["release_ready"] is False
    gate2 = json.loads((out / "gate-report.json").read_text())
    blockers2 = "\n".join(gate2["blockers"])
    assert "kata/sparring panel verdicts pending" not in blockers2 and "panel: no judge results" not in blockers2
    assert gate2["panel"]["rated"] > 0 and gate2["panel"]["verdict_gated"]
    assert gate2["gate_report_sha256"] != gate["gate_report_sha256"]


def test_driver_refuses_a_manifest_that_is_not_the_pinned_one(run_dir, tmp_path):
    inp, report, key, adapter_dir, run = run_dir
    with FakeLLM("openai", lambda body, index: Say("x")) as fake:
        args = args_for(inp, report, key, adapter_dir, run, fake.base_url, tmp_path / "o")
        args.exam_manifest_sha256 = "0" * 64
        args.candidate = RA._endpoint(args, "candidate", RA.ARM_CANDIDATE, None)
        args.parent = RA._endpoint(args, "parent", RA.ARM_PARENT, args.candidate)
        with pytest.raises(ManifestIntegrityError):
            RA.run(args, progress=lambda line: None)
        assert fake.requests == []  # refused before any model was asked


def test_waivers_cover_insufficient_evidence_only_and_never_by_an_agent(tmp_path):
    matrix = {"families": {"a": {"verdict": "insufficient-evidence", "missing": ["n"], "blockers": []},
                           "b": {"verdict": "fail", "missing": [], "blockers": ["floor"]}}}
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"waivers": [{"family": "a", "operator": "toasterson", "reason": "small n accepted", "at": "t"},
                                            {"family": "b", "operator": "toasterson", "reason": "x", "at": "t"}]}))
    waivers = RA.load_waivers(path)
    out, applied = RA.apply_waivers(matrix, waivers)
    assert out["families"]["a"]["verdict"] == "pass" and out["families"]["a"]["waived"]["operator"] == "toasterson"
    assert out["families"]["b"]["verdict"] == "fail" and [w["family"] for w in applied] == ["a"]
    path.write_text(json.dumps({"waivers": [{"family": "a", "operator": "opus", "reason": "r"}]}))
    with pytest.raises(Exception, match="only the human operator"):
        RA.load_waivers(path)


def test_a_manifest_without_pins_is_unscorable(frozen):
    inp, _ = frozen
    manifest = ExamManifest.from_dict(json.loads((inp.out / FZ.MANIFEST_FILE).read_text()))
    import dataclasses

    broken = dataclasses.replace(manifest, sentinels={}, rubric_sha256=None)
    with pytest.raises(UnpinnedManifestError):
        broken.require_scorable()
