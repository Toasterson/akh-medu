"""`runner.py reverify`: re-score a finished run against the current builders and verifiers, without a model."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from dojo import baseline as B
from dojo import families as F
from dojo import reverify as RV
from dojo import runner as R
from dojo.contracts import REPO_ROOT
from dojo.errors import DojoError
from dojo.families import BuildContext
from fixtures import T0, WI, Driver

ZERO_USAGE = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


def triage_doc(receipts, cause="conflict", action="rebase"):
    return {"contract": "lilith/dojo-park-triage@1", "language": "en", "work_item": "p", "cause": cause,
            "next_action": action, "receipts": receipts}


def played(world, tmp_path, spec, name="a"):
    d = Driver(spec, world.store(), tmp_path / f"{name}.jsonl")
    _, r = d.call("get_work_item_detail", {"id": "p"})
    d.emit(triage_doc([d.receipt(r)]))
    return d


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_run(tmp_path, world, spec, driver, *, recorded_spec=None, user_prompt_sha=None, tools_sha=None,
              tamper_receipt=False, name="run", extra=()):
    """A finished run directory the way `run_batch` writes one: episode records, verification records, provenance
    and summary. `recorded_spec` is the spec (label) the run scored it against; `spec` is what the builders make now."""
    run = tmp_path / name
    run.mkdir()
    recorded = recorded_spec or spec
    episode = driver.episode()
    snapshot = world.store().get(spec.snapshot_id)
    receipts = [r.to_dict() for r in episode.receipts]
    if tamper_receipt:
        receipts[0] = receipts[0] | {"response_sha256": "0" * 64}
    record = {
        "episode_id": spec.episode_id, "family": spec.family.value, "split": "sealed", "snapshot": spec.snapshot_id,
        "subjects": list(spec.subject_ids), "spec_sha256": recorded.spec_sha256, "run": "t",
        "steps": [s.to_dict() for s in episode.steps], "receipts": receipts, "status": "ok", "error": None,
        "answer_status": "document", "rounds": 2, "contract_ok": True, "schema_errors": [], "usage": ZERO_USAGE,
        "latency_ms": 10, "protocol_errors": [],
        "user_prompt_sha256": user_prompt_sha or sha(R.user_prompt(spec, snapshot.unix_time)),
        "tools_sha256": tools_sha or R.Broker(driver.servers).tools_sha256,
    }
    base = {"episode_id": spec.episode_id, "family": spec.family.value, "split": "sealed", "status": "ok",
            "answer_status": "document", "rounds": 2, "contract_ok": True, "schema_errors": [],
            "weight_policy": "cos-v1@1", "error": None}
    verification = B.score_episode(base, recorded, episode, snapshot, None)
    for rec in extra:
        record |= rec
    (run / B.EPISODES_FILE).write_text(json.dumps(record) + "\n")
    (run / B.VERIFICATION_FILE).write_text(json.dumps(verification) + "\n")
    provenance = {"schema": "lilith/dojo-run@1", "families": [spec.family.value], "seed": 0, "model_id": "m",
                  "endpoint_host": "h", "git_sha": "a" * 40, "git_dirty": False, "snapshot_root_sha256": "b" * 64,
                  "prompt_sha256": "c" * 64, "system_prompt_sha256": "d" * 64, "split": "sealed"}
    (run / B.PROVENANCE_FILE).write_text(json.dumps(provenance))
    (run / B.SUMMARY_FILE).write_text(json.dumps(B.build_summary(run, provenance, "source-arm")))
    return run, verification


def fingerprint(path: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.iterdir()) if p.is_file()}


@pytest.fixture
def parked(world):
    sid = world.snapshot(T0, [WI("p", "parked", parked_reason="rebase-conflict")])
    spec = F.build_park_triage(world.store(), sid, BuildContext()).specs[0]
    return world, spec


def reverify(world, run, tmp_path, **kw):
    return RV.run_reverify(run, world.root, REPO_ROOT, tmp_path / "out", allow_dirty=True, **kw)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_an_unchanged_episode_is_rescored_to_the_same_weight(parked, tmp_path):
    world, spec = parked
    run, before = write_run(tmp_path, world, spec, played(world, tmp_path, spec))
    done = reverify(world, run, tmp_path)
    (new,) = read_jsonl(tmp_path / "out" / B.VERIFICATION_FILE)
    assert (done["reverified"], done["skipped"]) == (1, 0)
    assert new["relabelled"] is False and new["prompt_drift"] is False
    assert new["weight"] == before["weight"] and new["previous_weight"] == before["weight"] > 0
    row = done["table"][0]
    assert row["family"] == "park-triage" and row["before_same_episodes"] == row["after"] == before["weight"]


def test_a_changed_label_is_rescored_and_marked_relabelled(parked, tmp_path):
    world, spec = parked
    old = dataclasses.replace(spec, label={**spec.label, "cause": "ci-red", "next_action": "fix-ci"})
    run, before = write_run(tmp_path, world, spec, played(world, tmp_path, spec), recorded_spec=old)
    assert before["weight"] == 0.0  # the old label called the right answer wrong
    done = reverify(world, run, tmp_path)
    (new,) = read_jsonl(tmp_path / "out" / B.VERIFICATION_FILE)
    assert new["relabelled"] is True and new["weight"] > 0.5 and new["previous_weight"] == 0.0
    assert new["previous_spec_sha256"] == old.spec_sha256 and new["spec_sha256"] == spec.spec_sha256
    assert done["table"][0]["relabelled"] == 1
    assert done["table"][0]["before_same_episodes"] == 0.0 and done["table"][0]["after"] > 0.5


def test_a_changed_prompt_is_skipped_unless_drift_is_allowed(parked, tmp_path):
    world, spec = parked
    run, _ = write_run(tmp_path, world, spec, played(world, tmp_path, spec), user_prompt_sha="e" * 64)
    done = reverify(world, run, tmp_path)
    assert done["reverified"] == 0 and done["skip_reasons"] == {RV.SkipReason.PROMPT_CHANGED: 1}
    assert not (tmp_path / "out" / B.VERIFICATION_FILE).exists() or not (tmp_path / "out" / B.VERIFICATION_FILE).read_text()
    (skip,) = read_jsonl(tmp_path / "out" / RV.SKIPPED_FILE)
    assert skip["episode_id"] == spec.episode_id and "model read" in skip["detail"]

    done = RV.run_reverify(run, world.root, REPO_ROOT, tmp_path / "out2", allow_dirty=True, allow_prompt_drift=True)
    (new,) = read_jsonl(tmp_path / "out2" / B.VERIFICATION_FILE)
    assert done["reverified"] == 1 and new["prompt_drift"] is True


def test_a_changed_tool_surface_is_skipped_and_never_waived(parked, tmp_path):
    world, spec = parked
    run, _ = write_run(tmp_path, world, spec, played(world, tmp_path, spec), tools_sha="f" * 64)
    done = RV.run_reverify(run, world.root, REPO_ROOT, tmp_path / "out", allow_dirty=True, allow_prompt_drift=True)
    assert done["skip_reasons"] == {RV.SkipReason.TOOLS_CHANGED: 1}


def test_replay_bytes_that_differ_from_what_the_model_read_are_skipped(parked, tmp_path):
    world, spec = parked
    run, _ = write_run(tmp_path, world, spec, played(world, tmp_path, spec), tamper_receipt=True)
    done = reverify(world, run, tmp_path)
    assert done["skip_reasons"] == {RV.SkipReason.REPLAY_DIFFERS: 1}
    (skip,) = read_jsonl(tmp_path / "out" / RV.SKIPPED_FILE)
    assert skip["detail"].startswith("r1 get_work_item_detail")


def rewrite(run, **fields):
    for name in (B.EPISODES_FILE, B.VERIFICATION_FILE):
        rec = json.loads((run / name).read_text()) | fields
        (run / name).write_text(json.dumps(rec) + "\n")


def test_episodes_the_builders_no_longer_make_are_skipped(parked, tmp_path):
    world, spec = parked
    run, _ = write_run(tmp_path, world, spec, played(world, tmp_path, spec))
    rewrite(run, episode_id="park-triage:20260928T080000Z:gone")
    done = reverify(world, run, tmp_path)
    assert done["skip_reasons"] == {RV.SkipReason.NOT_BUILT: 1}


def test_an_episode_the_source_run_did_not_score_is_skipped(parked, tmp_path):
    world, spec = parked
    run, _ = write_run(tmp_path, world, spec, played(world, tmp_path, spec))
    (run / B.VERIFICATION_FILE).write_text(json.dumps(
        json.loads((run / B.VERIFICATION_FILE).read_text()) | {"status": "error", "error": "endpoint failed"}) + "\n")
    done = reverify(world, run, tmp_path)
    assert done["skip_reasons"] == {RV.SkipReason.NOT_SCORED: 1}
    (skip,) = read_jsonl(tmp_path / "out" / RV.SKIPPED_FILE)
    assert skip["detail"] == "endpoint failed"


def test_the_source_run_is_never_written_and_the_output_is_new(parked, tmp_path):
    world, spec = parked
    run, _ = write_run(tmp_path, world, spec, played(world, tmp_path, spec))
    before = fingerprint(run)
    entries = sorted(p.name for p in tmp_path.iterdir())
    reverify(world, run, tmp_path)
    assert fingerprint(run) == before and sorted(p.name for p in run.iterdir()) == sorted(before)
    with pytest.raises(DojoError, match="already holds files"):
        reverify(world, run, tmp_path)
    with pytest.raises(DojoError, match="source run"):
        RV.run_reverify(run, world.root, REPO_ROOT, run, allow_dirty=True)
    default = RV.run_reverify(run, world.root, REPO_ROOT, None, allow_dirty=True)
    assert Path(default["out"]).name.startswith("run-reverify-") and Path(default["out"]).name.endswith(default["git_sha"][:8])
    assert Path(default["out"]).parent == tmp_path and entries


def test_provenance_names_the_source_run_and_the_reverify_commit(parked, tmp_path):
    world, spec = parked
    run, _ = write_run(tmp_path, world, spec, played(world, tmp_path, spec))
    done = reverify(world, run, tmp_path)
    prov = json.loads((tmp_path / "out" / B.PROVENANCE_FILE).read_text())
    meta = prov["reverify"]
    assert meta["source_run"] == str(run) and meta["source_git_sha"] == "a" * 40
    assert meta["git_sha"] == done["git_sha"] and len(meta["git_sha"]) == 40 and meta["git_dirty"] in (True, False)
    assert meta["prompt_sha256"] and meta["snapshot_root_sha256"] and meta["source_snapshot_root_sha256"] == "b" * 64
    assert (meta["reverified"], meta["skipped"], meta["relabelled"]) == (1, 0, 0)
    assert prov["git_sha"] == "a" * 40  # the producer of the answers stays the source run
    summary = json.loads((tmp_path / "out" / B.SUMMARY_FILE).read_text())
    assert summary["families"]["park-triage"]["n"] == 1 and summary["label"] == f"source-arm (reverified {done['git_sha'][:8]})"
    assert read_jsonl(tmp_path / "out" / B.EPISODES_FILE)[0]["reverified_from"] == str(run)
    assert "park-triage" in RV.render_table(done["table"], done["overall"])


def test_steps_round_trip_through_the_recorded_form(parked, tmp_path):
    world, spec = parked
    d = played(world, tmp_path, spec)
    d.send("imhotep", F.RecipientClass.EXECUTOR, {"objective": "o", "about": ["p"], "receipts": [], "ack_deadline": 3},
           sent_at=5, follow_up_at=9)
    for step in d.steps:
        assert RV.parse_step(step.to_dict(), step.episode_id) == step


def test_contract_status_follows_the_variant_and_the_current_schema(parked):
    world, spec = parked
    ok = F.StepRecord("x", 0, F.EmitDocument(triage_doc([{"kind": "tool-call", "reference": "r1", "observed_at": 1}])))
    assert RV.contract_status(spec, [ok]) == (True, [])
    bad = F.StepRecord("x", 0, F.EmitDocument({**ok.action.document, "cause": "nonsense"}))
    good_contract, errors = RV.contract_status(spec, [bad])
    assert good_contract is False and errors
