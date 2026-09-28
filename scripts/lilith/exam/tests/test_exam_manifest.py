"""Exam manifest: pins, refusal to score, tiers, case identity, gates, decontamination hook."""
import dataclasses
import json
import shutil

import pytest

from dojo.floors import FLOORS_V1
from exam.decontam import decontaminate_training
from exam.errors import InvalidPlanError, ManifestIntegrityError, UnpinnedManifestError
from exam.manifest import (
    CheckpointRef,
    DecontaminationPolicy,
    ExamManifest,
    ExamPlan,
    SentinelPin,
    Tier,
    stable_seed,
)
from exam_helpers import CONTRACT, bare_manifest, pinned_manifest, write_sentinel


def test_stable_seed_is_fnv1a_with_terminators():
    # FNV-1a 64 of the single byte 0xff (an empty part) and a hand-checked two-part key.
    assert stable_seed([""]) == ((0xCBF29CE484222325 ^ 0xFF) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    assert stable_seed(["a", "b"]) != stable_seed(["ab"])


def test_four_tiers_and_sentinel_identity_excludes_checkpoint(tmp_path):
    m = pinned_manifest(tmp_path)
    assert [t.value for t in Tier] == ["sentinel", "rolling_probe", "frontier", "audit"]
    s1, s2 = m.plan(Tier.SENTINEL, "park-triage", CheckpointRef(10, "a" * 64)), \
        m.plan(Tier.SENTINEL, "park-triage", CheckpointRef(20, "b" * 64))
    assert s1.fingerprint() == s2.fingerprint() == m.sentinels["park-triage"].fingerprint
    assert s1.checkpoint.step == 10  # the checkpoint is still recorded
    r1, r2 = (m.plan(Tier.ROLLING_PROBE, "park-triage", CheckpointRef(s, "c" * 64)) for s in (10, 20))
    assert r1.seed != r2.seed and r1.fingerprint() != r2.fingerprint()
    frontier = m.plan(Tier.FRONTIER, "park-triage", CheckpointRef(10, "c" * 64), difficulty=3)
    assert frontier.difficulty == 3 and frontier.sample_count == 50
    assert ExamPlan.from_dict(frontier.to_dict()) == frontier
    with pytest.raises(InvalidPlanError):
        m.plan(Tier.FRONTIER, "park-triage", CheckpointRef(10, "c" * 64), difficulty=9)
    with pytest.raises(InvalidPlanError):
        m.plan(Tier.AUDIT, "unknown", CheckpointRef(10, "c" * 64))


def test_refuses_to_score_with_every_missing_pin_listed(tmp_path):
    m = bare_manifest(tool_contract_sha256=None, rubric_sha256="TBD")
    with pytest.raises(UnpinnedManifestError) as err:
        m.require_scorable()
    missing = " ".join(err.value.missing)
    for expected in ("tool_contract_sha256", "rubric_sha256", "sentinels[park-triage].sha256",
                     "sentinels[tool:memory_write].sha256"):
        assert expected in missing
    assert "lilith::exam::unpinned" in err.value.render()
    with pytest.raises(UnpinnedManifestError):
        m.save(tmp_path / "exam.json")  # an unpinned manifest is never written


def test_sentinel_pin_must_match_plan_n_and_floor(tmp_path):
    good = pinned_manifest(tmp_path)
    good.require_scorable()
    pin = good.sentinels["park-triage"]
    wrong_fp = dataclasses.replace(good, sentinels={**good.sentinels,
                                                    "park-triage": SentinelPin(pin.sha256, pin.n, "exam-fnv1a64-0")})
    assert any("fingerprint" in m for m in wrong_fp.missing_pins())
    small = dataclasses.replace(good, sampling={f: dataclasses.replace(s, tiers={**s.tiers, Tier.SENTINEL: 60})
                                                for f, s in good.sampling.items()})
    assert any("sealed floor of 100" in m for m in small.missing_pins())
    no_audit = dataclasses.replace(good, sampling={f: dataclasses.replace(s, tiers={t: n for t, n in s.tiers.items()
                                                                                   if t is not Tier.AUDIT})
                                                   for f, s in good.sampling.items()})
    assert any(m.endswith(".audit") for m in no_audit.missing_pins())


def test_save_load_round_trip_and_hash_pin(tmp_path):
    m = pinned_manifest(tmp_path)
    path = tmp_path / "exam.json"
    sha = m.save(path)
    loaded = ExamManifest.load(path, sha)
    assert loaded.manifest_sha256 == sha and loaded.tool_contract_sha256 == CONTRACT
    with pytest.raises(ManifestIntegrityError):
        ExamManifest.load(path, None)  # the run WI's pin is required
    with pytest.raises(ManifestIntegrityError):
        ExamManifest.load(path, "0" * 64)
    data = json.loads(path.read_text())
    data["statistics"]["fdr_alpha"] = 0.2  # a silent threshold change
    path.write_text(json.dumps(data))
    with pytest.raises(ManifestIntegrityError):
        ExamManifest.load(path, sha)


def test_gates_a_to_e_and_family_gates(tmp_path):
    m = pinned_manifest(tmp_path, families=("park-triage", "briefing", "tool:memory_write", "restart-recall"))
    body = m.body()
    assert set(body["gates"]) == {"A", "B", "C", "D", "E"}
    assert body["gates"]["B"]["families"]["briefing"]["blocked_recall"] == {"min": 0.95}
    assert body["floors"]["sha256"] == "5db3576893c97454b59f8f5d906c78937083aea29053a511c0abd18b5253be89"
    park = m.family_gate("park-triage")
    assert park.catastrophic_ceiling == 0.0 and [f.metric for f in park.metric_floors] == ["accuracy"]
    assert {f.metric for f in park.optional_metric_floors} >= {"language", "effect_success", "within_budget"}
    tool = m.family_gate("tool:memory_write")
    assert (tool.pass_floor, tool.catastrophic_ceiling, tool.gate) == (0.85, 0.01, "A.tool-matrix")
    assert m.family_gate("restart-recall").pass_floor == 0.95
    retention = next(g for g in body["gates"]["D"]["gates"] if g["id"] == "D.retention")
    assert retention["sentinel_regression_budget"] == 0.03


def test_tampered_floors_file_is_an_unpinned_manifest(tmp_path):
    copy = tmp_path / "floors.json"
    shutil.copy(FLOORS_V1, copy)
    copy.write_text(copy.read_text().replace('"min": 0.95', '"min": 0.50', 1))
    m = dataclasses.replace(pinned_manifest(tmp_path), floors_path=str(copy))
    assert any(p.startswith("floors:") for p in m.missing_pins())


def test_decontamination_against_pinned_sentinels(tmp_path):
    m = pinned_manifest(tmp_path, families=("park-triage",))
    sentinel = tmp_path / "sentinel-park-triage.jsonl"
    leaked = json.loads(sentinel.read_text().splitlines()[3])["prompt"]
    records = [("clean", ["a fresh sibling about a different repository and a different failure entirely"]),
               ("leak", [leaked]),
               ("shingle", ["prefix words " + " ".join(leaked.split()[:14])])]
    kept, report = decontaminate_training(records, m, {"park-triage": sentinel})
    assert kept == ["clean"]
    assert dict(report.dropped) == {"leak": "exact", "shingle": "13-word-shingle"}

    other = write_sentinel(tmp_path / "other.jsonl", "park-triage", n=99)
    with pytest.raises(ManifestIntegrityError):
        decontaminate_training(records, m, {"park-triage": other})
    with pytest.raises(ManifestIntegrityError):
        decontaminate_training(records, m, {})
    baseline = tmp_path / "baseline.txt"
    baseline.write_text(leaked + "\n")
    with pytest.raises(ManifestIntegrityError):  # baseline given but not pinned
        decontaminate_training(records, m, {"park-triage": sentinel}, baseline)
    from dojo.snapshot import sha256_hex
    pinned = dataclasses.replace(m, decontamination=DecontaminationPolicy(
        sentinel_baseline_sha256=sha256_hex(baseline.read_bytes())))
    kept, _ = decontaminate_training(records, pinned, {"park-triage": sentinel}, baseline)
    assert "leak" in kept  # boilerplate subtracted by the pinned baseline
