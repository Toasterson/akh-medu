"""Candidate / active manifests: activation only by operator acceptance; serving loads active only."""
import dataclasses
import hashlib
import json
import subprocess

import pytest

from exam.certify import (
    C5_TRAJECTORIES_V3,
    GEMMA4_12B_PARENT,
    AdapterRegistry,
    DataManifestPin,
    ModelManifest,
    OperatorAcceptance,
    State,
    gate_report,
    verify_on_main,
)
from exam.errors import CertificationError, NoActiveAdapterError

EXAM_SHA = "e" * 64
MAIN_SHA = "1" * 40


def candidate(adapter_sha: str, **overrides) -> ModelManifest:
    body = dict(run_id="lilith-train-r1", checkpoint_step=1200, adapter_sha256=adapter_sha,
                tool_contract_sha256="c" * 64, data_manifests=(C5_TRAJECTORIES_V3, DataManifestPin("dev", "d" * 64, "dev")),
                training_config_sha256="f" * 64, code_main_sha=MAIN_SHA, exam_manifest_sha256=EXAM_SHA,
                created_at="2026-09-28T12:00:00Z")
    body.update(overrides)
    return ModelManifest(**body)


def passing_report(exam_sha=EXAM_SHA, blockers=()):
    matrix = {"arm": "candidate", "families": {"park-triage": {"verdict": "pass", "blockers": [], "missing": []}}}
    return gate_report(exam_sha, [matrix], panel={"blockers": []}, extra_blockers=blockers)


@pytest.fixture
def adapter(tmp_path):
    path = tmp_path / "adapter_model.safetensors"
    path.write_bytes(b"lora weights")
    return path, hashlib.sha256(b"lora weights").hexdigest()


def test_candidate_pins_parent_and_c5_corpus(adapter):
    m = candidate(adapter[1])
    m.validate()
    body = m.to_dict()
    assert body["parent"]["weights_sha256"].startswith("5a84cb31")
    assert body["parent"]["config_sha256"].startswith("478c46e8")
    assert body["parent"]["tokenizer_sha256"].startswith("cc8d3a0c")
    assert any(p["sha256"].startswith("a89e98ea") for p in body["data_manifests"])
    assert body["reward_policy"]["name"] == "cos-v1"
    assert body["numerics"]["sdpa_backend"] == "efficient" and body["numerics"]["sdpa_math_fallback"] is True
    assert ModelManifest.from_dict(json.loads(json.dumps(body))) == m
    assert GEMMA4_12B_PARENT.revision.startswith("5926caa4")


def test_candidate_problems_are_listed(adapter):
    bad = candidate("nope", code_main_sha="main", data_manifests=(), reward_policy="kl-ppo")
    problems = " ".join(bad.problems())
    for expected in ("adapter_sha256", "code_main_sha", "no data manifests", "reward policy"):
        assert expected in problems
    with pytest.raises(CertificationError):
        AdapterRegistry(adapter[0].parent / "reg").write_candidate(bad)


def test_activation_only_by_operator_acceptance(tmp_path, adapter):
    reg = AdapterRegistry(tmp_path / "reg")
    m = candidate(adapter[1])
    reg.write_candidate(m)
    with pytest.raises(CertificationError):
        reg.write_candidate(m)  # write-once
    with pytest.raises(NoActiveAdapterError):
        reg.load_active()  # no silent fallback to the candidate
    report = passing_report()

    def accept(operator="toasty", candidate_sha=m.manifest_sha256, report_=report, reason="all gates green"):
        return reg.accept(m.manifest_sha256, report_, OperatorAcceptance(
            operator, "2026-09-28T13:00:00Z", candidate_sha, report_["gate_report_sha256"], reason))

    for who in ("lilith", "Supervisor", "fable", "", "astra"):
        with pytest.raises(CertificationError, match="only the human operator"):
            accept(operator=who)
    with pytest.raises(CertificationError, match="different candidate"):
        accept(candidate_sha="0" * 64)
    with pytest.raises(CertificationError, match="gates not passed"):
        accept(report_=passing_report(blockers=["candidate/briefing: fail"]))
    with pytest.raises(CertificationError, match="different exam manifest"):
        accept(report_=passing_report(exam_sha="9" * 64))
    tampered = dict(report, release_ready=True, blockers=[])
    tampered["matrices"] = []
    with pytest.raises(CertificationError, match="hash does not match"):
        accept(report_=tampered)
    with pytest.raises(CertificationError, match="no reason"):
        accept(reason=" ")
    assert not (tmp_path / "reg" / "active.json").exists()

    active = accept()
    assert active.state is State.ACTIVE and active.candidate_manifest_sha256 == m.manifest_sha256
    loaded = reg.load_active(adapter[0])
    assert loaded == active and loaded.acceptance.operator == "toasty"
    with pytest.raises(CertificationError, match="already accepted"):
        accept()


def test_load_active_refuses_tampering(tmp_path, adapter):
    reg = AdapterRegistry(tmp_path / "reg")
    m = candidate(adapter[1])
    reg.write_candidate(m)
    report = passing_report()
    reg.accept(m.manifest_sha256, report, OperatorAcceptance("toasty", "t", m.manifest_sha256,
                                                             report["gate_report_sha256"], "ok"))
    other = tmp_path / "other.safetensors"
    other.write_bytes(b"different weights")
    with pytest.raises(NoActiveAdapterError):
        reg.load_active(other)
    active_path = tmp_path / "reg" / "active.json"
    data = json.loads(active_path.read_text())
    data["adapter_sha256"] = "0" * 64
    active_path.write_text(json.dumps(data))
    with pytest.raises(CertificationError):
        reg.load_active()
    # a hand-written active manifest without an acceptance record is refused
    forged = dataclasses.replace(m, state=State.ACTIVE, candidate_manifest_sha256="1" * 64,
                                 gate_report_sha256="2" * 64,
                                 acceptance=OperatorAcceptance("toasty", "t", "1" * 64, "2" * 64, "forged"))
    active_path.write_text(json.dumps(forged.to_dict()))
    with pytest.raises(NoActiveAdapterError, match="no matching acceptance"):
        reg.load_active()


def test_verify_on_main(tmp_path):
    repo = tmp_path / "repo"
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@example.invalid", "PATH": "/usr/bin:/bin:/opt/homebrew/bin"}

    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                              env=env).stdout.strip()

    repo.mkdir()
    git("init", "-q")
    git("commit", "-q", "--allow-empty", "-m", "one")
    first = git("rev-parse", "HEAD")
    git("update-ref", "refs/remotes/origin/main", first)
    git("commit", "-q", "--allow-empty", "-m", "branch only")
    second = git("rev-parse", "HEAD")
    assert verify_on_main(first, repo)
    assert not verify_on_main(second, repo)
