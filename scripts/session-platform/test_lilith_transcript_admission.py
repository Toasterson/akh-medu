import importlib.util
from pathlib import Path

P = Path(__file__).with_name("lilith_transcript_admission.py")
spec = importlib.util.spec_from_file_location("admission", P)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def event(kind, seq, render):
    return {"stream_id": "main", "seq": seq, "subseq": None, "kind": kind, "render": render}


def session(session_id="s1", **extra):
    return {"source": "codex", "session_id": session_id, "events": [event("prompt", 1, "fix x"), event("reply", 2, "done")], "truncated": False, "continue_from": None, **extra}


def test_session_split_dedup_and_eval_boundaries():
    s1, s2 = session(), session("s2")
    denied = m._session_hash("codex", "s2")
    rows, manifest = m.admit([s1, s1, s2], deny_sessions={denied}, deny_families=set(), eval_fingerprints=set(), evidence={}, salt="frozen")
    assert len(rows) == 1 and rows[0]["training_kind"] == "supervised_imitation"
    assert rows[0]["evidence"] == {"kind": "none"}
    assert manifest["rejected"] == {"duplicate_content": 1, "eval_session": 1}


def test_redaction_and_incomplete_pages():
    s = session(); s["events"][0]["render"] = "token=abc me@example.com /home/alice/key"
    incomplete = session("partial"); incomplete["truncated"] = True
    rows, manifest = m.admit([s, incomplete], deny_sessions=set(), deny_families=set(), eval_fingerprints=set(), evidence={}, salt="x")
    assert "abc" not in rows[0]["input"] and "example.com" not in rows[0]["input"] and "alice" not in rows[0]["input"]
    assert manifest["rejected"]["incomplete_or_invalid_session"] == 1


def test_verified_outcome_requires_bound_external_evidence():
    s = session(); base, _ = m.admit([s], deny_sessions=set(), deny_families=set(), eval_fingerprints=set(), evidence={}, salt="x")
    record_id = base[0]["record_id"]
    steps = [{"action": "observe", "receipt": {"passed": True, "sha256": "a" * 64}}]
    ext = {record_id: {"kind": "external_verifier", "passed": True, "verifier_sha256": "b" * 64, "outcome_sha256": m.sha(steps), "trajectory": {"steps": steps}}}
    rows, _ = m.admit([s], deny_sessions=set(), deny_families=set(), eval_fingerprints=set(), evidence=ext, salt="x")
    assert rows[0]["training_kind"] == "verified_outcome" and rows[0]["trajectory"]["steps"] == steps
    ext[record_id]["outcome_sha256"] = "c" * 64
    rows, manifest = m.admit([s], deny_sessions=set(), deny_families=set(), eval_fingerprints=set(), evidence=ext, salt="x")
    assert not rows and manifest["rejected"] == {"invalid_external_evidence": 1}


def test_eval_content_and_family_are_excluded():
    s = session(family_tags=["heldout"])
    rows, manifest = m.admit([s], deny_sessions=set(), deny_families={"heldout"}, eval_fingerprints=set(), evidence={}, salt="x")
    assert not rows and manifest["rejected"] == {"eval_family": 1}
    content = m.sha({"input": "fix x", "output": "done"})
    rows, manifest = m.admit([session()], deny_sessions=set(), deny_families=set(), eval_fingerprints={content}, evidence={}, salt="x")
    assert not rows and manifest["rejected"] == {"eval_content_collision": 1}
