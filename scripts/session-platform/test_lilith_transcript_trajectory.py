import argparse
import json
import os
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import lilith_transcript_admission as admission  # noqa: E402
import lilith_transcript_trajectory as t  # noqa: E402

SESSION_ID = "11111111-2222-4333-8444-555555555555"
AGENT = "a0123456789abcdef"


def _write_jsonl(path, docs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(d) + "\n" for d in docs))


def _user(text, cwd="/Users/alice/ws/proj"):
    return {"type": "user", "cwd": cwd, "timestamp": "2026-09-01T00:00:00Z", "message": {"role": "user", "content": text}}


def _assistant(*blocks):
    return {"type": "assistant", "timestamp": "2026-09-01T00:00:01Z", "message": {"role": "assistant", "content": list(blocks)}}


def _tool_result(call_id, content, agent=None):
    doc = {"type": "user", "timestamp": "2026-09-01T00:00:02Z",
           "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call_id, "content": content}]}}
    if agent:
        doc["toolUseResult"] = {"agentId": agent, "status": "async_launched"}
    return doc


def make_session(root, big_result="ok"):
    """Main stream: prompt, thinking, Bash call/result, Agent call/result (spawns), reply; one subagent."""
    main = root / "-Users-alice-ws-proj" / f"{SESSION_ID}.jsonl"
    _write_jsonl(main, [
        {"type": "queue-operation", "content": "noise"},
        _user("status of run 7?"),
        _assistant({"type": "thinking", "thinking": "", "signature": "x"}),
        _assistant({"type": "tool_use", "id": "c1", "name": "Bash", "input": {"command": "ssh 192.168.1.20 uptime"}}),
        _tool_result("c1", big_result),
        _assistant({"type": "tool_use", "id": "c2", "name": "Agent", "input": {"prompt": "check logs"}}),
        _tool_result("c2", [{"type": "text", "text": "launched"}], agent=AGENT),
        _assistant({"type": "text", "text": "Run 7 is at step 400 on host 10.0.0.5."}),
        _assistant({"type": "tool_use", "id": "c3", "name": "Bash", "input": {"command": "tail log"}}),
        _tool_result("c3", "step 401"),
        _assistant({"type": "text", "text": "Now at 401; next checkpoint at 500."}),
        _user([{"type": "image", "source": {}}]),
        _assistant({"type": "text", "text": "an image arrived"}),
    ])
    _write_jsonl(main.with_suffix("") / "subagents" / f"agent-{AGENT}.jsonl", [
        {**_user("check logs"), "timestamp": "2026-09-01T00:00:03Z"},
        _assistant({"type": "tool_use", "id": "s1", "name": "Read", "input": {"file_path": "/Users/alice/log"}}),
        _tool_result("s1", "log line from alice"),
        _assistant({"type": "text", "text": "logs clean"}),
    ])
    return main


def _admit(root, **kw):
    session = t.load_local_session(make_session(root, **kw))
    rows, rejected = t.admit_trajectories([session], salt="s")
    return session, rows, rejected


# ── redaction ────────────────────────────────────────────────────────────────


def test_lan_ipv4_is_redacted_only_for_rfc1918_ranges():
    text = "a 10.1.2.3 b 172.16.0.1 c 172.31.255.254 d 192.168.1.20:5432 e 172.15.0.1 f 172.32.0.1 g 8.8.8.8 h 1.10.0.0.1 i 11.10.0.1"
    out = t.redact(text)
    assert out.count("[LAN_IP]") == 4
    for kept in ("172.15.0.1", "172.32.0.1", "8.8.8.8", "1.10.0.0.1", "11.10.0.1"):
        assert kept in out
    assert ":5432" in out


def test_path_slug_home_paths_and_credentials_are_redacted():
    out = t.redact("/private/tmp/claude-501/-Users-alice-ws-proj/x -home-bob-src- /Users/carol /home/dave/x "
                   f"me@example.org gho_{'a' * 36} postgres://u:secretpw@db/x")
    for leaked in ("alice", "bob", "carol", "dave", "example.org", "gho_", "secretpw"):
        assert leaked not in out
    assert "-Users-[USER]-ws-proj" in out and "[REDACTED:github_pat]" in out


def test_token_count_prose_is_not_a_credential():
    assert t.redact("token_count 17311 and token: 42") == "token_count 17311 and token: 42"


def test_account_names_come_from_cwd_structure():
    accounts = t.account_names(["/Users/alice/ws/proj", "/opt/x", "/home/bob"])
    assert accounts == frozenset({"alice", "bob"})
    out = t.redact("alice  staff  com.alice.agent  malice", accounts)
    assert out == "[USER]  staff  com.[USER].agent  malice"


def test_unterminated_private_key_is_withheld():
    assert t.redact("-----BEGIN OPENSSH PRIVATE KEY-----\nabc") == "[WITHHELD: private_key]"


def test_bound_keeps_head_and_tail_with_marker():
    text, truncated = t.bound("h" * 6000 + "t" * 6000, 8192)
    assert truncated and text.startswith("h" * 4096) and text.endswith("t" * 4096) and "3808 bytes elided" in text
    assert t.bound("short", 8192) == ("short", False)


# ── local adapter + trajectories ─────────────────────────────────────────────


def test_local_adapter_seq_subseq_and_kinds(tmp_path):
    session = t.load_local_session(make_session(tmp_path))
    assert list(session.streams) == ["main", f"agent-{AGENT}"]
    main = session.streams["main"]
    assert [e.kind for e in main] == ["prompt", "tool_call", "tool_result", "tool_call", "tool_result", "reply",
                                      "tool_call", "tool_result", "reply", "prompt", "reply"]
    assert main[0].ref == t.EventRef("main", 1, 0)  # seq counts the noise line, thinking yields no event
    assert main[4].spawned_agent == AGENT
    assert main[9].text is None  # degraded image block still resets the pending prompt
    assert session.accounts == frozenset({"alice"})


def test_trajectory_steps_splice_subagent_after_spawn(tmp_path):
    session, rows, _ = _admit(tmp_path)
    first = next(r for r in rows if r["input"] == "status of run 7?")
    steps = first["trajectory"]["steps"]
    assert [(s["kind"], s["name"], s["event_ref"]["stream_id"], s["train"]) for s in steps] == [
        ("tool_call", "Bash", "main", True), ("tool_result", "Bash", "main", False),
        ("tool_call", "Agent", "main", True), ("tool_result", "Agent", "main", False),
        ("tool_call", "Read", f"agent-{AGENT}", False), ("tool_result", "Read", f"agent-{AGENT}", False),
        ("assistant_text", None, "main", True),
        ("tool_call", "Bash", "main", True), ("tool_result", "Bash", "main", False),
        ("assistant_text", None, "main", True)]
    assert first["trajectory"]["subagent_streams"] == [f"agent-{AGENT}"]
    assert first["rated_reply"]["step_index"] == 6 and first["rated_reply"]["text"].startswith("Run 7 is")
    assert first["turn_final_reply"]["step_index"] == 9 and first["turn_extended"] is True
    assert first["output"] == "Now at 401; next checkpoint at 500." == steps[9]["text"]
    assert first["turn_final_reply"]["event_ref"] == {"stream_id": "main", "seq": 10, "subseq": 0}
    assert "192.168" not in json.dumps(first) and "10.0.0.5" not in json.dumps(first)
    image_turn = next(r for r in rows if r["output"] == "an image arrived")
    assert image_turn["turn_extended"] is False and image_turn["trajectory"]["steps_total"] == 1
    assert "alice" not in json.dumps(rows)
    assert steps[0]["args"] and steps[0]["result_text"] is None and steps[1]["args"] is None
    assert first["schema"] == t.SCHEMA and first["provenance"]["redaction_version"] == t.REDACTION_VERSION
    assert first["provenance"]["event_refs"] == [{"stream_id": "main", "seq": 1, "subseq": 0},
                                                 {"stream_id": "main", "seq": 7, "subseq": 0}]


def test_large_result_is_bounded_with_full_digest(tmp_path):
    big = "x" * 20000
    _, rows, _ = _admit(tmp_path, big_result=big)
    step = next(r for r in rows if r["input"] == "status of run 7?")["trajectory"]["steps"][1]
    assert step["truncated"] and len(step["result_text"].encode()) < 8300
    assert step["full_bytes"] == 20000 and step["full_sha256"] == admission.sha(big.encode())


def test_record_ids_match_admission_v1_for_the_same_turns(tmp_path):
    session, rows, _ = _admit(tmp_path)
    mcp_events = []
    for events in session.streams.values():
        for e in events:
            body = {"prompt": "t", "reply": "t", "tool_call": "a", "tool_result": "o"}[e.kind]
            render = f"{e.ref.seq} {e.kind}" + (f" | {body} {' '.join(e.text.split())}" if e.text else "")
            mcp_events.append({**e.ref.as_json(), "kind": e.kind, "render": render})
    mcp = {"source": "claude_code", "session_id": SESSION_ID, "events": mcp_events, "truncated": False, "continue_from": None}
    v1, _ = admission.admit([mcp], deny_sessions=set(), deny_families=set(), eval_fingerprints=set(), evidence={}, salt="s")
    assert {r["record_id"] for r in v1} == {r["record_id"] for r in rows}
    via_mcp, _ = t.admit_trajectories([t.events_from_mcp_session(mcp)], salt="s")
    assert {r["record_id"] for r in via_mcp} == {r["record_id"] for r in rows}
    assert all(s.get("upstream_truncated") for r in via_mcp for s in r["trajectory"]["steps"]
               if s["kind"] != "assistant_text")


def _long_turn(root, calls, result_bytes):
    main = root / "-Users-alice-ws-proj" / f"{SESSION_ID}.jsonl"
    docs = [_user("go"), _assistant({"type": "text", "text": "starting"})]
    for i in range(calls):
        docs += [_assistant({"type": "tool_use", "id": f"k{i}", "name": "Bash", "input": {"n": i}}),
                 _tool_result(f"k{i}", f"{i}:" + "r" * result_bytes)]
    docs += [_assistant({"type": "text", "text": "done"}), _user("next")]
    _write_jsonl(main, docs)
    rows, _ = t.admit_trajectories([t.load_local_session(main)], salt="s")
    return rows[0]


def test_step_cap_keeps_rated_and_final_reply(tmp_path):
    row = _long_turn(tmp_path, 100, 10)
    traj = row["trajectory"]
    assert traj["steps_total"] == 202 and len(traj["steps"]) == t.MAX_STEPS == 128 and traj["steps_elided"] == 74
    assert traj["steps"][row["rated_reply"]["step_index"]]["text"] == "starting"
    assert traj["steps"][row["turn_final_reply"]["step_index"]]["text"] == "done" == row["output"]
    seqs = [s["event_ref"]["seq"] for s in traj["steps"]]
    assert seqs == sorted(seqs)  # order preserved; the middle of the turn is what goes


def test_step_cap_is_pair_atomic(tmp_path):
    """Parallel calls (call, call, result, result) plus an odd number of texts: no half pair survives."""
    main = tmp_path / "-Users-alice-ws-proj" / f"{SESSION_ID}.jsonl"
    docs = [_user("go"), _assistant({"type": "text", "text": "starting"})]
    pairs = []  # (call line, result line) = seq, since every line yields one event
    for i in range(0, 90, 2):
        pairs += [(len(docs), len(docs) + 2), (len(docs) + 1, len(docs) + 3)]
        docs += [_assistant({"type": "tool_use", "id": f"p{i}", "name": "Bash", "input": {"n": i}}),
                 _assistant({"type": "tool_use", "id": f"p{i + 1}", "name": "Read", "input": {"n": i + 1}}),
                 _tool_result(f"p{i}", "a"), _tool_result(f"p{i + 1}", "b")]
        if i % 20 == 0:
            docs.append(_assistant({"type": "text", "text": f"interim {i}"}))
    docs += [_assistant({"type": "text", "text": "done"}), _user("next")]
    _write_jsonl(main, docs)
    rows, _ = t.admit_trajectories([t.load_local_session(main)], salt="s")
    traj = rows[0]["trajectory"]
    assert traj["steps_total"] > t.MAX_STEPS and len(traj["steps"]) <= t.MAX_STEPS
    assert traj["steps_elided"] == traj["steps_total"] - len(traj["steps"])
    calls = {s["args"] for s in traj["steps"] if s["kind"] == "tool_call"}
    results = [s for s in traj["steps"] if s["kind"] == "tool_result"]
    assert len(calls) == len(results) > 0
    names = {"Bash": "a", "Read": "b"}
    assert sorted(names[s["name"]] for s in traj["steps"] if s["kind"] == "tool_call") == \
        sorted(s["result_text"] for s in results)
    kept_seqs = {s["event_ref"]["seq"] for s in traj["steps"]}
    assert any(call in kept_seqs for call, _ in pairs) and not all(call in kept_seqs for call, _ in pairs)
    for call, result in pairs:
        assert (call in kept_seqs) == (result in kept_seqs), (call, result)


def test_validation_pins_require_a_recorded_decision(tmp_path):
    args = argparse.Namespace(output=tmp_path / "o", manifest=tmp_path / "m", validation_sessions=["aaaaaaaa"],
                              pin_decided_by=None, pin_decided_on=None)
    try:
        t.cos_export(args)
    except SystemExit as error:
        assert "--pin-decided-by" in str(error)
    else:
        raise AssertionError("pins without a recorded decision must be rejected")


def test_record_budget_trims_longest_results_and_keeps_digests(tmp_path):
    row = _long_turn(tmp_path, 40, 20000)
    traj = row["trajectory"]
    assert len(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()) <= t.RECORD_BUDGET_BYTES
    assert traj["bytes_trimmed"] > 0 and traj["over_budget"] is False and traj["steps_elided"] == 0
    result = traj["steps"][2]
    assert result["kind"] == "tool_result" and result["truncated"] and result["full_bytes"] == 20002
    assert result["full_sha256"] == admission.sha(("0:" + "r" * 20000).encode())


def test_validation_pinning_and_prefixes():
    counts = {"a" * 64: 10, "b" * 64: 50, "c" * 64: 40}
    default = t.validation_sessions(counts, "salt")
    pinned = t.validation_sessions(counts, "salt", pinned=frozenset({"b" * 64, "z" * 64}))
    assert pinned == default | {"b" * 64}  # pins add on top, never displace; unknown sessions ignored
    assert t.resolve_session_prefixes(["aaaaaaaa"], set(counts)) == frozenset({"a" * 64})
    for bad in (["aaaa"], ["dddddddd"]):
        try:
            t.resolve_session_prefixes(bad, set(counts))
        except SystemExit:
            continue
        raise AssertionError(f"{bad} must be rejected")


def test_resolver_maps_session_hash_to_file(tmp_path):
    main = make_session(tmp_path)
    digest = admission._session_hash("claude_code", SESSION_ID)
    assert t.resolve_session_files(tmp_path, {digest, "f" * 64}) == {digest: main}
    assert t.resolve_session_files(tmp_path / "absent", {digest}) == {}


# ── selection, cap, split, manifest ──────────────────────────────────────────


def test_session_cap_is_the_largest_quarter_share_fixed_point():
    counts = {"a": 91, "b": 36, "c": 21, "d": 11, "e": 4, "f": 4, "g": 3, "h": 2, "i": 1}
    k = t.session_cap(counts)
    kept = sum(min(c, k) for c in counts.values())
    assert k == 23 and k <= 0.25 * kept
    assert t.session_cap({"a": 5, "b": 5, "c": 5}) == 0


def test_validation_split_is_session_bound_and_near_ten_percent():
    counts = {f"s{i}": n for i, n in enumerate([23, 23, 21, 11, 4, 4, 3, 2, 1])}
    chosen = t.validation_sessions(counts, "salt")
    share = sum(counts[s] for s in chosen) / sum(counts.values())
    assert chosen and share <= 0.12
    assert chosen == t.validation_sessions(counts, "salt")


def _row(record_id, session):
    return {"record_id": record_id, "split": "train", "provenance": {"session_sha256": session}, "turn_extended": False,
            "trajectory": {"steps": [{"kind": "tool_call"}], "steps_elided": 0, "bytes_trimmed": 0,
                           "over_budget": False, "subagent_streams": []}}


def test_selection_excludes_with_reasons_and_caps_per_session():
    sessions = {f"{c * 64}": n for c, n in zip("abcdef", [12, 3, 3, 3, 3, 3])}
    rows, panel, record_sessions = {}, [], {}
    for session, n in sessions.items():
        for i in range(n):
            rid = admission.sha({"s": session, "i": i})
            rows[rid] = _row(rid, session)
            record_sessions[rid] = session
            panel.append({"record_id": rid, "cos_grade": True, "cos_class": "triage", "operator_correction": i == 0})
    panel.append({"record_id": "9" * 64, "cos_grade": False, "cos_class": "other"})
    lost, unparsed = "7" * 64, admission.sha({"unparsed": 1})
    record_sessions.update({"8" * 64: lost, unparsed: "a" * 64})
    panel += [{"record_id": "8" * 64, "cos_grade": True, "cos_class": "triage"},
              {"record_id": unparsed, "cos_grade": True, "cos_class": "triage"}]
    kept, manifest = t.select_cos_corpus(rows, panel, record_sessions, {lost: "missing_session_file"}, salt="x")
    assert manifest["panel_cos_records"] == 29
    assert manifest["cap_policy"]["per_session_cap"] == 5 and manifest["records"] == 20
    assert manifest["excluded_counts"] == {"missing_session_file": 1, "parse_error": 1, "session_cap": 7}
    assert {r["provenance"]["session_sha256"] for r in kept if r["split"] == "validation"} <= set(sessions)
    per_session = {}
    for r in kept:
        per_session.setdefault(r["provenance"]["session_sha256"], set()).add(r["split"])
    assert all(len(splits) == 1 for splits in per_session.values())
    assert all(r["labels"]["cos_class"] == "triage" for r in kept)
    assert manifest["operator_corrections"] == sum(r["labels"]["operator_correction"] for r in kept)


def test_cos_export_end_to_end_is_private_and_skips_mcp_without_token(tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    make_session(projects)
    session, rows, _ = _admit(tmp_path / "scratch")
    missing = "e" * 64
    admitted = [{"record_id": r["record_id"], "provenance": {"session_sha256": session.session_sha256}} for r in rows]
    admitted.append({"record_id": "d" * 64, "provenance": {"session_sha256": missing}})
    panel = [{"record_id": r["record_id"], "cos_grade": True, "cos_class": "briefing", "operator_correction": False} for r in rows]
    panel.append({"record_id": "d" * 64, "cos_grade": True, "cos_class": "triage"})
    (tmp_path / "admitted.jsonl").write_text("".join(json.dumps(a) + "\n" for a in admitted))
    (tmp_path / "panel.jsonl").write_text("".join(json.dumps(p) + "\n" for p in panel))

    def boom(*_a, **_k):
        raise AssertionError("MCP must not be called without a token")

    monkeypatch.setattr(t, "fetch_mcp_sessions", boom)
    monkeypatch.delenv("AKH_TRANSCRIPTS_MCP_TOKEN", raising=False)
    args = argparse.Namespace(panel=tmp_path / "panel.jsonl", admitted=tmp_path / "admitted.jsonl", projects_root=projects,
                              output=tmp_path / "out" / "c.jsonl", manifest=tmp_path / "out" / "m.json", salt="x",
                              mcp_url="https://mcp.invalid/mcp", mcp_token_env="AKH_TRANSCRIPTS_MCP_TOKEN", mcp_repo="r",
                              validation_sessions=[], pin_decided_by=None, pin_decided_on=None, pin_note="")
    manifest = t.cos_export(args)
    assert manifest["excluded_counts"]["missing_session_file"] == 1 and manifest["mcp_fallback"] == "not_configured"
    for path in (args.output, args.manifest):
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert manifest["dataset_sha256"] == admission.sha(args.output.read_bytes())
    try:
        t.cos_export(args)
    except SystemExit:
        pass
    else:
        raise AssertionError("must refuse to overwrite")
