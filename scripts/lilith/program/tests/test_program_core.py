"""Schema, voice mapping, masking, preference prefixes, splits, exposure. Synthetic data only."""
import copy

import pytest

from program import encode as E
from program import schema as S
from program import splits as SPL
from program.export import apply_boundary


# -- schema ---------------------------------------------------------------------------------------------------


def test_valid_record_and_stable_identity(make):
    r = make()
    assert S.validate_record(r) == []
    again = make()
    assert r["record_id"] == again["record_id"]
    assert make(stage="yellow")["record_id"] != r["record_id"]  # the stage is part of the identity
    moved = dict(r, split="dev", component_id="c-x", provenance={"a": 1})
    assert S.sha256_hex(S.identity_payload(moved)) == r["record_id"]  # split and provenance are not


def test_validation_rejects_bad_records(make):
    assert any("weight" in p for p in S.validate_record({**make(), "weight": 1.5}))
    assert any("positive weight" in p for p in S.validate_record(make(weight=0.0)))
    assert any("source" in p for p in S.validate_record({**make(), "source": "mystery"}))
    assert any("voice" in p for p in S.validate_record({**make(), "voice": "creator"}))
    assert any("record_id" in p for p in S.validate_record({**make(), "families": ["other"]}))
    assert any("no supervised" in p for p in S.validate_record(make(messages=[S.user("q"), S.assistant("a", train=False)])))
    assert any("role" in p for p in S.validate_record(make(messages=[{"role": "bot", "content": "x"}])))


def test_voice_mapping_covers_the_crew_and_rejects_unknown():
    source = ["creator", "sage", "explorer", "rebel", "caregiver", "mirror", "witness", "teacher"]
    assert [S.canonical_voice(v) for v in source] == list(S.VOICES)
    assert S.canonical_voice("Sacred-Clown") == "sacred-clown"
    assert S.canonical_voice(None) is None
    with pytest.raises(S.ProgramError):
        S.canonical_voice("jester")


def test_preference_record_needs_a_shared_prompt(make):
    prompt = [S.system("s"), S.user("q")]
    ok = make(kind="preference", messages=None, chosen_messages=[*prompt, S.assistant("good")],
              rejected_messages=[*prompt, S.assistant("bad")])
    assert S.validate_record(ok) == []
    diverged = make(kind="preference", messages=None, chosen_messages=[*prompt, S.assistant("good")],
                    rejected_messages=[S.system("s"), S.user("other"), S.assistant("bad")])
    assert any("share the same prompt" in p for p in S.validate_record(diverged))
    same = make(kind="preference", messages=None, chosen_messages=[*prompt, S.assistant("x")],
                rejected_messages=[*prompt, S.assistant("x")])
    assert S.validate_record(same)


# -- masking ---------------------------------------------------------------------------------------------------


def _decode_spans(codec, enc):
    """The masked-in and masked-out token runs, as the ids the codec would give for known texts."""
    sup = [t for t, m in zip(enc.input_ids, enc.target_mask) if m]
    return sup


def test_masking_supervises_only_assistant_actions(codec):
    calls = S.assistant_calls([("c1", "get_thing", {"id": "x"})])
    msgs = [S.system("SYSTEM TEXT"), S.user("USER TEXT"), calls, S.tool_message("c1", "OBSERVATION"), S.assistant("FINAL")]
    enc = E.encode_sft(codec, msgs)
    supervised = _decode_spans(codec, enc)
    expected = codec.encode(codec.call_text("get_thing", {"id": "x"}) + codec.tool_stop) + codec.encode("FINAL" + codec.end_of_turn)
    assert supervised == expected
    assert len(enc.input_ids) == len(enc.target_mask)
    n_prompt = enc.prompt_tokens
    assert not any(enc.target_mask[:n_prompt])  # system, user and generation boundary
    obs = codec.encode(codec.response_text("get_thing", "OBSERVATION"))
    start = n_prompt + len(codec.encode(codec.call_text("get_thing", {"id": "x"}) + codec.tool_stop))
    assert enc.input_ids[start:start + len(obs)] == obs
    assert not any(enc.target_mask[start:start + len(obs)])  # tool observation is masked


def test_context_only_assistant_steps_are_masked(codec):
    msgs = [S.user("q"), S.assistant_calls([("a", "sub", {})], train=False), S.tool_message("a", "R"), S.assistant("done")]
    enc = E.encode_sft(codec, msgs)
    sup = _decode_spans(codec, enc)
    assert sup == codec.encode("done" + codec.end_of_turn)


def test_dialogue_window_trains_only_the_last_turn(codec):
    msgs = [S.system("s"), S.user("u1"), S.assistant("a1"), S.user("u2"), S.assistant("a2")]
    enc = E.encode_sft(codec, msgs)
    assert _decode_spans(codec, enc) == codec.encode("a2" + codec.end_of_turn)
    assert enc.input_ids[: enc.prompt_tokens] == codec.encode(codec.context(msgs[:-1], None))


def test_unfit_records_are_reported_not_truncated(codec):
    msgs = [S.user("q"), S.assistant("x" * 400)]
    with pytest.raises(E.Unfit) as info:
        E.encode_sft(codec, msgs, limit=100)
    assert info.value.length > 400 and info.value.limit == 100


def test_block_errors(codec):
    with pytest.raises(S.ProgramError):
        E.encode_sft(codec, [S.user("q")])
    with pytest.raises(S.ProgramError):
        E.encode_sft(codec, [S.user("q"), S.assistant_calls([("a", "t", {})])])
    with pytest.raises(S.ProgramError):
        E.encode_sft(codec, [S.user("q"), S.assistant_calls([("a", "t", {})]), S.tool_message("zzz", "r"), S.assistant("x")])


def test_preference_tokens_share_the_prompt(codec):
    prompt = [S.system("s"), S.user("q")]
    c, r = E.encode_preference(codec, [*prompt, S.assistant("good answer")], [*prompt, S.assistant("bad")])
    assert c.input_ids[: c.prompt_tokens] == r.input_ids[: r.prompt_tokens]
    tok = {"record_id": "r", "stage": "green", "kind": "preference", "weight": 1.0, "source": "onpolicy", "families": ["scope"],
           "voice": None, "language": "en", "component_id": "c", "split": "train", "chosen": c.to_dict(), "rejected": r.to_dict()}
    assert S.validate_tokens(tok) == []
    broken = copy.deepcopy(tok)
    broken["rejected"]["input_ids"][1] += 1
    assert any("share" in p for p in S.validate_tokens(broken))
    with pytest.raises(S.ProgramError):
        E.encode_preference(codec, [*prompt, S.assistant("a")], [S.system("s"), S.user("different"), S.assistant("b")])


def test_token_validation_rejects_unsupervised_and_excluded():
    base = {"record_id": "r", "stage": "white", "kind": "sft", "weight": 1.0, "source": "kata_drill", "families": [],
            "voice": None, "language": "en", "component_id": "c", "split": "train", "input_ids": [1, 2], "target_mask": [0, 0]}
    assert any("supervises nothing" in p for p in S.validate_tokens(base))
    assert any("train and dev" in p for p in S.validate_tokens({**base, "target_mask": [0, 1], "split": "excluded"}))
    assert S.validate_tokens({**base, "target_mask": [0, 1]}) == []


# -- components and splits ---------------------------------------------------------------------------------------


def _linked(make, text, session=None, subjects=(), **prov):
    return make(messages=[S.user(text), S.assistant("a" + text)],
                provenance={"session_ids": [session] if session else [], "subject_ids": list(subjects), **prov})


def test_component_split_isolation(make):
    recs = []
    for i in range(60):
        recs.append(_linked(make, f"solo{i}"))
    chain = [_linked(make, "a", session="S1"), _linked(make, "b", session="S1", subjects=["W1"]),
             _linked(make, "c", subjects=["W1"]), _linked(make, "d", subjects=["W2"], session="S2"),
             _linked(make, "e", session="S2", subjects=["W3"])]
    recs += chain
    SPL.assign_splits(recs)
    ids = {r["record_id"]: r for r in recs}
    first = {ids[c["record_id"]]["component_id"] for c in chain[:3]}
    assert len(first) == 1
    assert len({ids[c["record_id"]]["split"] for c in chain[:3]}) == 1
    assert ids[chain[3]["record_id"]]["component_id"] == ids[chain[4]["record_id"]]["component_id"]
    assert ids[chain[3]["record_id"]]["component_id"] != ids[chain[0]["record_id"]]["component_id"]
    assert SPL.components_isolated(recs)
    assert {r["split"] for r in recs} <= {"train", "dev"}


def test_forced_splits_and_conflicts(make):
    a = _linked(make, "tr", session="S", force_split="train")
    b = _linked(make, "dv", session="S", force_split="dev")
    c = _linked(make, "dv2", session="T", force_split="dev")
    d = _linked(make, "tr2", session="T")
    rows = SPL.assign_splits([a, b, c, d])
    assert a["split"] == "train" and b["split"] == "excluded"
    assert c["split"] == "dev" and d["split"] == "dev"
    assert [r["reason"] for r in rows] == ["component_conflict"]
    assert SPL.components_isolated([a, b, c, d])


def test_splits_are_deterministic(make):
    def build():
        return [_linked(make, f"x{i}") for i in range(80)]

    one, two = build(), build()
    SPL.assign_splits(one)
    SPL.assign_splits(two)
    assert [r["split"] for r in one] == [r["split"] for r in two]
    assert "dev" in {r["split"] for r in one}


# -- exposure -----------------------------------------------------------------------------------------------------

WI = "01a0c932-73ae-7d92-a346-7e06c7a4d122"


def test_exposure_by_tool_response_text_hash(make):
    held_out = "{\"phase\": \"working\", \"note\": \"held-out observation\"}"
    exposure = SPL.Exposure()
    exposure.add_text("held_out_case", held_out)
    leaked = make(messages=[S.user("q"), S.assistant_calls([("c", "get", {})]), S.tool_message("c", held_out), S.assistant("a")])
    clean = make(messages=[S.user("q"), S.assistant_calls([("c", "get", {})]), S.tool_message("c", "other"), S.assistant("a")])
    kept, rows = apply_boundary([leaked, clean], exposure)
    assert [r["record_id"] for r in kept] == [clean["record_id"]]
    assert rows[0]["reason"] == "exposure_tool_response"


def test_exposure_by_work_item_id_inside_a_tool_response_and_session(make):
    exposure = SPL.Exposure()
    exposure.add(origin="eval", work_items=[WI], sessions=["sess-1"])
    in_tool = make(messages=[S.user("q"), S.assistant_calls([("c", "list", {})]),
                             S.tool_message("c", '{"items": [{"id": "%s"}]}' % WI), S.assistant("a")])
    by_session = make(messages=[S.user("q2"), S.assistant("a2")], provenance={"session_ids": ["sess-1"]})
    system_only = make(messages=[S.system("example id " + WI), S.user("q3"), S.assistant("a3")])
    kept, rows = apply_boundary([in_tool, by_session, system_only], exposure)
    assert [r["record_id"] for r in kept] == [system_only["record_id"]]  # the shared system prompt is not situation content
    assert sorted(r["reason"] for r in rows) == ["exposure_session", "exposure_work_item"]


def test_project_and_plan_ids_do_not_count_as_work_items():
    project = "019d9844-ce74-7993-a9f0-c953b8f04a4e"
    text = '{"project_id": "%s", "plan_id": null, "id": "%s"}' % (project, WI)
    assert SPL.work_item_uuids(text) == {WI}
    escaped = '{\\"project_id\\":\\"%s\\"}' % project
    assert SPL.work_item_uuids(escaped) == set()


def test_snapshots_at_or_after_the_cutoff_are_excluded(make):
    late = make(messages=[S.user("l"), S.assistant("l")], provenance={"snapshot_id": "20261002T000000Z"})
    early = make(messages=[S.user("e"), S.assistant("e")], provenance={"snapshot_id": "20261001T235959Z"})
    kept, rows = apply_boundary([late, early], SPL.Exposure())
    assert [r["record_id"] for r in kept] == [early["record_id"]]
    assert rows[0]["reason"] == "snapshot_after_cutoff"
