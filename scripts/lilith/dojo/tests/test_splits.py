import pytest

from dojo.blake3_compat import blake3_digest, blake3_pure, native_available
from dojo.episode import EpisodeSpec, Family
from dojo.errors import SplitLeakError
from dojo.snapshot import Provenance
from dojo.splits import (
    SealedIndex,
    Split,
    SplitConfig,
    SplitManifest,
    Assignment,
    assign_splits,
    decontaminate,
    normalized_words,
    sealed_index,
)


def spec(snapshot, plan="p1", horizon=(), subject="s", family=Family.PARK_TRIAGE, situation=None):
    return EpisodeSpec(family, snapshot, tuple(horizon), (plan,) if plan else (), (subject,),
                       situation or {"s": subject}, {"l": subject}, Provenance())


WEEK1 = "20260929T080000Z"
WEEK2 = "20261006T080000Z"
WEEK3 = "20261013T080000Z"


def test_newest_window_sealed_previous_dev_rest_train():
    specs = [spec(WEEK1, subject="a"), spec(WEEK2, subject="b"), spec(WEEK3, subject="c")]
    m = assign_splits(specs)
    assert {a.spec.subject_ids[0]: a.split for a in m.assignments} == {"a": Split.TRAIN, "b": Split.DEV, "c": Split.SEALED}


def test_held_out_plan_is_sealed_in_every_window_and_splits_are_disjoint():
    specs = [spec(WEEK1, plan="hold", subject="a"), spec(WEEK1, plan="p1", subject="b"),
             spec(WEEK3, plan="p1", subject="c"), spec(WEEK1, plan=None, subject="d")]
    m = assign_splits(specs, SplitConfig(held_out_plans=("hold",)))
    by = {a.spec.subject_ids[0]: (a.split, a.reason) for a in m.assignments}
    assert by["a"] == (Split.SEALED, "held-out-plan")
    assert by["b"][0] is Split.TRAIN and by["d"][0] is Split.TRAIN
    ids = [a.spec.episode_id for a in m.assignments]
    assert len(ids) == len(set(ids))
    sets = {s: {a.spec.episode_id for a in m.members(s)} for s in Split}
    assert not (sets[Split.TRAIN] & sets[Split.DEV]) and not (sets[Split.TRAIN] & sets[Split.SEALED])
    assert not (sets[Split.DEV] & sets[Split.SEALED])


def test_multi_plan_episode_is_split_by_time_only():
    brief = EpisodeSpec(Family.BRIEFING, WEEK1, (), ("hold", "p1"), ("x",), {}, {}, Provenance())
    m = assign_splits([brief, spec(WEEK3, subject="c")], SplitConfig(held_out_plans=("hold",)))
    assert m.members(Split.TRAIN)[0].spec.family is Family.BRIEFING


def test_labels_reading_a_later_window_are_dropped_not_moved():
    leaky_train = spec(WEEK1, horizon=(WEEK2,), subject="a")
    leaky_dev = spec(WEEK2, horizon=(WEEK3,), subject="b")
    m = assign_splits([leaky_train, leaky_dev, spec(WEEK3, subject="c")])
    assert {e for e, _ in m.dropped} == {leaky_train.episode_id, leaky_dev.episode_id}
    assert [a.spec.subject_ids[0] for a in m.assignments] == ["c"]


def test_manifest_hashes_are_deterministic_and_sensitive():
    specs = [spec(WEEK1, subject="a"), spec(WEEK3, subject="c")]
    one = assign_splits(specs).to_dict()
    two = assign_splits(list(reversed(specs))).to_dict()
    assert one == two
    three = assign_splits([spec(WEEK1, subject="a2"), spec(WEEK3, subject="c")]).to_dict()
    assert three["splits"]["train"]["sha256"] != one["splits"]["train"]["sha256"]
    assert three["manifest_sha256"] != one["manifest_sha256"]


def test_disjointness_check_catches_a_held_out_plan_in_train():
    held = spec(WEEK1, plan="hold", subject="a")
    other = spec(WEEK1, plan="hold", subject="b")
    bad = SplitManifest(SplitConfig(), (Assignment(held, Split.SEALED, 0, "held-out-plan"),
                                        Assignment(other, Split.TRAIN, 0, "older-window")), ())
    with pytest.raises(SplitLeakError):
        bad.check_disjoint()


def test_normalization_matches_p63():
    assert normalized_words("Ｈello,  WORLD!  é ß") == ["hello", "world", "é", "ß"]


def test_blake3_vectors_and_native_parity():
    assert blake3_pure(b"").hex() == "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
    data = bytes(i % 251 for i in range(3000))
    assert blake3_pure(data).hex() == "5fade288bf27444bee55ba2babb98c3c922c1e84c2e445e7d1f6da24756f5060"
    if native_available():
        for n in (1, 64, 65, 1024, 1025, 4097):
            payload = bytes(i % 251 for i in range(n))
            assert blake3_pure(payload) == blake3_digest(payload)


def test_decontamination_exact_shingle_and_baseline():
    sealed_text = " ".join(f"w{i}" for i in range(20))
    index = SealedIndex.from_texts([sealed_text, "short sealed prompt"])
    records = [
        ("exact", ["Short   SEALED prompt!"]),
        ("shingle", ["prefix " + " ".join(f"w{i}" for i in range(3, 16)) + " suffix"]),
        ("twelve", ["prefix " + " ".join(f"w{i}" for i in range(3, 15)) + " suffix"]),
        ("clean", ["nothing in common here"]),
    ]
    kept, report = decontaminate(records, index)
    assert kept == ["twelve", "clean"]
    assert dict(report.dropped) == {"exact": "exact", "shingle": "13-word-shingle"}
    index.subtract_baseline([sealed_text])
    kept, _ = decontaminate(records, index)
    assert "shingle" in kept and "exact" not in kept


def test_sealed_index_from_manifest_covers_situation_and_label():
    sealed = spec(WEEK3, subject="c", situation={"work_item": "wi-sealed", "phase": "parked"})
    m = assign_splits([spec(WEEK1, subject="a"), sealed])
    index = sealed_index(m)
    assert index.collision(sealed.prompt_text()) == "exact"
    assert index.collision(spec(WEEK1, subject="a").prompt_text()) is None


def test_sealed_index_reads_p63_jsonl(tmp_path):
    path = tmp_path / "sentinels.jsonl"
    path.write_text('{"prompt": "alpha beta", "target": "gamma"}\nnot json\n{"expected_call": {"tool": "x"}}\n')
    index = SealedIndex.from_jsonl([path])
    assert index.texts == 3 and index.collision("ALPHA beta") == "exact"
