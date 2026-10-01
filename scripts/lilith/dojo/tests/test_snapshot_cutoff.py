"""`--snapshot-until`: a snapshot taken after the cutoff must not reach spec building, hindsight labels, replay
lookups or the run identity; `compare` says which cutoffs and episode sets it paired."""
from __future__ import annotations

import json
import types
from datetime import timedelta
from pathlib import Path

import pytest

from dojo import baseline as B
from dojo import families as F
from dojo import hindsight as H
from dojo import memory_epistemics as ME
from dojo import replay_anima as RA
from dojo import reverify as RV
from dojo import runner as R
from dojo.errors import CompareError, RunProvenanceError, SnapshotFormatError, UnknownSnapshotError
from dojo.families import BuildContext
from dojo.replay_server import StaticAncestryOracle
from dojo.snapshot import NO_CUTOFF, SnapshotStore
from fake_llm import FakeLLM
from fixtures import PROJECT, T0, WI, World, hours
from test_reverify import played, write_run
from test_runner import park_script, provenance

PARKED = dict(parked_reason="rebase-conflict")


def stamp(when):
    return when.strftime("%Y%m%dT%H%M%SZ")


@pytest.fixture
def grown(world):
    """Three snapshots; the third (after the cutoff) resolves `p` and parks a new item `z`."""
    world.snapshot(T0, [WI("p", "parked", **PARKED), WI("q", "parked", **PARKED)])
    world.snapshot(T0 + hours(2), [WI("p", "parked", **PARKED), WI("q", "parked", **PARKED)])
    world.snapshot(T0 + hours(4), [WI("p", "working"), WI("q", "parked", **PARKED), WI("z", "parked", **PARKED)])
    return world


@pytest.fixture
def twin(tmp_path):
    """The same world as `grown` without its third snapshot: what a cutoff at T0+2h must be indistinguishable from."""
    world = World(tmp_path / "twin")
    world.snapshot(T0, [WI("p", "parked", **PARKED), WI("q", "parked", **PARKED)])
    world.snapshot(T0 + hours(2), [WI("p", "parked", **PARKED), WI("q", "parked", **PARKED)])
    return world


CUT = stamp(T0 + hours(2))


def spec_identities(store):
    results = F.build_all(store, BuildContext())
    return {(s.episode_id, s.spec_sha256) for res in results.values() for s in res.specs}


# --- the store ------------------------------------------------------------------------------------------------------


def test_cutoff_forms_and_visibility(grown):
    full = SnapshotStore.open(grown.root)
    assert len(full) == 3 and full.cutoff is None
    by_id = SnapshotStore.open(grown.root, CUT)
    assert by_id.ids == full.ids[:2] and by_id.cutoff == CUT
    assert SnapshotStore.open(grown.root, (T0 + hours(2)).isoformat()).ids == by_id.ids
    assert SnapshotStore.open(grown.root, "2026-09-28T10:00:00").ids == by_id.ids  # naive = UTC
    assert SnapshotStore.open(grown.root, (T0 + hours(3)).isoformat()).ids == by_id.ids  # between two snapshots
    assert SnapshotStore.open(grown.root, stamp(T0 + hours(2) - timedelta(seconds=1))).ids == full.ids[:1]
    with pytest.raises(UnknownSnapshotError):
        by_id.get(full.ids[2])


def test_auto_takes_the_newest_snapshot_and_records_its_id(grown):
    store = SnapshotStore.open(grown.root, "auto")
    assert store.cutoff == stamp(T0 + hours(4)) and len(store) == 3
    later = grown.snapshot(T0 + hours(6), [WI("p", "working")])
    assert store.ids[-1] != later and SnapshotStore.open(grown.root, store.cutoff).ids == store.ids


def test_a_bad_or_too_early_cutoff_is_refused(grown):
    with pytest.raises(SnapshotFormatError):
        SnapshotStore.open(grown.root, "yesterday-ish")
    with pytest.raises(SnapshotFormatError):
        SnapshotStore.open(grown.root, stamp(T0 - hours(1)))


# --- no leakage ----------------------------------------------------------------------------------------------------------


def test_specs_and_labels_are_what_the_world_without_the_later_snapshot_gives(grown, twin):
    assert spec_identities(grown.store()) != spec_identities(twin.store())  # the later snapshot does change things
    assert spec_identities(SnapshotStore.open(grown.root, CUT)) == spec_identities(twin.store())


def test_hindsight_never_reads_a_snapshot_after_the_cutoff(grown):
    first = SnapshotStore.open(grown.root).ids[0]
    (whole, *_), _ = H.park_labels(grown.store(), first)
    assert whole.work_item_id == "p" and whole.resolution is not None
    assert whole.resolution.snapshot_id == stamp(T0 + hours(4))
    cut = SnapshotStore.open(grown.root, CUT)
    labels, _ = H.park_labels(cut, first)
    for label in labels:
        assert stamp(T0 + hours(4)) not in label.provenance.snapshot_ids
        assert label.resolution is None or label.resolution.snapshot_id in cut.ids
    assert all(s.id in cut.ids for s in cut.later(first))
    assert not any(s.id == stamp(T0 + hours(4)) for s in cut.snapshots())


def test_selected_episodes_and_split_manifest_match_the_twin(grown, twin):
    oracle = StaticAncestryOracle({})
    pick = lambda store: B.select_episodes(store, oracle, [F.Family.PARK_TRIAGE], "all", None, 0)  # noqa: E731
    cut_chosen, cut_sel = pick(SnapshotStore.open(grown.root, CUT))
    twin_chosen, twin_sel = pick(twin.store())
    assert [c.spec.episode_id for c in cut_chosen] == [c.spec.episode_id for c in twin_chosen]
    assert cut_sel == twin_sel
    assert len(pick(grown.store())[0]) != len(cut_chosen)


def test_newest_and_first_at_or_after_stay_below_the_cutoff(grown):
    cut = SnapshotStore.open(grown.root, CUT)
    full = grown.store()
    newest = types.SimpleNamespace(snapshot=ME.NEWEST)
    assert ME.pick_snapshot(full, newest, T0) == stamp(T0 + hours(4))
    assert ME.pick_snapshot(cut, newest, T0) == CUT
    first_after = types.SimpleNamespace(snapshot=ME.FIRST_AFTER)
    seed = T0 + hours(3)
    assert ME.pick_snapshot(full, first_after, seed) == stamp(T0 + hours(4))
    assert ME.pick_snapshot(cut, first_after, seed) is None  # the only snapshot after the seed is past the cutoff
    pinned = types.SimpleNamespace(snapshot=stamp(T0 + hours(4)))
    assert ME.pick_snapshot(full, pinned, T0) == pinned.snapshot and ME.pick_snapshot(cut, pinned, T0) is None


def test_replay_project_metadata_cannot_come_from_a_later_snapshot(world):
    projects = ("list_projects", {}, {"projects": [{"id": PROJECT, "name": "Akh-Medu"}]}, {})
    world.snapshot(T0, [WI("p", "parked", **PARKED)])
    world.snapshot(T0 + hours(4), [WI("p", "parked", **PARKED)], extra=(projects,))
    RA._NAMES.clear()
    whole = RA.project_metadata(world.store().get(stamp(T0 + hours(4))))
    assert whole is not None and whole[0] == stamp(T0 + hours(4))
    RA._NAMES.clear()
    cut = SnapshotStore.open(world.root, stamp(T0 + hours(2)))
    assert RA.project_metadata(cut.get(stamp(T0))) is None
    RA._NAMES.clear()


# --- identity and resume -------------------------------------------------------------------------------------------------


def test_a_run_without_a_recorded_cutoff_is_none_and_a_cutoff_is_part_of_the_identity():
    assert B.run_identity(provenance())["snapshot_until"] == NO_CUTOFF
    assert B.run_identity(provenance(snapshot_until=None))["snapshot_until"] == NO_CUTOFF
    assert B.run_identity(provenance(snapshot_until=CUT))["snapshot_until"] == CUT


def test_resume_with_a_different_cutoff_is_refused(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / B.PROVENANCE_FILE).write_text(json.dumps(provenance(snapshot_until=CUT)))
    assert B.check_resume(out, provenance(snapshot_until=CUT)) is not None
    for other in (None, stamp(T0 + hours(4))):
        with pytest.raises(RunProvenanceError, match="snapshot_until"):
            B.check_resume(out, provenance(snapshot_until=other))


def test_a_run_started_before_cutoffs_existed_resumes_only_without_one(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / B.PROVENANCE_FILE).write_text(json.dumps(provenance()))
    assert B.check_resume(out, provenance(snapshot_until=None)) is not None
    with pytest.raises(RunProvenanceError, match="snapshot_until"):
        B.check_resume(out, provenance(snapshot_until=CUT))


# --- the CLI ---------------------------------------------------------------------------------------------------------------


def cli(grown, tmp_path, *extra):
    key = tmp_path / "key"
    key.write_text("sk-test-not-a-real-key")
    key.chmod(0o600)
    return ["run", "--root", str(grown.root), "--repo", str(Path(R.REPO_ROOT)), "--family", "park-triage", "--split", "all",
            "--model-kind", "openai", "--model", "lilith", "--key-file", str(key), "--allow-dirty", "--attempts", "1",
            *extra]


def dry(grown, tmp_path, capsys, *extra):
    assert B.main([*cli(grown, tmp_path, *extra), "--dry-run"]) == 0
    return json.loads(capsys.readouterr().out)


def test_dry_run_counts_follow_the_cutoff(grown, tmp_path, capsys):
    whole = dry(grown, tmp_path, capsys)
    cut = dry(grown, tmp_path, capsys, "--snapshot-until", CUT)
    auto = dry(grown, tmp_path, capsys, "--snapshot-until", "auto")
    assert whole["snapshot_until"] is None and whole["snapshots"] == 3
    assert cut["snapshot_until"] == CUT and cut["snapshots"] == 2
    assert cut["selection"]["available"] != whole["selection"]["available"]
    assert auto["snapshot_until"] == stamp(T0 + hours(4)) and auto["selection"] == whole["selection"]


def test_cli_records_the_cutoff_and_refuses_a_resume_with_another(grown, tmp_path, capsys):
    out = tmp_path / "out"
    with FakeLLM("openai", park_script("openai")) as fake:
        args = [*cli(grown, tmp_path, "--base-url", fake.base_url, "--out", str(out))]
        assert B.main([*args, "--snapshot-until", CUT]) == 0
        prov = json.loads((out / B.PROVENANCE_FILE).read_text())
        assert prov["snapshot_until"] == CUT and prov["snapshots"] == 2
        assert B.main([*args, "--snapshot-until", CUT]) == 0  # same cutoff resumes
        capsys.readouterr()
        assert B.main(args) == 2
        assert "snapshot_until" in capsys.readouterr().err
        assert B.main([*args, "--snapshot-until", "auto"]) == 2


def test_reverify_inherits_the_runs_cutoff_and_records_an_explicit_one(grown, tmp_path):
    cut = SnapshotStore.open(grown.root, CUT)
    spec = next(s for s in F.build_park_triage(cut, CUT, BuildContext()).specs if s.subject_ids == ("p",))
    run, _ = write_run(tmp_path, grown, spec, played(grown, tmp_path, spec))
    prov_path = run / B.PROVENANCE_FILE
    prov_path.write_text(json.dumps({**json.loads(prov_path.read_text()), "snapshot_until": CUT}))
    RV.run_reverify(run, grown.root, R.REPO_ROOT, tmp_path / "inherit", allow_dirty=True)
    inherited = json.loads((tmp_path / "inherit" / B.PROVENANCE_FILE).read_text())
    assert inherited["snapshot_until"] == CUT and inherited["reverify"]["source_snapshot_until"] == CUT
    RV.run_reverify(run, grown.root, R.REPO_ROOT, tmp_path / "explicit", allow_dirty=True,
                    snapshot_until=stamp(T0 + hours(4)))
    explicit = json.loads((tmp_path / "explicit" / B.PROVENANCE_FILE).read_text())
    assert explicit["snapshot_until"] == stamp(T0 + hours(4)) and explicit["reverify"]["source_snapshot_until"] == CUT


# --- compare -------------------------------------------------------------------------------------------------------------------


def arm(label, ids, cutoff=None, weight=0.5):
    per_episode = {i: {"family": "park-triage", "status": "ok", "weight": weight, "hard": []} for i in ids}
    prov = {"snapshot_root_sha256": "r", "system_prompt_sha256": "s"} | ({"snapshot_until": cutoff} if cutoff else {})
    return B.Arm(label, Path(label), {"per_episode": per_episode, "provenance": prov})


def test_compare_reports_the_intersection_and_warns_on_different_cutoffs_and_sets():
    parent = arm("parent", ["a", "b", "c"], weight=0.4)
    dry_run = arm("later", ["b", "c", "d", "e"], cutoff=CUT, weight=0.6)
    result = B.compare_arms([parent, dry_run])
    assert result["snapshot_until"] == {"parent": NO_CUTOFF, "later": CUT}
    assert result["episodes_per_arm"] == {"parent": 3, "later": 4}
    assert result["intersection"] == {"episode_ids_in_every_arm": 2, "scored_ok_in_every_arm": 2, "union_of_episode_ids": 5}
    assert result["episodes_in_common"] == 2 and result["rows"][-1]["n"] == 2
    warnings = " ".join(result["warnings"])
    assert "different snapshot cutoffs" in warnings and "different episode-id sets" in warnings
    rendered = B.render_compare(result)
    assert "snapshot cutoffs: parent=none, later=" + CUT in rendered and "parent=3, later=4" in rendered


def test_compare_is_quiet_for_arms_over_the_same_cutoff_and_ids():
    result = B.compare_arms([arm("a", ["x", "y"], CUT), arm("b", ["x", "y"], CUT, 0.7)], strict=True)
    assert not [w for w in result["warnings"] if "cutoff" in w or "episode-id" in w]
    assert result["intersection"]["episode_ids_in_every_arm"] == 2


@pytest.mark.parametrize("second", [arm("b", ["x", "y"], CUT), arm("b", ["x", "z"])])
def test_compare_strict_refuses_a_mismatch(second):
    with pytest.raises(CompareError, match="differ"):
        B.compare_arms([arm("a", ["x", "y"]), second], strict=True)


def test_compare_cli_strict_exit_code(tmp_path, capsys):
    dirs = []
    for label, ids, cutoff in (("a", ["x", "y"], None), ("b", ["x", "y"], CUT)):
        built = arm(label, ids, cutoff)
        d = tmp_path / label
        d.mkdir()
        (d / B.SUMMARY_FILE).write_text(json.dumps({"schema": B.SUMMARY_SCHEMA, "label": label, **built.summary}))
        dirs.append(str(d))
    assert B.main(["compare", *dirs]) == 0
    assert "different snapshot cutoffs" in capsys.readouterr().out
    assert B.main(["compare", "--strict", *dirs]) == 2
