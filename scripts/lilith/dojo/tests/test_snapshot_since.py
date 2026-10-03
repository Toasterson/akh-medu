"""`--snapshot-since`: the lower bound of the sealed exam window. A snapshot taken before it must not reach spec
building, hindsight labels, replay lookups or the run identity (the mirror of `test_snapshot_cutoff.py`)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dojo import baseline as B
from dojo import families as F
from dojo import replay_anima as RA
from dojo import runner as R
from dojo.errors import RunProvenanceError, SnapshotFormatError, UnknownSnapshotError
from dojo.families import BuildContext
from dojo.replay_server import StaticAncestryOracle
from dojo.snapshot import NO_CUTOFF, SnapshotStore
from fake_llm import FakeLLM
from fixtures import PROJECT, T0, WI, World, hours
from test_runner import park_script, provenance

PARKED = dict(parked_reason="rebase-conflict")


def stamp(when):
    return when.strftime("%Y%m%dT%H%M%SZ")


FLOOR = stamp(T0 + hours(2))


@pytest.fixture
def grown(world):
    """Three snapshots; the first (before the floor) is the training-era world, the later two the sealed window."""
    world.snapshot(T0, [WI("old", "parked", **PARKED), WI("p", "parked", **PARKED)])
    world.snapshot(T0 + hours(2), [WI("p", "parked", **PARKED), WI("q", "parked", **PARKED)])
    world.snapshot(T0 + hours(4), [WI("p", "working"), WI("q", "parked", **PARKED), WI("z", "parked", **PARKED)])
    return world


def test_floor_forms_and_visibility(grown):
    full = SnapshotStore.open(grown.root)
    assert len(full) == 3 and full.floor is None
    by_id = SnapshotStore.open(grown.root, None, FLOOR)
    assert by_id.ids == full.ids[1:] and by_id.floor == FLOOR and by_id.cutoff is None
    assert SnapshotStore.open(grown.root, None, (T0 + hours(2)).isoformat()).ids == by_id.ids
    assert SnapshotStore.open(grown.root, None, "2026-09-28T10:00:00").ids == by_id.ids  # naive = UTC
    assert SnapshotStore.open(grown.root, None, (T0 + hours(1)).isoformat()).ids == by_id.ids  # between two snapshots
    with pytest.raises(UnknownSnapshotError):
        by_id.get(full.ids[0])


def test_floor_and_cutoff_make_a_closed_window(grown):
    window = SnapshotStore.open(grown.root, stamp(T0 + hours(3)), FLOOR)
    assert window.ids == (FLOOR,) and window.cutoff == stamp(T0 + hours(3)) and window.floor == FLOOR
    auto = SnapshotStore.open(grown.root, "auto", FLOOR)
    assert auto.ids == SnapshotStore.open(grown.root).ids[1:] and auto.cutoff == stamp(T0 + hours(4))


def test_a_bad_or_empty_window_is_refused(grown):
    with pytest.raises(SnapshotFormatError):
        SnapshotStore.open(grown.root, None, "last-tuesday")
    with pytest.raises(SnapshotFormatError, match="after --snapshot-until"):
        SnapshotStore.open(grown.root, stamp(T0 + hours(2)), stamp(T0 + hours(4)))
    with pytest.raises(SnapshotFormatError, match="at or after"):
        SnapshotStore.open(grown.root, None, stamp(T0 + hours(9)))


def test_specs_come_only_from_the_post_floor_worlds(grown):
    full = {s.subject_ids for res in F.build_all(grown.store(), BuildContext()).values() for s in res.specs}
    window = SnapshotStore.open(grown.root, None, FLOOR)
    sealed = F.build_all(window, BuildContext())
    ids = {s.subject_ids for res in sealed.values() for s in res.specs}
    assert ("old",) in full and ("old",) not in ids  # the item that lived only before the floor is not a sealed world
    assert all(s.snapshot_id in window.ids for res in sealed.values() for s in res.specs)


def test_selected_episodes_all_anchor_at_or_after_the_floor(grown):
    oracle = StaticAncestryOracle({})
    chosen, _ = B.select_episodes(SnapshotStore.open(grown.root, None, FLOOR), oracle, [F.Family.PARK_TRIAGE], "all", None, 0)
    assert chosen and all(c.spec.snapshot_id >= FLOOR for c in chosen)
    whole, _ = B.select_episodes(grown.store(), oracle, [F.Family.PARK_TRIAGE], "all", None, 0)
    assert len(whole) > len(chosen)


def test_replay_project_metadata_cannot_come_from_before_the_floor(world):
    projects = ("list_projects", {}, {"projects": [{"id": PROJECT, "name": "Akh-Medu"}]}, {})
    world.snapshot(T0, [WI("p", "parked", **PARKED)], extra=(projects,))
    world.snapshot(T0 + hours(4), [WI("p", "parked", **PARKED)])
    RA._NAMES.clear()
    whole = RA.project_metadata(world.store().get(stamp(T0 + hours(4))))
    assert whole is not None and whole[0] == stamp(T0)
    RA._NAMES.clear()
    window = SnapshotStore.open(world.root, None, stamp(T0 + hours(2)))
    assert RA.project_metadata(window.get(stamp(T0 + hours(4)))) is None
    RA._NAMES.clear()


def test_a_run_without_a_floor_is_none_and_a_floor_is_part_of_the_identity():
    assert B.run_identity(provenance())["snapshot_since"] == NO_CUTOFF
    assert B.run_identity(provenance(snapshot_since=None))["snapshot_since"] == NO_CUTOFF
    assert B.run_identity(provenance(snapshot_since=FLOOR))["snapshot_since"] == FLOOR


def test_resume_with_a_different_floor_is_refused(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / B.PROVENANCE_FILE).write_text(json.dumps(provenance(snapshot_since=FLOOR)))
    assert B.check_resume(out, provenance(snapshot_since=FLOOR)) is not None
    for other in (None, stamp(T0 + hours(4))):
        with pytest.raises(RunProvenanceError, match="snapshot_since"):
            B.check_resume(out, provenance(snapshot_since=other))


def test_a_run_started_before_floors_existed_resumes_only_without_one(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / B.PROVENANCE_FILE).write_text(json.dumps(provenance()))
    assert B.check_resume(out, provenance(snapshot_since=None)) is not None
    with pytest.raises(RunProvenanceError, match="snapshot_since"):
        B.check_resume(out, provenance(snapshot_since=FLOOR))


def cli(grown, tmp_path, *extra):
    key = tmp_path / "key"
    key.write_text("sk-test-not-a-real-key")
    key.chmod(0o600)
    return ["run", "--root", str(grown.root), "--repo", str(Path(R.REPO_ROOT)), "--family", "park-triage", "--split", "all",
            "--model-kind", "openai", "--model", "lilith", "--key-file", str(key), "--allow-dirty", "--attempts", "1",
            *extra]


def test_dry_run_counts_follow_the_floor(grown, tmp_path, capsys):
    def dry(*extra):
        assert B.main([*cli(grown, tmp_path, *extra), "--dry-run"]) == 0
        return json.loads(capsys.readouterr().out)

    whole = dry()
    window = dry("--snapshot-since", FLOOR)
    assert whole["snapshots"] == 3 and window["snapshots"] == 2
    assert window["selection"]["available"] != whole["selection"]["available"]


def test_cli_records_the_floor_and_refuses_a_resume_with_another(grown, tmp_path, capsys):
    out = tmp_path / "out"
    with FakeLLM("openai", park_script("openai")) as fake:
        args = [*cli(grown, tmp_path, "--base-url", fake.base_url, "--out", str(out))]
        assert B.main([*args, "--snapshot-since", FLOOR]) == 0
        prov = json.loads((out / B.PROVENANCE_FILE).read_text())
        assert prov["snapshot_since"] == FLOOR and prov["snapshots"] == 2
        assert B.main([*args, "--snapshot-since", FLOOR]) == 0  # same floor resumes
        capsys.readouterr()
        assert B.main(args) == 2
        assert "snapshot_since" in capsys.readouterr().err
