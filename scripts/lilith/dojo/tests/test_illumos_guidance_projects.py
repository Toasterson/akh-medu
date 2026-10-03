"""`illumos-guidance` resolves pack entries across every recorded Anima project collector, so the
illumos and OpenIndiana projects produce episodes; the Akh-Medu behaviour is unchanged."""
from __future__ import annotations

import json

import cos_recorder as rec
import dojo.families as F
from dojo.families import BuildContext
from dojo.hindsight import Outcome, later_outcome
from dojo.snapshot import SnapshotStore
from fixtures import T0, WI, Driver, hours

ILLUMOS = "019dda27-b1c5-79b0-929b-f9debcbd0b0b"
OPENINDIANA = "019de483-cdd9-7f52-9f49-7db6177102b3"


def add_project_items(world, sid, project, items, revisions=None):
    """Add an `anima:<project>` collector to snapshot `sid`: a listing plus one detail per item, as the recorder does.
    `items` = `[(id, phase)]`; `revisions` = `{id: change revisions}`."""
    revisions = revisions or {}
    path = world.root / sid / "manifest.json"
    manifest = json.loads(path.read_text())
    responses = []

    def put(call, args, payload):
        digest, _ = rec.store_object(world.root, payload)
        responses.append({"call": call, "args": args, "sha256": digest, "project_id": project})

    put("list_work_items", {"project_id": project}, {"work_items": [
        {"id": wid, "code": f"{project[:4]}-{n}", "current_phase": phase, "status": "open", "plan_id": None,
         "priority": "high", "title": f"item {wid}", "project_id": project} for n, (wid, phase) in enumerate(items)],
        "next_page_token": None})
    for wid, phase in items:
        put("get_work_item_detail", {"id": wid}, {
            "work_item": {"id": wid, "project_id": project, "current_phase": phase, "status": "open",
                          "executor": "x", "depends_on_ids": [], "dependency_edges": []},
            "prs": [], "change_history": [{"id": f"ch-{n}"} for n in range(revisions.get(wid, 0))], "ci_runs": []})
    manifest["collectors"][f"anima:{project}"] = {"status": "ok", "errors": [], "extra": {"project_id": project},
                                                  "responses": responses}
    rec.atomic_write_json(path, manifest)


def multi_project_world(world):
    s1 = world.snapshot(T0, [WI("am", "working", revisions=1)])
    add_project_items(world, s1, ILLUMOS, [("ill", "working")], {"ill": 1})
    add_project_items(world, s1, OPENINDIANA, [("oi", "working")], {"oi": 0})
    s2 = world.snapshot(T0 + hours(6), [WI("am", "working", revisions=2)])
    add_project_items(world, s2, ILLUMOS, [("ill", "working")], {"ill": 2})
    add_project_items(world, s2, OPENINDIANA, [("oi", "done")], {"oi": 0})
    return s1, s2


def test_pack_entries_resolve_across_project_collectors(world):
    s1, _ = multi_project_world(world)
    pack = (F.IllumosPackEntry("am", "smf", "operator"), F.IllumosPackEntry("ill", "zfs", "operator"),
            F.IllumosPackEntry("oi", "ips", "panel"), F.IllumosPackEntry("nowhere", "smf", "operator"))
    res = F.build_illumos_guidance(world.store(), s1, BuildContext(illumos_pack=pack))
    by = {s.subject_ids[0]: s for s in res.specs}
    assert set(by) == {"am", "ill", "oi"}
    assert by["am"].label["outcome"] == "reworked" and by["am"].label["flag_expected"] is True  # unchanged
    assert by["ill"].label["outcome"] == "reworked" and by["ill"].label["subsystem"] == "zfs"
    assert by["oi"].label["outcome"] == "clean" and by["oi"].label["flag_expected"] is False
    assert by["oi"].label["pack_source"] == "panel"
    assert by["oi"].situation == {"work_item": "oi", "phase": "working"}
    assert by["oi"].label_snapshot_ids and by["oi"].provenance.object_hashes


def test_replay_answers_detail_and_listing_for_the_extra_project(world, tmp_path):
    s1, _ = multi_project_world(world)
    pack = (F.IllumosPackEntry("ill", "zfs", "operator"),)
    res = F.build_illumos_guidance(world.store(), s1, BuildContext(illumos_pack=pack))
    spec = res.specs[0]
    d = Driver(spec, world.store(), tmp_path / "a.jsonl")
    detail, _ = d.call("get_work_item_detail", {"id": "ill"})
    assert ILLUMOS in json.dumps(detail)
    listing, _ = d.call("list_work_items", {"project_id": ILLUMOS})
    assert "ill" in json.dumps(listing) and '"am"' not in json.dumps(listing)


def test_extra_project_item_unsettled_is_excluded_and_terminal_skipped(world):
    s1 = world.snapshot(T0, [WI("am", "working")])
    add_project_items(world, s1, ILLUMOS, [("open", "working"), ("fin", "done")])
    world.snapshot(T0 + hours(6), [WI("am", "working")])
    pack = (F.IllumosPackEntry("open", "smf", "operator"), F.IllumosPackEntry("fin", "smf", "operator"))
    res = F.build_illumos_guidance(world.store(), s1, BuildContext(illumos_pack=pack))
    assert res.specs == [] and [e.subject for e in res.excluded] == ["open"]


def test_later_outcome_follows_an_extra_project_item(world):
    s1, _ = multi_project_world(world)
    label = later_outcome(world.store(), "oi", s1)
    assert label.outcome is Outcome.CLEAN and label.settled_snapshot is not None


def test_primary_wins_on_id_collision_and_primary_views_stay_clean(world):
    s1 = world.snapshot(T0, [WI("am", "working")])
    add_project_items(world, s1, ILLUMOS, [("am", "parked"), ("ill", "working")])
    snap = SnapshotStore(world.root).get(s1)
    assert set(snap.work_items) == {"am"}  # no leakage into the Akh-Medu views
    assert snap.work_item_any_project("am").phase == "working"
    assert snap.work_item_any_project("ill").phase == "working"
    assert snap.work_item_any_project("zzz") is None
