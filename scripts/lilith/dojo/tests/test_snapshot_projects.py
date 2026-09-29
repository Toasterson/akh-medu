"""Multi-project snapshots: old (single-project) and new (`anima:<project_id>`) both load,
and extra-project work items never leak into the Akh-Medu views."""
from __future__ import annotations

import json

import cos_recorder as rec
from fixtures import T0, WI, World
from dojo.snapshot import DEFAULT_PROJECT_ID, SnapshotStore

ILLUMOS = "019dda27-b1c5-79b0-929b-f9debcbd0b0b"


def add_project(world: World, sid: str) -> None:
    """Rewrite the manifest of `sid` the way the multi-project recorder would."""
    path = world.root / sid / "manifest.json"
    manifest = json.loads(path.read_text())
    list_sha, _ = rec.store_object(world.root, {"work_items": [
        {"id": "ill-1", "code": "ILL-1", "current_phase": "working", "status": "open", "plan_id": None,
         "priority": "high", "title": "illumos item", "project_id": ILLUMOS}], "next_page_token": None})
    plans_sha, _ = rec.store_object(world.root, {"plans": [{"id": "ill-plan", "title": "P"}], "next_page_token": None})
    # tag the existing primary records with their project, as the new recorder does
    for r in manifest["collectors"]["anima"]["responses"]:
        r["project_id"] = DEFAULT_PROJECT_ID
    manifest["collectors"]["anima"]["extra"] = {"project_id": DEFAULT_PROJECT_ID}
    manifest["collectors"][f"anima:{ILLUMOS}"] = {
        "status": "ok", "errors": [], "extra": {"project_id": ILLUMOS},
        "responses": [
            {"call": "list_work_items", "args": {"project_id": ILLUMOS}, "sha256": list_sha, "project_id": ILLUMOS},
            {"call": "list_plans", "args": {"project_id": ILLUMOS}, "sha256": plans_sha, "project_id": ILLUMOS},
        ],
    }
    rec.atomic_write_json(path, manifest)


def test_old_single_project_snapshot_loads_and_maps_default_project(world):
    sid = world.snapshot(T0, [WI("a", "working")])
    snap = world.store().get(sid)
    assert set(snap.work_items) == {"a"}
    assert snap.project_collectors == {DEFAULT_PROJECT_ID: "anima"}
    assert all(r.project_id is None for r in snap.responses)


def test_extra_project_is_separate_from_the_primary_views(world):
    sid = world.snapshot(T0, [WI("a", "working")])
    add_project(world, sid)
    snap = SnapshotStore(world.root).get(sid)
    assert set(snap.work_items) == {"a"}  # no illumos leakage into Akh-Medu families
    assert set(snap.plans()) == {"plan-aaaa", "plan-bbbb"}
    assert snap.project_collectors == {DEFAULT_PROJECT_ID: "anima", ILLUMOS: f"anima:{ILLUMOS}"}
    extra = snap.work_items_of(f"anima:{ILLUMOS}")
    assert set(extra) == {"ill-1"} and extra["ill-1"].title == "illumos item"
    assert set(snap.plans(f"anima:{ILLUMOS}")) == {"ill-plan"}
    projects = {r.project_id for r in snap.responses if r.collector.startswith("anima")}
    assert projects == {DEFAULT_PROJECT_ID, ILLUMOS}
