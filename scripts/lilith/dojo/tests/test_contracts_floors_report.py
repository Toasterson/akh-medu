import json
import shutil

import pytest

from dojo import contracts as C
from dojo.contracts import PermissionProfile, Tier
from dojo.dojo_report import build_report, render
from dojo.errors import FrozenFloorsError
from dojo.floors import FLOORS_V1, floors, load_floors
from fixtures import T0, WI, hours


def test_c1_schemas_load_and_validate():
    schemas = C.load_cos_schemas()
    assert {C.BRIEFING, C.ACTION_ITEM, C.ANNOUNCEMENT, C.COMMITMENT, C.GUIDANCE, C.PROPOSAL} <= set(schemas)
    ok = {"contract": C.ACTION_ITEM, "language": "de-CH", "action": "a", "location": "l",
          "trigger": {"kind": "deadline", "due_at": 5}, "cost_of_skipping": "c",
          "receipts": [{"kind": "tool-call", "reference": "r1", "observed_at": 1}]}
    assert C.validate_document(ok) == []
    assert C.validate_document({**ok, "trigger": {"kind": "deadline"}})
    assert C.validate_document({**ok, "receipts": []})
    assert C.validate_document({**ok, "language": "German"})
    assert C.validate_document({**ok, "extra": 1})
    assert C.validate_document({"contract": "lilith/unknown@1"}) == ["$: unknown contract 'lilith/unknown@1'"]


def test_free_text_reads_prose_fields_only():
    doc = {"contract": C.PARK_TRIAGE, "language": "en", "work_item": "wi-1", "cause": "ci-red",
           "next_action": "fix-ci", "note": "prose here",
           "receipts": [{"kind": "tool-call", "reference": "r1", "observed_at": 1}]}
    assert C.free_text(doc) == ["prose here"]


def test_permission_profile_tiers():
    profile = PermissionProfile.load()
    assert profile.server_tier("anima", "get_work_item_detail") is Tier.T0_READ
    assert profile.server_tier("anima", "list_work_items") is Tier.T0_READ
    assert profile.server_tier("anima", "approve_and_merge") is Tier.T3_OPERATOR_CLASS
    assert profile.server_tier("anima", "post_message") is Tier.T2_COORDINATION
    assert profile.tier("lilith_adapter_load") is Tier.OUTSIDE_GRANT
    assert profile.tier("totally_new_tool") is Tier.UNLISTED


def test_floors_are_frozen_and_carry_the_plan_values(tmp_path):
    doc = load_floors()
    table = {(f.family, f.metric): f for f in floors(doc)}
    assert table[("briefing", "blocked_recall")].minimum == 0.95
    assert table[("briefing", "precision")].minimum == 0.90
    assert table[("briefing", "stale_recall")].minimum == 0.90
    assert table[("commitment", "on_time")].minimum == 0.90
    assert table[("park-triage", "accuracy")].minimum == 0.80
    assert table[("action-item", "completeness")].minimum == 0.95
    assert table[("all_families", "language")].minimum == 0.98
    assert table[("illumos-guidance", "flagged_precision")].minimum == 0.80
    assert table[("illumos-guidance", "later_failure_recall")].minimum == 0.60
    assert table[("all_families", "fabricated_status_claims")].maximum == 0
    assert not table[("all_families", "fabricated_status_claims")].passes(1)
    copy = tmp_path / "floors.json"
    shutil.copy(FLOORS_V1, copy)
    edited = json.loads(copy.read_text())
    edited["gate_b_chief_of_staff"]["briefing"]["precision"]["min"] = 0.5
    copy.write_text(json.dumps(edited))
    with pytest.raises(FrozenFloorsError):
        load_floors(copy)


def test_report_on_fixture_snapshots(world):
    world.snapshot(T0, [WI("p", "parked", parked_reason="retry-budget"), WI("w", "working")])
    world.snapshot(T0 + hours(2), [WI("p", "working"), WI("w", "done")])
    report = build_report(world.root, None, None)
    assert report["snapshots"] == 2 and report["phase_transitions"] == 2
    fams = report["families"]
    assert fams["briefing"]["episodes"] == 2
    assert fams["park-triage"]["labels"] == {"retry-s3": 1}
    assert fams["correction"]["episodes"] == 0 and "C2" in fams["correction"]["note"]
    assert report["splits"]["sealed"] == sum(f["episodes"] for f in fams.values())
    text = render(report)
    assert "recorder started on 2026-09-28" in text and "everything is sealed" in text


def test_report_on_an_empty_root(tmp_path):
    (tmp_path / "empty").mkdir()
    report = build_report(tmp_path / "empty", None, None)
    assert report["snapshots"] == 0 and "nothing can be built" in render(report)
