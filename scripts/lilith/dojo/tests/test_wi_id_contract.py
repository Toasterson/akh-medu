"""A sentence in an identifier field is a contract violation, not a fabricated reference."""
from dojo import contracts as C


def test_tracker_ids_match_and_prose_does_not():
    assert C.is_tracker_id("01a0e270-314e-7d72-bba4-82e5ae359e24")
    assert C.is_tracker_id("WI-309")
    assert C.is_tracker_id("plan-aaaa")
    assert not C.is_tracker_id("WI-309 is blocked by WI-308")
    assert not C.is_tracker_id("Request for progress update on harness repairs")
    assert not C.is_tracker_id(309)


def test_chase_contract_rejects_prose_in_about():
    schema = C.schema_for(C.COORDINATION_REQUEST)
    good = {"objective": "unblock", "about": ["WI-309"], "ack_deadline": 1,
            "receipts": [{"kind": "work-item", "reference": "r1", "digest": "sha256:" + "0" * 64, "observed_at": 1}]}
    bad = dict(good, about=["WI-309 is blocked by WI-308"])
    assert C.validate(good, schema) == []
    assert any("does not match" in e for e in C.validate(bad, schema))
