import random

import pytest

from dojo.episode import Family
from dojo.families import EpisodeVerdict, HardConstraint, StepVector, StepVerdict
from dojo.reward import COS_V1, ReductionPolicy, reduce_step, sft_records, sibling_pairs, weights_in_bounds

VALUES = (None, 0.0, 0.3, 1.0, -0.5, 1.7)


def sv(step=0, hard=(), **components):
    return StepVerdict(step, StepVector(**components), frozenset(hard))


def test_weight_always_in_unit_interval_and_zero_under_hard_constraints():
    rng = random.Random(7)
    weights = []
    for _ in range(2000):
        comps = {name: rng.choice(VALUES) for name in ("effect", "validity", "provenance", "disposition", "budget")}
        hard = rng.choice([(), (HardConstraint.FABRICATION,), (HardConstraint.RULE_BREAK, HardConstraint.UNANNOUNCED)])
        w = reduce_step(sv(hard=hard, **comps))
        weights.append(w)
        assert 0.0 <= w.weight <= 1.0
        if hard:
            assert w.weight == 0.0 and w.corrected_sibling_required
    assert weights_in_bounds(weights)


@pytest.mark.parametrize("hard", list(HardConstraint))
def test_each_hard_constraint_zeroes_and_asks_for_a_sibling(hard):
    w = reduce_step(sv(hard=(hard,), effect=1.0, validity=1.0, provenance=1.0, disposition=1.0, budget=1.0))
    assert (w.weight, w.failed, w.corrected_sibling_required) == (0.0, True, True)


def test_failed_steps_train_at_zero():
    assert reduce_step(sv(effect=1.0, validity=0.0)).weight == 0.0
    assert reduce_step(sv(effect=0.0, validity=1.0)).weight == 0.0


def test_missing_components_are_renormalised():
    only_validity_budget = reduce_step(sv(validity=1.0, budget=0.5))
    expected = (0.15 * 1.0 + 0.10 * 0.5) / 0.25
    assert only_validity_budget.weight == pytest.approx(expected)
    assert reduce_step(sv()).weight == 0.0


def test_ablation_by_zero_coefficient():
    no_disposition = ReductionPolicy("cos-v1-nodisp", 1, {"effect": 0.4, "validity": 0.15, "provenance": 0.2,
                                                          "disposition": 0.0, "budget": 0.1})
    a = reduce_step(sv(effect=1.0, validity=1.0, provenance=1.0, disposition=0.0, budget=1.0), no_disposition)
    assert a.weight == 1.0


def test_policy_is_named_versioned_and_hashed():
    assert COS_V1.name == "cos-v1" and COS_V1.version == 1
    assert COS_V1.policy_sha256 == ReductionPolicy("cos-v1", 1, dict(COS_V1.coefficients)).policy_sha256
    with pytest.raises(ValueError):
        ReductionPolicy("bad", 1, {"effect": -1.0, "validity": 1, "provenance": 1, "disposition": 1, "budget": 1})
    with pytest.raises(ValueError):
        ReductionPolicy("bad", 1, {"effect": 1.0})


def episode(steps, eid="briefing:x:1"):
    return EpisodeVerdict(eid, Family.BRIEFING, tuple(steps), {})


def test_sibling_pairs_prefer_the_corrected_sibling_only():
    raw = episode([sv(0, validity=1.0, budget=1.0), sv(1, hard=(HardConstraint.FABRICATION,), effect=1.0, validity=1.0)])
    fixed = episode([sv(0, validity=1.0, budget=1.0), sv(1, effect=1.0, validity=1.0, provenance=1.0, budget=1.0)])
    pairs = sibling_pairs(raw, fixed)
    assert [(p.step, p.chosen, p.rejected) for p in pairs] == [(1, "corrected-sibling", "raw")]
    assert all(p.margin > 0 and p.chosen_weight >= 0 and p.rejected_weight >= 0 for p in pairs)
    assert sibling_pairs(fixed, raw) == []  # a failing "sibling" is never chosen
    records = sft_records(raw, fixed)
    assert {(r.source, r.step): r.weight for r in records}[("raw", 1)] == 0.0
    assert {(r.source, r.step): r.weight for r in records}[("corrected-sibling", 1)] == 1.0
    assert all(0.0 <= r.weight <= 1.0 for r in records)


def test_sibling_contrast_requires_the_same_situation():
    with pytest.raises(ValueError):
        sibling_pairs(episode([], "a"), episode([], "b"))
