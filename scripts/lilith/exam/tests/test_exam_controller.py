"""Controller decisions: golden parity with adaptive_exam.rs, its documented behaviour, and
the Lilith additions (stop latch, catastrophic classification, matrix, paired differences)."""
import json
import math
from pathlib import Path

import pytest

from exam.controller import (
    AdaptiveExamController,
    CapabilitySpec,
    Catastrophic,
    ControllerConfig,
    ExamObservation,
    ExamRun,
    ItemOutcome,
    Precursor,
    Verdict,
    WarningLevel,
    benjamini_hochberg,
    binomial_upper_tail,
    derive_training_sibling,
    family_matrix,
    paired_difference,
    steer_budget,
    wilson_interval,
)
from exam.errors import (
    ControllerConfigError,
    InvalidCountsError,
    ProtectedTierError,
    StoppedSegmentError,
    UnknownFamilyError,
)
from exam.manifest import CheckpointRef, FamilyGate, MetricFloor, StatisticalRule, Tier

GOLDEN = json.loads((Path(__file__).parent / "golden" / "adaptive_exam_rust.json").read_text())
FAMILIES = ("reason.chain.depth", "vision.text.line")


def controller(config: ControllerConfig = ControllerConfig()) -> AdaptiveExamController:
    return AdaptiveExamController("test-generator-v1", "run-42", 42, config,
                                  [CapabilitySpec(f, 0, 5, 2) for f in FAMILIES])


def ck(step: int) -> CheckpointRef:
    return CheckpointRef(step, f"sha256:{step:064x}")


def obs(step, tier, family, pass_, pre=0, cat=0, n=400) -> ExamObservation:
    return ExamObservation(ck(step), tier, family, 2, n, pass_, pre, cat)


def close(a, b, path="$"):
    """Deep equality with float tolerance; reports the first differing path."""
    if isinstance(a, dict) and isinstance(b, dict):
        assert set(a) == set(b), f"{path}: keys {sorted(set(a) ^ set(b))}"
        for k in a:
            close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), f"{path}: length"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert a is not None and b is not None and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


# --- golden parity with the Rust controller -------------------------------------------------


def test_default_config_matches_rust():
    close(ControllerConfig().to_dict(), GOLDEN["default_config"])


def test_plans_and_fingerprints_match_rust():
    c = controller()
    for row in GOLDEN["plans"]:
        p = row["plan"]
        plan = c.plan_for_family(Tier(p["tier"]), CheckpointRef.from_dict(p["checkpoint"]), p["family"], p["sample_count"])
        assert plan.to_dict() == p
        assert plan.fingerprint() == row["fingerprint"]


def test_wilson_matches_rust():
    for row in GOLDEN["wilson"]:
        close(wilson_interval(row["s"], row["n"]).to_dict(), row["i"])


def test_benjamini_hochberg_matches_rust():
    for row in GOLDEN["benjamini_hochberg"]:
        assert benjamini_hochberg(row["p"], row["alpha"]) == row["accepted"]


@pytest.mark.parametrize("name", sorted(GOLDEN["scenarios"]))
def test_scenarios_match_rust(name):
    scenario = GOLDEN["scenarios"][name]
    c = controller(ControllerConfig.from_dict(scenario["config"]))
    for step in scenario["steps"]:
        decision = c.observe(ExamObservation.from_dict(step["observation"]))
        close(decision.to_dict(), step["decision"])
        plan = c.next_plan(Tier.FRONTIER, CheckpointRef.from_dict(step["observation"]["checkpoint"]), 24)
        assert plan.to_dict() == step["next_frontier_plan"]
    state = c.to_dict()
    state.pop("seed_namespace")
    close(state, scenario["final_state"])


def test_state_round_trip_replays_the_same_plan():
    c = controller()
    c.observe(obs(10, Tier.FRONTIER, "vision.text.line", 390))
    restored = AdaptiveExamController.from_dict(json.loads(json.dumps(c.to_dict())))
    assert restored.next_plan(Tier.FRONTIER, ck(20), 200) == c.next_plan(Tier.FRONTIER, ck(20), 200)


def test_legacy_catastrophic_alias_is_readable():
    data = ControllerConfig().to_dict()
    data["catastrophic_precursor_limit"] = data.pop("catastrophic_failure_limit")
    assert ControllerConfig.from_dict(data).catastrophic_failure_limit == 0.05


# --- documented behaviour (ports of the Rust unit tests) ------------------------------------------


def test_invalid_configuration_and_duplicate_families_fail_closed():
    with pytest.raises(ControllerConfigError):
        controller(ControllerConfig(min_samples_per_window=0))
    with pytest.raises(ControllerConfigError):
        AdaptiveExamController("g", "r", 1, ControllerConfig(), [CapabilitySpec("a", 0, 5, 0)] * 2)
    with pytest.raises(ControllerConfigError):
        AdaptiveExamController("g", "r", 1, ControllerConfig(), [])
    with pytest.raises(ControllerConfigError):
        AdaptiveExamController("g", "r", 1, ControllerConfig(), [CapabilitySpec("a", 3, 1, 2)])


def test_passing_and_catastrophic_counts_cannot_overlap():
    with pytest.raises(InvalidCountsError):
        controller().observe(obs(10, Tier.FRONTIER, "vision.text.line", 390, 20, cat=20))
    with pytest.raises(UnknownFamilyError):
        controller().observe(obs(10, Tier.FRONTIER, "nope", 1))


def test_under_sampled_window_does_not_change_trend_state():
    c = controller()
    before = c.families["vision.text.line"]
    priority, difficulty = before.priority, before.difficulty
    d = c.observe(obs(10, Tier.FRONTIER, "vision.text.line", 0, 20, n=20))
    assert not d.statistically_eligible and d.warning is WarningLevel.WATCH
    after = c.families["vision.text.line"]
    history = after.tier_history[Tier.FRONTIER]
    assert (after.priority, after.difficulty) == (priority, difficulty)
    assert history.ewma_pass_rate is None and history.last_pass_interval is None
    assert history.steer_streak == history.stop_streak == 0


def test_stationary_frontier_noise_does_not_false_steer():
    c = controller()
    for i, p in enumerate([220, 224, 218, 226, 221, 223]):
        assert c.observe(obs(i * 10, Tier.FRONTIER, "reason.chain.depth", p, 10)).warning.rank < WarningLevel.STEER.rank


def test_precursor_is_shadow_only_until_enabled():
    c = controller()
    for step in (10, 20, 30):
        d = c.observe(obs(step, Tier.FRONTIER, "vision.text.line", 360, 120))
        assert d.warning is WarningLevel.WATCH and not d.precursor_steering_enabled


def test_sentinel_identity_is_checkpoint_invariant_and_survives_difficulty_changes():
    c = controller()
    first = c.plan_for_family(Tier.SENTINEL, ck(100), "vision.text.line", 200)
    assert c.observe(obs(150, Tier.FRONTIER, "vision.text.line", 390)).recommended_difficulty == 3
    second = c.plan_for_family(Tier.SENTINEL, ck(200), "vision.text.line", 200)
    assert (first.seed, first.fingerprint(), first.difficulty) == (second.seed, second.fingerprint(), 2)
    r1 = c.plan_for_family(Tier.ROLLING_PROBE, ck(100), "vision.text.line", 200)
    r2 = c.plan_for_family(Tier.ROLLING_PROBE, ck(200), "vision.text.line", 200)
    assert r1.fingerprint() != r2.fingerprint()


def test_tiers_do_not_share_trend():
    c = controller()
    c.observe(obs(10, Tier.FRONTIER, "vision.text.line", 220))
    c.observe(obs(20, Tier.SENTINEL, "vision.text.line", 390))
    d = c.observe(obs(30, Tier.FRONTIER, "vision.text.line", 220))
    assert d.warning is WarningLevel.GREEN and not d.reasons
    assert c.families["vision.text.line"].tier_history[Tier.FRONTIER].observations == 2


def test_siblings_only_from_frontier_and_steer_share():
    c = controller()
    frontier = c.next_plan(Tier.FRONTIER, ck(10), 200)
    sibling = derive_training_sibling(frontier, 50)
    assert sibling.seed != frontier.seed and sibling.source_exam_fingerprint == frontier.fingerprint()
    for tier in (Tier.SENTINEL, Tier.ROLLING_PROBE, Tier.AUDIT):
        with pytest.raises(ProtectedTierError):
            derive_training_sibling(c.next_plan(tier, ck(10), 200), 50)
    assert steer_budget(1000) == 50 and steer_budget(19) == 0


# --- Lilith additions ------------------------------------------------------------------------------


def test_lilith_config_stops_on_catastrophic_above_one_percent():
    # 12/400 catastrophic: Wilson lower ~0.017, above 0.01 but below the BabyLLM 0.05.
    lilith, babyllm = controller(ControllerConfig.lilith()), controller()
    for step in (10, 20):
        a = lilith.observe(obs(step, Tier.ROLLING_PROBE, "vision.text.line", 380, 0, cat=12))
        b = babyllm.observe(obs(step, Tier.ROLLING_PROBE, "vision.text.line", 380, 0, cat=12))
    assert a.warning is WarningLevel.STOP and "catastrophic hard-failure rate is significant" in a.reasons
    assert b.warning is WarningLevel.GREEN
    assert ControllerConfig.lilith().sentinel_regression_budget == 0.03


def test_stop_is_never_cleared_silently():
    # The Rust controller returns green on a clean window after a stop; ExamRun latches it.
    run = ExamRun(controller(), "2026-09-28T10:00:00Z")
    for step, p in ((0, 380), (10, 300)):
        run.observe(obs(step, Tier.SENTINEL, "reason.chain.depth", p))
    stop = run.observe(obs(20, Tier.SENTINEL, "reason.chain.depth", 295))
    assert stop.warning is WarningLevel.STOP and run.stopped
    with pytest.raises(StoppedSegmentError) as err:
        run.observe(obs(30, Tier.SENTINEL, "reason.chain.depth", 379))
    assert "lilith::exam::stopped" in err.value.render()
    with pytest.raises(StoppedSegmentError):
        run.resume(operator="toasty", reason="  ", at="2026-09-28T11:00:00Z")

    restored = ExamRun.from_dict(json.loads(json.dumps(run.to_dict())))
    assert restored.stopped  # the latch survives a restart of the supervisor
    segment = restored.resume(operator="toasty", reason="sentinel drift traced to a template change; reverted",
                              at="2026-09-28T11:00:00Z")
    assert segment.index == 1 and segment.resumed_from_stops[0].family == "reason.chain.depth"
    state = restored.controller.families["reason.chain.depth"]
    assert state.sentinel_baseline is not None  # budget stays cumulative from the first baseline
    assert all(h.stop_streak == 0 for h in state.tier_history.values())
    assert restored.observe(obs(30, Tier.SENTINEL, "reason.chain.depth", 379)).warning is WarningLevel.GREEN
    with pytest.raises(StoppedSegmentError):
        restored.resume(operator="toasty", reason="again", at="x")  # nothing to resume


def test_catastrophic_item_never_passes():
    item = ItemOutcome("i1", "chase", True, frozenset({Catastrophic.WRONG_RECIPIENT}))
    assert not item.passed
    assert ItemOutcome.from_dict(item.to_dict()) == item


def outcomes(family, n, passed, cat=0, pre=0, metrics=None):
    out = []
    for i in range(n):
        kinds = frozenset({Catastrophic.FABRICATED_STATUS}) if i < cat else frozenset()
        precursor = frozenset({Precursor.MISCOPIED_VALUE}) if i < pre else frozenset()
        out.append(ItemOutcome(f"{family}-{i}", family, i < passed, kinds, precursor,
                               {k: v(i) for k, v in (metrics or {}).items()}))
    return out


def test_family_matrix_blockers_bh_and_missing_evidence():
    gates = {
        "tool:memory_write": FamilyGate("tool:memory_write", "A.tool-matrix", 0.85, 0.01),
        "tool:reactor_manage": FamilyGate("tool:reactor_manage", "A.tool-matrix", 0.85, 0.01),
        "park-triage": FamilyGate("park-triage", "B", None, 0.0, (MetricFloor("B", "accuracy", 0.80),)),
        "commitment": FamilyGate("commitment", "B", None, 0.0, (MetricFloor("B", "on_time", 0.90),)),
    }
    items = (outcomes("tool:memory_write", 120, 119)  # clearly above 0.85
             + outcomes("tool:reactor_manage", 120, 104)  # 0.867: above the point floor, not confirmed
             + outcomes("park-triage", 120, 110, cat=1, metrics={"accuracy": lambda i: 1.0}))
    matrix = family_matrix(items, gates, sealed=True)
    rows = matrix.rows
    assert rows["tool:memory_write"].verdict is Verdict.PASS
    assert rows["tool:reactor_manage"].verdict is Verdict.INSUFFICIENT
    assert any("not confirmed after BH" in m for m in rows["tool:reactor_manage"].missing)
    assert rows["park-triage"].verdict is Verdict.FAIL  # one fabricated status is a blocker
    assert "fabricated-status" in rows["park-triage"].blockers[0]
    assert rows["commitment"].verdict is Verdict.INSUFFICIENT and "no scored items" in rows["commitment"].missing
    assert not matrix.release_ready and set(matrix.blockers) == {"park-triage"}
    point = family_matrix(items, gates, rule=StatisticalRule(release_rule="point"))
    assert point.rows["tool:reactor_manage"].verdict is Verdict.PASS
    small = family_matrix(outcomes("tool:memory_write", 60, 60), gates, sealed=True, families=["tool:memory_write"])
    assert any("< 100 per core family" in m for m in small.rows["tool:memory_write"].missing)
    json.dumps(matrix.to_dict())  # serializable


def test_binomial_tail_and_paired_difference():
    assert binomial_upper_tail(0, 10, 0.5) == 1.0
    assert math.isclose(binomial_upper_tail(10, 10, 0.5), 0.5 ** 10)
    assert math.isclose(binomial_upper_tail(9, 10, 0.5), 11 / 1024)
    better = paired_difference([(True, False)] * 30 + [(True, True)] * 60 + [(False, False)] * 10)
    assert better.better and better.verdict == "better"
    same = paired_difference([(True, True)] * 90 + [(False, False)] * 10)
    assert same.equivalent is None and same.verdict == "no significant difference"
    equal = paired_difference([(True, True)] * 190 + [(False, False)] * 10, margin=0.05)
    assert equal.equivalent and equal.verdict == "equivalent"
    assert paired_difference([]).verdict == "missing"
