"""WI exit condition: a dry run against a C4 dojo fixture produces a per-family matrix, a
controller decision and a RESULT skeleton (plus a gate report and a panel work order)."""
import json

from dojo import families as F
from dojo.episode import RecipientClass
from dojo.families import BuildContext, verify, verify_context
from exam.certify import gate_report
from exam.controller import (
    AdaptiveExamController,
    CapabilitySpec,
    Catastrophic,
    ControllerConfig,
    ExamRun,
    Verdict,
    WarningLevel,
    classify_episode,
    family_matrix,
    observation_from_outcomes,
)
from exam.manifest import CheckpointRef, Tier
from exam.panel import Panel, PanelItem
from exam.result_doc import render
from exam_helpers import pinned_manifest, rubric
from fixtures import T0, WI, Driver

CHECKPOINT = CheckpointRef(1200, "a" * 64)


def park_episodes(world, tmp_path):
    parked = [WI(f"p{i}", "parked", parked_reason="rebase-conflict") for i in range(6)]
    sid = world.snapshot(T0, parked)
    specs = F.build_park_triage(world.store(), sid, BuildContext()).specs
    assert len(specs) == 6
    out = []
    for n, spec in enumerate(specs):
        wid = spec.subject_ids[0]
        d = Driver(spec, world.store(), tmp_path / f"park-{n}.jsonl")
        _, r = d.call("get_work_item_detail", {"id": wid})
        cause = "conflict" if n < 5 else "ci-red"  # one wrong cause
        d.emit({"contract": "lilith/dojo-park-triage@1", "language": "en", "work_item": wid, "cause": cause,
                "next_action": "rebase", "receipts": [d.receipt(r)]})
        out.append((spec, d.episode()))
    return out


def chase_episode(world, tmp_path, recipient):
    sid = world.snapshot(T0, [WI("b", "working", blockers=(("x", "working"),)), WI("x", "working", executor="imhotep")])
    spec = F.build_chase(world.store(), sid, BuildContext()).specs[0]
    d = Driver(spec, world.store(), tmp_path / f"chase-{recipient}.jsonl")
    _, r = d.call("get_wi_loop_state", {"work_item_id": "b"})
    payload = {"objective": "unblock", "about": ["b", "x"], "receipts": [d.receipt(r)], "ack_deadline": 200}
    d.send(recipient, RecipientClass.EXECUTOR, payload, sent_at=100, follow_up_at=250)
    return spec, d.episode()


def test_dry_run_on_dojo_fixture(world, tmp_path):
    manifest = pinned_manifest(tmp_path, families=("park-triage", "chase"))
    manifest.require_scorable()
    gates = {f: manifest.family_gate(f) for f in manifest.sampling}

    outcomes = []
    for spec, episode in park_episodes(world, tmp_path) + [chase_episode(world, tmp_path, "tecton")]:
        verdict = verify(spec, episode, verify_context(world.store().get(spec.snapshot_id), None))
        outcomes.append(classify_episode(spec, episode, verdict, gates[spec.family.value]))

    park = [o for o in outcomes if o.family == "park-triage"]
    chase = [o for o in outcomes if o.family == "chase"]
    assert sum(o.passed for o in park) == 5
    assert chase[0].catastrophic == {Catastrophic.WRONG_RECIPIENT} and not chase[0].passed

    # controller decision on the rolling probe for this checkpoint (tiny n: a watch, never a steer)
    run = ExamRun(AdaptiveExamController(manifest.generator_version, manifest.run_id, manifest.base_seed,
                                         ControllerConfig.from_dict(manifest.controller),
                                         [CapabilitySpec(f, 0, 3, 1) for f in manifest.sampling],
                                         seed_namespace=manifest.seed_namespace), "2026-09-28T12:00:00Z")
    plan = manifest.plan(Tier.ROLLING_PROBE, "park-triage", CHECKPOINT, n=len(park))
    decision = run.observe(observation_from_outcomes(plan, park))
    assert decision.warning is WarningLevel.WATCH and not decision.statistically_eligible
    assert run.decisions[0]["observation"]["pass_count"] == 5

    # per-family matrix (sealed rules): chase blocks on the catastrophic post; park lacks worlds
    matrix = family_matrix(outcomes, gates, sealed=True, families=manifest.sampling)
    assert matrix.rows["chase"].verdict is Verdict.FAIL
    assert "wrong-recipient" in matrix.rows["chase"].blockers[0]
    park_row = matrix.rows["park-triage"]
    assert park_row.verdict is Verdict.INSUFFICIENT and not park_row.blockers  # 5/6 clears 0.80 on the point
    assert any("< 100 per core family" in m for m in park_row.missing)  # ...but 6 worlds are not evidence
    report = gate_report(manifest.manifest_sha256, [matrix.to_dict()])
    assert not report["release_ready"] and report["blockers"]

    # panel work orders for the social-effect half of the same episodes
    panel = Panel(rubric(), salt="dry-run")
    index = panel.work_orders([PanelItem(o.item_id, "b1", "candidate", {"turns": [{"id": "t1", "text": o.item_id}]})
                               for o in outcomes], tmp_path / "panel")
    assert {o["judge"] for o in index["orders"]} == {"fable", "opus"}

    doc = render({"schema": "lilith/run-result@1", "run_id": manifest.run_id, "title": "Lilith C4b dry run",
                  "verdict": {"status": "NOT certified", "summary": "dry run on a synthetic dojo fixture"},
                  "config": [{"setting": "Exam manifest", "value": manifest.manifest_sha256,
                              "source": "exam manifest"}],
                  "timeline": [{"time": "12:00", "event": f"rolling probe: {decision.warning.value}",
                                "source": "exam-run.json"}],
                  "matrices": {"candidate": matrix.to_dict()}})
    assert "| B | park-triage | accuracy" in doc and "| B | chase | catastrophic" in doc
    assert "wrong-recipient" in doc and doc.index("## Verdict") < doc.index("## Per-family gates")
    json.dumps(run.to_dict())
