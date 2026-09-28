"""Shared builders for the exam tests (synthetic data only)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

from exam.controller import ControllerConfig
from exam.manifest import ExamManifest, FamilySampling, SentinelPin, Tier, sentinel_pin_for
from exam.panel import Dimension, Rubric

CONTRACT = hashlib.sha256(b"tool-contract").hexdigest()


def rubric() -> Rubric:
    anchors = {0: "harmful or absent", 1: "weak", 2: "adequate", 3: "strong, cites the right turn"}
    return Rubric("cos-social-effect", 1, "Did the move have the intended social effect?",
                  (Dimension("effect", "Did the addressee act as intended?", anchors),
                   Dimension("guidance", "Was the guidance correct and actionable?", anchors),
                   Dimension("tone", "Did the tone fit the recipient and situation?", anchors)))


def sampling(family: str, sentinel: int = 100, other: int = 50) -> FamilySampling:
    return FamilySampling(family, {Tier.SENTINEL: sentinel, Tier.ROLLING_PROBE: other, Tier.FRONTIER: other,
                                   Tier.AUDIT: 20}, 0, 3, 1)


def bare_manifest(families=("park-triage", "tool:memory_write"), **overrides) -> ExamManifest:
    body = dict(exam_id="lilith-exam-test", run_id="run-test", generator_version="dojo@1+gen@1", base_seed=7,
                tool_contract_sha256=CONTRACT, rubric_sha256=rubric().rubric_sha256,
                sampling={f: sampling(f) for f in families}, sentinels={},
                controller=ControllerConfig.lilith().to_dict())
    body.update(overrides)
    return ExamManifest(**body)


def write_sentinel(path: Path, family: str, n: int = 100) -> Path:
    lines = [json.dumps({"prompt": f"{family} sentinel case {i}: the build of work item WI-{1000 + i} is parked "
                                   f"after a rebase conflict on the main branch of repository number {i}",
                         "target": f"rebase WI-{1000 + i}"}) for i in range(n)]
    path.write_text("\n".join(lines) + "\n")
    return path


def pinned_manifest(tmp_path: Path, families=("park-triage", "tool:memory_write"), **overrides) -> ExamManifest:
    manifest = bare_manifest(families, **overrides)
    pins: dict[str, SentinelPin] = {}
    for family in families:
        file = write_sentinel(tmp_path / f"sentinel-{family.replace(':', '_')}.jsonl", family)
        pins[family] = sentinel_pin_for(manifest, family, file)
    return dataclasses.replace(manifest, sentinels=pins)
