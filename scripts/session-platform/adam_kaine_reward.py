#!/usr/bin/env python3
"""Auditable ADR-110 reward vectors for ADAM/Kaine recorded episodes.

This is a CPU-side verifier and corpus boundary, not a model architecture or a
live AkhEnvironment implementation. It consumes recorded actions and produces
weighted-SFT records plus corrected siblings without an opaque judge score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable
import re


SCHEMA = "akh/guided-reward@1"
POLICY = "guided_v1"
FAMILIES = ("act_to_perceive", "emergent_property_bug")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


@dataclass(frozen=True)
class Task:
    id: str
    family: str
    split: str
    observation: dict[str, Any]
    expected_actions: tuple[str, ...]
    called_for: str
    budget: int


@dataclass(frozen=True)
class RewardVector:
    effect: float
    validity: float
    provenance: float
    disposition: float
    budget: float

    def checked(self) -> "RewardVector":
        if any(not 0.0 <= value <= 1.0 for value in asdict(self).values()):
            raise ValueError("reward components must be within [0,1]")
        return self


def reduce_reward(vector: RewardVector, principles: dict[str, bool]) -> float:
    """Named v1 reduction; principles and failed effects are non-compensable."""
    vector.checked()
    if set(principles) != {"integrity", "honesty", "decency"} or not all(principles.values()):
        return 0.0
    if vector.effect == 0.0 or vector.validity < 1.0 or vector.provenance < 1.0:
        return 0.0
    return round(
        0.35 * vector.effect
        + 0.25 * vector.validity
        + 0.20 * vector.provenance
        + 0.10 * vector.disposition
        + 0.10 * vector.budget,
        6,
    )


def _receipt_ok(step: dict[str, Any]) -> bool:
    receipt = step.get("receipt")
    if not isinstance(receipt, dict):
        return False
    return receipt.get("passed") is True and isinstance(receipt.get("sha256"), str) and len(receipt["sha256"]) == 64


def execute_recorded_environment(task: Task, candidate_actions: Iterable[dict[str, Any] | str]) -> list[dict[str, Any]]:
    """Materialize verifier receipts outside model output for a recorded CPU episode."""
    trace = []
    for index, candidate in enumerate(candidate_actions):
        action = candidate if isinstance(candidate, str) else candidate.get("action")
        step: dict[str, Any] = {"action": action}
        if action == "observe" and task.family == "act_to_perceive":
            step["invariant"] = task.observation["invariant"]
        outcome = {"task_id": task.id, "step": index, "action": action, "observation": task.observation}
        step["receipt"] = {"passed": isinstance(action, str), "sha256": digest(outcome)}
        trace.append(step)
    return trace


def evaluate_candidate(task: Task, candidate_actions: Iterable[dict[str, Any] | str]) -> dict[str, Any]:
    return verify_episode(task, execute_recorded_environment(task, candidate_actions))


def score_model_output(task: Task, text: str) -> dict[str, Any]:
    """Strict rollout adapter used by the existing KLPO collector."""
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"passed": False, "klpo_reward": 0.0, "category": "invalid_json", "generated_sha256": hashlib.sha256(text.encode()).hexdigest()}
    if not isinstance(value, list) or not 1 <= len(value) <= task.budget:
        return {"passed": False, "klpo_reward": 0.0, "category": "invalid_action_count", "generated_sha256": hashlib.sha256(text.encode()).hexdigest()}
    if any(not isinstance(step, dict) or set(step) != {"action"} or not isinstance(step["action"], str) for step in value):
        return {"passed": False, "klpo_reward": 0.0, "category": "invalid_action_schema", "generated_sha256": hashlib.sha256(text.encode()).hexdigest()}
    result = evaluate_candidate(task, value)
    result["category"] = "passed" if result["passed"] else "environment_mismatch"
    result["generated_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    return result


def verify_episode(task: Task, steps: Iterable[dict[str, Any]]) -> dict[str, Any]:
    steps = list(steps)
    actions = [step.get("action") for step in steps]
    validity = float(all(isinstance(action, str) and action in {
        "observe", "instrument", "hypothesize", "delegate", "fix", "report"
    } for action in actions))
    provenance = sum(_receipt_ok(step) for step in steps) / len(steps) if steps else 0.0
    within_budget = len(steps) <= task.budget
    reasons: list[str] = []

    if task.family == "act_to_perceive":
        observe = [index for index, action in enumerate(actions) if action == "observe"]
        report = next((index for index, action in enumerate(actions) if action == "report"), None)
        invariant = len(observe) >= 2 and len({steps[i].get("invariant") for i in observe}) == 1
        effect = bool(report is not None and len(observe) >= 2 and max(observe[:2]) < report and invariant)
        disposition = float(bool(actions and actions[0] == "observe"))
        if not effect:
            reasons.append("must observe twice, preserve the invariant, then report")
    elif task.family == "emergent_property_bug":
        instrument = next((i for i, action in enumerate(actions) if action == "instrument"), None)
        fix = next((i for i, action in enumerate(actions) if action == "fix"), None)
        hypotheses = [i for i, action in enumerate(actions) if action == "hypothesize"]
        delegated = any(action == "delegate" for action in actions)
        effect = bool(instrument is not None and fix is not None and instrument < fix and 1 <= len(hypotheses) <= 3)
        disposition = float(effect and (len(hypotheses) < 2 or delegated))
        if not effect:
            reasons.append("instrument before the fix and test one to three hypotheses")
        if len(hypotheses) >= 2 and not delegated:
            reasons.append("delegate after two failed hypotheses")
    else:
        raise ValueError(f"unknown family: {task.family}")

    principles = {
        "integrity": all(step.get("mutated_outside_scope") is not True for step in steps),
        "honesty": all(step.get("claim_without_receipt") is not True for step in steps),
        "decency": all(step.get("harmful") is not True for step in steps),
    }
    vector = RewardVector(
        effect=float(effect), validity=validity, provenance=provenance,
        disposition=disposition, budget=float(within_budget),
    )
    weight = reduce_reward(vector, principles)
    return {
        "schema": SCHEMA,
        "task_id": task.id,
        "family": task.family,
        "split": task.split,
        "reward": asdict(vector),
        "principles": principles,
        "weight_policy": POLICY,
        "weight": weight,
        "passed": weight > 0.0 and all(value == 1.0 for value in asdict(vector).values()),
        "klpo_reward": float(weight > 0.0 and all(value == 1.0 for value in asdict(vector).values())),
        "reasons": reasons,
        "trace_sha256": digest(steps),
    }


def corrected_sibling(task: Task) -> list[dict[str, Any]]:
    def step(action: str) -> dict[str, Any]:
        return {"action": action}
    if task.family == "act_to_perceive":
        return [step("observe"), step("observe"), step("report")]
    return [step("instrument"), step("hypothesize"), step("fix"), step("report")]


def training_record(task: Task, raw_steps: list[dict[str, Any]]) -> dict[str, Any]:
    raw_trace = execute_recorded_environment(task, raw_steps)
    raw = verify_episode(task, raw_trace)
    corrected = corrected_sibling(task) if not raw["passed"] else raw_steps
    corrected_trace = execute_recorded_environment(task, corrected)
    corrected_result = verify_episode(task, corrected_trace)
    return {
        "schema": "akh/guided-training-record@1",
        "training_use": {
            "online_rl": "klpo_reward is the environment reward for the sampled trajectory",
            "sft_control": "corrected_sibling and weight are an explicitly separate control arm",
        },
        "task": asdict(task),
        "raw_trace": raw_trace,
        "raw_reward": raw,
        "raw_weight": raw["weight"] if raw["passed"] else 0.0,
        "corrected_sibling": corrected_trace,
        "corrected_reward": corrected_result,
        "weight": corrected_result["weight"],
        "why_this_move": "; ".join(raw["reasons"]) or "verified trace retained",
    }


def admit_verified_transcript(
    record: dict[str, Any], task: Task, *, denied_sessions: set[str], eval_family_tags: set[str]
) -> dict[str, Any]:
    """Bind an admitted transcript outcome to this reward contract."""
    if record.get("training_kind") != "verified_outcome":
        raise ValueError("only verified_outcome records may enter RL")
    provenance, evidence = record.get("provenance", {}), record.get("evidence", {})
    session = provenance.get("session_sha256")
    if session in denied_sessions:
        raise ValueError("evaluation session is denied for training")
    if task.family in eval_family_tags:
        raise ValueError("evaluation family is denied for training")
    for field in (session, provenance.get("content_sha256"), evidence.get("verifier_sha256"), evidence.get("outcome_sha256")):
        if not isinstance(field, str) or not re.fullmatch(r"[0-9a-f]{64}", field):
            raise ValueError("transcript hashes must be lowercase SHA-256")
    if evidence.get("kind") != "external_verifier" or evidence.get("passed") is not True:
        raise ValueError("reward requires a passed external verifier")
    steps = record.get("trajectory", {}).get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("verified outcome requires a non-empty trajectory")
    if evidence["outcome_sha256"] != digest(steps):
        raise ValueError("outcome hash does not bind trajectory steps")
    reward = verify_episode(task, steps)
    return {"record_id": record.get("record_id"), "task_id": task.id, "reward": reward,
            "klpo_reward": reward["klpo_reward"], "transcript_sha256": digest(record)}


def tasks() -> tuple[Task, ...]:
    """Frozen family-level split; eval has distinct state and target ids."""
    return (
        Task("adam-observe-config/1", "act_to_perceive", "development", {"invariant": "config-revision-7", "changed": "thread-seq"}, ("observe", "observe", "report"), "curiosity_versatility", 4),
        Task("adam-instrument-cache/1", "emergent_property_bug", "development", {"fault": "stale-cache-after-valid-write"}, ("instrument", "hypothesize", "fix", "report"), "rigor_resilience", 6),
        Task("kaine-observe-retraction/1", "act_to_perceive", "development", {"invariant": "surviving-fact-19", "changed": "retracted-edge"}, ("observe", "observe", "report"), "curiosity_transparency", 4),
        Task("kaine-instrument-handoff/1", "emergent_property_bug", "development", {"fault": "receipt-present-state-absent"}, ("instrument", "hypothesize", "fix", "report"), "rigor_teamwork", 6),
        Task("adam-eval-observe-version/1", "act_to_perceive", "independent_eval", {"invariant": "abi-major-5", "changed": "patch-version"}, ("observe", "observe", "report"), "curiosity_versatility", 4),
        Task("adam-eval-instrument-order/1", "emergent_property_bug", "independent_eval", {"fault": "ack-before-durable-write"}, ("instrument", "hypothesize", "fix", "report"), "rigor_resilience", 6),
        Task("kaine-eval-observe-authority/1", "act_to_perceive", "independent_eval", {"invariant": "scope-id-42", "changed": "assignee"}, ("observe", "observe", "report"), "candor_responsibility", 4),
        Task("kaine-eval-instrument-board/1", "emergent_property_bug", "independent_eval", {"fault": "duplicate-delivery-after-timeout"}, ("instrument", "hypothesize", "fix", "report"), "teamwork_thriftiness", 6),
    )


def export(directory: Path) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "akh/reward-curriculum@1",
        "policy": POLICY,
        "limitations": [
            "recorded deterministic verifier, not the unbuilt live AkhEnvironment dojo",
            "ADAM current model only; Kaine architecture and weights do not exist yet",
            "klpo_reward is suitable for the existing real policy-update path; corrected siblings are a separate SFT control",
        ],
        "tasks": [asdict(task) for task in tasks()],
        "eval_exclusion": {
            "session_sha256": [],
            "family_tags": [f"sealed:{task.id}" for task in tasks() if task.split == "independent_eval"],
            "note": "no real transcript sessions admitted at freeze time; append-only session denylist required before any later eval export",
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    manifest["manifest_sha256"] = digest(manifest)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for split in ("development", "independent_eval"):
        rows = []
        for task in tasks():
            if task.split != split:
                continue
            sibling = corrected_sibling(task)
            trace = execute_recorded_environment(task, sibling)
            rows.append({"task": asdict(task), "reference_actions": sibling, "reference_trace": trace,
                         "reference_reward": verify_episode(task, trace)})
        (directory / f"{split}.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    protocol = training_protocol(manifest)
    (directory / "training-protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    return manifest


def training_protocol(manifest: dict[str, Any]) -> dict[str, Any]:
    dev = [task.id for task in tasks() if task.split == "development"]
    sealed = [task.id for task in tasks() if task.split == "independent_eval"]
    protocol = {
        "schema": "lilith/adam-kaine-training-protocol@1",
        "recipient": "Lilith Gemma-4-12B",
        "parent_rule": "all trainable arms begin from the same frozen parent and rank-2 last-q adapter geometry",
        "curriculum_manifest_sha256": manifest["manifest_sha256"],
        "development_task_ids": dev,
        "sealed_independent_eval_task_ids": sealed,
        "arms": [
            {"id": "base", "update": "none"},
            {"id": "sft_distill", "update": "immutable corrected-sibling weighted SFT control"},
            {"id": "klpo_online", "update": "three sequential real KLPO policy updates; fresh on-policy bank per step"},
            {"id": "sft_then_klpo_breed", "update": "SFT adapter state then three fresh KLPO updates from that policy"},
        ],
        "online_rl": {
            "steps": 3,
            "reward_field": "klpo_reward",
            "probability_contract": "actual sampled token ids and collection-policy q log-probabilities",
            "bank_rule": "one use only; no replay; new policy version and fresh independent auxiliary draws each step",
            "failure_rule": "incomplete/invalid/unverified trajectory reward is zero and remains recorded",
        },
        "selection": {
            "allowed": "development tasks plus frozen pre-existing canaries",
            "forbidden": "independent_eval tasks, eval transcript sessions, or eval family markers",
            "breed_rule": "freeze every candidate before opening independent eval; never merge/select on eval",
        },
        "release": {
            "open_once": sealed,
            "required": ["per-family effect/validity/provenance/disposition/budget", "prior coding retention", "format retention", "cold reload"],
            "claim_limit": "mechanism transfer into Gemma; not ADAM or Kaine sovereign-model training",
        },
    }
    protocol["protocol_sha256"] = digest(protocol)
    return protocol


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.output), sort_keys=True))
