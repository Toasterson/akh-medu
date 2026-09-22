#!/usr/bin/env python3
"""Compatibility boundary from ADAM/Kaine rewards to Lilith's KLPO runner."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import adam_kaine_reward as reward

canonical = reward.canonical
digest = reward.digest


@dataclass(frozen=True)
class Task:
    id: str
    family: str
    split: str
    prompt: str
    reference: str
    cases: tuple
    reward_task: reward.Task


def tasks() -> tuple[Task, ...]:
    rows = []
    for task in reward.tasks():
        split = "train" if task.split == "development" else "development_holdout"
        prompt = (
            f"You are acting in the {task.family} environment. Observation: "
            f"{json.dumps(task.observation, sort_keys=True)}. Return only a JSON array with 1..{task.budget} "
            "objects, each having exactly one key named action. Allowed actions: observe, instrument, "
            "hypothesize, delegate, fix, report. Verifier receipts are supplied by the environment."
        )
        reference = json.dumps(reward.corrected_sibling(task), separators=(",", ":"))
        rows.append(Task(task.id, task.family, split, prompt, reference,
                         ((reward.digest(task.observation), task.budget),), task))
    return tuple(rows)


def verify(text: str, task: Task) -> dict:
    result = reward.score_model_output(task.reward_task, text)
    return {
        "passed": result["passed"],
        "category": result["category"],
        "counterexample": None if result["passed"] else {"reason": result.get("reasons", [result["category"]])},
        "cases": 1,
        "total_cases": 1,
        "reward_vector": result.get("reward"),
        "klpo_reward": result["klpo_reward"],
        "generated_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def export(path: Path) -> dict:
    source = Path(__file__)
    rows = []
    for task in tasks():
        evidence = verify(task.reference, task)
        if not evidence["passed"]:
            raise AssertionError(task.id)
        rows.append({"id": task.id, "family": task.family, "split": task.split,
                     "prompt": task.prompt, "prompt_sha256": hashlib.sha256(task.prompt.encode()).hexdigest(),
                     "cases_sha256": digest(canonical(task.cases)), "reference_receipt": evidence})
    manifest = {"schema": "lilith/rl-tasks@1", "tasks": rows,
                "verifier_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "reward_manifest_sha256": "a491b9bd86fd52b1f4ca5dd756e218bd31e9b6e4ff5baae5c9f7961d917a2734"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.output), sort_keys=True))
