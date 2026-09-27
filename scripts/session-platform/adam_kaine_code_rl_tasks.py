#!/usr/bin/env python3
"""Frozen real-effect coding tasks in the shape expected by lilith_rl_probe."""

import hashlib
import json
from pathlib import Path

import adam_kaine_code_environment as environment

canonical = environment.canonical
digest = environment.digest


def verifier_components():
    interpreter = Path(environment.coding.__file__).with_name("lilith_rl_tasks.py")
    return {
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment_sha256": hashlib.sha256(Path(environment.__file__).read_bytes()).hexdigest(),
        "task_source_sha256": hashlib.sha256(Path(environment.coding.__file__).read_bytes()).hexdigest(),
        "interpreter_sha256": hashlib.sha256(interpreter.read_bytes()).hexdigest(),
    }


def verifier_identity():
    return hashlib.sha256(canonical(verifier_components())).hexdigest()


def tasks():
    return environment.tasks()


def verify(text, task):
    result = environment.score(text, task)
    evidence = dict(result["evidence"])
    evidence["reward_vector"] = result["reward"]
    evidence["klpo_reward"] = result["klpo_reward"]
    evidence["episode_steps"] = result["episode_steps"]
    return evidence


def export(path: Path):
    rows = []
    for task in tasks():
        receipt = verify(task.reference, task)
        if not receipt["passed"]:
            raise AssertionError(task.id)
        rows.append({"id": task.id, "family": task.family, "split": task.split,
                     "prompt": task.prompt, "prompt_sha256": hashlib.sha256(task.prompt.encode()).hexdigest(),
                     "cases_sha256": digest(task.cases), "reference_receipt": receipt})
    components = verifier_components()
    manifest = {"schema": "lilith/rl-tasks@2", "tasks": rows,
                "verifier_sha256": verifier_identity(), "verifier_components": components,
                "environment_manifest_sha256": "e0e80b64281322320f784d435ceccdf365c69b32fc036e4bef21cc51ca956fce"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.output), sort_keys=True))
