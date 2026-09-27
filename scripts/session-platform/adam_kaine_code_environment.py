#!/usr/bin/env python3
"""Real coding-effect reward environment for transferring ADAM/Kaine mechanisms."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import itertools
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import lilith_rl_tasks_v2 as coding

SCHEMA = "lilith/adam-kaine-code-reward@1"
POLICY = "verified_code_effect_v1"


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value: Any) -> str:
    return digest_bytes(canonical(value))


def _task(id, family, split, spec, reference, negative, inputs, oracle):
    return coding.Task(id, family, split, spec + " " + coding.GRAMMAR, reference, negative,
                       tuple((a, b, oracle(a, b)) for a, b in inputs))


def tasks():
    base = tuple(replace(task, split="development_holdout") if task.split != "train" else task
                 for task in coding.tasks())
    edge = (-1025, -257, -256, -129, -128, -65, -2, -1, 0, 1, 2, 63, 64, 127, 128, 255, 256, 1024)
    independent = (
        _task("saturating-sub/3", "signed_arithmetic", "independent_eval",
              "Inputs are arbitrary integers. If either is outside -128..127 return -1; otherwise return a-b clamped to -128..127.",
              "def f(a, b):\n    if not (-128 <= a <= 127 and -128 <= b <= 127):\n        return -1\n    if a - b < -128:\n        return -128\n    if a - b > 127:\n        return 127\n    return a - b\n",
              "def f(a, b):\n    return a - b\n", itertools.product(edge, repeat=2),
              lambda a,b: -1 if not(-128<=a<=127 and -128<=b<=127) else max(-128,min(127,a-b))),
        _task("power-two-in-range/3", "bit_properties", "independent_eval",
              "Return bool indicating that a is a power of two from 1 through 1024 inclusive and b equals the base-2 exponent; inputs are arbitrary integers.",
              "def f(a, b):\n    return 1 <= a <= 1024 and a & (a - 1) == 0 and 0 <= b <= 10 and a == 1 << b\n",
              "def f(a, b):\n    return a & (a - 1) == 0\n", itertools.product(edge+(512,), range(-1,12)),
              lambda a,b: 1<=a<=1024 and a&(a-1)==0 and 0<=b<=10 and a==1<<b),
        _task("unsigned-field-at-offset/3", "packed_offsets", "independent_eval",
              "a is arbitrary and b packs width in the low byte and bit offset in the next byte. Return -1 unless 0<=a<=65535, width is 1..8, offset is 0..15, and width+offset<=16; otherwise return that unsigned field.",
              "def f(a, b):\n    if not (0 <= a <= 65535 and 1 <= (b & 255) <= 8 and 0 <= (b >> 8) <= 15 and (b & 255) + (b >> 8) <= 16):\n        return -1\n    return (a >> (b >> 8)) & ((1 << (b & 255)) - 1)\n",
              "def f(a, b):\n    return a >> (b >> 8)\n",
              itertools.product(edge+(4095,32768,65535,65536), (0,1,8,9,256,257,264,520,1032,2056,3841,3842)),
              lambda a,b: -1 if not(0<=a<=65535 and 1<=b&255<=8 and 0<=b>>8<=15 and (b&255)+(b>>8)<=16) else (a>>(b>>8))&((1<<(b&255))-1)),
        _task("intervals-adjacent/3", "interval_adjacency", "independent_eval",
              "a and b pack half-open byte intervals: high byte is start and low byte is length. Return bool when both lengths are positive and the intervals touch at exactly one boundary without overlapping.",
              "def f(a, b):\n    return a & 255 > 0 and b & 255 > 0 and ((a >> 8) + (a & 255) == b >> 8 or (b >> 8) + (b & 255) == a >> 8)\n",
              "def f(a, b):\n    return (a >> 8) + (a & 255) <= b >> 8\n",
              itertools.product((0,1,255,256,257,258,511,512,513,514,1025,1281,65535), repeat=2),
              lambda a,b: a&255>0 and b&255>0 and ((a>>8)+(a&255)==b>>8 or (b>>8)+(b&255)==a>>8)),
    )
    return base + independent


def score(source: str, task: coding.Task) -> dict[str, Any]:
    """Reward only a real verifier effect; receipts are built outside output."""
    evidence = coding.verify(source, task)
    artifact, presentation = coding.extract_artifact(source)
    grammar_valid = evidence["category"] not in {
        "syntax", "grammar", "source_too_large", "ast_too_large", "missing_f", "forbidden_syntax"
    }
    steps = [
        {"action": "observe", "receipt_sha256": digest({"prompt": task.prompt, "cases": task.cases})},
        {"action": "patch", "receipt_sha256": digest_bytes(artifact.encode())},
        {"action": "test", "receipt_sha256": digest(evidence), "passed": evidence["passed"]},
        {"action": "report", "receipt_sha256": digest({"category": evidence["category"], "cases": evidence["cases"]})},
    ]
    failed_case = 1 if evidence["counterexample"] is not None else 0
    correct_cases = evidence["total_cases"] if evidence["passed"] else max(0, evidence["cases"] - failed_case)
    semantic_fraction = correct_cases / evidence["total_cases"]
    vector = {
        "effect": semantic_fraction,
        "validity": float(grammar_valid),
        "provenance": 1.0,
        "disposition": float(evidence["cases"] > 0),
        "budget": float(len(source.encode()) <= 4096),
    }
    passed = evidence["passed"]
    # Online reward uses only observed semantic correctness and grammar admission.
    # The other vector fields remain audit dimensions, not scalar reward padding.
    klpo_reward = 0.95 * semantic_fraction + 0.05 * float(grammar_valid)
    return {
        "schema": SCHEMA, "policy": POLICY, "task_id": task.id, "family": task.family,
        "split": task.split, "reward": vector, "klpo_reward": klpo_reward, "passed": passed,
        "evidence": evidence, "episode_steps": steps, "presentation": presentation,
        "generated_text_sha256": digest_bytes(source.encode()),
    }


def native_audit(source: str, task: coding.Task, executor: Any, repository: Path) -> dict[str, Any]:
    reward = score(source, task)
    if not reward["passed"]:
        return {"executed": False, "reason": "bounded verifier did not admit a passing artifact", "reward": reward}
    artifact, _ = coding.extract_artifact(source)
    harness = artifact + "\n\n_cases = " + repr(task.cases) + "\n" + (
        "for _a,_b,_expected in _cases:\n"
        "    _actual=f(_a,_b)\n"
        "    if type(_actual) is not type(_expected) or _actual != _expected:\n"
        "        raise AssertionError((_a,_b,_expected,_actual))\n"
        "print('native-cases-ok',len(_cases))\n"
    )
    receipt = executor.execute(repository, "candidate.py", harness, ["python3", "candidate.py"], timeout_s=30)
    return {"executed": True, "reward": reward, "native_receipt": receipt,
            "passed": receipt["passed"], "artifact_sha256": digest_bytes(artifact.encode())}


def export(directory: Path) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    task_rows = tasks()
    rows = []
    for task in task_rows:
        result = score(task.reference, task)
        negative = score(task.negative_control, task)
        if not result["passed"] or negative["passed"]:
            raise AssertionError(task.id)
        rows.append({"id": task.id, "family": task.family, "split": task.split,
                     "prompt": task.prompt, "prompt_sha256": digest_bytes(task.prompt.encode()),
                     "cases_sha256": digest(task.cases), "reference_sha256": digest_bytes(task.reference.encode()),
                     "reference_reward": result, "negative_reward": negative})
    manifest = {
        "schema": "lilith/adam-kaine-code-curriculum@1", "policy": POLICY,
        "task_source_sha256": digest_bytes(Path(coding.__file__).read_bytes()),
        "reward_source_sha256": digest_bytes(Path(__file__).read_bytes()), "tasks": rows,
        "training_ids": [row["id"] for row in rows if row["split"] == "train"],
        "development_ids": [row["id"] for row in rows if row["split"] == "development_holdout"],
        "sealed_eval_ids": [row["id"] for row in rows if row["split"] == "independent_eval"],
        "limitations": ["finite bounded cases are not proofs", "native illumos is an audit after bounded admission",
                        "transfers mechanisms into Gemma; does not train sovereign ADAM/Kaine weights"],
    }
    manifest["manifest_sha256"] = digest(manifest)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.output), sort_keys=True))
