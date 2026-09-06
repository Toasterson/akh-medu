#!/usr/bin/env python3
"""Run the first checkpoint-bound adaptive probe, never promotion.

The script waits for `run_tool_native_post_foundation.py` to finish one SFT
epoch, then produces rolling (not sentinel) closed-loop evidence for all seven
knowledge workflows and every captured tool. It updates a serialized adaptive
controller and stops, leaving the next training-mixture decision inspectable.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


KNOWLEDGE_FAMILIES = (
    "knowledge.acquire_textbook",
    "knowledge.recall_after_restart",
    "knowledge.revise_from_new_edition",
    "knowledge.provider_fallback",
    "knowledge.grounded_abstention",
    "knowledge.textbook_inventory",
    "knowledge.note_navigation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--knowledge-workflow-binary", type=Path, required=True)
    parser.add_argument("--tool-exam-generator-binary", type=Path, required=True)
    parser.add_argument("--knowledge-eval-binary", type=Path, required=True)
    parser.add_argument("--tool-eval-binary", type=Path, required=True)
    parser.add_argument("--controller-binary", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--maximum-wait-hours", type=float, default=15.0)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("knowledge-model-small-v1"),
        help="knowledge SFT bundle directory (relative paths are resolved under --base)",
    )
    parser.add_argument(
        "--probe-dir",
        type=Path,
        default=Path("adaptive-probe-sft-epoch-1-v1"),
        help="adaptive probe output directory (relative paths are resolved under --base)",
    )
    parser.add_argument(
        "--post-foundation-json",
        type=Path,
        default=Path("post-foundation-supervisor.json"),
        help="upstream post-foundation status file (relative paths are resolved under --base)",
    )
    parser.add_argument(
        "--status-json",
        type=Path,
        default=Path("adaptive-probe-supervisor.json"),
        help="supervisor status file (relative paths are resolved under --base)",
    )
    parser.add_argument(
        "--knowledge-families",
        type=str,
        default=",".join(KNOWLEDGE_FAMILIES),
        help="comma-separated knowledge workflow families to probe",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="generation budget for probe evaluations (long note-writing targets need >=512)",
    )
    args = parser.parse_args()
    if args.poll_seconds < 5 or args.maximum_wait_hours <= 0:
        parser.error("--poll-seconds must be >= 5 and --maximum-wait-hours positive")
    families = tuple(
        family.strip() for family in args.knowledge_families.split(",") if family.strip()
    )
    if not families:
        parser.error("--knowledge-families must list at least one family")
    args.knowledge_families = families
    return args


def resolve_under(base: Path, value: Path) -> Path:
    return value if value.is_absolute() else base / value


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_logged(command: list[str], log_path: Path) -> None:
    with log_path.open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command exited {completed.returncode}; inspect {log_path}: {command[0]}"
        )


def run_stdout_new(command: list[str], output: Path, error_log: Path) -> None:
    with output.open("x", encoding="utf-8") as stdout, error_log.open(
        "x", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command exited {completed.returncode}; inspect {error_log}: {command[0]}"
        )


def wait_for_candidate(
    upstream_status: Path, poll_seconds: int, maximum_hours: float
) -> dict[str, Any]:
    deadline = time.monotonic() + maximum_hours * 3600.0
    while time.monotonic() < deadline:
        if upstream_status.exists():
            status = load_json(upstream_status)
            stage = status.get("stage")
            if stage == "ready_for_adaptive_probe":
                return status
            if stage == "failed":
                raise RuntimeError("post-foundation supervisor reported failure")
        time.sleep(poll_seconds)
    raise RuntimeError("adaptive probe wait exceeded the configured deadline")


def binary(path: Path) -> str:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise RuntimeError(f"binary is not executable: {resolved}")
    return str(resolved)


def run_probe(
    args: argparse.Namespace,
    base: Path,
    status_path: Path,
    bundle: Path,
    probe: Path,
    post_foundation_json: Path,
    knowledge_families: tuple[str, ...],
) -> None:
    max_new_tokens = int(args.max_new_tokens)
    workflow = binary(args.knowledge_workflow_binary)
    generate_tools = binary(args.tool_exam_generator_binary)
    eval_knowledge = binary(args.knowledge_eval_binary)
    eval_tools = binary(args.tool_eval_binary)
    controller = binary(args.controller_binary)
    upstream = wait_for_candidate(
        post_foundation_json, args.poll_seconds, args.maximum_wait_hours
    )

    candidate_path = bundle / "candidate-model.json"
    candidate = load_json(candidate_path)
    checkpoint_step = int(candidate["checkpoint_step"])
    checkpoint_hash = str(candidate["checkpoint_model_hash"])
    if (
        checkpoint_step != int(upstream["candidate_checkpoint_step"])
        or checkpoint_hash != upstream["candidate_model_hash"]
        or candidate.get("certification") is not None
        or (bundle / "active-model.json").exists()
    ):
        raise RuntimeError("candidate identity changed or was activated before its probe")

    contracts = bundle / "contracts.json"
    probe.mkdir()
    (probe / "knowledge").mkdir()
    (probe / "tools").mkdir()
    atomic_json(
        status_path,
        {
            "schema_version": 1,
            "stage": "evaluating_knowledge_workflows",
            "checkpoint_step": checkpoint_step,
            "checkpoint_model_hash": checkpoint_hash,
        },
    )

    observations: list[dict[str, Any]] = []
    evidence_paths: list[str] = []
    for index, family in enumerate(knowledge_families):
        stem = family.replace(".", "-")
        exam = probe / "knowledge" / f"{stem}.exam.json"
        evidence = probe / "knowledge" / f"{stem}.evidence.json"
        predictions = probe / "knowledge" / f"{stem}.predictions.json"
        run_stdout_new(
            [
                workflow,
                "exam",
                "--contracts",
                str(contracts),
                "--family",
                family,
                "--tier",
                "rolling-probe",
                "--run-id",
                "tool-native-small-sft-epoch-1-v1",
                "--seed",
                str(62_001 + index),
                "--difficulty",
                "1",
                "--samples",
                "20",
                "--checkpoint-step",
                str(checkpoint_step),
                "--checkpoint-hash",
                checkpoint_hash,
            ],
            exam,
            probe / "knowledge" / f"{stem}.generate.log",
        )
        run_logged(
            [
                eval_knowledge,
                "--bundle",
                str(bundle),
                "--exam",
                str(exam),
                "--evidence-out",
                str(evidence),
                "--predictions-out",
                str(predictions),
                "--max-new-tokens",
                str(max_new_tokens),
            ],
            probe / "knowledge" / f"{stem}.evaluate.log",
        )
        item = load_json(evidence)
        observation = item.get("observation")
        if not isinstance(observation, dict):
            raise RuntimeError(f"knowledge evidence lacks observation: {evidence}")
        observations.append(observation)
        evidence_paths.append(str(evidence.relative_to(probe)))

    atomic_json(
        status_path,
        {
            "schema_version": 1,
            "stage": "evaluating_all_tools",
            "checkpoint_step": checkpoint_step,
            "checkpoint_model_hash": checkpoint_hash,
            "knowledge_evidence": evidence_paths,
        },
    )
    tool_exam = probe / "tools" / "all-tools.exam.json"
    tool_evidence = probe / "tools" / "all-tools.evidence.json"
    tool_observations = probe / "tools" / "all-tools.observations.jsonl"
    run_logged(
        [
            generate_tools,
            "--contracts",
            str(contracts),
            "--out",
            str(tool_exam),
            "--run-id",
            "tool-native-small-sft-epoch-1-v1",
            "--checkpoint-step",
            str(checkpoint_step),
            "--checkpoint-hash",
            checkpoint_hash,
            "--tier",
            "rolling-probe",
            "--difficulty",
            "3",
            "--seed",
            "62043",
            "--samples-per-tool",
            "12",
            "--offered-tools",
            "4",
        ],
        probe / "tools" / "generate.log",
    )
    run_logged(
        [
            eval_tools,
            "--bundle",
            str(bundle),
            "--exam",
            str(tool_exam),
            "--evidence-out",
            str(tool_evidence),
            "--observations-out",
            str(tool_observations),
            "--predictions-out",
            str(probe / "tools" / "all-tools.predictions.json"),
            "--max-new-tokens",
            str(max_new_tokens),
        ],
        probe / "tools" / "evaluate.log",
    )
    tool_result = load_json(tool_evidence)
    for observation in tool_result.get("observations", []):
        if not isinstance(observation, dict):
            raise RuntimeError("tool evidence contains a non-object observation")
        observations.append(observation)

    combined_observations = probe / "observations.jsonl"
    with combined_observations.open("x", encoding="utf-8") as handle:
        for observation in observations:
            json.dump(observation, handle, sort_keys=True)
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    capabilities = ",".join(f"{family}:0:5:1" for family in knowledge_families)
    initial_controller = probe / "controller.initial.json"
    run_stdout_new(
        [
            controller,
            "init",
            "--run-id",
            "tool-native-small-sft-v1",
            "--generator-version",
            "akh-tool-native-adaptive-v1",
            "--seed",
            "62000",
            "--capabilities",
            capabilities,
            "--contracts",
            str(contracts),
            "--min-samples-per-window",
            "12",
            "--enable-precursor-steering",
        ],
        initial_controller,
        probe / "controller.init.log",
    )
    controller_step = probe / "controller.step.json"
    run_stdout_new(
        [
            controller,
            "step",
            "--state",
            str(initial_controller),
            "--observations",
            str(combined_observations),
            "--tier",
            "frontier",
            "--checkpoint-step",
            str(checkpoint_step),
            "--checkpoint-hash",
            checkpoint_hash,
            "--samples",
            "24",
        ],
        controller_step,
        probe / "controller.step.log",
    )
    step_result = load_json(controller_step)
    atomic_json(probe / "controller.after.json", step_result["controller"])
    warnings: dict[str, int] = {}
    for decision in step_result.get("decisions", []):
        warning = str(decision.get("warning", "unknown"))
        warnings[warning] = warnings.get(warning, 0) + 1
    atomic_json(
        status_path,
        {
            "schema_version": 1,
            "stage": "ready_for_curriculum_decision",
            "checkpoint_step": checkpoint_step,
            "checkpoint_model_hash": checkpoint_hash,
            "knowledge_families": len(knowledge_families),
            "tool_families": len(tool_result.get("observations", [])),
            "observations": len(observations),
            "warnings": warnings,
            "next_plan": step_result["next_plan"],
            "probe_dir": str(probe),
        },
    )


def main() -> int:
    args = parse_args()
    base = args.base.resolve(strict=True)
    status_path = resolve_under(base, args.status_json)
    lock_path = base / ".adaptive-probe-supervisor.lock"
    bundle = resolve_under(base, args.bundle)
    probe = resolve_under(base, args.probe_dir)
    post_foundation_json = resolve_under(base, args.post_foundation_json)
    if probe.exists():
        raise RuntimeError(f"refusing existing adaptive probe directory {probe}")
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another adaptive probe supervisor holds the lock") from error
        atomic_json(
            status_path,
            {"schema_version": 1, "stage": "waiting_for_first_sft_epoch"},
        )
        try:
            run_probe(
                args,
                base,
                status_path,
                bundle,
                probe,
                post_foundation_json,
                args.knowledge_families,
            )
        except Exception as error:
            atomic_json(
                status_path,
                {"schema_version": 1, "stage": "failed", "error": str(error)},
            )
            raise
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"adaptive probe supervisor: {error}", file=sys.stderr)
        raise SystemExit(1)
