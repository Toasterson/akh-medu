#!/usr/bin/env python3
"""Run bounded BabyLLM exam/train windows without ever promoting a model.

This supervisor starts from the first rolling-probe controller artifact. Every
window evaluates one controller-selected frontier plus the immutable knowledge
and all-tool sentinels. A confirmed frontier ``steer`` creates a separated
training sibling and a new candidate segment; ``watch``/``green`` continue one
ordinary epoch; ``stop`` preserves all state and performs no update.

The loop never invokes a provider, changes an active-model pointer, or resumes
after a controller stop. Every exam, decision, sibling and segment is retained
under a new output directory for replay.

Ordinary-epoch training may use scaled learning rates and optional warmup
override: pass ``--ordinary-learning-rate-scale`` (0 < value <= 1.0, default
1.0) and ``--ordinary-warmup-steps`` (int, default None = keep run.json value)
to tune ordinary-window SFT hyperparameters independently. Effective values are
recorded in each window's state JSON. Whenever either differs from the
resumed bundle's run.json, ``train_ordinary`` passes
``train_knowledge --allow-schedule-change`` so the otherwise fail-closed
resume path (ADR 104 D5) accepts the change; ``train_knowledge`` itself
records the from/to values in the bundle's ``run-history``.

Pass ``--length-buckets`` (comma-separated, strictly increasing positive
integers, e.g. ``2048,4096,6144``) to bound the target token-count buckets the
``knowledge.length_generalization`` family (ADR 104 D5(b)) probes. The
``knowledge_workflow`` binary's own default (2048,4096,8192,12288) includes
buckets larger than a smaller lane's context window; those get compacted and
fail spuriously, so an 8192-context lane should pass
``--length-buckets 2048,4096,6144``. Left unset (default ``None``), the
binary's own default is used. The value is forwarded verbatim as
``--length-buckets`` to every ``knowledge_workflow exam``/``sibling``
invocation this loop issues (sentinel preparation, frontier materialization,
and training-sibling separation), and is persisted in ``launch-config.json``
and ``loop-config.json`` so ``--resume`` can verify it did not change.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


KNOWLEDGE_FAMILIES = (
    "knowledge.acquire_textbook",
    "knowledge.recall_after_restart",
    "knowledge.revise_from_new_edition",
    "knowledge.provider_fallback",
    "knowledge.grounded_abstention",
    "knowledge.textbook_inventory",
    "knowledge.note_navigation",
    "knowledge.length_generalization",
)
UNBOUND_SENTINEL = "unbound:immutable-sentinel"
MAX_STEERING_SHARE_PPM = 50_000


def validate_length_buckets(raw: str) -> None:
    """Validate ``--length-buckets`` as comma-separated, strictly increasing positive ints.

    This mirrors the format the ``knowledge_workflow`` binary itself parses
    (``crates/grug-data/examples/knowledge_workflow.rs``
    ``parse_length_buckets``), with the additional strictly-increasing check
    applied here so a malformed budget fails fast in the supervisor instead
    of silently reaching the binary.
    """
    parts = raw.split(",")
    if not parts or any(not part.strip() for part in parts):
        raise ValueError("--length-buckets must be a comma-separated list of positive integers")
    try:
        values = [int(part.strip()) for part in parts]
    except ValueError as error:
        raise ValueError(
            f"--length-buckets must be a comma-separated list of positive integers: {error}"
        ) from error
    if any(value <= 0 for value in values):
        raise ValueError("--length-buckets entries must be positive integers")
    if any(later <= earlier for earlier, later in zip(values, values[1:])):
        raise ValueError("--length-buckets entries must be strictly increasing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-binary", type=Path, required=True)
    parser.add_argument("--knowledge-workflow-binary", type=Path, required=True)
    parser.add_argument("--tool-exam-generator-binary", type=Path, required=True)
    parser.add_argument("--knowledge-eval-binary", type=Path, required=True)
    parser.add_argument("--tool-eval-binary", type=Path, required=True)
    parser.add_argument("--controller-binary", type=Path, required=True)
    parser.add_argument("--tool-sibling-binary", type=Path, required=True)
    parser.add_argument("--initial-bundle", type=Path)
    parser.add_argument("--initial-step", type=Path)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--maximum-wait-hours", type=float, default=18.0)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--ordinary-epochs-per-window", type=int, default=1)
    parser.add_argument(
        "--ordinary-learning-rate-scale",
        type=float,
        default=1.0,
        help="scale factor for learning rate during ordinary-window training (must be > 0 and <= 1.0)",
    )
    parser.add_argument(
        "--ordinary-warmup-steps",
        type=int,
        default=None,
        help="override warmup steps for ordinary-window training (default: use run.json value)",
    )
    parser.add_argument("--frontier-samples", type=int, default=24)
    parser.add_argument("--knowledge-sentinel-samples", type=int, default=100)
    parser.add_argument("--tool-sentinel-samples", type=int, default=24)
    parser.add_argument("--max-sibling-records", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only replay-safe work from an existing append-only loop directory",
    )
    parser.add_argument(
        "--knowledge-families",
        type=str,
        default=",".join(KNOWLEDGE_FAMILIES),
        help="comma-separated knowledge workflow families for sentinels and frontier plans",
    )
    parser.add_argument(
        "--length-buckets",
        type=str,
        default=None,
        help=(
            "comma-separated, strictly increasing positive target token-count buckets "
            "(e.g. 2048,4096,6144) forwarded as --length-buckets to every "
            "knowledge_workflow exam/sibling invocation; bounds the "
            "knowledge.length_generalization family (ADR 104 D5(b)) to buckets the "
            "checkpoint's context can hold -- an 8192-context lane should pass "
            "2048,4096,6144. Default: unset, which leaves the knowledge_workflow binary's "
            "own default (2048,4096,8192,12288) in place. Persisted in launch-config.json "
            "and loop-config.json; --resume requires an unchanged value."
        ),
    )
    args = parser.parse_args()
    positive = (
        args.poll_seconds,
        args.maximum_wait_hours,
        args.max_windows,
        args.ordinary_epochs_per_window,
        args.frontier_samples,
        args.knowledge_sentinel_samples,
        args.tool_sentinel_samples,
        args.max_sibling_records,
        args.max_new_tokens,
    )
    if args.poll_seconds < 5 or any(value <= 0 for value in positive[1:]):
        parser.error("poll must be >=5 and all budgets/sample counts must be positive")
    if args.ordinary_learning_rate_scale <= 0 or args.ordinary_learning_rate_scale > 1.0:
        parser.error("--ordinary-learning-rate-scale must be > 0 and <= 1.0")
    if args.ordinary_warmup_steps is not None and args.ordinary_warmup_steps < 0:
        parser.error("--ordinary-warmup-steps must be non-negative")
    if args.length_buckets is not None:
        try:
            validate_length_buckets(args.length_buckets)
        except ValueError as error:
            parser.error(str(error))
    if (args.initial_bundle is None) != (args.initial_step is None):
        parser.error("--initial-bundle and --initial-step must be supplied together")
    families = tuple(
        family.strip() for family in args.knowledge_families.split(",") if family.strip()
    )
    if not families:
        parser.error("--knowledge-families must list at least one family")
    args.knowledge_families = families
    return args


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


def write_jsonl_new(path: Path, values: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for value in values:
            json.dump(value, handle, sort_keys=True)
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(f"{path} contains a blank row at line {index}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path} row {index} is not a JSON object")
            values.append(value)
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


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


def executable(path: Path) -> str:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise RuntimeError(f"binary is not executable: {resolved}")
    return str(resolved)


def launch_configuration(
    args: argparse.Namespace, base: Path, binaries: dict[str, str]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "base": str(base),
        "binaries": {
            name: {"path": path, "sha256": sha256_file(Path(path))}
            for name, path in sorted(binaries.items())
        },
        "poll_seconds": args.poll_seconds,
        "maximum_wait_hours": args.maximum_wait_hours,
        "max_windows": args.max_windows,
        "ordinary_epochs_per_window": args.ordinary_epochs_per_window,
        "ordinary_learning_rate_scale": args.ordinary_learning_rate_scale,
        "ordinary_warmup_steps": args.ordinary_warmup_steps,
        "frontier_samples": args.frontier_samples,
        "knowledge_sentinel_samples": args.knowledge_sentinel_samples,
        "tool_sentinel_samples": args.tool_sentinel_samples,
        "max_sibling_records": args.max_sibling_records,
        "max_new_tokens": args.max_new_tokens,
        "knowledge_families": list(args.knowledge_families),
        "length_buckets": args.length_buckets,
        "promotion_enabled": False,
    }


def wait_for_initial(args: argparse.Namespace, base: Path) -> tuple[Path, Path]:
    if args.initial_bundle is not None and args.initial_step is not None:
        return (
            args.initial_bundle.resolve(strict=True),
            args.initial_step.resolve(strict=True),
        )
    status_path = base / "adaptive-probe-supervisor.json"
    deadline = time.monotonic() + args.maximum_wait_hours * 3600.0
    while time.monotonic() < deadline:
        if status_path.exists():
            status = load_json(status_path)
            if status.get("stage") == "ready_for_curriculum_decision":
                probe = Path(str(status["probe_dir"])).resolve(strict=True)
                return (
                    (base / "knowledge-model-small-v1").resolve(strict=True),
                    (probe / "controller.step.json").resolve(strict=True),
                )
            if status.get("stage") == "failed":
                raise RuntimeError("first adaptive-probe supervisor reported failure")
        time.sleep(args.poll_seconds)
    raise RuntimeError("adaptive-loop wait exceeded the configured deadline")


def load_candidate(bundle: Path) -> dict[str, Any]:
    if (bundle / "active-model.json").exists():
        raise RuntimeError(f"refusing activated bundle in adaptive loop: {bundle}")
    artifact = load_json(bundle / "candidate-model.json")
    if artifact.get("certification") is not None:
        raise RuntimeError(f"candidate is already certified: {bundle}")
    if not str(artifact.get("checkpoint_model_hash", "")).startswith("blake3:"):
        raise RuntimeError(f"candidate model hash is invalid: {bundle}")
    if not str(artifact.get("checkpoint_optimizer_hash", "")).startswith("blake3:"):
        raise RuntimeError(f"candidate has no content-addressed Adam state: {bundle}")
    checkpoint = bundle / str(artifact["checkpoint_dir"])
    for name in ("meta.json", "model.mpk", "optim.mpk"):
        if not (checkpoint / name).is_file():
            raise RuntimeError(f"candidate checkpoint is missing {checkpoint / name}")
    return artifact


def sentinel_inventory(
    output: Path, knowledge_paths: list[Path], tool_path: Path
) -> dict[str, Any]:
    return {
        "knowledge": [
            {
                "path": str(path.relative_to(output)),
                "sha256": sha256_file(path),
            }
            for path in knowledge_paths
        ],
        "tools": {
            "path": str(tool_path.relative_to(output)),
            "sha256": sha256_file(tool_path),
        },
    }


def load_sentinel_inventory(
    output: Path,
    inventory: dict[str, Any],
    families: tuple[str, ...] = KNOWLEDGE_FAMILIES,
) -> tuple[list[Path], Path]:
    knowledge = inventory.get("knowledge")
    tools = inventory.get("tools")
    if not isinstance(knowledge, list) or not isinstance(tools, dict):
        raise RuntimeError("loop config has no sentinel inventory")

    def verified(entry: Any) -> Path:
        if not isinstance(entry, dict):
            raise RuntimeError("sentinel inventory entry is not an object")
        relative = Path(str(entry.get("path", "")))
        if (
            not relative.parts
            or relative.is_absolute()
            or any(part in ("", ".", "..") for part in relative.parts)
        ):
            raise RuntimeError("sentinel inventory contains an unsafe path")
        path = (output / relative).resolve(strict=True)
        if output.resolve(strict=True) not in path.parents:
            raise RuntimeError("sentinel inventory escaped the loop directory")
        if sha256_file(path) != entry.get("sha256"):
            raise RuntimeError(f"sentinel hash mismatch: {path}")
        return path

    knowledge_paths = [verified(entry) for entry in knowledge]
    if len(knowledge_paths) != len(families):
        raise RuntimeError("sentinel inventory does not cover every knowledge family")
    return knowledge_paths, verified(tools)


def next_window_directory(windows: Path, number: int) -> Path:
    for attempt in range(1, 10_000):
        candidate = windows / f"window-{number:04d}-attempt-{attempt:04d}"
        if not candidate.exists():
            candidate.mkdir()
            return candidate
    raise RuntimeError(f"window {number} exhausted its attempt namespace")


def append_window_record(ledger: Path, record: dict[str, Any]) -> None:
    rows = read_jsonl(ledger)
    number = int(record["window"])
    if any(int(row.get("window", -1)) == number for row in rows):
        raise RuntimeError(f"window {number} is already committed in {ledger}")
    if number != len(rows) + 1:
        raise RuntimeError(
            f"window {number} is not the next append-only ledger position {len(rows) + 1}"
        )
    append_jsonl(ledger, record)


def restore_completed_windows(
    output: Path,
    initial_bundle: Path,
    initial_step_path: Path,
    maximum_windows: int,
) -> tuple[Path, dict[str, Any], dict[str, Any], Path, int]:
    bundle = initial_bundle
    artifact = load_candidate(bundle)
    step_path = initial_step_path
    step = load_json(step_path)
    validate_controller_step(step, artifact)
    rows = read_jsonl(output / "windows.jsonl")
    if len(rows) > maximum_windows:
        raise RuntimeError("window ledger exceeds the configured budget")
    for index, row in enumerate(rows, start=1):
        if int(row.get("window", -1)) != index:
            raise RuntimeError("window ledger is not contiguous")
        if row.get("action") not in ("continue", "steer"):
            raise RuntimeError("terminal or unknown window cannot be resumed")
        bundle = Path(str(row["bundle_after"])).resolve(strict=True)
        artifact = load_candidate(bundle)
        if row.get("checkpoint_after") != checkpoint(artifact):
            raise RuntimeError("window ledger checkpoint does not match its candidate")
        step_path = Path(str(row["controller_rebound"])).resolve(strict=True)
        step = load_json(step_path)
        validate_controller_step(step, artifact)
    return bundle, artifact, step, step_path, len(rows) + 1


def checkpoint(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": int(artifact["checkpoint_step"]),
        "artifact_hash": str(artifact["checkpoint_model_hash"]),
    }


def validate_controller_step(
    step: dict[str, Any], artifact: dict[str, Any]
) -> dict[str, Any]:
    controller = step.get("controller")
    plan = step.get("next_plan")
    decisions = step.get("decisions")
    if not isinstance(controller, dict) or not isinstance(plan, dict):
        raise RuntimeError("controller step lacks controller or next_plan")
    if not isinstance(decisions, list) or plan.get("tier") != "frontier":
        raise RuntimeError("controller step lacks a frontier plan/decision list")
    if plan.get("checkpoint") != checkpoint(artifact):
        raise RuntimeError("controller frontier plan is bound to another candidate")
    family = str(plan.get("family", ""))
    families = controller.get("families")
    if not isinstance(families, dict) or family not in families:
        raise RuntimeError("controller frontier family is not registered")
    if int(plan.get("sample_count", 0)) <= 0:
        raise RuntimeError("controller frontier plan has no samples")
    return plan


def safe_stem(family: str) -> str:
    return family.replace(".", "-").replace("/", "-")


def generate_knowledge_exam(
    workflow: str,
    contracts: Path,
    plan: dict[str, Any],
    output: Path,
    error_log: Path,
    length_buckets: str | None = None,
) -> None:
    point = plan["checkpoint"]
    command = [
        workflow,
        "exam",
        "--contracts",
        str(contracts),
        "--family",
        str(plan["family"]),
        "--tier",
        str(plan["tier"]),
        "--run-id",
        str(plan["run_id"]),
        "--plan-generator-version",
        str(plan["generator_version"]),
        "--seed",
        str(plan["seed"]),
        "--difficulty",
        str(plan["difficulty"]),
        "--samples",
        str(plan["sample_count"]),
        "--checkpoint-step",
        str(point["step"]),
        "--checkpoint-hash",
        str(point["artifact_hash"]),
    ]
    if length_buckets is not None:
        command.extend(["--length-buckets", length_buckets])
    run_stdout_new(command, output, error_log)
    exam = load_json(output)
    if exam.get("plan") != plan:
        raise RuntimeError(f"materialized knowledge exam does not equal its plan: {output}")


def generate_tool_exam(
    generator: str,
    contracts: Path,
    plan: dict[str, Any],
    output: Path,
    log_path: Path,
    targeted: bool,
) -> None:
    point = plan["checkpoint"]
    command = [
        generator,
        "--contracts",
        str(contracts),
        "--out",
        str(output),
        "--run-id",
        str(plan["run_id"]),
        "--checkpoint-step",
        str(point["step"]),
        "--checkpoint-hash",
        str(point["artifact_hash"]),
        "--tier",
        str(plan["tier"]),
        "--difficulty",
        str(plan["difficulty"]),
        "--seed",
        str(plan["seed"]),
        "--samples-per-tool",
        str(plan["sample_count"]),
        "--offered-tools",
        "4",
    ]
    if targeted:
        command.extend(["--family", str(plan["family"])])
    run_logged(command, log_path)
    exam = load_json(output)
    expected_scope = [str(plan["family"]).removeprefix("tool_contract.")]
    if (
        exam.get("run_id") != plan["run_id"]
        or exam.get("checkpoint") != point
        or exam.get("tier") != plan["tier"]
        or int(exam.get("difficulty", -1)) != int(plan["difficulty"])
        or int(exam.get("seed", -1)) != int(plan["seed"])
        or int(exam.get("samples_per_tool", -1)) != int(plan["sample_count"])
        or (targeted and exam.get("target_tools") != expected_scope)
    ):
        raise RuntimeError(f"materialized tool exam does not equal its plan: {output}")


def prepare_sentinels(
    args: argparse.Namespace,
    binaries: dict[str, str],
    contracts: Path,
    controller: dict[str, Any],
    output: Path,
    knowledge_families: tuple[str, ...] = KNOWLEDGE_FAMILIES,
) -> tuple[list[Path], Path]:
    sentinels = output / "sentinels"
    if sentinels.exists():
        raise RuntimeError(f"refusing existing sentinel directory {sentinels}")
    temporary = output / f".sentinels.tmp-{os.getpid()}"
    temporary.mkdir()
    knowledge_paths: list[Path] = []
    base_seed = int(controller["base_seed"])
    run_id = f"{controller['run_id']}-immutable-sentinel-v1"
    planner = str(controller["generator_version"])
    for index, family in enumerate(knowledge_families):
        plan = {
            "schema_version": int(controller["schema_version"]),
            "generator_version": planner,
            "run_id": run_id,
            "checkpoint": {"step": 0, "artifact_hash": UNBOUND_SENTINEL},
            "tier": "sentinel",
            "family": family,
            "difficulty": 1,
            "seed": base_seed + 10_000 + index,
            "sample_count": args.knowledge_sentinel_samples,
        }
        path = temporary / f"{safe_stem(family)}.json"
        generate_knowledge_exam(
            binaries["workflow"],
            contracts,
            plan,
            path,
            temporary / f"{safe_stem(family)}.generate.log",
            length_buckets=args.length_buckets,
        )
        knowledge_paths.append(path)

    tool_path = temporary / "all-tools.json"
    tool_plan = {
        "run_id": run_id,
        "checkpoint": {"step": 0, "artifact_hash": UNBOUND_SENTINEL},
        "tier": "sentinel",
        "family": "all_tools",
        "difficulty": 3,
        "seed": base_seed + 20_000,
        "sample_count": args.tool_sentinel_samples,
    }
    generate_tool_exam(
        binaries["tool_exam"],
        contracts,
        tool_plan,
        tool_path,
        temporary / "all-tools.generate.log",
        targeted=False,
    )
    os.replace(temporary, sentinels)
    return (
        [sentinels / path.name for path in knowledge_paths],
        sentinels / tool_path.name,
    )


def evaluate_knowledge(
    evaluator: str,
    bundle: Path,
    exam: Path,
    prefix: Path,
    max_new_tokens: int,
) -> dict[str, Any]:
    evidence = prefix.with_suffix(".evidence.json")
    predictions = prefix.with_suffix(".predictions.json")
    log = prefix.with_suffix(".evaluate.log")
    run_logged(
        [
            evaluator,
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
        log,
    )
    observation = load_json(evidence).get("observation")
    if not isinstance(observation, dict):
        raise RuntimeError(f"knowledge evidence lacks an observation: {evidence}")
    return observation


def evaluate_tools(
    evaluator: str,
    bundle: Path,
    exam: Path,
    prefix: Path,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    evidence = prefix.with_suffix(".evidence.json")
    observations_path = prefix.with_suffix(".observations.jsonl")
    predictions = prefix.with_suffix(".predictions.json")
    log = prefix.with_suffix(".evaluate.log")
    run_logged(
        [
            evaluator,
            "--bundle",
            str(bundle),
            "--exam",
            str(exam),
            "--evidence-out",
            str(evidence),
            "--observations-out",
            str(observations_path),
            "--predictions-out",
            str(predictions),
            "--max-new-tokens",
            str(max_new_tokens),
        ],
        log,
    )
    observations = load_json(evidence).get("observations")
    if not isinstance(observations, list) or not all(
        isinstance(value, dict) for value in observations
    ):
        raise RuntimeError(f"tool evidence lacks observations: {evidence}")
    return observations


def materialize_frontier(
    binaries: dict[str, str],
    contracts: Path,
    plan: dict[str, Any],
    path: Path,
    length_buckets: str | None = None,
) -> None:
    family = str(plan["family"])
    if family.startswith("knowledge."):
        generate_knowledge_exam(
            binaries["workflow"],
            contracts,
            plan,
            path,
            path.with_suffix(".generate.log"),
            length_buckets=length_buckets,
        )
    elif family.startswith("tool_contract."):
        generate_tool_exam(
            binaries["tool_exam"],
            contracts,
            plan,
            path,
            path.with_suffix(".generate.log"),
            targeted=True,
        )
    else:
        raise RuntimeError(f"unsupported frontier family {family!r}")


def evaluate_frontier(
    args: argparse.Namespace,
    binaries: dict[str, str],
    bundle: Path,
    plan: dict[str, Any],
    exam: Path,
    window: Path,
) -> dict[str, Any]:
    family = str(plan["family"])
    prefix = window / "frontier"
    if family.startswith("knowledge."):
        observation = evaluate_knowledge(
            binaries["knowledge_eval"], bundle, exam, prefix, args.max_new_tokens
        )
    else:
        observations = evaluate_tools(
            binaries["tool_eval"], bundle, exam, prefix, args.max_new_tokens
        )
        if len(observations) != 1:
            raise RuntimeError("a targeted tool frontier must emit exactly one observation")
        observation = observations[0]
    if (
        observation.get("tier") != "frontier"
        or observation.get("family") != family
        or observation.get("checkpoint") != plan["checkpoint"]
    ):
        raise RuntimeError("frontier observation does not match its controller plan")
    return observation


def evaluate_sentinels(
    args: argparse.Namespace,
    binaries: dict[str, str],
    bundle: Path,
    artifact: dict[str, Any],
    knowledge_sentinels: list[Path],
    tool_sentinel: Path,
    window: Path,
) -> list[dict[str, Any]]:
    directory = window / "sentinels"
    directory.mkdir()
    observations = []
    for path in knowledge_sentinels:
        family = path.stem.replace("knowledge-", "knowledge.")
        observation = evaluate_knowledge(
            binaries["knowledge_eval"],
            bundle,
            path,
            directory / path.stem,
            args.max_new_tokens,
        )
        observations.append(observation)
    observations.extend(
        evaluate_tools(
            binaries["tool_eval"],
            bundle,
            tool_sentinel,
            directory / "all-tools",
            args.max_new_tokens,
        )
    )
    expected_checkpoint = checkpoint(artifact)
    if any(
        value.get("tier") != "sentinel"
        or value.get("checkpoint") != expected_checkpoint
        for value in observations
    ):
        raise RuntimeError("sentinel evidence was not bound to the executing candidate")
    return observations


def controller_step(
    controller_binary: str,
    prior_step: dict[str, Any],
    observations: list[dict[str, Any]],
    artifact: dict[str, Any],
    samples: int,
    window: Path,
    stem: str,
) -> dict[str, Any]:
    state = window / f"{stem}.state.json"
    observations_path = window / f"{stem}.observations.jsonl"
    output = window / f"{stem}.step.json"
    atomic_json(state, prior_step["controller"])
    write_jsonl_new(observations_path, observations)
    point = checkpoint(artifact)
    run_stdout_new(
        [
            controller_binary,
            "step",
            "--state",
            str(state),
            "--observations",
            str(observations_path),
            "--tier",
            "frontier",
            "--checkpoint-step",
            str(point["step"]),
            "--checkpoint-hash",
            str(point["artifact_hash"]),
            "--samples",
            str(samples),
        ],
        output,
        window / f"{stem}.step.log",
    )
    result = load_json(output)
    validate_controller_step(result, artifact)
    return result


def selected_frontier_steer(step: dict[str, Any]) -> bool:
    family = str(step["next_plan"]["family"])
    selected = [
        decision
        for decision in step["decisions"]
        if decision.get("family") == family and decision.get("tier") == "frontier"
    ]
    return (
        len(selected) == 1
        and selected[0].get("warning") == "steer"
        and selected[0].get("statistically_eligible") is True
    )


def has_stop(step: dict[str, Any]) -> bool:
    return any(decision.get("warning") == "stop" for decision in step["decisions"])


def resolve_file(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError("run descriptor contains an invalid file path")
    return Path(value).resolve(strict=True)


def source_files(run: dict[str, Any], include_current_sibling: bool) -> dict[str, Any]:
    supplements = [resolve_file(value) for value in run.get("supplemental_jsonl", [])]
    if run.get("steering_sibling_jsonl"):
        sibling = resolve_file(run["steering_sibling_jsonl"])
        if include_current_sibling and sibling not in supplements:
            supplements.append(sibling)
        elif not include_current_sibling:
            supplements = [path for path in supplements if path != sibling]
    return {
        "train": resolve_file(run["train_jsonl"]),
        "supplements": supplements,
        "eval": resolve_file(run["eval_jsonl"]) if run.get("eval_jsonl") else None,
        "eval_supplements": [
            resolve_file(value) for value in run.get("supplemental_eval_jsonl", [])
        ],
    }


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def source_record_count(sources: dict[str, Any]) -> int:
    return count_jsonl(sources["train"]) + sum(
        count_jsonl(path) for path in sources["supplements"]
    )


def verified_base_sources(
    run: dict[str, Any], bundle: Path | None = None
) -> tuple[dict[str, Any], int]:
    """Verify the full parent corpus, then separate its temporary sibling.

    A steered run's training_records includes its sibling. New segments
    replace that sibling while preserving the base; comparing the base alone
    with the full count incorrectly rejects every steered starting bundle.
    """
    lineage = run.get("steered_from") or {}
    if lineage and not run.get("steering_sibling_jsonl"):
        # Ordinary resume folds the sibling into supplemental_jsonl and clears
        # its dedicated path. Recover that path only from the same corpus and
        # steering lineage, never from a filename or record-count guess.
        if bundle is None or not run.get("training_corpus_hash"):
            raise RuntimeError("resumed steering lineage requires its bundle run history")
        siblings = set()
        for path in (bundle / "run-history").glob("before-resume-step_*.json"):
            previous = load_json(path)
            if (
                previous.get("steering_sibling_jsonl")
                and previous.get("steered_from") == lineage
                and previous.get("training_corpus_hash") == run["training_corpus_hash"]
                and previous.get("training_records") == run["training_records"]
            ):
                siblings.add(resolve_file(previous["steering_sibling_jsonl"]))
        if len(siblings) != 1:
            raise RuntimeError("run history does not identify one original steering sibling")
        sibling = siblings.pop()
        if sibling not in source_files(run, include_current_sibling=True)["supplements"]:
            raise RuntimeError("historical steering sibling is absent from the resumed corpus")
        run = {**run, "steering_sibling_jsonl": str(sibling)}

    full_count = source_record_count(source_files(run, include_current_sibling=True))
    if full_count != int(run["training_records"]):
        raise RuntimeError("initial run descriptor does not reproduce its training record count")
    sources = source_files(run, include_current_sibling=False)
    base_count = source_record_count(sources)
    if run.get("steering_sibling_jsonl"):
        if (
            lineage.get("base_training_records") != base_count
            or lineage.get("sibling_training_records") != full_count - base_count
        ):
            raise RuntimeError("initial steering lineage does not reproduce its base/sibling counts")
    return sources, base_count


def common_training_command(
    binary: str,
    run: dict[str, Any],
    sources: dict[str, Any],
    contracts: Path,
    artifacts: Path,
    epochs: int,
) -> list[str]:
    command = [
        binary,
        "--contracts",
        str(contracts),
        "--train-jsonl",
        str(sources["train"]),
        "--artifacts-dir",
        str(artifacts),
        "--model",
        str(run["model_size"]),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(run["batch_size"]),
        "--learning-rate",
        str(run["learning_rate"]),
        "--warmup-steps",
        str(run["warmup_steps"]),
        "--bpe-vocab-size",
        str(run["tokenizer_vocab_size"]),
        "--eval-every",
        str(run.get("eval_every", 1000)),
        "--eval-samples",
        str(run.get("eval_samples", 8)),
        "--eval-max-new-tokens",
        str(run.get("eval_max_new_tokens", 128)),
        "--ckpt-every",
        str(run.get("ckpt_every", 1000)),
        "--loss-cap",
        str(run["loss_cap"]),
        "--grad-norm-cap",
        str(run["grad_norm_cap"]),
        "--seed",
        str(run["seed"]),
    ]
    for path in sources["supplements"]:
        command.extend(["--supplemental-jsonl", str(path)])
    if sources["eval"] is not None:
        command.extend(["--eval-jsonl", str(sources["eval"])])
    for path in sources["eval_supplements"]:
        command.extend(["--supplemental-eval-jsonl", str(path)])
    return command


def sibling_record_budget(base_records: int, requested: int) -> int:
    if base_records <= 0 or requested <= 0:
        raise RuntimeError("base and requested sibling record counts must be positive")
    budget = min(requested, base_records // 19)
    if budget == 0:
        raise RuntimeError("base corpus is too small for one bounded steering record")
    return budget


def make_sibling(
    args: argparse.Namespace,
    binaries: dict[str, str],
    contracts: Path,
    family: str,
    exam: Path,
    base_records: int,
    window: Path,
) -> tuple[Path, Path, int]:
    budget = sibling_record_budget(base_records, args.max_sibling_records)
    output = window / "training-sibling"
    if family.startswith("knowledge."):
        source = load_json(exam)
        episodes = source.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            raise RuntimeError("knowledge frontier has no episodes")
        step_counts = {len(episode.get("steps", [])) for episode in episodes}
        if len(step_counts) != 1 or next(iter(step_counts)) <= 0:
            raise RuntimeError("knowledge frontier episode width is not stable")
        units = budget // next(iter(step_counts))
        if units == 0:
            raise RuntimeError("steering budget cannot fit one knowledge episode")
        sibling_command = [
            binaries["workflow"],
            "sibling",
            "--contracts",
            str(contracts),
            "--exam",
            str(exam),
            "--out",
            str(output),
            "--samples",
            str(units),
        ]
        if args.length_buckets is not None:
            sibling_command.extend(["--length-buckets", args.length_buckets])
        run_logged(
            sibling_command,
            window / "training-sibling.generate.log",
        )
    elif family.startswith("tool_contract."):
        run_logged(
            [
                binaries["tool_sibling"],
                "--contracts",
                str(contracts),
                "--exam",
                str(exam),
                "--family",
                family,
                "--out",
                str(output),
                "--samples",
                str(budget),
            ],
            window / "training-sibling.generate.log",
        )
    else:
        raise RuntimeError(f"unsupported sibling family {family!r}")
    manifest_path = output / "manifest.json"
    manifest = load_json(manifest_path)
    records = int(manifest.get("records", 0))
    if (
        manifest.get("complete") is not True
        or manifest.get("family") != family
        or records <= 0
        or records > budget
        or records * 1_000_000
        > (base_records + records) * MAX_STEERING_SHARE_PPM
    ):
        raise RuntimeError("generated sibling exceeds its bounded record budget")
    return manifest_path, output / "train.sft.jsonl", records


def train_ordinary(
    args: argparse.Namespace,
    binaries: dict[str, str],
    bundle: Path,
    artifact: dict[str, Any],
    window: Path,
) -> tuple[Path, float, int, bool]:
    run = load_json(bundle / "run.json")
    sources = source_files(run, include_current_sibling=True)

    # Apply ordinary-window learning rate scale and warmup override
    effective_learning_rate = float(run["learning_rate"]) * args.ordinary_learning_rate_scale
    effective_warmup_steps = (
        args.ordinary_warmup_steps
        if args.ordinary_warmup_steps is not None
        else int(run["warmup_steps"])
    )
    # ADR 104 D5 deliberately trains ordinary windows at a reduced learning
    # rate (and optionally a different warmup). train_knowledge's --resume
    # path is fail-closed on any schedule drift unless explicitly sanctioned
    # via --allow-schedule-change, so pass it whenever this window's
    # effective schedule actually differs from the resumed run.json.
    schedule_change = (
        effective_learning_rate != float(run["learning_rate"])
        or args.ordinary_warmup_steps is not None
    )

    # Build command with effective values
    command = [
        binaries["train"],
        "--contracts",
        str(bundle / "contracts.json"),
        "--train-jsonl",
        str(sources["train"]),
        "--artifacts-dir",
        str(bundle),
        "--model",
        str(run["model_size"]),
        "--epochs",
        str(int(run["epochs"]) + args.ordinary_epochs_per_window),
        "--batch-size",
        str(run["batch_size"]),
        "--learning-rate",
        f"{effective_learning_rate:.8g}",
        "--warmup-steps",
        str(effective_warmup_steps),
        "--bpe-vocab-size",
        str(run["tokenizer_vocab_size"]),
        "--eval-every",
        str(run.get("eval_every", 1000)),
        "--eval-samples",
        str(run.get("eval_samples", 8)),
        "--eval-max-new-tokens",
        str(run.get("eval_max_new_tokens", 128)),
        "--ckpt-every",
        str(run.get("ckpt_every", 1000)),
        "--loss-cap",
        str(run["loss_cap"]),
        "--grad-norm-cap",
        str(run["grad_norm_cap"]),
        "--seed",
        str(run["seed"]),
    ]
    for path in sources["supplements"]:
        command.extend(["--supplemental-jsonl", str(path)])
    if sources["eval"] is not None:
        command.extend(["--eval-jsonl", str(sources["eval"])])
    for path in sources["eval_supplements"]:
        command.extend(["--supplemental-eval-jsonl", str(path)])

    command.extend(["--resume", str(bundle / str(artifact["checkpoint_dir"]))])
    if schedule_change:
        command.append("--allow-schedule-change")
    run_logged(command, window / "ordinary-training.log")
    return bundle, effective_learning_rate, effective_warmup_steps, schedule_change


def train_steered_segment(
    binaries: dict[str, str],
    fixed_run: dict[str, Any],
    fixed_sources: dict[str, Any],
    parent_bundle: Path,
    contracts: Path,
    decision: Path,
    frontier: Path,
    sibling_manifest: Path,
    sibling_jsonl: Path,
    segment: Path,
    window: Path,
) -> Path:
    command = common_training_command(
        binaries["train"], fixed_run, fixed_sources, contracts, segment, 1
    )
    command.extend(
        [
            "--steer-from-bundle",
            str(parent_bundle),
            "--steering-decision",
            str(decision),
            "--steering-frontier-exam",
            str(frontier),
            "--steering-sibling-manifest",
            str(sibling_manifest),
            "--steering-sibling-jsonl",
            str(sibling_jsonl),
        ]
    )
    run_logged(command, window / "steered-training.log")
    return segment


def rebind_controller(
    args: argparse.Namespace,
    controller_binary: str,
    evaluated_step: dict[str, Any],
    artifact: dict[str, Any],
    window: Path,
) -> tuple[dict[str, Any], Path]:
    result = controller_step(
        controller_binary,
        evaluated_step,
        [],
        artifact,
        args.frontier_samples,
        window,
        "controller-rebound",
    )
    if result.get("decisions") != []:
        raise RuntimeError("checkpoint rebind unexpectedly emitted decisions")
    return result, window / "controller-rebound.step.json"


def main() -> int:
    args = parse_args()
    base = args.base.resolve(strict=True)
    output = args.out.resolve()
    fresh = not output.exists()
    if fresh and args.resume:
        raise RuntimeError(f"cannot resume missing adaptive-loop directory {output}")
    if not fresh and not args.resume:
        raise RuntimeError(
            f"refusing existing adaptive-loop directory {output}; pass --resume for replay-safe recovery"
        )
    if fresh:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir()
    elif not output.is_dir():
        raise RuntimeError(f"adaptive-loop output is not a directory: {output}")
    status_path = output / "status.json"
    lock_path = base / ".tool-native-adaptive-loop.lock"
    binaries = {
        "train": executable(args.train_binary),
        "workflow": executable(args.knowledge_workflow_binary),
        "tool_exam": executable(args.tool_exam_generator_binary),
        "knowledge_eval": executable(args.knowledge_eval_binary),
        "tool_eval": executable(args.tool_eval_binary),
        "controller": executable(args.controller_binary),
        "tool_sibling": executable(args.tool_sibling_binary),
    }
    launch = launch_configuration(args, base, binaries)

    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another tool-native adaptive loop holds the lock") from error

        launch_path = output / "launch-config.json"
        if launch_path.exists():
            stored_launch = load_json(launch_path)
            # Loop directories created before --knowledge-families existed have
            # no such key; they were always built from the hardcoded default.
            stored_launch.setdefault("knowledge_families", list(KNOWLEDGE_FAMILIES))
            if stored_launch != launch:
                raise RuntimeError(
                    "resume launch arguments or executable hashes differ from the original loop"
                )
        elif fresh:
            atomic_json(launch_path, launch)
        else:
            # Compatibility for the first waiting-only supervisor written
            # before launch-config.json existed. No generated exam or model
            # state may be present when adopting it.
            entries = {path.name for path in output.iterdir()}
            status = load_json(status_path)
            if entries != {"status.json"} or status.get("stage") != "waiting_for_initial_probe":
                raise RuntimeError("existing loop has no verifiable launch configuration")
            atomic_json(launch_path, launch)

        previous_status = load_json(status_path) if status_path.exists() else None
        if fresh:
            atomic_json(
                status_path, {"schema_version": 1, "stage": "waiting_for_initial_probe"}
            )
        bundle, step_path = wait_for_initial(args, base)
        artifact = load_candidate(bundle)
        step = load_json(step_path)
        validate_controller_step(step, artifact)
        fixed_run = load_json(bundle / "run.json")
        fixed_sources, base_records = verified_base_sources(fixed_run, bundle)
        contracts = (bundle / "contracts.json").resolve(strict=True)
        loop_config_path = output / "loop-config.json"
        if loop_config_path.exists():
            stored_loop_config = load_json(loop_config_path)
            # Loop directories created before --knowledge-families existed have
            # no such key; they were always built from the hardcoded default.
            stored_loop_config.setdefault(
                "knowledge_families", list(KNOWLEDGE_FAMILIES)
            )
            # Backfill legacy configs that don't have ordinary learning rate scale/warmup overrides.
            stored_loop_config.setdefault("ordinary_learning_rate_scale", 1.0)
            stored_loop_config.setdefault("ordinary_warmup_steps", None)
            # Backfill legacy configs that predate --length-buckets.
            stored_loop_config.setdefault("length_buckets", None)
            knowledge_sentinels, tool_sentinel = load_sentinel_inventory(
                output,
                stored_loop_config.get("sentinels", {}),
                args.knowledge_families,
            )
        else:
            sentinels_path = output / "sentinels"
            if sentinels_path.exists():
                knowledge_sentinels = [
                    sentinels_path / f"{safe_stem(family)}.json"
                    for family in args.knowledge_families
                ]
                tool_sentinel = sentinels_path / "all-tools.json"
                for path in [*knowledge_sentinels, tool_sentinel]:
                    path.resolve(strict=True)
            else:
                knowledge_sentinels, tool_sentinel = prepare_sentinels(
                    args,
                    binaries,
                    contracts,
                    step["controller"],
                    output,
                    args.knowledge_families,
                )
            stored_loop_config = {
                "schema_version": 1,
                "base": str(base),
                "initial_bundle": str(bundle),
                "initial_step": str(step_path),
                "base_training_records": base_records,
                "max_windows": args.max_windows,
                "ordinary_epochs_per_window": args.ordinary_epochs_per_window,
                "ordinary_learning_rate_scale": args.ordinary_learning_rate_scale,
                "ordinary_warmup_steps": args.ordinary_warmup_steps,
                "frontier_samples": args.frontier_samples,
                "knowledge_sentinel_samples": args.knowledge_sentinel_samples,
                "tool_sentinel_samples": args.tool_sentinel_samples,
                "max_sibling_records": args.max_sibling_records,
                "maximum_steering_share_ppm": MAX_STEERING_SHARE_PPM,
                "knowledge_families": list(args.knowledge_families),
                "length_buckets": args.length_buckets,
                "sentinels": sentinel_inventory(
                    output, knowledge_sentinels, tool_sentinel
                ),
                "promotion_enabled": False,
            }
            atomic_json(loop_config_path, stored_loop_config)
        expected_loop_fields = {
            "schema_version": 1,
            "base": str(base),
            "initial_bundle": str(bundle),
            "initial_step": str(step_path),
            "base_training_records": base_records,
            "max_windows": args.max_windows,
            "ordinary_epochs_per_window": args.ordinary_epochs_per_window,
            "ordinary_learning_rate_scale": args.ordinary_learning_rate_scale,
            "ordinary_warmup_steps": args.ordinary_warmup_steps,
            "frontier_samples": args.frontier_samples,
            "knowledge_sentinel_samples": args.knowledge_sentinel_samples,
            "tool_sentinel_samples": args.tool_sentinel_samples,
            "max_sibling_records": args.max_sibling_records,
            "maximum_steering_share_ppm": MAX_STEERING_SHARE_PPM,
            "knowledge_families": list(args.knowledge_families),
            "length_buckets": args.length_buckets,
            "promotion_enabled": False,
        }
        if any(stored_loop_config.get(key) != value for key, value in expected_loop_fields.items()):
            raise RuntimeError("stored loop configuration differs from this invocation")

        ledger = output / "windows.jsonl"
        windows_dir = output / "windows"
        windows_dir.mkdir(exist_ok=True)
        (output / "segments").mkdir(exist_ok=True)
        bundle, artifact, step, current_step_path, first_window = restore_completed_windows(
            output, bundle, step_path, args.max_windows
        )
        committed_windows = first_window - 1
        if previous_status is not None:
            previous_stage = str(previous_status.get("stage", ""))
            previous_window = int(previous_status.get("window", 0))
            if previous_stage in ("stopped_by_controller", "window_budget_exhausted"):
                raise RuntimeError(f"refusing to resume terminal loop stage {previous_stage}")
            if previous_stage in ("training", "rebinding") and previous_window > committed_windows:
                raise RuntimeError(
                    f"interrupted {previous_stage} window requires explicit artifact recovery; no update was retried"
                )
            if previous_stage == "failed":
                raise RuntimeError("refusing to silently resume a failed adaptive loop")

        if first_window > args.max_windows:
            atomic_json(
                status_path,
                {
                    "schema_version": 1,
                    "stage": "window_budget_exhausted",
                    "windows": args.max_windows,
                    "bundle": str(bundle),
                    "checkpoint": checkpoint(artifact),
                    "controller_step": str(current_step_path),
                    "promotion_enabled": False,
                },
            )
            return 0

        for number in range(first_window, args.max_windows + 1):
            window = next_window_directory(windows_dir, number)
            plan = validate_controller_step(step, artifact)
            atomic_json(
                status_path,
                {
                    "schema_version": 1,
                    "stage": "evaluating",
                    "window": number,
                    "bundle": str(bundle),
                    "checkpoint": checkpoint(artifact),
                    "frontier_family": plan["family"],
                    "window_dir": str(window),
                    "promotion_enabled": False,
                },
            )
            frontier = window / "frontier.exam.json"
            materialize_frontier(
                binaries, contracts, plan, frontier, length_buckets=args.length_buckets
            )
            sentinel_observations = evaluate_sentinels(
                args,
                binaries,
                bundle,
                artifact,
                knowledge_sentinels,
                tool_sentinel,
                window,
            )
            frontier_observation = evaluate_frontier(
                args, binaries, bundle, plan, frontier, window
            )
            observations = sentinel_observations + [frontier_observation]
            evaluated = controller_step(
                binaries["controller"],
                step,
                observations,
                artifact,
                args.frontier_samples,
                window,
                "controller-evaluated",
            )
            evaluated_path = window / "controller-evaluated.step.json"
            warnings: dict[str, int] = {}
            for decision in evaluated["decisions"]:
                warning = str(decision.get("warning", "unknown"))
                warnings[warning] = warnings.get(warning, 0) + 1

            if has_stop(evaluated):
                record = {
                    "schema_version": 1,
                    "window": number,
                    "action": "stop",
                    "bundle": str(bundle),
                    "checkpoint": checkpoint(artifact),
                    "controller_step": str(evaluated_path),
                    "warnings": warnings,
                    "window_dir": str(window),
                    "promotion_attempted": False,
                }
                append_window_record(ledger, record)
                atomic_json(status_path, {**record, "stage": "stopped_by_controller"})
                return 0

            old_checkpoint = checkpoint(artifact)
            sibling_records = 0
            ordinary_learning_rate = None
            ordinary_warmup_steps = None
            ordinary_schedule_change = False
            if selected_frontier_steer(evaluated):
                steering_plan = evaluated["next_plan"]
                steering_frontier = window / "steering-source.exam.json"
                materialize_frontier(
                    binaries,
                    contracts,
                    steering_plan,
                    steering_frontier,
                    length_buckets=args.length_buckets,
                )
                sibling_manifest, sibling_jsonl, sibling_records = make_sibling(
                    args,
                    binaries,
                    contracts,
                    str(steering_plan["family"]),
                    steering_frontier,
                    base_records,
                    window,
                )
                segment = output / "segments" / f"segment-{number:04d}"
                atomic_json(
                    status_path,
                    {
                        "schema_version": 1,
                        "stage": "training",
                        "window": number,
                        "action": "steer",
                        "bundle": str(bundle),
                        "checkpoint": old_checkpoint,
                        "next_bundle": str(segment),
                        "controller_step": str(evaluated_path),
                        "window_dir": str(window),
                        "promotion_enabled": False,
                    },
                )
                next_bundle = train_steered_segment(
                    binaries,
                    fixed_run,
                    fixed_sources,
                    bundle,
                    contracts,
                    evaluated_path,
                    steering_frontier,
                    sibling_manifest,
                    sibling_jsonl,
                    segment,
                    window,
                )
                action = "steer"
            else:
                atomic_json(
                    status_path,
                    {
                        "schema_version": 1,
                        "stage": "training",
                        "window": number,
                        "action": "continue",
                        "bundle": str(bundle),
                        "checkpoint": old_checkpoint,
                        "next_bundle": str(bundle),
                        "controller_step": str(evaluated_path),
                        "window_dir": str(window),
                        "promotion_enabled": False,
                    },
                )
                (
                    next_bundle,
                    ordinary_learning_rate,
                    ordinary_warmup_steps,
                    ordinary_schedule_change,
                ) = train_ordinary(args, binaries, bundle, artifact, window)
                action = "continue"

            next_artifact = load_candidate(next_bundle)
            if (
                int(next_artifact["checkpoint_step"]) <= int(artifact["checkpoint_step"])
                or next_artifact["checkpoint_model_hash"]
                == artifact["checkpoint_model_hash"]
            ):
                raise RuntimeError("training did not advance candidate step and model identity")
            atomic_json(
                status_path,
                {
                    "schema_version": 1,
                    "stage": "rebinding",
                    "window": number,
                    "action": action,
                    "bundle": str(next_bundle),
                    "checkpoint": checkpoint(next_artifact),
                    "controller_step": str(evaluated_path),
                    "window_dir": str(window),
                    "promotion_enabled": False,
                },
            )
            rebound, rebound_path = rebind_controller(
                args, binaries["controller"], evaluated, next_artifact, window
            )
            record = {
                "schema_version": 1,
                "window": number,
                "action": action,
                "bundle_before": str(bundle),
                "checkpoint_before": old_checkpoint,
                "bundle_after": str(next_bundle),
                "checkpoint_after": checkpoint(next_artifact),
                "controller_step": str(evaluated_path),
                "controller_rebound": str(rebound_path),
                "frontier_family": plan["family"],
                "next_frontier_family": rebound["next_plan"]["family"],
                "sibling_records": sibling_records,
                "warnings": warnings,
                "window_dir": str(window),
                "promotion_attempted": False,
            }
            if action == "continue" and ordinary_learning_rate is not None:
                record["ordinary_learning_rate"] = ordinary_learning_rate
                record["ordinary_warmup_steps"] = ordinary_warmup_steps
                record["schedule_change"] = ordinary_schedule_change
            append_window_record(ledger, record)
            atomic_json(status_path, {**record, "stage": "window_complete"})
            bundle = next_bundle
            artifact = next_artifact
            step = rebound
            current_step_path = rebound_path

        atomic_json(
            status_path,
            {
                "schema_version": 1,
                "stage": "window_budget_exhausted",
                "windows": args.max_windows,
                "bundle": str(bundle),
                "checkpoint": checkpoint(artifact),
                "controller_step": str(current_step_path),
                "promotion_enabled": False,
            },
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        try:
            failure_args = parse_args()
            failure_out = failure_args.out.resolve()
            if failure_out.is_dir():
                atomic_json(
                    failure_out / "status.json",
                    {
                        "schema_version": 1,
                        "stage": "failed",
                        "error": str(error),
                        "promotion_enabled": False,
                    },
                )
        except Exception as status_error:
            print(
                f"adaptive loop: could not persist failed status: {status_error}",
                file=sys.stderr,
            )
        print(f"adaptive loop: {error}", file=sys.stderr)
        raise SystemExit(1)
