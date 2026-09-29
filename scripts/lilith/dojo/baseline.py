"""Batch driver, baseline summary and paired compare for the dojo runner (plan slice C4).

`run` selects episodes from the snapshots (family, split, limit, seed), lets a model act on each
through `runner.ModelEpisodeRunner`, scores it with `families.verify` and the `cos-v1` reduction,
and writes into `--out`:

- `episodes.jsonl`        one record per episode: steps, receipts, transcript, usage, latency
- `verification.jsonl`    per episode: verdict, `cos-v1` weight, hard constraints, contract validity
- `provenance.json`       what produced the run (git SHA, snapshot root hash, floors, prompt, model)
- `baseline-summary.json` per family: n, mean weight and bootstrap 95% CI, hard-constraint counts,
                          contract-validity and language-check rates, tokens, latency

Runs resume: an episode whose latest verification record has status `ok` is skipped. A record with
status `error` (the endpoint failed) is retried on the next run. A resumed run must match the
provenance of the first (same model, prompt, snapshots, seed, commit), otherwise it is refused.

`compare` prints the paired per-family table the exam uses (parent / teacher / candidate): means
with intervals, hard constraints, and paired differences with 95% bootstrap intervals over the
shared episode ids. It reuses `exam.controller.PairedDifference` for the verdict wording.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import dataclasses
import datetime as dt
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dojo import DOJO_VERSION, floors as floors_mod
from dojo.contracts import REPO_ROOT
from dojo.episode import EmitDocument, Family, SendMessage
from dojo.errors import CompareError, DojoError, RunProvenanceError
from dojo.families import BuildContext, HardConstraint, build_all, load_corrections, load_illumos_pack, verify, verify_context
from dojo.model_adapters import (
    ANTHROPIC_BASE_URL,
    DEFAULT_ANTHROPIC_MODEL,
    AnthropicAdapter,
    HttpPolicy,
    ModelAdapter,
    OpenAIChatAdapter,
    load_secret,
)
from dojo.replay_server import DojoLedger, GitAncestryOracle, episode_servers
from dojo.reward import COS_V1, reduce_episode
from dojo.runner import (
    RUNNER_VERSION,
    ModelEpisodeRunner,
    RunnerConfig,
    SystemPrompt,
    compose_system_prompt,
    prompt_bundle_sha256,
)
from dojo.snapshot import SnapshotStore, canonical_json, sha256_hex
from dojo.splits import Split, SplitConfig, assign_splits

SUMMARY_SCHEMA = "lilith/dojo-baseline@1"
EPISODES_FILE = "episodes.jsonl"
VERIFICATION_FILE = "verification.jsonl"
PROVENANCE_FILE = "provenance.json"
SUMMARY_FILE = "baseline-summary.json"
BOOTSTRAP_RESAMPLES = 2000
RECORDER_DEFAULT_ROOT = "/Volumes/akh-medu-runs/lilith/cos-snapshots"
ANSWERED = frozenset({"document", "message", "multiple"})


# --- selection -----------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Chosen:
    spec: Any  # EpisodeSpec
    split: Split


def parse_families(value: str) -> tuple[Family, ...]:
    if value == "all":
        return tuple(f for f in Family if f is not Family.CORRECTION)
    try:
        return tuple(Family(v) for v in value.split(","))
    except ValueError:
        raise DojoError(f"unknown family in {value!r}", help=f"use 'all' or a comma list of {[f.value for f in Family]}") from None


def select_episodes(store: SnapshotStore, oracle, families: Sequence[Family], split: str, limit: int | None,
                    seed: int, illumos_pack=(), corrections=()) -> tuple[list[Chosen], dict]:
    """Deterministic for a given snapshot root, family, split, limit and seed: every model in an
    exam gets the same episodes."""
    results = build_all(store, BuildContext(oracle=oracle, illumos_pack=tuple(illumos_pack),
                                                  corrections=tuple(corrections)))
    all_specs = [s for res in results.values() for s in res.specs]
    manifest = assign_splits(all_specs, SplitConfig())
    wanted = {Split(split)} if split != "all" else set(Split)
    by_family: dict[Family, list[Chosen]] = {f: [] for f in families}
    for assignment in manifest.assignments:
        if assignment.spec.family in by_family and assignment.split in wanted:
            by_family[assignment.spec.family].append(Chosen(assignment.spec, assignment.split))
    chosen: list[Chosen] = []
    available: dict[str, int] = {}
    for family, pool in by_family.items():
        pool.sort(key=lambda c: c.spec.episode_id)
        available[family.value] = len(pool)
        if limit is not None and len(pool) > limit:
            random.Random(f"{seed}:{family.value}").shuffle(pool)
            pool = pool[:limit]
        chosen.extend(pool)
    chosen.sort(key=lambda c: c.spec.episode_id)
    return chosen, {"available": available, "selected": dict(collections.Counter(c.spec.family.value for c in chosen)),
                    "split_manifest_sha256": manifest.to_dict()["manifest_sha256"]}


# --- provenance ----------------------------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> tuple[int, str]:
    try:
        done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        return 127, str(error)
    return done.returncode, done.stdout.strip()


def git_provenance(repo: Path, allow_dirty: bool) -> dict:
    code, sha = _git(repo, "rev-parse", "HEAD")
    if code != 0:
        raise RunProvenanceError(f"{repo} is not a git checkout: {sha}")
    _, status = _git(repo, "status", "--porcelain", "--untracked-files=no")
    dirty = bool(status)
    if dirty and not allow_dirty:
        raise RunProvenanceError(f"the working tree at {repo} has uncommitted tracked changes; a baseline must name a commit")
    code, _ = _git(repo, "merge-base", "--is-ancestor", "HEAD", "origin/main")
    return {"git_sha": sha, "git_dirty": dirty, "on_origin_main": True if code == 0 else (False if code == 1 else None)}


def snapshot_root_sha256(store: SnapshotStore) -> str:
    return sha256_hex(canonical_json([[i, store.get(i).manifest_sha256] for i in store.ids]))


IDENTITY_KEYS = ("runner", "dojo", "git_sha", "snapshot_root_sha256", "floors_sha256", "prompt_sha256",
                 "system_prompt_sha256", "model_kind", "model_id", "endpoint_host", "temperature", "max_tokens",
                 "max_rounds", "episode_token_cap", "seed", "split", "families", "limit")


def check_resume(out: Path, provenance: Mapping[str, Any]) -> None:
    path = out / PROVENANCE_FILE
    if not path.exists():
        return
    earlier = json.loads(path.read_text())
    differing = [k for k in IDENTITY_KEYS if earlier.get(k) != provenance.get(k)]
    if differing:
        raise RunProvenanceError(f"{out} was started with a different {', '.join(differing)}")


# --- records --------------------------------------------------------------------------------------------------


def _jsonl_append(path: Path, lock: threading.Lock, record: Mapping[str, Any]) -> None:
    line = json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
    with lock, path.open("a", encoding="utf-8") as stream:
        stream.write(line)


def read_jsonl_latest(path: Path) -> dict[str, dict]:
    """episode_id -> the last record written for it (a retry supersedes an earlier error)."""
    latest: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                latest[record["episode_id"]] = record
    return latest


def episode_weight(verdict, weights) -> tuple[float, str, int | None]:
    """The episode's cos-v1 weight: the weight of the step that carries the effect (the decision),
    forced to 0 when the episode holds any hard constraint (cos-v1 zeroes a step that broke a rule
    and asks for a sibling, so the episode does not count as earned)."""
    answer = next((s.step for s in reversed(verdict.steps) if s.vector.effect is not None), None)
    if verdict.hard_constraints:
        return 0.0, "hard-constraint", answer
    if answer is None:
        return 0.0, "no-answer-step", None
    return next(w.weight for w in weights if w.step == answer), "answer-step", answer


def _episode_record(chosen: Chosen, result, provenance_id: str) -> dict:
    episode, trace = result.episode, result.trace
    return {
        "episode_id": chosen.spec.episode_id,
        "family": chosen.spec.family.value,
        "split": chosen.split.value,
        "snapshot": chosen.spec.snapshot_id,
        "subjects": list(chosen.spec.subject_ids),
        "spec_sha256": chosen.spec.spec_sha256,
        "steps": [s.to_dict() for s in episode.steps],
        "receipts": [r.to_dict() for r in episode.receipts],
        "run": provenance_id,
        **trace.to_dict(),
    }


def _verification_record(chosen: Chosen, result, snapshot, oracle) -> dict:
    spec, episode, trace = chosen.spec, result.episode, result.trace
    base = {"episode_id": spec.episode_id, "family": spec.family.value, "split": chosen.split.value,
            "status": trace.status, "answer_status": trace.answer_status, "rounds": trace.rounds,
            "contract_ok": trace.contract_ok, "schema_errors": trace.schema_errors,
            "weight_policy": f"{COS_V1.name}@{COS_V1.version}", "error": trace.error}
    if trace.status != "ok":
        return base | {"weight": None, "weight_basis": "not-scored", "hard": [], "metrics": {}, "step_weights": []}
    try:
        verdict = verify(spec, episode, verify_context(snapshot, oracle))
    except Exception as error:  # model output the verifiers cannot read scores 0; it is counted, not hidden
        return base | {"weight": 0.0, "weight_basis": "verifier-error", "hard": [], "metrics": {}, "step_weights": [],
                       "verify_error": f"{type(error).__name__}: {error}"}
    weights = reduce_episode(verdict)
    weight, basis, answer_step = episode_weight(verdict, weights)
    return base | {
        "weight": weight, "weight_basis": basis, "answer_step": answer_step,
        "hard": sorted(h.value for h in verdict.hard_constraints),
        "metrics": dict(verdict.metrics),
        "step_weights": [w.to_dict() for w in weights],
        "verdict": verdict.to_dict(),
    }


# --- statistics ---------------------------------------------------------------------------------------------------


def bootstrap_ci(values: Sequence[float], seed: str, resamples: int = BOOTSTRAP_RESAMPLES,
                 level: float = 0.95) -> tuple[float, float] | None:
    """Percentile bootstrap interval of the mean; deterministic for a given seed."""
    n = len(values)
    if n == 0:
        return None
    if n == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    tail = (1 - level) / 2
    return means[int(tail * resamples)], means[min(resamples - 1, int((1 - tail) * resamples))]


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, max(0, math.ceil(q * len(sorted_values)) - 1))]


def _family_stats(family: str, verifications: Sequence[dict], episodes: Mapping[str, dict], seed: str) -> dict:
    scored = [v for v in verifications if v["status"] == "ok"]
    errors = len(verifications) - len(scored)
    weights = [float(v["weight"]) for v in scored]
    hard = [set(v["hard"]) for v in scored]
    languages = [v["metrics"]["language"] for v in scored if "language" in v["metrics"]]
    recs = [episodes[v["episode_id"]] for v in verifications if v["episode_id"] in episodes]
    prompt = sum(r["usage"]["prompt_tokens"] for r in recs)
    completion = sum(r["usage"]["completion_tokens"] for r in recs)
    latencies = sorted(r["latency_ms"] for r in recs)
    ci = bootstrap_ci(weights, f"{seed}:{family}")
    return {
        "n": len(verifications),
        "scored": len(scored),
        "errors": errors,
        "mean_weight": statistics.fmean(weights) if weights else None,
        "ci95": list(ci) if ci else None,
        "mean_weight_errors_as_zero": (sum(weights) / len(verifications)) if verifications else None,
        "hard_constraints": {
            "fabrication": sum(HardConstraint.FABRICATION.value in h for h in hard),
            "rule_break": sum(HardConstraint.RULE_BREAK.value in h for h in hard),
            "unannounced_operator_action": sum(HardConstraint.UNANNOUNCED.value in h for h in hard),
            "episodes_with_any": sum(bool(h) for h in hard),
        },
        "answered_rate": (sum(v["answer_status"] in ANSWERED for v in scored) / len(scored)) if scored else None,
        "contract_validity": {"rate": (sum(bool(v["contract_ok"]) for v in scored) / len(scored)) if scored else None,
                              "n": len(scored)},
        "language_check": {"rate": statistics.fmean(languages) if languages else None, "n": len(languages)},
        "verifier_errors": sum(v.get("weight_basis") == "verifier-error" for v in scored),
        "protocol_errors": sum(len(r.get("protocol_errors") or []) for r in recs),
        "rounds_mean": statistics.fmean(r["rounds"] for r in recs) if recs else None,
        "tokens": {"prompt_total": prompt, "completion_total": completion,
                   "prompt_mean": prompt / len(recs) if recs else None,
                   "completion_mean": completion / len(recs) if recs else None},
        "latency_ms": {"mean": statistics.fmean(latencies) if latencies else None,
                       "p50": _percentile(latencies, 0.5) if latencies else None,
                       "p95": _percentile(latencies, 0.95) if latencies else None},
    }


def build_summary(out: Path, provenance: Mapping[str, Any], label: str) -> dict:
    verifications = read_jsonl_latest(out / VERIFICATION_FILE)
    episodes = read_jsonl_latest(out / EPISODES_FILE)
    seed = str(provenance.get("seed", 0))
    by_family: dict[str, list[dict]] = collections.defaultdict(list)
    for v in verifications.values():
        by_family[v["family"]].append(v)
    families = {f: _family_stats(f, sorted(vs, key=lambda v: v["episode_id"]), episodes, seed)
                for f, vs in sorted(by_family.items())}
    everything = sorted(verifications.values(), key=lambda v: v["episode_id"])
    overall = _family_stats("overall", everything, episodes, seed)
    means = [f["mean_weight"] for f in families.values() if f["mean_weight"] is not None]
    overall["macro_mean_weight"] = statistics.fmean(means) if means else None
    return {
        "schema": SUMMARY_SCHEMA,
        "label": label,
        "reduction": f"{COS_V1.name}@{COS_V1.version}",
        "reduction_sha256": COS_V1.policy_sha256,
        "weight_definition": "weight of the answer step (the step carrying the family effect); 0 when the episode "
                             "holds any hard constraint or no answer step",
        "provenance": dict(provenance),
        "families": families,
        "overall": overall,
        "per_episode": {v["episode_id"]: {"family": v["family"], "status": v["status"], "weight": v["weight"],
                                          "hard": v["hard"]} for v in everything},
        "files": {"episodes": EPISODES_FILE, "verification": VERIFICATION_FILE, "provenance": PROVENANCE_FILE},
    }


# --- the batch ------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BatchConfig:
    out: Path
    concurrency: int = 1
    runner: RunnerConfig = RunnerConfig()


def run_batch(store: SnapshotStore, chosen: Sequence[Chosen], adapter: ModelAdapter, system: SystemPrompt,
              oracle, config: BatchConfig, provenance: Mapping[str, Any], label: str,
              progress=lambda line: None) -> dict:
    out = config.out
    out.mkdir(parents=True, exist_ok=True)
    check_resume(out, provenance)
    (out / PROVENANCE_FILE).write_text(json.dumps(dict(provenance), indent=2, sort_keys=True) + "\n")
    done = {eid for eid, v in read_jsonl_latest(out / VERIFICATION_FILE).items() if v["status"] == "ok"}
    todo = [c for c in chosen if c.spec.episode_id not in done]
    progress(f"{len(chosen)} episodes selected, {len(chosen) - len(todo)} already complete, {len(todo)} to run")
    for snapshot_id in {c.spec.snapshot_id for c in todo}:  # warm the shared store before threads start
        store.get(snapshot_id).tool_contract()
    runner = ModelEpisodeRunner(adapter, system, config.runner)
    lock = threading.Lock()
    run_id = str(provenance.get("started_at", ""))

    def one(item: Chosen) -> tuple[dict, dict]:
        spec = item.spec
        snapshot = store.get(spec.snapshot_id)
        ledger_path = out / "ledgers" / f"{sha256_hex(spec.episode_id.encode())[:16]}.jsonl"
        ledger_path.unlink(missing_ok=True)  # a retry is a fresh tenancy
        servers, _ = episode_servers(snapshot, DojoLedger(ledger_path, forbidden_roots=(store.root,)), oracle)
        result = runner.run_traced(spec, servers)
        return _episode_record(item, result, run_id), _verification_record(item, result, snapshot, oracle)

    finished = 0
    interrupted = False
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, config.concurrency))
    futures = {pool.submit(one, c): c for c in todo}
    try:
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                episode_record, verification = future.result()
            except DojoError as error:
                progress(f"[abort] {item.spec.episode_id}: {error.render()}")
                raise
            _jsonl_append(out / EPISODES_FILE, lock, episode_record)
            _jsonl_append(out / VERIFICATION_FILE, lock, verification)
            finished += 1
            weight = verification["weight"]
            progress(f"[{finished}/{len(todo)}] {item.spec.episode_id} {verification['status']} "
                     f"weight={'-' if weight is None else f'{weight:.3f}'} hard={','.join(verification['hard']) or '-'}")
    except KeyboardInterrupt:
        interrupted = True
        for future in futures:
            future.cancel()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    summary = build_summary(out, provenance, label)
    (out / SUMMARY_FILE).write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    summary["_interrupted"] = interrupted
    return summary


# --- compare --------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Arm:
    label: str
    path: Path
    summary: Mapping[str, Any]

    def weights(self, family: str | None, ids: Iterable[str]) -> dict[str, float]:
        table = self.summary["per_episode"]
        return {i: float(table[i]["weight"]) for i in ids if i in table and table[i]["status"] == "ok"
                and (family is None or table[i]["family"] == family)}


def load_arm(path: Path) -> Arm:
    file = Path(path)
    if file.is_dir():
        file = file / SUMMARY_FILE
    try:
        summary = json.loads(file.read_text())
    except (OSError, ValueError) as error:
        raise CompareError(f"cannot read {file}: {error}") from None
    if summary.get("schema") != SUMMARY_SCHEMA:
        raise CompareError(f"{file} is not a {SUMMARY_SCHEMA} summary")
    return Arm(str(summary.get("label") or file.parent.name), file, summary)


def paired_bootstrap(diffs: Sequence[float], seed: str):
    """`exam.controller.PairedDifference` over the mean of per-episode weight differences."""
    from exam.controller import PairedDifference

    n = len(diffs)
    if n == 0:
        return PairedDifference(0, 0.0, -1.0, 1.0, None)
    delta = sum(diffs) / n
    if n < 2:
        return PairedDifference(n, delta, -1.0, 1.0, None)  # one pair cannot exclude zero
    lo, hi = bootstrap_ci(diffs, seed)  # type: ignore[misc]
    return PairedDifference(n, delta, lo, hi, None)


def compare_arms(arms: Sequence[Arm]) -> dict:
    if len(arms) < 2:
        raise CompareError("compare needs at least two summaries")
    ok_ids = [{i for i, e in a.summary["per_episode"].items() if e["status"] == "ok"} for a in arms]
    all_ids = [set(a.summary["per_episode"]) for a in arms]
    shared = set.intersection(*ok_ids)
    families = sorted({arms[0].summary["per_episode"][i]["family"] for i in shared})
    roots = {a.summary["provenance"].get("snapshot_root_sha256") for a in arms}
    result: dict[str, Any] = {
        "arms": [a.label for a in arms],
        "warnings": [],
        "episodes_in_common": len(shared),
        "dropped_for_errors_or_absence": len(set.union(*all_ids)) - len(shared),
        "rows": [],
    }
    if len(roots) > 1:
        result["warnings"].append("the arms ran over different snapshot roots; only shared episode ids are paired")
    if len({a.summary["provenance"].get("system_prompt_sha256") for a in arms}) > 1:
        result["warnings"].append("the arms used different system prompts; they were not run through the same broker")
    for family in [*families, None]:
        ids = sorted(i for i in shared if family is None or arms[0].summary["per_episode"][i]["family"] == family)
        row: dict[str, Any] = {"family": family or "ALL", "n": len(ids), "arms": {}, "pairs": []}
        per_arm = []
        for arm in arms:
            w = arm.weights(family, ids)
            values = [w[i] for i in ids]
            ci = bootstrap_ci(values, f"compare:{family}:{arm.label}")
            hard = [arm.summary["per_episode"][i]["hard"] for i in ids]
            row["arms"][arm.label] = {
                "mean": statistics.fmean(values) if values else None, "ci95": list(ci) if ci else None,
                "fabrication": sum(HardConstraint.FABRICATION.value in h for h in hard),
                "rule_break": sum(bool({HardConstraint.RULE_BREAK.value, HardConstraint.UNANNOUNCED.value} & set(h)) for h in hard),
            }
            per_arm.append(w)
        for later in range(1, len(arms)):
            for earlier in range(later):
                diffs = [per_arm[later][i] - per_arm[earlier][i] for i in ids]
                pd = paired_bootstrap(diffs, f"compare:{family}:{arms[later].label}:{arms[earlier].label}")
                row["pairs"].append({"label": f"{arms[later].label} - {arms[earlier].label}", **pd.to_dict()})
        result["rows"].append(row)
    return result


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def render_compare(result: Mapping[str, Any]) -> str:
    labels = result["arms"]
    lines = [f"Paired comparison over {result['episodes_in_common']} shared episodes "
             f"({result['dropped_for_errors_or_absence']} dropped: absent in an arm or endpoint error)"]
    lines += [f"WARNING: {w}" for w in result["warnings"]]
    lines += ["", "Mean cos-v1 weight [95% bootstrap CI]; fab = episodes with fabrication, rule = rule-break/unannounced", ""]
    width = max(22, *(len(label) for label in labels))
    lines.append(f"{'family':<20} {'n':>4}  " + "  ".join(f"{label:<{width + 22}}" for label in labels))
    for row in result["rows"]:
        cells = []
        for label in labels:
            arm = row["arms"][label]
            ci = arm["ci95"]
            interval = f"[{_fmt(ci[0])},{_fmt(ci[1])}]" if ci else "[-,-]"
            cells.append(f"{_fmt(arm['mean'])} {interval} fab={arm['fabrication']} rule={arm['rule_break']}".ljust(width + 22))
        lines.append(f"{row['family']:<20} {row['n']:>4}  " + "  ".join(cells))
    lines += ["", "Paired difference of means (first minus second), 95% bootstrap interval", ""]
    lines.append(f"{'family':<20} {'n':>4}  {'pair':<{2 * width + 3}} {'delta':>7}  {'interval':<18} verdict")
    for row in result["rows"]:
        for pair in row["pairs"]:
            interval = f"[{pair['lower']:+.3f},{pair['upper']:+.3f}]"
            lines.append(f"{row['family']:<20} {row['n']:>4}  {pair['label']:<{2 * width + 3}} {pair['delta']:>+7.3f}  "
                         f"{interval:<18} {pair['verdict']}")
    lines += ["", "'better' needs an interval above zero; 'no significant difference' is not equality."]
    return "\n".join(lines)


# --- CLI --------------------------------------------------------------------------------------------------------------------


def build_adapter(args: argparse.Namespace) -> ModelAdapter:
    secret = load_secret(env=args.key_env, file=args.key_file)
    policy = HttpPolicy(timeout_s=args.timeout, max_attempts=args.attempts, backoff_s=args.backoff)
    temperature = None if args.omit_temperature else args.temperature
    if args.model_kind == "openai":
        base = args.base_url or (args.base_url_file.read_text().strip() if args.base_url_file else None)
        if not base:
            raise DojoError("the openai adapter needs --base-url or --base-url-file",
                            help="the serve pod URL is in ~/Library/Application Support/roman-akhs/secrets/lilith-base-url")
        if not args.model:
            raise DojoError("the openai adapter needs --model", help="the Lilith serve pod serves the model name `lilith`")
        return OpenAIChatAdapter(base, args.model, secret, temperature=temperature, max_tokens=args.max_tokens, policy=policy)
    base = args.base_url or (args.base_url_file.read_text().strip() if args.base_url_file else ANTHROPIC_BASE_URL)
    return AnthropicAdapter(args.model or DEFAULT_ANTHROPIC_MODEL, secret, base_url=base, temperature=temperature,
                            max_tokens=args.max_tokens, policy=policy)


def _add_run_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", type=Path, default=None, help="snapshot root (default: $LILITH_SNAPSHOT_ROOT)")
    p.add_argument("--repo", type=Path, default=REPO_ROOT, help="local akh-medu clone: ancestry oracle and provenance")
    p.add_argument("--illumos-pack", type=Path, default=None)
    p.add_argument("--corrections", type=Path, default=None,
                   help="C2 corrections.jsonl (c2/harvest_sessions.py); select them with --family correction")
    p.add_argument("--family", default="all", help="'all' or a comma list of family names")
    p.add_argument("--split", required=True, choices=["sealed", "dev", "train", "all"])
    p.add_argument("--limit", type=int, default=None, help="episodes per family")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--model-kind", choices=["openai", "anthropic"], required=True)
    p.add_argument("--base-url", default=None)
    p.add_argument("--base-url-file", type=Path, default=None)
    p.add_argument("--model", default=None)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--key-env", default=None, help="name of the environment variable holding the API key")
    src.add_argument("--key-file", type=Path, default=None, help="file (mode 0600) holding the API key")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--omit-temperature", action="store_true", help="do not send `temperature` (some models reject it)")
    p.add_argument("--max-tokens", type=int, default=4096, help="completion tokens per model call")
    p.add_argument("--max-rounds", type=int, default=20, help="model turns per episode")
    p.add_argument("--episode-token-cap", type=int, default=64_000, help="completion tokens per episode")
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--attempts", type=int, default=4, help="HTTP attempts per model call (429/5xx/transport)")
    p.add_argument("--backoff", type=float, default=2.0)
    p.add_argument("--label", default=None, help="name of this arm in compare tables (default: model id)")
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="select episodes, size the prompt, call no model")


def cmd_run(args: argparse.Namespace) -> int:
    root = args.root or Path(os.environ.get("LILITH_SNAPSHOT_ROOT", RECORDER_DEFAULT_ROOT))
    store = SnapshotStore(root)
    oracle = GitAncestryOracle(args.repo)
    pack = load_illumos_pack(args.illumos_pack) if args.illumos_pack else ()
    corrections = load_corrections(args.corrections) if args.corrections else ()
    families = parse_families(args.family)
    chosen, selection = select_episodes(store, oracle, families, args.split, args.limit, args.seed, pack, corrections)
    system = compose_system_prompt(args.repo)
    floors_mod.load_floors()  # refuses floors that no longer hash to their pin
    if args.dry_run:
        print(json.dumps(dry_run_report(store, chosen, selection, system, oracle), indent=2, sort_keys=True))
        return 0
    if args.out is None:
        raise DojoError("--out is required", help="give a fresh directory, or an earlier run's directory to resume it")
    if args.key_env is None and args.key_file is None:
        raise DojoError("no key source", help="pass --key-env NAME or --key-file PATH")
    adapter = build_adapter(args)
    provenance = {
        "schema": "lilith/dojo-run@1",
        "runner": RUNNER_VERSION,
        "dojo": DOJO_VERSION,
        **git_provenance(args.repo, args.allow_dirty),
        "snapshot_root_sha256": snapshot_root_sha256(store),
        "snapshots": len(store.ids),
        "floors_sha256": floors_mod.FLOORS_V1_SHA256,
        "system_prompt_sha256": system.sha256,
        "prompt_sha256": prompt_bundle_sha256(system),
        "prompt_sources": dict(system.sources),
        "model_kind": adapter.kind,
        "model_id": adapter.model_id,
        "endpoint_host": adapter.endpoint_host,
        "temperature": None if args.omit_temperature else args.temperature,
        "max_tokens": args.max_tokens,
        "max_rounds": args.max_rounds,
        "episode_token_cap": args.episode_token_cap,
        "seed": args.seed,
        "split": args.split,
        "families": [f.value for f in families],
        "limit": args.limit,
        "selection": selection,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    config = BatchConfig(args.out, args.concurrency,
                         RunnerConfig(max_rounds=args.max_rounds, episode_token_cap=args.episode_token_cap))
    summary = run_batch(store, chosen, adapter, system, oracle, config, provenance,
                        args.label or adapter.model_id, progress=lambda line: print(line, file=sys.stderr, flush=True))
    interrupted = summary.pop("_interrupted", False)
    print(render_summary(summary))
    if interrupted:
        return 130
    return 1 if summary["overall"]["errors"] else 0


def dry_run_report(store: SnapshotStore, chosen: Sequence[Chosen], selection: Mapping[str, Any],
                   system: SystemPrompt, oracle) -> dict:
    """Sizes the fixed prompt (system + tool list) with a throwaway ledger. chars/4 is a rough token count."""
    tools_chars = None
    names: list[str] = []
    if chosen:
        from dojo.runner import Broker

        with tempfile.TemporaryDirectory() as tmp:
            snapshot = store.get(chosen[0].spec.snapshot_id)
            servers, _ = episode_servers(snapshot, DojoLedger(Path(tmp) / "l.jsonl", forbidden_roots=(store.root,)), oracle)
            broker = Broker(servers)
            specs = broker.tool_specs()
            tools_chars = sum(len(json.dumps([s.name, s.description, dict(s.input_schema)])) for s in specs)
            names = [s.name for s in specs]
    return {
        "episodes": len(chosen),
        "selection": dict(selection),
        "system_prompt_sha256": system.sha256,
        "prompt_sha256": prompt_bundle_sha256(system),
        "system_prompt_chars": len(system.text),
        "tools": {"count": len(names), "chars": tools_chars, "longest_name": max(map(len, names), default=0)},
        "fixed_prompt_tokens_estimate": (len(system.text) + (tools_chars or 0)) // 4,
        "note": "tokens are chars/4; every model round re-sends the fixed prompt plus the growing tool results",
    }


def render_summary(summary: Mapping[str, Any]) -> str:
    p = summary["provenance"]
    lines = [f"baseline {summary['label']}  model={p['model_id']} host={p['endpoint_host']} "
             f"git={p['git_sha'][:12]}{'+dirty' if p['git_dirty'] else ''} prompt={p['system_prompt_sha256'][:12]}",
             f"{'family':<20} {'n':>4} {'err':>3} {'mean':>6} {'ci95':<15} {'fab':>3} {'rule':>4} {'contract':>8} {'lang':>5} "
             f"{'tok_out':>8} {'lat_p50s':>8}"]
    rows = [*summary["families"].items(), ("OVERALL", summary["overall"])]
    for name, f in rows:
        ci = f["ci95"]
        interval = f"[{_fmt(ci[0])},{_fmt(ci[1])}]" if ci else "-"
        hc = f["hard_constraints"]
        p50 = f["latency_ms"]["p50"]
        lines.append(f"{name:<20} {f['n']:>4} {f['errors']:>3} {_fmt(f['mean_weight']):>6} {interval:<15} "
                     f"{hc['fabrication']:>3} {hc['rule_break'] + hc['unannounced_operator_action']:>4} "
                     f"{_fmt(f['contract_validity']['rate']):>8} {_fmt(f['language_check']['rate']):>5} "
                     f"{f['tokens']['completion_total']:>8} {'-' if p50 is None else f'{p50 / 1000:.1f}':>8}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lilith CoS dojo: model-driven episode runner, baselines, compare")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a model over dojo episodes and write a baseline")
    _add_run_arguments(run)
    run.add_argument("--out", type=Path, default=None, help="output directory (an existing one resumes)")
    cmp = sub.add_parser("compare", help="paired per-family table over baseline summaries")
    cmp.add_argument("summaries", nargs="+", type=Path, help="baseline-summary.json (or its directory), reference first")
    cmp.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return cmd_run(args)
        result = compare_arms([load_arm(p) for p in args.summaries])
        print(json.dumps(result, indent=2, sort_keys=True) if args.json else render_compare(result))
        return 0
    except DojoError as error:
        print(error.render(), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
