"""Resumable driver: `python3 -m program.generate run|report` from `scripts/lilith`.

    python3 -m program.generate run --out ~/lilith-private/program-20261003/generated --workers 4
    python3 -m program.generate report --out ~/lilith-private/program-20261003/generated

Output stays under `~/lilith-private/program-20261003/` (refused elsewhere unless `--allow-any-out`, used by tests).
Re-running skips finished seeds (progress/<kind>.jsonl); `--retry-excluded` retries seeds that exhausted their attempts.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Sequence

from . import guidance, lineage, sparring, variation
from .acp_worker import AcpConfig, AcpTeacher, Teacher, run_pool
from .checks import CitationResolver, LanguageProfiles
from .records import KINDS, PRIVATE_ROOT_NAME, RunDir, dump_jsonl, load_jsonl
from .tasking import Context, Outcome, Task

REPO_ROOT = Path(__file__).resolve().parents[4]
DRILLS = REPO_ROOT / "data" / "phase51" / "character" / "kata-drills.jsonl"
SPARRING_SEEDS = REPO_ROOT / "data" / "phase51" / "character" / "sparring-green.jsonl"
DEFAULT_OUT = Path.home() / "lilith-private" / PRIVATE_ROOT_NAME / "generated"
DEFAULT_SNAPSHOTS = Path.home() / "Library" / "Application Support" / "akh-medu" / "lilith" / "cos-snapshots"
DEFAULT_GATE = Path.home() / "ws" / "illumos" / "repos" / "illumos" / "illumos-gate"
MAX_WORKERS = 4


def language_profiles(drills: Path = DRILLS, sparring_seeds: Path = SPARRING_SEEDS) -> LanguageProfiles:
    texts: dict[str, list[str]] = collections.defaultdict(list)
    for row in load_jsonl(drills):
        texts[row["lang"]] += [row["situation"], row["move_chosen"]]
    if sparring_seeds.is_file():
        for row in load_jsonl(sparring_seeds):
            texts[row["lang"]] += [t["text"] for t in row["dialogue"]]
    return LanguageProfiles.from_texts(texts)


class Progress:
    def __init__(self, log: Callable[[str], None]):
        self.lock = threading.Lock()
        self.counts: collections.Counter = collections.Counter()
        self.failed: collections.Counter = collections.Counter()
        self.log = log
        self.started = time.monotonic()

    def record(self, outcome: Outcome) -> None:
        with self.lock:
            self.counts[(outcome.kind, outcome.status)] += 1
            for name in outcome.failed_checks:
                self.failed[(outcome.kind, name)] += 1
            done = sum(self.counts.values())
        self.log(f"[{time.monotonic() - self.started:7.0f}s] #{done} {outcome.kind} {outcome.status}"
                 + (f" failed={','.join(outcome.failed_checks)}" if outcome.failed_checks else "")
                 + (f" detail={outcome.detail}" if outcome.detail else ""))


def run_tasks(tasks: Sequence[Task], make_teacher: Callable[[], Teacher], workers: int, progress: Progress) -> None:
    def work(teacher: Teacher, task: object) -> None:
        assert isinstance(task, Task)
        progress.record(task.run(teacher))

    def on_error(task: object, error: Exception) -> None:
        progress.log(f"ERROR {getattr(task, 'kind', '?')} {getattr(task, 'key', '?')}: {type(error).__name__}: {str(error)[:200]}")

    run_pool(list(tasks), make_teacher, work, min(workers, MAX_WORKERS), on_error)


def admitted(ctx: Context, kind: str, generator: str | None = None) -> int:
    rows = ctx.run.read(f"{kind}.jsonl")
    return sum(1 for r in rows if generator is None or r["source_ref"]["generator"] == generator)


def run_guidance_waves(ctx: Context, candidates: Sequence[guidance.Candidate], quota: int, make_teacher: Callable[[], Teacher],
                       workers: int, progress: Progress, first_wave_extra: Sequence[Task] = ()) -> None:
    """Run guidance candidates in waves sized to the remaining quota, so generation stops at the quota."""
    extra = list(first_wave_extra)
    while True:
        need = quota - admitted(ctx, guidance.KIND)
        tasks = guidance.plan(ctx, candidates)
        if need <= 0 or (not tasks and not extra):
            if extra:
                run_tasks(extra, make_teacher, workers, progress)
            return
        run_tasks(extra + tasks[:need], make_teacher, workers, progress)
        extra = []
        progress.log(f"guidance admitted {admitted(ctx, guidance.KIND)}/{quota}")


def write_pack(ctx: Context) -> int:
    entries = guidance.pack_entries(ctx.run.read("guidance.jsonl"))
    dump_jsonl(ctx.run.out / "illumos-pack.jsonl", entries)
    return len(entries)


def run(args: argparse.Namespace, make_teacher: Callable[[], Teacher] | None = None) -> int:
    out = Path(args.out).expanduser()
    private = Path.home() / "lilith-private" / PRIVATE_ROOT_NAME
    if not args.allow_any_out and private not in (out, *out.parents):
        print(f"refusing --out {out}: generated data stays under {private}", file=sys.stderr)
        return 2
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    unknown = sorted(set(kinds) - set(KINDS))
    if unknown:
        print(f"unknown kinds {unknown}; choose from {list(KINDS)}", file=sys.stderr)
        return 2
    log = lambda line: print(line, file=sys.stderr, flush=True)  # noqa: E731
    ctx = Context(RunDir(out), language_profiles(), CitationResolver(Path(args.gate).expanduser() if args.gate else None),
                  REPO_ROOT, Path(args.snapshots).expanduser(), args.snapshot_cutoff, retry_excluded=args.retry_excluded)
    config = AcpConfig(binary=Path(args.acp_bin).expanduser()) if args.acp_bin else AcpConfig()
    make_teacher = make_teacher or (lambda: AcpTeacher(config))
    progress = Progress(log)
    workers = min(args.workers, MAX_WORKERS)

    first: list[Task] = []
    if "variation" in kinds:
        first += variation.plan(ctx, DRILLS)
    if "sparring" in kinds:
        first += sparring.plan(ctx)
    if "lineage" in kinds:
        first += lineage.plan(ctx, DRILLS, out / "guidance.jsonl", origins=("kata-drill",))
    log(f"planned {len(first)} fixed tasks; guidance in waves; workers={workers}")
    if "guidance" in kinds:
        candidates = guidance.select_candidates(ctx.snapshot_root, ctx.snapshot_cutoff, args.guidance_quota)
        log(f"guidance candidates: {len(candidates)} (snapshot {candidates[0].snapshot_id if candidates else '-'})")
        run_guidance_waves(ctx, candidates, args.guidance_quota, make_teacher, workers, progress, first)
    else:
        run_tasks(first, make_teacher, workers, progress)
    if "lineage" in kinds and (out / "guidance.jsonl").is_file():
        run_tasks(lineage.plan(ctx, DRILLS, out / "guidance.jsonl", origins=("guidance",)), make_teacher, workers, progress)
    if "guidance" in kinds:
        log(f"illumos-pack.jsonl: {write_pack(ctx)} entries")
    print(json.dumps(report(ctx.run), indent=1, ensure_ascii=False))
    return 0


def report(run_dir: RunDir) -> dict:
    out: dict = {"admitted": {}, "excluded": {}, "excluded_by_check": {}, "by_stage": {}}
    for kind in KINDS:
        rows, excluded = run_dir.read(f"{kind}.jsonl"), run_dir.read(f"{kind}.excluded.jsonl")
        out["admitted"][kind] = len(rows)
        out["excluded"][kind] = len(excluded)
        counter: collections.Counter = collections.Counter()
        for row in excluded:
            counter.update(row["verification"]["checks_failed"])
        out["excluded_by_check"][kind] = dict(counter)
        for row in rows:
            out["by_stage"][row["stage"]] = out["by_stage"].get(row["stage"], 0) + 1
    gen = collections.Counter(r["source_ref"]["generator"] for r in run_dir.read("lineage.jsonl"))
    out["lineage_by_generator"] = dict(gen)
    guid = run_dir.read("guidance.jsonl") + run_dir.read("guidance.excluded.jsonl")
    methods: collections.Counter = collections.Counter()
    for row in guid:
        for res in row["provenance"].get("citation_resolutions", []):
            methods[(res["method"], res["ok"])] += 1
    out["citation_resolutions"] = {f"{m}:{'ok' if ok else 'fail'}": n for (m, ok), n in sorted(methods.items())}
    out["guidance_subsystems"] = dict(collections.Counter(r["provenance"].get("subsystem") for r in run_dir.read("guidance.jsonl")))
    out["variation_by_variant"] = dict(collections.Counter(r["provenance"]["variant"] for r in run_dir.read("variation.jsonl")))
    out["errors_pending"] = {k: sum(1 for line in run_dir.progress(k).values() if line["status"] == "error") for k in KINDS}
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="program.generate")
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="generate (resumable)")
    r.add_argument("--out", default=str(DEFAULT_OUT))
    r.add_argument("--kinds", default=",".join(KINDS))
    r.add_argument("--workers", type=int, default=MAX_WORKERS, help=f"concurrent teacher sessions (capped at {MAX_WORKERS})")
    r.add_argument("--guidance-quota", type=int, default=guidance.QUOTA)
    r.add_argument("--snapshots", default=str(DEFAULT_SNAPSHOTS), help="cos-snapshots root (read-only)")
    r.add_argument("--snapshot-cutoff", default="20261002T000000Z")
    r.add_argument("--gate", default=str(DEFAULT_GATE), help="local illumos-gate checkout for citation resolution")
    r.add_argument("--acp-bin", default=None, help="akh-acp binary (default: the pinned main build in roman-akhs/bin)")
    r.add_argument("--retry-excluded", action="store_true")
    r.add_argument("--allow-any-out", action="store_true", help=argparse.SUPPRESS)
    s = sub.add_parser("report", help="counts and exclusions of a run directory")
    s.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args)
    print(json.dumps(report(RunDir(Path(args.out).expanduser())), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
