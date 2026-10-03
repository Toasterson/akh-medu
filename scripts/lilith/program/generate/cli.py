"""Resumable driver: `python3 -m program.generate run|report` from `scripts/lilith`.

    python3 -m program.generate run --out ~/lilith-private/program-20261003/generated --akhs dojo-teacher
    python3 -m program.generate report --out ~/lilith-private/program-20261003/generated

Output stays under `~/lilith-private/program-20261003/` (refused elsewhere unless `--allow-any-out`, used by tests).
Re-running skips finished seeds (progress/<kind>.jsonl; error lines are dropped at start so those seeds are retried);
`--retry-excluded` retries seeds whose attempts all failed a structural check. Concurrency is the number of `--akhs`.
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
from .acp_worker import DEFAULT_ROOT, AcpTeacher, Teacher, akh_config, run_pool
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


DEFAULT_ORDER = ("guidance", "sparring", "lineage", "variation")  # Black-stage guidance first
STATUS_EVERY_S = 1200.0
MAX_CONSECUTIVE_ERRORS = 5  # tasks that exhausted their retries, in a row, before the run stops itself


class Progress:
    """Counts outcomes and retries, logs each outcome, prints an admitted/errors/throughput line every 20 minutes and
    asks the pool to stop after too many exhausted seeds in a row (a dead core must not be hammered)."""

    def __init__(self, log: Callable[[str], None], status: Callable[[], str] = lambda: "",
                 max_consecutive_errors: int = MAX_CONSECUTIVE_ERRORS, clock: Callable[[], float] = time.monotonic):
        self.lock = threading.Lock()
        self.counts: collections.Counter = collections.Counter()
        self.failed: collections.Counter = collections.Counter()
        self.error_kinds: collections.Counter = collections.Counter()
        self.retries: collections.Counter = collections.Counter()
        self.errored: set[tuple[str, str]] = set()  # (kind, key) that exhausted retries in this run
        self.consecutive_errors = 0
        self.max_consecutive_errors = max_consecutive_errors
        self.log = log
        self.status = status
        self.clock = clock
        self.started = clock()
        self._last_status = self.started

    @property
    def should_stop(self) -> bool:
        return self.consecutive_errors >= self.max_consecutive_errors

    def retry(self, kind: str, delay: float, attempt: int) -> None:
        with self.lock:
            self.retries[kind] += 1

    def record(self, outcome: Outcome) -> None:
        with self.lock:
            self.counts[(outcome.kind, outcome.status)] += 1
            for name in outcome.failed_checks:
                self.failed[(outcome.kind, name)] += 1
            if outcome.status == "error":
                self.consecutive_errors += 1
                self.error_kinds[outcome.extra.get("error_kind", "other")] += 1
                self.errored.add((outcome.kind, outcome.key))
            else:
                self.consecutive_errors = 0
            done = sum(self.counts.values())
            now = self.clock()
            due = now - self._last_status >= STATUS_EVERY_S
            if due:
                self._last_status = now
        elapsed = now - self.started
        self.log(f"[{elapsed:7.0f}s] #{done} {outcome.kind} {outcome.status}"
                 + (f" failed={','.join(outcome.failed_checks)}" if outcome.failed_checks else "")
                 + (f" detail={outcome.detail}" if outcome.detail else ""))
        if due:
            self.log(self.status_line())

    def status_line(self) -> str:
        with self.lock:
            elapsed = max(self.clock() - self.started, 1.0)
            finished = sum(n for (k, st), n in self.counts.items() if st != "error")
            errors, retries = dict(self.error_kinds), dict(self.retries)
        return (f"STATUS {self.status()} | finished {finished} ({finished / elapsed * 3600:.0f}/h) | errors by class "
                f"{errors} | retries {retries}")


def run_tasks(tasks: Sequence[Task], makers: Sequence[Callable[[], Teacher]], progress: Progress) -> None:
    def work(teacher: Teacher, task: object) -> None:
        assert isinstance(task, Task)
        progress.record(task.run(teacher))

    def on_error(task: object, error: Exception) -> None:
        progress.log(f"ERROR {getattr(task, 'kind', '?')} {getattr(task, 'key', '?')}: {type(error).__name__}: {str(error)[:200]}")

    run_pool(list(tasks), makers, work, on_error, stop=lambda: progress.should_stop)


def admitted(ctx: Context, kind: str, generator: str | None = None) -> int:
    rows = ctx.run.read(f"{kind}.jsonl")
    return sum(1 for r in rows if generator is None or r["source_ref"]["generator"] == generator)


def run_guidance_waves(ctx: Context, candidates: Sequence[guidance.Candidate], quota: int,
                       makers: Sequence[Callable[[], Teacher]], progress: Progress) -> None:
    """Run guidance candidates in waves sized to the remaining quota, so generation stops at the quota. Candidates that
    errored (retries exhausted) in this run are not planned again in it; a wave that cannot make progress ends the loop."""
    while not progress.should_stop:
        need = quota - admitted(ctx, guidance.KIND)
        tasks = [t for t in guidance.plan(ctx, candidates) if (guidance.KIND, t.key) not in progress.errored]
        if need <= 0 or not tasks:
            return
        run_tasks(tasks[:need], makers, progress)
        progress.log(f"guidance admitted {admitted(ctx, guidance.KIND)}/{quota}")


def write_pack(ctx: Context) -> int:
    entries = guidance.pack_entries(ctx.run.read("guidance.jsonl"))
    dump_jsonl(ctx.run.out / "illumos-pack.jsonl", entries)
    return len(entries)


def split_names(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def validate(args: argparse.Namespace) -> tuple[list[str], list[str]] | str:
    """(akhs, order) or an error message."""
    akhs, order = split_names(args.akhs), split_names(args.order)
    if not akhs or len(set(akhs)) != len(akhs):
        return f"--akhs needs one or more distinct akh names, got {akhs}"
    if len(akhs) > MAX_WORKERS:
        return f"at most {MAX_WORKERS} akhs (one worker each), got {len(akhs)}"
    unknown = sorted(set(order) - set(KINDS))
    if unknown or not order or len(set(order)) != len(order):
        return f"--order lists distinct kinds from {list(KINDS)}; got {order}"
    return akhs, order


def run(args: argparse.Namespace, make_teacher: Callable[[str], Teacher] | None = None) -> int:
    out = Path(args.out).expanduser()
    private = Path.home() / "lilith-private" / PRIVATE_ROOT_NAME
    if not args.allow_any_out and private not in (out, *out.parents):
        print(f"refusing --out {out}: generated data stays under {private}", file=sys.stderr)
        return 2
    checked = validate(args)
    if isinstance(checked, str):
        print(checked, file=sys.stderr)
        return 2
    akhs, order = checked
    log = lambda line: print(line, file=sys.stderr, flush=True)  # noqa: E731
    run_dir = RunDir(out)
    ctx = Context(run_dir, language_profiles(), CitationResolver(Path(args.gate).expanduser() if args.gate else None),
                  REPO_ROOT, Path(args.snapshots).expanduser(), args.snapshot_cutoff, retry_excluded=args.retry_excluded)
    for kind in KINDS:
        dropped = run_dir.drop_errors(kind)
        if dropped:
            log(f"progress/{kind}.jsonl: dropped {dropped} error lines (those seeds are retried)")
    progress = Progress(log, status=lambda: " ".join(f"{k}={admitted(ctx, k)}" for k in KINDS))
    if make_teacher is None:
        binary = Path(args.acp_bin).expanduser() if args.acp_bin else None
        missing = [a for a in akhs if not akh_config(a, binary).token_file.is_file()]
        if missing:
            print(f"no principal token file for {missing} (expected secrets/<akh>.key under {DEFAULT_ROOT})", file=sys.stderr)
            return 2
        make_teacher = lambda akh: AcpTeacher(akh_config(akh, binary), on_retry=progress.retry)  # noqa: E731
    makers = [(lambda akh=akh: make_teacher(akh)) for akh in akhs]
    log(f"akhs={akhs} (one worker each) order={order}")

    for kind in order:
        if progress.should_stop:
            break
        log(f"== {kind} ==")
        if kind == "guidance":
            candidates = guidance.select_candidates(ctx.snapshot_root, ctx.snapshot_cutoff, args.guidance_quota)
            log(f"guidance candidates: {len(candidates)} (snapshot {candidates[0].snapshot_id if candidates else '-'})")
            run_guidance_waves(ctx, candidates, args.guidance_quota, makers, progress)
            log(f"illumos-pack.jsonl: {write_pack(ctx)} entries")
        elif kind == "sparring":
            run_tasks(sparring.plan(ctx), makers, progress)
        elif kind == "lineage":
            run_tasks(lineage.plan(ctx, DRILLS, out / "guidance.jsonl"), makers, progress)
        elif kind == "variation":
            run_tasks(variation.plan(ctx, DRILLS), makers, progress)
    if progress.should_stop:
        log(f"STOPPED after {progress.consecutive_errors} exhausted seeds in a row; the core or the teacher akhs need attention")
    log(progress.status_line())
    print(json.dumps(report(ctx.run), indent=1, ensure_ascii=False))
    return 1 if progress.should_stop else 0


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
    r.add_argument("--akhs", default="dojo-teacher",
                   help=f"comma-separated teacher akhs, one worker (and one ACP session) each, at most {MAX_WORKERS}; token = secrets/<akh>.key")
    r.add_argument("--order", default=",".join(DEFAULT_ORDER), help="kinds to run, in order (default: guidance first)")
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
