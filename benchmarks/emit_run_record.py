#!/usr/bin/env python3
"""Phase 50 W5 adapter: convert a harness's own results format into the
`harness-core` `RawTaskRow` JSON shape and call `bench-tool record-run`, which
does the real schema-validated, provenance-stamped write (ADR 086 §4) via
`harness_core::store::write_run`. Keeps the "what does a real run's
provenance need" logic in one place (Rust, `bench_tool.rs`) instead of
re-implementing it once per Python harness.

Two source formats are understood:
  * frontiercode: `benchmarks/frontiercode/results/*.json` (one file per
    instance_id, written by `runner.py`'s `_save()`), NOT `summary.json`
    (that's a roll-up, not per-task detail).
  * memory (locomo/longmemeval): `benchmarks/results_<label>.json` written by
    `run_memory_bench.py`'s `run()` — one row per (item, question); rows are
    grouped by "item" here into one task each, since ADR 086's `TaskResult`
    is one row per *task*, and the natural task unit for a memory benchmark
    is "one conversation/haystack item", not "one question".

Usage:
  python3 emit_run_record.py frontiercode --model glm-5.2 --provider-tag glm-5.2 \\
      --ids sympy__sympy-17630 sympy__sympy-13852
  python3 emit_run_record.py memory --model glm-5.2 --provider-tag glm-5.2 \\
      --results-json benchmarks/results_longmemeval-s6.json --harness longmemeval \\
      --task-set-dir benchmarks/longmemeval
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


def emit_frontiercode(model: str, provider_tag: str, ids: list[str] | None,
                      results_dir: Path, task_set_dir: Path, results_out: Path) -> Path:
    rows = []
    for f in sorted(results_dir.glob("*.json")):
        if f.name == "summary.json":
            continue
        rec = json.loads(f.read_text())
        iid = rec["instance_id"]
        if ids and iid not in ids:
            continue
        score = rec.get("score")
        model_stage = (rec.get("stages", {}) or {}).get("model", {}) or {}
        elapsed = model_stage.get("elapsed_s") or 0.0
        requests_used = model_stage.get("requests_used")
        if score is None and rec.get("note") and not rec.get("error"):
            # `--no-tests` scope-only smoke run: grading never ran at all
            # (not even a blocker verdict). This is genuinely ungraded, not a
            # capability failure — record as aborted so it can never be
            # mistaken for a real 0-scored FrontierCode attempt.
            outcome = "aborted"
            reason = rec["note"]
            score = 0.0
        elif score:
            # A real pass: all blocker dimensions evaluated and passed.
            outcome = "pass"
            score = 1.0
            reason = None
        else:
            # Every other path (checkout failed, no usable edit extracted,
            # patch didn't apply, or a graded blocker failed) is a genuine
            # FrontierCode-style failure — score 0, per the README's "never a
            # false pass" rule. This is the *expected* common case for a
            # small local model against Diamond-style tasks, not an error in
            # our harness.
            outcome = "fail"
            score = 0.0
            reason = rec.get("error")
        notes = rec.get("error") or (rec.get("grade") or {}).get("note") or None
        rows.append({
            "task_id": iid,
            "outcome": outcome,
            "abort_reason": reason if outcome == "aborted" else None,
            "score": score,
            "wall_time_secs": float(elapsed),
            "tokens_used": None,  # ACP session/update does not report token counts
            "requests_used": requests_used,
            "notes": notes,
        })
    if ids:
        missing = set(ids) - {r["task_id"] for r in rows}
        if missing:
            print(f"WARNING: no result file found for {sorted(missing)} — "
                  f"they will be ABSENT from this RunRecord, not silently scored", file=sys.stderr)
    results_out.write_text(json.dumps(rows, indent=2))
    return results_out


def emit_memory(model: str, provider_tag: str, results_json: Path, results_out: Path) -> Path:
    data = json.loads(results_json.read_text())
    by_item: dict[str, list[dict]] = {}
    for r in data["results"]:
        by_item.setdefault(str(r["item"]), []).append(r)
    rows = []
    for item_id, item_rows in by_item.items():
        aborted_rows = [r for r in item_rows if r.get("aborted")]
        scored_rows = [r for r in item_rows if not r.get("aborted")]
        if aborted_rows and not scored_rows:
            rows.append({
                "task_id": item_id,
                "outcome": "aborted",
                "abort_reason": aborted_rows[0]["aborted"],
                "score": 0.0,
                "wall_time_secs": 0.0,
                "tokens_used": None,
                "requests_used": None,
                "notes": f"{len(aborted_rows)} question(s) aborted by budget cap",
            })
            continue
        # One task = one conversation/haystack item; score = fraction of its
        # questions answered correctly (partial credit, matching TaskResult's
        # 0.0..=1.0 contract); pass iff ALL scored questions were correct AND
        # none were aborted by the budget cap.
        n = len(scored_rows)
        correct = sum(1 for r in scored_rows if r["correct"])
        score = correct / n if n else 0.0
        outcome = "pass" if (score == 1.0 and not aborted_rows) else "fail"
        wall_time = sum(r.get("elapsed", 0.0) for r in scored_rows)
        notes = f"{correct}/{n} questions correct"
        if aborted_rows:
            notes += f"; {len(aborted_rows)} additional question(s) aborted by budget cap"
        rows.append({
            "task_id": item_id,
            "outcome": outcome,
            "abort_reason": None,
            "score": score,
            "wall_time_secs": wall_time,
            "tokens_used": None,
            "requests_used": n,  # one ACP request per scored question
            "notes": notes,
        })
    results_out.write_text(json.dumps(rows, indent=2))
    return results_out


def call_bench_tool(harness: str, model: str, provider_tag: str, tasks_json: Path,
                    task_set_dir: Path, results_dir: Path, grading: str,
                    repo_dir: Path) -> None:
    cmd = [
        "cargo", "run", "--quiet", "--manifest-path",
        str(REPO_ROOT / "benchmarks" / "harness-core" / "Cargo.toml"),
        "--bin", "bench-tool", "--",
        "record-run",
        "--harness", harness,
        "--model", model,
        "--provider-tag", provider_tag,
        "--tasks-json", str(tasks_json),
        "--task-set-dir", str(task_set_dir),
        "--results-dir", str(results_dir),
        "--grading", grading,
        "--repo-dir", str(repo_dir),
    ]
    env = dict(os.environ)  # inherits caller's CARGO_TARGET_DIR if set
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["frontiercode", "memory"])
    ap.add_argument("--model", required=True, help="filesystem-safe model tag, e.g. glm-5.2")
    ap.add_argument("--provider-tag", required=True, help="the cloud API's own model tag")
    ap.add_argument("--grading", default="autonomous")
    ap.add_argument("--results-dir-out", default=str(REPO_ROOT / "benchmarks" / "results"))
    ap.add_argument("--repo-dir", default=str(REPO_ROOT))
    # frontiercode
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--fc-results-dir", default=str(HERE / "frontiercode" / "results"))
    ap.add_argument("--fc-task-set-dir", default=str(HERE / "frontiercode" / "tasks"))
    # memory
    ap.add_argument("--harness", help="locomo | longmemeval (memory kind only)")
    ap.add_argument("--results-json", help="path to results_<label>.json (memory kind only)")
    ap.add_argument("--task-set-dir", help="dataset dir to hash (memory kind only)")
    args = ap.parse_args(argv)

    with tempfile.TemporaryDirectory() as td:
        tasks_json = Path(td) / "tasks.json"
        if args.kind == "frontiercode":
            emit_frontiercode(args.model, args.provider_tag, args.ids,
                              Path(args.fc_results_dir), Path(args.fc_task_set_dir), tasks_json)
            harness = "frontiercode"
            task_set_dir = Path(args.fc_task_set_dir)
        else:
            if not (args.harness and args.results_json and args.task_set_dir):
                sys.exit("memory kind requires --harness --results-json --task-set-dir")
            emit_memory(args.model, args.provider_tag, Path(args.results_json), tasks_json)
            harness = args.harness
            task_set_dir = Path(args.task_set_dir)

        n = len(json.loads(tasks_json.read_text()))
        if n == 0:
            sys.exit(f"no task rows produced for {args.kind} — nothing to record")
        print(f"{n} task row(s) prepared for harness={harness} model={args.model}")

        call_bench_tool(harness, args.model, args.provider_tag, tasks_json,
                        task_set_dir, Path(args.results_dir_out), args.grading,
                        Path(args.repo_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
