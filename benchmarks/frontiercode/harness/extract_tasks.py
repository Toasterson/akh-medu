#!/usr/bin/env python3
"""Extract a small curated task-set from SWE-bench Verified into self-contained JSON.

FrontierCode Diamond is private. We replicate its *methodology* (hard repo-edit
tasks, behavioral + regression + reverse-classical test-correctness + scope grading)
on public SWE-bench Verified instances, which ship the gold patch, the test patch
(injected unit tests / FAIL_TO_PASS), the base commit and the difficulty label.

This script downloads the Verified parquet from HuggingFace (single file, ~2 MB),
filters to the *hard* difficulty bands, and writes one self-contained task JSON per
selected instance into ../tasks/. Re-running is idempotent.

Fetch (network needed once):
    python3 harness/extract_tasks.py            # default curated 10
    python3 harness/extract_tasks.py --all-hard # every >4h / 1-4h instance
    python3 harness/extract_tasks.py --ids sympy__sympy-17630 pytest-dev__pytest-6197

If the HuggingFace download fails (intermittent LAN), retry the exact command; the
parquet is cached under ~/.cache/huggingface so subsequent runs are offline.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

HF_REPO = "princeton-nlp/SWE-bench_Verified"
HF_FILE = "data/test-00000-of-00001.parquet"
HERE = Path(__file__).resolve().parent
TASKS_DIR = HERE.parent / "tasks"

# Curated default: 10 hard tasks, biased to lightweight pip-installable repos
# (sympy / pytest / xarray / pylint) so a self-hoster can build the test env
# without the heavy django/matplotlib conda stacks. Mix of difficulty bands.
DEFAULT_IDS = [
    "sympy__sympy-17630",      # >4h-adjacent BlockMatrix ZeroMatrix; 1-line-ish, 2 F2P
    "sympy__sympy-13852",      # 1-4h polylog expansion; 1 F2P, single file
    "sympy__sympy-18199",      # 1-4h nthroot_mod; 1 F2P, single file
    "sympy__sympy-14248",      # 1-4h MatrixExpr printing; 2 F2P, 3 files
    "pytest-dev__pytest-6197", # 1-4h __init__ collection regression; 2 F2P
    "pytest-dev__pytest-10356", # 1-4h markers MRO; 1 F2P
    "pylint-dev__pylint-8898", # 1-4h bad-name-rgxs config; 1 F2P
    "pydata__xarray-3993",     # 1-4h integrate dim arg; 2 F2P
    "pydata__xarray-6992",     # >4h dataset index; 12 F2P (regression-heavy)
    "sphinx-doc__sphinx-7590", # >4h C++ user-defined literals; 1 F2P
]


def load_rows():
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow missing. Install: pip3 install --user pyarrow")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub missing. Install: pip3 install --user huggingface_hub")
    path = hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset")
    d = pq.read_table(path).to_pydict()
    n = len(d["instance_id"])
    return [{k: d[k][i] for k in d} for i in range(n)]


def to_task(r: dict) -> dict:
    """Project a SWE-bench row to our self-contained task schema."""
    return {
        "instance_id": r["instance_id"],
        "repo": r["repo"],
        "base_commit": r["base_commit"],
        "environment_setup_commit": r.get("environment_setup_commit") or r["base_commit"],
        "version": r.get("version"),
        "difficulty": r.get("difficulty"),
        "problem_statement": r["problem_statement"],
        "hints_text": r.get("hints_text", ""),
        # gold solution patch — used ONLY by the grader to locate edited files for
        # the context prompt and as an oracle for the optional gold-replay mode.
        # Tecton never sees this.
        "gold_patch": r["patch"],
        # injected unit tests (reverse-classical): these must PASS after Tecton's
        # patch and FAIL on the unpatched base. Applied by the harness, not Tecton.
        "test_patch": r["test_patch"],
        "FAIL_TO_PASS": json.loads(r["FAIL_TO_PASS"]),
        "PASS_TO_PASS": json.loads(r["PASS_TO_PASS"]),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all-hard", action="store_true",
                    help="select every >4h and 1-4h instance")
    ap.add_argument("--ids", nargs="*", help="explicit instance ids")
    ap.add_argument("--out", default=str(TASKS_DIR))
    args = ap.parse_args(argv)

    rows = load_rows()
    by_id = {r["instance_id"]: r for r in rows}

    if args.ids:
        wanted = args.ids
    elif args.all_hard:
        wanted = [r["instance_id"] for r in rows
                  if r.get("difficulty") in (">4 hours", "1-4 hours")]
    else:
        wanted = DEFAULT_IDS

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for iid in wanted:
        if iid not in by_id:
            print(f"  SKIP (not in dataset): {iid}", file=sys.stderr)
            continue
        task = to_task(by_id[iid])
        (out / f"{iid}.json").write_text(json.dumps(task, indent=2))
        print(f"  wrote {iid}  [{task['difficulty']}]  "
              f"F2P={len(task['FAIL_TO_PASS'])} repo={task['repo']}")
        written += 1
    print(f"\n{written} task(s) written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
