# FrontierCode-Diamond-style agentic-coding harness for Tecton

A self-hostable benchmark that evaluates the akh **Tecton** (GLM via ollama.com
cloud, served by the running `akhomed` daemon over ACP on `tcp:127.0.0.1:8201`)
on **hard repository-editing tasks**, replicating the *methodology* of Cognition's
**FrontierCode Diamond** benchmark.

> FrontierCode itself is **private** (not downloadable). This harness does **not**
> use FrontierCode data. It reproduces FrontierCode's grading methodology on
> **public** SWE-bench Verified instances, which ship everything the methodology
> needs: a real GitHub issue, the base commit, the gold solution patch, and the
> held-out unit tests (`test_patch` / `FAIL_TO_PASS` / `PASS_TO_PASS`).

## Why this shape (Tecton is a "model", not a repo-editing agent)

Tecton's sandboxed file writes are confined to a scratch dir — it **cannot** edit
an arbitrary target repo. So we treat Tecton as a *model*: given the issue plus the
relevant source files in the prompt, it returns a **unified diff** in its answer.
The harness then:

1. checks out the repo at `base_commit` in a temp dir,
2. applies Tecton's patch,
3. runs the repo's tests + lint,
4. grades against the 6 FrontierCode dimensions.

This is the realistic division of labour: **Tecton produces the solution, the
harness verifies it.**

## The 6 grading dimensions (and how each maps here)

FrontierCode Diamond grades on 6 dimensions; some are **blockers** (hard gates).
A solution scores **zero unless ALL blockers pass**. We reproduce that gating in
`grader.Grade.finalize()`.

| # | FrontierCode dimension | Gate | How we evaluate it | Faithful / approximated |
|---|------------------------|------|--------------------|-------------------------|
| 1 | Behavioral correctness | **blocker** | `FAIL_TO_PASS` tests pass after Tecton's patch + the held-out `test_patch` are applied | **Faithful** — these are the injected unit tests |
| 2 | Regression safety | **blocker** | a sampled 25 of `PASS_TO_PASS` still pass | **Faithful** (sampled for runtime; configurable) |
| 3 | Mechanical cleanliness | non-blocker | run the repo's linter (ruff → flake8 → pyflakes) on changed files | Approximated — uses whatever linter the venv has |
| 4 | Test correctness (reverse-classical) | **blocker** | apply **only** the held-out `test_patch` to the **unpatched** base and confirm the injected tests **FAIL** (they must bite) | **Faithful in spirit** — see note below |
| 5 | Scope | **blocker** | diff file-count ≤ 10 and churn ≤ 600 lines, > 0 files | Approximated — semantic-locality reduced to size limits |
| 6 | Code quality | non-blocker | LLM rubric on idiomaticity/design | **Stubbed** (`grader.judge_stub`) — TODO, see below |

### Reverse-classical note (Dim 4)

FrontierCode grades the **agent's own** added tests with the reverse-classical
check. SWE-bench does not separate "agent-authored" from "held-out" tests — it
ships one canonical `test_patch`. We therefore run the reverse-classical gate on
that canonical held-out test patch: apply it alone to the clean base and require
the `FAIL_TO_PASS` tests to **fail**. This proves the tests genuinely exercise the
fix (the same property FrontierCode is after). If a future task-set includes
model-authored tests, `grader.reverse_classical()` can be pointed at them instead.

## Task set (10 hard public tasks)

Source: **SWE-bench Verified** (`princeton-nlp/SWE-bench_Verified` on HuggingFace,
a single ~2 MB parquet). We filter to the hardest difficulty bands (`>4 hours`,
`1-4 hours`) and bias toward lightweight, pip-installable repos so a self-hoster
can build the test environment without the heavy django/matplotlib conda stacks.

| instance_id | repo | difficulty | F2P |
|-------------|------|------------|-----|
| sympy__sympy-17630 | sympy/sympy | 1-4 hours | 2 |
| sympy__sympy-13852 | sympy/sympy | 1-4 hours | 1 |
| sympy__sympy-18199 | sympy/sympy | 1-4 hours | 1 |
| sympy__sympy-14248 | sympy/sympy | 1-4 hours | 2 |
| pytest-dev__pytest-6197 | pytest-dev/pytest | 1-4 hours | 2 |
| pytest-dev__pytest-10356 | pytest-dev/pytest | 1-4 hours | 1 |
| pylint-dev__pylint-8898 | pylint-dev/pylint | 1-4 hours | 1 |
| pydata__xarray-3993 | pydata/xarray | 1-4 hours | 2 |
| pydata__xarray-6992 | pydata/xarray | **>4 hours** | 12 |
| sphinx-doc__sphinx-7590 | sphinx-doc/sphinx | **>4 hours** | 1 |

Each task is stored **self-contained** as `tasks/<instance_id>.json` (problem,
base_commit, gold_patch, test_patch, FAIL_TO_PASS, PASS_TO_PASS). The gold patch is
used only by the harness (to pick which source files to show Tecton, and for the
`--dry-run-gold` oracle); **Tecton never sees it**.

## Layout

```
benchmarks/frontiercode/
  harness/
    extract_tasks.py   # download SWE-bench Verified parquet → tasks/*.json
    util.py            # checkout, patch extraction, apply, diff stats
    grader.py          # the 6-dimension grader + env build + reverse-classical
    runner.py          # orchestrates checkout → prompt → drive Tecton → grade
  tasks/               # 10 self-contained task JSONs (committed-ready)
  results/             # per-task result JSONs + summary.json + report.md
  report-template.md   # markdown report skeleton
  README.md            # this file
```

## Prerequisites

- The `akhomed` daemon must be running and serving `tecton` on `tcp:127.0.0.1:8201`
  (it is — do **not** restart it).
- Python 3.9+, with `pyarrow` + `huggingface_hub` (for `extract_tasks.py` only):
  ```
  pip3 install --user pyarrow huggingface_hub
  ```
  (`huggingface_hub` was already present; `pyarrow` was installed via
  `pip3 install --user pyarrow` during setup.)
- `git` and `patch` on PATH (both present).
- Network for the one-time SWE-bench download and the per-repo `git clone`
  (cached under `~/.cache/frontiercode-repos`).

## How to run

```bash
cd benchmarks/frontiercode

# 1. (re)build the task set — needs network once; idempotent
python3 harness/extract_tasks.py                # default curated 10
python3 harness/extract_tasks.py --all-hard     # all 45 hard instances
python3 harness/extract_tasks.py --ids sympy__sympy-17630

# 2a. validate the grader + test env WITHOUT the model (gold oracle):
python3 harness/runner.py --ids sympy__sympy-17630 --dry-run-gold

# 2b. fast smoke (drive Tecton, apply patch, score scope only — no test env):
python3 harness/runner.py --ids sympy__sympy-17630 --no-tests

# 3. full run against Tecton (drives the live daemon over ACP):
python3 harness/runner.py                       # all tasks
python3 harness/runner.py --ids sympy__sympy-17630 pytest-dev__pytest-6197

# results land in results/<instance_id>.json + results/summary.json
```

Useful flags: `--timeout <s>` (per-prompt ACP timeout, default 600),
`--keep` (retain temp workdirs for debugging), `--workspace <name>` (default
`tecton`).

## Expected results

Frontier models score **low** on Diamond (Opus 4.8 ≈ 13.4%, GPT-5.5 ≈ 6.3%). Tecton
is expected to score **at or near zero** — that is the documented, expected outcome.
The deliverable is a **faithful, working harness** plus a feasibility run that
captures *why* a task fails (no patch / bad patch / patch-doesn't-apply / test-infra),
not a competitive score.

## What's faithful vs approximated (summary)

- **Faithful:** blocker-gated scoring; behavioral correctness via injected unit
  tests; regression via PASS_TO_PASS; reverse-classical test-correctness gate;
  Tecton-produces-patch / harness-verifies split.
- **Approximated:** scope = size limits (not full semantic-locality); cleanliness =
  whatever linter the venv has; regression sampled to 25 tests for runtime.
- **Stubbed (TODO):** Dim 6 code-quality LLM rubric — `grader.judge_stub`. Wire a
  judge (e.g. a second akh, or a rubric prompt back to Tecton) returning
  `{"pass": bool, "detail": str}` and pass it as `judge=` to `grader.grade`.

## Known limitations / gotchas

- **Per-repo test envs are heavy.** SWE-bench's canonical harness uses pinned
  Docker images. We build a fresh venv + editable install per task, which works for
  pure-Python repos (sympy/pytest/pylint/xarray) but may fail to install older
  pinned builds. When the env can't build, blocker dimensions are marked
  **not-evaluable** and the task scores 0 with a clear infra note (never a false
  pass). This is intentional and surfaced in the report.
- The runner makes **two** checkouts per graded task (patched + a clean one for the
  reverse-classical base check) and **two** venvs. Expect several minutes per task.
- Repo clones are cached under `~/.cache/frontiercode-repos`; first run of each repo
  is slow.
```
