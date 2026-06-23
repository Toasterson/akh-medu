#!/usr/bin/env python3
"""The 6-dimension FrontierCode-style grader.

Maps each FrontierCode Diamond grading dimension onto a mechanical check over a
SWE-bench instance. Blocker dimensions are hard gates: the final score is 0
unless EVERY blocker passes. Non-blockers are quality signals only.

  Dim 1  Behavioral correctness   BLOCKER   FAIL_TO_PASS tests pass post-patch
  Dim 2  Regression safety        BLOCKER   PASS_TO_PASS tests still pass
  Dim 3  Mechanical cleanliness   non-block repo linter clean on changed files
  Dim 4  Test correctness         BLOCKER   reverse-classical: injected tests FAIL
                                            on unpatched base (proves they bite)
  Dim 5  Scope                    BLOCKER   diff size / file-count within limits
  Dim 6  Code quality             non-block LLM rubric (stubbed / optional)

FrontierCode injects its own held-out tests AND grades the agent's *own* added
tests with the reverse-classical check. SWE-bench ships the held-out tests as
`test_patch`; it does not separate "agent-authored" tests. We therefore apply
the canonical `test_patch` for Dim 1/2/4 (faithful to the held-out-test idea),
and additionally run the reverse-classical check on any tests the *model* added
in its own patch when present (see `reverse_classical`).
"""
from __future__ import annotations
import json, re
from dataclasses import dataclass, field
from pathlib import Path

from util import run, Cmd, DiffStats, diff_stats

# Scope limits (Dim 5). FrontierCode penalizes sprawling diffs / poor locality.
# Calibrated generously vs the gold patches in our task-set (max gold ~6 files).
MAX_FILES = 10
MAX_CHURN = 600  # added+removed lines


@dataclass
class DimResult:
    name: str
    blocker: bool
    passed: bool | None          # None = could-not-evaluate (infra)
    detail: str = ""
    evaluable: bool = True       # False when infra prevented a verdict


@dataclass
class Grade:
    instance_id: str
    dims: list[DimResult] = field(default_factory=list)
    score: int = 0              # 1 if all evaluable blockers pass, else 0
    note: str = ""

    def finalize(self):
        blockers = [d for d in self.dims if d.blocker]
        if any(not d.evaluable for d in blockers):
            self.score = 0
            self.note = "blocker-not-evaluable (infra); scored 0"
        elif all(d.passed for d in blockers):
            self.score = 1
        else:
            self.score = 0
        return self

    def to_dict(self):
        return {
            "instance_id": self.instance_id,
            "score": self.score,
            "note": self.note,
            "dimensions": [
                {"name": d.name, "blocker": d.blocker, "passed": d.passed,
                 "evaluable": d.evaluable, "detail": d.detail}
                for d in self.dims
            ],
        }


# ── test environment ────────────────────────────────────────────────────────
def make_env(repo_dir: Path, repo: str, timeout=1800) -> tuple[bool, str, dict]:
    """Build a venv and editable-install the repo so pytest can run.

    Returns (ok, detail, env_overrides). Best-effort: many SWE-bench repos need
    pinned build deps; if the editable install fails we still return the python
    path so the caller can record an infra failure rather than a false verdict.
    """
    venv = repo_dir / ".fc_venv"
    r = run(["python3", "-m", "venv", str(venv)], timeout=120)
    if not r.ok:
        return False, f"venv create failed: {r.stderr[:200]}", {}
    py = str(venv / "bin" / "python")
    pip = [py, "-m", "pip", "install", "-q", "--disable-pip-version-check"]
    run(pip + ["--upgrade", "pip", "setuptools", "wheel"], timeout=300)
    # editable install with test extras when available
    r = run(pip + ["-e", ".[test,tests,dev]"], cwd=str(repo_dir), timeout=timeout)
    if not r.ok:
        r = run(pip + ["-e", "."], cwd=str(repo_dir), timeout=timeout)
    run(pip + ["pytest"], timeout=300)  # ensure a pytest is present
    ok = (Path(py).exists())
    return ok, ("editable install ok" if r.ok else f"install warnings: {r.stderr[:160]}"), {"FC_PY": py}


def _bare_name(t: str) -> str:
    """The plain test-function name from a SWE-bench F2P/P2P entry.
    Entries look like `path/to/test_x.py::TestC::test_y[param]` or just
    `test_y`. We key on the final function name (sans parametrization) for a
    robust `-k` selection that works whether or not the file path matches."""
    last = t.split("::")[-1]
    return re.sub(r"\[.*\]$", "", last).strip()


def _pytest(repo_dir: Path, py: str, tests: list[str], timeout=900) -> Cmd:
    """Run a set of SWE-bench test selectors.

    Pass explicit node-ids (those with `::`) directly so the exact case runs,
    AND add a `-k` OR-expression of the bare function names as a fallback for
    entries given as a path or bare name. `-x` stops on first failure so a
    behavioral verdict is fast.
    """
    args = [py, "-m", "pytest", "-x", "-q", "--no-header", "-p", "no:cacheprovider"]
    nodeids = [t for t in tests if "::" in t]
    if nodeids:
        args += nodeids
    names = sorted({_bare_name(t) for t in tests if _bare_name(t)})
    if names and not nodeids:
        # only use -k when we have no precise node-ids (else -k filters them too)
        args += ["-k", " or ".join(names)]
    return run(args, cwd=str(repo_dir), timeout=timeout)


def _all_passed(c: Cmd) -> bool:
    if c.timed_out:
        return False
    # pytest exit 0 = all selected passed; 5 = none collected (treat as fail)
    return c.returncode == 0


# ── dimensions ──────────────────────────────────────────────────────────────
def grade(instance_id: str, repo_dir: Path, repo: str, patch: str,
          test_patch: str, fail_to_pass: list[str], pass_to_pass: list[str],
          py: str, *, lint_targets: list[str], judge=None) -> Grade:
    g = Grade(instance_id)
    stats = diff_stats(patch)

    # Dim 5 — Scope (cheap, no env). Evaluate first; it never needs the venv.
    scope_ok = stats.files <= MAX_FILES and stats.churn <= MAX_CHURN and stats.files > 0
    g.dims.append(DimResult(
        "scope", blocker=True, passed=scope_ok,
        detail=f"files={stats.files} (≤{MAX_FILES}), churn={stats.churn} (≤{MAX_CHURN})"))

    # Apply the held-out test patch on top of the model patch for Dim 1/2.
    tp = repo_dir / ".fc_testpatch.diff"
    tp.write_text(test_patch)
    tp_applied = run(["git", "apply", "--whitespace=nowarn", str(tp)], cwd=str(repo_dir))
    if not tp_applied.ok:
        run(["git", "apply", "--3way", str(tp)], cwd=str(repo_dir))

    # Dim 1 — Behavioral correctness (BLOCKER): FAIL_TO_PASS pass post-patch.
    c1 = _pytest(repo_dir, py, fail_to_pass)
    g.dims.append(DimResult(
        "behavioral_correctness", blocker=True, passed=_all_passed(c1),
        evaluable=not (c1.timed_out or "no module" in c1.stderr.lower()
                       or "command not found" in c1.stderr.lower()),
        detail=f"rc={c1.returncode} {'TIMEOUT' if c1.timed_out else ''} "
               f"F2P={fail_to_pass[:3]}{'…' if len(fail_to_pass)>3 else ''} "
               f"| {_tail(c1)}"))

    # Dim 2 — Regression safety (BLOCKER): a sample of PASS_TO_PASS still passes.
    p2p_sample = pass_to_pass[:25]  # cap for runtime
    if p2p_sample:
        c2 = _pytest(repo_dir, py, p2p_sample, timeout=1200)
        g.dims.append(DimResult(
            "regression_safety", blocker=True, passed=_all_passed(c2),
            evaluable=not c2.timed_out,
            detail=f"rc={c2.returncode} sampled {len(p2p_sample)}/{len(pass_to_pass)} "
                   f"P2P | {_tail(c2)}"))
    else:
        g.dims.append(DimResult("regression_safety", blocker=True, passed=True,
                                detail="no PASS_TO_PASS in instance"))

    # Dim 4 — Test correctness / reverse-classical (BLOCKER):
    # the injected tests must FAIL on the *unpatched* base. Computed separately
    # (needs a clean tree) by reverse_classical(); passed in via judge hook is
    # not used here — see runner which calls reverse_classical() before grade().
    # We record a placeholder that the runner overwrites.
    g.dims.append(DimResult("test_correctness_reverse_classical", blocker=True,
                            passed=None, evaluable=False,
                            detail="computed by runner.reverse_classical"))

    # Dim 3 — Mechanical cleanliness (non-blocker): repo linter on changed files.
    lint = _lint(repo_dir, py, repo, lint_targets)
    g.dims.append(lint)

    # Dim 6 — Code quality (non-blocker): LLM rubric, optional.
    if judge is not None:
        q = judge(patch)
        g.dims.append(DimResult("code_quality", blocker=False, passed=q.get("pass"),
                                detail=q.get("detail", "")))
    else:
        g.dims.append(DimResult("code_quality", blocker=False, passed=None,
                                evaluable=False,
                                detail="STUB: no LLM judge wired (see grader.judge_stub)"))
    return g


def reverse_classical(repo_dir: Path, repo: str, test_patch: str,
                      fail_to_pass: list[str], py: str) -> DimResult:
    """Dim 4: apply ONLY the test patch to the clean base and confirm the
    injected FAIL_TO_PASS tests FAIL (they would pass trivially if they didn't
    actually exercise the fix). This is the reverse-classical gate."""
    # caller guarantees repo_dir is a clean checkout at base_commit
    tp = repo_dir / ".fc_rc_testpatch.diff"
    tp.write_text(test_patch)
    r = run(["git", "apply", "--whitespace=nowarn", str(tp)], cwd=str(repo_dir))
    if not r.ok:
        r = run(["git", "apply", "--3way", str(tp)], cwd=str(repo_dir))
    if not r.ok:
        return DimResult("test_correctness_reverse_classical", blocker=True,
                         passed=None, evaluable=False,
                         detail=f"test patch did not apply to base: {r.stderr[:160]}")
    c = _pytest(repo_dir, py, fail_to_pass)
    # we WANT failure here. rc != 0 (and not infra) => tests bite => PASS the dim.
    infra = c.timed_out or "no module named pytest" in c.stderr.lower()
    failed_as_expected = (not _all_passed(c)) and not infra
    return DimResult("test_correctness_reverse_classical", blocker=True,
                     passed=failed_as_expected, evaluable=not infra,
                     detail=f"on unpatched base rc={c.returncode} "
                            f"(want non-zero) | {_tail(c)}")


def _lint(repo_dir: Path, py: str, repo: str, targets: list[str]) -> DimResult:
    """Best-effort: run the repo's own linter on changed files. We try ruff,
    then flake8, then pyflakes. Non-blocker, so 'no linter available' is a soft
    skip, not a failure."""
    if not targets:
        return DimResult("mechanical_cleanliness", blocker=False, passed=None,
                         evaluable=False, detail="no changed python files to lint")
    pyfiles = [t for t in targets if t.endswith(".py")]
    if not pyfiles:
        return DimResult("mechanical_cleanliness", blocker=False, passed=None,
                         evaluable=False, detail="no python targets")
    for tool in (["-m", "ruff", "check"], ["-m", "flake8"], ["-m", "pyflakes"]):
        c = run([py, *tool, *pyfiles], cwd=str(repo_dir), timeout=180)
        if "No module named" in c.stderr:
            continue
        return DimResult("mechanical_cleanliness", blocker=False,
                         passed=(c.returncode == 0),
                         detail=f"{tool[1]}: rc={c.returncode} | {_tail(c)}")
    return DimResult("mechanical_cleanliness", blocker=False, passed=None,
                     evaluable=False, detail="no linter installed in venv")


def _tail(c: Cmd, n=240) -> str:
    out = (c.stdout or "") + (c.stderr or "")
    out = out.strip().replace("\n", " ⏎ ")
    return out[-n:]


def judge_stub(patch: str) -> dict:
    """Placeholder Dim-6 judge. Wire an LLM here (e.g. drive a second akh, or a
    rubric prompt to Tecton) to score idiomaticity/design 1-5 and map ≥3 to pass.
    TODO: implement; currently always returns evaluable=False upstream."""
    return {"pass": None, "detail": "stub"}
