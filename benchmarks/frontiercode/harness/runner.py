#!/usr/bin/env python3
"""FrontierCode-style runner: drive akh `tecton` on hard repo-edit tasks.

Per task:
  1. checkout repo@base_commit into a temp dir
  2. build a context-bearing prompt (problem + the source files the fix touches)
  3. drive Tecton over ACP to get a unified diff (Tecton is a "model", not a
     repo-editing agent — its sandbox cannot touch the target repo, so it
     returns a patch which THIS harness applies and verifies)
  4. apply the patch to the checkout
  5. run the 6-dimension grader; blocker-gated score (all blockers → 1 else 0)

Usage:
    python3 harness/runner.py --tasks ../tasks                 # all tasks
    python3 harness/runner.py --ids sympy__sympy-17630         # a slice
    python3 harness/runner.py --ids sympy__sympy-17630 --dry-run-gold
        ^ replay the GOLD patch instead of asking Tecton — sanity-checks the
          grader + test env independently of the model.
    python3 harness/runner.py --ids ... --no-tests
        ^ skip env build + test execution (scope+patch-applies only); fast smoke.

Outputs JSON per task into ../results/ and a roll-up results/summary.json.
"""
from __future__ import annotations
import argparse, json, os, sys, tempfile, time, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import util, grader

# import the ACP adapter from the repo's scripts/
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

HERE = Path(__file__).resolve().parent
TASKS_DIR = HERE.parent / "tasks"
RESULTS_DIR = HERE.parent / "results"


# ── prompt construction ─────────────────────────────────────────────────────
PROMPT_TEMPLATE = """\
You are an expert software engineer fixing a real issue in the `{repo}` repository.
Everything you need is in THIS message — do NOT call any tools, do NOT run shell
commands, do NOT fetch URLs, do NOT search files. Reason internally, then answer.

# Issue
{problem}

# Relevant source files (current contents at the affected paths)
{context}

# Your task
Produce a fix as a single unified diff (git format) that resolves the issue.
Rules:
- Edit ONLY the source files needed; do not modify or add tests.
- Your ENTIRE reply must be exactly one fenced ```diff block — the complete patch
  and nothing before or after it. Paths must be relative to the repo root, using
  the `a/<path>` and `b/<path>` prefixes and `@@` hunk headers.
- Keep the change minimal and idiomatic for this codebase.

Output the ```diff block now and nothing else."""

# SEARCH/REPLACE edit format — the model emits CODE, not a diff (PewDiePie's
# Qwen-coding insight; aider/opencode). `git diff` makes the patch afterwards.
PROMPT_TEMPLATE_SR = """\
You are an expert software engineer fixing a real issue in the `{repo}` repository.
Everything you need is in THIS message — do NOT call any tools, do NOT run shell
commands, do NOT search files. Reason internally, then answer.

# Issue
{problem}

# Relevant source files (current contents at the affected paths)
{context}

# Your task
Fix the issue by editing the source files. Do NOT write a diff. Express each edit
as a SEARCH/REPLACE block: the file path on its own line, then the EXACT existing
lines to find, then the replacement lines, like this:

path/to/file.py
<<<<<<< SEARCH
<lines copied verbatim from the file contents above, including indentation>
=======
<the replacement lines>
>>>>>>> REPLACE

Rules:
- Copy the SEARCH text EXACTLY from the file contents shown above — same
  indentation and enough surrounding lines to be unique. This is critical.
- Edit ONLY source files; do not modify or add tests.
- Use one block per contiguous change; you may emit several.
- Keep changes minimal and idiomatic.

Output the SEARCH/REPLACE block(s) now and nothing else."""


def gold_touched_files(gold_patch: str) -> list[str]:
    """Files the gold solution edits — these are the relevant source files to
    show Tecton (we never show the patch itself)."""
    files = []
    for line in gold_patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                b = parts[3]
                files.append(b[2:] if b.startswith("b/") else b)
    return files


def build_context(repo_dir: Path, files: list[str], max_chars=24000) -> str:
    blocks = []
    budget = max_chars
    for f in files:
        p = repo_dir / f
        if not p.exists():
            continue
        text = p.read_text(errors="replace")
        if len(text) > budget:
            text = text[:budget] + "\n# … (truncated for context budget) …\n"
        blocks.append(f"## {f}\n```python\n{text}\n```")
        budget -= len(text)
        if budget <= 0:
            break
    return "\n\n".join(blocks) if blocks else "(no source files could be loaded)"


def build_prompt(task: dict, repo_dir: Path, max_context=24000,
                 edit_format="search-replace") -> tuple[str, list[str]]:
    files = gold_touched_files(task["gold_patch"])
    ctx = build_context(repo_dir, files, max_chars=max_context)
    tmpl = PROMPT_TEMPLATE_SR if edit_format == "search-replace" else PROMPT_TEMPLATE
    prompt = tmpl.format(
        repo=task["repo"],
        problem=task["problem_statement"].strip()[:8000],
        context=ctx,
    )
    return prompt, files


# ── per-task run ────────────────────────────────────────────────────────────
def run_task(task: dict, *, dry_run_gold=False, no_tests=False, workspace="tecton",
             timeout=600, keep=False, max_context=24000, judge_workspace=None,
             edit_format="search-replace") -> dict:
    iid = task["instance_id"]
    print(f"\n{'='*70}\n[{iid}] {task['repo']} @ {task['base_commit'][:10]} "
          f"({task['difficulty']})")
    rec: dict = {"instance_id": iid, "repo": task["repo"],
                 "difficulty": task["difficulty"], "stages": {}}
    workdir = Path(tempfile.mkdtemp(prefix=f"fc_{iid}_"))
    rec["workdir"] = str(workdir)
    try:
        repo_dir = workdir / "repo"
        # 1. checkout
        t0 = time.time()
        co = util.checkout(task["repo"], task["base_commit"], repo_dir)
        rec["stages"]["checkout"] = {"ok": co.ok, "secs": round(time.time()-t0, 1),
                                     "detail": "" if co.ok else co.stderr[-300:]}
        if not co.ok:
            rec["error"] = "checkout failed (network? bad sha?)"
            print(f"  checkout FAILED: {co.stderr[-200:]}")
            return rec
        print(f"  checkout ok ({rec['stages']['checkout']['secs']}s)")

        # 2/3. get a patch — gold (oracle) or Tecton
        if dry_run_gold:
            patch = task["gold_patch"]
            rec["stages"]["model"] = {"mode": "gold-oracle", "patch_chars": len(patch)}
            print("  using GOLD patch (oracle mode)")
        else:
            prompt, files = build_prompt(task, repo_dir, max_context=max_context,
                                         edit_format=edit_format)
            rec["stages"]["prompt"] = {"chars": len(prompt), "context_files": files,
                                       "edit_format": edit_format}
            if edit_format == "search-replace":
                followup = ("Output your edits NOW as SEARCH/REPLACE blocks (file path "
                            "line, then <<<<<<< SEARCH / ======= / >>>>>>> REPLACE). "
                            "No diff, no tools, no prose.")
                target = lambda t: len(util.parse_search_replace(t)) > 0
            else:
                followup = ("Output the COMPLETE unified diff for your fix NOW, as a "
                            "single fenced ```diff block and nothing else — no tools, "
                            "no prose.")
                target = lambda t: util.extract_patch(t) is not None
            text, mrec = drive_tecton(prompt, workspace, timeout, followup, target)
            rec["stages"]["model"] = mrec
            # Convert the model's CODE output into a clean unified diff via git
            # (the "don't make the LLM generate patches" path).
            patch = None
            if text is not None and edit_format == "search-replace":
                edits = util.parse_search_replace(text)
                applied, failures = util.apply_search_replace(repo_dir, edits)
                mrec.update({"edits_parsed": len(edits), "edits_applied": applied,
                             "edit_failures": failures})
                if applied:
                    patch = util.git_diff(repo_dir) or None
                    util.git_revert(repo_dir)  # grader re-applies the clean git diff
            elif text is not None:
                patch = util.extract_patch(text)
            mrec["patch_extracted"] = bool(patch and patch.strip())
            if not mrec["patch_extracted"]:
                patch = None
                mrec.setdefault("failure_kind", "no_usable_edit")
                mrec["answer_head"] = (text or "")[:400]
                rec["error"] = "no usable edit from Tecton"
                print("  Tecton produced NO usable patch")
                _save(rec)
                return rec

        (workdir / "candidate.patch").write_text(patch)
        stats = util.diff_stats(patch)
        rec["patch_stats"] = {"files": stats.files, "added": stats.added,
                              "removed": stats.removed, "files_list": stats.files_list}

        # 4. apply
        applied, detail = util.apply_patch(patch, repo_dir)
        rec["stages"]["apply"] = {"ok": applied, "detail": detail}
        print(f"  apply: {'ok' if applied else 'FAILED'} — {detail[:140]}")
        if not applied:
            rec["score"] = 0
            rec["error"] = "patch did not apply"
            _save(rec)
            return rec

        if no_tests:
            rec["note"] = "--no-tests: scope+apply only; not graded"
            _save(rec)
            return rec

        # 5a. reverse-classical (Dim 4) needs a SEPARATE clean checkout at base
        rc_dir = workdir / "repo_rc"
        util.checkout(task["repo"], task["base_commit"], rc_dir)

        # build env once in the patched repo
        print("  building test env (venv + editable install)…")
        t1 = time.time()
        env_ok, env_detail, env_over = grader.make_env(repo_dir, task["repo"])
        py = env_over.get("FC_PY", "python3")
        rec["stages"]["env"] = {"ok": env_ok, "secs": round(time.time()-t1, 1),
                                "detail": env_detail}
        print(f"  env: {env_detail} ({rec['stages']['env']['secs']}s)")

        # reuse the same venv python for the rc checkout (install editable there too)
        rc_env_ok, _, rc_env = grader.make_env(rc_dir, task["repo"])
        rc_py = rc_env.get("FC_PY", py)

        lint_targets = stats.files_list
        # Dim-6 idiomaticity judge (non-blocker): a separate akh, if requested.
        judge = grader.make_akh_judge(judge_workspace) if judge_workspace else None
        g = grader.grade(iid, repo_dir, task["repo"], patch, task["test_patch"],
                         task["FAIL_TO_PASS"], task["PASS_TO_PASS"], py,
                         lint_targets=lint_targets, judge=judge)
        # overwrite Dim 4 placeholder with the real reverse-classical verdict
        rc = grader.reverse_classical(rc_dir, task["repo"], task["test_patch"],
                                      task["FAIL_TO_PASS"], rc_py)
        for i, d in enumerate(g.dims):
            if d.name == "test_correctness_reverse_classical":
                g.dims[i] = rc
        g.finalize()
        rec["grade"] = g.to_dict()
        rec["score"] = g.score
        _print_grade(g)
        _save(rec)
        return rec
    finally:
        if not keep:
            shutil.rmtree(workdir, ignore_errors=True)


def drive_tecton(prompt: str, workspace: str, timeout: int,
                 followup_prompt: str, target_present) -> tuple[str | None, dict]:
    """Run the akh and return (model_text, rec). `model_text` is the best output
    (answer; reasoning appended as a fallback, since glm-5.2 is a reasoning model
    that may leave the edit in the thinking stream). If `target_present(text)` is
    false after the first turn, do ONE tool-free elicitation follow-up in the same
    session. Format-agnostic — the caller turns the text into a patch."""
    from acp_adapter import AkhClient

    def best(t):
        return t.answer if target_present(t.answer) else \
            (t.answer or "") + "\n" + (t.reasoning or "")

    rec: dict = {"mode": "tecton"}
    turn = followup = None
    try:
        with AkhClient(workspace, timeout=timeout) as c:
            turn = c.prompt(prompt)
            text = best(turn)
            if not target_present(text):
                followup = c.prompt(followup_prompt)
                text = best(followup)
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        return None, rec
    rec.update({"stop_reason": turn.stop_reason, "elapsed_s": round(turn.elapsed_s, 1),
                "timed_out": turn.timed_out, "answer_chars": len(turn.answer),
                "reasoning_chars": len(turn.reasoning), "tools": sorted(set(turn.tools_used))})
    if followup is not None:
        rec["followup"] = {"answer_chars": len(followup.answer),
                           "reasoning_chars": len(followup.reasoning),
                           "elapsed_s": round(followup.elapsed_s, 1)}
    head = (turn.answer or "").lstrip()[:80].lower()
    if head.startswith("error[") or ("backend" in head and "failed" in head):
        rec["failure_kind"] = "backend_error"
    elif turn.timed_out:
        rec["failure_kind"] = "timeout"
    return text, rec


# ── output ──────────────────────────────────────────────────────────────────
def _print_grade(g: grader.Grade):
    print(f"  SCORE: {g.score}  {('— ' + g.note) if g.note else ''}")
    for d in g.dims:
        mark = {True: "PASS", False: "FAIL", None: "----"}[d.passed]
        kind = "B" if d.blocker else " "
        ev = "" if d.evaluable else " (not-evaluable)"
        print(f"    [{kind}] {mark:4} {d.name}{ev}: {d.detail[:120]}")


def _save(rec: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{rec['instance_id']}.json").write_text(json.dumps(rec, indent=2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default=str(TASKS_DIR))
    ap.add_argument("--ids", nargs="*", help="subset of instance ids")
    ap.add_argument("--dry-run-gold", action="store_true",
                    help="replay gold patch to validate grader+env")
    ap.add_argument("--no-tests", action="store_true",
                    help="scope+apply only; no env build / test exec")
    ap.add_argument("--workspace", default="tecton")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--keep", action="store_true", help="keep temp workdirs")
    ap.add_argument("--max-context", type=int, default=24000,
                    help="max chars of source context in the prompt; shrink this "
                         "(e.g. 6000) if the cloud LLM backend times out on large prompts")
    ap.add_argument("--judge-workspace", default=None,
                    help="akh to score Dim-6 idiomaticity (non-blocker); use a "
                         "different akh than --workspace to avoid self-grading")
    ap.add_argument("--edit-format", choices=["search-replace", "diff"],
                    default="search-replace",
                    help="how the model expresses its fix: search-replace (model "
                         "emits code, git makes the diff — default) or diff (model "
                         "writes the unified diff itself)")
    args = ap.parse_args(argv)

    tdir = Path(args.tasks)
    files = sorted(tdir.glob("*.json"))
    tasks = [json.loads(f.read_text()) for f in files]
    if args.ids:
        tasks = [t for t in tasks if t["instance_id"] in args.ids]
    if not tasks:
        sys.exit("no tasks selected")

    summary = []
    for t in tasks:
        rec = run_task(t, dry_run_gold=args.dry_run_gold, no_tests=args.no_tests,
                       workspace=args.workspace, timeout=int(args.timeout),
                       keep=args.keep, max_context=args.max_context,
                       judge_workspace=args.judge_workspace,
                       edit_format=args.edit_format)
        summary.append({"instance_id": rec["instance_id"],
                        "score": rec.get("score", 0),
                        "error": rec.get("error")})
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    total = sum(s["score"] or 0 for s in summary)
    roll = {"n": len(summary), "passed": total,
            "pass_rate": round(total / len(summary), 3) if summary else 0,
            "tasks": summary}
    (RESULTS_DIR / "summary.json").write_text(json.dumps(roll, indent=2))
    print(f"\n{'='*70}\nDIAMOND-STYLE RESULT: {total}/{len(summary)} "
          f"({roll['pass_rate']*100:.1f}%) — see results/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
