#!/usr/bin/env python3
"""Shared helpers: repo checkout, patch extraction, diff stats, command runner."""
from __future__ import annotations
import json, os, re, subprocess, tempfile, shutil
from dataclasses import dataclass, field
from pathlib import Path

GITHUB = "https://github.com/{repo}.git"


# ── command runner ──────────────────────────────────────────────────────────
@dataclass
class Cmd:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def run(cmd, cwd=None, timeout=600, env=None) -> Cmd:
    """Run a command, capturing output. `cmd` is a list (no shell)."""
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    try:
        p = subprocess.run(cmd, cwd=cwd, timeout=timeout, env=full_env,
                           capture_output=True, text=True)
        return Cmd(p.returncode, p.stdout, p.stderr)
    except subprocess.TimeoutExpired as e:
        return Cmd(124, e.stdout or "", e.stderr or "", timed_out=True)


# ── repo checkout ───────────────────────────────────────────────────────────
def checkout(repo: str, base_commit: str, dest: Path, timeout=600) -> Cmd:
    """Clone `repo` and hard-reset to `base_commit` in `dest`.

    Uses a full clone (SWE-bench base commits are often deep in history, so a
    shallow clone of a single sha is unreliable across mirrors). Falls back to
    a local cache under ~/.cache/frontiercode-repos to avoid re-cloning.
    """
    cache = Path.home() / ".cache" / "frontiercode-repos" / repo.replace("/", "__")
    url = GITHUB.format(repo=repo)
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        r = run(["git", "clone", url, str(cache)], timeout=timeout)
        if not r.ok:
            return r
    else:
        run(["git", "fetch", "origin"], cwd=str(cache), timeout=timeout)
    # clone-local (fast, no network) into dest
    if dest.exists():
        shutil.rmtree(dest)
    r = run(["git", "clone", str(cache), str(dest)], timeout=timeout)
    if not r.ok:
        return r
    r = run(["git", "fetch", url, base_commit], cwd=str(dest), timeout=timeout)
    # base_commit may already be present from the cache; reset regardless
    r = run(["git", "reset", "--hard", base_commit], cwd=str(dest), timeout=120)
    if not r.ok:
        # try fetching the specific sha then reset
        run(["git", "fetch", "origin", base_commit], cwd=str(dest), timeout=timeout)
        r = run(["git", "reset", "--hard", base_commit], cwd=str(dest), timeout=120)
    run(["git", "clean", "-fdx"], cwd=str(dest), timeout=120)
    return r


# ── patch extraction from a model answer ────────────────────────────────────
_FENCE = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.S)
_DIFF_HDR = re.compile(r"^(diff --git |--- )", re.M)


def extract_patch(answer: str) -> str | None:
    """Pull a unified diff out of a free-text model answer.

    Strategy: prefer the contents of a ```diff/```patch fence; else the first
    fenced block that *looks* like a diff; else a raw `diff --git`/`---` region
    to end of text. Returns the patch text (newline-terminated) or None.
    """
    candidates: list[str] = []
    for m in _FENCE.finditer(answer):
        candidates.append(m.group(1))
    for c in candidates:
        if _DIFF_HDR.search(c):
            return _normalize(c)
    # no usable fenced diff — look for a raw diff region
    m = _DIFF_HDR.search(answer)
    if m:
        return _normalize(answer[m.start():])
    # last resort: a fenced block even without a clear header (may be a hunk)
    if candidates:
        return _normalize(candidates[0])
    return None


def _normalize(p: str) -> str:
    p = p.strip("\n")
    if not p.endswith("\n"):
        p += "\n"
    return p


def apply_patch(patch: str, cwd: Path) -> tuple[bool, str]:
    """Apply a unified diff with progressively looser strategies.

    Returns (applied, detail). Tries `git apply`, then `git apply --3way`, then
    `git apply -p0`, then GNU `patch -p1 --fuzz`.
    """
    pf = cwd / ".fc_patch.diff"
    pf.write_text(patch)
    attempts = [
        ["git", "apply", "--whitespace=nowarn", str(pf)],
        ["git", "apply", "--3way", "--whitespace=nowarn", str(pf)],
        ["git", "apply", "-p0", "--whitespace=nowarn", str(pf)],
        ["patch", "-p1", "--fuzz=3", "--no-backup-if-mismatch", "-i", str(pf)],
    ]
    detail = []
    for a in attempts:
        r = run(a, cwd=str(cwd), timeout=120)
        if r.ok:
            return True, f"applied via: {' '.join(a[:3])}"
        detail.append(f"{a[0]} {a[2] if len(a)>2 else ''}: {r.stderr.strip()[:200]}")
    return False, " | ".join(detail)


# ── diff statistics (scope dimension) ───────────────────────────────────────
@dataclass
class DiffStats:
    files: int = 0
    files_list: list[str] = field(default_factory=list)
    added: int = 0
    removed: int = 0

    @property
    def churn(self) -> int:
        return self.added + self.removed


def diff_stats(patch: str) -> DiffStats:
    s = DiffStats()
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            s.files += 1
            parts = line.split()
            if len(parts) >= 3:
                s.files_list.append(parts[2][2:] if parts[2].startswith("a/") else parts[2])
        elif line.startswith("+") and not line.startswith("+++"):
            s.added += 1
        elif line.startswith("-") and not line.startswith("---"):
            s.removed += 1
    if s.files == 0:
        # patch without diff --git headers (e.g. plain ---/+++): count file pairs
        s.files = patch.count("\n--- ")
    return s


# ── SEARCH/REPLACE edit format ──────────────────────────────────────────────
# "Don't make the LLM generate patches" (aider/opencode; PewDiePie's Qwen
# coding insight). The model emits CODE (old→new snippets); we apply them with
# plain string ops and let `git diff` produce the unified diff — no LLM-authored
# line numbers, so no "corrupt patch".

_SR_BLOCK = re.compile(
    r'(?P<path>[^\n`]+?)[ \t]*\n+'
    r'<{5,}[ \t]*SEARCH[ \t]*\n(?P<search>.*?)\n={5,}[ \t]*\n(?P<replace>.*?)\n>{5,}[ \t]*REPLACE',
    re.DOTALL,
)


def parse_search_replace(text: str) -> list[dict]:
    """Parse aider-style edit blocks:

        path/to/file.py
        <<<<<<< SEARCH
        <exact existing lines>
        =======
        <replacement lines>
        >>>>>>> REPLACE

    Returns [{path, search, replace}]. The path is the last non-fence token on
    the line preceding the SEARCH marker (tolerates a ```lang fence above it)."""
    edits: list[dict] = []
    for m in _SR_BLOCK.finditer(text):
        raw = m.group("path").strip()
        # take the last line, strip backticks/markdown noise
        path = raw.splitlines()[-1].strip().strip("`").strip().lstrip("#").strip()
        if path.startswith("```"):
            continue
        edits.append({"path": path, "search": m.group("search"),
                      "replace": m.group("replace")})
    return edits


def _apply_one(content: str, search: str, replace: str) -> tuple[str, bool]:
    """Exact match first; fall back to per-line trailing-whitespace-insensitive
    window match (models often drift on trailing spaces)."""
    if search in content:
        return content.replace(search, replace, 1), True
    clines = content.split("\n")
    slines = [l.rstrip() for l in search.split("\n")]
    n = len(slines)
    if n == 0:
        return content, False
    for i in range(len(clines) - n + 1):
        if [l.rstrip() for l in clines[i:i + n]] == slines:
            new = clines[:i] + replace.split("\n") + clines[i + n:]
            return "\n".join(new), True
    return content, False


def apply_search_replace(repo_dir: Path, edits: list[dict]) -> tuple[int, list[str]]:
    """Apply edits to files under `repo_dir`. Returns (applied, failures)."""
    applied, failures = 0, []
    for e in edits:
        f = repo_dir / e["path"]
        if not f.exists():
            failures.append(f"no such file: {e['path']}")
            continue
        new, ok = _apply_one(f.read_text(), e["search"], e["replace"])
        if ok:
            f.write_text(new)
            applied += 1
        else:
            failures.append(f"SEARCH not found in {e['path']}")
    return applied, failures


def git_diff(repo_dir: Path) -> str:
    """The working-tree diff (the patch git itself generates from our edits)."""
    return run(["git", "diff"], cwd=repo_dir).stdout


def git_revert(repo_dir: Path) -> None:
    """Reset the working tree so the grader can re-apply the clean git diff."""
    run(["git", "checkout", "--", "."], cwd=repo_dir)
