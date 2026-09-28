"""Tests for scripts/lilith/ancestry_fetch.sh: bootstrap, idempotent fetch,
new-commit pickup, and the portable mkdir-based fetch lock (stale-lock
recovery and concurrent-run skip). Runs real `git` in temp directories; no
network access (the "remote" is a local bare repo)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("ancestry_fetch.sh")

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")


def run_git(*args: str, cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def make_remote(tmp_path: Path) -> Path:
    remote = tmp_path / "origin-mirror"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    work = tmp_path / "work"
    subprocess.run(["git", "clone", "-q", str(remote), str(work)], check=True)
    run_git("config", "user.email", "test@example.com", cwd=work)
    run_git("config", "user.name", "Test", cwd=work)
    run_git("commit", "-q", "--allow-empty", "-m", "first", cwd=work)
    run_git("push", "-q", "origin", "HEAD:main", cwd=work)
    return remote


def run_fetch(clone: Path, remote: Path, *, timeout: float = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(SCRIPT), str(clone), str(remote)],
        capture_output=True, text=True, timeout=timeout,
    )


def test_bootstraps_bare_clone_and_fetches_main(tmp_path):
    remote = make_remote(tmp_path)
    clone = tmp_path / "clone-mirror"
    expected_sha = run_git("rev-parse", "HEAD", cwd=tmp_path / "work")

    result = run_fetch(clone, remote)
    assert result.returncode == 0, result.stderr

    assert clone.is_dir()
    got_sha = run_git("rev-parse", "--verify", "refs/remotes/origin/main", cwd=clone)
    assert got_sha == expected_sha

    # Bare: no working tree checked out.
    assert not (clone / ".git").exists()


def test_idempotent_refetch_and_new_commit_pickup(tmp_path):
    remote = make_remote(tmp_path)
    clone = tmp_path / "clone-mirror"
    run_fetch(clone, remote)

    # Re-running with nothing new must still succeed.
    result = run_fetch(clone, remote)
    assert result.returncode == 0, result.stderr

    work = tmp_path / "work"
    run_git("commit", "-q", "--allow-empty", "-m", "second", cwd=work)
    run_git("push", "-q", "origin", "HEAD:main", cwd=work)
    new_sha = run_git("rev-parse", "HEAD", cwd=work)

    result = run_fetch(clone, remote)
    assert result.returncode == 0, result.stderr
    got_sha = run_git("rev-parse", "--verify", "refs/remotes/origin/main", cwd=clone)
    assert got_sha == new_sha


def test_repairs_missing_or_wrong_origin_remote(tmp_path):
    remote = make_remote(tmp_path)
    clone = tmp_path / "clone-mirror"
    run_fetch(clone, remote)

    # Simulate a clone restored without its remote configured.
    subprocess.run(["git", "remote", "remove", "origin"], cwd=clone, check=True)
    result = run_fetch(clone, remote)
    assert result.returncode == 0, result.stderr
    assert run_git("remote", "get-url", "origin", cwd=clone) == str(remote)


def test_stale_lock_is_cleared_and_fetch_proceeds(tmp_path):
    remote = make_remote(tmp_path)
    clone = tmp_path / "clone-mirror"
    run_fetch(clone, remote)  # bootstrap first so the lock dir sits beside a real clone

    lock_dir = tmp_path / "clone-mirror.fetch.lock"
    lock_dir.mkdir()
    (lock_dir / "pid").write_text("999999999")  # a pid that cannot be alive

    result = run_fetch(clone, remote)
    assert result.returncode == 0, result.stderr
    assert "clearing a stale lock" in result.stdout
    assert not lock_dir.exists()  # released after the run


def test_concurrent_fetch_is_skipped_and_lock_left_intact(tmp_path):
    remote = make_remote(tmp_path)
    clone = tmp_path / "clone-mirror"
    run_fetch(clone, remote)  # bootstrap first

    holder = subprocess.Popen(["sleep", "20"])
    try:
        lock_dir = tmp_path / "clone-mirror.fetch.lock"
        lock_dir.mkdir()
        (lock_dir / "pid").write_text(str(holder.pid))

        result = run_fetch(clone, remote)
        assert result.returncode == 0, result.stderr
        assert "already running" in result.stdout
        # The skipped run must not have torn down the still-live lock.
        assert lock_dir.exists()
        assert (lock_dir / "pid").read_text() == str(holder.pid)
    finally:
        holder.terminate()
        holder.wait(timeout=5)


def test_refuses_a_non_git_directory(tmp_path):
    remote = make_remote(tmp_path)
    clone = tmp_path / "not-a-repo"
    clone.mkdir()
    (clone / "stray-file").write_text("hello")

    result = run_fetch(clone, remote)
    assert result.returncode == 1
    assert "not a git repository" in result.stdout
