"""supervisor_status.sh against 1Password / curl doubles (no network, no pods)."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "supervisor_status.sh"
FAKE_KEY = "fake-runpod-key-9c1e"
PODS = [
    {"id": "p1", "name": "akh-medu-lilith-train-r1", "desiredStatus": "RUNNING", "costPerHr": 2.69,
     "machine": {"gpuDisplayName": "H100 SXM"}, "networkVolumeId": "c6kdciozex", "lastStartedAt": "2026-09-28"},
    {"id": "p2", "name": "akh-medu-babyllm-old", "desiredStatus": "EXITED", "costPerHr": 1.59},
]
FAKE_OP = f"""#!/usr/bin/env bash
[[ "$1 $2" == "read op://Homelab/Runpod API Key/credential" ]] || exit 9
[[ -e "$FAKE_STATE/op-down" ]] && exit 1
echo "{FAKE_KEY}"
"""
FAKE_CURL = r"""#!/usr/bin/env bash
printf '%s\n' "$@" >> "$FAKE_STATE/curl-args"
cat >> "$FAKE_STATE/curl-stdin"
case "${@: -1}" in
  */pods) cat "$FAKE_STATE/pods.json" ;;
  */graphql) printf '%s' '{"data":{"myself":{"clientBalance":41.5,"currentSpendPerHr":2.75,"spendLimit":80}}}' ;;
  *) exit 22 ;;
esac
"""


@pytest.fixture
def env(tmp_path):
    state, bin_dir, home = tmp_path / "state", tmp_path / "bin", tmp_path / "home"
    for d in (state, bin_dir, home):
        d.mkdir()
    (bin_dir / "op-service-account").write_text(FAKE_OP)
    (bin_dir / "curl").write_text(FAKE_CURL)
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    (state / "pods.json").write_text(json.dumps(PODS))
    snapshots = tmp_path / "snapshots"
    for sid in ("20260928T080000Z", "20260928T090000Z"):
        (snapshots / sid).mkdir(parents=True)
        (snapshots / sid / "manifest.json").write_text("{}")
    watchdogs = home / "watchdogs"
    watchdogs.mkdir()
    (watchdogs / "lilith-train-r1.pid").write_text(f"{os.getpid()}\n")
    (watchdogs / "lilith-train-r0.pid").write_text("999999\n")
    return {"PATH": f"{bin_dir}:/usr/bin:/bin:/opt/homebrew/bin", "HOME": str(home), "FAKE_STATE": str(state),
            "OPSA": str(bin_dir / "op-service-account"), "LILITH_SNAPSHOT_ROOT": str(snapshots),
            "LILITH_WATCHDOG_DIR": str(watchdogs), "RUNPOD_REST": "https://rest.test/v1",
            "RUNPOD_GRAPHQL": "https://api.test/graphql"}


def run(env, *args):
    return subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=60)


def files_containing(root: Path, needle: str) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and needle in p.read_text(errors="ignore")]


def test_report_sections_and_key_hygiene(env, tmp_path):
    out = run(env)
    assert out.returncode == 0, out.stderr
    text = out.stdout
    assert "pods: 2 total, 1 running, $2.69/h while running" in text
    assert " * p1 akh-medu-lilith-train-r1 RUNNING H100 SXM $2.69/h volume=c6kdciozex" in text
    assert "account: credit $41.50, spending $2.750/h, spend limit 80, ~15 h of credit" in text
    assert "recorder: 2 snapshots, newest 20260928T090000Z" in text and "STALE" in text
    assert "run: pod akh-medu-lilith-train-r1 (p1) RUNNING" in text
    assert "run: watchdog lilith-train-r1 alive" in text
    assert "run: watchdog lilith-train-r0 NOT running" in text
    assert "active runs: 2" in text
    # the key went to curl on stdin only: never on a command line, stdout/stderr, or a file of ours
    state = Path(env["FAKE_STATE"])
    assert FAKE_KEY in (state / "curl-stdin").read_text()
    assert FAKE_KEY not in (state / "curl-args").read_text()
    assert FAKE_KEY not in out.stdout + out.stderr
    assert files_containing(Path(env["HOME"]), FAKE_KEY) == []
    # only the doubles hold it: the fake op (its source) and the fake curl's stdin capture
    assert sorted(p.name for p in files_containing(tmp_path, FAKE_KEY)) == ["curl-stdin", "op-service-account"]


def test_key_unavailable_and_offline_modes(env):
    (Path(env["FAKE_STATE"]) / "op-down").touch()
    out = run(env)
    assert out.returncode == 0
    assert "pods: unavailable (cannot read the Runpod key" in out.stdout
    assert not (Path(env["FAKE_STATE"]) / "curl-args").exists()
    offline = run(env, "--no-runpod")
    assert "pods: skipped" in offline.stdout and "active runs: 1" in offline.stdout
    assert run(env, "--bogus").returncode == 64


def test_recorder_root_missing(env):
    env["LILITH_SNAPSHOT_ROOT"] = "/nonexistent/snapshots"
    assert "not mounted" in run(env, "--no-runpod").stdout


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_shellcheck_clean():
    result = subprocess.run(["shellcheck", str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout
