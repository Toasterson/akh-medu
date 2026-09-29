"""cos_mirror.py against temp dirs (Lilith C3, ADR 116). Uses the real /usr/bin/rsync."""
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

P = Path(__file__).with_name("cos_mirror.py")
spec = importlib.util.spec_from_file_location("cos_mirror", P)
mm = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mm
spec.loader.exec_module(mm)

pytestmark = pytest.mark.skipif(not os.path.exists("/usr/bin/rsync") and shutil.which("rsync") is None, reason="rsync required")
RSYNC = "/usr/bin/rsync" if os.path.exists("/usr/bin/rsync") else shutil.which("rsync")
NO_SLEEP = lambda _s: None


def make_object(root: Path, payload: dict) -> str:
    data = json.dumps(payload, sort_keys=True).encode()
    sha = hashlib.sha256(data).hexdigest()
    (root / "objects").mkdir(parents=True, exist_ok=True)
    (root / "objects" / f"{sha}.json").write_bytes(data)
    return sha


def make_snapshot(root: Path, name: str, shas: list[str]) -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "lilith/cos-snapshot@1", "collectors": {"anima": {
        "status": "ok", "tool_contract_sha256": shas[0], "errors": [],
        "responses": [{"call": "c", "args": {}, "sha256": s} for s in shas[1:]]}}}
    (d / "manifest.json").write_text(json.dumps(manifest))


def runner(**kw):
    return mm.RsyncRunner(binary=RSYNC, sleep=NO_SLEEP, **kw)


@pytest.fixture
def roots(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    return src, dest


def tree(root: Path):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def test_mirrors_completed_snapshots_and_verifies(roots):
    src, dest = roots
    a, b, c = (make_object(src, {"n": i}) for i in range(3))
    make_snapshot(src, "20260928T000000Z", [a, b])
    make_snapshot(src, "20260928T020000Z", [a, c])  # shares object a
    report = mm.mirror(src, dest, runner=runner(), sleep=NO_SLEEP)
    assert report.ok and report.snapshots_mirrored == 2 and report.objects_copied == 3
    assert tree(dest) == sorted([f"objects/{s}.json" for s in (a, b, c)] + ["20260928T000000Z/manifest.json", "20260928T020000Z/manifest.json"])
    for s in (a, b, c):
        assert hashlib.sha256((dest / "objects" / f"{s}.json").read_bytes()).hexdigest() == s


def test_second_run_is_a_noop(roots):
    src, dest = roots
    a, b = make_object(src, {"n": 1}), make_object(src, {"n": 2})
    make_snapshot(src, "20260928T000000Z", [a, b])
    mm.mirror(src, dest, runner=runner(), sleep=NO_SLEEP)
    calls = []

    class Spy(mm.RsyncRunner):
        def copy(self, *args, **kwargs):
            calls.append(args[2])
            return super().copy(*args, **kwargs)

    report = mm.mirror(src, dest, runner=Spy(binary=RSYNC, sleep=NO_SLEEP), sleep=NO_SLEEP)
    assert report.ok and report.snapshots_mirrored == 0 and report.snapshots_already_present == 1
    assert report.objects_copied == 0 and all(not files for files in calls)


def test_incomplete_snapshot_without_manifest_is_ignored(roots):
    src, dest = roots
    a = make_object(src, {"n": 1})
    (src / "20260928T040000Z").mkdir()  # recorder still writing: no manifest yet
    (src / "20260928T040000Z" / ".manifest.json.tmp-1-x").write_text("{")
    make_snapshot(src, "20260928T000000Z", [a, a])
    report = mm.mirror(src, dest, runner=runner(), sleep=NO_SLEEP)
    assert report.snapshots_seen == 1
    assert not (dest / "20260928T040000Z").exists()


def test_unreadable_manifest_is_skipped_and_reported(roots):
    src, dest = roots
    (src / "20260928T000000Z").mkdir()
    (src / "20260928T000000Z" / "manifest.json").write_text("{not json")
    report = mm.mirror(src, dest, runner=runner(), sleep=NO_SLEEP)
    assert report.skipped_unreadable == ["20260928T000000Z"] and report.snapshots_seen == 0


def test_never_deletes_on_either_side(roots):
    src, dest = roots
    a = make_object(src, {"n": 1})
    make_snapshot(src, "20260928T000000Z", [a, a])
    stale = dest / "objects" / ("f" * 64 + ".json")
    stale.parent.mkdir(parents=True)
    stale.write_text("only on the share")
    old = dest / "20250101T000000Z"
    old.mkdir()
    (old / "manifest.json").write_text("{}")
    mm.mirror(src, dest, runner=runner(), sleep=NO_SLEEP)
    assert stale.read_text() == "only on the share" and (old / "manifest.json").exists()
    assert (src / "objects" / f"{a}.json").exists()  # source untouched


def test_damaged_object_on_share_is_repaired(roots):
    src, dest = roots
    a, b = make_object(src, {"n": 1}), make_object(src, {"n": 2})
    (dest / "objects").mkdir(parents=True)
    (dest / "objects" / f"{a}.json").write_bytes(b"truncat")  # wrong size
    make_snapshot(src, "20260928T000000Z", [a, b])
    report = mm.mirror(src, dest, runner=runner(), sleep=NO_SLEEP)
    assert report.ok and report.objects_repaired == 1 and report.objects_copied == 1
    assert (dest / "objects" / f"{a}.json").read_bytes() == (src / "objects" / f"{a}.json").read_bytes()


def test_existing_different_manifest_is_conflict_never_overwritten(roots):
    src, dest = roots
    a = make_object(src, {"n": 1})
    make_snapshot(src, "20260928T000000Z", [a, a])
    (dest / "20260928T000000Z").mkdir(parents=True)
    (dest / "20260928T000000Z" / "manifest.json").write_text('{"different": true}')
    report = mm.mirror(src, dest, runner=runner(), sleep=NO_SLEEP)
    assert report.manifest_conflicts == ["20260928T000000Z"] and not report.ok
    assert json.loads((dest / "20260928T000000Z" / "manifest.json").read_text()) == {"different": True}


def test_object_missing_locally_defers_snapshot_and_manifest_not_copied(roots):
    src, dest = roots
    a = make_object(src, {"n": 1})
    ghost = "a" * 64
    make_snapshot(src, "20260928T000000Z", [a, ghost])
    report = mm.mirror(src, dest, runner=runner(), sleep=NO_SLEEP)
    assert not report.ok and report.snapshots_deferred == ["20260928T000000Z"]
    assert not (dest / "20260928T000000Z" / "manifest.json").exists()  # manifest never precedes its objects


def test_corrupt_copy_fails_verification_and_defers_manifest(roots):
    src, dest = roots
    a = make_object(src, {"n": 1})
    make_snapshot(src, "20260928T000000Z", [a, a])

    class Corrupting(mm.RsyncRunner):
        def copy(self, s, d, files, **kw):
            super().copy(s, d, files, **kw)
            for f in files:
                if f.startswith("objects/"):
                    (d / f).write_bytes((s / f).read_bytes()[:-1] + b"X")  # same size, wrong bytes

    report = mm.mirror(src, dest, runner=Corrupting(binary=RSYNC, sleep=NO_SLEEP), sleep=NO_SLEEP)
    assert not report.ok and report.objects_failed and report.snapshots_deferred == ["20260928T000000Z"]
    assert not (dest / "20260928T000000Z" / "manifest.json").exists()


def test_rsync_transient_failure_is_retried(roots, tmp_path):
    src, dest = roots
    a = make_object(src, {"n": 1})
    make_snapshot(src, "20260928T000000Z", [a, a])
    counter = tmp_path / "count"
    wrapper = tmp_path / "flaky-rsync"
    wrapper.write_text(f"""#!/bin/sh
n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}
if [ "$n" -le 2 ]; then echo "rsync: read error: Interrupted system call (4)" >&2; exit 12; fi
exec {RSYNC} "$@"
""")
    wrapper.chmod(0o755)
    report = mm.mirror(src, dest, runner=mm.RsyncRunner(binary=str(wrapper), sleep=NO_SLEEP), sleep=NO_SLEEP)
    assert report.ok and report.snapshots_mirrored == 1
    assert int(counter.read_text()) >= 3


def test_rsync_permanent_failure_reports_error(roots, tmp_path):
    src, dest = roots
    a = make_object(src, {"n": 1})
    make_snapshot(src, "20260928T000000Z", [a, a])
    failing = tmp_path / "bad-rsync"
    failing.write_text("#!/bin/sh\necho nope >&2\nexit 23\n")
    failing.chmod(0o755)
    report = mm.mirror(src, dest, runner=mm.RsyncRunner(binary=str(failing), attempts=2, sleep=NO_SLEEP), sleep=NO_SLEEP)
    assert not report.ok and any("after 2 attempts" in e for e in report.errors)
    assert report.snapshots_deferred


def test_unmounted_share_refuses_before_touching_anything(roots):
    src, dest = roots
    a = make_object(src, {"n": 1})
    make_snapshot(src, "20260928T000000Z", [a, a])
    with pytest.raises(mm.ShareNotMounted) as info:
        mm.mirror(src, dest, runner=runner(), mount=Path("/Volumes/akh-medu-runs"), is_mount=lambda _p: False, sleep=NO_SLEEP)
    assert "not a mount point" in str(info.value) and not dest.exists()


def test_cli_exit_codes_and_summary(roots, capsys):
    src, dest = roots
    a = make_object(src, {"n": 1})
    make_snapshot(src, "20260928T000000Z", [a, a])
    args = ["--src", str(src), "--dest", str(dest), "--rsync", RSYNC]
    assert mm.main([*args, "--require-mount", "/definitely/not/a/mount"]) == mm.EXIT_NOT_MOUNTED
    assert not dest.exists() and "not a mount point" in capsys.readouterr().err
    assert mm.main([*args, "--no-require-mount"]) == mm.EXIT_OK
    assert json.loads(capsys.readouterr().out)["snapshots_mirrored"] == 1
    assert mm.main([*args, "--no-require-mount"]) == mm.EXIT_OK  # idempotent
    assert mm.main(["--src", str(src / "nope"), "--dest", str(dest), "--no-require-mount"]) == mm.EXIT_FAILED
