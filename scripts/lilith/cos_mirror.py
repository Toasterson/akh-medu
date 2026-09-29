#!/usr/bin/env python3
"""One-shot mirror of Lilith CoS snapshots from the local root to the NFS share
(Anima WI 01a0e270-137c, plan slice C3, ADR 116 Decision 3).

Why this exists: `cos_recorder.py` writes to a *local* root because writing the
NFS share directly from a launchd job hit EINTR mid-run and left half-written
snapshots. The plan wants the private data on NFS, so this second, separate job
copies only *completed* snapshots across and verifies them.

Guarantees:

* Completed snapshots only: a `<UTC-ts>/manifest.json` is written last and
  atomically by the recorder, so its presence means the snapshot is complete.
* Objects before manifests: a snapshot's `manifest.json` is copied only after
  every object it references is present and verified on the share. The share
  therefore never holds a manifest that points at a missing object.
* Verified: every object copied is re-read from the share and its sha256 must
  equal its file name (objects are content-addressed); a copied manifest must
  match the local manifest's sha256.
* Idempotent: existing files are skipped (`--ignore-existing`); a second run
  copies nothing. An existing object whose size differs from the local one is
  treated as damaged and re-copied with `--checksum`.
* Never deletes on either side. A differing manifest already on the share is a
  conflict: reported, never overwritten.
* Retries transient failures (EINTR, NFS hiccups, rsync non-zero exits) with
  exponential backoff.
* Refuses to run when the share is not mounted (exit 2): writing under an
  unmounted `/Volumes/<share>` would silently fill the boot volume.

Uses `/usr/bin/rsync` (macOS openrsync, rsync 2.6.9 compatible), so only flags
that exist there: `-t --ignore-existing --files-from`. No `-a` (ownership and
permission bits are not preserved on NFS and would fail as non-root).

Exit codes: 0 all mirrored and verified; 1 usage; 2 share not mounted;
3 copy/verification failure or conflict (details in the JSON summary).

Usage:
    cos_mirror.py [--src DIR] [--dest DIR] [--require-mount PATH]
"""
from __future__ import annotations

import argparse
import dataclasses
import errno
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Sequence

DEFAULT_SRC = "~/Library/Application Support/akh-medu/lilith/cos-snapshots"
DEFAULT_DEST = "/Volumes/akh-medu-runs/lilith/cos-snapshots"
DEFAULT_MOUNT = "/Volumes/akh-medu-runs"
DEFAULT_RSYNC = "/usr/bin/rsync"

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_NOT_MOUNTED = 2
EXIT_FAILED = 3

_SNAPSHOT_RE = re.compile(r"^\d{8}T\d{6}Z$")
_OBJECT_RE = re.compile(r"^[0-9a-f]{64}\.json$")
_TRANSIENT_ERRNOS = {errno.EINTR, errno.EIO, errno.ESTALE, errno.ETIMEDOUT, errno.EAGAIN}


class MirrorError(RuntimeError):
    """Base for mirror failures that map to an exit code."""

    exit_code = EXIT_FAILED


class ShareNotMounted(MirrorError):
    exit_code = EXIT_NOT_MOUNTED


def sha256_file(path: Path, attempts: int = 4, sleep: Callable[[float], None] = time.sleep) -> str:
    """Streaming sha256, retrying transient read errors (EINTR/ESTALE/EIO on NFS)."""
    last: OSError | None = None
    for attempt in range(attempts):
        try:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except OSError as error:
            if error.errno not in _TRANSIENT_ERRNOS:
                raise
            last = error
            sleep(0.2 * (2**attempt))
    assert last is not None
    raise last


@dataclasses.dataclass
class RsyncRunner:
    """Runs `rsync -t --ignore-existing --files-from=<list> src/ dest/` with retries."""

    binary: str = DEFAULT_RSYNC
    attempts: int = 4
    initial_delay: float = 2.0
    sleep: Callable[[float], None] = time.sleep

    def copy(self, src: Path, dest: Path, relative_files: Sequence[str], *, checksum: bool = False,
             ignore_existing: bool = True) -> None:
        if not relative_files:
            return
        with tempfile.NamedTemporaryFile("w", suffix=".list", delete=False) as handle:
            handle.write("\n".join(relative_files) + "\n")
            list_path = handle.name
        try:
            cmd = [self.binary, "-t"]
            if ignore_existing:
                cmd.append("--ignore-existing")
            if checksum:
                cmd.append("--checksum")
            cmd += [f"--files-from={list_path}", f"{src}/", f"{dest}/"]
            last = ""
            for attempt in range(self.attempts):
                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                except (OSError, subprocess.SubprocessError) as error:
                    last = f"{type(error).__name__}: {error}"
                else:
                    if proc.returncode == 0:
                        return
                    last = f"rsync exit {proc.returncode}: {(proc.stderr or proc.stdout).strip()[:300]}"
                if attempt + 1 < self.attempts:
                    self.sleep(self.initial_delay * (2**attempt))
            raise MirrorError(f"rsync failed after {self.attempts} attempts: {last}")
        finally:
            os.unlink(list_path)


@dataclasses.dataclass
class MirrorReport:
    snapshots_seen: int = 0
    snapshots_mirrored: int = 0
    snapshots_already_present: int = 0
    snapshots_deferred: list[str] = dataclasses.field(default_factory=list)
    objects_copied: int = 0
    objects_repaired: int = 0
    objects_failed: list[str] = dataclasses.field(default_factory=list)
    manifest_conflicts: list[str] = dataclasses.field(default_factory=list)
    skipped_unreadable: list[str] = dataclasses.field(default_factory=list)
    errors: list[str] = dataclasses.field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.snapshots_deferred or self.objects_failed or self.manifest_conflicts or self.errors)

    def to_dict(self) -> dict:
        return {**dataclasses.asdict(self), "ok": self.ok}


def require_mounted(mount: Path | None, is_mount: Callable[[str], bool] = os.path.ismount) -> None:
    if mount is not None and not is_mount(str(mount)):
        raise ShareNotMounted(
            f"{mount} is not a mount point; refusing to write to {mount} on the local disk. "
            "Mount the NFS share (Finder or `mount_nfs`) and rerun."
        )


def completed_snapshots(src: Path, report: MirrorReport) -> list[Path]:
    out = []
    for child in sorted(src.iterdir()) if src.is_dir() else []:
        if not (_SNAPSHOT_RE.match(child.name) and child.is_dir()):
            continue
        manifest = child / "manifest.json"
        if not manifest.is_file():
            continue  # no manifest yet = recorder still writing (manifest is written last)
        try:
            json.loads(manifest.read_bytes())
        except (OSError, ValueError):
            report.skipped_unreadable.append(child.name)
            continue
        out.append(child)
    return out


def referenced_objects(manifest_path: Path) -> set[str]:
    manifest = json.loads(manifest_path.read_bytes())
    shas: set[str] = set()
    for collector in manifest.get("collectors", {}).values():
        shas.update(r["sha256"] for r in collector.get("responses", []) if r.get("sha256"))
        if collector.get("tool_contract_sha256"):
            shas.add(collector["tool_contract_sha256"])
    return shas


def _dest_sizes(directory: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    if directory.is_dir():
        with os.scandir(directory) as entries:
            for entry in entries:
                if _OBJECT_RE.match(entry.name):
                    sizes[entry.name] = entry.stat().st_size
    return sizes


def mirror(
    src: Path,
    dest: Path,
    *,
    runner: RsyncRunner | None = None,
    mount: Path | None = None,
    is_mount: Callable[[str], bool] = os.path.ismount,
    sleep: Callable[[float], None] = time.sleep,
) -> MirrorReport:
    runner = runner or RsyncRunner(sleep=sleep)
    require_mounted(mount, is_mount)
    report = MirrorReport()
    if not src.is_dir():
        raise MirrorError(f"source root {src} does not exist")
    dest.mkdir(parents=True, exist_ok=True)  # safe: the mount check above proves the share is present
    snapshots = completed_snapshots(src, report)
    report.snapshots_seen = len(snapshots)

    dest_sizes = _dest_sizes(dest / "objects")
    todo = [s for s in snapshots if not (dest / s.name / "manifest.json").exists()]
    report.snapshots_already_present = len(snapshots) - len(todo)

    # Existing manifests must agree with local ones; a difference is a conflict, never overwritten.
    for snap in snapshots:
        existing = dest / snap.name / "manifest.json"
        if existing.exists() and sha256_file(existing, sleep=sleep) != sha256_file(snap / "manifest.json", sleep=sleep):
            report.manifest_conflicts.append(snap.name)

    # Objects needed by snapshots still to mirror: copy those missing on the share, and
    # re-copy (with --checksum) any whose size differs from the local object.
    needed: set[str] = set()
    for snap in todo:
        needed |= {f"{sha}.json" for sha in referenced_objects(snap / "manifest.json")}
    missing, damaged = [], []
    for name in sorted(needed):
        local = src / "objects" / name
        try:
            local_size = local.stat().st_size
        except OSError:
            report.objects_failed.append(f"{name}: missing locally")
            continue
        if name not in dest_sizes:
            missing.append(name)
        elif dest_sizes[name] != local_size:
            damaged.append(name)

    failed_names = {f.split(":", 1)[0] for f in report.objects_failed}
    try:
        runner.copy(src, dest, [f"objects/{n}" for n in missing])
        if damaged:
            runner.copy(src, dest, [f"objects/{n}" for n in damaged], checksum=True, ignore_existing=False)
    except MirrorError as error:
        report.errors.append(str(error))

    for name in missing + damaged:
        target = dest / "objects" / name
        try:
            ok = target.is_file() and sha256_file(target, sleep=sleep) == name[: -len(".json")]
        except OSError as error:
            ok = False
            report.errors.append(f"verify {name}: {error}")
        if ok:
            if name in damaged:
                report.objects_repaired += 1
            else:
                report.objects_copied += 1
        else:
            report.objects_failed.append(f"{name}: failed verification after copy")
            failed_names.add(name)

    # Manifests last, and only for snapshots whose every object is verified on the share.
    bad = failed_names | {f.split(":", 1)[0] for f in report.objects_failed}
    ready: list[Path] = []
    for snap in todo:
        wanted = {f"{sha}.json" for sha in referenced_objects(snap / "manifest.json")}
        if wanted & bad:
            report.snapshots_deferred.append(snap.name)
        else:
            ready.append(snap)
    try:
        runner.copy(src, dest, [f"{s.name}/manifest.json" for s in ready])
    except MirrorError as error:
        report.errors.append(str(error))
    for snap in ready:
        target = dest / snap.name / "manifest.json"
        try:
            good = target.is_file() and sha256_file(target, sleep=sleep) == sha256_file(snap / "manifest.json", sleep=sleep)
        except OSError as error:
            good = False
            report.errors.append(f"verify {snap.name}/manifest.json: {error}")
        if good:
            report.snapshots_mirrored += 1
        else:
            report.snapshots_deferred.append(snap.name)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mirror completed Lilith CoS snapshots to the NFS share")
    parser.add_argument("--src", type=Path, default=None, help=f"local snapshot root (env LILITH_SNAPSHOT_ROOT, default {DEFAULT_SRC})")
    parser.add_argument("--dest", type=Path, default=None, help=f"share snapshot root (env LILITH_MIRROR_DEST, default {DEFAULT_DEST})")
    parser.add_argument("--require-mount", type=Path, default=None,
                        help=f"mount point that must exist before writing (env LILITH_MIRROR_MOUNT, default {DEFAULT_MOUNT})")
    parser.add_argument("--no-require-mount", action="store_true", help="skip the mount check (tests / non-NFS targets)")
    parser.add_argument("--rsync", default=DEFAULT_RSYNC, help="rsync binary")
    parser.add_argument("--attempts", type=int, default=4, help="rsync attempts per batch")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    src = (args.src or Path(os.environ.get("LILITH_SNAPSHOT_ROOT", DEFAULT_SRC))).expanduser()
    dest = args.dest or Path(os.environ.get("LILITH_MIRROR_DEST", DEFAULT_DEST))
    mount = None if args.no_require_mount else (args.require_mount or Path(os.environ.get("LILITH_MIRROR_MOUNT", DEFAULT_MOUNT)))
    runner = RsyncRunner(binary=args.rsync, attempts=args.attempts)
    try:
        report = mirror(src, dest, runner=runner, mount=mount)
    except MirrorError as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True), file=sys.stderr)
        return error.exit_code
    print(json.dumps(report.to_dict(), sort_keys=True))
    return EXIT_OK if report.ok else EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
