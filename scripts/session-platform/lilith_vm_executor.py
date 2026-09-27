#!/usr/bin/env python3
"""Bounded SSH executor for Lilith's disposable illumos validation VM."""

from __future__ import annotations

import hashlib
import json
import base64
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tarfile
import tempfile
import time
import uuid


MAX_TIMEOUT_S = 60
MAX_OUTPUT_BYTES = 65_536
MAX_SNAPSHOT_BYTES = 32 * 1024 * 1024

SUPERVISOR = r'''import base64,json,os,resource,signal,subprocess,sys
argv=json.loads(base64.b64decode(sys.argv[1]))
seconds=int(sys.argv[2])
status=sys.argv[3]
resource.setrlimit(resource.RLIMIT_FSIZE,(65536,65536))
resource.setrlimit(resource.RLIMIT_CPU,(seconds,seconds))
try:
    resource.setrlimit(resource.RLIMIT_NPROC,(64,64))
except (AttributeError,ValueError):
    pass
p=subprocess.Popen(argv,start_new_session=True)
try:
    code=p.wait(timeout=seconds)
    open(status,'w').write('exit\n')
    raise SystemExit(code)
except subprocess.TimeoutExpired:
    os.killpg(p.pid,signal.SIGKILL)
    p.wait()
    open(status,'w').write('timeout\n')
    raise SystemExit(124)
'''


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_target(target: str) -> PurePosixPath:
    path = PurePosixPath(target)
    if not target or path.is_absolute() or ".." in path.parts or "." == target:
        raise ValueError("target must be a non-empty relative guest path")
    return path


def _snapshot(repository: Path, archive: Path) -> tuple[str, int, int]:
    repository = repository.resolve(strict=True)
    if not repository.is_dir():
        raise ValueError("repository must be a directory")
    entries: list[Path] = []
    total = 0
    for root, dirs, files in os.walk(repository, followlinks=False):
        root_path = Path(root)
        for name in sorted(dirs + files):
            item = root_path / name
            if item.is_symlink():
                raise ValueError(f"snapshot contains symlink: {item.relative_to(repository)}")
            if item.is_file():
                total += item.stat().st_size
                if total > MAX_SNAPSHOT_BYTES:
                    raise ValueError("snapshot exceeds 32 MiB limit")
            elif not item.is_dir():
                raise ValueError(f"snapshot contains special file: {item.relative_to(repository)}")
            entries.append(item)
    with tarfile.open(archive, "w") as tar:
        for item in sorted(entries, key=lambda p: p.relative_to(repository).as_posix()):
            tar.add(item, arcname=item.relative_to(repository), recursive=False)
    payload = archive.read_bytes()
    return _sha256(payload), len(payload), len(entries)


class IllumosVmExecutor:
    execution_boundary = "ssh_guest"

    def __init__(
        self, *, host: str, port: int, user: str, identity: Path,
        rootfs_sha256: str, host_key_fingerprint: str, known_hosts: Path,
    ):
        self.host = host
        self.port = int(port)
        self.user = user
        self.identity = identity.resolve(strict=True)
        if not re.fullmatch(r"[0-9a-f]{64}", rootfs_sha256):
            raise ValueError("rootfs_sha256 must be a SHA-256 hex digest")
        self.rootfs_sha256 = rootfs_sha256
        self.host_key_fingerprint = host_key_fingerprint
        self.known_hosts = known_hosts.resolve(strict=True)
        listing = subprocess.run(
            ["ssh-keygen", "-lf", str(self.known_hosts)], capture_output=True, text=True, check=True
        ).stdout
        if host_key_fingerprint not in listing:
            raise ValueError("declared SSH host-key fingerprint is absent from known_hosts")

    def contract(self) -> dict[str, object]:
        return {
            "schema": "lilith/native-executor-contract@1",
            "execution_boundary": self.execution_boundary,
            "guest_identity": "lilith-illumos-20260922",
            "rootfs_sha256": self.rootfs_sha256,
            "executor_source_sha256": _sha256(Path(__file__).read_bytes()),
            "supervisor_sha256": _sha256(SUPERVISOR.encode()),
            "known_hosts_sha256": _sha256(self.known_hosts.read_bytes()),
            "ssh": {
                "host": self.host,
                "port": self.port,
                "user": self.user,
                "host_key_fingerprint": self.host_key_fingerprint,
            },
            "staging_protocol": "tar-stdin+exact-content-v1",
            "max_timeout_s": MAX_TIMEOUT_S,
            "max_output_bytes": MAX_OUTPUT_BYTES,
            "max_snapshot_bytes": MAX_SNAPSHOT_BYTES,
            "cleanup_policy": "unique-run-directory-removed-after-every-attempt",
            "privilege_policy": "unprivileged akh-exec; executor never invokes sudo",
            "host_exposure": "SLIRP loopback SSH only; no host mounts or forwarded secrets",
        }

    def _ssh(self) -> list[str]:
        return [
            "ssh", "-T", "-i", str(self.identity), "-p", str(self.port),
            "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=yes",
            "-o", f"UserKnownHostsFile={self.known_hosts}",
            "-o", "ConnectTimeout=5",
            f"{self.user}@{self.host}",
        ]

    def execute(
        self,
        repository: Path,
        target: str,
        content: str,
        command: list[str] | tuple[str, ...],
        *,
        timeout_s: int = 30,
        output_limit: int = MAX_OUTPUT_BYTES,
    ) -> dict[str, object]:
        target_path = _safe_target(target)
        if not command or any("\x00" in arg for arg in command):
            raise ValueError("command must contain valid argv")
        if not 1 <= timeout_s <= MAX_TIMEOUT_S:
            raise ValueError("timeout must be between 1 and 60 seconds")
        if not 1 <= output_limit <= MAX_OUTPUT_BYTES:
            raise ValueError("output limit must be between 1 and 65536 bytes")

        run_id = uuid.uuid4().hex
        remote = f"/var/tmp/lilith-native/{run_id}"
        started = time.monotonic()
        cleanup_succeeded = False
        with tempfile.TemporaryDirectory(prefix="lilith-native-") as temp:
            archive = Path(temp) / "snapshot.tar"
            snapshot_sha, snapshot_bytes, entries = _snapshot(Path(repository), archive)
            try:
                mkdir = subprocess.run(
                    self._ssh() + [f"mkdir -p {shlex.quote(remote)}"],
                    capture_output=True, timeout=10,
                )
                if mkdir.returncode:
                    raise RuntimeError(mkdir.stderr.decode("utf-8", "replace")[:output_limit])
                unpack = subprocess.run(
                    self._ssh() + [f"cd {shlex.quote(remote)} && gtar -xf -"],
                    input=archive.read_bytes(), capture_output=True, timeout=20,
                )
                if unpack.returncode:
                    raise RuntimeError(unpack.stderr.decode("utf-8", "replace")[:output_limit])
                remote_target = f"{remote}/{target_path.as_posix()}"
                write = subprocess.run(
                    self._ssh() + [
                        f"mkdir -p {shlex.quote(str(PurePosixPath(remote_target).parent))} && "
                        f"dd of={shlex.quote(remote_target)}"
                    ],
                    input=content.encode(), capture_output=True, timeout=10,
                )
                if write.returncode:
                    raise RuntimeError(write.stderr.decode("utf-8", "replace")[:output_limit])
                supervisor = subprocess.run(
                    self._ssh() + [f"dd of={shlex.quote(remote + '/.supervisor.py')}"],
                    input=SUPERVISOR.encode(), capture_output=True, timeout=10,
                )
                if supervisor.returncode:
                    raise RuntimeError(supervisor.stderr.decode("utf-8", "replace")[:output_limit])
                argv_b64 = base64.b64encode(json.dumps(list(command)).encode()).decode()
                remote_command = (
                    f"cd {shlex.quote(remote)} && "
                    f"python3 .supervisor.py {shlex.quote(argv_b64)} {timeout_s} .supervisor-status "
                    f">.stdout 2>.stderr; printf '%s' $? >.exit"
                )
                host_timed_out = False
                try:
                    run = subprocess.run(
                        self._ssh() + [remote_command], capture_output=True,
                        timeout=timeout_s + 8,
                    )
                except subprocess.TimeoutExpired:
                    host_timed_out = True
                    run = subprocess.CompletedProcess(command, 124, b"", b"host SSH timeout")
                if run.returncode and not host_timed_out:
                    raise RuntimeError(run.stderr.decode("utf-8", "replace")[:output_limit])

                def fetch(name: str) -> bytes:
                    fetched = subprocess.run(
                        self._ssh() + [f"head -c {MAX_OUTPUT_BYTES} {shlex.quote(remote + '/' + name)}"],
                        capture_output=True, timeout=10,
                    )
                    return fetched.stdout

                stdout_full = fetch(".stdout") if not host_timed_out else b""
                stderr_full = fetch(".stderr") if not host_timed_out else b"host SSH timeout"
                exit_raw = fetch(".exit") if not host_timed_out else b"124"
                exit_code = int(exit_raw.decode().strip() or "125")
                supervisor_status = fetch(".supervisor-status").decode().strip() if not host_timed_out else "host_timeout"
                timed_out = host_timed_out or supervisor_status == "timeout"
                sizes = subprocess.run(
                    self._ssh() + [f"wc -c {shlex.quote(remote + '/.stdout')} {shlex.quote(remote + '/.stderr')}"],
                    capture_output=True, text=True, timeout=10,
                ).stdout
                size_values = [int(line.split()[0]) for line in sizes.splitlines()[:2]]
            finally:
                cleanup = subprocess.run(
                    self._ssh() + [f"rm -rf {shlex.quote(remote)}"],
                    capture_output=True, timeout=10,
                )
                cleanup_succeeded = cleanup.returncode == 0
        return {
            "schema": "lilith/native-execution@1",
            "run_id": run_id,
            "guest": {"host": self.host, "port": self.port, "user": self.user},
            "target": target_path.as_posix(),
            "command": list(command),
            "timeout_s": timeout_s,
            "timed_out": timed_out,
            "supervisor_status": supervisor_status,
            "exit_code": exit_code,
            "passed": exit_code == 0 and not timed_out and cleanup_succeeded,
            "duration_ms": round((time.monotonic() - started) * 1000),
            "copy_in": {
                "snapshot_sha256": snapshot_sha,
                "snapshot_bytes": snapshot_bytes,
                "snapshot_entries": entries,
                "content_sha256": _sha256(content.encode()),
                "content_bytes": len(content.encode()),
            },
            "stdout": stdout_full[:output_limit].decode("utf-8", "replace"),
            "stderr": stderr_full[:output_limit].decode("utf-8", "replace"),
            "stdout_sha256": _sha256(stdout_full),
            "stderr_sha256": _sha256(stderr_full),
            "stdout_truncated": size_values[0] >= MAX_OUTPUT_BYTES or size_values[0] > output_limit,
            "stderr_truncated": size_values[1] >= MAX_OUTPUT_BYTES or size_values[1] > output_limit,
            "remote_cleanup_succeeded": cleanup_succeeded,
        }


def write_receipt(receipt: dict[str, object], destination: Path) -> None:
    destination.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def local_executor() -> IllumosVmExecutor:
    """Return the pinned local Lilith illumos executor (credentials stay outside git)."""
    state = Path("/home/toasty/.local/share/akh-medu/lilith-native-20260922")
    return IllumosVmExecutor(
        host="127.0.0.1",
        port=10115,
        user="akh-exec",
        identity=state / "credentials/id_ed25519",
        known_hosts=state / "known_hosts",
        rootfs_sha256="437edebceb20931ffee43d0838b0caf284e174368a1def421128a4375ac4f8ee",
        host_key_fingerprint="SHA256:9XvZc8Pd/zkdM9zOqOQEXp2Z0ZDej3IPI3ommnQeVj0",
    )
