"""Atomic training-run status and checkpoint-and-stop control protocol.

The daemon reads ``training-status.json`` and may atomically publish
``training-control.json``.  The reporter never evaluates policy: it only
publishes facts and honors the single supported command after a checkpoint is
known to exist.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import threading
import time
from pathlib import Path
from typing import Any


class TrainingRunReporter:
    def __init__(
        self,
        run_id: str,
        run_dir: Path,
        *,
        total_steps: int | None = None,
        heartbeat_seconds: float = 10.0,
        checkpoint_stable_seconds: float = 2.0,
    ) -> None:
        self.run_id = run_id
        self.run_dir = run_dir
        self.status_path = run_dir / "training-status.json"
        self.control_path = run_dir / "training-control.json"
        self.total_steps = total_steps
        self.heartbeat_seconds = heartbeat_seconds
        self.checkpoint_stable_seconds = checkpoint_stable_seconds
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._status: dict[str, Any] = {
            "run_id": run_id,
            "state": "starting",
            "updated_at": int(time.time()),
            "progress_updated_at": int(time.time()),
            "step": 0,
            "total_steps": total_steps,
            "metrics": {},
        }
        self._control_seen = False
        self._control_pending_reported = False
        self._last_checkpoint_mtime_ns: int | None = None
        self._checkpoint_lock = threading.Lock()
        self._checkpoint_fingerprint: tuple[str, int, int] | None = None
        self._checkpoint_unchanged_since: float | None = None
        self._child_process_group: int | None = None

    def start(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.publish()
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"training-status-{self.run_id}",
            daemon=True,
        )
        self._thread.start()

    def update(self, state: str, **facts: Any) -> None:
        with self._lock:
            self._status["state"] = state
            self._status["progress_updated_at"] = int(time.time())
            self._status.update(facts)
        self.publish()

    def publish(self) -> None:
        with self._lock:
            status = dict(self._status)
        status["updated_at"] = int(time.time())
        try:
            status["disk_free_bytes"] = shutil.disk_usage(self.run_dir).free
        except OSError:
            pass
        checkpoint = self._latest_checkpoint_candidate()
        if checkpoint is not None:
            status["checkpoint_path"] = str(checkpoint)
            try:
                checkpoint_mtime_ns = checkpoint.stat().st_mtime_ns
            except OSError:
                checkpoint_mtime_ns = None
            if checkpoint_mtime_ns is not None:
                with self._lock:
                    if checkpoint_mtime_ns != self._last_checkpoint_mtime_ns:
                        self._last_checkpoint_mtime_ns = checkpoint_mtime_ns
                        self._status["progress_updated_at"] = int(time.time())
                    status["progress_updated_at"] = self._status["progress_updated_at"]
                status.setdefault("metrics", {})["checkpoint_mtime_ns"] = checkpoint_mtime_ns
        status["checkpoint_durable"] = self._durable_checkpoint() is not None
        _atomic_json(self.status_path, status)

    def finish(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.heartbeat_seconds * 2))
        self.publish()

    @property
    def checkpoint_stop_requested(self) -> bool:
        return self._control_seen

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            self.publish()
            self._honor_control_if_safe()

    def attach_child_process_group(self, process_group: int) -> None:
        with self._checkpoint_lock:
            self._child_process_group = process_group

    def detach_child_process_group(self, process_group: int) -> None:
        with self._checkpoint_lock:
            if self._child_process_group == process_group:
                self._child_process_group = None

    def _latest_checkpoint_candidate(self) -> Path | None:
        candidates = [
            self.run_dir / "adapter.safetensors",
            self.run_dir / "adapter.bin",
        ]
        checkpoint_dir = self.run_dir / "checkpoints"
        if checkpoint_dir.exists():
            candidates.extend(path for path in checkpoint_dir.iterdir() if path.is_file())
        existing: list[Path] = []
        for path in candidates:
            try:
                if path.is_file() and path.stat().st_size > 0:
                    existing.append(path)
            except OSError:
                continue
        return max(existing, key=lambda path: path.stat().st_mtime_ns, default=None)

    def _durable_checkpoint(self) -> Path | None:
        checkpoint = self._latest_checkpoint_candidate()
        if checkpoint is None:
            return None
        try:
            stat = checkpoint.stat()
        except OSError:
            return None
        fingerprint = (str(checkpoint), stat.st_size, stat.st_mtime_ns)
        now = time.monotonic()
        with self._checkpoint_lock:
            if fingerprint != self._checkpoint_fingerprint:
                self._checkpoint_fingerprint = fingerprint
                self._checkpoint_unchanged_since = now
                return None
            unchanged_since = self._checkpoint_unchanged_since
        if unchanged_since is None:
            return None
        if now - unchanged_since < self.checkpoint_stable_seconds:
            return None
        return checkpoint

    @property
    def latest_durable_checkpoint(self) -> Path | None:
        return self._durable_checkpoint()

    def wait_for_durable_checkpoint(self, timeout_seconds: float = 30.0) -> Path:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            checkpoint = self._durable_checkpoint()
            if checkpoint is not None:
                self.publish()
                return checkpoint
            time.sleep(min(0.1, max(0.01, self.checkpoint_stable_seconds / 2)))
        raise TimeoutError("checkpoint did not become stable before timeout")

    def _honor_control_if_safe(self) -> None:
        if self._control_seen or not self.control_path.exists():
            return
        try:
            request = json.loads(self.control_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        if request.get("run_id") != self.run_id:
            return
        if request.get("command") != "checkpoint_and_stop":
            return
        checkpoint = self._durable_checkpoint()
        if checkpoint is None:
            if not self._control_pending_reported:
                self._control_pending_reported = True
                self.update(
                    "checkpoint_stop_pending",
                    message="checkpoint-and-stop requested; waiting for a durable checkpoint",
                )
            return
        self._control_seen = True
        self.update(
            "stopping",
            checkpoint_path=str(checkpoint),
            message="checkpoint-and-stop requested by deterministic supervisor",
        )
        with self._checkpoint_lock:
            child_process_group = self._child_process_group
        if child_process_group is not None and hasattr(os, "killpg"):
            os.killpg(child_process_group, signal.SIGINT)
        # Python delivers SIGINT to the main thread. The training wrapper catches
        # KeyboardInterrupt, preserves the checkpoint path, and exits cleanly.
        os.kill(os.getpid(), signal.SIGINT)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}.{threading.get_ident()}")
    data = json.dumps(value, indent=2, sort_keys=True)
    with open(tmp, "w") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
