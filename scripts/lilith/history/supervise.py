#!/usr/bin/env python3
"""Host cost guard for an explicitly approved pilot; uses own OpenBao delivery.

Never creates/restarts/deletes a pod. Stops the approved pod on completion or
the two-hour deadline, including persistent SSH failure, and verifies stop state.
Run this before starting the pod workload. Store evidence on the network volume.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

from run_pod import RUN_ID

AGENT_TOOLS = Path.home() / ".local/bin/agent-tools"


def stop_pod(pod, runner=subprocess.run):
    stop = runner(
        [str(AGENT_TOOLS), "runpod", "--", "runpodctl", "pod", "stop", pod],
        capture_output=True,
        text=True,
        timeout=45,
    )
    if stop.returncode:
        raise RuntimeError("pod stop failed; operator must check billing immediately")
    for _ in range(6):
        state = runner(
            [str(AGENT_TOOLS), "runpod", "--", "runpodctl", "pod", "get", pod],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if state.returncode == 0:
            body = json.loads(state.stdout)
            if (
                body.get("runtimeStatus") == "stopped"
                and body.get("desiredStatus") == "EXITED"
            ):
                return {"runtimeStatus": "stopped", "desiredStatus": "EXITED"}
        time.sleep(5)
    raise RuntimeError(
        "pod stop was requested but stopped state is unconfirmed; check billing"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--approval", type=Path, required=True)
    p.add_argument("--pod-id", required=True)
    p.add_argument("--ssh", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--identity", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    args = p.parse_args()
    consent = json.loads(args.approval.read_text())
    if (
        consent.get("decision") != "go"
        or consent.get("run_id") != RUN_ID
        or consent.get("pod_id") != args.pod_id
        or consent.get("stop_pod_at_completion_or_deadline") is not True
    ):
        p.error("explicit go binding this pod and its stop deadline is required")
    if (
        not re.fullmatch(r"[a-zA-Z0-9]+", args.pod_id)
        or not re.fullmatch(r"[\w.-]+@[\w.:-]+", args.ssh)
        or not 1 <= args.port <= 65535
    ):
        p.error("invalid pod/SSH endpoint")
    created = datetime.fromisoformat(consent["pod_created_at"].replace("Z", "+00:00"))
    if created.tzinfo is None or created > datetime.now(timezone.utc):
        p.error(
            "pod creation time must be a dated, non-future control-plane observation"
        )
    deadline = created.timestamp() + 7200
    args.dest.mkdir(parents=True, mode=0o700, exist_ok=False)
    known = args.dest / "known_hosts"
    opts = [
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={known}",
        "-i",
        str(args.identity),
        "-o",
        "IdentitiesOnly=yes",
    ]
    remote = f"/workspace/lilith/runs/{RUN_ID}"
    report = {
        "schema": "lilith/history-supervisor@1",
        "pod_id": args.pod_id,
        "run_id": RUN_ID,
        "deadline_utc": datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
        "artifacts_verified": False,
    }
    try:
        while time.time() < deadline:
            poll = subprocess.run(
                [
                    "ssh",
                    *opts,
                    "-p",
                    str(args.port),
                    args.ssh,
                    "cat " + shlex.quote(remote + "/history-result.json"),
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if poll.returncode == 0:
                marker = json.loads(poll.stdout)
                if marker.get("run_id") != RUN_ID or marker.get("status") not in (
                    "passed",
                    "failed",
                ):
                    raise ValueError("invalid completion marker")
                report["run_status"] = marker["status"]
                copy = subprocess.run(
                    [
                        "scp",
                        *opts,
                        "-P",
                        str(args.port),
                        "-r",
                        args.ssh + ":" + remote,
                        str(args.dest / "artifacts"),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=min(120, max(1, deadline - time.time())),
                )
                if copy.returncode:
                    raise RuntimeError(
                        "artifact copy failed; persistent volume retains original"
                    )
                from train import sha

                for name, expected in marker["files_sha256"].items():
                    relative = Path(name)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("unsafe artifact path in marker")
                    if sha(args.dest / "artifacts" / relative) != expected:
                        raise ValueError("artifact checksum mismatch")
                report["artifacts_verified"] = True
                break
            time.sleep(min(30, max(0, deadline - time.time())))
        else:
            report["run_status"] = "deadline_exhausted"
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            report["pod_stop"] = stop_pod(args.pod_id)
        except BaseException as error:
            report["stop_error"] = f"{type(error).__name__}: {error}"
        (args.dest / "supervisor.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.dest / "supervisor.json").chmod(0o600)
    print(json.dumps(report))
    return (
        0
        if report.get("artifacts_verified")
        and report.get("run_status") == "passed"
        and "pod_stop" in report
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
