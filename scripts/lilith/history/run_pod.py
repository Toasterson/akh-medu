#!/usr/bin/env python3
"""One-shot historical pilot on an already approved pod; never provisions one.

The host supervisor must already be running with a two-hour stop deadline.
All outputs stay on the network volume; the completion marker is written last.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import train

RUN_ID = "lilith-history-r1-20261001"


def seal(out, report):
    files = {
        str(p.relative_to(out)): train.sha(p)
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name not in ("history-result.json", "SHA256SUMS")
    }
    report["files_sha256"] = files
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    # Includes failures and partial results, without pretending missing artifacts passed.
    (out / "SHA256SUMS").write_text("".join(f"{v}  {k}\n" for k, v in files.items()))
    staged = out / ".history-result.tmp"
    staged.write_text(json.dumps(report, indent=2) + "\n")
    staged.replace(out / "history-result.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--approval", type=Path, required=True)
    p.add_argument("--main-sha", required=True)
    p.add_argument("--root", type=Path, default=Path("/workspace/lilith"))
    args = p.parse_args()
    os.umask(0o077)
    train.check_inputs(args.data, args.approval)
    consent = json.loads(args.approval.read_text())
    if (
        consent.get("main_sha") != args.main_sha
        or consent.get("stop_pod_at_completion_or_deadline") is not True
    ):
        p.error("go must bind the main source and authorize the bounded pod stop")
    if not str(args.root.resolve()).startswith("/workspace/"):
        p.error("outputs must live on the persistent /workspace network volume")
    payload = Path(__file__).resolve().parents[3]
    provenance = train.verify_payload.verify(
        payload, payload / "source-manifest.json", args.main_sha, False
    )
    if provenance["status"] != "verified":
        p.error(
            "main-only payload provenance failed: " + "; ".join(provenance["problems"])
        )
    out = args.root / "runs" / RUN_ID
    out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + 6600
    report = {
        "schema": "lilith/historical-pilot@1",
        "run_id": RUN_ID,
        "status": "failed",
        "source": provenance,
        "quality_claim": False,
        "release_selected": False,
        "adapter_activated": False,
        "steps": {},
    }
    python = str(args.root / "env/bin/python")
    env = {
        **os.environ,
        "LILITH_ROOT": str(args.root),
        "HF_HUB_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1",
        "PATH": str(args.root / "env/bin") + os.pathsep + os.environ["PATH"],
    }

    def step(name, command, cap):
        available = min(cap, int(deadline - time.monotonic()))
        if available <= 0:
            raise TimeoutError("pilot global budget exhausted")
        with (out / f"{name}.log").open("w") as log:
            try:
                result = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    timeout=available,
                )
                report["steps"][name] = {"exit_code": result.returncode}
            except subprocess.TimeoutExpired:
                report["steps"][name] = {
                    "status": "timeout",
                    "limit_seconds": available,
                }
                raise
        if result.returncode:
            raise RuntimeError(f"{name} exited {result.returncode}; evidence retained")

    try:
        step(
            "bootstrap-check",
            ["bash", str(payload / "scripts/lilith/runpod/bootstrap.sh"), "--check"],
            600,
        )
        step(
            "train",
            [
                python,
                str(payload / "scripts/lilith/history/train.py"),
                "--data",
                str(args.data),
                "--parent",
                str(args.root / "parent"),
                "--approval",
                str(args.approval),
                "--payload-root",
                str(payload),
                "--main-sha",
                args.main_sha,
                "--out",
                str(out / "training"),
            ],
            5500,
        )
        step(
            "cold-reload",
            [
                python,
                str(payload / "scripts/lilith/runpod/reference_microrun.py"),
                "--mode",
                "cold-reload",
                "--parent",
                str(args.root / "parent"),
                "--adapter-dir",
                str(out / "training/candidate-adapter"),
                "--probe",
                str(out / "training/cold-reload.probe.json"),
                "--output",
                str(out / "cold-reload.json"),
                "--device",
                "cuda:0",
            ],
            900,
        )
        cold = json.loads((out / "cold-reload.json").read_text())
        if cold.get("status") != "passed" or cold.get("optimizer_steps") != 0:
            raise RuntimeError("fresh-process reload control did not pass")
        report["status"] = "passed"
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        seal(out, report)
    print(json.dumps({"run_id": RUN_ID, "status": report["status"], "out": str(out)}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
