#!/usr/bin/env python3
"""Record a local Candle probe's identity and GPU/process telemetry. No services."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["hip", "vulkan"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--gpu", type=Path,
                        default=Path("/sys/bus/pci/devices/0000:03:00.0"))
    # Extra probe flags stay explicit in identity.json rather than hidden in a
    # shell wrapper, so evidence records exactly what was measured.
    parser.add_argument("--probe-arg", action="append", default=[],
                        metavar="ARG",
                        help="append one argument to the probe command; use "
                             "--probe-arg=--flag for values starting with a dash")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    # Exclusive creation prevents silently replacing previous evidence.
    args.output.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve(strict=True)
    env = dict(os.environ, AKH_HIP_Q8_GEMV="1", AKH_HIP_QWEN35_FUSED_DELTA="0",
               AKH_VULKAN_BATCH_ENCODER="1", AKH_VULKAN_BATCH_PASS="1",
               AKH_VULKAN_BATCH_SUBMIT="1")
    command = [str(binary), "--device", args.device, "--model", str(args.model),
               "--tokenizer", str(args.tokenizer), "--gpu-sysfs", str(args.gpu),
               "--tokens", "64", "--continuation", "--idle-samples", "60",
               "--settle-seconds", "15", "--baseline-settle-seconds", "15",
               *args.probe_arg]
    identity = {"command": command, "source_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
        "binary_sha256": sha256(binary), "tokenizer_sha256": sha256(args.tokenizer),
        "model_path": str(args.model), "model_size": args.model.stat().st_size,
        "model_identity_note": "Path and size only; model hash not recomputed by this recorder",
        "settings": {key: env[key] for key in (
            "AKH_HIP_Q8_GEMV", "AKH_HIP_QWEN35_FUSED_DELTA",
            "AKH_VULKAN_BATCH_ENCODER", "AKH_VULKAN_BATCH_PASS", "AKH_VULKAN_BATCH_SUBMIT")}}
    (args.output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    monitor = next((args.gpu / "hwmon").iterdir())
    started = time.monotonic()
    timed_out = False
    with (args.output / "stdout.jsonl").open("x") as stdout, \
            (args.output / "stderr.log").open("x") as stderr, \
            (args.output / "telemetry.jsonl").open("x") as telemetry:
        child = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
        print(json.dumps({"pid": child.pid, "output": str(args.output)}), flush=True)
        try:
            while child.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed > args.timeout:
                    timed_out = True
                    break
                sample = {"seconds": elapsed}
                for name, path in {
                    "gpu_busy_percent": args.gpu / "gpu_busy_percent",
                    "vram_bytes": args.gpu / "mem_info_vram_used",
                    "power_microwatts": monitor / "power1_average",
                    "fan_rpm": monitor / "fan1_input",
                }.items():
                    sample[name] = int(path.read_text().strip())
                try:
                    for line in Path(f"/proc/{child.pid}/status").read_text().splitlines():
                        if line.startswith("VmRSS:"):
                            sample["rss_kib"] = int(line.split()[1])
                except FileNotFoundError:
                    pass  # Child exited between poll and status read.
                telemetry.write(json.dumps(sample) + "\n")
                telemetry.flush()
                time.sleep(1)
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        result = {"returncode": child.returncode, "timed_out": timed_out,
                  "seconds": time.monotonic() - started}
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if timed_out or child.returncode:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
