#!/usr/bin/env python3
"""Replay the exact model-run3 reporter turn in an isolated candidate stack."""
import argparse, copy, hashlib, json, os, secrets, shutil, socket, subprocess, time, urllib.request, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "training/lilith/coordination-20260922/model-run3"
BASE = Path.home() / ".local/share/akh-medu/lilith-dev-vulkan-20260921"
NAMES = ("akhexecd", "akhsbxd", "akhrtd", "akhsessd")
SYSTEM = "Report only the supplied bounded evidence capsule. Preserve the objective and reviewer opinion, distinguish independent verification from reviewer judgment, and do not call tools."


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def free_ports(count):
    sockets = []
    try:
        for _ in range(count):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-akhrtd", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)

    assignment = json.loads((SOURCE / "assignments.json").read_text())[0]
    sessions = json.loads((SOURCE / "sessions.json").read_text())
    original = sessions["lilith_report"]
    request = assignment["return_request"]["text"]
    assert original["instructions"] == SYSTEM and original["tools"] == []
    assert original["akh"] == "lilith-report-only"
    assert assignment["report_prompt_tokens"] == 735
    assert digest(request) == digest(original["turns"][0]["input"]["text"])

    binaries = {name: ROOT / "target/release" / name for name in NAMES}
    binaries["akhrtd"] = args.candidate_akhrtd.resolve()
    for path in binaries.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    model_run3_protocol = json.loads((SOURCE / "protocol.json").read_text())
    protocol = {
        "schema": "lilith/candidate-full-report-session@1",
        "driver_sha256": sha(__file__),
        "source_protocol_sha256": sha(SOURCE / "protocol.json"),
        "source_assignments_sha256": sha(SOURCE / "assignments.json"),
        "source_sessions_sha256": sha(SOURCE / "sessions.json"),
        "request_sha256": digest(request),
        "system_sha256": digest(SYSTEM),
        "prompt_tokens_pinned": 735,
        "max_output_tokens": 768,
        "turn_deadline_seconds": 120,
        "binaries": {name: {"path": str(path), "sha256": sha(path)} for name, path in binaries.items()},
        "model_pins": model_run3_protocol["pins"]["runtimes"]["lilith"],
        "environment": {
            "AKH_GPU_MATMUL": "0", "AKH_GPU_QUANT": "0",
            "AKH_VULKAN_PIPELINE_CACHE_ENTRIES": "16384",
            "AKH_VULKAN_REUSABLE_KV": "1", "AKH_VULKAN_GROUPED_DECODE": "1",
            "AKH_VULKAN_MATMUL_STRIDES": "1",
            "VK_DRIVER_FILES": "/usr/share/vulkan/icd.d/radeon_icd.x86_64.json",
        },
        "scope": "Exact model-run3 tool-free reporter system/user replay in a fresh private Vulkan-only stack. Candidate binary only; no shared deployment or Ptah/HIP claim.",
    }
    write(args.out / "protocol.json", protocol)
    (args.out / "request.txt").write_text(request)
    (args.out / "system.txt").write_text(SYSTEM + "\n")
    if args.check:
        write(args.out / "result.json", {"status": "check_passed", "gpu_used": False})
        return

    private = Path.home() / ".local/share/akh-medu" / ("lilith-candidate-report-" + uuid.uuid4().hex)
    private.mkdir(mode=0o700)
    for directory in ("credentials", "workspaces"):
        (private / directory).mkdir(mode=0o700)
    ports = free_ports(4)
    configs = {}
    for index, name in enumerate(NAMES):
        text = (BASE / (name + ".json")).read_text().replace(str(BASE), str(private))
        for old, new in zip((18420, 18421, 18422, 18423), ports):
            text = text.replace("127.0.0.1:" + str(old), "127.0.0.1:" + str(new))
        configs[name] = json.loads(text)
    for name in ("operator",) + NAMES:
        token = secrets.token_urlsafe(32)
        for suffix, value in (("token", token), ("sha256", digest(token))):
            path = private / "credentials" / (name + "." + suffix)
            path.write_text(value + "\n")
            path.chmod(0o600)
    for config in configs.values():
        for principal in config["principals"]:
            principal["akhs"] = ["lilith-report-only"]

    # model-run3 derived its reporter from the base `lilith` profile.
    profile = copy.deepcopy(configs["akhsessd"]["akhs"]["lilith"])
    profile.update(instructions=SYSTEM, tools=[])
    profile["runtime"]["profile"] = "reporter"
    configs["akhsessd"]["akhs"] = {"lilith-report-only": profile}
    runtime = configs["akhrtd"]["runtimes"]["lilith"]
    configs["akhrtd"]["runtimes"] = {"reporter": runtime}
    runtime["timeout_secs"] = 120
    native = runtime["native_candle"]
    native.update(max_context_tokens=2048, max_output_tokens=768, max_iterations=1)
    for name in ("model", "tokenizer"):
        assert sha(native[name + "_path"]) == protocol["model_pins"][name + "_sha256"]
    for name, config in configs.items():
        write(private / (name + ".json"), config)

    operator = (private / "credentials/operator.token").read_text().strip()
    def api(name, path, body=None):
        request_obj = urllib.request.Request(
            "http://" + configs[name]["listen"] + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + operator, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request_obj, timeout=15) as response:
            return json.load(response)

    processes, logs = [], []
    result = {"status": "running", "truth_review": "pending", "services_stopped": False}
    write(args.out / "result.json", result)
    try:
        env = dict(os.environ, **protocol["environment"])
        for name in NAMES:
            log = (args.out / (name + ".log")).open("w")
            logs.append(log)
            process = subprocess.Popen(
                [str(binaries[name]), "--config", str(private / (name + ".json"))],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env,
            )
            processes.append(process)
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError(name + " exited during readiness")
                try:
                    if api(name, "/healthz")["service_id"] == configs[name]["service_id"]:
                        break
                except (OSError, ValueError):
                    time.sleep(0.2)
            else:
                raise TimeoutError(name + " readiness")

        session_id = api("akhsessd", "/v1/sessions", {
            "request_id": uuid.uuid4().hex, "akh": "lilith-report-only", "workspace_profile": "local-dev"
        })["id"]
        path = "/v1/sessions/" + session_id
        started = time.monotonic()
        turn = api("akhsessd", path + "/inputs", {"request_id": uuid.uuid4().hex, "text": request})
        while time.monotonic() - started < 120:
            session = api("akhsessd", path)
            current = next(item for item in session["turns"] if item["id"] == turn["id"])
            if current["status"] in ("completed", "failed", "cancelled", "unknown"):
                break
            time.sleep(0.25)
        else:
            raise TimeoutError("report exceeded 120-second gate")
        elapsed = time.monotonic() - started
        write(args.out / "session.json", session)
        tools = api("akhsessd", path + "/tools")["records"]
        write(args.out / "tools.json", tools)
        output = current.get("output") or ""
        truth = {
            "status_completed": current["status"] == "completed",
            "nonempty_output": bool(output.strip()), "no_tools": tools == [],
            "within_120_seconds": elapsed < 120,
            "preserves_objective": "range_within_u64" in output,
            "preserves_reviewer_opinion": "no defect found" in output.lower(),
            "reports_passed_verification": "pass" in output.lower() and "exit" in output.lower(),
            "reports_guest_boundary": "127.0.0.1" in output and "akh-exec" in output,
            "avoids_repository_modification_claim": "modified the original repository" not in output.lower(),
        }
        truth["passed"] = all(truth.values())
        write(args.out / "truth-review.json", truth)
        if not truth["passed"]:
            raise RuntimeError("report truth review failed")
        result.update(status="completed", elapsed_seconds=elapsed, report=output,
                      session_id=session_id, turn_id=turn["id"], truth_review="passed")
    except BaseException as error:
        result.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        for log in logs:
            log.close()
        result["services_stopped"] = all(process.poll() is not None for process in processes)
        shutil.rmtree(private)
        result["private_root_removed"] = not private.exists()
        write(args.out / "result.json", result)


if __name__ == "__main__":
    main()
