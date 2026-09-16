#!/usr/bin/env python3
"""Prepare an isolated native deployment without starting services or reading keys.

Requires an explicit storage selection. SQLite applies to service records; configure_core.py installs the separate
SeaORM SQLite backing store used by the cognitive core. Rendering is safe to repeat:
existing credentials and mutable state are never replaced.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import plistlib
import secrets
import shutil

ROLES = ("janus", "minerva", "vesta")
SERVICES = {"sessions": ("akhsessd", 18431), "runtime": ("akhrtd", 18432),
            "sandbox": ("akhsbxd", 18433), "executor": ("akhexecd", 18434)}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    path.chmod(0o600)


def prepare(root, binaries, user, storage):
    source = Path(__file__).resolve().parent
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    for name in ("config", "secrets", "state", "logs", "workspaces", "bin", "launchd", "roles", "evidence"):
        (root / name).mkdir(exist_ok=True, mode=0o700)
    for name in (*SERVICES, "core", "operator"):
        path = root / "secrets" / (name + ".key")
        if not path.exists():
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as file:
                file.write(secrets.token_urlsafe(48) + "\n")
    for path in source.joinpath("roles").glob("*.md"):
        shutil.copy2(path, root / "roles" / path.name)
    manifest = {}
    for binary in [v[0] for v in SERVICES.values()] + ["akh-trigger"]:
        original = binaries / binary
        target = root / "bin" / binary
        shutil.copy2(original, target)
        manifest[binary] = hashlib.sha256(target.read_bytes()).hexdigest()
    def endpoint(name):
        return {"url": f"http://127.0.0.1:{SERVICES[name][1]}",
                "token_file": str(root / "secrets" / (name + ".key"))}
    endpoints = {name: endpoint(name) for name in SERVICES}
    endpoints["sandboxes"] = endpoint("sandbox")
    endpoints["core"] = {"url": "http://127.0.0.1:18430",
                         "token_file": str(root / "secrets/core.key")}
    profiles = {}
    for role in ROLES:
        profiles[role] = {"revision": "roman-mini-v1", "site": "armini",
            "provisioner": {"kind": "existing", "executor": "executor",
                            "root": str(root / "workspaces" / role)},
            "retain_on_release": True}
        (root / "workspaces" / role).mkdir(exist_ok=True, mode=0o700)
    common = source.joinpath("roles/common.md").read_text()
    akhs = {}
    automations = {}
    for role in ROLES:
        akhs[role] = {"core_service": "core", "runtime": {
            "runtime": "runtime", "profile": "ollama", "revision": "roman-mini-v1",
            "kind": "internal", "model": "mistral-large-3:675b", "execution_mode": "brokered"},
            "instructions": common + "\n" + source.joinpath(f"roles/{role}.md").read_text(),
            "workspace_profiles": [role], "tools": [], "approval_policy": "allow_granted"}
        automations[role] = {"id": role, "enabled": False, "principal": "operator",
            "akh": role, "workspace_profile": role,
            "instructions": "Review current evidence and commitments. Act within your role and granted tools. Persist findings and the next check.",
            "sources": ["roman-clock", "roman-handoff", "roman-alerts"],
            "target": {"kind": "conversation", "key": "roman-" + role},
            "max_active": 1, "max_age_ms": 900000}
    runtimes = {}
    for provider, url in (("ollama", "https://ollama.com/v1/chat/completions"),
                          ("openrouter", "https://openrouter.ai/api/v1/chat/completions")):
        runtimes[provider] = {"revision": "roman-mini-v1", "kind": "internal",
            "working_dir": str(root / "workspaces"), "binary": "/usr/bin/false",
            "execution_mode": "brokered", "timeout_secs": 300,
            "isolation": {"directory": str(root / "state" / ("provider-" + provider))},
            "model_endpoint": {"url": url,
                "token_env": "ROMAN_" + provider.upper() + "_KEY",
                "max_iterations": 12, "max_output_tokens": 2048}}
    for name, (binary, port) in SERVICES.items():
        config = {"service_id": name, "listen": f"127.0.0.1:{port}",
            "public_url": f"http://127.0.0.1:{port}",
            "data_dir": str(root / "state" / name),
            "principals": [{"id": name + "-caller", "token_file": str(root / "secrets" / (name + ".key")),
                            "roles": ["session", "runtime", "sandbox", "client", "trigger"], "akhs": list(ROLES)},
                           {"id": "operator", "token_file": str(root / "secrets/operator.key"),
                            "roles": ["admin"], "akhs": list(ROLES)}],
            "endpoints": endpoints, "workspaces": profiles}
        if storage == "sqlite":
            config["store"] = {"kind": "sqlite"}
        if name == "sessions":
            config.update(akhs=akhs, automations=automations)
            config["principals"][0]["id"] = "runtime"
            config["principals"][0]["roles"] = ["runtime"]
        if name == "runtime":
            config["runtimes"] = runtimes
        if name == "executor":
            config["workspace_root"] = str(root / "workspaces")
        write_json(root / "config" / (name + ".json"), config)
        plist = {"Label": "dev.roman-akhs." + name, "UserName": user,
            "ProgramArguments": [str(root / "bin" / binary), "--config", str(root / "config" / (name + ".json"))],
            "RunAtLoad": True, "KeepAlive": True, "ThrottleInterval": 30,
            "WorkingDirectory": str(root), "Umask": 63,
            "StandardOutPath": str(root / "logs" / (name + ".out.log")),
            "StandardErrorPath": str(root / "logs" / (name + ".err.log")),
            "EnvironmentVariables": {"RUST_LOG": "info"}}
        with (root / "launchd" / ("dev.roman-akhs." + name + ".plist")).open("wb") as file:
            plistlib.dump(plist, file)
    write_json(root / "config/core.json", {"listen": "127.0.0.1:18430",
        "data_dir": str(root / "state/core"), "principals": [{"id": "sessions",
        "token_file": str(root / "secrets/core.key"), "akhs": list(ROLES)}]})
    write_json(root / "config/operator.json", {"url": "http://127.0.0.1:18431",
        "token_file": str(root / "secrets/operator.key")})
    write_json(root / "evidence/binaries.json", manifest)
    write_json(root / "evidence/prepared.json", {"status": "prepared_not_activated",
        "storage": storage, "provider_keys_read": False, "roles": list(ROLES),
        "pending": ["configure and verify SeaORM SQLite core", "akh core and awakening", "scoped operational tools",
                    "provider credentials", "live tool/continuation/recovery verification",
                    "enable automations after verification"]})
    print("Prepared isolated candidate deployment:", root)
    print("No launchd service started; no cloud inference performed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--binaries", type=Path, required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--storage", choices=["redb", "sqlite"], required=True)
    args = parser.parse_args()
    prepare(args.root.resolve(), args.binaries.resolve(), args.user, args.storage)
