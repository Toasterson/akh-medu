#!/usr/bin/env python3
"""Render the isolated SeaORM SQLite core. Does not awaken or start services."""
import argparse
import hashlib
import json
from pathlib import Path
import plistlib
import shutil


def configure(root, binaries):
    source = Path(__file__).resolve().parent
    for name in ("core-config/akh-medu/workspaces", "core-data", "core-state",
                 "core-runtime", "core-cache", "state/core", "awakening"):
        (root / name).mkdir(parents=True, exist_ok=True, mode=0o700)
    url = "sqlite://" + str(root / "state/core/akh.sqlite3") + "?mode=rwc"
    config = root / "core-config/akh-medu/config.toml"
    config.write_text('[database]\nurl = ' + json.dumps(url) + '''
migrate_on_start = true
pool_size = 1
wal_flush_interval_secs = 1
snapshot_interval_secs = 30

[daemon]
auto_start = []

[exec]
kind = "none"
''')
    config.chmod(0o600)
    for role in ("janus", "minerva", "vesta"):
        prompt = (source / "roles/common.md").read_text() + "\n" + (source / f"roles/{role}.md").read_text()
        path = root / f"core-config/akh-medu/workspaces/{role}.toml"
        path.write_text(f'name = "{role}"\nsystem_prompt = ' + json.dumps(prompt) + '\n[exec]\nkind = "none"\n')
        path.chmod(0o600)
    core = json.loads((root / "config/core.json").read_text())
    core["sqlite"] = True
    (root / "config/core.json").write_text(json.dumps(core, indent=2) + "\n")
    original = binaries / "akhomed"
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    executable = root / "bin" / ("akhomed-" + digest[:12])
    if not executable.exists():
        shutil.copy2(original, executable)
    env = {"AKH_SYSTEM_PATHS": "0", "AKH_AUTO_START": "",
           "AKH_SERVER_BIND": "127.0.0.1", "AKH_SERVER_PORT": "18430",
           "AKH_CORE_CONFIG": str(root / "config/core.json"), "DATABASE_URL": url,
           "RUST_LOG": "info"}
    for kind in ("CONFIG", "DATA", "STATE", "CACHE"):
        env["XDG_" + kind + "_HOME"] = str(root / ("core-" + kind.lower()))
    env["XDG_RUNTIME_DIR"] = str(root / "core-runtime")
    plist = {"Label": "dev.roman-akhs.core", "UserName": "toasty",
             "ProgramArguments": [str(executable)], "RunAtLoad": True,
             "KeepAlive": True, "ThrottleInterval": 30, "Umask": 63,
             "WorkingDirectory": str(root), "EnvironmentVariables": env,
             "StandardOutPath": str(root / "logs/core.out.log"),
             "StandardErrorPath": str(root / "logs/core.err.log")}
    with (root / "launchd/dev.roman-akhs.core.plist").open("wb") as file:
        plistlib.dump(plist, file)
    (root / "evidence/core-binary.json").write_text(json.dumps({"sha256": digest, "path": str(executable)}, indent=2) + "\n")
    shutil.copy2(source / "awakening.json", root / "awakening.json")
    print("Configured isolated SQLite core:", root)
    print("Not started. Awakening requires the offline roman_awaken helper.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--binaries", type=Path, required=True)
    args = parser.parse_args()
    configure(args.root.resolve(), args.binaries.resolve())
