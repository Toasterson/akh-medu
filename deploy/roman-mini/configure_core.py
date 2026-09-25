#!/usr/bin/env python3
"""Render the full-akh Roman core (Phase 53f, ADR 113). Does not start services.

The core is one fully featured akhomed built from origin/main with the single
`server` profile (`cargo build --release --locked --features server --bin
akhomed`). It hosts janus, minerva and vesta with their daemon loops, in-process
turns (GLM-5.3 on Ollama Cloud), full tool registries, per-akh MCP servers,
reactors, ACP and the additive `/v1` core API.

Refuses unless `--repo` is a clean checkout whose HEAD is an ancestor of
origin/main (.claude/rules/deploy-from-main.md); the binary is installed as
`bin/akhomed-main-<sha>` and its provenance recorded in `config/core-release.json`.

Renders (existing unrelated keys and secrets are preserved; rerunning is safe):
- core-config/akh-medu/config.toml         host config (deployed, auto_start, [exec], [acp])
- core-config/akh-medu/workspaces/<r>.toml  full-akh workspace per Roman
- config/core.json                          sessions / events / operator principals
- config/reactor-seeds/<r>.json             reactor_manage payloads (apply_reactor_seeds.py)
- config/events.akh-target.json             staged akheventd routes (retarget_events.py)
- config/t3-acp.json                        T3 transport -> akhomed ACP (t3_entrypoint.py)
- launchd/dev.roman-akhs.core.plist         execs through daemon_credentials.py core
Never reads a provider key; token files are only concatenated into the ACP
shared-secret file, never printed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import plistlib
import secrets
import shutil
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11 (macOS /usr/bin/python3 is 3.9)
    tomllib = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
import full_akh as fa  # noqa: E402


def _save_json(path, value):
    fa.write_atomic(path, json.dumps(value, indent=2) + "\n")


def _save_text(path, text):
    fa.write_atomic(path, text)


def _ensure_key(path):
    fa.private_dir(path.parent)
    if not path.exists():
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as file:
            file.write(secrets.token_urlsafe(48) + "\n")
    return path


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_binary(built, target):
    """Copy `built` to `target` once; refuse to replace a different binary."""
    digest = _sha256(built)
    if target.exists():
        if _sha256(target) != digest:
            raise fa.RenderError(f"{target} exists with different content; the same main SHA must "
                                 "produce the same install (remove it only if it was never started)")
    else:
        shutil.copy2(built, target)
        target.chmod(0o755)
    return digest


def render(root, provenance, binary, binary_sha256, acp_binary=None, source=None,
           acp_port=fa.ACP_TCP_PORT):
    """Render every full-akh file under `root`. Returns warnings."""
    source = source or Path(__file__).resolve().parent
    warnings = []
    for name in ("core-config/akh-medu/workspaces", "core-data", "core-state", "core-runtime",
                 "core-cache", "state/core", "config/reactor-seeds", "launchd", "evidence", "logs"):
        (root / name).mkdir(parents=True, exist_ok=True, mode=0o700)
    fa.private_dir(root / "secrets")
    config = root / "config"
    session = json.loads((config / "sessions.json").read_text())
    events = json.loads((config / "events.json").read_text())
    core = json.loads((config / "core.json").read_text())
    url = "sqlite://" + str(root / "state/core/akh.sqlite3") + "?mode=rwc"

    host = fa.render_host_config(url, fa.acp_principals(session, root), acp_port)
    _save_text(root / "core-config/akh-medu/config.toml", fa.dump_toml(
        host, fa.HOST_COMMENTS,
        header="Rendered by deploy/roman-mini/configure_core.py (Phase 53f, ADR 113).\n"
               f"akhomed-main-{provenance.short} = {provenance.commit}"))

    for role in fa.ROLES:
        path = root / f"core-config/akh-medu/workspaces/{role}.toml"
        if tomllib is None:
            raise SystemExit(
                "configure_core.py needs Python >= 3.11 (tomllib) to read the existing "
                f"workspace TOMLs; this is {sys.version.split()[0]}. Run it with "
                "/opt/homebrew/bin/python3 (cutover.sh selects one automatically)."
            )
        existing = tomllib.loads(path.read_text()) if path.exists() else None
        servers, server_warnings = fa.mcp_servers(session, role, root)
        warnings.extend(server_warnings)
        doc = fa.render_workspace(existing, role, fa.system_prompt(source / "roles", role), servers,
                                  fa.board_identity(events, role))
        _save_text(path, fa.dump_toml(doc, fa.WORKSPACE_COMMENTS,
                                      header=f"{role}: full akh (ADR 113). Rendered by configure_core.py."))
        _save_json(config / f"reactor-seeds/{role}.json", fa.reactor_seeds(events, session, role))

    _ensure_key(root / "secrets/core-events.key")
    _save_json(config / "core.json", fa.render_core_config(core, root))
    _save_json(config / "events.akh-target.json", fa.retarget_events(events, root))

    # ACP shared secret: one accepted token per line (akhomed reads
    # AKHOMED_SHARED_SECRET_FILE). Until per-principal ACP scoping lands
    # (phase53/fixups) any of these keys can open a session on any Roman.
    keys = [Path(p["token_file"]) for p in fa.acp_principals(session, root)]
    missing = [str(k) for k in keys if not k.exists()]
    if missing:
        raise fa.RenderError("ACP client key files are missing: " + ", ".join(missing))
    _save_text(root / "secrets/acp-clients.keys", "".join(k.read_text().strip() + "\n" for k in keys))

    acp_path = acp_binary or (root / "bin/akh-acp")
    if acp_binary is None:
        warnings.append("no akh-acp built from main was installed; T3 keeps bin/akh-acp, which must "
                        "support --akhomed before the transport switch works")
    _save_json(config / "t3-acp.json", {
        "mode": "akhomed", "endpoint": f"tcp://127.0.0.1:{acp_port}",
        "health_url": f"http://{fa.CORE_LISTEN}/healthz", "binary": str(acp_path)})

    release = {"binary": str(binary), "commit": provenance.commit, "short": provenance.short,
               "origin_main": provenance.origin_main, "sha256": binary_sha256,
               "build_command": fa.BUILD_COMMAND,
               "acp_binary": str(acp_binary) if acp_binary else None}
    _save_json(config / "core-release.json", release)
    _save_json(root / "evidence/core-binary.json", release)

    env = {"AKH_SYSTEM_PATHS": "0", "AKH_DEPLOYMENT": "deployed",
           "AKH_SERVER_BIND": "127.0.0.1", "AKH_SERVER_PORT": fa.CORE_LISTEN.rsplit(":", 1)[1],
           "AKH_CORE_CONFIG": str(config / "core.json"), "DATABASE_URL": url,
           "AKHOMED_SHARED_SECRET_FILE": str(root / "secrets/acp-clients.keys"),
           "HOME": "/Users/toasty", "PATH": "/opt/homebrew/bin:/usr/bin:/bin:/Users/toasty/.local/bin",
           "RUST_LOG": "info"}
    for kind in ("CONFIG", "DATA", "STATE", "CACHE"):
        env["XDG_" + kind + "_HOME"] = str(root / ("core-" + kind.lower()))
    env["XDG_RUNTIME_DIR"] = str(root / "core-runtime")
    plist = {"Label": "dev.roman-akhs.core", "UserName": "toasty",
             "ProgramArguments": ["/usr/bin/python3", str(root / "bin/daemon_credentials.py"), "core", str(root)],
             "RunAtLoad": True, "KeepAlive": True, "ThrottleInterval": 30, "ExitTimeOut": 60,
             "Umask": 63, "WorkingDirectory": str(root), "EnvironmentVariables": env,
             "StandardOutPath": str(root / "logs/core.out.log"),
             "StandardErrorPath": str(root / "logs/core.err.log")}
    fa.write_atomic(root / "launchd/dev.roman-akhs.core.plist", plistlib.dumps(plist), mode=0o644)
    for name in ("daemon_credentials.py", "t3_entrypoint.py"):
        if (source / name).exists() and (root / "bin").is_dir():
            shutil.copy2(source / name, root / "bin" / name)
    return warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--binaries", type=Path, required=True,
                        help="target/release of the clean origin/main checkout")
    parser.add_argument("--repo", type=Path, required=True, help="the clean origin/main checkout")
    parser.add_argument("--no-fetch", action="store_true", help="skip `git fetch origin`")
    parser.add_argument("--acp-port", type=int, default=fa.ACP_TCP_PORT)
    args = parser.parse_args()
    # Everything rendered here is private unless written with an explicit mode.
    os.umask(0o077)
    root = args.root.resolve()
    provenance = fa.main_provenance(args.repo.resolve(), fetch=not args.no_fetch)
    built = args.binaries.resolve() / "akhomed"
    if not built.is_file():
        raise fa.RenderError(f"{built} not found; run `{fa.BUILD_COMMAND}` in {args.repo}")
    if built.stat().st_mtime < provenance.committed_at:
        raise fa.RenderError(f"{built} is older than commit {provenance.short}; rebuild it from this checkout")
    binary = root / "bin" / provenance.binary_name
    digest = install_binary(built, binary)
    acp_binary = None
    if (args.binaries.resolve() / "akh-acp").is_file():
        acp_binary = root / "bin" / provenance.acp_binary_name
        install_binary(args.binaries.resolve() / "akh-acp", acp_binary)
    warnings = render(root, provenance, binary, digest, acp_binary, acp_port=args.acp_port)
    print(f"Configured full akh core {provenance.binary_name} ({provenance.commit}); "
          f"merge-base --is-ancestor {provenance.short} origin/main ({provenance.origin_main[:12]}): ok")
    for warning in warnings:
        print("warning:", warning)
    print("Not started. Follow deploy/roman-mini/CUTOVER.md.")


if __name__ == "__main__":
    try:
        main()
    except fa.RenderError as error:
        print("configure_core: " + str(error), file=sys.stderr)
        sys.exit(1)
