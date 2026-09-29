#!/usr/bin/env python3
"""Render the full-akh Roman core (Phase 53f, ADR 113). Does not start services.

The core is one fully featured akhomed built from origin/main with the single
`server` profile (`cargo build --release --locked --features server --bin
akhomed`). It hosts janus, minerva and vesta with their daemon loops, in-process
turns (GLM-5.3 on Ollama Cloud), full tool registries, per-akh MCP servers,
reactors, ACP and the additive `/v1` core API.

Refuses unless `--repo` is a clean checkout whose HEAD is an ancestor of
origin/main (.claude/rules/deploy-from-main.md). The core execs a stable path,
`bin/akhomed`, signed with the identity from `config/codesign.json` (absent:
ad-hoc, with a warning) so its macOS Local Network TCC approval survives
future deploys; a versioned copy `bin/akhomed-main-<sha>` is kept alongside it
for rollback and audit. Provenance (commit, both the unsigned build's and the
installed signed binary's sha256, the codesign identity/identifier/designated
requirement) is recorded in `config/core-release.json` and
`evidence/core-binary.json`.

Renders (existing unrelated keys and secrets are preserved; rerunning is safe):
- core-config/akh-medu/config.toml         host config (deployed, auto_start, [exec], [acp])
- core-config/akh-medu/workspaces/<r>.toml  full-akh workspace per Roman
- config/core.json                          sessions / events / operator principals
- config/reactor-seeds/<r>.json             reactor_manage payloads (apply_reactor_seeds.py)
- config/events.akh-target.json             staged akheventd routes + roman-alerts monitor + roman-anima
                                            source (retarget_events.py); shadow unless --live (PR 6)
- core-data/…/workspaces/<r>/kg/skills/laya-intervention   the PR 6 skillpack (activate via ACP)
- config/t3-acp.json                        T3 transport -> akhomed ACP (t3_entrypoint.py); lists the
                                            hosted `akhs` and, with Lilith, her `lilith_serve` checkout
- secrets/t3-lilith.key + bin/roman-lilith  Lilith's T3 client key (created once) and provider binary
- launchd/dev.roman-akhs.core.plist         execs through daemon_credentials.py core
- skills/{astra-escalation,mergiraf}/       Janus's worker skillpacks (53g PR 7)
- secrets/sessions-janus-delegation.key     akhsessd client token for Janus's delegation (local
                                            akhsessd; not created when config/janus-delegation.json
                                            switches Janus onto the cluster akhsessd, PR 7 cluster
                                            variant)
- skills/vesta-remediation/                  Vesta's sandbox procedures (53g PR 8)
- workspaces/lilith.toml + core-data/…/workspaces/lilith/kg/skills/chief-of-staff
                                            Lilith C2 (ADR 116), only once config/lilith.json exists;
                                            auto-started only once her secrets and ancestry clone exist
- workspaces/dojo-teacher.toml + secrets/dojo-teacher.key
                                            Lilith C4 sealed dojo teacher (tool_surface = session-only,
                                            ADR 116), only once config/dojo-teacher.json exists; its ACP
                                            principal `dojo-teacher` reaches that akh alone
Never reads a provider key; token files are only concatenated into the ACP
shared-secret file, never printed. (Lilith's endpoint base URL is read from
secrets/lilith-base-url, because `[openai_endpoint] base_url` has no `env:` form.)

`--plan` renders into memory and prints the TOML diff per file (host config and
every workspace) without writing, installing or signing anything. Run it before
every full render: re-renders have dropped hand-applied holds before.
"""
import argparse
from dataclasses import dataclass, field
import difflib
import hashlib
import json
import os
from pathlib import Path
import plistlib
import secrets
import shutil
import subprocess
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


# ── Stable-path code signing (operator decision 2026-09-27) ─────────────────
#
# macOS Local Network privacy (TCC) keys its per-binary approval off the
# installed file's path and code signature together. The old flow installed
# a fresh `bin/akhomed-main-<sha>` on every deploy, so every deploy was a new
# TCC identity and the operator had to re-approve it by hand. Signing a
# *stable* path (`bin/akhomed`) with a *stable* identity keeps the same TCC
# identity across deploys; the versioned copy stays for rollback and audit,
# per .claude/rules/deploy-from-main.md ("SHA in its name, tag or evidence
# record").
DEFAULT_CODESIGN_IDENTIFIER = "dev.akh-medu.akhomed"


def load_codesign_config(root):
    """Read `config/codesign.json` -> {"identity", "identifier"}, or None.

    Absent file: the caller ad-hoc signs (`codesign --sign -`) and must warn
    that the Local Network TCC approval will not survive the next deploy.
    """
    path = root / "config/codesign.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    identity = data.get("identity")
    if not identity:
        raise fa.RenderError(f"{path} is missing required key 'identity'")
    return {"identity": identity, "identifier": data.get("identifier", DEFAULT_CODESIGN_IDENTIFIER)}


def _codesign(args):
    result = subprocess.run(["codesign", *args], capture_output=True, text=True)
    if result.returncode:
        raise fa.RenderError("codesign " + " ".join(args) + " failed: " + result.stderr.strip())
    return result


def codesign_binary(path, codesign_config):
    """Sign `path` in place, replacing any existing signature.

    With `codesign_config` (from `config/codesign.json`), signs with the
    named identity and identifier, no hardened runtime (`--options runtime`
    is deliberately omitted: it is not needed without notarization, and could
    break library loading) and no timestamp server (`--timestamp=none`, so
    signing works offline). With `codesign_config=None`, signs ad-hoc
    (`--sign -`); the resulting signature carries no persistent identity, so
    macOS treats it as new on every deploy.

    Returns {"identity", "identifier", "designated_requirement"}; the latter
    two are None for an ad-hoc signature.
    """
    if codesign_config is None:
        _codesign(["--force", "--sign", "-", str(path)])
        return {"identity": "-", "identifier": None, "designated_requirement": None}
    identity, identifier = codesign_config["identity"], codesign_config["identifier"]
    _codesign(["--force", "--sign", identity, "--identifier", identifier, "--timestamp=none", str(path)])
    requirement = _codesign(["-dr", "-", str(path)]).stdout.strip().removeprefix("designated => ")
    return {"identity": identity, "identifier": identifier, "designated_requirement": requirement}


def install_core_binary(built, root, provenance, codesign_config, warnings):
    """Install the core at the stable, signed `bin/akhomed` (the only path the
    daemon execs) and the versioned `bin/akhomed-main-<sha>` (unsigned, kept
    for rollback and audit; refuses to replace a different binary under the
    same SHA name, like `install_binary`).

    The stable path is written by copy-to-temp + codesign + atomic rename, so
    a crash or a concurrent reader never sees a partially written or
    unsigned file at `bin/akhomed`, and the running process's inode is never
    modified in place.

    Returns {"binary", "versioned_binary", "build_sha256", "sha256",
    "codesign"} for `render()`'s `config/core-release.json` /
    `evidence/core-binary.json`.
    """
    versioned = root / "bin" / provenance.binary_name
    build_sha256 = install_binary(built, versioned)

    stable = root / "bin" / "akhomed"
    stable.parent.mkdir(parents=True, exist_ok=True)
    temp = stable.with_name(f".{stable.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(built, temp)
        temp.chmod(0o755)
        codesign_info = codesign_binary(temp, codesign_config)
        os.replace(temp, stable)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    if codesign_config is None:
        warnings.append(f"{stable} is ad-hoc signed (no config/codesign.json); its Local Network "
                        "TCC approval will not survive the next deploy")
    return {"binary": str(stable), "versioned_binary": str(versioned), "build_sha256": build_sha256,
            "sha256": _sha256(stable), "codesign": codesign_info}


def install_skillpack(source, root, role):
    """Copy the laya-intervention skillpack into the role's engine skills dir
    (`core-data/akh-medu/workspaces/<role>/kg/skills/`). Activation stays an
    operator step (`akh acp skill activate laya-intervention`, INVENTORY PR 6)."""
    src = source / "skills" / fa.LAYA_SKILL
    dest = root / f"core-data/akh-medu/workspaces/{role}/kg/skills/{fa.LAYA_SKILL}"
    dest.mkdir(parents=True, exist_ok=True, mode=0o700)
    for item in sorted(src.iterdir()):
        if item.is_file():
            fa.write_atomic(dest / item.name, item.read_bytes(), mode=0o600)
    return dest


def install_skill_tree(src, dest):
    """Copy a whole skillpack directory (files and subdirectories such as the
    chief-of-staff `schemas/`) into an engine skills dir: dirs 0700, files
    0600, each file replaced atomically. Files the pack no longer ships are
    left alone (the engine may keep state next to them)."""
    if not (src / "skill.json").is_file():
        raise fa.RenderError(f"skillpack {src} has no skill.json")
    for item in sorted(src.rglob("*")):
        target = dest / item.relative_to(src)
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True, mode=0o700)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fa.write_atomic(target, item.read_bytes(), mode=0o600)
    return dest


def cos_skill_source(source):
    """The chief-of-staff skillpack ships at the repository root
    (`skills/chief-of-staff`), two levels above deploy/roman-mini."""
    return source.resolve().parents[1] / "skills" / fa.COS_SKILL


@dataclass
class Rendered:
    """Everything `render()` writes that is derived from the operator inputs,
    computed without touching the deployment root (shared with `--plan`)."""
    toml: dict            # Path -> rendered TOML text (host config + workspaces)
    seeds: dict           # role -> reactor_manage payloads
    staged: dict          # config/events.akh-target.json
    core: dict            # config/core.json
    akhs: tuple           # every akh this core hosts
    specs: dict           # role -> fa.RoleSpec
    auto_start: list
    warnings: list = field(default_factory=list)
    lilith: object = None  # fa.LilithSettings when she is configured
    database: object = None  # fa.DatabaseSettings (ADR 120 layout switch)
    teacher: object = None  # fa.DojoTeacherSettings when the dojo teacher is configured
    acp_akhs: tuple = ()  # every akh an ACP principal may name (akhs + the dojo teacher)


def render_configs(root, provenance, source, acp_port=fa.ACP_TCP_PORT, live=False):
    """Compute every rendered configuration file without writing anything.

    Reads config/{sessions,events,core}.json, the existing workspace TOMLs
    (their unrelated keys are preserved) and the operator's switch files
    (janus-delegation.json, vesta-sandbox.json, lilith.json,
    per-akh-databases.json). A malformed switch raises RenderError here,
    before `render()` writes a single file.
    """
    if tomllib is None:
        raise SystemExit(
            "configure_core.py needs Python >= 3.11 (tomllib) to read the existing "
            f"workspace TOMLs; this is {sys.version.split()[0]}. Run it with "
            "/opt/homebrew/bin/python3 (cutover.sh selects one automatically)."
        )
    warnings = []
    config = root / "config"
    session = json.loads((config / "sessions.json").read_text())
    events = json.loads((config / "events.json").read_text())
    core = json.loads((config / "core.json").read_text())
    registry = config / "janus-escalation.json"
    projects = list(json.loads(registry.read_text()).get("projects", {})) if registry.exists() else []
    if not projects:
        warnings.append("config/janus-escalation.json lists no projects; Janus's escalation tools stay disabled")
    # config/janus-delegation.json (PR 7 cluster variant, operator decision
    # 2026-09-27 option a): switches Janus onto the cluster akh-platform
    # akhsessd instead of the Mac's local one.
    cluster = fa.janus_delegation_switch(root)

    # Lilith C2 (ADR 116): rendered only once config/lilith.json exists; she
    # auto-starts only once her secrets and ancestry clone exist too.
    specs = dict(fa.ROLE_SPECS)
    auto_start = list(fa.ROLES)
    lilith = fa.lilith_settings(root)
    cos_source = None
    if lilith is not None:
        cos_source = cos_skill_source(source)
        if not (cos_source / "skill.json").is_file():
            raise fa.RenderError(f"the chief-of-staff skillpack is missing at {cos_source}; render from a "
                                 "full akh-medu checkout of origin/main")
        specs[fa.LILITH] = lilith.spec()
        events = fa.with_board_endpoint(events, fa.LILITH, fa.LILITH_BOARD_TOKEN_ENV)
        missing = fa.lilith_missing_inputs(root)
        if missing:
            warnings.append("lilith is rendered but NOT auto-started until these exist: " + ", ".join(missing)
                            + " (see CUTOVER.md \"Lilith\"); rerun configure_core.py afterwards")
        else:
            auto_start.append(fa.LILITH)
    akhs = tuple(specs)
    # Lilith C4 (ADR 116): the sealed dojo teacher, only once config/dojo-teacher.json
    # exists. It is an ACP-only exam instrument: no T3 role, no core /v1 scope,
    # no events, no reactor seeds.
    teacher = fa.dojo_teacher_settings(root)
    acp_akhs = akhs + ((fa.DOJO_TEACHER,) if teacher is not None else ())
    if teacher is not None:
        auto_start.append(fa.DOJO_TEACHER)

    # ADR 120: config/per-akh-databases.json switches the core onto the
    # per-akh layout (host.sqlite3 + akh.<name>.sqlite3), gated on the split
    # evidence; without it the shared akh.sqlite3 render is unchanged.
    database = fa.database_settings(root)
    if database.layout is not None:
        warnings.append(f"per-akh databases (ADR 120): [database] and DATABASE_URL point at "
                        f"{fa.HOST_DATABASE} with layout = \"per_akh\" (split evidence "
                        f"{database.evidence}); {fa.SHARED_DATABASE} stays for rollback only")
    retargeted = fa.retarget_events(events, root)
    staged = fa.render_anima_events(fa.render_monitoring_events(retargeted, root), live=live)
    retired_routes = [r for r in retargeted.get("routes", []) if fa.legacy_route(r)] if live else []
    retired_servers = (fa.LAYA_SERVER,) if live else ()

    toml = {}
    host = fa.render_host_config(database.url, fa.acp_principals(session, root, acp_akhs), acp_port,
                                 auto_start, database_layout=database.layout)
    host_comments = dict(fa.HOST_COMMENTS)
    if database.layout is not None:
        host_comments["database"] = (
            "ADR 120 per-akh layout: host-level tables here, each akh in state/core/akh.<name>.sqlite3.\n"
            f"Split evidence: {database.evidence}. Rollback: CUTOVER.md \"Per-akh databases\".")
    toml[root / "core-config/akh-medu/config.toml"] = fa.dump_toml(
        host, host_comments,
        header="Rendered by deploy/roman-mini/configure_core.py (Phase 53f, ADR 113).\n"
               f"akhomed-main-{provenance.short} = {provenance.commit}")

    if teacher is not None:
        path = root / f"core-config/akh-medu/workspaces/{fa.DOJO_TEACHER}.toml"
        existing = tomllib.loads(path.read_text()) if path.exists() else None
        toml[path] = fa.dump_toml(fa.render_dojo_teacher_workspace(existing, teacher), fa.DOJO_TEACHER_COMMENTS,
                                  header=f"{fa.DOJO_TEACHER}: sealed Lilith C4 dojo teacher (ADR 116; "
                                         "tool_surface = session-only, an exam instrument, not a working akh). "
                                         "Rendered by configure_core.py.")

    seeds = {}
    for role, spec in specs.items():
        path = root / f"core-config/akh-medu/workspaces/{role}.toml"
        existing = tomllib.loads(path.read_text()) if path.exists() else None
        if not spec.session_grants:
            # Lilith: her own identities, no akhsessd grants, no reactor seeds
            # yet (her wakes are a later slice), exec as the Romans use.
            servers = fa.lilith_mcp_servers(session)
            prompt = fa.system_prompt(source / "roles", role, servers, cos_source, files=spec.prompt_files)
            agent_id, id_warnings = fa.board_agent_id(root, role)
            warnings.extend(id_warnings)
            board = fa.with_board_agent_id(fa.board_identity(events, role), agent_id)
            doc = fa.render_workspace(existing, role, prompt, servers, board,
                                      exec_table={"kind": "none"}, inference=spec.inference,
                                      chief_of_staff=fa.render_chief_of_staff(root, lilith))
            toml[path] = fa.dump_toml(doc, fa.LILITH_WORKSPACE_COMMENTS,
                                      header=f"{role}: full akh (ADR 113), Chief of Staff (ADR 116). "
                                             "Rendered by configure_core.py.")
            continue
        servers, server_warnings = fa.mcp_servers(session, role, root, retired_servers)
        warnings.extend(server_warnings)
        delegation = None
        if role in fa.DELEGATING_ROLES:
            delegation, delegation_warnings = fa.render_delegation(session, root, projects, role, cluster=cluster)
            warnings.extend(delegation_warnings)
        # 53g PR 8: after the operator writes config/vesta-sandbox.json, Vesta's
        # [exec] targets her akhsbxd sandboxes and her policy prompt is the
        # vesta-remediation skillpack instead of roles/addenda/vesta.md.
        prompt = fa.system_prompt(source / "roles", role, servers, fa.role_skill(source, role, root),
                                  files=spec.prompt_files)
        agent_id, id_warnings = fa.board_agent_id(root, role)
        warnings.extend(id_warnings)
        doc = fa.render_workspace(existing, role, prompt, servers,
                                  fa.with_board_agent_id(fa.board_identity(events, role), agent_id),
                                  delegation=delegation,
                                  monitoring=fa.monitoring_config(session, role, root, observer_retired=live),
                                  exec_table=fa.render_exec(role, fa.vesta_sandbox(root)),
                                  inference=spec.inference)
        toml[path] = fa.dump_toml(doc, fa.WORKSPACE_COMMENTS,
                                  header=f"{role}: full akh (ADR 113). Rendered by configure_core.py.")
        seeds[role] = fa.reactor_seeds(staged, session, role, source=source, retired=retired_routes)
    return Rendered(toml=toml, seeds=seeds, staged=staged, core=fa.render_core_config(core, root, akhs),
                    akhs=akhs, specs=specs, auto_start=auto_start, warnings=warnings, lilith=lilith,
                    database=database, teacher=teacher, acp_akhs=acp_akhs)


def plan(root, provenance, source=None, acp_port=fa.ACP_TCP_PORT, live=False):
    """`--plan`: the rendered-TOML diff per file, without writing anything.

    Full re-renders have dropped hand-applied holds before (Janus's broker
    hold, 2026-09-27), so the operator reviews exactly what a render would
    replace. Returns (report lines, warnings).
    """
    source = source or Path(__file__).resolve().parent
    rendered = render_configs(root, provenance, source, acp_port, live)
    lines = []
    for path, text in rendered.toml.items():
        rel = str(path.relative_to(root))
        old = path.read_text() if path.exists() else None
        if old == text:
            lines.append(f"= {rel}: unchanged")
            continue
        if old is None:
            lines.append(f"+ {rel}: new file")
        else:
            lines.append(f"~ {rel}: changes")
        lines.extend(line.rstrip("\n") for line in difflib.unified_diff(
            (old or "").splitlines(keepends=True), text.splitlines(keepends=True),
            fromfile="/dev/null" if old is None else f"a/{rel}", tofile=f"b/{rel}"))
    lines.append("daemon.auto_start = " + json.dumps(rendered.auto_start))
    return lines, rendered.warnings


def release_provenance(root):
    """The provenance of the release currently installed (config/core-release.json),
    for `--plan` without `--repo`: the host config header then diffs as unchanged."""
    path = root / "config/core-release.json"
    if not path.exists():
        raise fa.RenderError(f"{path} not found; pass --repo <clean origin/main checkout> with --plan")
    release = json.loads(path.read_text())
    return fa.Provenance(commit=release["commit"], short=release["short"],
                         origin_main=release.get("origin_main", release["commit"]), committed_at=0)


def render(root, provenance, binary, binary_sha256, acp_binary=None, source=None,
           acp_port=fa.ACP_TCP_PORT, live=False, build_sha256=None, versioned_binary=None,
           codesign=None):
    """Render every full-akh file under `root`. Returns warnings.

    `binary` is the path the core actually execs (`bin/akhomed`, the stable
    signed path, in production; a test may pass any path directly).
    `binary_sha256` is that file's own sha256 — `daemon_credentials.py` role
    `core` checks it before exec. `build_sha256` (defaults to `binary_sha256`
    when omitted) is the unsigned `target/release` artifact's sha256, kept
    for reproducible-build provenance. `versioned_binary` and `codesign` are
    recorded as-is (both may be None, e.g. in tests that render without
    going through `install_core_binary`).

    `live=False` (default) stages the PR 6 Anima source in shadow mode next
    to the observer/laya routes. `live=True` is the PR 6 cutover: the source
    ingests, the watchdog moves into akheventd, the legacy routes, their
    reactor seeds (as `remove` operations) and the `laya` MCP server go, and
    `[monitoring]` stops reading the observer's status export.
    """
    source = source or Path(__file__).resolve().parent
    # Everything derived from the operator inputs is computed first, so a
    # malformed switch file fails before a single file is written.
    rendered = render_configs(root, provenance, source, acp_port, live)
    warnings = list(rendered.warnings)
    for name in ("core-config/akh-medu/workspaces", "core-data", "core-state", "core-runtime",
                 "core-cache", "state/core", "config/reactor-seeds", "launchd", "evidence", "logs"):
        (root / name).mkdir(parents=True, exist_ok=True, mode=0o700)
    fa.private_dir(root / "secrets")
    config = root / "config"
    session = json.loads((config / "sessions.json").read_text())
    url = rendered.database.url

    # Skillpacks ship next to the rendered config so the operator can read
    # exactly the procedures a role runs with: Janus's escalation and
    # merge-repair workers (53g PR 7, read by path on every dispatch; the
    # registered projects are the keys of the retired broker's
    # config/janus-escalation.json) and Vesta's remediation procedures
    # (53g PR 8).
    if (source / "skills").is_dir():
        shutil.copytree(source / "skills", root / "skills", dirs_exist_ok=True)
    # The cluster akhsessd token (config/janus-delegation.json) is the
    # cluster's own issued secret, so it must already exist; only the
    # local-akhsessd path gets a freshly generated one here.
    if fa.janus_delegation_switch(root) is None:
        _ensure_key(root / fa.DELEGATION_TOKEN)

    for path, text in rendered.toml.items():
        _save_text(path, text)
    for role, seeds in rendered.seeds.items():
        _save_json(config / f"reactor-seeds/{role}.json", seeds)
    for role, spec in rendered.specs.items():
        if spec.skill is None:
            install_skillpack(source, root, role)
        else:
            # Lilith: the chief-of-staff pack, whose permission-profile.json
            # her [chief_of_staff] table names.
            install_skill_tree(cos_skill_source(source), fa.workspace_skill_dir(root, role, spec.skill))
    if fa.LILITH in rendered.specs:
        # The log's directory (outside her write roots); akhomed creates the file.
        fa.private_dir((root / fa.LILITH_ANNOUNCEMENTS).parent)

    _ensure_key(root / "secrets/core-events.key")
    _ensure_key(root / f"secrets/{fa.MONITOR_READER}.key")
    _save_json(config / "core.json", rendered.core)
    _save_json(config / "events.akh-target.json", rendered.staged)
    missing = fa.missing_monitor_files(json.loads((config / "events.akh-target.json").read_text()))
    if missing:
        warnings.append("akheventd would not start with the staged events config until these exist "
                        "(PR 5 / D2, see INVENTORY.md): " + ", ".join(missing))

    # ACP shared secret: one accepted token per line (akhomed reads
    # AKHOMED_SHARED_SECRET_FILE). Until per-principal ACP scoping lands
    # (phase53/fixups) any of these keys can open a session on any Roman.
    if fa.LILITH in rendered.specs:
        # T3 chat with Lilith: her scoped client key, created once, never overwritten.
        _ensure_key(root / fa.LILITH_T3_KEY_FILE)
    if rendered.teacher is not None:
        # The dojo runner's scoped key (Lilith C4), created once, never overwritten.
        _ensure_key(root / fa.DOJO_TEACHER_KEY_FILE)
    keys = [Path(p["token_file"]) for p in fa.acp_principals(session, root, rendered.acp_akhs or rendered.akhs)]
    missing = [str(k) for k in keys if not k.exists()]
    if missing:
        raise fa.RenderError("ACP client key files are missing: " + ", ".join(missing))
    _save_text(root / "secrets/acp-clients.keys", "".join(k.read_text().strip() + "\n" for k in keys))

    acp_path = acp_binary or (root / "bin/akh-acp")
    if acp_binary is None:
        warnings.append("no akh-acp built from main was installed; T3 keeps bin/akh-acp, which must "
                        "support --akhomed before the transport switch works")
    t3_acp = {"mode": "akhomed", "endpoint": f"tcp://127.0.0.1:{acp_port}",
              "health_url": f"http://{fa.CORE_LISTEN}/healthz", "binary": str(acp_path),
              # The roles t3_entrypoint.py accepts: every akh this core hosts.
              "akhs": list(rendered.akhs)}
    if rendered.lilith is not None:
        # Her serve pod is normally stopped; the entrypoint starts it on demand
        # from this checkout (docs/ops/lilith-serve.md).
        t3_acp["lilith_serve"] = rendered.lilith.serve_checkout
    _save_json(config / "t3-acp.json", t3_acp)

    release = {"binary": str(binary), "versioned_binary": str(versioned_binary) if versioned_binary else None,
               "commit": provenance.commit, "short": provenance.short,
               "origin_main": provenance.origin_main,
               "build_sha256": build_sha256 if build_sha256 is not None else binary_sha256,
               "sha256": binary_sha256, "codesign": codesign,
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
    if fa.LILITH in rendered.specs and (source / "t3_entrypoint.py").exists() and (root / "bin").is_dir():
        # T3's `roman-lilith` provider binary: the same file, its role is the basename.
        # (The Romans' roman-<role> copies are installed by cutover.sh / configure_t3.py.)
        entrypoint = root / "bin" / f"roman-{fa.LILITH}"
        shutil.copy2(source / "t3_entrypoint.py", entrypoint)
        entrypoint.chmod(0o700)
    return warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--binaries", type=Path,
                        help="target/release of the clean origin/main checkout (required unless --plan)")
    parser.add_argument("--repo", type=Path,
                        help="the clean origin/main checkout (required unless --plan; with --plan it "
                             "sets the provenance in the host config header, default: the installed release)")
    parser.add_argument("--no-fetch", action="store_true", help="skip `git fetch origin`")
    parser.add_argument("--acp-port", type=int, default=fa.ACP_TCP_PORT)
    parser.add_argument("--live", action="store_true",
                        help="PR 6 cutover: the roman-anima source ingests (default: shadow), "
                             "drops the legacy observer/laya routes, seeds and laya MCP server")
    parser.add_argument("--plan", action="store_true",
                        help="print the rendered-TOML diff per workspace and the host config; write nothing")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.plan:
        provenance = (fa.main_provenance(args.repo.resolve(), fetch=not args.no_fetch) if args.repo
                      else release_provenance(root))
        lines, warnings = plan(root, provenance, acp_port=args.acp_port, live=args.live)
        for line in lines:
            print(line)
        for warning in warnings:
            print("warning:", warning)
        print("Plan only: nothing was written. Re-apply any hand-held edit the diff removes, or "
              "render and restore it afterwards (see CUTOVER.md).")
        return
    if args.binaries is None or args.repo is None:
        parser.error("--binaries and --repo are required unless --plan is given")
    # Everything rendered here is private unless written with an explicit mode.
    os.umask(0o077)
    provenance = fa.main_provenance(args.repo.resolve(), fetch=not args.no_fetch)
    built = args.binaries.resolve() / "akhomed"
    if not built.is_file():
        raise fa.RenderError(f"{built} not found; run `{fa.BUILD_COMMAND}` in {args.repo}")
    if built.stat().st_mtime < provenance.committed_at:
        raise fa.RenderError(f"{built} is older than commit {provenance.short}; rebuild it from this checkout")
    codesign_config = load_codesign_config(root)
    warnings = []
    core_install = install_core_binary(built, root, provenance, codesign_config, warnings)
    acp_binary = None
    if (args.binaries.resolve() / "akh-acp").is_file():
        acp_binary = root / "bin" / provenance.acp_binary_name
        install_binary(args.binaries.resolve() / "akh-acp", acp_binary)
    warnings += render(root, provenance, Path(core_install["binary"]), core_install["sha256"], acp_binary,
                       acp_port=args.acp_port, live=args.live, build_sha256=core_install["build_sha256"],
                       versioned_binary=Path(core_install["versioned_binary"]), codesign=core_install["codesign"])
    print(f"Configured full akh core {provenance.binary_name} ({provenance.commit}); "
          f"merge-base --is-ancestor {provenance.short} origin/main ({provenance.origin_main[:12]}): ok")
    print("signed bin/akhomed with identity " + str(core_install["codesign"]["identity"]))
    for warning in warnings:
        print("warning:", warning)
    print("roman-anima source: " + ("LIVE (cutover)" if args.live else "shadow"))
    print("Not started. Follow deploy/roman-mini/CUTOVER.md.")


if __name__ == "__main__":
    try:
        main()
    except fa.RenderError as error:
        print("configure_core: " + str(error), file=sys.stderr)
        sys.exit(1)
