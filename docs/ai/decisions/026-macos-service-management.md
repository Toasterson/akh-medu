# ADR-026: macOS Service Management via launchd

**Date:** 2026-03-01
**Status:** Accepted
**Deciders:** toasty

## Context

akh-medu's daemon (`akhomed`) needs to run autonomously on a Mac Mini M2 for days or weeks without manual intervention. macOS does not keep arbitrary user processes alive — without integration, `akhomed` gets reaped after logout, won't survive reboots, and won't auto-restart after crashes.

## Decision

Implement a `src/service.rs` module that generates macOS launchd plists and exposes `akh service {install,start,stop,status,uninstall,show}` CLI subcommands. No new dependencies — we generate the XML plist directly and shell out to `launchctl`.

### Key Design Choices

1. **User LaunchAgent, not system LaunchDaemon** — `~/Library/LaunchAgents/` requires no root. The service runs as the logged-in user, which matches the per-user XDG data model.

2. **`KeepAlive: true` + `RunAtLoad: true`** — launchd restarts on crash and starts on login. `ThrottleInterval: 10` prevents restart loops.

3. **`Nice: 10` + `ProcessType: Background`** — low CPU priority so the agent doesn't interfere with interactive use.

4. **`ExitTimeOut: 30`** — gives akhomed 30 seconds to persist sessions on SIGTERM before SIGKILL.

5. **Non-macOS stubs** — on Linux, all operations return `ServiceError::NotMacOS` with a diagnostic suggesting systemd. This keeps the module compilable everywhere.

6. **No new dependencies** — plist XML is simple enough to template. Avoids pulling in a plist crate for 150 lines of XML generation.

## Alternatives Considered

| Alternative | Why Not |
|------------|---------|
| systemd user service | Linux-only; Mac Mini M2 is macOS |
| Homebrew service formula | Requires Homebrew; adds external dependency |
| `launchd` via plist crate | Extra dependency for trivial XML generation |
| Docker container | Overkill for single-binary daemon; complicates data persistence |
| `nohup` / `screen` | No auto-restart, no boot persistence |

## Consequences

- `akh service install && akh service start` is the complete deployment story for macOS
- Crash recovery is automatic (launchd restart within 10-12s)
- Boot persistence is automatic (RunAtLoad)
- Logs land in `~/Library/Logs/akh-medu/` — standard macOS location
- Linux deployment needs a separate systemd unit (not implemented yet)

## Enhanced Monitoring

Alongside the service management, `DaemonStatus` was extended with:
- `last_persist_at`, `last_learning_at`, `last_sleep_at`, `last_goal_gen_at` — timestamps for each background task
- `active_goals`, `kg_symbols`, `kg_triples` — live KG statistics

This enables one-command health checks: `akh agent daemon-status` shows everything needed to verify autonomous operation.
