# Phase 48 — Privilege-separated executor Backend (per-akh OS principal, SSH transport)

- **Status:** DESIGN (grilled 2026-07-04; not started). Ready for a first spike.
- **Anima work item:** Akh-Medu project `019d9844-ce74-7993-a9f0-c953b8f04a4e` (work-item id TBD — create + link)
- **ADR:** [082-privilege-separated-executor-backend](../decisions/082-privilege-separated-executor-backend.md)
- **Builds on / touches:** `src/fs/mod.rs` (`Filesystem`/`LocalFs`), `src/agent/tools/{shell_exec,file_io,glob,grep,edit}.rs`, `src/agent/permission.rs`, `src/agent/tool_manifest.rs`, `src/audit.rs`, ACP transport (`agent-client-protocol`), Phase 44 context-assembly runtime
- **New crates:** `russh` (client), `akh-executor` (new binary), a small privilege-drop helper (local backend)

## Why

`shell_exec` runs `/bin/sh -c` in the akhomed process as the akhomed user with
full host access (`Critical`, no confinement). The real topology is distributed
— brain near Postgres on a small box, heavy `cargo test`/`build` on a bigger box
— and the current model can't express "run over there," can't attribute actions
to an OS principal, and can't confine a misbehaving akh. This phase makes the akh
a **sysadmin who's already logged in**: every action runs as a real per-akh OS
principal on a real host, the **OS is the security boundary**, and akh-medu
**rides existing infra** (LDAP/Authd, RBAC/SELinux, sshd, auditd) instead of
building its own sandbox. See ADR 082 for the ten resolved design decisions.

## Architecture (target state)

```
                 ┌─────────────────────────── akhomed (brain) ───────────────────────────┐
   tools:        │  file_io / glob / grep / edit / shell_exec   (signatures UNCHANGED)    │
   dep injected: │            Arc<dyn Backend>  ◄── replaces Arc<dyn Filesystem>          │
                 │                     │  read/write/exec/glob/grep + handshake            │
                 └─────────────────────┼──────────────────────────────────────────────────┘
                                       │  (bound at session attach — client, pre-LLM)
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
   InProcess backend            LocalDrop backend             Remote backend (russh)
   (today's behavior,           akhomed su→akh user,          OIDC→short-lived SSH cert
    unconfined, Critical)       spawn akh-executor as         → sshd → spawn akh-executor
                                that uid (SUID/PAM)           1 conn/session, 1 chan/agent
                                                              RPC=stdout, logs=stderr(ext)
                                                              ACP-shaped wire protocol
```

- **Session** = stateful, born already `attach`ed to a `(backend, host, os_user,
  cwd)`. The LLM never sees the host.
- **Executor** = ephemeral per-agent process; dies with the connection/sleep;
  relaunched on wake with cwd/env replayed.
- **Effective danger** = worst(admin confinement assertion, executor handshake
  posture); absent assertion ⇒ unconfined/Critical; root ⇒ operator warning +
  LLM-context banner.

## Sub-phases (each independently landable)

### 48a — `Backend` seam + in-process backend (byte-identical to today)
- Define `trait Backend: Send + Sync` = the union of the current `Filesystem`
  verbs **plus** `exec(cmd, timeout, cwd, env) -> ExecResult` (streamed) and
  `handshake() -> Posture`.
- `InProcessBackend` wraps today's `LocalFs` + the current `shell_exec` logic
  verbatim (still `Critical`, unconfined). **Goal: zero behavior change.**
- Rewire `file_io`/`glob`/`grep`/`edit` from `Arc<dyn Filesystem>` →
  `Arc<dyn Backend>`; rewrite `shell_exec` to call `backend.exec(...)` instead of
  `Command::new`. Tool signatures + model-facing surface unchanged.
- **Exit criteria:** all existing tool tests pass unchanged; `shell_exec` still
  `Critical`; no `Command::new` left in the tool.

### 48b — LocalDrop backend (privilege drop, no network)
- akhomed (privileged service account) launches `akh-executor` as the akh uid via
  a vetted mechanism (SUID helper validating akh→uid, or PAM/`setuid`). Config:
  akh→uid mapping sourced from existing directory (assume-exists; do **not**
  provision users).
- **Config flags (ADR decision 10):** default `InProcess`; `prefer_local_drop`;
  `forbid_in_process` (hard-disable unconfined exec where security warrants).
- **Exit criteria:** `whoami` via the tool returns the akh user; writes outside
  `$HOME` denied by the OS (not by akhomed); `forbid_in_process` refuses to start
  an in-process backend with a miette diagnostic.

### 48c — Remote backend (russh) + ACP wire protocol  ← **first spike lives here**
- **Spike first (before 48a refactor even):** russh client → stock sshd → `exec`
  the `akh-executor` binary; **one channel**, RPC frames on stdout, executor logs
  on stderr (extended-data). Prove: streamed stdout of a long `cargo build`,
  model-issued timeout, cancel (signal over channel), and a handshake posture
  report. Verify russh's forwarding story is irrelevant on this path.
- `akh-executor` binary: reads ACP-shaped frames on stdin, executes verbs as its
  own (already-dropped) uid, streams results. **Reuse ACP** (`fs/read_text_file`
  etc.) + new `exec`/`glob`/`grep`/`handshake` verbs. Version-negotiate so newer
  akhomed talks to slightly older executor.
- Session = 1 russh connection; each agent (main/subagent/bg job) = 1 exec
  channel = 1 executor process.
- Ephemeral: connection drop / session sleep kills executors; wake ⇒ reconnect +
  relaunch + replay cwd/env.
- **Exit criteria:** a remote `cargo test` runs on the cloud box, streams output,
  respects a 20-min timeout, and cancels cleanly; a dropped connection surfaces a
  miette diagnostic and the next turn re-orients.

### 48d — Auth: OIDC→cert + raw certs (no key-hoard)
- Cert-based SSH auth as the akh principal; OIDC token exchange → short-lived
  cert from a CA/signing service; raw-cert path for existing CAs.
- **No** long-lived per-akh key vault in akhomed (dev-only single-key path may
  exist but is loudly fenced). GSSAPI explicitly out.
- **Exit criteria:** akhomed obtains a minutes-valid cert scoped to one akh
  principal, authenticates, and the cert **key-id carries the correlation id**;
  cert expiry mid-session is handled (re-mint on reconnect).

### 48e — Audit envelope + correlation propagation
- Extend `AuditKind::ToolInvocation` with `{ akh_id, session_id, agent_id,
  backend, host, os_user, exit_code, correlation_id }` (append-only; migrate the
  redb schema additively).
- Propagate `correlation_id` → SSH cert key-id (48d) + executor env var + OIDC
  `jti`. Verify the id appears in the target's `auth.log`/auditd.
- Doc + runtime miette warning when host-side auditing is absent (forensic gap).
- **Exit criteria:** an akh action can be joined from akhomed's ledger to the
  target's sshd/auditd logs by `correlation_id`.

### 48f — Permission downgrade + posture handshake + root banner
- `PermissionPolicy::decide` gains an execution-context arg (backend + asserted
  confinement + observed posture). Confined ⇒ downgrade `Ask`→`Allow` for the
  autonomy dividend.
- Executor `handshake()` returns posture (`euid`, caps, SELinux domain, writable
  roots); `effective_danger = worst(assertion, posture)`; absent assertion ⇒
  Critical.
- Root/unconfined ⇒ (a) operator warning, (b) **standing LLM-context banner** via
  the Phase-44 context runtime (high-retention contribution).
- **Exit criteria:** the same `shell_exec` resolves `Allow` behind a confined
  executor and `Ask`/`Critical` in-process; an euid-0 executor forces Critical +
  emits the banner even when the config *asserts* confined.

### 48g — Worktree-isolation tool (git-only, model-launchable)
- Explicit tool the model calls to spawn an isolated parallel coding agent in its
  own git worktree on the bound target; merge-back on success; conflicts surfaced
  to the spawning agent.
- Available **only** when the bound tree is a git repo; absent otherwise.
- **Exit criteria:** two model-launched sub-agents build in separate worktrees
  without clobbering one `target/`; clean results merge back.

## Deferred (not v1)

- Durable / reattachable executors (ephemeral is the decision).
- Worktree isolation for non-git trees (overlays / per-agent scratch).
- The standalone FLOSS CA/OIDC turnkey helper project (separate repo).
- Stronger local containment backends (namespaces, seccomp, containers) — the
  `Backend` trait leaves room; not scoped here.

## Open questions to resolve during the spike

- ACP frame shape for **streaming** exec + **cancel** + **handshake** — does ACP
  stretch cleanly, or do we fork a minimal executor protocol?
- Exact privilege-drop mechanism for LocalDrop across Linux **and** illumos (SUID
  helper vs PAM vs SMF/systemd `User=`), given the cross-platform (RBAC/SELinux)
  ambition.
- Cert re-mint cadence vs. session length; what a mid-build cert expiry does.

## Documentation flow (per CLAUDE.md)

1. This plan + ADR 082 committed to develop (backups).
2. Push both to Seshat (`seshat_ingest`) so the corpus stays in sync.
3. Create the Anima work item, link the file paths, set `current_phase`
   appropriately, and record the work-item id above.
