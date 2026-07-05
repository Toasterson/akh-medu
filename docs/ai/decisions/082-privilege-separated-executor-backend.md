# ADR 082 — Privilege-separated executor Backend (per-akh OS principal, SSH transport)

> Date: 2026-07-04
> Status: Design (grilled; not yet built). Supersedes the in-process
>          `shell_exec` model and folds in the deferred `RemoteFs` (P4.5) note.
> Related: `src/agent/tools/shell_exec.rs` (the hole this closes),
>          `src/fs/mod.rs` (`Filesystem`/`LocalFs` — the seam this generalizes),
>          `src/agent/permission.rs` (`PermissionPolicy`/`PermissionMode`),
>          `src/agent/tool_manifest.rs` (`DangerLevel`/`Capability`),
>          `src/audit.rs` (`AuditKind` — extended here),
>          ACP transport (`agent-client-protocol`, the wire we reuse),
>          Phase 44 context-assembly runtime (the banner sink)
> Plan: [2026-07-04-phase48-privilege-separated-executor](../plans/2026-07-04-phase48-privilege-separated-executor.md)

## Context

Today the akh's "hands" are two very different shapes:

- **`file_io`/`glob`/`grep`/`edit`** go through a `Filesystem` trait (`LocalFs`,
  separate read/write roots, symlink-safe canonical containment). A real seam.
- **`shell_exec`** runs `/bin/sh -c <cmd>` **directly in the akhomed process**,
  as the akhomed OS user, with full host access. `DangerLevel::Critical`,
  `Capability::ProcessExec`, **no confinement at all.**

So the process-execution surface has no isolation boundary, and file vs. exec
can't be guaranteed to target the same place. Meanwhile the real deployment
topology is **distributed**: the brain (engine, KG, memory, WAL) runs on a
small box close to Postgres, while heavy `cargo test`/`build` wants a bigger
box. The current model can't express "run that over there."

The request that started this: replace the simplistic in-process shell/fs calls
with an **executor process per akh + session** that **drops to a limited per-akh
OS user** (provisioned via existing LDAP/Authd), leaning on **OS capabilities
admins already know** (illumos RBAC, Linux SELinux) for confinement, and able to
**SSH to a remote host transparently** so the model never hand-formats `ssh
user@host '…'`.

The mental model, refined during design: **the akh is a sysadmin who is already
logged in.** Every action is attributable to a real OS principal on a real host;
the OS — not akhomed — is the security boundary; akh-medu **rides existing
infrastructure** rather than building its own sandbox.

## Decision

Introduce a unified **`Backend`** (files *and* exec verbs, one bound target)
behind the existing tool surface, with pluggable backends that establish an OS
principal identity. Ten decisions, each resolved by grilling:

### 1. Driver & scope
Primary drivers are **fitting existing multi-tenant/networked hosting** and
**keeping actors identified for audit**; the SSH ergonomics are a third. This
targets **networked deployments that already have a shared-user setup**, not
single hosts. Single-host / Mac path is the *dev* case and gets a simpler
backend. `Backend` is a trait so stronger containment (namespaces, seccomp,
containers) can be added later without touching tools.

### 2. Contract & transparency
Structured verbs (`read`/`write`/`exec`/`glob`/`grep`/`edit`), **host bound to
the session, never a per-call argument.** The `attach` that binds a session to a
target is a **client action performed pre-session-start — invisible to the
LLM.** The session is *born logged-in*. Fully transparent: on the remote it
feels local; the model cannot tell where it runs and has **no per-call local
escape hatch.** A session is therefore a **stateful, lifecycle-managed object**
(target host, cwd, env); file verbs and exec verbs route to the **same** target.

### 3. Mechanism
A pre-installed **`akh-executor` binary** (owned/versioned by existing config
management — Ansible / illumos IPS / pkg), with the **process launched
per-session** over SSH via **russh** (pure-Rust, FLOSS-clean; the target keeps
its stock sshd). This is literally "executor process per akh *and* session":
the binary is durable infra, the process is session-scoped.

### 4. Lifecycle: ephemeral
The executor is **ephemeral** — it dies on connection drop / session sleep. On
wake, relaunch and **replay cwd/env** from the persisted session record; the LLM
re-orients (the Claude-Code-resume model — sessions die, you resume, the model
re-checks `git status` and re-runs). This deletes durable-executor machinery
(no orphan GC, no output spooling, no reattach protocol, no lingering).

### 5. Channel & exec model
Ephemeral collapses the earlier two-channel/socket idea to **one exec channel**:
RPC on **stdout**, executor logs/panics on **stderr** (SSH extended-data, a
distinct stream in the same channel — no socket, no `direct-streamlocal`
dependency). Exec is **block-until-exit** with **streamed** progress, a
**model-issued timeout** (models reflexively reach for `timeout(1)`) plus a
server-side ceiling backstop, and a **cancel** path. Concurrency is per-agent:
main + each subagent + each background job = **1..N executor processes, same OS
user**, multiplexed as **one SSH connection per session, one channel per agent**
(this is where russh multiplexing earns its keep). **No-sleep markers** pin
wakefulness while background jobs run; markers **do not survive session stop**.

### 6. Identity & credentials
- **Local-drop backend:** akhomed runs as a privileged service account
  authorized to `su`/become akh users (sudoers / PAM / a vetted SUID helper that
  validates the akh→uid mapping). No secret stored; the OS does the drop.
- **Remote backend:** akhomed **authenticates to the target's sshd as the akh
  principal**. Blessed path is **OIDC → short-lived SSH certificate**
  (CA-signed, principal-scoped, minutes–hours), plus **raw SSH certs** for sites
  already running a CA. **GSSAPI/Kerberos is dropped** (too hard to deploy).
  **Long-lived per-akh key-hoarding is discouraged by design** — akhomed must
  not become a permanent vault of every akh's identity; forcing certs/OIDC
  pushes operators toward correct auth. A **separate FLOSS helper project** to
  make CA/OIDC turnkey for companies is anticipated (the real adoption blocker;
  unbundling keeps akh-medu out of the identity-infra business).

### 7. Audit: correlation-index, not duplicate
A single remote exec produces **two** trails: akhomed's ledger (intent) and the
target's native sshd/auditd/SELinux (enforcement + effect). akhomed's ledger
becomes a **correlation index over the native audit**, not a re-implementation.
`AuditKind::ToolInvocation` gains an **identity/locus envelope**
`{ akh_id, session_id, agent_id, backend, host, os_user, exit_code,
correlation_id }`. The `correlation_id` is propagated into the remote principal
via the **SSH cert key-id** (sshd logs it on every cert auth — free join), plus
an **env var** and the OIDC **`jti`**. For local-drop / in-process there is no
cert; **env var + OS process accounting (the akh uid in `ps`/auditd) suffices.**
Consequence: a deployment with **no host-side auditing has a forensic gap
akhomed cannot fill** — documented loudly (miette warning).

### 8. Security boundary: OS-authoritative, confinement buys autonomy
The **OS (RBAC/SELinux + the per-akh user's own permissions) is the sole
security boundary.** akhomed does **not** replicate OS policy. Its
`PermissionPolicy`/`DangerLevel` is demoted from a security control to a
**backend-aware friction/audit knob**: `decide` gains an execution-context
input, and **confinement downgrades human-approval friction** — a `ProcessExec`
op that resolves to `Ask`/`Critical` **in-process** resolves to
`Allow`/downgraded behind a **confined** per-akh executor. That downgrade is the
**autonomy dividend** the isolation buys.

Confinement is an **admin-asserted, audited deployment claim** akhomed cannot
remotely prove, refined by an **executor self-report at handshake** (`euid`/uid,
gid, groups, SELinux/AppArmor domain or `unconfined`, capability bounding set,
whether anything outside `$HOME` is writable). **Effective danger = worst(admin
assertion, observed posture)**; **absent assertion ⇒ fail-safe to
unconfined/Critical.** When the executor reports **root / unconfined**, akhomed
(a) raises danger and warns the operator out-of-band, and (b) **injects a
standing banner into the LLM context** ("you are running as root on host X;
unconfined; the OS will not stop destructive commands") — recruiting the model's
own trained caution as a defense layer and socially discouraging root logins.
akhomed keeps exactly **one hard control**: authenticating the executor and
channel integrity (so a hijacked target cannot feed forged results).

### 9. Concurrency coherence
Main + subagents of one akh are the **same OS user → same target → same bound
working tree**; the OS boundary does nothing between them, so intra-akh
coherence is akhomed's problem. **Default: shared tree** (subagents exist to
keep context clean, not to parallelize writes). **Worktree isolation is an
explicit, model-launchable tool, git-only** — absent when the bound tree is not
a repo, expandable later (overlays for non-git). This is the deliberate lever
toward **frontier-style parallel-coding fan-out with openweight models.**

### 10. Landing
Unify `Filesystem` + exec behind one bound-session **`Backend`** so a remote
`file_io` write and a remote `shell_exec` are guaranteed coherent (same host,
cwd, auth). File tools move their injected dependency from `Arc<dyn Filesystem>`
→ `Arc<dyn Backend>`; `shell_exec` stops calling `Command::new` and calls
`backend.exec(...)`. **Tool signatures and the model-facing surface are
unchanged.** The wire protocol **reuses the ACP shape** (`fs/read_text_file` et
al. already exist) extended with exec/glob/grep/handshake verbs. Backends:
`InProcess` (today's behavior), `LocalDrop` (su to akh user), `Remote` (russh).
**Default = in-process, unconfined, honestly `Critical`/gated** (preserves the
Mac / standalone-akh path; no false confinement claim), with config to **(a)
prefer local-drop** and **(b) forbid in-process entirely** where security
warrants.

## Why this shape

- **OS as the boundary, akhomed as a friction knob** avoids maintaining a second
  policy model that drifts from the real SELinux/RBAC policy, and turns
  isolation into *earned autonomy* rather than pure overhead.
- **Ephemeral** matches how resumable agents already behave and removes the
  largest pile of distributed-systems machinery (reattach, spooling, GC).
- **Correlation-index audit** refuses to make the brain the single audit
  chokepoint; the cert key-id makes the join free where it matters (networked).
- **Reuse ACP + `Filesystem` seam** keeps this a refactor, not a fork: ~6 tools
  rewire, zero model-facing change.
- **Discourage key-hoarding / defer the CA helper** keeps akh-medu from owning
  an identity-infra problem it would do badly, and pushes operators to correct
  auth.
- **Root ⇒ context banner** is the cheapest, most honest backstop for the one
  claim we can't verify: it makes the unsafe posture visibly uncomfortable to
  both human and model instead of silently trusting it.

## Risks / watch-items

- **Block-until-exit vs. the brain's other duties** — a multi-minute build holds
  a channel + thread; confirm daemon background tasks (consolidation) run on
  independent threads so a long exec doesn't stall them (the blocked call is
  IO-bound on the channel, which helps).
- **The confinement dividend rests on an unverifiable claim** — a subtly
  over-broad SELinux domain still grants autonomy unsafely; observed-posture
  catches only the obvious cases (euid 0). Banner + audit are the honest
  backstop, not a proof.
- **ACP protocol stretch** — ACP was shaped for file ops + a supervising client;
  streaming exec + cancellation + posture-handshake may strain it. Spike the
  wire format before committing.
- **russh forwarding capability** — verify `direct-tcpip`/streamlocal before
  designing on it; the single-channel-stdio path sidesteps this and is the
  baseline.

## Status / next

- **Design:** complete (this ADR).
- **Not yet:** any code. See the plan for the phased build (48a `Backend` seam +
  in-process backend byte-identical to today → 48b local-drop → 48c russh remote
  + ACP wire → 48d audit envelope + cert key-id → 48e permission downgrade +
  posture handshake + root banner → 48f worktree tool).
- **First de-risking spike:** ACP-over-russh single exec channel with streamed
  stdout + stderr-via-extended-data + a handshake posture report — proves the
  transport and the wire in one shot before the trait refactor lands.
