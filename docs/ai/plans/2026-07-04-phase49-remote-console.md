# Phase 49 — Remote Akh Console & Co-Piloted Terminal

> Date: 2026-07-04
> Status: Proposed. **Integrated with ADR 082 — DEPENDS on executor 48c/48d
>          (shared russh + OIDC/cert) + an `akh-executor` PTY service. Sequence
>          after those; not independently shippable.**
> ADR: docs/ai/decisions/083-remote-akh-console.md
> Anima WI: _TODO — create under project "Akh-Medu" (019d9844-…)_
> Related: ADR 059/061/071 (client/server, ACP-only, CLI tree), ADR 049/Phase 38
>          (relocation), **ADR 082 / Phase 48 (privilege-separated executor
>          `Backend`) — the concurrent "B" thread this console integrates with;
>          see "Coordination with the executor" below.**

## Coordination with the executor (ADR 082 / Phase 48) — RESOLVED

The "B" thread is a *grilled, locked design* (`docs/ai/plans/2026-07-04-phase48-privilege-separated-executor.md`).
It uses **russh + stock sshd**, an **ephemeral per-session executor process**,
**ACP-wire reuse**, and **OIDC→short-lived SSH cert** auth (key-hoarding
discouraged). Resolved with the user:

- **Transport SHARED** — one russh/SSH transport layer (common crate), not two.
- **Auth ALIGNED** — OIDC→SSH-cert primary; ssh-agent/known_hosts is a dev-only
  fallback (revises v1a's auth bullets).
- **PTY on the EXECUTOR TARGET** — the console shell runs at the executor's bound
  target (true sandbox even when remote); akhomed holds the emulator + relays.

**Dependency:** this console **requires executor 48c (russh remote + ACP wire) +
48d (OIDC/cert) + a PTY service on `akh-executor`**. Build order below assumes
those exist; the shared transport crate should be authored once (by whichever
thread reaches it first) and reused.

## Goal

Let an operator drive a **remote `akhomed`** from `akh` over **SSH** (no exposed
ports), and get a **co-piloted operator console**: an interactive terminal *into
the akh's own sandbox* that the operator runs with manual care, that the **agent
observes read-only**, and for which the agent can prepare a **runbook** the
operator steps through and adjusts.

## Non-goals (this thread)

- **Agent-driven** terminals / autonomous command execution — that is B
  (Executor). Here the agent is **read-only on the PTY**.
- Sandbox confinement beyond `shell_exec`'s current authority — B's job.
- Migrating a live terminal with a relocating akh (impossible; PTY is host-bound).
- A web/thin client (would motivate grid-diff render — deferred).

## Architecture (see ADR 083 for rationale)

```
 laptop                   shared russh (OIDC-cert)          brain host                   executor target
┌───────────────┐                                   ┌──────────────────────┐        ┌───────────────────┐
│ akh TUI       │ ─ exec: `akhomed acp-bridge` ───▶ │ akhomed daemon       │        │  akh-executor     │
│  chat pane    │     (ACP JSON-RPC, self-locating) │  ├ ACP control plane │        │   ├ agent exec RPC│
│  term pane ◀──┼─ direct-streamlocal term.sock ──▶ │  ├ vt100 emulator ◀──┼─ raw ──┤   └ console PTY   │
│   (vt100)     │     (raw PTY bytes; resize on ctl)│  │    ├▶ agent (reads) │  PTY   │  (akh-user shell) │
│  runbook panel│                                   │  │    └▶ relay→laptop │ (exec  └───────────────────┘
└───────────────┘                                   │  └ runbook queue (RW)│  leg)   (local-drop ⇒ same host)
                                                     └──────────────────────┘
```

**One emulator, in akhomed.** The console PTY child lives at the **executor
target** (true sandbox); its raw bytes stream to akhomed's `vt100`, which the
**agent reads** and which akhomed **relays** to the laptop. "The agent can read
the terminal" is true *by construction* — the human's screen *is* the agent's
read target. When the executor is *local-drop*, the exec leg is in-process.

## Milestones

> **Prereq (from ADR 082):** executor **48c** (russh remote + ACP wire), **48d**
> (OIDC/cert auth), and a **PTY service on `akh-executor`**. The shared russh
> transport crate is authored once and reused; do not build a second SSH stack.

### v1a — Remote control over the shared transport
- Reuse the **shared russh layer** (OIDC→SSH-cert primary). **Dev-only fallback**
  behind a flag: ssh-agent → `IdentityFile` (passphrase) → keyboard-interactive,
  no password; known_hosts TOFU/**fail-on-mismatch**; `~/.ssh/config` alias parse
  with **`ProxyJump` → miette error** (parser-only borrow; chaining deferred).
- **Control channel** = `exec` `akhomed acp-bridge` (new subcommand: relays local
  `akh.sock` over stdio, self-locating). Prove ACP round-trip over the cert path.
- **Daemon lifecycle over SSH:** `akh --host X daemon start|status|stop`.
  **Require running daemon**; missing socket → diagnostic naming the fix. No
  silent auto-start.
- CLI: **bare `akh` → local TUI**; `akh --host <alias|user@host>` → remote TUI.

### v1b — Console PTY (executor target) + emulator (akhomed)
- **`akh-executor` PTY service:** spawn an akh-user login-shell PTY at the
  executor's bound target (local-drop + remote), relay raw bytes + accept
  input/resize over the executor leg. (Shares the executor's identity/audit
  posture — same OS user, same handshake.)
- **akhomed** consumes that raw stream into its **own `vt100`** (the agent's
  source of truth) and exposes a client-facing terminal on `term-<id>.sock`
  (`0600`) for `direct-streamlocal` attach — **raw PTY byte pipe**, resize on the
  control channel.
- Control-plane methods: `terminal.create` (→ `{id, socket_path}`),
  `terminal.list`, `terminal.resize`, `terminal.detach`, `terminal.kill`.
- Session lifecycle: tmux-like **detach/reattach with scrollback replay** (PTY
  outlives a dropped *laptop* link); **single attach**, refuse-with-holder-identity
  + **force-detach**; **dies when the executor session drops/sleeps** (ADR 082
  ephemeral model); ephemeral.
- In-akhomed **subscribe/replay + resize API** — the seam the agent read path (and
  B) consume.

### v1c — Agent read path + privacy
- **Breadcrumb** (changed-flag + last-N clean lines) into agent context (Hybrid).
- **`console_read`** tool (screen + scrollback range), **read-only**.
- Privacy: **opt-in** per session + indicator; **`ECHO`-off suppression**;
  clean-text only; **ephemeral** (no KG persist unless archived).

### v1d — Runbook + TUI surface
- Runbook queue (agent RW, PTY read-only): `runbook_append/revise/reorder/
  annotate` tools; `{command, agent_note, status}`; **manual advance, no
  mutation**; agent revises pending from console reads.
- Client TUI: **chat-primary + summonable terminal split** (`vt100`/`tui-term`);
  **runbook panel when non-empty**; **raw-to-PTY focus** + **configurable
  tmux-style prefix** for client verbs.
- Operator verbs (under prefix): toggle/focus terminal, **"ask akh about this"**
  (snapshot → chat), **paste-to-prompt (unexecuted)**, stage/skip runbook step,
  resize.

## Deferred fast-follows (parked, with reasons)

| Item | Why deferred |
|------|--------------|
| **ProxyJump chaining** (recursive connect over `direct-tcpip`) | Real russh work; parser-only borrow. Dev-fallback path only; v1 errors on it. |
| **OSC-133 per-command status** | Needs shell integration; v1 uses manual advance + console reads. |
| **Runbook → Phase-30 save** | Reusable named procedures build on procedural-skills. |
| **Multi-attach mirroring** | Shared-input + winsize conflicts; v1 is single-attach + force-detach. |
| **Grid-diff render** | Only pays off for lossy links / thin/web clients; reliable-TCP-in-SSH doesn't need it. |
| **Opt-in remote daemon auto-start** | Principle: don't spawn daemons on someone's box behind their back. |

## Risks & mitigations

- **Dependency on ADR 082 (48c/48d + PTY service).** The console cannot ship
  before the executor transport + a PTY service exist. Mitigate by co-designing
  the shared transport crate + the `akh-executor` PTY verb with the B thread, and
  by spiking ACP-over-russh (B's first spike) jointly.
- **Two-hop relay latency** (laptop → akhomed → executor target). akhomed
  emulates + relays; extra hop only when the executor is remote. Acceptable;
  grid-diff/predictive-echo deferred.
- **Emulator drift** (akhomed vs client, both `vt100`). Mitigated: identical raw
  byte feed; single size authority (attached client → `terminal.resize`).
- **Secret leakage** into agent/KG/logs. Mitigated: opt-in, `ECHO`-off
  suppression, clean-text, ephemeral. Accepted residual: the console has the same
  authority the agent already has via its backend; "main SSH client" misuse is
  the operator's call.
- **PTY-service posture in `akh-executor`.** The interactive PTY must inherit the
  executor's identity/audit/handshake posture (ADR 082), not bypass it — verify
  the cert key-id correlation-id propagates to the console shell too.

## Verification

See ADR 083 "Verification". Adds: clap parse tests for `--host` / bare-`akh`-TUI;
integration test over the shared transport (cert path + dev ssh-agent fallback);
a console PTY-at-target stream test (local-drop + remote); a detach/reattach
replay test; an `ECHO`-off suppression test; a runbook staging/advance test
asserting no agent-side execution.

## Documentation tasks (per CLAUDE.md workflow)

- [ ] Create the Anima work item under "Akh-Medu"; paste this plan + ADR 083;
      set `current_phase`. Cross-link the executor WI (ADR 082 / Phase 48).
- [ ] `seshat_ingest` this plan and ADR 083 so the corpus stays in sync.
- [ ] On implementation: bump `docs/ai/architecture.md` (new `russh` client
      transport + console module) — *not now* (that doc tracks built state).
