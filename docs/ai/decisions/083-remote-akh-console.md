# ADR 083 — Remote Akh Console (russh client, session mgmt, co-piloted terminal)

> Date: 2026-07-04
> Status: Proposed. **Integrated with ADR 082 (executor) — DEPENDS on executor
>          48c/48d (shared russh + OIDC/cert) + an `akh-executor` PTY service.
>          Not independently shippable; sequence after those.**
> Numbering: develop-wide (079/080 used on main; memory-schema→081; executor→082;
>            this = 083). Provisional "Phase 49" (executor claimed Phase 48).
> Related: ADR 059 (client/server split), ADR 061 (ACP-only `akh` client),
>          ADR 071 (CLI command tree), ADR 049 / Phase 38 (relocatable Akh),
>          ADR 070 (selectable backings), **ADR 082 (privilege-separated executor
>          `Backend`) — the concurrent "B" thread; shares the russh transport +
>          auth model (see "Coordination with ADR 082").**
> Plan: docs/ai/plans/2026-07-04-phase49-remote-console.md

## Context

`akh` is already a **pure ACP client** (ADR 061): it discovers a running
`akhomed` and speaks JSON-RPC over a Unix socket / loopback TCP. Two capabilities
are missing for operating an akh that lives on a *remote* brain host:

1. **Remote control ("A").** Drive a remote `akhomed` from a laptop without
   exposing a network port or building TLS.
2. **Co-piloted operator console ("C").** A terminal *into the akh's own
   sandbox* that the operator drives with their own muscle memory (look things
   up, debug, run deploy commands under manual care) and that **the agent can
   read** — so when the operator asks for help, the agent already sees the
   screen, and can prepare a multi-step **runbook** the operator runs and adjusts.

A third capability — the **agent** autonomously driving terminals as a tool — is
the *concurrent* thread **"B" (ADR 082, executor `Backend`)**: tool calls
dispatched to an executor that may run on another host, over **russh +
OIDC→SSH-cert**. After reconciliation (see "Coordination with ADR 082"), **this
console is *integrated* with B, not independent of it**: it reuses B's shared
russh transport + cert auth, and the console PTY runs **on the executor's bound
target** (the true akh sandbox). The console therefore **depends on executor
48c/48d** landing first.

### What already exists (do not rebuild)

- ACP transport over Unix socket / TCP; per-request `-w` workspace routing
  (`src/acp/`, `src/bin/akhomed.rs`).
- `ShellExecTool` — headless `/bin/sh -c`, 30 s timeout, 64 KB cap
  (`src/agent/tools/shell_exec.rs`). The console is the *interactive* cousin.
- Relocation is **Postgres-based**, not SSH (ADR 049). SSH is new.
- The `akh` CLI tree + `daemon`/`akh`/`workspace` groups (ADR 071).
- No PTY / terminal emulation anywhere; TUI is ratatui/crossterm *rendering*.

## Decision

### 1. Transport — one shared russh layer (with ADR 082), multiplexed

Use **russh** (embedded, not system `ssh`) via a **single shared SSH transport
layer** — the same crate/module ADR 082 builds; there is **one** SSH stack, not
two. SSH is the **sole network trust boundary**; no exposed ports, no TLS. Two
network legs, each a russh connection on that layer:

- **Laptop ↔ akhomed (remote control, "A").** **Control channel** = russh `exec`
  of **`akhomed acp-bridge`**, relaying the daemon's *own* local `akh.sock` over
  stdio (self-locating → zero remote-path discovery). ACP JSON-RPC rides this. A
  **terminal channel** = russh `direct-streamlocal` to a per-session akhomed
  socket (`0600`) whose path `terminal.create` returns — a **pure raw PTY byte
  pipe** to the client; **resize rides the control channel**.
- **akhomed ↔ executor target (console shell, "C").** The console PTY lives at
  the executor's bound target; akhomed sources its raw bytes over the
  **executor's** russh connection (a PTY service on `akh-executor`) and **relays**
  them to the laptop's terminal channel while feeding its own emulator (§3). When
  the executor is *local-drop* (same host), this leg is in-process.

Channel separation gives HOL isolation (a terminal flood never stalls chat) and
avoids base64-into-JSON overhead for the byte stream.

### 2. Auth & trust — OIDC→SSH-cert (per ADR 082)

- **Primary (remote):** **OIDC → short-lived SSH certificate** (ADR 082 decision
  6) — no key-hoarding; the cert's key-id carries the audit correlation-id. The
  console reuses the executor's cert path verbatim.
- **Fallback (local/dev only):** ssh-agent → `IdentityFile` (passphrase prompt),
  known_hosts TOFU-on-unknown / **fail-on-mismatch**, no password — behind an
  explicit dev flag so it never becomes the default.
- **`~/.ssh/config`** parsing (aliases; `ProxyJump` **hard-errors** in v1, no
  silent bypass) applies to the **fallback** path only; the cert path resolves
  targets from executor config, not `~/.ssh/config`.

### 3. Console — PTY on the executor target, emulator in akhomed

- The console shell is a **PTY on the executor's bound target** (the true akh
  sandbox), spawned by `akh-executor` as the akh user with the **same authority
  as the agent's own backend** (no *extra* confinement here — confinement is ADR
  082's OS-authoritative model). Rationale: the operator's deploy commands must
  run *in the same sandbox* the agent operates in — which ADR 082 defines — so
  the shell lives at the executor target, not a second environment. Rejected:
  client-side emulation (round-trips bytes laptop-ward) and an akhomed-host PTY
  (wrong sandbox when the executor is remote).
- **akhomed owns the `vt100` emulator.** It consumes the target PTY's **raw
  bytes** (streamed over the executor leg), maintains the screen+scrollback, and
  the **agent reads that emulator** (clean-text source of truth). akhomed **fans
  the same raw bytes** to the laptop's terminal channel; the client runs its own
  `vt100`/`tui-term` to display. All emulators eat identical bytes ⇒ in sync.
  (Grid/cell-diff single-emulator — deferred; weak payoff over reliable TCP.)
- **Session model:** session-scoped, **tmux-like detach/reattach** — the PTY
  outlives a dropped *laptop* link because it lives at the target under the
  executor's per-session process; reattach replays scrollback from akhomed.
  Bound to the executor session: **dies when the executor session drops/sleeps**
  (ADR 082's ephemeral model) — wake relaunches a fresh shell, no live migration.
  **Ephemeral by default** (no secrets at rest).
- **Attach policy:** **single human client** per console. A second attach is
  **refused naming who is connected** (client/user/host/since); takeover requires
  an **explicit force-detach**. The **agent is a separate reader** of akhomed's
  emulator, unaffected by human attach state.

### 4. Agent access — read-only on the PTY, Hybrid engagement

- **Read-only on the terminal.** The agent never sends keystrokes to the console
  (that is B).
- **Hybrid:** an ambient **breadcrumb** (a "console changed" flag + the last few
  **clean-text** lines) for near-free situational awareness, plus a
  **`console_read`** tool (current screen + optional scrollback range) for the
  deep look, plus an operator **"ask akh about this"** hotkey that snapshots the
  screen into the chat turn.
- **Privacy:** **opt-in** sharing per session (off until enabled, visible
  indicator); **automatic `ECHO`-off suppression** (password/passphrase prompts
  are not captured); **clean text only** (escape codes stripped); **ephemeral**
  (RAM-only shadow; not persisted to KG unless the operator archives).

### 5. Runbook — agent writes the queue, never the PTY

- The agent is **read-write on a runbook queue** but **read-only on the PTY** —
  the queue is a *separate artifact*, not the terminal. This preserves "the agent
  cannot run anything" while letting it prep a multi-step plan.
- Session-scoped, ephemeral, ordered steps `{ command, agent_note,
  status: pending|staged|done|skipped }`. The **agent** (via `runbook_*` tools)
  appends / revises / reorders / annotates *pending* steps; the **operator**
  edits, stages (paste-to-prompt), runs (presses Enter themselves), skips,
  reorders, deletes. The agent **never executes and never marks a step done.**
- **Advance = manual, no command mutation.** Staging the next step consumes the
  current one; the agent judges success by *reading the console*, not a captured
  exit code. (Precise per-command status via **OSC-133** shell-integration marks
  is a fast-follow.) **Paste-to-prompt (unexecuted)** is in v1.
- **Lifetime:** ephemeral v1; a **Phase-30 (procedural skills)** save path
  (runbook → reusable named procedure) is deferred.

### 6. Client surface (TUI-only) + CLI

- **Chat-primary** ratatui client with a **summonable terminal split**; the
  runbook panel shows **only when non-empty**. **Focus:** keys go **raw to the
  PTY** when the terminal pane is focused; a **configurable tmux-style prefix**
  escapes to client verbs (switch focus, "ask akh about this", stage/skip runbook
  step, resize). Configurable because the `akh` TUI itself usually runs inside an
  outer tmux.
- Console + runbook are **TUI-only**; `--headless` and the daemon are untouched
  (the daemon only *serves* the console socket; `console_read` has content only
  when a TUI operator has a live console).
- **CLI (folds into ADR 071):** **bare `akh` → local TUI**;
  **`akh --host <alias|user@host>` → TUI against the remote daemon**;
  `akh --host X daemon start|status|stop` execs the akhomed lifecycle over SSH;
  `--host` resolves via `~/.ssh/config`; `-w` still applies. No dedicated
  `console` subcommand in v1 (summon from within the TUI).

## Coordination with ADR 082 (executor `Backend`) — RESOLVED

ADR 082 (the "B" thread) is a grilled, locked design using **russh + stock
sshd**, an **ephemeral per-session executor process**, **ACP-wire reuse**, and
**OIDC→short-lived SSH cert** auth. The three overlaps are resolved as:

1. **Transport — SHARED.** One russh/SSH transport layer (common crate); the
   console's channels/legs ride it. No second SSH stack.
2. **Auth — ALIGNED.** OIDC→SSH-cert is primary; ssh-agent/known_hosts is a
   dev-only fallback (§2). This supersedes the grilled key-based answers.
3. **PTY placement — EXECUTOR TARGET.** The console shell runs at the executor's
   bound target (true sandbox); akhomed holds the emulator and relays (§3).

**Dependency:** the console **requires executor 48c (russh remote + ACP wire) and
48d (OIDC/cert auth)** to land first, plus a **PTY service on `akh-executor`**. It
is *not* independently shippable — sequence it after those.

## Consequences

- **One shared russh layer** with ADR 082 (self-locating exec-bridge for control;
  raw byte pipe for the terminal), zero exposed ports, cert-based auth + `0600`
  socket perms. No second SSH stack to maintain or secure.
- The **one-emulator-in-akhomed** invariant makes "the agent can read the
  terminal" true *by construction* — the human's screen *is* the agent's read
  target — even though the PTY child lives at the executor target.
- **Console is gated on ADR 082 (executor 48c/48d + an `akh-executor` PTY
  service).** Cost of choosing true-sandbox parity: it ships *after* the executor
  transport, not standalone. Benefit: no throwaway auth stack, real sandbox.
- Deferred, explicitly: ProxyJump **chaining** (dev-fallback path), OSC-133
  status, runbook→Phase-30 save, multi-attach mirroring, grid-diff render.

## Verification

- OIDC→SSH-cert connect via the shared transport (reuse ADR 082's harness); the
  dev-fallback ssh-agent path connects against a test `sshd`, resolves a fixture
  alias, and **errors** on a fixture `ProxyJump`.
- Console PTY spawns at the executor target (local-drop + remote), streams raw
  bytes to akhomed's emulator; attach → `vim`/`htop` → detach (laptop link drop)
  → reattach replays scrollback; a second attach is **refused naming the holder**;
  force-detach takes over. `ECHO`-off produces **no** captured bytes.
- Agent: breadcrumb reflects recent lines; `console_read` returns the screen;
  the agent cannot emit a keystroke to the PTY (no such tool).
- Runbook: agent appends steps; operator stages (paste-to-prompt, unexecuted) →
  edits → runs; agent revises pending steps; no step is agent-executed or
  agent-marked-done.
- Errors are miette diagnostics (no daemon → names `akh --host X daemon start`).
