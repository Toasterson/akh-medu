# ADR 060 — Claude-Code tool parity on a sandboxed Filesystem with a permission layer

> Date: 2026-06-16
> Status: Accepted
> Phase: P4 (Pure-ACP Agentic Harness pivot — Filesystem abstraction + tool parity + hardening)
> Depends on: ADR-053 (first-class AkhConfig), ADR-059 (akh client/akhomed server), ADR-027 (one Agent mutex per workspace, lock inside spawn_blocking)
> Plan: [`docs/ai/plans/2026-06-14-pivot-acp-agentic-harness.md`](../plans/2026-06-14-pivot-acp-agentic-harness.md)

## Context

To be an *effective coding agent on illumos that Anima drives over ACP*, an akh
needs Claude-Code-grade tools (Edit/Glob/Grep/WebSearch/structured task list) on
a filesystem it cannot escape, with a supervising client able to gate what it
does. The pre-P4 state (mapped by a parallel-reader pass) had concrete holes:

- **No filesystem abstraction.** Every tool used raw `std::fs`. `FileIoTool`'s
  read path applied **no** containment (it would read any path — `/etc/passwd`,
  `~/.ssh/id_rsa`), and its write path canonicalized only the parent with a
  non-canonical fallback, so a symlinked leaf or `..` tail escaped.
- **`ToolInput` was a flat `HashMap<String,String>`** — no way to express the
  structured input a task-list tool needs, with a load-bearing blast radius
  (~197 `.with_param` sites, a flat `{string:string}` WASM/CLI wire contract).
- **Missing tools**: Edit, Glob, Grep, WebSearch, structured task list.
- **No permission gating** of side-effecting tools.

A hard constraint runs through all of it (ADR-027): tool execution happens
inside `spawn_blocking` under a `std::sync::Mutex<Agent>` — no `.await`, no guard
across an await — while ACP client calls (`session/request_permission`, `fs/*`)
are async on `ConnectionTo<Client>`.

## Decision

**1. Typed `ToolInput` as a backward-compatible shim, not a big-bang.**
`ToolInput.params` is re-backed by `serde_json::Map<String, Value>`. The string
API is preserved verbatim — `with_param(k, v: Into<String>)` stores a JSON
string; `get`/`require` return `&str` only for JSON-string values — so all ~285
existing call sites and the WASM/CLI flat-string-map contracts compile
unchanged. Typed/structured accessors (`with_value`, `get_bool/i64/u64/f64`,
`get_str_list`, `get_value`) are additive, for tools whose input is genuinely
structured (the task-list tool's array of `{content,status,active_form}`).

**2. A `Filesystem` trait with a sandboxed `LocalFs`, Local-first.**
All file-touching tools go through `Arc<dyn Filesystem>` instead of raw
`std::fs`, so containment lives in one audited place. `LocalFs` keeps **separate
read-roots and write-roots** with symlink-safe canonical containment: reads
`canonicalize` (resolving symlinks + `..`) then check `starts_with` a root;
writes reject `..`, canonicalize the deepest existing ancestor, then check
containment. This closes the read-anywhere hole and the parent-only write
bypass while letting an akh read across its workspace but write only to a
narrower area. Directory-walking tools **do not follow symlinks** (`find`/`rg`
default) and bound recursion (depth cap), preventing in-root symlink cycles
from overflowing the stack. A `RemoteFs` (delegating to the client's `fs/*`)
implements the same trait — **deferred** (see Consequences).

**3. Five tools on the sandbox + typed input.** Edit (exact-unique replace),
Glob and Grep (walk *via* the `Filesystem`, so sandboxed for free; `regex` for
matching, no ripgrep subprocess), WebSearch (keyless DuckDuckGo HTML via
`ureq`+`scraper`), and a TodoWrite-style structured `task_list`. Failures are
data (`Ok(ToolOutput::err)`); `Err` is reserved for missing-required params.

**4. A two-layer permission model split along the sync/async boundary.**
A **synchronous `PermissionPolicy`** (modes `Bypass`/`Standard`/`AcceptEdits`/
`Plan`) decides allow/deny/ask from each tool's manifest (capabilities + danger
level) with no client round-trip, enforced at the single `ToolRegistry::execute`
chokepoint. Default mode is **`Bypass`** (allow all) so a standalone autonomous
akh is unaffected; a supervising client tightens it via the
`_akh/permission/mode` data-plane method (the first agent-mutating data-plane
op — it takes the agent lock inside `spawn_blocking`). The **interactive `Ask`**
is resolved by an optional `PermissionAsker`: an ACP bridge installs a sync
asker for the prompt turn that `blocking_send`s over an mpsc channel to an async
task holding `cx`, which calls `session/request_permission` and relays the
decision back over a oneshot — crossing the ADR-027 boundary without holding a
guard across an await. With no asker attached, `Ask` fails closed (deny).

## Consequences

- The agent's general-purpose file tools (file_io read+write, edit, glob, grep)
  are sandbox-confined and verified escape-proof (adversarial review: no escape;
  fixed a char-boundary audit panic, a grep symlink-cycle stack overflow, a
  grep read-before-size-check OOM, and glob symlink result-inflation).
- A client can supervise an akh over ACP: set the mode, and approve/deny
  side-effecting tools interactively.
- **Deferred, intentionally:**
  - **(c) Remote-FS** (`RemoteFs` over ACP `fs/*`): the illumos coding agent
    edits its *own* filesystem (LocalFs); editing the *client's* files is a
    distinct, lower-priority use case. `ClientCapabilities` are captured at
    `initialize` so it can be built later.
  - **Ingest-tool sandboxing** (text/csv/journal/code/pattern_mine still read
    via raw `std::fs`): ingestion is heavy and headed toward **LLM-assisted,
    subagent-offloaded** processing (a separate later effort); its sandboxing
    lands with that rework. The simple single-file ingest reads are the next
    increment; the tree-walk/glob ingesters route through the sandbox the way
    grep/glob do.
  - **Full mock-client `request_permission` round-trip test**: the bridge's
    sync↔async relay is unit-tested; the ACP wire types are compile-checked; an
    end-to-end test needs a prompt that deterministically triggers a gated tool.

## Status of work

P4.1–P4.5b landed on `claude/naughty-johnson-1add1a` (commits 8d04652, a5d1778,
ee09f62, 573e7a6, 4b20b18, 0d3bcba, e531d0b). Lib test count 2500 → 2548.
