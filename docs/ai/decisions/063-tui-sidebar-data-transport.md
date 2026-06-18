# ADR 063 — TUI OpenCode-style sidebar: data transport & agent-read dispatch

> Date: 2026-06-18
> Status: Planned
> Depends on: ADR-059 (akh client/server split), ADR-061 (ACP-only),
>             ADR-062 (LLM agentic chat loop), ADR-027 (agent lock discipline)
> Plan: [`docs/ai/plans/2026-06-18-tui-opencode-sidebar.md`](../plans/2026-06-18-tui-opencode-sidebar.md)

## Context

The TUI (`src/tui/`) is a pure single-column ACP chat client: header, message
stream, input box. OpenCode's TUI surfaces a rich right sidebar — context/token
meter, a todo/plan list, MCP + LSP status, and a model·branch footer — that akh
lacks. We want the same affordances, fed by akh's own internals (goals, the OODA
plan, the `ToolRegistry`, loaded skills), plus usage instrumentation that does
not yet exist.

Three facts from the code constrain the design:

1. **`dispatch_akh` is engine-only.** `dispatch_akh(engine, workspace, …)`
   (`src/acp/mod.rs:506`) has no `Agent` handle; only `_akh/permission/mode`
   reaches the agent, via `spawn_blocking` + `Arc<Mutex<Agent>>` lock
   (`mod.rs:1127`). Goals/plan/tools/projects all live on `Agent`.
2. **The agent `Mutex` is held for a whole turn** (ADR-027/ADR-062): an LLM
   agentic turn holds the lock for seconds–minutes. Anything that blocks on the
   agent lock blocks behind the live turn.
3. **`control_call` blocks the UI thread** — it builds a fresh current-thread
   runtime per call (`src/tui/mod.rs:456`). Acceptable for a one-shot `/model`,
   unacceptable for periodic sidebar refresh.

Token/context usage is **not tracked anywhere**: Ollama returns
`prompt_eval_count`/`eval_count` per response (`src/agent/ollama.rs`) and we
discard them; `OodaCycleResult` (`src/agent/ooda.rs:27`) carries no token field.

## Decision

### 1. Hybrid data transport

- **Slow-changing data (goals, plan, tools, skills, projects, daemon health)**
  is **polled** by the client from a new combined read endpoint `_akh/status`,
  on a ~1.5 s timer, from a dedicated background thread (a sibling of the
  existing `acp-remote` thread). The snapshot is delivered to the UI loop as a
  new non-blocking `ChatUpdate::Snapshot(StatusSnapshot)` variant, so the render
  thread never issues a blocking `control_call`.
- **Fast-changing usage (tokens this turn)** is **pushed live** over the
  existing ACP `session/update` stream as the turn consumes tokens, so the meter
  moves during a turn rather than jumping once per poll.

Rejected — *pure polling*: a 1.5 s usage meter feels dead mid-turn. Rejected —
*pure ACP push*: would force protocol additions for data that changes once every
several seconds and is just as easily pulled.

### 2. Agent reads via a non-blocking snapshot path

New read-only methods (`_akh/goals`, `_akh/tools`, `_akh/skills`,
`_akh/projects`, and the combined `_akh/status`) reach the agent the same way
`_akh/permission/mode` does — threading `Arc<Mutex<Agent>>` into an agent-aware
dispatch branch — **but take the lock with `try_lock`**. If the agent is busy in
a turn (lock held), `_akh/status` returns immediately with the engine-derivable
fields populated (skills, health) and the agent-derived fields marked `stale`
(last known / empty), rather than blocking behind the turn. This keeps ADR-027's
"the turn is the priority" invariant: a sidebar refresh can never stall a live
turn or be stalled by one.

### 3. Usage instrumentation

Capture Ollama's `prompt_eval_count` (prompt tokens) and `eval_count`
(completion tokens) in `OllamaClient::chat`; surface them on `AssistantTurn`;
accumulate a per-session `UsageLedger` (prompt/completion/total tokens, turn
count) in the agentic loop; expose `context_window` (from model metadata, with a
config default) for the `% used` figure and an optional per-model `cost_per_1k`
(config) for the `$` estimate. Serve via `_akh/usage`, fold into `_akh/status`,
and emit a live `session/update` usage delta per turn.

## Consequences

- Un-stubs `/goals` and `/status` over ACP (currently `remote.rs:199` returns
  "not yet available over ACP").
- `_akh/status` becomes the single round-trip that backs the whole sidebar —
  cheap, and degrades gracefully under an in-flight turn.
- Usage tracking is additive and Ollama-specific for now; the remote-fallback
  (OpenAI-compatible) path reports no tokens until wired (it already reports none
  for tool calling — ADR-062).
- Client gains typed view structs (`GoalView`, `PlanView`, `ToolView`,
  `SkillView`, `UsageView`) deserialized from the endpoint JSON, keeping the
  render layer in strongly-typed Rust rather than poking at `serde_json::Value`.
