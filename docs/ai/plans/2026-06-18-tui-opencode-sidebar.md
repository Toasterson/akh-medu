# Plan — OpenCode-style sidebar for the akh TUI

> Date: 2026-06-18
> Status: Planned
> ADR: [`063-tui-sidebar-data-transport.md`](../decisions/063-tui-sidebar-data-transport.md)
> Anima WI: `019edc5c-4155-7f43-933c-7831c1286fac`

## Goal

Bring OpenCode's TUI affordances to akh's own TUI (`src/tui/`): a collapsible
right sidebar with a **context/usage meter**, a **live goals/plan panel**, a
**capabilities (tools + skills) panel**, and a **statusline/footer + command
palette**. Three of the four panels surface data akh already exposes; the usage
meter requires new LLM-layer instrumentation (tokens are currently discarded).

## Backing APIs (what exists today)

| Panel | API | Symbol |
| --- | --- | --- |
| Goals / plan | ✅ | `Agent::goals() -> &[Goal]` (`agent.rs:1598`), `Agent::plan_for_goal(id)` (`agent.rs:2360`) |
| Tools | ✅ | `Agent::list_tools() -> Vec<ToolSignature>` (`agent.rs:1613`) |
| Skills / projects | ✅ | `Engine::list_skills()` (`engine.rs:959`), `Agent::projects()`, `Agent::agenda()` |
| Usage | ❌ | not tracked; Ollama returns `prompt_eval_count`/`eval_count` (`ollama.rs`), discarded; add to `OodaCycleResult` (`ooda.rs:27`) |

Dispatch seam: `dispatch_akh` (`src/acp/mod.rs:506`) routes `_akh/*` data-plane
methods (engine-only); capabilities advertised at `mod.rs:528`; agent reached via
`Arc<Mutex<Agent>>` (the `_akh/permission/mode` precedent, `mod.rs:1127`). Client
control calls: `ext_request_unix`/`ext_request_tcp` (`mod.rs:1513/1531`), used by
`control_call` (`tui/mod.rs:456`).

## Layout

```
┌─ akh-medu :: <session title> ────────────────────────────┬─ Context ──────────┐
│ chat stream …                                            │ 12.4k tok · 41% ▓▓░│
│ ▸ 4 steps  (Tab to expand)                               ├─ Goals ────────────┤
│                                                          │ ● diagnose / ○ fix │
│                                                          ├─ Capabilities ─────┤
│                                                          │ 11 tools · 3 skills│
├──────────────────────────────────────────────────────────┴────────────────────┤
│ ☥ ⠹ thinking…                       GLM-5.2 · med · ~/infra:master · pid 1234   │
└─────────────────────────────────────────────────────────────────────────────────┘
  ctrl+b sidebar · ctrl+p palette · Tab thinking · esc interrupt
```

Sidebar = a `Constraint::Length(~32)` column carved from the right, toggled with
`ctrl+b`, auto-hidden below a min terminal width.

## Phases

### Phase A — Shell + footer + palette (client-only, no daemon changes)

1. Split layout in `widgets::render`: main column + collapsible sidebar; new
   `show_sidebar: bool` on `AkhTui`, toggled by `ctrl+b`. Sidebar panels render
   from whatever `StatusSnapshot` is cached (empty/placeholder until Phase C).
2. Statusline footer (one row above input): model·effort (cached from
   `_akh/model/list` `current`), workspace `git -C <ws> rev-parse --abbrev-ref
   HEAD` + dirty flag, daemon pid (`ServerInfo`) + connection health
   (`remote.is_disconnected()`).
3. Session-title heuristic: first user prompt → truncated title shown in header.
4. Command palette (`ctrl+p`): centered overlay reusing the `ModelPicker`
   pattern (filtered list + `↑/↓`/Enter/Esc) over the command set, with one-line
   descriptions; `/`-prefix autocomplete dropdown above the input.

### Phase B — Read endpoints + combined `_akh/status` (daemon)

5. Agent-aware dispatch branch: thread `Arc<Mutex<Agent>>` into the data-plane
   path and add `_akh/goals`, `_akh/tools`, `_akh/skills`, `_akh/projects`, each
   serializing the corresponding struct to JSON. Use **`try_lock`** (ADR-063 §2):
   if the agent is busy in a turn, return engine-only fields + `stale: true`.
6. `_akh/status`: one snapshot combining goals + active plan + tools + skills +
   projects + usage (Phase D) for a single client round-trip.
7. Register all new methods + groups in `_akh/meta/capabilities` (`mod.rs:528`).

### Phase C — Sidebar poller + goals/capabilities panels (client)

8. Background poller thread (sibling of `acp-remote`): call `_akh/status` every
   ~1.5 s, send `ChatUpdate::Snapshot(StatusSnapshot)` (new variant) over the
   existing inbound channel — UI thread never blocks.
9. Typed client view structs (`GoalView`, `PlanView`, `ToolView`, `SkillView`,
   `UsageView`) deserialized from the snapshot JSON.
10. Render the Goals panel (status-colored: ● active / ○ pending / ✓ done /
    ✗ failed), active plan steps, and the Capabilities panel (tool + skill
    counts, daemon health). Un-stub `/goals`, `/status` to focus the panel.

### Phase D — Context/usage meter (LLM instrumentation)

11. `OllamaClient::chat` captures `prompt_eval_count`/`eval_count` →
    `AssistantTurn`; per-session `UsageLedger` (prompt/completion/total, turns)
    accumulated in `run_agentic_turn`.
12. `context_window` (model metadata + config default) for `% used`; optional
    `cost_per_1k` (config) for the `$` estimate.
13. Serve `_akh/usage`, fold into `_akh/status`, and push a live per-turn usage
    delta over `session/update` so the bar moves during a turn.

### Phase E (stretch) — Cognitive panels beyond OpenCode

Reasoning-trace view, confidence/evidence on facts, KG mini-view, active-
microtheory context. Deferred.

## Verification

- Per phase: `cargo test --features "acp server"`; new unit tests for the
  `_akh/status` handlers (incl. the `try_lock`-busy path returning `stale`) and
  for `StatusSnapshot` (de)serialization round-trips.
- Manual: run the TUI against a live daemon; confirm sidebar toggles, panels
  populate within one poll interval, the usage bar advances during a turn, and a
  long in-flight turn never blocks a sidebar refresh.

## Risks / open questions

- **Lock contention** under autonomous OODA cycles — mitigated by `try_lock` +
  `stale` (ADR-063 §2).
- **Cost estimate accuracy** — model pricing is config-supplied; cloud
  (`ollama.com`) vs local differ; show `$` only when configured.
- **Remote-fallback usage** — OpenAI-compatible path reports no tokens yet; meter
  reads 0 for that backend until wired (consistent with ADR-062's tool-calling
  gap).
- **Palette discoverability vs. the existing `/`-commands** — keep both; palette
  is additive.

## Follow-ups (not in this plan)

- Phase E cognitive panels.
- Per-turn streamed tool-progress in the sidebar (depends on the mid-turn
  `session/update` sink follow-up noted in the LLM-agentic-chat plan).
- Mouse interaction with sidebar panels (click a goal to expand).
