# Phase 8 — Agent Autonomy & Core Infrastructure

Status: **Complete**

Phases 1-7 are complete (engine + agent scaffold). Phase 8 evolved the placeholder
decision-making core into real autonomous agent infrastructure.

## Phase 8a — Fix wiring bugs (prerequisites for real autonomy)

- [x] Wire up `reference_count` in Decide phase (increment when WM entries are consulted)
- [x] Fix status triple accumulation: restore_goals now picks highest SymbolId (most recent) deterministically
- [x] Connect recalled episodes from Observe to Orient/Decide (full EpisodicEntry data flows through)
- [x] Enable all 5 tools in `select_tool()` — anti-repetition, memory_recall, kg_mutate, synthesize_triple
- [x] Fix Act phase: evaluate_goal_progress() checks criteria keywords against non-metadata tool output
- [x] Fix self-referential criteria match: agent-metadata labels (desc:, status:, criteria:, goal:) filtered out

## Phase 8b — Success criteria & goal autonomy

- [x] Parse and evaluate success criteria against KG state (pattern matching on triples)
- [x] Let Act produce Completed/Failed based on criteria evaluation (two-signal: tool output + KG state)
- [x] Stall detection: track `cycles_worked` and `last_progress_cycle` per goal, `is_stalled()` method
- [x] Integrate goal decomposition into OODA loop (`decompose_stalled_goals()` auto-fires after each cycle)
- [x] Goal decomposition splits on commas/"and", suspends parent, creates active children
- [x] Add `suspend_goal()`, `fail_goal()`, `decompose_stalled_goal()` to Agent public API
- [x] Metadata label filtering: agent-metadata (desc:, status:, criteria:, goal:, episode:, summary:, tag:) excluded from criteria matching

## Phase 8c — Intelligent decision-making

- [x] Replace if/else `select_tool()` with utility-based scoring (`ToolCandidate` with `total_score()`)
- [x] Score each tool by: base_score (state-dependent), recency_penalty, novelty_bonus, episodic_bonus, pressure_bonus
- [x] Add loop detection: `GoalToolHistory` tracks per-goal (tool, count, recency) from WM Decision entries
- [x] Strategy rotation: novelty_bonus (+0.15) for tools never used on this goal; recency_penalty (-0.4/-0.2/-0.1) prevents repetition
- [x] Use recalled episodic memories: `extract_episodic_tool_hints()` parses tool names from episode summaries, applies episodic_bonus (+0.2)
- [x] Score breakdown in reasoning string for full transparency

## Phase 8d — Session persistence & REPL

- [x] Serialize/deserialize WorkingMemory to engine's durable store (bincode via `put_meta`/`get_meta`)
- [x] Add agent REPL mode (interactive loop with user input between cycles): `agent repl`
- [x] Persist cycle_count and restore on agent restart
- [x] CLI session continuity: `agent resume` picks up where it left off
- [x] `Agent::persist_session()` and `Agent::resume()` constructors
- [x] `Agent::has_persisted_session()` static check
- [x] All agent CLI commands (`cycle`, `run`, `repl`) auto-persist session on exit

## Phase 8e — External tools & world interaction

- [x] File I/O tool: read/write files with scratch-dir sandboxing, 4KB read truncation
- [x] HTTP tool: sync GET via ureq with 256KB response limit and configurable timeout
- [x] Shell exec tool: poll-based timeout (default 30s), 64KB output limit, process kill on timeout
- [x] User interaction tool: stdout prompt + stdin readline with EOF/empty handling
- [x] All 9 tools (5 core + 4 external) registered in Agent, wired into OODA utility scoring
- [x] Keyword-based tool selection for file_io, http_fetch, shell_exec, user_interact

## Phase 8f — Planning & reflection

- [x] Multi-step planning: Plan type with ordered PlanSteps, auto-generated per goal before OODA cycle
- [x] Two alternating strategies (explore-first vs reason-first) based on attempt number
- [x] Reflection: after every N cycles (configurable), reviews tool effectiveness and goal progress
- [x] Meta-reasoning: auto-adjusts goal priorities (boost progressing, demote stagnant), suggests decomposition
- [x] Backtracking: on plan step failure, marks plan as failed, generates alternative with incremented attempt
- [x] CLI commands: `agent plan`, `agent reflect`; REPL commands: `p`/`plan`, `r`/`reflect`
