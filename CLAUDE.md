 use miette error handling pattern and provide rich diagnotics

use strongly typed idiomatic rust. Think to express the function in datatypes
do error handling
try to express options on how to handle issues for calling functions

## Documentation Maintenance

- Keep `docs/ai/architecture.md` updated when making structural changes (new modules, protocol changes, phase transitions). Bump the "last updated" date.
- Create a new timestamped plan in `docs/ai/plans/` before starting a new phase or significant feature.
- Create a new timestamped ADR in `docs/ai/decisions/` when making meaningful technology or design choices. Number sequentially from the last ADR.
- Never delete old plans or decisions. Mark superseded plans with status `Superseded` and link to the replacement.

## Library Usage Patterns

### Error handling (miette + thiserror)
- All error enums derive `#[derive(Debug, Error, Diagnostic)]`
- Each variant has `#[diagnostic(code(...), help(...))]`
- Subsystem errors chain via `#[error(transparent)] #[diagnostic(transparent)]`
- Return `AkhResult<T>` (alias for `Result<T, AkhError>`) from public APIs
- Subsystem functions return their own result type (e.g., `VsaResult<T>`, `StoreResult<T>`)

### HNSW (hnsw_rs + anndists)
- Import distances from `anndists::dist::DistHamming` (NOT `hnsw_rs::dist`)
- Hnsw requires lifetime: `Hnsw<'static, u32, DistHamming>`
- `insert()` takes `(&[T], usize)` tuple and uses `&self` (not `&mut self`)
- `search()` takes `(&[T], k, ef_search)` and returns `Vec<Neighbour>`

### Oxigraph (SPARQL store)
- `Store::new()` for in-memory, `Store::open(path)` for persistent
- Triples use `Quad::new(subject, predicate, object, GraphNameRef::DefaultGraph)`
- SymbolId maps to IRI: `https://akh-medu.dev/sym/{id}`
- Query with `store.query("SELECT ...")` → `QueryResults::Solutions`

### Petgraph (knowledge graph)
- Must `use petgraph::visit::EdgeRef` to access `.source()` and `.target()` on edges
- Use `DiGraph<SymbolId, EdgeData>` with separate `DashMap<SymbolId, NodeIndex>` index

### Egg (e-graph reasoning)
- Define language with `define_language!` macro
- Create rules with `rewrite!` macro
- Run with `Runner::default().with_expr(&expr).run(&rules)`
- Extract best with `Extractor::new(&runner.egraph, AstSize)`

### Redb (durable store)
- Define tables as const: `const TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("name")`
- Use `begin_write()` / `begin_read()` for transactions
- Drop table handles before calling `txn.commit()`
- When checking if remove() found something, extract the bool inside the scope

### Edition 2024
- `unsafe fn` bodies are NOT implicitly unsafe — wrap each unsafe op in `unsafe {}`
- Use `edition = "2024"` in Cargo.toml

### Agent module patterns
- Agent concepts (goals, memories, episodes) are **regular Entity symbols** with well-known relation predicates (prefixed `agent:`) — no new `SymbolKind` variants
- `AgentPredicates` holds 12 well-known relation SymbolIds resolved at agent init
- `AgentError::Engine` wraps `Box<AkhError>` to break the recursive type cycle; manual `From<AkhError>` impl handles boxing
- Memory consolidation stores `DerivationKind::AgentConsolidation` provenance; decisions store `DerivationKind::AgentDecision`
- Tools are compile-time trait impls (`impl Tool for X`) with runtime registration in `ToolRegistry`
- The OODA loop is synchronous — no async runtime

## Agent Roadmap

Phases 1–7 are complete (engine + agent scaffold). The agent has working infrastructure
(memory, goals, tools, consolidation, provenance) but the decision-making core is a
placeholder. The phases below evolve it into a real autonomous agent.

### Phase 8a — Fix wiring bugs (prerequisites for real autonomy) ✓
- [x] Wire up `reference_count` in Decide phase (increment when WM entries are consulted)
- [x] Fix status triple accumulation: restore_goals now picks highest SymbolId (most recent) deterministically
- [x] Connect recalled episodes from Observe to Orient/Decide (full EpisodicEntry data flows through)
- [x] Enable all 5 tools in `select_tool()` — anti-repetition, memory_recall, kg_mutate, synthesize_triple
- [x] Fix Act phase: evaluate_goal_progress() checks criteria keywords against non-metadata tool output
- [x] Fix self-referential criteria match: agent-metadata labels (desc:, status:, criteria:, goal:) filtered out

### Phase 8b — Success criteria & goal autonomy ✓
- [x] Parse and evaluate success criteria against KG state (pattern matching on triples)
- [x] Let Act produce Completed/Failed based on criteria evaluation (two-signal: tool output + KG state)
- [x] Stall detection: track `cycles_worked` and `last_progress_cycle` per goal, `is_stalled()` method
- [x] Integrate goal decomposition into OODA loop (`decompose_stalled_goals()` auto-fires after each cycle)
- [x] Goal decomposition splits on commas/"and", suspends parent, creates active children
- [x] Add `suspend_goal()`, `fail_goal()`, `decompose_stalled_goal()` to Agent public API
- [x] Metadata label filtering: agent-metadata (desc:, status:, criteria:, goal:, episode:, summary:, tag:) excluded from criteria matching

### Phase 8c — Intelligent decision-making ✓
- [x] Replace if/else `select_tool()` with utility-based scoring (`ToolCandidate` with `total_score()`)
- [x] Score each tool by: base_score (state-dependent), recency_penalty, novelty_bonus, episodic_bonus, pressure_bonus
- [x] Add loop detection: `GoalToolHistory` tracks per-goal (tool, count, recency) from WM Decision entries
- [x] Strategy rotation: novelty_bonus (+0.15) for tools never used on this goal; recency_penalty (-0.4/-0.2/-0.1) prevents repetition
- [x] Use recalled episodic memories: `extract_episodic_tool_hints()` parses tool names from episode summaries, applies episodic_bonus (+0.2)
- [x] Score breakdown in reasoning string for full transparency

### Phase 8d — Session persistence & REPL ✓
- [x] Serialize/deserialize WorkingMemory to engine's durable store (bincode via `put_meta`/`get_meta`)
- [x] Add agent REPL mode (interactive loop with user input between cycles): `agent repl`
- [x] Persist cycle_count and restore on agent restart
- [x] CLI session continuity: `agent resume` picks up where it left off
- [x] `Agent::persist_session()` and `Agent::resume()` constructors
- [x] `Agent::has_persisted_session()` static check
- [x] All agent CLI commands (`cycle`, `run`, `repl`) auto-persist session on exit

### Phase 8e — External tools & world interaction ✓
- [x] File I/O tool: read/write files with scratch-dir sandboxing, 4KB read truncation
- [x] HTTP tool: sync GET via ureq with 256KB response limit and configurable timeout
- [x] Shell exec tool: poll-based timeout (default 30s), 64KB output limit, process kill on timeout
- [x] User interaction tool: stdout prompt + stdin readline with EOF/empty handling
- [x] All 9 tools (5 core + 4 external) registered in Agent, wired into OODA utility scoring
- [x] Keyword-based tool selection for file_io, http_fetch, shell_exec, user_interact

### Phase 8f — Planning & reflection ✓
- [x] Multi-step planning: Plan type with ordered PlanSteps, auto-generated per goal before OODA cycle
- [x] Two alternating strategies (explore-first vs reason-first) based on attempt number
- [x] Reflection: after every N cycles (configurable), reviews tool effectiveness and goal progress
- [x] Meta-reasoning: auto-adjusts goal priorities (boost progressing, demote stagnant), suggests decomposition
- [x] Backtracking: on plan step failure, marks plan as failed, generates alternative with incremented attempt
- [x] CLI commands: `agent plan`, `agent reflect`; REPL commands: `p`/`plan`, `r`/`reflect`

## Phase 9 — Cyc-Inspired HOL Enhancements

Motivated by analysis of Lenat & Marcus 2023 and CycL higher-order logic features.
15 sub-phases (9a–9o) covering microtheories, predicate hierarchy, TMS, defeasibility,
argumentation, reasoner dispatch, rule macros, skolem functions, arg-based truth,
arity constraints, temporal projection, contradiction detection, circumscription,
second-order quantification, and NARTs.

- **Gap analysis**: `docs/ai/decisions/001-cyc-paper-analysis.md`
- **Implementation plan**: `docs/ai/plans/2026-02-17-phase9-cyc-inspired.md`

## Phase 10 — Generative Functions (Rust Code Generation)

`RustCodeGrammar` linearizer, `code_gen` agent tool, code-aware planning,
iterative refinement via compiler feedback, and reusable code templates.
5 sub-phases (10a–10e). Builds on existing `AbsTree::CodeSignature/CodeModule`,
`code_ingest`, `doc_gen`, `file_io`, `shell_exec` tools.

- **Implementation plan**: `docs/ai/plans/2026-02-17-phase10-code-generation.md`

## Phase 11 — Autonomous Task System with Self-Goal Setting

Goal generation from observation (gaps, anomalies, opportunities), intelligent
decomposition with dependency tracking, argumentation-backed priority reasoning,
project abstraction for cross-session continuity, world monitoring with reactive
goals, self-evaluation with goal questioning, and resource awareness.
7 sub-phases (11a–11g). Converges all prior phases into true autonomy.

- **Implementation plan**: `docs/ai/plans/2026-02-17-phase11-autonomous-tasks.md`
