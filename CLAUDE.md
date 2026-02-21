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

Core generation: `RustCodeGrammar` linearizer, `code_gen` agent tool, code-aware
planning, iterative refinement via compiler feedback, reusable code templates.
Pattern learning: non-ML code2vec via VSA path-context encoding, frequent AST
pattern mining from examples (blog posts/tutorials), DreamCoder/LILO-inspired
library learning cycle with e-graph anti-unification.
8 sub-phases (10a–10h). Builds on existing `AbsTree::CodeSignature/CodeModule`,
`code_ingest`, `doc_gen`, `file_io`, `shell_exec`, VSA encode/grounding.

- **Implementation plan**: `docs/ai/plans/2026-02-17-phase10-code-generation.md`
- **Research**: `docs/ai/decisions/002-code-generation-research.md`

## Phase 11 — Autonomous Task System with Self-Goal Setting

Drive-based goal generation (curiosity/coherence/completeness/efficiency) with
GDA discrepancy detection, HTN-based intelligent decomposition with dependency
DAGs, value-based argumentation for priority reasoning (Dung/VAF), project
abstraction via microtheories with Soar/ACT-R memory integration, GDA
expectation monitoring for reactive goals, metacognitive self-evaluation
(Nelson-Narens monitoring/control, ZPD, AGM belief revision), VOC-based
resource reasoning with CBR effort estimation, and Soar-inspired procedural
learning (chunking).
8 sub-phases (11a–11h). Converges all prior phases into true autonomy.

- **Implementation plan**: `docs/ai/plans/2026-02-17-phase11-autonomous-tasks.md`
- **Research**: `docs/ai/decisions/003-autonomous-tasks-research.md`

## Phase 12 — Interaction: Communication Protocols and Social Reasoning

Chat-as-operator-protocol with capability-secured channel abstraction
(Goblins/OCapN-inspired). Grounded dialogue backed by KG + provenance.
Pre-communication constraint checking (Winter-inspired, full inference stack).
Social knowledge graph with per-interlocutor theory of mind via microtheories.
Federation via oxifed (akh as app inside oxifed, AMQP + REST). Transparent reasoning with provenance-to-prose
explanations. Multi-agent communication with capability tokens.
7 sub-phases (12a–12g). Gives the agent an interaction surface.

- **Implementation plan**: `docs/ai/plans/2026-02-17-phase12-interaction.md`

### Phase 12a — Communication channel abstraction ✓
- [x] `CommChannel` trait (Send, not Sync): channel_id, channel_kind, capabilities, try_receive, send, is_connected
- [x] `ChannelKind` enum: Operator (singleton), Trusted, Social, Public — with Display, Serialize/Deserialize
- [x] `ChannelCapabilities` — 10 boolean flags + rate limit, factory methods per kind, `require()` gating
- [x] `ChannelRegistry` — HashMap-based, exactly-one-operator invariant, drain_all
- [x] `OperatorChannel` — wraps MessageSink + InboundHandle (Arc<Mutex<VecDeque>>)
- [x] Protocol messages: InboundMessage (InterlocutorId, MessageContent::Text/Command), OutboundMessage (ResponseContent::Messages, provenance, confidence, ConstraintCheckStatus)
- [x] `Agent::setup_operator_channel()` → InboundHandle; `Agent::drain_inbound()`
- [x] TUI wired: operator_handle in ChatBackend::Local, process_inbound_local drains channel
- [x] Headless chat wired: push_text → drain_inbound → classify
- [x] `AkhMessage::into_outbound()` bridge method
- [x] `AgentError::Channel` transparent variant
- [x] 35 unit tests across 3 new modules

## Phase 13 — Personal Assistant

Email as bidirectional CommChannel (JMAP primary, IMAP fallback). OnlineHD VSA-native
spam/ham classification with single-pass incremental learning. Email triage with sender
reputation KG, four-feature importance model, HEY-style screening. Structured extraction
(dates, events, tracking numbers, action items) via rule-based NER + e-graph rules.
Personal task management (GTD + Eisenhower + PARA) with petgraph dependency DAGs.
Calendar integration (RFC 5545 iCalendar, Allen interval algebra, temporal e-graph rules).
HyperRec-style VSA preference profiles with temporal decay, JITIR (Remembrance Agent),
serendipity engine (near-miss HNSW). Structured output via JSON-LD briefings, notifications,
SPARQL endpoints. Delegated agent spawning with scoped knowledge compartments, own
identities, and email composition pipeline via grammar module + constraint checking.
9 sub-phases (13a–13i). Builds on Phase 12 CommChannel and capability model.

- **Implementation plan**: `docs/ai/plans/2026-02-17-phase13-personal-assistant.md`
- **Research**: `docs/ai/decisions/004-personal-assistant-research.md`

## Phase 14 — Purpose-Driven Bootstrapping with Identity

Autonomous domain knowledge acquisition AND identity construction from operator statements
like "You are the Architect of the System based on Ptah" or "Be like Gandalf — a GCC
compiler expert". Purpose + identity parser extracts domain, competence level, seed
concepts, and character reference. Identity resolution via multi-source cascade
(Wikidata SPARQL + DBpedia categories + Wikipedia extraction with Hearst patterns)
resolves cultural references (mythology, fiction, history) into structured personality:
12 Jungian archetypes → OCEAN Big Five → behavioral parameters → Psyche construction
(Persona + Shadow + ArchetypeWeights). The Ritual of Awakening: self-naming via
culture-specific morpheme composition (Egyptian, Greek, Norse, Latin patterns),
provenance-tracked as `DerivationKind::RitualOfAwakening` — the akh's creation myth. Domain expansion, prerequisite
discovery (Vygotsky ZPD), resource discovery (Semantic Scholar + OpenAlex + Open Library),
curriculum-ordered ingestion, and Dreyfus competence assessment — all shaped by the
constructed personality (Creator archetype weights building resources, Sage weights
theoretical depth). Bootstrap orchestrator runs meta-OODA with personality-adaptive
exploration-exploitation. Community purpose recipes (TOML with identity section) shared
via ActivityPub/oxifed. 9 sub-phases (14a-14i). Builds on existing Psyche model in
`compartment/psyche.rs`.

- **Implementation plan**: `docs/ai/plans/2026-02-17-phase14-bootstrapping.md`
- **Research**: `docs/ai/decisions/005-bootstrapping-research.md`, `docs/ai/decisions/006-identity-bootstrapping-research.md`
