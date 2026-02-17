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

## Phase 9 — Cyc-Inspired Enhancements

Motivated by analysis of Lenat & Marcus 2023 ("From Generative AI to Trustworthy AI") and
deep dive into CycL higher-order logic features.
See `docs/ai/decisions/001-cyc-paper-analysis.md` for the full gap analysis.
See `docs/ai/plans/2026-02-17-phase9-cyc-inspired.md` for the detailed implementation plan.

### Phase 9a — Microtheories (`ist` + `genlMt` + lifting rules)
- [ ] Promote `Compartment` to first-class `Entity` with `ctx:` predicates (`ctx:specializes`, `ctx:assumes`, `ctx:domain`, `ctx:disjoint`)
- [ ] `ist` operator: context-relative truth as first-class assertion
- [ ] Context inheritance via `ctx:specializes` chains (multiple inheritance, transitive)
- [ ] Lifting rules for cross-context entailment propagation
- [ ] Domain assumption factoring: in-context triples implicitly carry `ctx:assumes`
- [ ] `create_context()`, `add_context_assumption()`, `query_in_context()`, `add_lifting_rule()` engine APIs

### Phase 9b — Predicate hierarchy (`genlPreds` + `genlInverse`)
- [ ] `rel:generalizes` predicate subsumption lattice (biologicalMother ⊂ mother ⊂ parent)
- [ ] `rel:inverse` argument-swapped equivalence (parent ↔ child)
- [ ] Predicate-aware triple queries: search up hierarchy, check inverses
- [ ] Cached transitive closure of predicate hierarchy

### Phase 9c — Truth Maintenance System (TMS)
- [ ] Support sets: each triple/inference carries premises that justify it, multiple alternative justifications
- [ ] Retraction cascade: removing a triple auto-retracts unsupported dependents, re-evaluates alternatives
- [ ] Transactional cascades: full cascade succeeds or nothing changes
- [ ] Provenance becomes bidirectional (recording + driving retraction)

### Phase 9d — Defeasible reasoning
- [ ] Specificity-based override via `is-a` chain depth (needs 9b)
- [ ] Exception registration via `defeasible:overrides` / `defeasible:except` / `defeasible:monotonic`
- [ ] Override resolution: monotonic > default; specific > general; confidence as tiebreaker
- [ ] `resolve_conflict()` on KnowledgeGraph

### Phase 9e — Pro/Con argumentation
- [ ] `ArgumentSet` collects pro/con evidence chains from provenance + TMS support chains
- [ ] Meta-rules: monotonicity > specificity > recency > depth > source quality > constructive
- [ ] `Verdict` type with winning answer, confidence, ranked pro/con arguments
- [ ] `engine.argue()` public API
- [ ] Grammar linearization of verdicts into natural language

### Phase 9f — Competitive reasoner dispatch
- [ ] `Reasoner` trait with `can_handle()` → `Bid` and `solve()` with time budget
- [ ] `ReasonerRegistry` with competitive dispatch (cheapest applicable reasoner wins)
- [ ] Wrap existing strategies (spreading, backward, superposition, e-graph) as `Reasoner` impls
- [ ] New specialized reasoners: `TransitiveClosureReasoner`, `TypeHierarchyReasoner`, `PredicateHierarchyReasoner`
- [ ] Resource-bounded reasoning: per-query time limits, interrupt slow reasoners

### Phase 9g — Rule macro predicates
- [ ] `RuleMacro` trait: compact meta-predicates expanding to quantified patterns
- [ ] Built-in macros: `relationAllExists`, `relationExistsAll`, `genls`
- [ ] Each macro registers a specialized reasoner via dispatch (9f)

### Phase 9h — Skolem functions (existential witnesses)
- [ ] Create concrete Skolem symbols for existential quantification
- [ ] Skolem grounding: link to real entity when concrete match found
- [ ] `DerivationKind::SkolemWitness`, `DerivationKind::SkolemGrounding`

### Phase 9i — Argumentation-based truth values
- [ ] Truth emerges from argument balance, not single `f64` confidence
- [ ] `query_with_argumentation()` dynamically computes truth via pro/con weighing
- [ ] TMS retraction triggers re-argumentation for affected symbols

### Phase 9j — Arity and type constraints
- [ ] `onto:arity`, `onto:arg1type`, `onto:arg2type` on relations
- [ ] Optional type checking on `add_triple()` via `is-a` chain (needs 9b)
- [ ] `declare_relation(relation, arity, arg_types)` engine API

### Phase 9k — Temporal projection
- [ ] `TemporalProfile` on relations: `Stable`, `Decaying { half_life }`, `Ephemeral { ttl }`, `Periodic { period }`
- [ ] Temporal confidence decay applied at query time
- [ ] Default profiles for well-known relations (`is-a` → Stable, `located-at` → Ephemeral)

### Phase 9l — Contradiction detection
- [ ] `check_contradiction()` on triple insertion (opt-in)
- [ ] Detect functional violations, disjointness violations, temporal conflicts, intra-microtheory conflicts
- [ ] Report contradictions with `DerivationKind::ContradictionDetected` provenance
- [ ] Caller decides: add anyway, replace, or abort

### Phase 9m — Circumscription and closed world assumption
- [ ] CWA per-microtheory: failure to find = negation (not unknown)
- [ ] Unique names assumption per-microtheory: different SymbolIds = different entities
- [ ] Circumscription per-collection: only known/derivable instances exist

### Phase 9n — Second-order quantification
- [ ] Extend `AkhLang` with predicate variables (`ForAllPred`, `PredVar`)
- [ ] Built-in second-order rules: transitivity, symmetry, reflexivity
- [ ] Auto-generate specialized reasoners from second-order rules via dispatch (9f)

### Phase 9o — Non-atomic reified terms (NARTs)
- [ ] Extend `SymbolKind::Composite` with function + args structure
- [ ] Structural equality and deduplication for NARTs
- [ ] Structural unification in triple queries
- [ ] VSA encoding: bind function vector with permuted arg vectors
