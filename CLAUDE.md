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

Phases 1-7 are complete (engine + agent scaffold). The phases below evolve it into a real autonomous agent.
Detailed completion checklists for each phase live under `docs/ai/phases/`.

### Completed Phases

| Phase | Description | Details |
|-------|-------------|---------|
| 8a-8f | Agent Autonomy (wiring, goals, decisions, persistence, tools, planning) | [phase-8](docs/ai/phases/phase-8-agent-autonomy.md) |
| 9 | Cyc-Inspired HOL Enhancements (microtheories, TMS, defeasibility, argumentation) | [plan](docs/ai/plans/2026-02-17-phase9-cyc-inspired.md), [gap analysis](docs/ai/decisions/001-cyc-paper-analysis.md) |
| 10 | Generative Functions (Rust code generation, pattern learning) | [plan](docs/ai/plans/2026-02-17-phase10-code-generation.md), [research](docs/ai/decisions/002-code-generation-research.md) |
| 11 | Autonomous Task System (drives, HTN, argumentation, metacognition) | [plan](docs/ai/plans/2026-02-17-phase11-autonomous-tasks.md), [research](docs/ai/decisions/003-autonomous-tasks-research.md) |
| 12a-12g | Interaction (channels, grounded dialogue, constraints, social KG, oxifed, explanations, multi-agent) | [phase-12](docs/ai/phases/phase-12-interaction.md), [plan](docs/ai/plans/2026-02-17-phase12-interaction.md) |
| 13a-13g | Personal Assistant (email, spam, triage, extraction, PIM, calendar, preferences) | [phase-13](docs/ai/phases/phase-13-personal-assistant.md), [plan](docs/ai/plans/2026-02-17-phase13-personal-assistant.md) |
| 14a-14d | Identity, domain expansion, prerequisite discovery, ZPD classification | [phase-14](docs/ai/phases/phase-14-identity.md), [plan](docs/ai/plans/2026-02-17-phase14-bootstrapping.md) |
| 15a | Causal World Model (schemas, preconditions, effects, VSA encoding) | [phase-15](docs/ai/phases/phase-15-causal.md), [plan](docs/ai/plans/2026-02-22-phase15-causal-world-model.md) |

### Upcoming Phases

| Phase | Description | Plan | Research |
|-------|-------------|------|----------|
| 14e-14i | Bootstrap orchestrator, resource discovery, curriculum ingestion, community recipes | [plan](docs/ai/plans/2026-02-17-phase14-bootstrapping.md) | [research](docs/ai/decisions/005-bootstrapping-research.md) |
| Release Alpha | Kubernetes-deployed autonomous agent (Docker, Helm, Prometheus, continuous learning) | [plan](docs/ai/plans/2026-02-23-release-alpha.md) | — |
| 15b-15c | Event calculus, counterfactual reasoning, prediction tracking | [plan](docs/ai/plans/2026-02-22-phase15-causal-world-model.md) | [research](docs/ai/decisions/020-predictive-planning-epistemic-research.md) |
| 16 | Predictive Multi-Step Planning (MCTS + TD Learning) | [plan](docs/ai/plans/2026-02-22-phase16-predictive-planning.md) | [research](docs/ai/decisions/020-predictive-planning-epistemic-research.md) |
| 17 | Dempster-Shafer Evidence Theory & Belief Intervals | [plan](docs/ai/plans/2026-02-22-phase17-evidence-theory.md) | [research](docs/ai/decisions/020-predictive-planning-epistemic-research.md) |
| 18 | Source Reliability, ACH & Credibility Assessment | [plan](docs/ai/plans/2026-02-22-phase18-source-reliability.md) | [research](docs/ai/decisions/020-predictive-planning-epistemic-research.md) |
| 19 | Epistemic Logic & Theory of Mind | [plan](docs/ai/plans/2026-02-22-phase19-epistemic-reasoning.md) | [research](docs/ai/decisions/020-predictive-planning-epistemic-research.md) |
| 20 | Active Inference OODA Enhancement | [plan](docs/ai/plans/2026-02-22-phase20-active-inference.md) | [research](docs/ai/decisions/020-predictive-planning-epistemic-research.md) |
| 21 | Game-Theoretic Social Reasoning | [plan](docs/ai/plans/2026-02-22-phase21-game-theoretic-reasoning.md) | [research](docs/ai/decisions/020-predictive-planning-epistemic-research.md) |
| 22 | Akhipedia: Internal Knowledge Wiki | [plan](docs/ai/plans/2026-02-23-phase22-akhipedia.md) | [ADR](docs/ai/decisions/021-akhipedia-architecture.md) |
