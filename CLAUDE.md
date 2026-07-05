 use miette error handling pattern and provide rich diagnotics

use strongly typed idiomatic rust. Think to express the function in datatypes
do error handling
try to express options on how to handle issues for calling functions

## Documentation Maintenance

**Anima is the primary store for plans and decisions** (project `019d9844-ce74-7993-a9f0-c953b8f04a4e`, "Akh-Medu"). Every plan and ADR also lives in Seshat's corpus so AI agents can search across them. The `docs/ai/plans/` and `docs/ai/decisions/` directories in this repo are **backups** — they exist for offline access and as the canonical paste source when ingesting into Anima/Seshat. Code that references a plan or ADR should cite both: the Anima work-item ID *and* the local file path.

Authoring flow for new plans / ADRs:
1. Open or pick the relevant Anima work item (or create one).
2. Write the plan / ADR markdown.
3. Save it under `docs/ai/plans/<YYYY-MM-DD>-phase-name.md` or `docs/ai/decisions/<NNN>-name.md`.
4. Push to Seshat via `seshat_ingest` (or eventually `anima_import` once that MCP tool lands — Anima WI `019dc4e5-df5a`) so the corpus stays in sync.
5. Update the Anima work item's description to reference the file path; promote `current_phase` to `triaged` or `agent-ready` as appropriate.

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

Phases 1-24 + 14i + 15d + 22a + 25a-25e are complete. Detailed completion checklists live under `docs/ai/phases/`.
Plans live under `docs/ai/plans/`, ADRs under `docs/ai/decisions/`.

### Completed Phases

- **1-7**: Engine + agent scaffold
- **8a-8f**: Agent Autonomy — [phase](docs/ai/phases/phase-8-agent-autonomy.md), [plan](docs/ai/plans/2026-02-17-phase8-agent-autonomy.md)
- **9**: Cyc-Inspired HOL — [plan](docs/ai/plans/2026-02-17-phase9-cyc-inspired.md), [ADR 001](docs/ai/decisions/001-cyc-paper-analysis.md)
- **10**: Generative Functions — [plan](docs/ai/plans/2026-02-17-phase10-code-generation.md), [ADR 002](docs/ai/decisions/002-code-generation-research.md)
- **11 + 11i-11j**: Autonomous Tasks + lifeform extensions — [plan](docs/ai/plans/2026-02-17-phase11-autonomous-tasks.md), [ADR 003](docs/ai/decisions/003-autonomous-tasks-research.md)
- **12a-12g**: Interaction — [phase](docs/ai/phases/phase-12-interaction.md), [plan](docs/ai/plans/2026-02-17-phase12-interaction.md)
- **13a-13g**: Personal Assistant — [phase](docs/ai/phases/phase-13-personal-assistant.md), [plan](docs/ai/plans/2026-02-17-phase13-personal-assistant.md)
- **14a-14m**: Identity, bootstrapping, NLU (all 4 tiers) — [phase](docs/ai/phases/phase-14-identity.md), [bootstrap plan](docs/ai/plans/2026-02-17-phase14-bootstrapping.md), [NLU plan](docs/ai/plans/2026-02-24-phase14-nlu-extension.md), [ADR 022](docs/ai/decisions/022-nlu-architecture.md)
- **15a-15c**: Causal World Model + Event Calculus + Counterfactuals — [phase](docs/ai/phases/phase-15-causal.md), [plan](docs/ai/plans/2026-02-22-phase15-causal-world-model.md), [ADR 032](docs/ai/decisions/032-event-calculus-engine.md), [ADR 033](docs/ai/decisions/033-counterfactual-reasoning.md)
- **16a-16c**: MCTS Planning + TD Values + Plan Monitoring — [plan](docs/ai/plans/2026-02-22-phase16-predictive-planning.md), [ADR 034](docs/ai/decisions/034-mcts-planning.md)
- **17**: Dempster-Shafer Evidence Theory — [plan](docs/ai/plans/2026-02-22-phase17-evidence-theory.md), [ADR 035](docs/ai/decisions/035-evidence-theory.md)
- **18a-18c**: Source Reliability + ACH + Credibility — [plan](docs/ai/plans/2026-02-22-phase18-source-reliability.md), [ADR 036](docs/ai/decisions/036-source-reliability.md)
- **24**: Sensory Grounding (24a vision, 24b audio, 24c structured) — [plan](docs/ai/plans/2026-02-24-phase24-sensory-grounding.md)
- **25a-25e**: People & Contacts — [phase](docs/ai/phases/phase-25-people-contacts.md), [ADR 028](docs/ai/decisions/028-people-contacts-architecture.md)
- **32**: Knowledge Extraction (T5 Knowledge model for bootstrap elicitation, multi-pass, hallucination detection) — [plan](docs/ai/plans/2026-03-26-phase32-llm-knowledge-extraction.md), [ADR 042](docs/ai/decisions/042-llm-knowledge-extraction.md)
- **38a–c, e, f (partial), g**: Relocatable Akh — Postgres-backed truth + WAL + snapshot/wake; `SleepLevel` + HNSW local-disk wake cache; workspace membership + `WorkspaceJoin/Leave`; relocation with `last_seen` + advisory-lock split-brain guard + [runbook](docs/ops/relocate-akh.md) + recovery harness. 38d (AGE) deferred; live `pg_notify` convergence + daemon auto-idle remain. — [plan](docs/ai/plans/2026-04-25-phase38-relocatable-akh.md), [ADR 049](docs/ai/decisions/049-relocatable-akh.md), [ADR 071](docs/ai/decisions/071-akh-cli-command-tree.md), [ADR 072](docs/ai/decisions/072-normalized-akhomed-store.md)
- **41 (COMPLETE 41a–c, e–g; 41d = seam only)**: The Thinking Engine — `src/thinking/`: unified `ThinkingLoop` + `ThoughtState` (HyperVec focus + DS-massed hypotheses + faithful `ThoughtStep` trace) + `ThinkingBudget`; `engine.deliberate(goal, budget)`; goal-shape/weighted/round-robin schedulers; structural `CompositeVerifier` (9l contradictions + 17 belief intervals + 18a source trust + convergence); 41e VOC controller (**two-regime rule**: diminishing-returns halting only in refinement, never during frontier search — first sweep reproduced the illusion-of-thinking failure); 41d Postgres persistence deferred behind `ThoughtPersistence` seam. **Measured scaling law**: multi-hop solvable depth 4→16→32 at budgets 100→300→1000 (steps ≈ 18.7·d+20), flat curves on analogy (one-shot VSA) + factual (fusion saturates) — flat curve = representation diagnostic. Bench: `src/thinking/bench.rs` + `examples/bench_thinking.rs`; real loop replaced the 41g pipeline stub. — [plan](docs/ai/plans/2026-04-27-phase41-thinking-engine.md), [ADR 052](docs/ai/decisions/052-unified-thinking-loop.md), [ADR 085 findings](docs/ai/decisions/085-thinking-engine-findings.md), [41f report](docs/ai/2026-07-05-phase41f-scaling-law.md)
- **Candle↔Ollama reconciliation**: selectable per-akh chat/agentic backing — Ollama (default, cloud-capable) or Candle (in-process GGUF, full grammar-constrained tool calling; device-limited). Forward-ported candle boundary + ROCm/HIP AMD-GPU + Gemma-4 audio. — [ADR 067](docs/ai/decisions/067-candle-conversational-boundary.md), [ADR 068](docs/ai/decisions/068-rocm-candle-gpu-backend.md), [ADR 069](docs/ai/decisions/069-gemma4-audio-native.md), [ADR 070](docs/ai/decisions/070-selectable-chat-backing.md)

### Upcoming Phases

See individual plan docs for details. Phases are ordered by dependency, not priority.

- **Release Alpha**: K8s deployment — [plan](docs/ai/plans/2026-02-23-release-alpha.md)
- **19**: Epistemic Logic & Theory of Mind — [plan](docs/ai/plans/2026-02-22-phase19-epistemic-reasoning.md)
- **20**: Active Inference — [plan](docs/ai/plans/2026-02-22-phase20-active-inference.md)
- **21**: Game-Theoretic Social Reasoning — [plan](docs/ai/plans/2026-02-22-phase21-game-theory.md)
- **22**: Akhipedia — [plan](docs/ai/plans/2026-02-23-phase22-akhipedia.md), [ADR 021](docs/ai/decisions/021-akhipedia-architecture.md)
- **23**: Affective System — [plan](docs/ai/plans/2026-02-24-phase23-affective-system.md)
- **26**: LLM-VSA Deep Integration (Candle migration, Neural→VSA bridge, PolarQuant, NEON SIMD) — [plan](docs/ai/plans/2026-03-26-phase26-llm-vsa-deep-integration.md), [ADR 037](docs/ai/decisions/037-llm-vsa-deep-integration.md)
- **27**: Live KG→LLM Training (Burn framework, LoRA fine-tuning, depends on Phase 26) — preliminary plan in Phase 26 doc
- **28**: `n akh` Semantic Router (dynamic model routing, LinUCB bandit, cost-aware) — [plan](docs/ai/plans/2026-03-26-phase28-semantic-router.md), [ADR 038](docs/ai/decisions/038-semantic-router.md)
- **29**: Capability Exposure (microtheories, skillpacks, psyche, argumentation, reasoning traces, hypotheses, causal explorer, evidence, source trust) — [plan](docs/ai/plans/2026-03-26-phase29-capability-exposure.md), [ADR 039](docs/ai/decisions/039-capability-exposure.md)
- **30**: Procedural Skills (actions.json + plans.json, LLM-assisted authoring, rule refinement, quality metrics) — [plan](docs/ai/plans/2026-03-26-phase30-procedural-skills.md), [ADR 040](docs/ai/decisions/040-procedural-skills.md)
- **31**: Skill Intelligence (auto-discovery, dependency graph, neural embeddings, skill-aware routing, dynamic synthesis, sharing protocol, cross-skill inference, curriculum) — [plan](docs/ai/plans/2026-03-26-phase31-skill-intelligence.md), [ADR 041](docs/ai/decisions/041-skill-intelligence.md)
- **33**: Grounded Verbalization (grammar aggregation, T5/mT5 data-to-text NLG, NeuroLogic constrained decoding, T5 NLU replaces Qwen) — [plan](docs/ai/plans/2026-03-26-phase33-grounded-verbalization.md), [ADR 043](docs/ai/decisions/043-grounded-verbalization.md)
- **34**: Communicative Intelligence (response strategy selection, psyche style transform, archetype framing, conversation trajectory, shadow integration) — [plan](docs/ai/plans/2026-03-26-phase34-communicative-intelligence.md), [ADR 044](docs/ai/decisions/044-communicative-intelligence.md)
- **35**: T5 Training Infrastructure (offline training for NLU + NLG + Knowledge models, Wikidata + ConceptNet + WebNLG data, GGUF export) — [plan](docs/ai/plans/2026-03-27-phase35-t5-training-infrastructure.md), [ADR 045](docs/ai/decisions/045-t5-training-infrastructure.md)
- **36**: Delegation & External Scheduling (DelegationManager, local Claude Code subprocess, remote triggers, peer coordination, cost control, bidirectional MCP bridge) — [plan](docs/ai/plans/2026-03-28-phase36-delegation-external-scheduling.md), [ADR 046](docs/ai/decisions/046-delegation-external-scheduling.md)
- **37**: Seshat's Archive (shared corpus library, pgvector RAG, MCP service, wish→microtheory synthesis, Candle/T5 LLM, multi-akh sharing) — [plan](docs/ai/plans/2026-04-09-phase37-seshat-archive.md), [ADR 047](docs/ai/decisions/047-seshat-archive.md)
- **43**: Code-Understanding KG (Shen/`shend` out-of-process ingester daemon, tree-sitter polyglot extraction retires the `syn` `code_ingest` tool, `codex.*` relational truth + Apache AGE `codex_kg` read-model, live edits stay in the LLM not the KG) — [ADR 057](docs/ai/decisions/057-code-understanding-kg.md)
- **44 (COMPLETE 44a–44j; 44k/44l recorded-deferred)**: Working-Context Assembly Runtime — swappable `ContextManager` seam in `run_agentic_turn` + `TokenCounter` + byte-identical `PassThroughManager`; `BudgetAllocator` (atomic all-or-nothing) + per-turn telemetry; provenance `Retention` + `SafetyEnforced` eviction guard. Three live, per-akh-selectable strategies (`AKH_CONTEXT_STRATEGY`, default `compacting`): **CompactingManager** (seed→KG→recalled→history + tools, budget-degraded; 44e recency/graph-proximity discourse; 44d opt-in KG delta `AKH_KG_DELTA`), **PassiveManager** (44h, lean + background consolidation), **VirtualContextManager** (44i, MemGPT core/recall/archival + `core_memory_append`/`core_memory_replace`/`archival_search` self-edit tools over KG-backed core memory). 44g recall = VSA-expanded + recency/confidence blend + TMS supersede filter (`TruthMaintenanceSystem::was_retracted`). 44j deterministic A/B harness (`src/bench_phase44/`) + [report](docs/ai/2026-06-21-phase44j-strategy-ab.md). 44k (Candle KV) / 44l (Heka retrieval swap) = `context::policy` trait stubs gated on ADR 037 / Phase 42. — [plan](docs/ai/plans/2026-06-18-phase44-context-assembly-runtime.md), [ADR 064](docs/ai/decisions/064-context-runtime-lessons.md), [ADR 065](docs/ai/decisions/065-context-assembly-runtime.md)
- **Postgres test harness + multi-host**: standalone CloudNativePG (`scripts/pg-standalone.sh`) validates the whole Phase-38 workflow E2E; advisory-locked migrations (`akhomed_store::run_migrations_locked`) for concurrent multi-host start; 3-akh starter ([examples/three-akhs](examples/three-akhs/), [runbook](docs/ops/three-akhs-postgres.md)) — Tecton (GLM/ollama-cloud), Thoth (reasoning cloud), Gemma (Gemma-4 12B candle).
