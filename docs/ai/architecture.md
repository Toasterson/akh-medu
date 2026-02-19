# Akh-medu Architecture

> Last updated: 2026-02-19 (Phase 10h — Library learning cycle)

## Overview

Akh-medu is a neuro-symbolic AI engine that runs entirely on CPU with no LLM dependency. It hybridizes:

- **Vector Symbolic Architecture (VSA)** — 10,000-bit binary hypervectors for distributed representation
- **Knowledge Graphs** — dual-indexed (petgraph + oxigraph/SPARQL) for structured symbolic reasoning
- **E-graph Reasoning** — equality saturation via `egg` for symbolic rewriting
- **Autonomous Agent** — OODA-loop agent with 23+ tools, working/episodic memory, planning, reflection
- **Code Generation** — KG-to-Rust pipeline: code_gen tool, RustCodeGrammar, compiler feedback loop, parameterized templates, VSA code pattern encoding, pattern mining from examples, library learning cycle
- **Multilingual Grammar** — GF-inspired abstract/concrete syntax split for 5 languages
- **Content Library** — document ingestion (PDF, EPUB, HTML) with chunking and semantic enrichment
- **Tiered Storage** — hot (DashMap) → warm (mmap) → cold (redb) for scalability

## Module Map

```
src/
├── agent/              28 modules — OODA loop, tools (code_gen, code_ingest, compile_feedback, pattern_mine), memory, goals, planning, psyche, library learning
├── autonomous/          6 modules — background learning, confidence fusion, grounding
├── argumentation/       1 module  — pro/con argumentation (Phase 9e): meta-rules, verdicts, evidence chains
├── compartment/         5 modules — knowledge isolation, Jungian psyche, microtheories (Phase 9a, per-repo code scoping), CWA/circumscription (Phase 9m)
├── dispatch/            1 module  — competitive reasoner dispatch (Phase 9f): Reasoner trait, bid-based registry, 7 built-in reasoners
├── grammar/            22 modules — GF-inspired parsing/generation, entity resolution, Rust code gen (Phase 10a), templates (Phase 10e)
├── graph/               9 modules — KG (petgraph), SPARQL (oxigraph), analytics, predicate hierarchy (Phase 9b), defeasible reasoning (Phase 9d), arity constraints (Phase 9j), contradiction detection (Phase 9l), argumentation truth (Phase 9i), NARTs (Phase 9o)
├── infer/               3 modules — spreading activation, backward chaining, superposition
├── library/            12 modules — document parsing, chunking, concept extraction
├── reason/              3 modules — e-graph language (AkhLang), rewrite rules, second-order quantification (Phase 9n), anti-unification (Phase 10h)
├── simd/                5 modules — runtime SIMD kernel dispatch (AVX2 / generic)
├── skills/              1 module  — skillpack lifecycle (Cold/Warm/Hot)
├── store/               3 modules — tiered storage (hot/warm/cold)
├── tui/                 6 modules — ratatui terminal UI, WebSocket remote
├── vsa/                 5 modules — HyperVec, VsaOps, encoding, item memory (HNSW), code pattern encoding (Phase 10f)
├── engine.rs                      — facade composing all subsystems
├── error.rs                       — miette + thiserror rich diagnostics
├── rule_macro.rs                  — rule macro predicates (Phase 9g): RuleMacro trait, registry, genls/relationAllExists/relationExistsAll
├── temporal.rs                    — temporal projection (Phase 9k): TemporalProfile, decay computation, registry
├── provenance.rs                  — persistent explanation ledger (redb, multi-index, 40 derivation kinds)
├── skolem.rs                      — Skolem functions (Phase 9h): existential witnesses, grounding, auto-ground
├── tms.rs                         — truth maintenance system (Phase 9c): support sets, retraction cascades
├── symbol.rs                      — SymbolId (NonZeroU64), SymbolKind, allocator
├── pipeline.rs                    — composable stage pipelines
└── main.rs                        — CLI (clap) with 50+ subcommands
```

## Core Data Model

### Symbols
- **SymbolId**: `NonZeroU64` (niche-optimized for `Option` packing)
- **SymbolKind**: `Entity`, `Relation`, `Composite`, `Glyph(codepoint)`
- **AtomicSymbolAllocator**: thread-safe monotonic ID generator

### Triples
- `(subject: SymbolId, predicate: SymbolId, object: SymbolId)` with confidence, timestamp, provenance_id, compartment_id
- Stored in both petgraph (in-memory graph ops) and oxigraph (SPARQL queries)
- Each triple carries provenance linking back to how it was derived

### Hypervectors (VSA)
- 10,000-bit binary vectors, SIMD-accelerated (AVX2 with generic fallback)
- Operations: bind (XOR), unbind, bundle (majority vote), permute, similarity (Hamming)
- Item Memory: HNSW approximate nearest-neighbor search

## Reasoning Systems

| System | Strategy | Use Case |
|--------|----------|----------|
| Spreading Activation | Seeds → expand frontier via graph edges + VSA recovery | Forward inference, "what follows from X?" |
| Backward Chaining | Goal ← find supporting evidence recursively | Why-questions, evidence chains |
| Superposition | Parallel competing hypotheses, constructive/destructive interference | Multi-path exploration |
| E-graph Rewriting | `AkhLang` + `egg` equality saturation | Symbolic simplification, equivalence |
| Confidence Fusion | Noisy-OR and consensus across multi-source evidence | Combining evidence |

### Phase 9 — Cyc-Inspired HOL Enhancements

| System | Status | Description |
|--------|--------|-------------|
| **Microtheories** (9a) | Complete | `ist` operator, `genlMt` inheritance, lifting rules, 6 `ctx:` predicates, context-scoped queries, ancestry cache |
| **Predicate Hierarchy** (9b) | Complete | `genlPreds` subsumption, `genlInverse`, hierarchy-aware queries, transitive closure cache, 2 `rel:` predicates |
| **Truth Maintenance System** (9c) | Complete | Support sets with alternative justifications, BFS retraction cascade, re-evaluation, `remove_triple()` |
| **Defeasible Reasoning** (9d) | Complete | 5 override reasons (Monotonic, Specificity, Exception, Recency, Confidence), `DefeasiblePredicates` (4 well-known), BFS specificity, conflict resolution |
| **Pro/Con Argumentation** (9e) | Complete | 6 meta-rules (monotonicity, specificity, recency, depth, source quality, constructiveness), `Argument`/`ArgumentSet`/`Verdict` types, pro/con collection, weighted scoring, decisive rule detection |
| **Reasoner Dispatch** (9f) | Complete | `Reasoner` trait with bid-based dispatch, 7 built-in reasoners (spreading-activation, backward-chaining, superposition, egraph, transitive-closure, type-hierarchy, predicate-hierarchy), fallback on failure |
| **Rule Macro Predicates** (9g) | Complete | `RuleMacro` trait, `RuleMacroRegistry`, 3 built-in macros (Genls, RelationAllExists, RelationExistsAll), virtual expansion + query answering, 4 `macro:` predicates |
| **Arity & Type Constraints** (9j) | Complete | `ConstraintRegistry` with per-relation arity/arg-type declarations, `is-a` chain type checking (BFS), 3 `onto:` predicates, opt-in enforcement with diagnostic errors |
| **Temporal Projection** (9k) | Complete | `TemporalProfile` (Stable, Decaying, Ephemeral, Periodic), `TemporalRegistry` with default profiles, `apply_temporal_decay()`, filter-by-time, 1 `temporal:` predicate |
| **Contradiction Detection** (9l) | Complete | 4 contradiction kinds (functional, disjointness, temporal, intra-microtheory), `FunctionalPredicates`/`DisjointnessConstraints`, `check_contradictions()`, 2 `onto:` predicates |
| **Skolem Functions** (9h) | Complete | `SkolemSymbol`, `SkolemRegistry` with deduplication, create/ground/unground/auto_ground, check_grounding from KG, existential witness lifecycle |
| **Argumentation-Based Truth** (9i) | Complete | `ArgumentationCache` with verdict caching, `query_with_argumentation()`, cache invalidation by symbol or (subject, predicate), custom meta-rule queries |
| **Circumscription / CWA** (9m) | Complete | `ContextAssumptions` (CWA, UNA, circumscription), `AssumptionRegistry`, negation-as-failure queries, circumscribed instance enumeration, UNA entity identity, 3 `ctx:` predicates |
| **Second-Order Quantification** (9n) | Complete | `SecondOrderRule` with `RelationProperty` constraints, `SecondOrderRegistry` with 3 built-in rules (transitivity, symmetry, reflexivity), qualifying predicate enumeration, rule instantiation |
| **NARTs** (9o) | Complete | `NartDef` (function + args), `NartRegistry` with structural deduplication, structural unification with wildcards, function/arg lookup |

## Agent Architecture

OODA loop (synchronous, no async runtime):
1. **Observe** — scan KG for active goals, recall episodic memories
2. **Orient** — assess working memory, build context
3. **Decide** — utility-based tool scoring with recency penalty, novelty bonus, episodic hints
4. **Act** — execute selected tool, evaluate goal progress

Supporting infrastructure: working memory (ephemeral), episodic memory (consolidated), goal management with decomposition, multi-step planning with backtracking, periodic reflection, Jungian psyche model.

## Storage Architecture

```
Hot  (DashMap)     — sub-microsecond, volatile
Warm (mmap)        — memory-mapped files, persistent, fast reads
Cold (redb)        — ACID transactions, durable, slower writes
Provenance (redb)  — multi-index ledger (derived/source/kind)
SPARQL (oxigraph)  — persistent RDF store for structured queries
```

## Provenance

Every inference, agent decision, and knowledge derivation creates a `ProvenanceRecord`:
- Derived symbol, derivation kind (29 variants), confidence, depth, source symbols, metadata
- Full traceback from any result to its original sources
- Indices by derived symbol, source symbol, and kind for fast lookup

## Development Phases

Phases 1–7: Engine foundation (VSA, KG, reasoning, storage, provenance, inference, pipeline, skills)
Phase 8a–8f: Agent evolution (wiring, goals, decision-making, persistence, external tools, planning)
Phase 9a–9o: Cyc-inspired HOL enhancements (15 sub-phases):
- **High**: 9a microtheories, 9b predicate hierarchy, 9c TMS, 9d defeasibility, 9e argumentation, 9f reasoner dispatch
- **Medium**: 9g rule macros, 9h skolem functions, 9i arg-based truth, 9j arity/types, 9k temporal projection, 9l contradiction detection
- **Lower**: 9m circumscription/CWA, 9n second-order quantification, 9o NARTs
Phase 10a–10h: Rust code generation (8 sub-phases):
- **Core (Wave 1 complete)**: 10a RustCodeGrammar, 10b code_gen tool, 10c code-aware planning, 10d iterative refinement
- **Pattern infrastructure (Wave 2 complete)**: 10e parameterized templates (7 built-in), 10f VSA code pattern encoding (path-contexts, multi-granularity)
- **Infra**: code_ingest per-repo microtheory scoping (mt:repo:<name> specializes mt:rust-code, ContextDomain::Code, clean re-ingestion)
- **Pattern learning**: 10g pattern mining from examples, 10h library learning cycle
Phase 11a–11h: Autonomous task system (8 sub-phases):
- **Core**: 11a drive-based goal generation (CLARION/GDA/Soar), 11b HTN decomposition with dependency DAGs, 11c value-based argumentation priority (Dung/VAF)
- **Structure**: 11d projects as microtheories with Soar/ACT-R memory, 11e GDA expectation monitoring
- **Meta**: 11f metacognitive self-evaluation (Nelson-Narens/AGM), 11g VOC resource reasoning with CBR effort estimation, 11h procedural learning (Soar chunking)
Phase 12a–12g: Interaction — communication protocols and social reasoning (7 sub-phases):
- **Core**: 12a channel abstraction (capability-secured, Chat = operator protocol), 12b grounded operator dialogue, 12c pre-communication constraint checking
- **Social**: 12d social KG with theory of mind (microtheories), 12e ActivityPub federation via oxifed, 12f transparent reasoning / explanations, 12g multi-agent communication
Phase 13a–13i: Personal assistant (9 sub-phases):
- **Email**: 13a email channel (JMAP/IMAP + MIME + JWZ threading), 13b OnlineHD spam classification (VSA-native), 13c email triage & priority (sender reputation + HEY-style screening), 13d structured extraction (dates, events, action items → KG)
- **PIM**: 13e personal task & project management (GTD + Eisenhower + PARA), 13f calendar & temporal reasoning (RFC 5545, Allen interval algebra)
- **Intelligence**: 13g preference learning & proactive assistance (HyperRec-style VSA profiles, serendipity engine), 13h structured output & operator dashboards (JSON-LD, briefings, notifications)
- **Delegation**: 13i delegated agent spawning (scoped knowledge, own identity, email composition pipeline)
Phase 14a–14i: Purpose-driven bootstrapping with identity (9 sub-phases):
- **Identity**: 14a purpose + identity parser (NL → PurposeModel + IdentityRef, character reference extraction), 14b identity resolution (Wikidata + DBpedia + Wikipedia cascade → 12 Jungian archetypes → OCEAN → Psyche construction, Ritual of Awakening: self-naming via cultural morphemes)
- **Domain**: 14c domain expansion (Wikidata + Wikipedia + ConceptNet, VSA boundary detection), 14d prerequisite discovery + ZPD classification (Vygotsky zones, curriculum generation)
- **Acquisition**: 14e resource discovery (Semantic Scholar + OpenAlex + Open Library, quality scoring), 14f iterative ingestion (curriculum-ordered, NELL-style multi-extractor cross-validation, personality-biased resource selection)
- **Assessment**: 14g competence assessment (Dreyfus model, competency questions, graph completeness, VSA structural analysis)
- **Orchestration**: 14h bootstrap orchestrator (meta-OODA loop, personality shapes exploration style, Dreyfus-adaptive exploration), 14i community recipe sharing (TOML purpose recipes with identity section, ActivityPub federation, skillpack export)
