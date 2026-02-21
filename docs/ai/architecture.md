# Akh-medu Architecture

> Last updated: 2026-02-21 (Phase 13d complete — structured email extraction)

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
├── agent/              46 modules — OODA loop, tools (code_gen, code_ingest, compile_feedback, pattern_mine), memory, goals, drives, goal_generation, HTN decomposition, priority reasoning (argumentation), projects (microtheory-backed), planning, psyche, library learning, watch (GDA expectation monitoring), metacognition (Nelson-Narens monitoring/control, ZPD, AGM belief revision), resource awareness (VOC, CBR effort estimation), chunking (procedural learning), channel abstraction (CommChannel trait, ChannelRegistry, OperatorChannel), conversation (grounded dialogue, ConversationState, GroundedResponse), constraint_check (pre-communication constraint pipeline), interlocutor (social KG, InterlocutorRegistry, theory-of-mind microtheories, VSA interest vectors), oxifed (ActivityPub federation via AMQP bridge, feature-gated), explain (provenance-to-prose pipeline, DerivationNode trees, 5 query types), multi_agent (capability tokens, AgentProtocolMessage, TokenRegistry, trust bootstrap)
├── email/              9 modules — email channel (feature-gated): EmailConnector trait (JMAP/IMAP/Mock), MIME parsing (mail-parser), JWZ threading (RFC 5256), email composition (lettre), EmailChannel implementing CommChannel, EmailPredicates (14 well-known relations), OnlineHD spam classifier (VSA + Bayesian + deterministic rules), email triage & priority (sender reputation, four-feature importance scoring, VSA prototypes, HEY-style screening), structured extraction (regex + grammar hybrid, multi-language temporal/action NER, compartment-scoped KG persistence)
├── autonomous/          6 modules — background learning, confidence fusion, grounding
├── argumentation/       1 module  — pro/con argumentation (Phase 9e): meta-rules, verdicts, evidence chains
├── compartment/         5 modules — knowledge isolation, Jungian psyche, microtheories (Phase 9a, per-repo code scoping), CWA/circumscription (Phase 9m)
├── dispatch/            1 module  — competitive reasoner dispatch (Phase 9f): Reasoner trait, bid-based registry, 7 built-in reasoners
├── grammar/            22 modules — GF-inspired parsing/generation, entity resolution, Rust code gen (Phase 10a), templates (Phase 10e)
├── graph/               9 modules — KG (petgraph), SPARQL (oxigraph), analytics, predicate hierarchy (Phase 9b), defeasible reasoning (Phase 9d), arity constraints (Phase 9j), contradiction detection (Phase 9l), argumentation truth (Phase 9i), NARTs (Phase 9o)
├── infer/               3 modules — spreading activation (with Phase 9 hierarchy + temporal context), backward chaining, superposition
├── library/            12 modules — document parsing, chunking, concept extraction
├── reason/              3 modules — e-graph language (AkhLang), rewrite rules, second-order quantification (Phase 9n), anti-unification (Phase 10h)
├── simd/                5 modules — runtime SIMD kernel dispatch (AVX2 / generic)
├── skills/              1 module  — skillpack lifecycle (Cold/Warm/Hot)
├── store/               3 modules — tiered storage (hot/warm/cold)
├── tui/                 6 modules — ratatui terminal UI, WebSocket remote
├── vsa/                 5 modules — HyperVec, VsaOps, encoding, item memory (HNSW), code pattern encoding (Phase 10f)
├── engine.rs                      — facade composing all subsystems (Phase9Config, 9 stored registries, wired add_triple/remove_triple pipeline)
├── error.rs                       — miette + thiserror rich diagnostics
├── rule_macro.rs                  — rule macro predicates (Phase 9g): RuleMacro trait, registry, genls/relationAllExists/relationExistsAll
├── temporal.rs                    — temporal projection (Phase 9k): TemporalProfile, decay computation, registry
├── provenance.rs                  — persistent explanation ledger (redb, multi-index, 53 derivation kinds)
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

### Phase 9 — Production Pipeline (Wired into Engine Lifecycle)

All Phase 9 systems are wired into the engine's `add_triple()` and `remove_triple()` paths, controlled by `Phase9Config` flags (all enabled by default):

**`add_triple()` pipeline** (pre/post hooks):
1. **Constraint check** — `ConstraintRegistry::check_triple()` rejects arity/type violations
2. **Contradiction detection** — `check_contradictions()` with configurable policy (Warn/Reject/Replace)
3. Core KG + SPARQL insertion
4. **Hierarchy invalidation** — `rel:generalizes`/`rel:inverse` triples mark hierarchy dirty for lazy rebuild
5. **Skolem auto-grounding** — `SkolemRegistry::auto_ground()` checks newly satisfiable existentials

**`remove_triple()` with TMS cascade**:
1. Core KG + SPARQL removal
2. **TMS retraction** — `TruthMaintenanceSystem::retract()` cascades to derived symbols
3. Cascaded triples removed from KG + SPARQL

**Inference integration** — `Engine::infer()` builds `InferPhase9Context`:
- Hierarchy-aware spreading activation (specialization + inverse predicates)
- Temporal decay of triple confidences based on `TemporalRegistry` profiles

**Persistence** — 7 registries serialized via bincode to `TieredStore::put_meta()`:
TMS, TemporalRegistry, ConstraintRegistry, FunctionalPredicates, DisjointnessConstraints, SkolemRegistry, NartRegistry. PredicateHierarchy rebuilt from KG on startup.

### Phase 9 — Cyc-Inspired HOL Enhancements (Module Status)

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

Supporting infrastructure: working memory (ephemeral), episodic memory (consolidated), goal management with HTN decomposition (6+ built-in + learned methods, dependency DAGs, VSA-based method selection), multi-step planning with backtracking, periodic reflection, Jungian psyche model, autonomous goal generation (CLARION-inspired drives: curiosity, coherence, completeness, efficiency), metacognitive self-evaluation (Nelson-Narens monitoring/control, ZPD, competence tracking, AGM belief revision, e-graph goal reformulation), resource awareness (VOC-based goal switching, CBR effort estimation, dynamic stall thresholds, diminishing returns detection, opportunity cost recording), procedural learning (Soar-inspired chunking: trace extraction → generalization → method compilation, success/failure tracking, dormancy pruning).

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
- Derived symbol, derivation kind (50 variants), confidence, depth, source symbols, metadata
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
Phase 11a–11h: Autonomous task system (8 sub-phases, all complete):
- **Core (Waves 1–3)**: 11a drive-based goal generation with impasse detection (CLARION/GDA/Soar), 11b HTN decomposition with dependency DAGs (6 built-in methods, VSA method selection, petgraph TaskTree), 11c value-based argumentation priority (Dung/VAF), 11d projects as microtheories with Soar/ACT-R memory, 11e GDA expectation monitoring with pattern-based KG watches, VSA semantic trigger matching, fluent-style state tracking
- **Meta (Wave 4)**: 11f metacognitive self-evaluation (Nelson-Narens monitoring/control, CompetenceModel, HNSW failure patterns, ZPD scoring, autoepistemic goal questioning, AGM belief revision with entrenchment cascade, e-graph goal reformulation)
- **Economic (Wave 5)**: 11g VOC resource reasoning (CBR effort estimation via HNSW EffortIndex, compute_voc, dynamic stall thresholds, diminishing returns detection, marginal-value goal ranking, opportunity cost recording, project budget tracking), 11h procedural learning/chunking (Soar-inspired trace extraction → generalization → method compilation, HNSW MethodIndex, success/failure tracking with retraction, HTN registry integration, dormancy pruning, compilation opportunity detection)
Phase 12a–12g: Interaction — communication protocols and social reasoning (7 sub-phases):
- **Core (12a–12d complete)**: 12a channel abstraction (CommChannel trait, ChannelRegistry with operator invariant, OCapN-inspired ChannelCapabilities per ChannelKind, OperatorChannel wrapping MessageSink with InboundHandle, TUI+headless wired transparently), 12b grounded operator dialogue (ConversationState with bounded turn history, ResponseDetail levels, GroundedResponse with provenance + confidence, ground_query pipeline, grounded-first query path in TUI+headless, SetDetail intent), 12c pre-communication constraint checking (6-check pipeline: consistency/confidence/rate/relevance/sensitivity/provenance, ConstraintChecker with CommunicationBudget, per-channel-kind emission decisions, SensitivityLevel, CheckOutcome, ConstraintCheckStatus evolution), 12d social KG with theory of mind (InterlocutorRegistry, per-interlocutor microtheories via Phase 9a, InterlocutorPredicates with 6 well-known relations, VSA interest bundling, Hamming-based similarity search, trust-level management with operator immutability, auto-registration in TUI+headless)
- **Federation (12e complete)**: 12e ActivityPub federation via oxifed (OxifedChannel implementing CommChannel with ChannelKind::Social, AMQP consumer/publisher background tasks via lapin, serde-compatible oxifed message types, activity↔InboundMessage bridge, constraint-checked outbound Notes, OxifedConfig, feature-gated under `oxifed`)
- **Transparency (12f complete)**: 12f transparent reasoning and explanations (ExplanationQuery with 5 query types: Why/How/WhatKnown/HowConfident/WhatChanged, DerivationNode tree built by recursive provenance walk, render_derivation_tree for indented hierarchy rendering, render_derivation_prose for concise output, derivation_kind_prose covering all 50 DerivationKind variants, explain_entity/explain_known/explain_confidence/explain_changes, ExplanationQuery::parse for NL recognition, Explain UserIntent variant, wired into TUI+headless chat)
- **Multi-Agent (12g complete)**: 12g multi-agent communication with OCapN-inspired capability tokens (CapabilityToken with scoped permissions, expiry, revocation; 6 CapabilityScope variants; TokenRegistry with pair indexing and validation; AgentProtocolMessage with 10 structured message types: Query/QueryResponse/Assert/ProposeGoal/Subscribe/Unsubscribe/GrantCapability/RevokeCapability/Ack/Error; InterlocutorKind Human/Agent on InterlocutorProfile; MessageContent::AgentMessage variant bypassing NLP; UserIntent::AgentProtocol; can_propose_goals capability flag; trust bootstrap via operator introduction)
Phase 13a–13i: Personal assistant (9 sub-phases):
- **Email (13d complete)**: 13a email channel (JMAP/IMAP + MIME + JWZ threading), 13b OnlineHD spam classification (VSA-native), 13c email triage & priority (sender reputation + HEY-style screening), 13d structured extraction (regex + grammar hybrid, multi-language temporal/action NER, compartment-scoped KG)
- **PIM**: 13e personal task & project management (GTD + Eisenhower + PARA), 13f calendar & temporal reasoning (RFC 5545, Allen interval algebra)
- **Intelligence**: 13g preference learning & proactive assistance (HyperRec-style VSA profiles, serendipity engine), 13h structured output & operator dashboards (JSON-LD, briefings, notifications)
- **Delegation**: 13i delegated agent spawning (scoped knowledge, own identity, email composition pipeline)

### Phase 13a — Email Channel (JMAP/IMAP + MIME) ✓
- [x] `EmailError` miette diagnostic enum (7 variants: Connection, Authentication, Parse, Send, Threading, Config, Engine) with `EmailResult<T>` alias
- [x] `EmailConnector` trait: `fetch_new()`, `fetch_by_id()`, `send_email()`, `sync_state()` — with RawEmail, EmailConfig, EmailCredentials
- [x] `JmapConnector` — JMAP over ureq (sync HTTP), session discovery, Email/query + Email/get, delta sync via Email/changes
- [x] `ImapConnector` — sync IMAP via `imap` crate with `native-tls`, TLS connection, UID-based delta sync
- [x] `MockConnector` — in-memory queue for testing (`push_raw()`, `mock_send()`)
- [x] `ParsedEmail` struct (15 fields) with `parse_raw()` via `mail-parser` — multipart/alternative, multipart/mixed, nested MIME, 4KB text / 8KB HTML truncation
- [x] `extract_domain()` utility for email address domain extraction
- [x] JWZ threading (RFC 5256): `ThreadNode`, `ThreadTree`, `build_threads()` — 5-step algorithm with cycle protection, phantom parent nodes
- [x] `ComposedEmail` with `compose_reply()` (In-Reply-To, References chain, Re: prefix, quoted body), `compose_new()`, `to_mime()` via lettre
- [x] `EmailPredicates` — 14 well-known relation SymbolIds (message-id, from, to, cc, subject, date, thread-id, in-reply-to, has-attachment, content-type, body-text, list-id, dkim-pass, spf-pass)
- [x] `EmailInboundHandle` — cloneable `Arc<Mutex<VecDeque<InboundMessage>>>` with `push_email()` converting ParsedEmail → InboundMessage
- [x] `EmailChannel` implementing `CommChannel` — ChannelKind::Social, background std::thread polling, AtomicBool connected/shutdown, Drop cleanup
- [x] `DerivationKind::EmailIngested` (tag 48) and `DerivationKind::EmailThreaded` (tag 49) provenance variants
- [x] Feature-gated: `--features email` (adds `mail-parser`, `imap`, `native-tls`, `lettre`)
- [x] `AgentError::Email` transparent variant (cfg-gated)
- [x] 62 new unit tests across 6 modules

### Phase 13b — OnlineHD Spam & Relevance Classification ✓
- [x] `SpamDecision` enum: Spam, Ham, Uncertain — with Display, Serialize/Deserialize
- [x] `ClassificationResult` — decision + vsa_spam_similarity + vsa_ham_similarity + bayesian_score + confidence + rule_override + reasoning
- [x] `SpamRoleVectors` — 7 deterministic role HyperVecs (sender, domain, subject, body, has_attachments, has_list_id, time_bucket) via `encode_token(ops, "email-role:X")`
- [x] `TokenProbabilityTable` — per-token spam/ham counts, Robinson chi-square combination, MAX_TOKEN_TABLE_SIZE eviction
- [x] `SpamClassifier` — OnlineHD prototype vectors (spam/ham) + Bayesian supplement + whitelist/blacklist + persistence
- [x] `encode_email()` pipeline: 6-feature role-filler binding (domain, subject, body, attachments, list-id, time bucket) → bundle
- [x] `classify()` pipeline: deterministic rules → VSA similarity → Robinson chi-square → combined score (0.7 VSA + 0.3 Bayesian) → threshold
- [x] `train()` — OnlineHD adaptive update via majority-vote bundling + token table training
- [x] Whitelist/blacklist domain management with case-insensitive matching and dedup
- [x] `persist()`/`restore()` via bincode + `put_meta`/`get_meta` on engine's durable store
- [x] `record_classification_provenance()` — `DerivationKind::SpamClassification` (tag 50)
- [x] 24 new unit tests

### Phase 13c — Email Triage & Priority ✓
- [x] `EmailRoute` enum: Important, Feed, PaperTrail, ScreeningQueue, Spam — with Display, Serialize/Deserialize
- [x] `SenderRelationship` enum: Colleague, Friend, Service, Newsletter, Unknown — with Display, Serialize/Deserialize, weight()
- [x] `SenderStats` — per-sender reputation: address, message_count, reply_count, reply_rate (EMA), avg_reply_time_secs (EMA), relationship, routing, symbol_id
- [x] `TriageRoleVectors` — 8 deterministic role HyperVecs via `encode_token(ops, "triage-role:X")`
- [x] `ImportanceWeights` — configurable social/content/thread/label weights (default 0.35/0.25/0.20/0.20)
- [x] `TriagePredicates` — 7 well-known KG relations (sender: namespace)
- [x] `TriageEngine` — sender stats HashMap + OnlineHD important/low-priority prototypes + role vectors + weights
- [x] Four-feature importance scoring: social (reply_rate, frequency, recency, relationship), content (VSA prototype similarity), thread (in_reply_to, references depth), label (operator-assigned route)
- [x] HEY-style screening: `needs_screening()` → ScreeningQueue for first-time unrouted senders
- [x] `encode_email()` — 7-feature triage-oriented role-filler binding → bundle
- [x] `train_important()`/`train_low_priority()` — OnlineHD adaptive update via majority-vote bundling
- [x] `persist()`/`restore()` via bincode + `put_meta`/`get_meta`
- [x] `record_triage_provenance()` — `DerivationKind::EmailTriaged` (tag 51)
- [x] `sync_sender_to_kg()` — KG triple sync for SPARQL queryability
- [x] 26 new unit tests

### Phase 13d — Structured Email Extraction ✓
- [x] `ExtractedItemKind` enum: Date, RelativeDate, TrackingNumber, Url, PhoneNumber, ActionItem, EmailAddress — with Display, Serialize/Deserialize
- [x] `SourceField` enum: Subject, BodyText — with Display, Serialize/Deserialize
- [x] `ExtractedItem` — kind + raw_text + normalized + symbol_id + offset + confidence + source_field + language
- [x] `ExtractionResult` — items + counts + reasoning + detected_language
- [x] `ExtractionPredicates` — 8 well-known KG relations (extract: namespace)
- [x] `ExtractionScope` — account_compartment + correspondent_compartment for microtheory scoping
- [x] `ActionItemGoalSpec` — goal specification from action items (does NOT create goals)
- [x] 10 regex patterns (LazyLock): ISO dates, US/EU dates, written dates, UPS/FedEx/USPS tracking, URLs, phones, emails
- [x] FedEx false-positive mitigation: context keyword gating within 100-char window
- [x] `extract_temporal_via_grammar()` — multi-language relative date extraction (EN/RU/FR/ES/AR) + "in N days/weeks" patterns
- [x] `extract_actions_via_grammar()` — multi-language action item extraction (EN/RU/FR/ES/AR) with urgency boost
- [x] `extract_all()` — full pipeline: regex + grammar on subject + body, deduplication by (kind, normalized)
- [x] `ensure_extraction_scope()` — create account + correspondent microtheories via `engine.create_context()`
- [x] `store_extractions()` — compartment-scoped KG triples + carrier triples for tracking numbers
- [x] `record_extraction_provenance()` — `DerivationKind::EmailExtracted` (tag 52)
- [x] `action_items_to_goals()` — goal specs with multi-language urgency detection
- [x] Quick predicates: `has_action_items()`, `has_calendar_event()`, `has_shipment_info()`
- [x] ~26 new unit tests

Phase 14a–14i: Purpose-driven bootstrapping with identity (9 sub-phases):
- **Identity**: 14a purpose + identity parser (NL → PurposeModel + IdentityRef, character reference extraction), 14b identity resolution (Wikidata + DBpedia + Wikipedia cascade → 12 Jungian archetypes → OCEAN → Psyche construction, Ritual of Awakening: self-naming via cultural morphemes)
- **Domain**: 14c domain expansion (Wikidata + Wikipedia + ConceptNet, VSA boundary detection), 14d prerequisite discovery + ZPD classification (Vygotsky zones, curriculum generation)
- **Acquisition**: 14e resource discovery (Semantic Scholar + OpenAlex + Open Library, quality scoring), 14f iterative ingestion (curriculum-ordered, NELL-style multi-extractor cross-validation, personality-biased resource selection)
- **Assessment**: 14g competence assessment (Dreyfus model, competency questions, graph completeness, VSA structural analysis)
- **Orchestration**: 14h bootstrap orchestrator (meta-OODA loop, personality shapes exploration style, Dreyfus-adaptive exploration), 14i community recipe sharing (TOML purpose recipes with identity section, ActivityPub federation, skillpack export)
