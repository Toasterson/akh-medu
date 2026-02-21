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

### Phase 12b — Grounded operator dialogue ✓
- [x] `ConversationState` — bounded turn ring buffer (VecDeque), active referents, active topic, grammar, ResponseDetail
- [x] `ResponseDetail` enum: Concise / Normal / Full with `from_str_loose` parser
- [x] `GroundedResponse` — prose + GroundedTriple entries + confidence + provenance IDs + grammar, `render(detail)` method
- [x] `ground_query(subject, engine, grammar)` pipeline: resolve → collect triples → filter metadata → provenance → synthesize → GroundedResponse
- [x] `ResponseContent::Grounded` variant on OutboundMessage, `OutboundMessage::grounded()` constructor
- [x] `UserIntent::SetDetail` variant in NLP + channel message classifier
- [x] `Agent::conversation_state()` / `set_response_detail()` API
- [x] TUI grounded-first query path + SetDetail handler
- [x] Headless grounded-first query path + SetDetail handler
- [x] 17 new unit tests

### Phase 12c — Pre-communication constraint checking ✓
- [x] `ConstraintChecker` with configurable `ConstraintConfig` and per-channel-kind `ConfidenceThresholds`
- [x] 6-check pipeline: consistency (contradiction detection), confidence (per-kind threshold), rate (CommunicationBudget with sliding window + cooldown), relevance (placeholder), sensitivity (SensitivityLevel + channel kind gating), provenance (ungrounded claim warnings)
- [x] `CheckOutcome` with violations + warnings; `EmissionDecision` enum (Emit/Suppress)
- [x] Per-channel-kind emission behavior: operator=annotate, trusted=suppress violations, social/public=suppress entirely
- [x] `ConstraintCheckStatus` evolved: Unchecked / Passed { warning_count } / Failed { violation_count, warning_count }
- [x] `GroundedTriple` enriched with subject_id/predicate_id/object_id for consistency checking
- [x] `Agent::check_and_wrap_grounded()` — full constraint check + outbound message construction
- [x] TUI + headless wired through constraint pipeline
- [x] `AgentError::ConstraintCheck` transparent variant
- [x] 17 new unit tests

### Phase 12d — Social KG with theory of mind ✓
- [x] `InterlocutorProfile` with symbol_id, channel_ids, trust_level, knowledge_mt, interests, interaction tracking
- [x] `InterlocutorPredicates` — 6 well-known KG relations (has-channel, has-trust-level, has-interest, last-interaction, interaction-count, has-knowledge-mt)
- [x] `InterlocutorRegistry` — HashMap-based profile store with lazy predicate initialization
- [x] Per-interlocutor microtheory creation via `engine.create_context()` (ContextDomain::Belief) for theory of mind
- [x] VSA interest bundling: rebuild interest vectors on `add_interest()`, Hamming-based `find_similar()` and `interest_overlap()`
- [x] Trust level management with operator immutability enforcement
- [x] `record_knowledge()` — compartment-scoped "knows" triples in personal microtheory
- [x] Auto-registration in TUI `process_inbound_local()` and headless chat via `agent.ensure_interlocutor()`
- [x] `AgentError::Interlocutor` transparent variant
- [x] 10 new unit tests

### Phase 12e — Federation via oxifed ✓
- [x] `OxifedChannel` implementing `CommChannel` with `ChannelKind::Social` — AMQP consumer/publisher via background tokio tasks
- [x] `OxifedConfig` — AMQP URL, admin API URL, domain, actor username, API token, custom inbox queue
- [x] Serde-compatible oxifed message types: `OxifedMessage` enum, `NoteCreate`, `NoteUpdate`, `NoteDelete`, `ProfileCreate`, `ProfileUpdate`, `FollowActivity`, `LikeActivity`, `AnnounceActivity`, `IncomingObject`, `IncomingActivity`
- [x] AMQP exchange/queue constants matching oxifed (`EXCHANGE_INTERNAL_PUBLISH`, `EXCHANGE_ACTIVITYPUB_PUBLISH`, `EXCHANGE_INCOMING_PROCESS`, `QUEUE_ACTIVITIES`)
- [x] Activity → InboundMessage bridge: `incoming_object_to_inbound()` (Note/Article content extraction with HTML stripping), `incoming_activity_to_inbound()` (Create/Follow/Like/Announce/Undo mapping)
- [x] OutboundMessage → NoteCreate bridge: `outbound_to_note()` with prose linearization, constraint-check gating in `send()`
- [x] `OxifedInboundHandle` for test injection
- [x] Feature-gated: `--features oxifed` (depends on `daemon`, adds `lapin`, `deadpool-lapin`, `reqwest`)
- [x] `AgentError::Oxifed` transparent variant (cfg-gated)
- [x] 16 new unit tests (bridge functions, serde round-trip, channel FIFO, handle push/receive)

### Phase 12f — Transparent Reasoning and Explanations ✓
- [x] `ExplanationQuery` enum: Why, How, WhatKnown, HowConfident, WhatChanged — with `parse()` for natural language recognition
- [x] `DerivationNode` tree built by recursive provenance walk with cycle detection and max_depth
- [x] `render_derivation_tree()` — indented hierarchy with box-drawing connectors
- [x] `render_derivation_prose()` — concise comma-separated prose format
- [x] `derivation_kind_prose()` — human-readable strings for all 48 DerivationKind variants
- [x] `explain_entity()` — derivation tree + known facts with provenance tags
- [x] `explain_known()` — enumerate all non-metadata triples with confidence and provenance
- [x] `explain_confidence()` — aggregate confidence, range, assessment, evidence source breakdown
- [x] `explain_changes()` — KG diff since timestamp (filters metadata)
- [x] `execute_query()` — dispatch ExplanationQuery to appropriate explain function
- [x] `UserIntent::Explain` variant in NLP classifier (checked before Query to intercept "why"/"explain")
- [x] Wired into TUI `process_input_local()` and headless chat in `main.rs`
- [x] `AgentError::Explain` transparent variant
- [x] 18 new unit tests (parsing, rendering, derivation prose, helpers)

### Phase 12g — Multi-Agent Communication ✓
- [x] `CapabilityScope` enum: QueryAll, QueryTopics, AssertTopics, ProposeGoals, Subscribe, ViewProvenance
- [x] `CapabilityToken` struct with scoped permissions, expiry, revocation — `is_valid()`, `permits()`, `revoke()`
- [x] `AgentProtocolMessage` enum: Query, QueryResponse, Assert, ProposeGoal, Subscribe, Unsubscribe, GrantCapability, RevokeCapability, Ack, Error — `requires_token()`, `token_id()`
- [x] `InterlocutorKind` enum: Human, Agent — with Default, Display, serde
- [x] `TokenRegistry` — grant/revoke/get/tokens_for_pair/validate_message with pair indexing
- [x] Trust bootstrap: `initial_trust_for_agent()`, `should_promote_trust()`
- [x] `MessageContent::AgentMessage` variant in channel_message.rs, bypasses NLP classifier
- [x] `UserIntent::AgentProtocol` variant in nlp.rs
- [x] `InterlocutorProfile.kind: InterlocutorKind` field, `is_agent()` method
- [x] `can_propose_goals` capability flag on `ChannelCapabilities` (Operator/Trusted: true, Social/Public: false)
- [x] `Agent.token_registry` field with `token_registry()` / `token_registry_mut()` accessors
- [x] `AgentError::MultiAgent` transparent variant
- [x] Wired into TUI `process_input_local()` and headless chat in `main.rs`
- [x] 22 new unit tests (tokens, registry, protocol messages, interlocutor kind, trust bootstrap)

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

### Phase 13a — Email Channel (JMAP/IMAP + MIME) ✓
- [x] `EmailError` miette diagnostic enum (7 variants: Connection, Authentication, Parse, Send, Threading, Config, Engine) with `EmailResult<T>`
- [x] `EmailConnector` trait: `fetch_new()`, `fetch_by_id()`, `send_email()`, `sync_state()` — RawEmail, EmailConfig, EmailCredentials
- [x] `JmapConnector` — JMAP over ureq (sync HTTP), session discovery, delta sync via Email/changes
- [x] `ImapConnector` — sync IMAP via `imap` + `native-tls`, TLS, UID-based delta sync
- [x] `MockConnector` — in-memory queue for testing
- [x] `ParsedEmail` (15 fields) with `parse_raw()` via `mail-parser` — multipart/alternative, multipart/mixed, nested MIME
- [x] JWZ threading (RFC 5256): `ThreadNode`, `ThreadTree`, `build_threads()` — 5-step algorithm with cycle protection
- [x] `ComposedEmail` with `compose_reply()`, `compose_new()`, `to_mime()` via lettre
- [x] `EmailPredicates` — 14 well-known relation SymbolIds (email: namespace)
- [x] `EmailInboundHandle` — Arc<Mutex<VecDeque<InboundMessage>>> with `push_email()`
- [x] `EmailChannel` implementing `CommChannel` — ChannelKind::Social, background std::thread polling
- [x] `DerivationKind::EmailIngested` (tag 48) and `DerivationKind::EmailThreaded` (tag 49)
- [x] Feature-gated: `--features email` (mail-parser, imap, native-tls, lettre)
- [x] `AgentError::Email` transparent variant (cfg-gated)
- [x] 62 new unit tests across 6 modules

### Phase 13b — OnlineHD Spam & Relevance Classification ✓
- [x] `SpamDecision` enum: Spam, Ham, Uncertain — with Display, Serialize/Deserialize
- [x] `ClassificationResult` — decision + vsa_spam_similarity + vsa_ham_similarity + bayesian_score + confidence + rule_override + reasoning
- [x] `SpamRoleVectors` — 7 deterministic role HyperVecs via `encode_token(ops, "email-role:X")`
- [x] `TokenProbabilityTable` — per-token spam/ham counts, Robinson chi-square combination, MAX_TOKEN_TABLE_SIZE eviction
- [x] `SpamClassifier` — OnlineHD prototype vectors + Bayesian supplement + whitelist/blacklist + persistence
- [x] `encode_email()` — 6-feature role-filler binding → bundle
- [x] `classify()` — deterministic rules → VSA similarity → Robinson chi-square → combined score (0.7/0.3) → threshold
- [x] `train()` — OnlineHD adaptive update via majority-vote bundling + token table training
- [x] Whitelist/blacklist domain management (case-insensitive, dedup)
- [x] `persist()`/`restore()` via bincode + `put_meta`/`get_meta`
- [x] `record_classification_provenance()` — `DerivationKind::SpamClassification` (tag 50)
- [x] 24 new unit tests

### Phase 13c — Email Triage & Priority ✓
- [x] `EmailRoute` enum: Important, Feed, PaperTrail, ScreeningQueue, Spam — with Display, Serialize/Deserialize
- [x] `SenderRelationship` enum: Colleague, Friend, Service, Newsletter, Unknown — with Display, Serialize/Deserialize, weight()
- [x] `SenderStats` — per-sender reputation: address, message_count, reply_count, reply_rate (EMA), avg_reply_time_secs (EMA), relationship, routing, symbol_id, needs_screening(), record_message(), record_reply()
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
- [x] `ExtractionScope` — compartment-scoped microtheories (account + correspondent)
- [x] `ActionItemGoalSpec` — goal specification from action items (caller decides whether to create goals)
- [x] 10 regex patterns (LazyLock): ISO/US/EU/written dates, UPS/FedEx/USPS tracking, URLs, phones, emails
- [x] FedEx false-positive mitigation: context keyword gating within 100-char window
- [x] `extract_temporal_via_grammar()` — multi-language relative dates (EN/RU/FR/ES/AR) + "in N days/weeks"
- [x] `extract_actions_via_grammar()` — multi-language action items (EN/RU/FR/ES/AR) with urgency boost
- [x] `extract_all()` — full pipeline: regex + grammar on subject + body, deduplicate by (kind, normalized)
- [x] `ensure_extraction_scope()` — create account + correspondent microtheories via `engine.create_context()`
- [x] `store_extractions()` — compartment-scoped KG triples + carrier triples for tracking numbers
- [x] `record_extraction_provenance()` — `DerivationKind::EmailExtracted` (tag 52)
- [x] `action_items_to_goals()` — goal specs with multi-language urgency detection
- [x] Quick predicates: `has_action_items()`, `has_calendar_event()`, `has_shipment_info()`
- [x] ~26 new unit tests

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
