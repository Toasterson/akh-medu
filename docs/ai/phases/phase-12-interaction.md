# Phase 12 — Interaction: Communication Protocols and Social Reasoning

Status: **Complete**

Chat-as-operator-protocol with capability-secured channel abstraction
(Goblins/OCapN-inspired). Grounded dialogue backed by KG + provenance.
Pre-communication constraint checking (Winter-inspired, full inference stack).
Social knowledge graph with per-interlocutor theory of mind via microtheories.
Federation via oxifed (akh as app inside oxifed, AMQP + REST). Transparent reasoning with provenance-to-prose
explanations. Multi-agent communication with capability tokens.
7 sub-phases (12a-12g). Gives the agent an interaction surface.

- **Implementation plan**: `docs/ai/plans/2026-02-17-phase12-interaction.md`

## Phase 12a — Communication channel abstraction

- [x] `CommChannel` trait (Send, not Sync): channel_id, channel_kind, capabilities, try_receive, send, is_connected
- [x] `ChannelKind` enum: Operator (singleton), Trusted, Social, Public — with Display, Serialize/Deserialize
- [x] `ChannelCapabilities` — 10 boolean flags + rate limit, factory methods per kind, `require()` gating
- [x] `ChannelRegistry` — HashMap-based, exactly-one-operator invariant, drain_all
- [x] `OperatorChannel` — wraps MessageSink + InboundHandle (Arc<Mutex<VecDeque>>)
- [x] Protocol messages: InboundMessage (InterlocutorId, MessageContent::Text/Command), OutboundMessage (ResponseContent::Messages, provenance, confidence, ConstraintCheckStatus)
- [x] `Agent::setup_operator_channel()` -> InboundHandle; `Agent::drain_inbound()`
- [x] TUI wired: operator_handle in ChatBackend::Local, process_inbound_local drains channel
- [x] Headless chat wired: push_text -> drain_inbound -> classify
- [x] `AkhMessage::into_outbound()` bridge method
- [x] `AgentError::Channel` transparent variant
- [x] 35 unit tests across 3 new modules

## Phase 12b — Grounded operator dialogue

- [x] `ConversationState` — bounded turn ring buffer (VecDeque), active referents, active topic, grammar, ResponseDetail
- [x] `ResponseDetail` enum: Concise / Normal / Full with `from_str_loose` parser
- [x] `GroundedResponse` — prose + GroundedTriple entries + confidence + provenance IDs + grammar, `render(detail)` method
- [x] `ground_query(subject, engine, grammar)` pipeline: resolve -> collect triples -> filter metadata -> provenance -> synthesize -> GroundedResponse
- [x] `ResponseContent::Grounded` variant on OutboundMessage, `OutboundMessage::grounded()` constructor
- [x] `UserIntent::SetDetail` variant in NLP + channel message classifier
- [x] `Agent::conversation_state()` / `set_response_detail()` API
- [x] TUI grounded-first query path + SetDetail handler
- [x] Headless grounded-first query path + SetDetail handler
- [x] 17 new unit tests

## Phase 12c — Pre-communication constraint checking

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

## Phase 12d — Social KG with theory of mind

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

## Phase 12e — Federation via oxifed

- [x] `OxifedChannel` implementing `CommChannel` with `ChannelKind::Social` — AMQP consumer/publisher via background tokio tasks
- [x] `OxifedConfig` — AMQP URL, admin API URL, domain, actor username, API token, custom inbox queue
- [x] Serde-compatible oxifed message types: `OxifedMessage` enum, `NoteCreate`, `NoteUpdate`, `NoteDelete`, `ProfileCreate`, `ProfileUpdate`, `FollowActivity`, `LikeActivity`, `AnnounceActivity`, `IncomingObject`, `IncomingActivity`
- [x] AMQP exchange/queue constants matching oxifed (`EXCHANGE_INTERNAL_PUBLISH`, `EXCHANGE_ACTIVITYPUB_PUBLISH`, `EXCHANGE_INCOMING_PROCESS`, `QUEUE_ACTIVITIES`)
- [x] Activity -> InboundMessage bridge: `incoming_object_to_inbound()` (Note/Article content extraction with HTML stripping), `incoming_activity_to_inbound()` (Create/Follow/Like/Announce/Undo mapping)
- [x] OutboundMessage -> NoteCreate bridge: `outbound_to_note()` with prose linearization, constraint-check gating in `send()`
- [x] `OxifedInboundHandle` for test injection
- [x] Feature-gated: `--features oxifed` (depends on `daemon`, adds `lapin`, `deadpool-lapin`, `reqwest`)
- [x] `AgentError::Oxifed` transparent variant (cfg-gated)
- [x] 16 new unit tests (bridge functions, serde round-trip, channel FIFO, handle push/receive)

## Phase 12f — Transparent Reasoning and Explanations

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

## Phase 12g — Multi-Agent Communication

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
