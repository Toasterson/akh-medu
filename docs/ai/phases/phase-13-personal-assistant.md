# Phase 13 — Personal Assistant

Status: **Complete** (13a-13g)

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
9 sub-phases (13a-13i). Builds on Phase 12 CommChannel and capability model.

- **Implementation plan**: `docs/ai/plans/2026-02-17-phase13-personal-assistant.md`
- **Research**: `docs/ai/decisions/004-personal-assistant-research.md`

## Phase 13a — Email Channel (JMAP/IMAP + MIME)

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

## Phase 13b — OnlineHD Spam & Relevance Classification

- [x] `SpamDecision` enum: Spam, Ham, Uncertain — with Display, Serialize/Deserialize
- [x] `ClassificationResult` — decision + vsa_spam_similarity + vsa_ham_similarity + bayesian_score + confidence + rule_override + reasoning
- [x] `SpamRoleVectors` — 7 deterministic role HyperVecs via `encode_token(ops, "email-role:X")`
- [x] `TokenProbabilityTable` — per-token spam/ham counts, Robinson chi-square combination, MAX_TOKEN_TABLE_SIZE eviction
- [x] `SpamClassifier` — OnlineHD prototype vectors + Bayesian supplement + whitelist/blacklist + persistence
- [x] `encode_email()` — 6-feature role-filler binding -> bundle
- [x] `classify()` — deterministic rules -> VSA similarity -> Robinson chi-square -> combined score (0.7/0.3) -> threshold
- [x] `train()` — OnlineHD adaptive update via majority-vote bundling + token table training
- [x] Whitelist/blacklist domain management (case-insensitive, dedup)
- [x] `persist()`/`restore()` via bincode + `put_meta`/`get_meta`
- [x] `record_classification_provenance()` — `DerivationKind::SpamClassification` (tag 50)
- [x] 24 new unit tests

## Phase 13c — Email Triage & Priority

- [x] `EmailRoute` enum: Important, Feed, PaperTrail, ScreeningQueue, Spam — with Display, Serialize/Deserialize
- [x] `SenderRelationship` enum: Colleague, Friend, Service, Newsletter, Unknown — with Display, Serialize/Deserialize, weight()
- [x] `SenderStats` — per-sender reputation: address, message_count, reply_count, reply_rate (EMA), avg_reply_time_secs (EMA), relationship, routing, symbol_id, needs_screening(), record_message(), record_reply()
- [x] `TriageRoleVectors` — 8 deterministic role HyperVecs via `encode_token(ops, "triage-role:X")`
- [x] `ImportanceWeights` — configurable social/content/thread/label weights (default 0.35/0.25/0.20/0.20)
- [x] `TriagePredicates` — 7 well-known KG relations (sender: namespace)
- [x] `TriageEngine` — sender stats HashMap + OnlineHD important/low-priority prototypes + role vectors + weights
- [x] Four-feature importance scoring: social (reply_rate, frequency, recency, relationship), content (VSA prototype similarity), thread (in_reply_to, references depth), label (operator-assigned route)
- [x] HEY-style screening: `needs_screening()` -> ScreeningQueue for first-time unrouted senders
- [x] `encode_email()` — 7-feature triage-oriented role-filler binding -> bundle
- [x] `train_important()`/`train_low_priority()` — OnlineHD adaptive update via majority-vote bundling
- [x] `persist()`/`restore()` via bincode + `put_meta`/`get_meta`
- [x] `record_triage_provenance()` — `DerivationKind::EmailTriaged` (tag 51)
- [x] `sync_sender_to_kg()` — KG triple sync for SPARQL queryability
- [x] 26 new unit tests

## Phase 13d — Structured Email Extraction

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

## Phase 13e — Personal Task & Project Management

- [x] `PimError` miette diagnostic enum (5 variants: TaskNotFound, InvalidTransition, CycleDetected, RecurrenceParse, Engine) with `PimResult<T>`
- [x] `GtdState` enum (Inbox/Next/Waiting/Someday/Reference/Done) with validated transitions via `can_transition_to()`
- [x] `EisenhowerQuadrant` enum (Do/Schedule/Delegate/Eliminate) with `classify(urgency, importance)` and `priority_bonus()`
- [x] `ParaCategory`, `PimContext`, `EnergyLevel`, `Recurrence` enums — all with as_label/from_label, Display, Serialize
- [x] `PimMetadata` struct (14 fields) — per-goal overlay with GTD state, urgency/importance, quadrant, PARA, contexts, energy, deadline, recurrence
- [x] `PimPredicates` — 14 well-known KG relations in `pim:` namespace
- [x] `PimRoleVectors` — 6 deterministic VSA role vectors for priority encoding
- [x] `PimManager` — HashMap metadata + petgraph DiGraph DAG + predicates + roles; add_task, transition_gtd, update_eisenhower, set_para/context/energy/recurrence, available_tasks, add/remove_dependency, topological_order, critical_path, ready_tasks, process_recurring_completions, overdue_tasks, encode_priority, find_similar_priority
- [x] `SerializableDag` — custom Serialize/Deserialize for petgraph DiGraph
- [x] `GtdReviewResult` + `gtd_weekly_review()` — weekly review analysis
- [x] `action_items_to_pim_tasks()` — bridge from Phase 13d ActionItemGoalSpec
- [x] `persist()`/`restore()` via bincode + `put_meta`/`get_meta`
- [x] `DerivationKind::PimTaskManaged` (tag 53) provenance variant
- [x] `pim_rules()` — 2 e-graph rewrite rules (pim-unblock, pim-deadline-chain)
- [x] `Agent.pim_manager` field with init/resume/persist lifecycle
- [x] `reflect()` extended with pim/projects params, `gtd_review` on ReflectionResult
- [x] `UserIntent::PimCommand` in NLP, wired into TUI + headless
- [x] CLI: `Commands::Pim` with 9 subcommands (Inbox, Next, Review, Project, Add, Transition, Matrix, Deps, Overdue)
- [x] `AgentError::Pim` transparent variant
- [x] ~30 new unit tests

## Phase 13f — Calendar & Temporal Reasoning

- [x] `CalendarError` miette diagnostic enum (5 variants: EventNotFound, Conflict, ParseError, SyncError, Engine) with `CalendarResult<T>`
- [x] `AllenRelation` enum: 13 variants (Before, After, Meets, MetBy, Overlaps, OverlappedBy, During, Contains, Starts, StartedBy, Finishes, FinishedBy, Equals) — pure `compute()`, `inverse()`, `is_overlapping()`, `as_label()`/`from_label()`
- [x] `CalendarEvent` struct (symbol_id, summary, dtstart/dtend, location, description, recurrence, ical_uid, confirmed) — `duration_secs()`, `overlaps()`
- [x] `CalendarPredicates` — 13 Allen predicates (`time:` namespace) + 6 calendar metadata (`cal:` namespace) + `allen_predicate()` mapper
- [x] `CalendarRoleVectors` — 4 deterministic VSA role vectors (day_of_week, time_of_day, activity_type, duration)
- [x] `CalendarManager` — HashMap event store; add_event, remove_event, get_event, events_in_range, today_events, week_events, detect_conflicts (sweep-line), compute_allen_relations, encode_temporal_pattern, sync_to_kg, persist/restore
- [x] `import_ical()` — RFC 5545 parsing via `icalendar` crate, dedup by ical_uid (`#[cfg(feature = "calendar")]`)
- [x] `sync_caldav()` — HTTP GET + Basic auth + iCal parse (`#[cfg(feature = "calendar")]`)
- [x] `DerivationKind::CalendarEventManaged` (tag 54) provenance variant
- [x] `calendar_rules()` — 2 e-graph rewrite rules (before-trans, cal-conflict)
- [x] `Agent.calendar_manager` field with init/resume/persist lifecycle
- [x] `UserIntent::CalCommand` in NLP, wired into TUI + headless
- [x] CLI: `Commands::Cal` with 6 subcommands (Today, Week, Conflicts, Add, Import, Sync)
- [x] `AgentError::Calendar` transparent variant
- [x] Feature: `calendar = ["icalendar", "chrono"]`
- [x] 32 new unit tests

## Phase 13g — Preference Learning & Proactive Assistance

- [x] `PreferenceError` miette diagnostic enum (4 variants: ProfileNotFound, EmptyContext, SerendipityFailed, Engine) with `PreferenceResult<T>`
- [x] `ProactivityLevel` enum: Ambient, Nudge, Offer, Scheduled, Autonomous — with as_label/from_label, Display, Default(Ambient)
- [x] `FeedbackSignal` enum: Replied, ReadTime, ArchivedUnread, Starred, GoalCompleted, ExplicitPreference, SuggestionDismissed — with entity(), strength()
- [x] `PreferencePredicates` — 8 well-known KG relations in `pref:` namespace
- [x] `PreferenceRoleVectors` — 6 deterministic VSA role vectors (topic, interaction_type, recency, frequency, source_channel, entity_kind)
- [x] `PreferenceProfile` — interest_prototype (Option<HyperVec>), decay_rate, interaction_count, interaction_history, proactivity_level, suggestions stats
- [x] `Suggestion`, `JitirResult`, `PreferenceReview` types
- [x] `PreferenceManager` — record_feedback (OnlineHD adaptive update), encode_interaction (6-feature role-filler), apply_temporal_decay (exponential), jitir_query (direct+serendipity+KG multi-hop), interest_similarity, top_interests, review
- [x] JITIR: encode context from goals+WM -> direct HNSW search (k=5) + serendipity search (k=50, filter [0.3,0.6]) + KG BFS (depth 3)
- [x] `DerivationKind::PreferenceLearned` (tag 55), `JitirSuggestion` (tag 56), `ProactiveAssistance` (tag 57)
- [x] Agent integration: preference_manager field, init/resume/persist lifecycle, accessors
- [x] OODA integration: JITIR query in observe() guarded by interaction_count > 0
- [x] Trigger extensions: ContextMatch, UrgencyThreshold conditions; SurfaceSuggestions, RefreshPreferences actions
- [x] Reflection: preference parameter, preference_review field on ReflectionResult
- [x] `UserIntent::PrefCommand` in NLP, wired into TUI + headless
- [x] CLI: `Commands::Pref` with 5 subcommands (Status, Train, Level, Interests, Suggest)
- [x] `AgentError::Preference` transparent variant
- [x] 22 new unit tests
