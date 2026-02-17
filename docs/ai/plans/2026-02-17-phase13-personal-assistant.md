# Phase 13 — Personal Assistant: Email, Planning, Preferences, and Delegation

- **Date**: 2026-02-17
- **Status**: Planned
- **Depends on**: Phase 12 (CommChannel abstraction, capability model, constraint checking, social KG, oxifed integration), Phase 9 (microtheories, TMS, temporal projection), Phase 11 (autonomous goals, metacognition, HTN decomposition)
- **Research**: `docs/ai/decisions/004-personal-assistant-research.md`

## Goal

Make the agent a personal assistant that handles email, manages tasks and calendar, learns preferences, and provides structured output to operator dashboards — all on-device with full provenance. Email is a bidirectional CommChannel: the agent reads, classifies, triages, extracts structured information, composes replies, and sends messages. The operator can spawn delegated agents with their own identities to help specific people with domain-specific problems.

The core value proposition: **a local-first knowledge companion that accumulates understanding over time, respects privacy, and can explain every conclusion** — without cloud APIs or LLMs.

## Design Principles

1. **Email is a CommChannel** — JMAP/IMAP are transport layers for a `CommChannel` with `ChannelKind::Social` or `ChannelKind::Trusted` capabilities depending on sender classification
2. **Classification is VSA-native** — OnlineHD prototype vectors for spam/ham, single-pass incremental learning, no batch retraining
3. **Everything traces to provenance** — spam classification explains which features triggered it, priority explains which sender signals mattered, extraction explains which patterns matched
4. **Graduated proactivity** — ambient suggestions (Level 1) by default; escalate to nudges, offers, scheduled actions only with explicit opt-in
5. **Operator controls the send pipeline** — agent drafts, operator approves (Social level); auto-send only for learned routine patterns (Trusted+delegated rule)
6. **Local-first, always** — all email content processed on-device; federation via oxifed is optional enhancement
7. **Knowledge accumulates** — every email interaction enriches the sender reputation graph, preference profile, and temporal patterns

## Existing Building Blocks

| Component | Location | Reuse for Phase 13 |
|-----------|----------|-------------------|
| CommChannel trait | Phase 12a | Email implements CommChannel |
| Capability model | Phase 12a | Controls email send approval flow |
| Constraint checking | Phase 12c | Validates outbound email before send |
| Social KG | Phase 12d | Sender profiles, relationship tracking |
| Oxifed integration | Phase 12e | Forward email notifications to ActivityPub |
| Grammar module | `grammar/` | Compose email prose from KG content |
| Entity resolution | `grammar/entity_resolution.rs` | Match sender names to KG entities |
| Library parsers | `library/parser/` | Pattern for building email parser |
| VSA encode | `vsa/encode.rs` | `encode_role_filler`, `encode_label`, `encode_sequence` |
| HNSW item memory | `vsa/item_memory.rs` | Similarity search for classification + preference matching |
| SPARQL store | `graph/sparql.rs` | Sender reputation queries, temporal queries |
| Petgraph | `graph/index.rs` | Task dependency DAG, temporal interval graph |
| E-graph rewriting | `reason/mod.rs` | Spam rules, priority rules, extraction rules, temporal rules |
| Trigger system | `agent/trigger.rs` | Event-driven notifications and reminders |
| Reflection | `agent/reflect.rs` | Weekly review, habit tracking, preference review |
| Goal system | `agent/goal.rs` | Action items from emails become goals |
| Working memory | `agent/memory.rs` | Email context in OODA cycle |
| Episodic memory | `agent/memory.rs` | Historical email patterns, interaction episodes |
| Tiered storage | `store/` | Spam prototypes (hot), sender stats (warm), email index (cold) |
| Provenance | `provenance.rs` | Classification, extraction, and recommendation explanations |
| Compartments | `compartment/` | Scoped knowledge for delegated agents |
| Agent | `agent/agent.rs` | Delegated agent spawning reuses Agent infrastructure |

---

## Phase 13a — Email Channel (JMAP/IMAP + MIME)

**Problem**: The agent has no way to access email. No email connector, no MIME parser, no threading.

**Design**:

### Email Module Structure
```
src/email/
├── mod.rs          — EmailChannel (impl CommChannel), EmailPredicates
├── error.rs        — EmailError with miette diagnostics
├── connector.rs    — EmailConnector trait, JmapConnector, ImapConnector
├── parser.rs       — MIME parsing via mail-parser, header extraction
├── threading.rs    — JWZ threading algorithm (RFC 5256)
└── compose.rs      — Email composition from AbsTree, MIME encoding
```

### EmailChannel as CommChannel
```rust
pub struct EmailChannel {
    connector: Box<dyn EmailConnector>,
    capabilities: ChannelCapabilities,
    predicates: EmailPredicates,
    /// Mapping from email addresses to SymbolId (sender entities)
    sender_index: HashMap<String, SymbolId>,
}

impl CommChannel for EmailChannel {
    fn channel_kind(&self) -> ChannelKind {
        // Default to Social; upgrade to Trusted for screened senders
        ChannelKind::Social
    }
    // ...
}
```

### EmailConnector Trait
```rust
pub trait EmailConnector: Send + Sync {
    /// Fetch new messages since last sync
    fn fetch_new(&mut self) -> EmailResult<Vec<RawEmail>>;
    /// Fetch a specific message by ID
    fn fetch_by_id(&self, id: &str) -> EmailResult<Option<RawEmail>>;
    /// Send a composed message
    fn send(&self, message: &ComposedEmail) -> EmailResult<()>;
    /// Get current sync state (for delta sync)
    fn sync_state(&self) -> Option<String>;
}
```

Two implementations: `JmapConnector` (primary, using `jmap-client` crate) and `ImapConnector` (fallback, using `imap` crate). JMAP preferred because JSON-native, batched, delta sync.

### Email Entity Model
Emails become Entity symbols with well-known `email:` predicates:
```
email:message_id    → string (RFC 5322 Message-ID)
email:from          → Entity (sender)
email:to            → Entity (recipient, multiple)
email:cc            → Entity (cc recipient, multiple)
email:subject       → string
email:date          → timestamp
email:thread_id     → Entity (JWZ thread)
email:in_reply_to   → Entity (parent message)
email:has_attachment → bool
email:content_type  → string (text/plain, text/html, multipart/*)
email:body_text     → string (plain text body, truncated for KG)
email:list_id       → string (mailing list identification)
email:dkim_pass     → bool
email:spf_pass      → bool
```

### JWZ Threading
Implement the JWZ algorithm to build conversation trees from Message-ID, In-Reply-To, and References headers. Thread entities link to all participating messages.

**Deliverables**:
- [ ] `EmailConnector` trait with `JmapConnector` and `ImapConnector` implementations
- [ ] MIME parser using `mail-parser` crate, extracting headers and body parts
- [ ] `EmailPredicates` (analogous to `AgentPredicates`) — well-known relation SymbolIds
- [ ] JWZ threading algorithm producing thread Entity graphs
- [ ] `EmailChannel` implementing `CommChannel` trait from Phase 12a
- [ ] Email composition from `AbsTree` → MIME-encoded message
- [ ] Credential storage in redb (OAuth2 tokens, app passwords)

**Estimated scope**: ~600–900 lines

---

## Phase 13b — Spam & Relevance Classification

**Problem**: The agent needs to distinguish spam from legitimate email without an LLM or cloud service.

**Design**:

### OnlineHD Prototype Classification (VSA-Native)
Maintain two class prototype vectors in 10,000-bit binary space:

```rust
pub struct SpamClassifier {
    spam_prototype: HyperVec,
    ham_prototype: HyperVec,
    /// Number of training examples seen per class
    spam_count: u64,
    ham_count: u64,
    /// Bayesian token probabilities (supplement for edge cases)
    token_probs: TokenProbabilityTable,
}
```

Email encoding as composite hypervector:
```rust
fn encode_email(ops: &VsaOps, email: &ParsedEmail) -> HyperVec {
    let sender_hv = ops.encode_token(&email.from);
    let domain_hv = ops.encode_token(&email.from_domain);
    let subject_hv = ops.encode_label(&email.subject);
    let body_hv = ops.encode_label(&email.body_preview);

    ops.bundle(&[
        ops.bind(&role_vectors.sender, &sender_hv),
        ops.bind(&role_vectors.domain, &domain_hv),
        ops.bind(&role_vectors.subject, &subject_hv),
        ops.bind(&role_vectors.body, &body_hv),
        ops.bind(&role_vectors.has_links, &bool_hv(email.has_links)),
        ops.bind(&role_vectors.time_bucket, &time_bucket_hv(email.date)),
    ])
}
```

Classification: compare new email's hypervector against both prototypes using Hamming similarity. Closer prototype wins. OnlineHD adaptive update weights contribution by distance from current prototype (avoids saturation).

### Bayesian Token Probabilities (Supplement)
Robinson's chi-square method as a secondary signal. Token probability table stored in redb for durability. Merged with VSA similarity score for final classification.

### E-graph Spam Rules
Deterministic rules that override statistical classification:
```rust
// Trusted sender → always ham
rewrite!("ham-trusted"; "(and (from ?email ?sender) (triple ?sender reputation high))"
    => "(ham ?email)");

// Failed authentication → likely spam
rewrite!("spam-auth-fail"; "(and (not (dkim_pass ?email)) (not (spf_pass ?email)))"
    => "(spam_likely ?email)");
```

### User Feedback Training
When the user marks an email as spam/ham:
1. Encode email as hypervector
2. Bundle into appropriate prototype (OnlineHD adaptive weight)
3. Update token probability table
4. Record as `DerivationKind::UserFeedback` provenance

**Deliverables**:
- [ ] `SpamClassifier` with OnlineHD prototype vectors (spam/ham)
- [ ] Email-to-hypervector encoding function using role-filler bindings
- [ ] Bayesian token probability table (redb-durable, Robinson chi-square)
- [ ] E-graph spam reasoning rules in `AkhLang`
- [ ] Combined scoring: VSA similarity + Bayesian + e-graph rules
- [ ] User feedback training loop (mark spam/ham → update prototypes)
- [ ] Provenance for every classification decision

**Estimated scope**: ~500–750 lines

---

## Phase 13c — Email Triage & Priority

**Problem**: Even non-spam email needs prioritization. The agent should surface what matters and route the rest.

**Design**:

### Four-Feature Importance Model (Gmail-inspired)
1. **Social features**: Reply rate, interaction frequency, recency of last interaction, relationship type
2. **Content features**: Similarity to recently-acted-on emails (VSA), subject keyword match to active goals
3. **Thread features**: User started thread, user has replied, thread length
4. **Label features**: Sender routing decisions (HEY-style screening)

### Sender Reputation Subgraph
```
sender:reply_rate        → float
sender:msg_count         → u64
sender:avg_reply_time    → u64 (seconds)
sender:last_interaction  → timestamp
sender:relationship      → Entity (colleague/friend/service/newsletter/unknown)
sender:routing           → Entity (important/feed/paper_trail/screened_out)
sender:importance_score  → float (computed, cached)
```

### VSA Priority Prototypes
Encode sender interaction signatures as composite hypervectors. Maintain "important" and "low-priority" prototype vectors, updated from user behavior (quick replies = important; archived without reading = low priority).

### HEY-Style Screening
First-time senders trigger operator approval via CommChannel:
- Approved → sender routing decision stored as KG triple
- Screened out → future emails auto-archived

### Priority Routing
```rust
pub enum EmailRoute {
    Important,    // Direct to operator attention
    Feed,         // Newsletters, digests — read when convenient
    PaperTrail,   // Receipts, confirmations — searchable archive
    ScreeningQueue, // Unknown senders awaiting approval
    Spam,         // Classified spam (from 13b)
}
```

**Deliverables**:
- [ ] Sender reputation graph with `sender:` predicates in Oxigraph
- [ ] Four-feature importance scoring (social + content + thread + label)
- [ ] VSA priority prototype vectors (important/low-priority)
- [ ] HEY-style screening flow via operator CommChannel
- [ ] `EmailRoute` enum with routing logic
- [ ] Incremental sender statistics update on each email
- [ ] SPARQL queries for sender reputation aggregation

**Estimated scope**: ~500–700 lines

---

## Phase 13d — Structured Extraction from Messages

**Problem**: Emails contain actionable structured information (dates, events, tracking numbers, action items) buried in prose.

**Design**:

### Rule-Based Extractors
Three-layer extraction:

1. **Pattern matchers** (regex-like, expressed as KG pattern entities):
   - Dates: ISO 8601, US/EU formats, relative ("next Tuesday", "in 3 days")
   - Times: 12h/24h formats, timezone abbreviations
   - Tracking numbers: UPS (1Z...), FedEx (12-22 digits), USPS (20-22 digits)
   - Phone numbers, URLs, email addresses

2. **Gazetteer lookup** (KG entities):
   - City/country names, organization names, month/day names
   - Populated from engine's existing entity store

3. **Contextual rules** (e-graph):
   ```rust
   // Date + time + location in proximity → calendar event
   rewrite!("extract-event";
       "(and (has_date ?email ?date) (and (has_time ?email ?time) (has_location ?email ?loc)))"
       => "(calendar_event ?email ?date ?time ?loc)");

   // Action verb + deadline → action item
   rewrite!("extract-action";
       "(and (has_action_verb ?email ?verb) (has_deadline ?email ?date))"
       => "(action_item ?email ?verb ?date)");
   ```

### Extracted Entities as KG Triples
```
# Calendar event extracted from email
email:mentions_event    → Entity (event)
cal:date                → timestamp
cal:time                → string
cal:location            → Entity or string
cal:organizer           → Entity (sender)

# Shipment extracted from email
email:mentions_shipment → Entity (shipment)
ship:carrier            → Entity (UPS/FedEx/USPS)
ship:tracking           → string
ship:eta                → timestamp

# Action item → agent goal
email:action_item       → Entity (goal)
```

### Action Items as Agent Goals
Extracted action items (e.g., "Please review the document by Friday") convert to agent goals:
```rust
Goal {
    description: "Review document from alice@example.com",
    criteria: "document reviewed, reply sent",
    deadline: Some(friday_timestamp),
    provenance: DerivationKind::EmailExtraction(email_symbol_id),
}
```

**Deliverables**:
- [ ] Pattern matcher registry (date, time, tracking, phone, URL extractors)
- [ ] Gazetteer integration with existing KG entities
- [ ] E-graph extraction rules for calendar events, shipments, action items
- [ ] Extracted entities stored as KG triples with provenance
- [ ] Action item → Goal conversion pipeline
- [ ] VSA fuzzy pattern matching for email type classification ("meeting invite", "shipping notification")

**Estimated scope**: ~600–800 lines

---

## Phase 13e — Personal Task & Project Management

**Problem**: The agent has goals but no personal information management (PIM) framework for the operator's tasks, projects, and organizational needs.

**Design**:

### PIM Predicates
```rust
pub struct PimPredicates {
    // GTD workflow
    pub task_status: SymbolId,     // inbox/next/waiting/someday/reference/done
    pub task_context: SymbolId,    // @home, @office, @computer, @phone
    pub task_energy: SymbolId,     // low/medium/high (GTD energy contexts)
    pub task_time_est: SymbolId,   // estimated minutes

    // Eisenhower
    pub task_urgency: SymbolId,    // 0.0–1.0
    pub task_importance: SymbolId, // 0.0–1.0

    // PARA
    pub pim_category: SymbolId,    // project/area/resource/archive
    pub pim_deadline: SymbolId,    // timestamp (projects)
    pub pim_standard: SymbolId,    // description (areas)

    // Dependencies
    pub dep_blocked_by: SymbolId,  // task B blocked by task A
    pub dep_blocks: SymbolId,      // task A blocks task B

    // Recurrence
    pub task_recurrence: SymbolId, // RRULE string
    pub task_next_due: SymbolId,   // timestamp
    pub task_last_done: SymbolId,  // timestamp
}
```

### Eisenhower Matrix with VSA
Encode urgency and importance as separate dimensions:
```rust
let task_priority = ops.bundle(&[
    ops.bind(&v_urgent, &level_encode(ops, urgency)),
    ops.bind(&v_important, &level_encode(ops, importance)),
]);
```
Similarity search finds tasks with similar priority profiles. Quadrant 2 (important but not urgent) tasks receive a long-term priority bonus.

### Task Dependencies as Petgraph DAG
Reuse existing `DiGraph<SymbolId, EdgeData>` with dependency edge type. Topological sort yields execution order. Critical path analysis identifies schedule-driving tasks.

### E-graph Dependency Propagation
```rust
// If A blocks B and A is completed, B is unblocked
rewrite!("unblock"; "(and (triple ?a dep:blocks ?b) (triple ?a task:status done))"
    => "(triple ?b task:status ready)");

// Deadline inheritance
rewrite!("deadline-chain";
    "(and (triple ?b dep:blocked_by ?a) (triple ?a pim:deadline ?d))"
    => "(triple ?b task:earliest_start ?d)");
```

### GTD Workflow Integration
The existing OODA reflection system drives the GTD weekly review:
1. **Collect**: Gather unprocessed items from working memory
2. **Process**: Classify as actionable or reference
3. **Review**: Check all active goals, waiting-for items, projects
4. **Calendar review**: Past week and next two weeks
5. **Someday/Maybe**: Promote or archive

### Bullet Journal Migration
Extend reflection with a "migration" step: review open tasks and force a keep/drop decision. Maps to existing `reflect.rs` infrastructure.

**Deliverables**:
- [ ] `PimPredicates` struct (analogous to `AgentPredicates`, `EmailPredicates`)
- [ ] GTD workflow states as well-known predicates
- [ ] Eisenhower matrix with VSA urgency/importance encoding
- [ ] PARA categorization predicates
- [ ] Task dependency DAG with topological sort and critical path
- [ ] Recurring task infrastructure (RRULE parsing, next-due computation)
- [ ] E-graph dependency propagation rules
- [ ] GTD weekly review as reflection extension
- [ ] CLI commands: `pim inbox`, `pim next`, `pim review`, `pim project <name>`

**Estimated scope**: ~700–1,000 lines

---

## Phase 13f — Calendar & Temporal Reasoning

**Problem**: The agent has no temporal model. It can't reason about time intervals, detect scheduling conflicts, or integrate with external calendars.

**Design**:

### iCalendar Parser
Build a parser for RFC 5545 VEVENT/VTODO/VALARM/RRULE following the library parser pattern:
```rust
pub fn parse_ical(data: &str) -> EmailResult<Vec<CalendarComponent>> {
    // Parse VEVENT → Entity + triples (dtstart, dtend, summary, location)
    // Parse VTODO → maps to Goal with deadline
    // Parse VALARM → maps to TriggerCondition
    // Parse RRULE → recurrence string for recurring event/task
}
```

### Allen Interval Algebra as KG Predicates
13 temporal relations as well-known predicates:
```
time:before, time:after, time:meets, time:met_by,
time:overlaps, time:overlapped_by, time:during, time:contains,
time:starts, time:started_by, time:finishes, time:finished_by,
time:equals
```

SPARQL queries: "Find all events overlapping with my meeting", "What tasks are due during this project?"

### Temporal E-graph Rules
```rust
// Before is transitive
rewrite!("before-trans";
    "(and (triple ?a time:before ?b) (triple ?b time:before ?c))"
    => "(triple ?a time:before ?c)");

// Conflict detection
rewrite!("conflict";
    "(and (triple ?a time:overlaps ?b) (and (triple ?a cal:requires ?r) (triple ?b cal:requires ?r)))"
    => "(triple ?a cal:conflicts_with ?b)");
```

### Scheduling as Constraint Satisfaction
Model time slots as entities with `time:start` and `time:end`. Graph traversal checks hard constraints (no overlapping committed slots). VSA similarity matches task requirements to slot characteristics ("deep work" tasks similar to "morning focus" slots).

### Proactive Scheduling
The OODA loop detects scheduling opportunities:
1. **Observe**: Upcoming deadlines, current progress, historical patterns
2. **Orient**: Match context (time, energy) to task requirements
3. **Decide**: Utility-based scoring selects highest-value scheduling action
4. **Act**: Propose schedule modifications to operator

VSA temporal patterns: bind day-of-week × time-of-day × activity-type vectors. Over time, ItemMemory accumulates these composites, revealing habitual patterns via similarity search.

**Deliverables**:
- [ ] iCalendar parser (RFC 5545 VEVENT, VTODO, VALARM, RRULE)
- [ ] CalDAV sync adapter (read events → KG triples)
- [ ] Allen interval algebra as 13 well-known temporal predicates
- [ ] Temporal reasoning e-graph rules (transitivity, conflict detection)
- [ ] Scheduling constraint checker (hard constraints via graph traversal)
- [ ] VSA temporal pattern encoding (day × time × activity)
- [ ] Proactive scheduling in OODA loop
- [ ] CLI commands: `cal today`, `cal week`, `cal conflicts`, `cal sync`

**Estimated scope**: ~600–900 lines

---

## Phase 13g — Preference Learning & Proactive Assistance

**Problem**: The agent doesn't learn what the operator cares about over time. It can't anticipate needs or surface relevant information proactively.

**Design**:

### VSA Preference Profiles (HyperRec-Style)
Encode operator interests as composite hypervectors with temporal decay:

```rust
pub struct PreferenceProfile {
    /// Topic interest vectors (bundled from interactions)
    interest_prototype: HyperVec,
    /// Temporal decay factor (weight recent interactions higher)
    decay_rate: f32,
    /// Interaction count for adaptive update weighting
    interaction_count: u64,
}
```

Updating preferences: when the user acts on information (replies to email, completes a goal, reads a document), encode that interaction and bundle into the interest prototype with higher weight for recent interactions.

### Implicit Feedback Signals
Track user behavior as KG triples:
```
user:replied_to       → email Entity (strong positive)
user:read_time        → seconds (>30s = interest)
user:archived_without_reading → email Entity (negative)
user:starred          → Entity (explicit positive)
user:completed_goal   → goal Entity (positive for related topics)
```

### Just-in-Time Information Retrieval (Remembrance Agent Pattern)
On every OODA cycle, encode current context as a hypervector and search ItemMemory:

```rust
// Current context from working memory + active goals
let context_hv = encode_context(wm, active_goals);

// Search for similar concepts (top-5)
let suggestions = item_memory.search(&context_hv, k=5, ef_search=200);

// Filter by compartment permissions
let accessible = filter_by_compartment(suggestions);

// Present as ambient suggestions (Level 1 — never interrupt)
dashboard.add_ambient(accessible);
```

### Serendipity Engine
Find non-obvious connections via HNSW near-miss matches:
```rust
// Standard search: closest matches (k=5)
let exact = item_memory.search(&query, k=5, ef_search=200);

// Serendipity: intermediate-distance matches
let broader = item_memory.search(&query, k=50, ef_search=400);
let serendipitous = broader.iter()
    .filter(|m| m.distance > 0.3 && m.distance < 0.6) // "serendipity zone"
    .take(3);
```

Additionally, KG path-based serendipity finds unexpected multi-hop connections.

### Context-Aware Reminders
Extend `TriggerCondition` with context-sensitive triggers:
```rust
// Proposed extensions to trigger.rs
BreakpointReached,                       // Activity transition detected
BatchReady { min_items: usize },         // Low-priority items accumulated
UrgencyThreshold { deadline_hours: u64 }, // Deadline approaching
ContextMatch { context_hv: HyperVec },   // VSA context similarity match
```

Breakpoint-sensitive timing (Iqbal & Bailey 2008): deliver notifications at task transitions, not mid-task.

### Graduated Proactivity Model
```rust
pub enum ProactivityLevel {
    Ambient,      // Suggestions visible in dashboard, no interruption
    Nudge,        // Brief notification at next breakpoint
    Offer,        // Structured suggestion with accept/dismiss
    Scheduled,    // Planned action requiring operator approval
    Autonomous,   // Auto-execute (only with explicit delegated rule)
}
```
Default to `Ambient`. Escalation requires explicit opt-in per category.

**Deliverables**:
- [ ] `PreferenceProfile` with VSA interest prototypes and temporal decay
- [ ] Implicit feedback tracking (reply, read-time, archive, star) as KG triples
- [ ] JITIR on every OODA cycle (encode context → search → ambient suggestions)
- [ ] Serendipity engine (near-miss HNSW matches + KG multi-hop paths)
- [ ] Extended `TriggerCondition` with breakpoint, batch, urgency, context triggers
- [ ] `ProactivityLevel` enum with graduated escalation model
- [ ] Preference profile persistence in tiered storage

**Estimated scope**: ~600–800 lines

---

## Phase 13h — Structured Output & Operator Dashboards

**Problem**: The agent's knowledge and insights need to be presented in structured, machine-readable formats that external operator dashboards can consume.

**Design**:

### JSON-LD Export
The engine already stores RDF in Oxigraph; JSON-LD is its JSON serialization:
```json
{
  "@context": "https://akh-medu.dev/context",
  "@type": "DailyBriefing",
  "date": "2026-02-17",
  "activeGoals": [
    {
      "@id": "akh:sym/42",
      "name": "Review Q4 financials",
      "status": "active",
      "progress": 0.6,
      "source": "email from alice@example.com"
    }
  ],
  "emailSummary": {
    "unread": 12,
    "important": 3,
    "screening_queue": 1,
    "spam_filtered": 47
  },
  "newInsights": [
    {
      "connection": "Paper on distributed consensus relates to active project goal",
      "confidence": 0.85,
      "provenance": "spreading_activation from sym/100 to sym/42"
    }
  ]
}
```

### Three-Tier Summarization
```rust
pub enum BriefingTier {
    Daily,   // Quick hits: 5-7 items max (Miller's Law)
    Weekly,  // Pattern synthesis: connecting dots, emerging themes
    Monthly, // Strategic view: goal progress, knowledge growth, preference evolution
}

pub struct Briefing {
    pub tier: BriefingTier,
    pub date_range: (u64, u64),
    pub goals: Vec<GoalSummary>,
    pub email_stats: EmailStatistics,
    pub new_knowledge: Vec<InsightSummary>,
    pub upcoming_deadlines: Vec<DeadlineSummary>,
    pub habit_streaks: Vec<HabitSummary>,
    pub suggested_actions: Vec<ActionSuggestion>,
}
```

### Notification System
Every notification carries full context:
```rust
pub struct Notification {
    pub priority: NotificationPriority, // Critical / Important / Informational
    pub summary: String,                // Human-readable one-liner
    pub context: Vec<Triple>,           // Supporting KG triples
    pub suggested_actions: Vec<Action>, // What the operator can do
    pub provenance: ProvenanceRecord,   // Full derivation chain
    pub expiry: Option<u64>,            // When this becomes irrelevant
    pub compartment: String,            // Source compartment
}
```

### Compartment-Organized Dashboard
Dashboard organized by compartments (Cowan's limit: 4±1 items per section):
```
[Research]            [Personal]           [Work]
 3 active goals       2 tasks due today    1 email needs reply
 5 new insights       Next: gym at 6pm     Meeting at 14:00
```

### Machine-Readable Endpoints
- **SPARQL** (already exists via Oxigraph): any client can query the KG
- **JSON-LD export**: serialize subgraphs as linked data
- **Webhook events**: trigger system emits structured JSON events
- **AMQP via oxifed**: push structured events to federated consumers

**Deliverables**:
- [ ] JSON-LD serializer for KG subgraphs
- [ ] `Briefing` struct with daily/weekly/monthly tiers
- [ ] Briefing generator (SPARQL aggregation + WM analysis + reflection data)
- [ ] `Notification` struct with priority/context/action/provenance/expiry
- [ ] Compartment-organized dashboard data structure
- [ ] Webhook event emitter as `TriggerAction` variant
- [ ] CLI commands: `briefing daily`, `briefing weekly`, `briefing monthly`, `notifications`

**Estimated scope**: ~500–700 lines

---

## Phase 13i — Delegated Agent Spawning

**Problem**: The operator wants to create specialized agents with their own identities to help specific people (e.g., a GCC/SPARC expert that corresponds with a friend under its own identity, rather than the operator taking credit).

**Design**:

### Agent Identity
```rust
pub struct AgentIdentity {
    /// Display name for this agent
    pub name: String,
    /// Description of expertise/purpose
    pub description: String,
    /// Unique identifier
    pub identity_id: SymbolId,
    /// Which compartments this agent can access (read)
    pub readable_compartments: Vec<String>,
    /// Which compartments this agent can write to
    pub writable_compartments: Vec<String>,
    /// Capability set for this agent's actions
    pub capabilities: ChannelCapabilities,
    /// The operator who created this agent
    pub creator: SymbolId,
}
```

### Scoped Knowledge
Each delegated agent gets access to specific compartments:
- **Dedicated compartment**: Agent's own working knowledge (conversations, derived insights)
- **Shared compartments**: Read access to operator's relevant compartments (e.g., "gcc-knowledge", "sparc-architecture")
- **No access**: Operator's personal compartments remain private

### CommChannel Forwarding
The operator routes messages to/from delegated agents:
```
Friend → Email → Operator's agent → routes to Delegated Agent
Delegated Agent → composes reply → Constraint Check → Send under agent's identity
```

For email: the delegated agent uses a sub-address or alias (e.g., `sparc-helper@operator-domain.com`) as its outbound identity.

For ActivityPub (via oxifed): the delegated agent gets its own Actor identity in the oxifed system.

### Email Composition Pipeline
When a delegated agent needs to send email:
1. **Grammar module** generates prose from KG-grounded content (AbsTree → linearize)
2. **Constraint checking** (Phase 12c) validates content before send
3. **Capability check**: Agent must have send permission for this channel
4. **Approval flow**:
   - Default: draft queued for operator approval
   - Trusted rule: operator can whitelist patterns for auto-send (e.g., "this agent can auto-reply to user X about topic Y")

### Lifecycle Management
```rust
pub enum DelegatedAgentAction {
    Create { identity: AgentIdentity, initial_goal: Option<String> },
    Pause { agent_id: SymbolId },
    Resume { agent_id: SymbolId },
    Terminate { agent_id: SymbolId, merge_knowledge: bool },
}
```

On termination with `merge_knowledge: true`, the delegated agent's working knowledge is consolidated into the operator's KG with provenance tracking the delegation.

### Delegation as Provenance
Every action by a delegated agent carries `DerivationKind::DelegatedAgent(agent_id)` provenance, maintaining the full audit trail.

**Deliverables**:
- [ ] `AgentIdentity` struct with name, compartments, capabilities
- [ ] Delegated agent creation from operator command
- [ ] Scoped `CompartmentManager` for restricted knowledge access
- [ ] CommChannel forwarding (operator routes messages to delegated agent)
- [ ] Email composition pipeline: grammar → constraint check → capability check → approval
- [ ] Agent-specific outbound identity (email alias, AP actor)
- [ ] Lifecycle management (create, pause, resume, terminate, merge knowledge)
- [ ] `DerivationKind::DelegatedAgent` provenance variant
- [ ] CLI commands: `agent delegate create`, `agent delegate list`, `agent delegate terminate`

**Estimated scope**: ~700–1,000 lines

---

## Architecture Diagram

```
                    ┌─────────────────────────────────────────┐
                    │              Operator                     │
                    │    (Chat CommChannel — full capabilities) │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │           Agent (OODA Loop)              │
                    │                                          │
                    │  ┌──────────┐  ┌──────────┐  ┌────────┐│
                    │  │ Observe  │→ │  Orient  │→ │ Decide ││
                    │  │ (email,  │  │ (classify,│  │ (route,││
                    │  │  tasks,  │  │  extract, │  │ triage,││
                    │  │  calendar)│  │  priority)│  │ suggest││
                    │  └──────────┘  └──────────┘  └───┬────┘│
                    │                                   │     │
                    │  ┌────────────────────────────────▼───┐ │
                    │  │              Act                    │ │
                    │  │  • Route email to inbox/feed/spam   │ │
                    │  │  • Create goals from action items   │ │
                    │  │  • Compose replies (grammar module) │ │
                    │  │  • Surface suggestions (JITIR)      │ │
                    │  │  • Generate briefings (JSON-LD)     │ │
                    │  │  • Forward to delegated agents      │ │
                    │  └────────────────────────────────────┘ │
                    └──┬──────────┬──────────┬───────────┬────┘
                       │          │          │           │
          ┌────────────▼──┐ ┌────▼────┐ ┌───▼───┐ ┌────▼──────────┐
          │ Email Channel │ │   PIM   │ │ Cal   │ │ Delegated     │
          │ (JMAP/IMAP)   │ │ (tasks, │ │ (RFC  │ │ Agents        │
          │               │ │  GTD,   │ │ 5545, │ │ (scoped       │
          │ • fetch       │ │  PARA)  │ │ Allen)│ │  knowledge,   │
          │ • parse       │ │         │ │       │ │  own identity) │
          │ • classify    │ │         │ │       │ │               │
          │ • compose     │ │         │ │       │ │               │
          │ • send        │ │         │ │       │ │               │
          └───────────────┘ └─────────┘ └───────┘ └───────────────┘
                       │          │          │           │
          ┌────────────▼──────────▼──────────▼───────────▼────┐
          │                Engine (VSA + KG + E-graph)         │
          │                                                    │
          │  VSA: spam/ham prototypes, priority prototypes,    │
          │       preference profiles, temporal patterns,      │
          │       serendipity (near-miss HNSW)                │
          │                                                    │
          │  KG:  sender reputation, email threads,            │
          │       task dependencies, calendar events,          │
          │       Allen intervals, preference triples          │
          │                                                    │
          │  E-graph: spam rules, priority rules,              │
          │           extraction rules, temporal reasoning,    │
          │           dependency propagation                   │
          └───────┬────────────────────────────────────────────┘
                  │
          ┌───────▼────────────────────────────┐
          │        Structured Output            │
          │                                     │
          │  • JSON-LD briefings (daily/weekly) │
          │  • SPARQL endpoints                 │
          │  • Notifications (prioritized)      │
          │  • AMQP events (via oxifed)         │
          │  • Webhook events                   │
          └─────────────────────────────────────┘
```

## Scope Summary

| Sub-phase | Description | Estimated Lines |
|-----------|-------------|----------------|
| 13a | Email Channel (JMAP/IMAP + MIME) | 600–900 |
| 13b | Spam & Relevance Classification | 500–750 |
| 13c | Email Triage & Priority | 500–700 |
| 13d | Structured Extraction from Messages | 600–800 |
| 13e | Personal Task & Project Management | 700–1,000 |
| 13f | Calendar & Temporal Reasoning | 600–900 |
| 13g | Preference Learning & Proactive Assistance | 600–800 |
| 13h | Structured Output & Operator Dashboards | 500–700 |
| 13i | Delegated Agent Spawning | 700–1,000 |
| **Total** | | **~5,300–7,550** |

## Implementation Priority

**Core** (enables the rest):
- 13a Email Channel — everything depends on email access
- 13e PIM — task management is the foundation for personal assistance

**High value** (what users care about most):
- 13b Spam Classification — immediate quality-of-life improvement
- 13c Email Triage — surface what matters
- 13d Structured Extraction — turns emails into actionable knowledge

**Enhancement** (builds on core):
- 13f Calendar — temporal reasoning enriches everything
- 13g Preference Learning — accumulates value over time
- 13h Dashboards — structured output for operator systems

**Advanced** (depends on Phase 12 maturity):
- 13i Delegated Agent Spawning — requires stable CommChannel + capability model + social KG
