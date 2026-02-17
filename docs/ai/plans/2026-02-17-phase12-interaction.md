# Phase 12 — Interaction: Communication Protocols and Social Reasoning

- **Date**: 2026-02-17
- **Status**: Planned
- **Depends on**: Phase 9 (microtheories for theory of mind, argumentation for reasoning transparency), Phase 11 (autonomous goals, metacognition)
- **Inspired by**: Winter (Quinn Wilton's neurosymbolic Bluesky bot), Sussman-Radul propagator model, Spritely Goblins capability security, Christine Lemmer-Webber's analysis

## Goal

Give the agent an interaction surface. The existing Chat is not just a UI — it is the **operator communication protocol**, the highest-privilege channel through which the akh's human steward converses, directs, inspects, and collaborates. Additional channels (ActivityPub via oxifed, future ATProto, agent-to-agent) get progressively restricted capability sets. Every channel shares a common `CommChannel` abstraction with capability-secured permissions, pre-communication constraint checking, and grounded dialogue backed by the full KG + VSA + e-graph reasoning stack.

Winter demonstrated that interesting neurosymbolic behavior emerges in social context — but Winter lacks provenance, TMS, contradiction detection, and similarity search. Akh-medu has all of those. Phase 12 provides the interaction surface that makes them useful for communication.

## Design Principles

1. **Chat is a protocol, not just a UI** — the TUI, headless REPL, WebSocket remote, and future SSH access are all transport layers for the same operator protocol
2. **The operator is the akh's steward** — operator channel has full capabilities: set goals, modify KG, approve proposals, inspect reasoning, override decisions
3. **External channels get attenuated capabilities** — ActivityPub followers can query but not mutate; other agents can propose but not assert
4. **Every utterance is grounded** — responses trace back to KG state via provenance, not generated from thin air
5. **Constraint checking before output** — inspired by Winter's Datalog self-check, but using the full inference stack
6. **Propagator-inspired accumulation** — conversations accumulate partial information into the KG monotonically, narrowing beliefs over time

## Existing Building Blocks

| Component | Location | What It Does |
|-----------|----------|-------------|
| TUI | `tui/mod.rs` | Ratatui event loop, local + WebSocket remote backends |
| Message protocol | `message.rs` | `AkhMessage` enum (Fact, Reasoning, Gap, ToolResult, Narrative, etc.) |
| MessageSink trait | `message.rs` | `StdoutSink`, `JsonSink`, `VecSink`, `TuiSink` |
| Grammar registry | `grammar/mod.rs` | 3 built-in archetypes (formal, terse, narrative), `ConcreteGrammar` trait |
| Abstract syntax | `grammar/abs.rs` | `AbsTree` — shared interlingua across all grammars |
| Parser | `grammar/parser.rs` | `parse_prose()` — priority cascade: commands → questions → compounds → facts |
| Lexicon | `grammar/lexer.rs` | Language-aware tokenization, `QuestionFrame`, 5 languages |
| Discourse | `grammar/discourse.rs` | `DiscourseContext`, pronoun resolution, POV/focus, response detail levels |
| Entity resolution | `grammar/entity_resolution.rs` | Fuzzy/learned alias matching via VSA + KG |
| Intent classifier | `agent/nlp.rs` | `classify_intent()` — Query, Assert, SetGoal, RunAgent, etc. |
| User interact tool | `agent/tools/user_interact.rs` | Blocking stdin prompt/read |
| Agent REPL | `agent/agent.rs` + `main.rs` | TUI mode and headless mode, session persistence |
| Chat module | `agent/chat.rs` | Conversation persistence, SSH fingerprint discovery |
| Provenance | `provenance.rs` | Full derivation history, multi-index ledger |
| Oxifed (external) | `github.com/toasterson/oxifed` | AP federation server — domainservd (HTTP), publisherd (delivery), AMQP bus, MongoDB; akh-medu is an app inside this system |

## What's Missing

1. **Channel abstraction** — no unified protocol layer; TUI, headless, and WebSocket are separate code paths
2. **Capability model** — no permission differentiation between operator and external actors
3. **Grounded responses** — intent classifier dispatches to tools but responses aren't KG-grounded with provenance trails
4. **Constraint checking** — agent outputs are not validated before emission
5. **Social knowledge** — no model of interlocutors, their knowledge state, trust, or interests
6. **ActivityPub integration** — no connection to oxifed; the agent cannot participate in federated social networks
7. **Explanation interface** — provenance exists but isn't wired into conversational "why?" queries

---

## Phase 12a — Communication Channel Abstraction

**Problem**: The TUI, headless REPL, WebSocket remote, and future protocols are each wired differently. There's no shared protocol layer with a unified capability model.

**Design**:

### CommChannel Trait
A unified abstraction over all interaction surfaces:
```rust
pub trait CommChannel: Send + Sync {
    fn channel_id(&self) -> &str;
    fn capabilities(&self) -> &ChannelCapabilities;
    fn receive(&mut self) -> ChannelResult<Option<InboundMessage>>;
    fn send(&self, msg: &OutboundMessage) -> ChannelResult<()>;
    fn channel_kind(&self) -> ChannelKind;
}

pub enum ChannelKind {
    Operator,       // Full capabilities — the steward
    Trusted,        // Read + propose + limited mutate — trusted agents
    Social,         // Read + converse + propose goals — ActivityPub followers
    Public,         // Read-only — unauthenticated queries
}
```

### Capability Model (Goblins/OCapN-inspired)
Each channel carries an immutable capability set, determined at creation:

```rust
pub struct ChannelCapabilities {
    pub can_query_kg: bool,           // Read KG state
    pub can_assert_triples: bool,     // Add knowledge directly
    pub can_retract_triples: bool,    // Remove knowledge
    pub can_set_goals: bool,          // Create goals directly (active)
    pub can_propose_goals: bool,      // Suggest goals (proposed, needs approval)
    pub can_approve_proposals: bool,  // Promote proposed → active goals
    pub can_inspect_reasoning: bool,  // See provenance chains, tool scores, plans
    pub can_override_decisions: bool, // Veto agent decisions, force tool selection
    pub can_execute_tools: bool,      // Trigger tool execution directly
    pub can_modify_config: bool,      // Change agent parameters (reflection config, etc.)
    pub max_message_rate: Option<u32>,  // Rate limit (messages per minute)
}
```

Capability presets:

| Preset | Query | Assert | Retract | SetGoal | Propose | Approve | Inspect | Override | Tools | Config | Rate |
|--------|-------|--------|---------|---------|---------|---------|---------|----------|-------|--------|------|
| **Operator** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | None |
| **Trusted** | ✓ | ✓ | — | — | ✓ | — | ✓ | — | — | — | 30/min |
| **Social** | ✓ | — | — | — | ✓ | — | — | — | — | — | 10/min |
| **Public** | ✓ | — | — | — | — | — | — | — | — | — | 5/min |

### InboundMessage / OutboundMessage
Unified message types that all channels produce/consume:

```rust
pub struct InboundMessage {
    pub channel_id: String,
    pub sender: InterlocutorId,
    pub content: MessageContent,
    pub timestamp: u64,
}

pub enum MessageContent {
    Text(String),                           // Natural language input
    Command(CommandKind),                   // Structured command
    GoalProposal(GoalProposal),            // From Phase 11a
    Activity(Box<ActivityPub>),             // From oxifed
    Reaction { target: SymbolId, kind: ReactionKind },
}

pub struct OutboundMessage {
    pub content: ResponseContent,
    pub provenance: Option<Vec<SymbolId>>,  // Derivation chain backing this response
    pub confidence: Option<f32>,
    pub constraint_check: ConstraintCheckResult,
}
```

### Channel Registry
The agent maintains a registry of active channels. The operator channel is always present. Additional channels are registered at runtime:

```rust
pub struct ChannelRegistry {
    channels: HashMap<String, Box<dyn CommChannel>>,
    operator_channel: String,  // Always exactly one operator
}
```

### Transport Implementations
The existing backends become transports for the operator channel:
- `TuiTransport` — ratatui event loop (existing `AkhTui`, refactored)
- `HeadlessTransport` — stdin/stdout (existing headless REPL, refactored)
- `WebSocketTransport` — existing `RemoteChat`, refactored
- Future: `SshTransport`, `UnixSocketTransport`

All produce/consume `InboundMessage`/`OutboundMessage` through the `CommChannel` trait.

**Changes**:
- [ ] New file: `src/agent/channel.rs` — `CommChannel` trait, `ChannelCapabilities`, `ChannelKind`, `ChannelRegistry`
- [ ] New file: `src/agent/message_types.rs` — `InboundMessage`, `OutboundMessage`, `MessageContent`, `ResponseContent`
- [ ] `src/tui/mod.rs` — refactor `AkhTui` to implement `CommChannel` (operator preset)
- [ ] `src/agent/agent.rs` — `ChannelRegistry` field, multi-channel message loop
- [ ] `src/message.rs` — bridge between existing `AkhMessage` and new `OutboundMessage`

**Estimated scope**: ~600–800 lines

---

## Phase 12b — Grounded Operator Dialogue

**Problem**: The intent classifier dispatches to tools, but responses aren't systematically grounded in KG state with provenance. The operator deserves provenance-backed answers, not pattern-matched output.

**Design**:

### Conversational Grounding Pipeline
Every operator query flows through:

1. **Parse**: `parse_prose()` → `AbsTree` (existing, works well)
2. **Resolve**: `EntityResolver` maps labels → SymbolIds (existing)
3. **Ground**: Query KG for all triples involving resolved entities. Collect provenance for each.
4. **Infer**: If grounding is sparse, run spreading activation from resolved entities to discover related knowledge
5. **Compose**: Build response `AbsTree` from grounded triples + inferences
6. **Linearize**: `GrammarRegistry::linearize()` → prose (existing)
7. **Annotate**: Attach provenance chain to the `OutboundMessage`

### Discourse-Aware Multi-Turn Context
Extend `DiscourseContext` to maintain conversation history:

```rust
pub struct ConversationState {
    pub channel_id: String,
    pub turns: VecDeque<ConversationTurn>,     // Bounded ring buffer
    pub active_referents: Vec<SymbolId>,       // Entities "on the table"
    pub active_topic: Option<SymbolId>,        // Current focus entity
    pub grammar: String,                       // Current linearization grammar
    pub response_detail: ResponseDetail,       // Concise | Normal | Full
}

pub struct ConversationTurn {
    pub speaker: Speaker,                      // Operator | Agent
    pub content: AbsTree,                      // Parsed abstract syntax
    pub resolved_entities: Vec<SymbolId>,
    pub timestamp: u64,
}
```

Pronoun resolution uses `active_referents` from recent turns. "It", "that", "this" resolve to the most recent entity mentioned. "They" resolves to the most recent plural or collection.

### Operator-Specific Commands
The operator channel recognizes commands that other channels cannot:

| Command | Effect | Capability Required |
|---------|--------|-------------------|
| `why {statement}?` | Show provenance chain for a triple | `can_inspect_reasoning` |
| `how did you decide {X}?` | Show tool scoring from last relevant Decision | `can_inspect_reasoning` |
| `approve {goal}` | Promote Proposed → Active | `can_approve_proposals` |
| `reject {goal}` | Dismiss a proposed goal | `can_approve_proposals` |
| `assert {triple}` | Add triple directly to KG | `can_assert_triples` |
| `retract {triple}` | Remove triple from KG | `can_retract_triples` |
| `override: use {tool}` | Force next cycle to use a specific tool | `can_override_decisions` |
| `set detail {level}` | Change response verbosity | `can_modify_config` |

### Response Quality Levels
The operator can request different detail levels:
- **Concise**: Just the answer (existing `ResponseDetail::Concise`)
- **Normal**: Answer + brief rationale
- **Full**: Answer + rationale + provenance IDs + confidence scores + supporting triples

**Changes**:
- [ ] New file: `src/agent/conversation.rs` — `ConversationState`, `ConversationTurn`, grounding pipeline
- [ ] `src/grammar/discourse.rs` — wire `ConversationState` into pronoun resolution
- [ ] `src/agent/nlp.rs` — extend `classify_intent()` with operator commands (why, approve, reject, assert, retract, override)
- [ ] `src/agent/ooda.rs` — expose Decision/Orientation data for "how did you decide?" queries
- [ ] `src/tui/mod.rs` — display provenance annotations in Full detail mode

**Estimated scope**: ~700–900 lines

---

## Phase 12c — Pre-Communication Constraint Checking

**Problem**: The agent can output anything. There's no symbolic validation before communication. Winter's key insight — checking Datalog rules before every post — should be done with akh-medu's full inference stack.

**Design**:

### Constraint Check Pipeline
Before any `OutboundMessage` is emitted on any channel, run:

1. **Consistency check**: Does the response contradict anything in the KG? Query e-graph for the response's key assertions. If any assertion is in the same e-class as a known falsehood, flag it.
2. **Confidence check**: Are all assertions in the response above a minimum confidence threshold? (Configurable per channel — operator might tolerate lower confidence with a caveat; social channel requires higher.)
3. **Rate check**: Has the agent exceeded the channel's `max_message_rate`? If so, queue or drop.
4. **Relevance check**: Is the response actually about what was asked? Compute VSA similarity between the query vector and the response vector. If below threshold, flag as potentially off-topic.
5. **Sensitivity check**: Does the response involve entities marked with sensitivity predicates (e.g., `is-a:private`, `has-sensitivity:high`)? If so, check that the channel has appropriate capabilities.
6. **Provenance check**: Can every factual claim in the response be traced to a provenance record? Ungrounded claims get flagged.

### ConstraintCheckResult
```rust
pub struct ConstraintCheckResult {
    pub passed: bool,
    pub violations: Vec<ConstraintViolation>,
    pub warnings: Vec<ConstraintWarning>,
}

pub enum ConstraintViolation {
    Contradiction { claim: AbsTree, contradicts: SymbolId },
    BelowConfidence { claim: AbsTree, confidence: f32, threshold: f32 },
    RateLimitExceeded { channel: String, limit: u32, current: u32 },
    SensitivityBreach { entity: SymbolId, required_kind: ChannelKind, actual_kind: ChannelKind },
}

pub enum ConstraintWarning {
    Ungrounded { claim: AbsTree },
    LowRelevance { similarity: f32, threshold: f32 },
    LowConfidence { claim: AbsTree, confidence: f32 },
}
```

### Behavior on Violation
- **Operator channel**: Emit anyway but annotate with warnings (the steward decides)
- **Trusted channel**: Suppress violated claims, emit remainder with caveats
- **Social channel**: Suppress entirely, log the suppression for operator review
- **Public channel**: Suppress entirely, return a "cannot answer" response

### Winter's Spamminess Rule, Generalized
Winter's Datalog rule "am I posting too much?" becomes a general **communication budget** tracked per channel per time window:

```rust
pub struct CommunicationBudget {
    pub channel_id: String,
    pub window: Duration,
    pub max_messages: u32,
    pub max_assertions: u32,     // Factual claims per window
    pub cooldown_after_burst: Duration,
}
```

**Changes**:
- [ ] New file: `src/agent/constraint_check.rs` — `ConstraintCheckResult`, check pipeline, communication budget
- [ ] `src/agent/channel.rs` — wire constraint check into `CommChannel::send()` (mandatory for all channels)
- [ ] `src/agent/conversation.rs` — insert constraint check before response emission
- [ ] `src/reason/` — expose e-graph contradiction query for consistency check

**Estimated scope**: ~500–700 lines

---

## Phase 12d — Social Knowledge Graph and Theory of Mind

**Problem**: The agent has no model of who it's talking to, what they know, what they're interested in, or how much to trust them.

**Research basis**: Cyc's theory of mind (desideratum #6), microtheories (9a) for per-interlocutor belief contexts, VSA for interest modeling.

**Design**:

### Interlocutor Model
Each person or agent the akh interacts with becomes a KG entity with well-known predicates:

```
interlocutor:alice  is-a               Interlocutor
interlocutor:alice  has-channel         channel:activitypub:alice@example.com
interlocutor:alice  has-trust-level     "social"
interlocutor:alice  has-interest        concept:datalog
interlocutor:alice  has-interest        concept:distributed-systems
interlocutor:alice  last-interaction    "1708200000"
interlocutor:alice  interaction-count   "42"
```

### Interest Mapping via VSA
Bundle each interlocutor's known interests into a single hypervector:
```
alice_interests = bundle(datalog_hv, distributed_systems_hv, ...)
```
Discover interest overlaps between interlocutors via Hamming similarity. "Alice and Bob both care about X" — exactly what Winter does with Datalog queries, but via HNSW over bundled interest vectors.

### Theory of Mind via Microtheories (9a)
Each interlocutor gets a dedicated microtheory (from Phase 9a) representing **what the akh believes they know**:

```
mt:alice-knowledge  genlMt  mt:common-knowledge
mt:alice-knowledge  contains  (alice knows concept:datalog)
mt:alice-knowledge  contains  (alice does-not-know concept:propagators)
```

When responding to Alice, the agent can:
- Avoid explaining concepts she already knows
- Provide background on concepts she doesn't know
- Tailor vocabulary to her expertise level

This is Cyc's "theory of mind" desideratum implemented with existing microtheory infrastructure.

### Trust as Capability Mapping
Trust isn't a single number — it's a mapping to `ChannelKind`:

```rust
pub struct InterlocutorProfile {
    pub id: SymbolId,
    pub channel_ids: Vec<String>,
    pub trust_level: ChannelKind,       // Determines capability preset
    pub interest_vector: HyperVec,
    pub knowledge_mt: Option<SymbolId>, // Microtheory of what they know
    pub interaction_history: Vec<SymbolId>,  // Episodic entries
}
```

The operator can promote/demote trust levels. Default for new interlocutors: `Public`. The operator is always `Operator`.

### Impression Formation (Winter-inspired)
When first encountering a new interlocutor, form an initial impression:
- Encode their first messages as VSA vectors
- Search HNSW for similar past interlocutors
- Inherit initial interest estimates from the nearest match
- Refine monotonically as more interactions occur (propagator-style accumulation)

**Changes**:
- [ ] New file: `src/agent/interlocutor.rs` — `InterlocutorProfile`, interest mapping, impression formation
- [ ] `src/agent/channel.rs` — associate `InterlocutorId` with each inbound message
- [ ] `src/agent/conversation.rs` — theory-of-mind-aware response tailoring
- [ ] `src/compartment/` — per-interlocutor microtheory creation (via 9a)
- [ ] `src/vsa/encoding.rs` — `encode_interest_bundle()` for interlocutor interest vectors

**Depends on**: 9a (microtheories)

**Estimated scope**: ~600–800 lines

---

## Phase 12e — Federation via Oxifed

**Problem**: The agent exists only in the terminal. It cannot participate in federated social networks where neurosymbolic interaction (as Winter demonstrates) becomes most interesting.

**Integration model**: Akh-medu is an **application inside the oxifed system** — not a standalone ActivityPub implementation. Oxifed handles all AP protocol details (HTTP signatures, WebFinger, inbox/outbox, federation, delivery). Akh-medu communicates with oxifed through its internal APIs and RabbitMQ/LavinMQ message bus. The akh never touches ActivityPub directly.

```
                    ┌──────────────────────────────┐
                    │  The Fediverse                │
                    │  (remote AP servers)          │
                    └──────────┬───────────────────┘
                               │ ActivityPub (HTTP)
                    ┌──────────▼───────────────────┐
                    │  oxifed                       │
                    │  ┌─────────────┐              │
                    │  │ domainservd │ AP endpoints  │
                    │  └──────┬──────┘              │
                    │         │                     │
                    │  ┌──────▼──────┐              │
                    │  │  MongoDB    │ state         │
                    │  └──────┬──────┘              │
                    │         │                     │
                    │  ┌──────▼──────┐              │
                    │  │ publisherd  │ delivery      │
                    │  └─────────────┘              │
                    └──────────┬───────────────────┘
                               │ AMQP (RabbitMQ)
                               │ + oxifed REST API
                    ┌──────────▼───────────────────┐
                    │  akh-medu                     │
                    │  (app in oxifed ecosystem)    │
                    │                               │
                    │  OxifedChannel ──→ CommChannel │
                    │  AMQP consumer ← inbox events │
                    │  AMQP producer → outbox posts  │
                    │  REST client  → user/profile   │
                    └──────────────────────────────┘
```

**Design**:

### Akh-medu as an Oxifed App
The akh registers as an application within oxifed's domain:
- **User/actor**: Created via `oxiadm` or oxifed's API — the akh gets a user identity (e.g., `@akh@yourdomain.example`)
- **Profile**: Generated from KG state (name, summary from `self:description`, interests as tags), pushed to oxifed via API
- **No AP protocol handling**: Akh-medu never sees HTTP signatures, WebFinger, or raw AP JSON — oxifed abstracts all of that

### AMQP Message Bus Integration
Akh-medu connects to the same RabbitMQ/LavinMQ instance that oxifed uses:

**Consuming (inbox events)**:
- Subscribe to a queue for the akh's actor (e.g., `inbox.akh-actor-id`)
- Oxifed's `domainservd` routes incoming activities to this queue after AP processing
- The akh receives pre-parsed activity payloads — mentions, follows, replies, likes, questions

**Publishing (outbox)**:
- Publish activity payloads to an outbox exchange (e.g., `outbox.akh-actor-id`)
- Oxifed's `publisherd` picks them up, signs them, and delivers to remote inboxes
- The akh constructs activity content (Note text, Article body) but oxifed handles delivery, retries, and signatures

```rust
pub struct OxifedChannel {
    amqp_connection: lapin::Connection,  // or amqprs
    inbox_consumer: Consumer,
    outbox_publisher: Publisher,
    api_client: reqwest::Client,         // For oxifed REST API
    actor_id: String,
    domain: String,
    capabilities: ChannelCapabilities,   // Social preset
}

impl CommChannel for OxifedChannel {
    fn channel_kind(&self) -> ChannelKind { ChannelKind::Social }
    fn capabilities(&self) -> &ChannelCapabilities { &self.capabilities }

    fn receive(&mut self) -> ChannelResult<Option<InboundMessage>> {
        // Poll AMQP inbox queue, convert oxifed activity → InboundMessage
    }

    fn send(&self, msg: &OutboundMessage) -> ChannelResult<()> {
        // Convert OutboundMessage → oxifed activity payload, publish to AMQP outbox
    }
}
```

### Oxifed REST API Usage
For operations beyond message passing:
- **Profile management**: `PUT /api/users/{id}/profile` — update the akh's bio, avatar, interests
- **User lookup**: `GET /api/users/{handle}` — resolve remote actors to interlocutor profiles
- **Timeline reading** (future 12e+): `GET /api/timelines/home` — consume the akh's home timeline to stay informed about topics of interest

### Activity → InboundMessage Bridge
Incoming activities from the AMQP queue are converted:

| Oxifed Event | InboundMessage Mapping |
|---|---|
| Mention/Reply (Note) | `MessageContent::Text` — parsed through grammar, social-channel input |
| Follow | Create `InterlocutorProfile` with `Social` trust, acknowledge via AMQP |
| Question | `MessageContent::Text` — parsed as query |
| Like / Boost | `MessageContent::Reaction` — update engagement metrics |
| Unfollow | Deactivate interlocutor profile |

### OutboundMessage → Oxifed Activity Bridge
Constraint-checked responses are published as AMQP messages:
- **Notes**: Prose linearized via grammar, published as `Create(Note)` activity
- **Articles**: Longer reasoning (Full detail mode), published as `Create(Article)`
- **Replies**: Threaded via `inReplyTo` from the original activity's URI

### What Gets Published
Only constraint-checked (12c), grounded responses:
- Replies to mentions (after full constraint pipeline)
- Periodic reasoning journal entries (opt-in, Winter-style blog)
- Goal proposals marked for public deliberation
- Interest discovery ("I found that X and Y share interest in Z")

### What Doesn't Get Published
- Operator-only conversations
- Raw KG mutations
- Failed constraint checks (logged internally for operator review)
- Anything involving entities marked `has-sensitivity:private`

### Future: Timeline Consumption (12e+)
In a later iteration, the akh can read its home timeline via oxifed's API:
- Poll timeline periodically or subscribe to a timeline AMQP queue
- Parse posts through the grammar module
- Extract entities and assertions → add to KG as low-confidence observations
- Discover trending topics in the akh's social circle → feed into goal generation (11a) as curiosity signals
- Build richer interlocutor interest profiles from what people post about

### Rate Limiting
Communication budget (from 12c) is critical. Default: 10 messages per 15-minute window. The operator adjusts via the operator channel. Oxifed may impose its own rate limits — respect both.

**Changes**:
- [ ] New file: `src/agent/oxifed.rs` — `OxifedChannel`, AMQP consumer/publisher, activity ↔ message bridges
- [ ] `Cargo.toml` — add `lapin` (or `amqprs`) for AMQP, `reqwest` for REST (feature-gated: `oxifed`)
- [ ] `src/agent/channel.rs` — register OxifedChannel in `ChannelRegistry`
- [ ] `src/agent/interlocutor.rs` — create profiles from Follow events
- [ ] `src/main.rs` — CLI: `agent federate --domain X --actor Y --amqp-url Z` to join the oxifed system
- [ ] Configuration: AMQP URL, oxifed API base URL, domain, actor identity

**Depends on**: 12a (channel abstraction), 12c (constraint checking), 12d (interlocutor model), running oxifed instance

**Estimated scope**: ~600–900 lines

---

## Phase 12f — Transparent Reasoning and Explanations

**Problem**: The provenance ledger exists but isn't wired into conversational "why?" queries. The agent can't explain itself.

**Design**:

### Explanation Queries
The operator (and trusted channels) can ask:

- **"Why X?"** → Trace provenance for triple X, return the derivation chain as prose
- **"How did you decide X?"** → Retrieve the `Decision` from the relevant OODA cycle, show tool scoring breakdown
- **"What do you know about X?"** → Enumerate all triples involving entity X with confidence and provenance
- **"How confident are you about X?"** → Show confidence score + the evidence chain that produced it
- **"What changed since last time?"** → Diff KG state between current and last session summary

### Provenance-to-Prose Pipeline
```
provenance_id
  → ProvenanceRecord (derived_symbol, kind, confidence, sources, depth)
    → Recursively resolve source records
      → Build derivation tree (AbsTree nodes)
        → Linearize via grammar
          → "X because Y, which was inferred from Z (confidence 0.87)"
```

### Derivation Visualization
For the TUI operator channel, render derivation trees as indented hierarchies:
```
[0.92] concept:rust is-a programming-language
  └─ [0.95] Inferred via spreading activation from:
       ├─ [1.00] concept:rust has-paradigm systems-programming (asserted)
       └─ [0.88] concept:systems-programming is-a programming-paradigm
            └─ [0.88] Derived from library ingest: "The Rust Book" chunk 3
```

### Reasoning Journal (Winter-inspired)
Optionally publish the agent's reasoning process as structured entries:
- Each OODA cycle produces a journal entry: what was observed, what was decided, why, what happened
- Journal entries are `AkhMessage::Reasoning` variants with provenance
- On the operator channel: displayed inline
- On the AP channel: published as periodic blog-style Articles (if configured)

**Changes**:
- [ ] New file: `src/agent/explain.rs` — provenance-to-prose pipeline, derivation tree building
- [ ] `src/agent/nlp.rs` — recognize "why", "how", "what changed" as explanation queries
- [ ] `src/agent/conversation.rs` — route explanation queries through the explain module
- [ ] `src/tui/widgets.rs` — derivation tree rendering for Full detail mode
- [ ] `src/provenance.rs` — add `build_derivation_tree()` utility

**Estimated scope**: ~500–700 lines

---

## Phase 12g — Multi-Agent Communication

**Problem**: The agent can talk to humans but not to other agents in a structured way. Agent-to-agent interaction requires capability-secured message passing.

**Design**:

### Agent-to-Agent Protocol
A lightweight protocol for structured agent communication, inspired by Goblins/OCapN but simpler:

- **Capability tokens**: When an operator introduces two agents, each receives a capability token scoped to specific permissions (e.g., "you may query this agent's KG about topic X")
- **Message types**: Query, Assert (propose), Subscribe (to topic), Unsubscribe
- **Transport**: Initially over oxifed (agents as AP actors in the same or different oxifed instances, communicating via federation). Future: direct WebSocket or OCapN.

### Structured Interaction Patterns
- **Knowledge exchange**: Agent A queries Agent B about topic X. B responds with grounded triples + provenance.
- **Goal delegation**: Agent A proposes a goal to Agent B (if B's channel has `can_propose_goals`). B's operator must approve.
- **Collaborative reasoning**: Two agents share triples about a topic, each running their own inference. Results are exchanged and merged with confidence fusion.

### Trust Bootstrapping
When agent A first encounters agent B:
1. Both are `Public` trust level
2. If their operators explicitly introduce them (via a signed capability token), promote to `Trusted`
3. Trust can be attenuated: "Agent B may query topics tagged 'public-research' only"

### VSA for Agent Similarity
Encode each agent's knowledge profile as a bundled hypervector of its top concepts. Agents with similar knowledge profiles are natural collaboration partners. The akh can suggest: "Agent B has deep knowledge about X, which is relevant to your current goal."

**Changes**:
- [ ] New file: `src/agent/multi_agent.rs` — capability tokens, structured message types, trust bootstrapping
- [ ] `src/agent/oxifed.rs` — extend oxifed bridge for agent-to-agent structured messages (via structured Note content or AP extensions)
- [ ] `src/agent/channel.rs` — `AgentChannel` implementation with `Trusted` preset
- [ ] `src/agent/interlocutor.rs` — distinguish human vs agent interlocutors

**Depends on**: 12e (oxifed federation), 12d (interlocutor model)

**Estimated scope**: ~500–700 lines

---

## Implementation Order

```
12a (Channel Abstraction) ──→ 12b (Grounded Dialogue) ──→ 12f (Explanations)
         │                                                       │
         │                    12c (Constraint Checking) ←────────┘
         │                         │
         ▼                         ▼
12d (Social KG + ToM) ──→ 12e (Federation via Oxifed) ──→ 12g (Multi-Agent)
```

**Wave 1**: 12a (channel abstraction) — the foundation; refactors existing TUI/REPL into the protocol model
**Wave 2**: 12b (grounded dialogue) + 12c (constraint checking) — the operator experience
**Wave 3**: 12d (social KG) + 12f (explanations) — understanding interlocutors, explaining reasoning
**Wave 4**: 12e (federation via oxifed) — the akh joins the fediverse as an oxifed app
**Wave 5**: 12g (multi-agent) — agent-to-agent collaboration

## Total Estimated Scope

~4,000–5,600 lines across 7 sub-phases.

## Relationship to Prior Phases

Phase 12 is where akh-medu **meets the world**:
- **Phase 9a** (microtheories) → per-interlocutor theory-of-mind contexts
- **Phase 9c** (TMS) → constraint checking traces contradictions before communication
- **Phase 9e** (argumentation) → "why?" explanations backed by argument chains
- **Phase 9l** (contradiction detection) → pre-communication consistency validation
- **Phase 11a** (goal generation) → social interactions generate goal proposals
- **Phase 11c** (argumentation priority) → goal proposals from external channels argued against existing goals
- **Phase 11f** (metacognition) → self-awareness about communication quality and failure patterns
- **Phase 8 agent infrastructure** → OODA loop, grammar, entity resolution, provenance are the substrate
- **Oxifed** → AP federation server; akh-medu is an app consuming oxifed's AMQP bus and REST API

## Research Grounding

| Sub-phase | Primary Inspiration |
|-----------|-------------------|
| 12a Channel Abstraction | Goblins/OCapN capability security, object capability model |
| 12b Grounded Dialogue | Existing grammar module, Sussman-Radul propagator model (monotonic accumulation) |
| 12c Constraint Checking | Winter's Datalog self-check, extended with full inference stack |
| 12d Social KG / ToM | Cyc theory of mind (desideratum #6), microtheories, VSA interest bundling |
| 12e Oxifed Federation | Winter on Bluesky/ATProto, oxifed (AMQP + REST app model), Lemmer-Webber's ActivityPub + capability security vision |
| 12f Explanations | Provenance ledger (existing), Cyc explanation (desideratum #1) |
| 12g Multi-Agent | Goblins distributed objects, OCapN capability tokens, FIPA agent communication |

This phase transforms akh-medu from an engine that reasons in isolation into an agent that **communicates, explains, and collaborates** — with its operator as the trusted steward, and the federated social web as its broader context.
