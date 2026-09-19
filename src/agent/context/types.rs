//! Core datatypes for the working-context assembly runtime (Phase 44a, ADR 065).
//!
//! These express the prompt as **sections and spans**, not a flat string, so the
//! budget allocator (44b) and safety enforcement (44f) can reason about what
//! enters the window and at what cost — rather than three isolated `truncate`
//! caps and a `len/4` guess (the pre-Phase-44 state).

use std::collections::BTreeMap;

use serde_json::Value;

use crate::agent::conversation::ConversationState;
use crate::engine::Engine;
use crate::library::note::Note;
use crate::symbol::SymbolId;

/// Default model context window (tokens) assumed until 44b measures the real one
/// per deployed model. Conservative; the allocator clamps to it.
pub const DEFAULT_CONTEXT_WINDOW: usize = 8_192;

/// Tokens held back from the window for the model's own output.
pub const DEFAULT_RESERVED_OUTPUT: usize = 1_024;

/// Which section of the assembled prompt a token count / span belongs to.
///
/// Declared most-essential-first; [`Ord`] follows declaration order so the
/// allocator can iterate sections in keep-first priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ContextSection {
    /// The system prompt (psyche / explicit). Assembled by the backend, counted here.
    System,
    /// The `get_graph_schema`-style digest of the akh's in-force schema (ADR
    /// 081 slice 7): every installed type + relation, so the model always
    /// knows the in-force vocabulary. Core tier — funded right after `System`.
    Schema,
    /// Frontmatter (never bodies) of notes the caller judges relevant to the
    /// current turn (ADR 081 slice 7 decision 1). Core tier.
    ActiveNotes,
    /// The current user turn.
    UserPrompt,
    /// Retrieved knowledge-graph facts.
    KnowledgeGraph,
    /// Frontmatter (never bodies) of recently-touched notes (ADR 081 slice 7
    /// decision 2), alongside recalled episodes. Recall tier.
    RecentNotes,
    /// Prior conversation turns.
    History,
    /// Recalled episodic memories.
    RecalledEpisodes,
    /// The tool catalog.
    Tools,
}

impl ContextSection {
    /// Stable lower-case label for telemetry.
    pub fn as_str(self) -> &'static str {
        match self {
            ContextSection::System => "system",
            ContextSection::Schema => "schema",
            ContextSection::ActiveNotes => "active_notes",
            ContextSection::UserPrompt => "user_prompt",
            ContextSection::KnowledgeGraph => "kg",
            ContextSection::RecentNotes => "recent_notes",
            ContextSection::History => "history",
            ContextSection::RecalledEpisodes => "recalled",
            ContextSection::Tools => "tools",
        }
    }
}

/// Identifies a context-assembly strategy (telemetry + the 44j A/B harness).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StrategyId(String);

impl StrategyId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for StrategyId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Read-only snapshot of everything a [`ContextManager`](super::manager::ContextManager)
/// may draw on to assemble one turn. Borrows the agent's live state; the manager
/// must not mutate it (curation that writes back to memory is the `PassiveManager`
/// strategy, 44h, which uses the consolidation pipeline, not this).
pub struct EpisodeState<'a> {
    /// Keys the backend's per-akh system-prompt cache.
    pub agent_id: &'a str,
    /// The current user turn text.
    pub user_prompt: &'a str,
    /// The engine, for KG context / provenance / recall lookups.
    pub engine: &'a Engine,
    /// Prior conversation turns (consumed by the history/discourse sub-phases,
    /// 44c/44e). `None` when the caller has no conversation state.
    pub conversation: Option<&'a ConversationState>,
    /// Pre-retrieved KG context for the current input (44d), rendered as a
    /// leading context block. `None` ⇒ omitted. The agent retrieves it (it owns
    /// the engine + caches); the manager only budgets + renders.
    pub kg_context: Option<&'a str>,
    /// Pre-ranked recalled episode summaries (44g), rendered as a context block.
    pub recalled: &'a [String],
    /// Session-continuity seed restored on wakeup (44e). `None` ⇒ omitted.
    pub session_seed: Option<&'a str>,
    /// The tool catalog available this turn (44b). The manager budgets it as the
    /// lowest-priority [`Tools`](ContextSection::Tools) section and returns the
    /// selected subset in [`AssembledContext::tools`]. Empty ⇒ no tools.
    pub tool_catalog: &'a [Value],
    /// A ready message array for backends that speak one, such as the native
    /// session path, where a turn carries tool calls, tool results, pinned
    /// system instructions and `neural_recall` markers that the speaker/text
    /// conversation model cannot represent. `None` for the agentic loop, which
    /// supplies [`conversation`](Self::conversation) instead. Strategies that
    /// select over messages read this; the rest ignore it.
    pub messages: Option<&'a [Value]>,
    /// Notes the caller judges **active** for the current turn (ADR 081 slice
    /// 7). Only [`Note::frontmatter`] is ever rendered — [`Note::body`] is
    /// never read by the assembler, regardless of what a caller puts here, so
    /// a body can never leak into the resident context through this field.
    ///
    /// **Minimal "active note" notion** (no persistent note store/linker
    /// exists yet — see [`crate::library::note_store`]): the caller decides
    /// relevance, e.g. notes whose wikilinks resolve to a symbol mentioned in
    /// the current conversation, or a note the user is actively editing. The
    /// context manager does not compute this itself; it only renders
    /// frontmatter for whatever the caller supplies. Empty ⇒ no active-notes
    /// block. Core tier — rendered in [`ContextSection::ActiveNotes`].
    pub active_notes: &'a [Note],
    /// Notes **recently touched** (written/read) this session, independent of
    /// topical relevance (ADR 081 slice 7 decision 2). Same frontmatter-only
    /// guarantee as [`Self::active_notes`]. Recall tier — rendered in
    /// [`ContextSection::RecentNotes`], alongside recalled episodes.
    pub recent_notes: &'a [Note],
}

impl<'a> EpisodeState<'a> {
    /// Minimal episode (no conversation / retrieved context) — the common base
    /// for callers that only have the current prompt + engine (tests, embeds).
    pub fn new(agent_id: &'a str, user_prompt: &'a str, engine: &'a Engine) -> Self {
        Self {
            agent_id,
            user_prompt,
            engine,
            conversation: None,
            messages: None,
            kg_context: None,
            recalled: &[],
            session_seed: None,
            tool_catalog: &[],
            active_notes: &[],
            recent_notes: &[],
        }
    }

    /// Attach prior conversation turns.
    #[must_use]
    pub fn with_conversation(mut self, conversation: Option<&'a ConversationState>) -> Self {
        self.conversation = conversation;
        self
    }

    /// Attach a ready message array, for backends that speak one.
    #[must_use]
    pub fn with_messages(mut self, messages: Option<&'a [Value]>) -> Self {
        self.messages = messages;
        self
    }

    /// Attach the retrieved context sections (44d KG / 44g recalled / 44e seed).
    #[must_use]
    pub fn with_context(
        mut self,
        kg_context: Option<&'a str>,
        recalled: &'a [String],
        session_seed: Option<&'a str>,
    ) -> Self {
        self.kg_context = kg_context;
        self.recalled = recalled;
        self.session_seed = session_seed;
        self
    }

    /// Attach the tool catalog for budget-driven selection (44b).
    #[must_use]
    pub fn with_tools(mut self, tool_catalog: &'a [Value]) -> Self {
        self.tool_catalog = tool_catalog;
        self
    }

    /// Attach the active-note and recently-touched-note frontmatter sources
    /// (ADR 081 slice 7 — see [`Self::active_notes`] / [`Self::recent_notes`]).
    #[must_use]
    pub fn with_notes(mut self, active_notes: &'a [Note], recent_notes: &'a [Note]) -> Self {
        self.active_notes = active_notes;
        self.recent_notes = recent_notes;
        self
    }
}

/// The token budget for one assembly: total window minus reserved output.
#[derive(Debug, Clone, Copy)]
pub struct TokenBudget {
    /// The model's full context window, in tokens.
    pub window: usize,
    /// Tokens reserved for the model's response.
    pub reserved_output: usize,
}

impl TokenBudget {
    pub fn new(window: usize, reserved_output: usize) -> Self {
        Self {
            window,
            reserved_output,
        }
    }
    /// Tokens available for the assembled context (window minus reserved output).
    pub fn available(&self) -> usize {
        self.window.saturating_sub(self.reserved_output)
    }
}

impl Default for TokenBudget {
    fn default() -> Self {
        Self {
            window: DEFAULT_CONTEXT_WINDOW,
            reserved_output: DEFAULT_RESERVED_OUTPUT,
        }
    }
}

/// Why a span was evicted from the working context (Phase 44m, ADR 065 D12).
///
/// Per-span, not per-section: `CompactingManager` can drop one section's spans
/// for two different reasons in a single turn (e.g. an atomic block that did
/// not fit the budget ⇒ `Budget`, plus an older history turn aged out by the
/// tail cap ⇒ `Age`). The per-section dominant reason is derived as a histogram
/// in [`crate::agent::context::manager::ContextWindowStats`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DropReason {
    /// Evicted because the section did not fit the token budget (the allocator
    /// denied it, whole or in part). The common reason under pressure.
    Budget,
    /// Evicted because the retention classifier refused it (a `Recoverable`
    /// span that recall can bring back is preferred over a `PinnedSafety`
    /// span — but the classifier itself never drops; this reason records the
    /// rare case the manager chose to drop *because* the retention class
    /// permitted it). Today no shipped manager emits this; it is reserved.
    Retention,
    /// Evicted by an age / tail cap (e.g. `PassiveManager`'s `tail`, which
    /// leaves older turns to recall rather than carrying them verbatim).
    Age,
    /// Evicted by a virtual-context / VSA policy (the model manages its own
    /// context window and chose to page a span out of the resident tier).
    /// Reserved for `VirtualContextManager`'s future paging work; today it
    /// does not emit this either.
    Vsa,
}

impl DropReason {
    /// Stable lower-case label for telemetry and the ACP surface.
    pub fn as_str(self) -> &'static str {
        match self {
            DropReason::Budget => "budget",
            DropReason::Retention => "retention",
            DropReason::Age => "age",
            DropReason::Vsa => "vsa",
        }
    }
}

/// A unit of content the assembler considered but did **not** include, with its
/// anchoring symbol (if any) so retention/safety (44f) can classify it. A drop
/// of a `PinnedSafety` span is a hard error.
///
/// `reason` (Phase 44m, ADR 065 D12) is per-span; the dominant reason per
/// section is derived as a histogram in [`ContextWindowStats::sections`]. See
/// ADR 065 open question 5 for the per-span-vs-dominant decision.
#[derive(Debug, Clone)]
pub struct DroppedSpan {
    pub section: ContextSection,
    pub text: String,
    pub anchor: Option<SymbolId>,
    /// Why this span was evicted. Defaults to [`DropReason::Budget`] for
    /// backward compatibility with callers that did not specify (the common
    /// reason under pressure); set explicitly by managers that age-out or
    /// VSA-page spans.
    pub reason: DropReason,
}

impl DroppedSpan {
    /// Construct a dropped span with an explicit [`DropReason`].
    pub fn with_reason(
        section: ContextSection,
        text: impl Into<String>,
        anchor: Option<SymbolId>,
        reason: DropReason,
    ) -> Self {
        Self {
            section,
            text: text.into(),
            anchor,
            reason,
        }
    }

    /// Construct a dropped span attributed to the budget (the common reason).
    /// Prefer [`DroppedSpan::with_reason`] when the manager knows the actual
    /// reason (age / retention / VSA).
    pub fn new(section: ContextSection, text: impl Into<String>, anchor: Option<SymbolId>) -> Self {
        Self::with_reason(section, text, anchor, DropReason::Budget)
    }
}

/// Per-section accounting for one assembled turn (Phase 44m, ADR 065 D12).
///
/// `tokens_in` is what the section *asked for* (the cost of the candidate span
/// before the allocator ran); `tokens_kept` is what survived; `tokens_dropped`
/// is `tokens_in - tokens_kept`. `drop_reason` is the **dominant** reason for
/// the section (the mode of the per-span `DroppedSpan::reason` histogram, with
/// ties broken in declaration order `Budget > Retention > Age > Vsa`); `None`
/// when nothing was dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SectionStats {
    pub section: ContextSection,
    pub tokens_in: usize,
    pub tokens_kept: usize,
    pub tokens_dropped: usize,
    pub drop_reason: Option<DropReason>,
}

/// One deferred-tool group for the ACP `_akh/usage` panel (Phase 44m, ADR 065
/// D12.1). A tool is "deferred" when it was in `full_catalog` (the pre-
/// retrieval-scoping catalog) but was NOT in the `tools` set actually offered
/// this round — ADR 089 M1 narrowed it. The reason distinguishes
/// `retrieval_scoped` (`AKH_AGENT_TOOL_RETRIEVAL`) from `capped`
/// (`AKH_AGENT_MAX_TOOLS`) from `unadvertised` (`ToolRegistry::unadvertised`).
/// Deferred rows carry tokens and a count but **no percent** — they are not in
/// the window.
#[derive(Debug, Clone)]
pub struct DeferredToolGroup {
    /// Row id: `tools` (System tools / Skills) or `mcp` (MCP tools).
    pub id: &'static str,
    /// Human-readable label for the row.
    pub label: String,
    /// Total tokens of the deferred tools in this group (heuristic estimate).
    pub tokens: usize,
    /// Number of deferred tools in this group.
    pub count: usize,
    /// Why these tools were deferred.
    pub reason: DeferredReason,
}

/// Why a tool was not offered this turn (Phase 44m, ADR 065 D12.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeferredReason {
    /// `AKH_AGENT_TOOL_RETRIEVAL=<k>` kept only the top-K relevant tools.
    RetrievalScoped,
    /// `AKH_AGENT_MAX_TOOLS=<cap>` truncated the catalog (legacy blind cap).
    Capped,
    /// `ToolRegistry::unadvertised` hid the tool from the catalog entirely.
    Unadvertised,
}

impl DeferredReason {
    /// Stable label for the ACP surface.
    pub fn as_str(self) -> &'static str {
        match self {
            DeferredReason::RetrievalScoped => "retrieval_scoped",
            DeferredReason::Capped => "capped",
            DeferredReason::Unadvertised => "unadvertised",
        }
    }
}

/// Normalized context-window statistics for one assembled turn (Phase 44m,
/// ADR 065 D12). Every [`ContextManager`](super::manager::ContextManager)
/// reports the same shape so the 44j A/B harness and the `_akh/usage` panel can
/// compare strategies apples-to-apples.
///
/// Strategy extras are all `Option` so a new manager is never a breaking
/// change: a caller that doesn't know about `compacted` just sees `None`.
#[derive(Debug, Clone)]
pub struct ContextWindowStats {
    /// Which strategy produced this account.
    pub strategy: StrategyId,
    /// The model's full context window, in tokens.
    pub window_total: usize,
    /// Tokens reserved for the model's response.
    pub reserved: usize,
    /// `window_total - reserved` — what the assembler may spend.
    pub available: usize,
    /// `tokens_kept / available` (0.0–1.0, NaN-free; >1.0 clamped if a manager
    /// over-spent, which is itself a signal). 1.0 = the window is full.
    pub budget_pressure: f32,
    /// One entry per section the manager considered, in `ContextSection`'s
    /// declaration order. Sections the manager never touched are omitted.
    pub sections: Vec<SectionStats>,
    /// How many spans the [`SafetyEnforced`](super::safety::SafetyEnforced)
    /// guard would have pinned (anchors a decision / safety veto / dangerous
    /// tool execution) — counted whether or not they were actually dropped.
    pub provenance_pinned_count: usize,
    /// How many `PinnedSafety` spans the guard caught being evicted. Must be 0
    /// in a healthy backend; >0 means the manager tried and the guard refused.
    pub safety_violations: usize,
    /// D10 KV-cache token count (Candle-only). `None` until the Candle-in-loop
    /// work lands a real `KvCachePolicy`.
    pub kv_cache_tokens: Option<usize>,

    // ── CompactingManager extras ──
    /// Number of history turns that were *compacted* (summarized / dropped
    /// wholesale) rather than kept verbatim. `None` for managers that do not
    /// compact.
    pub compacted: Option<usize>,
    /// `kept_tokens / requested_tokens` for the compactable section(s) —
    /// 0.0 means everything was compacted away, 1.0 nothing was. `None` for
    /// managers that do not compact.
    pub compaction_ratio: Option<f32>,

    // ── VirtualContextManager extras ──
    /// How many virtual spans the model *referenced* this turn (paged in by
    /// name). `None` for non-virtual managers.
    pub virtual_referenced: Option<usize>,
    /// How many virtual spans were actually *materialized* into the window
    /// (vs. just referenced by handle). `None` for non-virtual managers.
    pub physical_materialized: Option<usize>,
    /// `virtual_referenced / physical_materialized` hit rate (0.0–1.0).
    /// `None` for non-virtual managers.
    pub vsa_hit_rate: Option<f32>,
}

/// The result of assembling context for one turn.
#[derive(Debug)]
pub struct AssembledContext {
    /// Conversation messages to send (the system prompt stays in the backend).
    pub messages: Vec<Value>,
    /// The tool catalog to send, after budget-driven selection (44b). The
    /// pass-through manager echoes [`EpisodeState::tool_catalog`] unchanged;
    /// curating managers may trim it to the `Tools` section budget.
    pub tools: Vec<Value>,
    /// Tokens consumed per section (telemetry; should sum to ≤ `budget.available()`).
    pub section_tokens: BTreeMap<ContextSection, usize>,
    /// Spans the manager declined to include (telemetry + safety enforcement).
    pub dropped: Vec<DroppedSpan>,
}

impl AssembledContext {
    /// Total assembled tokens across all sections.
    pub fn total_tokens(&self) -> usize {
        self.section_tokens.values().copied().sum()
    }
}

/// Outcome of a completed turn, fed back via
/// [`ContextManager::commit`](super::manager::ContextManager::commit) so stateful
/// strategies can learn.
#[derive(Debug, Clone, Default)]
pub struct TurnOutcome {
    /// Whether the turn produced a final answer (vs. hit the round cap).
    pub answered: bool,
    /// Tool-call rounds the turn took.
    pub rounds: usize,
    /// Prompt tokens reported by the backend, if known (post-hoc, for calibration).
    pub prompt_tokens: Option<usize>,
    /// Completion tokens reported by the backend, if known.
    pub completion_tokens: Option<usize>,
}
