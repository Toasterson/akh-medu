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
    /// The current user turn.
    UserPrompt,
    /// Retrieved knowledge-graph facts.
    KnowledgeGraph,
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
            ContextSection::UserPrompt => "user_prompt",
            ContextSection::KnowledgeGraph => "kg",
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
            kg_context: None,
            recalled: &[],
            session_seed: None,
            tool_catalog: &[],
        }
    }

    /// Attach prior conversation turns.
    #[must_use]
    pub fn with_conversation(mut self, conversation: Option<&'a ConversationState>) -> Self {
        self.conversation = conversation;
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

/// A unit of content the assembler considered but did **not** include, with its
/// anchoring symbol (if any) so retention/safety (44f) can classify it. A drop
/// of a `PinnedSafety` span is a hard error.
#[derive(Debug, Clone)]
pub struct DroppedSpan {
    pub section: ContextSection,
    pub text: String,
    pub anchor: Option<SymbolId>,
}

impl DroppedSpan {
    pub fn new(section: ContextSection, text: impl Into<String>, anchor: Option<SymbolId>) -> Self {
        Self {
            section,
            text: text.into(),
            anchor,
        }
    }
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
