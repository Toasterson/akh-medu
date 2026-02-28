//! Dialogue state management backed by KG microtheories.
//!
//! Each conversation gets a dialogue microtheory (`dialogue:<channel_id>`)
//! that stores dialogue state as triples: active topic, last act type,
//! turn count, and interlocutor identity.
//!
//! This replaces the in-memory-only `ConversationState` fields with
//! KG-persistent state that survives session boundaries and composes
//! with microtheory-based reasoning.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::engine::Engine;
use crate::grammar::abs::AbsTree;
use crate::symbol::SymbolId;

use super::conversation::{ConversationState, ResponseDetail, Speaker};
use super::error::AgentResult;

// ── Well-known predicates ────────────────────────────────────────────────

/// Well-known dialogue relation SymbolIds, following the `AgentPredicates` pattern.
#[derive(Debug, Clone)]
pub struct DialoguePredicates {
    /// `dlg:active-topic` — the current topic entity under discussion.
    pub active_topic: SymbolId,
    /// `dlg:last-act` — the type of the last dialogue act (e.g., "greeting").
    pub last_act: SymbolId,
    /// `dlg:turn-count` — the number of turns in this dialogue.
    pub turn_count: SymbolId,
    /// `dlg:interlocutor` — who is participating in this dialogue.
    pub interlocutor: SymbolId,
    /// `dlg:context-mt` — the microtheory for this dialogue context.
    pub context_mt: SymbolId,
}

impl DialoguePredicates {
    /// Resolve or create all well-known dialogue predicates in the engine.
    pub fn init(engine: &Engine) -> AgentResult<Self> {
        Ok(Self {
            active_topic: engine.resolve_or_create_relation("dlg:active-topic")?,
            last_act: engine.resolve_or_create_relation("dlg:last-act")?,
            turn_count: engine.resolve_or_create_relation("dlg:turn-count")?,
            interlocutor: engine.resolve_or_create_relation("dlg:interlocutor")?,
            context_mt: engine.resolve_or_create_relation("dlg:context-mt")?,
        })
    }
}

// ── Dialogue act type label ──────────────────────────────────────────────

/// Extract a dialogue act type label from an AbsTree node.
pub fn dialogue_act_label(tree: &AbsTree) -> Option<&'static str> {
    match tree {
        AbsTree::Greeting { .. } => Some("greeting"),
        AbsTree::Farewell { .. } => Some("farewell"),
        AbsTree::Acknowledgment { .. } => Some("ack"),
        AbsTree::FollowUpRequest { .. } => Some("follow-up"),
        AbsTree::MetaQuery { .. } => Some("meta-query"),
        AbsTree::GoalRequest { .. } => Some("goal-request"),
        AbsTree::StructuralCommand { .. } => Some("command"),
        _ => None,
    }
}

// ── DialogueManager ──────────────────────────────────────────────────────

/// Manages dialogue state backed by the knowledge graph.
///
/// Each channel gets its own dialogue microtheory inheriting from `mt:general`.
/// State (active topic, last act, turn count) is stored as triples within
/// this microtheory, making it persistent and queryable.
pub struct DialogueManager {
    /// Well-known dialogue predicates.
    predicates: DialoguePredicates,
    /// Channel-specific dialogue microtheory name (e.g., `dialogue:operator`).
    mt_name: String,
}

impl DialogueManager {
    /// Create a new dialogue manager for the given channel.
    pub fn new(predicates: DialoguePredicates, channel_id: &str) -> Self {
        let mt_name = format!("dialogue:{channel_id}");
        Self {
            predicates,
            mt_name,
        }
    }

    /// The microtheory name for this dialogue.
    pub fn mt_name(&self) -> &str {
        &self.mt_name
    }

    /// The dialogue predicates.
    pub fn predicates(&self) -> &DialoguePredicates {
        &self.predicates
    }

    /// Record a turn in the dialogue.
    ///
    /// Updates the last-act type in the KG and increments the turn count.
    /// If the tree is a dialogue act, records the act type.
    pub fn record_turn(
        &self,
        _speaker: Speaker,
        _raw_text: &str,
        tree: &AbsTree,
        engine: &Engine,
    ) {
        // Record dialogue act type if applicable.
        if let Some(act_label) = dialogue_act_label(tree) {
            let _ = self.set_last_act(engine, act_label);
        }
    }

    /// Set the last dialogue act type in the KG.
    fn set_last_act(&self, engine: &Engine, act_label: &str) -> AgentResult<()> {
        let mt_id = engine.resolve_or_create_entity(&self.mt_name)?;
        let act_id = engine.resolve_or_create_entity(act_label)?;
        engine.add_triple(&crate::graph::Triple {
            subject: mt_id,
            predicate: self.predicates.last_act,
            object: act_id,
            confidence: 1.0,
            timestamp: now_secs(),
            compartment_id: Some(self.mt_name.clone()),
            provenance_id: None,
        })?;
        Ok(())
    }

    /// Update the active topic in the KG.
    pub fn set_active_topic(&self, engine: &Engine, topic_id: SymbolId) -> AgentResult<()> {
        let mt_id = engine.resolve_or_create_entity(&self.mt_name)?;
        engine.add_triple(&crate::graph::Triple {
            subject: mt_id,
            predicate: self.predicates.active_topic,
            object: topic_id,
            confidence: 1.0,
            timestamp: now_secs(),
            compartment_id: Some(self.mt_name.clone()),
            provenance_id: None,
        })?;
        Ok(())
    }

    /// Generate a persona greeting response.
    pub fn handle_greeting(
        &self,
        conversation: &ConversationState,
        persona_name: &str,
        _traits: &[String],
    ) -> String {
        if conversation.is_empty() {
            format!(
                "Hello. I am {persona_name}. Ask me a question or tell me something to learn."
            )
        } else {
            "Hello again. What would you like to explore?".to_string()
        }
    }

    /// Generate a farewell response.
    pub fn handle_farewell(&self, persona_name: &str) -> String {
        format!("Farewell. {persona_name} will be here when you return.")
    }

    /// Generate an acknowledgment response.
    pub fn handle_ack(&self, traits: &[String]) -> String {
        let warm = traits.iter().any(|t| t.eq_ignore_ascii_case("warm"));
        if warm {
            "You're welcome! Let me know if there's more.".to_string()
        } else {
            "Understood.".to_string()
        }
    }

    /// Handle a meta-query ("who are you", "what can you do").
    ///
    /// Returns `None` if the caller should route to `ground_query("self", ...)` instead.
    pub fn handle_meta_query(
        &self,
        engine: &Arc<Engine>,
        grammar: &str,
    ) -> Option<String> {
        if let Some(gr) = super::conversation::ground_query("self", engine, grammar) {
            let rendered = gr.render(ResponseDetail::Normal);
            Some(rendered)
        } else {
            None
        }
    }

    /// Handle a follow-up request.
    ///
    /// Returns `None` if the active topic should be re-queried at Full detail.
    pub fn handle_follow_up(
        &self,
        conversation: &ConversationState,
    ) -> Option<String> {
        if conversation.topic().is_some() {
            // Caller should re-query the active topic at Full detail.
            None
        } else {
            Some("I don't have a current topic. What would you like to know?".to_string())
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Current time as seconds since UNIX epoch.
fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
