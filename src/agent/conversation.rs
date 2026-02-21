//! Conversation state management and grounding pipeline (Phase 12b).
//!
//! Provides multi-turn discourse context with bounded turn history,
//! configurable response detail levels, and a grounding pipeline that
//! collects KG triples + provenance for query responses.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::engine::Engine;
use crate::graph::Triple;
use crate::symbol::SymbolId;

// ── ResponseDetail ───────────────────────────────────────────────────────

/// How much detail to include in agent responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResponseDetail {
    /// Just the core answer, no supporting information.
    Concise,
    /// Answer with brief rationale (default).
    Normal,
    /// Answer + provenance IDs + confidence scores + supporting triples.
    Full,
}

impl Default for ResponseDetail {
    fn default() -> Self {
        Self::Normal
    }
}

impl std::fmt::Display for ResponseDetail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Concise => write!(f, "concise"),
            Self::Normal => write!(f, "normal"),
            Self::Full => write!(f, "full"),
        }
    }
}

impl ResponseDetail {
    /// Parse from a string (case-insensitive).
    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "concise" | "brief" | "short" => Some(Self::Concise),
            "normal" | "default" | "standard" => Some(Self::Normal),
            "full" | "verbose" | "detailed" => Some(Self::Full),
            _ => None,
        }
    }
}

// ── Speaker ──────────────────────────────────────────────────────────────

/// Who produced a conversation turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Speaker {
    /// The operator / human user.
    Operator,
    /// The agent.
    Agent,
}

// ── ConversationTurn ─────────────────────────────────────────────────────

/// A single turn in the conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    /// Who spoke.
    pub speaker: Speaker,
    /// The raw text of this turn.
    pub text: String,
    /// Entity SymbolIds resolved from this turn.
    pub resolved_entities: Vec<SymbolId>,
    /// Unix timestamp (seconds).
    pub timestamp: u64,
}

// ── ConversationState ────────────────────────────────────────────────────

/// Multi-turn discourse context for a communication channel.
///
/// Tracks recent turns, active referents (entities "on the table"),
/// and the operator's preferred response detail level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationState {
    /// Which channel this conversation belongs to.
    pub channel_id: String,
    /// Bounded ring buffer of recent turns.
    pub turns: VecDeque<ConversationTurn>,
    /// Maximum number of turns to keep.
    pub max_turns: usize,
    /// Entities recently mentioned and still "on the table".
    pub active_referents: Vec<SymbolId>,
    /// Current focus entity (most recently discussed).
    pub active_topic: Option<SymbolId>,
    /// Current linearization grammar archetype.
    pub grammar: String,
    /// How much detail to include in responses.
    pub response_detail: ResponseDetail,
}

impl ConversationState {
    /// Create a new conversation state.
    pub fn new(channel_id: impl Into<String>, grammar: impl Into<String>) -> Self {
        Self {
            channel_id: channel_id.into(),
            turns: VecDeque::with_capacity(10),
            max_turns: 10,
            active_referents: Vec::new(),
            active_topic: None,
            grammar: grammar.into(),
            response_detail: ResponseDetail::default(),
        }
    }

    /// Record a turn in the conversation.
    pub fn add_turn(
        &mut self,
        speaker: Speaker,
        text: impl Into<String>,
        resolved_entities: Vec<SymbolId>,
    ) {
        // Update active referents: new entities go to front.
        if !resolved_entities.is_empty() {
            self.active_topic = Some(resolved_entities[0]);
            // Merge: prepend new entities, dedup, cap at 10.
            let mut new_referents = resolved_entities.clone();
            for r in &self.active_referents {
                if !new_referents.contains(r) {
                    new_referents.push(*r);
                }
            }
            new_referents.truncate(10);
            self.active_referents = new_referents;
        }

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let turn = ConversationTurn {
            speaker,
            text: text.into(),
            resolved_entities,
            timestamp,
        };

        self.turns.push_back(turn);
        if self.turns.len() > self.max_turns {
            self.turns.pop_front();
        }
    }

    /// Set the response detail level.
    pub fn set_detail(&mut self, level: ResponseDetail) {
        self.response_detail = level;
    }

    /// Number of recorded turns.
    pub fn len(&self) -> usize {
        self.turns.len()
    }

    /// Whether no turns have been recorded.
    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    /// Get the most recent active topic entity.
    pub fn topic(&self) -> Option<SymbolId> {
        self.active_topic
    }

    /// Convenience: record an agent turn with no resolved entities.
    pub fn record_agent_turn(&mut self, text: &str) {
        self.add_turn(Speaker::Agent, text, Vec::new());
    }

    /// Convenience: record an operator turn with no resolved entities.
    pub fn record_operator_turn(&mut self, text: &str) {
        self.add_turn(Speaker::Operator, text, Vec::new());
    }

    /// Track a subject string as an active referent (by resolving to SymbolId).
    ///
    /// This is a best-effort operation — if the subject cannot be resolved,
    /// the referent is silently skipped.
    pub fn track_referent(&mut self, _subject: String) {
        // Referent tracking requires engine access to resolve names to SymbolIds.
        // For now, the subject is tracked through the conversation turn text.
        // Full entity resolution will be wired in Phase 12c.
    }
}

impl Default for ConversationState {
    fn default() -> Self {
        Self::new("operator", "narrative")
    }
}

// ── GroundedResponse ─────────────────────────────────────────────────────

/// A response grounded in KG triples with provenance.
///
/// Produced by the grounding pipeline and consumed by the response
/// renderer to produce detail-level-appropriate output.
#[derive(Debug, Clone)]
pub struct GroundedResponse {
    /// The main prose response (linearized from triples).
    pub prose: String,
    /// Supporting triples that back this response.
    pub supporting_triples: Vec<GroundedTriple>,
    /// Aggregate confidence across supporting triples.
    pub confidence: Option<f32>,
    /// SymbolIds from the provenance ledger.
    pub provenance_ids: Vec<SymbolId>,
    /// Knowledge gaps discovered during grounding.
    pub gaps: Vec<String>,
}

/// A single triple with resolved labels and provenance metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundedTriple {
    /// Subject label.
    pub subject_label: String,
    /// Predicate label.
    pub predicate_label: String,
    /// Object label.
    pub object_label: String,
    /// Subject SymbolId (if resolved).
    pub subject_id: Option<SymbolId>,
    /// Predicate SymbolId (if resolved).
    pub predicate_id: Option<SymbolId>,
    /// Object SymbolId (if resolved).
    pub object_id: Option<SymbolId>,
    /// Confidence score [0.0, 1.0] (None if unknown).
    pub confidence: Option<f32>,
    /// How this triple was derived (human-readable tag, None if seed data).
    pub derivation_tag: Option<String>,
}

impl GroundedResponse {
    /// Render this response at the given detail level.
    pub fn render(&self, detail: ResponseDetail) -> String {
        match detail {
            ResponseDetail::Concise => {
                // Just the first line of prose.
                self.prose
                    .lines()
                    .next()
                    .unwrap_or(&self.prose)
                    .to_string()
            }
            ResponseDetail::Normal => {
                if self.supporting_triples.is_empty() {
                    return self.prose.clone();
                }
                let triple_count = self.supporting_triples.len();
                let conf_str = self
                    .confidence
                    .map(|c| format!(" (confidence: {c:.2})"))
                    .unwrap_or_default();
                format!(
                    "{}\n[grounded in {triple_count} triple(s){conf_str}]",
                    self.prose,
                )
            }
            ResponseDetail::Full => {
                let mut lines = vec![self.prose.clone()];

                if !self.supporting_triples.is_empty() {
                    lines.push(String::new());
                    lines.push("Supporting facts:".to_string());
                    for gt in &self.supporting_triples {
                        let conf = gt.confidence.unwrap_or(1.0);
                        let deriv = gt.derivation_tag.as_deref().unwrap_or("seed");
                        lines.push(format!(
                            "  - {} {} {} [confidence: {conf:.2}] ({deriv})",
                            gt.subject_label, gt.predicate_label, gt.object_label,
                        ));
                    }
                }

                if let Some(conf) = self.confidence {
                    lines.push(format!("\nAggregate confidence: {conf:.2}"));
                }

                if !self.provenance_ids.is_empty() {
                    lines.push(format!(
                        "Provenance chain: {} record(s)",
                        self.provenance_ids.len(),
                    ));
                }

                if !self.gaps.is_empty() {
                    lines.push(String::new());
                    lines.push("Knowledge gaps:".to_string());
                    for gap in &self.gaps {
                        lines.push(format!("  - {gap}"));
                    }
                }

                lines.join("\n")
            }
        }
    }
}

// ── Grounding pipeline ───────────────────────────────────────────────────

/// Ground a query in the KG: resolve entities, collect triples with
/// provenance, compose a `GroundedResponse`.
///
/// This is the core of Phase 12b — every query flows through here so
/// responses are systematically backed by KG state.
pub fn ground_query(
    subject: &str,
    engine: &Arc<Engine>,
    grammar_name: &str,
) -> Option<GroundedResponse> {
    // Step 1: Resolve the subject entity.
    let sym_id = engine.resolve_symbol(subject).ok()?;

    // Step 2: Collect all triples involving this entity.
    let from_triples = engine.triples_from(sym_id);
    let to_triples = engine.triples_to(sym_id);
    let mut all_triples = from_triples;
    all_triples.extend(to_triples);

    if all_triples.is_empty() {
        return None;
    }

    // Step 3: Resolve labels and collect provenance for each triple.
    let mut grounded = Vec::new();
    let mut provenance_ids = Vec::new();

    for triple in &all_triples {
        let subj_label = engine.resolve_label(triple.subject);
        let pred_label = engine.resolve_label(triple.predicate);
        let obj_label = engine.resolve_label(triple.object);

        // Skip agent-internal metadata.
        if super::synthesize::is_metadata_label(&pred_label)
            || super::synthesize::is_metadata_label(&obj_label)
            || super::synthesize::is_metadata_label(&subj_label)
        {
            continue;
        }

        // Collect provenance.
        let derivation = collect_derivation_tag(engine, triple);
        if let Some(prov_id) = triple.provenance_id {
            if let Some(id) = SymbolId::new(prov_id) {
                provenance_ids.push(id);
            }
        }

        grounded.push(GroundedTriple {
            subject_label: subj_label,
            predicate_label: pred_label,
            object_label: obj_label,
            subject_id: Some(triple.subject),
            predicate_id: Some(triple.predicate),
            object_id: Some(triple.object),
            confidence: Some(triple.confidence),
            derivation_tag: if derivation.is_empty() { None } else { Some(derivation) },
        });
    }

    if grounded.is_empty() {
        return None;
    }

    // Step 4: Compute aggregate confidence.
    let confidence = if grounded.is_empty() {
        None
    } else {
        let sum: f32 = grounded.iter().map(|g| g.confidence.unwrap_or(1.0)).sum();
        Some(sum / grounded.len() as f32)
    };

    // Step 5: Use existing synthesis to produce prose.
    let summary = super::synthesize::synthesize_from_triples(
        subject,
        &all_triples,
        engine,
        grammar_name,
    );

    // Step 6: Also try discourse-aware rendering for richer prose.
    let prose = if summary.overview.is_empty() {
        // Fallback: manual composition.
        grounded
            .iter()
            .map(|g| format!("{} {} {}", g.subject_label, g.predicate_label, g.object_label))
            .collect::<Vec<_>>()
            .join(". ")
    } else {
        let mut parts = vec![summary.overview];
        for section in &summary.sections {
            parts.push(format!("{}: {}", section.heading, section.prose));
        }
        parts.join("\n")
    };

    let gaps = summary.gaps;

    Some(GroundedResponse {
        prose,
        supporting_triples: grounded,
        confidence,
        provenance_ids,
        gaps,
    })
}

/// Derive a human-readable tag for how a triple was derived.
fn collect_derivation_tag(engine: &Engine, triple: &Triple) -> String {
    if let Some(prov_id) = triple.provenance_id {
        if let Some(id) = SymbolId::new(prov_id) {
            if let Ok(records) = engine.provenance_of(id) {
                if let Some(rec) = records.first() {
                    return derivation_kind_tag(&rec.kind);
                }
            }
        }
    }

    // Fallback: infer from confidence.
    if triple.confidence >= 0.99 {
        "asserted".to_string()
    } else if triple.confidence >= 0.8 {
        "high-confidence".to_string()
    } else {
        "inferred".to_string()
    }
}

/// Convert a DerivationKind to a short human-readable tag.
fn derivation_kind_tag(kind: &crate::provenance::DerivationKind) -> String {
    use crate::provenance::DerivationKind;
    match kind {
        DerivationKind::Extracted => "extracted".to_string(),
        DerivationKind::Seed => "seed knowledge".to_string(),
        DerivationKind::Reasoned => "reasoned".to_string(),
        DerivationKind::Aggregated => "aggregated".to_string(),
        DerivationKind::VsaRecovery { similarity, .. } => {
            format!("VSA similarity ({similarity:.2})")
        }
        DerivationKind::GraphEdge { .. } => "graph inference".to_string(),
        DerivationKind::Analogy { .. } => "analogy".to_string(),
        DerivationKind::RuleInference { rule_name, .. } => {
            format!("rule: {rule_name}")
        }
        DerivationKind::FusedInference { .. } => "fused inference".to_string(),
        DerivationKind::GapIdentified { .. } => "gap identified".to_string(),
        DerivationKind::AgentDecision { .. } => "agent decision".to_string(),
        DerivationKind::AgentConsolidation { .. } => "consolidation".to_string(),
        _ => format!("{kind:?}"),
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(n: u64) -> SymbolId {
        SymbolId::new(n).unwrap()
    }

    #[test]
    fn conversation_state_new() {
        let state = ConversationState::new("ch-1", "narrative");
        assert_eq!(state.channel_id, "ch-1");
        assert_eq!(state.grammar, "narrative");
        assert!(state.is_empty());
        assert_eq!(state.len(), 0);
        assert!(state.topic().is_none());
        assert!(matches!(state.response_detail, ResponseDetail::Normal));
    }

    #[test]
    fn conversation_state_default() {
        let state = ConversationState::default();
        assert_eq!(state.channel_id, "operator");
        assert_eq!(state.grammar, "narrative");
    }

    #[test]
    fn add_turn_updates_state() {
        let mut state = ConversationState::new("ch", "formal");
        state.add_turn(Speaker::Operator, "What is a dog?", vec![sym(42)]);

        assert_eq!(state.len(), 1);
        assert_eq!(state.topic(), Some(sym(42)));
        assert_eq!(state.active_referents, vec![sym(42)]);
    }

    #[test]
    fn add_turn_preserves_order() {
        let mut state = ConversationState::new("ch", "formal");
        state.add_turn(Speaker::Operator, "first", vec![sym(1)]);
        state.add_turn(Speaker::Agent, "second", vec![sym(2)]);

        assert_eq!(state.len(), 2);
        assert_eq!(state.topic(), Some(sym(2)));
        // New entities prepended, old kept.
        assert_eq!(state.active_referents, vec![sym(2), sym(1)]);
    }

    #[test]
    fn bounded_history() {
        let mut state = ConversationState::new("ch", "formal");
        state.max_turns = 5;
        for i in 1..=8 {
            state.add_turn(Speaker::Operator, format!("turn {i}"), vec![sym(i)]);
        }
        assert_eq!(state.len(), 5);
        // Oldest turn should be turn 4 (1,2,3 dropped).
        assert_eq!(state.turns.front().unwrap().text, "turn 4");
    }

    #[test]
    fn active_referents_dedup() {
        let mut state = ConversationState::new("ch", "formal");
        state.add_turn(Speaker::Operator, "first", vec![sym(1), sym(2)]);
        state.add_turn(Speaker::Operator, "second", vec![sym(2), sym(3)]);

        // sym(2) appears in both, should not be duplicated.
        assert_eq!(state.active_referents.len(), 3);
        assert_eq!(state.active_referents, vec![sym(2), sym(3), sym(1)]);
    }

    #[test]
    fn active_referents_capped() {
        let mut state = ConversationState::new("ch", "formal");
        // Add 15 unique entities across turns.
        for i in 1..=15 {
            state.add_turn(Speaker::Operator, format!("t{i}"), vec![sym(i)]);
        }
        assert!(state.active_referents.len() <= 10);
    }

    #[test]
    fn set_detail() {
        let mut state = ConversationState::new("ch", "formal");
        assert!(matches!(state.response_detail, ResponseDetail::Normal));

        state.set_detail(ResponseDetail::Full);
        assert!(matches!(state.response_detail, ResponseDetail::Full));

        state.set_detail(ResponseDetail::Concise);
        assert!(matches!(state.response_detail, ResponseDetail::Concise));
    }

    #[test]
    fn response_detail_display() {
        assert_eq!(ResponseDetail::Concise.to_string(), "concise");
        assert_eq!(ResponseDetail::Normal.to_string(), "normal");
        assert_eq!(ResponseDetail::Full.to_string(), "full");
    }

    #[test]
    fn response_detail_from_str_loose() {
        assert_eq!(
            ResponseDetail::from_str_loose("concise"),
            Some(ResponseDetail::Concise)
        );
        assert_eq!(
            ResponseDetail::from_str_loose("BRIEF"),
            Some(ResponseDetail::Concise)
        );
        assert_eq!(
            ResponseDetail::from_str_loose("verbose"),
            Some(ResponseDetail::Full)
        );
        assert_eq!(
            ResponseDetail::from_str_loose("standard"),
            Some(ResponseDetail::Normal)
        );
        assert_eq!(ResponseDetail::from_str_loose("bogus"), None);
    }

    fn test_grounded_triple(subj: &str, pred: &str, obj: &str, conf: f32, deriv: &str) -> GroundedTriple {
        GroundedTriple {
            subject_label: subj.to_string(),
            predicate_label: pred.to_string(),
            object_label: obj.to_string(),
            subject_id: None,
            predicate_id: None,
            object_id: None,
            confidence: Some(conf),
            derivation_tag: if deriv.is_empty() { None } else { Some(deriv.to_string()) },
        }
    }

    #[test]
    fn grounded_response_render_concise() {
        let resp = GroundedResponse {
            prose: "Dogs are mammals.\nThey are canine.".to_string(),
            supporting_triples: vec![test_grounded_triple("dog", "is-a", "mammal", 0.95, "asserted")],
            confidence: Some(0.95),
            provenance_ids: vec![],
            gaps: vec![],
        };
        let rendered = resp.render(ResponseDetail::Concise);
        assert_eq!(rendered, "Dogs are mammals.");
    }

    #[test]
    fn grounded_response_render_normal() {
        let resp = GroundedResponse {
            prose: "Dogs are mammals.".to_string(),
            supporting_triples: vec![test_grounded_triple("dog", "is-a", "mammal", 0.95, "asserted")],
            confidence: Some(0.95),
            provenance_ids: vec![],
            gaps: vec![],
        };
        let rendered = resp.render(ResponseDetail::Normal);
        assert!(rendered.contains("Dogs are mammals."));
        assert!(rendered.contains("1 triple(s)"));
        assert!(rendered.contains("0.95"));
    }

    #[test]
    fn grounded_response_render_full() {
        let resp = GroundedResponse {
            prose: "Dogs are mammals.".to_string(),
            supporting_triples: vec![test_grounded_triple("dog", "is-a", "mammal", 0.95, "extracted")],
            confidence: Some(0.95),
            provenance_ids: vec![sym(42)],
            gaps: vec!["diet unknown".to_string()],
        };
        let rendered = resp.render(ResponseDetail::Full);
        assert!(rendered.contains("Supporting facts:"));
        assert!(rendered.contains("dog is-a mammal"));
        assert!(rendered.contains("extracted"));
        assert!(rendered.contains("Aggregate confidence: 0.95"));
        assert!(rendered.contains("1 record(s)"));
        assert!(rendered.contains("Knowledge gaps:"));
        assert!(rendered.contains("diet unknown"));
    }

    #[test]
    fn grounded_response_render_normal_no_triples() {
        let resp = GroundedResponse {
            prose: "No grounded facts.".to_string(),
            supporting_triples: vec![],
            confidence: None,
            provenance_ids: vec![],
            gaps: vec![],
        };
        let rendered = resp.render(ResponseDetail::Normal);
        assert_eq!(rendered, "No grounded facts.");
    }

    #[test]
    fn speaker_variants() {
        assert_ne!(Speaker::Operator, Speaker::Agent);
    }

    #[test]
    fn conversation_turn_no_entities() {
        let mut state = ConversationState::new("ch", "formal");
        state.add_turn(Speaker::Operator, "hello", vec![]);

        assert_eq!(state.len(), 1);
        assert!(state.topic().is_none());
        assert!(state.active_referents.is_empty());
    }
}
