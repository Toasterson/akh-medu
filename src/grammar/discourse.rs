//! Discourse-aware response generation.
//!
//! Resolves pronouns, determines point-of-view, classifies query focus,
//! and builds [`AbsTree::DiscourseFrame`] nodes that carry conversational
//! metadata through the grammar pipeline.
//!
//! ## Worked Example
//!
//! "Who are you?" → resolve "you" via `refers-to` chain → "self" (first person)
//! → focus: Identity → prioritize `is-a`, `has-name` triples
//! → wrap in `DiscourseFrame { FirstPerson, Identity, Conjunction[...] }`
//! → narrative grammar → "I am Akh, powered by Akh-Medu."

use std::collections::HashSet;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::agent::nlp::QuestionWord;
use crate::agent::synthesize::is_metadata_label;
use crate::engine::Engine;
use crate::grammar::abs::AbsTree;
use crate::grammar::bridge::triple_to_abs;
use crate::graph::Triple;
use crate::symbol::SymbolId;

/// Point of view for the response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PointOfView {
    /// "I am..." — subject is self-referential.
    FirstPerson,
    /// "You are..." — subject is the interlocutor.
    SecondPerson,
    /// "X is..." — subject is a third-party entity.
    ThirdPerson,
}

/// What kind of information the query is seeking.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryFocus {
    /// "Who/What is X?" — identity, classification, nature.
    Identity,
    /// "What does X mean?" — definition or explanation.
    Definition,
    /// "How does X work?" — method, process, mechanism.
    Method,
    /// "Why does X?" — cause, reason, motivation.
    Cause,
    /// "Where is X?" — location, spatial context.
    Location,
    /// "When did X?" — temporal context.
    Time,
    /// "Is X a Y?" / "Does X have Y?" — yes/no confirmation.
    Confirmation,
    /// "What can X do?" — capability, ability, affordances.
    Capability,
    /// Catch-all for unclassified queries.
    General,
}

/// Resolved discourse context for a query.
#[derive(Debug, Clone)]
pub struct DiscourseContext {
    /// The resolved subject label (after pronoun resolution).
    pub resolved_subject: String,
    /// The resolved subject's SymbolId in the KG.
    pub subject_id: SymbolId,
    /// The original subject string from the user's query.
    pub original_subject: String,
    /// Whether pronoun resolution was applied.
    pub pronoun_resolved: bool,
    /// Point of view for the response.
    pub pov: PointOfView,
    /// What kind of information is sought.
    pub focus: QueryFocus,
    /// The question word, if any.
    pub question_word: Option<QuestionWord>,
    /// The original user input.
    pub original_input: String,
}

/// Errors from the discourse resolution pipeline.
#[derive(Debug, Error, Diagnostic)]
pub enum DiscourseError {
    #[error("could not resolve pronoun \"{pronoun}\" — no `refers-to` chain found")]
    #[diagnostic(
        code(akh::discourse::unresolved_pronoun),
        help("Ensure the discourse skillpack is loaded, which defines `you refers-to self`.")
    )]
    UnresolvedPronoun { pronoun: String },

    #[error("subject \"{subject}\" not found in the knowledge graph")]
    #[diagnostic(
        code(akh::discourse::subject_not_found),
        help("The subject could not be resolved to a known symbol. Try asserting it first.")
    )]
    SubjectNotFound { subject: String },

    #[error(transparent)]
    #[diagnostic(transparent)]
    Engine(#[from] Box<crate::error::AkhError>),
}

/// Result type for discourse operations.
pub type DiscourseResult<T> = std::result::Result<T, DiscourseError>;

/// Well-known predicate labels used in discourse resolution.
const REFERS_TO: &str = "refers-to";
const SELF_LABEL: &str = "self";

/// Predicates that indicate identity (prioritized for Identity focus).
const IDENTITY_PREDICATES: &[&str] = &["is-a", "has-name", "powered-by", "named"];

/// Predicates that should be deprioritized (infrastructure/state).
const DEPRIORITIZED_PREDICATES: &[&str] = &["has-state", "has-status", "refers-to"];

/// Labels that indicate discourse infrastructure triples (not user-facing).
const INFRASTRUCTURE_PREDICATES: &[&str] = &[
    "asks-about",
    "is-question-word",
    "discourse-type",
    "has-discourse-role",
    "discourse:response-detail",
];

/// Configurable verbosity for discourse responses.
///
/// Stored as a KG triple: `self discourse:response-detail normal`.
/// Users can change at runtime by asserting a new value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ResponseDetail {
    /// Max ~3 triples — identity only, very terse.
    Concise,
    /// Max ~8 triples — balanced default.
    #[default]
    Normal,
    /// No cap — include everything that passes filters.
    Full,
}

impl ResponseDetail {
    /// Maximum number of triples to include in the response.
    pub fn max_triples(self) -> usize {
        match self {
            Self::Concise => 3,
            Self::Normal => 8,
            Self::Full => usize::MAX,
        }
    }

    /// Parse from a label string (case-insensitive). Unknown values default to `Normal`.
    pub fn from_label(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "concise" => Self::Concise,
            "normal" => Self::Normal,
            "full" => Self::Full,
            _ => Self::default(),
        }
    }
}

/// Category of a predicate, used to group related facts in responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum PredicateCategory {
    /// is-a, has-name, named — core identity facts.
    Identity,
    /// powered-by — power source / engine.
    Power,
    /// has-role — assigned roles.
    Role,
    /// has-capability — what the entity can do.
    Capability,
    /// Anything not in the above categories.
    Other,
    /// has-state, has-status — lowest priority state info.
    State,
}

/// Classify a predicate label into a category for grouping.
fn categorize_predicate(pred_label: &str) -> PredicateCategory {
    match pred_label {
        "is-a" | "has-name" | "named" => PredicateCategory::Identity,
        "powered-by" => PredicateCategory::Power,
        "has-role" => PredicateCategory::Role,
        "has-capability" => PredicateCategory::Capability,
        "has-state" | "has-status" => PredicateCategory::State,
        _ => PredicateCategory::Other,
    }
}

/// Resolve the `discourse:response-detail` level from the KG.
///
/// Looks up the triple `<subject> discourse:response-detail <value>` and
/// parses the object label into a [`ResponseDetail`].
fn resolve_response_detail(subject_id: SymbolId, engine: &Engine) -> ResponseDetail {
    let pred_id = match engine.resolve_symbol("discourse:response-detail") {
        Ok(id) => id,
        Err(_) => return ResponseDetail::default(),
    };
    let objects = engine.knowledge_graph().objects_of(subject_id, pred_id);
    objects
        .first()
        .map(|&obj_id| {
            let label = engine.resolve_label(obj_id);
            ResponseDetail::from_label(&label)
        })
        .unwrap_or_default()
}

/// Resolve discourse context from a query subject and question word.
///
/// Walks `refers-to` chains in the KG to resolve pronouns, determines POV,
/// and classifies query focus. Falls back gracefully if the discourse
/// skillpack isn't loaded.
pub fn resolve_discourse(
    subject: &str,
    question_word: Option<QuestionWord>,
    original_input: &str,
    engine: &Engine,
    capability_signal: bool,
) -> DiscourseResult<DiscourseContext> {
    let original_subject = subject.to_string();

    // Step 1: Try to resolve the subject directly.
    let (resolved_label, subject_id, pronoun_resolved) =
        resolve_pronoun_chain(subject, engine)?;

    // Step 2: Determine POV.
    let pov = determine_pov(&resolved_label, pronoun_resolved);

    // Step 3: Classify query focus from question word and capability signal.
    let focus = classify_focus_with_modal(question_word, capability_signal);

    Ok(DiscourseContext {
        resolved_subject: resolved_label,
        subject_id,
        original_subject,
        pronoun_resolved,
        pov,
        focus,
        question_word,
        original_input: original_input.to_string(),
    })
}

/// Walk `refers-to` chains to resolve pronouns.
///
/// Returns (resolved_label, symbol_id, was_pronoun_resolved).
fn resolve_pronoun_chain(
    subject: &str,
    engine: &Engine,
) -> DiscourseResult<(String, SymbolId, bool)> {
    // Try direct resolution first.
    let subject_id = engine
        .resolve_symbol(subject)
        .map_err(|_| DiscourseError::SubjectNotFound {
            subject: subject.to_string(),
        })?;

    // Look for a `refers-to` chain from this subject.
    let refers_to_id = engine.resolve_symbol(REFERS_TO).ok();

    if let Some(ref_pred) = refers_to_id {
        let triples = engine.triples_from(subject_id);
        for triple in &triples {
            if triple.predicate == ref_pred {
                // Found a refers-to target — follow one hop.
                let target_label = engine.resolve_label(triple.object);
                return Ok((target_label, triple.object, true));
            }
        }
    }

    // No pronoun resolution needed — use subject as-is.
    let label = engine.resolve_label(subject_id);
    Ok((label, subject_id, false))
}

/// Determine point-of-view based on resolved subject.
fn determine_pov(resolved_label: &str, pronoun_resolved: bool) -> PointOfView {
    let lower = resolved_label.to_lowercase();

    // If the resolved subject is "self", this is first person.
    if lower == SELF_LABEL {
        return PointOfView::FirstPerson;
    }

    // If pronoun resolution happened but didn't land on "self",
    // it's second person (talking about the user).
    if pronoun_resolved {
        return PointOfView::SecondPerson;
    }

    PointOfView::ThirdPerson
}

/// Map question word to query focus.
fn classify_focus(question_word: Option<QuestionWord>) -> QueryFocus {
    match question_word {
        Some(QuestionWord::Who) | Some(QuestionWord::What) => QueryFocus::Identity,
        Some(QuestionWord::How) => QueryFocus::Method,
        Some(QuestionWord::Why) => QueryFocus::Cause,
        Some(QuestionWord::Where) => QueryFocus::Location,
        Some(QuestionWord::When) => QueryFocus::Time,
        Some(QuestionWord::Which) => QueryFocus::Definition,
        Some(QuestionWord::YesNo) => QueryFocus::Confirmation,
        None => QueryFocus::General,
    }
}

/// Map question word to query focus, overriding with `Capability` when a
/// capability modal (e.g., "can", "peut") is detected.
pub fn classify_focus_with_modal(
    question_word: Option<QuestionWord>,
    capability_signal: bool,
) -> QueryFocus {
    if capability_signal {
        return QueryFocus::Capability;
    }
    classify_focus(question_word)
}

/// Build a discourse-framed `AbsTree` from triples and discourse context.
///
/// Filters infrastructure triples, ranks by focus relevance, and wraps
/// in a `DiscourseFrame` node.
pub fn build_discourse_response(
    triples: &[Triple],
    ctx: &DiscourseContext,
    engine: &Engine,
) -> Option<AbsTree> {
    let registry = engine.registry();
    let detail = resolve_response_detail(ctx.subject_id, engine);

    // Dedup: track unique (subject, predicate, object) tuples.
    let mut seen = HashSet::new();

    // Carry predicate label alongside each item for later grouping.
    let mut abs_items: Vec<(AbsTree, i32, String)> = Vec::new();

    for triple in triples {
        // Dedup check.
        if !seen.insert((triple.subject, triple.predicate, triple.object)) {
            continue;
        }

        let pred_label = registry.resolve_label(triple.predicate);
        let subj_label = registry.resolve_label(triple.subject);
        let obj_label = registry.resolve_label(triple.object);

        // Filter: skip infrastructure triples.
        if INFRASTRUCTURE_PREDICATES
            .iter()
            .any(|p| pred_label == *p)
        {
            continue;
        }

        // Filter: skip metadata labels.
        if is_metadata_label(&pred_label)
            || is_metadata_label(&subj_label)
            || is_metadata_label(&obj_label)
        {
            continue;
        }

        // Filter: skip self-referential refers-to (already resolved).
        if pred_label == REFERS_TO {
            continue;
        }

        // Filter: skip triples with unresolved sym:N labels.
        if subj_label.starts_with("sym:") || subj_label.starts_with("Sym:")
            || pred_label.starts_with("sym:") || pred_label.starts_with("Sym:")
            || obj_label.starts_with("sym:") || obj_label.starts_with("Sym:")
        {
            continue;
        }

        let abs = triple_to_abs(triple, registry);
        let score = score_triple_for_focus(&pred_label, &ctx.focus);
        abs_items.push((abs, score, pred_label));
    }

    // Filter: exclude negatively-scored triples.
    abs_items.retain(|(_, score, _)| *score >= 0);

    if abs_items.is_empty() {
        return None;
    }

    // Sort by relevance score (highest first).
    abs_items.sort_by(|a, b| b.1.cmp(&a.1));

    // Truncate to the configured response detail level.
    abs_items.truncate(detail.max_triples());

    // Group by predicate category, preserving score order within each group.
    abs_items.sort_by(|a, b| {
        let cat_a = categorize_predicate(&a.2);
        let cat_b = categorize_predicate(&b.2);
        cat_a.cmp(&cat_b).then(b.1.cmp(&a.1))
    });

    let items: Vec<AbsTree> = abs_items.into_iter().map(|(tree, _, _)| tree).collect();

    let inner = if items.len() == 1 {
        items.into_iter().next().unwrap()
    } else {
        AbsTree::Conjunction {
            items,
            is_and: true,
        }
    };

    Some(AbsTree::DiscourseFrame {
        pov: ctx.pov,
        focus: ctx.focus.clone(),
        inner: Box::new(inner),
    })
}

/// Score a triple's relevance to the query focus.
/// Higher scores appear first in the response.
fn score_triple_for_focus(predicate_label: &str, focus: &QueryFocus) -> i32 {
    let base = match focus {
        QueryFocus::Identity => {
            if IDENTITY_PREDICATES.iter().any(|p| predicate_label == *p) {
                10
            } else if DEPRIORITIZED_PREDICATES
                .iter()
                .any(|p| predicate_label == *p)
            {
                -5
            } else {
                0
            }
        }
        QueryFocus::Location => {
            if predicate_label == "located-in" || predicate_label == "part-of" {
                10
            } else {
                0
            }
        }
        QueryFocus::Method => {
            if predicate_label.contains("method")
                || predicate_label.contains("process")
                || predicate_label == "has-capability"
            {
                10
            } else {
                0
            }
        }
        QueryFocus::Capability => {
            if predicate_label == "has-capability" {
                10
            } else if IDENTITY_PREDICATES.iter().any(|p| predicate_label == *p) {
                3
            } else if DEPRIORITIZED_PREDICATES
                .iter()
                .any(|p| predicate_label == *p)
            {
                -5
            } else {
                0
            }
        }
        _ => 0,
    };

    // Boost is-a universally (it's almost always relevant).
    if predicate_label == "is-a" {
        base + 5
    } else {
        base
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::symbol::SymbolKind;
    use crate::vsa::Dimension;

    #[test]
    fn classify_focus_who() {
        assert_eq!(
            classify_focus(Some(QuestionWord::Who)),
            QueryFocus::Identity
        );
    }

    #[test]
    fn classify_focus_how() {
        assert_eq!(classify_focus(Some(QuestionWord::How)), QueryFocus::Method);
    }

    #[test]
    fn classify_focus_none() {
        assert_eq!(classify_focus(None), QueryFocus::General);
    }

    #[test]
    fn classify_focus_yes_no() {
        assert_eq!(
            classify_focus(Some(QuestionWord::YesNo)),
            QueryFocus::Confirmation
        );
    }

    #[test]
    fn classify_focus_with_modal_capability() {
        assert_eq!(
            classify_focus_with_modal(Some(QuestionWord::What), true),
            QueryFocus::Capability
        );
    }

    #[test]
    fn classify_focus_with_modal_no_signal() {
        assert_eq!(
            classify_focus_with_modal(Some(QuestionWord::What), false),
            QueryFocus::Identity
        );
    }

    #[test]
    fn score_capability_has_capability() {
        assert_eq!(
            score_triple_for_focus("has-capability", &QueryFocus::Capability),
            10
        );
    }

    #[test]
    fn score_capability_identity_predicate() {
        // Identity predicates get a moderate boost under Capability focus.
        assert!(score_triple_for_focus("is-a", &QueryFocus::Capability) > 0);
    }

    #[test]
    fn score_capability_deprioritized() {
        assert!(score_triple_for_focus("has-state", &QueryFocus::Capability) < 0);
    }

    #[test]
    fn pov_self_is_first_person() {
        assert_eq!(determine_pov("self", true), PointOfView::FirstPerson);
    }

    #[test]
    fn pov_third_party() {
        assert_eq!(determine_pov("dog", false), PointOfView::ThirdPerson);
    }

    #[test]
    fn score_identity_is_a() {
        assert!(score_triple_for_focus("is-a", &QueryFocus::Identity) > 0);
    }

    #[test]
    fn score_identity_deprioritized() {
        assert!(score_triple_for_focus("has-state", &QueryFocus::Identity) < 0);
    }

    // --- ResponseDetail unit tests ---

    #[test]
    fn response_detail_max_triples() {
        assert_eq!(ResponseDetail::Concise.max_triples(), 3);
        assert_eq!(ResponseDetail::Normal.max_triples(), 8);
        assert_eq!(ResponseDetail::Full.max_triples(), usize::MAX);
    }

    #[test]
    fn response_detail_from_label_case_insensitive() {
        assert_eq!(ResponseDetail::from_label("concise"), ResponseDetail::Concise);
        assert_eq!(ResponseDetail::from_label("NORMAL"), ResponseDetail::Normal);
        assert_eq!(ResponseDetail::from_label("Full"), ResponseDetail::Full);
        assert_eq!(ResponseDetail::from_label("unknown"), ResponseDetail::Normal);
    }

    #[test]
    fn response_detail_default_is_normal() {
        assert_eq!(ResponseDetail::default(), ResponseDetail::Normal);
    }

    // --- PredicateCategory unit tests ---

    #[test]
    fn categorize_identity_predicates() {
        assert_eq!(categorize_predicate("is-a"), PredicateCategory::Identity);
        assert_eq!(categorize_predicate("has-name"), PredicateCategory::Identity);
    }

    #[test]
    fn categorize_orders_identity_before_capability() {
        assert!(PredicateCategory::Identity < PredicateCategory::Capability);
        assert!(PredicateCategory::Capability < PredicateCategory::State);
    }

    // --- Integration tests using Engine ---

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    fn identity_ctx(subject_id: SymbolId) -> DiscourseContext {
        DiscourseContext {
            resolved_subject: "self".to_string(),
            subject_id,
            original_subject: "you".to_string(),
            pronoun_resolved: true,
            pov: PointOfView::FirstPerson,
            focus: QueryFocus::Identity,
            question_word: Some(QuestionWord::Who),
            original_input: "Who are you?".to_string(),
        }
    }

    #[test]
    fn dedup_removes_duplicate_triples() {
        let engine = test_engine();
        let self_sym = engine.create_symbol(SymbolKind::Entity, "self").unwrap();
        let is_a = engine.create_symbol(SymbolKind::Relation, "is-a").unwrap();
        let akh = engine.create_symbol(SymbolKind::Entity, "Akh").unwrap();

        let triple = Triple::new(self_sym.id, is_a.id, akh.id);
        // Supply the same triple twice.
        let triples = vec![triple.clone(), triple];
        let ctx = identity_ctx(self_sym.id);

        let result = build_discourse_response(&triples, &ctx, &engine).unwrap();
        let labels = result.collect_labels();
        // "is-a" should appear only once (not duplicated).
        let is_a_count = labels.iter().filter(|&&l| l == "is-a").count();
        assert_eq!(is_a_count, 1);
    }

    #[test]
    fn sym_n_labels_filtered() {
        let engine = test_engine();
        let self_sym = engine.create_symbol(SymbolKind::Entity, "self").unwrap();
        let is_a = engine.create_symbol(SymbolKind::Relation, "is-a").unwrap();
        let akh = engine.create_symbol(SymbolKind::Entity, "Akh").unwrap();

        // Create a triple with a good predicate and known labels.
        let good = Triple::new(self_sym.id, is_a.id, akh.id);

        // Create a SymbolId that won't resolve (simulate unresolved sym:N).
        // Use an ID that's much higher than anything registered.
        let bad_id = SymbolId::new(99999).unwrap();
        let bad_triple = Triple::new(self_sym.id, is_a.id, bad_id);

        let ctx = identity_ctx(self_sym.id);
        let triples = vec![bad_triple, good];
        let result = build_discourse_response(&triples, &ctx, &engine).unwrap();
        let labels = result.collect_labels();
        // The bad triple's unresolvable object should be filtered out.
        assert!(!labels.iter().any(|l| l.starts_with("sym:")));
    }

    #[test]
    fn negative_score_triples_excluded_for_identity() {
        let engine = test_engine();
        let self_sym = engine.create_symbol(SymbolKind::Entity, "self").unwrap();
        let is_a = engine.create_symbol(SymbolKind::Relation, "is-a").unwrap();
        let akh = engine.create_symbol(SymbolKind::Entity, "Akh").unwrap();
        let has_state = engine.create_symbol(SymbolKind::Relation, "has-state").unwrap();
        let active = engine.create_symbol(SymbolKind::Entity, "active").unwrap();

        let good = Triple::new(self_sym.id, is_a.id, akh.id);
        let bad = Triple::new(self_sym.id, has_state.id, active.id);

        let ctx = identity_ctx(self_sym.id);
        let triples = vec![good, bad];
        let result = build_discourse_response(&triples, &ctx, &engine).unwrap();
        let labels = result.collect_labels();
        // has-state should be excluded (score < 0 for Identity focus).
        assert!(!labels.contains(&"has-state"));
        assert!(labels.contains(&"is-a"));
    }

    #[test]
    fn response_detail_from_graph_truncates() {
        let engine = test_engine();
        let self_sym = engine.create_symbol(SymbolKind::Entity, "self").unwrap();
        let detail_pred = engine
            .create_symbol(SymbolKind::Relation, "discourse:response-detail")
            .unwrap();
        let concise_val = engine
            .create_symbol(SymbolKind::Entity, "concise")
            .unwrap();

        // Assert the detail triple into the KG.
        let detail_triple = Triple::new(self_sym.id, detail_pred.id, concise_val.id);
        engine.add_triple(&detail_triple).unwrap();

        // Create 5 identity triples.
        let is_a = engine.create_symbol(SymbolKind::Relation, "is-a").unwrap();
        let mut triples = Vec::new();
        for i in 0..5 {
            let obj = engine
                .create_symbol(SymbolKind::Entity, &format!("thing-{i}"))
                .unwrap();
            triples.push(Triple::new(self_sym.id, is_a.id, obj.id));
        }

        let ctx = identity_ctx(self_sym.id);
        let result = build_discourse_response(&triples, &ctx, &engine).unwrap();
        let labels = result.collect_labels();
        // "concise" limits to 3 triples. Each triple contributes 3 labels (subj, pred, obj).
        // Count "is-a" occurrences as a proxy.
        let is_a_count = labels.iter().filter(|&&l| l == "is-a").count();
        assert!(
            is_a_count <= 3,
            "concise should limit to 3 triples, got {is_a_count}"
        );
    }

    #[test]
    fn predicate_grouping_identity_before_capability() {
        let engine = test_engine();
        let self_sym = engine.create_symbol(SymbolKind::Entity, "self").unwrap();
        let is_a = engine.create_symbol(SymbolKind::Relation, "is-a").unwrap();
        let has_cap = engine
            .create_symbol(SymbolKind::Relation, "has-capability")
            .unwrap();
        let akh = engine.create_symbol(SymbolKind::Entity, "Akh").unwrap();
        let reason = engine.create_symbol(SymbolKind::Entity, "reasoning").unwrap();

        // Supply capability first, identity second — grouping should reorder.
        let triples = vec![
            Triple::new(self_sym.id, has_cap.id, reason.id),
            Triple::new(self_sym.id, is_a.id, akh.id),
        ];
        let ctx = identity_ctx(self_sym.id);
        let result = build_discourse_response(&triples, &ctx, &engine).unwrap();
        let labels = result.collect_labels();

        // is-a should appear before has-capability in the output.
        let is_a_pos = labels.iter().position(|l| *l == "is-a");
        let cap_pos = labels.iter().position(|l| *l == "has-capability");
        assert!(
            is_a_pos < cap_pos,
            "is-a ({is_a_pos:?}) should come before has-capability ({cap_pos:?})"
        );
    }
}
