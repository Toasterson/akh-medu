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
];

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
) -> DiscourseResult<DiscourseContext> {
    let original_subject = subject.to_string();

    // Step 1: Try to resolve the subject directly.
    let (resolved_label, subject_id, pronoun_resolved) =
        resolve_pronoun_chain(subject, engine)?;

    // Step 2: Determine POV.
    let pov = determine_pov(&resolved_label, pronoun_resolved);

    // Step 3: Classify query focus from question word.
    let focus = classify_focus(question_word);

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

    // Convert triples to AbsTree nodes, filtering out infrastructure and metadata.
    let mut abs_items: Vec<(AbsTree, i32)> = Vec::new();

    for triple in triples {
        let pred_label = registry
            .get(triple.predicate)
            .map(|m| m.label.clone())
            .unwrap_or_default();
        let subj_label = registry
            .get(triple.subject)
            .map(|m| m.label.clone())
            .unwrap_or_default();
        let obj_label = registry
            .get(triple.object)
            .map(|m| m.label.clone())
            .unwrap_or_default();

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

        let abs = triple_to_abs(triple, registry);
        let score = score_triple_for_focus(&pred_label, &ctx.focus);
        abs_items.push((abs, score));
    }

    if abs_items.is_empty() {
        return None;
    }

    // Sort by relevance score (highest first).
    abs_items.sort_by(|a, b| b.1.cmp(&a.1));

    let items: Vec<AbsTree> = abs_items.into_iter().map(|(tree, _)| tree).collect();

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
}
