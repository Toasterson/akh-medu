//! Phase 34f — Communicative pipeline: composition of the three stages.
//!
//! Given a grounded response (Phase 12b), this module runs the full
//! communicative-intelligence pipeline:
//!
//! 1. Build a [`StrategyContext`] from the grounded response, the discourse
//!    trajectory (34d), and the affective state (Phase 23).
//! 2. Select a response strategy (34a) — psyche-modulated symbolic scoring.
//! 3. Run the shadow filter (34e) — soften inhibited strategies + journal.
//! 4. Select content per strategy (34c).
//! 5. Compose strategy-framed prose and apply the psyche style transform
//!    (34b): opening move, register rewrite, closing move.
//!
//! Everything is deterministic and symbolic — no LLM, no network. The
//! `ChatProcessor` calls [`compose_grounded`] on the grounded path so all
//! chat surfaces (TUI, daemon WS, headless CLI, ADR 024) get the same
//! behaviour through one wiring point.
//!
//! Plan: docs/ai/plans/2026-03-26-phase34-communicative-intelligence.md
//! ADR:  docs/ai/decisions/044-communicative-intelligence.md

use crate::compartment::psyche::Psyche;
use crate::grammar::style_transform::StyleTransform;

use super::conversation::{GroundedResponse, GroundedTriple, triple_to_prose};
use super::evidence::MassFunction;
use super::response_strategy::{
    ConfidenceLevel, QueryType, ResponseStrategy, SelectedContent, Sentiment, ShadowFilter,
    ShadowFilterResult, StrategyContext, StrategySelector, evidence_conflict, select_content,
};
use super::trajectory::DiscourseTrajectory;

// ── Composed response ────────────────────────────────────────────────────

/// The outcome of the communicative pipeline for one grounded response.
#[derive(Debug, Clone)]
pub struct ComposedResponse {
    /// The final prose, styled and framed.
    pub text: String,
    /// Name of the primary strategy actually used (post shadow filter).
    pub strategy: &'static str,
    /// Name of the companion conversational move, if one rode along.
    pub companion: Option<&'static str>,
    /// Shadow journal entry when the original strategy was inhibited.
    pub shadow_journal: Option<String>,
    /// The selector's scored deliberation `(strategy, score)`, descending.
    pub deliberation: Vec<(&'static str, f32)>,
    /// The content triples actually verbalized.
    pub content: SelectedContent,
}

// ── Context construction ─────────────────────────────────────────────────

/// Classify the query type from the raw input (cheap keyword heuristic).
pub fn classify_query_type(raw_input: &str, subject: &str) -> QueryType {
    let lower = raw_input.trim().to_lowercase();
    if subject.eq_ignore_ascii_case("self") {
        return QueryType::Meta;
    }
    if lower.starts_with("why") {
        return QueryType::Causal;
    }
    if lower.starts_with("how") {
        return QueryType::Procedural;
    }
    if lower.starts_with("is there") || lower.starts_with("does ") || lower.starts_with("do ") {
        return QueryType::Existential;
    }
    QueryType::Factual
}

/// Build a [`StrategyContext`] from a grounded response plus conversation
/// signals. This is the Stage-0 glue: every signal the selector evaluates
/// is derived here from real subsystem state.
pub fn context_from_grounded(
    gr: &GroundedResponse,
    subject: &str,
    query_type: QueryType,
    trajectory: &DiscourseTrajectory,
    sentiment: Option<Sentiment>,
) -> StrategyContext {
    let confidence = gr.confidence.unwrap_or(0.5);
    let mut ctx = StrategyContext::for_answer(subject, gr.supporting_triples.len(), confidence);
    ctx.query_type = query_type;

    // Dempster-Shafer signals from triple confidences (Phase 17 primitives).
    let confidences: Vec<f32> = gr
        .supporting_triples
        .iter()
        .map(|t| t.confidence.unwrap_or(1.0))
        .collect();
    ctx.evidence_conflict = evidence_conflict(&confidences);
    let interval = MassFunction::from_confidence(confidence).belief_interval();
    ctx.belief_interval = Some((interval.belief, interval.plausibility));

    // Trajectory signals (34d).
    ctx.conversation_turns = trajectory.turn_count();
    ctx.topic_repeated = trajectory.is_repeated(subject);
    ctx.discourse_mode = trajectory.discourse_mode();
    ctx.engagement = trajectory.user_engagement();
    ctx.last_strategy = trajectory.last_strategy().map(str::to_string);

    // Affective signal (Phase 23).
    ctx.user_sentiment = sentiment;

    // Related topic: a taxonomic neighbour of the subject worth exploring.
    ctx.related_topic = find_related_topic(&gr.supporting_triples, subject);

    // Grounding gaps + provenance.
    ctx.gaps = gr.gaps.clone();
    ctx.provenance = gr.provenance_ids.clone();

    ctx
}

/// Find a taxonomic neighbour `(topic, connection)` of the subject.
fn find_related_topic(triples: &[GroundedTriple], subject: &str) -> Option<(String, String)> {
    const TAXONOMIC: [&str; 6] = [
        "is-a",
        "subclass-of",
        "part-of",
        "member-of",
        "similar-to",
        "instance-of",
    ];
    triples
        .iter()
        .filter(|t| {
            t.subject_label.eq_ignore_ascii_case(subject)
                && TAXONOMIC.contains(&t.predicate_label.as_str())
                && !t.object_label.eq_ignore_ascii_case(subject)
        })
        .max_by(|a, b| {
            a.confidence
                .unwrap_or(1.0)
                .partial_cmp(&b.confidence.unwrap_or(1.0))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.object_label.cmp(&a.object_label))
        })
        .map(|t| (t.object_label.clone(), t.predicate_label.clone()))
}

// ── Composition ──────────────────────────────────────────────────────────

/// Run stages 1–3 of the communicative pipeline over a grounded response.
///
/// Returns the composed, psyche-styled prose plus the full decision record
/// (strategy, companion, shadow journal, deliberation) for auditing.
pub fn compose_grounded(
    gr: &GroundedResponse,
    subject: &str,
    psyche: &Psyche,
    ctx: &StrategyContext,
) -> ComposedResponse {
    // Stage 1: strategy selection (34a).
    let selector = StrategySelector::from_psyche(psyche);
    let choice = selector.select(ctx);

    // Shadow filter (34e) — needs a provisional content slice to inspect.
    let provisional = select_content(&choice.primary, &gr.supporting_triples);
    let shadow = ShadowFilter::from_psyche(psyche);
    let (strategy, shadow_journal) = match shadow.filter(&choice.primary, &provisional.primary) {
        ShadowFilterResult::Pass => (choice.primary.clone(), None),
        ShadowFilterResult::Inhibited {
            replacement,
            journal_entry,
        } => (replacement, Some(journal_entry)),
    };

    // Stage 2: content selection (34c) for the final strategy.
    let content = select_content(&strategy, &gr.supporting_triples);

    // Stage 3: strategy-framed core prose + psyche style transform (34b).
    let core = strategy_core_prose(&strategy, gr, &content, subject, ctx);
    let style = StyleTransform::from_psyche(psyche, ctx.conversation_turns as u64);
    let mut text = style.apply(&core);

    // Companion conversational move (e.g. Empathy, ExplorationPrompt).
    let companion_name = choice.companion.as_ref().map(|c| c.name());
    if let Some(companion) = &choice.companion {
        if let Some(suffix) = companion_suffix(companion, subject) {
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(&suffix);
        }
    }

    ComposedResponse {
        text,
        strategy: strategy.name(),
        companion: companion_name,
        shadow_journal,
        deliberation: choice.deliberation,
        content,
    }
}

/// Sentences from selected triples.
///
/// Phase 33a: shared-subject triples are folded into fluent aggregated
/// sentences ("Dog, a mammal, has fur and is similar to wolf.") before the
/// psyche style transform (34b) is applied. With aggregation disabled
/// (`AKH_GRAMMAR_AGGREGATE=0`) this degrades to one sentence per triple.
fn triples_to_sentences(triples: &[GroundedTriple]) -> String {
    let cfg = crate::grammar::aggregate::AggregationConfig::from_env();

    let sentences: Vec<String> = if cfg.enabled {
        // Lift into AbsTree statements, aggregate, and realize neutrally.
        // Confidence is intentionally not lifted: this surface never
        // rendered per-triple confidence, and band-splitting here would
        // only fragment the aggregation.
        let items: Vec<crate::grammar::AbsTree> = triples
            .iter()
            .map(|t| {
                crate::grammar::AbsTree::triple(
                    crate::grammar::AbsTree::entity(&t.subject_label),
                    crate::grammar::AbsTree::relation(&t.predicate_label),
                    crate::grammar::AbsTree::entity(&t.object_label),
                )
            })
            .collect();
        crate::grammar::aggregate::aggregate_statements(&items, &cfg)
            .iter()
            .filter_map(crate::grammar::aggregate::realize_statement)
            .collect()
    } else {
        triples
            .iter()
            .map(|t| triple_to_prose(&t.subject_label, &t.predicate_label, &t.object_label))
            .collect()
    };

    let sentences: Vec<String> = sentences
        .into_iter()
        .map(|mut s| {
            // Sentence-case the first character.
            if let Some(first) = s.get(0..1) {
                let upper = first.to_uppercase();
                s.replace_range(0..1, &upper);
            }
            s
        })
        .collect();
    if sentences.is_empty() {
        String::new()
    } else {
        format!("{}.", sentences.join(". "))
    }
}

/// Build the core prose for a strategy (before style transform).
fn strategy_core_prose(
    strategy: &ResponseStrategy,
    gr: &GroundedResponse,
    content: &SelectedContent,
    subject: &str,
    ctx: &StrategyContext,
) -> String {
    let facts = triples_to_sentences(&content.primary);
    match strategy {
        ResponseStrategy::DirectAnswer => facts,
        ResponseStrategy::Explanation { .. } => {
            // The grounding synthesis already builds explanatory prose;
            // fall back to fact sentences when it is empty.
            if gr.prose.trim().is_empty() {
                facts
            } else {
                gr.prose.trim().to_string()
            }
        }
        ResponseStrategy::Contextualization { broader_topic } => {
            format!("In the broader context of {broader_topic}: {facts}")
        }
        ResponseStrategy::Comparison { compare_to } => {
            format!("Compared with {compare_to}: {facts}")
        }
        ResponseStrategy::Analogy { source_domain } => {
            format!("Think of it through {source_domain}. {facts}")
        }
        ResponseStrategy::ConfidentAssertion => {
            format!("{facts} This is well corroborated.")
        }
        ResponseStrategy::HedgedStatement { reason } => {
            format!("{facts} I hold this loosely — {reason}.")
        }
        ResponseStrategy::UncertaintyAdmission {
            confidence,
            what_is_missing,
        } => {
            let pct = (confidence * 100.0).round() as u32;
            let mut s = format!("I'm not certain about this — my confidence is around {pct}%.");
            if !facts.is_empty() {
                s.push(' ');
                s.push_str(&format!("What I have: {facts}"));
            }
            s.push(' ');
            s.push_str(&format!("What I'm missing: {what_is_missing}."));
            s
        }
        ResponseStrategy::EvidenceWeighing { .. } => {
            let mut s = String::from("The evidence points in different directions.");
            if !content.primary.is_empty() {
                s.push_str(&format!(
                    " Supporting: {}",
                    triples_to_sentences(&content.primary)
                ));
            }
            if !content.contrast.is_empty() {
                s.push_str(&format!(
                    " However, with lower confidence: {}",
                    triples_to_sentences(&content.contrast)
                ));
            }
            s
        }
        ResponseStrategy::KnowledgeGap { topic, can_learn } => {
            let mut s = format!("I don't know enough about {topic} yet.");
            if *can_learn {
                s.push_str(" I can investigate it if you'd like.");
            }
            s
        }
        ResponseStrategy::ClarificationRequest { ambiguity, options } => {
            let mut s = format!("Could you clarify — {ambiguity}?");
            if !options.is_empty() {
                s.push_str(&format!(" Did you mean: {}?", options.join(", ")));
            }
            s
        }
        ResponseStrategy::ExplorationPrompt {
            related_topic,
            connection,
        } => {
            format!(
                "{facts} This connects to {related_topic} (via {connection}) — want to explore that?"
            )
        }
        ResponseStrategy::Challenge { reasoning } => {
            let mut s = format!("I see this differently: {reasoning}.");
            if !content.contrast.is_empty() {
                s.push_str(&format!(
                    " Counter-evidence: {}",
                    triples_to_sentences(&content.contrast)
                ));
            } else if !facts.is_empty() {
                s.push(' ');
                s.push_str(&facts);
            }
            s
        }
        ResponseStrategy::Redirect {
            from_topic,
            to_topic,
            reason,
        } => {
            format!("I can't say much about {from_topic}, but {reason}: {to_topic}. {facts}")
        }
        ResponseStrategy::Empathy { recognized_state } => {
            let mut s = format!("I recognise {recognized_state} here.");
            if !facts.is_empty() {
                s.push(' ');
                s.push_str(&facts);
            }
            s
        }
        ResponseStrategy::SelfReflection { new_connection } => {
            format!("{facts} Curiously, while answering this I noticed {new_connection}.")
        }
        ResponseStrategy::ConfidenceCalibration { level, summary } => {
            format!(
                "My confidence here is {} ({}). {facts}",
                level.as_label(),
                summary
            )
        }
        ResponseStrategy::ReasoningTrace { steps } => {
            let tags: Vec<String> = content
                .primary
                .iter()
                .filter_map(|t| t.derivation_tag.clone())
                .collect();
            let mut s = facts;
            if !tags.is_empty() {
                s.push_str(&format!(" Derived via: {}.", tags.join(", ")));
            }
            if !steps.is_empty() {
                s.push_str(&format!(" ({} provenance record(s).)", steps.len()));
            }
            let _ = ctx; // context reserved for richer traces
            let _ = subject;
            s
        }
    }
}

/// Suffix sentence contributed by a companion conversational move.
fn companion_suffix(companion: &ResponseStrategy, subject: &str) -> Option<String> {
    match companion {
        ResponseStrategy::Empathy { .. } => Some(format!(
            "Is there anything specific about {subject} you'd like to know more about?"
        )),
        ResponseStrategy::ExplorationPrompt {
            related_topic,
            connection,
        } => Some(format!(
            "Want to explore how {subject} relates to {related_topic} ({connection})?"
        )),
        ResponseStrategy::ClarificationRequest { ambiguity, .. } => {
            Some(format!("Also — could you clarify {ambiguity}?"))
        }
        _ => None,
    }
}

/// Format a compact calibration tag for Full-detail output.
pub fn calibration_tag(confidence: f32) -> String {
    format!(
        "[confidence: {} ({confidence:.2})]",
        ConfidenceLevel::from_confidence(confidence).as_label()
    )
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compartment::psyche::{ArchetypeWeights, OceanProfile};

    fn gt(subj: &str, pred: &str, obj: &str, conf: f32) -> GroundedTriple {
        GroundedTriple {
            subject_label: subj.into(),
            predicate_label: pred.into(),
            object_label: obj.into(),
            subject_id: None,
            predicate_id: None,
            object_id: None,
            confidence: Some(conf),
            derivation_tag: None,
        }
    }

    fn grounded_dog() -> GroundedResponse {
        GroundedResponse {
            prose: "Dogs are domesticated mammals descended from wolves.".into(),
            supporting_triples: vec![
                gt("dog", "is-a", "mammal", 0.95),
                gt("dog", "has-a", "fur", 0.9),
                gt("dog", "similar-to", "wolf", 0.85),
                gt("dog", "has-property", "loyal", 0.8),
                gt("dog", "part-of", "pack", 0.75),
            ],
            confidence: Some(0.9),
            provenance_ids: vec![],
            gaps: vec![],
        }
    }

    fn psyche(archetypes: [f32; 4], ocean: [f32; 5]) -> Psyche {
        let mut p = Psyche::default();
        p.archetypes = ArchetypeWeights {
            healer: archetypes[0],
            sage: archetypes[1],
            guardian: archetypes[2],
            explorer: archetypes[3],
        };
        p.ocean = Some(OceanProfile {
            openness: ocean[0],
            conscientiousness: ocean[1],
            extraversion: ocean[2],
            agreeableness: ocean[3],
            neuroticism: ocean[4],
        });
        p
    }

    fn sage() -> Psyche {
        psyche([0.4, 0.9, 0.4, 0.4], [0.7, 0.8, 0.4, 0.6, 0.3])
    }

    fn explorer() -> Psyche {
        psyche([0.4, 0.4, 0.3, 0.9], [0.9, 0.4, 0.7, 0.5, 0.2])
    }

    fn healer() -> Psyche {
        psyche([0.9, 0.4, 0.4, 0.4], [0.5, 0.6, 0.5, 0.9, 0.5])
    }

    fn ctx_for(gr: &GroundedResponse) -> StrategyContext {
        let trajectory = DiscourseTrajectory::default();
        context_from_grounded(gr, "dog", QueryType::Factual, &trajectory, None)
    }

    #[test]
    fn context_derives_related_topic() {
        let gr = grounded_dog();
        let ctx = ctx_for(&gr);
        assert_eq!(
            ctx.related_topic,
            Some(("mammal".to_string(), "is-a".to_string()))
        );
        assert_eq!(ctx.answer_triples, 5);
        assert!(ctx.belief_interval.is_some());
    }

    #[test]
    fn same_query_three_psyches_differ() {
        let gr = grounded_dog();
        let ctx = ctx_for(&gr);
        let a = compose_grounded(&gr, "dog", &sage(), &ctx);
        let b = compose_grounded(&gr, "dog", &explorer(), &ctx);
        let c = compose_grounded(&gr, "dog", &healer(), &ctx);
        assert_ne!(a.text, b.text);
        assert_ne!(b.text, c.text);
        assert_ne!(a.text, c.text);
        // Strategies must differ between sage and explorer per the plan.
        assert_eq!(a.strategy, "Explanation");
        assert_eq!(b.strategy, "Contextualization");
        assert_eq!(b.companion, Some("ExplorationPrompt"));
        assert_eq!(c.companion, Some("Empathy"));
    }

    #[test]
    fn composition_is_deterministic() {
        let gr = grounded_dog();
        let ctx = ctx_for(&gr);
        let a = compose_grounded(&gr, "dog", &sage(), &ctx);
        let b = compose_grounded(&gr, "dog", &sage(), &ctx);
        assert_eq!(a.text, b.text);
    }

    #[test]
    fn sage_output_carries_pedagogical_framing() {
        let gr = grounded_dog();
        let ctx = ctx_for(&gr);
        let out = compose_grounded(&gr, "dog", &sage(), &ctx);
        assert!(
            out.text.starts_with("Let me explain."),
            "sage must open pedagogically, got: {}",
            out.text
        );
        assert!(out.text.contains("Does that clarify things?"));
    }

    #[test]
    fn healer_output_checks_in() {
        let gr = grounded_dog();
        let ctx = ctx_for(&gr);
        let out = compose_grounded(&gr, "dog", &healer(), &ctx);
        assert!(
            out.text.contains("anything specific about dog"),
            "healer companion Empathy must check in, got: {}",
            out.text
        );
    }

    #[test]
    fn low_confidence_admits_uncertainty_in_prose() {
        let mut gr = grounded_dog();
        gr.confidence = Some(0.25);
        for t in &mut gr.supporting_triples {
            t.confidence = Some(0.25);
        }
        gr.gaps = vec!["diet information".into()];
        let ctx = ctx_for(&gr);
        let out = compose_grounded(&gr, "dog", &healer(), &ctx);
        assert_eq!(out.strategy, "UncertaintyAdmission");
        assert!(out.text.contains("not certain"), "got: {}", out.text);
        assert!(out.text.contains("diet information"), "got: {}", out.text);
    }

    #[test]
    fn shadow_journal_recorded_when_inhibited() {
        // Disagreeable psyche + argumentative context + conflict → Challenge,
        // which the default high-severity shadow pattern then softens.
        let p = psyche([0.2, 0.5, 0.4, 0.5], [0.6, 0.6, 0.6, 0.1, 0.2]);
        let mut gr = grounded_dog();
        gr.supporting_triples
            .push(gt("dog", "is-a", "reptile", 0.1));
        let mut ctx = ctx_for(&gr);
        ctx.discourse_mode = crate::agent::trajectory::DiscourseMode::Argumentative;
        let out = compose_grounded(&gr, "dog", &p, &ctx);
        assert_eq!(out.strategy, "EvidenceWeighing");
        let journal = out
            .shadow_journal
            .expect("shadow must journal the inhibition");
        assert!(journal.contains("considered Challenge"), "got: {journal}");
    }

    #[test]
    fn query_type_classification() {
        assert_eq!(
            classify_query_type("Why do dogs bark?", "dog"),
            QueryType::Causal
        );
        assert_eq!(
            classify_query_type("How does a dog run?", "dog"),
            QueryType::Procedural
        );
        assert_eq!(
            classify_query_type("What is a dog?", "dog"),
            QueryType::Factual
        );
        assert_eq!(classify_query_type("who are you?", "self"), QueryType::Meta);
    }
}
