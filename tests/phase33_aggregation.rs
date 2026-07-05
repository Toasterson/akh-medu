//! Phase 33a — Grammar aggregation rules (integration).
//!
//! Verifies the plan's 33a promise end-to-end: N related triples produce
//! one fluent combined sentence instead of one robotic sentence per triple,
//! through every concrete grammar archetype AND through the full grounded
//! chat path — with Phase 34's psyche style transform still applying
//! cleanly AFTER aggregation. Fully deterministic: no network, no LLM.
//!
//! Plan: docs/ai/plans/2026-03-26-phase33-grounded-verbalization.md (33a)
//! ADR:  docs/ai/decisions/043-grounded-verbalization.md

use std::sync::Arc;

use akh_medu::agent::{Agent, AgentConfig};
use akh_medu::chat::ChatProcessor;
use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::grammar::style_transform::{CommunicationRegister, apply_register};
use akh_medu::grammar::{
    AbsTree, AggregationConfig, GrammarRegistry, aggregate_statements,
};
use akh_medu::graph::Triple;
use akh_medu::message::AkhMessage;
use akh_medu::nlu::NluPipeline;
use akh_medu::symbol::SymbolKind;
use akh_medu::vsa::Dimension;

// ── Fixtures ────────────────────────────────────────────────────────────

fn t(s: &str, p: &str, o: &str) -> AbsTree {
    AbsTree::triple(AbsTree::entity(s), AbsTree::relation(p), AbsTree::entity(o))
}

/// The plan's canonical example: four facts about one subject.
fn dog_facts() -> Vec<AbsTree> {
    vec![
        t("dog", "is-a", "mammal"),
        t("dog", "has-a", "fur"),
        t("dog", "has-a", "tail"),
        t("dog", "lives-in", "forests"),
    ]
}

// ── Before/after fixture through the grammar registry ───────────────────

#[test]
fn before_after_narrative_single_fluent_sentence() {
    // BEFORE: one sentence per triple (transition-cycled). Fresh registry
    // per linearization — the narrative transition counter is stateful.
    let before_tree = AbsTree::and(dog_facts());
    let before = GrammarRegistry::new()
        .linearize("narrative", &before_tree)
        .unwrap();
    let before_sentences = before.matches('.').count();
    assert!(
        before_sentences >= 4,
        "before: expected one sentence per triple, got: {before}"
    );

    // AFTER: aggregation folds them into ONE fluent sentence.
    let aggregated = aggregate_statements(&dog_facts(), &AggregationConfig::default());
    assert_eq!(aggregated.len(), 1);
    let after_tree = AbsTree::and(aggregated);
    let after = GrammarRegistry::new()
        .linearize("narrative", &after_tree)
        .unwrap();
    assert_eq!(
        after, "dog, a mammal, has fur and tail and lives in forests.",
        "aggregated narrative output"
    );
}

#[test]
fn aggregated_through_formal_grammar() {
    let registry = GrammarRegistry::new();
    let aggregated = aggregate_statements(&dog_facts(), &AggregationConfig::default());
    let prose = registry.linearize("formal", &aggregated[0]).unwrap();
    assert_eq!(
        prose,
        "The entity 'dog', a 'mammal', has 'fur' and 'tail' and lives in 'forests'."
    );
}

#[test]
fn aggregated_through_terse_grammar() {
    let registry = GrammarRegistry::new();
    let aggregated = aggregate_statements(&dog_facts(), &AggregationConfig::default());
    let prose = registry.linearize("terse", &aggregated[0]).unwrap();
    assert_eq!(prose, "dog → {is-a → mammal; has-a → fur, tail; lives-in → forests}");
}

#[test]
fn continuation_sentence_uses_pronoun_in_narrative() {
    let registry = GrammarRegistry::new();
    let facts = vec![
        t("dog", "is-a", "mammal"),
        t("dog", "has-a", "fur"),
        t("dog", "lives-in", "forests"),
        t("dog", "chases", "cats"),
        t("dog", "guards", "houses"),
    ];
    let mut cfg = AggregationConfig::default();
    cfg.max_groups_per_sentence = 2;
    let aggregated = aggregate_statements(&facts, &cfg);
    assert_eq!(aggregated.len(), 2);

    let prose = registry
        .linearize("narrative", &AbsTree::and(aggregated))
        .unwrap();
    // First sentence: subject + apposition; second: pronoun + connective
    // (the narrative transition machinery supplies the connective).
    assert!(
        prose.starts_with("dog, a mammal, has fur and lives in forests."),
        "got: {prose}"
    );
    assert!(
        prose.contains("it chases cats and guards houses."),
        "continuation must use the pronoun, got: {prose}"
    );
}

#[test]
fn aggregated_confidence_qualifier_survives() {
    // Confidence-banded facts keep an honest qualifier on the combined
    // sentence (narrative renders the band's minimum).
    let registry = GrammarRegistry::new();
    let facts = vec![
        AbsTree::triple_with_confidence(
            AbsTree::entity("dog"),
            AbsTree::relation("is-a"),
            AbsTree::entity("mammal"),
            0.95,
        ),
        AbsTree::triple_with_confidence(
            AbsTree::entity("dog"),
            AbsTree::relation("has-a"),
            AbsTree::entity("fur"),
            0.92,
        ),
    ];
    let aggregated = aggregate_statements(&facts, &AggregationConfig::default());
    assert_eq!(aggregated.len(), 1);
    let prose = registry.linearize("narrative", &aggregated[0]).unwrap();
    assert_eq!(prose, "dog, a mammal, has fur, with high confidence.");
}

// ── Phase 34 style transform applies cleanly AFTER aggregation ──────────

#[test]
fn style_transform_applies_after_aggregation() {
    let hedging = CommunicationRegister {
        formality: 0.5,
        hedging_level: 0.8,
        elaboration: 0.5,
        warmth: 0.5,
        metaphor_tendency: 0.5,
    };

    // Pure-copula aggregation keeps "is a", so the hedging rewrite (34b)
    // still fires on the combined sentence.
    let copular = aggregate_statements(
        &[t("dog", "is-a", "mammal"), t("dog", "is-a", "predator")],
        &AggregationConfig::default(),
    );
    let core = GrammarRegistry::new()
        .linearize("narrative", &copular[0])
        .unwrap();
    assert_eq!(core, "dog is a mammal and a predator.");
    let hedged = apply_register(&core, &hedging);
    assert!(
        hedged.contains("appears to be a mammal"),
        "hedging must rewrite the aggregated copula, got: {hedged}"
    );

    // Apposition-style aggregation has no copula to hedge; the transform
    // must pass it through cleanly (no garbling).
    let apposed = aggregate_statements(&dog_facts(), &AggregationConfig::default());
    let core = GrammarRegistry::new()
        .linearize("narrative", &apposed[0])
        .unwrap();
    let styled = apply_register(&core, &hedging);
    assert_eq!(
        styled, core,
        "apposition sentence must survive the register untouched"
    );
}

// ── Full grounded chat path ──────────────────────────────────────────────

fn test_engine(compartments_dir: Option<std::path::PathBuf>) -> Arc<Engine> {
    Arc::new(
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            compartments_dir,
            ..Default::default()
        })
        .unwrap(),
    )
}

/// Seed the same dog micro-domain as the Phase 34 fixture.
fn seed_dog(engine: &Engine) {
    let sym = |kind: SymbolKind, name: &str| engine.create_symbol(kind, name).unwrap().id;
    let dog = sym(SymbolKind::Entity, "dog");
    let mammal = sym(SymbolKind::Entity, "mammal");
    let fur = sym(SymbolKind::Entity, "fur");
    let wolf = sym(SymbolKind::Entity, "wolf");
    let loyal = sym(SymbolKind::Entity, "loyal");
    let pack = sym(SymbolKind::Entity, "pack");
    let is_a = sym(SymbolKind::Relation, "is-a");
    let has_a = sym(SymbolKind::Relation, "has-a");
    let similar_to = sym(SymbolKind::Relation, "similar-to");
    let has_property = sym(SymbolKind::Relation, "has-property");
    let part_of = sym(SymbolKind::Relation, "part-of");

    for (s, p, o, c) in [
        (dog, is_a, mammal, 0.95),
        (dog, has_a, fur, 0.92),
        (dog, similar_to, wolf, 0.88),
        (dog, has_property, loyal, 0.85),
        (dog, part_of, pack, 0.80),
    ] {
        engine
            .add_triple(&Triple::new(s, p, o).with_confidence(c))
            .unwrap();
    }
}

/// Drive the real ChatProcessor (rule-tier NLU, read-only) and return the
/// narrative response text.
fn chat_response(input: &str, communicative: bool) -> String {
    let tmp = tempfile::tempdir().unwrap();
    let engine = test_engine(Some(tmp.path().to_path_buf()));
    seed_dog(&engine);

    let mut agent = Agent::new(engine.clone(), AgentConfig::default()).unwrap();
    let mut processor = ChatProcessor::new(&engine, NluPipeline::new());
    processor.set_create_goal(false);
    processor.set_communicative(communicative);

    let msgs = processor.process_input(input, &mut agent, &engine);
    msgs.iter()
        .filter_map(|m| match m {
            AkhMessage::Narrative { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn grounded_chat_path_emits_aggregated_prose() {
    let out = chat_response("What is a dog?", false);
    assert!(!out.is_empty(), "grounded query produced no narrative");
    eprintln!("CHAT-OUTPUT>>> {out}");
    // The subject-merge apposition is the signature of aggregation: the
    // is-a fact rides inside the combined sentence instead of standing
    // alone as "dog is a mammal."
    assert!(
        out.contains(", a mammal,"),
        "grounded prose must carry the aggregated apposition, got: {out}"
    );
    // Predicate coordination: at least one combined verb-phrase list.
    assert!(out.contains(" and "), "no coordination in: {out}");
}

#[test]
fn grounded_chat_path_aggregation_composes_with_communicative() {
    // Aggregation (33a) runs before the communicative pipeline (34); the
    // styled output must still contain grounded, aggregated content.
    let out = chat_response("What is a dog?", true);
    assert!(!out.is_empty(), "communicative grounded query produced no narrative");
    assert!(
        out.to_lowercase().contains("mammal"),
        "grounded content missing from styled output: {out}"
    );
}
