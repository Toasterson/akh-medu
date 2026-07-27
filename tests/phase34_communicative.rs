//! Phase 34 — Communicative Intelligence A/B fixture (integration).
//!
//! Verifies the plan's core promise: the SAME query, grounded in the SAME
//! knowledge, produces genuinely different responses under different
//! psyches — different strategies, different framings, different content
//! slices — and that the wired ChatProcessor pipeline (34f) carries the
//! transform end-to-end. Fully deterministic: no network, no LLM (the NLU
//! pipeline runs its rule tier only).
//!
//! Plan: docs/ai/plans/2026-03-26-phase34-communicative-intelligence.md
//! ADR:  docs/ai/decisions/044-communicative-intelligence.md

use std::sync::Arc;

use akh_medu::agent::communicative::{compose_grounded, context_from_grounded};
use akh_medu::agent::conversation::ground_query;
use akh_medu::agent::response_strategy::QueryType;
use akh_medu::agent::trajectory::DiscourseTrajectory;
use akh_medu::agent::{Agent, AgentConfig};
use akh_medu::chat::{ChatProcessor, SessionContext};
use akh_medu::compartment::psyche::{ArchetypeWeights, OceanProfile, Psyche};
use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::graph::Triple;
use akh_medu::message::AkhMessage;
use akh_medu::nlu::NluPipeline;
use akh_medu::symbol::SymbolKind;
use akh_medu::vsa::Dimension;

// ── Fixtures ────────────────────────────────────────────────────────────

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

/// Seed a well-corroborated "dog" micro-domain.
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

fn psyche(name: &str, archetypes: [f32; 4], ocean: [f32; 5]) -> Psyche {
    let mut p = Psyche::default();
    p.persona.name = name.to_string();
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
    psyche("Sage", [0.4, 0.9, 0.4, 0.4], [0.7, 0.8, 0.4, 0.6, 0.3])
}

fn explorer() -> Psyche {
    psyche("Explorer", [0.4, 0.4, 0.3, 0.9], [0.9, 0.4, 0.7, 0.5, 0.2])
}

fn healer() -> Psyche {
    psyche("Healer", [0.9, 0.4, 0.4, 0.4], [0.5, 0.6, 0.5, 0.9, 0.5])
}

// ── A/B fixture over the composition API ────────────────────────────────

#[test]
fn same_query_three_psyches_three_beings() {
    let engine = test_engine(None);
    seed_dog(&engine);

    let gr = ground_query("dog", &engine, "narrative").expect("dog must ground");
    let trajectory = DiscourseTrajectory::default();
    let ctx = context_from_grounded(&gr, "dog", QueryType::Factual, &trajectory, None);

    let a = compose_grounded(&gr, "dog", &sage(), &ctx);
    let b = compose_grounded(&gr, "dog", &explorer(), &ctx);
    let c = compose_grounded(&gr, "dog", &healer(), &ctx);

    // 1. The strategy chosen differs where the plan promises it.
    assert_eq!(a.strategy, "Explanation", "sage explains");
    assert_eq!(b.strategy, "Contextualization", "explorer contextualises");
    assert_eq!(
        b.companion,
        Some("ExplorationPrompt"),
        "explorer invites exploration"
    );
    assert_eq!(c.companion, Some("Empathy"), "healer checks in");

    // 2. Framing markers differ per archetype.
    assert!(
        a.text.starts_with("Let me explain."),
        "sage framing, got: {}",
        a.text
    );
    assert!(a.text.contains("Does that clarify things?"));
    assert!(
        b.text.contains("more to explore") || b.text.contains("Want to explore"),
        "explorer framing, got: {}",
        b.text
    );
    assert!(
        c.text.contains("I hope that helps.") || c.text.contains("anything specific"),
        "healer framing, got: {}",
        c.text
    );

    // 3. All three texts genuinely differ.
    assert_ne!(a.text, b.text);
    assert_ne!(b.text, c.text);
    assert_ne!(a.text, c.text);

    // 4. Content is filtered per strategy: contextualization leads with
    //    taxonomic triples.
    assert!(
        matches!(
            b.content.primary.first().map(|t| t.predicate_label.as_str()),
            Some("is-a") | Some("part-of") | Some("similar-to")
        ),
        "contextualization leads taxonomic"
    );

    // 5. Deterministic replay.
    let a2 = compose_grounded(&gr, "dog", &sage(), &ctx);
    assert_eq!(a.text, a2.text);
}

#[test]
fn repeated_topic_shifts_strategy() {
    let engine = test_engine(None);
    seed_dog(&engine);
    let gr = ground_query("dog", &engine, "narrative").expect("dog must ground");

    let mut trajectory = DiscourseTrajectory::default();
    let ctx1 = context_from_grounded(&gr, "dog", QueryType::Factual, &trajectory, None);
    let first = compose_grounded(&gr, "dog", &sage(), &ctx1);
    trajectory.record_turn(Some("dog"), None, first.strategy);

    let ctx2 = context_from_grounded(&gr, "dog", QueryType::Factual, &trajectory, None);
    assert!(ctx2.topic_repeated, "trajectory must flag the repeat");
    let second = compose_grounded(&gr, "dog", &sage(), &ctx2);

    assert_ne!(
        first.strategy, second.strategy,
        "asking the same question twice must change the strategy"
    );
}

// ── Wired pipeline (34f) through the ChatProcessor ──────────────────────

/// Drive the real ChatProcessor (rule-tier NLU only, read-only mode) and
/// return the narrative response text for a query.
fn chat_response(input: &str, psyche_cfg: Psyche) -> String {
    let tmp = tempfile::tempdir().unwrap();
    let engine = test_engine(Some(tmp.path().to_path_buf()));
    seed_dog(&engine);
    engine
        .compartments()
        .expect("compartments configured")
        .force_set_psyche(psyche_cfg);

    let mut agent = Agent::new(engine.clone(), AgentConfig::default()).unwrap();
    let mut processor = ChatProcessor::new(&engine, NluPipeline::new());
    processor.set_create_goal(false);
    processor.set_communicative(true);

    let msgs = processor.process_input(&mut SessionContext::new(), input, &mut agent, &engine);
    msgs.iter()
        .filter_map(|m| match m {
            AkhMessage::Narrative { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn chat_processor_carries_psyche_style_end_to_end() {
    let sage_out = chat_response("What is a dog?", sage());
    let healer_out = chat_response("What is a dog?", healer());

    assert!(!sage_out.is_empty(), "sage produced no narrative");
    assert!(!healer_out.is_empty(), "healer produced no narrative");
    assert_ne!(
        sage_out, healer_out,
        "the wired pipeline must produce psyche-dependent output"
    );

    // Archetype framing markers survive the whole pipeline (constraint
    // check + rendering).
    assert!(
        sage_out.contains("Let me explain.")
            || sage_out.contains("Does that clarify things?"),
        "sage marker missing from wired output: {sage_out}"
    );
    assert!(
        healer_out.contains("I hope that helps.")
            || healer_out.contains("anything specific")
            || healer_out.contains("I understand what you're asking.")
            || healer_out.contains("That's a thoughtful question."),
        "healer marker missing from wired output: {healer_out}"
    );
}

#[test]
fn communicative_flag_off_restores_plain_grounding() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = test_engine(Some(tmp.path().to_path_buf()));
    seed_dog(&engine);
    engine.compartments().unwrap().force_set_psyche(sage());

    let mut agent = Agent::new(engine.clone(), AgentConfig::default()).unwrap();
    let mut processor = ChatProcessor::new(&engine, NluPipeline::new());
    processor.set_create_goal(false);
    processor.set_communicative(false);

    let msgs = processor.process_input(&mut SessionContext::new(), "What is a dog?", &mut agent, &engine);
    let text: String = msgs
        .iter()
        .filter_map(|m| match m {
            AkhMessage::Narrative { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");

    assert!(!text.is_empty());
    assert!(
        !text.contains("Let me explain.") && !text.contains("Does that clarify things?"),
        "disabled pipeline must not inject archetype framing: {text}"
    );
}
