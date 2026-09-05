//! Phase 34 demo — same query, different psyches, different beings.
//!
//! Seeds a small "dog" knowledge domain, grounds the query "What is a
//! dog?" once, then runs the communicative pipeline (strategy selection →
//! shadow filter → content filter → psyche style transform) under four
//! psyches and three conversation situations. Prints a comparison table.
//!
//! Deterministic: no network, no LLM, no model files.
//!
//! Run: cargo run --example phase34_demo
//!
//! Plan: docs/ai/plans/2026-03-26-phase34-communicative-intelligence.md
//! ADR:  docs/ai/decisions/044-communicative-intelligence.md

use std::sync::Arc;

use akh_medu::agent::communicative::{compose_grounded, context_from_grounded};
use akh_medu::agent::conversation::ground_query;
use akh_medu::agent::response_strategy::{QueryType, Sentiment};
use akh_medu::agent::trajectory::{DiscourseMode, DiscourseTrajectory};
use akh_medu::compartment::psyche::{ArchetypeWeights, OceanProfile, Psyche};
use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::graph::Triple;
use akh_medu::symbol::SymbolKind;
use akh_medu::vsa::Dimension;

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

fn main() {
    let engine = Arc::new(
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .expect("engine"),
    );

    // Seed a well-corroborated micro-domain.
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

    let gr = ground_query("dog", &engine, "narrative").expect("dog must ground");

    // O C E A N per the plan's worked example.
    let psyches = vec![
        psyche("Sage", [0.4, 0.9, 0.4, 0.4], [0.7, 0.8, 0.4, 0.6, 0.3]),
        psyche("Explorer", [0.4, 0.4, 0.3, 0.9], [0.9, 0.4, 0.7, 0.5, 0.2]),
        psyche("Healer", [0.9, 0.4, 0.4, 0.4], [0.5, 0.6, 0.5, 0.9, 0.5]),
        psyche("Guardian", [0.4, 0.4, 0.9, 0.4], [0.4, 0.9, 0.4, 0.7, 0.3]),
    ];

    println!("Phase 34 — Communicative Intelligence demo");
    println!(
        "KG: {} triples about 'dog', aggregate confidence {:.2}\n",
        gr.supporting_triples.len(),
        gr.confidence.unwrap_or(0.0)
    );

    // ── Situation 1: first ask, neutral ─────────────────────────────
    println!("── Query: \"What is a dog?\" (first ask, neutral mood) ──\n");
    println!(
        "{:<10} {:<18} {:<18} RESPONSE",
        "PSYCHE", "STRATEGY", "COMPANION"
    );
    println!("{}", "-".repeat(110));
    let trajectory = DiscourseTrajectory::default();
    for p in &psyches {
        let ctx = context_from_grounded(&gr, "dog", QueryType::Factual, &trajectory, None);
        let out = compose_grounded(&gr, "dog", p, &ctx);
        println!(
            "{:<10} {:<18} {:<18} {}",
            p.persona.name,
            out.strategy,
            out.companion.unwrap_or("-"),
            out.text
        );
        println!();
    }

    // ── Situation 2: same question asked again ──────────────────────
    println!("── Same question asked AGAIN (topic repeated) ──\n");
    for p in &psyches {
        let mut trajectory = DiscourseTrajectory::default();
        let ctx1 = context_from_grounded(&gr, "dog", QueryType::Factual, &trajectory, None);
        let first = compose_grounded(&gr, "dog", p, &ctx1);
        trajectory.record_turn(Some("dog"), None, first.strategy);
        let ctx2 = context_from_grounded(&gr, "dog", QueryType::Factual, &trajectory, None);
        let second = compose_grounded(&gr, "dog", p, &ctx2);
        println!(
            "{:<10} {} -> {}",
            p.persona.name, first.strategy, second.strategy
        );
    }
    println!();

    // ── Situation 3: argumentative user, conflicting evidence ───────
    println!("── Argumentative user + conflicting evidence ──\n");
    let mut gr_conflict = gr.clone();
    {
        let mut bad = gr_conflict.supporting_triples[0].clone();
        bad.predicate_label = "is-a".into();
        bad.object_label = "reptile".into();
        bad.confidence = Some(0.15);
        gr_conflict.supporting_triples.push(bad);
    }
    let mut trajectory = DiscourseTrajectory::default();
    trajectory.observe_input("No, that's wrong — dogs are reptiles!");
    assert_eq!(trajectory.discourse_mode(), DiscourseMode::Argumentative);
    for p in &psyches {
        let ctx = context_from_grounded(
            &gr_conflict,
            "dog",
            QueryType::Factual,
            &trajectory,
            Some(Sentiment::Negative),
        );
        let out = compose_grounded(&gr_conflict, "dog", p, &ctx);
        println!(
            "{:<10} {:<18} shadow: {}",
            p.persona.name,
            out.strategy,
            out.shadow_journal.as_deref().unwrap_or("-")
        );
        println!("           {}", out.text);
        println!();
    }

    // ── Deliberation transparency ────────────────────────────────────
    println!("── Selector deliberation (Sage, first ask) — top 5 ──\n");
    let ctx = context_from_grounded(
        &gr,
        "dog",
        QueryType::Factual,
        &DiscourseTrajectory::default(),
        None,
    );
    let out = compose_grounded(&gr, "dog", &psyches[0], &ctx);
    for (name, score) in out.deliberation.iter().take(5) {
        println!("  {score:>5.2}  {name}");
    }
}
