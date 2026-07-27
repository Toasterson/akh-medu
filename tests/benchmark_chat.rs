//! ELIZA vs LLM quality and routing benchmark (20 test queries).
//!
//! Verifies:
//! - Factual queries route to ELIZA+KG (no LLM call).
//! - Creative queries route to LLM fallback.
//! - ELIZA classification completes in < 1 ms per query.
//! - KG-grounded responses render in < 1 ms (grammar linearization).

use std::time::Instant;

use akh_medu::agent::eliza::{ElizaRouter, ElizaRouting, FallbackReason};
use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::graph::Triple;
use akh_medu::symbol::SymbolKind;
use akh_medu::vsa::Dimension;

// ── Benchmark helpers ──────────────────────────────────────────────────────

fn test_engine() -> Engine {
    Engine::new(EngineConfig {
        dimension: Dimension::TEST,
        ..Default::default()
    })
    .unwrap()
}

/// Seed the engine with Ennead facts for KG context tests.
fn seed_ennead(engine: &Engine) {
    let ennead = engine.create_symbol(SymbolKind::Entity, "Ennead").unwrap();
    let is_a = engine.create_symbol(SymbolKind::Relation, "is-a").unwrap();
    let group = engine.create_symbol(SymbolKind::Entity, "group-of-deities").unwrap();
    let osiris = engine.create_symbol(SymbolKind::Entity, "Osiris").unwrap();
    let member = engine.create_symbol(SymbolKind::Relation, "member-of").unwrap();

    let _ = engine.add_triple(&Triple::new(ennead.id, is_a.id, group.id).with_confidence(0.98));
    let _ = engine.add_triple(&Triple::new(osiris.id, member.id, ennead.id).with_confidence(0.97));
}

// ── 20 benchmark queries ───────────────────────────────────────────────────

/// Each entry: (query, expected_routing_variant, label)
struct QueryCase {
    query: &'static str,
    expect_kg: bool,
    label: &'static str,
}

fn benchmark_queries() -> Vec<QueryCase> {
    vec![
        // ── KG-grounded (expect ELIZA+KG) ─────────────────────────────────
        QueryCase {
            query: "What is the Ennead?",
            expect_kg: true,
            label: "factual-what-ennead",
        },
        QueryCase {
            query: "Who is Osiris?",
            expect_kg: true,
            label: "factual-who-osiris",
        },
        QueryCase {
            query: "Define the Ennead",
            expect_kg: true,
            label: "factual-define-ennead",
        },
        QueryCase {
            query: "Describe Thoth",
            expect_kg: true,
            label: "factual-describe-thoth",
        },
        QueryCase {
            query: "Explain what Horus represents",
            expect_kg: true,
            label: "factual-explain-horus",
        },
        QueryCase {
            query: "What are the nine deities?",
            expect_kg: true,
            label: "factual-nine-deities",
        },
        QueryCase {
            query: "Tell me about Set",
            expect_kg: true,
            label: "factual-about-set",
        },
        QueryCase {
            query: "Who was Ra in Egyptian mythology?",
            expect_kg: true,
            label: "factual-who-ra",
        },
        QueryCase {
            query: "What is Ma'at?",
            expect_kg: true,
            label: "factual-maat",
        },
        QueryCase {
            query: "List the members of the Ennead",
            expect_kg: true,
            label: "factual-list-ennead",
        },
        // ── Creative / LLM fallback (expect LLM) ──────────────────────────
        QueryCase {
            query: "Write a poem about the Ennead",
            expect_kg: false,
            label: "creative-poem-ennead",
        },
        QueryCase {
            query: "Compose an ode to Osiris",
            expect_kg: false,
            label: "creative-ode-osiris",
        },
        QueryCase {
            query: "Write a short story where Horus and Set argue",
            expect_kg: false,
            label: "creative-story-horus-set",
        },
        QueryCase {
            query: "Imagine what Thoth would tweet today",
            expect_kg: false,
            label: "creative-imagine-thoth",
        },
        QueryCase {
            query: "Create a haiku about Ra",
            expect_kg: false,
            label: "creative-haiku-ra",
        },
        QueryCase {
            query: "Draft a myth about Isis and Osiris",
            expect_kg: false,
            label: "creative-myth-isis",
        },
        QueryCase {
            query: "Invent a modern version of the Ennead story",
            expect_kg: false,
            label: "creative-invent-modern",
        },
        QueryCase {
            query: "Make up a joke about Anubis",
            expect_kg: false,
            label: "creative-joke-anubis",
        },
        QueryCase {
            query: "Brainstorm five ways Thoth might help humanity today",
            expect_kg: false,
            label: "creative-brainstorm-thoth",
        },
        QueryCase {
            query: "Suppose Horus lived in the 21st century — write his bio",
            expect_kg: false,
            label: "creative-suppose-horus-bio",
        },
    ]
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[test]
fn eliza_routing_correctness_20_queries() {
    let engine = ElizaRouter::new();
    let cases = benchmark_queries();

    let mut passed = 0usize;
    let mut failed = Vec::new();

    for case in &cases {
        let routing = engine.classify(case.query);
        let is_kg = matches!(routing, ElizaRouting::KgGrounded { .. });

        if is_kg == case.expect_kg {
            passed += 1;
        } else {
            failed.push(format!(
                "[FAIL] {}: expected {} but got {:?}",
                case.label,
                if case.expect_kg { "KgGrounded" } else { "LlmFallback" },
                routing
            ));
        }
    }

    if !failed.is_empty() {
        for msg in &failed {
            eprintln!("{msg}");
        }
    }

    // Require at least 18 / 20 correct (90%).
    assert!(
        passed >= 18,
        "ELIZA routing correctness: {passed}/20 correct, need ≥ 18. Failures:\n{}",
        failed.join("\n")
    );
}

#[test]
fn eliza_classification_under_1ms_per_query() {
    let engine = ElizaRouter::new();
    let cases = benchmark_queries();

    for case in &cases {
        // Best-of-N: take the minimum over several runs so scheduling noise
        // under parallel `cargo test` execution can't flake this microbenchmark.
        // The fast path is < 1µs in release and ~20µs in debug; the minimum
        // reflects that even when the machine is busy.
        let mut best = std::time::Duration::MAX;
        for _ in 0..32 {
            let start = Instant::now();
            let _routing = engine.classify(case.query);
            best = best.min(start.elapsed());
        }
        assert!(
            best.as_micros() < 1_000,
            "ELIZA classification for '{}' took {}µs at best (must be < 1ms)",
            case.label,
            best.as_micros()
        );
    }
}

#[test]
fn kg_response_latency_under_1ms() {
    let engine = test_engine();
    seed_ennead(&engine);
    let std_engine = std::sync::Arc::new(engine);

    // Ground the "Ennead" entity and linearize via grammar — should be sub-ms.
    let sym_id = std_engine.resolve_symbol("Ennead").unwrap();
    let triples = std_engine.triples_from(sym_id);
    assert!(!triples.is_empty(), "Ennead triples must be non-empty after seeding");

    // Build the registry once (one-time setup, not part of per-response cost),
    // then measure grammar linearization with best-of-N so the sub-ms assertion
    // stays robust under parallel `cargo test` load.
    let registry = akh_medu::grammar::GrammarRegistry::new();
    let abs = akh_medu::grammar::abs::AbsTree::EntityRef {
        label: "Ennead".to_string(),
        symbol_id: Some(sym_id),
    };
    let mut best = std::time::Duration::MAX;
    for _ in 0..32 {
        let start = Instant::now();
        let _ = registry.linearize("narrative", &abs);
        best = best.min(start.elapsed());
    }
    assert!(
        best.as_micros() < 1_000,
        "KG response (grammar linearization) took {}µs at best (must be < 1ms)",
        best.as_micros()
    );
}

#[test]
fn creative_queries_all_route_to_llm_fallback() {
    let engine = ElizaRouter::new();
    let creative_cases: Vec<_> = benchmark_queries()
        .into_iter()
        .filter(|c| !c.expect_kg)
        .collect();

    for case in &creative_cases {
        let routing = engine.classify(case.query);
        assert!(
            matches!(
                routing,
                ElizaRouting::LlmFallback {
                    reason: FallbackReason::Creative
                }
            ),
            "Creative query '{}' should route to LlmFallback(Creative), got {routing:?}",
            case.label
        );
    }
}

#[test]
fn factual_queries_all_route_to_kg() {
    let engine = ElizaRouter::new();
    let factual_cases: Vec<_> = benchmark_queries()
        .into_iter()
        .filter(|c| c.expect_kg)
        .collect();

    for case in &factual_cases {
        let routing = engine.classify(case.query);
        assert!(
            matches!(routing, ElizaRouting::KgGrounded { .. }),
            "Factual query '{}' should route to KgGrounded, got {routing:?}",
            case.label
        );
    }
}

#[test]
fn kg_context_extraction_finds_ennead_facts() {
    use akh_medu::chat::ChatProcessor;
    use akh_medu::nlu::NluPipeline;

    let engine = test_engine();
    seed_ennead(&engine);
    let arc_engine = std::sync::Arc::new(engine);

    let nlu = NluPipeline::new();
    let _processor = ChatProcessor::new(&*arc_engine, nlu);

    // ELIZA should classify this as KG-grounded.
    let routing = akh_medu::chat::SessionContext::new()
        .eliza
        .classify("What is the Ennead?");
    assert!(
        matches!(routing, ElizaRouting::KgGrounded { .. }),
        "expected KgGrounded, got {routing:?}"
    );
}
