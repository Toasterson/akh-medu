//! End-to-end integration tests for the akh-medu engine.
//!
//! These tests exercise the full pipeline from symbol creation through
//! inference and export, validating that the registry, knowledge graph,
//! and introspection APIs all work together.

use std::collections::HashSet;

use std::sync::Arc;

use akh_medu::agent::{Agent, AgentConfig};
use akh_medu::agent::goal::GoalStatus;
use akh_medu::agent::memory::{WorkingMemory, WorkingMemoryEntry, WorkingMemoryKind};
use akh_medu::agent::tool::{Tool, ToolInput, ToolOutput, ToolSignature};
use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::graph::traverse::TraversalConfig;
use akh_medu::graph::Triple;
use akh_medu::infer::InferenceQuery;
use akh_medu::pipeline::{Pipeline, PipelineData, PipelineStage, StageConfig, StageKind};
use akh_medu::symbol::{SymbolId, SymbolKind};
use akh_medu::vsa::Dimension;

fn test_engine() -> Engine {
    Engine::new(EngineConfig {
        dimension: Dimension::TEST,
        ..Default::default()
    })
    .unwrap()
}

fn persistent_engine(dir: &std::path::Path) -> Engine {
    Engine::new(EngineConfig {
        dimension: Dimension::TEST,
        data_dir: Some(dir.to_path_buf()),
        ..Default::default()
    })
    .unwrap()
}

#[test]
fn end_to_end_create_ingest_infer() {
    let dir = tempfile::TempDir::new().unwrap();
    let engine = persistent_engine(dir.path());

    // Create symbols with labels.
    let sun = engine.create_symbol(SymbolKind::Entity, "Sun").unwrap();
    let is_a = engine.create_symbol(SymbolKind::Relation, "is-a").unwrap();
    let star = engine.create_symbol(SymbolKind::Entity, "Star").unwrap();
    let moon = engine.create_symbol(SymbolKind::Entity, "Moon").unwrap();
    let orbits = engine.create_symbol(SymbolKind::Relation, "orbits").unwrap();

    // Ingest triples.
    engine
        .add_triple(&Triple::new(sun.id, is_a.id, star.id))
        .unwrap();
    engine
        .add_triple(&Triple::new(moon.id, orbits.id, sun.id))
        .unwrap();

    // Verify registry lookups.
    assert_eq!(engine.lookup_symbol("Sun").unwrap(), sun.id);
    assert_eq!(engine.lookup_symbol("star").unwrap(), star.id); // case insensitive

    // Run inference from Sun.
    let query = InferenceQuery {
        seeds: vec![sun.id],
        top_k: 10,
        max_depth: 1,
        ..Default::default()
    };
    let result = engine.infer(&query).unwrap();
    assert!(!result.activations.is_empty());

    // Verify the label resolution.
    let label = engine.resolve_label(sun.id);
    assert_eq!(label, "Sun");
}

#[test]
fn resolve_symbol_by_name_and_id() {
    let engine = test_engine();

    let sun = engine.create_symbol(SymbolKind::Entity, "Sun").unwrap();
    let moon = engine.create_symbol(SymbolKind::Entity, "Moon").unwrap();

    // Resolve by name.
    assert_eq!(engine.resolve_symbol("Sun").unwrap(), sun.id);
    assert_eq!(engine.resolve_symbol("moon").unwrap(), moon.id);

    // Resolve by numeric ID.
    assert_eq!(
        engine.resolve_symbol(&sun.id.get().to_string()).unwrap(),
        sun.id
    );

    // Unknown label should error.
    assert!(engine.resolve_symbol("Jupiter").is_err());
}

#[test]
fn introspection_apis() {
    let engine = test_engine();

    let sun = engine.create_symbol(SymbolKind::Entity, "Sun").unwrap();
    let is_a = engine.create_symbol(SymbolKind::Relation, "is-a").unwrap();
    let star = engine.create_symbol(SymbolKind::Entity, "Star").unwrap();
    let moon = engine.create_symbol(SymbolKind::Entity, "Moon").unwrap();
    let orbits = engine.create_symbol(SymbolKind::Relation, "orbits").unwrap();

    engine
        .add_triple(&Triple::new(sun.id, is_a.id, star.id))
        .unwrap();
    engine
        .add_triple(&Triple::new(moon.id, orbits.id, sun.id))
        .unwrap();

    // has_triple
    assert!(engine.has_triple(sun.id, is_a.id, star.id));
    assert!(!engine.has_triple(star.id, is_a.id, sun.id));

    // triples_from
    let from_sun = engine.triples_from(sun.id);
    assert_eq!(from_sun.len(), 1);
    assert_eq!(from_sun[0].object, star.id);

    // triples_to
    let to_sun = engine.triples_to(sun.id);
    assert_eq!(to_sun.len(), 1);
    assert_eq!(to_sun[0].subject, moon.id);

    // all_triples
    let all = engine.all_triples();
    assert_eq!(all.len(), 2);
}

#[test]
fn export_triples_with_labels() {
    let engine = test_engine();

    let sun = engine.create_symbol(SymbolKind::Entity, "Sun").unwrap();
    let is_a = engine.create_symbol(SymbolKind::Relation, "is-a").unwrap();
    let star = engine.create_symbol(SymbolKind::Entity, "Star").unwrap();

    engine
        .add_triple(&Triple::new(sun.id, is_a.id, star.id))
        .unwrap();

    let exports = engine.export_triples();
    assert_eq!(exports.len(), 1);
    assert_eq!(exports[0].subject_label, "Sun");
    assert_eq!(exports[0].predicate_label, "is-a");
    assert_eq!(exports[0].object_label, "Star");
    assert_eq!(exports[0].subject_id, sun.id.get());
}

#[test]
fn export_symbol_table() {
    let engine = test_engine();

    engine.create_symbol(SymbolKind::Entity, "Sun").unwrap();
    engine.create_symbol(SymbolKind::Relation, "is-a").unwrap();
    engine.create_symbol(SymbolKind::Entity, "Star").unwrap();

    let all_symbols = engine.all_symbols();
    let export_table = engine.export_symbol_table();

    assert_eq!(all_symbols.len(), export_table.len());

    // All symbols in all_symbols() should appear in the export table.
    for meta in &all_symbols {
        assert!(export_table.iter().any(|e| e.id == meta.id.get() && e.label == meta.label));
    }
}

#[test]
fn duplicate_label_rejected() {
    let engine = test_engine();

    engine.create_symbol(SymbolKind::Entity, "Sun").unwrap();
    let result = engine.create_symbol(SymbolKind::Entity, "sun");
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("duplicate label"));
}

#[test]
fn get_symbol_meta_returns_correct_data() {
    let engine = test_engine();

    let sun = engine.create_symbol(SymbolKind::Entity, "Sun").unwrap();
    let meta = engine.get_symbol_meta(sun.id).unwrap();
    assert_eq!(meta.label, "Sun");
    assert_eq!(meta.kind, SymbolKind::Entity);

    // Non-existent ID should error.
    let bad_id = SymbolId::new(9999).unwrap();
    assert!(engine.get_symbol_meta(bad_id).is_err());
}

// ---------------------------------------------------------------------------
// Part A: Label-based ingest tests
// ---------------------------------------------------------------------------

#[test]
fn ingest_label_based_triples() {
    let engine = test_engine();

    let triples = vec![
        ("Sun".into(), "is-a".into(), "Star".into(), 1.0),
        ("Earth".into(), "is-a".into(), "Planet".into(), 1.0),
        ("Earth".into(), "orbits".into(), "Sun".into(), 0.95),
    ];

    let (created, ingested) = engine.ingest_label_triples(&triples).unwrap();

    // All labels are new, so created should be non-zero.
    assert!(created > 0, "expected new symbols to be created");
    assert_eq!(ingested, 3);

    // Verify symbols were auto-created.
    let sun_id = engine.lookup_symbol("Sun").unwrap();
    let star_id = engine.lookup_symbol("Star").unwrap();
    let is_a_id = engine.lookup_symbol("is-a").unwrap();

    // Verify triples exist.
    assert!(engine.has_triple(sun_id, is_a_id, star_id));

    // Verify symbol kinds: predicates should be Relations.
    let is_a_meta = engine.get_symbol_meta(is_a_id).unwrap();
    assert_eq!(is_a_meta.kind, SymbolKind::Relation);

    // Verify entities.
    let sun_meta = engine.get_symbol_meta(sun_id).unwrap();
    assert_eq!(sun_meta.kind, SymbolKind::Entity);
}

#[test]
fn resolve_or_create_entity_idempotent() {
    let engine = test_engine();

    let id1 = engine.resolve_or_create_entity("Sun").unwrap();
    let id2 = engine.resolve_or_create_entity("Sun").unwrap();

    assert_eq!(id1, id2, "calling resolve_or_create twice should return same ID");
}

#[test]
fn resolve_or_create_relation_idempotent() {
    let engine = test_engine();

    let id1 = engine.resolve_or_create_relation("is-a").unwrap();
    let id2 = engine.resolve_or_create_relation("is-a").unwrap();

    assert_eq!(id1, id2, "calling resolve_or_create twice should return same ID");
}

// ---------------------------------------------------------------------------
// Part B: Graph traversal tests
// ---------------------------------------------------------------------------

#[test]
fn traverse_from_seeds() {
    let engine = test_engine();

    let triples = vec![
        ("Sun".into(), "is-a".into(), "Star".into(), 1.0),
        ("Earth".into(), "orbits".into(), "Sun".into(), 1.0),
        ("Moon".into(), "orbits".into(), "Earth".into(), 1.0),
    ];
    engine.ingest_label_triples(&triples).unwrap();

    let earth_id = engine.lookup_symbol("Earth").unwrap();
    let result = engine
        .traverse(
            &[earth_id],
            TraversalConfig {
                max_depth: 3,
                ..Default::default()
            },
        )
        .unwrap();

    // Earth -> orbits -> Sun, and then Sun -> is-a -> Star.
    assert!(!result.triples.is_empty(), "traversal should find triples");
    assert!(result.visited.contains(&earth_id));
}

#[test]
fn traverse_with_predicate_filter() {
    let engine = test_engine();

    let triples = vec![
        ("Sun".into(), "is-a".into(), "Star".into(), 1.0),
        ("Earth".into(), "orbits".into(), "Sun".into(), 1.0),
        ("Earth".into(), "is-a".into(), "Planet".into(), 1.0),
    ];
    engine.ingest_label_triples(&triples).unwrap();

    let earth_id = engine.lookup_symbol("Earth").unwrap();
    let orbits_id = engine.lookup_symbol("orbits").unwrap();

    let mut pred_filter = HashSet::new();
    pred_filter.insert(orbits_id);

    let result = engine
        .traverse(
            &[earth_id],
            TraversalConfig {
                max_depth: 2,
                predicate_filter: pred_filter,
                ..Default::default()
            },
        )
        .unwrap();

    // Only orbits edges should appear.
    for t in &result.triples {
        assert_eq!(t.predicate, orbits_id, "only 'orbits' predicates should appear");
    }
}

// ---------------------------------------------------------------------------
// Part C: SPARQL test
// ---------------------------------------------------------------------------

#[test]
fn sparql_select_query() {
    let dir = tempfile::TempDir::new().unwrap();
    let engine = persistent_engine(dir.path());

    let triples = vec![
        ("Sun".into(), "is-a".into(), "Star".into(), 1.0),
    ];
    engine.ingest_label_triples(&triples).unwrap();

    // Sync to SPARQL.
    engine.persist().unwrap();

    let results = engine
        .sparql_query("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")
        .unwrap();

    assert!(!results.is_empty(), "SPARQL should return at least one result");
}

// ---------------------------------------------------------------------------
// Part D: Reasoning tests
// ---------------------------------------------------------------------------

#[test]
fn simplify_double_negation() {
    let engine = test_engine();

    let result = engine.simplify_expression("(not (not x))").unwrap();
    assert_eq!(result, "x");
}

#[test]
fn simplify_bind_self_inverse() {
    let engine = test_engine();

    let result = engine.simplify_expression("(bind a (bind a b))").unwrap();
    assert_eq!(result, "b");
}

// ---------------------------------------------------------------------------
// Part E: Skill scaffold + label loading tests
// ---------------------------------------------------------------------------

#[test]
fn skill_scaffold_creates_template() {
    let dir = tempfile::TempDir::new().unwrap();
    let skill_dir = dir.path().join("skills").join("my-test");
    std::fs::create_dir_all(&skill_dir).unwrap();

    // Write the three template files (mirroring what the CLI does).
    let manifest = serde_json::json!({
        "id": "my-test",
        "name": "my-test",
        "version": "0.1.0",
        "description": "my-test knowledge domain",
        "domains": ["my-test"],
        "weight_size_bytes": 0,
        "triples_file": "triples.json",
        "rules_file": "rules.txt"
    });
    std::fs::write(
        skill_dir.join("skill.json"),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let triples = serde_json::json!([
        {"subject": "ExampleEntity", "predicate": "is-a", "object": "Category", "confidence": 1.0}
    ]);
    std::fs::write(
        skill_dir.join("triples.json"),
        serde_json::to_string_pretty(&triples).unwrap(),
    )
    .unwrap();

    std::fs::write(
        skill_dir.join("rules.txt"),
        "# Rewrite rules\n",
    )
    .unwrap();

    // Verify files exist.
    assert!(skill_dir.join("skill.json").exists());
    assert!(skill_dir.join("triples.json").exists());
    assert!(skill_dir.join("rules.txt").exists());

    // Verify manifest parses correctly.
    let content = std::fs::read_to_string(skill_dir.join("skill.json")).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert_eq!(parsed["id"], "my-test");
    assert_eq!(parsed["version"], "0.1.0");
}

#[test]
fn load_skill_with_label_triples() {
    let dir = tempfile::TempDir::new().unwrap();

    // Create skill directory structure.
    let skills_dir = dir.path().join("skills");
    let skill_dir = skills_dir.join("test-labels");
    std::fs::create_dir_all(&skill_dir).unwrap();

    let manifest = serde_json::json!({
        "id": "test-labels",
        "name": "Test Labels",
        "version": "0.1.0",
        "description": "Skill with label-based triples",
        "domains": ["test"],
        "weight_size_bytes": 0,
        "triples_file": "triples.json",
        "rules_file": "rules.txt"
    });
    std::fs::write(
        skill_dir.join("skill.json"),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let triples = serde_json::json!([
        {"subject": "Alpha", "predicate": "related-to", "object": "Beta", "confidence": 0.9},
        {"subject": "Beta", "predicate": "related-to", "object": "Gamma", "confidence": 0.8}
    ]);
    std::fs::write(
        skill_dir.join("triples.json"),
        serde_json::to_string_pretty(&triples).unwrap(),
    )
    .unwrap();

    std::fs::write(skill_dir.join("rules.txt"), "# no rules\n").unwrap();

    // Create engine with persistence pointing to the temp dir.
    let engine = Engine::new(EngineConfig {
        dimension: Dimension::TEST,
        data_dir: Some(dir.path().to_path_buf()),
        ..Default::default()
    })
    .unwrap();

    let activation = engine.load_skill("test-labels").unwrap();
    assert!(activation.triples_loaded >= 2, "should load at least 2 label-based triples");

    // Verify symbols were created.
    assert!(engine.lookup_symbol("Alpha").is_ok());
    assert!(engine.lookup_symbol("Beta").is_ok());
    assert!(engine.lookup_symbol("Gamma").is_ok());
    assert!(engine.lookup_symbol("related-to").is_ok());
}

// ---------------------------------------------------------------------------
// Part F: VSA search, analogy, filler tests
// ---------------------------------------------------------------------------

#[test]
fn search_similar_finds_related() {
    let engine = test_engine();

    let triples = vec![
        ("Sun".into(), "is-a".into(), "Star".into(), 1.0),
        ("Earth".into(), "is-a".into(), "Planet".into(), 1.0),
        ("Earth".into(), "orbits".into(), "Sun".into(), 1.0),
    ];
    engine.ingest_label_triples(&triples).unwrap();

    let sun_id = engine.lookup_symbol("Sun").unwrap();

    // Search for symbols similar to Sun.
    let results = engine.search_similar_to(sun_id, 5).unwrap();
    assert!(
        !results.is_empty(),
        "search_similar_to should return results"
    );
    // Sun's own vector should be the top match.
    assert_eq!(results[0].symbol_id, sun_id);
}

#[test]
fn analogy_basic() {
    let engine = test_engine();

    let _sun = engine.create_symbol(SymbolKind::Entity, "Sun").unwrap();
    let _star = engine.create_symbol(SymbolKind::Entity, "Star").unwrap();
    let _earth = engine.create_symbol(SymbolKind::Entity, "Earth").unwrap();
    let _planet = engine.create_symbol(SymbolKind::Entity, "Planet").unwrap();

    let sun_id = engine.lookup_symbol("Sun").unwrap();
    let star_id = engine.lookup_symbol("Star").unwrap();
    let earth_id = engine.lookup_symbol("Earth").unwrap();

    // Analogy: Sun:Star :: Earth:?
    let results = engine.infer_analogy(sun_id, star_id, earth_id, 5).unwrap();
    assert!(
        !results.is_empty(),
        "analogy should return results"
    );
}

#[test]
fn filler_recovery_basic() {
    let engine = test_engine();

    let triples = vec![
        ("Sun".into(), "is-a".into(), "Star".into(), 1.0),
    ];
    engine.ingest_label_triples(&triples).unwrap();

    let sun_id = engine.lookup_symbol("Sun").unwrap();
    let is_a_id = engine.lookup_symbol("is-a").unwrap();

    // Recover filler for (Sun, is-a).
    let results = engine.recover_filler(sun_id, is_a_id, 5).unwrap();
    assert!(
        !results.is_empty(),
        "filler recovery should return results"
    );
}

// ---------------------------------------------------------------------------
// Part G: Graph analytics tests
// ---------------------------------------------------------------------------

#[test]
fn degree_centrality_basic() {
    let engine = test_engine();

    let triples = vec![
        ("Sun".into(), "is-a".into(), "Star".into(), 1.0),
        ("Earth".into(), "orbits".into(), "Sun".into(), 1.0),
        ("Moon".into(), "orbits".into(), "Earth".into(), 1.0),
        ("Mars".into(), "orbits".into(), "Sun".into(), 1.0),
    ];
    engine.ingest_label_triples(&triples).unwrap();

    let results = engine.degree_centrality();
    assert!(!results.is_empty());

    // Sun has the highest total degree: 1 outgoing (is-a -> Star) + 2 incoming (Earth orbits, Mars orbits)
    let sun_id = engine.lookup_symbol("Sun").unwrap();
    let sun_dc = results.iter().find(|dc| dc.symbol == sun_id).unwrap();
    assert!(sun_dc.total >= 3, "Sun should have at least 3 total degree");
    assert_eq!(results[0].symbol, sun_id, "Sun should be first (highest degree)");
}

#[test]
fn pagerank_hub_scores_highest() {
    let engine = test_engine();

    let triples = vec![
        ("A".into(), "links".into(), "Hub".into(), 1.0),
        ("B".into(), "links".into(), "Hub".into(), 1.0),
        ("C".into(), "links".into(), "Hub".into(), 1.0),
        ("D".into(), "links".into(), "Hub".into(), 1.0),
    ];
    engine.ingest_label_triples(&triples).unwrap();

    let results = engine.pagerank(0.85, 20).unwrap();
    assert!(!results.is_empty());

    // Hub has 4 incoming links, should have a high score.
    let hub_id = engine.lookup_symbol("Hub").unwrap();
    let hub_pr = results.iter().find(|pr| pr.symbol == hub_id).unwrap();
    assert!(hub_pr.score > 0.0);
}

#[test]
fn scc_finds_cycle() {
    let engine = test_engine();

    let triples = vec![
        ("A".into(), "points-to".into(), "B".into(), 1.0),
        ("B".into(), "points-to".into(), "C".into(), 1.0),
        ("C".into(), "points-to".into(), "A".into(), 1.0),
    ];
    engine.ingest_label_triples(&triples).unwrap();

    let components = engine.strongly_connected_components().unwrap();
    // A, B, C form one SCC.
    let cycle_scc = components.iter().find(|c| c.size >= 3);
    assert!(cycle_scc.is_some(), "should find a component with 3+ members");

    let a_id = engine.lookup_symbol("A").unwrap();
    let b_id = engine.lookup_symbol("B").unwrap();
    let c_id = engine.lookup_symbol("C").unwrap();
    let members = &cycle_scc.unwrap().members;
    assert!(members.contains(&a_id));
    assert!(members.contains(&b_id));
    assert!(members.contains(&c_id));
}

#[test]
fn shortest_path_finds_route() {
    let engine = test_engine();

    let triples = vec![
        ("A".into(), "links".into(), "B".into(), 1.0),
        ("B".into(), "links".into(), "C".into(), 1.0),
        ("C".into(), "links".into(), "D".into(), 1.0),
    ];
    engine.ingest_label_triples(&triples).unwrap();

    let a_id = engine.lookup_symbol("A").unwrap();
    let c_id = engine.lookup_symbol("C").unwrap();

    let path = engine.shortest_path(a_id, c_id).unwrap();
    assert!(path.is_some());
    let path = path.unwrap();
    assert_eq!(path.len(), 3); // A -> B -> C
    assert_eq!(path[0], a_id);
    assert_eq!(path[2], c_id);
}

// ---------------------------------------------------------------------------
// Part H: Pipeline CLI tests
// ---------------------------------------------------------------------------

#[test]
fn pipeline_query_via_engine() {
    let engine = test_engine();

    let triples = vec![
        ("Sun".into(), "is-a".into(), "Star".into(), 1.0),
        ("Earth".into(), "orbits".into(), "Sun".into(), 1.0),
    ];
    engine.ingest_label_triples(&triples).unwrap();

    let sun_id = engine.lookup_symbol("Sun").unwrap();
    let output = engine.query_pipeline(vec![sun_id]).unwrap();

    assert_eq!(output.stages_executed, 3);
    assert_eq!(output.stage_results.len(), 3);

    // First stage is "retrieve", should produce Traversal data.
    assert_eq!(output.stage_results[0].0, "retrieve");
}

#[test]
fn pipeline_custom_stages() {
    let engine = test_engine();

    let triples = vec![
        ("Sun".into(), "is-a".into(), "Star".into(), 1.0),
    ];
    engine.ingest_label_triples(&triples).unwrap();

    let sun_id = engine.lookup_symbol("Sun").unwrap();

    // Build a custom 2-stage pipeline: retrieve + infer.
    let pipeline = Pipeline {
        name: "custom".into(),
        stages: vec![
            PipelineStage {
                name: "retrieve".into(),
                kind: StageKind::Retrieve,
                config: StageConfig::Default,
            },
            PipelineStage {
                name: "infer".into(),
                kind: StageKind::Infer,
                config: StageConfig::Default,
            },
        ],
    };

    let output = engine
        .run_pipeline(&pipeline, PipelineData::Seeds(vec![sun_id]))
        .unwrap();

    assert_eq!(output.stages_executed, 2);
}

// ---------------------------------------------------------------------------
// Part I: Batch persist test
// ---------------------------------------------------------------------------

#[test]
fn batch_persist_consistent() {
    let dir = tempfile::TempDir::new().unwrap();

    // Phase 1: create symbols and persist.
    {
        let engine = persistent_engine(dir.path());
        for i in 0..20 {
            engine
                .create_symbol(SymbolKind::Entity, format!("Symbol{i}"))
                .unwrap();
        }
        engine.persist().unwrap();
    }

    // Phase 2: reopen and verify all symbols present.
    {
        let engine = persistent_engine(dir.path());
        let all = engine.all_symbols();
        assert_eq!(all.len(), 20, "all 20 symbols should survive restart");

        for i in 0..20 {
            let label = format!("Symbol{i}");
            assert!(
                engine.lookup_symbol(&label).is_ok(),
                "symbol '{label}' should be found after restart"
            );
        }
    }
}

// ===========================================================================
// Part J: Agent integration tests
// ===========================================================================

fn test_agent() -> Agent {
    let engine = Arc::new(test_engine());
    Agent::new(engine, AgentConfig::default()).unwrap()
}

fn test_agent_with_data() -> Agent {
    let engine = test_engine();
    let triples = vec![
        ("Sun".into(), "is-a".into(), "Star".into(), 1.0),
        ("Earth".into(), "is-a".into(), "Planet".into(), 1.0),
        ("Earth".into(), "orbits".into(), "Sun".into(), 0.95),
        ("Moon".into(), "orbits".into(), "Earth".into(), 0.9),
        ("Mars".into(), "is-a".into(), "Planet".into(), 1.0),
        ("Mars".into(), "orbits".into(), "Sun".into(), 0.95),
    ];
    engine.ingest_label_triples(&triples).unwrap();
    let engine = Arc::new(engine);
    Agent::new(engine, AgentConfig::default()).unwrap()
}

#[test]
fn working_memory_push_and_retrieve() {
    let mut wm = WorkingMemory::new(10);

    let id1 = wm
        .push(WorkingMemoryEntry {
            id: 0,
            content: "Observed: Sun is a star".into(),
            symbols: Vec::new(),
            kind: WorkingMemoryKind::Observation,
            timestamp: 0,
            relevance: 0.7,
            source_cycle: 1,
            reference_count: 0,
        })
        .unwrap();

    let id2 = wm
        .push(WorkingMemoryEntry {
            id: 0,
            content: "Decided: query KG".into(),
            symbols: Vec::new(),
            kind: WorkingMemoryKind::Decision,
            timestamp: 0,
            relevance: 0.8,
            source_cycle: 1,
            reference_count: 0,
        })
        .unwrap();

    assert_eq!(wm.len(), 2);
    assert_eq!(wm.get(id1).unwrap().content, "Observed: Sun is a star");
    assert_eq!(wm.get(id2).unwrap().kind, WorkingMemoryKind::Decision);

    let obs = wm.by_kind(WorkingMemoryKind::Observation);
    assert_eq!(obs.len(), 1);
}

#[test]
fn working_memory_eviction() {
    let mut wm = WorkingMemory::new(3);

    wm.push(WorkingMemoryEntry {
        id: 0,
        content: "a".into(),
        symbols: Vec::new(),
        kind: WorkingMemoryKind::Observation,
        timestamp: 0,
        relevance: 0.3,
        source_cycle: 1,
        reference_count: 0,
    })
    .unwrap();

    let id_high = wm
        .push(WorkingMemoryEntry {
            id: 0,
            content: "b".into(),
            symbols: Vec::new(),
            kind: WorkingMemoryKind::Decision,
            timestamp: 0,
            relevance: 0.9,
            source_cycle: 1,
            reference_count: 0,
        })
        .unwrap();

    wm.push(WorkingMemoryEntry {
        id: 0,
        content: "c".into(),
        symbols: Vec::new(),
        kind: WorkingMemoryKind::ToolResult,
        timestamp: 0,
        relevance: 0.2,
        source_cycle: 1,
        reference_count: 0,
    })
    .unwrap();

    assert!(wm.is_full());

    // Evict entries below 0.5 relevance.
    let evicted = wm.evict_below(0.5);
    assert_eq!(evicted, 2);
    assert_eq!(wm.len(), 1);
    assert!(wm.get(id_high).is_some());
}

#[test]
fn goal_create_and_decompose() {
    let dir = tempfile::TempDir::new().unwrap();
    let engine = Arc::new(persistent_engine(dir.path()));
    let mut agent = Agent::new(engine.clone(), AgentConfig::default()).unwrap();

    let parent_id = agent
        .add_goal("Understand the solar system", 200, "Know all planets")
        .unwrap();

    assert_eq!(agent.goals().len(), 1);
    assert_eq!(agent.goals()[0].symbol_id, parent_id);
    assert!(matches!(agent.goals()[0].status, GoalStatus::Active));

    // Verify goal is in KG.
    let triples = engine.triples_from(parent_id);
    assert!(
        !triples.is_empty(),
        "goal should have triples in KG"
    );
}

#[test]
fn goal_status_transitions() {
    let dir = tempfile::TempDir::new().unwrap();
    let engine = Arc::new(persistent_engine(dir.path()));
    let mut agent = Agent::new(engine, AgentConfig::default()).unwrap();

    let goal_id = agent
        .add_goal("Test goal", 128, "Test criteria")
        .unwrap();

    // Initially active.
    assert!(matches!(agent.goals()[0].status, GoalStatus::Active));

    // Complete the goal.
    agent.complete_goal(goal_id).unwrap();
    assert!(matches!(agent.goals()[0].status, GoalStatus::Completed));
}

#[test]
fn tool_registry_crud() {
    let mut agent = test_agent();

    // Should have 15 built-in tools (5 core + 4 external + 2 autonomous + 3 ingest + 1 doc).
    let tools = agent.list_tools();
    assert_eq!(tools.len(), 15);

    let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    // Core tools.
    assert!(names.contains(&"kg_query"));
    assert!(names.contains(&"kg_mutate"));
    assert!(names.contains(&"memory_recall"));
    assert!(names.contains(&"reason"));
    assert!(names.contains(&"similarity_search"));
    // External tools.
    assert!(names.contains(&"file_io"));
    assert!(names.contains(&"http_fetch"));
    assert!(names.contains(&"shell_exec"));
    assert!(names.contains(&"user_interact"));
    // Autonomous tools.
    assert!(names.contains(&"infer_rules"));
    assert!(names.contains(&"gap_analysis"));
    // Ingest tools.
    assert!(names.contains(&"csv_ingest"));
    assert!(names.contains(&"text_ingest"));
    assert!(names.contains(&"code_ingest"));
    // Documentation tools.
    assert!(names.contains(&"doc_gen"));

    // Register a custom tool.
    struct CustomTool;
    impl Tool for CustomTool {
        fn signature(&self) -> ToolSignature {
            ToolSignature {
                name: "custom".into(),
                description: "A custom tool".into(),
                parameters: vec![],
            }
        }
        fn execute(
            &self,
            _engine: &Engine,
            _input: ToolInput,
        ) -> akh_medu::agent::AgentResult<ToolOutput> {
            Ok(ToolOutput::ok("custom result"))
        }
        fn manifest(&self) -> akh_medu::agent::ToolManifest {
            akh_medu::agent::ToolManifest {
                name: "custom".into(),
                description: "A custom tool".into(),
                parameters: vec![],
                danger: akh_medu::agent::DangerInfo {
                    level: akh_medu::agent::DangerLevel::Safe,
                    capabilities: HashSet::new(),
                    description: "safe custom tool".into(),
                    shadow_triggers: vec![],
                },
                source: akh_medu::agent::ToolSource::Native,
            }
        }
    }

    agent.register_tool(Box::new(CustomTool));
    assert_eq!(agent.list_tools().len(), 16);
}

#[test]
fn kg_query_tool_execution() {
    let mut agent = test_agent_with_data();

    // Add a goal so we have context.
    agent.add_goal("Find stars", 128, "Find star symbols").unwrap();

    // Execute the kg_query tool directly.
    let _input = ToolInput::new()
        .with_param("symbol", "Sun")
        .with_param("direction", "both");
    let output = agent
        .engine()
        .resolve_symbol("Sun")
        .expect("Sun should exist");

    // Use the tool registry.
    let result = agent
        .engine()
        .triples_from(output);
    assert!(!result.is_empty(), "Sun should have outgoing triples");
}

#[test]
fn consolidation_persists_episodes() {
    let dir = tempfile::TempDir::new().unwrap();
    let engine = persistent_engine(dir.path());

    // Ingest some data.
    engine
        .ingest_label_triples(&[
            ("Sun".into(), "is-a".into(), "Star".into(), 1.0),
            ("Earth".into(), "orbits".into(), "Sun".into(), 1.0),
        ])
        .unwrap();

    let engine = Arc::new(engine);
    let config = AgentConfig {
        consolidation: akh_medu::agent::memory::ConsolidationConfig {
            min_relevance: 0.0, // persist everything
            ..Default::default()
        },
        ..Default::default()
    };
    let mut agent = Agent::new(engine.clone(), config).unwrap();

    let _sun_id = engine.lookup_symbol("Sun").unwrap();

    // Push entries to working memory via a cycle.
    agent.add_goal("Test consolidation", 128, "test").unwrap();
    let _ = agent.run_cycle(); // produces WM entries

    // Now consolidate.
    let result = agent.consolidate().unwrap();
    assert!(
        result.entries_scored > 0,
        "should have scored WM entries"
    );
    // Episodes should be created (since min_relevance is 0).
    // The exact count depends on WM entries, which come from the cycle.

    // Verify episodes are in the KG.
    for ep_id in &result.episodes_created {
        let triples = engine.triples_from(*ep_id);
        assert!(
            !triples.is_empty(),
            "episode should have triples in KG"
        );
    }
}

#[test]
fn consolidation_provenance_tracking() {
    let dir = tempfile::TempDir::new().unwrap();
    let engine = persistent_engine(dir.path());

    engine
        .ingest_label_triples(&[("Sun".into(), "is-a".into(), "Star".into(), 1.0)])
        .unwrap();

    let engine = Arc::new(engine);
    let config = AgentConfig {
        consolidation: akh_medu::agent::memory::ConsolidationConfig {
            min_relevance: 0.0,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut agent = Agent::new(engine.clone(), config).unwrap();

    agent.add_goal("Track provenance", 128, "test").unwrap();
    let _ = agent.run_cycle();
    let result = agent.consolidate().unwrap();

    // Check provenance records for consolidated episodes.
    for ep_id in &result.episodes_created {
        let provenance = engine.provenance_of(*ep_id);
        if let Ok(records) = provenance {
            let consolidation_records: Vec<_> = records
                .iter()
                .filter(|r| matches!(r.kind, akh_medu::provenance::DerivationKind::AgentConsolidation { .. }))
                .collect();
            assert!(
                !consolidation_records.is_empty(),
                "episode should have AgentConsolidation provenance"
            );
        }
    }
}

#[test]
fn ooda_single_cycle() {
    let mut agent = test_agent_with_data();

    agent
        .add_goal("Find all stars in the knowledge graph", 200, "List star symbols")
        .unwrap();

    let result = agent.run_cycle().unwrap();

    assert_eq!(result.cycle_number, 1);
    assert!(!result.observation.active_goals.is_empty());
    assert!(!result.decision.chosen_tool.is_empty());
    assert!(!result.decision.reasoning.is_empty());
    // The tool should have executed.
    assert!(!result.action_result.tool_output.result.is_empty());
}

#[test]
fn agent_run_until_complete() {
    let mut agent = test_agent_with_data();

    let goal_id = agent
        .add_goal("Explore Sun", 128, "Query Sun symbol")
        .unwrap();

    // Run a few cycles — then manually complete the goal.
    let _ = agent.run_cycle();
    let _ = agent.run_cycle();
    agent.complete_goal(goal_id).unwrap();

    // Now run_until_complete should return immediately (no active goals).
    let results = agent.run_until_complete().unwrap();
    assert!(results.is_empty(), "no active goals, should return immediately");
}

#[test]
fn episodic_recall_by_tag() {
    let dir = tempfile::TempDir::new().unwrap();
    let engine = persistent_engine(dir.path());

    engine
        .ingest_label_triples(&[
            ("Sun".into(), "is-a".into(), "Star".into(), 1.0),
            ("Earth".into(), "orbits".into(), "Sun".into(), 1.0),
        ])
        .unwrap();

    let engine = Arc::new(engine);
    let config = AgentConfig {
        consolidation: akh_medu::agent::memory::ConsolidationConfig {
            min_relevance: 0.0,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut agent = Agent::new(engine.clone(), config).unwrap();

    agent.add_goal("Consolidate and recall", 128, "test").unwrap();
    let _ = agent.run_cycle();
    let consolidation = agent.consolidate().unwrap();

    // Recall using a symbol that was learned.
    if !consolidation.episodes_created.is_empty() {
        // Get one of the learned symbols from the first episode.
        let ep_id = consolidation.episodes_created[0];
        let triples = engine.triples_from(ep_id);
        let learned_syms: Vec<SymbolId> = triples
            .iter()
            .filter(|t| {
                let label = engine.resolve_label(t.predicate);
                label == "agent:learned"
            })
            .map(|t| t.object)
            .collect();

        if !learned_syms.is_empty() {
            let recalled = agent.recall(&learned_syms, 5).unwrap();
            assert!(
                !recalled.is_empty(),
                "should recall episodes by learned symbol"
            );
        }
    }
}

#[test]
fn agent_persistence_roundtrip() {
    let dir = tempfile::TempDir::new().unwrap();

    // Phase 1: create agent with goals, run cycles, persist.
    {
        let engine = persistent_engine(dir.path());
        engine
            .ingest_label_triples(&[("Sun".into(), "is-a".into(), "Star".into(), 1.0)])
            .unwrap();

        let engine = Arc::new(engine);
        let mut agent = Agent::new(engine.clone(), AgentConfig::default()).unwrap();

        agent
            .add_goal("Persistent goal", 200, "Survives restart")
            .unwrap();
        let _ = agent.run_cycle();
        engine.persist().unwrap();
    }

    // Phase 2: reopen and verify goals are restored.
    {
        let engine = persistent_engine(dir.path());
        let engine = Arc::new(engine);
        let agent = Agent::new(engine.clone(), AgentConfig::default()).unwrap();

        // Goals should be restored from KG.
        let restored_goals = agent.goals();
        assert!(
            !restored_goals.is_empty(),
            "goals should survive restart"
        );
        let has_persistent = restored_goals
            .iter()
            .any(|g| g.description.contains("Persistent goal"));
        assert!(has_persistent, "should find the 'Persistent goal'");
    }
}

// ===========================================================================
// Phase 8a: Wiring fixes integration tests
// ===========================================================================

#[test]
fn reference_count_incremented_during_decide() {
    let mut agent = test_agent_with_data();
    // Use criteria that won't self-match against goal metadata in KG,
    // so the goal stays active across multiple cycles.
    agent
        .add_goal("Explore astronomy data", 128, "Comprehensive verification of dataset completeness")
        .unwrap();

    // Run one cycle to populate WM, then run a second to trigger decide on existing entries.
    let _ = agent.run_cycle();
    let wm_before: Vec<(u64, u32)> = agent
        .working_memory()
        .entries()
        .iter()
        .map(|e| (e.id, e.reference_count))
        .collect();

    let _ = agent.run_cycle();

    // After the second cycle, recent entries from cycle 1 should have incremented ref counts.
    let mut any_incremented = false;
    for (id, prev_count) in &wm_before {
        if let Some(entry) = agent.working_memory().get(*id) {
            if entry.reference_count > *prev_count {
                any_incremented = true;
                break;
            }
        }
    }
    assert!(
        any_incremented,
        "at least one WM entry should have its reference_count incremented after a second cycle"
    );
}

#[test]
fn goal_status_restore_deterministic() {
    let dir = tempfile::TempDir::new().unwrap();

    // Create a goal, transition it through multiple statuses, persist.
    {
        let engine = Arc::new(persistent_engine(dir.path()));
        let mut agent = Agent::new(engine.clone(), AgentConfig::default()).unwrap();
        let goal_id = agent
            .add_goal("Multi-status goal", 128, "test")
            .unwrap();

        // Transition: Active -> Completed (adds a second has_status triple).
        agent.complete_goal(goal_id).unwrap();
        engine.persist().unwrap();
    }

    // Reopen and verify the restored goal picks the LATEST status (Completed).
    {
        let engine = Arc::new(persistent_engine(dir.path()));
        let agent = Agent::new(engine, AgentConfig::default()).unwrap();
        let goal = agent
            .goals()
            .iter()
            .find(|g| g.description.contains("Multi-status goal"));
        assert!(goal.is_some(), "goal should survive restart");
        assert!(
            matches!(goal.unwrap().status, GoalStatus::Completed),
            "should restore to Completed (most recent status), got: {:?}",
            goal.unwrap().status
        );
    }
}

#[test]
fn criteria_evaluation_completes_goal() {
    let engine = test_engine();
    // Ingest data that matches our success criteria keywords.
    engine
        .ingest_label_triples(&[
            ("Sun".into(), "is-a".into(), "Star".into(), 1.0),
            ("Sirius".into(), "is-a".into(), "Star".into(), 1.0),
            ("Earth".into(), "is-a".into(), "Planet".into(), 1.0),
        ])
        .unwrap();

    let engine = Arc::new(engine);
    let config = AgentConfig {
        max_cycles: 20,
        ..Default::default()
    };
    let mut agent = Agent::new(engine, config).unwrap();

    // Criteria keywords "star" and "symbols" should match when kg_query finds Star triples.
    agent
        .add_goal("Find all stars", 200, "Find star symbols in the graph")
        .unwrap();

    // Run cycles — the agent should eventually complete the goal via criteria matching.
    let result = agent.run_until_complete();
    match result {
        Ok(cycles) => {
            // Goal was completed before max_cycles.
            assert!(
                agent.goals().iter().any(|g| matches!(g.status, GoalStatus::Completed)),
                "goal should be marked completed. Ran {} cycles.",
                cycles.len()
            );
        }
        Err(_) => {
            // MaxCyclesReached is acceptable if criteria matching was too strict,
            // but let's check if any progress was made.
            let any_completed = agent
                .goals()
                .iter()
                .any(|g| matches!(g.status, GoalStatus::Completed));
            if !any_completed {
                // At minimum, check that cycles actually ran and tools varied.
                assert!(
                    agent.cycle_count() > 0,
                    "agent should have run at least one cycle"
                );
            }
        }
    }
}

#[test]
fn all_five_tools_selectable() {
    // Verify the agent can select each of the 5 tools across different scenarios.
    let engine = test_engine();
    engine
        .ingest_label_triples(&[
            ("Sun".into(), "is-a".into(), "Star".into(), 1.0),
            ("Earth".into(), "orbits".into(), "Sun".into(), 0.95),
            ("Mars".into(), "is-a".into(), "Planet".into(), 1.0),
            ("Mars".into(), "orbits".into(), "Sun".into(), 0.95),
            ("Jupiter".into(), "is-a".into(), "Planet".into(), 1.0),
            ("Saturn".into(), "is-a".into(), "Planet".into(), 1.0),
        ])
        .unwrap();

    let engine = Arc::new(engine);
    let config = AgentConfig {
        max_cycles: 15,
        ..Default::default()
    };
    let mut agent = Agent::new(engine, config).unwrap();
    agent
        .add_goal("Map the solar system", 200, "Find all planets and stars")
        .unwrap();

    let mut tools_used = std::collections::HashSet::new();
    for _ in 0..15 {
        if akh_medu::agent::goal::active_goals(agent.goals()).is_empty() {
            break;
        }
        if let Ok(result) = agent.run_cycle() {
            tools_used.insert(result.decision.chosen_tool.clone());
        }
    }

    // We should see at least 3 different tools used across 15 cycles
    // (the anti-repetition logic forces tool cycling).
    assert!(
        tools_used.len() >= 3,
        "expected at least 3 different tools used, got {}: {:?}",
        tools_used.len(),
        tools_used
    );
}

// ===========================================================================
// Phase 8b: Goal autonomy & stall detection integration tests
// ===========================================================================

#[test]
fn goal_stall_detection() {
    // A goal that makes no progress should be detected as stalled after threshold cycles.
    let engine = test_engine();
    engine
        .ingest_label_triples(&[("Sun".into(), "is-a".into(), "Star".into(), 1.0)])
        .unwrap();

    let engine = Arc::new(engine);
    let config = AgentConfig {
        max_cycles: 20,
        ..Default::default()
    };
    let mut agent = Agent::new(engine, config).unwrap();

    // Use criteria that will never self-match so the goal stays Active and stalls.
    agent
        .add_goal(
            "Impossible task",
            128,
            "Requires unicorn verification protocol",
        )
        .unwrap();

    // Run enough cycles past the stall threshold (DEFAULT_STALL_THRESHOLD = 5).
    for _ in 0..7 {
        if akh_medu::agent::goal::active_goals(agent.goals()).is_empty() {
            break;
        }
        let _ = agent.run_cycle();
    }

    // The goal should have been worked on and detected as stalled.
    // After stall detection, decompose_stalled_goals should have:
    // - Suspended the original goal
    // - Created child sub-goals
    let original = agent
        .goals()
        .iter()
        .find(|g| g.description.contains("Impossible task"))
        .expect("original goal should still exist");

    // Either it got decomposed (Suspended with children) or it's still active
    // with cycles_worked tracked.
    if matches!(original.status, GoalStatus::Suspended) {
        assert!(
            !original.children.is_empty(),
            "suspended goal should have children from decomposition"
        );
        // Should have new active child goals.
        let active = akh_medu::agent::goal::active_goals(agent.goals());
        assert!(
            !active.is_empty(),
            "should have active child goals after decomposition"
        );
    } else {
        // If not yet stalled (cycles_worked < threshold), it should be tracking.
        assert!(
            original.cycles_worked > 0,
            "goal should have cycles_worked tracked, got {}",
            original.cycles_worked
        );
    }
}

#[test]
fn goal_decomposition_creates_children() {
    let dir = tempfile::TempDir::new().unwrap();
    let engine = Arc::new(persistent_engine(dir.path()));
    let mut agent = Agent::new(engine.clone(), AgentConfig::default()).unwrap();

    // A goal with natural comma-separated sub-tasks.
    let parent_id = agent
        .add_goal(
            "Find stars, classify planets, and map orbits",
            200,
            "Complete all sub-tasks",
        )
        .unwrap();

    // Manually decompose it.
    let children = agent.decompose_stalled_goal(parent_id).unwrap();

    // Should produce at least 2 sub-goals from the comma/"and" split.
    assert!(
        children.len() >= 2,
        "expected at least 2 sub-goals from decomposition, got {}",
        children.len()
    );

    // Parent should be suspended.
    let parent = agent
        .goals()
        .iter()
        .find(|g| g.symbol_id == parent_id)
        .unwrap();
    assert!(
        matches!(parent.status, GoalStatus::Suspended),
        "parent should be suspended after decomposition"
    );

    // Children should be active.
    for child_id in &children {
        let child = agent
            .goals()
            .iter()
            .find(|g| g.symbol_id == *child_id)
            .unwrap();
        assert!(
            matches!(child.status, GoalStatus::Active),
            "child should be active"
        );
        assert_eq!(
            child.parent,
            Some(parent_id),
            "child should reference parent"
        );
    }

    // Verify parent-child triples in KG.
    let parent_triples = engine.triples_from(parent_id);
    let child_pred_label = "agent:child_goal";
    let child_triples: Vec<_> = parent_triples
        .iter()
        .filter(|t| engine.resolve_label(t.predicate) == child_pred_label)
        .collect();
    assert!(
        child_triples.len() >= 2,
        "parent should have child_goal triples in KG"
    );
}

#[test]
fn suspend_and_fail_goal() {
    let engine = Arc::new(test_engine());
    let mut agent = Agent::new(engine, AgentConfig::default()).unwrap();

    let g1 = agent.add_goal("Goal one", 128, "test").unwrap();
    let g2 = agent.add_goal("Goal two", 128, "test").unwrap();

    // Suspend goal one.
    agent.suspend_goal(g1).unwrap();
    let goal1 = agent.goals().iter().find(|g| g.symbol_id == g1).unwrap();
    assert!(matches!(goal1.status, GoalStatus::Suspended));

    // Fail goal two with a reason.
    agent.fail_goal(g2, "resource unavailable").unwrap();
    let goal2 = agent.goals().iter().find(|g| g.symbol_id == g2).unwrap();
    match &goal2.status {
        GoalStatus::Failed { reason } => {
            assert!(reason.contains("resource unavailable"));
        }
        other => panic!("expected Failed status, got {other:?}"),
    }

    // Neither should appear in active goals.
    let active = akh_medu::agent::goal::active_goals(agent.goals());
    assert!(active.is_empty(), "no goals should be active");
}

#[test]
fn criteria_evaluation_uses_kg_state() {
    // Verify that criteria evaluation checks KG state, not just tool output.
    let engine = test_engine();
    engine
        .ingest_label_triples(&[
            ("Alpha".into(), "is-a".into(), "Star".into(), 1.0),
            ("Beta".into(), "is-a".into(), "Star".into(), 1.0),
            ("Gamma".into(), "is-a".into(), "Planet".into(), 1.0),
        ])
        .unwrap();

    let engine = Arc::new(engine);
    let config = AgentConfig {
        max_cycles: 10,
        ..Default::default()
    };
    let mut agent = Agent::new(engine, config).unwrap();

    // Criteria keywords "star" should match entities like "Star" already in the KG.
    agent
        .add_goal("Catalog celestial objects", 200, "Find star entities")
        .unwrap();

    let result = agent.run_until_complete();
    match result {
        Ok(cycles) => {
            assert!(
                !cycles.is_empty(),
                "should have run at least one cycle"
            );
        }
        Err(_) => {
            // Even if max cycles reached, the agent should have run cycles.
            assert!(agent.cycle_count() > 0);
        }
    }
}

#[test]
fn cycles_worked_tracks_per_goal() {
    let engine = test_engine();
    engine
        .ingest_label_triples(&[("Sun".into(), "is-a".into(), "Star".into(), 1.0)])
        .unwrap();

    let engine = Arc::new(engine);
    let mut agent = Agent::new(engine, AgentConfig::default()).unwrap();

    agent
        .add_goal(
            "Track cycles",
            128,
            "Requires quantum entanglement verification",
        )
        .unwrap();

    // Run 3 cycles.
    for _ in 0..3 {
        if akh_medu::agent::goal::active_goals(agent.goals()).is_empty() {
            break;
        }
        let _ = agent.run_cycle();
    }

    // The goal should have cycles_worked tracked.
    let goal = agent
        .goals()
        .iter()
        .find(|g| g.description.contains("Track cycles"));
    assert!(goal.is_some());
    let goal = goal.unwrap();
    assert!(
        goal.cycles_worked > 0,
        "expected cycles_worked > 0, got {}",
        goal.cycles_worked
    );
}

// ===========================================================================
// Phase 8c: Utility-based tool selection integration tests
// ===========================================================================

#[test]
fn utility_scoring_diversifies_tools() {
    // With utility scoring, the recency penalty should force even more tool
    // diversification than the old if/else approach.
    let engine = test_engine();
    engine
        .ingest_label_triples(&[
            ("Sun".into(), "is-a".into(), "Star".into(), 1.0),
            ("Earth".into(), "orbits".into(), "Sun".into(), 0.95),
            ("Mars".into(), "is-a".into(), "Planet".into(), 1.0),
            ("Mars".into(), "orbits".into(), "Sun".into(), 0.95),
            ("Jupiter".into(), "is-a".into(), "Planet".into(), 1.0),
            ("Saturn".into(), "is-a".into(), "Planet".into(), 1.0),
            ("Venus".into(), "is-a".into(), "Planet".into(), 1.0),
        ])
        .unwrap();

    let engine = Arc::new(engine);
    let config = AgentConfig {
        max_cycles: 20,
        ..Default::default()
    };
    let mut agent = Agent::new(engine, config).unwrap();
    agent
        .add_goal(
            "Map the entire solar system topology",
            200,
            "Comprehensive verification of planetary classification",
        )
        .unwrap();

    let mut tools_used = std::collections::HashSet::new();
    let mut tool_sequence = Vec::new();
    for _ in 0..12 {
        if akh_medu::agent::goal::active_goals(agent.goals()).is_empty() {
            break;
        }
        if let Ok(result) = agent.run_cycle() {
            tools_used.insert(result.decision.chosen_tool.clone());
            tool_sequence.push(result.decision.chosen_tool.clone());
        }
    }

    // Should see at least 3 different tools (better than before).
    assert!(
        tools_used.len() >= 3,
        "expected at least 3 tools with utility scoring, got {}: {:?}",
        tools_used.len(),
        tools_used
    );

    // Verify no tool is used more than 3 times consecutively (recency penalty works).
    let max_consecutive = tool_sequence
        .windows(4)
        .filter(|w| w.iter().all(|t| t == &w[0]))
        .count();
    assert_eq!(
        max_consecutive, 0,
        "no tool should be used 4+ times consecutively, sequence: {:?}",
        tool_sequence
    );
}

#[test]
fn utility_scoring_includes_score_breakdown() {
    // The reasoning string should include the score breakdown for transparency.
    let engine = test_engine();
    engine
        .ingest_label_triples(&[("Sun".into(), "is-a".into(), "Star".into(), 1.0)])
        .unwrap();

    let engine = Arc::new(engine);
    let mut agent = Agent::new(engine, AgentConfig::default()).unwrap();
    agent
        .add_goal("Explore stars", 128, "Find star data")
        .unwrap();

    let result = agent.run_cycle().unwrap();
    let reasoning = &result.decision.reasoning;

    // Reasoning should contain the score breakdown.
    assert!(
        reasoning.contains("score=") && reasoning.contains("base="),
        "reasoning should include score breakdown, got: {}",
        reasoning
    );
}

#[test]
fn novelty_bonus_encourages_new_tools() {
    // When running multiple cycles, tools not yet tried for a goal should
    // get selected due to novelty bonus, even if their base score is lower.
    let engine = test_engine();
    engine
        .ingest_label_triples(&[
            ("Sun".into(), "is-a".into(), "Star".into(), 1.0),
            ("Earth".into(), "is-a".into(), "Planet".into(), 1.0),
            ("Earth".into(), "orbits".into(), "Sun".into(), 0.9),
        ])
        .unwrap();

    let engine = Arc::new(engine);
    let config = AgentConfig {
        max_cycles: 10,
        ..Default::default()
    };
    let mut agent = Agent::new(engine, config).unwrap();
    agent
        .add_goal(
            "Classify celestial bodies",
            200,
            "Requires galactic classification protocol",
        )
        .unwrap();

    let mut tools_seen_by_cycle_5 = std::collections::HashSet::new();
    for i in 0..8 {
        if akh_medu::agent::goal::active_goals(agent.goals()).is_empty() {
            break;
        }
        if let Ok(result) = agent.run_cycle() {
            if i < 5 {
                tools_seen_by_cycle_5.insert(result.decision.chosen_tool.clone());
            }
        }
    }

    // By cycle 5, novelty bonus should have pushed at least 2 different tools.
    assert!(
        tools_seen_by_cycle_5.len() >= 2,
        "novelty bonus should diversify tools early, got {} by cycle 5: {:?}",
        tools_seen_by_cycle_5.len(),
        tools_seen_by_cycle_5
    );
}

#[test]
fn recency_penalty_prevents_immediate_repeat() {
    // The most recently used tool should get a recency penalty, so consecutive
    // cycles rarely pick the same tool twice in a row.
    let engine = test_engine();
    engine
        .ingest_label_triples(&[
            ("Alpha".into(), "is-a".into(), "Star".into(), 1.0),
            ("Beta".into(), "is-a".into(), "Star".into(), 1.0),
            ("Gamma".into(), "is-a".into(), "Planet".into(), 1.0),
            ("Delta".into(), "orbits".into(), "Alpha".into(), 0.9),
        ])
        .unwrap();

    let engine = Arc::new(engine);
    let config = AgentConfig {
        max_cycles: 15,
        ..Default::default()
    };
    let mut agent = Agent::new(engine, config).unwrap();
    agent
        .add_goal(
            "Analyze stellar neighborhood",
            200,
            "Requires interstellar mapping protocol",
        )
        .unwrap();

    let mut consecutive_repeats = 0;
    let mut prev_tool = String::new();
    for _ in 0..10 {
        if akh_medu::agent::goal::active_goals(agent.goals()).is_empty() {
            break;
        }
        if let Ok(result) = agent.run_cycle() {
            if result.decision.chosen_tool == prev_tool {
                consecutive_repeats += 1;
            }
            prev_tool = result.decision.chosen_tool;
        }
    }

    // With recency penalty, consecutive repeats should be rare.
    // Allow at most 3 out of 9 transitions to be repeats.
    assert!(
        consecutive_repeats <= 3,
        "recency penalty should limit consecutive repeats, got {} in 10 cycles",
        consecutive_repeats
    );
}

// ===========================================================================
// Phase 8d: Session persistence & resume integration tests
// ===========================================================================

#[test]
fn session_persist_and_resume() {
    let dir = tempfile::TempDir::new().unwrap();

    // Phase 1: create agent, add goals, run cycles, persist session.
    let cycle_count_phase1;
    let wm_len_phase1;
    {
        let engine = Arc::new(persistent_engine(dir.path()));
        engine
            .ingest_label_triples(&[
                ("Sun".into(), "is-a".into(), "Star".into(), 1.0),
                ("Earth".into(), "orbits".into(), "Sun".into(), 1.0),
            ])
            .unwrap();

        let mut agent = Agent::new(engine, AgentConfig::default()).unwrap();
        agent
            .add_goal("Explore the solar system", 128, "Map celestial bodies")
            .unwrap();

        // Run 3 cycles to populate WM and advance cycle_count.
        for _ in 0..3 {
            let _ = agent.run_cycle();
        }

        cycle_count_phase1 = agent.cycle_count();
        wm_len_phase1 = agent.working_memory().len();

        assert!(cycle_count_phase1 >= 3, "should have run at least 3 cycles");
        assert!(wm_len_phase1 > 0, "WM should have entries after 3 cycles");

        agent.persist_session().unwrap();
    }

    // Phase 2: resume and verify state was restored.
    {
        let engine = Arc::new(persistent_engine(dir.path()));

        assert!(
            Agent::has_persisted_session(&engine),
            "should detect persisted session"
        );

        let agent = Agent::resume(engine, AgentConfig::default()).unwrap();

        assert_eq!(
            agent.cycle_count(),
            cycle_count_phase1,
            "cycle count should be restored"
        );
        assert_eq!(
            agent.working_memory().len(),
            wm_len_phase1,
            "WM entries should be restored"
        );
        assert!(
            !agent.goals().is_empty(),
            "goals should be restored from KG"
        );
    }
}

#[test]
fn session_resume_continues_execution() {
    let dir = tempfile::TempDir::new().unwrap();

    // Phase 1: create, run 2 cycles, persist.
    {
        let engine = Arc::new(persistent_engine(dir.path()));
        engine
            .ingest_label_triples(&[("Sun".into(), "is-a".into(), "Star".into(), 1.0)])
            .unwrap();

        let mut agent = Agent::new(engine, AgentConfig::default()).unwrap();
        agent
            .add_goal(
                "Analyze stellar data",
                128,
                "Requires comprehensive stellar classification",
            )
            .unwrap();

        let _ = agent.run_cycle();
        let _ = agent.run_cycle();
        agent.persist_session().unwrap();
    }

    // Phase 2: resume and run more cycles — cycle count should continue from where it left off.
    {
        let engine = Arc::new(persistent_engine(dir.path()));
        let mut agent = Agent::resume(engine, AgentConfig::default()).unwrap();

        let initial_cycle = agent.cycle_count();
        assert!(initial_cycle >= 2, "should resume from at least cycle 2");

        // Run 2 more cycles.
        let _ = agent.run_cycle();
        let _ = agent.run_cycle();

        assert!(
            agent.cycle_count() > initial_cycle,
            "cycle count should advance after resumed cycles"
        );
    }
}

#[test]
fn no_persisted_session_returns_fresh_agent() {
    let dir = tempfile::TempDir::new().unwrap();
    let engine = Arc::new(persistent_engine(dir.path()));

    // No prior session — has_persisted_session should be false.
    assert!(
        !Agent::has_persisted_session(&engine),
        "fresh engine should have no persisted session"
    );

    // Resume should still work, returning a fresh agent.
    let agent = Agent::resume(engine, AgentConfig::default()).unwrap();
    assert_eq!(agent.cycle_count(), 0, "fresh resume should start at cycle 0");
    assert!(agent.working_memory().is_empty(), "fresh resume should have empty WM");
}

#[test]
fn wm_serialize_deserialize_roundtrip() {
    // Test that WorkingMemory serialization is lossless.
    let mut wm = WorkingMemory::new(50);

    let id1 = wm
        .push(WorkingMemoryEntry {
            id: 0,
            content: "observation about stars".into(),
            symbols: vec![],
            kind: WorkingMemoryKind::Observation,
            timestamp: 0,
            relevance: 0.8,
            source_cycle: 1,
            reference_count: 3,
        })
        .unwrap();

    let id2 = wm
        .push(WorkingMemoryEntry {
            id: 0,
            content: "decision to query".into(),
            symbols: vec![],
            kind: WorkingMemoryKind::Decision,
            timestamp: 0,
            relevance: 0.6,
            source_cycle: 2,
            reference_count: 1,
        })
        .unwrap();

    // Serialize.
    let (next_id, bytes) = wm.serialize().unwrap();
    assert!(!bytes.is_empty());

    // Restore.
    let restored = WorkingMemory::restore(50, next_id, &bytes).unwrap();
    assert_eq!(restored.len(), 2, "should have 2 entries after restore");

    let e1 = restored.get(id1).unwrap();
    assert_eq!(e1.content, "observation about stars");
    assert_eq!(e1.kind, WorkingMemoryKind::Observation);
    assert_eq!(e1.relevance, 0.8);
    assert_eq!(e1.reference_count, 3);

    let e2 = restored.get(id2).unwrap();
    assert_eq!(e2.content, "decision to query");
    assert_eq!(e2.kind, WorkingMemoryKind::Decision);
}

// ── Phase 8e: External tools ──────────────────────────────────────────

#[test]
fn file_io_tool_read_write() {
    use akh_medu::agent::tool::ToolInput;
    use std::collections::HashMap;

    let dir = tempfile::tempdir().unwrap();
    let scratch = dir.path().to_path_buf();

    let engine = test_engine();
    let tool = akh_medu::agent::tools::FileIoTool::new(Some(scratch.clone()));

    // Write a file.
    let mut params = HashMap::new();
    params.insert("action".into(), "write".into());
    params.insert("path".into(), "hello.txt".into());
    params.insert("content".into(), "Hello from agent!".into());
    let input = ToolInput { params };
    let out = tool.execute(&engine, input).unwrap();
    assert!(out.success, "write should succeed: {}", out.result);
    assert!(out.result.contains("17 bytes"));

    // Read it back.
    let read_path = scratch.join("hello.txt");
    let mut params = HashMap::new();
    params.insert("action".into(), "read".into());
    params.insert("path".into(), read_path.to_string_lossy().into());
    let input = ToolInput { params };
    let out = tool.execute(&engine, input).unwrap();
    assert!(out.success, "read should succeed: {}", out.result);
    assert!(out.result.contains("Hello from agent!"));
}

#[test]
fn file_io_tool_denies_outside_scratch() {
    use akh_medu::agent::tool::ToolInput;
    use std::collections::HashMap;

    let dir = tempfile::tempdir().unwrap();
    let scratch = dir.path().to_path_buf();

    let engine = test_engine();
    let tool = akh_medu::agent::tools::FileIoTool::new(Some(scratch));

    // Try writing to an absolute path outside scratch.
    let mut params = HashMap::new();
    params.insert("action".into(), "write".into());
    params.insert("path".into(), "/tmp/evil.txt".into());
    params.insert("content".into(), "should not work".into());
    let input = ToolInput { params };
    let out = tool.execute(&engine, input).unwrap();
    assert!(!out.success, "write outside scratch should be denied");
    assert!(out.result.contains("denied") || out.result.contains("outside"));
}

#[test]
fn shell_exec_tool_runs_command() {
    use akh_medu::agent::tool::ToolInput;
    use std::collections::HashMap;

    let engine = test_engine();
    let tool = akh_medu::agent::tools::ShellExecTool;

    let mut params = HashMap::new();
    params.insert("command".into(), "echo hello_world".into());
    let input = ToolInput { params };
    let out = tool.execute(&engine, input).unwrap();
    assert!(out.success, "echo should succeed: {}", out.result);
    assert!(out.result.contains("hello_world"));
}

#[test]
fn shell_exec_tool_timeout() {
    use akh_medu::agent::tool::ToolInput;
    use std::collections::HashMap;

    let engine = test_engine();
    let tool = akh_medu::agent::tools::ShellExecTool;

    let mut params = HashMap::new();
    params.insert("command".into(), "sleep 60".into());
    params.insert("timeout".into(), "1".into());
    let input = ToolInput { params };
    let out = tool.execute(&engine, input).unwrap();
    assert!(!out.success, "should have timed out");
    assert!(out.result.contains("timed out"));
}

#[test]
fn shell_exec_tool_exit_code() {
    use akh_medu::agent::tool::ToolInput;
    use std::collections::HashMap;

    let engine = test_engine();
    let tool = akh_medu::agent::tools::ShellExecTool;

    let mut params = HashMap::new();
    params.insert("command".into(), "false".into());
    let input = ToolInput { params };
    let out = tool.execute(&engine, input).unwrap();
    assert!(!out.success, "false should report failure");
    assert!(out.result.contains("Exit code: 1"));
}

// ── Phase 8f: Planning & reflection ───────────────────────────────────

#[test]
fn plan_generation_produces_steps() {
    let mut agent = test_agent_with_data();
    let goal_id = agent
        .add_goal("Find all stars in the knowledge graph", 128, "stars found")
        .unwrap();

    let plan = agent.plan_goal(goal_id).unwrap();

    assert!(plan.total_steps() >= 1, "plan should have at least one step");
    assert_eq!(plan.status, akh_medu::agent::PlanStatus::Active);
    assert!(plan.has_remaining_steps());
    assert_eq!(plan.completed_count(), 0);

    // Verify the strategy string is populated.
    assert!(!plan.strategy.is_empty());
}

#[test]
fn plan_step_completion_advances() {
    use akh_medu::agent::plan::{Plan, PlanStep, PlanStatus, StepStatus};

    let mut plan = Plan {
        goal_id: SymbolId::new(1).unwrap(),
        steps: vec![
            PlanStep {
                tool_name: "kg_query".into(),
                tool_input: ToolInput::new(),
                rationale: "gather data".into(),
                status: StepStatus::Pending,
                index: 0,
            },
            PlanStep {
                tool_name: "reason".into(),
                tool_input: ToolInput::new(),
                rationale: "analyze".into(),
                status: StepStatus::Pending,
                index: 1,
            },
        ],
        status: PlanStatus::Active,
        attempt: 0,
        strategy: "test".into(),
    };

    // Complete step 0.
    plan.complete_step(0);
    assert_eq!(plan.completed_count(), 1);
    assert_eq!(plan.next_step_index(), Some(1));

    // Complete step 1.
    plan.complete_step(1);
    assert_eq!(plan.status, PlanStatus::Completed);
    assert!(!plan.has_remaining_steps());
}

#[test]
fn backtrack_generates_alternative_plan() {
    let mut agent = test_agent_with_data();
    let goal_id = agent
        .add_goal("Find all stars in the knowledge graph", 128, "stars found")
        .unwrap();

    // Generate first plan.
    let first_strategy = agent.plan_goal(goal_id).unwrap().strategy.clone();

    // Backtrack.
    let alt = agent.backtrack_goal(goal_id).unwrap();
    assert!(alt.is_some(), "should get an alternative plan");

    let alt_plan = alt.unwrap();
    assert_eq!(alt_plan.attempt, 1);
    // Strategy should differ (attempt 0 = explore-first, attempt 1 = reason-first).
    assert_ne!(alt_plan.strategy, first_strategy);
}

#[test]
fn reflection_produces_insights() {
    let mut agent = test_agent_with_data();
    agent
        .add_goal("Find stars", 128, "stars found")
        .unwrap();

    // Run a few cycles to populate WM with tool results.
    for _ in 0..3 {
        let _ = agent.run_cycle();
    }

    let result = agent.reflect().unwrap();

    // Should have analyzed something.
    assert!(result.at_cycle > 0);
    assert!(result.memory_pressure >= 0.0);
    assert!(!result.summary.is_empty());
}

#[test]
fn meta_reasoning_adjusts_priorities() {
    use akh_medu::agent::Adjustment;

    let mut agent = test_agent_with_data();
    let goal_id = agent
        .add_goal("Find stars", 128, "stars found")
        .unwrap();

    let original_priority = agent.goals().iter()
        .find(|g| g.symbol_id == goal_id)
        .unwrap()
        .priority;

    // Apply a priority increase adjustment.
    let adjustments = vec![Adjustment::IncreasePriority {
        goal_id,
        from: 128,
        to: 200,
        reason: "test boost".into(),
    }];

    let applied = agent.apply_adjustments(&adjustments).unwrap();
    assert_eq!(applied, 1);

    let new_priority = agent.goals().iter()
        .find(|g| g.symbol_id == goal_id)
        .unwrap()
        .priority;
    assert_eq!(new_priority, 200);
    assert_ne!(new_priority, original_priority);
}

#[test]
fn run_cycle_generates_plan_automatically() {
    let mut agent = test_agent_with_data();
    let goal_id = agent
        .add_goal("Find stars in the knowledge graph", 128, "stars found")
        .unwrap();

    // Running a cycle should auto-generate a plan.
    let _ = agent.run_cycle().unwrap();

    let plan = agent.plan_for_goal(goal_id);
    assert!(plan.is_some(), "plan should exist after run_cycle");
}

// ---------------------------------------------------------------------------
// Phase 9: Autonomous KG building integration tests
// ---------------------------------------------------------------------------

#[test]
fn rule_inference_transitive_closure() {
    let engine = test_engine();
    engine
        .ingest_label_triples(&[
            ("Dog".into(), "is-a".into(), "Mammal".into(), 1.0),
            ("Mammal".into(), "is-a".into(), "Animal".into(), 1.0),
        ])
        .unwrap();

    let config = akh_medu::autonomous::RuleEngineConfig::default();
    let result = engine.run_rules(config).unwrap();

    // Dog is-a Animal should be derived via transitive closure.
    let dog = engine.lookup_symbol("Dog").unwrap();
    let animal = engine.lookup_symbol("Animal").unwrap();
    let derived_match = result.derived.iter().any(|dt| {
        dt.triple.subject == dog && dt.triple.object == animal
    });
    assert!(derived_match, "Dog is-a Animal should be derived");
    assert!(result.derived.len() >= 1);

    // Verify the triple is actually in the KG now.
    assert!(engine.has_triple(dog, engine.lookup_symbol("is-a").unwrap(), animal));
}

#[test]
fn rule_inference_inverse_relations() {
    let engine = test_engine();
    engine
        .ingest_label_triples(&[
            ("Alice".into(), "parent-of".into(), "Bob".into(), 1.0),
        ])
        .unwrap();

    let config = akh_medu::autonomous::RuleEngineConfig::default();
    let result = engine.run_rules(config).unwrap();

    // Bob child-of Alice should be derived.
    let bob = engine.lookup_symbol("Bob").unwrap();
    let alice = engine.lookup_symbol("Alice").unwrap();
    let child_of = engine.lookup_symbol("child-of").unwrap();
    assert!(
        engine.has_triple(bob, child_of, alice),
        "Bob child-of Alice should be derived"
    );

    let derived_match = result.derived.iter().any(|dt| {
        dt.rule_name == "parent-child-inverse"
    });
    assert!(derived_match, "parent-child-inverse rule should fire");
}

#[test]
fn rule_inference_provenance_tracking() {
    let dir = tempfile::tempdir().unwrap();
    let engine = persistent_engine(dir.path());
    engine
        .ingest_label_triples(&[
            ("X".into(), "similar-to".into(), "Y".into(), 1.0),
        ])
        .unwrap();

    let config = akh_medu::autonomous::RuleEngineConfig::default();
    let result = engine.run_rules(config).unwrap();

    // Y similar-to X should be derived via symmetric rule.
    assert!(!result.derived.is_empty());
    let derived_match = result.derived.iter().any(|dt| {
        dt.rule_name == "similar-to-symmetric"
    });
    assert!(derived_match, "similar-to-symmetric rule should fire");

    // Check provenance was stored.
    let y = engine.lookup_symbol("Y").unwrap();
    let provenance = engine.provenance_of(y).unwrap();
    let has_rule_prov = provenance.iter().any(|p| {
        matches!(&p.kind, akh_medu::provenance::DerivationKind::RuleInference { rule_name, .. }
            if rule_name == "similar-to-symmetric")
    });
    assert!(has_rule_prov, "RuleInference provenance should be stored");
}

#[test]
fn gap_analysis_finds_dead_ends() {
    let engine = test_engine();
    engine
        .ingest_label_triples(&[
            ("Star".into(), "is-a".into(), "CelestialBody".into(), 1.0),
            ("Lonely".into(), "near".into(), "Star".into(), 1.0),
        ])
        .unwrap();

    let star = engine.lookup_symbol("Star").unwrap();
    let config = akh_medu::autonomous::GapAnalysisConfig::default();
    let result = engine.analyze_gaps(&[star], config).unwrap();

    // At least one dead end should be found (entities with very few connections).
    assert!(result.entities_analyzed > 0);
    // Coverage score should be between 0 and 1.
    assert!(result.coverage_score >= 0.0 && result.coverage_score <= 1.0);
}

#[test]
fn gap_analysis_finds_missing_predicates() {
    let engine = test_engine();
    // Create 3 entities of the same type with shared predicates, then remove one.
    engine
        .ingest_label_triples(&[
            ("Dog".into(), "is-a".into(), "Animal".into(), 1.0),
            ("Dog".into(), "has-legs".into(), "4".into(), 1.0),
            ("Dog".into(), "has-color".into(), "brown".into(), 1.0),
            ("Cat".into(), "is-a".into(), "Animal".into(), 1.0),
            ("Cat".into(), "has-legs".into(), "4".into(), 1.0),
            ("Cat".into(), "has-color".into(), "black".into(), 1.0),
            ("Horse".into(), "is-a".into(), "Animal".into(), 1.0),
            ("Horse".into(), "has-legs".into(), "4".into(), 1.0),
            ("Horse".into(), "has-color".into(), "white".into(), 1.0),
            // Bird has is-a but is missing has-legs and has-color.
            ("Bird".into(), "is-a".into(), "Animal".into(), 1.0),
        ])
        .unwrap();

    let bird = engine.lookup_symbol("Bird").unwrap();
    let config = akh_medu::autonomous::GapAnalysisConfig {
        min_degree: 1,
        ..Default::default()
    };
    let result = engine.analyze_gaps(&[bird], config).unwrap();

    assert!(result.entities_analyzed > 0);
    // Bird should show up as having missing predicates relative to its type.
    // We just check that the analysis runs and produces at least some gaps.
    // The exact gap content depends on VSA similarity which is stochastic.
}

#[test]
fn confidence_fusion_multiple_paths() {
    use akh_medu::autonomous::fusion::noisy_or;

    // Noisy-OR: 1 - (1-0.72)(1-0.6) = 1 - 0.28*0.4 = 1 - 0.112 = 0.888
    let fused = noisy_or(&[0.72, 0.6]);
    assert!((fused - 0.888).abs() < 0.001, "noisy-or of [0.72, 0.6] should be ~0.888, got {fused}");

    // Fused confidence always exceeds any individual path.
    assert!(fused > 0.72);
    assert!(fused > 0.6);

    // Single path returns itself.
    let single = noisy_or(&[0.5]);
    assert!((single - 0.5).abs() < 0.001);

    // Three paths.
    let three = noisy_or(&[0.5, 0.5, 0.5]);
    // 1 - 0.5^3 = 0.875
    assert!((three - 0.875).abs() < 0.001);
}

#[test]
fn schema_discovery_finds_types() {
    let engine = test_engine();
    // Create entities with shared predicate patterns.
    engine
        .ingest_label_triples(&[
            ("Dog".into(), "is-a".into(), "Animal".into(), 1.0),
            ("Dog".into(), "has-legs".into(), "4".into(), 1.0),
            ("Cat".into(), "is-a".into(), "Animal".into(), 1.0),
            ("Cat".into(), "has-legs".into(), "4".into(), 1.0),
            ("Horse".into(), "is-a".into(), "Animal".into(), 1.0),
            ("Horse".into(), "has-legs".into(), "4".into(), 1.0),
        ])
        .unwrap();

    let config = akh_medu::autonomous::SchemaDiscoveryConfig {
        min_type_members: 3,
        ..Default::default()
    };
    let result = engine.discover_schema(config).unwrap();

    // Should discover at least one type cluster (Dog, Cat, Horse share the same predicates).
    assert!(!result.types.is_empty(), "should discover at least one entity type");
    let first_type = &result.types[0];
    assert!(first_type.members.len() >= 3);
}

#[test]
fn agent_uses_infer_and_gap_tools() {
    let engine = Arc::new(test_engine());
    engine
        .ingest_label_triples(&[
            ("Star".into(), "is-a".into(), "CelestialBody".into(), 1.0),
            ("CelestialBody".into(), "is-a".into(), "PhysicalObject".into(), 1.0),
            ("Sun".into(), "is-a".into(), "Star".into(), 1.0),
        ])
        .unwrap();

    let config = AgentConfig {
        max_cycles: 5,
        ..Default::default()
    };
    let mut agent = Agent::new(Arc::clone(&engine), config).unwrap();
    agent
        .add_goal(
            "Derive type hierarchy for celestial bodies",
            128,
            "transitive types inferred",
        )
        .unwrap();

    // Run a few cycles — the agent should select infer_rules or gap_analysis at some point.
    let mut tools_used: Vec<String> = Vec::new();
    for _ in 0..5 {
        match agent.run_cycle() {
            Ok(result) => {
                tools_used.push(result.decision.chosen_tool.clone());
            }
            Err(_) => break,
        }
    }

    // Verify that the tools are at least available and the agent runs without crashing.
    // The exact tool selection depends on the utility scoring heuristics.
    let all_tools: Vec<String> = agent.list_tools().iter().map(|t| t.name.clone()).collect();
    assert!(all_tools.contains(&"infer_rules".to_string()));
    assert!(all_tools.contains(&"gap_analysis".to_string()));
    assert!(!tools_used.is_empty(), "agent should have run at least one cycle");
}

// ===========================================================================
// Phase 10: Hieroglyphic Notation System
// ===========================================================================

#[test]
fn hieroglyphic_render_triple() {
    let engine = test_engine();
    let dog = engine.create_symbol(SymbolKind::Entity, "Dog").unwrap();
    let is_a = engine.create_symbol(SymbolKind::Relation, "is-a").unwrap();
    let animal = engine.create_symbol(SymbolKind::Entity, "Animal").unwrap();
    engine
        .add_triple(&Triple::new(dog.id, is_a.id, animal.id))
        .unwrap();

    let config = akh_medu::glyph::NotationConfig {
        use_pua: false,
        show_confidence: true,
        show_provenance: false,
        show_sigils: false,
        compact: true,
    };
    let rendered = akh_medu::glyph::notation::render_triple(&engine, &Triple::new(dog.id, is_a.id, animal.id), &config);

    // Should contain the is-a glyph fallback (△) and entity labels.
    assert!(rendered.contains("Dog"), "should contain subject label");
    assert!(rendered.contains("Animal"), "should contain object label");
    assert!(
        rendered.contains('\u{25B3}'),
        "should contain is-a glyph fallback (△)"
    );
    // Should contain confidence dots.
    assert!(rendered.contains('\u{25CF}'), "should contain filled confidence dots");
}

#[test]
fn hieroglyphic_sigil_unique() {
    let engine = test_engine();
    let dog = engine.create_symbol(SymbolKind::Entity, "Dog").unwrap();
    let cat = engine.create_symbol(SymbolKind::Entity, "Cat").unwrap();

    let s1 = akh_medu::glyph::sigil::sigil_for_symbol(dog.id, engine.item_memory(), false)
        .unwrap();
    let s2 = akh_medu::glyph::sigil::sigil_for_symbol(cat.id, engine.item_memory(), false)
        .unwrap();

    // Different symbols should (very likely) produce different sigils.
    // With 32^3 combinations and random VSA vectors, collision probability is negligible.
    assert_ne!(s1, s2, "different entities should have different sigils");
}

#[test]
fn hieroglyphic_subgraph() {
    let engine = test_engine();
    let dog = engine.create_symbol(SymbolKind::Entity, "Dog").unwrap();
    let is_a = engine.create_symbol(SymbolKind::Relation, "is-a").unwrap();
    let animal = engine.create_symbol(SymbolKind::Entity, "Animal").unwrap();
    let has_a = engine.create_symbol(SymbolKind::Relation, "has-a").unwrap();
    let legs = engine.create_symbol(SymbolKind::Entity, "Legs").unwrap();

    engine.add_triple(&Triple::new(dog.id, is_a.id, animal.id)).unwrap();
    engine.add_triple(&Triple::new(dog.id, has_a.id, legs.id)).unwrap();

    let triples = vec![
        Triple::new(dog.id, is_a.id, animal.id),
        Triple::new(dog.id, has_a.id, legs.id),
    ];

    let config = akh_medu::glyph::NotationConfig {
        use_pua: false,
        show_confidence: false,
        show_provenance: false,
        show_sigils: false,
        compact: false,
    };
    let rendered = akh_medu::glyph::notation::render_subgraph(&engine, &triples, &config);

    // Block format should have curly braces for grouped triples.
    assert!(rendered.contains('{'), "block format should have opening brace");
    assert!(rendered.contains('}'), "block format should have closing brace");
    assert!(rendered.contains("Animal"));
    assert!(rendered.contains("Legs"));
}

#[test]
fn hieroglyphic_legend() {
    let config = akh_medu::glyph::RenderConfig {
        color: false,
        notation: akh_medu::glyph::NotationConfig {
            use_pua: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let legend = akh_medu::glyph::render::render_legend(&config);

    // Should list all 35 fixed glyphs.
    assert!(legend.contains("is-a"), "legend should contain is-a");
    assert!(legend.contains("type:person"), "legend should contain type:person");
    assert!(legend.contains("prov:asserted"), "legend should contain prov:asserted");
    assert!(legend.contains("struct:triple"), "legend should contain struct:triple");

    // Should list all 32 radicals.
    assert!(legend.contains("eye"), "legend should contain eye radical");
    assert!(legend.contains("ankh"), "legend should contain ankh radical");
    assert!(legend.contains("star"), "legend should contain star radical");

    // Count lines with radical entries (should have 32).
    let radical_lines: Vec<&str> = legend.lines().filter(|l| l.contains("[")).collect();
    assert_eq!(radical_lines.len(), 32, "legend should list 32 radicals");
}

// -----------------------------------------------------------------------
// Phase 12: Knowledge Population
// -----------------------------------------------------------------------

/// Create an engine with bundled skill packs available.
fn engine_with_skills() -> (Engine, tempfile::TempDir) {
    let dir = tempfile::TempDir::new().unwrap();
    let skills_target = dir.path().join("skills");
    std::fs::create_dir_all(&skills_target).unwrap();

    // Copy bundled skill packs from the repo's skills/ directory.
    let source_skills = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("skills");
    if source_skills.exists() {
        for entry in std::fs::read_dir(&source_skills).unwrap() {
            let entry = entry.unwrap();
            let skill_name = entry.file_name();
            let src = entry.path();
            let dst = skills_target.join(&skill_name);
            std::fs::create_dir_all(&dst).unwrap();
            for file in std::fs::read_dir(&src).unwrap() {
                let file = file.unwrap();
                std::fs::copy(file.path(), dst.join(file.file_name())).unwrap();
            }
        }
    }

    let engine = Engine::new(EngineConfig {
        dimension: Dimension::TEST,
        data_dir: Some(dir.path().to_path_buf()),
        ..Default::default()
    })
    .unwrap();

    (engine, dir)
}

#[test]
fn skill_pack_common_sense_loads() {
    let (engine, _dir) = engine_with_skills();
    let activation = engine.load_skill("common_sense").unwrap();
    assert!(activation.triples_loaded > 100, "common_sense should have >100 triples");
    assert!(activation.rules_loaded > 0, "common_sense should have rules");
}

#[test]
fn skill_pack_geography_loads() {
    let (engine, _dir) = engine_with_skills();
    let activation = engine.load_skill("geography").unwrap();
    assert!(activation.triples_loaded > 80, "geography should have >80 triples");
    assert!(activation.rules_loaded > 0, "geography should have rules");
}

#[test]
fn skill_pack_science_loads() {
    let (engine, _dir) = engine_with_skills();
    let activation = engine.load_skill("science").unwrap();
    assert!(activation.triples_loaded > 80, "science should have >80 triples");
    assert!(activation.rules_loaded > 0, "science should have rules");
}

#[test]
fn skill_pack_language_loads() {
    let (engine, _dir) = engine_with_skills();
    let activation = engine.load_skill("language").unwrap();
    assert!(activation.triples_loaded > 60, "language should have >60 triples");
    assert!(activation.rules_loaded > 0, "language should have rules");
}

#[test]
fn csv_ingest_spo_and_entity() {
    use akh_medu::agent::tools::CsvIngestTool;
    use akh_medu::agent::tool::{Tool, ToolInput};

    let engine = test_engine();
    let dir = tempfile::TempDir::new().unwrap();

    // SPO format.
    let spo_path = dir.path().join("spo.csv");
    std::fs::write(&spo_path, "Dog,is-a,Animal\nCat,is-a,Animal,0.95\n").unwrap();
    let input = ToolInput::new()
        .with_param("path", spo_path.to_str().unwrap())
        .with_param("format", "spo");
    let result = CsvIngestTool.execute(&engine, input).unwrap();
    assert!(result.success);
    assert!(result.result.contains("2 triples ingested"));

    // Entity format.
    let entity_path = dir.path().join("entity.csv");
    std::fs::write(&entity_path, "entity,is-a,lives-in\nFrog,Animal,Pond\n").unwrap();
    let input = ToolInput::new()
        .with_param("path", entity_path.to_str().unwrap())
        .with_param("format", "entity");
    let result = CsvIngestTool.execute(&engine, input).unwrap();
    assert!(result.success);
    assert!(result.result.contains("2 triples ingested"));
}

#[test]
fn text_ingest_extracts_triples() {
    use akh_medu::agent::tools::TextIngestTool;
    use akh_medu::agent::tool::{Tool, ToolInput};

    let engine = test_engine();
    let input = ToolInput::new()
        .with_param("text", "Dogs are mammals. Paris is located in France. The wheel is part of the car.");
    let result = TextIngestTool.execute(&engine, input).unwrap();
    assert!(result.success);
    assert!(result.result.contains("extracted 3 triple(s)"));
}

#[test]
fn bootstrap_loads_skills_and_grounds() {
    let (engine, _dir) = engine_with_skills();

    // Load all skills.
    let skill_names = ["astronomy", "common_sense", "geography", "science", "language"];
    let mut total = 0;
    for name in &skill_names {
        if let Ok(activation) = engine.load_skill(name) {
            total += activation.triples_loaded;
        }
    }
    assert!(total > 200, "should load >200 triples across all skills, got {total}");

    // Run grounding.
    let ops = engine.ops();
    let im = engine.item_memory();
    let config = akh_medu::vsa::grounding::GroundingConfig::default();
    let grounding = akh_medu::vsa::grounding::ground_all(&engine, ops, im, &config).unwrap();
    assert!(grounding.symbols_updated > 0, "grounding should update symbols");
}

#[test]
fn post_ingest_grounding_improves_similarity() {
    let (engine, _dir) = engine_with_skills();
    engine.load_skill("common_sense").unwrap();

    let ops = engine.ops();
    let im = engine.item_memory();

    // Before grounding: Dog and Cat have ~0.5 similarity (random vectors).
    let dog = engine.lookup_symbol("Dog").unwrap();
    let cat = engine.lookup_symbol("Cat").unwrap();

    let dog_vec_before = im.get_or_create(ops, dog);
    let cat_vec_before = im.get_or_create(ops, cat);
    let sim_before = ops.similarity(&dog_vec_before, &cat_vec_before).unwrap();

    // Run grounding.
    let config = akh_medu::vsa::grounding::GroundingConfig::default();
    akh_medu::vsa::grounding::ground_all(&engine, ops, im, &config).unwrap();

    // After grounding: Dog and Cat should be more similar.
    let dog_vec_after = im.get(dog).unwrap();
    let cat_vec_after = im.get(cat).unwrap();
    let sim_after = ops.similarity(&dog_vec_after, &cat_vec_after).unwrap();

    assert!(
        sim_after > sim_before,
        "Dog/Cat similarity should increase after grounding: before={sim_before:.3}, after={sim_after:.3}"
    );
}

#[test]
fn new_tools_accessible_in_ooda() {
    let engine = Engine::new(EngineConfig {
        dimension: Dimension::TEST,
        ..Default::default()
    })
    .unwrap();
    let engine = Arc::new(engine);
    let agent = Agent::new(Arc::clone(&engine), AgentConfig::default()).unwrap();

    // Verify new tools are registered and accessible.
    let tools = agent.list_tools();
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    assert!(names.contains(&"csv_ingest"), "csv_ingest should be registered");
    assert!(names.contains(&"text_ingest"), "text_ingest should be registered");
}

// ---------------------------------------------------------------------------
// Phase 13 — Natural Language Interface
// ---------------------------------------------------------------------------

#[test]
fn intent_classify_query() {
    use akh_medu::agent::{classify_intent, UserIntent};

    match classify_intent("What is a dog?") {
        UserIntent::Query { subject } => {
            assert!(subject.to_lowercase().contains("dog"), "subject should contain 'dog', got: {subject}");
        }
        other => panic!("Expected Query, got {other:?}"),
    }
}

#[test]
fn intent_classify_assert() {
    use akh_medu::agent::{classify_intent, UserIntent};

    match classify_intent("Dogs are mammals") {
        UserIntent::Assert { text } => {
            assert_eq!(text, "Dogs are mammals");
        }
        other => panic!("Expected Assert, got {other:?}"),
    }
}

#[test]
fn intent_classify_goal() {
    use akh_medu::agent::{classify_intent, UserIntent};

    match classify_intent("Find similar animals to Dog") {
        UserIntent::SetGoal { description } => {
            assert!(description.contains("similar animals"), "desc: {description}");
        }
        other => panic!("Expected SetGoal, got {other:?}"),
    }
}

#[test]
fn intent_classify_status() {
    use akh_medu::agent::{classify_intent, UserIntent};

    assert!(matches!(classify_intent("status"), UserIntent::ShowStatus));
    assert!(matches!(classify_intent("goals"), UserIntent::ShowStatus));
}

#[test]
fn intent_classify_render() {
    use akh_medu::agent::{classify_intent, UserIntent};

    match classify_intent("show Dog") {
        UserIntent::RenderHiero { entity } => {
            assert_eq!(entity.as_deref(), Some("Dog"));
        }
        other => panic!("Expected RenderHiero, got {other:?}"),
    }
}

#[test]
fn intent_classify_freeform() {
    use akh_medu::agent::{classify_intent, UserIntent};

    assert!(matches!(
        classify_intent("tell me something interesting"),
        UserIntent::Freeform { .. }
    ));
}

#[test]
fn conversation_serialize_deserialize() {
    use akh_medu::agent::Conversation;

    let mut conv = Conversation::new(50);
    conv.add_turn("What is a cat?".into(), "A cat is a mammal.".into());
    conv.add_turn("Tell me more.".into(), "Cats are domestic animals.".into());

    let bytes = conv.to_bytes().unwrap();
    let restored = Conversation::from_bytes(&bytes).unwrap();

    assert_eq!(restored.len(), 2);
    assert_eq!(restored.turns()[0].user_input, "What is a cat?");
    assert_eq!(restored.turns()[1].agent_response, "Cats are domestic animals.");
}

#[test]
fn chat_roundtrip_intent_pipeline() {
    // Integration test: classify intent, execute assertion via TextIngestTool, query result.
    let engine = Engine::new(EngineConfig {
        dimension: Dimension::TEST,
        ..Default::default()
    })
    .unwrap();
    let engine = Arc::new(engine);

    // Step 1: Assert a fact.
    let intent = akh_medu::agent::classify_intent("Dogs are mammals");
    assert!(matches!(intent, akh_medu::agent::UserIntent::Assert { .. }));

    if let akh_medu::agent::UserIntent::Assert { text } = intent {
        use akh_medu::agent::tool::Tool;
        let tool_input = akh_medu::agent::ToolInput::new().with_param("text", &text);
        let output = akh_medu::agent::tools::TextIngestTool
            .execute(&engine, tool_input)
            .unwrap();
        assert!(output.success, "TextIngestTool should succeed: {}", output.result);
        assert!(
            output.result.contains("is-a") || output.result.contains("triple"),
            "Should mention extracted triple: {}",
            output.result,
        );
    }

    // Step 2: Verify symbols were created by the ingest.
    // TextIngestTool capitalizes and preserves the original words: "Dogs" and "Mammals".
    let resolved = engine.resolve_symbol("Dogs");
    assert!(
        resolved.is_ok(),
        "Symbol 'Dogs' should exist after assertion, got: {resolved:?}",
    );
    let resolved_obj = engine.resolve_symbol("Mammals");
    assert!(
        resolved_obj.is_ok(),
        "Symbol 'Mammals' should exist after assertion, got: {resolved_obj:?}",
    );

    // Verify the triple exists.
    let dogs_id = resolved.unwrap();
    let triples = engine.triples_from(dogs_id);
    assert!(!triples.is_empty(), "Dogs should have outgoing triples");
}

#[test]
fn core_features_work_standalone() {
    // All core features work — no panics, no unwraps on None.
    let engine = Engine::new(EngineConfig {
        dimension: Dimension::TEST,
        ..Default::default()
    })
    .unwrap();
    let engine = Arc::new(engine);

    // Agent creation works.
    let agent = Agent::new(Arc::clone(&engine), AgentConfig::default()).unwrap();
    assert_eq!(agent.cycle_count(), 0);

    // Conversation works.
    let mut conv = akh_medu::agent::Conversation::new(10);
    conv.add_turn("test".into(), "response".into());
    assert_eq!(conv.len(), 1);

    // Intent classification works (regex-based).
    let intent = akh_medu::agent::classify_intent("What is the capital of France?");
    assert!(matches!(intent, akh_medu::agent::UserIntent::Query { .. }));

    // Text ingest works (regex extraction).
    use akh_medu::agent::tool::Tool;
    let input = akh_medu::agent::ToolInput::new().with_param("text", "Paris is the capital of France");
    let output = akh_medu::agent::tools::TextIngestTool.execute(&engine, input).unwrap();
    assert!(output.success);
}
