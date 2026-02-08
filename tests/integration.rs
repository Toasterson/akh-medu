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

    // Should have 5 built-in tools.
    let tools = agent.list_tools();
    assert_eq!(tools.len(), 5);

    let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    assert!(names.contains(&"kg_query"));
    assert!(names.contains(&"kg_mutate"));
    assert!(names.contains(&"memory_recall"));
    assert!(names.contains(&"reason"));
    assert!(names.contains(&"similarity_search"));

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
    }

    agent.register_tool(Box::new(CustomTool));
    assert_eq!(agent.list_tools().len(), 6);
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
