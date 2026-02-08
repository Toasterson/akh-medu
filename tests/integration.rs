//! End-to-end integration tests for the akh-medu engine.
//!
//! These tests exercise the full pipeline from symbol creation through
//! inference and export, validating that the registry, knowledge graph,
//! and introspection APIs all work together.

use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::graph::Triple;
use akh_medu::infer::InferenceQuery;
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
