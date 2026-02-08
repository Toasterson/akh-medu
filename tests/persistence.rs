//! Persistence and recovery tests for the akh-medu engine.
//!
//! These tests verify that symbols, registry state, and allocator state
//! survive engine restart (persist + reopen cycle).

use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::graph::Triple;
use akh_medu::infer::InferenceQuery;
use akh_medu::symbol::SymbolKind;
use akh_medu::vsa::Dimension;

fn persistent_engine(dir: &std::path::Path) -> Engine {
    Engine::new(EngineConfig {
        dimension: Dimension::TEST,
        data_dir: Some(dir.to_path_buf()),
        ..Default::default()
    })
    .unwrap()
}

#[test]
fn symbols_survive_restart() {
    let dir = tempfile::TempDir::new().unwrap();

    // First session: create symbols and persist.
    {
        let engine = persistent_engine(dir.path());
        engine.create_symbol(SymbolKind::Entity, "Sun").unwrap();
        engine.create_symbol(SymbolKind::Relation, "is-a").unwrap();
        engine.create_symbol(SymbolKind::Entity, "Star").unwrap();
        engine.persist().unwrap();
    }

    // Second session: reopen and verify.
    {
        let engine = persistent_engine(dir.path());
        assert_eq!(engine.all_symbols().len(), 3);

        let sun_id = engine.lookup_symbol("Sun").unwrap();
        let meta = engine.get_symbol_meta(sun_id).unwrap();
        assert_eq!(meta.label, "Sun");
        assert_eq!(meta.kind, SymbolKind::Entity);

        // Case-insensitive lookup should still work.
        assert_eq!(engine.lookup_symbol("sun").unwrap(), sun_id);
        assert_eq!(engine.lookup_symbol("SUN").unwrap(), sun_id);

        // resolve_symbol by name should work.
        assert_eq!(engine.resolve_symbol("Star").unwrap(), engine.lookup_symbol("star").unwrap());
    }
}

#[test]
fn allocator_resumes_after_restart() {
    let dir = tempfile::TempDir::new().unwrap();

    let max_id_before;
    // First session: create symbols, persist.
    {
        let engine = persistent_engine(dir.path());
        engine.create_symbol(SymbolKind::Entity, "Alpha").unwrap();
        engine.create_symbol(SymbolKind::Entity, "Beta").unwrap();
        let gamma = engine.create_symbol(SymbolKind::Entity, "Gamma").unwrap();
        max_id_before = gamma.id.get();
        engine.persist().unwrap();
    }

    // Second session: new symbols should have higher IDs.
    {
        let engine = persistent_engine(dir.path());
        let delta = engine.create_symbol(SymbolKind::Entity, "Delta").unwrap();
        assert!(
            delta.id.get() > max_id_before,
            "new ID {} should be > pre-restart max {}",
            delta.id.get(),
            max_id_before
        );
    }
}

#[test]
fn provenance_survives_restart() {
    let dir = tempfile::TempDir::new().unwrap();

    let derived_id;
    // First session: create symbols, infer, persist.
    {
        let engine = persistent_engine(dir.path());
        let sun = engine.create_symbol(SymbolKind::Entity, "Sun").unwrap();
        let is_a = engine.create_symbol(SymbolKind::Relation, "is-a").unwrap();
        let star = engine.create_symbol(SymbolKind::Entity, "Star").unwrap();
        engine
            .add_triple(&Triple::new(sun.id, is_a.id, star.id))
            .unwrap();

        let query = InferenceQuery {
            seeds: vec![sun.id],
            top_k: 10,
            max_depth: 1,
            ..Default::default()
        };
        let result = engine.infer(&query).unwrap();

        // Pick the first activation with provenance.
        if let Some(record) = result.provenance.first() {
            derived_id = Some(record.derived_id);
        } else {
            derived_id = None;
        }

        engine.persist().unwrap();
    }

    // Second session: verify provenance survived.
    if let Some(sym_id) = derived_id {
        let engine = persistent_engine(dir.path());
        let records = engine.provenance_of(sym_id).unwrap();
        assert!(
            !records.is_empty(),
            "provenance records for {} should survive restart",
            sym_id
        );
    }
}
