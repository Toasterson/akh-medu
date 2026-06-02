//! Workspace hermeticity tests.
//!
//! Verifies that two independent workspaces enforce strict data isolation:
//! symbols, triples, and labels ingested into workspace A must never appear
//! when querying workspace B, and deleting one workspace must not corrupt
//! the other.

use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::paths::AkhPaths;
use akh_medu::symbol::SymbolKind;
use akh_medu::vsa::Dimension;
use akh_medu::workspace::{WorkspaceConfig, WorkspaceManager};

fn workspace_engine(dir: &std::path::Path) -> Engine {
    Engine::new(EngineConfig {
        dimension: Dimension::TEST,
        data_dir: Some(dir.to_path_buf()),
        ..Default::default()
    })
    .unwrap()
}

/// Symbols created in workspace A must not appear in workspace B's registry.
#[test]
fn symbol_registry_is_hermetic() {
    let dir_a = tempfile::TempDir::new().unwrap();
    let dir_b = tempfile::TempDir::new().unwrap();

    let engine_a = workspace_engine(dir_a.path());
    let engine_b = workspace_engine(dir_b.path());

    engine_a
        .create_symbol(SymbolKind::Entity, "Sirius")
        .unwrap();
    engine_a.create_symbol(SymbolKind::Entity, "Rigel").unwrap();
    engine_a
        .create_symbol(SymbolKind::Relation, "brighter-than")
        .unwrap();

    let b_symbols = engine_b.all_symbols();
    assert!(
        b_symbols.is_empty(),
        "workspace B must have zero symbols after ingesting into A; got {}",
        b_symbols.len(),
    );
    assert!(
        engine_b.lookup_symbol("Sirius").is_err(),
        "workspace B must not resolve 'Sirius' from workspace A",
    );
    assert!(
        engine_b.lookup_symbol("Rigel").is_err(),
        "workspace B must not resolve 'Rigel' from workspace A",
    );
}

/// Triples added to workspace A must not appear in workspace B's triple store.
#[test]
fn triple_store_is_hermetic() {
    let dir_a = tempfile::TempDir::new().unwrap();
    let dir_b = tempfile::TempDir::new().unwrap();

    let engine_a = workspace_engine(dir_a.path());
    let engine_b = workspace_engine(dir_b.path());

    let triples = vec![
        ("Earth".into(), "orbits".into(), "Sun".into(), 1.0_f32),
        ("Moon".into(), "orbits".into(), "Earth".into(), 1.0),
        ("Sun".into(), "is-a".into(), "Star".into(), 1.0),
    ];
    engine_a.ingest_label_triples(&triples).unwrap();

    assert_eq!(
        engine_a.all_triples().len(),
        3,
        "workspace A must contain the 3 ingested triples",
    );

    let b_triples = engine_b.all_triples();
    assert!(
        b_triples.is_empty(),
        "workspace B must have zero triples after ingesting into A; got {}",
        b_triples.len(),
    );
}

/// Persistence of workspace A must not leak data into workspace B on reopen.
#[test]
fn hermetic_across_persist_and_reopen() {
    let dir_a = tempfile::TempDir::new().unwrap();
    let dir_b = tempfile::TempDir::new().unwrap();

    // Session 1: populate workspace A and flush to disk.
    {
        let engine_a = workspace_engine(dir_a.path());
        engine_a
            .create_symbol(SymbolKind::Entity, "Andromeda")
            .unwrap();
        engine_a
            .create_symbol(SymbolKind::Entity, "Milky Way")
            .unwrap();
        engine_a.persist().unwrap();
    }

    // Session 2: reopen workspace A; open a fresh workspace B.
    {
        let engine_a = workspace_engine(dir_a.path());
        let engine_b = workspace_engine(dir_b.path());

        assert_eq!(
            engine_a.all_symbols().len(),
            2,
            "workspace A should recover both symbols after reopen",
        );

        let b_symbols = engine_b.all_symbols();
        assert!(
            b_symbols.is_empty(),
            "workspace B must remain empty even after workspace A persists; got {}",
            b_symbols.len(),
        );
        assert!(
            engine_b.lookup_symbol("Andromeda").is_err(),
            "workspace B must not find 'Andromeda' persisted in workspace A",
        );
        assert!(
            engine_b.lookup_symbol("Milky Way").is_err(),
            "workspace B must not find 'Milky Way' persisted in workspace A",
        );
    }
}

/// Deleting workspace A must leave workspace B's data completely intact.
#[test]
fn delete_workspace_a_does_not_corrupt_workspace_b() {
    let tmp = tempfile::TempDir::new().unwrap();
    let akh_paths = AkhPaths {
        config_dir: tmp.path().join("config"),
        data_dir: tmp.path().join("data"),
        state_dir: tmp.path().join("state"),
        runtime_dir: tmp.path().join("run"),
        cache_dir: tmp.path().join("cache"),
    };
    akh_paths.ensure_dirs().unwrap();

    let mgr = WorkspaceManager::new(akh_paths);
    let ws_a = mgr.create(WorkspaceConfig::with_name("alpha")).unwrap();
    let ws_b = mgr.create(WorkspaceConfig::with_name("beta")).unwrap();

    // Populate both workspaces with distinct data.
    {
        let engine_a = workspace_engine(&ws_a.kg_dir);
        engine_a
            .create_symbol(SymbolKind::Entity, "PrivateA")
            .unwrap();
        engine_a.persist().unwrap();

        let engine_b = workspace_engine(&ws_b.kg_dir);
        engine_b
            .create_symbol(SymbolKind::Entity, "PrivateB")
            .unwrap();
        engine_b.persist().unwrap();
    }

    // Remove workspace A.
    mgr.delete("alpha").unwrap();
    assert!(!ws_a.exists(), "workspace A directory should be gone");
    assert!(ws_b.exists(), "workspace B directory should still exist");

    // Reopen workspace B and confirm its data survived.
    {
        let engine_b = workspace_engine(&ws_b.kg_dir);
        let b_symbols = engine_b.all_symbols();
        assert_eq!(
            b_symbols.len(),
            1,
            "workspace B must still have its 1 symbol after deleting workspace A; got {}",
            b_symbols.len(),
        );
        assert!(
            engine_b.lookup_symbol("PrivateB").is_ok(),
            "workspace B's 'PrivateB' must survive deletion of workspace A",
        );
        assert!(
            engine_b.lookup_symbol("PrivateA").is_err(),
            "workspace B must never have contained workspace A's 'PrivateA'",
        );
    }
}
