//! Live-Postgres test for the ADR 081 Slice 2 schema-pack registry (the
//! Postgres half): admit the seed pack → fetch it back → install into a
//! fresh engine → its types are present → reinstalling is idempotent.
//!
//! Compiled only with the `postgres` feature; ignored unless AKH_TEST_PG_URL is
//! set. Run against a throwaway database:
//!   AKH_TEST_PG_URL=postgres://user@127.0.0.1:5439/akhmedu?sslmode=disable \
//!     cargo test --features postgres --test schema_registry_live -- --ignored
#![cfg(feature = "postgres")]

use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::schema::registry_pg::PgSchemaRegistry;
use akh_medu::schema::{RegistryRow, SchemaRegistry as _, compile, install_pack};
use akh_medu::vsa::Dimension;
use akhomed_store::{connect, run_migrations};

/// Admits the bundled seed pack (`data/schema-packs/schema-org-core.json`)
/// under a per-run unique `version` (so the test is safely rerunnable against
/// a persistent database — registry admission is insert-only), then proves
/// the full loop: admit → duplicate rejected → fetch → install into a fresh
/// engine → types queryable → reinstall is idempotent → pack is listed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires AKH_TEST_PG_URL"]
async fn admit_install_and_reinstall_the_seed_pack_over_postgres() {
    let Ok(url) = std::env::var("AKH_TEST_PG_URL") else {
        eprintln!("skipping: set AKH_TEST_PG_URL");
        return;
    };
    let db = connect(&akhomed_store::PoolOptions::new(url))
        .await
        .unwrap();
    run_migrations(&db).await.unwrap();

    // The seed pack is admitted under its REAL version, exactly once per
    // database lifetime: global type names are immutable across the whole
    // registry (ADR 081 decision 5 — `check_admission` rejects ANY later
    // pack, including a newer version of this one, re-declaring `Thing`),
    // so a per-run-unique version would be correctly rejected on the second
    // run. Admit-or-reuse keeps the test rerunnable against a persistent
    // database while still exercising the admission path on a fresh one.
    let text = include_str!("../data/schema-packs/schema-org-core.json");
    let pack = compile(text).unwrap();
    assert_eq!(pack.name, "schema-org-core");

    // `PgSchemaRegistry::admit` is a sync trait method (it bridges its own
    // async DB hop internally — see `schema::registry_pg::block_on`), so it
    // is called directly, no explicit `block_in_place` needed at the call
    // site (mirrors `PgTurnBodyReader::turn_body` in `src/bin/akhomed.rs`).
    let registry = PgSchemaRegistry::new(&db);
    if registry.fetch(&pack.name, &pack.version).unwrap().is_none() {
        registry
            .admit(RegistryRow::from_pack(&pack, "live-test-operator").unwrap())
            .unwrap();
    }

    // Re-admitting the exact same (name, version) is rejected — the registry
    // gate ADR 081 decision 9 requires (rows are written exactly once).
    let dup = registry.admit(RegistryRow::from_pack(&pack, "someone-else").unwrap());
    assert!(
        dup.is_err(),
        "duplicate (name, version) admission must be rejected"
    );

    // Fetch it back and decode.
    let fetched = registry
        .fetch(&pack.name, &pack.version)
        .unwrap()
        .expect("admitted pack is fetchable");
    let decoded = fetched.decode().unwrap();
    assert_eq!(decoded.triples, pack.triples);

    // Install into a fresh engine: the core types + relations land in the KG.
    let engine = Engine::new(EngineConfig {
        dimension: Dimension::TEST,
        data_dir: None,
        ..Default::default()
    })
    .unwrap();
    let report = install_pack(&engine, &decoded).unwrap();
    assert!(!report.already_installed);
    assert!(report.triples_applied > 30);

    for ty in ["Thing", "Person", "Organization", "Akh", "Goal"] {
        assert!(engine.resolve_symbol(ty).is_ok(), "missing type {ty}");
    }
    assert!(
        engine.resolve_symbol("worksFor").is_ok(),
        "missing relation worksFor"
    );

    // Reinstalling the exact same pack@version is an idempotent no-op
    // (`install::install_pack`'s already-installed guard).
    let report2 = install_pack(&engine, &decoded).unwrap();
    assert!(report2.already_installed);
    assert_eq!(report2.triples_applied, 0);

    // The pack is listed.
    let listed = registry.list().unwrap();
    assert!(
        listed
            .iter()
            .any(|(name, version)| name == &pack.name && version == &pack.version)
    );
}
