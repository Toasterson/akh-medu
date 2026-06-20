//! Live-Postgres test for the Phase 38c snapshot writer.
//!
//! Compiled only with the `postgres` feature; ignored unless AKH_TEST_PG_URL is
//! set. Run against a throwaway database:
//!   AKH_TEST_PG_URL=postgres://user@127.0.0.1:5432/akhmedu?sslmode=disable \
//!     cargo test --features server --test snapshot_live -- --ignored
#![cfg(feature = "postgres")]

use std::sync::Arc;

use akh_medu::agent::{Agent, AgentConfig};
use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::graph::Triple;
use akh_medu::snapshot::write_snapshot;
use akh_medu::symbol::SymbolKind;
use akhomed_store::sea_orm::{self, ConnectionTrait, DbBackend, Statement};
use akhomed_store::uuid::Uuid;
use akhomed_store::{MutationRow, append_mutations, connect, ensure_akh, run_migrations};

async fn count(db: &sea_orm::DatabaseConnection, sql: &str) -> i64 {
    let row = db
        .query_one(Statement::from_string(DbBackend::Postgres, sql.to_owned()))
        .await
        .unwrap()
        .unwrap();
    row.try_get::<i64>("", "c").unwrap()
}

#[tokio::test]
#[ignore = "requires AKH_TEST_PG_URL"]
async fn snapshot_populates_relational_tables() {
    let Ok(url) = std::env::var("AKH_TEST_PG_URL") else {
        eprintln!("skipping: set AKH_TEST_PG_URL");
        return;
    };
    let db = connect(&akhomed_store::PoolOptions::new(url)).await.unwrap();
    run_migrations(&db).await.unwrap();

    let akh_id = Uuid::new_v4();
    let workspace_id = Uuid::new_v4();

    // Build an engine with a few symbols + triples.
    let tmp = tempfile::tempdir().unwrap();
    let engine = Arc::new(
        Engine::new(EngineConfig {
            data_dir: Some(tmp.path().join("kg")),
            ..Default::default()
        })
        .unwrap(),
    );
    let sun = engine.create_symbol(SymbolKind::Entity, "Sun").unwrap();
    let is_a = engine.create_symbol(SymbolKind::Relation, "is-a").unwrap();
    let star = engine.create_symbol(SymbolKind::Entity, "Star").unwrap();
    engine.add_triple(&Triple::new(sun.id, is_a.id, star.id)).unwrap();

    let agent = Agent::new(Arc::clone(&engine), AgentConfig::default()).expect("agent");

    // A WAL row that must survive the snapshot (38c does not truncate).
    ensure_akh(&db, akh_id, "snap-test", None).await.unwrap();
    append_mutations(
        &db,
        akh_id,
        workspace_id,
        &[MutationRow {
            kind: "assert".into(),
            payload: serde_json::json!({ "subject": sun.id.get() }),
        }],
    )
    .await
    .unwrap();

    let expected_triples = engine.all_triples().len() as i64;
    let expected_symbols = engine.dump_symbols().len() as i64;

    write_snapshot(&agent, akh_id, workspace_id, &db).await.expect("snapshot");

    assert_eq!(
        count(&db, &format!("SELECT count(*)::bigint AS c FROM akhomed_facts WHERE workspace_id = '{workspace_id}'")).await,
        expected_triples,
        "facts rows must equal live KG triple count",
    );
    assert_eq!(
        count(&db, &format!("SELECT count(*)::bigint AS c FROM akhomed_symbols WHERE akh_id = '{akh_id}'")).await,
        expected_symbols,
        "symbols rows must equal registry size",
    );
    assert_eq!(
        count(&db, &format!("SELECT count(*)::bigint AS c FROM akhomed_drives WHERE akh_id = '{akh_id}'")).await,
        4,
        "four drive rows",
    );
    assert_eq!(
        count(&db, &format!("SELECT count(*)::bigint AS c FROM akhomed_akh_state WHERE akh_id = '{akh_id}'")).await,
        1,
        "one akh_state row",
    );
    // WAL not truncated by the snapshot.
    assert_eq!(
        count(&db, &format!("SELECT count(*)::bigint AS c FROM akhomed_mutation_log WHERE akh_id = '{akh_id}'")).await,
        1,
        "snapshot must not truncate the WAL",
    );
}
