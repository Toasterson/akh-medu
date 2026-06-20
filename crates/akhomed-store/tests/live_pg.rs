//! Live-Postgres integration tests for the 38b write API.
//!
//! Ignored by default; run against a throwaway database:
//!   AKH_TEST_PG_URL=postgres://user@127.0.0.1:5439/akhmedu?sslmode=disable \
//!     cargo test -p akhomed-store --test live_pg -- --ignored

use akhomed_store::uuid::Uuid;
use akhomed_store::{
    MutationRow, PoolOptions, append_mutations, connect, ensure_akh, run_migrations,
};
use sea_orm::{ConnectionTrait, DbBackend, Statement};

async fn mutation_count(db: &sea_orm::DatabaseConnection, akh: Uuid) -> i64 {
    let row = db
        .query_one(Statement::from_string(
            DbBackend::Postgres,
            format!("SELECT count(*)::bigint AS c FROM akhomed_mutation_log WHERE akh_id = '{akh}'"),
        ))
        .await
        .unwrap()
        .unwrap();
    row.try_get::<i64>("", "c").unwrap()
}

#[tokio::test]
#[ignore = "requires AKH_TEST_PG_URL"]
async fn ensure_akh_is_idempotent_and_append_writes_rows() {
    let Ok(url) = std::env::var("AKH_TEST_PG_URL") else {
        eprintln!("skipping: set AKH_TEST_PG_URL");
        return;
    };
    let db = connect(&PoolOptions::new(url)).await.expect("connect");
    run_migrations(&db).await.expect("migrate");

    let akh = Uuid::new_v4();
    let workspace = Uuid::new_v4();

    // Idempotent ensure: two calls, no error, one row.
    ensure_akh(&db, akh, "live-test-akh", Some("test-host")).await.unwrap();
    ensure_akh(&db, akh, "live-test-akh", Some("test-host")).await.unwrap();

    assert_eq!(mutation_count(&db, akh).await, 0);

    let rows = vec![
        MutationRow {
            kind: "assert".into(),
            payload: serde_json::json!({ "subject": 1, "predicate": 2, "object": 3, "confidence": 1.0 }),
        },
        MutationRow {
            kind: "retract".into(),
            payload: serde_json::json!({ "subject": 1, "predicate": 2, "object": 3 }),
        },
    ];
    let n = append_mutations(&db, akh, workspace, &rows).await.unwrap();
    assert_eq!(n, 2);
    assert_eq!(mutation_count(&db, akh).await, 2);

    // Empty batch is a no-op.
    assert_eq!(append_mutations(&db, akh, workspace, &[]).await.unwrap(), 0);
    assert_eq!(mutation_count(&db, akh).await, 2);
}
