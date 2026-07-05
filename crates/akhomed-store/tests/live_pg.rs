//! Live-Postgres integration tests for the 38b write API.
//!
//! Ignored by default; run against a throwaway database:
//!   AKH_TEST_PG_URL=postgres://user@127.0.0.1:5439/akhmedu?sslmode=disable \
//!     cargo test -p akhomed-store --test live_pg -- --ignored

use akhomed_store::uuid::Uuid;
use akhomed_store::{
    MutationRow, PoolOptions, append_mutations, connect, ensure_akh, join_workspace,
    leave_workspace, list_members, release_akh_lock, run_migrations, run_migrations_locked,
    touch_last_seen, try_acquire_akh_lock,
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

#[tokio::test]
#[ignore = "requires AKH_TEST_PG_URL"]
async fn membership_join_list_leave_roundtrip() {
    let Ok(url) = std::env::var("AKH_TEST_PG_URL") else {
        eprintln!("skipping: set AKH_TEST_PG_URL");
        return;
    };
    let db = connect(&PoolOptions::new(url)).await.expect("connect");
    run_migrations(&db).await.expect("migrate");

    let workspace = Uuid::new_v4();
    let akh = Uuid::new_v4();
    ensure_akh(&db, akh, "member-test-akh", None).await.unwrap();

    // Initially no members.
    assert!(list_members(&db, workspace).await.unwrap().is_empty());

    // Join is idempotent: two joins -> one row.
    let now = akhomed_store::chrono::Utc::now();
    join_workspace(&db, workspace, akh, "member", now).await.unwrap();
    join_workspace(&db, workspace, akh, "member", now).await.unwrap();
    let members = list_members(&db, workspace).await.unwrap();
    assert_eq!(members.len(), 1);
    assert_eq!(members[0].akh_id, akh);
    assert_eq!(members[0].role, "member");

    // Leave removes the row (and reports it).
    assert!(leave_workspace(&db, workspace, akh).await.unwrap());
    assert!(list_members(&db, workspace).await.unwrap().is_empty());
    // Leaving again is a no-op (reports false).
    assert!(!leave_workspace(&db, workspace, akh).await.unwrap());
}

#[tokio::test]
#[ignore = "requires AKH_TEST_PG_URL"]
async fn advisory_lock_acquire_release_and_heartbeat() {
    let Ok(url) = std::env::var("AKH_TEST_PG_URL") else {
        eprintln!("skipping: set AKH_TEST_PG_URL");
        return;
    };
    // The advisory lock is session-scoped — a single-connection pool keeps all
    // statements on one backend so acquire/contend/release behave deterministically.
    let opts = PoolOptions {
        max_connections: 1,
        ..PoolOptions::new(url.clone())
    };
    let db = connect(&opts).await.expect("connect");
    run_migrations(&db).await.expect("migrate");

    let akh = Uuid::new_v4();
    ensure_akh(&db, akh, "lock-test-akh", None).await.unwrap();

    // First acquire succeeds; a second connection cannot take it.
    assert!(try_acquire_akh_lock(&db, akh).await.unwrap());
    let other = connect(&PoolOptions { max_connections: 1, ..PoolOptions::new(url) })
        .await
        .expect("connect other");
    assert!(!try_acquire_akh_lock(&other, akh).await.unwrap());

    // Release on the holder, then the other connection can take it.
    assert!(release_akh_lock(&db, akh).await.unwrap());
    assert!(try_acquire_akh_lock(&other, akh).await.unwrap());
    assert!(release_akh_lock(&other, akh).await.unwrap());

    // Heartbeat updates last_seen without error.
    touch_last_seen(&db, akh, akhomed_store::chrono::Utc::now()).await.unwrap();
}

/// Swap the database name in a `postgres://…/db?query` URL, preserving the query
/// string (e.g. `?sslmode=disable`).
fn swap_db(url: &str, db: &str) -> String {
    let (base, query) = match url.split_once('?') {
        Some((b, q)) => (b, Some(q)),
        None => (url, None),
    };
    let prefix = &base[..base.rfind('/').expect("url has a path /")];
    match query {
        Some(q) => format!("{prefix}/{db}?{q}"),
        None => format!("{prefix}/{db}"),
    }
}

/// Several daemons starting against the same *fresh* cluster database race on
/// creating `seaql_migrations` — without serialization some fail with a
/// duplicate-key error on `pg_type`. [`run_migrations_locked`] takes a Postgres
/// advisory lock so concurrent migrators queue; all must succeed. We isolate the
/// first-migration window on a throwaway database so the race is real (the shared
/// test DB is already migrated by the time other tests run).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires AKH_TEST_PG_URL"]
async fn concurrent_locked_migrators_do_not_race() {
    let Ok(url) = std::env::var("AKH_TEST_PG_URL") else {
        eprintln!("skipping: set AKH_TEST_PG_URL");
        return;
    };

    // Throwaway database carved off the same cluster.
    let admin = connect(&PoolOptions::new(url.clone())).await.expect("admin connect");
    let dbname = "akh_mig_race_test";
    let drop = format!("DROP DATABASE IF EXISTS {dbname} WITH (FORCE)");
    admin.execute_unprepared(&drop).await.expect("drop pre-existing");
    admin
        .execute_unprepared(&format!("CREATE DATABASE {dbname}"))
        .await
        .expect("create throwaway db");

    let opts = PoolOptions::new(swap_db(&url, dbname));

    // Fire several locked migrators at once against the empty schema.
    let handles: Vec<_> = (0..6)
        .map(|_| {
            let o = opts.clone();
            tokio::spawn(async move { run_migrations_locked(&o).await })
        })
        .collect();
    for h in handles {
        h.await
            .expect("join migrator task")
            .expect("each concurrent migrator must succeed under the advisory lock");
    }

    // Cleanup (FORCE terminates the migrators' now-idle pooled connections).
    admin.execute_unprepared(&drop).await.expect("drop throwaway db");
}
