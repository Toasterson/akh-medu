//! Live-Postgres test for the full WAL path (Phase 38b): buffer → flush → DB.
//!
//! Compiled only with the `postgres` feature; ignored unless AKH_TEST_PG_URL is
//! set. Run against a throwaway database:
//!   AKH_TEST_PG_URL=postgres://user@127.0.0.1:5439/akhmedu?sslmode=disable \
//!     cargo test --features postgres --test wal_live -- --ignored
#![cfg(feature = "postgres")]

use std::sync::Arc;
use std::time::Duration;

use akh_medu::mutation::{MutationKind, MutationRecord, MutationSink};
use akh_medu::wal::{MutationLog, WalConfig};
use akhomed_store::sea_orm::{self, ConnectionTrait, DbBackend, Statement};
use akhomed_store::uuid::Uuid;

async fn count(db: &sea_orm::DatabaseConnection, akh: Uuid) -> i64 {
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
async fn record_then_flush_writes_rows() {
    let Ok(url) = std::env::var("AKH_TEST_PG_URL") else {
        eprintln!("skipping: set AKH_TEST_PG_URL");
        return;
    };
    let db = akhomed_store::connect(&akhomed_store::PoolOptions::new(url))
        .await
        .unwrap();
    akhomed_store::run_migrations(&db).await.unwrap();

    let akh = Uuid::new_v4();
    let workspace = Uuid::new_v4();
    akhomed_store::ensure_akh(&db, akh, "wal-live", None).await.unwrap();
    let session = Arc::new(
        akhomed_store::StablePgSession::connect(&akhomed_store::PoolOptions::new(
            std::env::var("AKH_TEST_PG_URL").unwrap(),
        ))
        .await
        .unwrap(),
    );
    assert!(session.try_acquire_akh_lock(akh).await.unwrap());
    let epoch = akhomed_store::claim_akh_residency(&db, akh, workspace)
        .await
        .unwrap();

    let log = Arc::new(MutationLog::new(
        Arc::clone(&session),
        akh,
        workspace,
        epoch,
        WalConfig::new(Duration::from_millis(50), 64, 1 << 20),
    ));

    // Record through the MutationSink trait (same path the engine uses).
    for i in 0..10 {
        log.record(MutationRecord::new(
            MutationKind::Assert,
            serde_json::json!({ "subject": i, "predicate": 1, "object": 2 }),
        ));
    }
    assert_eq!(log.pending(), 10);
    assert_eq!(count(&db, akh).await, 0, "nothing flushed yet");

    let flushed = log.flush_once().await;
    assert_eq!(flushed, 10);
    assert_eq!(log.pending(), 0);
    assert_eq!(count(&db, akh).await, 10);

    // A second flush with an empty buffer is a no-op.
    assert_eq!(log.flush_once().await, 0);
    assert_eq!(count(&db, akh).await, 10);
    assert!(session.release_akh_lock(akh).await.unwrap());
}

/// ADR 091: a `MutationKind::Turn` record flows through the same
/// `MutationSink` → `MutationLog` → Postgres path as `Assert`/`Retract` and
/// lands with `kind='turn'`, readable back via the store's turn-read API.
#[tokio::test]
#[ignore = "requires AKH_TEST_PG_URL"]
async fn turn_mutation_flushes_with_turn_kind() {
    let Ok(url) = std::env::var("AKH_TEST_PG_URL") else {
        eprintln!("skipping: set AKH_TEST_PG_URL");
        return;
    };
    let db = akhomed_store::connect(&akhomed_store::PoolOptions::new(url))
        .await
        .unwrap();
    akhomed_store::run_migrations(&db).await.unwrap();

    let akh = Uuid::new_v4();
    let workspace = Uuid::new_v4();
    akhomed_store::ensure_akh(&db, akh, "wal-live-turn", None).await.unwrap();
    let session = Arc::new(
        akhomed_store::StablePgSession::connect(&akhomed_store::PoolOptions::new(
            std::env::var("AKH_TEST_PG_URL").unwrap(),
        ))
        .await
        .unwrap(),
    );
    assert!(session.try_acquire_akh_lock(akh).await.unwrap());
    let epoch = akhomed_store::claim_akh_residency(&db, akh, workspace)
        .await
        .unwrap();

    let log = Arc::new(MutationLog::new(
        Arc::clone(&session),
        akh,
        workspace,
        epoch,
        WalConfig::new(Duration::from_millis(50), 64, 1 << 20),
    ));

    let payload = serde_json::json!({
        "v": 1,
        "ts": "2026-07-10T00:00:00.000000000Z",
        "workspace": "wal-live-turn",
        "model": null,
        "user_prompt": "hi",
        "agent_answer": "hello",
        "tools": [],
    });
    log.record(MutationRecord::new(MutationKind::Turn, payload.clone()));
    assert_eq!(log.pending(), 1);

    let flushed = log.flush_once().await;
    assert_eq!(flushed, 1);
    assert_eq!(count(&db, akh).await, 1);

    let turns = akhomed_store::load_turns_since(&db, akh, 0, 10).await.unwrap();
    assert_eq!(turns.len(), 1, "the flushed row is visible as a kind='turn' row");
    assert_eq!(turns[0].payload, payload);
    assert!(session.release_akh_lock(akh).await.unwrap());
}

#[tokio::test]
#[ignore = "requires AKH_TEST_PG_URL"]
async fn stale_owner_wal_is_rejected_after_independent_epoch_handoff() {
    let Ok(url) = std::env::var("AKH_TEST_PG_URL") else {
        eprintln!("skipping: set AKH_TEST_PG_URL");
        return;
    };
    let opts = akhomed_store::PoolOptions::new(url);
    let db = akhomed_store::connect(&opts).await.unwrap();
    akhomed_store::run_migrations(&db).await.unwrap();

    let akh = Uuid::new_v4();
    let workspace = Uuid::new_v4();
    akhomed_store::ensure_akh(&db, akh, "wal-fence-live", None)
        .await
        .unwrap();
    let old_session = Arc::new(akhomed_store::StablePgSession::connect(&opts).await.unwrap());
    let new_session = Arc::new(akhomed_store::StablePgSession::connect(&opts).await.unwrap());
    assert!(old_session.try_acquire_akh_lock(akh).await.unwrap());
    assert!(
        !new_session.try_acquire_akh_lock(akh).await.unwrap(),
        "a second physical session must not share the owner's advisory lock"
    );
    let old_epoch = akhomed_store::claim_akh_residency(&db, akh, workspace)
        .await
        .unwrap();
    let old_log = Arc::new(MutationLog::new(
        Arc::clone(&old_session),
        akh,
        workspace,
        old_epoch,
        WalConfig::new(Duration::from_millis(50), 64, 1 << 20),
    ));
    old_log.record(MutationRecord::new(
        MutationKind::Assert,
        serde_json::json!({ "subject": 1, "predicate": 2, "object": 3 }),
    ));

    assert!(old_session.release_akh_lock(akh).await.unwrap());
    assert!(new_session.try_acquire_akh_lock(akh).await.unwrap());
    let new_epoch = akhomed_store::claim_akh_residency(&db, akh, workspace)
        .await
        .unwrap();
    assert!(new_epoch > old_epoch);
    let new_log = Arc::new(MutationLog::new(
        Arc::clone(&new_session),
        akh,
        workspace,
        new_epoch,
        WalConfig::new(Duration::from_millis(50), 64, 1 << 20),
    ));
    new_log.record(MutationRecord::new(
        MutationKind::Assert,
        serde_json::json!({ "subject": 4, "predicate": 5, "object": 6 }),
    ));
    assert_eq!(new_log.flush_once().await, 1);
    assert!(new_session.release_akh_lock(akh).await.unwrap());

    // Re-acquiring the physical session is deliberately not enough: the log
    // still carries the former owner's immutable generation and must fail
    // closed without consuming its buffered record.
    assert!(old_session.try_acquire_akh_lock(akh).await.unwrap());
    assert_eq!(old_log.flush_once().await, 0);
    assert_eq!(old_log.pending(), 1);
    assert_eq!(count(&db, akh).await, 1);
    assert_eq!(
        old_log
            .metrics()
            .flush_failures
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
    assert!(old_session.release_akh_lock(akh).await.unwrap());
}
