//! Live-Postgres integration tests for the 38b write API.
//!
//! Ignored by default; run against a throwaway database:
//!   AKH_TEST_PG_URL=postgres://user@127.0.0.1:5439/akhmedu?sslmode=disable \
//!     cargo test -p akhomed-store --test live_pg -- --ignored

use akhomed_store::entity::{fact, symbol};
use akhomed_store::snapshots::{
    load_symbols, load_workspace_facts, replace_symbols, replace_workspace_facts,
};
use akhomed_store::uuid::Uuid;
use akhomed_store::{
    MutationRow, PoolOptions, append_mutations, connect, ensure_akh, get_harvest_watermark,
    join_workspace, leave_workspace, list_members, load_turn_by_seq, load_turns_since,
    release_akh_lock, run_migrations, run_migrations_locked, set_harvest_watermark,
    touch_last_seen, try_acquire_akh_lock,
};
use sea_orm::{ConnectionTrait, DbBackend, Statement};

async fn mutation_count(db: &sea_orm::DatabaseConnection, akh: Uuid) -> i64 {
    let row = db
        .query_one(Statement::from_string(
            DbBackend::Postgres,
            format!(
                "SELECT count(*)::bigint AS c FROM akhomed_mutation_log WHERE akh_id = '{akh}'"
            ),
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
    ensure_akh(&db, akh, "live-test-akh", Some("test-host"))
        .await
        .unwrap();
    ensure_akh(&db, akh, "live-test-akh", Some("test-host"))
        .await
        .unwrap();

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

/// A fact has ten bound columns, so 7,000 rows exceed Postgres's 65,535 bind
/// parameter ceiling and force `replace_workspace_facts` to issue more than
/// one INSERT. Repeating the replacement proves the normal periodic-snapshot
/// path remains exact after the first write.
#[tokio::test]
#[ignore = "requires AKH_TEST_PG_URL"]
async fn workspace_fact_replace_chunks_past_postgres_bind_limit() {
    let Ok(url) = std::env::var("AKH_TEST_PG_URL") else {
        eprintln!("skipping: set AKH_TEST_PG_URL");
        return;
    };
    let db = connect(&PoolOptions::new(url)).await.expect("connect");
    run_migrations(&db).await.expect("migrate");

    const ROW_COUNT: usize = 7_000;
    let workspace = Uuid::new_v4();
    let akh = Uuid::new_v4();
    let asserted_at = akhomed_store::chrono::Utc::now();
    let mut rows: Vec<fact::Model> = (0..ROW_COUNT)
        .map(|index| fact::Model {
            workspace_id: workspace,
            subject: index as i64,
            predicate: 1,
            object: index as i64 + 10_000,
            akh_id: akh,
            confidence: 1.0,
            asserted_at,
            retracted_at: None,
            provenance_id: None,
            compartment: None,
        })
        .collect();

    replace_workspace_facts(&db, workspace, rows.clone())
        .await
        .expect("first chunked replacement");
    let first = load_workspace_facts(&db, workspace)
        .await
        .expect("load first replacement");
    assert_eq!(first.len(), ROW_COUNT);

    for row in &mut rows {
        row.confidence = 0.25;
    }
    replace_workspace_facts(&db, workspace, rows)
        .await
        .expect("repeat chunked replacement");
    let second = load_workspace_facts(&db, workspace)
        .await
        .expect("load repeat replacement");
    assert_eq!(second.len(), ROW_COUNT);
    assert!(second.iter().all(|row| row.confidence == 0.25));
}

/// Arbitrary knowledge text is a valid symbol label. The Postgres migration
/// must keep a label larger than a B-tree index tuple from aborting snapshots.
#[tokio::test]
#[ignore = "requires AKH_TEST_PG_URL"]
async fn symbol_snapshot_accepts_label_larger_than_postgres_btree_tuple() {
    let Ok(url) = std::env::var("AKH_TEST_PG_URL") else {
        eprintln!("skipping: set AKH_TEST_PG_URL");
        return;
    };
    let db = connect(&PoolOptions::new(url)).await.expect("connect");
    run_migrations(&db).await.expect("migrate");

    let akh = Uuid::new_v4();
    ensure_akh(&db, akh, "long-label-test-akh", None)
        .await
        .unwrap();
    let label = "knowledge ".repeat(500);
    replace_symbols(
        &db,
        akh,
        vec![symbol::Model {
            akh_id: akh,
            id: 1,
            kind: "Entity".into(),
            glyph_codepoint: None,
            label: label.clone(),
        }],
    )
    .await
    .expect("long symbol label must persist");

    let restored = load_symbols(&db, akh)
        .await
        .expect("load long symbol label");
    assert_eq!(restored.len(), 1);
    assert_eq!(restored[0].label, label);
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
    join_workspace(&db, workspace, akh, "member", now)
        .await
        .unwrap();
    join_workspace(&db, workspace, akh, "member", now)
        .await
        .unwrap();
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
    let other = connect(&PoolOptions {
        max_connections: 1,
        ..PoolOptions::new(url)
    })
    .await
    .expect("connect other");
    assert!(!try_acquire_akh_lock(&other, akh).await.unwrap());

    // Release on the holder, then the other connection can take it.
    assert!(release_akh_lock(&db, akh).await.unwrap());
    assert!(try_acquire_akh_lock(&other, akh).await.unwrap());
    assert!(release_akh_lock(&other, akh).await.unwrap());

    // Heartbeat updates last_seen without error.
    touch_last_seen(&db, akh, akhomed_store::chrono::Utc::now())
        .await
        .unwrap();
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
    let admin = connect(&PoolOptions::new(url.clone()))
        .await
        .expect("admin connect");
    let dbname = "akh_mig_race_test";
    let drop = format!("DROP DATABASE IF EXISTS {dbname} WITH (FORCE)");
    admin
        .execute_unprepared(&drop)
        .await
        .expect("drop pre-existing");
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
    admin
        .execute_unprepared(&drop)
        .await
        .expect("drop throwaway db");
}

/// ADR 091: a `kind='turn'` WAL row lands via `append_mutations` and is
/// readable back through `load_turns_since` / `load_turn_by_seq`.
#[tokio::test]
#[ignore = "requires AKH_TEST_PG_URL"]
async fn turn_rows_land_and_are_readable() {
    let Ok(url) = std::env::var("AKH_TEST_PG_URL") else {
        eprintln!("skipping: set AKH_TEST_PG_URL");
        return;
    };
    let db = connect(&PoolOptions::new(url)).await.expect("connect");
    run_migrations(&db).await.expect("migrate");

    let akh = Uuid::new_v4();
    let workspace = Uuid::new_v4();
    ensure_akh(&db, akh, "turn-row-test-akh", None)
        .await
        .unwrap();

    // Nothing captured yet.
    assert!(load_turns_since(&db, akh, 0, 100).await.unwrap().is_empty());

    let payload = serde_json::json!({
        "v": 1,
        "ts": "2026-07-10T00:00:00.000000000Z",
        "workspace": "turn-row-test-akh",
        "model": "glm-5.2",
        "user_prompt": "what is the capital of France?",
        "agent_answer": "Paris.",
        "tools": [],
    });
    let rows = vec![MutationRow {
        kind: "turn".into(),
        payload: payload.clone(),
    }];
    assert_eq!(
        append_mutations(&db, akh, workspace, &rows).await.unwrap(),
        1
    );

    let turns = load_turns_since(&db, akh, 0, 100).await.unwrap();
    assert_eq!(turns.len(), 1, "one turn row visible");
    assert_eq!(turns[0].payload, payload);

    let by_seq = load_turn_by_seq(&db, akh, turns[0].seq).await.unwrap();
    assert_eq!(by_seq, Some(payload));

    // `after_seq` at-or-past the row's seq excludes it.
    assert!(
        load_turns_since(&db, akh, turns[0].seq, 100)
            .await
            .unwrap()
            .is_empty()
    );

    // A non-`turn` seq (or one that doesn't exist) is `None`.
    assert_eq!(
        load_turn_by_seq(&db, akh, turns[0].seq + 999_999)
            .await
            .unwrap(),
        None
    );
}

/// ADR 091: the harvest watermark round-trips through get/set, and a full
/// `akh_state` snapshot `upsert` never clobbers a value `set_harvest_watermark`
/// already wrote (the clobber-guard this slice depends on).
#[tokio::test]
#[ignore = "requires AKH_TEST_PG_URL"]
async fn harvest_watermark_roundtrips_and_survives_state_upsert() {
    let Ok(url) = std::env::var("AKH_TEST_PG_URL") else {
        eprintln!("skipping: set AKH_TEST_PG_URL");
        return;
    };
    let db = connect(&PoolOptions::new(url)).await.expect("connect");
    run_migrations(&db).await.expect("migrate");

    let akh = Uuid::new_v4();
    ensure_akh(&db, akh, "watermark-test-akh", None)
        .await
        .unwrap();

    // 0 with no akh_state row at all.
    assert_eq!(get_harvest_watermark(&db, akh).await.unwrap(), 0);

    // Setting with no row yet inserts a minimal default row.
    set_harvest_watermark(&db, akh, 7).await.unwrap();
    assert_eq!(get_harvest_watermark(&db, akh).await.unwrap(), 7);

    // Advancing again updates in place.
    set_harvest_watermark(&db, akh, 42).await.unwrap();
    assert_eq!(get_harvest_watermark(&db, akh).await.unwrap(), 42);

    // A full-state snapshot upsert (as `persist_snapshot` performs every sleep
    // tick) with a stale `harvest_watermark` value in the in-memory model must
    // NOT clobber the 42 already durable — `update_columns` excludes the
    // column, so only the dedicated setter can move it.
    let stale_state = akhomed_store::entity::akh_state::Model {
        akh_id: akh,
        watermark: 100,
        allocator_next: 5,
        cycle_count: 3,
        session_start_cycle: 0,
        sleep_phase: "Wake".into(),
        last_sleep_cycle: 1,
        psyche: None,
        extra: None,
        harvest_watermark: 0, // stale/wrong on purpose
        librarian_watermark: 0,
        updated_at: akhomed_store::chrono::Utc::now(),
    };
    akhomed_store::snapshots::upsert_akh_state(&db, stale_state)
        .await
        .unwrap();

    assert_eq!(
        get_harvest_watermark(&db, akh).await.unwrap(),
        42,
        "snapshot upsert must not clobber the harvest watermark"
    );
    // The columns the upsert IS responsible for did land.
    let state = akhomed_store::snapshots::load_akh_state(&db, akh)
        .await
        .unwrap()
        .expect("akh_state row exists");
    assert_eq!(state.watermark, 100);
    assert_eq!(state.cycle_count, 3);
}
