//! Live-Postgres integration tests for increment 1 (schema + migrations).
//!
//! Ignored by default; run against throwaway databases from
//! `scripts/pg-standalone.sh` (see its header for the CNPG-driven boot dance):
//!
//! ```text
//! # Full-featured image: both timescaledb and vector present.
//! AKH_PG_IMAGE=code.aopc.cloud/toasterson/cnpg-timescale:18-2.25.1 \
//!   AKH_PG_CONTAINER=akh-th-pg AKH_PG_VOLUME=akh-th-pg-data AKH_PG_PORT=55432 \
//!   AKH_PG_DB=transcripts scripts/pg-standalone.sh up
//! AKH_TEST_TH_PG_URL='postgres://akh:akh@127.0.0.1:55432/transcripts?sslmode=disable' \
//!   cargo test -p akh-transcripts --test live_pg -- --ignored
//!
//! # Plain image: vector present, timescaledb absent — proves the guarded
//! # warn-and-continue path.
//! AKH_PG_IMAGE=ghcr.io/cloudnative-pg/postgresql:18.3-system-trixie \
//!   AKH_PG_CONTAINER=akh-th-pg-plain AKH_PG_VOLUME=akh-th-pg-plain-data \
//!   AKH_PG_PORT=55433 AKH_PG_DB=transcripts scripts/pg-standalone.sh up
//! AKH_TEST_TH_PG_URL_PLAIN='postgres://akh:akh@127.0.0.1:55433/transcripts?sslmode=disable' \
//!   cargo test -p akh-transcripts --test live_pg -- --ignored
//! ```

use akh_transcripts::entity::raw_record;
use akh_transcripts::pool::{connect, run_migrations_locked};
use sea_orm::sea_query::OnConflict;
use sea_orm::{ActiveValue::Set, ConnectionTrait, DbBackend, EntityTrait, Statement};

fn sample_raw_record() -> raw_record::ActiveModel {
    let now = chrono::Utc::now();
    raw_record::ActiveModel {
        observed_at: Set(now),
        source: Set("claude_code".to_owned()),
        session_id: Set("session-live-pg-test".to_owned()),
        stream_id: Set("main".to_owned()),
        seq: Set(0),
        stream_kind: Set("main".to_owned()),
        machine: Set("live-pg-test-runner".to_owned()),
        raw: Set(r#"{"type":"user","message":{"content":"synthetic fixture"}}"#.to_owned()),
        payload: Set(Some(serde_json::json!({"type": "user"}))),
        record_type: Set(Some("user".to_owned())),
        schema_marker: Set(Some("v1".to_owned())),
        origin_path: Set("/fixtures/synthetic-session.jsonl".to_owned()),
        observed_at_synthetic: Set(false),
        harvested_at: Set(now),
    }
}

/// Insert the same record twice via `INSERT ... ON CONFLICT DO NOTHING`
/// (the raw table's insert-only, idempotent-by-construction contract from
/// the ingest-transport ticket) and assert exactly one row survives.
///
/// Builds the record **once** and reuses it for both attempts: `observed_at`
/// is part of the primary key, and the whole point of the forward-fill rule
/// is that it must never be recomputed between runs — generating a fresh
/// `Utc::now()` per attempt would silently insert two rows instead of
/// colliding, which is exactly the hazard the store-schema ticket warns
/// against.
async fn insert_twice_with_do_nothing(
    db: &sea_orm::DatabaseConnection,
) -> Result<(), sea_orm::DbErr> {
    let record = sample_raw_record();
    for _ in 0..2 {
        let result = raw_record::Entity::insert(record.clone())
            .on_conflict(
                OnConflict::columns([
                    raw_record::Column::ObservedAt,
                    raw_record::Column::Source,
                    raw_record::Column::SessionId,
                    raw_record::Column::StreamId,
                    raw_record::Column::Seq,
                ])
                .do_nothing()
                .to_owned(),
            )
            .exec(db)
            .await;
        match result {
            Ok(_) => {}
            // do_nothing's second, colliding insert reports zero affected
            // rows as RecordNotInserted — that IS the no-op success case.
            Err(sea_orm::DbErr::RecordNotInserted) => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[tokio::test]
#[ignore = "requires AKH_TEST_TH_PG_URL (cnpg-timescale image: timescaledb + vector)"]
async fn migrations_apply_and_hypertable_and_indexes_exist() {
    let Ok(url) = std::env::var("AKH_TEST_TH_PG_URL") else {
        eprintln!("skipping: set AKH_TEST_TH_PG_URL");
        return;
    };

    run_migrations_locked(&url).await.expect("migrate");
    let db = connect(&url).await.expect("connect");

    // th_raw_records IS a hypertable on this image.
    let hypertable_row = db
        .query_one(Statement::from_string(
            DbBackend::Postgres,
            "SELECT hypertable_name FROM timescaledb_information.hypertables \
             WHERE hypertable_name = 'th_raw_records'"
                .to_owned(),
        ))
        .await
        .expect("query hypertables")
        .expect("th_raw_records must be a hypertable when timescaledb is present");
    let hypertable_name: String = hypertable_row.try_get("", "hypertable_name").unwrap();
    assert_eq!(hypertable_name, "th_raw_records");
    println!("ASSERTED: th_raw_records is a TimescaleDB hypertable ({hypertable_name})");

    // Idempotent insert: two writes, one row.
    insert_twice_with_do_nothing(&db)
        .await
        .expect("idempotent insert");
    let count_row = db
        .query_one(Statement::from_string(
            DbBackend::Postgres,
            "SELECT count(*)::bigint AS c FROM th_raw_records \
             WHERE session_id = 'session-live-pg-test'"
                .to_owned(),
        ))
        .await
        .expect("count query")
        .expect("count row");
    let count: i64 = count_row.try_get("", "c").unwrap();
    assert_eq!(
        count, 1,
        "ON CONFLICT DO NOTHING must leave exactly one row"
    );
    println!("ASSERTED: double insert via ON CONFLICT DO NOTHING left exactly {count} row");

    // th_events' unique key — the derivation worker's anti-join index —
    // exists as a real Postgres index.
    let index_row = db
        .query_one(Statement::from_string(
            DbBackend::Postgres,
            "SELECT indexname FROM pg_indexes WHERE tablename = 'th_events' \
             AND indexname LIKE '%pkey%'"
                .to_owned(),
        ))
        .await
        .expect("query pg_indexes")
        .expect("th_events must have a primary-key-backed index");
    let index_name: String = index_row.try_get("", "indexname").unwrap();
    println!("ASSERTED: th_events unique index exists ({index_name})");
}

#[tokio::test]
#[ignore = "requires AKH_TEST_TH_PG_URL_PLAIN (plain CNPG image: vector present, timescaledb absent)"]
async fn timescaledb_absent_warns_and_stays_plain_while_pgvector_succeeds() {
    let Ok(url) = std::env::var("AKH_TEST_TH_PG_URL_PLAIN") else {
        eprintln!("skipping: set AKH_TEST_TH_PG_URL_PLAIN");
        return;
    };

    // Must not fail: TimescaleDB absence is warn-and-continue, and this image
    // has pgvector (the CNPG base image ships it), so migrations succeed.
    run_migrations_locked(&url)
        .await
        .expect("migrate on timescaledb-absent image");
    let db = connect(&url).await.expect("connect");

    // Confirm the premise: timescaledb really is absent on this image, so the
    // warn-and-continue path is the one that actually ran.
    let ext_row = db
        .query_one(Statement::from_string(
            DbBackend::Postgres,
            "SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb'".to_owned(),
        ))
        .await
        .expect("query pg_available_extensions");
    assert!(
        ext_row.is_none(),
        "this test's premise is that timescaledb is NOT available on the plain image"
    );

    // th_raw_records must still exist as a plain table with the identical PK.
    let not_hypertable = db
        .query_one(Statement::from_string(
            DbBackend::Postgres,
            "SELECT to_regclass('th_raw_records')::text AS reg".to_owned(),
        ))
        .await
        .expect("query to_regclass")
        .expect("row");
    let reg: Option<String> = not_hypertable.try_get("", "reg").unwrap();
    assert_eq!(reg.as_deref(), Some("th_raw_records"));
    println!("ASSERTED: th_raw_records exists as a plain table when timescaledb is absent");

    // Idempotent insert still holds on the plain-table path.
    insert_twice_with_do_nothing(&db)
        .await
        .expect("idempotent insert on plain table");
    let count_row = db
        .query_one(Statement::from_string(
            DbBackend::Postgres,
            "SELECT count(*)::bigint AS c FROM th_raw_records \
             WHERE session_id = 'session-live-pg-test'"
                .to_owned(),
        ))
        .await
        .expect("count query")
        .expect("count row");
    let count: i64 = count_row.try_get("", "c").unwrap();
    assert_eq!(count, 1);

    // pgvector succeeded: th_chunks' embedding column and HNSW index exist.
    let vector_index = db
        .query_one(Statement::from_string(
            DbBackend::Postgres,
            "SELECT indexname FROM pg_indexes WHERE tablename = 'th_chunks' \
             AND indexname = 'idx_th_chunks_embedding'"
                .to_owned(),
        ))
        .await
        .expect("query pg_indexes")
        .expect("th_chunks HNSW index must exist: pgvector succeeded on this image");
    let _: String = vector_index.try_get("", "indexname").unwrap();
    println!("ASSERTED: pgvector HNSW index exists even though timescaledb is absent");
}
