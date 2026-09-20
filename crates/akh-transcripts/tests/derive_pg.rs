//! Live-Postgres integration tests for increment 5 (the `akh-derive`
//! worker). Ignored by default; run against a scratch database from
//! `scripts/pg-standalone.sh` exactly as `tests/live_pg.rs` documents,
//! **plus** the one extra step the cnpg-timescale wrapper image needs:
//! TimescaleDB is compiled in but not preloaded by default, so
//! `shared_preload_libraries = 'timescaledb'` must be appended to
//! `postgresql.conf` and the container restarted before the extension will
//! actually activate (`CREATE EXTENSION timescaledb` otherwise fails with
//! "not listed in shared_preload_libraries").
//!
//! ```text
//! AKH_PG_IMAGE=code.aopc.cloud/toasterson/cnpg-timescale:18-2.25.1 \
//!   AKH_PG_CONTAINER=akh-derive-test-pg AKH_PG_VOLUME=akh-derive-test-pg-data \
//!   AKH_PG_PORT=55490 AKH_PG_DB=transcripts scripts/pg-standalone.sh up
//! docker exec --user root akh-derive-test-pg bash -c \
//!   "echo \"shared_preload_libraries = 'timescaledb'\" >> /var/lib/postgresql/data/pgdata/postgresql.conf"
//! docker restart akh-derive-test-pg
//! AKH_TEST_TH_PG_URL='postgres://akh:akh@127.0.0.1:55490/transcripts?sslmode=disable' \
//!   cargo test -p akh-transcripts --test derive_pg -- --ignored --nocapture
//! ```
//!
//! No real transcript content anywhere below — shapes only, taken from
//! `src/normalize/tests.rs`'s fixtures per the plan increment's acceptance
//! criteria.

use chrono::Utc;
use sea_orm::{ActiveValue::Set, EntityTrait, PaginatorTrait};
use serde_json::json;

use akh_transcripts::derive::{self, DeriveConfig, pass};
use akh_transcripts::entity::{chunk, event, raw_record};
use akh_transcripts::pool::{connect, run_migrations_locked};

/// Insert one synthetic `th_raw_records` row. `source`/`record_type`/`payload`
/// mirror the shapes `src/normalize/tests.rs` builds for the same scenarios;
/// nothing here is a real transcript.
async fn insert_raw(
    db: &sea_orm::DatabaseConnection,
    source: &str,
    session_id: &str,
    stream_id: &str,
    seq: i64,
    record_type: &str,
    payload: serde_json::Value,
) {
    let now = Utc::now() + chrono::Duration::milliseconds(seq);
    let model = raw_record::ActiveModel {
        observed_at: Set(now),
        source: Set(source.to_string()),
        session_id: Set(session_id.to_string()),
        stream_id: Set(stream_id.to_string()),
        seq: Set(seq),
        stream_kind: Set("main".to_string()),
        machine: Set("derive-pg-test".to_string()),
        raw: Set(payload.to_string()),
        payload: Set(Some(payload)),
        record_type: Set(Some(record_type.to_string())),
        schema_marker: Set(None),
        origin_path: Set("/fixtures/synthetic.jsonl".to_string()),
        observed_at_synthetic: Set(false),
        harvested_at: Set(now),
    };
    raw_record::Entity::insert(model)
        .exec(db)
        .await
        .expect("insert synthetic raw record");
}

/// Seed one raw record per scenario the acceptance criteria list: a
/// multi-block Claude record, a Codex `token_count` noise record, a Codex
/// `world_state` undetermined record, and a malformed record (an
/// unrecognised `source` column — the one shape `akh-derive` itself can hit
/// without depending on a normalizer bug to panic).
async fn seed_corpus(db: &sea_orm::DatabaseConnection) {
    // Claude Code: one `user` record with three `tool_result` blocks ->
    // Outcome::Events(3), subseq 0..2, sharing (source, session, stream, seq).
    insert_raw(
        db,
        "claude_code",
        "sess-cc",
        "main",
        0,
        "user",
        json!({"message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "out1"},
            {"type": "tool_result", "tool_use_id": "t2", "content": "out2"},
            {"type": "tool_result", "tool_use_id": "t3", "content": "out3"},
        ]}}),
    )
    .await;
    // A prompt/reply pair in the same stream so there is something for the
    // chunker to fold into `th_chunks`.
    insert_raw(
        db,
        "claude_code",
        "sess-cc",
        "main",
        1,
        "user",
        json!({"message": {"content": "what does this function do?"}}),
    )
    .await;
    insert_raw(
        db,
        "claude_code",
        "sess-cc",
        "main",
        2,
        "assistant",
        json!({"message": {"content": [{"type": "text", "text": "it parses the transcript."}]}}),
    )
    .await;

    // Codex: `event_msg/token_count` -> Noise (not a drift signal).
    insert_raw(
        db,
        "codex",
        "sess-codex",
        "main",
        0,
        "event_msg",
        json!({"type": "event_msg", "payload": {"type": "token_count", "count": 17311}}),
    )
    .await;
    // Codex: `world_state` -> Undetermined (counted, never guessed).
    insert_raw(
        db,
        "codex",
        "sess-codex",
        "main",
        1,
        "world_state",
        json!({"type": "world_state"}),
    )
    .await;

    // OpenCode: a third source, so all three are exercised.
    insert_raw(
        db,
        "opencode",
        "sess-oc",
        "main",
        0,
        "message.updated",
        json!({
            "id": "evt_1",
            "aggregate_id": "sess-oc",
            "seq": 0,
            "type": "message.updated",
            "data": {"role": "user", "text": "hello"},
        }),
    )
    .await;

    // A malformed record: our own `source` column drifted from
    // claude_code|codex|opencode. `derive_record` catches this before ever
    // calling `normalize`, and it must still produce a row
    // (`disposition = 'failed'`) rather than being silently skipped.
    insert_raw(
        db,
        "not-a-real-source",
        "sess-bad",
        "main",
        0,
        "user",
        json!({}),
    )
    .await;
}

/// `(id, embedding-as-pgvector-text, embedded_at.is_some())` for every
/// `th_chunks` row, via an explicit `embedding::text` cast — see the caller's
/// comment for why a plain SeaORM `Entity::find()` cannot be used here once
/// any row has a real vector.
async fn chunk_summaries(
    db: &sea_orm::DatabaseConnection,
) -> Vec<(i64, Option<String>, Option<chrono::DateTime<Utc>>)> {
    use sea_orm::{ConnectionTrait, DatabaseBackend, Statement};
    let rows = db
        .query_all(Statement::from_string(
            DatabaseBackend::Postgres,
            "SELECT id, embedding::text AS embedding, embedded_at FROM th_chunks ORDER BY id"
                .to_owned(),
        ))
        .await
        .expect("query chunk summaries");
    rows.into_iter()
        .map(|r| {
            let id: i64 = r.try_get("", "id").expect("id");
            let embedding: Option<String> = r.try_get("", "embedding").expect("embedding");
            let embedded_at: Option<chrono::DateTime<Utc>> =
                r.try_get("", "embedded_at").expect("embedded_at");
            (id, embedding, embedded_at)
        })
        .collect()
}

fn maybe_embedder() -> Option<akh_embed::SharedEmbedder> {
    let dir = std::env::var("AKH_TRANSCRIPTS_MODEL_DIR")
        .or_else(|_| std::env::var("AKH_MINILM_DIR"))
        .ok()?;
    let embedder =
        akh_embed::SentenceEmbedder::load(&std::path::PathBuf::from(dir)).expect("load embedder");
    Some(std::sync::Arc::new(std::sync::Mutex::new(embedder)))
}

#[tokio::test]
#[ignore = "requires AKH_TEST_TH_PG_URL (cnpg-timescale image with shared_preload_libraries set — see module docs)"]
async fn one_pass_derives_every_scenario_and_a_second_pass_is_a_no_op() {
    let Ok(url) = std::env::var("AKH_TEST_TH_PG_URL") else {
        eprintln!("skipping: set AKH_TEST_TH_PG_URL");
        return;
    };

    run_migrations_locked(&url).await.expect("migrate");
    let db = connect(&url).await.expect("connect");

    seed_corpus(&db).await;

    // --- anti-join is index-backed, verified via a real EXPLAIN --------
    let plan = {
        use sea_orm::{ConnectionTrait, DatabaseBackend, Statement};
        let rows = db
            .query_all(Statement::from_string(
                DatabaseBackend::Postgres,
                "EXPLAIN SELECT DISTINCT r.source, r.session_id, r.stream_id \
                 FROM th_raw_records r \
                 WHERE NOT EXISTS ( \
                     SELECT 1 FROM th_events e \
                     WHERE e.derivation_version = 1 AND e.source = r.source \
                       AND e.session_id = r.session_id AND e.stream_id = r.stream_id \
                       AND e.seq = r.seq AND e.subseq = 0 \
                 ) LIMIT 200"
                    .to_owned(),
            ))
            .await
            .expect("explain");
        rows.into_iter()
            .map(|r| {
                r.try_get::<String>("", "QUERY PLAN")
                    .expect("EXPLAIN output has a QUERY PLAN column")
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    println!("EXPLAIN pending-streams anti-join:\n{plan}");
    assert!(
        !plan.contains("Seq Scan on th_events"),
        "the anti-join must not full-scan th_events at any corpus size: {plan}"
    );

    // --- pass 1: everything derives -----------------------------------
    let embedder = maybe_embedder();
    let config = DeriveConfig {
        batch_streams: 100,
        embed_batch: embedder.as_ref().map(|_| 64),
    };
    let summary = derive::run_pass(&db, &config, embedder.as_ref())
        .await
        .expect("first pass");
    println!("pass 1: {summary}");

    assert_eq!(summary.streams, 4, "four distinct streams were seeded");
    assert_eq!(summary.records, 7, "seven raw records were seeded");
    // Claude: 3 tool_result events + 1 prompt + 1 reply; OpenCode: 1 prompt.
    // = 6 disposition='event' rows.
    assert_eq!(summary.events, 6);
    assert_eq!(summary.noise, 1, "the token_count record");
    assert_eq!(summary.undetermined, 1, "the world_state record");
    assert_eq!(summary.failed, 1, "the unrecognised-source record");

    // Anti-join now returns nothing: every raw row has a matching th_events
    // row at the current version.
    let pending = pass::find_pending_streams(&db, derive::DERIVATION_VERSION, 100)
        .await
        .expect("query pending");
    assert!(
        pending.is_empty(),
        "anti-join must be empty after a full pass, found: {pending:?}"
    );

    // Every raw row has a matching th_events row.
    let raw_count = raw_record::Entity::find()
        .count(&db)
        .await
        .expect("count raw");
    let event_rows = event::Entity::find().all(&db).await.expect("all events");
    assert_eq!(raw_count, 7);
    // subseq=0 rows are exactly one per raw record (the anti-join's own
    // invariant: every record writes at least a subseq=0 row).
    let subseq_zero_count = event_rows.iter().filter(|e| e.subseq == 0).count();
    assert_eq!(subseq_zero_count as u64, raw_count);

    // Dispositions match the summary.
    let by_disposition = |d: &str| event_rows.iter().filter(|e| e.disposition == d).count();
    assert_eq!(by_disposition("event"), 6);
    assert_eq!(by_disposition("noise"), 1);
    assert_eq!(by_disposition("undetermined"), 1);
    assert_eq!(by_disposition("failed"), 1);
    // `kind` is NULL for every non-'event' disposition, never a pseudo-kind.
    assert!(
        event_rows
            .iter()
            .filter(|e| e.disposition != "event")
            .all(|e| e.kind.is_none())
    );
    assert!(
        event_rows
            .iter()
            .filter(|e| e.disposition == "event")
            .all(|e| e.kind.is_some())
    );

    // th_sessions rolled up for the two "real" sessions (the failed row's
    // session has no events, so it still gets a session row from the raw
    // count, but with no title/repo — just asserting it does not error is
    // enough here).
    let sessions = akh_transcripts::entity::session::Entity::find()
        .all(&db)
        .await
        .expect("sessions");
    assert!(sessions.iter().any(|s| s.session_id == "sess-cc"));
    assert!(sessions.iter().any(|s| s.session_id == "sess-codex"));
    assert!(sessions.iter().any(|s| s.session_id == "sess-oc"));

    // th_chunks: one prompt + one reply in sess-cc plus one prompt in
    // sess-oc, turn-boundary chunked into (at least) two chunks.
    //
    // Deliberately **not** `chunk::Entity::find().all(&db)`: SeaORM/sqlx has
    // no `Decode` for pgvector's `vector` type into the entity's
    // `Option<String>` column mapping, so a plain `SELECT *` panics as soon
    // as any row has a non-NULL embedding (`ColumnDecode` — "mismatched
    // types ... TEXT is not compatible with SQL type vector"). Reading rows
    // back once they may carry a real vector needs the same `::text` cast
    // the entity's own doc comment describes for writing one; this worker's
    // production code never hits this because `embed::embed_pending`'s
    // pending-lookup filters to `embedded_at IS NULL` rows, which are
    // guaranteed to have a NULL (never a real) embedding.
    let chunks_before = chunk_summaries(&db).await;
    assert!(
        !chunks_before.is_empty(),
        "the prompt/reply pairs must produce at least one chunk"
    );
    if embedder.is_some() {
        for (id, embedding, embedded_at) in &chunks_before {
            let embedding = embedding.as_deref().unwrap_or_else(|| {
                panic!("chunk {id} must be embedded when an embedder was configured")
            });
            let vec = chunk::pgvector_to_vec(embedding);
            assert_eq!(vec.len(), 384, "MiniLM vectors are 384-dimensional");
            assert!(embedded_at.is_some());
        }
        println!(
            "ASSERTED: {} chunk(s) embedded at 384 dimensions",
            chunks_before.len()
        );
    } else {
        eprintln!(
            "AKH_TRANSCRIPTS_MODEL_DIR/AKH_MINILM_DIR not set — skipping embedding assertions"
        );
    }

    // --- pass 2: no-op ---------------------------------------------------
    let summary2 = derive::run_pass(&db, &config, embedder.as_ref())
        .await
        .expect("second pass");
    println!("pass 2: {summary2}");
    assert_eq!(
        summary2.streams, 0,
        "nothing should be pending on a second pass"
    );
    assert_eq!(summary2.records, 0);

    let event_count_after = event::Entity::find()
        .count(&db)
        .await
        .expect("count events");
    assert_eq!(
        event_count_after,
        event_rows.len() as u64,
        "a no-op pass must not write additional th_events rows"
    );
    let chunk_count_after = chunk::Entity::find()
        .count(&db)
        .await
        .expect("count chunks");
    assert_eq!(
        chunk_count_after,
        chunks_before.len() as u64,
        "a no-op pass must not write additional th_chunks rows"
    );

    // --- version bump re-pends everything --------------------------------
    // `derive::DERIVATION_VERSION` is a compile-time constant, so the
    // version-bump trigger is exercised by calling the version-parameterized
    // `pass::find_pending_streams` directly at a version nothing has been
    // derived at yet, rather than recompiling with a different constant.
    let pending_at_next_version =
        pass::find_pending_streams(&db, derive::DERIVATION_VERSION + 1, 100)
            .await
            .expect("query pending at bumped version");
    assert_eq!(
        pending_at_next_version.len(),
        4,
        "bumping the derivation version must re-pend every stream, found: {pending_at_next_version:?}"
    );

    // And processing one of them at the bumped version writes new rows
    // alongside the old-version ones (no orphans, no deletes — the
    // derivation-worker ticket's re-derivation contract).
    let next_version = derive::DERIVATION_VERSION + 1;
    let stats = pass::process_stream(&db, next_version, "codex", "sess-codex", "main")
        .await
        .expect("process at bumped version");
    assert_eq!(stats.noise, 1);
    assert_eq!(stats.undetermined, 1);
    let all_rows_after_bump = event::Entity::find().all(&db).await.expect("all events");
    let old_version_rows = all_rows_after_bump
        .iter()
        .filter(|e| e.derivation_version == derive::DERIVATION_VERSION)
        .count();
    let new_version_rows = all_rows_after_bump
        .iter()
        .filter(|e| e.derivation_version == next_version)
        .count();
    assert_eq!(
        old_version_rows,
        event_rows.len(),
        "old-version rows must survive a bumped-version derivation untouched, not deleted or orphaned"
    );
    assert_eq!(
        new_version_rows, 2,
        "the bumped-version pass wrote its own noise + undetermined rows"
    );
}
