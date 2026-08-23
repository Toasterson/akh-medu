//! Explicit, disposable Phase 42c runtime acceptance.
//!
//! This module is ignored by the ordinary test suite. The companion script
//! provisions a loopback-only pgvector instance and model, then opts in with
//! guarded environment variables. All SQL is executed through SeaORM; no
//! database shell is used.

use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use miette::{IntoDiagnostic, WrapErr};
use sea_orm::{ConnectionTrait, DatabaseBackend, DatabaseConnection, Statement, TransactionTrait};
use sea_orm_migration::MigratorTrait;
use sha2::{Digest, Sha256};
use tokio::sync::Barrier;
use uuid::Uuid;

use super::*;
use crate::capture::{CapturePolicy, parse_capture_input_with_policy};
use crate::config::{
    DatabaseConfig, EmbeddingConfig, EvalConfig, HekaConfig, ServerConfig, connect_database,
};
use crate::embed_worker::{count_pending_for_akh, embed_pending_batch_for_akh};
use crate::embedder::{SentenceEmbedder, SharedEmbedder};
use crate::entity::heka_drawer_embeddings::vec_to_pgvector;
use crate::ingest::write_drawer;
use crate::migration::Migrator;

const LIVE_OPT_IN: &str = "HEKA_PHASE42C_LIVE";
const ADMIN_URL_ENV: &str = "HEKA_PHASE42C_ADMIN_DATABASE_URL";
const RESOURCE_ENV: &str = "HEKA_PHASE42C_RESOURCE_ID";
const MODEL_DIR_ENV: &str = "HEKA_PHASE42C_MODEL_DIR";
const DATASET_ENV: &str = "HEKA_PHASE42C_LONGMEMEVAL_DATASET";
const HEKAD_BIN_ENV: &str = "HEKA_PHASE42C_HEKAD_BIN";
const MODEL_NAME: &str = "all-MiniLM-L6-v2";
const LEGACY_REPAIR_AUDIT_KEY: &str = "heka_phase42c_legacy_title_hash_repair";
const LEGACY_REPAIR_AUDIT_VALUE: &str = "repaired:m20260604_000002_drawer_ingest";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires the explicitly disposable PostgreSQL+pgvector and ONNX harness"]
async fn phase42c_live_acceptance() {
    let result = guarded_live_acceptance().await;
    if let Err(error) = result {
        panic!("Phase 42c live acceptance failed: {error:?}");
    }
}

async fn guarded_live_acceptance() -> miette::Result<()> {
    miette::ensure!(
        matches!(std::env::var(LIVE_OPT_IN).as_deref(), Ok("1")),
        "{LIVE_OPT_IN}=1 is required"
    );
    let resource = std::env::var(RESOURCE_ENV)
        .into_diagnostic()
        .wrap_err_with(|| format!("{RESOURCE_ENV} is required"))?;
    validate_resource_id(&resource)?;
    let admin_url = std::env::var(ADMIN_URL_ENV)
        .into_diagnostic()
        .wrap_err_with(|| format!("{ADMIN_URL_ENV} is required"))?;
    let database_url = disposable_database_url(&admin_url, &resource)?;
    let model_dir = PathBuf::from(
        std::env::var_os(MODEL_DIR_ENV)
            .ok_or_else(|| miette::miette!("{MODEL_DIR_ENV} is required"))?,
    );
    validate_model_dir(&model_dir)?;
    let dataset = PathBuf::from(
        std::env::var_os(DATASET_ENV)
            .ok_or_else(|| miette::miette!("{DATASET_ENV} is required"))?,
    );
    miette::ensure!(
        dataset.is_file(),
        "official LongMemEval dataset is unavailable"
    );

    let admin = connect_without_url_logging(&admin_url, 2, "admin database").await?;
    // The resource name is strict lower-hex data under a fixed prefix, so it
    // is safe to quote as an identifier. Cleaning a stale exact-name database
    // is confined to the loopback-only disposable server.
    drop_database(&admin, &resource).await?;
    admin
        .execute_unprepared(&format!("CREATE DATABASE \"{resource}\""))
        .await
        .into_diagnostic()
        .wrap_err("creating disposable Phase 42c database")?;

    let connected = connect_without_url_logging(&database_url, 8, "test database").await;
    let outcome = match connected {
        Ok(db) => {
            let outcome = run_acceptance(&db, &database_url, &model_dir, &dataset, &resource).await;
            let close = db
                .close()
                .await
                .into_diagnostic()
                .wrap_err("closing disposable Phase 42c database");
            match (outcome, close) {
                (Ok(()), Ok(())) => Ok(()),
                (Err(error), Ok(())) => Err(error),
                (Ok(()), Err(error)) => Err(error),
                (Err(error), Err(close_error)) => Err(miette::miette!(
                    "{error}; database close also failed: {close_error}"
                )),
            }
        }
        Err(error) => Err(error),
    };

    let cleanup = drop_database(&admin, &resource).await;
    let admin_close = admin
        .close()
        .await
        .into_diagnostic()
        .wrap_err("closing disposable admin database");
    match (outcome, cleanup, admin_close) {
        (Ok(()), Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(()), Ok(())) => Err(error),
        (result, cleanup, close) => Err(miette::miette!(
            "acceptance result: {result:?}; cleanup result: {cleanup:?}; admin close result: {close:?}"
        )),
    }
}

async fn run_acceptance(
    db: &DatabaseConnection,
    database_url: &str,
    model_dir: &Path,
    dataset: &Path,
    resource: &str,
) -> miette::Result<()> {
    verify_initial_migration_ladder(db, resource).await?;
    verify_legacy_migration_upgrade(db, database_url, resource).await?;

    let capture_akh = insert_akh(db, &format!("{resource}_capture")).await?;
    let batch = realistic_two_hundred_turn_capture(resource)?;
    miette::ensure!(
        batch.turns.len() == 200,
        "Stop capture did not yield 200 turns"
    );
    for turn in &batch.turns {
        let summary = write_drawer(
            db,
            capture_akh,
            batch.session_id.as_deref(),
            turn.speaker.as_deref(),
            turn.body.as_deref().unwrap_or_default(),
            None,
            &[],
        )
        .await
        .into_diagnostic()
        .wrap_err("capturing a production-path drawer")?;
        miette::ensure!(
            !summary.duplicated,
            "first capture unexpectedly deduplicated"
        );
    }
    miette::ensure!(
        count_drawers(db, capture_akh).await? == 200,
        "first capture did not persist exactly 200 drawers"
    );

    for turn in &batch.turns {
        let summary = write_drawer(
            db,
            capture_akh,
            batch.session_id.as_deref(),
            turn.speaker.as_deref(),
            turn.body.as_deref().unwrap_or_default(),
            None,
            &[],
        )
        .await
        .into_diagnostic()
        .wrap_err("replaying a production-path drawer")?;
        miette::ensure!(summary.duplicated, "replay inserted a duplicate drawer");
    }
    miette::ensure!(
        count_drawers(db, capture_akh).await? == 200,
        "replay changed the 200-drawer cardinality"
    );

    let embedder = SharedEmbedder::new(
        SentenceEmbedder::load(model_dir)
            .into_diagnostic()
            .wrap_err("loading the real MiniLM ONNX model")?,
    );
    let drain_started = Instant::now();
    drain_akh(db, &embedder, capture_akh).await?;
    let drain_elapsed = drain_started.elapsed();
    miette::ensure!(
        drain_elapsed < Duration::from_secs(30),
        "200-chunk embedding drain took {drain_elapsed:?}, exceeding 30 seconds"
    );
    miette::ensure!(
        count_embeddings(db, capture_akh).await? == 200,
        "200-chunk drain did not create exactly 200 embeddings"
    );

    verify_concurrent_and_retry_workers(db, &embedder, resource).await?;
    verify_longmemeval_retrieval_and_isolation(db, &embedder, dataset).await?;
    verify_public_longmemeval_smoke(db, database_url, model_dir, dataset, resource).await?;

    eprintln!(
        "Phase 42c live acceptance: captured/replayed 200 rows and embedded 200 chunks in {drain_elapsed:?}"
    );
    Ok(())
}

async fn verify_initial_migration_ladder(
    db: &DatabaseConnection,
    resource: &str,
) -> miette::Result<()> {
    Migrator::up(db, Some(1))
        .await
        .into_diagnostic()
        .wrap_err("applying migration 000001")?;

    let akh_id = insert_akh(db, &format!("{resource}_migration_ladder")).await?;
    let first_drawer_id = Uuid::new_v4();
    let second_drawer_id = Uuid::new_v4();
    let chunk_id = Uuid::new_v4();
    let now = chrono::Utc::now();
    for drawer_id in [first_drawer_id, second_drawer_id] {
        db.execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "INSERT INTO heka_drawers \
                 (id, akh_id, title, tags, metadata, created_at, updated_at) \
             VALUES ($1, $2, $3, '[]'::jsonb, '{}'::jsonb, $4, $5)",
            [
                drawer_id.into(),
                akh_id.into(),
                "same pre-000002 legacy title".to_string().into(),
                now.clone().into(),
                now.clone().into(),
            ],
        ))
        .await
        .into_diagnostic()
        .wrap_err("seeding one of two same-title migration-000001 drawers")?;
    }
    db.execute(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        "INSERT INTO heka_drawer_chunks \
             (id, drawer_id, chunk_index, text, word_count, embedded_at, created_at) \
         VALUES ($1, $2, 0, $3, 3, $4, $5)",
        [
            chunk_id.into(),
            first_drawer_id.into(),
            "pre-000003 embedding chunk".to_string().into(),
            now.clone().into(),
            now.into(),
        ],
    ))
    .await
    .into_diagnostic()
    .wrap_err("seeding a migration-000001 chunk")?;

    Migrator::up(db, Some(1))
        .await
        .into_diagnostic()
        .wrap_err("upgrading migration 000001 through 000002")?;
    let first_hash = drawer_hash(db, first_drawer_id).await?;
    let second_hash = drawer_hash(db, second_drawer_id).await?;
    miette::ensure!(
        first_hash == format!("legacy:{first_drawer_id}")
            && second_hash == format!("legacy:{second_drawer_id}")
            && first_hash != second_hash,
        "migration 000002 did not preserve both same-title drawers with collision-free hashes"
    );
    miette::ensure!(
        count_drawers(db, akh_id).await? == 2,
        "migration 000002 lost one of two same-title migration-000001 drawers"
    );

    let vector = vec_to_pgvector(&vec![0.125; crate::embedder::EMBEDDING_DIM]);
    let lower_embedding_id =
        Uuid::parse_str("00000000-0000-0000-0000-000000000001").into_diagnostic()?;
    let higher_embedding_id =
        Uuid::parse_str("00000000-0000-0000-0000-000000000002").into_diagnostic()?;
    for embedding_id in [higher_embedding_id, lower_embedding_id] {
        db.execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "INSERT INTO heka_drawer_embeddings \
                 (id, chunk_id, model_name, embedding, created_at) \
             VALUES ($1, $2, $3, $4::vector(384), $5)",
            [
                embedding_id.into(),
                chunk_id.into(),
                MODEL_NAME.to_string().into(),
                vector.clone().into(),
                chrono::Utc::now().into(),
            ],
        ))
        .await
        .into_diagnostic()
        .wrap_err("seeding a duplicate pre-000003 embedding")?;
    }

    verify_embedding_migration_writer_race(
        db,
        chunk_id,
        lower_embedding_id,
        higher_embedding_id,
        vector,
    )
    .await?;

    Migrator::up(db, Some(1))
        .await
        .into_diagnostic()
        .wrap_err("applying migration 000004 after the full upgrade ladder")?;
    Ok(())
}

async fn verify_embedding_migration_writer_race(
    db: &DatabaseConnection,
    chunk_id: Uuid,
    lower_embedding_id: Uuid,
    higher_embedding_id: Uuid,
    vector: String,
) -> miette::Result<()> {
    // SELECT FOR UPDATE is compatible with migration 000003's table lock but
    // blocks its duplicate DELETE. That creates a deterministic window after
    // the real migration owns SHARE ROW EXCLUSIVE and before its unique index
    // exists. A concurrent INSERT must remain blocked until the index commits.
    let blocker = db
        .begin()
        .await
        .into_diagnostic()
        .wrap_err("starting migration-000003 row blocker")?;
    blocker
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "SELECT id FROM heka_drawer_embeddings WHERE id = $1 FOR UPDATE",
            [higher_embedding_id.into()],
        ))
        .await
        .into_diagnostic()
        .wrap_err("locking the higher duplicate embedding")?
        .ok_or_else(|| miette::miette!("higher duplicate embedding disappeared"))?;

    let migration_db = db.clone();
    let migration = tokio::spawn(async move { Migrator::up(&migration_db, Some(1)).await });
    if let Err(error) = wait_for_embedding_migration_lock(db).await {
        let _ = blocker.rollback().await;
        let _ = tokio::time::timeout(Duration::from_secs(5), migration).await;
        return Err(error);
    }

    let writer_id = Uuid::parse_str("00000000-0000-0000-0000-000000000003").into_diagnostic()?;
    let writer_db = db.clone();
    let mut writer = tokio::spawn(async move {
        writer_db
            .execute(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                "INSERT INTO heka_drawer_embeddings \
                     (id, chunk_id, model_name, embedding, created_at) \
                 VALUES ($1, $2, $3, $4::vector(384), $5)",
                [
                    writer_id.into(),
                    chunk_id.into(),
                    MODEL_NAME.to_string().into(),
                    vector.into(),
                    chrono::Utc::now().into(),
                ],
            ))
            .await
    });
    let (writer_blocked, early_writer_result) =
        match tokio::time::timeout(Duration::from_millis(150), &mut writer).await {
            Err(_) => (true, None),
            Ok(result) => (false, Some(result)),
        };

    blocker
        .rollback()
        .await
        .into_diagnostic()
        .wrap_err("releasing migration-000003 row blocker")?;
    tokio::time::timeout(Duration::from_secs(5), migration)
        .await
        .into_diagnostic()
        .wrap_err("migration 000003 remained stuck after row release")?
        .into_diagnostic()
        .wrap_err("joining migration 000003")?
        .into_diagnostic()
        .wrap_err("migration 000003 failed")?;

    let writer_result = match early_writer_result {
        Some(result) => result,
        None => tokio::time::timeout(Duration::from_secs(5), writer)
            .await
            .into_diagnostic()
            .wrap_err("embedding writer remained stuck after migration 000003")?,
    };
    let writer_result = writer_result
        .into_diagnostic()
        .wrap_err("joining the racing embedding writer")?;
    miette::ensure!(
        writer_blocked,
        "embedding writer bypassed migration 000003's table lock"
    );
    miette::ensure!(
        writer_result.is_err(),
        "racing embedding writer inserted a duplicate after migration 000003"
    );
    miette::ensure!(
        count_chunk_embeddings(db, chunk_id).await? == 1,
        "migration 000003 did not reduce duplicate embeddings to one row"
    );
    miette::ensure!(
        embedding_id_for_chunk(db, chunk_id).await? == lower_embedding_id,
        "migration 000003 did not retain the lower deterministic embedding UUID"
    );
    miette::ensure!(
        embedding_unique_index_exists(db).await?,
        "migration 000003 completed without its per-chunk unique index"
    );
    Ok(())
}

async fn wait_for_embedding_migration_lock(db: &DatabaseConnection) -> miette::Result<()> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let row = db
            .query_one(Statement::from_string(
                DatabaseBackend::Postgres,
                "SELECT EXISTS ( \
                     SELECT 1 FROM pg_locks \
                      WHERE relation = 'heka_drawer_embeddings'::regclass \
                        AND mode = 'ShareRowExclusiveLock' \
                        AND granted \
                        AND pid <> pg_backend_pid() \
                 ) AS granted",
            ))
            .await
            .into_diagnostic()
            .wrap_err("observing migration 000003 table lock")?
            .ok_or_else(|| miette::miette!("migration lock query returned no row"))?;
        if row.try_get::<bool>("", "granted").into_diagnostic()? {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(miette::miette!(
                "migration 000003 never acquired its table lock"
            ));
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

async fn verify_legacy_migration_upgrade(
    db: &DatabaseConnection,
    database_url: &str,
    resource: &str,
) -> miette::Result<()> {
    Migrator::down(db, Some(1))
        .await
        .into_diagnostic()
        .wrap_err("rolling back repair migration for upgrade fixture")?;

    let akh_id = insert_akh(db, &format!("migration-fixture-{}", Uuid::new_v4())).await?;
    let approved_id = Uuid::new_v4();
    let approved_title = "legacy title written by the rejected 000002 backfill";
    let approved_rejected_hash = hex_sha256(approved_title);
    let unapproved_id = Uuid::new_v4();
    let unapproved_title = "legitimate same-second one-line drawer";
    let unapproved_hash = hex_sha256(unapproved_title);
    let malformed_id = Uuid::new_v4();
    let malformed_title = "post-ingest drawer that must fail legacy-shape validation";
    let malformed_hash = hex_sha256(malformed_title);
    for (drawer_id, title, body_hash) in [
        (approved_id, approved_title, approved_rejected_hash.as_str()),
        (unapproved_id, unapproved_title, unapproved_hash.as_str()),
    ] {
        db.execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "INSERT INTO heka_drawers \
                 (id, akh_id, title, tags, metadata, body_hash, created_at, updated_at) \
             SELECT $1, $2, $3, '[]'::jsonb, '{}'::jsonb, $4, \
                    to_timestamp(applied_at) + interval '500 milliseconds', \
                    to_timestamp(applied_at) + interval '500 milliseconds' \
               FROM seaql_migrations \
              WHERE version = 'm20260604_000002_drawer_ingest'",
            [
                drawer_id.into(),
                akh_id.into(),
                title.to_string().into(),
                body_hash.to_string().into(),
            ],
        ))
        .await
        .into_diagnostic()
        .wrap_err("inserting one ambiguous same-second title-hash fixture")?;
    }
    db.execute(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        "INSERT INTO heka_drawers \
             (id, akh_id, title, tags, metadata, body_hash, session_id, topics, created_at, updated_at) \
         SELECT $1, $2, $3, '[]'::jsonb, '{}'::jsonb, $4, 'post-ingest', '[]'::jsonb, \
                to_timestamp(applied_at) + interval '500 milliseconds', \
                to_timestamp(applied_at) + interval '500 milliseconds' \
           FROM seaql_migrations \
          WHERE version = 'm20260604_000002_drawer_ingest'",
        [
            malformed_id.into(),
            akh_id.into(),
            malformed_title.to_string().into(),
            malformed_hash.clone().into(),
        ],
    ))
    .await
    .into_diagnostic()
    .wrap_err("inserting malformed explicit-repair control")?;
    let inserted_xmin = drawer_xmin(db, approved_id).await?;
    // Change xmin after insertion, reproducing a real legacy drawer that was
    // edited between the rejected 000002 and repair 000004 migrations.
    db.execute(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        "UPDATE heka_drawers \
            SET metadata = '{\"updated_after_000002\":true}'::jsonb, updated_at = clock_timestamp() \
          WHERE id = $1",
        [approved_id.into()],
    ))
    .await
    .into_diagnostic()
    .wrap_err("updating same-second rejected-migration fixture")?;
    miette::ensure!(
        drawer_xmin(db, approved_id).await? != inserted_xmin,
        "same-second migration fixture did not receive a later committed update"
    );

    Migrator::up(db, Some(1))
        .await
        .into_diagnostic()
        .wrap_err("registering fail-closed legacy repair migration")?;
    miette::ensure!(
        drawer_hash(db, approved_id).await? == approved_rejected_hash,
        "migration 000004 heuristically rewrote an unapproved ambiguous row"
    );
    miette::ensure!(
        drawer_hash(db, unapproved_id).await? == unapproved_hash,
        "migration 000004 rewrote a legitimate same-second drawer"
    );

    invoke_public_legacy_repair(database_url, resource, &[approved_id, malformed_id], false)
        .await?;
    miette::ensure!(
        drawer_hash(db, approved_id).await? == approved_rejected_hash
            && drawer_hash(db, malformed_id).await? == malformed_hash,
        "failed explicit repair batch committed a partial update"
    );

    invoke_public_legacy_repair(database_url, resource, &[approved_id], true).await?;
    let repaired = format!("legacy:{approved_id}");
    miette::ensure!(
        drawer_hash(db, approved_id).await? == repaired,
        "explicit public repair command did not repair its approved drawer"
    );
    miette::ensure!(
        drawer_repair_marker(db, approved_id).await?.as_deref() == Some(LEGACY_REPAIR_AUDIT_VALUE),
        "explicit public repair command omitted its durable audit marker"
    );
    miette::ensure!(
        drawer_hash(db, unapproved_id).await? == unapproved_hash
            && drawer_repair_marker(db, unapproved_id).await?.is_none(),
        "explicit repair crossed from the approved UUID to an unapproved legitimate drawer"
    );

    // The same public operator request is a no-op after its durable marker was
    // committed, and a migration ledger down/up does not mutate either row.
    invoke_public_legacy_repair(database_url, resource, &[approved_id], true).await?;
    miette::ensure!(
        drawer_hash(db, approved_id).await? == repaired,
        "explicit repair command was not idempotent"
    );
    Migrator::down(db, Some(1))
        .await
        .into_diagnostic()
        .wrap_err("rolling back repair ledger entry")?;
    Migrator::up(db, Some(1))
        .await
        .into_diagnostic()
        .wrap_err("reapplying idempotent repair")?;
    miette::ensure!(
        drawer_hash(db, approved_id).await? == repaired,
        "repair changed on down/up retry"
    );
    miette::ensure!(
        drawer_hash(db, unapproved_id).await? == unapproved_hash,
        "migration ledger retry changed the unapproved legitimate drawer"
    );
    Ok(())
}

async fn verify_concurrent_and_retry_workers(
    db: &DatabaseConnection,
    embedder: &SharedEmbedder,
    resource: &str,
) -> miette::Result<()> {
    let akh_id = insert_akh(db, &format!("{resource}_workers")).await?;
    for index in 0..64 {
        write_drawer(
            db,
            akh_id,
            Some("concurrency"),
            Some("user"),
            &format!("concurrent embedding chunk {index}"),
            None,
            &[],
        )
        .await
        .into_diagnostic()
        .wrap_err("seeding concurrent worker fixture")?;
    }

    let barrier = Arc::new(Barrier::new(3));
    let first = spawn_worker(db, embedder, barrier.clone(), akh_id, 32);
    let second = spawn_worker(db, embedder, barrier.clone(), akh_id, 32);
    barrier.wait().await;
    let first = first
        .await
        .into_diagnostic()
        .wrap_err("joining first concurrent worker")?
        .into_diagnostic()?;
    let second = second
        .await
        .into_diagnostic()
        .wrap_err("joining second concurrent worker")?
        .into_diagnostic()?;
    miette::ensure!(
        first.chunks_processed + second.chunks_processed == 64,
        "concurrent workers did not claim exactly 64 distinct chunks"
    );
    miette::ensure!(
        count_embeddings(db, akh_id).await? == 64,
        "concurrent workers created duplicate or missing embeddings"
    );
    miette::ensure!(
        count_pending_for_akh(db, akh_id).await.into_diagnostic()? == 0,
        "concurrent workers left pending claims"
    );

    db.execute(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        "UPDATE heka_drawer_chunks AS c SET embedded_at = NULL \
           FROM heka_drawers AS d \
          WHERE c.drawer_id = d.id AND d.akh_id = $1",
        [akh_id.into()],
    ))
    .await
    .into_diagnostic()
    .wrap_err("simulating an interrupted pre-marker retry")?;
    let retry_created = drain_akh(db, embedder, akh_id).await?;
    miette::ensure!(retry_created == 0, "retry duplicated existing embeddings");
    miette::ensure!(
        count_embeddings(db, akh_id).await? == 64,
        "retry changed embedding cardinality"
    );

    write_drawer(
        db,
        akh_id,
        Some("failure-recovery"),
        Some("user"),
        "chunk whose first embedding transaction must fail",
        None,
        &[],
    )
    .await
    .into_diagnostic()
    .wrap_err("seeding transaction-failure fixture")?;
    db.execute_unprepared(
        "ALTER TABLE heka_drawer_embeddings \
         ADD CONSTRAINT phase42c_force_embedding_failure \
         CHECK (model_name <> 'force-failure')",
    )
    .await
    .into_diagnostic()
    .wrap_err("installing transaction-failure fixture")?;
    let failed = embed_pending_batch_for_akh(db, embedder, 1, "force-failure", akh_id).await;
    miette::ensure!(
        failed.is_err(),
        "forced embedding transaction unexpectedly committed"
    );
    miette::ensure!(
        count_pending_for_akh(db, akh_id).await.into_diagnostic()? == 1,
        "failed transaction left a stuck claim or marker"
    );
    db.execute_unprepared(
        "ALTER TABLE heka_drawer_embeddings \
         DROP CONSTRAINT phase42c_force_embedding_failure",
    )
    .await
    .into_diagnostic()
    .wrap_err("removing transaction-failure fixture")?;
    drain_akh(db, embedder, akh_id).await?;
    miette::ensure!(
        count_pending_for_akh(db, akh_id).await.into_diagnostic()? == 0,
        "retry did not recover the failed chunk"
    );
    miette::ensure!(
        count_embeddings(db, akh_id).await? == 65,
        "failure recovery produced the wrong embedding cardinality"
    );
    Ok(())
}

fn spawn_worker(
    db: &DatabaseConnection,
    embedder: &SharedEmbedder,
    barrier: Arc<Barrier>,
    akh_id: Uuid,
    batch_size: usize,
) -> tokio::task::JoinHandle<crate::HekaResult<crate::embed_worker::EmbedStats>> {
    let db = db.clone();
    let embedder = embedder.clone();
    tokio::spawn(async move {
        barrier.wait().await;
        embed_pending_batch_for_akh(&db, &embedder, batch_size, MODEL_NAME, akh_id).await
    })
}

async fn verify_longmemeval_retrieval_and_isolation(
    db: &DatabaseConnection,
    embedder: &SharedEmbedder,
    official_dataset: &Path,
) -> miette::Result<()> {
    let full = retrieval_case(
        "repeatable-question",
        "Where is the cobalt key?",
        &[
            ("evidence", "The cobalt key is inside the blue cabinet."),
            (
                "removed-distractor",
                "The garden meeting begins after lunch.",
            ),
        ],
        &["evidence"],
    );
    validate_case(&full)?;
    let full_result =
        evaluate_case_isolated(db, embedder, 32, MODEL_NAME, Uuid::new_v4(), &full).await?;
    miette::ensure!(
        full_result.recalled_all,
        "real retrieval produced zero recall_all@5"
    );
    miette::ensure!(
        full_result
            .retrieved_session_ids
            .iter()
            .any(|id| id == "removed-distractor"),
        "full fixture did not exercise the removable distractor"
    );

    let revised = retrieval_case(
        "repeatable-question",
        "Where is the cobalt key?",
        &[("evidence", "The cobalt key is inside the blue cabinet.")],
        &["evidence"],
    );
    validate_case(&revised)?;
    let revised_result =
        evaluate_case_isolated(db, embedder, 32, MODEL_NAME, Uuid::new_v4(), &revised).await?;
    miette::ensure!(
        revised_result.recalled_all,
        "revised corpus lost its evidence session"
    );
    miette::ensure!(
        revised_result
            .retrieved_session_ids
            .iter()
            .all(|id| id != "removed-distractor"),
        "revised corpus retained a distractor from the previous run"
    );

    let identical = retrieval_case(
        "identical-body-question",
        "Which sessions contain the shared statement?",
        &[
            ("same-body-a", "The shared evidence body is identical."),
            ("same-body-b", "The shared evidence body is identical."),
        ],
        &["same-body-a", "same-body-b"],
    );
    validate_case(&identical)?;
    let identical_result =
        evaluate_case_isolated(db, embedder, 32, MODEL_NAME, Uuid::new_v4(), &identical).await?;
    let retrieved: HashSet<&str> = identical_result
        .retrieved_session_ids
        .iter()
        .map(String::as_str)
        .collect();
    miette::ensure!(
        identical_result.recalled_all
            && retrieved.contains("same-body-a")
            && retrieved.contains("same-body-b"),
        "body-hash dedupe collapsed identical evidence from distinct sessions"
    );

    let metrics = aggregate_metrics(
        &[full_result, revised_result, identical_result],
        RetrievalParams {
            mode: "raw".to_string(),
            top_k: TOP_K.to_string(),
            top_k_explicit: None,
            auto_top_k_cap: None,
        },
    )?;
    match metrics.scores {
        EvalScores::EvidenceSessionRetrieval {
            recall_all_at_5, ..
        } => {
            miette::ensure!(recall_all_at_5 > 0.0, "published recall_all@5 was zero");
        }
        _ => return Err(miette::miette!("LongMemEval emitted the wrong metric kind")),
    }

    let official_cases = load_dataset(official_dataset)?;
    validate_dataset(&official_cases)?;
    let official = official_cases
        .iter()
        .find(|case| !is_abstention(case))
        .ok_or_else(|| miette::miette!("official dataset contains no evaluable case"))?;
    let official_result =
        evaluate_case_isolated(db, embedder, 32, MODEL_NAME, Uuid::new_v4(), official).await?;
    miette::ensure!(
        official_result.recalled_all,
        "official LongMemEval oracle case produced zero recall_all@5"
    );
    miette::ensure!(
        count_longmemeval_akhs(db).await? == 0,
        "normal evaluation left an isolated corpus behind"
    );
    Ok(())
}

async fn invoke_public_legacy_repair(
    database_url: &str,
    resource: &str,
    drawer_ids: &[Uuid],
    expect_success: bool,
) -> miette::Result<()> {
    miette::ensure!(!drawer_ids.is_empty(), "repair invocation needs drawer IDs");
    let provided_bin = validated_hekad_bin()?;
    let public = TempPublicEval::new(resource)?;
    let config = HekaConfig {
        database: DatabaseConfig {
            url: database_url.to_string(),
            max_connections: 4,
        },
        server: ServerConfig::default(),
        embedding: EmbeddingConfig::default(),
        eval: EvalConfig::default(),
    };
    write_private_config(&public.config, &config)?;

    let mut command = tokio::process::Command::new(&provided_bin);
    command.arg("repair-legacy-title-hashes");
    for drawer_id in drawer_ids {
        command.arg("--drawer-id").arg(drawer_id.to_string());
    }
    command
        .arg("--config-path")
        .arg(&public.config)
        .env_remove("DATABASE_URL")
        .kill_on_drop(true);
    let output = tokio::time::timeout(Duration::from_secs(30), command.output())
        .await
        .map_err(|_| miette::miette!("public legacy repair command exceeded 30 seconds"))?
        .into_diagnostic()
        .wrap_err("executing public legacy repair command")?;
    miette::ensure!(
        output.status.success() == expect_success,
        "public legacy repair command returned unexpected status {} (output withheld to protect configuration secrets)",
        output.status
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    miette::ensure!(
        !stdout.contains(database_url) && !stderr.contains(database_url),
        "public legacy repair command emitted the raw database URL"
    );
    Ok(())
}

async fn verify_public_longmemeval_smoke(
    db: &DatabaseConnection,
    database_url: &str,
    model_dir: &Path,
    official_dataset: &Path,
    resource: &str,
) -> miette::Result<()> {
    let provided_bin = validated_hekad_bin()?;

    let public = TempPublicEval::new(resource)?;
    let model_name = model_dir
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .ok_or_else(|| miette::miette!("model directory has no UTF-8 final component"))?;
    let model_base = model_dir
        .parent()
        .and_then(Path::to_str)
        .ok_or_else(|| miette::miette!("model directory has no UTF-8 parent"))?;
    let config = HekaConfig {
        database: DatabaseConfig {
            url: database_url.to_string(),
            max_connections: 8,
        },
        server: ServerConfig::default(),
        embedding: EmbeddingConfig {
            model: model_name.to_string(),
            batch_size: 32,
            poll_interval_secs: 1,
            model_dir: Some(model_base.to_string()),
        },
        eval: EvalConfig::default(),
    };
    write_private_config(&public.config, &config)?;

    let mut command = tokio::process::Command::new(&provided_bin);
    command
        .arg("eval")
        .arg("longmemeval")
        .arg("--mode")
        .arg("raw")
        .arg("--sessions")
        .arg("1")
        .arg("--data-dir")
        .arg(official_dataset)
        .arg("--output")
        .arg(&public.output)
        .env("XDG_CONFIG_HOME", &public.xdg_config_home)
        .env_remove("DATABASE_URL")
        .kill_on_drop(true);
    let output = tokio::time::timeout(Duration::from_secs(120), command.output())
        .await
        .map_err(|_| miette::miette!("public hekad LongMemEval smoke exceeded 120 seconds"))?
        .into_diagnostic()
        .wrap_err("executing public hekad LongMemEval smoke")?;
    miette::ensure!(
        output.status.success(),
        "public hekad LongMemEval smoke failed with status {} (output withheld to protect configuration secrets)",
        output.status
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    miette::ensure!(
        !stdout.contains(database_url) && !stderr.contains(database_url),
        "public LongMemEval smoke emitted the raw database URL"
    );

    let report_path = public.output.join("longmemeval_report.json");
    let report: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&report_path)
            .into_diagnostic()
            .wrap_err("reading public LongMemEval report")?,
    )
    .into_diagnostic()
    .wrap_err("parsing public LongMemEval report")?;
    miette::ensure!(
        report
            .pointer("/metrics/scores/kind")
            .and_then(|value| value.as_str())
            == Some("evidence_session_retrieval"),
        "public LongMemEval smoke report emitted the wrong metric kind"
    );
    let recall_all = report
        .pointer("/metrics/scores/recall_all@5")
        .and_then(|value| value.as_f64())
        .ok_or_else(|| miette::miette!("public report omitted recall_all@5"))?;
    miette::ensure!(
        recall_all > 0.0,
        "public LongMemEval smoke report produced zero recall_all@5"
    );
    miette::ensure!(
        count_longmemeval_akhs(db).await? == 0,
        "public LongMemEval smoke left an isolated corpus behind"
    );
    Ok(())
}

fn validated_hekad_bin() -> miette::Result<PathBuf> {
    let provided_bin = PathBuf::from(
        std::env::var_os(HEKAD_BIN_ENV)
            .ok_or_else(|| miette::miette!("{HEKAD_BIN_ENV} is required"))?,
    )
    .canonicalize()
    .into_diagnostic()
    .wrap_err("canonicalizing the public hekad acceptance binary")?;
    let expected_bin = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/debug/hekad")
        .canonicalize()
        .into_diagnostic()
        .wrap_err("locating the workspace-built hekad acceptance binary")?;
    miette::ensure!(
        provided_bin == expected_bin,
        "public command binary is not the exact workspace target/debug/hekad"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = provided_bin
            .metadata()
            .into_diagnostic()?
            .permissions()
            .mode();
        miette::ensure!(mode & 0o111 != 0, "public hekad binary is not executable");
    }
    Ok(provided_bin)
}

fn write_private_config(path: &Path, config: &HekaConfig) -> miette::Result<()> {
    let config_text = toml::to_string(config)
        .into_diagnostic()
        .wrap_err("serializing disposable public-command config")?;
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(path)
        .into_diagnostic()
        .wrap_err("creating private disposable public-command config")?;
    file.write_all(config_text.as_bytes())
        .into_diagnostic()
        .wrap_err("writing private disposable public-command config")?;
    file.sync_all()
        .into_diagnostic()
        .wrap_err("syncing private disposable public-command config")?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        miette::ensure!(
            file.metadata().into_diagnostic()?.permissions().mode() & 0o077 == 0,
            "disposable public-command config permissions are not private"
        );
    }
    Ok(())
}

fn retrieval_case(
    question_id: &str,
    question: &str,
    sessions: &[(&str, &str)],
    answers: &[&str],
) -> LongMemEvalCase {
    LongMemEvalCase {
        question_id: question_id.to_string(),
        question_type: "single-session-user".to_string(),
        question: question.to_string(),
        haystack_session_ids: sessions.iter().map(|(id, _)| (*id).to_string()).collect(),
        haystack_dates: Vec::new(),
        haystack_sessions: sessions
            .iter()
            .map(|(_, body)| {
                vec![LongMemEvalTurn {
                    role: "user".to_string(),
                    content: (*body).to_string(),
                    has_answer: true,
                }]
            })
            .collect(),
        answer_session_ids: answers.iter().map(|id| (*id).to_string()).collect(),
    }
}

async fn drain_akh(
    db: &DatabaseConnection,
    embedder: &SharedEmbedder,
    akh_id: Uuid,
) -> miette::Result<usize> {
    let mut created = 0usize;
    for _ in 0..10_000 {
        let stats = embed_pending_batch_for_akh(db, embedder, 32, MODEL_NAME, akh_id)
            .await
            .into_diagnostic()
            .wrap_err("draining disposable embedding corpus")?;
        created += stats.embeddings_created;
        if stats.chunks_processed == 0 {
            let pending = count_pending_for_akh(db, akh_id).await.into_diagnostic()?;
            miette::ensure!(
                pending == 0,
                "embedding drain stalled with {pending} pending chunks"
            );
            return Ok(created);
        }
    }
    Err(miette::miette!(
        "embedding drain exceeded its iteration limit"
    ))
}

fn realistic_two_hundred_turn_capture(
    resource: &str,
) -> miette::Result<crate::capture::CaptureBatch> {
    let tree = TempCaptureTree::new(resource)?;
    let records = (0..200)
        .map(|index| {
            let role = if index % 2 == 0 { "user" } else { "assistant" };
            let cwd = match index % 3 {
                0 => &tree.cwd,
                1 => &tree.nested_cwd,
                _ => &tree.alternate_cwd,
            };
            serde_json::json!({
                "type": role,
                "sessionId": tree.session_id,
                "cwd": cwd,
                "isSidechain": false,
                "message": {
                    "role": role,
                    "content": [{"type": "text", "text": format!("unique captured turn {index}")}]
                }
            })
            .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(&tree.transcript, records)
        .into_diagnostic()
        .wrap_err("writing disposable Stop transcript")?;
    let payload = serde_json::json!({
        "session_id": tree.session_id,
        "transcript_path": tree.transcript,
        "cwd": tree.nested_cwd,
        "hook_event_name": "Stop"
    })
    .to_string();
    parse_capture_input_with_policy(&payload, &CapturePolicy::new(tree.root.clone()))
        .into_diagnostic()
        .wrap_err("parsing realistic Stop transcript")
}

struct TempCaptureTree {
    base: PathBuf,
    root: PathBuf,
    cwd: PathBuf,
    nested_cwd: PathBuf,
    alternate_cwd: PathBuf,
    session_id: String,
    transcript: PathBuf,
}

impl TempCaptureTree {
    fn new(resource: &str) -> miette::Result<Self> {
        let base = std::env::temp_dir().join(format!("{resource}-capture-{}", Uuid::new_v4()));
        miette::ensure!(!base.exists(), "disposable capture path already exists");
        let root = base.join("claude-projects");
        let cwd = base.join("workspace");
        let nested_cwd = cwd.join("nested/source");
        let alternate_cwd = base.join("alternate-worktree");
        std::fs::create_dir_all(&root).into_diagnostic()?;
        std::fs::create_dir_all(&nested_cwd).into_diagnostic()?;
        std::fs::create_dir_all(&alternate_cwd).into_diagnostic()?;
        let cwd = cwd.canonicalize().into_diagnostic()?;
        let nested_cwd = nested_cwd.canonicalize().into_diagnostic()?;
        let alternate_cwd = alternate_cwd.canonicalize().into_diagnostic()?;
        let project = root.join(claude_project_component(&cwd));
        std::fs::create_dir_all(&project).into_diagnostic()?;
        let session_id = Uuid::new_v4().to_string();
        let transcript = project.join(format!("{session_id}.jsonl"));
        Ok(Self {
            base,
            root,
            cwd,
            nested_cwd,
            alternate_cwd,
            session_id,
            transcript,
        })
    }
}

fn claude_project_component(project_root: &Path) -> String {
    project_root.to_string_lossy().replace(['/', '\\'], "-")
}

impl Drop for TempCaptureTree {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.base);
    }
}

struct TempPublicEval {
    base: PathBuf,
    xdg_config_home: PathBuf,
    config: PathBuf,
    output: PathBuf,
}

impl TempPublicEval {
    fn new(resource: &str) -> miette::Result<Self> {
        validate_resource_id(resource)?;
        let base = std::env::temp_dir().join(format!("{resource}-public-eval-{}", Uuid::new_v4()));
        miette::ensure!(!base.exists(), "disposable public-eval path already exists");
        let xdg_config_home = base.join("xdg-config");
        let config_dir = xdg_config_home.join("heka");
        let output = base.join("reports");
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            std::fs::DirBuilder::new()
                .mode(0o700)
                .create(&base)
                .into_diagnostic()?;
        }
        #[cfg(not(unix))]
        std::fs::create_dir(&base).into_diagnostic()?;
        std::fs::create_dir_all(&config_dir).into_diagnostic()?;
        std::fs::create_dir_all(&output).into_diagnostic()?;
        let config = config_dir.join("heka.toml");
        Ok(Self {
            base,
            xdg_config_home,
            config,
            output,
        })
    }
}

impl Drop for TempPublicEval {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.base);
    }
}

async fn insert_akh(db: &DatabaseConnection, name: &str) -> miette::Result<Uuid> {
    let id = Uuid::new_v4();
    let now = chrono::Utc::now();
    db.execute(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        "INSERT INTO heka_akhs \
             (id, name, description, personality, avatar_url, created_at, updated_at) \
         VALUES ($1, $2, NULL, NULL, NULL, $3, $4)",
        [
            id.into(),
            name.to_string().into(),
            now.clone().into(),
            now.into(),
        ],
    ))
    .await
    .into_diagnostic()
    .wrap_err("registering disposable akh")?;
    Ok(id)
}

async fn count_drawers(db: &DatabaseConnection, akh_id: Uuid) -> miette::Result<i64> {
    scalar_count(
        db,
        "SELECT COUNT(*)::bigint AS count FROM heka_drawers WHERE akh_id = $1",
        akh_id,
    )
    .await
}

async fn count_embeddings(db: &DatabaseConnection, akh_id: Uuid) -> miette::Result<i64> {
    scalar_count(
        db,
        "SELECT COUNT(*)::bigint AS count \
           FROM heka_drawer_embeddings AS e \
           JOIN heka_drawer_chunks AS c ON c.id = e.chunk_id \
           JOIN heka_drawers AS d ON d.id = c.drawer_id \
          WHERE d.akh_id = $1",
        akh_id,
    )
    .await
}

async fn count_chunk_embeddings(db: &DatabaseConnection, chunk_id: Uuid) -> miette::Result<i64> {
    let row = db
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "SELECT COUNT(*)::bigint AS count \
               FROM heka_drawer_embeddings \
              WHERE chunk_id = $1",
            [chunk_id.into()],
        ))
        .await
        .into_diagnostic()?
        .ok_or_else(|| miette::miette!("chunk embedding count returned no row"))?;
    row.try_get("", "count").into_diagnostic()
}

async fn embedding_id_for_chunk(db: &DatabaseConnection, chunk_id: Uuid) -> miette::Result<Uuid> {
    let row = db
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "SELECT id FROM heka_drawer_embeddings WHERE chunk_id = $1",
            [chunk_id.into()],
        ))
        .await
        .into_diagnostic()?
        .ok_or_else(|| miette::miette!("retained embedding row disappeared"))?;
    row.try_get("", "id").into_diagnostic()
}

async fn embedding_unique_index_exists(db: &DatabaseConnection) -> miette::Result<bool> {
    let row = db
        .query_one(Statement::from_string(
            DatabaseBackend::Postgres,
            "SELECT EXISTS ( \
                 SELECT 1 FROM pg_indexes \
                  WHERE schemaname = current_schema() \
                    AND tablename = 'heka_drawer_embeddings' \
                    AND indexname = 'uq_heka_embeddings_chunk' \
             ) AS present",
        ))
        .await
        .into_diagnostic()?
        .ok_or_else(|| miette::miette!("unique-index lookup returned no row"))?;
    row.try_get("", "present").into_diagnostic()
}

async fn scalar_count(db: &DatabaseConnection, sql: &str, akh_id: Uuid) -> miette::Result<i64> {
    let row = db
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            sql,
            [akh_id.into()],
        ))
        .await
        .into_diagnostic()?
        .ok_or_else(|| miette::miette!("count query returned no row"))?;
    row.try_get("", "count").into_diagnostic()
}

async fn count_longmemeval_akhs(db: &DatabaseConnection) -> miette::Result<i64> {
    let row = db
        .query_one(Statement::from_string(
            DatabaseBackend::Postgres,
            "SELECT COUNT(*)::bigint AS count FROM heka_akhs WHERE name LIKE 'longmemeval:%'",
        ))
        .await
        .into_diagnostic()?
        .ok_or_else(|| miette::miette!("LongMemEval cleanup count returned no row"))?;
    row.try_get("", "count").into_diagnostic()
}

async fn drawer_hash(db: &DatabaseConnection, drawer_id: Uuid) -> miette::Result<String> {
    let row = db
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "SELECT body_hash FROM heka_drawers WHERE id = $1",
            [drawer_id.into()],
        ))
        .await
        .into_diagnostic()?
        .ok_or_else(|| miette::miette!("migration fixture drawer disappeared"))?;
    row.try_get("", "body_hash").into_diagnostic()
}

async fn drawer_repair_marker(
    db: &DatabaseConnection,
    drawer_id: Uuid,
) -> miette::Result<Option<String>> {
    let row = db
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "SELECT metadata ->> $1 AS marker FROM heka_drawers WHERE id = $2",
            [LEGACY_REPAIR_AUDIT_KEY.into(), drawer_id.into()],
        ))
        .await
        .into_diagnostic()?
        .ok_or_else(|| miette::miette!("migration fixture drawer disappeared"))?;
    row.try_get("", "marker").into_diagnostic()
}

async fn drawer_xmin(db: &DatabaseConnection, drawer_id: Uuid) -> miette::Result<String> {
    let row = db
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "SELECT xmin::text AS xmin FROM heka_drawers WHERE id = $1",
            [drawer_id.into()],
        ))
        .await
        .into_diagnostic()?
        .ok_or_else(|| miette::miette!("migration fixture drawer disappeared"))?;
    row.try_get("", "xmin").into_diagnostic()
}

fn hex_sha256(value: &str) -> String {
    let digest = Sha256::digest(value.as_bytes());
    let mut hex = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

async fn connect_without_url_logging(
    url: &str,
    max_connections: u32,
    label: &str,
) -> miette::Result<DatabaseConnection> {
    connect_database(url, max_connections)
        .await
        .into_diagnostic()
        .wrap_err_with(|| format!("connecting to disposable {label} failed"))
}

async fn drop_database(admin: &DatabaseConnection, resource: &str) -> miette::Result<()> {
    validate_resource_id(resource)?;
    admin
        .execute_unprepared(&format!(
            "DROP DATABASE IF EXISTS \"{resource}\" WITH (FORCE)"
        ))
        .await
        .into_diagnostic()
        .wrap_err("dropping exact disposable Phase 42c database")?;
    Ok(())
}

fn validate_resource_id(resource: &str) -> miette::Result<()> {
    let suffix = resource
        .strip_prefix("heka_phase42c_")
        .ok_or_else(|| miette::miette!("disposable resource must start with heka_phase42c_"))?;
    miette::ensure!(
        (12..=32).contains(&suffix.len())
            && suffix
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "disposable resource suffix must be 12-32 lowercase hexadecimal characters"
    );
    miette::ensure!(
        resource.len() <= 63,
        "disposable database identifier is too long"
    );
    Ok(())
}

fn disposable_database_url(admin_url: &str, resource: &str) -> miette::Result<String> {
    validate_resource_id(resource)?;
    miette::ensure!(
        !admin_url.chars().any(char::is_control)
            && !admin_url.contains('%')
            && !admin_url.contains('#'),
        "disposable admin URL contains forbidden characters"
    );
    let (base, query) = admin_url
        .split_once('?')
        .map_or((admin_url, None), |(base, query)| (base, Some(query)));
    if let Some(query) = query {
        miette::ensure!(
            query == "sslmode=disable",
            "disposable admin URL query must be exactly sslmode=disable; credential-bearing and mutable connection parameters are forbidden"
        );
    }
    let remainder = base
        .strip_prefix("postgres://")
        .or_else(|| base.strip_prefix("postgresql://"))
        .ok_or_else(|| miette::miette!("disposable database URL must use PostgreSQL"))?;
    let (authority, database) = remainder
        .split_once('/')
        .ok_or_else(|| miette::miette!("disposable admin URL has no database path"))?;
    miette::ensure!(
        database == "postgres",
        "disposable admin URL must target postgres"
    );
    let (userinfo, host_port) = authority
        .rsplit_once('@')
        .ok_or_else(|| miette::miette!("disposable admin URL requires scoped credentials"))?;
    let (user, password) = userinfo
        .split_once(':')
        .ok_or_else(|| miette::miette!("disposable admin URL requires scoped credentials"))?;
    miette::ensure!(
        user == "heka"
            && password.len() == 64
            && password
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "disposable admin URL credentials do not match the scoped contract"
    );
    let port = if let Some(port) = host_port.strip_prefix("[::1]:") {
        port
    } else {
        let (host, port) = host_port
            .rsplit_once(':')
            .ok_or_else(|| miette::miette!("disposable database server requires a port"))?;
        miette::ensure!(
            matches!(host, "127.0.0.1" | "localhost"),
            "disposable database server must be loopback-only"
        );
        port
    };
    miette::ensure!(
        !port.is_empty() && port.bytes().all(|byte| byte.is_ascii_digit()),
        "disposable database server requires a valid port"
    );
    let port = port
        .parse::<u16>()
        .into_diagnostic()
        .wrap_err("disposable database server requires a valid port")?;
    miette::ensure!(
        port != 0,
        "disposable database server requires a valid port"
    );

    let prefix_len = base.len() - database.len();
    let mut url = format!("{}{resource}", &base[..prefix_len]);
    if let Some(query) = query {
        url.push('?');
        url.push_str(query);
    }
    Ok(url)
}

fn validate_model_dir(model_dir: &Path) -> miette::Result<()> {
    let model = model_dir.join("model.onnx");
    let tokenizer = model_dir.join("tokenizer.json");
    miette::ensure!(model.is_file(), "disposable model.onnx is unavailable");
    miette::ensure!(
        tokenizer.is_file(),
        "disposable tokenizer.json is unavailable"
    );
    Ok(())
}

#[test]
fn disposable_url_accepts_only_scoped_scram_credentials() {
    let resource = "heka_phase42c_0123456789ab";
    let password = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    for host in ["127.0.0.1", "localhost", "[::1]"] {
        let url = format!("postgres://heka:{password}@{host}:5432/postgres?sslmode=disable");
        assert!(disposable_database_url(&url, resource).is_ok());
    }

    let invalid = [
        "postgres://heka@127.0.0.1:5432/postgres?sslmode=disable".to_string(),
        "postgres://heka:@127.0.0.1:5432/postgres?sslmode=disable".to_string(),
        "postgres://heka:secret@127.0.0.1:5432/postgres?sslmode=disable".to_string(),
        format!(
            "postgres://heka:{}@127.0.0.1:5432/postgres?sslmode=disable",
            password.to_ascii_uppercase()
        ),
        format!(
            "postgres://heka:{}@127.0.0.1:5432/postgres?sslmode=disable",
            &password[..63]
        ),
        format!("postgres://heka:{password}0@127.0.0.1:5432/postgres?sslmode=disable"),
        format!("postgres://heka:{password}@extra@127.0.0.1:5432/postgres?sslmode=disable"),
        format!("postgres://root:{password}@127.0.0.1:5432/postgres?sslmode=disable"),
        format!("postgres://heka:{password}@127.0.0.1:0/postgres?sslmode=disable"),
        format!("postgres://heka:{password}@127.0.0.1:65536/postgres?sslmode=disable"),
        format!("postgres://heka:{password}@127.0.0.2:5432/postgres?sslmode=disable"),
        format!("postgres://heka:{password}@localhost.example:5432/postgres?sslmode=disable"),
        format!("postgres://heka:{password}@127.0.0.1/postgres?sslmode=disable"),
        format!("postgres://heka:{password}@127.0.0.1:5432/other?sslmode=disable"),
        format!("postgres://heka:{password}@127.0.0.1:5432/postgres?password=secret"),
        format!("postgres://heka:{password}@127.0.0.1:5432/postgres?sslmode=disable&token=secret"),
        format!("postgres://heka:{password}@127.0.0.1:5432/postgres?sslmode=disable#fragment"),
        format!("postgres://heka:%30{password}@127.0.0.1:5432/postgres?sslmode=disable"),
        format!("postgres://heka:{password}@127.0.0.1:5432/postgres\n?sslmode=disable"),
    ];
    for url in invalid {
        assert!(disposable_database_url(&url, resource).is_err());
    }
}

#[cfg(unix)]
#[test]
fn disposable_public_config_is_created_private() {
    use std::os::unix::fs::PermissionsExt;

    let public = TempPublicEval::new("heka_phase42c_0123456789ab").unwrap();
    let config = HekaConfig {
        database: DatabaseConfig {
            url: "postgres://heka:secret@127.0.0.1:5432/disposable".to_string(),
            max_connections: 1,
        },
        server: ServerConfig::default(),
        embedding: EmbeddingConfig::default(),
        eval: EvalConfig::default(),
    };
    write_private_config(&public.config, &config).unwrap();
    assert_eq!(
        public.config.metadata().unwrap().permissions().mode() & 0o077,
        0
    );
}
