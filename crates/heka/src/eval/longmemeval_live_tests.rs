//! Explicit, disposable Phase 42c runtime acceptance.
//!
//! This module is ignored by the ordinary test suite. The companion script
//! provisions a loopback-only pgvector instance and model, then opts in with
//! guarded environment variables. All SQL is executed through SeaORM; no
//! database shell is used.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use miette::{IntoDiagnostic, WrapErr};
use sea_orm::{ConnectionTrait, DatabaseBackend, DatabaseConnection, Statement, TransactionTrait};
use sea_orm_migration::MigratorTrait;
use sha2::{Digest, Sha256};
use tokio::sync::Barrier;
use uuid::Uuid;

use super::*;
use crate::capture::{CapturePolicy, parse_capture_input_with_policy};
use crate::config::connect_database;
use crate::embed_worker::{count_pending_for_akh, embed_pending_batch_for_akh};
use crate::embedder::{SentenceEmbedder, SharedEmbedder};
use crate::ingest::write_drawer;
use crate::migration::Migrator;

const LIVE_OPT_IN: &str = "HEKA_PHASE42C_LIVE";
const ADMIN_URL_ENV: &str = "HEKA_PHASE42C_ADMIN_DATABASE_URL";
const RESOURCE_ENV: &str = "HEKA_PHASE42C_RESOURCE_ID";
const MODEL_DIR_ENV: &str = "HEKA_PHASE42C_MODEL_DIR";
const DATASET_ENV: &str = "HEKA_PHASE42C_LONGMEMEVAL_DATASET";
const MODEL_NAME: &str = "all-MiniLM-L6-v2";

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
            let outcome = run_acceptance(&db, &model_dir, &dataset, &resource).await;
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
    model_dir: &Path,
    dataset: &Path,
    resource: &str,
) -> miette::Result<()> {
    Migrator::up(db, None)
        .await
        .into_diagnostic()
        .wrap_err("applying Heka migrations")?;

    verify_legacy_migration_upgrade(db).await?;
    verify_migration_writer_lock(db).await?;

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

    let embedder = Arc::new(Mutex::new(
        SentenceEmbedder::load(model_dir)
            .into_diagnostic()
            .wrap_err("loading the real MiniLM ONNX model")?,
    ));
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

    eprintln!(
        "Phase 42c live acceptance: captured/replayed 200 rows and embedded 200 chunks in {drain_elapsed:?}"
    );
    Ok(())
}

async fn verify_legacy_migration_upgrade(db: &DatabaseConnection) -> miette::Result<()> {
    Migrator::down(db, Some(1))
        .await
        .into_diagnostic()
        .wrap_err("rolling back repair migration for upgrade fixture")?;

    let akh_id = insert_akh(db, &format!("migration-fixture-{}", Uuid::new_v4())).await?;
    let drawer_id = Uuid::new_v4();
    let title = "legacy title written before Phase 42c";
    let rejected_hash = hex_sha256(title);
    db.execute(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        "INSERT INTO heka_drawers \
             (id, akh_id, title, tags, metadata, body_hash, created_at, updated_at) \
         SELECT $1, $2, $3, '[]'::jsonb, '{}'::jsonb, $4, \
                to_timestamp(applied_at - 1), to_timestamp(applied_at - 1) \
           FROM seaql_migrations \
          WHERE version = 'm20260604_000002_drawer_ingest'",
        [
            drawer_id.into(),
            akh_id.into(),
            title.to_string().into(),
            rejected_hash.into(),
        ],
    ))
    .await
    .into_diagnostic()
    .wrap_err("inserting rejected-migration fixture")?;

    Migrator::up(db, Some(1))
        .await
        .into_diagnostic()
        .wrap_err("upgrading rejected legacy hash")?;
    let repaired = drawer_hash(db, drawer_id).await?;
    miette::ensure!(
        repaired == format!("legacy:{drawer_id}"),
        "legacy hash was not repaired"
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
        drawer_hash(db, drawer_id).await? == repaired,
        "repair changed on down/up retry"
    );
    Ok(())
}

async fn verify_migration_writer_lock(db: &DatabaseConnection) -> miette::Result<()> {
    Migrator::down(db, Some(1))
        .await
        .into_diagnostic()
        .wrap_err("rolling back repair migration for live-writer fixture")?;
    let akh_id = insert_akh(db, &format!("lock-fixture-{}", Uuid::new_v4())).await?;
    let candidate_id = Uuid::new_v4();
    let candidate_title = "legacy title held during the real repair migration";
    db.execute(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        "INSERT INTO heka_drawers \
             (id, akh_id, title, tags, metadata, body_hash, created_at, updated_at) \
         SELECT $1, $2, $3, '[]'::jsonb, '{}'::jsonb, $4, \
                to_timestamp(applied_at - 1), to_timestamp(applied_at - 1) \
           FROM seaql_migrations \
          WHERE version = 'm20260604_000002_drawer_ingest'",
        [
            candidate_id.into(),
            akh_id.into(),
            candidate_title.to_string().into(),
            hex_sha256(candidate_title).into(),
        ],
    ))
    .await
    .into_diagnostic()
    .wrap_err("inserting live-writer migration candidate")?;

    // A row lock is compatible with the migration's table lock but holds the
    // candidate UPDATE. This gives the harness a deterministic window in which
    // the real migration owns SHARE ROW EXCLUSIVE and an application writer
    // must wait behind it.
    let blocker = db
        .begin()
        .await
        .into_diagnostic()
        .wrap_err("starting migration lock fixture")?;
    blocker
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "SELECT id FROM heka_drawers WHERE id = $1 FOR UPDATE",
            [candidate_id.into()],
        ))
        .await
        .into_diagnostic()
        .wrap_err("locking migration candidate row")?
        .ok_or_else(|| miette::miette!("migration candidate disappeared"))?;

    let migration_db = db.clone();
    let migration = tokio::spawn(async move { Migrator::up(&migration_db, Some(1)).await });
    if let Err(error) = wait_for_repair_table_lock(db).await {
        let _ = blocker.rollback().await;
        let _ = tokio::time::timeout(Duration::from_secs(5), migration).await;
        return Err(error);
    }

    let writer_db = db.clone();
    let mut writer = tokio::spawn(async move {
        write_drawer(
            &writer_db,
            akh_id,
            Some("live-writer"),
            Some("user"),
            "writer must wait for migration repair",
            None,
            &[],
        )
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
        .wrap_err("releasing migration candidate row")?;
    tokio::time::timeout(Duration::from_secs(5), migration)
        .await
        .into_diagnostic()
        .wrap_err("real repair migration remained stuck after row release")?
        .into_diagnostic()
        .wrap_err("joining real repair migration")?
        .into_diagnostic()
        .wrap_err("real repair migration failed")?;

    let writer_result = match early_writer_result {
        Some(result) => result,
        None => tokio::time::timeout(Duration::from_secs(5), writer)
            .await
            .into_diagnostic()
            .wrap_err("writer remained stuck after real migration completed")?,
    };
    writer_result
        .into_diagnostic()
        .wrap_err("joining migration writer fixture")?
        .into_diagnostic()
        .wrap_err("writer failed after real migration completed")?;
    miette::ensure!(
        writer_blocked,
        "drawer writer bypassed the real migration table lock"
    );
    miette::ensure!(
        drawer_hash(db, candidate_id).await? == format!("legacy:{candidate_id}"),
        "real repair migration did not update its blocked candidate"
    );
    Ok(())
}

async fn wait_for_repair_table_lock(db: &DatabaseConnection) -> miette::Result<()> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let row = db
            .query_one(Statement::from_string(
                DatabaseBackend::Postgres,
                "SELECT EXISTS ( \
                     SELECT 1 FROM pg_locks \
                      WHERE relation = 'heka_drawers'::regclass \
                        AND mode = 'ShareRowExclusiveLock' \
                        AND granted \
                        AND pid <> pg_backend_pid() \
                 ) AS granted",
            ))
            .await
            .into_diagnostic()
            .wrap_err("observing real repair migration lock")?
            .ok_or_else(|| miette::miette!("migration lock query returned no row"))?;
        if row.try_get::<bool>("", "granted").into_diagnostic()? {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(miette::miette!(
                "real repair migration never acquired its table lock"
            ));
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
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
    let embedder = Arc::clone(embedder);
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
    );
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
            serde_json::json!({
                "type": role,
                "sessionId": tree.session_id,
                "cwd": tree.cwd,
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
        "cwd": tree.cwd,
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
    session_id: String,
    transcript: PathBuf,
}

impl TempCaptureTree {
    fn new(resource: &str) -> miette::Result<Self> {
        let base = std::env::temp_dir().join(format!("{resource}-capture-{}", Uuid::new_v4()));
        miette::ensure!(!base.exists(), "disposable capture path already exists");
        let root = base.join("claude-projects");
        let cwd = base.join("workspace");
        std::fs::create_dir_all(&root).into_diagnostic()?;
        std::fs::create_dir_all(&cwd).into_diagnostic()?;
        let cwd = cwd.canonicalize().into_diagnostic()?;
        let project = root.join(cwd.to_string_lossy().replace(['/', '\\'], "-"));
        std::fs::create_dir_all(&project).into_diagnostic()?;
        let session_id = Uuid::new_v4().to_string();
        let transcript = project.join(format!("{session_id}.jsonl"));
        Ok(Self {
            base,
            root,
            cwd,
            session_id,
            transcript,
        })
    }
}

impl Drop for TempCaptureTree {
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
    let (base, query) = admin_url
        .split_once('?')
        .map_or((admin_url, None), |(base, query)| (base, Some(query)));
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
        .map_or((None, authority), |(userinfo, host)| (Some(userinfo), host));
    if let Some(userinfo) = userinfo {
        miette::ensure!(
            !userinfo.contains(':'),
            "disposable admin URL must not contain a password"
        );
    }
    let local = host_port.starts_with("127.0.0.1:")
        || host_port.starts_with("localhost:")
        || host_port.starts_with("[::1]:");
    miette::ensure!(local, "disposable database server must be loopback-only");

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
