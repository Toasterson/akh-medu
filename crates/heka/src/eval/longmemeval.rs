//! LongMemEval session-retrieval evaluation.
//!
//! The runner loads the official LongMemEval JSON schema, ingests each
//! evaluation history through the real drawer path, drains its embeddings, and
//! measures strict session-level recall against `answer_session_ids`.

use std::collections::HashSet;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::eval::{EvalMetrics, EvalScores, RetrievalParams};

#[cfg(feature = "embedding")]
use {
    crate::HekaConfig,
    crate::config::{DatabaseConfig, EmbeddingConfig, EvalConfig, ServerConfig, connect_database},
    crate::embed_worker::{count_pending_for_akh, embed_pending_batch_for_akh},
    crate::embedder::{SentenceEmbedder, SharedEmbedder, resolve_model_dir},
    crate::entity::heka_drawer_embeddings::vec_to_pgvector,
    crate::eval::reporter::{ReportContext, Reporter},
    crate::ingest::write_drawer_scoped,
    miette::{IntoDiagnostic, WrapErr},
    sea_orm::{ConnectionTrait, DatabaseBackend, DatabaseConnection, FromQueryResult, Statement},
    std::sync::{Arc, Mutex},
    std::time::{Duration, Instant},
    uuid::Uuid,
};

const DEFAULT_DATA_DIR: &str = "eval_data/longmemeval";
const DEFAULT_OUTPUT_DIR: &str = "eval_reports";
const TOP_K: usize = 5;
const MAX_DATASET_BYTES: usize = 512 * 1024 * 1024;
const MAX_EVALUATION_QUESTIONS: usize = 10_000;
const MAX_SESSIONS_PER_QUESTION: usize = 10_000;
const MAX_TURNS_PER_SESSION: usize = 10_000;
const MAX_IDENTIFIER_BYTES: usize = 1_024;
const MAX_QUESTION_BYTES: usize = 1024 * 1024;
const MAX_TURN_BYTES: usize = 1024 * 1024;

#[cfg(feature = "embedding")]
const SEARCH_SESSIONS_SQL: &str = "SELECT d.session_id, \
            (1.0 - MIN(e.embedding <=> $3::vector(384)))::real AS similarity \
       FROM heka_drawer_embeddings AS e \
       JOIN heka_drawer_chunks AS c ON c.id = e.chunk_id \
       JOIN heka_drawers AS d ON d.id = c.drawer_id \
      WHERE d.akh_id = $1 AND e.model_name = $2 AND d.session_id IS NOT NULL \
      GROUP BY d.session_id \
      ORDER BY MIN(e.embedding <=> $3::vector(384)) ASC \
      LIMIT $4";

#[derive(Debug, Clone, Deserialize)]
struct LongMemEvalCase {
    question_id: String,
    #[serde(default)]
    question_type: String,
    question: String,
    #[serde(default)]
    haystack_session_ids: Vec<String>,
    #[serde(default)]
    haystack_dates: Vec<String>,
    #[serde(default)]
    haystack_sessions: Vec<Vec<LongMemEvalTurn>>,
    #[serde(default)]
    answer_session_ids: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct LongMemEvalTurn {
    role: String,
    content: String,
    #[allow(dead_code)]
    #[serde(default)]
    has_answer: bool,
}

#[derive(Debug, Clone)]
struct EvidenceRetrievalResult {
    recalled_any: bool,
    recalled_all: bool,
    retrieved_count: usize,
    latency_secs: f64,
    retrieved_session_ids: Vec<String>,
}

/// Run the LongMemEval benchmark.
///
/// `sessions` is retained as the public CLI name for compatibility; it limits
/// the number of non-abstention evaluation instances (questions).
pub async fn run(
    mode: &str,
    sessions: usize,
    data_dir: Option<&str>,
    output: Option<&str>,
) -> miette::Result<()> {
    if mode != "raw" {
        return Err(miette::miette!(
            "LongMemEval mode `{mode}` is unavailable: Phase 42c implements only raw vector retrieval; refusing to publish a mislabeled hybrid report"
        ));
    }
    if sessions == 0 {
        return Err(miette::miette!("sessions must be greater than zero"));
    }
    if sessions > MAX_EVALUATION_QUESTIONS {
        return Err(miette::miette!(
            "requested {sessions} questions, exceeding the safety limit {MAX_EVALUATION_QUESTIONS}"
        ));
    }

    #[cfg(feature = "embedding")]
    {
        run_with_embeddings(mode, sessions, data_dir, output).await
    }

    #[cfg(not(feature = "embedding"))]
    {
        let _ = (sessions, data_dir, output);
        Err(miette::miette!(
            "LongMemEval retrieval requires the `embedding` feature; rebuild hekad with default features"
        ))
    }
}

#[cfg(feature = "embedding")]
async fn run_with_embeddings(
    mode: &str,
    sessions: usize,
    data_dir: Option<&str>,
    output: Option<&str>,
) -> miette::Result<()> {
    let dataset_path = resolve_dataset_path(Path::new(data_dir.unwrap_or(DEFAULT_DATA_DIR)))?;
    let output_dir = output.unwrap_or(DEFAULT_OUTPUT_DIR);
    let cases = load_dataset(&dataset_path)?;
    validate_dataset(&cases)?;
    let dataset_hash = compute_dataset_hash(&dataset_path);
    let selected: Vec<_> = cases
        .iter()
        // The official retrieval protocol excludes the 30 abstention questions
        // because they have no evidence-session ground truth.
        .filter(|case| !is_abstention(case))
        .take(sessions)
        .collect();
    if selected.len() != sessions {
        return Err(miette::miette!(
            "LongMemEval requested {sessions} non-abstention questions but the validated dataset contains only {}",
            selected.len()
        ));
    }

    tracing::info!(
        mode,
        requested_sessions = sessions,
        evaluated_questions = selected.len(),
        dataset = %dataset_path.display(),
        "starting LongMemEval"
    );
    let config = load_runtime_config()?;
    let db = connect_database(&config.database.url, config.database.max_connections)
        .await
        .into_diagnostic()
        .wrap_err("connecting to PostgreSQL for LongMemEval")?;

    let model_name = config.embedding.model.clone();
    let model_dir = resolve_model_dir(&config.model_dir(), &model_name)
        .into_diagnostic()
        .wrap_err_with(|| format!("resolving embedding model `{model_name}`"))?;
    let embedder: SharedEmbedder = Arc::new(Mutex::new(
        SentenceEmbedder::load(&model_dir)
            .into_diagnostic()
            .wrap_err("loading ONNX embedding model")?,
    ));

    let run_id = Uuid::new_v4();
    let mut results = Vec::with_capacity(selected.len());
    for case in &selected {
        let result = evaluate_case_isolated(
            &db,
            &embedder,
            config.embedding.batch_size,
            &model_name,
            run_id,
            case,
        )
        .await?;

        tracing::debug!(
            question_id = %case.question_id,
            recalled_any = result.recalled_any,
            recalled_all = result.recalled_all,
            retrieved = result.retrieved_count,
            "LongMemEval question complete"
        );
        results.push(result);
    }

    let metrics = aggregate_metrics(
        &results,
        RetrievalParams {
            mode: mode.to_string(),
            top_k: TOP_K.to_string(),
            top_k_explicit: None,
            auto_top_k_cap: None,
        },
    );
    tracing::info!(
        total_questions = metrics.total_questions,
        "LongMemEval evidence-session retrieval complete; primary metric is recall_all@5"
    );

    let reporter = Reporter::new_with_embedding_model(
        ReportContext {
            benchmark: "longmemeval".to_string(),
            mode: mode.to_string(),
            dataset_hash,
            sessions: selected.len(),
            metrics,
        },
        model_name,
    );
    std::fs::create_dir_all(output_dir)
        .into_diagnostic()
        .wrap_err_with(|| format!("creating output dir {output_dir}"))?;
    let json_path = format!("{output_dir}/longmemeval_report.json");
    let markdown_path = format!("{output_dir}/longmemeval_report.md");
    reporter.write_json(&json_path)?;
    reporter.write_markdown(&markdown_path)?;
    println!("Report written to {json_path} and {markdown_path}");
    Ok(())
}

fn resolve_dataset_path(path: &Path) -> miette::Result<PathBuf> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    if !path.is_dir() {
        return Err(miette::miette!(
            "LongMemEval dataset not found at {} — run eval_data/fetch_datasets.sh",
            path.display()
        ));
    }

    // Prefer the full cleaned small split when present. The fetch helper uses
    // the official oracle split for a lightweight smoke run.
    for filename in [
        "longmemeval_s_cleaned.json",
        "longmemeval_s.json",
        "longmemeval_oracle.json",
        "longmemeval.json",
    ] {
        let candidate = path.join(filename);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }

    Err(miette::miette!(
        "no supported LongMemEval JSON file found in {}",
        path.display()
    ))
}

fn load_dataset(path: &Path) -> miette::Result<Vec<LongMemEvalCase>> {
    let mut file = std::fs::File::open(path)
        .map_err(|error| miette::miette!("opening {}: {error}", path.display()))?;
    let metadata = file
        .metadata()
        .map_err(|error| miette::miette!("inspecting {}: {error}", path.display()))?;
    if !metadata.is_file() {
        return Err(miette::miette!(
            "LongMemEval dataset must be a regular file: {}",
            path.display()
        ));
    }
    if metadata.len() > MAX_DATASET_BYTES as u64 {
        return Err(miette::miette!(
            "LongMemEval dataset exceeds the {MAX_DATASET_BYTES} byte safety limit"
        ));
    }

    let mut bytes = Vec::with_capacity((metadata.len() as usize).min(64 * 1024));
    file.by_ref()
        .take(MAX_DATASET_BYTES.saturating_add(1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| miette::miette!("reading {}: {error}", path.display()))?;
    if bytes.len() > MAX_DATASET_BYTES {
        return Err(miette::miette!(
            "LongMemEval dataset exceeds the {MAX_DATASET_BYTES} byte safety limit"
        ));
    }
    serde_json::from_slice(&bytes)
        .map_err(|error| miette::miette!("parsing {}: {error}", path.display()))
}

fn validate_dataset(cases: &[LongMemEvalCase]) -> miette::Result<()> {
    if cases.is_empty() {
        return Err(miette::miette!("LongMemEval dataset is empty"));
    }
    if cases.len() > MAX_EVALUATION_QUESTIONS {
        return Err(miette::miette!(
            "LongMemEval contains {} questions, exceeding the safety limit {MAX_EVALUATION_QUESTIONS}",
            cases.len()
        ));
    }
    let mut question_ids = HashSet::with_capacity(cases.len());
    for case in cases {
        if !question_ids.insert(case.question_id.as_str()) {
            return Err(miette::miette!(
                "LongMemEval contains duplicate question_id `{}`",
                case.question_id
            ));
        }
        validate_case(case)?;
    }
    Ok(())
}

fn validate_case(case: &LongMemEvalCase) -> miette::Result<()> {
    if case.question_id.trim().is_empty() || case.question_id.len() > MAX_IDENTIFIER_BYTES {
        return Err(miette::miette!(
            "LongMemEval contains an empty or oversized question_id"
        ));
    }
    if case.question.trim().is_empty() || case.question.len() > MAX_QUESTION_BYTES {
        return Err(miette::miette!(
            "LongMemEval case {} has an empty or oversized question",
            case.question_id
        ));
    }
    if case.question_type.len() > MAX_IDENTIFIER_BYTES {
        return Err(miette::miette!(
            "LongMemEval case {} has an oversized question_type",
            case.question_id
        ));
    }
    if case.haystack_session_ids.len() > MAX_SESSIONS_PER_QUESTION {
        return Err(miette::miette!(
            "LongMemEval case {} exceeds the {MAX_SESSIONS_PER_QUESTION} session limit",
            case.question_id
        ));
    }
    if case.haystack_session_ids.len() != case.haystack_sessions.len() {
        return Err(miette::miette!(
            "LongMemEval case {} has {} session ids but {} session bodies",
            case.question_id,
            case.haystack_session_ids.len(),
            case.haystack_sessions.len()
        ));
    }
    if !case.haystack_dates.is_empty()
        && case.haystack_dates.len() != case.haystack_session_ids.len()
    {
        return Err(miette::miette!(
            "LongMemEval case {} has {} session ids but {} session dates",
            case.question_id,
            case.haystack_session_ids.len(),
            case.haystack_dates.len()
        ));
    }
    if case
        .haystack_dates
        .iter()
        .any(|date| date.len() > MAX_IDENTIFIER_BYTES)
    {
        return Err(miette::miette!(
            "LongMemEval case {} contains an oversized session date",
            case.question_id
        ));
    }

    let mut session_ids = HashSet::with_capacity(case.haystack_session_ids.len());
    for session_id in &case.haystack_session_ids {
        if session_id.trim().is_empty() || session_id.len() > MAX_IDENTIFIER_BYTES {
            return Err(miette::miette!(
                "LongMemEval case {} contains an empty or oversized haystack session id",
                case.question_id
            ));
        }
        if !session_ids.insert(session_id.as_str()) {
            return Err(miette::miette!(
                "LongMemEval case {} contains duplicate haystack session id `{session_id}`",
                case.question_id
            ));
        }
    }

    for (session_id, turns) in case
        .haystack_session_ids
        .iter()
        .zip(&case.haystack_sessions)
    {
        if turns.is_empty() {
            return Err(miette::miette!(
                "LongMemEval case {} session `{session_id}` has no turns",
                case.question_id
            ));
        }
        if turns.len() > MAX_TURNS_PER_SESSION {
            return Err(miette::miette!(
                "LongMemEval case {} session `{session_id}` exceeds the {MAX_TURNS_PER_SESSION} turn limit",
                case.question_id
            ));
        }
        for (turn_index, turn) in turns.iter().enumerate() {
            if !matches!(turn.role.as_str(), "user" | "assistant") {
                return Err(miette::miette!(
                    "LongMemEval case {} session `{session_id}` turn {turn_index} has invalid role `{}`",
                    case.question_id,
                    turn.role
                ));
            }
            if turn.content.trim().is_empty() || turn.content.len() > MAX_TURN_BYTES {
                return Err(miette::miette!(
                    "LongMemEval case {} session `{session_id}` turn {turn_index} has empty or oversized content",
                    case.question_id
                ));
            }
        }
    }

    if is_abstention(case) {
        if !case.answer_session_ids.is_empty() {
            return Err(miette::miette!(
                "LongMemEval abstention case {} unexpectedly has answer_session_ids",
                case.question_id
            ));
        }
        return Ok(());
    }

    if case.answer_session_ids.is_empty() {
        return Err(miette::miette!(
            "LongMemEval non-abstention case {} has no answer_session_ids",
            case.question_id
        ));
    }
    let mut answers = HashSet::with_capacity(case.answer_session_ids.len());
    for answer_session_id in &case.answer_session_ids {
        if answer_session_id.trim().is_empty()
            || answer_session_id.len() > MAX_IDENTIFIER_BYTES
            || !answers.insert(answer_session_id.as_str())
        {
            return Err(miette::miette!(
                "LongMemEval case {} has an empty or duplicate answer session id",
                case.question_id
            ));
        }
        if !session_ids.contains(answer_session_id.as_str()) {
            return Err(miette::miette!(
                "LongMemEval case {} answer session `{answer_session_id}` is absent from the haystack",
                case.question_id
            ));
        }
    }
    Ok(())
}

fn is_abstention(case: &LongMemEvalCase) -> bool {
    // Match the official metric aggregator, which excludes every question ID
    // containing the `_abs` marker before computing session recall.
    case.question_id.contains("_abs")
}

#[cfg(feature = "embedding")]
fn load_runtime_config() -> miette::Result<HekaConfig> {
    let config_path = default_config_path();
    if config_path.is_file() {
        return HekaConfig::from_file(&config_path)
            .into_diagnostic()
            .wrap_err_with(|| format!("loading config from {}", config_path.display()));
    }

    let database_url = std::env::var("DATABASE_URL")
        .into_diagnostic()
        .wrap_err_with(|| {
            format!(
                "DATABASE_URL is not set and {} does not exist",
                config_path.display()
            )
        })?;
    Ok(HekaConfig {
        database: DatabaseConfig {
            url: database_url,
            max_connections: 10,
        },
        server: ServerConfig::default(),
        embedding: EmbeddingConfig::default(),
        eval: EvalConfig::default(),
    })
}

#[cfg(feature = "embedding")]
fn default_config_path() -> PathBuf {
    std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("heka")
        .join("heka.toml")
}

#[cfg(feature = "embedding")]
async fn create_eval_akh(
    db: &DatabaseConnection,
    run_id: Uuid,
    question_id: &str,
) -> miette::Result<Uuid> {
    let id = Uuid::new_v4();
    let name = eval_akh_name(run_id, question_id);
    let now = chrono::Utc::now();
    let row = db
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "INSERT INTO heka_akhs \
                (id, name, description, personality, avatar_url, created_at, updated_at) \
             VALUES ($1, $2, $3, NULL, NULL, $4, $5) \
             RETURNING id",
            [
                id.into(),
                name.into(),
                "LongMemEval isolated evaluation corpus".to_string().into(),
                now.clone().into(),
                now.into(),
            ],
        ))
        .await
        .into_diagnostic()
        .wrap_err("registering LongMemEval evaluation akh")?
        .ok_or_else(|| miette::miette!("LongMemEval akh registration returned no id"))?;
    row.try_get_by_index::<Uuid>(0)
        .into_diagnostic()
        .wrap_err("reading LongMemEval akh id")
}

#[cfg(feature = "embedding")]
fn eval_akh_name(run_id: Uuid, question_id: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(question_id.as_bytes());
    let question_hash = format!("{:x}", hasher.finalize());
    format!("longmemeval:{run_id}:{}", &question_hash[..16])
}

#[cfg(feature = "embedding")]
async fn cleanup_eval_akh(db: &DatabaseConnection, akh_id: Uuid) -> miette::Result<()> {
    db.execute(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        "DELETE FROM heka_akhs WHERE id = $1",
        [akh_id.into()],
    ))
    .await
    .into_diagnostic()
    .wrap_err("cleaning up isolated LongMemEval corpus")?;
    Ok(())
}

#[cfg(feature = "embedding")]
async fn evaluate_case_isolated(
    db: &DatabaseConnection,
    embedder: &SharedEmbedder,
    batch_size: usize,
    model_name: &str,
    run_id: Uuid,
    case: &LongMemEvalCase,
) -> miette::Result<EvidenceRetrievalResult> {
    let akh_id = create_eval_akh(db, run_id, &case.question_id).await?;
    let outcome = evaluate_case_in_akh(db, embedder, batch_size, model_name, akh_id, case).await;
    let cleanup = cleanup_eval_akh(db, akh_id).await;
    match (outcome, cleanup) {
        (Ok(result), Ok(())) => Ok(result),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(cleanup_error)) => Err(cleanup_error),
        (Err(error), Err(cleanup_error)) => Err(miette::miette!(
            "{error}; isolated-corpus cleanup also failed: {cleanup_error}"
        )),
    }
}

#[cfg(feature = "embedding")]
async fn evaluate_case_in_akh(
    db: &DatabaseConnection,
    embedder: &SharedEmbedder,
    batch_size: usize,
    model_name: &str,
    akh_id: Uuid,
    case: &LongMemEvalCase,
) -> miette::Result<EvidenceRetrievalResult> {
    ingest_case(db, akh_id, case).await?;
    drain_case_embeddings(db, embedder, batch_size, model_name, akh_id).await?;

    let started = Instant::now();
    let query_embedding = embed_question(embedder, &case.question).await?;
    let retrieved = search_sessions(db, akh_id, model_name, &query_embedding, TOP_K).await?;
    let latency_secs = started.elapsed().as_secs_f64();
    let (recalled_any, recalled_all) = evidence_recall(&retrieved, &case.answer_session_ids);
    Ok(EvidenceRetrievalResult {
        recalled_any,
        recalled_all,
        retrieved_count: retrieved.len(),
        latency_secs,
        retrieved_session_ids: retrieved
            .into_iter()
            .map(|result| result.session_id)
            .collect(),
    })
}

#[cfg(feature = "embedding")]
async fn ingest_case(
    db: &DatabaseConnection,
    akh_id: Uuid,
    case: &LongMemEvalCase,
) -> miette::Result<()> {
    let topics = vec!["longmemeval".to_string(), case.question_type.clone()];
    for (session_id, turns) in case
        .haystack_session_ids
        .iter()
        .zip(&case.haystack_sessions)
    {
        for (turn_index, turn) in turns.iter().enumerate() {
            let dedupe_scope = format!("longmemeval:{session_id}:{turn_index}");
            write_drawer_scoped(
                db,
                akh_id,
                Some(session_id),
                Some(&turn.role),
                &turn.content,
                Some("longmemeval"),
                &topics,
                &dedupe_scope,
            )
            .await
            .into_diagnostic()
            .wrap_err_with(|| {
                format!(
                    "ingesting LongMemEval case {} session {}",
                    case.question_id, session_id
                )
            })?;
        }
    }
    Ok(())
}

#[cfg(feature = "embedding")]
async fn drain_case_embeddings(
    db: &DatabaseConnection,
    embedder: &SharedEmbedder,
    batch_size: usize,
    model_name: &str,
    akh_id: Uuid,
) -> miette::Result<()> {
    let stalled_since = Instant::now();
    loop {
        let stats = embed_pending_batch_for_akh(db, embedder, batch_size, model_name, akh_id)
            .await
            .into_diagnostic()
            .wrap_err("embedding LongMemEval drawer chunks")?;
        if stats.chunks_processed > 0 {
            continue;
        }

        let pending = count_pending_for_akh(db, akh_id)
            .await
            .into_diagnostic()
            .wrap_err("counting LongMemEval pending chunks")?;
        if pending == 0 {
            return Ok(());
        }
        if stalled_since.elapsed() > Duration::from_secs(30) {
            return Err(miette::miette!(
                "timed out waiting for {pending} LongMemEval chunks claimed by another worker"
            ));
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

#[cfg(feature = "embedding")]
async fn embed_question(embedder: &SharedEmbedder, question: &str) -> miette::Result<Vec<f32>> {
    let embedder = Arc::clone(embedder);
    let question = question.to_string();
    let inference = tokio::task::spawn_blocking(move || {
        let mut guard = embedder
            .lock()
            .map_err(|error| crate::HekaError::Embedding(format!("embedder lock: {error}")))?;
        guard.embed_text(&question)
    });
    tokio::time::timeout(Duration::from_secs(30), inference)
        .await
        .map_err(|_| miette::miette!("LongMemEval query embedding exceeded 30 seconds"))?
        .into_diagnostic()
        .wrap_err("joining LongMemEval query embedding task")?
        .into_diagnostic()
        .wrap_err("embedding LongMemEval question")
}

#[cfg(feature = "embedding")]
#[derive(Debug, FromQueryResult)]
struct RetrievedSession {
    session_id: String,
    similarity: f32,
}

#[cfg(feature = "embedding")]
async fn search_sessions(
    db: &DatabaseConnection,
    akh_id: Uuid,
    model_name: &str,
    query_embedding: &[f32],
    top_k: usize,
) -> miette::Result<Vec<RetrievedSession>> {
    let statement = Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        SEARCH_SESSIONS_SQL,
        [
            akh_id.into(),
            model_name.to_string().into(),
            vec_to_pgvector(query_embedding).into(),
            i64::try_from(top_k).unwrap_or(i64::MAX).into(),
        ],
    );
    RetrievedSession::find_by_statement(statement)
        .all(db)
        .await
        .into_diagnostic()
        .wrap_err("running LongMemEval pgvector retrieval")
}

#[cfg(feature = "embedding")]
fn evidence_recall(retrieved: &[RetrievedSession], answer_session_ids: &[String]) -> (bool, bool) {
    debug_assert!(!answer_session_ids.is_empty());
    let gold: HashSet<&str> = answer_session_ids.iter().map(String::as_str).collect();
    let recalled: HashSet<&str> = retrieved
        .iter()
        .map(|result| result.session_id.as_str())
        .collect();
    (
        gold.iter().any(|session_id| recalled.contains(session_id)),
        gold.iter().all(|session_id| recalled.contains(session_id)),
    )
}

fn compute_dataset_hash(path: &Path) -> Option<String> {
    let mut file = std::fs::File::open(path).ok()?;
    let metadata = file.metadata().ok()?;
    if !metadata.is_file() || metadata.len() > MAX_DATASET_BYTES as u64 {
        return None;
    }
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut total = 0usize;
    loop {
        let read = file.read(&mut buffer).ok()?;
        if read == 0 {
            break;
        }
        total = total.checked_add(read)?;
        if total > MAX_DATASET_BYTES {
            return None;
        }
        hasher.update(&buffer[..read]);
    }
    Some(format!("{:x}", hasher.finalize()))
}

fn aggregate_metrics(results: &[EvidenceRetrievalResult], params: RetrievalParams) -> EvalMetrics {
    let total = results.len();
    let any_hits = results.iter().filter(|result| result.recalled_any).count();
    let all_hits = results.iter().filter(|result| result.recalled_all).count();
    let (recall_any, recall_all) = if total > 0 {
        (
            any_hits as f64 / total as f64,
            all_hits as f64 / total as f64,
        )
    } else {
        (0.0, 0.0)
    };

    let mut latencies: Vec<f64> = results.iter().map(|result| result.latency_secs).collect();
    latencies.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));

    EvalMetrics {
        total_questions: total,
        scores: EvalScores::EvidenceSessionRetrieval {
            retrieved_any_at_5: any_hits,
            retrieved_all_at_5: all_hits,
            recall_any_at_5: recall_any,
            recall_all_at_5: recall_all,
        },
        p50_latency_secs: percentile(&latencies, 50),
        p99_latency_secs: percentile(&latencies, 99),
        retrieval_params: params,
    }
}

fn percentile(sorted: &[f64], percentile: usize) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((percentile as f64 / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

#[cfg(all(test, feature = "embedding"))]
#[path = "longmemeval_live_tests.rs"]
mod live_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> &'static str {
        r#"[
          {
            "question_id": "q1",
            "question_type": "single-session-user",
            "question": "Where did I leave the key?",
            "answer": "in the blue drawer",
            "question_date": "2026/01/01",
            "haystack_dates": ["2025/12/31"],
            "haystack_session_ids": ["evidence-session"],
            "haystack_sessions": [[
              {"role": "user", "content": "I left the key in the blue drawer.", "has_answer": true},
              {"role": "assistant", "content": "I will remember that.", "has_answer": false}
            ]],
            "answer_session_ids": ["evidence-session"]
          }
        ]"#
    }

    #[test]
    fn parses_official_dataset_shape_and_evidence_sessions() {
        let cases: Vec<LongMemEvalCase> = serde_json::from_str(fixture()).unwrap();
        assert_eq!(cases.len(), 1);
        validate_case(&cases[0]).unwrap();
        assert_eq!(cases[0].haystack_sessions[0].len(), 2);
        assert_eq!(
            cases[0].answer_session_ids,
            vec!["evidence-session".to_string()]
        );
    }

    #[test]
    fn malformed_parallel_session_arrays_fail_closed() {
        let mut cases: Vec<LongMemEvalCase> = serde_json::from_str(fixture()).unwrap();
        cases[0].haystack_session_ids.clear();
        assert!(validate_case(&cases[0]).is_err());
    }

    #[test]
    fn official_abstention_cases_are_excluded_from_retrieval_recall() {
        let mut cases: Vec<LongMemEvalCase> = serde_json::from_str(fixture()).unwrap();
        cases[0].question_id = "q1_abs".to_string();
        cases[0].answer_session_ids.clear();
        validate_case(&cases[0]).unwrap();
        assert!(is_abstention(&cases[0]));
    }

    #[cfg(feature = "embedding")]
    #[test]
    fn official_any_and_all_evidence_metrics_are_distinct() {
        let retrieved = vec![
            RetrievedSession {
                session_id: "distractor".to_string(),
                similarity: 0.9,
            },
            RetrievedSession {
                session_id: "evidence-session".to_string(),
                similarity: 0.8,
            },
        ];
        let gold = vec![
            "evidence-session".to_string(),
            "missing-session".to_string(),
        ];
        assert_eq!(evidence_recall(&retrieved, &gold), (true, false));

        let metrics = aggregate_metrics(
            &[EvidenceRetrievalResult {
                recalled_any: true,
                recalled_all: false,
                retrieved_count: 2,
                latency_secs: 0.01,
                retrieved_session_ids: vec!["evidence-session".to_string()],
            }],
            RetrievalParams {
                mode: "raw".to_string(),
                top_k: "5".to_string(),
                top_k_explicit: None,
                auto_top_k_cap: None,
            },
        );
        match metrics.scores {
            EvalScores::EvidenceSessionRetrieval {
                retrieved_any_at_5,
                retrieved_all_at_5,
                recall_any_at_5,
                recall_all_at_5,
            } => {
                assert_eq!(retrieved_any_at_5, 1);
                assert_eq!(retrieved_all_at_5, 0);
                assert_eq!(recall_any_at_5, 1.0);
                assert_eq!(recall_all_at_5, 0.0);
            }
            other => panic!("unexpected metric kind: {other:?}"),
        }

        let scores = EvalScores::EvidenceSessionRetrieval {
            retrieved_any_at_5: 1,
            retrieved_all_at_5: 0,
            recall_any_at_5: 1.0,
            recall_all_at_5: 0.0,
        };
        let published = serde_json::to_value(scores).unwrap();
        assert_eq!(published["recall_any@5"], 1.0);
        assert_eq!(published["recall_all@5"], 0.0);
        assert!(published.get("accuracy").is_none());
    }

    #[test]
    fn malformed_roles_sessions_answers_and_question_ids_fail_closed() {
        let cases: Vec<LongMemEvalCase> = serde_json::from_str(fixture()).unwrap();

        let mut invalid_role = cases[0].clone();
        invalid_role.haystack_sessions[0][0].role = "system".to_string();
        assert!(validate_case(&invalid_role).is_err());

        let mut duplicate_session = cases[0].clone();
        duplicate_session
            .haystack_session_ids
            .push("evidence-session".to_string());
        duplicate_session
            .haystack_sessions
            .push(duplicate_session.haystack_sessions[0].clone());
        duplicate_session
            .haystack_dates
            .push("2026/01/01".to_string());
        assert!(validate_case(&duplicate_session).is_err());

        let mut missing_answer = cases[0].clone();
        missing_answer.answer_session_ids = vec!["not-in-haystack".to_string()];
        assert!(validate_case(&missing_answer).is_err());

        assert!(validate_dataset(&[cases[0].clone(), cases[0].clone()]).is_err());
    }

    #[cfg(feature = "embedding")]
    #[test]
    fn repeated_runs_use_distinct_corpus_names() {
        let question_id = "same-question";
        let first = eval_akh_name(Uuid::new_v4(), question_id);
        let second = eval_akh_name(Uuid::new_v4(), question_id);
        assert_ne!(first, second);
        assert!(first.starts_with("longmemeval:"));
    }

    #[tokio::test]
    async fn hybrid_mode_is_refused_before_dataset_or_database_access() {
        let error = run("hybrid", 1, Some("does-not-exist"), None)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("refusing to publish a mislabeled hybrid report"));
    }

    #[cfg(feature = "embedding")]
    #[test]
    fn retrieval_query_is_parameterized_and_scoped() {
        // The production query must never interpolate dataset or vector text,
        // and it must isolate each question's evaluation akh.
        assert!(SEARCH_SESSIONS_SQL.contains("d.akh_id = $1"));
        assert!(SEARCH_SESSIONS_SQL.contains("e.model_name = $2"));
        assert!(SEARCH_SESSIONS_SQL.contains("$3::vector(384)"));
    }
}
