//! LongMemEval session-retrieval evaluation.
//!
//! The runner loads the official LongMemEval JSON schema, ingests each
//! evaluation history through the real drawer path, drains its embeddings, and
//! measures strict session-level recall against `answer_session_ids`.

use std::io::Read;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::eval::{EvalMetrics, RetrievalParams, RetrievalResult};

#[cfg(feature = "embedding")]
use {
    crate::HekaConfig,
    crate::config::{DatabaseConfig, EmbeddingConfig, EvalConfig, ServerConfig},
    crate::embed_worker::{count_pending_for_akh, embed_pending_batch_for_akh},
    crate::embedder::{SentenceEmbedder, SharedEmbedder, resolve_model_dir},
    crate::entity::heka_drawer_embeddings::vec_to_pgvector,
    crate::eval::reporter::{ReportContext, Reporter},
    crate::ingest::write_drawer,
    miette::{IntoDiagnostic, WrapErr},
    sea_orm::{
        ConnectionTrait, Database, DatabaseBackend, DatabaseConnection, FromQueryResult, Statement,
    },
    std::collections::HashSet,
    std::sync::{Arc, Mutex},
    std::time::{Duration, Instant},
    uuid::Uuid,
};

const DEFAULT_DATA_DIR: &str = "eval_data/longmemeval";
const DEFAULT_OUTPUT_DIR: &str = "eval_reports";
const TOP_K: usize = 5;

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
    if mode != "raw" && mode != "hybrid" {
        return Err(miette::miette!(
            "invalid mode `{mode}` — must be `raw` or `hybrid`"
        ));
    }
    if sessions == 0 {
        return Err(miette::miette!("sessions must be greater than zero"));
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
    let dataset_hash = compute_dataset_hash(&dataset_path);
    let cases = load_dataset(&dataset_path)?;
    let selected: Vec<_> = cases
        .into_iter()
        // The official retrieval protocol excludes abstention questions because
        // they have no evidence-session ground truth.
        .filter(is_evaluable_case)
        .take(sessions)
        .collect();
    if selected.is_empty() {
        return Err(miette::miette!(
            "LongMemEval dataset contains no non-abstention cases with answer_session_ids"
        ));
    }

    tracing::info!(
        mode,
        requested_sessions = sessions,
        evaluated_questions = selected.len(),
        dataset = %dataset_path.display(),
        "starting LongMemEval"
    );
    if mode == "hybrid" {
        tracing::warn!(
            "Phase 42c has no symbolic reranker; hybrid uses the same raw vector candidate stage"
        );
    }

    let config = load_runtime_config()?;
    let db = Database::connect(&config.database.url)
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

    let mut results = Vec::with_capacity(selected.len());
    for case in &selected {
        validate_case(case)?;
        let akh_id = ensure_eval_akh(&db, &case.question_id).await?;
        ingest_case(&db, akh_id, case).await?;
        drain_case_embeddings(
            &db,
            &embedder,
            config.embedding.batch_size,
            &model_name,
            akh_id,
        )
        .await?;

        let started = Instant::now();
        let query_embedding = embed_question(&embedder, &case.question).await?;
        let retrieved = search_sessions(&db, akh_id, &model_name, &query_embedding, TOP_K).await?;
        let latency_secs = started.elapsed().as_secs_f64();
        let hit = session_hit(&retrieved, &case.answer_session_ids);

        tracing::debug!(
            question_id = %case.question_id,
            hit,
            retrieved = retrieved.len(),
            best_similarity = retrieved.first().map(|row| row.similarity),
            "LongMemEval question complete"
        );
        results.push(RetrievalResult {
            hit,
            retrieved_count: retrieved.len(),
            latency_secs,
        });
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
        hits = metrics.hits,
        recall = %format!("{:.4}", metrics.recall_at_k),
        "LongMemEval complete"
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
    let file = std::fs::File::open(path)
        .map_err(|error| miette::miette!("opening {}: {error}", path.display()))?;
    serde_json::from_reader(file)
        .map_err(|error| miette::miette!("parsing {}: {error}", path.display()))
}

fn validate_case(case: &LongMemEvalCase) -> miette::Result<()> {
    if case.question.trim().is_empty() {
        return Err(miette::miette!(
            "LongMemEval case {} has an empty question",
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
    Ok(())
}

fn is_evaluable_case(case: &LongMemEvalCase) -> bool {
    !case.question_id.ends_with("_abs") && !case.answer_session_ids.is_empty()
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
async fn ensure_eval_akh(db: &DatabaseConnection, question_id: &str) -> miette::Result<Uuid> {
    let id = Uuid::new_v4();
    let name = format!("longmemeval:{question_id}");
    let now = chrono::Utc::now();
    let row = db
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "INSERT INTO heka_akhs \
                (id, name, description, personality, avatar_url, created_at, updated_at) \
             VALUES ($1, $2, $3, NULL, NULL, $4, $5) \
             ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name \
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
        for turn in turns {
            if turn.content.trim().is_empty() {
                continue;
            }
            write_drawer(
                db,
                akh_id,
                Some(session_id),
                Some(&turn.role),
                &turn.content,
                Some("longmemeval"),
                &topics,
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
    tokio::task::spawn_blocking(move || {
        let mut guard = embedder
            .lock()
            .map_err(|error| crate::HekaError::Embedding(format!("embedder lock: {error}")))?;
        guard.embed_text(&question)
    })
    .await
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
fn session_hit(retrieved: &[RetrievedSession], answer_session_ids: &[String]) -> bool {
    let gold: HashSet<&str> = answer_session_ids.iter().map(String::as_str).collect();
    retrieved
        .iter()
        .any(|result| gold.contains(result.session_id.as_str()))
}

fn compute_dataset_hash(path: &Path) -> Option<String> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).ok()?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Some(format!("{:x}", hasher.finalize()))
}

fn aggregate_metrics(results: &[RetrievalResult], params: RetrievalParams) -> EvalMetrics {
    let total = results.len();
    let hits = results.iter().filter(|result| result.hit).count();
    let recall = if total > 0 {
        hits as f64 / total as f64
    } else {
        0.0
    };

    let mut latencies: Vec<f64> = results.iter().map(|result| result.latency_secs).collect();
    latencies.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));

    EvalMetrics {
        total_questions: total,
        hits,
        recall_at_k: recall,
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
        assert!(!is_evaluable_case(&cases[0]));
    }

    #[cfg(feature = "embedding")]
    #[test]
    fn strict_evidence_session_scoring_can_produce_nonzero_recall() {
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
        assert!(session_hit(&retrieved, &["evidence-session".to_string()]));

        let metrics = aggregate_metrics(
            &[RetrievalResult {
                hit: true,
                retrieved_count: 2,
                latency_secs: 0.01,
            }],
            RetrievalParams {
                mode: "raw".to_string(),
                top_k: "5".to_string(),
                top_k_explicit: None,
                auto_top_k_cap: None,
            },
        );
        assert_eq!(metrics.hits, 1);
        assert_eq!(metrics.recall_at_k, 1.0);
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
