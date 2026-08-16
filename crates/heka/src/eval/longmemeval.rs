//! LongMemEval evaluation harness.
//!
//! LongMemEval is a multi-session conversational memory benchmark. The harness
//! loads sessions from the dataset, runs retrieval against the memory store,
//! and measures recall@K + latency.
//!
//! Until Phase 42c lands, the retrieval stub returns empty — recall is 0.0.
//! Each subsequent sub-phase pushes the number up.

use std::path::Path;

use crate::eval::reporter::{ReportContext, Reporter};
use crate::eval::{EvalMetrics, RetrievalParams, RetrievalResult};

/// Run the LongMemEval benchmark.
///
/// # Arguments
/// * `mode` - "raw" (direct embedding search) or "hybrid" (embedding + symbolic re-rank).
/// * `sessions` - Number of sessions to evaluate (10 for smoke test).
/// * `data_dir` - Path to the dataset directory (defaults to `eval_data/longmemeval`).
/// * `output` - Output directory for reports (defaults to `eval_reports/`).
pub async fn run(
    mode: &str,
    sessions: usize,
    data_dir: Option<&str>,
    output: Option<&str>,
) -> miette::Result<()> {
    let data_dir = data_dir.unwrap_or("eval_data/longmemeval");
    let output_dir = output.unwrap_or("eval_reports");

    tracing::info!(mode, sessions, data_dir, "starting LongMemEval");

    // Validate mode.
    if mode != "raw" && mode != "hybrid" {
        return Err(miette::miette!(
            "invalid mode `{mode}` — must be `raw` or `hybrid`"
        ));
    }

    // Check that the dataset directory exists.
    let data_path = Path::new(data_dir);
    if !data_path.exists() {
        tracing::warn!(
            data_dir,
            "dataset directory not found — run eval_data/fetch_datasets.sh to download. \
             Running in stub mode (0% recall baseline)."
        );
    }

    // Collect dataset hash if the directory exists.
    let dataset_hash = compute_dataset_hash(data_path);

    // Run evaluation sessions.
    let mut all_results: Vec<RetrievalResult> = Vec::new();

    for i in 0..sessions {
        let session_results = run_session(i, mode).await;
        all_results.extend(session_results);
    }

    // Aggregate metrics.
    let metrics = aggregate_metrics(&all_results, RetrievalParams {
        mode: mode.to_string(),
        top_k: "auto".to_string(),
        top_k_explicit: None,
        auto_top_k_cap: None,
    });

    tracing::info!(
        total_questions = metrics.total_questions,
        hits = metrics.hits,
        recall = %format!("{:.4}", metrics.recall_at_k),
        "LongMemEval complete"
    );

    // Generate report.
    let ctx = ReportContext {
        benchmark: "longmemeval".to_string(),
        mode: mode.to_string(),
        dataset_hash,
        sessions,
        metrics: metrics.clone(),
    };

    let reporter = Reporter::new(ctx);
    std::fs::create_dir_all(output_dir)
        .map_err(|e| miette::miette!("creating output dir {output_dir}: {e}"))?;

    let json_path = format!("{output_dir}/longmemeval_report.json");
    let md_path = format!("{output_dir}/longmemeval_report.md");

    reporter.write_json(&json_path)?;
    reporter.write_markdown(&md_path)?;

    println!("Report written to {json_path} and {md_path}");
    Ok(())
}

/// Run a single evaluation session.
///
/// Until 42c, the retrieval stub returns empty (no hits).
async fn run_session(session_idx: usize, _mode: &str) -> Vec<RetrievalResult> {
    tracing::debug!(session_idx, "running session (stub retrieval)");

    // Stub: simulate 5 questions per session with empty retrieval.
    let mut results = Vec::with_capacity(5);
    for _ in 0..5 {
        let start = std::time::Instant::now();
        // Stub retrieval — returns empty. 42c will wire this to the real store.
        let _retrieved = stub_retrieve().await;
        let elapsed = start.elapsed().as_secs_f64();

        results.push(RetrievalResult {
            hit: false,
            retrieved_count: 0,
            latency_secs: elapsed,
        });
    }
    results
}

/// Stub retrieval — returns empty Vec. Phase 42c will replace this with
/// real pgvector search against the heka store.
async fn stub_retrieve() -> Vec<String> {
    // Simulate minimal latency so p50/p99 have non-trivial values.
    tokio::time::sleep(std::time::Duration::from_micros(100)).await;
    Vec::new()
}

/// Compute a simple hash of the dataset directory contents for reproducibility.
fn compute_dataset_hash(data_dir: &Path) -> Option<String> {
    if !data_dir.exists() {
        return None;
    }

    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::io::Read;

    let mut hasher = DefaultHasher::new();

    // Walk the directory and hash file names + sizes.
    fn walk_and_hash(dir: &Path, hasher: &mut DefaultHasher) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            let mut sorted_entries: Vec<_> = entries.filter_map(|e| e.ok()).collect();
            sorted_entries.sort_by_key(|e| e.path());
            for entry in sorted_entries {
                let path = entry.path();
                if let Some(name) = path.file_name() {
                    name.to_string_lossy().hash(hasher);
                }
                if path.is_dir() {
                    walk_and_hash(&path, hasher);
                } else if let Ok(metadata) = entry.metadata() {
                    metadata.len().hash(hasher);
                    // Hash first 4KB of file for content fingerprinting.
                    if let Ok(mut file) = std::fs::File::open(&path) {
                        let mut buf = [0u8; 4096];
                        if let Ok(n) = file.read(&mut buf) {
                            buf[..n].hash(hasher);
                        }
                    }
                }
            }
        }
    }

    walk_and_hash(data_dir, &mut hasher);
    Some(format!("{:016x}", hasher.finish()))
}

/// Aggregate per-query results into EvalMetrics.
fn aggregate_metrics(results: &[RetrievalResult], params: RetrievalParams) -> EvalMetrics {
    let total = results.len();
    let hits = results.iter().filter(|r| r.hit).count();
    let recall = if total > 0 {
        hits as f64 / total as f64
    } else {
        0.0
    };

    let mut latencies: Vec<f64> = results.iter().map(|r| r.latency_secs).collect();
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let p50 = percentile(&latencies, 50);
    let p99 = percentile(&latencies, 99);

    EvalMetrics {
        total_questions: total,
        hits,
        recall_at_k: recall,
        p50_latency_secs: p50,
        p99_latency_secs: p99,
        retrieval_params: params,
    }
}

/// Compute a percentile from a sorted slice.
fn percentile(sorted: &[f64], pct: usize) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((pct as f64 / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
