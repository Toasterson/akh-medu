//! LoCoMo evaluation harness.
//!
//! LoCoMo (Long Conversational Memory) is a benchmark for long-horizon
//! conversational memory retrieval. The harness loads conversations, runs
//! retrieval, and measures recall@K + latency.
//!
//! ## Top-K auto-cap
//!
//! When `--top-k auto` (default), the harness computes `min(5, conv_len / 4)`
//! per session. This is a structural guard against the MemPalace
//! `top_k=50`-on-30-items exploit: retrieving more items than the conversation
//! has turns is nonsensical and inflates recall artificially.
//!
//! `--top-k-explicit N` overrides the auto-cap but emits a **loud warning** in
//! the report so reviewers can see when a run bypasses the safety guard.
//!
//! Until Phase 42c lands, the retrieval stub returns empty — recall is 0.0.

use std::path::Path;

use crate::eval::reporter::{ReportContext, Reporter};
use crate::eval::{EvalMetrics, EvalScores, RetrievalParams, RetrievalResult};

/// Run the LoCoMo benchmark.
///
/// # Arguments
/// * `top_k` - "auto" (default) or a numeric string.
/// * `top_k_explicit` - If set, overrides auto top-K with an explicit value.
/// * `sessions` - Number of sessions to evaluate (10 for smoke test).
/// * `data_dir` - Path to the dataset directory (defaults to `eval_data/locomo`).
/// * `output` - Output directory for reports (defaults to `eval_reports/`).
pub async fn run(
    top_k: &str,
    top_k_explicit: Option<usize>,
    sessions: usize,
    data_dir: Option<&str>,
    output: Option<&str>,
) -> miette::Result<()> {
    let data_dir = data_dir.unwrap_or("eval_data/locomo");
    let output_dir = output.unwrap_or("eval_reports");

    tracing::info!(
        top_k,
        ?top_k_explicit,
        sessions,
        data_dir,
        "starting LoCoMo"
    );

    // Validate and resolve top-K strategy.
    let use_explicit = top_k_explicit.is_some();
    if use_explicit {
        // ⚠ LOUD WARNING — explicit top-K bypasses the auto-cap safety guard.
        let explicit_val = top_k_explicit.unwrap();
        tracing::warn!(
            top_k_explicit = explicit_val,
            "⚠⚠⚠  EXPLICIT TOP-K OVERRIDE: top_k={explicit_val} bypasses the auto-cap \
             (min(5, conv_len/4)). This can inflate recall if K exceeds the \
             conversation length. Review the report carefully.  ⚠⚠⚠"
        );
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

    let dataset_hash = compute_dataset_hash(data_path);

    // Run evaluation sessions.
    let mut all_results: Vec<RetrievalResult> = Vec::new();
    let mut auto_caps: Vec<usize> = Vec::new();

    for i in 0..sessions {
        // Simulate a conversation length (stub: 20 turns per session).
        let conv_len = 20;
        let effective_top_k = if let Some(explicit) = top_k_explicit {
            explicit
        } else {
            // Auto-cap: min(5, conv_len/4) — refuses the MemPalace exploit by construction.
            let cap = std::cmp::min(5, conv_len / 4);
            auto_caps.push(cap);
            cap
        };

        let session_results = run_session(i, effective_top_k).await;
        all_results.extend(session_results);
    }

    // Determine the auto top-K cap for reporting (use the most common value).
    let auto_top_k_cap = if auto_caps.is_empty() {
        None
    } else {
        auto_caps.sort();
        Some(auto_caps[auto_caps.len() / 2])
    };

    // Aggregate metrics.
    let metrics = aggregate_metrics(
        &all_results,
        RetrievalParams {
            mode: "locomo".to_string(),
            top_k: top_k.to_string(),
            top_k_explicit,
            auto_top_k_cap,
        },
    );

    tracing::info!(total_questions = metrics.total_questions, "LoCoMo complete");

    // Generate report.
    let ctx = ReportContext {
        benchmark: "locomo".to_string(),
        mode: format!(
            "top_k={top_k}{}",
            top_k_explicit
                .map(|v| format!(", explicit={v}"))
                .unwrap_or_default()
        ),
        dataset_hash,
        sessions,
        metrics: metrics.clone(),
    };

    let reporter = Reporter::new(ctx);
    std::fs::create_dir_all(output_dir)
        .map_err(|e| miette::miette!("creating output dir {output_dir}: {e}"))?;

    let json_path = format!("{output_dir}/locomo_report.json");
    let md_path = format!("{output_dir}/locomo_report.md");

    reporter.write_json(&json_path)?;
    reporter.write_markdown(&md_path)?;

    println!("Report written to {json_path} and {md_path}");
    Ok(())
}

/// Run a single evaluation session.
///
/// Until 42c, the retrieval stub returns empty (no hits).
async fn run_session(session_idx: usize, top_k: usize) -> Vec<RetrievalResult> {
    tracing::debug!(session_idx, top_k, "running session (stub retrieval)");

    // Stub: simulate 5 questions per session with empty retrieval.
    let mut results = Vec::with_capacity(5);
    for _ in 0..5 {
        let start = std::time::Instant::now();
        let _retrieved = stub_retrieve(top_k).await;
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
async fn stub_retrieve(_top_k: usize) -> Vec<String> {
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
        scores: EvalScores::BinaryRetrieval {
            hits,
            recall_at_k: recall,
        },
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
