//! Evaluation report generator.
//!
//! Emits both JSON (machine-readable for CI delta comparison) and Markdown
//! (human-readable for review). Reports capture:
//! - Hardware metadata (CPU, memory, OS)
//! - Model versions (embedding model, LLM model if applicable)
//! - Dataset hashes for reproducibility
//! - Exact retrieval parameters
//! - p50/p99 latency percentiles
//!
//! The JSON format is stable across runs so the CI `eval-smoke` job can diff
//! against a baseline and post the delta as a non-blocking comment.

use serde::Serialize;

use crate::eval::EvalMetrics;

/// Context for a single evaluation report.
#[derive(Debug, Clone, Serialize)]
pub struct ReportContext {
    /// Benchmark name: "longmemeval" or "locomo".
    pub benchmark: String,
    /// Mode string (e.g., "raw", "hybrid", "top_k=auto").
    pub mode: String,
    /// SHA-256-style hash of the dataset directory, if available.
    pub dataset_hash: Option<String>,
    /// Number of sessions evaluated.
    pub sessions: usize,
    /// Aggregated metrics.
    pub metrics: EvalMetrics,
}

/// Hardware metadata captured at report time.
#[derive(Debug, Clone, Serialize)]
pub struct HardwareInfo {
    pub os: String,
    pub cpu_brand: String,
    pub cpu_cores: usize,
    pub total_memory_mb: u64,
    pub hostname: String,
}

/// Model version info for reproducibility.
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    /// Embedding model identifier (e.g., "all-MiniLM-L6-v2").
    pub embedding_model: String,
    /// LLM model identifier if used in evaluation (e.g., "gpt-4o").
    pub llm_model: Option<String>,
}

/// Full report structure serialized to JSON.
#[derive(Debug, Clone, Serialize)]
pub struct EvalReport {
    pub benchmark: String,
    pub mode: String,
    pub timestamp: String,
    pub hardware: HardwareInfo,
    pub models: ModelInfo,
    pub dataset_hash: Option<String>,
    pub sessions: usize,
    pub metrics: EvalMetrics,
}

/// Reporter handles writing eval results to JSON and Markdown.
pub struct Reporter {
    report: EvalReport,
}

impl Reporter {
    /// Build a reporter from a context, capturing hardware info at construction time.
    pub fn new(ctx: ReportContext) -> Self {
        Self::new_with_embedding_model(ctx, "stub (no retrieval)")
    }

    /// Build a reporter and record the concrete embedding model used.
    pub fn new_with_embedding_model(
        ctx: ReportContext,
        embedding_model: impl Into<String>,
    ) -> Self {
        let hardware = capture_hardware();
        let models = ModelInfo {
            embedding_model: embedding_model.into(),
            llm_model: None,
        };

        let timestamp = chrono::Utc::now().to_rfc3339();

        let report = EvalReport {
            benchmark: ctx.benchmark,
            mode: ctx.mode,
            timestamp,
            hardware,
            models,
            dataset_hash: ctx.dataset_hash,
            sessions: ctx.sessions,
            metrics: ctx.metrics,
        };

        Self { report }
    }

    /// Write the JSON report to `path`.
    pub fn write_json(&self, path: &str) -> miette::Result<()> {
        let json = serde_json::to_string_pretty(&self.report)
            .map_err(|e| miette::miette!("serializing report to JSON: {e}"))?;
        std::fs::write(path, json)
            .map_err(|e| miette::miette!("writing JSON report to {path}: {e}"))?;
        Ok(())
    }

    /// Write the Markdown report to `path`.
    pub fn write_markdown(&self, path: &str) -> miette::Result<()> {
        let md = self.render_markdown();
        std::fs::write(path, md)
            .map_err(|e| miette::miette!("writing Markdown report to {path}: {e}"))?;
        Ok(())
    }

    /// Render the report as Markdown.
    fn render_markdown(&self) -> String {
        let r = &self.report;
        let mut md = String::new();

        md.push_str(&format!("# Evaluation Report: {}\n\n", r.benchmark));
        md.push_str(&format!("**Generated:** {}\n\n", r.timestamp));
        md.push_str(&format!("**Mode:** {}\n\n", r.mode));

        // ── Summary ──
        md.push_str("## Summary\n\n");
        md.push_str(&format!(
            "| Metric | Value |\n|---|---|\n"
        ));
        md.push_str(&format!(
            "| Sessions | {} |\n", r.sessions
        ));
        md.push_str(&format!(
            "| Total Questions | {} |\n", r.metrics.total_questions
        ));
        md.push_str(&format!(
            "| Hits | {} |\n", r.metrics.hits
        ));
        md.push_str(&format!(
            "| Recall@K | {:.4} ({:.2}%) |\n",
            r.metrics.recall_at_k,
            r.metrics.recall_at_k * 100.0
        ));
        md.push_str(&format!(
            "| p50 Latency | {:.6}s |\n", r.metrics.p50_latency_secs
        ));
        md.push_str(&format!(
            "| p99 Latency | {:.6}s |\n", r.metrics.p99_latency_secs
        ));
        md.push_str("\n");

        // ── Retrieval Parameters ──
        md.push_str("## Retrieval Parameters\n\n");
        let p = &r.metrics.retrieval_params;
        md.push_str(&format!(
            "| Parameter | Value |\n|---|---|\n"
        ));
        md.push_str(&format!("| Mode | {} |\n", p.mode));
        md.push_str(&format!("| Top-K | {} |\n", p.top_k));
        if let Some(explicit) = p.top_k_explicit {
            md.push_str(&format!(
                "| Top-K Explicit | **{}** ⚠ WARNING: bypasses auto-cap |\n",
                explicit
            ));
        }
        if let Some(cap) = p.auto_top_k_cap {
            md.push_str(&format!("| Auto Top-K Cap | {} |\n", cap));
        }
        md.push_str("\n");

        // ── Hardware ──
        md.push_str("## Hardware\n\n");
        md.push_str(&format!(
            "| Property | Value |\n|---|---|\n"
        ));
        md.push_str(&format!("| OS | {} |\n", r.hardware.os));
        md.push_str(&format!("| CPU | {} |\n", r.hardware.cpu_brand));
        md.push_str(&format!("| Cores | {} |\n", r.hardware.cpu_cores));
        md.push_str(&format!("| Memory | {} MB |\n", r.hardware.total_memory_mb));
        md.push_str(&format!("| Hostname | {} |\n", r.hardware.hostname));
        md.push_str("\n");

        // ── Models ──
        md.push_str("## Models\n\n");
        md.push_str(&format!(
            "| Component | Model |\n|---|---|\n"
        ));
        md.push_str(&format!(
            "| Embedding | {} |\n",
            r.models.embedding_model
        ));
        if let Some(llm) = &r.models.llm_model {
            md.push_str(&format!("| LLM | {} |\n", llm));
        }
        md.push_str("\n");

        // ── Dataset ──
        md.push_str("## Dataset\n\n");
        match &r.dataset_hash {
            Some(hash) => {
                md.push_str(&format!("**Hash:** `{hash}`\n\n"));
            }
            None => {
                md.push_str("**Hash:** _not available (dataset directory not found)_\n\n");
            }
        }

        // ── Warning for explicit top-K ──
        if r.metrics.retrieval_params.top_k_explicit.is_some() {
            md.push_str("## ⚠⚠⚠ Explicit Top-K Warning ⚠⚠⚠\n\n");
            md.push_str(&format!(
                "This run used `--top-k-explicit {}`, bypassing the auto-cap safety guard \
                 `min(5, conv_len/4)`. This can inflate recall if K exceeds the \
                 conversation length. Review the results carefully.\n\n",
                r.metrics.retrieval_params.top_k_explicit.unwrap()
            ));
        }

        // ── Baseline note ──
        if r.metrics.recall_at_k == 0.0 && r.models.embedding_model.starts_with("stub") {
            md.push_str("## Baseline Note\n\n");
            md.push_str(
                "Recall is 0.0 because retrieval is a stub (Phase 42b). \
                 Phase 42c will wire real pgvector search and push this number up.\n\n",
            );
        }

        md
    }
}

/// Capture hardware metadata using sysinfo.
fn capture_hardware() -> HardwareInfo {
    use sysinfo::System;

    let mut sys = System::new_all();
    sys.refresh_all();

    let os = format!(
        "{} {}",
        System::name().unwrap_or_else(|| "unknown".to_string()),
        System::os_version().unwrap_or_else(|| String::new())
    );

    let cpu_brand = sys
        .cpus()
        .first()
        .map(|c| c.brand().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let cpu_cores = sys.cpus().len();

    let total_memory_mb = sys.total_memory() / 1024 / 1024;

    let hostname = System::host_name().unwrap_or_else(|| "unknown".to_string());

    HardwareInfo {
        os,
        cpu_brand,
        cpu_cores,
        total_memory_mb,
        hostname,
    }
}
