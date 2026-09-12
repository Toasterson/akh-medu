//! Phase 44n.1 — context-rot curve bench (WI-386).
//!
//! Measures per-backing-model context-rot curves (needle-depth × context-length
//! grid) and emits two regression-trackable numbers per (model, filler) pair:
//!
//! - **`positional_sensitivity(L)`** — relative middle-vs-edge accuracy drop at
//!   length L: `max(0, edge − mid) / max(edge, ε)`.
//! - **`effective_length`** — largest swept L whose mean depth hit-rate is ≥
//!   `threshold` (default 0.8). This is the practical MECW — the number
//!   `CompactingManager` should budget against, not the advertised window.
//!
//! Needles are seeded nonce codes (reproducible per model × filler × L × depth
//! × repeat), filler is neutral prose or same-domain vault distractors, and
//! cells where the backend truncated the prompt (`l_actual` > 20 % below
//! `l_target`) are excluded from every metric so truncation cannot masquerade
//! as recall.
//!
//! Backends: the Ollama probe sets `num_ctx` per request (Ollama's default
//! silently truncates long prompts, faking good edge recall); the Candle probe
//! loads the local GGUF (`--features candle-backend`) and reads its context
//! window from the model metadata. A `--dry-run` MockProbe emits a real CSV
//! with no model attached, for plumbing validation.
//!
//! Follows the `bench_thinking` template: logic here, entry point in
//! `examples/bench_context_rot.rs`, CSVs to `target/bench-results/context-rot/`.
//!
//! Recovered and completed from the WI-276 parent branch
//! (`claude/wi-019fbf62-phase-44n-context-rot-curve-bench-per-mo`); additions
//! here are the Candle probe, schema-versioned CSV with source/git metadata,
//! a truncation-resilient effective-length, and the `examples/` entry point the
//! parent never shipped.

use std::io::{self, Write as _};
use std::path::Path;
use std::time::Instant;

use miette::Diagnostic;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use thiserror::Error;

use crate::agent::context::{HeuristicTokenCounter, TokenCounter};
use crate::agent::ollama::{GenerateResult, OllamaClient};
use crate::nlu::error::NluError;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors from the context-rot bench.
#[derive(Debug, Error, Diagnostic)]
pub enum RotBenchError {
    #[error("Ollama probe failed: {0}")]
    #[diagnostic(code(akh::rot::ollama), help("Check that Ollama is running and the model is pulled."))]
    Ollama(#[from] crate::agent::ollama::OllamaError),

    #[error("No lengths to sweep after capping to advertised window")]
    #[diagnostic(code(akh::rot::empty_grid), help("Set AKH_ROT_LENGTHS to smaller values or use a model with a larger context window."))]
    EmptyGrid,

    #[error("CSV write failed: {0}")]
    #[diagnostic(code(akh::rot::csv), help("Check disk space and permissions on target/bench-results/."))]
    Csv(#[from] io::Error),
}

// ---------------------------------------------------------------------------
// Probe trait + impls
// ---------------------------------------------------------------------------

/// Response from a probe completion.
#[derive(Debug, Clone)]
pub struct ProbeResponse {
    /// The generated text.
    pub text: String,
    /// Actual prompt token count if reported by the backend.
    pub prompt_tokens: Option<u64>,
}

/// A prompt→text completion backend for the rot bench.
///
/// Neither the full `ChatBackend::chat_with_tools` nor the agentic loop is
/// needed — probes are plain prompt→text. This small trait keeps the bench
/// decoupled from the daemon's tool plumbing.
pub trait RotProbe {
    /// Model id shown in reports.
    fn label(&self) -> &str;
    /// Backend family recorded in `runs.csv` (`"ollama"`, `"candle"`, `"mock"`).
    fn backend(&self) -> &str;
    /// Advertised context window (caps the sweep). `None` = unknown.
    fn context_window(&self) -> Option<usize>;
    /// Complete a (system, user) prompt with `max_tokens` output budget.
    fn complete(
        &mut self,
        system: &str,
        user: &str,
        max_tokens: u32,
    ) -> miette::Result<ProbeResponse>;
}

/// Ollama-backed probe. Sets `num_ctx` per probe to avoid silent truncation.
pub struct OllamaProbe {
    client: OllamaClient,
    label: String,
}

impl OllamaProbe {
    pub fn new(client: OllamaClient) -> Self {
        let label = client.config().model.clone();
        Self { client, label }
    }
}

impl RotProbe for OllamaProbe {
    fn label(&self) -> &str {
        &self.label
    }

    fn backend(&self) -> &str {
        "ollama"
    }

    fn context_window(&self) -> Option<usize> {
        self.client.capabilities().context_window
    }

    fn complete(
        &mut self,
        system: &str,
        user: &str,
        max_tokens: u32,
    ) -> miette::Result<ProbeResponse> {
        // Set num_ctx to the prompt's estimated length + headroom for output.
        let est_tokens = HeuristicTokenCounter.count(user) + HeuristicTokenCounter.count(system);
        let num_ctx = est_tokens + max_tokens as usize + 512;

        let result: GenerateResult = self
            .client
            .generate_with_options(system, user, num_ctx)
            .map_err(|e| miette::miette!("Ollama probe: {e}"))?;

        Ok(ProbeResponse {
            text: result.text,
            prompt_tokens: if result.prompt_tokens > 0 {
                Some(result.prompt_tokens)
            } else {
                None
            },
        })
    }
}

/// Candle-backed probe (local GGUF, `candle-backend` feature).
///
/// Uses [`CandleBackend::generate`] for prompt→text and reads the context
/// window from the model's own GGUF `context_length` metadata — the sweep is
/// always real for Candle, whose window is authoritative. Prompt tokens are not
/// surfaced by the backend, so `prompt_tokens` is `None` and truncation
/// detection is skipped (Candle's `context_window` is always known, and the
/// grid is capped to it).
#[cfg(feature = "candle-backend")]
pub struct CandleProbe {
    backend: crate::nlu::candle_backend::CandleBackend,
    label: String,
}

#[cfg(feature = "candle-backend")]
impl CandleProbe {
    pub fn new(backend: crate::nlu::candle_backend::CandleBackend) -> Self {
        let label = format!("candle-{:?}", backend.family());
        Self { backend, label }
    }
}

#[cfg(feature = "candle-backend")]
impl RotProbe for CandleProbe {
    fn label(&self) -> &str {
        &self.label
    }

    fn backend(&self) -> &str {
        "candle"
    }

    fn context_window(&self) -> Option<usize> {
        Some(self.backend.context_length())
    }

    fn complete(
        &mut self,
        _system: &str,
        user: &str,
        max_tokens: u32,
    ) -> miette::Result<ProbeResponse> {
        // CandleBackend::generate takes a bare prompt and applies its own
        // sampling config; fold the system instruction into the user turn so
        // the harness does not depend on the backend's chat-template plumbing.
        let prompt = format!(
            "Answer only from the provided text. Reply with the answer and nothing else.\n\n{user}"
        );
        let text = self
            .backend
            .generate(&prompt, max_tokens)
            .map_err(|e| miette::miette!("Candle probe: {e}"))?;
        Ok(ProbeResponse {
            text,
            prompt_tokens: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Haystack construction
// ---------------------------------------------------------------------------

/// Filler mode: same-domain distractors or unrelated neutral prose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum FillerMode {
    Distractor,
    Neutral,
}

impl FillerMode {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Distractor => "distractor",
            Self::Neutral => "neutral",
        }
    }
}

/// A synthetic needle: one seeded fact to retrieve.
#[derive(Debug, Clone)]
pub struct Needle {
    /// The vault name (used in the question).
    pub vault_name: String,
    /// The nonce access code (the answer the model must retrieve).
    pub access_code: String,
    /// The question to ask.
    pub question: String,
    /// The sentence to embed at the target depth.
    pub sentence: String,
}

impl Needle {
    /// Generate a deterministic needle from the RNG.
    fn generate(rng: &mut StdRng) -> Self {
        let vault_idx = rng.gen_range(1..=999);
        let vault_name = format!("vault-{vault_idx:03}");

        // 6-char alphanumeric nonce (uppercase + digits).
        const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        let mut code = String::with_capacity(6);
        for _ in 0..6 {
            code.push(CHARS[rng.gen_range(0..CHARS.len())] as char);
        }

        let sentence = format!(
            "The access code for vault-{vault_idx:03} is {code}."
        );
        let question = format!(
            "What is the access code for vault-{vault_idx:03}?"
        );

        Self {
            vault_name,
            access_code: code,
            question,
            sentence,
        }
    }

    /// Check whether the response contains the nonce (normalized).
    fn grade(&self, response: &str) -> bool {
        let resp_lower = response.to_lowercase();
        let code_lower = self.access_code.to_lowercase();
        resp_lower.contains(&code_lower)
    }
}

/// Neutral filler corpus — generic unrelated prose.
const NEUTRAL_CORPUS: &[&str] = &[
    "The old lighthouse stood on the cliff, its beam sweeping the dark waters.",
    "Coffee is best brewed at 92 degrees Celsius with freshly ground beans.",
    "The train departed at dawn, carrying cargo through the mountain pass.",
    "A team of researchers studied the migration patterns of arctic terns.",
    "The library's oldest manuscript dates back to the twelfth century.",
    "Gardening requires patience: tomatoes take seventy days to ripen.",
    "The architect designed the building with natural ventilation in mind.",
    "Saturday's concert featured a string quartet performing late Beethoven.",
    "The river carved a canyon over millennia, layer by layer of sediment.",
    "A baker's dozen is thirteen, a tradition from medieval guilds.",
    "The telescope captured images of a distant galaxy cluster.",
    "Olive trees can live for centuries, their roots deep in dry soil.",
    "The chess match lasted four hours, ending in a draw by repetition.",
    "Maple syrup is collected in early spring when nights are still cold.",
    "The ferry crossed the fjord daily, connecting two small fishing villages.",
    "A potter shaped the clay on the wheel, hands steady and wet.",
    "The monastery's bells rang at six, calling the monks to prayer.",
    "Desert foxes are nocturnal, avoiding the heat of the midday sun.",
    "The bridge was painted red every five years to prevent rust.",
    "A meteor streaked across the sky, visible for several seconds.",
];

/// Generate distractor filler — verbalized KG triples about other vaults.
fn generate_distractor_sentence(rng: &mut StdRng) -> String {
    let vault_idx = rng.gen_range(1..=999);
    let props = [
        "was inspected on Monday",
        "has a steel-reinforced door",
        "is located in sub-basement two",
        "contains archived documents",
        "requires dual authentication",
        "was last accessed three weeks ago",
        "has a humidity-controlled interior",
        "is shared by two departments",
    ];
    let prop = props[rng.gen_range(0..props.len())];
    format!("Vault-{vault_idx:03} {prop}.")
}

/// Build a haystack prompt of approximately `target_tokens` tokens, with the
/// needle sentence inserted at fractional depth `depth` (0.0 = beginning,
/// 1.0 = end).
///
/// Returns the full user prompt (haystack + question).
fn build_haystack(
    rng: &mut StdRng,
    needle: &Needle,
    target_tokens: usize,
    depth: f64,
    filler: FillerMode,
    counter: &HeuristicTokenCounter,
) -> String {
    let question = &needle.question;
    let needle_sentence = &needle.sentence;

    // Build filler until we reach target_tokens (minus room for needle + question).
    let question_tokens = counter.count(question);
    let needle_tokens = counter.count(needle_sentence);
    let overhead = question_tokens + needle_tokens + 20; // +20 for newlines/formatting
    let filler_target = target_tokens.saturating_sub(overhead);

    let mut filler_chunks: Vec<String> = Vec::new();
    let mut accumulated = 0usize;

    while accumulated < filler_target {
        let chunk = match filler {
            FillerMode::Distractor => generate_distractor_sentence(rng),
            FillerMode::Neutral => {
                let idx = rng.gen_range(0..NEUTRAL_CORPUS.len());
                NEUTRAL_CORPUS[idx].to_string()
            }
        };
        accumulated += counter.count(&chunk);
        filler_chunks.push(chunk);
    }

    // Insert the needle at the target depth.
    let insert_pos = ((filler_chunks.len() as f64) * depth.clamp(0.0, 1.0)).round() as usize;
    let insert_pos = insert_pos.min(filler_chunks.len());

    let mut parts: Vec<String> = Vec::with_capacity(filler_chunks.len() + 3);
    for (i, chunk) in filler_chunks.iter().enumerate() {
        if i == insert_pos {
            parts.push(needle_sentence.clone());
        }
        parts.push(chunk.clone());
    }
    if insert_pos >= filler_chunks.len() {
        parts.push(needle_sentence.clone());
    }

    let haystack = parts.join("\n");
    format!("{haystack}\n\nQuestion: {question}")
}

// ---------------------------------------------------------------------------
// Grid configuration
// ---------------------------------------------------------------------------

/// Default output directory for CSV artefacts.
pub const OUTPUT_DIR: &str = "target/bench-results/context-rot";

/// Default swept lengths (tokens).
pub const DEFAULT_LENGTHS: &[usize] = &[1024, 2048, 4096, 8192, 16384, 32768];

/// Default swept depths (fractional position in the haystack).
pub const DEFAULT_DEPTHS: &[f64] = &[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];

/// Default repeats.
pub const DEFAULT_REPEATS: usize = 3;

/// Default accuracy threshold for effective_length.
pub const DEFAULT_THRESHOLD: f64 = 0.8;

/// Edge depth values for positional sensitivity.
pub const EDGE_DEPTHS: &[f64] = &[0.0, 0.1, 0.9, 1.0];

/// Middle depth values for positional sensitivity.
pub const MID_DEPTHS: &[f64] = &[0.25, 0.5, 0.75];

/// Grid configuration, env-overridable.
#[derive(Debug, Clone)]
pub struct GridConfig {
    pub lengths: Vec<usize>,
    pub depths: Vec<f64>,
    pub repeats: usize,
    pub filler_modes: Vec<FillerMode>,
    pub threshold: f64,
    pub quick: bool,
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            lengths: DEFAULT_LENGTHS.to_vec(),
            depths: DEFAULT_DEPTHS.to_vec(),
            repeats: DEFAULT_REPEATS,
            filler_modes: vec![FillerMode::Distractor, FillerMode::Neutral],
            threshold: DEFAULT_THRESHOLD,
            quick: false,
        }
    }
}

impl GridConfig {
    /// Quick smoke mode: fewer lengths, depths, repeats, distractor only.
    pub fn quick() -> Self {
        Self {
            lengths: vec![1024, 4096, 8192],
            depths: vec![0.0, 0.5, 1.0],
            repeats: 1,
            filler_modes: vec![FillerMode::Distractor],
            threshold: DEFAULT_THRESHOLD,
            quick: true,
        }
    }

    /// Load from environment variables.
    pub fn from_env() -> Self {
        let quick = std::env::var("AKH_BENCH_QUICK").is_ok_and(|v| v != "0");
        if quick {
            return Self::quick();
        }

        let mut cfg = Self::default();

        if let Ok(v) = std::env::var("AKH_ROT_LENGTHS") {
            cfg.lengths = v
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
        }
        if let Ok(v) = std::env::var("AKH_ROT_DEPTHS") {
            cfg.depths = v
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
        }
        if let Ok(v) = std::env::var("AKH_ROT_REPEATS") {
            if let Ok(r) = v.parse() {
                cfg.repeats = r;
            }
        }
        if let Ok(v) = std::env::var("AKH_ROT_FILLER") {
            cfg.filler_modes = match v.as_str() {
                "distractor" => vec![FillerMode::Distractor],
                "neutral" => vec![FillerMode::Neutral],
                _ => vec![FillerMode::Distractor, FillerMode::Neutral],
            };
        }
        if let Ok(v) = std::env::var("AKH_ROT_THRESHOLD") {
            if let Ok(t) = v.parse() {
                cfg.threshold = t;
            }
        }

        cfg
    }

    /// Cap lengths to `min(advertised_window - 512, 32768)`.
    pub fn cap_to_window(&mut self, window: Option<usize>) {
        if let Some(w) = window {
            let max_len = w.saturating_sub(512).min(32768);
            self.lengths.retain(|&l| l <= max_len);
        }
    }

    /// Total number of probes (lengths × depths × repeats × filler_modes).
    pub fn total_probes(&self) -> usize {
        self.lengths.len() * self.depths.len() * self.repeats * self.filler_modes.len()
    }
}

// ---------------------------------------------------------------------------
// Run records + metrics
// ---------------------------------------------------------------------------

/// CSV schema version for `runs.csv` / `summary.csv`. Bump when columns change.
pub const CSV_SCHEMA_VERSION: u32 = 1;

/// One probe result row (maps to `runs.csv`).
#[derive(Debug, Clone, Serialize)]
pub struct RunRow {
    pub model: String,
    pub backend: String,
    pub filler: String,
    pub l_target: usize,
    pub l_actual: Option<u64>,
    pub depth: f64,
    pub repeat: usize,
    pub seed: u64,
    pub hit: bool,
    pub invalid: bool,
    pub latency_ms: u64,
}

/// Per-model×filler summary (maps to `summary.csv`).
#[derive(Debug, Clone, Serialize)]
pub struct SummaryRow {
    pub model: String,
    pub filler: String,
    pub effective_length: usize,
    pub positional_sensitivity_max: f64,
    pub per_length_sensitivity: String,
}

/// Truncation detector: a cell is invalid if `l_actual` is > 20% below
/// `l_target`. Backends that don't report the actual prompt token count
/// (`None`) cannot be truncated-detected this way.
fn is_truncated(l_target: usize, l_actual: Option<u64>) -> bool {
    match l_actual {
        Some(actual) => {
            let threshold = (l_target as f64) * 0.8;
            (actual as f64) < threshold
        }
        None => false, // can't detect if backend doesn't report
    }
}

/// Compute hit_rate per (L, depth) across repeats, excluding invalid cells.
fn hit_rate(rows: &[RunRow], model: &str, filler: &str, l: usize, d: f64) -> Option<f64> {
    let cells: Vec<&RunRow> = rows
        .iter()
        .filter(|r| {
            r.model == model
                && r.filler == filler
                && r.l_target == l
                && (r.depth - d).abs() < 1e-6
                && !r.invalid
        })
        .collect();
    if cells.is_empty() {
        return None;
    }
    let hits = cells.iter().filter(|r| r.hit).count();
    Some(hits as f64 / cells.len() as f64)
}

/// Compute positional_sensitivity(L) = max(0, edge_rate - mid_rate) / max(edge_rate, ε).
///
/// Edge = {0.0, 0.1, 0.9, 1.0}, mid = {0.25, 0.5, 0.75}.
fn positional_sensitivity(
    rows: &[RunRow],
    model: &str,
    filler: &str,
    l: usize,
) -> f64 {
    let edge_rates: Vec<f64> = EDGE_DEPTHS
        .iter()
        .filter_map(|&&d| hit_rate(rows, model, filler, l, d))
        .collect();
    let mid_rates: Vec<f64> = MID_DEPTHS
        .iter()
        .filter_map(|&&d| hit_rate(rows, model, filler, l, d))
        .collect();

    if edge_rates.is_empty() || mid_rates.is_empty() {
        return 0.0;
    }

    let edge_rate: f64 = edge_rates.iter().sum::<f64>() / edge_rates.len() as f64;
    let mid_rate: f64 = mid_rates.iter().sum::<f64>() / mid_rates.len() as f64;

    let eps = 1e-9;
    (edge_rate - mid_rate).max(0.0) / edge_rate.max(eps)
}

/// Compute effective_length: largest L (ascending) whose mean hit-rate across
/// every swept depth is ≥ `threshold`.
///
/// **Truncation-resilient**: unlike the parent draft, a single truncated or
/// failing length does not end the search — lengths whose cells are *all*
/// invalid are skipped rather than treated as a cliff, and the largest valid
/// passing L is kept. The practical MECW is the largest length the model still
/// passes at, not the first length that happened to truncate or miss (recall
/// need not be a monotone cliff).
fn effective_length(
    rows: &[RunRow],
    model: &str,
    filler: &str,
    depths: &[f64],
    threshold: f64,
) -> usize {
    let mut lengths: Vec<usize> = rows
        .iter()
        .filter(|r| r.model == model && r.filler == filler && !r.invalid)
        .map(|r| r.l_target)
        .collect();
    lengths.sort();
    lengths.dedup();

    let mut best = 0usize;
    for &l in &lengths {
        // Mean hit_rate across all swept depths at this length.
        let rates: Vec<f64> = depths
            .iter()
            .filter_map(|&&d| hit_rate(rows, model, filler, l, d))
            .collect();
        if rates.is_empty() {
            // All cells for this length were invalid (truncated) — not evidence.
            continue;
        }
        let mean_rate: f64 = rates.iter().sum::<f64>() / rates.len() as f64;
        if mean_rate >= threshold {
            best = l;
        }
        // Do NOT break: a single failure (truncation, one bad needle) must not
        // preclude a larger length from still passing.
    }
    best
}

// ---------------------------------------------------------------------------
// Grid runner
// ---------------------------------------------------------------------------

/// Run the full grid against a probe, returning per-probe run rows.
///
/// Errors with [`RotBenchError::EmptyGrid`] when the configuration sweeps no
/// lengths (e.g. after `cap_to_window` pruned every length).
pub fn run_grid(probe: &mut dyn RotProbe, cfg: &GridConfig) -> miette::Result<Vec<RunRow>> {
    if cfg.lengths.is_empty() || cfg.depths.is_empty() {
        return Err(RotBenchError::EmptyGrid.into());
    }

    let counter = HeuristicTokenCounter;
    let model = probe.label().to_string();
    let backend_label = probe.backend();

    let mut rows = Vec::with_capacity(cfg.total_probes());

    for &filler_mode in &cfg.filler_modes {
        for &target_len in &cfg.lengths {
            for &depth in &cfg.depths {
                for rep in 0..cfg.repeats {
                    // Seed: deterministic per (model, filler, L, depth, repeat).
                    let seed = {
                        let mut h: u64 = 0xcbf29ce484222325; // FNV offset
                        h ^= model.len() as u64;
                        h = h.wrapping_mul(0x100000001b3);
                        h ^= filler_mode.as_str().len() as u64;
                        h = h.wrapping_mul(0x100000001b3);
                        h ^= target_len as u64;
                        h = h.wrapping_mul(0x100000001b3);
                        let depth_bits = (depth * 10000.0) as u64;
                        h ^= depth_bits;
                        h = h.wrapping_mul(0x100000001b3);
                        h ^= rep as u64;
                        h = h.wrapping_mul(0x100000001b3);
                        h
                    };

                    let mut rng = StdRng::seed_from_u64(seed);
                    let needle = Needle::generate(&mut rng);
                    let user_prompt = build_haystack(
                        &mut rng,
                        &needle,
                        target_len,
                        depth,
                        filler_mode,
                        &counter,
                    );

                    let system = "Answer only from the provided text. Reply with the answer and nothing else.";

                    let started = Instant::now();
                    let resp = probe.complete(system, &user_prompt, 64)?;
                    let latency_ms = started.elapsed().as_millis() as u64;

                    let hit = needle.grade(&resp.text);
                    let invalid = is_truncated(target_len, resp.prompt_tokens);

                    rows.push(RunRow {
                        model: model.clone(),
                        backend: backend_label.to_string(),
                        filler: filler_mode.as_str().to_string(),
                        l_target: target_len,
                        l_actual: resp.prompt_tokens,
                        depth,
                        repeat: rep,
                        seed,
                        hit,
                        invalid,
                        latency_ms,
                    });

                    // Progress.
                    let total = cfg.total_probes();
                    eprintln!(
                        "  [{}/{}] {} L={} d={:.2} r={} → hit={} invalid={} ({}ms)",
                        rows.len(),
                        total,
                        filler_mode.as_str(),
                        target_len,
                        depth,
                        rep,
                        hit,
                        invalid,
                        latency_ms,
                    );
                }
            }
        }
    }

    Ok(rows)
}

// ---------------------------------------------------------------------------
// CSV emission + stdout table
// ---------------------------------------------------------------------------

/// `#`-prefixed metadata header lines shared by both CSVs. Schema version +
/// source/git provenance so a row is always traceable to the code that made it.
fn metadata_lines() -> Vec<(String, String)> {
    let mut lines = vec![
        ("schema_version".to_string(), CSV_SCHEMA_VERSION.to_string()),
        ("source".to_string(), "bench_context_rot".to_string()),
        ("git_commit".to_string(), env!("VERGEN_GIT_SHA").to_string()),
    ];
    if let Ok(branch) = std::env::var("AKH_BENCH_GIT_BRANCH") {
        lines.push(("git_branch".to_string(), branch));
    }
    lines
}

/// Write per-probe rows to `dir/runs.csv`.
pub fn emit_runs_csv(rows: &[RunRow], dir: &Path) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut f = std::fs::File::create(dir.join("runs.csv"))?;
    for (k, v) in metadata_lines() {
        writeln!(f, "# {k}={v}")?;
    }
    writeln!(
        f,
        "model,backend,filler,l_target,l_actual,depth,repeat,seed,hit,invalid,latency_ms"
    )?;
    for r in rows {
        writeln!(
            f,
            "{},{},{},{},{},{},{},{},{},{},{}",
            r.model,
            r.backend,
            r.filler,
            r.l_target,
            r.l_actual.map(|v| v.to_string()).unwrap_or_default(),
            r.depth,
            r.repeat,
            r.seed,
            r.hit,
            r.invalid,
            r.latency_ms,
        )?;
    }
    Ok(())
}

/// Write per-model×filler summaries to `dir/summary.csv`.
pub fn emit_summary_csv(rows: &[RunRow], cfg: &GridConfig, dir: &Path) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut f = std::fs::File::create(dir.join("summary.csv"))?;
    for (k, v) in metadata_lines() {
        writeln!(f, "# {k}={v}")?;
    }
    writeln!(
        f,
        "model,filler,effective_length,positional_sensitivity_max,per_length_sensitivity"
    )?;

    let summaries = summarize(rows, cfg);
    for s in &summaries {
        writeln!(
            f,
            "{},{},{},{:.4},\"{}\"",
            s.model, s.filler, s.effective_length, s.positional_sensitivity_max, s.per_length_sensitivity,
        )?;
    }
    Ok(())
}

/// Compute summary rows from run rows.
pub fn summarize(rows: &[RunRow], cfg: &GridConfig) -> Vec<SummaryRow> {
    // Collect (model, filler) pairs.
    let mut pairs: Vec<(String, String)> = Vec::new();
    for r in rows {
        let pair = (r.model.clone(), r.filler.clone());
        if !pairs.contains(&pair) {
            pairs.push(pair);
        }
    }

    pairs
        .iter()
        .map(|(model, filler)| {
            let eff = effective_length(rows, model, filler, &cfg.depths, cfg.threshold);

            // Per-length sensitivity.
            let mut lengths: Vec<usize> = rows
                .iter()
                .filter(|r| r.model == *model && r.filler == *filler && !r.invalid)
                .map(|r| r.l_target)
                .collect();
            lengths.sort();
            lengths.dedup();

            let per_len: Vec<String> = lengths
                .iter()
                .map(|&l| {
                    let sens = positional_sensitivity(rows, model, filler, l);
                    format!("L={l}:{sens:.3}")
                })
                .collect();

            let max_sens = lengths
                .iter()
                .map(|&l| positional_sensitivity(rows, model, filler, l))
                .fold(0.0f64, f64::max);

            SummaryRow {
                model: model.clone(),
                filler: filler.clone(),
                effective_length: eff,
                positional_sensitivity_max: max_sens,
                per_length_sensitivity: per_len.join("; "),
            }
        })
        .collect()
}

/// Render the summary as a printable text table.
pub fn format_table(summaries: &[SummaryRow]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:<24} {:<12} {:>14} {:>22} {:>30}\n",
        "model", "filler", "effective_length", "max_pos_sensitivity", "per-L sensitivity"
    ));
    out.push_str(&"-".repeat(104));
    out.push('\n');
    for s in summaries {
        out.push_str(&format!(
            "{:<24} {:<12} {:>14} {:>21.4} {:>30}\n",
            s.model,
            s.filler,
            s.effective_length,
            s.positional_sensitivity_max,
            s.per_length_sensitivity,
        ));
    }
    out
}

// ---------------------------------------------------------------------------
// Tests (pure logic, no live LLM required)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn row(model: &str, filler: &str, l: usize, d: f64, hit: bool, invalid: bool) -> RunRow {
        RunRow {
            model: model.into(),
            backend: "ollama".into(),
            filler: filler.into(),
            l_target: l,
            l_actual: Some((l as f64 * 0.95) as u64),
            depth: d,
            repeat: 0,
            seed: 0,
            hit,
            invalid,
            latency_ms: 10,
        }
    }

    #[test]
    fn needle_generation_is_deterministic() {
        let mut rng1 = StdRng::seed_from_u64(42);
        let mut rng2 = StdRng::seed_from_u64(42);
        let n1 = Needle::generate(&mut rng1);
        let n2 = Needle::generate(&mut rng2);
        assert_eq!(n1.access_code, n2.access_code);
        assert_eq!(n1.vault_name, n2.vault_name);
        assert_eq!(n1.sentence, n2.sentence);
    }

    #[test]
    fn needle_grading_is_case_insensitive() {
        let mut rng = StdRng::seed_from_u64(1);
        let needle = Needle::generate(&mut rng);
        assert!(needle.grade(&format!("The code is {}", needle.access_code)));
        assert!(needle.grade(&format!("The code is {}", needle.access_code.to_lowercase())));
        assert!(!needle.grade("The code is WRONG"));
    }

    #[test]
    fn haystack_is_deterministic_same_seed() {
        let counter = HeuristicTokenCounter;
        let mut rng1 = StdRng::seed_from_u64(100);
        let mut rng2 = StdRng::seed_from_u64(100);
        let n1 = Needle::generate(&mut rng1);
        let n2 = Needle::generate(&mut rng2);
        assert_eq!(n1.sentence, n2.sentence);

        let h1 = build_haystack(&mut rng1, &n1, 512, 0.5, FillerMode::Distractor, &counter);
        let h2 = build_haystack(&mut rng2, &n2, 512, 0.5, FillerMode::Distractor, &counter);
        assert_eq!(h1, h2);
    }

    #[test]
    fn haystack_needle_at_depth_zero_is_first() {
        let counter = HeuristicTokenCounter;
        let mut rng = StdRng::seed_from_u64(7);
        let needle = Needle::generate(&mut rng);
        let h = build_haystack(&mut rng, &needle, 256, 0.0, FillerMode::Neutral, &counter);
        let needle_pos = h.find(&needle.sentence).unwrap();
        // At depth 0.0, needle should be in the first 10% of the text.
        assert!(needle_pos < h.len() / 10, "needle at depth 0 should be near the start, was at {needle_pos}/{}", h.len());
    }

    #[test]
    fn haystack_needle_at_depth_one_is_last() {
        let counter = HeuristicTokenCounter;
        let mut rng = StdRng::seed_from_u64(7);
        let needle = Needle::generate(&mut rng);
        let h = build_haystack(&mut rng, &needle, 256, 1.0, FillerMode::Neutral, &counter);
        let needle_pos = h.find(&needle.sentence).unwrap();
        // At depth 1.0, needle should be in the last 20% of the text (before question).
        assert!(needle_pos > h.len() * 4 / 5, "needle at depth 1.0 should be near the end, was at {needle_pos}/{}", h.len());
    }

    #[test]
    fn haystack_contains_question_at_end() {
        let counter = HeuristicTokenCounter;
        let mut rng = StdRng::seed_from_u64(3);
        let needle = Needle::generate(&mut rng);
        let h = build_haystack(&mut rng, &needle, 256, 0.5, FillerMode::Neutral, &counter);
        assert!(h.ends_with(&needle.question), "haystack should end with the question");
    }

    #[test]
    fn truncation_detector_fires_on_short_actual() {
        assert!(is_truncated(8192, Some(4000))); // 4000 < 8192*0.8
        assert!(!is_truncated(8192, Some(7000))); // 7000 > 8192*0.8
        assert!(!is_truncated(8192, None)); // can't detect
    }

    #[test]
    fn hit_rate_excludes_invalid_cells() {
        let mut rows = vec![
            row("m", "distractor", 1024, 0.0, true, false),
            row("m", "distractor", 1024, 0.0, false, false),
            row("m", "distractor", 1024, 0.0, true, true),
        ];
        // Make the third cell's l_actual genuinely truncated.
        rows[2].l_actual = Some(100);
        let rate = hit_rate(&rows, "m", "distractor", 1024, 0.0).unwrap();
        // 2 valid cells, 1 hit → 0.5
        assert!((rate - 0.5).abs() < 1e-6);
    }

    #[test]
    fn effective_length_finds_threshold_crossing() {
        // At L=1024 all hits → rate 1.0 ≥ 0.8 ✓. At L=2048 2/3 ≈ 0.67 < 0.8 ✗.
        let rows = vec![
            row("m", "distractor", 1024, 0.0, true, false),
            row("m", "distractor", 1024, 0.5, true, false),
            row("m", "distractor", 1024, 1.0, true, false),
            row("m", "distractor", 2048, 0.0, true, false),
            row("m", "distractor", 2048, 0.5, false, false),
            row("m", "distractor", 2048, 1.0, true, false),
        ];
        let eff = effective_length(&rows, "m", "distractor", DEFAULT_DEPTHS, 0.8);
        assert_eq!(eff, 1024);
    }

    #[test]
    fn effective_length_survives_a_truncated_gap() {
        // Regression for the parent's monotone-cliff break: L=2048 is entirely
        // truncated (no valid cells) but L=4096 passes — effective length must
        // be 4096, not stop at 1024.
        let rows = vec![
            row("m", "distractor", 1024, 0.0, true, false),
            row("m", "distractor", 1024, 0.5, true, false),
            row("m", "distractor", 2048, 0.0, true, true),
            row("m", "distractor", 2048, 0.5, true, true),
            row("m", "distractor", 4096, 0.0, true, false),
            row("m", "distractor", 4096, 0.5, true, false),
        ];
        let eff = effective_length(&rows, "m", "distractor", DEFAULT_DEPTHS, 0.8);
        assert_eq!(eff, 4096, "a truncated middle length must not cap effective_length");
    }

    #[test]
    fn positional_sensitivity_detects_middle_blindness() {
        // Edge depths all hit, mid depths all miss → sensitivity ≈ 1.0.
        let mut rows = Vec::new();
        for &d in EDGE_DEPTHS {
            rows.push(row("m", "distractor", 4096, d, true, false));
        }
        for &d in MID_DEPTHS {
            rows.push(row("m", "distractor", 4096, d, false, false));
        }
        let sens = positional_sensitivity(&rows, "m", "distractor", 4096);
        assert!((sens - 1.0).abs() < 1e-6, "full middle blindness → sensitivity 1.0, got {sens}");
    }

    #[test]
    fn positional_sensitivity_zero_when_uniform() {
        // All depths hit → sensitivity = 0.
        let mut rows = Vec::new();
        for &d in DEFAULT_DEPTHS {
            rows.push(row("m", "distractor", 4096, d, true, false));
        }
        let sens = positional_sensitivity(&rows, "m", "distractor", 4096);
        assert!(sens.abs() < 1e-6, "uniform accuracy → sensitivity 0, got {sens}");
    }

    #[test]
    fn grid_config_quick_mode() {
        let cfg = GridConfig::quick();
        assert!(cfg.quick);
        assert_eq!(cfg.lengths, vec![1024, 4096, 8192]);
        assert_eq!(cfg.depths, vec![0.0, 0.5, 1.0]);
        assert_eq!(cfg.repeats, 1);
        assert_eq!(cfg.filler_modes, vec![FillerMode::Distractor]);
        assert_eq!(cfg.total_probes(), 9);
    }

    #[test]
    fn grid_config_caps_to_window() {
        let mut cfg = GridConfig::default();
        cfg.cap_to_window(Some(4096));
        // max_len = min(4096-512, 32768) = 3584
        assert!(cfg.lengths.iter().all(|&l| l <= 3584));
        assert!(!cfg.lengths.is_empty());
    }

    #[test]
    fn grid_config_caps_to_small_window() {
        let mut cfg = GridConfig::default();
        cfg.cap_to_window(Some(1024));
        // max_len = min(1024-512, 32768) = 512 — all default lengths exceed it.
        assert!(cfg.lengths.is_empty());
    }

    #[test]
    fn run_grid_rejects_empty_config() {
        struct Noop;
        impl RotProbe for Noop {
            fn label(&self) -> &str { "noop" }
            fn backend(&self) -> &str { "mock" }
            fn context_window(&self) -> Option<usize> { None }
            fn complete(&mut self, _: &str, _: &str, _: u32) -> miette::Result<ProbeResponse> {
                Ok(ProbeResponse { text: String::new(), prompt_tokens: None })
            }
        }
        let mut probe = Noop;
        let cfg = GridConfig { lengths: vec![], ..GridConfig::quick() };
        let err = run_grid(&mut probe, &cfg).unwrap_err();
        assert!(err.to_string().contains("No lengths"), "expected EmptyGrid, got {err}");
    }

    #[test]
    fn dry_run_emits_schema_versioned_csv() {
        /// Mock probe: echoes the vault code when it can see it, so `hit` is
        /// always true and every metric has data — plumbing, not LLM quality.
        struct Mock;
        impl RotProbe for Mock {
            fn label(&self) -> &str { "mock" }
            fn backend(&self) -> &str { "mock" }
            fn context_window(&self) -> Option<usize> { None }
            fn complete(&mut self, _sys: &str, user: &str, _max: u32) -> miette::Result<ProbeResponse> {
                // Recover the needle's "is XXXXXX." tail from the prompt.
                let text = user
                    .lines()
                    .find_map(|l| l.strip_prefix("The access code for ").and_then(|l| l.split(" is ").nth(1)))
                    .map(|tail| tail.trim_end_matches('.').to_string())
                    .unwrap_or_default();
                Ok(ProbeResponse {
                    text,
                    prompt_tokens: Some(HeuristicTokenCounter.count(user) as u64),
                })
            }
        }

        let dir = std::env::temp_dir().join(format!("akh-rot-test-{}", std::process::id()));
        let mut probe = Mock;
        let mut cfg = GridConfig::quick();
        cfg.cap_to_window(probe.context_window());
        let rows = run_grid(&mut probe, &cfg).expect("grid");
        assert_eq!(rows.len(), cfg.total_probes());

        emit_runs_csv(&rows, &dir).expect("runs csv");
        emit_summary_csv(&rows, &cfg, &dir).expect("summary csv");

        let runs = std::fs::read_to_string(dir.join("runs.csv")).unwrap();
        let summary = std::fs::read_to_string(dir.join("summary.csv")).unwrap();

        // Schema-versioned header with source provenance.
        assert!(runs.starts_with("# schema_version=1\n"), "runs.csv header:\n{runs}");
        assert!(runs.contains("# source=bench_context_rot\n"));
        assert!(runs.contains("# git_commit="));
        assert!(summary.starts_with("# schema_version=1\n"));

        // One line per probe + header lines.
        let header = runs.lines().find(|l| !l.starts_with('#')).unwrap();
        assert!(header.starts_with("model,backend,filler,"));
        let data_lines = runs.lines().filter(|l| !l.starts_with('#')).skip(1).count();
        assert_eq!(data_lines, cfg.total_probes());

        // Mock sees every needle → effective_length is the max swept length.
        let summaries = summarize(&rows, &cfg);
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].effective_length, *cfg.lengths.iter().max().unwrap());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn summarize_produces_one_row_per_model_filler() {
        let rows = vec![
            row("m1", "distractor", 1024, 0.0, true, false),
            row("m1", "neutral", 1024, 0.0, true, false),
        ];
        let cfg = GridConfig::quick();
        let summaries = summarize(&rows, &cfg);
        assert_eq!(summaries.len(), 2);
    }
}
