//! Candle-based LLM backend for GGUF inference with hidden state access.
//!
//! Replaces `llama-cpp-2` with Candle's pure-Rust inference engine, giving us:
//!
//! 1. **Hidden state extraction** — access intermediate tensor representations
//!    for the neural→VSA bridge (Phase 26d)
//! 2. **Full Rust control** — KV cache is an explicit Rust struct (not opaque C),
//!    enabling PolarQuant compression (Phase 26c)
//! 3. **Metal support** — Apple M-series GPU acceleration via Candle's Metal backend
//!
//! The backend implements the same input/output boundary as [`LlmTranslator`]:
//! NL→AbsTree for NLU Tier 3, and symbols→NL for dialogue generation.

use std::path::{Path, PathBuf};

use super::error::{NluError, NluResult};
use super::llm_translator::{build_prompt_with_context, parse_abstree_json, LlmTranslation};

#[cfg(feature = "candle-backend")]
use candle_core::{Device, IndexOp, Tensor};

/// Which model family is loaded — drives chat-prompt formatting and EOS tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    /// Qwen2.5 (ChatML `<|im_start|>` template, `<|im_end|>` / `<|endoftext|>`).
    Qwen2,
    /// Gemma 2/3 (`<start_of_turn>` template, `<end_of_turn>` EOS).
    Gemma,
    /// Gemma 4 — unified encoder-free multimodal (native audio). Shares Gemma's
    /// `<start_of_turn>` template/EOS but a distinct decoder (per-layer attention
    /// geometry, partial rotary, GELU MLP, logit softcap). See ADR 054.
    Gemma4,
}

impl ModelFamily {
    /// Detect the family from a GGUF `general.architecture` string.
    #[cfg(feature = "candle-backend")]
    fn from_architecture(arch: &str) -> Self {
        let a = arch.to_lowercase();
        // Check the more specific `gemma4` before the generic `gemma` substring.
        if a.contains("gemma4") {
            ModelFamily::Gemma4
        } else if a.contains("gemma") {
            ModelFamily::Gemma
        } else {
            ModelFamily::Qwen2
        }
    }
}

/// A loaded quantized model, dispatched by family. Uses the vendored model
/// loaders (see [`crate::nlu::vendored`]) which expose hidden states. Both
/// backends share the same `forward(x, index_pos)` / `forward_with_hidden`
/// contract, so generation is family-agnostic.
#[cfg(feature = "candle-backend")]
enum LoadedModel {
    Qwen2(super::vendored::quantized_qwen2::ModelWeights),
    Gemma(super::vendored::quantized_gemma3::ModelWeights),
    Gemma4(super::vendored::quantized_gemma4::ModelWeights),
}

#[cfg(feature = "candle-backend")]
impl LoadedModel {
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> candle_core::Result<Tensor> {
        match self {
            LoadedModel::Qwen2(m) => m.forward(x, index_pos),
            LoadedModel::Gemma(m) => m.forward(x, index_pos),
            LoadedModel::Gemma4(m) => m.forward(x, index_pos),
        }
    }

    /// Run a forward pass returning `(logits, hidden_state)` where `hidden_state`
    /// is the post-norm last-position representation `[batch, hidden_dim]`.
    fn forward_with_hidden(
        &mut self,
        x: &Tensor,
        index_pos: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        match self {
            LoadedModel::Qwen2(m) => m.forward_with_hidden(x, index_pos),
            LoadedModel::Gemma(m) => m.forward_with_hidden(x, index_pos),
            LoadedModel::Gemma4(m) => m.forward_with_hidden(x, index_pos),
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Candle LLM backend.
#[derive(Debug, Clone)]
pub struct CandleConfig {
    /// Maximum tokens to generate per call (default: 512).
    pub max_tokens: u32,
    /// Sampling temperature (0.0 = greedy, default: 0.0).
    pub temperature: f32,
    /// Whether to capture hidden states during forward passes.
    pub extract_hidden: bool,
    /// Which layer to extract hidden states from (None = last layer).
    pub hidden_layer: Option<usize>,
}

impl Default for CandleConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.0,
            extract_hidden: false,
            hidden_layer: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Inference result
// ---------------------------------------------------------------------------

/// Result of a Candle inference run.
#[derive(Debug, Clone)]
pub struct CandleInferenceResult {
    /// Generated text.
    pub text: String,
    /// Number of tokens generated.
    pub tokens_generated: u32,
    /// Parsed AbsTree (if this was an NLU translation).
    pub abs_tree: Option<crate::grammar::abs::AbsTree>,
    /// Final hidden state tensor dimensions (if extracted).
    /// Stored as `[hidden_dim]` f32 values.
    pub final_hidden: Option<Vec<f32>>,
}

/// Decode-throughput statistics for the Phase 45a benchmark harness (ADR 073).
///
/// Exposes the raw token count and decode wall-time so callers can compute tok/s
/// and compare against the memory-bandwidth roofline — the metric by which the
/// HIP backend's speed is judged. Prefill is excluded so the number reflects
/// steady-state decode throughput, which is what the bus-bound op-offload path
/// (and the device-resident successor) actually differ on.
#[derive(Debug, Clone)]
pub struct GenBenchStats {
    /// Generated text (uncleaned).
    pub text: String,
    /// Tokens decoded (prefill excluded).
    pub tokens_generated: u32,
    /// Wall-clock in the decode loop, seconds (prefill excluded).
    pub decode_secs: f32,
}

impl GenBenchStats {
    /// Decode throughput in tokens/second (0 when nothing was decoded).
    pub fn tok_per_s(&self) -> f32 {
        if self.decode_secs > 0.0 {
            self.tokens_generated as f32 / self.decode_secs
        } else {
            0.0
        }
    }
}

/// Stats from the Phase 46c fixed-position HIP-graph capture/replay probe.
///
/// The decisive, cheap measurement for whether HIP-graph replay can close the
/// Candle↔Ollama decode gap. At a **fixed** decode position it compares eager
/// `forward_from_embed` (the per-token kernel storm, host-enqueue-gapped) against a single
/// `hipGraphLaunch` replay of the same forward. This is the *upper bound* on the graph win
/// — no per-token KV write / growing shapes to update — so if replay does not clearly beat
/// eager here, the dispatch-idle hypothesis is wrong and the full per-token graph decode is
/// not worth building (revert + record). If it wins big, that headroom pays for the
/// per-token update overhead the real integration needs.
#[cfg(all(feature = "candle-backend", feature = "hip"))]
#[derive(Debug, Clone)]
pub struct GraphBenchStats {
    /// Fixed decode position the forward was captured at (= prompt length).
    pub position: usize,
    /// Iterations timed for each of eager / replay (after warm-up).
    pub iters: u32,
    /// Eager `forward_from_embed` throughput at the fixed position (tok/s), drained
    /// per iteration so it reflects real dispatch+execute cost.
    pub eager_tok_s: f32,
    /// Single-dispatch graph-replay throughput at the fixed position (tok/s); 0 if
    /// capture failed (see [`captured`](Self::captured)).
    pub replay_tok_s: f32,
    /// hipMalloc cache *misses* on a *warm* eager forward (after warm-up) — diagnostic:
    /// > 0 means the forward still issues a real (host-synchronizing) hipMalloc through
    /// our caching pool even at a fixed position, which would abort graph capture.
    pub warm_malloc_miss: u64,
    /// Total caching-allocator calls on that warm forward (context for the miss count).
    pub warm_malloc_calls: u64,
    /// Whether the forward was successfully captured into a HIP graph.
    pub captured: bool,
    /// Capture failure reason, if any (e.g. a hipMalloc / rocBLAS-internal alloc that
    /// invalidated the capture).
    pub capture_error: Option<String>,
    /// Whether the replayed logits' argmax (greedy token) matches a *fresh* eager
    /// forward's at the same KV state (only meaningful when [`captured`](Self::captured)).
    pub argmax_matches: bool,
    /// Max absolute difference between the last replay and a fresh eager forward
    /// (byte-coherence check; the correctness gate).
    pub max_abs_diff: f32,
    /// Max absolute difference between the first and last replay outputs — 0 ⇒ replay is
    /// deterministic across launches; > 0 ⇒ a race (cross-stream / reused buffer).
    pub replay_determinism: f32,
    /// Max absolute difference between an eager forward run on a *private stream* (no
    /// graph) and the default-stream forward — 0 ⇒ every op honors the launch stream (a
    /// replay mismatch is then a graph-machinery/allocator fault); > 0 ⇒ some op ignored
    /// the stream and raced (the cause of the replay mismatch).
    pub stream_eager_diff: f32,
}

/// Byte-exactness of the Phase 46d-2c **graph-ready** decode forward — the
/// [`GraphDecodeCtx`](super::vendored::quantized_gemma3::GraphDecodeCtx)-driven path
/// (device-pos rope `index_select`, `kv_scatter_at`, fixed block-padded `read_len` + a
/// device-built keep-mask) run **eagerly** at a fixed position — against the normal decode
/// forward at the same position. The 46d-2c gate: both diffs must be `0` before adding
/// capture/replay (46d-2d). If they are not, the graph-ready math diverged and capturing it
/// would replay a wrong forward.
#[derive(Debug, Clone)]
pub struct GraphDecodeParity {
    /// Fixed decode position the forwards ran at (= prompt length).
    pub position: usize,
    /// max|Δlogits| between normal decode and the graph-ready forward reading the EXACT
    /// `read_len = pos + 1` (no padding) — isolates the device-pos rope + scatter from the
    /// padding/mask path. Expected `0`.
    pub exact_diff: f32,
    /// max|Δlogits| between normal decode and the graph-ready forward reading a PADDED
    /// `read_len > pos + 1` with the block tail masked — proves the fixed-shape + masked-
    /// tail attention is bit-identical to attention over the valid prefix. Expected `0`.
    pub padded_diff: f32,
    /// The block-padded `read_len` used for [`padded_diff`](Self::padded_diff).
    pub padded_read_len: usize,
    /// Whether the device-pos `kv_scatter_at` write path (hip) was exercised (vs the
    /// `slice_set` fallback). Always true in a `hip` build reaching the cuda KV path.
    pub used_device_scatter: bool,
}

/// Result of the Phase 46d-2d **real per-token graph decode loop** — the payoff measurement
/// for the whole 46d effort. One HIP graph is captured for the decode forward and replayed
/// once per token (advancing the device position + refreshing the fixed embedding buffer
/// between replays), generating a full sequence in single-dispatch steps instead of the
/// per-token kernel storm. Compared against eager generation of the same sequence.
///
/// The gate: [`tokens_match`](Self::tokens_match) must be true (graph replay decodes the
/// **same** greedy tokens as eager — correctness) and `graph_tok_s` should beat
/// `eager_tok_s` (the speed payoff). Per the measure-or-revert discipline, if replay does
/// not beat eager the integration is reverted and the finding recorded.
#[derive(Debug, Clone)]
pub struct GraphDecodeLoopStats {
    /// Prompt length (the first decode position).
    pub position: usize,
    /// Tokens requested.
    pub requested_new: usize,
    /// Tokens actually decoded (≤ requested; capped by the static cache or an early EOS).
    pub decoded: usize,
    /// Eager generation throughput (`forward_from_embed` per token, exact KV prefix) — the
    /// real baseline (tok/s).
    pub eager_tok_s: f32,
    /// Graph-replay generation throughput (one `hipGraphLaunch` per token) — the new path
    /// (tok/s).
    pub graph_tok_s: f32,
    /// hipMalloc cache *misses* on the warm forward right before capture — `0` ⇒ the forward
    /// is allocation-stable and the capture is clean.
    pub capture_miss: u64,
    /// Whether the graph-replayed greedy token sequence is identical to the eager one — the
    /// correctness gate. A captured graph that decodes different tokens is wrong.
    pub tokens_match: bool,
}

/// Phase 47a gate: byte-exactness of the **all-position verify forward** (the speculative-decode
/// primitive) against incremental single-token decode. The target scores `seq_len` candidate
/// positions in one forward (`forward_verify` → `[1, seq, vocab]`); each position's logits must
/// equal what a sequence of single-token `forward`s would produce at the same position (same K/V
/// writes, same per-position rope/mask). Both diffs must be `0` before the verify forward is used
/// to accept/reject draft tokens (47c) — a wrong verify forward accepts wrong tokens.
#[derive(Debug, Clone)]
pub struct VerifyParity {
    /// Prefix length the verify block was appended at (`index_pos`).
    pub position: usize,
    /// Number of candidate positions scored in the one verify forward.
    pub seq_len: usize,
    /// max over ALL positions of max|Δlogits| (verify vs incremental). `0` only at `seq_len==1`
    /// (both use the m=1 GEMV); for `seq_len>1` the verify projections use the m=n GEMM, a
    /// different quant kernel → small (~1e-2) divergence, NOT a bug.
    pub max_abs_diff: f32,
    /// max|Δlogits| at the LAST position.
    pub last_pos_diff: f32,
    /// Positions where `argmax(verify) == argmax(incremental)` — the greedy-token gate. The
    /// logit divergence is harmless iff this equals `seq_len` (the verify picks the same token a
    /// pure m=1 decode would).
    pub argmax_agree: usize,
}

/// Result of the Phase 47c greedy **speculative-decode loop** — the payoff measurement for
/// Phase 47. Compares plain greedy decode against prompt-lookup speculative decode (n-gram
/// drafter + one `forward_verify` per round, accept the matching greedy prefix, advance up to
/// `accepted+1` tokens). The win is workload-conditional: high where output echoes context.
#[derive(Debug, Clone)]
pub struct SpecDecodeStats {
    /// Prompt length (first decode position).
    pub position: usize,
    /// Tokens generated by plain greedy decode (the baseline).
    pub plain_tokens: usize,
    /// Tokens generated by speculative decode (compared over `min(plain, spec)`).
    pub spec_tokens: usize,
    /// Plain greedy decode throughput (tok/s) — one forward per token.
    pub plain_tok_s: f32,
    /// Speculative decode throughput (tok/s) — one `forward_verify` per round.
    pub spec_tok_s: f32,
    /// Verify forwards the spec run issued (rounds).
    pub rounds: usize,
    /// Total draft tokens proposed across all rounds.
    pub drafted: usize,
    /// Total draft tokens accepted (`accepted/drafted` = acceptance rate; `accepted/rounds`
    /// + 1 = mean tokens per forward).
    pub accepted: usize,
    /// Whether the spec-decoded token sequence equals the plain greedy one over the shared
    /// prefix. Usually true (47a: 100% argmax agreement); a rare flip is the m=1 GEMV vs m=n
    /// GEMM quant divergence, not a logic error.
    pub tokens_match: bool,
}

// ---------------------------------------------------------------------------
// Expected model paths
// ---------------------------------------------------------------------------

const LLM_SUBDIR: &str = "models/llm";
const CANDLE_MODEL_FILE: &str = "qwen2.5-1.5b-instruct-q4_k_m.gguf";
/// Fallback prompt ceiling (tokens) used only when a GGUF does not declare
/// `<arch>.context_length`. The real window is read from the model file at load
/// (`CandleBackend::context_length`) — e.g. Gemma-4-12B declares 262144 — so the
/// agentic path's full tool catalog (~6.6K tokens) fits comfortably; the earlier
/// hardcoded cap was wrong. 8192 is a conservative floor for metadata-less GGUFs.
const MAX_PROMPT_TOKENS: usize = 8192;
const TOKENIZER_FILE: &str = "tokenizer.json";

// ---------------------------------------------------------------------------
// CandleBackend
// ---------------------------------------------------------------------------

/// Candle-based GGUF LLM backend.
///
/// Wraps Candle's quantized Qwen2 model with tokenizer and inference loop.
/// Feature-gated behind `candle-backend`.
pub struct CandleBackend {
    #[cfg(feature = "candle-backend")]
    model: LoadedModel,
    #[cfg(feature = "candle-backend")]
    tokenizer: tokenizers::Tokenizer,
    #[cfg(feature = "candle-backend")]
    device: Device,
    /// Model family (Qwen2 / Gemma) — drives chat templating and EOS tokens.
    family: ModelFamily,
    /// Real context window (tokens) read from the GGUF `<arch>.context_length`
    /// metadata — e.g. Gemma-4-12B = 262144. Used as the prompt ceiling and as
    /// the budget window (Phase 44b), replacing a hardcoded guess. Falls back to
    /// [`MAX_PROMPT_TOKENS`] when the metadata is absent.
    context_length: usize,
    config: CandleConfig,
    model_path: PathBuf,
    /// Live, lock-free inference toggles shared (by `Arc`) with this akh's
    /// [`Engine`](crate::engine::Engine) (Phase 47 / ADR 075). `None` when the backend runs
    /// standalone (benches/tests) — eligibility then falls back to the `AKH_SPEC_DECODE` env
    /// flag. Wired by `CandleChatBackend` construction so a runtime tool flip is seen next turn.
    runtime_options: Option<std::sync::Arc<crate::inference_options::InferenceOptions>>,
}

impl std::fmt::Debug for CandleBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandleBackend")
            .field("config", &self.config)
            .field("model_path", &self.model_path)
            .finish_non_exhaustive()
    }
}

impl CandleBackend {
    /// Load a GGUF model and tokenizer from the given paths.
    #[cfg(feature = "candle-backend")]
    pub fn load(model_path: &Path, tokenizer_path: &Path) -> NluResult<Self> {
        Self::load_with_config(model_path, tokenizer_path, CandleConfig::default())
    }

    /// Load with explicit configuration.
    #[cfg(feature = "candle-backend")]
    pub fn load_with_config(
        model_path: &Path,
        tokenizer_path: &Path,
        config: CandleConfig,
    ) -> NluResult<Self> {
        if !model_path.exists() {
            return Err(NluError::ModelNotFound {
                path: model_path.to_path_buf(),
            });
        }
        if !tokenizer_path.exists() {
            return Err(NluError::ModelNotFound {
                path: tokenizer_path.to_path_buf(),
            });
        }

        // Select device: Metal on macOS, CPU elsewhere.
        let device = Self::select_device();

        tracing::info!(
            model = %model_path.display(),
            device = ?device,
            "loading Candle GGUF model"
        );

        // Load GGUF model.
        let mut file = std::fs::File::open(model_path).map_err(|e| NluError::ModelLoadFailed {
            reason: format!("open GGUF: {e}"),
        })?;
        let content =
            candle_core::quantized::gguf_file::Content::read(&mut file).map_err(|e| {
                NluError::ModelLoadFailed {
                    reason: format!("read GGUF: {e}"),
                }
            })?;

        // Detect model family from GGUF metadata before `from_gguf` consumes it.
        let architecture = content
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .cloned()
            .unwrap_or_default();
        let family = ModelFamily::from_architecture(&architecture);
        tracing::info!(architecture = %architecture, family = ?family, "detected GGUF model family");

        // Read the model's real context window from the GGUF metadata before
        // `from_gguf` consumes `content`. e.g. `gemma4.context_length = 262144`.
        // This is the model capability, not a guess — used as the prompt ceiling
        // and the Phase 44 budget window.
        let context_length = content
            .metadata
            .get(&format!("{architecture}.context_length"))
            .and_then(|v| {
                v.to_u32()
                    .map(|x| x as usize)
                    .or_else(|_| v.to_u64().map(|x| x as usize))
                    .or_else(|_| v.to_i32().map(|x| x as usize))
                    .or_else(|_| v.to_i64().map(|x| x as usize))
                    .ok()
            })
            .filter(|&n| n > 0)
            .unwrap_or(MAX_PROMPT_TOKENS);
        tracing::info!(context_length, architecture = %architecture, "model context window (from GGUF)");

        let model = match family {
            ModelFamily::Gemma4 => {
                let m = super::vendored::quantized_gemma4::ModelWeights::from_gguf(
                    content, &mut file, &device,
                )
                .map_err(|e| NluError::ModelLoadFailed {
                    reason: format!("load Gemma 4 weights: {e}"),
                })?;
                LoadedModel::Gemma4(m)
            }
            ModelFamily::Gemma => {
                let m = super::vendored::quantized_gemma3::ModelWeights::from_gguf(
                    content, &mut file, &device,
                )
                .map_err(|e| NluError::ModelLoadFailed {
                    reason: format!("load Gemma weights: {e}"),
                })?;
                LoadedModel::Gemma(m)
            }
            ModelFamily::Qwen2 => {
                let m = super::vendored::quantized_qwen2::ModelWeights::from_gguf(
                    content, &mut file, &device,
                )
                .map_err(|e| NluError::ModelLoadFailed {
                    reason: format!("load Qwen2 weights: {e}"),
                })?;
                LoadedModel::Qwen2(m)
            }
        };

        // Load tokenizer.
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).map_err(|e| {
            NluError::ModelLoadFailed {
                reason: format!("load tokenizer: {e}"),
            }
        })?;

        tracing::info!(family = ?family, "Candle GGUF model loaded successfully");

        Ok(Self {
            model,
            tokenizer,
            device,
            family,
            context_length,
            config,
            model_path: model_path.to_path_buf(),
            runtime_options: None,
        })
    }

    /// Attach this akh's shared [`InferenceOptions`](crate::inference_options::InferenceOptions)
    /// (Phase 47). Once set, [`spec_decode_eligible`](Self::spec_decode_eligible) reads the live
    /// shared flag instead of the `AKH_SPEC_DECODE` env fallback — so the `inference_config` tool
    /// can toggle speculative decode at runtime without a restart.
    pub fn set_inference_options(
        &mut self,
        opts: std::sync::Arc<crate::inference_options::InferenceOptions>,
    ) {
        self.runtime_options = Some(opts);
    }

    /// Override the sampling temperature (0.0 = greedy). Primarily for benches/tests that need to
    /// exercise the temperature path; production sets it via [`CandleConfig`] at load.
    #[cfg(feature = "candle-backend")]
    pub fn set_temperature(&mut self, temperature: f32) {
        self.config.temperature = temperature;
    }

    /// The model's context window (tokens), read from the GGUF
    /// `<arch>.context_length` metadata (fallback [`MAX_PROMPT_TOKENS`]).
    #[cfg(feature = "candle-backend")]
    pub fn context_length(&self) -> usize {
        self.context_length
    }

    /// Non-feature-gated stub.
    #[cfg(not(feature = "candle-backend"))]
    pub fn load(model_path: &Path, _tokenizer_path: &Path) -> NluResult<Self> {
        if !model_path.exists() {
            return Err(NluError::ModelNotFound {
                path: model_path.to_path_buf(),
            });
        }
        Err(NluError::ModelLoadFailed {
            reason: "candle-backend feature not enabled".to_string(),
        })
    }

    /// Gracefully attempt to load from the standard data directory.
    pub fn try_load(data_dir: &Path) -> Option<Self> {
        let (model_path, tokenizer_path) = Self::discover_model(data_dir)?;
        match Self::load(&model_path, &tokenizer_path) {
            Ok(b) => Some(b),
            Err(e) => {
                tracing::warn!(model = %model_path.display(), error = %e, "Candle model discovered but failed to load");
                None
            }
        }
    }

    /// Gracefully attempt to load with config.
    #[cfg(feature = "candle-backend")]
    pub fn try_load_with_config(data_dir: &Path, config: CandleConfig) -> Option<Self> {
        let (model_path, tokenizer_path) = Self::discover_model(data_dir)?;
        match Self::load_with_config(&model_path, &tokenizer_path, config) {
            Ok(b) => Some(b),
            Err(e) => {
                tracing::warn!(model = %model_path.display(), error = %e, "Candle model discovered but failed to load");
                None
            }
        }
    }

    /// Discover a GGUF model + tokenizer under `data_dir/models/llm/`.
    ///
    /// Selection order: a file whose name matches `AKH_LLM_MODEL` (if set),
    /// then the canonical Qwen filename, then any `.gguf`. The tokenizer is
    /// `tokenizer.json` in the same directory (or `<model-stem>.tokenizer.json`).
    fn discover_model(data_dir: &Path) -> Option<(PathBuf, PathBuf)> {
        // Accept either a base dir (…/<data>) or the shared models dir
        // (…/<data>/models) — the latter already contains `/models`, so try
        // both `models/llm` and `llm` to avoid a doubled `models` segment.
        let dir = [data_dir.join(LLM_SUBDIR), data_dir.join("llm")]
            .into_iter()
            .find(|d| d.is_dir())?;

        // Collect candidate .gguf files. Exclude multimodal projector sidecars
        // (`<model>.mmproj.gguf`, ADR 054) — they are not standalone models and
        // would otherwise match a `contains` hint like `gemma4-12b`.
        let mut ggufs: Vec<PathBuf> = std::fs::read_dir(&dir)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("gguf"))
            .filter(|p| {
                !p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.to_lowercase().contains(".mmproj."))
                    .unwrap_or(false)
            })
            .collect();
        if ggufs.is_empty() {
            // Fall back to the canonical path so the error message is specific.
            let canonical = dir.join(CANDLE_MODEL_FILE);
            if canonical.exists() {
                ggufs.push(canonical);
            } else {
                return None;
            }
        }

        // Prefer a model matching the AKH_LLM_MODEL hint (e.g. "gemma3").
        let hint = std::env::var("AKH_LLM_MODEL").ok();
        let pick = hint
            .as_deref()
            .and_then(|h| {
                let stem = h.split(':').next().unwrap_or(h).to_lowercase();
                // Prefer an exact filename-stem match (`<hint>.gguf`) so `gemma4-12b`
                // does not accidentally pick `gemma4-12b-something.gguf`; fall back to
                // a substring match.
                let exact = format!("{stem}.gguf");
                ggufs
                    .iter()
                    .find(|p| {
                        p.file_name()
                            .and_then(|n| n.to_str())
                            .map(|n| n.to_lowercase() == exact)
                            .unwrap_or(false)
                    })
                    .or_else(|| {
                        ggufs.iter().find(|p| {
                            p.file_name()
                                .and_then(|n| n.to_str())
                                .map(|n| n.to_lowercase().contains(&stem))
                                .unwrap_or(false)
                        })
                    })
            })
            .or_else(|| ggufs.iter().find(|p| {
                p.file_name().and_then(|n| n.to_str()) == Some(CANDLE_MODEL_FILE)
            }))
            .or_else(|| ggufs.first())?
            .clone();

        // Tokenizer: <stem>.tokenizer.json, else tokenizer.json in the dir.
        let stem_tok = pick.with_extension("tokenizer.json");
        let tokenizer = if stem_tok.exists() {
            stem_tok
        } else {
            dir.join(TOKENIZER_FILE)
        };
        Some((pick, tokenizer))
    }

    // ── Input boundary: NL → AbsTree ─────────────────────────────────────

    /// Translate natural language to AbsTree (NLU Tier 3 replacement).
    #[cfg(feature = "candle-backend")]
    pub fn translate(&mut self, input: &str) -> NluResult<LlmTranslation> {
        self.translate_with_context(input, None)
    }

    /// Translate with an optional KG context block prepended to the prompt.
    ///
    /// `kg_context` is typically generated by `nlu::kg_context::build_kg_context`.
    /// When `None` the behaviour is identical to [`translate`]. Kept so the
    /// develop tier-3 cascade (`NluPipeline::tier3_parse`) can ground the parse
    /// in live KG facts (the chat-e2e branch had dropped this overload).
    #[cfg(feature = "candle-backend")]
    pub fn translate_with_context(
        &mut self,
        input: &str,
        kg_context: Option<&str>,
    ) -> NluResult<LlmTranslation> {
        let prompt = build_prompt_with_context(input, kg_context);
        let result = self.generate_inner(&prompt, self.config.max_tokens, StopMode::Json)?;

        // Extract JSON from the output.
        let raw = result.text.trim();
        let json = extract_json_object(raw);

        tracing::debug!(json_len = json.len(), "attempting AbsTree JSON parse");

        let tree = parse_abstree_json(json)?;

        Ok(LlmTranslation {
            json: json.to_string(),
            tree,
            tokens_generated: result.tokens_generated,
        })
    }

    /// Non-feature-gated stub.
    #[cfg(not(feature = "candle-backend"))]
    pub fn translate(&mut self, _input: &str) -> NluResult<LlmTranslation> {
        Err(NluError::LlmGenerationFailed {
            reason: "candle-backend feature not enabled".to_string(),
        })
    }

    /// Non-feature-gated stub.
    #[cfg(not(feature = "candle-backend"))]
    pub fn translate_with_context(
        &mut self,
        _input: &str,
        _kg_context: Option<&str>,
    ) -> NluResult<LlmTranslation> {
        Err(NluError::LlmGenerationFailed {
            reason: "candle-backend feature not enabled".to_string(),
        })
    }

    // ── Output boundary: Symbols → NL ────────────────────────────────────

    /// Generate natural language from a prompt.
    #[cfg(feature = "candle-backend")]
    pub fn generate(&mut self, prompt: &str, max_tokens: u32) -> NluResult<String> {
        let result = self.generate_inner(prompt, max_tokens, StopMode::Text)?;
        let text = clean_generated_text(&result.text);
        tracing::debug!(
            tokens = result.tokens_generated,
            len = text.len(),
            "Candle text generation complete"
        );
        Ok(text)
    }

    /// Generate and return decode-throughput stats (Phase 45a / ADR 073).
    ///
    /// Like [`generate`](Self::generate) (greedy/temperature per config,
    /// [`StopMode::Text`]) but surfaces the token count and decode wall-time so the
    /// `candle_bench` example can report tok/s and % of the bandwidth roofline.
    /// Pair with [`super::hip::reset_pcie_counts`] / [`super::hip::pcie_counts`] to
    /// also measure PCIe crossings per token.
    #[cfg(feature = "candle-backend")]
    pub fn generate_bench(&mut self, prompt: &str, max_tokens: u32) -> NluResult<GenBenchStats> {
        let result = self.generate_inner(prompt, max_tokens, StopMode::Fixed)?;
        Ok(GenBenchStats {
            text: result.text,
            tokens_generated: result.tokens_generated,
            decode_secs: result.decode_secs,
        })
    }

    /// Phase 46c make-or-break probe: capture `forward_from_embed` at a **fixed** decode
    /// position and replay it, comparing eager vs single-dispatch-graph throughput and
    /// logits parity. gemma3 only (the model with the static-KV + embed/forward_from_embed
    /// split). `iters` forwards/replays are timed after warm-up. See [`GraphBenchStats`].
    ///
    /// The flow mirrors the proven `hip_graph_capture_replay_*` spikes scaled to the whole
    /// forward: prefill the prompt to reach a realistic position P, compute the next
    /// token's embedding OUTSIDE the captured region (it does a host-synchronizing D2H+H2D),
    /// warm the caching pool with eager forwards, then capture the forward at P (every
    /// allocation now hits the warm pool → zero hipMalloc → capturable) and replay it.
    #[cfg(all(feature = "candle-backend", feature = "hip"))]
    pub fn bench_graph_replay(&mut self, prompt: &str, iters: u32) -> NluResult<GraphBenchStats> {
        use candle_core::cuda;

        let iters = iters.max(1);
        let fail = |reason: String| NluError::LlmGenerationFailed { reason };

        // --- Prefill to a realistic position P (mirrors generate_inner) -------------
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| fail(format!("tokenization: {e}")))?;
        let mut prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
        if matches!(self.family, ModelFamily::Gemma | ModelFamily::Gemma4) {
            if let Some(bos) = self.tokenizer.token_to_id("<bos>") {
                if prompt_tokens.first() != Some(&bos) {
                    prompt_tokens.insert(0, bos);
                }
            }
        }
        let position = prompt_tokens.len();
        let input = Tensor::new(prompt_tokens.as_slice(), &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| fail(format!("prompt tensor: {e}")))?;
        let prefill = self
            .model
            .forward(&input, 0)
            .map_err(|e| fail(format!("prefill: {e}")))?;
        // Greedy next token = the token the decode loop would feed next.
        let next_token = prefill
            .i((0, ..))
            .and_then(|l| l.argmax(0))
            .and_then(|t| t.to_scalar::<u32>())
            .map_err(|e| fail(format!("prefill argmax: {e}")))?;
        let tok_input = Tensor::new(&[next_token], &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| fail(format!("token tensor: {e}")))?;

        // --- gemma3-only (the model with the static-KV + embed split) --------------
        let model = match &mut self.model {
            LoadedModel::Gemma(m) => m,
            _ => return Err(fail("graph replay probe currently supports gemma3 only".to_string())),
        };

        // The embedding lookup stays OUTSIDE the captured region (host-sync D2H+H2D).
        let embed = model.embed(&tok_input).map_err(|e| fail(format!("embed: {e}")))?;

        // Bisection toggle (Phase 46c): capture only the transformer stack + final norm
        // (`hidden_states_from_embed`, excluding the lm_head GEMV) when AKH_BENCH_GRAPH_HIDDEN
        // is set, so a replay mismatch can be localized to the layers vs the output head.
        let hidden_only = std::env::var("AKH_BENCH_GRAPH_HIDDEN").as_deref() == Ok("1");
        let run = |m: &mut super::vendored::quantized_gemma3::ModelWeights| -> candle_core::Result<Tensor> {
            if hidden_only {
                m.hidden_states_from_embed(&embed, position, None)
            } else {
                m.forward_from_embed(&embed, position, None)
            }
        };

        let to_vec0 = |t: &Tensor, what: &str| -> NluResult<Vec<f32>> {
            t.i((0, ..))
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| fail(format!("{what} logits: {e}")))
        };
        let argmax = |v: &[f32]| -> usize {
            v.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                    if x > bv { (i, x) } else { (bi, bv) }
                })
                .0
        };
        let max_diff = |a: &[f32], b: &[f32]| -> f32 {
            a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
        };

        // --- Eager baseline ---------------------------------------------------------
        // Warm-ups so every transient allocation has a recycled pool block before capture
        // (which aborts on any real hipMalloc). CRUCIAL: never hold a logits device tensor
        // live across the capture — doing so drains the pool of that block so the capture
        // forward's own logits alloc misses → a hipMalloc that aborts capture. So every
        // reference forward is read to a host Vec and its device tensor dropped immediately.
        for _ in 0..2 {
            run(model).map_err(|e| fail(format!("eager warmup: {e}")))?;
            cuda::device_synchronize().map_err(|e| fail(format!("sync: {e}")))?;
        }
        // Reference (warm) → host, device tensor dropped. Also the zero-miss check.
        cuda::reset_malloc_counts();
        let eager_vec = {
            let l = run(model).map_err(|e| fail(format!("warm forward: {e}")))?;
            cuda::device_synchronize().map_err(|e| fail(format!("sync: {e}")))?;
            to_vec0(&l, "eager")?
        };
        let (warm_malloc_calls, warm_malloc_miss, _) = cuda::malloc_profile();

        // Same forward on a PRIVATE stream (no graph) → host, device dropped. If this
        // matches the default-stream result, every op honors the launch stream.
        let stream_vec = {
            let l = cuda::run_on_private_stream(|| run(model))
                .map_err(|e| fail(format!("private-stream forward: {e}")))?;
            to_vec0(&l, "stream")?
        };
        let stream_eager_diff = max_diff(&eager_vec, &stream_vec);

        // Eager throughput (drained per iteration; results dropped each iter).
        let eager_t0 = std::time::Instant::now();
        for _ in 0..iters {
            run(model).map_err(|e| fail(format!("eager forward: {e}")))?;
            cuda::device_synchronize().map_err(|e| fail(format!("sync: {e}")))?;
        }
        let eager_secs = eager_t0.elapsed().as_secs_f32();
        let eager_tok_s = if eager_secs > 0.0 { iters as f32 / eager_secs } else { 0.0 };

        // --- Capture the forward at the fixed position (non-fatal on failure) -------
        let capture = cuda::HipGraph::capture(|| run(model));

        let mut replay_tok_s = 0.0;
        let mut captured = false;
        let mut capture_error = None;
        let mut argmax_matches = false;
        let mut max_abs_diff = f32::NAN;
        let mut replay_determinism = f32::NAN;

        match capture {
            Err(e) => capture_error = Some(format!("{e}")),
            Ok((graph, replay_logits)) => {
                captured = true;
                // First replay → its output (read before any further replays).
                graph.replay().map_err(|e| fail(format!("replay warmup: {e}")))?;
                let first_vec = to_vec0(&replay_logits, "replay-first")?;

                let replay_t0 = std::time::Instant::now();
                for _ in 0..iters {
                    graph.replay().map_err(|e| fail(format!("replay: {e}")))?;
                }
                let replay_secs = replay_t0.elapsed().as_secs_f32();
                replay_tok_s = if replay_secs > 0.0 { iters as f32 / replay_secs } else { 0.0 };
                let last_vec = to_vec0(&replay_logits, "replay-last")?;

                replay_determinism = max_diff(&first_vec, &last_vec);
                argmax_matches = argmax(&eager_vec) == argmax(&last_vec);
                max_abs_diff = max_diff(&eager_vec, &last_vec);
            }
        }

        Ok(GraphBenchStats {
            position,
            iters,
            eager_tok_s,
            replay_tok_s,
            warm_malloc_miss,
            warm_malloc_calls,
            captured,
            capture_error,
            argmax_matches,
            max_abs_diff,
            replay_determinism,
            stream_eager_diff,
        })
    }

    /// Phase 46d-2c gate: byte-exact A/B of the **graph-ready** decode forward against the
    /// normal decode forward at a fixed position (see [`GraphDecodeParity`]). The graph-ready
    /// path (device-pos rope `index_select`, `kv_scatter_at`, fixed block-padded `read_len` +
    /// a device-built keep-mask) is run EAGERLY here — no capture — so any divergence is
    /// attributed to the new math, not to graph machinery. Both diffs must be `0` before
    /// 46d-2d wraps this forward in a captured/replayed graph. gemma3 only.
    #[cfg(all(feature = "candle-backend", feature = "hip"))]
    pub fn bench_graph_decode_parity(&mut self, prompt: &str) -> NluResult<GraphDecodeParity> {
        use super::vendored::quantized_gemma3::GraphDecodeCtx;
        let fail = |reason: String| NluError::LlmGenerationFailed { reason };

        // --- Prefill to a realistic position P (mirrors bench_graph_replay) ---------
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| fail(format!("tokenization: {e}")))?;
        let mut prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
        if matches!(self.family, ModelFamily::Gemma | ModelFamily::Gemma4) {
            if let Some(bos) = self.tokenizer.token_to_id("<bos>") {
                if prompt_tokens.first() != Some(&bos) {
                    prompt_tokens.insert(0, bos);
                }
            }
        }
        let position = prompt_tokens.len();
        let input = Tensor::new(prompt_tokens.as_slice(), &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| fail(format!("prompt tensor: {e}")))?;
        let prefill = self
            .model
            .forward(&input, 0)
            .map_err(|e| fail(format!("prefill: {e}")))?;
        let next_token = prefill
            .i((0, ..))
            .and_then(|l| l.argmax(0))
            .and_then(|t| t.to_scalar::<u32>())
            .map_err(|e| fail(format!("prefill argmax: {e}")))?;
        let tok_input = Tensor::new(&[next_token], &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| fail(format!("token tensor: {e}")))?;

        // Clone the device handle before the mutable model borrow (GraphDecodeCtx::new needs it).
        let device = self.device.clone();
        let model = match &mut self.model {
            LoadedModel::Gemma(m) => m,
            _ => {
                return Err(fail(
                    "graph decode parity probe supports gemma3 only".to_string(),
                ))
            }
        };
        // The embedding lookup is a host-sync op; the graph-ready forward starts from it.
        let embed = model.embed(&tok_input).map_err(|e| fail(format!("embed: {e}")))?;

        let to_vec = |t: &Tensor| -> NluResult<Vec<f32>> {
            t.i((0, ..))
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| fail(format!("logits: {e}")))
        };
        let max_diff = |a: &[f32], b: &[f32]| -> f32 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };

        // Normal decode forward at P (writes K/V col P via slice_set, reads [0, P+1)).
        let normal = to_vec(
            &model
                .forward_from_embed(&embed, position, None)
                .map_err(|e| fail(format!("normal forward: {e}")))?,
        )?;

        // Graph-ready, EXACT read_len = P + 1: re-writes col P via the device-pos
        // kv_scatter_at and reads the same [0, P+1) — isolates device-pos rope + scatter
        // from the padding/mask path.
        let g_exact = GraphDecodeCtx::new(&device, position + 1)
            .map_err(|e| fail(format!("graph ctx (exact): {e}")))?;
        g_exact
            .set_pos(position)
            .map_err(|e| fail(format!("set_pos (exact): {e}")))?;
        let exact = to_vec(
            &model
                .forward_from_embed(&embed, position, Some(&g_exact))
                .map_err(|e| fail(format!("graph-ready forward (exact): {e}")))?,
        )?;

        // Graph-ready, PADDED read_len > P + 1: reads a fixed padded slice; the > P tail is
        // masked by the keep-mask — proves the block-padding + masked-tail attention is
        // bit-identical to attention over the valid prefix.
        let padded_read_len = (position + 1 + 8).min(model.kv_ctx().max(position + 1));
        let g_pad = GraphDecodeCtx::new(&device, padded_read_len)
            .map_err(|e| fail(format!("graph ctx (padded): {e}")))?;
        g_pad
            .set_pos(position)
            .map_err(|e| fail(format!("set_pos (padded): {e}")))?;
        let padded = to_vec(
            &model
                .forward_from_embed(&embed, position, Some(&g_pad))
                .map_err(|e| fail(format!("graph-ready forward (padded): {e}")))?,
        )?;

        Ok(GraphDecodeParity {
            position,
            exact_diff: max_diff(&normal, &exact),
            padded_diff: max_diff(&normal, &padded),
            padded_read_len,
            used_device_scatter: true,
        })
    }

    /// Phase 46d-2d: the real per-token graph decode loop (see [`GraphDecodeLoopStats`]).
    /// Captures the decode forward ONCE at `read_len = position + max_new` (block-padded to
    /// the longest position the run reaches) and replays it per token — because every
    /// position-dependent read now comes from the device `d_pos` (rope `index_select`,
    /// `kv_scatter_at` column, additive mask), one graph follows the whole generation: each
    /// token only updates `d_pos` (4-byte H2D) and refreshes a fixed embedding buffer in
    /// place, then a single `hipGraphLaunch` runs the forward. Generates the same sequence
    /// eagerly first for an A/B (identical greedy tokens ⇒ correct; faster ⇒ the payoff).
    /// gemma3 only.
    #[cfg(all(feature = "candle-backend", feature = "hip"))]
    pub fn bench_graph_decode_loop(
        &mut self,
        prompt: &str,
        max_new: u32,
    ) -> NluResult<GraphDecodeLoopStats> {
        use super::vendored::quantized_gemma3::GraphDecodeCtx;
        use candle_core::cuda;
        let fail = |reason: String| NluError::LlmGenerationFailed { reason };
        let max_new = max_new.max(1) as usize;

        // --- tokenize + bos --------------------------------------------------------
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| fail(format!("tokenization: {e}")))?;
        let mut prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
        if matches!(self.family, ModelFamily::Gemma | ModelFamily::Gemma4) {
            if let Some(bos) = self.tokenizer.token_to_id("<bos>") {
                if prompt_tokens.first() != Some(&bos) {
                    prompt_tokens.insert(0, bos);
                }
            }
        }
        let position = prompt_tokens.len();
        let input = Tensor::new(prompt_tokens.as_slice(), &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| fail(format!("prompt tensor: {e}")))?;
        let eos = self
            .tokenizer
            .token_to_id("<eos>")
            .or_else(|| self.tokenizer.token_to_id("<end_of_turn>"));

        let device = self.device.clone();
        let model = match &mut self.model {
            LoadedModel::Gemma(m) => m,
            _ => {
                return Err(fail(
                    "graph decode loop probe supports gemma3 only".to_string(),
                ))
            }
        };

        // One graph spans the whole run: read_len covers the deepest position decoded
        // (P + effective_new - 1 ⇒ read prefix length P + effective_new). Clamp to the cache.
        let kv_ctx = model.kv_ctx();
        let read_len = (position + max_new).min(kv_ctx);
        let effective_new = read_len.saturating_sub(position);

        let greedy = |logits: &Tensor| -> NluResult<u32> {
            logits
                .i((0, ..))
                .and_then(|l| l.argmax(0))
                .and_then(|t| t.to_scalar::<u32>())
                .map_err(|e| fail(format!("argmax: {e}")))
        };

        // ===== EAGER reference run (forward_from_embed None — the real baseline) =====
        let prefill = model
            .forward(&input, 0)
            .map_err(|e| fail(format!("prefill (eager): {e}")))?;
        let mut tok = greedy(&prefill)?;
        let mut eager_tokens: Vec<u32> = Vec::with_capacity(effective_new);
        let eager_t0 = std::time::Instant::now();
        for t in 0..effective_new {
            let pos = position + t;
            let tok_input = Tensor::new(&[tok], &device)
                .and_then(|x| x.unsqueeze(0))
                .map_err(|e| fail(format!("tok tensor: {e}")))?;
            let embed = model
                .embed(&tok_input)
                .map_err(|e| fail(format!("embed (eager): {e}")))?;
            let logits = model
                .forward_from_embed(&embed, pos, None)
                .map_err(|e| fail(format!("eager forward: {e}")))?;
            tok = greedy(&logits)?;
            eager_tokens.push(tok);
            if Some(tok) == eos {
                break;
            }
        }
        cuda::device_synchronize().ok();
        let eager_secs = eager_t0.elapsed().as_secs_f32();
        let eager_tok_s = if eager_secs > 0.0 {
            eager_tokens.len() as f32 / eager_secs
        } else {
            0.0
        };

        // ===== GRAPH run (single capture, replay per token) =========================
        // Reset the KV base to [0, P) so both runs decode from the same state.
        let prefill2 = model
            .forward(&input, 0)
            .map_err(|e| fail(format!("prefill (graph): {e}")))?;
        let tok0 = greedy(&prefill2)?;

        let gctx = GraphDecodeCtx::new(&device, read_len)
            .map_err(|e| fail(format!("graph ctx: {e}")))?;

        // Fixed embedding buffer the captured forward reads — overwritten in place each
        // token (its address stays stable, so the graph reads the refreshed embedding).
        let tok0_input = Tensor::new(&[tok0], &device)
            .and_then(|x| x.unsqueeze(0))
            .map_err(|e| fail(format!("tok0 tensor: {e}")))?;
        let embed_buf = model
            .embed(&tok0_input)
            .and_then(|e| e.contiguous())
            .map_err(|e| fail(format!("embed0: {e}")))?;
        gctx.set_pos(position)
            .map_err(|e| fail(format!("set_pos (capture): {e}")))?;

        // Warm the pool with the EXACT captured shapes. The first passes prime the pool; the
        // reported miss is from a final steady pass (0 ⇒ allocation-stable ⇒ a clean capture).
        for _ in 0..2 {
            model
                .forward_from_embed(&embed_buf, position, Some(&gctx))
                .map_err(|e| fail(format!("graph warmup: {e}")))?;
            cuda::device_synchronize().map_err(|e| fail(format!("sync: {e}")))?;
        }
        cuda::reset_malloc_counts();
        model
            .forward_from_embed(&embed_buf, position, Some(&gctx))
            .map_err(|e| fail(format!("graph warmup (measured): {e}")))?;
        cuda::device_synchronize().map_err(|e| fail(format!("sync: {e}")))?;
        let (_calls, capture_miss, _) = cuda::malloc_profile();

        // Capture the decode forward once.
        let (graph, logits_buf) =
            cuda::HipGraph::capture(|| model.forward_from_embed(&embed_buf, position, Some(&gctx)))
                .map_err(|e| fail(format!("graph capture: {e}")))?;

        // Decode by replay: update embed_buf + d_pos (outside capture), then one dispatch.
        let mut graph_tokens: Vec<u32> = Vec::with_capacity(effective_new);
        let mut tok = tok0;
        let graph_t0 = std::time::Instant::now();
        for t in 0..effective_new {
            let pos = position + t;
            if t > 0 {
                let tok_input = Tensor::new(&[tok], &device)
                    .and_then(|x| x.unsqueeze(0))
                    .map_err(|e| fail(format!("tok tensor: {e}")))?;
                let new_embed = model
                    .embed(&tok_input)
                    .map_err(|e| fail(format!("embed: {e}")))?;
                // In-place refresh — keeps embed_buf's address stable for the captured graph.
                embed_buf
                    .slice_set(&new_embed, 0, 0)
                    .map_err(|e| fail(format!("embed refresh: {e}")))?;
            }
            gctx.set_pos(pos)
                .map_err(|e| fail(format!("set_pos: {e}")))?;
            graph.replay().map_err(|e| fail(format!("replay: {e}")))?;
            tok = greedy(&logits_buf)?;
            graph_tokens.push(tok);
            if Some(tok) == eos {
                break;
            }
        }
        let graph_secs = graph_t0.elapsed().as_secs_f32();
        let graph_tok_s = if graph_secs > 0.0 {
            graph_tokens.len() as f32 / graph_secs
        } else {
            0.0
        };

        Ok(GraphDecodeLoopStats {
            position,
            requested_new: max_new,
            decoded: graph_tokens.len(),
            eager_tok_s,
            graph_tok_s,
            capture_miss,
            tokens_match: eager_tokens == graph_tokens,
        })
    }

    /// Phase 47a gate: byte-exact A/B of gemma4's all-position **verify forward** against
    /// incremental single-token decode (see [`VerifyParity`]). Prefills the prompt to position
    /// `P`, greedily decodes `n` tokens one-at-a-time recording each position's logits (the
    /// reference + the token sequence), resets the KV to `[0, P)` by re-prefilling, then scores
    /// the SAME `n` tokens in one `forward_verify` at `P` and compares every position. gemma4 only.
    #[cfg(feature = "candle-backend")]
    pub fn bench_verify_parity(&mut self, prompt: &str, n: usize) -> NluResult<VerifyParity> {
        let fail = |reason: String| NluError::LlmGenerationFailed { reason };
        let n = n.max(1);

        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| fail(format!("tokenization: {e}")))?;
        let mut prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
        if matches!(self.family, ModelFamily::Gemma | ModelFamily::Gemma4) {
            if let Some(bos) = self.tokenizer.token_to_id("<bos>") {
                if prompt_tokens.first() != Some(&bos) {
                    prompt_tokens.insert(0, bos);
                }
            }
        }
        let position = prompt_tokens.len();
        let input = Tensor::new(prompt_tokens.as_slice(), &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| fail(format!("prompt tensor: {e}")))?;

        let device = self.device.clone();
        let model = match &mut self.model {
            LoadedModel::Gemma4(m) => m,
            _ => return Err(fail("verify-parity probe supports gemma4 only".to_string())),
        };

        let argmax = |v: &[f32]| -> u32 {
            v.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                    if x > bv { (i, x) } else { (bi, bv) }
                })
                .0 as u32
        };
        let max_diff = |a: &[f32], b: &[f32]| -> f32 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        let row = |t: &Tensor, i: usize| -> NluResult<Vec<f32>> {
            t.i((0, i, ..))
                .and_then(|r| r.to_vec1::<f32>())
                .map_err(|e| fail(format!("logits row {i}: {e}")))
        };

        // Prefill -> P; first token = argmax(prefill).
        let prefill = model
            .forward(&input, 0)
            .map_err(|e| fail(format!("prefill: {e}")))?;
        let prefill_vec = prefill
            .i((0, ..))
            .and_then(|l| l.to_vec1::<f32>())
            .map_err(|e| fail(format!("prefill logits: {e}")))?;
        let mut s = argmax(&prefill_vec);

        // Incremental reference: feed the greedy sequence one token at a time, recording each
        // position's logits. This writes KV [P, P+n).
        let mut seq: Vec<u32> = Vec::with_capacity(n);
        let mut ref_logits: Vec<Vec<f32>> = Vec::with_capacity(n);
        for i in 0..n {
            let inp = Tensor::new(&[s], &device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| fail(format!("tok tensor: {e}")))?;
            let logits = model
                .forward(&inp, position + i)
                .map_err(|e| fail(format!("incremental forward {i}: {e}")))?;
            let lv = logits
                .i((0, ..))
                .and_then(|l| l.to_vec1::<f32>())
                .map_err(|e| fail(format!("incremental logits {i}: {e}")))?;
            seq.push(s);
            s = argmax(&lv);
            ref_logits.push(lv);
        }

        // Score the SAME seq in one verify forward. No re-prefill needed: the prefix [0, P) was
        // written once by the initial prefill and is never overwritten (both incremental and
        // verify only write [P, P+n)); verify just re-writes [P, P+n) with the same tokens. A
        // re-prefill here would re-run the (potentially non-deterministic rocBLAS) prefill GEMM
        // and give verify a slightly different prefix — a measurement confound, not a real diff.
        let seq_t = Tensor::new(seq.as_slice(), &device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| fail(format!("seq tensor: {e}")))?;
        let verify = model
            .forward_verify(&seq_t, position)
            .map_err(|e| fail(format!("verify forward: {e}")))?;

        let mut max_abs_diff = 0.0f32;
        let mut last_pos_diff = 0.0f32;
        let mut argmax_agree = 0usize;
        for i in 0..n {
            let vrow = row(&verify, i)?;
            let d = max_diff(&vrow, &ref_logits[i]);
            max_abs_diff = max_abs_diff.max(d);
            if i == n - 1 {
                last_pos_diff = d;
            }
            if argmax(&vrow) == argmax(&ref_logits[i]) {
                argmax_agree += 1;
            }
        }

        Ok(VerifyParity {
            position,
            seq_len: n,
            max_abs_diff,
            last_pos_diff,
            argmax_agree,
        })
    }

    /// Phase 47c: greedy speculative decode loop (see [`SpecDecodeStats`]). A/Bs plain greedy
    /// decode against prompt-lookup speculative decode — the n-gram drafter proposes a span from
    /// the token history, one `forward_verify` scores it, the matching greedy prefix is accepted,
    /// and the round advances `accepted+1` tokens (KV rollback is implicit: the next round just
    /// writes from the new position, overwriting the rejected tail). gemma4 only.
    #[cfg(all(feature = "candle-backend", feature = "hip"))]
    pub fn bench_spec_decode(&mut self, prompt: &str, max_new: u32) -> NluResult<SpecDecodeStats> {
        use super::spec_decode::{accept_run, DrafterConfig, NgramDrafter};
        use candle_core::cuda;
        let fail = |reason: String| NluError::LlmGenerationFailed { reason };
        let max_new = max_new.max(1) as usize;

        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| fail(format!("tokenization: {e}")))?;
        let mut prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
        if matches!(self.family, ModelFamily::Gemma | ModelFamily::Gemma4) {
            if let Some(bos) = self.tokenizer.token_to_id("<bos>") {
                if prompt_tokens.first() != Some(&bos) {
                    prompt_tokens.insert(0, bos);
                }
            }
        }
        let position = prompt_tokens.len();
        let input = Tensor::new(prompt_tokens.as_slice(), &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| fail(format!("prompt tensor: {e}")))?;
        let eos = self
            .tokenizer
            .token_to_id("<eos>")
            .or_else(|| self.tokenizer.token_to_id("<end_of_turn>"));

        let device = self.device.clone();
        let model = match &mut self.model {
            LoadedModel::Gemma4(m) => m,
            _ => return Err(fail("spec-decode probe supports gemma4 only".to_string())),
        };
        let kv_ctx = model.kv_ctx();

        // On-device argmax → only the token id(s) come back over PCIe (no full-logits D2H).
        // `[1, vocab]` -> 1 id; `[1, m, vocab]` -> m ids.
        let argmax_ids = |t: &Tensor| -> NluResult<Vec<u32>> {
            t.argmax(candle_core::D::Minus1)
                .and_then(|a| a.flatten_all())
                .and_then(|a| a.to_vec1::<u32>())
                .map_err(|e| fail(format!("argmax: {e}")))
        };

        // Prefill -> P; first generated token = argmax(prefill).
        let prefill = model
            .forward(&input, 0)
            .map_err(|e| fail(format!("prefill: {e}")))?;
        let cur0 = argmax_ids(&prefill)?[0];

        // ===== PLAIN greedy baseline (one forward per token) ========================
        let mut plain: Vec<u32> = vec![cur0];
        let mut cur = cur0;
        let mut pos = position;
        let plain_t0 = std::time::Instant::now();
        while plain.len() < max_new {
            let inp = Tensor::new(&[cur], &device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| fail(format!("tok tensor: {e}")))?;
            let logits = model
                .forward(&inp, pos)
                .map_err(|e| fail(format!("plain forward: {e}")))?;
            let next = argmax_ids(&logits)?[0];
            plain.push(next);
            if Some(next) == eos {
                break;
            }
            cur = next;
            pos += 1;
        }
        cuda::device_synchronize().ok();
        let plain_secs = plain_t0.elapsed().as_secs_f32();
        let plain_tok_s = plain.len() as f32 / plain_secs.max(1e-6);

        // ===== SPECULATIVE decode (prompt-lookup + verify per round) ================
        // No re-prefill: plain only wrote [P, ..]; the prefix [0, P) is the one shared prefill,
        // and the spec verify overwrites [P, ..] with its own K/V before reading.
        let drafter = NgramDrafter::new(DrafterConfig::default());
        let mut all: Vec<u32> = prompt_tokens.clone();
        all.push(cur0);
        let mut spec: Vec<u32> = vec![cur0];
        let mut cur = cur0;
        let mut pos = position;
        let (mut rounds, mut drafted, mut accepted) = (0usize, 0usize, 0usize);
        let spec_t0 = std::time::Instant::now();
        while spec.len() < max_new {
            // Propose, then cap the span so the verify write [pos, pos+gamma] fits the cache.
            let mut drafts = drafter.propose(&all);
            let max_gamma = kv_ctx.saturating_sub(pos + 1);
            if drafts.len() > max_gamma {
                drafts.truncate(max_gamma);
            }

            // Verify [cur, drafts..] at [pos, ..] in one forward; greedy-accept the prefix.
            let feed: Vec<u32> = std::iter::once(cur).chain(drafts.iter().copied()).collect();
            let feed_t = Tensor::new(feed.as_slice(), &device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| fail(format!("feed tensor: {e}")))?;
            let verify = model
                .forward_verify(&feed_t, pos)
                .map_err(|e| fail(format!("verify forward: {e}")))?;
            let argmaxes = argmax_ids(&verify)?; // length 1 + gamma
            let k = accept_run(&drafts, &argmaxes);
            rounds += 1;
            drafted += drafts.len();
            accepted += k;

            // Emit the k accepted drafts + the bonus/correction (argmaxes[k] always exists).
            for &d in drafts.iter().take(k) {
                spec.push(d);
                all.push(d);
            }
            let bonus = argmaxes[k];
            spec.push(bonus);
            all.push(bonus);
            pos += k + 1;
            cur = bonus;
            if Some(bonus) == eos || pos + 1 >= kv_ctx {
                break;
            }
        }
        cuda::device_synchronize().ok();
        let spec_secs = spec_t0.elapsed().as_secs_f32();
        let spec_tok_s = spec.len() as f32 / spec_secs.max(1e-6);

        let cmp = plain.len().min(spec.len());
        let tokens_match = plain[..cmp] == spec[..cmp];

        Ok(SpecDecodeStats {
            position,
            plain_tokens: plain.len(),
            spec_tokens: spec.len(),
            plain_tok_s,
            spec_tok_s,
            rounds,
            drafted,
            accepted,
            tokens_match,
        })
    }

    /// Generate and return the **typed channel stream** — the model's reasoning
    /// (`thought`) and user-facing answer (`final`) as ordered, typed segments —
    /// instead of collapsing to a single string. Lets the chat layer annotate the
    /// thought process distinctly (collapsible "thinking" sections) rather than
    /// dropping it. For Gemma 4's channel format; plain models yield a single
    /// [`crate::nlu::channel::Channel::Final`] segment. See ADR 054.
    #[cfg(feature = "candle-backend")]
    pub fn generate_channels(
        &mut self,
        prompt: &str,
        max_tokens: u32,
    ) -> NluResult<crate::nlu::channel::ChannelStream> {
        let result = self.generate_inner(prompt, max_tokens, StopMode::Text)?;
        Ok(crate::nlu::channel::ChannelStream::parse(&result.text))
    }

    /// Non-feature-gated stub.
    #[cfg(not(feature = "candle-backend"))]
    pub fn generate(&mut self, _prompt: &str, _max_tokens: u32) -> NluResult<String> {
        Err(NluError::LlmGenerationFailed {
            reason: "candle-backend feature not enabled".to_string(),
        })
    }

    /// Build a family-appropriate chat prompt from system + user turns.
    #[cfg(feature = "candle-backend")]
    fn build_chat_prompt(&self, system: &str, user: &str) -> String {
        match self.family {
            // Gemma 2/3 has no system role; fold it into the user turn.
            ModelFamily::Gemma => format!(
                "<start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n<start_of_turn>model\n"
            ),
            // Gemma 4 uses a new turn format: `<|turn>role\n ... <turn|>\n` with the
            // generation prompt `<|turn>model\n`. (BOS is prepended in generate_inner.)
            ModelFamily::Gemma4 => format!(
                "<|turn>user\n{system}\n\n{user}<turn|>\n<|turn>model\n"
            ),
            ModelFamily::Qwen2 => format!(
                "<|im_start|>system\n{system}<|im_end|>\n\
                 <|im_start|>user\n{user}<|im_end|>\n\
                 <|im_start|>assistant\n"
            ),
        }
    }

    /// Build the persona-framed chat prompt for a dialogue act (greeting,
    /// farewell, …). Shared by [`generate_dialogue`](Self::generate_dialogue) and
    /// [`generate_dialogue_channels`](Self::generate_dialogue_channels) so both
    /// paths produce byte-identical prompts.
    #[cfg(feature = "candle-backend")]
    fn dialogue_prompt(
        &self,
        act: &str,
        persona_name: &str,
        traits: &[String],
        context: Option<&str>,
    ) -> String {
        let traits_str = if traits.is_empty() {
            "knowledgeable".to_string()
        } else {
            traits.join(", ")
        };
        let context_line = context
            .map(|c| format!(" We were discussing {c}."))
            .unwrap_or_default();
        let system = format!(
            "You are {persona_name}, a neuro-symbolic knowledge engine that is {traits_str}. \
             Reply in one warm, natural sentence. Always speak as {persona_name}. No hashtags, no emoji."
        );
        let user = format!("Generate a {act} response.{context_line}");
        self.build_chat_prompt(&system, &user)
    }

    /// Generate a short conversational dialogue act (greeting, farewell, …) in
    /// the engine's persona. Output boundary.
    #[cfg(feature = "candle-backend")]
    pub fn generate_dialogue(
        &mut self,
        act: &str,
        persona_name: &str,
        traits: &[String],
        context: Option<&str>,
    ) -> NluResult<String> {
        let prompt = self.dialogue_prompt(act, persona_name, traits, context);
        let result = self.generate_inner(&prompt, 96, StopMode::Text)?;
        Ok(clean_generated_text(&result.text))
    }

    /// Channel-aware variant of [`generate_dialogue`](Self::generate_dialogue):
    /// returns the typed [`ChannelStream`] so the chat layer can surface Gemma 4's
    /// internal reasoning (`thought`) as a collapsible trace distinct from the
    /// answer (`final`), rather than dropping it. Builds the identical prompt. See
    /// ADR 054.
    #[cfg(feature = "candle-backend")]
    pub fn generate_dialogue_channels(
        &mut self,
        act: &str,
        persona_name: &str,
        traits: &[String],
        context: Option<&str>,
    ) -> NluResult<crate::nlu::channel::ChannelStream> {
        let prompt = self.dialogue_prompt(act, persona_name, traits, context);
        let result = self.generate_inner(&prompt, 96, StopMode::Text)?;
        Ok(crate::nlu::channel::ChannelStream::parse(&result.text))
    }

    /// Verbalize a grounded answer: rephrase KG-derived `facts` as a fluent,
    /// conversational reply to `question`, constrained to those facts.
    ///
    /// The system prompt forbids inventing information, so the KG stays the
    /// source of truth — this is grounded generation, not free reasoning.
    #[cfg(feature = "candle-backend")]
    pub fn verbalize(
        &mut self,
        question: &str,
        facts: &str,
        persona_name: &str,
        traits: &[String],
    ) -> NluResult<String> {
        let prompt = self.verbalize_prompt(question, facts, persona_name, traits);
        // Cap output to keep the answer tight and reduce embellishment drift.
        let cap = self.config.max_tokens.min(160);
        let result = self.generate_inner(&prompt, cap, StopMode::Text)?;
        Ok(clean_generated_text(&result.text))
    }

    /// Build the grounded-verbalization chat prompt. Shared by
    /// [`verbalize`](Self::verbalize) and
    /// [`verbalize_channels`](Self::verbalize_channels) so both paths produce a
    /// byte-identical prompt.
    #[cfg(feature = "candle-backend")]
    fn verbalize_prompt(
        &self,
        question: &str,
        facts: &str,
        persona_name: &str,
        traits: &[String],
    ) -> String {
        let traits_str = if traits.is_empty() {
            "precise and grounded".to_string()
        } else {
            traits.join(", ")
        };
        let system = format!(
            "You are {persona_name}, a neuro-symbolic knowledge engine that is {traits_str}.\n\
             Restate the KNOWLEDGE as a direct, natural answer to the QUESTION.\n\
             ABSOLUTE RULES:\n\
             - Use ONLY facts present in KNOWLEDGE. Add NOTHING — no examples, no\n  \
               elaboration, no general world knowledge, no adjectives that aren't given.\n\
             - If KNOWLEDGE is a single fact, answer in a single short sentence.\n\
             - If KNOWLEDGE does not answer the QUESTION, say so plainly.\n\
             Speak as {persona_name} in the first person. No markdown, no hashtags."
        );
        let knowledge = if facts.trim().is_empty() {
            "(no relevant facts were found in the knowledge graph)".to_string()
        } else {
            facts.trim().to_string()
        };
        let user = format!("KNOWLEDGE:\n{knowledge}\n\nQUESTION: {question}");
        self.build_chat_prompt(&system, &user)
    }

    /// Channel-aware variant of [`verbalize`](Self::verbalize): returns the typed
    /// [`ChannelStream`] so the chat layer can surface Gemma 4's grounded reasoning
    /// (`thought`) as a collapsible trace distinct from the answer (`final`),
    /// rather than dropping it. Builds the identical prompt. See ADR 054.
    #[cfg(feature = "candle-backend")]
    pub fn verbalize_channels(
        &mut self,
        question: &str,
        facts: &str,
        persona_name: &str,
        traits: &[String],
    ) -> NluResult<crate::nlu::channel::ChannelStream> {
        let prompt = self.verbalize_prompt(question, facts, persona_name, traits);
        let cap = self.config.max_tokens.min(160);
        let result = self.generate_inner(&prompt, cap, StopMode::Text)?;
        Ok(crate::nlu::channel::ChannelStream::parse(&result.text))
    }

    // ── Native audio input (Gemma 4, ADR 054) ───────────────────────────

    /// Understand a 16 kHz mono PCM clip and answer `question` about it.
    ///
    /// Gemma 4 is encoder-free: each 40 ms (640-sample) window is linearly
    /// projected (`640 → 3840`) directly into the text-embedding space and
    /// spliced into the prompt at the audio-placeholder positions. We build the
    /// prompt as
    /// `[bos] <|turn>user\n <boa> <audio>×n_frames <eoa> \n{question}<turn|>\n<|turn>model\n`,
    /// embed every token via `token_embd` (text tokens ×`sqrt(embedding_length)`),
    /// then **overwrite** the `n_frames` audio-placeholder rows with the projected
    /// (un-scaled) audio embeddings, prefill via `forward_embeds`, and decode
    /// normally.
    ///
    /// Requires the Gemma 4 family and a loadable mmproj sidecar
    /// (`<model-stem>.mmproj.gguf`).
    #[cfg(feature = "candle-backend")]
    pub fn understand_audio(
        &mut self,
        pcm_16k_mono: &[f32],
        question: &str,
    ) -> NluResult<String> {
        use super::audio::AudioProjector;

        if self.family != ModelFamily::Gemma4 {
            return Err(NluError::LlmGenerationFailed {
                reason: format!(
                    "native audio input requires the Gemma 4 model family, but \
                     the loaded model is {:?}",
                    self.family
                ),
            });
        }

        // Resolve the Gemma 4 audio special-token ids. Gemma 4's audio markers:
        //   <|audio>  (256000) begin-of-audio (BOA)
        //   <|audio|> (258881) per-frame placeholder (one per 40 ms window)
        //   <audio|>  (258883) end-of-audio (EOA)
        let tok_id = |t: &str| -> NluResult<u32> {
            self.tokenizer
                .token_to_id(t)
                .ok_or_else(|| NluError::LlmGenerationFailed {
                    reason: format!("Gemma 4 tokenizer is missing audio token `{t}`"),
                })
        };
        let boa = tok_id("<|audio>")?;
        let audio_placeholder = tok_id("<|audio|>")?;
        let eoa = tok_id("<audio|>")?;
        let bos = tok_id("<bos>")?;

        // Project the audio into embedding space → [n_frames, 3840].
        let mmproj_path = self.mmproj_path()?;
        let projector = AudioProjector::load(&mmproj_path, &self.device)?;
        let audio_embeds = projector.project_audio(pcm_16k_mono)?;
        let n_frames = audio_embeds
            .dim(0)
            .map_err(|e| NluError::LlmGenerationFailed {
                reason: format!("audio frame count: {e}"),
            })?;

        // Tokenize the surrounding text. encode(.., false) keeps the literal
        // markers we add ourselves out of the picture; we splice token ids
        // directly. (The Gemma 4 chat template is `<|turn>role\n…<turn|>`.)
        let encode = |s: &str| -> NluResult<Vec<u32>> {
            self.tokenizer
                .encode(s, false)
                .map(|e| e.get_ids().to_vec())
                .map_err(|e| NluError::LlmGenerationFailed {
                    reason: format!("tokenize `{s}`: {e}"),
                })
        };
        let pre = encode("<|turn>user\n")?;
        let post = encode(&format!("\n{question}<turn|>\n<|turn>model\n"))?;

        // Assemble the full id sequence with the audio frames as placeholders.
        let mut ids: Vec<u32> = Vec::with_capacity(
            1 + pre.len() + 1 + n_frames + 1 + post.len(),
        );
        ids.push(bos);
        ids.extend_from_slice(&pre);
        ids.push(boa);
        let audio_start = ids.len();
        ids.extend(std::iter::repeat(audio_placeholder).take(n_frames));
        ids.push(eoa);
        ids.extend_from_slice(&post);

        tracing::debug!(
            n_frames,
            seq_len = ids.len(),
            audio_start,
            "Gemma 4 audio prompt assembled"
        );

        if ids.len() > 4096 {
            return Err(NluError::LlmGenerationFailed {
                reason: format!(
                    "audio prompt too long ({} tokens, {n_frames} audio frames). \
                     Use a shorter clip (each 40 ms = 1 token).",
                    ids.len()
                ),
            });
        }

        // Embed every token, then scale ONLY text tokens by sqrt(embedding_length).
        // Audio frames are projected directly into embedding space (no scale).
        let id_tensor = Tensor::new(ids.as_slice(), &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| NluError::LlmGenerationFailed {
                reason: format!("id tensor: {e}"),
            })?;
        let audio_rows = audio_embeds
            .unsqueeze(0)
            .map_err(|e| NluError::LlmGenerationFailed {
                reason: format!("audio unsqueeze: {e}"),
            })?; // [1, n_frames, 3840]

        // Diagnostic toggle (ADR 054): by default audio frames are spliced
        // un-scaled (they are already projected into embedding space). Setting
        // AKH_AUDIO_SCALE=1 additionally multiplies them by sqrt(embedding_length)
        // — used to A/B the residual-stream magnitude question.
        let scale_audio = std::env::var("AKH_AUDIO_SCALE").ok().as_deref() == Some("1");

        // Do all Gemma 4-specific tensor work inside one match to keep a single
        // borrow of `self.model`, then prefill via forward_embeds.
        let logits = match &mut self.model {
            LoadedModel::Gemma4(m) => {
                let scale = m.embedding_scale();
                let embed_dim = m.embedding_length();
                let base = m.embed_tokens(&id_tensor).map_err(|e| {
                    NluError::LlmGenerationFailed {
                        reason: format!("embed_tokens: {e}"),
                    }
                })?; // [1, seq, 3840]
                let scaled = (base * scale).map_err(|e| NluError::LlmGenerationFailed {
                    reason: format!("scale text embeds: {e}"),
                })?;
                let audio_rows = if scale_audio {
                    (&audio_rows * scale).map_err(|e| NluError::LlmGenerationFailed {
                        reason: format!("scale audio embeds: {e}"),
                    })?
                } else {
                    audio_rows
                };
                // Overwrite the n_frames audio-placeholder rows with the projected
                // (un-scaled) audio embeddings.
                let embeds = scaled
                    .slice_assign(
                        &[0..1, audio_start..audio_start + n_frames, 0..embed_dim],
                        &audio_rows,
                    )
                    .map_err(|e| NluError::LlmGenerationFailed {
                        reason: format!("splice audio embeds: {e}"),
                    })?;
                m.forward_embeds(&embeds, 0)
            }
            _ => unreachable!("family checked above"),
        }
        .map_err(|e| NluError::LlmGenerationFailed {
            reason: format!("audio prefill (forward_embeds): {e}"),
        })?;

        let cap = self.config.max_tokens.min(256);
        let result = self.decode_loop(logits, ids.len(), cap, StopMode::Text)?;
        Ok(clean_generated_text(&result.text))
    }

    /// Path to the multimodal projector sidecar for the loaded model:
    /// `<model-stem>.mmproj.gguf` next to the GGUF.
    #[cfg(feature = "candle-backend")]
    fn mmproj_path(&self) -> NluResult<PathBuf> {
        // gemma4-12b.gguf -> gemma4-12b.mmproj.gguf
        let path = self.model_path.with_extension("mmproj.gguf");
        if !path.exists() {
            return Err(NluError::ModelNotFound { path });
        }
        Ok(path)
    }

    /// Non-feature-gated stub.
    #[cfg(not(feature = "candle-backend"))]
    pub fn understand_audio(
        &mut self,
        _pcm_16k_mono: &[f32],
        _question: &str,
    ) -> NluResult<String> {
        Err(NluError::LlmGenerationFailed {
            reason: "candle-backend feature not enabled".to_string(),
        })
    }

    /// Non-feature-gated stubs.
    #[cfg(not(feature = "candle-backend"))]
    pub fn generate_dialogue(
        &mut self,
        _act: &str,
        _persona_name: &str,
        _traits: &[String],
        _context: Option<&str>,
    ) -> NluResult<String> {
        Err(NluError::LlmGenerationFailed {
            reason: "candle-backend feature not enabled".to_string(),
        })
    }

    #[cfg(not(feature = "candle-backend"))]
    pub fn verbalize(
        &mut self,
        _question: &str,
        _facts: &str,
        _persona_name: &str,
        _traits: &[String],
    ) -> NluResult<String> {
        Err(NluError::LlmGenerationFailed {
            reason: "candle-backend feature not enabled".to_string(),
        })
    }

    #[cfg(not(feature = "candle-backend"))]
    pub fn generate_dialogue_channels(
        &mut self,
        _act: &str,
        _persona_name: &str,
        _traits: &[String],
        _context: Option<&str>,
    ) -> NluResult<crate::nlu::channel::ChannelStream> {
        Err(NluError::LlmGenerationFailed {
            reason: "candle-backend feature not enabled".to_string(),
        })
    }

    #[cfg(not(feature = "candle-backend"))]
    pub fn verbalize_channels(
        &mut self,
        _question: &str,
        _facts: &str,
        _persona_name: &str,
        _traits: &[String],
    ) -> NluResult<crate::nlu::channel::ChannelStream> {
        Err(NluError::LlmGenerationFailed {
            reason: "candle-backend feature not enabled".to_string(),
        })
    }

    /// The loaded model family (Qwen2 / Gemma).
    pub fn family(&self) -> ModelFamily {
        self.family
    }

    // ── Hidden state extraction ──────────────────────────────────────────

    /// Run inference and return the **true hidden-state** representation vector
    /// for the Neural→VSA bridge (Phase 26).
    ///
    /// Returns the post-norm hidden state of the last position
    /// (shape `[hidden_dim]`, e.g. 1536 for Qwen2.5-1.5B / 2560 for Gemma-3-4B),
    /// taken **before** the output (lm_head) projection. This is the dense
    /// semantic representation the bridge projects into VSA space — far more
    /// informative than the old vocab-sized logits proxy.
    ///
    /// Enabled by the vendored, hidden-state-exposing model loaders
    /// ([`crate::nlu::vendored`]): upstream candle-transformers keeps `norm` /
    /// `output` private, so this required owning the model code.
    #[cfg(feature = "candle-backend")]
    pub fn extract_hidden_state(&mut self, input: &str) -> NluResult<Vec<f32>> {
        let encoding = self.tokenizer.encode(input, true).map_err(|e| {
            NluError::LlmGenerationFailed {
                reason: format!("tokenization: {e}"),
            }
        })?;
        let token_ids = encoding.get_ids();
        let input_tensor =
            Tensor::new(token_ids, &self.device)
                .map_err(|e| NluError::LlmGenerationFailed {
                    reason: format!("tensor creation: {e}"),
                })?
                .unsqueeze(0)
                .map_err(|e| NluError::LlmGenerationFailed {
                    reason: format!("unsqueeze: {e}"),
                })?;

        let (_logits, hidden) = self
            .model
            .forward_with_hidden(&input_tensor, 0)
            .map_err(|e| NluError::LlmGenerationFailed {
                reason: format!("forward_with_hidden: {e}"),
            })?;

        // `hidden` is [batch, hidden_dim]; take row 0 → [hidden_dim].
        let hidden_vec: Vec<f32> = hidden
            .i((0, ..))
            .and_then(|h| h.to_vec1())
            .map_err(|e| NluError::LlmGenerationFailed {
                reason: format!("hidden to_vec1: {e}"),
            })?;

        Ok(hidden_vec)
    }

    // ── PolarQuant KV cache utilities ────────────────────────────────────

    /// Compress an f32 vector using the PolarQuant pipeline.
    ///
    /// This is a standalone utility for compressing KV cache vectors.
    /// Full integration into Candle's internal `Cache` struct requires forking
    /// `candle-transformers` to wrap read/write paths — deferred until a Candle
    /// hook API or forked model is available.
    ///
    /// The compressed representation uses 3-bit Lloyd-Max quantization with
    /// randomized Hadamard rotation, achieving ~5.3× compression.
    pub fn compress_kv_vector(values: &[f32], seed: u64) -> crate::vsa::polar_quant::QuantizedVector {
        crate::vsa::polar_quant::compress(values, seed)
    }

    /// Decompress a PolarQuant-compressed vector.
    pub fn decompress_kv_vector(
        qvec: &crate::vsa::polar_quant::QuantizedVector,
        seed: u64,
        std_dev_hint: f32,
    ) -> Vec<f32> {
        crate::vsa::polar_quant::decompress(qvec, seed, std_dev_hint)
    }

    // ── Internal generation loop ─────────────────────────────────────────

    #[cfg(feature = "candle-backend")]
    fn generate_inner(
        &mut self,
        prompt: &str,
        max_tokens: u32,
        stop: StopMode,
    ) -> NluResult<GenerationResult> {
        let encoding = self.tokenizer.encode(prompt, true).map_err(|e| {
            NluError::LlmGenerationFailed {
                reason: format!("tokenization: {e}"),
            }
        })?;
        let mut prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

        // Gemma is critically dependent on a leading BOS token: without it the
        // residual stream is un-anchored and greedy decoding collapses to a single
        // repeated token (e.g. "." / "1"). The Gemma 4 tokenizer's `TemplateProcessing`
        // post-processor does NOT prepend BOS (it relies on the model's `add_bos_token`
        // flag, which `encode(.., true)` does not honour here), so we prepend it
        // ourselves when missing. Other families (Qwen2) carry BOS via their template.
        if matches!(self.family, ModelFamily::Gemma | ModelFamily::Gemma4) {
            if let Some(bos) = self.tokenizer.token_to_id("<bos>") {
                if prompt_tokens.first() != Some(&bos) {
                    prompt_tokens.insert(0, bos);
                }
            }
        }

        tracing::debug!(prompt_tokens = prompt_tokens.len(), "Candle prompt tokenized");

        if prompt_tokens.len() > self.context_length {
            return Err(NluError::LlmGenerationFailed {
                reason: format!(
                    "Prompt too long ({} tokens, model context window is {}).",
                    prompt_tokens.len(),
                    self.context_length
                ),
            });
        }

        // Prefill: process all prompt tokens at once.
        let input = Tensor::new(prompt_tokens.as_slice(), &self.device)
            .map_err(|e| NluError::LlmGenerationFailed {
                reason: format!("tensor: {e}"),
            })?
            .unsqueeze(0)
            .map_err(|e| NluError::LlmGenerationFailed {
                reason: format!("unsqueeze: {e}"),
            })?;

        let logits = self.model.forward(&input, 0).map_err(|e| {
            NluError::LlmGenerationFailed {
                reason: format!("prefill: {e}"),
            }
        })?;

        // Continue token-by-token from the prefilled prompt. Eligible gemma4 turns take the
        // speculative (prompt-lookup) path (Phase 47 / ADR 075): greedy → byte-exact vs plain
        // greedy, temperature → distribution-lossless speculative sampling (47e). Every other
        // case keeps the proven plain decode loop unchanged.
        if self.spec_decode_eligible() {
            if self.config.temperature == 0.0 {
                self.spec_decode_loop(logits, prompt_tokens, max_tokens, stop)
            } else {
                self.spec_sample_decode_loop(logits, prompt_tokens, max_tokens, stop)
            }
        } else {
            self.decode_loop(logits, prompt_tokens.len(), max_tokens, stop)
        }
    }

    /// Emit one freshly-chosen `token` into `st` and apply the `StopMode` stopping rules,
    /// returning `true` when generation should stop. An EOS in any non-`Fixed` mode stops
    /// *before* the token is counted or detokenized (so the marker never reaches the output),
    /// matching plain greedy decode. Shared by [`decode_loop`](Self::decode_loop) and
    /// [`spec_decode_loop`](Self::spec_decode_loop) so their detokenization and stop semantics
    /// are byte-identical — the speculative loop just calls this once per accepted/bonus token.
    #[cfg(feature = "candle-backend")]
    fn process_token(
        &self,
        token: u32,
        eos: Option<u32>,
        stop: StopMode,
        st: &mut DecodeState,
    ) -> bool {
        if !matches!(stop, StopMode::Fixed) && Some(token) == eos {
            return true;
        }
        st.n_generated += 1;
        let piece = self.tokenizer.decode(&[token], false).unwrap_or_default();
        st.generated_text.push_str(&piece);
        match stop {
            StopMode::Json => {
                for ch in piece.chars() {
                    if ch == '{' {
                        st.brace_depth += 1;
                        st.saw_open_brace = true;
                    } else if ch == '}' {
                        st.brace_depth -= 1;
                    }
                }
                if st.saw_open_brace && st.brace_depth <= 0 {
                    return true;
                }
            }
            StopMode::Text => {
                if st.generated_text.contains("<|im_end|>")
                    || st.generated_text.contains("<end_of_turn>")
                {
                    return true;
                }
                for ch in piece.chars() {
                    if ch == '\n' {
                        st.consecutive_newlines += 1;
                    } else {
                        st.consecutive_newlines = 0;
                    }
                }
                if st.consecutive_newlines >= 2 {
                    return true;
                }
            }
            StopMode::Fixed => {}
        }
        false
    }

    /// Shared autoregressive decode loop: given the `logits` from a prefill of
    /// `prompt_len` positions, sample greedily/with temperature, apply the
    /// [`StopMode`] stopping rules, and continue token-by-token via the normal
    /// `forward` path until EOS / a stop condition / `max_tokens`.
    ///
    /// Factored out of [`generate_inner`] so multimodal prefill (e.g. native
    /// audio via [`understand_audio`](Self::understand_audio), which prefills
    /// with [`forward_embeds`]) reuses identical sampling/stop logic.
    #[cfg(feature = "candle-backend")]
    fn decode_loop(
        &mut self,
        mut logits: Tensor,
        prompt_len: usize,
        max_tokens: u32,
        stop: StopMode,
    ) -> NluResult<GenerationResult> {
        let mut st = DecodeState::default();
        let eos_token = self.find_eos_token();

        // Phase 45a: time the autoregressive decode (prefill excluded) for tok/s.
        let decode_start = std::time::Instant::now();
        while st.n_generated < max_tokens {
            // Sample next token. Both quantized_qwen2 and quantized_gemma3
            // `forward` already reduce to the last position internally, so
            // `logits` is `[batch, vocab]` — take row 0 to get `[vocab]`.
            // (Guard the 3-D shape too, in case a future model returns
            // `[batch, seq, vocab]`.)
            let last_logits = match logits.rank() {
                3 => {
                    let seq_dim = logits.dim(1).map_err(|e| NluError::LlmGenerationFailed {
                        reason: format!("dim: {e}"),
                    })?;
                    logits.i((0, seq_dim - 1, ..))
                }
                _ => logits.i((0, ..)),
            }
            .map_err(|e| NluError::LlmGenerationFailed {
                reason: format!("index: {e}"),
            })?;

            let next_token = if self.config.temperature == 0.0 {
                // Greedy
                last_logits
                    .argmax(0)
                    .map_err(|e| NluError::LlmGenerationFailed {
                        reason: format!("argmax: {e}"),
                    })?
                    .to_scalar::<u32>()
                    .map_err(|e| NluError::LlmGenerationFailed {
                        reason: format!("scalar: {e}"),
                    })?
            } else {
                // Temperature sampling
                let scaled = (&last_logits / self.config.temperature as f64)
                    .map_err(|e| NluError::LlmGenerationFailed {
                        reason: format!("scale: {e}"),
                    })?;
                let probs = candle_nn::ops::softmax(&scaled, 0).map_err(|e| {
                    NluError::LlmGenerationFailed {
                        reason: format!("softmax: {e}"),
                    }
                })?;
                let probs_vec: Vec<f32> = probs
                    .to_vec1()
                    .map_err(|e| NluError::LlmGenerationFailed {
                        reason: format!("to_vec1: {e}"),
                    })?;
                sample_from_probs(&probs_vec)
            };

            // Emit the token + apply the StopMode rules (EOS / brace / newline / markers).
            if self.process_token(next_token, eos_token, stop, &mut st) {
                break;
            }

            // Prepare next forward pass (single token).
            let next_input = Tensor::new(&[next_token], &self.device)
                .map_err(|e| NluError::LlmGenerationFailed {
                    reason: format!("tensor: {e}"),
                })?
                .unsqueeze(0)
                .map_err(|e| NluError::LlmGenerationFailed {
                    reason: format!("unsqueeze: {e}"),
                })?;

            logits = self
                .model
                .forward(&next_input, prompt_len + st.n_generated as usize - 1)
                .map_err(|e| NluError::LlmGenerationFailed {
                    reason: format!("decode step: {e}"),
                })?;
        }

        let decode_secs = decode_start.elapsed().as_secs_f32();
        let tok_s = if decode_secs > 0.0 {
            st.n_generated as f32 / decode_secs
        } else {
            0.0
        };
        tracing::debug!(
            tokens_generated = st.n_generated,
            decode_secs,
            tok_s,
            "Candle generation complete"
        );

        Ok(GenerationResult {
            text: st.generated_text,
            tokens_generated: st.n_generated,
            decode_secs,
        })
    }

    /// Whether this turn should use the speculative (prompt-lookup) decode path
    /// ([`spec_decode_loop`](Self::spec_decode_loop)) instead of plain
    /// [`decode_loop`](Self::decode_loop).
    ///
    /// Gated to the cases where it is **lossless and a potential win**:
    /// - gemma4 only — the only loaded family exposing `forward_verify` + the batched verify GEMV;
    /// - enabled — the live per-akh flag (set from `AkhConfig.spec_decode`, toggled at runtime by
    ///   the `inference_config` tool). When no shared options are wired (benches/tests), falls back
    ///   to the `AKH_SPEC_DECODE=1` env flag.
    ///
    /// `generate_inner` then routes by temperature: greedy (`== 0`) →
    /// [`spec_decode_loop`](Self::spec_decode_loop) (matches plain greedy exactly), `> 0` →
    /// [`spec_sample_decode_loop`](Self::spec_sample_decode_loop) (lossless speculative sampling —
    /// same *distribution* as plain sampling, Phase 47e). The throughput win is additionally
    /// conditional on the batched verify GEMV (default-on, Phase 47d).
    #[cfg(feature = "candle-backend")]
    fn spec_decode_eligible(&self) -> bool {
        let enabled = match &self.runtime_options {
            Some(opts) => opts.spec_decode(),
            None => std::env::var("AKH_SPEC_DECODE").as_deref() == Ok("1"),
        };
        enabled && matches!(self.family, ModelFamily::Gemma4)
    }

    /// Speculative (prompt-lookup) **greedy** decode loop — gemma4 only (Phase 47 / ADR 075).
    ///
    /// Produces up to `γ+1` tokens per target forward: each round drafts a continuation from the
    /// token history with an n-gram lookup (no draft model — works for gemma4, whose tokenizer
    /// has no compatible small sibling), verifies the whole span in one `forward_verify`, and
    /// greedily accepts the matching prefix plus one bonus/correction token. KV rollback is free
    /// — the next round simply writes from the new position, overwriting any rejected tail
    /// (device-pos static cache, Phase 46d-2). Every accepted/bonus token is emitted through the
    /// shared [`process_token`](Self::process_token), so stop/detokenization match plain decode
    /// exactly; the generated tokens match plain greedy decode in practice (validated
    /// token-identical on the Phase 47c bench prompts).
    ///
    /// Wins where output echoes context (agentic / RAG / code / structured output) and is
    /// near-neutral otherwise (an empty draft falls back to a cheap `m=1` verify). The throughput
    /// win additionally requires the batched verify GEMV (`AKH_GEMV_BATCH=1`, Phase 47d) — without
    /// it the verify is ~4× a decode and this *loses*; a warning is logged in that case.
    #[cfg(feature = "candle-backend")]
    fn spec_decode_loop(
        &mut self,
        logits: Tensor,
        prompt_tokens: Vec<u32>,
        max_tokens: u32,
        stop: StopMode,
    ) -> NluResult<GenerationResult> {
        use super::spec_decode::{accept_run, DrafterConfig, NgramDrafter};
        let fail = |reason: String| NluError::LlmGenerationFailed { reason };

        if std::env::var("AKH_GEMV_BATCH").as_deref() == Ok("0") {
            tracing::warn!(
                "speculative decode is enabled but AKH_GEMV_BATCH=0 disables the batched verify \
                 GEMV (Phase 47d): the verify forward is then ~4× a single decode, so spec decode \
                 will be SLOWER than plain decode. Unset AKH_GEMV_BATCH to restore the fast path."
            );
        }

        let device = self.device.clone();
        let eos = self.find_eos_token();
        // Echo-gate (min_ngram=2): only draft when at least a 2-token suffix recurs. Lone
        // single-token matches are usually rejected on free-form text — requiring a longer
        // match keeps the precision high so rejected γ-rounds stay rare.
        let drafter = NgramDrafter::new(DrafterConfig {
            min_ngram: 2,
            ..DrafterConfig::default()
        });
        let kv_ctx = match &self.model {
            LoadedModel::Gemma4(m) => m.kv_ctx(),
            _ => return Err(fail("spec decode: gemma4 only".to_string())),
        };
        // On-device argmax → only the token id(s) cross PCIe (no full-logits D2H).
        let argmax_ids = |t: &Tensor| -> NluResult<Vec<u32>> {
            t.argmax(candle_core::D::Minus1)
                .and_then(|a| a.flatten_all())
                .and_then(|a| a.to_vec1::<u32>())
                .map_err(|e| fail(format!("argmax: {e}")))
        };

        let mut st = DecodeState::default();
        let mut all = prompt_tokens; // drafter context: prompt + everything generated
        let mut pos = all.len();
        let decode_start = std::time::Instant::now();

        // First token = argmax(prefill) (last position).
        let cur0 = *argmax_ids(&logits)?
            .last()
            .ok_or_else(|| fail("empty prefill argmax".to_string()))?;
        all.push(cur0);
        let mut cur = cur0;
        let mut stop_now = self.process_token(cur0, eos, stop, &mut st);

        while !stop_now && st.n_generated < max_tokens {
            // Draft from the history, then verify [cur, drafts..] at [pos, ..] in ONE forward.
            // Scope the &mut self.model borrow so process_token (&self) can run after it.
            let (drafts, argmaxes) = {
                let model = match &mut self.model {
                    LoadedModel::Gemma4(m) => m,
                    _ => return Err(fail("spec decode: gemma4 only".to_string())),
                };
                let mut drafts = drafter.propose(&all);
                // Cap γ so the verify write [pos, pos+γ] fits the static KV cache.
                let max_gamma = kv_ctx.saturating_sub(pos + 1);
                if drafts.len() > max_gamma {
                    drafts.truncate(max_gamma);
                }
                let feed: Vec<u32> = std::iter::once(cur).chain(drafts.iter().copied()).collect();
                let feed_t = Tensor::new(feed.as_slice(), &device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| fail(format!("feed tensor: {e}")))?;
                let verify = model
                    .forward_verify(&feed_t, pos)
                    .map_err(|e| fail(format!("verify forward: {e}")))?;
                (drafts, argmax_ids(&verify)?) // argmaxes length = 1 + drafts.len()
            };

            let k = accept_run(&drafts, &argmaxes);
            // Emit the k accepted drafts then the bonus/correction (argmaxes[k] always exists),
            // each through the shared per-token path. Stop mid-run on EOS / a stop rule / budget.
            for i in 0..=k {
                let tok = if i < k { drafts[i] } else { argmaxes[k] };
                all.push(tok);
                if self.process_token(tok, eos, stop, &mut st) {
                    stop_now = true;
                    break;
                }
                if st.n_generated >= max_tokens {
                    stop_now = true;
                    break;
                }
            }
            cur = argmaxes[k]; // bonus becomes the next round's current token
            pos += k + 1;
            if pos + 1 >= kv_ctx {
                break;
            }
        }

        let decode_secs = decode_start.elapsed().as_secs_f32();
        tracing::debug!(
            tokens_generated = st.n_generated,
            decode_secs,
            tok_s = st.n_generated as f32 / decode_secs.max(1e-6),
            "Candle speculative generation complete"
        );
        Ok(GenerationResult {
            text: st.generated_text,
            tokens_generated: st.n_generated,
            decode_secs,
        })
    }

    /// Speculative (prompt-lookup) **sampling** decode loop — gemma4 only (Phase 47e / ADR 075).
    ///
    /// The temperature analogue of [`spec_decode_loop`](Self::spec_decode_loop). Each round drafts
    /// a span from the token history, verifies it in one `forward_verify`, then walks the drafted
    /// positions **accepting drafted token `d` with probability `p_target(d)`** and, on the first
    /// rejection, resampling from the residual `norm(p ∖ {d})` (speculative sampling specialized to
    /// a deterministic point-mass draft — see [`spec_sample_step`](super::spec_decode::spec_sample_step)).
    /// The emitted sequence has the **same distribution** as plain temperature sampling (not
    /// byte-identical, by nature — proven distribution-lossless by `spec_sample_step`'s test).
    /// Shares [`process_token`](Self::process_token) for stop/detokenization; KV rollback is free
    /// (device-pos cache). Engaged only for `temperature > 0`; the throughput win is conditional on
    /// output echoing context (and the batched verify GEMV, default-on Phase 47d).
    #[cfg(feature = "candle-backend")]
    fn spec_sample_decode_loop(
        &mut self,
        logits: Tensor,
        prompt_tokens: Vec<u32>,
        max_tokens: u32,
        stop: StopMode,
    ) -> NluResult<GenerationResult> {
        use super::spec_decode::{categorical, spec_sample_step, DrafterConfig, NgramDrafter};
        use rand::Rng;
        let fail = |reason: String| NluError::LlmGenerationFailed { reason };

        if std::env::var("AKH_GEMV_BATCH").as_deref() == Ok("0") {
            tracing::warn!(
                "speculative sampling is enabled but AKH_GEMV_BATCH=0 disables the batched verify \
                 GEMV (Phase 47d): the verify forward is ~4× a single decode, so this will be \
                 SLOWER than plain sampling. Unset AKH_GEMV_BATCH to restore the fast path."
            );
        }

        let device = self.device.clone();
        let eos = self.find_eos_token();
        // Temperature scale for softmax (guard against a degenerate 0 — eligibility only routes
        // here for temperature > 0, but stay defensive).
        let temp = (self.config.temperature.max(f32::MIN_POSITIVE)) as f64;
        let drafter = NgramDrafter::new(DrafterConfig {
            min_ngram: 2,
            ..DrafterConfig::default()
        });
        let kv_ctx = match &self.model {
            LoadedModel::Gemma4(m) => m.kv_ctx(),
            _ => return Err(fail("spec decode: gemma4 only".to_string())),
        };
        let mut rng = rand::thread_rng();

        // Temperature-softmax a logits tensor to a row-major host probability Vec (one row per
        // leading position). D2H of the full vocab rows is the same order as plain sampling's
        // per-token `to_vec1`; only the needed rows of a verify cross the bus.
        let to_probs = |t: &Tensor| -> NluResult<Vec<f32>> {
            let scaled = (t / temp).map_err(|e| fail(format!("temp scale: {e}")))?;
            candle_nn::ops::softmax(&scaled, candle_core::D::Minus1)
                .and_then(|p| p.flatten_all())
                .and_then(|p| p.to_vec1::<f32>())
                .map_err(|e| fail(format!("softmax/d2h: {e}")))
        };

        let mut st = DecodeState::default();
        let mut all = prompt_tokens;
        let mut pos = all.len();
        let decode_start = std::time::Instant::now();

        // First token: sample from the prefill distribution (last position).
        let cur0 = categorical(&to_probs(&logits)?, rng.r#gen::<f32>());
        all.push(cur0);
        let mut cur = cur0;
        let mut stop_now = self.process_token(cur0, eos, stop, &mut st);

        while !stop_now && st.n_generated < max_tokens {
            // Draft + verify in one forward; pull the per-position probabilities. Scope the
            // &mut self.model borrow so process_token (&self) can run after.
            let (drafts, probs, vocab) = {
                let model = match &mut self.model {
                    LoadedModel::Gemma4(m) => m,
                    _ => return Err(fail("spec decode: gemma4 only".to_string())),
                };
                let mut drafts = drafter.propose(&all);
                let max_gamma = kv_ctx.saturating_sub(pos + 1);
                if drafts.len() > max_gamma {
                    drafts.truncate(max_gamma);
                }
                let feed: Vec<u32> = std::iter::once(cur).chain(drafts.iter().copied()).collect();
                let feed_t = Tensor::new(feed.as_slice(), &device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| fail(format!("feed tensor: {e}")))?;
                let verify = model
                    .forward_verify(&feed_t, pos)
                    .map_err(|e| fail(format!("verify forward: {e}")))?;
                let (_, _seq, vocab) =
                    verify.dims3().map_err(|e| fail(format!("verify dims: {e}")))?;
                (drafts, to_probs(&verify)?, vocab) // probs: (drafts.len()+1) rows × vocab
            };

            // Speculative-sampling accept: accept drafted d_j with prob p_j[d_j]; on the first
            // rejection, resample from the residual and stop. `row(i)` is verify position i.
            let row = |i: usize| &probs[i * vocab..(i + 1) * vocab];
            let mut k = drafts.len();
            let mut bonus: Option<u32> = None;
            for j in 0..drafts.len() {
                let (u, rp) = (rng.r#gen::<f32>(), rng.r#gen::<f32>());
                let (tok, accepted) = spec_sample_step(row(j), drafts[j], u, rp);
                if !accepted {
                    bonus = Some(tok);
                    k = j;
                    break;
                }
            }
            // All drafts accepted → sample the extra (bonus) position; else use the residual draw.
            let bonus =
                bonus.unwrap_or_else(|| categorical(row(drafts.len()), rng.r#gen::<f32>()));

            // Emit the k accepted drafts + the bonus, each through the shared per-token path.
            for i in 0..k {
                all.push(drafts[i]);
                if self.process_token(drafts[i], eos, stop, &mut st) {
                    stop_now = true;
                    break;
                }
                if st.n_generated >= max_tokens {
                    stop_now = true;
                    break;
                }
            }
            if !stop_now {
                all.push(bonus);
                if self.process_token(bonus, eos, stop, &mut st) || st.n_generated >= max_tokens {
                    stop_now = true;
                }
            }
            cur = bonus;
            pos += k + 1;
            if pos + 1 >= kv_ctx {
                break;
            }
        }

        let decode_secs = decode_start.elapsed().as_secs_f32();
        tracing::debug!(
            tokens_generated = st.n_generated,
            decode_secs,
            tok_s = st.n_generated as f32 / decode_secs.max(1e-6),
            "Candle speculative-sampling generation complete"
        );
        Ok(GenerationResult {
            text: st.generated_text,
            tokens_generated: st.n_generated,
            decode_secs,
        })
    }

    /// Prefill the model with `prompt`; return its last-position logits and the
    /// prompt length. Mirrors [`generate_inner`](Self::generate_inner)'s prefill
    /// so the constrained decoder can reuse it without touching the (delicate,
    /// vendored) generate path.
    #[cfg(feature = "candle-backend")]
    fn prefill(&mut self, prompt: &str) -> NluResult<(Tensor, usize)> {
        let encoding = self.tokenizer.encode(prompt, true).map_err(|e| {
            NluError::LlmGenerationFailed {
                reason: format!("tokenization: {e}"),
            }
        })?;
        let mut prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

        if matches!(self.family, ModelFamily::Gemma | ModelFamily::Gemma4) {
            if let Some(bos) = self.tokenizer.token_to_id("<bos>") {
                if prompt_tokens.first() != Some(&bos) {
                    prompt_tokens.insert(0, bos);
                }
            }
        }

        if prompt_tokens.len() > self.context_length {
            return Err(NluError::LlmGenerationFailed {
                reason: format!(
                    "Prompt too long ({} tokens, model context window is {}).",
                    prompt_tokens.len(),
                    self.context_length
                ),
            });
        }

        let input = Tensor::new(prompt_tokens.as_slice(), &self.device)
            .map_err(|e| NluError::LlmGenerationFailed {
                reason: format!("tensor: {e}"),
            })?
            .unsqueeze(0)
            .map_err(|e| NluError::LlmGenerationFailed {
                reason: format!("unsqueeze: {e}"),
            })?;
        let logits = self.model.forward(&input, 0).map_err(|e| {
            NluError::LlmGenerationFailed {
                reason: format!("prefill: {e}"),
            }
        })?;
        Ok((logits, prompt_tokens.len()))
    }

    /// Generate a JSON object under grammar constraint (ADR 070).
    ///
    /// At each decode step only token candidates whose decoded text keeps the
    /// output a valid JSON prefix (per [`JsonConstraint`](super::json_constraint::JsonConstraint))
    /// are eligible; the highest-logit eligible token is chosen. The model still
    /// decides *which* valid token (the argument semantics); the grammar
    /// guarantees the bytes parse — so a tool call is never dropped to malformed
    /// JSON. This is the in-process advantage Candle has over an HTTP backend:
    /// we own the sampling loop. See `tool_call.gbnf`.
    #[cfg(feature = "candle-backend")]
    pub fn generate_constrained_json(&mut self, prompt: &str, max_tokens: u32) -> NluResult<String> {
        let (mut logits, prompt_len) = self.prefill(prompt)?;
        let mut constraint = super::json_constraint::JsonConstraint::object();
        let mut out = String::new();
        let mut n = 0u32;
        let eos = self.find_eos_token();

        while n < max_tokens {
            let last = match logits.rank() {
                3 => {
                    let seq = logits.dim(1).map_err(|e| NluError::LlmGenerationFailed {
                        reason: format!("dim: {e}"),
                    })?;
                    logits.i((0, seq - 1, ..))
                }
                _ => logits.i((0, ..)),
            }
            .map_err(|e| NluError::LlmGenerationFailed {
                reason: format!("index: {e}"),
            })?;
            let scores: Vec<f32> = last.to_vec1().map_err(|e| NluError::LlmGenerationFailed {
                reason: format!("to_vec1: {e}"),
            })?;

            // Highest-logit token whose decoded text keeps the JSON valid.
            let mut chosen: Option<(u32, String)> = None;
            for id in top_k_indices(&scores, 96) {
                if Some(id) == eos {
                    continue;
                }
                let piece = self.tokenizer.decode(&[id], false).unwrap_or_default();
                if piece.is_empty() {
                    continue;
                }
                if constraint.accepts(&piece) {
                    chosen = Some((id, piece));
                    break;
                }
            }
            let Some((id, piece)) = chosen else { break };
            let _ = constraint.push(&piece);
            out.push_str(&piece);
            n += 1;
            if constraint.is_complete() {
                break;
            }

            let next_input = Tensor::new(&[id], &self.device)
                .map_err(|e| NluError::LlmGenerationFailed {
                    reason: format!("tensor: {e}"),
                })?
                .unsqueeze(0)
                .map_err(|e| NluError::LlmGenerationFailed {
                    reason: format!("unsqueeze: {e}"),
                })?;
            logits = self
                .model
                .forward(&next_input, prompt_len + n as usize - 1)
                .map_err(|e| NluError::LlmGenerationFailed {
                    reason: format!("decode step: {e}"),
                })?;
        }
        Ok(out)
    }

    /// Select the best available device.
    #[cfg(feature = "candle-backend")]
    fn select_device() -> Device {
        // Try Metal first (macOS with Apple Silicon).
        #[cfg(feature = "metal")]
        {
            if let Ok(device) = Device::new_metal(0) {
                tracing::info!("using Metal device for Candle inference");
                return device;
            }
        }
        // Try CUDA.
        #[cfg(feature = "cuda")]
        {
            if let Ok(device) = Device::new_cuda(0) {
                tracing::info!("using CUDA device for Candle inference");
                return device;
            }
        }
        // AMD ROCm/HIP via the vendored candle-core HIP backend (Phase 45f / ADR 073).
        // The `hip` feature fills candle's `cuda` module slot, so `new_cuda(0)` builds a
        // real `HipDevice`. The whole forward then runs GPU-resident — quantized weights
        // live as `QHipStorage` (Q4_K/Q6_K), matmuls go through rocBLAS, elementwise/
        // reduce/rope/softmax/rms-norm stay on device — instead of the per-op PCIe offload
        // path (`AKH_GPU_MATMUL`), which round-tripped every op across the bus.
        #[cfg(feature = "hip")]
        {
            match Device::new_cuda(0) {
                Ok(device) => {
                    tracing::info!(
                        "using AMD HIP device (Device::Hip via candle cuda slot) for Candle inference"
                    );
                    return device;
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "HIP device construction failed; falling back to CPU. Verify the ROCm \
                         runtime (ROCM_PATH, default /opt/rocm) and that the GPU is visible \
                         (`rocminfo`)."
                    );
                }
            }
        }
        // No GPU backend selected (or HIP construction failed): probe the host so the log
        // tells the user *why* we are on CPU and what to do about it.
        Self::report_rocm_status();
        tracing::info!("using CPU device for Candle inference");
        Device::Cpu
    }

    /// Inspect the host ROCm stack and emit an honest, actionable log line.
    ///
    /// This does not change the device (no Candle HIP backend exists yet) — it surfaces the
    /// [`super::gpu`] diagnostic so a user on an AMD box knows whether the GPU is ready and
    /// what is still missing (userspace install vs. the `rocm` feature/fork).
    #[cfg(feature = "candle-backend")]
    fn report_rocm_status() {
        use super::gpu::{RocmStatus, detect_rocm};
        match detect_rocm() {
            RocmStatus::Ready { runtime } => {
                // Userspace is present; only the Candle HIP backend (fork) is missing.
                tracing::warn!(
                    runtime = %runtime,
                    "ROCm runtime present but the Candle HIP backend is not built \
                     (feature `rocm`, ADR 068) — running on CPU. See \
                     docs/ai/plans/2026-06-04-gpu-rocm-candle.md"
                );
            }
            RocmStatus::KernelReadyUserspaceMissing => {
                tracing::warn!(
                    "AMD GPU compute node /dev/kfd present but ROCm userspace missing — \
                     install it to enable GPU inference (no reboot needed): \
                     `sudo apt-get install -y rocminfo rocm-hip-runtime rocblas hipblas` \
                     after adding the ROCm 6.x repo. See ADR 068."
                );
            }
            RocmStatus::Unavailable { reason } => {
                tracing::debug!(%reason, "no ROCm compute stack — CPU inference");
            }
        }
    }

    /// Find the EOS token ID for the loaded model (family-aware).
    #[cfg(feature = "candle-backend")]
    fn find_eos_token(&self) -> Option<u32> {
        match self.family {
            // Gemma 2/3: <end_of_turn> (chat) or <eos>.
            ModelFamily::Gemma => self
                .tokenizer
                .token_to_id("<end_of_turn>")
                .or_else(|| self.tokenizer.token_to_id("<eos>"))
                .or_else(|| self.tokenizer.token_to_id("</s>")),
            // Gemma 4: the turn-end marker `<turn|>` (id 106) is the chat EOS;
            // fall back to <eos> for raw completion.
            ModelFamily::Gemma4 => self
                .tokenizer
                .token_to_id("<turn|>")
                .or_else(|| self.tokenizer.token_to_id("<eos>"))
                .or_else(|| self.tokenizer.token_to_id("</s>")),
            // Qwen2.5: <|endoftext|> (151643) or <|im_end|> (151645).
            ModelFamily::Qwen2 => self
                .tokenizer
                .token_to_id("<|endoftext|>")
                .or_else(|| self.tokenizer.token_to_id("<|im_end|>"))
                .or_else(|| self.tokenizer.token_to_id("</s>")),
        }
    }

    /// Access the config.
    pub fn config(&self) -> &CandleConfig {
        &self.config
    }

    /// Update the config (e.g., to enable hidden state extraction).
    pub fn set_config(&mut self, config: CandleConfig) {
        self.config = config;
    }
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// Stop mode for the generation loop.
#[derive(Debug, Clone, Copy)]
enum StopMode {
    /// Stop when a top-level JSON object closes (brace depth tracking).
    Json,
    /// Stop on EOS, double-newline, or chat template end marker.
    Text,
    /// Decode exactly `max_tokens`, ignoring EOS and stop markers — a pure
    /// throughput probe for the Phase 45 benchmark (steady-state decode tok/s,
    /// independent of generated content).
    Fixed,
}

/// Internal generation result.
struct GenerationResult {
    text: String,
    tokens_generated: u32,
    /// Wall-clock spent in the autoregressive decode loop (prefill excluded), in
    /// seconds. Drives the Phase 45a tok/s baseline. Zero when no token was decoded.
    decode_secs: f32,
}

/// Running state shared by the plain [`CandleBackend::decode_loop`] and the speculative
/// [`CandleBackend::spec_decode_loop`]: the accumulated text, the emitted-token count, and
/// the [`StopMode`] bookkeeping (brace depth / newline run). Bundled so both loops drive
/// [`CandleBackend::process_token`] identically — the speculative loop emits a *batch* of
/// accepted tokens per forward but each goes through the same per-token path.
#[cfg(feature = "candle-backend")]
#[derive(Default)]
struct DecodeState {
    generated_text: String,
    n_generated: u32,
    brace_depth: i32,
    saw_open_brace: bool,
    consecutive_newlines: u32,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the first JSON object from a string (find first `{`, match to closing `}`).
fn extract_json_object(raw: &str) -> &str {
    if let Some(start) = raw.find('{') {
        if let Some(end) = raw.rfind('}') {
            return &raw[start..=end];
        }
    }
    raw
}

/// Clean up generated text: strip chat template tokens and trailing noise.
fn clean_generated_text(raw: &str) -> String {
    // Gemma 4 wraps responses in a typed channel stream
    // (`<|channel>thought\n{reasoning}<channel|>{answer}`). Parse it and keep the
    // user-facing answer (non-thought segments); the reasoning is preserved for
    // callers that want it via [`CandleBackend::generate_channels`] /
    // [`crate::nlu::channel::ChannelStream`]. Then strip any stray markers.
    let answer = crate::nlu::channel::ChannelStream::parse(raw).answer();
    let text = answer
        .replace("<|im_end|>", "")
        .replace("<|im_start|>", "")
        .replace("<|turn>", "")
        .replace("<turn|>", "")
        .trim()
        .to_string();

    // Strip everything from the first '#' onward (hashtag spam).
    if let Some(hash_pos) = text.find('#') {
        let mut trimmed = text[..hash_pos].trim_end().to_string();
        // Trim trailing incomplete sentences.
        if !trimmed.is_empty()
            && !trimmed.ends_with('.')
            && !trimmed.ends_with('!')
            && !trimmed.ends_with('?')
        {
            if let Some(last_period) = trimmed.rfind(|c: char| c == '.' || c == '!' || c == '?') {
                trimmed.truncate(last_period + 1);
            }
        }
        return trimmed;
    }

    text
}

/// Indices of the `k` highest `scores`, in descending score order.
///
/// Used by the grammar-constrained decoder ([`CandleBackend::generate_constrained_json`])
/// to consider only the most-likely tokens when filtering for grammar validity.
#[cfg(feature = "candle-backend")]
fn top_k_indices(scores: &[f32], k: usize) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..scores.len() as u32).collect();
    let k = k.min(idx.len());
    if k == 0 {
        return Vec::new();
    }
    // Partition so the top-k by score are first, then sort just those.
    idx.select_nth_unstable_by(k - 1, |&a, &b| {
        scores[b as usize]
            .partial_cmp(&scores[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k);
    idx.sort_unstable_by(|&a, &b| {
        scores[b as usize]
            .partial_cmp(&scores[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx
}

/// Simple weighted random sampling from a probability distribution.
#[cfg(feature = "candle-backend")]
fn sample_from_probs(probs: &[f32]) -> u32 {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let r: f32 = rng.r#gen();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if cumulative >= r {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_json_from_mixed_output() {
        let raw = r#"Here is the result:
{"Triple":{"subject":{"EntityRef":{"label":"dogs","symbol_id":null}}}}
Some trailing text."#;
        let json = extract_json_object(raw);
        assert!(json.starts_with('{'));
        assert!(json.ends_with('}'));
        assert!(json.contains("Triple"));
    }

    #[test]
    fn extract_json_no_braces() {
        let raw = "no json here";
        let json = extract_json_object(raw);
        assert_eq!(json, "no json here");
    }

    #[test]
    fn clean_text_strips_chat_tokens() {
        let raw = "<|im_start|>assistant\nHello world.<|im_end|>";
        let cleaned = clean_generated_text(raw);
        assert_eq!(cleaned, "assistant\nHello world.");
    }

    #[test]
    fn clean_text_strips_hashtags() {
        let raw = "This is a response. #hashtag #spam";
        let cleaned = clean_generated_text(raw);
        assert_eq!(cleaned, "This is a response.");
    }

    #[test]
    fn try_load_returns_none_for_missing() {
        let nonexistent = PathBuf::from("/tmp/akh-medu-nonexistent-candle-test");
        assert!(CandleBackend::try_load(&nonexistent).is_none());
    }

    #[test]
    fn default_config() {
        let config = CandleConfig::default();
        assert_eq!(config.max_tokens, 512);
        assert_eq!(config.temperature, 0.0);
        assert!(!config.extract_hidden);
        assert!(config.hidden_layer.is_none());
    }
}
