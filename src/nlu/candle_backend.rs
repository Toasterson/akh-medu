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
use super::llm_translator::{LlmTranslation, build_prompt_with_context, parse_abstree_json};

#[cfg(feature = "candle-backend")]
use candle_core::{DType, Device, IndexOp, Tensor};

use super::span_projection::LayerMean;

/// Which model family is loaded — drives chat-prompt formatting and EOS tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    /// Qwen2.5 (ChatML `<|im_start|>` template, `<|im_end|>` / `<|endoftext|>`).
    Qwen2,
    /// Qwen3.5/Qwen3.8 hybrid decoder (ChatML, gated-delta + full-attention layers).
    ///
    /// Family recognition lands before the loader deliberately: until the dedicated
    /// tensor schema is implemented, loading fails closed instead of interpreting a
    /// Qwen3.8 GGUF as the superficially similar Qwen2 architecture.
    Qwen35,
    /// Gemma 2/3 (`<start_of_turn>` template, `<end_of_turn>` EOS).
    Gemma,
    /// Gemma 4 — unified encoder-free multimodal (native audio). Shares Gemma's
    /// `<start_of_turn>` template/EOS but a distinct decoder (per-layer attention
    /// geometry, partial rotary, GELU MLP, logit softcap). See ADR 054.
    Gemma4,
}

/// Token-level chat dialect selected once from the model family.
///
/// Keeping this mapping beside [`ModelFamily`] means adding a family cannot
/// silently require the same grouping change in every prompt renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(feature = "candle-backend")]
pub(crate) enum ChatTemplate {
    ChatMl,
    Gemma,
    Gemma4,
}

impl ModelFamily {
    /// Detect the family from a GGUF `general.architecture` string.
    fn from_architecture(arch: &str) -> Self {
        let a = arch.to_lowercase();
        // Check the more specific `gemma4` before the generic `gemma` substring.
        if a.contains("gemma4") {
            ModelFamily::Gemma4
        } else if a.contains("gemma") {
            ModelFamily::Gemma
        } else if a.contains("qwen35") || a.contains("qwen3.5") || a.contains("qwen3_5") {
            ModelFamily::Qwen35
        } else {
            ModelFamily::Qwen2
        }
    }

    #[cfg(feature = "candle-backend")]
    pub(crate) const fn chat_template(self) -> ChatTemplate {
        match self {
            ModelFamily::Qwen2 | ModelFamily::Qwen35 => ChatTemplate::ChatMl,
            ModelFamily::Gemma => ChatTemplate::Gemma,
            ModelFamily::Gemma4 => ChatTemplate::Gemma4,
        }
    }

    const fn eos_markers(self) -> &'static [&'static str] {
        match self {
            ModelFamily::Gemma => &["<end_of_turn>", "<eos>", "</s>"],
            ModelFamily::Gemma4 => &["<turn|>", "<eos>", "</s>"],
            ModelFamily::Qwen2 | ModelFamily::Qwen35 => &["<|endoftext|>", "<|im_end|>", "</s>"],
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
    Qwen35(super::vendored::quantized_qwen35::InferenceSession),
    Gemma(super::vendored::quantized_gemma3::ModelWeights),
    Gemma4(super::vendored::quantized_gemma4::ModelWeights),
}

#[cfg(feature = "candle-backend")]
impl LoadedModel {
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> candle_core::Result<Tensor> {
        match self {
            LoadedModel::Qwen2(m) => m.forward(x, index_pos),
            LoadedModel::Qwen35(m) => m.forward(x, index_pos),
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
            LoadedModel::Qwen35(m) => m.forward_with_hidden(x, index_pos),
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

/// Explicit placement for resident runtimes. Unsupported devices fail closed;
/// only the older convenience loader performs automatic device selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandleDevice {
    Cpu,
    Hip { ordinal: usize },
    Cuda { ordinal: usize },
    Metal { ordinal: usize },
    Vulkan { ordinal: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandleFinishReason {
    Stop,
    Length,
}

/// A complete, uncleaned model reply. `Length` must not authorize tool effects.
#[derive(Debug, Clone)]
pub struct CandleCompletion {
    pub text: String,
    pub prompt_tokens: usize,
    /// Exact previously forwarded prompt tokens reused by this call.
    pub reused_prompt_tokens: usize,
    pub tokens_generated: u32,
    pub finish_reason: CandleFinishReason,
    /// Tokenization, prefill and first sampled token, including device readback.
    /// Excludes model loading and the caller's queue wait.
    pub time_to_first_token: std::time::Duration,
    /// All subsequent decode/sample steps, including an EOS step when present.
    pub decode_time: std::time::Duration,
    /// Sampled positions after the first, including EOS (unlike tokens_generated).
    pub decode_steps: u32,
    /// Seconds per decode step, in order, one per position counted by
    /// `decode_steps`. Measured at the sampling point, which reads the result
    /// back and therefore includes the queued GPU work for that step.
    ///
    /// This is the resolution needed to tell a periodic expensive step, such as
    /// a cache reallocation, from a uniformly slower kernel: the aggregate rate
    /// cannot distinguish them.
    pub step_seconds: Vec<f64>,
    /// Pooled residual means of the requested prompt span, when it was newly
    /// forwarded by this call. `None` is an explicit miss, never a replay.
    pub prompt_span_taps: Option<PromptSpanTaps>,
}

/// Pooled post-block residual means for one newly forwarded prompt span,
/// read back from the device. Layer identity is preserved for projection.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptSpanTaps {
    pub layers: Vec<LayerMean>,
    /// Absolute token positions of the span inside the forwarded prompt.
    pub token_span: std::ops::Range<usize>,
}

/// Ask a keyed completion to pool the residual streams over a prompt byte range.
#[derive(Debug, Clone)]
pub struct CaptureRequest<'a> {
    /// Byte range inside the rendered prompt; must align with token boundaries.
    pub byte_span: std::ops::Range<usize>,
    /// Distinct target block indices, at most five.
    pub layers: &'a [usize],
}

/// Caller decision after the prompt taps were offered, before decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecallDecision {
    /// Decode from the prompt as forwarded.
    Continue,
    /// Rewind to the pre-call state and forward this re-rendered prompt instead.
    Reprompt(String),
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
    /// Optional smaller physical window for resident session generation.
    completion_context_limit: Option<usize>,
    #[cfg(feature = "candle-backend")]
    active_completion_session: Option<String>,
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
        Self::load_with_device(model_path, tokenizer_path, config, Self::select_device())
    }

    /// Load on exactly the requested device, without a CPU fallback.
    #[cfg(feature = "candle-backend")]
    pub fn load_on_device(
        model_path: &Path,
        tokenizer_path: &Path,
        config: CandleConfig,
        device: CandleDevice,
    ) -> NluResult<Self> {
        let unavailable = |name| NluError::ModelLoadFailed {
            reason: format!("{name} backend is not available in this build"),
        };
        #[cfg(any(feature = "hip", feature = "metal", feature = "vulkan"))]
        let failure = |error: candle_core::Error| NluError::ModelLoadFailed {
            reason: format!("requested device {device:?} failed: {error}"),
        };
        let selected = match device {
            CandleDevice::Cpu => Device::Cpu,
            #[cfg(feature = "hip")]
            CandleDevice::Hip { ordinal: 0 } => Device::new_cuda(0).map_err(failure)?,
            #[cfg(feature = "hip")]
            CandleDevice::Hip { .. } => return Err(NluError::ModelLoadFailed {
                reason: "resident HIP currently supports ordinal 0 only: allocator and rocBLAS state are process-wide".into(),
            }),
            #[cfg(not(feature = "hip"))]
            CandleDevice::Hip { .. } => return Err(unavailable("HIP")),
            CandleDevice::Cuda { .. } => return Err(unavailable("CUDA")),
            #[cfg(feature = "metal")]
            CandleDevice::Metal { ordinal } => Device::new_metal(ordinal).map_err(failure)?,
            #[cfg(not(feature = "metal"))]
            CandleDevice::Metal { .. } => return Err(unavailable("Metal")),
            #[cfg(feature = "vulkan")]
            CandleDevice::Vulkan { ordinal } => Device::new_vulkan(ordinal).map_err(failure)?,
            #[cfg(not(feature = "vulkan"))]
            CandleDevice::Vulkan { .. } => return Err(unavailable("Vulkan")),
        };
        Self::load_with_device(model_path, tokenizer_path, config, selected)
    }

    #[cfg(feature = "candle-backend")]
    fn load_with_device(
        model_path: &Path,
        tokenizer_path: &Path,
        config: CandleConfig,
        device: Device,
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

        tracing::info!(
            model = %model_path.display(),
            device = ?device,
            "loading Candle GGUF model"
        );

        // Load GGUF model.
        let mut file = std::fs::File::open(model_path).map_err(|e| NluError::ModelLoadFailed {
            reason: format!("open GGUF: {e}"),
        })?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file).map_err(|e| {
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
            ModelFamily::Qwen35 => {
                let m = super::vendored::quantized_qwen35::InferenceSession::from_gguf(
                    content, &mut file, &device,
                )
                .map_err(|e| NluError::ModelLoadFailed {
                    reason: format!("load Qwen3.5/Qwen3.8 weights: {e}"),
                })?;
                LoadedModel::Qwen35(m)
            }
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
            completion_context_limit: None,
            active_completion_session: None,
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

    /// Bound resident prompt plus output memory without changing model metadata
    /// or silently discarding history. Operators must choose a tested physical
    /// context window, which can be smaller than the model's advertised maximum.
    #[cfg(feature = "candle-backend")]
    pub fn set_context_limit(&mut self, limit: usize) -> NluResult<()> {
        if limit == 0 || limit > self.context_length {
            return Err(NluError::ModelLoadFailed {
                reason: format!(
                    "context limit {limit} must be positive and at most model capability {}",
                    self.context_length
                ),
            });
        }
        self.completion_context_limit = Some(limit);
        Ok(())
    }

    /// Shader specialization cache counters when this backend is resident on a
    /// Vulkan device. `None` on other devices. These count application-held
    /// residency and compilation events, not driver-internal caches or
    /// compiled pipeline bytes.
    #[cfg(feature = "vulkan")]
    pub fn vulkan_pipeline_cache_stats(&self) -> Option<candle_core::PipelineCacheStats> {
        match &self.device {
            Device::Vulkan(device) => device.pipeline_cache_stats().ok(),
            _ => None,
        }
    }

    /// Consume this backend and return unused cached device memory to the driver.
    ///
    /// HIP callers must serialize this with inference and graph capture, just as
    /// they serialize the process-wide HIP compute stream. The native runtime
    /// calls it only when replacing resident weights, never between ordinary
    /// turns. Other devices release their owned allocations through normal drop.
    /// The returned byte count covers the HIP free list, not all device memory.
    #[cfg(feature = "candle-backend")]
    pub fn unload(self) -> NluResult<usize> {
        #[cfg(feature = "hip")]
        let release_hip_cache = self.device.is_cuda();
        drop(self);
        #[cfg(feature = "hip")]
        if release_hip_cache {
            return candle_core::cuda::release_cached_memory().map_err(|error| {
                NluError::ModelLoadFailed {
                    reason: format!("releasing unused HIP memory after model unload: {error}"),
                }
            });
        }
        Ok(0)
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
            .or_else(|| {
                ggufs
                    .iter()
                    .find(|p| p.file_name().and_then(|n| n.to_str()) == Some(CANDLE_MODEL_FILE))
            })
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

    /// Like [`generate`](Self::generate) but returns the **raw** decoded text without
    /// [`clean_generated_text`] (which keeps only the answer channel and strips markers). The
    /// agentic tool path needs this: Gemma 4 emits its `<tool_call>` inside the *thought* channel,
    /// so cleaning to the answer channel discards the call before it can be parsed.
    #[cfg(feature = "candle-backend")]
    pub fn generate_raw(&mut self, prompt: &str, max_tokens: u32) -> NluResult<String> {
        Ok(self
            .generate_inner(prompt, max_tokens, StopMode::Text)?
            .text)
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
            _ => {
                return Err(fail(
                    "graph replay probe currently supports gemma3 only".to_string(),
                ));
            }
        };

        // The embedding lookup stays OUTSIDE the captured region (host-sync D2H+H2D).
        let embed = model
            .embed(&tok_input)
            .map_err(|e| fail(format!("embed: {e}")))?;

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
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
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
        let eager_tok_s = if eager_secs > 0.0 {
            iters as f32 / eager_secs
        } else {
            0.0
        };

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
                graph
                    .replay()
                    .map_err(|e| fail(format!("replay warmup: {e}")))?;
                let first_vec = to_vec0(&replay_logits, "replay-first")?;

                let replay_t0 = std::time::Instant::now();
                for _ in 0..iters {
                    graph.replay().map_err(|e| fail(format!("replay: {e}")))?;
                }
                let replay_secs = replay_t0.elapsed().as_secs_f32();
                replay_tok_s = if replay_secs > 0.0 {
                    iters as f32 / replay_secs
                } else {
                    0.0
                };
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
                ));
            }
        };
        // The embedding lookup is a host-sync op; the graph-ready forward starts from it.
        let embed = model
            .embed(&tok_input)
            .map_err(|e| fail(format!("embed: {e}")))?;

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
                ));
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

        let gctx =
            GraphDecodeCtx::new(&device, read_len).map_err(|e| fail(format!("graph ctx: {e}")))?;

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
        use super::spec_decode::{DrafterConfig, NgramDrafter, accept_run};
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
        match self.family.chat_template() {
            // Gemma 2/3 has no system role; fold it into the user turn.
            ChatTemplate::Gemma => format!(
                "<start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n<start_of_turn>model\n"
            ),
            // Gemma 4 uses a new turn format: `<|turn>role\n ... <turn|>\n` with the
            // generation prompt `<|turn>model\n`. (BOS is prepended in generate_inner.)
            ChatTemplate::Gemma4 => {
                format!("<|turn>user\n{system}\n\n{user}<turn|>\n<|turn>model\n")
            }
            ChatTemplate::ChatMl => format!(
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
    pub fn understand_audio(&mut self, pcm_16k_mono: &[f32], question: &str) -> NluResult<String> {
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
        let mut ids: Vec<u32> = Vec::with_capacity(1 + pre.len() + 1 + n_frames + 1 + post.len());
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
                let base =
                    m.embed_tokens(&id_tensor)
                        .map_err(|e| NluError::LlmGenerationFailed {
                            reason: format!("embed_tokens: {e}"),
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

    /// Path to the loaded model's multimodal projector.
    ///
    /// The Gemma sidecar (`<model-stem>.mmproj.gguf`) is tried first. A
    /// content-addressed model such as the pinned Ptah blob has no derivable
    /// sidecar name, so the projector is resolved through the manifest that
    /// lists both layers; see [`crate::nlu::projector_artifact`].
    #[cfg(feature = "candle-backend")]
    fn mmproj_path(&self) -> NluResult<PathBuf> {
        crate::nlu::projector_artifact::locate(&self.model_path).ok_or_else(|| {
            NluError::ModelNotFound {
                path: self.model_path.with_extension("mmproj.gguf"),
            }
        })
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
        let encoding =
            self.tokenizer
                .encode(input, true)
                .map_err(|e| NluError::LlmGenerationFailed {
                    reason: format!("tokenization: {e}"),
                })?;
        let token_ids = encoding.get_ids();
        let input_tensor = Tensor::new(token_ids, &self.device)
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
        let hidden_vec: Vec<f32> = hidden.i((0, ..)).and_then(|h| h.to_vec1()).map_err(|e| {
            NluError::LlmGenerationFailed {
                reason: format!("hidden to_vec1: {e}"),
            }
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
    pub fn compress_kv_vector(
        values: &[f32],
        seed: u64,
    ) -> crate::vsa::polar_quant::QuantizedVector {
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

    /// Resident KV bytes this model adds per additional context token.
    ///
    /// Only the attention blocks grow with context; the recurrent blocks hold a
    /// fixed-size state. `None` for families whose growth is not modelled here.
    /// The figure covers key and value tensors only, not activations, weights or
    /// allocator overhead, so it is a floor for planning rather than a
    /// prediction of device occupancy.
    #[cfg(feature = "candle-backend")]
    pub fn kv_bytes_per_token(&self) -> Option<usize> {
        match &self.model {
            LoadedModel::Qwen35(model) => Some(model.kv_bytes_per_token()),
            _ => None,
        }
    }

    /// Resident KV bytes for a full `tokens`-long context, when modelled.
    #[cfg(feature = "candle-backend")]
    pub fn projected_kv_bytes(&self, tokens: usize) -> Option<usize> {
        self.kv_bytes_per_token()
            .map(|per_token| per_token.saturating_mul(tokens))
    }

    /// Round-trip quality of PolarQuant on this model's resident KV cache.
    ///
    /// Measurement only: nothing is compressed in the live path. The unit is the
    /// per-head, per-position vector of `head_dim` values, which is what a paged
    /// KV compressor would actually quantize. Keys are post-RoPE and values are
    /// not, so they are reported separately rather than pooled.
    #[cfg(feature = "candle-backend")]
    pub fn kv_quantization_report(&self, seed: u64) -> NluResult<Vec<KvLayerQuality>> {
        self.kv_report_with(&move |values| {
            let mean = values.iter().map(|v| *v as f64).sum::<f64>() / values.len() as f64;
            let variance = values
                .iter()
                .map(|v| (*v as f64 - mean).powi(2))
                .sum::<f64>()
                / values.len() as f64;
            Self::decompress_kv_vector(
                &Self::compress_kv_vector(values, seed),
                seed,
                variance.sqrt() as f32,
            )
        })
    }

    /// The same report for TurboQuant at a chosen bit-width.
    ///
    /// The rotation and codebook are built once for the head dimension and
    /// reused across every vector, which is how an online quantizer would run.
    #[cfg(feature = "candle-backend")]
    pub fn kv_turboquant_report(&self, seed: u64, bits: u8) -> NluResult<Vec<KvLayerQuality>> {
        use std::cell::RefCell;
        // Head dimension is constant across layers, but build lazily so the
        // dimension comes from the tensors rather than an assumption.
        let quantizer: RefCell<Option<crate::nlu::turbo_quant::TurboQuant>> = RefCell::new(None);
        self.kv_report_with(&move |values| {
            let mut slot = quantizer.borrow_mut();
            let turbo = slot.get_or_insert_with(|| {
                crate::nlu::turbo_quant::TurboQuant::new(values.len(), bits, seed)
            });
            turbo.dequantize(&turbo.quantize(values))
        })
    }

    #[cfg(feature = "candle-backend")]
    fn kv_report_with(
        &self,
        round_trip: &dyn Fn(&[f32]) -> Vec<f32>,
    ) -> NluResult<Vec<KvLayerQuality>> {
        let LoadedModel::Qwen35(model) = &self.model else {
            return Err(NluError::ModelLoadFailed {
                reason: "KV quantization report requires the Qwen35 family".into(),
            });
        };
        let mut report = Vec::new();
        for (layer, key, value) in model.attention_states() {
            report.push(KvLayerQuality {
                layer,
                positions: key.dim(2).map_err(|e| NluError::ModelLoadFailed {
                    reason: format!("kv state shape: {e}"),
                })?,
                key: measure_kv_tensor(key, round_trip)?,
                value: measure_kv_tensor(value, round_trip)?,
            });
        }
        Ok(report)
    }

    // ── Internal generation loop ─────────────────────────────────────────

    /// Generate a session reply without NLU cleanup or blank-line stopping.
    ///
    /// The caller owns compute serialization. `check` runs before prefill and
    /// between GPU steps, and may return a cancellation/deadline error. It cannot
    /// interrupt a kernel already submitted to the device. Each call starts a
    /// fresh prefill, so no other conversation's position-dependent state is reused.
    #[cfg(feature = "candle-backend")]
    pub fn generate_completion(
        &mut self,
        prompt: &str,
        max_tokens: u32,
        check: &mut dyn FnMut() -> NluResult<()>,
    ) -> NluResult<CandleCompletion> {
        self.generate_completion_scoped(
            None,
            prompt,
            None,
            max_tokens,
            check,
            &mut |_| Ok(RecallDecision::Continue),
        )
    }

    /// Reuse one active conversation's exact forwarded prefix. The caller must
    /// supply a stable key scoped to its canonical session and provider binding.
    /// Another key evicts only mutable inference state, not immutable weights.
    #[cfg(feature = "candle-backend")]
    pub fn generate_completion_for_session(
        &mut self,
        session_key: &str,
        prompt: &str,
        max_tokens: u32,
        check: &mut dyn FnMut() -> NluResult<()>,
    ) -> NluResult<CandleCompletion> {
        self.generate_completion_for_session_with_capture(
            session_key,
            prompt,
            None,
            max_tokens,
            check,
            &mut |_| Ok(RecallDecision::Continue),
        )
    }

    /// Keyed completion that pools post-block residual means over one newly
    /// forwarded prompt byte span and offers them to `recall` before decoding.
    ///
    /// The taps are `None` when the span is not aligned to token boundaries or
    /// already lies in the reused prefix; generation still proceeds. `recall`
    /// runs inside the compute lane after prefill: returning
    /// [`RecallDecision::Reprompt`] rewinds to the state before this call and
    /// forwards the re-rendered prompt (reusing whatever prefix still matches),
    /// which is the honest memory-before-query layout rather than a same-pass
    /// splice. A `recall` error discards the conversation cache like any other
    /// generation failure. Only Qwen3.5 supports capture; other families report
    /// a miss.
    #[cfg(feature = "candle-backend")]
    pub fn generate_completion_for_session_with_capture(
        &mut self,
        session_key: &str,
        prompt: &str,
        capture: Option<CaptureRequest<'_>>,
        max_tokens: u32,
        check: &mut dyn FnMut() -> NluResult<()>,
        recall: &mut dyn FnMut(Option<&PromptSpanTaps>) -> NluResult<RecallDecision>,
    ) -> NluResult<CandleCompletion> {
        if session_key.is_empty() {
            self.discard_completion_state();
            return Err(NluError::LlmGenerationFailed {
                reason: "empty completion session key".into(),
            });
        }
        self.generate_completion_scoped(Some(session_key), prompt, capture, max_tokens, check, recall)
    }

    /// Target block count and residual width of a resident Qwen3.5 model.
    #[cfg(feature = "candle-backend")]
    pub fn qwen35_shape(&self) -> Option<(usize, usize)> {
        match &self.model {
            LoadedModel::Qwen35(model) => Some((model.block_count(), model.hidden_size())),
            _ => None,
        }
    }

    /// Discard uncommitted or invalid response state while retaining weights.
    #[cfg(feature = "candle-backend")]
    pub fn discard_completion_state(&mut self) {
        self.active_completion_session = None;
        if let LoadedModel::Qwen35(model) = &mut self.model {
            *model = model.new_session();
        }
    }

    #[cfg(feature = "candle-backend")]
    fn generate_completion_scoped(
        &mut self,
        session_key: Option<&str>,
        prompt: &str,
        capture: Option<CaptureRequest<'_>>,
        max_tokens: u32,
        check: &mut dyn FnMut() -> NluResult<()>,
        recall: &mut dyn FnMut(Option<&PromptSpanTaps>) -> NluResult<RecallDecision>,
    ) -> NluResult<CandleCompletion> {
        if session_key.is_none() || self.active_completion_session.as_deref() != session_key {
            self.discard_completion_state();
            self.active_completion_session = session_key.map(str::to_owned);
        }
        // Cached recurrence must remain consistent with full prompt prefill.
        // Q8 decode activations failed that gate even when greedy IDs agreed.
        // Scope precision to this synchronous call, leaving other users' Q8
        // policy and the compressed GGUF weights untouched.
        #[cfg(feature = "hip")]
        let result =
            if session_key.is_some() && self.family == ModelFamily::Qwen35 && self.device.is_cuda()
            {
                candle_core::cuda::with_full_precision_activations(|| {
                    self.generate_completion_inner(prompt, capture, max_tokens, check, recall)
                })
            } else {
                self.generate_completion_inner(prompt, capture, max_tokens, check, recall)
            };
        #[cfg(not(feature = "hip"))]
        let result = self.generate_completion_inner(prompt, capture, max_tokens, check, recall);
        if result.as_ref().map_or(true, |reply| {
            reply.finish_reason == CandleFinishReason::Length
        }) {
            self.discard_completion_state();
        }
        result
    }

    /// Map a prompt byte range onto the exact tokens that cover it.
    ///
    /// `None` when the range is empty, not on character boundaries, or when a
    /// token straddles either boundary; capture then reports an explicit miss.
    #[cfg(feature = "candle-backend")]
    fn token_span_for_bytes(
        &self,
        prompt: &str,
        byte_span: &std::ops::Range<usize>,
    ) -> NluResult<Option<std::ops::Range<usize>>> {
        if self.family != ModelFamily::Qwen35
            || byte_span.start >= byte_span.end
            || byte_span.end > prompt.len()
            || !prompt.is_char_boundary(byte_span.start)
            || !prompt.is_char_boundary(byte_span.end)
        {
            return Ok(None);
        }
        let encoding =
            self.tokenizer
                .encode(prompt, true)
                .map_err(|e| NluError::LlmGenerationFailed {
                    reason: format!("tokenization: {e}"),
                })?;
        let mut first = None;
        let mut last = None;
        for (index, (start, end)) in encoding.get_offsets().iter().enumerate() {
            if end <= start {
                continue;
            }
            if *start >= byte_span.start && *end <= byte_span.end {
                if first.is_none() {
                    if *start != byte_span.start {
                        return Ok(None);
                    }
                    first = Some(index);
                }
                last = Some((index, *end));
            } else if *start < byte_span.end && *end > byte_span.start {
                return Ok(None);
            }
        }
        Ok(match (first, last) {
            (Some(first), Some((last, end))) if end == byte_span.end => Some(first..last + 1),
            _ => None,
        })
    }

    #[cfg(feature = "candle-backend")]
    fn generate_completion_inner(
        &mut self,
        prompt: &str,
        capture: Option<CaptureRequest<'_>>,
        max_tokens: u32,
        check: &mut dyn FnMut() -> NluResult<()>,
        recall: &mut dyn FnMut(Option<&PromptSpanTaps>) -> NluResult<RecallDecision>,
    ) -> NluResult<CandleCompletion> {
        let started = std::time::Instant::now();
        check()?;
        let fail = |error: candle_core::Error| NluError::LlmGenerationFailed {
            reason: format!("session generation: {error}"),
        };
        // HIP device selection is thread-local. A resident model can be called
        // from different serialized spawn_blocking workers across session jobs.
        #[cfg(feature = "hip")]
        if let Device::Cuda(device) = &self.device {
            Device::new_cuda(device.ordinal()).map_err(fail)?;
        }
        if max_tokens == 0 || !self.config.temperature.is_finite() || self.config.temperature < 0.0
        {
            return Err(NluError::LlmGenerationFailed {
                reason: "generation needs positive token budget and finite nonnegative temperature"
                    .into(),
            });
        }
        let limit = self.completion_context_limit.unwrap_or(self.context_length);
        let mut prompt_tokens = self.encode_prompt(prompt)?;
        validate_completion_budget(prompt_tokens.len(), max_tokens, limit)?;
        let token_span = match &capture {
            Some(request) => self.token_span_for_bytes(prompt, &request.byte_span)?,
            None => None,
        };
        check()?;
        let mut prompt_span_taps = None;
        let (mut logits, reused_prompt_tokens) = if let LoadedModel::Qwen35(model) = &mut self.model
        {
            // Bound the attention score allocation and make long prefill
            // cancellation cooperative between chunks, not only after prefill.
            let checkpoint = model.checkpoint();
            let (logits, reused) = match (token_span, capture.as_ref()) {
                (Some(span), Some(request)) => {
                    let layers = request.layers.to_vec();
                    let captured = with_interrupt(check, |guard| {
                        model.resume_prompt_with_taps(&prompt_tokens, span.clone(), &layers, guard)
                    })?;
                    if let Some(taps) = captured.taps {
                        let mut means = Vec::with_capacity(taps.len());
                        for tap in taps {
                            means.push(LayerMean {
                                layer: tap.layer,
                                mean: tap
                                    .mean
                                    .to_dtype(DType::F32)
                                    .and_then(|mean| mean.to_vec1::<f32>())
                                    .map_err(fail)?,
                            });
                        }
                        prompt_span_taps = Some(PromptSpanTaps {
                            layers: means,
                            token_span: span,
                        });
                    }
                    (captured.logits, captured.reused_tokens)
                }
                _ => with_interrupt(check, |guard| model.resume_prompt(&prompt_tokens, guard))?,
            };
            check()?;
            match recall(prompt_span_taps.as_ref())? {
                RecallDecision::Continue => (logits, reused),
                RecallDecision::Reprompt(new_prompt) => {
                    model.rewind(checkpoint).map_err(fail)?;
                    let new_tokens = encode_prompt_tokens(&self.tokenizer, self.family, &new_prompt)?;
                    validate_completion_budget(new_tokens.len(), max_tokens, limit)?;
                    check()?;
                    let forwarded =
                        with_interrupt(check, |guard| model.resume_prompt(&new_tokens, guard))?;
                    prompt_tokens = new_tokens;
                    forwarded
                }
            }
        } else {
            recall(None)?;
            let input = Tensor::new(prompt_tokens.as_slice(), &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(fail)?;
            (self.model.forward(&input, 0).map_err(fail)?, 0)
        };
        let eos = self.find_eos_token();
        let mut generated = Vec::new();
        let mut finish_reason = CandleFinishReason::Length;
        let mut first_sample_at = None;
        let mut last_sample_at = None;
        let mut step_seconds: Vec<f64> = Vec::new();
        let mut decode_steps = 0;
        for index in 0..max_tokens {
            check()?;
            let row = match logits.rank() {
                3 => logits.i((0, logits.dim(1).map_err(fail)? - 1, ..)),
                _ => logits.i((0, ..)),
            }
            .map_err(fail)?;
            let token = if self.config.temperature == 0.0 {
                row.argmax(0)
                    .and_then(|t| t.to_scalar::<u32>())
                    .map_err(fail)?
            } else {
                let scaled = (&row / self.config.temperature as f64).map_err(fail)?;
                let probs = candle_nn::ops::softmax(&scaled, 0).map_err(fail)?;
                sample_from_probs(&probs.to_vec1::<f32>().map_err(fail)?)
            };
            // Sampling reads the result back and therefore includes queued GPU
            // work. Timing forward's return alone would measure enqueue time.
            let sampled_now = std::time::Instant::now();
            if first_sample_at.is_none() {
                first_sample_at = Some(sampled_now);
            } else {
                decode_steps += 1;
                if let Some(previous) = last_sample_at {
                    step_seconds.push(sampled_now.duration_since(previous).as_secs_f64());
                }
            }
            last_sample_at = Some(sampled_now);
            if eos.contains(&token) {
                finish_reason = CandleFinishReason::Stop;
                break;
            }
            generated.push(token);
            // No redundant forward after consuming the final output token.
            if index + 1 < max_tokens {
                check()?;
                logits = if let LoadedModel::Qwen35(model) = &mut self.model {
                    let mut interrupted = None;
                    let result = model.forward_token(token, &mut || {
                        check().map_err(|error| {
                            interrupted = Some(error);
                            candle_core::Error::Msg("session decode interrupted".into())
                        })
                    });
                    if let Some(error) = interrupted {
                        return Err(error);
                    }
                    result.map_err(fail)?
                } else {
                    let input = Tensor::new(&[token], &self.device)
                        .and_then(|t| t.unsqueeze(0))
                        .map_err(fail)?;
                    self.model
                        .forward(&input, prompt_tokens.len() + index as usize)
                        .map_err(fail)?
                };
            }
        }
        let sampled_at = std::time::Instant::now();
        check()?;
        // Decode the sequence together: byte-level tokens may split a UTF-8 code
        // point. Decoding individual tokens inserts replacement characters.
        let text = self.tokenizer.decode(&generated, false).map_err(|e| {
            NluError::LlmGenerationFailed {
                reason: format!("detokenization: {e}"),
            }
        })?;
        Ok(CandleCompletion {
            text,
            prompt_tokens: prompt_tokens.len(),
            reused_prompt_tokens,
            tokens_generated: generated.len() as u32,
            finish_reason,
            time_to_first_token: first_sample_at.unwrap().duration_since(started),
            decode_time: sampled_at.duration_since(first_sample_at.unwrap()),
            step_seconds,
            decode_steps,
            prompt_span_taps,
        })
    }

    #[cfg(feature = "candle-backend")]
    /// Owned, exact counter using the completion path's family-specific encoding.
    /// Safe to retain across a mutable generation call; encoding errors propagate.
    pub fn completion_prompt_counter(&self) -> impl Fn(&str) -> NluResult<usize> + use<> {
        let tokenizer = self.tokenizer.clone();
        let family = self.family;
        move |prompt| encode_prompt_tokens(&tokenizer, family, prompt).map(|tokens| tokens.len())
    }

    #[cfg(feature = "candle-backend")]
    fn encode_prompt(&self, prompt: &str) -> NluResult<Vec<u32>> {
        encode_prompt_tokens(&self.tokenizer, self.family, prompt)
    }

    #[cfg(feature = "candle-backend")]
    fn generate_inner(
        &mut self,
        prompt: &str,
        max_tokens: u32,
        stop: StopMode,
    ) -> NluResult<GenerationResult> {
        let prompt_tokens = self.encode_prompt(prompt)?;

        tracing::debug!(
            prompt_tokens = prompt_tokens.len(),
            "Candle prompt tokenized"
        );

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

        let logits = self
            .model
            .forward(&input, 0)
            .map_err(|e| NluError::LlmGenerationFailed {
                reason: format!("prefill: {e}"),
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
    /// matching plain greedy decode. `eos` is the **full set** of stop-token ids for the loaded
    /// family (see [`find_eos_token`](Self::find_eos_token)) — checking membership rather than a
    /// single id matters because a family can have more than one live stop marker (e.g. Qwen2.5's
    /// chat template ends turns with `<|im_end|>`, but its vocab also always defines
    /// `<|endoftext|>`; picking only one by lookup precedence let the other slip through the
    /// pre-decode check and leak into the output — see `find_eos_token`'s doc comment for the
    /// full story). The `StopMode::Text` string match below is a second, defense-in-depth net
    /// for any marker that still isn't caught by id (e.g. emitted as part of a multi-token
    /// piece); it truncates the marker back out of `generated_text` rather than merely
    /// detecting it, so a miss there still
    /// can't leak text to the caller. Shared by [`decode_loop`](Self::decode_loop) and
    /// [`spec_decode_loop`](Self::spec_decode_loop) so their detokenization and stop semantics
    /// are byte-identical — the speculative loop just calls this once per accepted/bonus token.
    #[cfg(feature = "candle-backend")]
    fn process_token(&self, token: u32, eos: &[u32], stop: StopMode, st: &mut DecodeState) -> bool {
        if !matches!(stop, StopMode::Fixed) && eos.contains(&token) {
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
                // Defense-in-depth: the id-based check above is primary. This string match
                // catches a stop marker that reached `generated_text` some other way (e.g.
                // decoded as part of a multi-token piece) — and *removes* it before stopping,
                // so the marker never survives into the returned text either way.
                for marker in ["<|im_end|>", "<end_of_turn>"] {
                    if let Some(idx) = st.generated_text.find(marker) {
                        st.generated_text.truncate(idx);
                        return true;
                    }
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
                let scaled = (&last_logits / self.config.temperature as f64).map_err(|e| {
                    NluError::LlmGenerationFailed {
                        reason: format!("scale: {e}"),
                    }
                })?;
                let probs = candle_nn::ops::softmax(&scaled, 0).map_err(|e| {
                    NluError::LlmGenerationFailed {
                        reason: format!("softmax: {e}"),
                    }
                })?;
                let probs_vec: Vec<f32> =
                    probs.to_vec1().map_err(|e| NluError::LlmGenerationFailed {
                        reason: format!("to_vec1: {e}"),
                    })?;
                sample_from_probs(&probs_vec)
            };

            // Emit the token + apply the StopMode rules (EOS / brace / newline / markers).
            if self.process_token(next_token, &eos_token, stop, &mut st) {
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
        use super::spec_decode::{DrafterConfig, NgramDrafter, accept_run};
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
        let mut stop_now = self.process_token(cur0, &eos, stop, &mut st);

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
                if self.process_token(tok, &eos, stop, &mut st) {
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
        use super::spec_decode::{DrafterConfig, NgramDrafter, categorical, spec_sample_step};
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
        let mut stop_now = self.process_token(cur0, &eos, stop, &mut st);

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
                let (_, _seq, vocab) = verify
                    .dims3()
                    .map_err(|e| fail(format!("verify dims: {e}")))?;
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
            let bonus = bonus.unwrap_or_else(|| categorical(row(drafts.len()), rng.r#gen::<f32>()));

            // Emit the k accepted drafts + the bonus, each through the shared per-token path.
            for i in 0..k {
                all.push(drafts[i]);
                if self.process_token(drafts[i], &eos, stop, &mut st) {
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
                if self.process_token(bonus, &eos, stop, &mut st) || st.n_generated >= max_tokens {
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
        let encoding =
            self.tokenizer
                .encode(prompt, true)
                .map_err(|e| NluError::LlmGenerationFailed {
                    reason: format!("tokenization: {e}"),
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
        let logits = self
            .model
            .forward(&input, 0)
            .map_err(|e| NluError::LlmGenerationFailed {
                reason: format!("prefill: {e}"),
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
    pub fn generate_constrained_json(
        &mut self,
        prompt: &str,
        max_tokens: u32,
    ) -> NluResult<String> {
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
                if eos.contains(&id) {
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
                        "HIP device construction failed; trying remaining backends. Verify the ROCm \
                         runtime (ROCM_PATH, default /opt/rocm) and that the GPU is visible \
                         (`rocminfo`)."
                    );
                }
            }
        }
        #[cfg(feature = "vulkan")]
        match Device::new_vulkan(0) {
            Ok(device) => {
                tracing::info!("using Vulkan device for Candle inference");
                return device;
            }
            Err(error) => tracing::warn!(%error, "Vulkan device construction failed"),
        }
        // No GPU backend selected: probe the host so the log
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

    /// Find the **full set** of EOS token IDs for the loaded model (family-aware).
    ///
    /// A family can have more than one live stop marker at once — e.g. Qwen2.5's
    /// vocab always defines `<|endoftext|>` *and* its ChatML template ends every
    /// assistant turn with `<|im_end|>`. An earlier version of this used
    /// `Option<u32>` with `or_else` fallback, so it returned only the
    /// first-resolving id in lookup-precedence order (`<|endoftext|>`, since it's
    /// always present) and never even looked up `<|im_end|>`. The chat model then
    /// emits `<|im_end|>` to end its turn, the pre-decode id check in
    /// [`process_token`](Self::process_token) compared it against the wrong single
    /// id and missed, and the token got decoded and appended to the output before
    /// the string-level `StopMode::Text` fallback caught it one token too late —
    /// the literal `<|im_end|>` leaking into the visible reply. Returning every id
    /// that resolves, not just the first, closes that gap.
    #[cfg(feature = "candle-backend")]
    fn find_eos_token(&self) -> Vec<u32> {
        eos_ids_for_family(self.family, |s| self.tokenizer.token_to_id(s))
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

/// Tokenize a prompt, inserting the Gemma BOS marker when the family needs it.
#[cfg(feature = "candle-backend")]
fn encode_prompt_tokens(
    tokenizer: &tokenizers::Tokenizer,
    family: ModelFamily,
    prompt: &str,
) -> NluResult<Vec<u32>> {
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| NluError::LlmGenerationFailed {
            reason: format!("tokenization: {e}"),
        })?;
    let mut tokens = encoding.get_ids().to_vec();
    if matches!(family, ModelFamily::Gemma | ModelFamily::Gemma4) {
        if let Some(bos) = tokenizer.token_to_id("<bos>") {
            if tokens.first() != Some(&bos) {
                tokens.insert(0, bos);
            }
        }
    }
    Ok(tokens)
}

/// Run a model operation whose cooperative check may raise a caller error.
/// The caller's cancellation/deadline error is returned verbatim instead of
/// the model's wrapped interruption message.
#[cfg(feature = "candle-backend")]
fn with_interrupt<T>(
    check: &mut dyn FnMut() -> NluResult<()>,
    run: impl FnOnce(&mut dyn FnMut() -> candle_core::Result<()>) -> candle_core::Result<T>,
) -> NluResult<T> {
    let mut interrupted = None;
    let result = run(&mut || {
        check().map_err(|error| {
            interrupted = Some(error);
            candle_core::Error::Msg("session generation interrupted".into())
        })
    });
    if let Some(error) = interrupted {
        return Err(error);
    }
    result.map_err(|error| NluError::LlmGenerationFailed {
        reason: format!("session generation: {error}"),
    })
}

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

/// Resolve the family-aware EOS/stop-marker id set: every marker string the family can end a
/// turn or completion with, mapped through `token_to_id` and kept when it resolves. Split out
/// of [`CandleBackend::find_eos_token`] as a pure function (vocab lookup injected via `token_to_id`)
/// so the marker→id resolution — the actual root cause of the EOS-leak bug (see
/// [`find_eos_token`](CandleBackend::find_eos_token)) — is unit-testable against a synthetic
/// vocab without a real tokenizer/model.
#[cfg_attr(not(feature = "candle-backend"), allow(dead_code))]
fn eos_ids_for_family(family: ModelFamily, token_to_id: impl Fn(&str) -> Option<u32>) -> Vec<u32> {
    // Qwen's <|endoftext|> base EOS and <|im_end|> ChatML turn end are both
    // live stop markers, not alternatives, so resolve every available marker.
    let markers = family.eos_markers();
    let mut ids: Vec<u32> = markers.iter().filter_map(|m| token_to_id(m)).collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

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

#[cfg(feature = "candle-backend")]
fn validate_completion_budget(prompt: usize, output: u32, context: usize) -> NluResult<()> {
    if prompt == 0
        || output == 0
        || prompt
            .checked_add(output as usize)
            .is_none_or(|n| n > context)
    {
        return Err(NluError::LlmGenerationFailed {
            reason: format!(
                "prompt ({prompt}) plus output budget ({output}) must fit model context ({context}); history is never silently truncated"
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Keyed capture/recall on the tiny CPU Qwen3.5 fixture: zero mixers make
    /// every post-block residual equal the token embedding (id 1 = ones), so
    /// pooled means are exactly known, and the argmax always lands on id 0,
    /// which the fixture tokenizer declares as `<|im_end|>`.
    #[cfg(feature = "candle-backend")]
    mod capture {
        use super::super::*;
        use crate::nlu::span_projection::CapturePolicy;

        /// Exact-model projection gate. Loads the pinned Ptah GGUF/tokenizer
        /// from `AKH_QWEN35_TEST_MODEL` / `AKH_QWEN35_TEST_TOKENIZER` on the
        /// device in `AKH_QWEN35_SESSION_DEVICE` (cpu|hip|vulkan), captures the
        /// user span of chat-rendered prompts through the keyed completion
        /// path, projects them with the production policy/projector, and
        /// checks that (1) the projection is deterministic across fresh
        /// sessions, (2) every paraphrase pair is closer than any unrelated
        /// pair, and (3) unrelated pairs are not near-duplicates. This is a
        /// same-pipeline neural-span similarity check, not grounded concept
        /// recall. Run serially with exclusive device access.
        #[test]
        #[ignore = "requires the exact pinned Ptah artifacts and exclusive device access"]
        fn installed_artifact_span_projection_ranks_paraphrases_above_unrelated_text() {
            use crate::nlu::span_projection::{Projector, hamming};
            let model = std::path::PathBuf::from(
                std::env::var_os("AKH_QWEN35_TEST_MODEL").expect("exact Ptah GGUF required"),
            );
            let tokenizer = std::path::PathBuf::from(
                std::env::var_os("AKH_QWEN35_TEST_TOKENIZER").expect("exact tokenizer required"),
            );
            let device = match std::env::var("AKH_QWEN35_SESSION_DEVICE")
                .unwrap_or_else(|_| "cpu".into())
                .as_str()
            {
                "hip" => CandleDevice::Hip { ordinal: 0 },
                "vulkan" => CandleDevice::Vulkan { ordinal: 0 },
                "cpu" => CandleDevice::Cpu,
                other => panic!("unsupported AKH_QWEN35_SESSION_DEVICE {other}"),
            };
            let started = std::time::Instant::now();
            let mut backend =
                CandleBackend::load_on_device(&model, &tokenizer, CandleConfig::default(), device)
                    .unwrap();
            backend.set_context_limit(2048).unwrap();
            eprintln!("loaded exact model in {:.2?}", started.elapsed());
            let (blocks, hidden) = backend.qwen35_shape().unwrap();
            assert_eq!((blocks, hidden), (64, 5120));
            let policy = CapturePolicy::for_model(blocks, hidden).unwrap();
            assert_eq!(policy.layers, [5, 19, 33, 47, 61]);
            let projector = Projector::default();

            let groups: [[&str; 2]; 3] = [
                [
                    "Please remember that my cat is named Cedar and she only eats tuna.",
                    "Note for later: the name of my cat is Cedar, and tuna is the only thing she eats.",
                ],
                [
                    "The deployment pipeline fails whenever the Docker layer cache is cold.",
                    "A cold Docker layer cache makes the deploy pipeline fall over every time.",
                ],
                [
                    "What is the capital city of Portugal, and roughly how many people live there?",
                    "Tell me Portugal's capital and its approximate population.",
                ],
            ];
            let fixture_path = std::env::var_os("AKH_SPAN_GATE_FIXTURE");
            let fact_query = fixture_path.is_some();
            let groups: [[String; 2]; 3] = if let Some(path) = fixture_path {
                use std::io::Read;
                let mut bytes = Vec::new();
                std::fs::File::open(path).unwrap().take(65537).read_to_end(&mut bytes).unwrap();
                assert!(bytes.len() <= 65536, "fixture exceeds 64 KiB");
                let groups: [[String; 2]; 3] = serde_json::from_slice(&bytes).unwrap();
                assert!(groups.iter().flatten().all(|text| !text.trim().is_empty() && text.len() <= 8192));
                groups
            } else {
                groups.map(|pair| pair.map(str::to_owned))
            };
            let render = |text: &str| {
                format!(
                    "<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                )
            };
            let mut vectors: Vec<Vec<Vec<u8>>> = Vec::new();
            for (g, group) in groups.iter().enumerate() {
                let mut projected = Vec::new();
                for (i, text) in group.iter().enumerate() {
                    let prompt = render(text);
                    let start = prompt.find(text).unwrap();
                    let key = format!("gate-{g}-{i}");
                    let mut capture_one = |backend: &mut CandleBackend, key: &str| {
                        let reply = backend
                            .generate_completion_for_session_with_capture(
                                key,
                                &prompt,
                                Some(CaptureRequest {
                                    byte_span: start..start + text.len(),
                                    layers: &policy.layers,
                                }),
                                1,
                                &mut || Ok(()),
                                &mut |_| Ok(RecallDecision::Continue),
                            )
                            .unwrap();
                        let taps = reply.prompt_span_taps.expect("fresh user span captured");
                        assert_eq!(taps.layers.len(), 5);
                        for layer in &taps.layers {
                            assert_eq!(layer.mean.len(), 5120);
                            assert!(layer.mean.iter().all(|v| v.is_finite()));
                            assert!(layer.mean.iter().any(|v| *v != 0.0));
                        }
                        projector.project(&policy, &taps.layers).unwrap()
                    };
                    let first = capture_one(&mut backend, &key);
                    // A fresh session over the same prompt must project identically.
                    let again = capture_one(&mut backend, &format!("{key}-again"));
                    assert_eq!(first, again, "projection must be deterministic");
                    let ones: u32 = first.iter().map(|b| b.count_ones()).sum();
                    eprintln!("group {g} text {i}: {ones} set bits of 10000");
                    projected.push(first);
                }
                vectors.push(projected);
            }
            // Bounded synthetic fixture output only; never export raw means.
            // create_new prevents an accidental overwrite of earlier evidence.
            if let Some(path) = std::env::var_os("AKH_SPAN_GATE_OUTPUT") {
                let file = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)
                    .unwrap();
                serde_json::to_writer_pretty(file, &serde_json::json!({
                    "schema": "akh-span-paraphrase-gate/1",
                    "task": if fact_query { "fact_query" } else { "paraphrase" },
                    "backend": std::env::var("AKH_QWEN35_SESSION_DEVICE").unwrap_or_else(|_| "cpu".into()),
                    "model_path": model,
                    "tokenizer_path": tokenizer,
                    "policy_sha256": policy.sha256(),
                    "projector_sha256": projector.sha256(),
                    "groups": groups,
                    "vectors": vectors,
                })).unwrap();
            }
            let mut within = Vec::new();
            let mut between = Vec::new();
            for g in 0..groups.len() {
                let d = hamming(&vectors[g][0], &vectors[g][1]);
                eprintln!("paraphrase distance group {g}: {d}");
                within.push(d);
                for h in (g + 1)..groups.len() {
                    for a in 0..2 {
                        for b in 0..2 {
                            let d = hamming(&vectors[g][a], &vectors[h][b]);
                            eprintln!("unrelated distance {g}.{a} vs {h}.{b}: {d}");
                            between.push(d);
                        }
                    }
                }
            }
            let max_within = *within.iter().max().unwrap();
            let min_between = *between.iter().min().unwrap();
            eprintln!("max paraphrase distance {max_within}, min unrelated distance {min_between}");
            if fact_query {
                // Questions are queries, not evidence candidates. Keep all
                // three closely related facts in the candidate pool.
                let mut correct = 0;
                for g in 0..groups.len() {
                    let distances: Vec<_> = vectors.iter()
                        .map(|pair| hamming(&vectors[g][1], &pair[0])).collect();
                    let expected = distances[g];
                    let unique = distances.iter().enumerate()
                        .all(|(other, &d)| other == g || expected < d);
                    eprintln!("fact query {g}: distances {distances:?}, unique correct={unique}");
                    correct += usize::from(unique);
                }
                assert_eq!(correct, 3, "all held-out questions must retrieve their unique supporting fact");
            } else {
                assert!(
                    max_within < min_between,
                    "every paraphrase pair must be closer than every unrelated pair"
                );
                assert!(min_between > 500, "unrelated spans must not look like near-duplicates");
            }
        }

        use crate::nlu::vendored::quantized_qwen35::test_fixture::{
            TINY_TOKENIZER_JSON as TOKENIZER, tiny_gguf_with_shape,
        };

        fn fixture(dir: &std::path::Path) -> CandleBackend {
            let model = dir.join("tiny.gguf");
            std::fs::write(&model, tiny_gguf_with_shape(64, 8)).unwrap();
            let tokenizer = dir.join("tokenizer.json");
            std::fs::write(&tokenizer, TOKENIZER).unwrap();
            CandleBackend::load_on_device(
                &model,
                &tokenizer,
                CandleConfig::default(),
                CandleDevice::Cpu,
            )
            .unwrap()
        }

        fn span(prompt: &str, text: &str) -> std::ops::Range<usize> {
            let start = prompt.find(text).unwrap();
            start..start + text.len()
        }

        #[test]
        fn keyed_capture_pools_the_span_and_cached_spans_are_misses() {
            let dir = tempfile::tempdir().unwrap();
            let mut backend = fixture(dir.path());
            let (blocks, hidden) = backend.qwen35_shape().unwrap();
            assert_eq!((blocks, hidden), (8, 8));
            let policy = CapturePolicy::for_model(blocks, hidden).unwrap();
            let prompt = "user hello world assistant";
            let mut offered = None;
            let reply = backend
                .generate_completion_for_session_with_capture(
                    "key",
                    prompt,
                    Some(CaptureRequest {
                        byte_span: span(prompt, "hello world"),
                        layers: &policy.layers,
                    }),
                    4,
                    &mut || Ok(()),
                    &mut |taps| {
                        offered = taps.cloned();
                        Ok(RecallDecision::Continue)
                    },
                )
                .unwrap();
            assert_eq!(reply.finish_reason, CandleFinishReason::Stop);
            let taps = reply.prompt_span_taps.clone().expect("newly forwarded span");
            assert_eq!(offered.as_ref(), Some(&taps));
            assert_eq!(taps.token_span, 1..3);
            assert_eq!(taps.layers.len(), 5);
            assert_eq!(policy.layers, [0, 2, 3, 5, 7]);
            for (mean, layer) in taps.layers.iter().zip(policy.layers) {
                assert_eq!(mean.layer, layer);
                assert_eq!(mean.mean.len(), 8);
                assert!(mean.mean.iter().all(|v| (v - 1.0).abs() < 1e-5), "{mean:?}");
            }
            assert_eq!(reply.reused_prompt_tokens, 0);

            // The same span is now inside the reused prefix: explicit miss,
            // generation still reuses the four forwarded tokens.
            let longer = "user hello world assistant more";
            let mut offered = Some(taps);
            let reply = backend
                .generate_completion_for_session_with_capture(
                    "key",
                    longer,
                    Some(CaptureRequest {
                        byte_span: span(longer, "hello world"),
                        layers: &policy.layers,
                    }),
                    4,
                    &mut || Ok(()),
                    &mut |taps| {
                        offered = taps.cloned();
                        Ok(RecallDecision::Continue)
                    },
                )
                .unwrap();
            assert_eq!(reply.reused_prompt_tokens, 4);
            assert!(reply.prompt_span_taps.is_none());
            assert!(offered.is_none());
        }

        #[test]
        fn unaligned_or_foreign_spans_are_misses_not_errors() {
            let dir = tempfile::tempdir().unwrap();
            let mut backend = fixture(dir.path());
            let prompt = "user hello world assistant";
            for byte_span in [
                span(prompt, "hel"),
                span(prompt, "lo world"),
                0..0,
                0..prompt.len() + 1,
            ] {
                let reply = backend
                    .generate_completion_for_session_with_capture(
                        "key",
                        prompt,
                        Some(CaptureRequest {
                            byte_span,
                            layers: &[0, 1, 2, 3, 4],
                        }),
                        4,
                        &mut || Ok(()),
                        &mut |taps| {
                            assert!(taps.is_none());
                            Ok(RecallDecision::Continue)
                        },
                    )
                    .unwrap();
                assert!(reply.prompt_span_taps.is_none());
            }
            // Zero embeddings pool to a zero mean: captured, but the projector
            // will reject it as degenerate rather than the model failing.
            let prompt = "user zero zero assistant";
            let reply = backend
                .generate_completion_for_session_with_capture(
                    "other",
                    prompt,
                    Some(CaptureRequest {
                        byte_span: span(prompt, "zero zero"),
                        layers: &[0, 1, 2, 3, 4],
                    }),
                    4,
                    &mut || Ok(()),
                    &mut |_| Ok(RecallDecision::Continue),
                )
                .unwrap();
            let taps = reply.prompt_span_taps.unwrap();
            assert!(taps.layers.iter().all(|m| m.mean.iter().all(|v| *v == 0.0)));
        }

        #[test]
        fn reprompt_rewinds_to_the_pre_call_state_and_forwards_the_new_prompt() {
            let dir = tempfile::tempdir().unwrap();
            let mut backend = fixture(dir.path());
            // First turn establishes a prefix that the rewind must return to.
            let first = "user greet assistant";
            let reply = backend
                .generate_completion_for_session("key", first, 4, &mut || Ok(()))
                .unwrap();
            assert_eq!((reply.prompt_tokens, reply.reused_prompt_tokens), (3, 0));

            let second = "user greet assistant cedar assistant";
            let rewritten = "user greet assistant note cedar assistant";
            let reply = backend
                .generate_completion_for_session_with_capture(
                    "key",
                    second,
                    Some(CaptureRequest {
                        byte_span: span(second, "cedar"),
                        layers: &[0, 1, 2, 3, 4],
                    }),
                    4,
                    &mut || Ok(()),
                    &mut |taps| {
                        assert_eq!(taps.unwrap().token_span, 3..4);
                        Ok(RecallDecision::Reprompt(rewritten.to_string()))
                    },
                )
                .unwrap();
            // Rewound to the three-token first turn, then forwarded the rewrite.
            assert_eq!((reply.prompt_tokens, reply.reused_prompt_tokens), (6, 3));
            assert!(reply.prompt_span_taps.is_some(), "taps come from the original forward");

            // The resident state now matches the rewritten prompt, not the original.
            let reply = backend
                .generate_completion_for_session(
                    "key",
                    "user greet assistant note cedar assistant next",
                    4,
                    &mut || Ok(()),
                )
                .unwrap();
            assert_eq!(reply.reused_prompt_tokens, 6);
            let reply = backend
                .generate_completion_for_session(
                    "key",
                    "user greet assistant cedar assistant next",
                    4,
                    &mut || Ok(()),
                )
                .unwrap();
            // Recurrent state cannot be sliced: divergence after the shared
            // prefix forces a fresh prefill rather than a partial reuse.
            assert_eq!(reply.reused_prompt_tokens, 0, "divergence forces fresh prefill");
        }

        #[test]
        fn recall_failure_discards_the_conversation_cache() {
            let dir = tempfile::tempdir().unwrap();
            let mut backend = fixture(dir.path());
            let prompt = "user hello assistant";
            let error = backend
                .generate_completion_for_session_with_capture(
                    "key",
                    prompt,
                    Some(CaptureRequest {
                        byte_span: span(prompt, "hello"),
                        layers: &[0, 1, 2, 3, 4],
                    }),
                    4,
                    &mut || Ok(()),
                    &mut |_| {
                        Err(NluError::LlmGenerationFailed {
                            reason: "recall transport failed".into(),
                        })
                    },
                )
                .unwrap_err();
            assert!(error.to_string().contains("recall transport failed"));
            let reply = backend
                .generate_completion_for_session("key", prompt, 4, &mut || Ok(()))
                .unwrap();
            assert_eq!(reply.reused_prompt_tokens, 0);
            // An oversized rewrite is rejected before any forward and also discards.
            let reply = backend.generate_completion_for_session_with_capture(
                "key",
                "user hello assistant again",
                Some(CaptureRequest {
                    byte_span: span("user hello assistant again", "again"),
                    layers: &[0, 1, 2, 3, 4],
                }),
                4,
                &mut || Ok(()),
                &mut |_| Ok(RecallDecision::Reprompt("x ".repeat(70))),
            );
            assert!(reply.is_err());
            assert!(backend.active_completion_session.is_none());
        }
    }

    #[test]
    #[cfg(feature = "candle-backend")]
    fn completion_budget_reserves_output_and_rejects_empty_or_overflowing_context() {
        assert!(validate_completion_budget(6, 4, 10).is_ok());
        assert!(validate_completion_budget(7, 4, 10).is_err());
        assert!(validate_completion_budget(0, 4, 10).is_err());
        assert!(validate_completion_budget(6, 0, 10).is_err());
        assert!(validate_completion_budget(usize::MAX, 1, usize::MAX).is_err());
    }

    #[test]
    #[cfg(all(feature = "candle-backend", not(feature = "vulkan")))]
    fn explicit_unavailable_device_never_falls_back_to_cpu() {
        let error = CandleBackend::load_on_device(
            Path::new("/not/a/model.gguf"),
            Path::new("/not/a/tokenizer.json"),
            CandleConfig::default(),
            CandleDevice::Vulkan { ordinal: 0 },
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Vulkan backend is not available")
        );
    }

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

    /// Regression test for the EOS-leak bug: Qwen2.5's vocab always defines the base
    /// `<|endoftext|>` EOS *and* its ChatML template ends assistant turns with `<|im_end|>`.
    /// The old `Option<u32>`/`or_else` version of this lookup stopped at the first-resolving
    /// id (`<|endoftext|>`) and never checked `<|im_end|>` at all, so the actual chat-template
    /// stop token was invisible to the pre-decode id check and leaked into decoded output. A
    /// synthetic vocab that defines both must resolve to *both* ids.
    #[test]
    fn qwen2_eos_ids_include_both_endoftext_and_im_end() {
        let vocab = [("<|endoftext|>", 151643u32), ("<|im_end|>", 151645u32)];
        let lookup = |s: &str| vocab.iter().find(|(k, _)| *k == s).map(|(_, v)| *v);
        let ids = eos_ids_for_family(ModelFamily::Qwen2, lookup);
        assert!(
            ids.contains(&151643),
            "must still resolve <|endoftext|>: {ids:?}"
        );
        assert!(
            ids.contains(&151645),
            "must also resolve <|im_end|> — the actual ChatML turn-end token: {ids:?}"
        );
        assert_eq!(ids.len(), 2, "no duplicates, no phantom ids: {ids:?}");
    }

    /// A vocab that only defines the base EOS (no ChatML `<|im_end|>` — e.g. a raw-completion
    /// tokenizer) should resolve to just that one id, not error or panic.
    #[test]
    fn qwen2_eos_ids_falls_back_when_im_end_absent() {
        let vocab = [("<|endoftext|>", 151643u32)];
        let lookup = |s: &str| vocab.iter().find(|(k, _)| *k == s).map(|(_, v)| *v);
        let ids = eos_ids_for_family(ModelFamily::Qwen2, lookup);
        assert_eq!(ids, vec![151643]);
    }

    /// Gemma's marker set resolves independently of Qwen2's — sanity check that family
    /// dispatch doesn't cross-contaminate the candidate lists.
    #[test]
    fn gemma_eos_ids_resolve_family_specific_markers() {
        let vocab = [("<end_of_turn>", 107u32), ("<eos>", 1u32)];
        let lookup = |s: &str| vocab.iter().find(|(k, _)| *k == s).map(|(_, v)| *v);
        let ids = eos_ids_for_family(ModelFamily::Gemma, lookup);
        assert_eq!(ids, vec![1, 107]);
    }

    /// Qwen3.8 GGUFs use the qwen35 architecture family. They must be recognized
    /// explicitly: treating an unknown architecture as Qwen2 can make the loader
    /// interpret a hybrid gated-delta model with the wrong tensor schema.
    #[test]
    fn qwen35_architecture_tags_never_fall_through_to_qwen2() {
        for architecture in ["qwen35", "qwen3.5", "qwen3_5", "QWEN35"] {
            assert_eq!(
                ModelFamily::from_architecture(architecture),
                ModelFamily::Qwen35,
                "{architecture} must select the dedicated hybrid-model family"
            );
        }
    }

    #[test]
    fn qwen35_uses_both_chatml_stop_markers() {
        let vocab = [("<|endoftext|>", 151643u32), ("<|im_end|>", 151645u32)];
        let lookup = |s: &str| vocab.iter().find(|(k, _)| *k == s).map(|(_, v)| *v);
        let ids = eos_ids_for_family(ModelFamily::Qwen35, lookup);
        assert_eq!(ids, vec![151643, 151645]);
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


/// Per-layer PolarQuant round-trip quality for one resident KV cache.
#[cfg(feature = "candle-backend")]
#[derive(Debug, Clone, serde::Serialize)]
pub struct KvLayerQuality {
    pub layer: usize,
    pub positions: usize,
    pub key: KvVectorQuality,
    pub value: KvVectorQuality,
}

/// Aggregate error over every per-head, per-position vector in one tensor.
#[cfg(feature = "candle-backend")]
#[derive(Debug, Clone, serde::Serialize)]
pub struct KvVectorQuality {
    /// Vectors measured.
    pub vectors: usize,
    /// Worst cosine similarity between a vector and its round trip.
    pub min_cosine: f64,
    /// Mean cosine similarity.
    pub mean_cosine: f64,
    /// Worst relative L2 error, `||x - x'|| / ||x||`.
    pub max_relative_l2: f64,
    /// Mean relative L2 error.
    pub mean_relative_l2: f64,
}

/// Compress and decompress every `head_dim` vector of a `(batch, heads, pos, dim)`
/// KV tensor and summarize the error. A zero vector is skipped rather than
/// scored, since cosine is undefined there and it carries no information.
#[cfg(feature = "candle-backend")]
fn measure_kv_tensor(
    tensor: &Tensor,
    round_trip: &dyn Fn(&[f32]) -> Vec<f32>,
) -> NluResult<KvVectorQuality> {
    let shape = |e: candle_core::Error| NluError::ModelLoadFailed {
        reason: format!("kv state read: {e}"),
    };
    let (_, heads, positions, dim) = tensor.dims4().map_err(shape)?;
    let flat: Vec<f32> = tensor
        .to_dtype(DType::F32)
        .map_err(shape)?
        .flatten_all()
        .map_err(shape)?
        .to_vec1()
        .map_err(shape)?;
    let mut vectors = 0usize;
    let mut min_cosine = f64::INFINITY;
    let mut sum_cosine = 0.0;
    let mut max_rel = 0.0f64;
    let mut sum_rel = 0.0;
    for head in 0..heads {
        for position in 0..positions {
            let start = (head * positions + position) * dim;
            let original = &flat[start..start + dim];
            let norm: f64 = original.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
            if norm == 0.0 {
                continue;
            }
            let restored = round_trip(original);
            let mut dot = 0.0f64;
            let mut restored_norm = 0.0f64;
            let mut diff = 0.0f64;
            for (a, b) in original.iter().zip(&restored) {
                let (a, b) = (*a as f64, *b as f64);
                dot += a * b;
                restored_norm += b * b;
                diff += (a - b) * (a - b);
            }
            let cosine = if restored_norm > 0.0 {
                dot / (norm * restored_norm.sqrt())
            } else {
                0.0
            };
            let relative = diff.sqrt() / norm;
            vectors += 1;
            sum_cosine += cosine;
            sum_rel += relative;
            min_cosine = min_cosine.min(cosine);
            max_rel = max_rel.max(relative);
        }
    }
    if vectors == 0 {
        return Err(NluError::ModelLoadFailed {
            reason: "kv state held no non-zero vectors to measure".into(),
        });
    }
    Ok(KvVectorQuality {
        vectors,
        min_cosine,
        mean_cosine: sum_cosine / vectors as f64,
        max_relative_l2: max_rel,
        mean_relative_l2: sum_rel / vectors as f64,
    })
}
