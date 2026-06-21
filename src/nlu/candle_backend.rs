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
        })
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

        // Continue token-by-token from the prefilled prompt.
        self.decode_loop(logits, prompt_tokens.len(), max_tokens, stop)
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
        let mut generated_text = String::new();
        let mut n_generated = 0u32;
        let mut brace_depth: i32 = 0;
        let mut saw_open_brace = false;
        let mut consecutive_newlines = 0u32;

        let eos_token = self.find_eos_token();

        // Phase 45a: time the autoregressive decode (prefill excluded) for tok/s.
        let decode_start = std::time::Instant::now();
        while n_generated < max_tokens {
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

            // Check EOS (ignored in Fixed mode: a throughput probe decodes the
            // full token budget regardless of content).
            if !matches!(stop, StopMode::Fixed) && Some(next_token) == eos_token {
                break;
            }

            n_generated += 1;

            // Decode token to text
            let piece = self
                .tokenizer
                .decode(&[next_token], false)
                .unwrap_or_default();
            generated_text.push_str(&piece);

            // Check stop condition
            match stop {
                StopMode::Json => {
                    for ch in piece.chars() {
                        if ch == '{' {
                            brace_depth += 1;
                            saw_open_brace = true;
                        } else if ch == '}' {
                            brace_depth -= 1;
                        }
                    }
                    if saw_open_brace && brace_depth <= 0 {
                        break;
                    }
                }
                StopMode::Text => {
                    if generated_text.contains("<|im_end|>")
                        || generated_text.contains("<end_of_turn>")
                    {
                        break;
                    }
                    for ch in piece.chars() {
                        if ch == '\n' {
                            consecutive_newlines += 1;
                        } else {
                            consecutive_newlines = 0;
                        }
                    }
                    if consecutive_newlines >= 2 {
                        break;
                    }
                }
                StopMode::Fixed => {}
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
                .forward(&next_input, prompt_len + n_generated as usize - 1)
                .map_err(|e| NluError::LlmGenerationFailed {
                    reason: format!("decode step: {e}"),
                })?;
        }

        let decode_secs = decode_start.elapsed().as_secs_f32();
        let tok_s = if decode_secs > 0.0 {
            n_generated as f32 / decode_secs
        } else {
            0.0
        };
        tracing::debug!(
            tokens_generated = n_generated,
            decode_secs,
            tok_s,
            "Candle generation complete"
        );

        Ok(GenerationResult {
            text: generated_text,
            tokens_generated: n_generated,
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
        // AMD ROCm/HIP: Candle 0.10 has no upstream HIP device, so we cannot construct one
        // here yet (the backend is a fork gated behind `feature = "rocm"`, ADR 068). Probe
        // the host so the log tells the user *why* we are on CPU and what to do about it.
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
