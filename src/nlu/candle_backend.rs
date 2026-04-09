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
use super::llm_translator::{build_prompt, parse_abstree_json, LlmTranslation};

#[cfg(feature = "candle-backend")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "candle-backend")]
use candle_transformers::models::quantized_qwen2::ModelWeights;

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

// ---------------------------------------------------------------------------
// Expected model paths
// ---------------------------------------------------------------------------

const LLM_SUBDIR: &str = "models/llm";
const CANDLE_MODEL_FILE: &str = "qwen2.5-1.5b-instruct-q4_k_m.gguf";
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
    model: ModelWeights,
    #[cfg(feature = "candle-backend")]
    tokenizer: tokenizers::Tokenizer,
    #[cfg(feature = "candle-backend")]
    device: Device,
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
        let model =
            ModelWeights::from_gguf(content, &mut file, &device).map_err(|e| {
                NluError::ModelLoadFailed {
                    reason: format!("load Qwen2 weights: {e}"),
                }
            })?;

        // Load tokenizer.
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).map_err(|e| {
            NluError::ModelLoadFailed {
                reason: format!("load tokenizer: {e}"),
            }
        })?;

        tracing::info!("Candle GGUF model loaded successfully");

        Ok(Self {
            model,
            tokenizer,
            device,
            config,
            model_path: model_path.to_path_buf(),
        })
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
        let model_path = data_dir.join(LLM_SUBDIR).join(CANDLE_MODEL_FILE);
        let tokenizer_path = data_dir.join(LLM_SUBDIR).join(TOKENIZER_FILE);
        Self::load(&model_path, &tokenizer_path).ok()
    }

    /// Gracefully attempt to load with config.
    #[cfg(feature = "candle-backend")]
    pub fn try_load_with_config(data_dir: &Path, config: CandleConfig) -> Option<Self> {
        let model_path = data_dir.join(LLM_SUBDIR).join(CANDLE_MODEL_FILE);
        let tokenizer_path = data_dir.join(LLM_SUBDIR).join(TOKENIZER_FILE);
        Self::load_with_config(&model_path, &tokenizer_path, config).ok()
    }

    // ── Input boundary: NL → AbsTree ─────────────────────────────────────

    /// Translate natural language to AbsTree (NLU Tier 3 replacement).
    #[cfg(feature = "candle-backend")]
    pub fn translate(&mut self, input: &str) -> NluResult<LlmTranslation> {
        let prompt = build_prompt(input);
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

    /// Non-feature-gated stub.
    #[cfg(not(feature = "candle-backend"))]
    pub fn generate(&mut self, _prompt: &str, _max_tokens: u32) -> NluResult<String> {
        Err(NluError::LlmGenerationFailed {
            reason: "candle-backend feature not enabled".to_string(),
        })
    }

    // ── Hidden state extraction ──────────────────────────────────────────

    /// Run inference and return the final hidden state (for neural→VSA bridge).
    ///
    /// This extracts the hidden state at the configured layer before the LM head,
    /// enabling concept identification via the neural bridge (Phase 26d).
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

        // Run forward pass to get logits (hidden states are internal to the model).
        // For now, we use the logits themselves as a proxy signal.
        // Phase 26d will add proper per-layer hidden state extraction
        // by modifying the Candle model's forward pass.
        let logits = self
            .model
            .forward(&input_tensor, 0)
            .map_err(|e| NluError::LlmGenerationFailed {
                reason: format!("forward: {e}"),
            })?;

        // Take the last token's logits as a representation.
        let seq_len = logits.dim(1).map_err(|e| NluError::LlmGenerationFailed {
            reason: format!("dim: {e}"),
        })?;
        let last_logits = logits
            .i((0, seq_len - 1, ..))
            .map_err(|e| NluError::LlmGenerationFailed {
                reason: format!("index: {e}"),
            })?;
        let hidden_vec: Vec<f32> = last_logits
            .to_vec1()
            .map_err(|e| NluError::LlmGenerationFailed {
                reason: format!("to_vec1: {e}"),
            })?;

        Ok(hidden_vec)
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
        let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

        tracing::debug!(prompt_tokens = prompt_tokens.len(), "Candle prompt tokenized");

        if prompt_tokens.len() > 1800 {
            return Err(NluError::LlmGenerationFailed {
                reason: format!(
                    "Prompt too long ({} tokens, max 1800).",
                    prompt_tokens.len()
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

        let mut logits = self.model.forward(&input, 0).map_err(|e| {
            NluError::LlmGenerationFailed {
                reason: format!("prefill: {e}"),
            }
        })?;

        let mut generated_text = String::new();
        let mut n_generated = 0u32;
        let mut brace_depth: i32 = 0;
        let mut saw_open_brace = false;
        let mut consecutive_newlines = 0u32;

        let eos_token = self.find_eos_token();

        while n_generated < max_tokens {
            // Sample next token (greedy: argmax of last position logits).
            let seq_dim = logits.dim(1).map_err(|e| NluError::LlmGenerationFailed {
                reason: format!("dim: {e}"),
            })?;
            let last_logits = logits
                .i((0, seq_dim - 1, ..))
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

            // Check EOS
            if Some(next_token) == eos_token {
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
                    if generated_text.contains("<|im_end|>") {
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
                .forward(&next_input, prompt_tokens.len() + n_generated as usize - 1)
                .map_err(|e| NluError::LlmGenerationFailed {
                    reason: format!("decode step: {e}"),
                })?;
        }

        tracing::debug!(tokens_generated = n_generated, "Candle generation complete");

        Ok(GenerationResult {
            text: generated_text,
            tokens_generated: n_generated,
        })
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
        tracing::info!("using CPU device for Candle inference");
        Device::Cpu
    }

    /// Find the EOS token ID for the loaded model.
    #[cfg(feature = "candle-backend")]
    fn find_eos_token(&self) -> Option<u32> {
        // Qwen2.5 EOS tokens: <|endoftext|> (151643) or <|im_end|> (151645)
        self.tokenizer
            .token_to_id("<|endoftext|>")
            .or_else(|| self.tokenizer.token_to_id("<|im_end|>"))
            .or_else(|| self.tokenizer.token_to_id("</s>"))
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
}

/// Internal generation result.
struct GenerationResult {
    text: String,
    tokens_generated: u32,
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
    let text = raw
        .replace("<|im_end|>", "")
        .replace("<|im_start|>", "")
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

/// Simple weighted random sampling from a probability distribution.
#[cfg(feature = "candle-backend")]
fn sample_from_probs(probs: &[f32]) -> u32 {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
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
