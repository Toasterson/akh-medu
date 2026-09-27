//! NLU pipeline error types with miette diagnostics.

use miette::Diagnostic;
use thiserror::Error;

/// Result type for NLU operations.
pub type NluResult<T> = Result<T, NluError>;

/// Errors from the NLU pipeline.
#[derive(Debug, Error, Diagnostic)]
pub enum NluError {
    /// No tier could parse the input into a structured representation.
    #[error("NLU parse failed: no tier could handle \"{input}\"")]
    #[diagnostic(
        code(akh::nlu::parse_failed),
        help(
            "The input could not be parsed by any NLU tier. It will be treated as freeform text."
        )
    )]
    ParseFailed { input: String },

    /// A requested NLU tier is not available (e.g., ML tier without the feature gate).
    #[error("NLU tier {tier} is not available")]
    #[diagnostic(
        code(akh::nlu::tier_unavailable),
        help(
            "Enable the corresponding feature gate: \
             tier 2 requires `nlu-ml`, tier 3 requires `nlu-llm`."
        )
    )]
    TierUnavailable { tier: u8 },

    /// The VSA parse ranker encountered an error.
    #[error("NLU ranker error: {reason}")]
    #[diagnostic(
        code(akh::nlu::ranker_error),
        help("The parse ranker's exemplar memory may be corrupted. Try resetting it.")
    )]
    RankerError { reason: String },

    /// A required model file was not found on disk.
    #[error("NLU model not found: {path}")]
    #[diagnostic(
        code(akh::nlu::model_not_found),
        help("Download with `akh setup models`")
    )]
    ModelNotFound { path: std::path::PathBuf },

    /// A model file exists but could not be loaded.
    #[error("NLU model load failed: {reason}")]
    #[diagnostic(code(akh::nlu::model_load), help("Check model file integrity"))]
    ModelLoadFailed { reason: String },

    /// A valid GGUF belongs to an architecture whose Candle loader has not landed.
    #[error("Candle does not yet support GGUF architecture {architecture:?}")]
    #[diagnostic(
        code(akh::nlu::candle_architecture_unsupported),
        help(
            "Keep this exact model on its Ollama/llama.cpp backing until the Qwen3.5/Qwen3.8 hybrid loader is implemented"
        )
    )]
    CandleArchitectureUnsupported { architecture: String },

    /// A request would need more resident KV cache than the device can hold.
    ///
    /// Distinct from the declared-window check: the GGUF advertises a context
    /// length the hardware cannot honour once the weights are resident, so a
    /// request can be well inside the model's window and still not fit.
    #[error(
        "context of {requested} tokens needs {} of KV cache, but only {} is free on the device",
        human_bytes(*needed_bytes),
        human_bytes(*available_bytes)
    )]
    #[diagnostic(
        code(akh::nlu::context_exceeds_device_memory),
        help(
            "This model declares a {declared_window}-token window, but on this device only \
             about {fits_tokens} tokens fit once the weights are resident: each token costs \
             about {} of device memory, of which {} is KV and the rest is prefill working \
             space. Either send at most {fits_tokens} tokens, \
             or free device memory, or turn on compressed KV paging with \
             AKH_QWEN35_KV_QUANT_BITS (note that compressed KV does not meet the 0.9999 \
             cosine gate at any width and has no agreed quality criterion yet). \
             AKH_CANDLE_KV_HEADROOM_BYTES adjusts the reserve held back for activations, \
             currently {}.",
            human_bytes(*per_token_bytes),
            human_bytes(*kv_bytes_per_token),
            human_bytes(*headroom_bytes)
        )
    )]
    ContextExceedsDeviceMemory {
        requested: usize,
        declared_window: usize,
        fits_tokens: usize,
        /// Total device cost per token, not the KV share alone: sizing on KV
        /// was wrong by more than 4x and admitted requests that then failed.
        per_token_bytes: usize,
        kv_bytes_per_token: usize,
        needed_bytes: usize,
        available_bytes: usize,
        headroom_bytes: usize,
    },

    /// The prompt does not carry exactly one well-formed image placeholder.
    ///
    /// Image tokens are placed by the caller's prompt, never inferred, so a
    /// prompt that is missing the placeholder, repeats it, or leaves it
    /// unbracketed is refused with the exact string to use.
    #[error(
        "prompt has {found} `<|image_pad|>` placeholder(s), bracketed: {bracketed}; \
         exactly one bracketed image is supported"
    )]
    #[diagnostic(
        code(akh::nlu::image_placeholder),
        help(
            "Put the image where it belongs in the prompt as the literal string \
             `<|vision_start|><|image_pad|><|vision_end|>`, exactly once. The single \
             `<|image_pad|>` is expanded to one token per merged patch - {grid_tokens} \
             tokens for this image - and the text before and after it is tokenized as \
             usual. Multiple images in one prompt are not supported yet."
        )
    )]
    ImagePlaceholder {
        found: usize,
        bracketed: bool,
        grid_tokens: usize,
    },

    /// The LLM could not produce valid structured output.
    #[error("LLM generation failed: {reason}")]
    #[diagnostic(
        code(akh::nlu::llm_generation),
        help("The LLM could not produce valid output")
    )]
    LlmGenerationFailed { reason: String },

    /// GBNF grammar initialization failed (malformed grammar file).
    #[error("GBNF grammar initialization failed: {reason}")]
    #[diagnostic(
        code(akh::nlu::grammar_init),
        help("The GBNF grammar file may be malformed. Check abstree.gbnf syntax.")
    )]
    GrammarInitFailed { reason: String },

    /// Wrapped grammar error.
    #[error(transparent)]
    #[diagnostic(transparent)]
    Grammar(#[from] crate::grammar::error::GrammarError),

    /// Wrapped VSA error.
    #[error(transparent)]
    #[diagnostic(transparent)]
    Vsa(#[from] crate::error::VsaError),
}

/// Render a byte count the way an operator reads one.
///
/// Binary units, because device memory is reported and sold in them and a gate
/// that says "24.0 GB" against a card labelled 24 GiB invites the wrong arithmetic.
fn human_bytes(bytes: usize) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod byte_rendering_tests {
    use super::NluError;
    use super::human_bytes;
    use miette::Diagnostic;

    #[test]
    fn the_image_placeholder_refusal_names_the_exact_string_to_use() {
        let error = NluError::ImagePlaceholder {
            found: 0,
            bracketed: false,
            grid_tokens: 256,
        };
        let help = error.help().expect("must carry the remedy").to_string();
        assert!(
            help.contains("<|vision_start|><|image_pad|><|vision_end|>"),
            "{help}"
        );
        assert!(help.contains("256 tokens"), "{help}");
        assert!(help.contains("not supported yet"), "{help}");
        assert!(error.to_string().contains("0 `<|image_pad|>`"), "{error}");
    }

    /// The diagnostic exists to tell an operator what to do, so assert on the
    /// text they will read rather than on the struct's fields.
    #[test]
    fn the_context_gate_tells_the_operator_what_to_do() {
        let error = NluError::ContextExceedsDeviceMemory {
            requested: 60_000,
            declared_window: 262_144,
            fits_tokens: 48_672,
            per_token_bytes: 589_824,
            kv_bytes_per_token: 131_072,
            needed_bytes: 60_000 * 589_824,
            available_bytes: 8_527_020_032,
            headroom_bytes: 2 * 1024 * 1024 * 1024,
        };

        let message = error.to_string();
        assert!(message.contains("60000 tokens"), "{message}");
        assert!(message.contains("32.96 GiB"), "needed memory: {message}");
        assert!(message.contains("7.94 GiB"), "free memory: {message}");

        let help = error
            .help()
            .expect("the gate must carry a remedy, not just a complaint")
            .to_string();
        // The limit that actually applies, not just the one that failed.
        assert!(help.contains("48672 tokens"), "{help}");
        assert!(help.contains("262144-token window"), "{help}");
        // Both figures, so the KV share is never mistaken for the whole cost.
        assert!(help.contains("576.00 KiB"), "total per-token cost: {help}");
        assert!(help.contains("128.00 KiB"), "KV share: {help}");
        assert!(help.contains("2.00 GiB"), "headroom: {help}");
        // Every remedy, including the caveat on the tempting one.
        assert!(help.contains("AKH_QWEN35_KV_QUANT_BITS"), "{help}");
        assert!(
            help.contains("0.9999"),
            "compression caveat missing: {help}"
        );
        assert!(help.contains("AKH_CANDLE_KV_HEADROOM_BYTES"), "{help}");
    }

    #[test]
    fn bytes_render_in_the_units_device_memory_is_reported_in() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(1024), "1.00 KiB");
        // The card in question, and the KV a full declared window would need.
        assert_eq!(human_bytes(25_753_026_560), "23.98 GiB");
        assert_eq!(human_bytes(34_359_738_368), "32.00 GiB");
        // Per-token KV for this model.
        assert_eq!(human_bytes(131_072), "128.00 KiB");
    }
}
