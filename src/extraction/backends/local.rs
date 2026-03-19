//! Local LLM backend for triple extraction (Qwen2.5-1.5B via llama-cpp-2).
//!
//! Shares the `LlmTranslator` model with the NLU pipeline — no second load.

use std::sync::Arc;

use crate::extraction::error::{ExtractionError, ExtractionResult};
use crate::extraction::prompt;
use crate::extraction::RawTriple;
use crate::nlu::llm_translator::LlmTranslator;

/// Backend using the shared local LLM.
pub struct LocalLlmBackend {
    translator: Arc<LlmTranslator>,
}

impl LocalLlmBackend {
    pub fn new(translator: Arc<LlmTranslator>) -> Self {
        Self { translator }
    }

    /// Extract triples from text using the local LLM.
    pub fn extract(&self, text: &str, max_triples: usize) -> ExtractionResult<Vec<RawTriple>> {
        let full_prompt = prompt::chatml_prompt(text, max_triples);

        let raw_output = self.translator.generate(&full_prompt, 512).map_err(|e| {
            ExtractionError::LlmUnavailable {
                reason: format!("local LLM generation failed: {e}"),
            }
        })?;

        Ok(prompt::parse_triples(&raw_output))
    }
}
