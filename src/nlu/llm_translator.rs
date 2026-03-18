//! LLM Translator — dual boundary translator.
//!
//! Uses a small local LLM (Qwen2.5-1.5B-Instruct via `llama-cpp-2`) for:
//!
//! 1. **Input boundary** (NLU Tier 3): natural language → [`AbsTree`] JSON
//! 2. **Output boundary**: symbolic facts + persona context → natural prose
//!
//! The LLM is a boundary translator — not a reasoning engine. All state
//! management and reasoning stays in VSA/KG space.
//!
//! Graceful degradation: if the model file is absent, `try_load()` returns
//! `None` and both translation directions are skipped.

use std::path::{Path, PathBuf};

use crate::grammar::abs::AbsTree;

use super::error::{NluError, NluResult};

// ── Types ──────────────────────────────────────────────────────────────────

/// The result of an LLM translation (input boundary).
#[derive(Debug, Clone)]
pub struct LlmTranslation {
    /// The raw JSON string produced by the LLM.
    pub json: String,
    /// The deserialized abstract syntax tree.
    pub tree: AbsTree,
    /// Number of tokens generated.
    pub tokens_generated: u32,
}

/// Stop condition for the generation loop.
enum StopCondition {
    /// Stop when a top-level JSON object closes (brace depth tracking).
    JsonObject,
    /// Stop on EOS or double-newline (for free-form text generation).
    Text,
}

// ── GBNF grammar ───────────────────────────────────────────────────────────

/// The GBNF grammar constraining LLM output to valid AbsTree JSON.
/// Kept for reference and future use if llama.cpp fixes abort behavior.
pub const ABSTREE_GBNF: &str = include_str!("abstree.gbnf");

// ── LlmTranslator ─────────────────────────────────────────────────────────

/// Dual boundary translator: NL ↔ Symbols via local LLM.
pub struct LlmTranslator {
    /// The loaded LLM model.
    #[cfg(feature = "nlu-llm")]
    model: llama_cpp_2::model::LlamaModel,
    /// The llama.cpp backend handle (must outlive model usage).
    #[cfg(feature = "nlu-llm")]
    backend: llama_cpp_2::llama_backend::LlamaBackend,
    /// Maximum tokens to generate per translation.
    max_tokens: u32,
    /// Path to the model file (for diagnostics).
    _model_path: PathBuf,
}

impl std::fmt::Debug for LlmTranslator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmTranslator")
            .field("max_tokens", &self.max_tokens)
            .field("model_path", &self._model_path)
            .finish_non_exhaustive()
    }
}

/// Expected model file path under `data_dir`.
const LLM_SUBDIR: &str = "models/llm";
const LLM_MODEL_FILE: &str = "qwen2.5-1.5b-instruct-q4_k_m.gguf";

impl LlmTranslator {
    /// Load the LLM model from a GGUF file.
    ///
    /// Returns `NluError::ModelNotFound` if the file is missing,
    /// or `NluError::ModelLoadFailed` if loading fails.
    #[cfg(feature = "nlu-llm")]
    pub fn load(model_path: &Path) -> NluResult<Self> {
        use llama_cpp_2::llama_backend::LlamaBackend;
        use llama_cpp_2::model::LlamaModel;
        use llama_cpp_2::model::params::LlamaModelParams;

        if !model_path.exists() {
            return Err(NluError::ModelNotFound {
                path: model_path.to_path_buf(),
            });
        }

        let backend = LlamaBackend::init().map_err(|e| NluError::ModelLoadFailed {
            reason: format!("LlamaBackend init: {e}"),
        })?;

        let params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, model_path, &params).map_err(|e| {
            NluError::ModelLoadFailed {
                reason: format!("GGUF model load: {e}"),
            }
        })?;

        Ok(Self {
            model,
            backend,
            max_tokens: 512,
            _model_path: model_path.to_path_buf(),
        })
    }

    /// Non-feature-gated load that always fails.
    #[cfg(not(feature = "nlu-llm"))]
    pub fn load(model_path: &Path) -> NluResult<Self> {
        if !model_path.exists() {
            return Err(NluError::ModelNotFound {
                path: model_path.to_path_buf(),
            });
        }
        Err(NluError::ModelLoadFailed {
            reason: "nlu-llm feature not enabled".to_string(),
        })
    }

    /// Gracefully attempt to load the LLM from the standard data directory path.
    /// Returns `None` if the model file is missing or loading fails.
    pub fn try_load(data_dir: &Path) -> Option<Self> {
        let model_path = data_dir.join(LLM_SUBDIR).join(LLM_MODEL_FILE);
        Self::load(&model_path).ok()
    }

    // ── Input boundary: NL → AbsTree ─────────────────────────────────

    /// Translate natural language input into an AbsTree (input boundary).
    ///
    /// Uses greedy sampling with brace-depth tracking to extract a JSON object.
    #[cfg(feature = "nlu-llm")]
    pub fn translate(&self, input: &str) -> NluResult<LlmTranslation> {
        let prompt = build_prompt(input);
        let (raw, n_generated) = self.run_generation(&prompt, self.max_tokens, StopCondition::JsonObject)?;

        // Extract the JSON object (trim anything before first '{' or after last '}')
        let raw = raw.trim();
        let json = if let Some(start) = raw.find('{') {
            if let Some(end) = raw.rfind('}') {
                &raw[start..=end]
            } else {
                raw
            }
        } else {
            raw
        };

        tracing::debug!(json_len = json.len(), raw = %json, "attempting AbsTree JSON parse");

        let tree = parse_abstree_json(json)?;

        Ok(LlmTranslation {
            json: json.to_string(),
            tree,
            tokens_generated: n_generated,
        })
    }

    /// Non-feature-gated stub.
    #[cfg(not(feature = "nlu-llm"))]
    pub fn translate(&self, _input: &str) -> NluResult<LlmTranslation> {
        Err(NluError::LlmGenerationFailed {
            reason: "nlu-llm feature not enabled".to_string(),
        })
    }

    // ── Output boundary: Symbols → NL ────────────────────────────────

    /// Generate natural language from a prompt (output boundary).
    ///
    /// Stops on EOS or double-newline. Returns the generated text.
    #[cfg(feature = "nlu-llm")]
    pub fn generate(&self, prompt: &str, max_tokens: u32) -> NluResult<String> {
        let (raw, n_generated) = self.run_generation(prompt, max_tokens, StopCondition::Text)?;
        let text = Self::clean_generated_text(&raw);
        tracing::debug!(tokens = n_generated, len = text.len(), "LLM text generation complete");
        Ok(text)
    }

    /// Non-feature-gated stub.
    #[cfg(not(feature = "nlu-llm"))]
    pub fn generate(&self, _prompt: &str, _max_tokens: u32) -> NluResult<String> {
        Err(NluError::LlmGenerationFailed {
            reason: "nlu-llm feature not enabled".to_string(),
        })
    }

    // ── Shared generation core ───────────────────────────────────────

    /// Run the LLM generation loop with the given prompt and stop condition.
    ///
    /// Returns `(generated_text, tokens_generated)`.
    #[cfg(feature = "nlu-llm")]
    fn run_generation(
        &self,
        prompt: &str,
        max_tokens: u32,
        stop: StopCondition,
    ) -> NluResult<(String, u32)> {
        use llama_cpp_2::context::params::LlamaContextParams;
        use llama_cpp_2::llama_batch::LlamaBatch;
        use llama_cpp_2::model::AddBos;
        use llama_cpp_2::sampling::LlamaSampler;

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(2048))
            .with_n_batch(2048)
            .with_n_ubatch(512);
        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| NluError::LlmGenerationFailed {
                reason: format!("Context creation: {e}"),
            })?;

        let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);

        let tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| NluError::LlmGenerationFailed {
                reason: format!("Tokenization: {e}"),
            })?;

        tracing::debug!(prompt_tokens = tokens.len(), "LLM prompt tokenized");

        if tokens.len() > 1800 {
            return Err(NluError::LlmGenerationFailed {
                reason: format!(
                    "Prompt too long ({} tokens, max 1800).",
                    tokens.len()
                ),
            });
        }

        let mut batch = LlamaBatch::new(tokens.len().max(2048), 1);
        for (i, &token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch
                .add(token, i as i32, &[0], is_last)
                .map_err(|_| NluError::LlmGenerationFailed {
                    reason: "Batch add failed".to_string(),
                })?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| NluError::LlmGenerationFailed {
                reason: format!("Decode: {e}"),
            })?;

        // Generation loop with stop-condition dispatch
        let mut output_tokens = Vec::new();
        let mut n_generated = 0u32;
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        // State for JsonObject stop condition
        let mut brace_depth: i32 = 0;
        let mut saw_open_brace = false;

        // State for Text stop condition
        let mut consecutive_newlines = 0u32;
        let mut consecutive_hashes = 0u32;
        let mut generated_text = String::new();

        while n_generated < max_tokens {
            let token = sampler.sample(&ctx, (batch.n_tokens() - 1) as i32);

            if token == self.model.token_eos() {
                break;
            }

            sampler.accept(token);
            output_tokens.push(token);
            n_generated += 1;

            // Check stop condition
            if let Ok(piece) = self.model.token_to_piece(token, &mut decoder, false, None) {
                match stop {
                    StopCondition::JsonObject => {
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
                    StopCondition::Text => {
                        generated_text.push_str(&piece);
                        // Stop on <|im_end|> (Qwen chat template end-of-turn)
                        if generated_text.contains("<|im_end|>") {
                            break;
                        }
                        for ch in piece.chars() {
                            if ch == '\n' {
                                consecutive_newlines += 1;
                            } else {
                                consecutive_newlines = 0;
                            }
                            if ch == '#' {
                                consecutive_hashes += 1;
                            } else if !ch.is_whitespace() {
                                consecutive_hashes = 0;
                            }
                        }
                        // Stop on double-newline, hashtag spam, or repetition
                        if consecutive_newlines >= 2
                            || consecutive_hashes >= 2
                            || Self::detect_repetition(&generated_text)
                        {
                            break;
                        }
                    }
                }
            }

            batch.clear();
            batch
                .add(
                    token,
                    (tokens.len() + n_generated as usize - 1) as i32,
                    &[0],
                    true,
                )
                .map_err(|_| NluError::LlmGenerationFailed {
                    reason: "Batch add failed".to_string(),
                })?;

            ctx.decode(&mut batch)
                .map_err(|e| NluError::LlmGenerationFailed {
                    reason: format!("Decode: {e}"),
                })?;
        }

        tracing::debug!(tokens_generated = n_generated, "LLM generation complete");

        let mut detok_decoder = encoding_rs::UTF_8.new_decoder();
        let text: String = output_tokens
            .iter()
            .filter_map(|t| {
                self.model
                    .token_to_piece(*t, &mut detok_decoder, false, None)
                    .ok()
            })
            .collect();

        Ok((text, n_generated))
    }

    /// Access the max_tokens setting.
    pub fn max_tokens(&self) -> u32 {
        self.max_tokens
    }

    /// Clean up generated text: strip chat template tokens, hashtags, numbered prefixes, and trailing noise.
    fn clean_generated_text(raw: &str) -> String {
        // Strip Qwen chat template tokens
        let mut text = raw
            .replace("<|im_end|>", "")
            .replace("<|im_start|>", "")
            .trim()
            .to_string();

        // Strip everything from the first '#' onward (hashtag spam)
        if let Some(hash_pos) = text.find('#') {
            text.truncate(hash_pos);
            text = text.trim_end().to_string();
        }

        // Strip numbered list prefixes: "1. " "2. " etc.
        let lines: Vec<&str> = text.lines().collect();
        if lines.len() > 1 && lines.iter().all(|l| {
            let t = l.trim();
            t.is_empty() || (t.len() > 2 && t.as_bytes()[0].is_ascii_digit() && t.as_bytes()[1] == b'.')
        }) {
            // All lines are numbered — strip the numbering and join
            text = lines
                .iter()
                .filter_map(|l| {
                    let t = l.trim();
                    if t.len() > 3 && t.as_bytes()[0].is_ascii_digit() && t.as_bytes()[1] == b'.' {
                        Some(t[2..].trim())
                    } else if t.is_empty() {
                        None
                    } else {
                        Some(t)
                    }
                })
                .collect::<Vec<_>>()
                .join(" ");
        }

        // Trim trailing incomplete sentences (no period at end)
        if !text.is_empty() && !text.ends_with('.') && !text.ends_with('!') && !text.ends_with('?') {
            if let Some(last_period) = text.rfind(|c: char| c == '.' || c == '!' || c == '?') {
                text.truncate(last_period + 1);
            }
        }

        text.trim().to_string()
    }

    /// Detect degenerate repetition in generated text.
    ///
    /// Returns `true` if the last 40 chars appear earlier in the text,
    /// indicating the model is stuck in a loop.
    fn detect_repetition(text: &str) -> bool {
        if text.len() < 80 {
            return false;
        }
        let tail = &text[text.len() - 40..];
        text[..text.len() - 40].contains(tail)
    }
}

// ── Input boundary prompt ──────────────────────────────────────────────────

/// Build the system + few-shot prompt for NL → AbsTree translation.
pub fn build_prompt(input: &str) -> String {
    format!(
        r#"You are a semantic parser. Convert natural language to AbsTree JSON.
Output ONLY valid JSON matching the AbsTree schema. No explanation.

## Examples

Input: "Dogs are mammals"
Output: {{"Triple":{{"subject":{{"EntityRef":{{"label":"dogs","symbol_id":null}}}},"predicate":{{"RelationRef":{{"label":"are","symbol_id":null}}}},"object":{{"EntityRef":{{"label":"mammals","symbol_id":null}}}}}}}}

Input: "Dogs are not cats"
Output: {{"Negation":{{"inner":{{"Triple":{{"subject":{{"EntityRef":{{"label":"dogs","symbol_id":null}}}},"predicate":{{"RelationRef":{{"label":"are","symbol_id":null}}}},"object":{{"EntityRef":{{"label":"cats","symbol_id":null}}}}}}}}}}}}

Input: "All dogs are mammals"
Output: {{"Quantified":{{"quantifier":"Universal","scope":{{"Triple":{{"subject":{{"EntityRef":{{"label":"dogs","symbol_id":null}}}},"predicate":{{"RelationRef":{{"label":"are","symbol_id":null}}}},"object":{{"EntityRef":{{"label":"mammals","symbol_id":null}}}}}}}}}}}}

Input: "Dogs can swim"
Output: {{"Modal":{{"modality":"Can","inner":{{"Triple":{{"subject":{{"EntityRef":{{"label":"dogs","symbol_id":null}}}},"predicate":{{"RelationRef":{{"label":"can","symbol_id":null}}}},"object":{{"EntityRef":{{"label":"swim","symbol_id":null}}}}}}}}}}}}

Input: "If it rains then the ground is wet"
Output: {{"Conditional":{{"condition":{{"Triple":{{"subject":{{"EntityRef":{{"label":"it","symbol_id":null}}}},"predicate":{{"RelationRef":{{"label":"rains","symbol_id":null}}}},"object":{{"Freeform":""}}}}}},"consequent":{{"Triple":{{"subject":{{"EntityRef":{{"label":"the ground","symbol_id":null}}}},"predicate":{{"RelationRef":{{"label":"is","symbol_id":null}}}},"object":{{"EntityRef":{{"label":"wet","symbol_id":null}}}}}}}}}}}}

Input: "Dogs are mammals and cats are mammals"
Output: {{"Conjunction":{{"items":[{{"Triple":{{"subject":{{"EntityRef":{{"label":"dogs","symbol_id":null}}}},"predicate":{{"RelationRef":{{"label":"are","symbol_id":null}}}},"object":{{"EntityRef":{{"label":"mammals","symbol_id":null}}}}}}}},{{"Triple":{{"subject":{{"EntityRef":{{"label":"cats","symbol_id":null}}}},"predicate":{{"RelationRef":{{"label":"are","symbol_id":null}}}},"object":{{"EntityRef":{{"label":"mammals","symbol_id":null}}}}}}}}],"is_and":true}}}}

Input: "Dogs are bigger than cats"
Output: {{"Comparison":{{"entity_a":{{"EntityRef":{{"label":"dogs","symbol_id":null}}}},"entity_b":{{"EntityRef":{{"label":"cats","symbol_id":null}}}},"property":"size","ordering":"GreaterThan"}}}}

Input: "Yesterday it rained"
Output: {{"Temporal":{{"time_expr":{{"Named":"yesterday"}},"inner":{{"Triple":{{"subject":{{"EntityRef":{{"label":"it","symbol_id":null}}}},"predicate":{{"RelationRef":{{"label":"rained","symbol_id":null}}}},"object":{{"Freeform":""}}}}}}}}}}

Input: "The dog that chased the cat"
Output: {{"RelativeClause":{{"head":{{"EntityRef":{{"label":"the dog","symbol_id":null}}}},"clause":{{"Triple":{{"subject":{{"EntityRef":{{"label":"the dog","symbol_id":null}}}},"predicate":{{"RelationRef":{{"label":"chased","symbol_id":null}}}},"object":{{"EntityRef":{{"label":"the cat","symbol_id":null}}}}}}}}}}}}

Input: "No dogs are reptiles"
Output: {{"Quantified":{{"quantifier":"None","scope":{{"Triple":{{"subject":{{"EntityRef":{{"label":"dogs","symbol_id":null}}}},"predicate":{{"RelationRef":{{"label":"are","symbol_id":null}}}},"object":{{"EntityRef":{{"label":"reptiles","symbol_id":null}}}}}}}}}}}}

## Task

Input: "{input}"
Output: "#
    )
}

// ── Output boundary prompt — dialogue acts only ───────────────────────────
//
// The LLM is NOT used for knowledge responses (the narrative grammar handles
// that). Only short dialogue acts (greeting, farewell) use LLM generation.

/// Build a short prompt for dialogue acts (greeting, farewell, etc.).
///
/// Uses Qwen2.5-Instruct chat template format.
pub fn build_dialogue_prompt(
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

    format!(
        "<|im_start|>system\n\
         You are {persona_name}, a knowledge engine that is {traits_str}.\n\
         Respond with exactly one sentence. No hashtags. Always introduce yourself as {persona_name}.<|im_end|>\n\
         <|im_start|>user\n\
         Generate a {act} response.{context_line}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

// ── JSON parsing ───────────────────────────────────────────────────────────

/// Parse a JSON string into an AbsTree.
///
/// Validates that the JSON is both syntactically valid and deserializes
/// into a recognized AbsTree variant.
pub fn parse_abstree_json(json: &str) -> NluResult<AbsTree> {
    let trimmed = json.trim();
    if trimmed.is_empty() {
        return Err(NluError::LlmGenerationFailed {
            reason: "Empty JSON output".to_string(),
        });
    }

    serde_json::from_str::<AbsTree>(trimmed).map_err(|e| NluError::LlmGenerationFailed {
        reason: format!("Invalid AbsTree JSON: {e}"),
    })
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::abs::{CompareOrd, Modality, Quantifier, TemporalExpr};

    // ── GBNF grammar file ──────────────────────────────────────────────

    #[test]
    fn gbnf_loads_as_valid_utf8() {
        assert!(!ABSTREE_GBNF.is_empty());
        assert!(ABSTREE_GBNF.contains("root"));
        assert!(ABSTREE_GBNF.contains("abstree"));
        assert!(ABSTREE_GBNF.contains("triple"));
    }

    // ── Prompt construction ────────────────────────────────────────────

    #[test]
    fn build_prompt_contains_input() {
        let prompt = build_prompt("dogs are mammals");
        assert!(prompt.contains("dogs are mammals"));
        assert!(prompt.contains("semantic parser"));
        assert!(prompt.contains("AbsTree"));
    }

    #[test]
    fn build_prompt_contains_few_shot_examples() {
        let prompt = build_prompt("test");
        assert!(prompt.contains("Triple"));
        assert!(prompt.contains("Negation"));
        assert!(prompt.contains("Quantified"));
        assert!(prompt.contains("Conditional"));
        assert!(prompt.contains("Modal"));
    }

    #[test]
    fn dialogue_prompt_contains_act() {
        let prompt = build_dialogue_prompt("greeting", "Akh", &["warm".into()], None);
        assert!(prompt.contains("greeting"));
        assert!(prompt.contains("Akh"));
        assert!(prompt.contains("warm"));
    }

    // ── JSON → AbsTree parsing ─────────────────────────────────────────

    #[test]
    fn parse_entity_ref() {
        let json = r#"{"EntityRef":{"label":"dogs","symbol_id":null}}"#;
        let tree = parse_abstree_json(json).unwrap();
        assert_eq!(tree, AbsTree::entity("dogs"));
    }

    #[test]
    fn parse_relation_ref() {
        let json = r#"{"RelationRef":{"label":"are","symbol_id":null}}"#;
        let tree = parse_abstree_json(json).unwrap();
        assert_eq!(tree, AbsTree::relation("are"));
    }

    #[test]
    fn parse_freeform() {
        let json = r#"{"Freeform":"hello world"}"#;
        let tree = parse_abstree_json(json).unwrap();
        assert_eq!(tree, AbsTree::Freeform("hello world".to_string()));
    }

    #[test]
    fn parse_triple() {
        let json = r#"{"Triple":{"subject":{"EntityRef":{"label":"dogs","symbol_id":null}},"predicate":{"RelationRef":{"label":"are","symbol_id":null}},"object":{"EntityRef":{"label":"mammals","symbol_id":null}}}}"#;
        let tree = parse_abstree_json(json).unwrap();
        assert_eq!(
            tree,
            AbsTree::triple(
                AbsTree::entity("dogs"),
                AbsTree::relation("are"),
                AbsTree::entity("mammals"),
            )
        );
    }

    #[test]
    fn parse_negation() {
        let json = r#"{"Negation":{"inner":{"Triple":{"subject":{"EntityRef":{"label":"dogs","symbol_id":null}},"predicate":{"RelationRef":{"label":"are","symbol_id":null}},"object":{"EntityRef":{"label":"cats","symbol_id":null}}}}}}"#;
        let tree = parse_abstree_json(json).unwrap();
        assert_eq!(
            tree,
            AbsTree::Negation {
                inner: Box::new(AbsTree::triple(
                    AbsTree::entity("dogs"),
                    AbsTree::relation("are"),
                    AbsTree::entity("cats"),
                ))
            }
        );
    }

    #[test]
    fn parse_quantified_universal() {
        let json = r#"{"Quantified":{"quantifier":"Universal","scope":{"Triple":{"subject":{"EntityRef":{"label":"dogs","symbol_id":null}},"predicate":{"RelationRef":{"label":"are","symbol_id":null}},"object":{"EntityRef":{"label":"mammals","symbol_id":null}}}}}}"#;
        let tree = parse_abstree_json(json).unwrap();
        assert_eq!(
            tree,
            AbsTree::Quantified {
                quantifier: Quantifier::Universal,
                scope: Box::new(AbsTree::triple(
                    AbsTree::entity("dogs"),
                    AbsTree::relation("are"),
                    AbsTree::entity("mammals"),
                ))
            }
        );
    }

    #[test]
    fn parse_conditional() {
        let json = r#"{"Conditional":{"condition":{"EntityRef":{"label":"rain","symbol_id":null}},"consequent":{"EntityRef":{"label":"wet","symbol_id":null}}}}"#;
        let tree = parse_abstree_json(json).unwrap();
        assert_eq!(
            tree,
            AbsTree::Conditional {
                condition: Box::new(AbsTree::entity("rain")),
                consequent: Box::new(AbsTree::entity("wet")),
            }
        );
    }

    #[test]
    fn parse_temporal_named() {
        let json = r#"{"Temporal":{"time_expr":{"Named":"yesterday"},"inner":{"EntityRef":{"label":"rain","symbol_id":null}}}}"#;
        let tree = parse_abstree_json(json).unwrap();
        assert_eq!(
            tree,
            AbsTree::Temporal {
                time_expr: TemporalExpr::Named("yesterday".to_string()),
                inner: Box::new(AbsTree::entity("rain")),
            }
        );
    }

    #[test]
    fn parse_modal() {
        let json = r#"{"Modal":{"modality":"Can","inner":{"EntityRef":{"label":"swim","symbol_id":null}}}}"#;
        let tree = parse_abstree_json(json).unwrap();
        assert_eq!(
            tree,
            AbsTree::Modal {
                modality: Modality::Can,
                inner: Box::new(AbsTree::entity("swim")),
            }
        );
    }

    #[test]
    fn parse_conjunction() {
        let json = r#"{"Conjunction":{"items":[{"EntityRef":{"label":"a","symbol_id":null}},{"EntityRef":{"label":"b","symbol_id":null}}],"is_and":true}}"#;
        let tree = parse_abstree_json(json).unwrap();
        assert_eq!(
            tree,
            AbsTree::Conjunction {
                items: vec![AbsTree::entity("a"), AbsTree::entity("b")],
                is_and: true,
            }
        );
    }

    #[test]
    fn parse_comparison() {
        let json = r#"{"Comparison":{"entity_a":{"EntityRef":{"label":"dogs","symbol_id":null}},"entity_b":{"EntityRef":{"label":"cats","symbol_id":null}},"property":"size","ordering":"GreaterThan"}}"#;
        let tree = parse_abstree_json(json).unwrap();
        assert_eq!(
            tree,
            AbsTree::Comparison {
                entity_a: Box::new(AbsTree::entity("dogs")),
                entity_b: Box::new(AbsTree::entity("cats")),
                property: "size".to_string(),
                ordering: CompareOrd::GreaterThan,
            }
        );
    }

    // ── Serde roundtrip tests ──────────────────────────────────────────

    #[test]
    fn serde_roundtrip_negation() {
        let tree = AbsTree::Negation {
            inner: Box::new(AbsTree::entity("x")),
        };
        let json = serde_json::to_string(&tree).unwrap();
        let restored = parse_abstree_json(&json).unwrap();
        assert_eq!(tree, restored);
    }

    #[test]
    fn serde_roundtrip_quantified() {
        let tree = AbsTree::Quantified {
            quantifier: Quantifier::Existential,
            scope: Box::new(AbsTree::entity("x")),
        };
        let json = serde_json::to_string(&tree).unwrap();
        let restored = parse_abstree_json(&json).unwrap();
        assert_eq!(tree, restored);
    }

    #[test]
    fn serde_roundtrip_modal() {
        let tree = AbsTree::Modal {
            modality: Modality::Must,
            inner: Box::new(AbsTree::entity("x")),
        };
        let json = serde_json::to_string(&tree).unwrap();
        let restored = parse_abstree_json(&json).unwrap();
        assert_eq!(tree, restored);
    }

    #[test]
    fn serde_roundtrip_temporal() {
        let tree = AbsTree::Temporal {
            time_expr: TemporalExpr::Relative(-3),
            inner: Box::new(AbsTree::entity("event")),
        };
        let json = serde_json::to_string(&tree).unwrap();
        let restored = parse_abstree_json(&json).unwrap();
        assert_eq!(tree, restored);
    }

    #[test]
    fn serde_roundtrip_conditional() {
        let tree = AbsTree::Conditional {
            condition: Box::new(AbsTree::entity("rain")),
            consequent: Box::new(AbsTree::entity("umbrella")),
        };
        let json = serde_json::to_string(&tree).unwrap();
        let restored = parse_abstree_json(&json).unwrap();
        assert_eq!(tree, restored);
    }

    #[test]
    fn serde_roundtrip_conjunction() {
        let tree = AbsTree::Conjunction {
            items: vec![AbsTree::entity("a"), AbsTree::entity("b"), AbsTree::entity("c")],
            is_and: false,
        };
        let json = serde_json::to_string(&tree).unwrap();
        let restored = parse_abstree_json(&json).unwrap();
        assert_eq!(tree, restored);
    }

    #[test]
    fn serde_roundtrip_triple() {
        let tree = AbsTree::triple(
            AbsTree::entity("sun"),
            AbsTree::relation("is"),
            AbsTree::entity("star"),
        );
        let json = serde_json::to_string(&tree).unwrap();
        let restored = parse_abstree_json(&json).unwrap();
        assert_eq!(tree, restored);
    }

    // ── Error cases ────────────────────────────────────────────────────

    #[test]
    fn parse_rejects_malformed_json() {
        let err = parse_abstree_json("{not valid json").unwrap_err();
        assert!(matches!(err, NluError::LlmGenerationFailed { .. }));
    }

    #[test]
    fn parse_rejects_valid_json_wrong_shape() {
        let err = parse_abstree_json(r#"{"NotAVariant": 42}"#).unwrap_err();
        assert!(matches!(err, NluError::LlmGenerationFailed { .. }));
    }

    #[test]
    fn parse_rejects_empty_string() {
        let err = parse_abstree_json("").unwrap_err();
        assert!(matches!(err, NluError::LlmGenerationFailed { .. }));
    }

    #[test]
    fn parse_rejects_whitespace_only() {
        let err = parse_abstree_json("   ").unwrap_err();
        assert!(matches!(err, NluError::LlmGenerationFailed { .. }));
    }

    // ── Graceful degradation ───────────────────────────────────────────

    #[test]
    fn try_load_returns_none_for_missing_model() {
        let nonexistent = PathBuf::from("/tmp/akh-medu-nonexistent-llm-test");
        assert!(LlmTranslator::try_load(&nonexistent).is_none());
    }

    #[test]
    fn load_returns_model_not_found_error() {
        let nonexistent = PathBuf::from("/tmp/akh-medu-nonexistent-llm-model.gguf");
        let err = LlmTranslator::load(&nonexistent).unwrap_err();
        assert!(
            matches!(err, NluError::ModelNotFound { .. }),
            "Expected ModelNotFound, got: {err:?}"
        );
    }
}
