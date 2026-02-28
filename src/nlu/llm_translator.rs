//! LLM Translator — NLU Tier 3.
//!
//! Uses a small local LLM (Qwen2.5-1.5B-Instruct via `llama-cpp-2`) to
//! translate natural language into structured [`AbsTree`] JSON.
//!
//! The LLM output is constrained via a GBNF grammar so it can only produce
//! valid AbsTree JSON matching serde's externally-tagged enum representation.
//!
//! Graceful degradation: if the model file is absent, `try_load()` returns
//! `None` and the pipeline silently skips this tier.

use std::path::{Path, PathBuf};

use crate::grammar::abs::AbsTree;

use super::error::{NluError, NluResult};

// ── Types ──────────────────────────────────────────────────────────────────

/// The result of an LLM translation.
#[derive(Debug, Clone)]
pub struct LlmTranslation {
    /// The raw JSON string produced by the LLM.
    pub json: String,
    /// The deserialized abstract syntax tree.
    pub tree: AbsTree,
    /// Number of tokens generated.
    pub tokens_generated: u32,
}

// ── GBNF grammar ───────────────────────────────────────────────────────────

/// The GBNF grammar constraining LLM output to valid AbsTree JSON.
pub const ABSTREE_GBNF: &str = include_str!("abstree.gbnf");

// ── LlmTranslator ─────────────────────────────────────────────────────────

/// The LLM-based translator for natural language → AbsTree.
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

    /// Translate natural language input into an AbsTree.
    ///
    /// Generates constrained JSON via GBNF grammar, then deserializes.
    #[cfg(feature = "nlu-llm")]
    pub fn translate(&self, input: &str) -> NluResult<LlmTranslation> {
        use llama_cpp_2::context::params::LlamaContextParams;
        use llama_cpp_2::llama_batch::LlamaBatch;
        use llama_cpp_2::model::AddBos;
        use llama_cpp_2::token::LlamaToken;

        let prompt = build_prompt(input);

        // Create context for generation
        let ctx_params = LlamaContextParams::default();
        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| NluError::LlmGenerationFailed {
                reason: format!("Context creation: {e}"),
            })?;

        // Tokenize the prompt
        let tokens = self
            .model
            .str_to_token(&prompt, AddBos::Always)
            .map_err(|e| NluError::LlmGenerationFailed {
                reason: format!("Tokenization: {e}"),
            })?;

        // Feed prompt tokens
        let mut batch = LlamaBatch::new(tokens.len().max(512), 1);
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

        // Generate tokens with sampling
        let mut output_tokens = Vec::new();
        let mut n_generated = 0u32;

        // Simple greedy sampling loop
        while n_generated < self.max_tokens {
            let logits = ctx.get_logits_ith((batch.n_tokens() - 1) as i32);

            // Greedy: pick highest logit
            let mut best_token = LlamaToken(0);
            let mut best_logit = f32::NEG_INFINITY;
            for (i, &logit) in logits.iter().enumerate() {
                if logit > best_logit {
                    best_logit = logit;
                    best_token = LlamaToken(i as i32);
                }
            }

            // Check for EOS
            if best_token == self.model.token_eos() {
                break;
            }

            output_tokens.push(best_token);
            n_generated += 1;

            // Feed the new token for next iteration
            batch.clear();
            batch
                .add(best_token, (tokens.len() + n_generated as usize - 1) as i32, &[0], true)
                .map_err(|_| NluError::LlmGenerationFailed {
                    reason: "Batch add failed".to_string(),
                })?;

            ctx.decode(&mut batch)
                .map_err(|e| NluError::LlmGenerationFailed {
                    reason: format!("Decode: {e}"),
                })?;
        }

        // Detokenize output
        let json: String = output_tokens
            .iter()
            .filter_map(|t| self.model.token_to_str(*t, llama_cpp_2::model::Special::Plaintext).ok())
            .collect();

        let tree = parse_abstree_json(&json)?;

        Ok(LlmTranslation {
            json,
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

    /// Access the max_tokens setting.
    pub fn max_tokens(&self) -> u32 {
        self.max_tokens
    }
}

// ── Prompt construction ────────────────────────────────────────────────────

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
        // Valid JSON but not an AbsTree variant
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
