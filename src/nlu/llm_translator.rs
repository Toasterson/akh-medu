//! LLM Translator — dual boundary translator.
//!
//! Uses the native Candle backend for:
//!
//! 1. **Input boundary** (NLU Tier 3): natural language → [`AbsTree`] JSON
//! 2. **Output boundary**: symbolic facts + persona context → natural prose
//!
//! The LLM is a boundary translator — not a reasoning engine. All state
//! management and reasoning stays in VSA/KG space.
//!
//! Graceful degradation: if the model file is absent, `try_load()` returns
//! `None` and both translation directions are skipped.

use std::path::Path;
#[cfg(test)]
use std::path::PathBuf;

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

/// Reference grammar for AbsTree JSON.
pub const ABSTREE_GBNF: &str = include_str!("abstree.gbnf");

/// Compatibility API for local boundary translation and extraction.
///
/// The shared mutex serializes access to Candle's mutable decode state.
pub struct LlmTranslator {
    backend: std::sync::Arc<std::sync::Mutex<super::candle_backend::CandleBackend>>,
}

impl std::fmt::Debug for LlmTranslator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmTranslator").finish_non_exhaustive()
    }
}

impl LlmTranslator {
    /// Share an already loaded Candle model, including its decode state.
    pub fn from_shared_backend(
        backend: std::sync::Arc<std::sync::Mutex<super::candle_backend::CandleBackend>>,
    ) -> Self {
        Self { backend }
    }

    /// Load a GGUF model with its adjacent tokenizer.
    pub fn load(model_path: &Path) -> NluResult<Self> {
        let sidecar = model_path.with_extension("tokenizer.json");
        let tokenizer_path = if sidecar.exists() {
            sidecar
        } else {
            model_path.with_file_name("tokenizer.json")
        };
        let backend = super::candle_backend::CandleBackend::load(model_path, &tokenizer_path)?;
        Ok(Self::from_shared_backend(std::sync::Arc::new(
            std::sync::Mutex::new(backend),
        )))
    }

    /// Discover and load a native model from the standard data directory.
    pub fn try_load(data_dir: &Path) -> Option<Self> {
        super::candle_backend::CandleBackend::try_load(data_dir).map(|backend| {
            Self::from_shared_backend(std::sync::Arc::new(std::sync::Mutex::new(backend)))
        })
    }

    fn lock_backend(
        &self,
    ) -> NluResult<std::sync::MutexGuard<'_, super::candle_backend::CandleBackend>> {
        self.backend
            .lock()
            .map_err(|_| NluError::LlmGenerationFailed {
                reason: "Candle model lock poisoned".to_string(),
            })
    }

    pub fn translate(&self, input: &str) -> NluResult<LlmTranslation> {
        self.translate_with_context(input, None)
    }

    pub fn translate_with_context(
        &self,
        input: &str,
        kg_context: Option<&str>,
    ) -> NluResult<LlmTranslation> {
        self.lock_backend()?
            .translate_with_context(input, kg_context)
    }

    pub fn generate(&self, prompt: &str, max_tokens: u32) -> NluResult<String> {
        self.lock_backend()?.generate(prompt, max_tokens)
    }

    /// Access the max_tokens setting.
    pub fn max_tokens(&self) -> u32 {
        self.backend
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .config()
            .max_tokens
    }
}

// ── Input boundary prompt ──────────────────────────────────────────────────

/// Build the system + few-shot prompt for NL → AbsTree translation.
///
/// Delegates to [`build_prompt_with_context`] with no KG context.
pub fn build_prompt(input: &str) -> String {
    build_prompt_with_context(input, None)
}

/// Build the NLU prompt with an optional KG context hint.
///
/// When `kg_context` is provided it is inserted as a compact entity-hint
/// section before the task instruction. The hint is intentionally brief
/// (just the "## Knowledge Graph Context" block header + bullet lines) so
/// that the total prompt stays within the 1 800-token generation limit.
pub fn build_prompt_with_context(input: &str, kg_context: Option<&str>) -> String {
    let context_section = kg_context
        .filter(|c| !c.is_empty())
        .map(|c| format!("{c}\n\n"))
        .unwrap_or_default();

    format!(
        r#"{context_section}You are a semantic parser. Convert natural language to AbsTree JSON.
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
/// Delegates to [`build_dialogue_prompt_with_kg`] with no KG context.
pub fn build_dialogue_prompt(
    act: &str,
    persona_name: &str,
    traits: &[String],
    context: Option<&str>,
    kg_context: Option<&str>,
) -> String {
    build_dialogue_prompt_with_kg(act, persona_name, traits, context, kg_context)
}

/// Build a dialogue-act prompt with an optional KG context block.
///
/// `kg_context` is injected into the system turn so the model can reference
/// actual knowledge-graph facts when composing the response. The combined
/// system prompt + KG context is designed to stay under 3 800 tokens.
pub fn build_dialogue_prompt_with_kg(
    act: &str,
    persona_name: &str,
    traits: &[String],
    context: Option<&str>,
    kg_context: Option<&str>,
) -> String {
    let traits_str = if traits.is_empty() {
        "knowledgeable".to_string()
    } else {
        traits.join(", ")
    };

    let context_line = context
        .map(|c| format!(" We were discussing {c}."))
        .unwrap_or_default();

    let kg_section = kg_context
        .filter(|c| !c.is_empty())
        .map(|c| format!("\n\n{c}"))
        .unwrap_or_default();

    format!(
        "<|im_start|>system\n\
         You are {persona_name}, a knowledge engine that is {traits_str}.\
         {kg_section}\n\
         Respond with exactly one sentence. No hashtags. Always introduce yourself as {persona_name}.<|im_end|>\n\
         <|im_start|>user\n\
         Generate a {act} response.{context_line}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

/// Build a prompt for open-ended generation (creative writing, hypotheticals, etc.).
///
/// Injects KG facts into the system context when available to ground the response
/// in known facts and reduce hallucination.
pub fn build_open_ended_prompt(
    query: &str,
    persona_name: &str,
    traits: &[String],
    kg_context: Option<&str>,
) -> String {
    let traits_str = if traits.is_empty() {
        "knowledgeable and thoughtful".to_string()
    } else {
        traits.join(", ")
    };

    let context_block = match kg_context {
        Some(ctx) if !ctx.trim().is_empty() => format!(
            "\n\nHere are relevant facts from my knowledge graph:\n{ctx}\n\nUse these facts where relevant but feel free to be creative."
        ),
        _ => String::new(),
    };

    format!(
        "<|im_start|>system\n\
         You are {persona_name}, a knowledge engine that is {traits_str}.\
         {context_block}\n\
         Respond helpfully and concisely. No hashtags. No numbered lists unless asked.<|im_end|>\n\
         <|im_start|>user\n\
         {query}<|im_end|>\n\
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
        let prompt = build_dialogue_prompt("greeting", "Akh", &["warm".into()], None, None);
        assert!(prompt.contains("greeting"));
        assert!(prompt.contains("Akh"));
        assert!(prompt.contains("warm"));
    }

    #[test]
    fn dialogue_prompt_includes_kg_context() {
        let kg = "## Knowledge Graph Context\n- Ennead is-a collective\n";
        let prompt = build_dialogue_prompt("greeting", "Akh", &[], None, Some(kg));
        assert!(prompt.contains("Knowledge Graph Context"));
        assert!(prompt.contains("Ennead is-a collective"));
    }

    #[test]
    fn dialogue_prompt_no_kg_context_unchanged() {
        let without = build_dialogue_prompt("farewell", "Akh", &[], None, None);
        assert!(!without.contains("Knowledge Graph Context"));
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
            items: vec![
                AbsTree::entity("a"),
                AbsTree::entity("b"),
                AbsTree::entity("c"),
            ],
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
