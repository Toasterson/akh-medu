//! NLU pipeline: cascading natural language understanding.
//!
//! The pipeline applies a sequence of tiers to parse natural language input
//! into structured [`AbsTree`](crate::grammar::abs::AbsTree) representations:
//!
//! 1. **Rule parser** (always available) — extended `parse_prose()` with
//!    negation, quantification, conditionals, temporal, modal, comparative
//! 2. **Micro-ML NER** (feature-gated: `nlu-ml`) — ONNX-based multilingual
//!    named entity recognition with intent classification
//! 3. **Small LLM translator** (feature-gated: `nlu-llm`) — local LLM-based
//!    semantic parsing with GBNF-constrained AbsTree JSON output
//! 4. **VSA parse ranker** — ranks candidate parses using exemplar memory

pub mod error;
pub mod parse_ranker;

#[cfg(feature = "nlu-ml")]
pub mod micro_ml;

pub mod llm_translator;

use std::path::Path;

use crate::grammar::abs::AbsTree;
use crate::grammar::concrete::ParseContext;
use crate::grammar::parser::{self, ParseResult};

use self::error::{NluError, NluResult};
use self::parse_ranker::ParseRanker;

/// The result of NLU parsing, including provenance about which tier succeeded.
#[derive(Debug, Clone)]
pub struct NluParseResult {
    /// The parsed abstract syntax tree.
    pub tree: AbsTree,
    /// Which tier produced this parse (1-4).
    pub source_tier: u8,
    /// Confidence score from the producing tier.
    pub confidence: f32,
    /// If the ranker contributed, the best exemplar similarity.
    pub exemplar_similarity: Option<f32>,
}

/// The NLU pipeline orchestrator.
///
/// Holds configuration and state for the cascading parse pipeline.
/// The ranker accumulates exemplars over time for self-improving parsing.
/// ML and LLM tiers are feature-gated and degrade gracefully when models
/// are absent.
pub struct NluPipeline {
    /// VSA parse ranker (Tier 4) — always available.
    ranker: ParseRanker,
    /// Micro-ML NER layer (Tier 2) — loaded if model files present.
    #[cfg(feature = "nlu-ml")]
    ml_layer: Option<micro_ml::MicroMlLayer>,
    /// Small LLM translator (Tier 3) — loaded if model file present.
    #[cfg(feature = "nlu-llm")]
    llm_translator: Option<llm_translator::LlmTranslator>,
}

impl NluPipeline {
    /// Create a new NLU pipeline with a fresh ranker and no ML models.
    pub fn new() -> Self {
        Self {
            ranker: ParseRanker::new(),
            #[cfg(feature = "nlu-ml")]
            ml_layer: None,
            #[cfg(feature = "nlu-llm")]
            llm_translator: None,
        }
    }

    /// Create a pipeline with a pre-existing ranker (restored from persistence).
    pub fn with_ranker(ranker: ParseRanker) -> Self {
        Self {
            ranker,
            #[cfg(feature = "nlu-ml")]
            ml_layer: None,
            #[cfg(feature = "nlu-llm")]
            llm_translator: None,
        }
    }

    /// Create a pipeline with a pre-existing ranker and attempt to load models
    /// from `data_dir`. Models that are absent are silently skipped.
    pub fn with_ranker_and_models(ranker: ParseRanker, data_dir: Option<&Path>) -> Self {
        let mut pipeline = Self::with_ranker(ranker);
        if let Some(dir) = data_dir {
            pipeline.load_models(dir);
        }
        pipeline
    }

    /// Create a pipeline with a fresh ranker, attempting to load models
    /// from `data_dir`. Models that are absent are silently skipped.
    pub fn new_with_models(data_dir: Option<&Path>) -> Self {
        let mut pipeline = Self::new();
        if let Some(dir) = data_dir {
            pipeline.load_models(dir);
        }
        pipeline
    }

    /// Attempt to load ML and LLM models from the given data directory.
    /// Missing models are silently skipped (graceful degradation).
    fn load_models(&mut self, _data_dir: &Path) {
        #[cfg(feature = "nlu-ml")]
        {
            self.ml_layer = micro_ml::MicroMlLayer::try_load(_data_dir);
        }
        #[cfg(feature = "nlu-llm")]
        {
            self.llm_translator = llm_translator::LlmTranslator::try_load(_data_dir);
        }
    }

    /// Access the ranker for persistence.
    pub fn ranker(&self) -> &ParseRanker {
        &self.ranker
    }

    /// Access the ranker mutably for recording successes.
    pub fn ranker_mut(&mut self) -> &mut ParseRanker {
        &mut self.ranker
    }

    /// Parse input through the NLU cascade.
    ///
    /// Returns a structured parse result if any tier succeeds, or `NluError::ParseFailed`
    /// if all tiers fail.
    pub fn parse(&mut self, input: &str, ctx: &ParseContext) -> NluResult<NluParseResult> {
        // Tier 1: Rule parser (extended parse_prose)
        let result = parser::parse_prose(input, ctx);

        match &result {
            // Structured parse succeeded
            ParseResult::Facts(facts) if !facts.is_empty() => {
                let tree = if facts.len() == 1 {
                    facts[0].clone()
                } else {
                    AbsTree::and(facts.clone())
                };

                let nlu_result = NluParseResult {
                    tree: tree.clone(),
                    source_tier: 1,
                    confidence: 0.85,
                    exemplar_similarity: None,
                };

                // Record success for the ranker
                self.ranker.record_success(input, &tree, 1, 0.85);

                return Ok(nlu_result);
            }
            ParseResult::Query { subject: _, tree } => {
                let nlu_result = NluParseResult {
                    tree: tree.clone(),
                    source_tier: 1,
                    confidence: 0.80,
                    exemplar_similarity: None,
                };
                self.ranker
                    .record_success(input, &nlu_result.tree, 1, 0.80);
                return Ok(nlu_result);
            }
            ParseResult::Goal { description } => {
                let tree = AbsTree::Freeform(description.clone());
                let nlu_result = NluParseResult {
                    tree,
                    source_tier: 1,
                    confidence: 0.75,
                    exemplar_similarity: None,
                };
                return Ok(nlu_result);
            }
            _ => {}
        }

        // Tier 2: Micro-ML NER (feature-gated)
        #[cfg(feature = "nlu-ml")]
        if let Some(ref mut ml) = self.ml_layer {
            if let Ok(augmented) = ml.augment_parse(input, ctx) {
                if let Some(tree) = augmented.tree {
                    self.ranker
                        .record_success(input, &tree, 2, augmented.confidence);
                    return Ok(NluParseResult {
                        tree,
                        source_tier: 2,
                        confidence: augmented.confidence,
                        exemplar_similarity: None,
                    });
                }
            }
        }

        // Tier 3: Small LLM translator (feature-gated)
        #[cfg(feature = "nlu-llm")]
        if let Some(ref llm) = self.llm_translator {
            if let Ok(translation) = llm.translate(input) {
                self.ranker
                    .record_success(input, &translation.tree, 3, 0.70);
                return Ok(NluParseResult {
                    tree: translation.tree,
                    source_tier: 3,
                    confidence: 0.70,
                    exemplar_similarity: None,
                });
            }
        }

        // Tier 4: VSA Parse Ranker — check if we have a similar exemplar
        if let Some(ranked) = self.ranker.find_similar(input) {
            return Ok(NluParseResult {
                tree: ranked.tree,
                source_tier: 4,
                confidence: ranked.confidence,
                exemplar_similarity: Some(ranked.similarity),
            });
        }

        // All tiers failed
        Err(NluError::ParseFailed {
            input: input.to_string(),
        })
    }
}

impl Default for NluPipeline {
    fn default() -> Self {
        Self::new()
    }
}
