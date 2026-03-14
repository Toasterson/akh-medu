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

    /// Attempt to load ML and LLM models from the given data directory,
    /// falling back to the shared XDG models directory.
    /// Missing models are logged and skipped (graceful degradation).
    fn load_models(&mut self, _data_dir: &Path) {
        // Build search paths: workspace-local first, then shared XDG models dir.
        let _shared_dir = crate::paths::AkhPaths::resolve()
            .ok()
            .map(|p| p.models_dir());

        #[cfg(feature = "nlu-ml")]
        {
            // Try workspace-local, then shared.
            let result = micro_ml::MicroMlLayer::load(_data_dir).or_else(|_| {
                if let Some(ref shared) = _shared_dir {
                    micro_ml::MicroMlLayer::load(shared)
                } else {
                    micro_ml::MicroMlLayer::load(_data_dir) // re-run to get error
                }
            });
            match result {
                Ok(layer) => {
                    tracing::info!(tier = 2, "ONNX NER model loaded");
                    self.ml_layer = Some(layer);
                }
                Err(e) => {
                    tracing::warn!(tier = 2, error = %e, "ONNX NER model not loaded");
                    self.ml_layer = None;
                }
            }
        }
        #[cfg(feature = "nlu-llm")]
        {
            let llm_file = "models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf";
            let result = llm_translator::LlmTranslator::load(&_data_dir.join(llm_file))
                .or_else(|_| {
                    if let Some(ref shared) = _shared_dir {
                        llm_translator::LlmTranslator::load(
                            &shared.join("llm/qwen2.5-1.5b-instruct-q4_k_m.gguf"),
                        )
                    } else {
                        llm_translator::LlmTranslator::load(&_data_dir.join(llm_file))
                    }
                });
            match result {
                Ok(translator) => {
                    tracing::info!(tier = 3, "LLM translator model loaded");
                    self.llm_translator = Some(translator);
                }
                Err(e) => {
                    tracing::warn!(tier = 3, error = %e, "LLM translator model not loaded");
                    self.llm_translator = None;
                }
            }
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
        tracing::debug!(tier = 1, "attempting rule parser");
        let result = parser::parse_prose(input, ctx);

        match &result {
            // Structured parse succeeded
            ParseResult::Facts(facts) if !facts.is_empty() => {
                let tree = if facts.len() == 1 {
                    facts[0].clone()
                } else {
                    AbsTree::and(facts.clone())
                };

                tracing::info!(tier = 1, confidence = 0.85, "rule parser succeeded");

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
                tracing::info!(tier = 1, confidence = 0.80, "rule parser query matched");
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
                tracing::info!(tier = 1, confidence = 0.75, "rule parser goal matched");
                let tree = AbsTree::Freeform(description.clone());
                let nlu_result = NluParseResult {
                    tree,
                    source_tier: 1,
                    confidence: 0.75,
                    exemplar_similarity: None,
                };
                return Ok(nlu_result);
            }
            _ => {
                tracing::debug!(tier = 1, "rule parser did not match");
            }
        }

        // Tier 2: Micro-ML NER (feature-gated)
        #[cfg(feature = "nlu-ml")]
        if let Some(ref mut ml) = self.ml_layer {
            tracing::debug!(tier = 2, "attempting ML NER augmentation");
            match ml.augment_parse(input, ctx) {
                Ok(augmented) => {
                    if let Some(tree) = augmented.tree {
                        tracing::info!(tier = 2, confidence = augmented.confidence, "ML NER succeeded");
                        self.ranker
                            .record_success(input, &tree, 2, augmented.confidence);
                        return Ok(NluParseResult {
                            tree,
                            source_tier: 2,
                            confidence: augmented.confidence,
                            exemplar_similarity: None,
                        });
                    }
                    tracing::debug!(tier = 2, "ML NER produced no tree");
                }
                Err(e) => {
                    tracing::warn!(tier = 2, error = %e, "ML NER failed");
                }
            }
        }

        // Tier 3: Small LLM translator (feature-gated)
        #[cfg(feature = "nlu-llm")]
        if let Some(ref llm) = self.llm_translator {
            tracing::debug!(tier = 3, "attempting LLM translation");
            match llm.translate(input) {
                Ok(translation) => {
                    tracing::info!(
                        tier = 3,
                        confidence = 0.70,
                        tokens = translation.tokens_generated,
                        "LLM translation succeeded"
                    );
                    self.ranker
                        .record_success(input, &translation.tree, 3, 0.70);
                    return Ok(NluParseResult {
                        tree: translation.tree,
                        source_tier: 3,
                        confidence: 0.70,
                        exemplar_similarity: None,
                    });
                }
                Err(e) => {
                    tracing::warn!(tier = 3, error = %e, "LLM translation failed");
                }
            }
        }

        // Tier 4: VSA Parse Ranker — check if we have a similar exemplar
        tracing::debug!(tier = 4, "checking parse ranker exemplars");
        if let Some(ranked) = self.ranker.find_similar(input) {
            tracing::info!(
                tier = 4,
                confidence = ranked.confidence,
                similarity = ranked.similarity,
                "ranker found similar exemplar"
            );
            return Ok(NluParseResult {
                tree: ranked.tree,
                source_tier: 4,
                confidence: ranked.confidence,
                exemplar_similarity: Some(ranked.similarity),
            });
        }

        // All tiers failed
        tracing::warn!("all NLU tiers failed for input");
        Err(NluError::ParseFailed {
            input: input.to_string(),
        })
    }

    /// Report which NLU tiers are currently available.
    pub fn tier_status(&self) -> NluTierStatus {
        NluTierStatus {
            tier1_rule_parser: true,
            tier2_ml_ner: {
                #[cfg(feature = "nlu-ml")]
                {
                    self.ml_layer.is_some()
                }
                #[cfg(not(feature = "nlu-ml"))]
                {
                    false
                }
            },
            tier3_llm: {
                #[cfg(feature = "nlu-llm")]
                {
                    self.llm_translator.is_some()
                }
                #[cfg(not(feature = "nlu-llm"))]
                {
                    false
                }
            },
            tier4_ranker: true,
        }
    }
}

/// Status of each NLU tier (loaded or unavailable).
#[derive(Debug, Clone)]
pub struct NluTierStatus {
    pub tier1_rule_parser: bool,
    pub tier2_ml_ner: bool,
    pub tier3_llm: bool,
    pub tier4_ranker: bool,
}

impl std::fmt::Display for NluTierStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "T1(rules)={} T2(NER)={} T3(LLM)={} T4(ranker)={}",
            if self.tier1_rule_parser { "ok" } else { "off" },
            if self.tier2_ml_ner { "ok" } else { "off" },
            if self.tier3_llm { "ok" } else { "off" },
            if self.tier4_ranker { "ok" } else { "off" },
        )
    }
}

impl Default for NluPipeline {
    fn default() -> Self {
        Self::new()
    }
}
