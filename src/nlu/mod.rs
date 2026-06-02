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
pub mod candle_backend;

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
    /// Concepts identified by the neural→VSA bridge (Phase 26e).
    /// Each entry is (symbol_id_u64, similarity_score).
    pub neural_concepts: Vec<(u64, f32)>,
}

/// A concept identified by the neural bridge after parsing.
#[derive(Debug, Clone)]
pub struct NeuralConcept {
    /// Symbol ID of the identified concept.
    pub symbol_id: u64,
    /// Similarity between encoded hidden state and grounded concept vector.
    pub similarity: f32,
}

/// The NLU pipeline orchestrator.
///
/// Holds configuration and state for the cascading parse pipeline.
/// The ranker accumulates exemplars over time for self-improving parsing.
/// ML and LLM tiers are feature-gated and degrade gracefully when models
/// are absent.
pub struct NluPipeline {
    /// VSA parse ranker (Tier 4) — always available, per-session (cloned).
    ranker: ParseRanker,
    /// Micro-ML NER layer (Tier 2) — loaded if model files present.
    /// Wrapped in Arc<Mutex<>> because loading is expensive (~130MB) and we
    /// want to share the model across WS/MCP sessions.
    #[cfg(feature = "nlu-ml")]
    ml_layer: Option<std::sync::Arc<std::sync::Mutex<micro_ml::MicroMlLayer>>>,
    /// Small LLM translator (Tier 3) — loaded if model file present.
    /// Wrapped in Arc because loading is expensive (~1GB GGUF).
    #[cfg(feature = "nlu-llm")]
    llm_translator: Option<std::sync::Arc<llm_translator::LlmTranslator>>,
    /// Candle LLM backend (Tier 3 alternative) — GGUF via Candle with hidden state access.
    /// Mutually exclusive with llm_translator at runtime; Candle preferred when both available.
    #[cfg(feature = "candle-backend")]
    candle_backend: Option<std::sync::Arc<std::sync::Mutex<candle_backend::CandleBackend>>>,
    /// Training data collector (Phase 27f) — logs parse successes as training pairs.
    training_collector: Option<std::sync::Arc<std::sync::Mutex<crate::training::TrainingDataCollector>>>,
}

impl Clone for NluPipeline {
    /// Clone the pipeline: ranker is cloned by value (per-session learning),
    /// model layers are shared via Arc (cheap).
    fn clone(&self) -> Self {
        Self {
            ranker: self.ranker.clone(),
            #[cfg(feature = "nlu-ml")]
            ml_layer: self.ml_layer.clone(),
            #[cfg(feature = "nlu-llm")]
            llm_translator: self.llm_translator.clone(),
            #[cfg(feature = "candle-backend")]
            candle_backend: self.candle_backend.clone(),
            training_collector: self.training_collector.clone(),
        }
    }
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
            #[cfg(feature = "candle-backend")]
            candle_backend: None,
            training_collector: None,
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
            #[cfg(feature = "candle-backend")]
            candle_backend: None,
            training_collector: None,
        }
    }

    /// Attach a training data collector for logging parse successes.
    pub fn set_training_collector(
        &mut self,
        collector: std::sync::Arc<std::sync::Mutex<crate::training::TrainingDataCollector>>,
    ) {
        self.training_collector = Some(collector);
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
            // Auto-detect ONNX Runtime if ORT_DYLIB_PATH is not set.
            if std::env::var("ORT_DYLIB_PATH").is_err() {
                let candidates = crate::setup::ort_lib_candidates();
                for candidate in &candidates {
                    if candidate.exists() {
                        tracing::info!(path = %candidate.display(), "auto-detected ONNX Runtime library");
                        // SAFETY: set_var is called during single-threaded pipeline init,
                        // before any ONNX threads are spawned. The `ort` crate reads this
                        // variable only once during its lazy init.
                        unsafe { std::env::set_var("ORT_DYLIB_PATH", candidate); }
                        break;
                    }
                }
            }

            // The `ort` crate panics if it cannot dlopen libonnxruntime.
            // Catch the panic so we degrade gracefully instead of crashing.
            let data_dir_owned = _data_dir.to_path_buf();
            let shared_owned = _shared_dir.clone();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                micro_ml::MicroMlLayer::load(&data_dir_owned).or_else(|_| {
                    if let Some(ref shared) = shared_owned {
                        micro_ml::MicroMlLayer::load(shared)
                    } else {
                        micro_ml::MicroMlLayer::load(&data_dir_owned)
                    }
                })
            }));
            match result {
                Ok(Ok(layer)) => {
                    tracing::info!(tier = 2, "ONNX NER model loaded");
                    self.ml_layer = Some(std::sync::Arc::new(std::sync::Mutex::new(layer)));
                }
                Ok(Err(e)) => {
                    tracing::warn!(tier = 2, error = %e, "ONNX NER model not loaded");
                    self.ml_layer = None;
                }
                Err(_) => {
                    tracing::warn!(
                        tier = 2,
                        "ONNX Runtime not available (dlopen failed). Run `akh setup onnx-runtime` to install."
                    );
                    self.ml_layer = None;
                }
            }
        }
        #[cfg(feature = "nlu-llm")]
        {
            let llm_file = "llm/qwen2.5-1.5b-instruct-q4_k_m.gguf";
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
                    self.llm_translator = Some(std::sync::Arc::new(translator));
                }
                Err(e) => {
                    tracing::warn!(tier = 3, error = %e, "LLM translator model not loaded");
                    self.llm_translator = None;
                }
            }
        }
        // Candle backend (Tier 3 alternative): preferred over llama-cpp-2 when available.
        #[cfg(feature = "candle-backend")]
        {
            let result = candle_backend::CandleBackend::try_load(_data_dir)
                .or_else(|| {
                    _shared_dir
                        .as_ref()
                        .and_then(|shared| candle_backend::CandleBackend::try_load(shared))
                });
            if let Some(backend) = result {
                tracing::info!(tier = 3, backend = "candle", "Candle LLM backend loaded");
                self.candle_backend =
                    Some(std::sync::Arc::new(std::sync::Mutex::new(backend)));
            } else {
                tracing::debug!(tier = 3, backend = "candle", "Candle backend not loaded (model absent)");
            }
        }
    }

    /// Log an NLG training pair (grammar linearization → natural text).
    ///
    /// Called from the chat processor after a successful linearization.
    pub fn log_nlg_pair(&self, triples_desc: &str, output_text: &str, grammar: &str) {
        if let Some(ref collector_arc) = self.training_collector {
            if let Ok(mut collector) = collector_arc.try_lock() {
                if let Err(e) = collector.record_nlg(triples_desc, output_text, grammar, 1.0) {
                    tracing::debug!(error = %e, "training data collector: NLG record failed");
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

    /// Build an NluParseResult and log to the training collector if attached.
    fn make_result(
        &mut self,
        input: &str,
        tree: AbsTree,
        source_tier: u8,
        confidence: f32,
        exemplar_similarity: Option<f32>,
    ) -> NluParseResult {
        // Log to training data collector (Phase 27f).
        if let Some(ref collector_arc) = self.training_collector {
            if let Ok(mut collector) = collector_arc.try_lock() {
                let json = serde_json::to_string(&tree).unwrap_or_default();
                if let Err(e) = collector.record_nlu(input, &json, source_tier, confidence) {
                    tracing::debug!(error = %e, "training data collector: NLU record failed");
                }
            }
        }

        NluParseResult {
            tree,
            source_tier,
            confidence,
            exemplar_similarity,
            neural_concepts: Vec::new(),
        }
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
                self.ranker.record_success(input, &tree, 1, 0.85);
                return Ok(self.make_result(input, tree, 1, 0.85, None));
            }
            ParseResult::Query { subject: _, tree } => {
                tracing::info!(tier = 1, confidence = 0.80, "rule parser query matched");
                let tree = tree.clone();
                self.ranker.record_success(input, &tree, 1, 0.80);
                return Ok(self.make_result(input, tree, 1, 0.80, None));
            }
            ParseResult::Goal { description } => {
                tracing::info!(tier = 1, confidence = 0.75, "rule parser goal matched");
                let tree = AbsTree::Freeform(description.clone());
                return Ok(self.make_result(input, tree, 1, 0.75, None));
            }
            _ => {
                tracing::debug!(tier = 1, "rule parser did not match");
            }
        }

        // Tier 2: Micro-ML NER (feature-gated)
        #[cfg(feature = "nlu-ml")]
        if let Some(ml_arc) = self.ml_layer.as_ref().map(std::sync::Arc::clone) {
            tracing::debug!(tier = 2, "attempting ML NER augmentation");
            let augmented_result = ml_arc.lock().unwrap().augment_parse(input, ctx);
            match augmented_result {
                Ok(augmented) => {
                    if let Some(tree) = augmented.tree {
                        let confidence = augmented.confidence;
                        tracing::info!(tier = 2, confidence, "ML NER succeeded");
                        self.ranker.record_success(input, &tree, 2, confidence);
                        return Ok(self.make_result(input, tree, 2, confidence, None));
                    }
                    tracing::debug!(tier = 2, "ML NER produced no tree");
                }
                Err(e) => {
                    tracing::warn!(tier = 2, error = %e, "ML NER failed");
                }
            }
        }

        // Tier 3: Small LLM translator (feature-gated)
        // Wrapped in catch_unwind because llama.cpp can GGML_ASSERT → abort().
        #[cfg(feature = "nlu-llm")]
        if let Some(ref llm_arc) = self.llm_translator {
            let llm = std::sync::Arc::clone(llm_arc);
            let input_owned = input.to_string();
            tracing::debug!(tier = 3, "attempting LLM translation");
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                llm.translate(&input_owned)
            }));
            match result {
                Ok(Ok(translation)) => {
                    tracing::info!(
                        tier = 3,
                        confidence = 0.70,
                        tokens = translation.tokens_generated,
                        "LLM translation succeeded"
                    );
                    self.ranker
                        .record_success(input, &translation.tree, 3, 0.70);
                    return Ok(self.make_result(input, translation.tree, 3, 0.70, None));
                }
                Ok(Err(e)) => {
                    tracing::warn!(tier = 3, error = %e, "LLM translation failed");
                }
                Err(_) => {
                    tracing::error!(tier = 3, "LLM translation panicked (llama.cpp assertion failure)");
                }
            }
        }

        // Tier 3 (Candle): Alternative LLM backend with hidden state access.
        // Preferred over llama-cpp-2 when both are loaded.
        #[cfg(feature = "candle-backend")]
        if let Some(candle_arc) = self.candle_backend.as_ref().map(std::sync::Arc::clone) {
            tracing::debug!(tier = 3, backend = "candle", "attempting Candle translation");
            let translation_result = candle_arc.lock().unwrap().translate(input);
            match translation_result {
                Ok(translation) => {
                    tracing::info!(
                        tier = 3,
                        backend = "candle",
                        confidence = 0.72,
                        tokens = translation.tokens_generated,
                        "Candle translation succeeded"
                    );
                    self.ranker
                        .record_success(input, &translation.tree, 3, 0.72);
                    return Ok(self.make_result(input, translation.tree, 3, 0.72, None));
                }
                Err(e) => {
                    tracing::warn!(tier = 3, backend = "candle", error = %e, "Candle translation failed");
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
            return Ok(self.make_result(
                input,
                ranked.tree,
                4,
                ranked.confidence,
                Some(ranked.similarity),
            ));
        }

        // All tiers failed
        tracing::warn!("all NLU tiers failed for input");
        Err(NluError::ParseFailed {
            input: input.to_string(),
        })
    }

    // ── Output boundary: dialogue acts only ──────────────────────────
    //
    // The LLM is NOT used for knowledge responses — the narrative grammar
    // handles that. The LLM is only used for short dialogue acts (greetings,
    // farewells) where hallucination risk is low and persona flavor helps.

    /// Generate natural language from VSA-structured text using the Candle backend.
    ///
    /// Intended for formatting VSA pipeline results and agent summaries as
    /// readable prose.  Returns `None` if the Candle backend is not loaded.
    ///
    /// # Latency
    /// The caller is responsible for measuring end-to-end latency from VSA
    /// output production to the returned string.  Internal decoding latency is
    /// logged at `DEBUG` level.
    #[allow(unused_variables)]
    pub fn generate_vsa_summary(&self, structured_text: &str, max_tokens: u32) -> Option<String> {
        #[cfg(feature = "candle-backend")]
        if let Some(ref backend_arc) = self.candle_backend {
            match backend_arc.lock() {
                Ok(mut backend) => {
                    let prompt = build_vsa_nl_prompt(structured_text);
                    match backend.generate(&prompt, max_tokens) {
                        Ok(text) if !text.trim().is_empty() => {
                            tracing::debug!(
                                approx_words = text.split_whitespace().count(),
                                "NluPipeline: VSA summary generated via Candle"
                            );
                            return Some(text);
                        }
                        Ok(_) => {}
                        Err(e) => {
                            tracing::debug!(
                                error = %e,
                                "NluPipeline: VSA summary generation failed"
                            );
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "NluPipeline: backend lock poisoned");
                }
            }
        }
        None
    }

    /// Generate a brief dialogue act response (greeting, farewell, etc.)
    /// using the LLM with persona context. Returns `None` if LLM unavailable.
    #[allow(unused_variables)] // params used only when nlu-llm feature is enabled
    pub fn generate_dialogue(
        &self,
        act: &str,
        persona_name: &str,
        traits: &[String],
        context: Option<&str>,
    ) -> Option<String> {
        #[cfg(feature = "nlu-llm")]
        if let Some(ref llm_arc) = self.llm_translator {
            let llm = std::sync::Arc::clone(llm_arc);
            let prompt = llm_translator::build_dialogue_prompt(act, persona_name, traits, context);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                llm.generate(&prompt, 64)
            }));
            match result {
                Ok(Ok(text)) if !text.is_empty() => {
                    return Some(text);
                }
                _ => {}
            }
        }
        None
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
                let mut available = false;
                #[cfg(feature = "nlu-llm")]
                {
                    available = self.llm_translator.is_some();
                }
                #[cfg(feature = "candle-backend")]
                {
                    available = available || self.candle_backend.is_some();
                }
                available
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

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build a Qwen2-style chat prompt for VSA → NL generation.
fn build_vsa_nl_prompt(structured_text: &str) -> String {
    format!(
        "<|im_start|>system\n\
         You are a concise assistant. Present knowledge graph findings clearly.\n\
         <|im_end|>\n\
         <|im_start|>user\n\
         {structured_text}\n\
         <|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

