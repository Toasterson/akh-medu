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

pub mod bandit;
pub mod cache;
pub mod error;
pub mod kg_context;
pub mod parse_ranker;
pub mod router;

#[cfg(feature = "nlu-ml")]
pub mod micro_ml;

pub mod llm_translator;
pub mod candle_backend;

// Candle GPU + audio + hidden-states forks (ADRs 067/068/069). `gpu` is a pure
// host-capability probe (no candle dep); `hip` needs the ROCm toolchain
// (`rocm` feature); `audio` (Gemma-4) and the `vendored` hidden-states model
// forks need the Candle backend.
pub mod channel;
pub mod gpu;
#[cfg(feature = "rocm")]
pub mod hip;
#[cfg(feature = "candle-backend")]
pub mod audio;
#[cfg(feature = "candle-backend")]
pub mod vendored;

use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::grammar::abs::AbsTree;
use crate::grammar::concrete::ParseContext;
use crate::grammar::parser::{self, ParseResult};

use self::error::{NluError, NluResult};
use self::parse_ranker::ParseRanker;
use self::router::{BackendId, NakhRouter};

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
    training_collector: Option<Arc<Mutex<crate::training::TrainingDataCollector>>>,
    /// n akh semantic router (Phase 28) — shared across clones so all sessions
    /// contribute to the same bandit.  `None` until `enable_router()` is called.
    router: Option<Arc<Mutex<NakhRouter>>>,
}

impl Clone for NluPipeline {
    /// Clone the pipeline: ranker is cloned by value (per-session learning),
    /// model layers and router are shared via Arc (cheap).
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
            router: self.router.clone(),
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
            router: None,
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
            router: None,
        }
    }

    /// Attach a training data collector for logging parse successes.
    pub fn set_training_collector(
        &mut self,
        collector: Arc<Mutex<crate::training::TrainingDataCollector>>,
    ) {
        self.training_collector = Some(collector);
    }

    /// Enable the n akh semantic router with production defaults.
    ///
    /// Once enabled the router is shared across all clones of this pipeline
    /// so every session contributes to the same bandit learning state.
    pub fn enable_router(&mut self) {
        let router = NakhRouter::with_defaults();
        self.router = Some(Arc::new(Mutex::new(router)));
        tracing::info!("n akh semantic router enabled");
    }

    /// Attach an externally created router (e.g., restored from persisted state).
    pub fn set_router(&mut self, router: Arc<Mutex<NakhRouter>>) {
        self.router = Some(router);
    }

    /// Signal that model weights have been hot-swapped (Phase 27 integration hook).
    ///
    /// Bumps the router's exploration rate so the bandit re-evaluates the
    /// `BurnLocal` backend with fresh eyes.
    pub fn on_model_weights_updated(&self, delta: f32) {
        if let Some(router_arc) = &self.router {
            if let Ok(mut router) = router_arc.lock() {
                router.on_model_weights_updated(delta);
            }
        }
    }

    /// Invalidate the router's semantic cache after KG mutations.
    pub fn invalidate_router_cache(&self) {
        if let Some(router_arc) = &self.router {
            if let Ok(mut router) = router_arc.lock() {
                router.invalidate_cache();
            }
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

    /// Parse input through the NLU pipeline.
    ///
    /// When the n akh router is enabled (Phase 28) the router selects the most
    /// likely backend, that backend runs first, and the full cascade is only
    /// used as a fallback on failure.  Without a router the classic sequential
    /// cascade runs unchanged.
    ///
    /// Returns a structured parse result or `NluError::ParseFailed` when every
    /// path fails.
    pub fn parse(&mut self, input: &str, ctx: &ParseContext) -> NluResult<NluParseResult> {
        self.parse_with_context(input, ctx, None)
    }

    /// Parse with an optional KG context block that is passed to LLM tiers.
    ///
    /// `kg_context` is forwarded to Tier 3 (LLM/Candle) prompt construction
    /// so the model can ground entity references against known KG facts.
    /// Tiers 1, 2, and 4 ignore the context (they are rule- or exemplar-based).
    ///
    /// When the n akh router is enabled (Phase 28) the router selects the most
    /// likely backend, that backend runs first, and the full cascade is only
    /// used as a fallback on failure.  Without a router the classic sequential
    /// cascade runs unchanged.
    #[allow(unused_variables)] // kg_context used only when nlu-llm or candle-backend is enabled
    pub fn parse_with_context(
        &mut self,
        input: &str,
        ctx: &ParseContext,
        kg_context: Option<&str>,
    ) -> NluResult<NluParseResult> {
        // n akh router path (Phase 28) ─────────────────────────────────────────
        let router_arc = self.router.as_ref().map(Arc::clone);
        if let Some(router_arc) = router_arc {
            // Semantic cache check — serve identical/near-identical queries instantly.
            if let Some(cached) = router_arc.lock().unwrap().find_cached(input) {
                tracing::debug!(router = "cache_hit", "n akh: semantic cache served result");
                return Ok(cached);
            }

            // Bandit selects the backend most likely to succeed for this query.
            let decision = match router_arc.lock().unwrap().route(input) {
                Ok(d) => d,
                Err(e) => {
                    tracing::warn!(error = %e, "n akh router failed; falling back to cascade");
                    return self.cascade_parse(input, ctx, kg_context);
                }
            };

            tracing::debug!(
                router = "dispatch",
                backend = %decision.backend,
                "n akh dispatching to backend"
            );

            // Execute the bandit-selected backend.
            let result = match decision.backend {
                BackendId::RuleParser => self.tier1_parse(input, ctx),
                BackendId::OnnxNer => self.tier2_parse(input, ctx),
                BackendId::CandleLlm => self.tier3_parse(input, kg_context),
                BackendId::VsaDirect => self.tier4_parse(input),
                // CloudApi and BurnLocal are not yet implemented; cascade handles them.
                BackendId::CloudApi | BackendId::BurnLocal => {
                    Err(NluError::ParseFailed { input: input.to_string() })
                }
            };

            let (result, reward_override) = match result {
                Ok(r) => (r, None),
                Err(_) => {
                    tracing::debug!(
                        backend = %decision.backend,
                        "n akh: selected backend failed; cascade fallback"
                    );
                    // Cascade may also fail; propagate that error.
                    let cascaded = self.cascade_parse(input, ctx, kg_context)?;
                    // Penalise the bandit arm that caused the fallback.
                    (cascaded, Some(0.0f32))
                }
            };

            // Update bandit and populate cache.
            router_arc
                .lock()
                .unwrap()
                .report_reward(&decision, &result, input, reward_override);

            return Ok(result);
        }

        // No router — sequential cascade (original behaviour).
        self.cascade_parse(input, ctx, kg_context)
    }

    // ── Per-tier helpers (called directly by router dispatch) ────────────────

    fn tier1_parse(&mut self, input: &str, ctx: &ParseContext) -> NluResult<NluParseResult> {
        tracing::debug!(tier = 1, "attempting rule parser");
        let result = parser::parse_prose(input, ctx);
        match &result {
            ParseResult::Facts(facts) if !facts.is_empty() => {
                let tree = if facts.len() == 1 {
                    facts[0].clone()
                } else {
                    AbsTree::and(facts.clone())
                };
                tracing::info!(tier = 1, confidence = 0.85, "rule parser succeeded");
                self.ranker.record_success(input, &tree, 1, 0.85);
                Ok(self.make_result(input, tree, 1, 0.85, None))
            }
            ParseResult::Query { subject: _, tree } => {
                tracing::info!(tier = 1, confidence = 0.80, "rule parser query matched");
                let tree = tree.clone();
                self.ranker.record_success(input, &tree, 1, 0.80);
                Ok(self.make_result(input, tree, 1, 0.80, None))
            }
            ParseResult::Goal { description } => {
                tracing::info!(tier = 1, confidence = 0.75, "rule parser goal matched");
                let tree = AbsTree::Freeform(description.clone());
                Ok(self.make_result(input, tree, 1, 0.75, None))
            }
            _ => {
                tracing::debug!(tier = 1, "rule parser did not match");
                Err(NluError::ParseFailed { input: input.to_string() })
            }
        }
    }

    fn tier2_parse(&mut self, input: &str, ctx: &ParseContext) -> NluResult<NluParseResult> {
        #[cfg(feature = "nlu-ml")]
        if let Some(ml_arc) = self.ml_layer.as_ref().map(Arc::clone) {
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
        #[cfg(not(feature = "nlu-ml"))]
        let _ = ctx;
        Err(NluError::ParseFailed { input: input.to_string() })
    }

    #[allow(unused_variables)] // kg_context used only when nlu-llm or candle-backend is enabled
    fn tier3_parse(&mut self, input: &str, kg_context: Option<&str>) -> NluResult<NluParseResult> {
        // Tier 3 (Candle): Alternative LLM backend with hidden state access.
        // Preferred over llama-cpp-2 when both are loaded.
        #[cfg(feature = "candle-backend")]
        if let Some(candle_arc) = self.candle_backend.as_ref().map(Arc::clone) {
            tracing::debug!(tier = 3, backend = "candle", has_kg_context = kg_context.is_some(), "attempting Candle translation");
            let translation_result = candle_arc.lock().unwrap().translate_with_context(input, kg_context);
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

        // Fallback: llama-cpp-2 LLM (Wrapped in catch_unwind: llama.cpp may abort).
        #[cfg(feature = "nlu-llm")]
        if let Some(ref llm_arc) = self.llm_translator {
            let llm = Arc::clone(llm_arc);
            let input_owned = input.to_string();
            let ctx_owned = kg_context.map(str::to_string);
            tracing::debug!(tier = 3, has_kg_context = ctx_owned.is_some(), "attempting LLM translation");
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                llm.translate_with_context(&input_owned, ctx_owned.as_deref())
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
                    tracing::error!(tier = 3, "LLM translation panicked (llama.cpp assertion)");
                }
            }
        }

        Err(NluError::ParseFailed { input: input.to_string() })
    }

    fn tier4_parse(&mut self, input: &str) -> NluResult<NluParseResult> {
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
        Err(NluError::ParseFailed { input: input.to_string() })
    }

    /// Sequential cascade: tries each tier in order, returns first success.
    #[allow(unused_variables)] // kg_context used only when nlu-llm or candle-backend is enabled
    fn cascade_parse(
        &mut self,
        input: &str,
        ctx: &ParseContext,
        kg_context: Option<&str>,
    ) -> NluResult<NluParseResult> {
        if let Ok(r) = self.tier1_parse(input, ctx) {
            return Ok(r);
        }
        if let Ok(r) = self.tier2_parse(input, ctx) {
            return Ok(r);
        }
        if let Ok(r) = self.tier3_parse(input, kg_context) {
            return Ok(r);
        }
        if let Ok(r) = self.tier4_parse(input) {
            return Ok(r);
        }
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
    /// using the LLM with persona and optional KG context.
    ///
    /// `context` is the active conversation topic (e.g. "the Ennead").
    /// `kg_context` is the pre-built KG context block from [`kg_context::build_kg_context`];
    /// when present it is injected into the system prompt to ground the response.
    ///
    /// Returns `None` when no LLM backend is loaded.
    #[allow(unused_variables)] // params used only when nlu-llm feature is enabled
    pub fn generate_dialogue(
        &self,
        act: &str,
        persona_name: &str,
        traits: &[String],
        context: Option<&str>,
        kg_context: Option<&str>,
    ) -> Option<String> {
        self.generate_dialogue_with_kg(act, persona_name, traits, context, None)
    }

    /// Generate a dialogue act response with an optional KG context block.
    ///
    /// `kg_context` is injected into the system prompt so the model can ground
    /// its response in actual knowledge-graph facts. The combined prompt is
    /// designed to stay under 3 800 tokens (Psyche prompt + KG context).
    #[allow(unused_variables)] // params used only when a backend feature is enabled
    pub fn generate_dialogue_with_kg(
        &self,
        act: &str,
        persona_name: &str,
        traits: &[String],
        context: Option<&str>,
        kg_context: Option<&str>,
    ) -> Option<String> {
        // Candle (native, in-process): the primary conversational backend when an
        // akh is Candle-backed. Prefer grounded KG context as the dialogue context.
        #[cfg(feature = "candle-backend")]
        if let Some(candle_arc) = self.candle_backend.as_ref().map(std::sync::Arc::clone) {
            if let Ok(mut backend) = candle_arc.lock() {
                let ctx = kg_context.or(context);
                match backend.generate_dialogue(act, persona_name, traits, ctx) {
                    Ok(text) if !text.is_empty() => return Some(text),
                    Ok(_) => {}
                    Err(e) => {
                        tracing::warn!(backend = "candle", error = %e, "dialogue generation failed");
                    }
                }
            }
        }

        // Fallback: llama.cpp llm_translator (legacy `nlu-llm`).
        #[cfg(feature = "nlu-llm")]
        if let Some(ref llm_arc) = self.llm_translator {
            let llm = std::sync::Arc::clone(llm_arc);
            let prompt = llm_translator::build_dialogue_prompt_with_kg(
                act,
                persona_name,
                traits,
                context,
                kg_context,
            );
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

    /// Channel-aware variant of [`generate_dialogue`](Self::generate_dialogue):
    /// returns the typed [`ChannelStream`](channel::ChannelStream) so the chat
    /// layer can surface the boundary model's internal reasoning (Gemma-4
    /// `thought` channel) as a collapsible trace distinct from the answer. Candle
    /// only; `None` when no Candle backend is loaded. See ADR 069.
    #[allow(unused_variables)]
    pub fn generate_dialogue_channels(
        &self,
        act: &str,
        persona_name: &str,
        traits: &[String],
        context: Option<&str>,
    ) -> Option<channel::ChannelStream> {
        #[cfg(feature = "candle-backend")]
        if let Some(candle_arc) = self.candle_backend.as_ref().map(std::sync::Arc::clone) {
            if let Ok(mut backend) = candle_arc.lock() {
                match backend.generate_dialogue_channels(act, persona_name, traits, context) {
                    Ok(stream) if !stream.answer().is_empty() || stream.has_thoughts() => {
                        return Some(stream);
                    }
                    Ok(_) => {}
                    Err(e) => {
                        tracing::warn!(backend = "candle", error = %e, "dialogue generation failed");
                    }
                }
            }
        }
        None
    }

    /// Whether a conversational verbalizer is available (the Candle backend).
    ///
    /// When `true`, [`verbalize`](Self::verbalize) can rephrase grounded facts
    /// into fluent, chat-like prose. When `false`, callers fall back to the
    /// grammar linearizer (deterministic, no LLM).
    pub fn has_verbalizer(&self) -> bool {
        let mut available = false;
        #[cfg(feature = "candle-backend")]
        {
            available = available || self.candle_backend.is_some();
        }
        available
    }

    /// Verbalize a grounded answer: turn KG-derived `facts` into a natural,
    /// conversational reply to `question`, constrained to those facts.
    ///
    /// Grounded generation — the facts come from the knowledge graph, so the KG
    /// remains the source of truth. Returns `None` when no verbalizer backend is
    /// loaded (caller falls back to grammar linearization).
    #[allow(unused_variables)]
    pub fn verbalize(
        &self,
        question: &str,
        facts: &str,
        persona_name: &str,
        traits: &[String],
    ) -> Option<String> {
        #[cfg(feature = "candle-backend")]
        if let Some(candle_arc) = self.candle_backend.as_ref().map(std::sync::Arc::clone) {
            if let Ok(mut backend) = candle_arc.lock() {
                match backend.verbalize(question, facts, persona_name, traits) {
                    Ok(text) if !text.is_empty() => return Some(text),
                    Ok(_) => {}
                    Err(e) => {
                        tracing::warn!(backend = "candle", error = %e, "verbalization failed");
                    }
                }
            }
        }
        None
    }

    /// Channel-aware variant of [`verbalize`](Self::verbalize): returns the typed
    /// [`ChannelStream`](channel::ChannelStream) so grounded reasoning can be
    /// surfaced as a collapsible trace. Candle only. See ADR 069.
    #[allow(unused_variables)]
    pub fn verbalize_channels(
        &self,
        question: &str,
        facts: &str,
        persona_name: &str,
        traits: &[String],
    ) -> Option<channel::ChannelStream> {
        #[cfg(feature = "candle-backend")]
        if let Some(candle_arc) = self.candle_backend.as_ref().map(std::sync::Arc::clone) {
            if let Ok(mut backend) = candle_arc.lock() {
                match backend.verbalize_channels(question, facts, persona_name, traits) {
                    Ok(stream) if !stream.answer().is_empty() || stream.has_thoughts() => {
                        return Some(stream);
                    }
                    Ok(_) => {}
                    Err(e) => {
                        tracing::warn!(backend = "candle", error = %e, "verbalization failed");
                    }
                }
            }
        }
        None
    }

    /// Generate an open-ended LLM response for creative or novel queries.
    ///
    /// Unlike `generate_dialogue` (which handles short dialogue acts), this method
    /// is designed for multi-sentence responses: creative writing, hypotheticals, etc.
    /// KG facts are injected into the prompt to reduce hallucination on factual claims.
    ///
    /// Returns `None` if no LLM is available (graceful degradation).
    #[allow(unused_variables)] // params used only when nlu-llm feature is enabled
    pub fn generate_open_ended(
        &self,
        query: &str,
        persona_name: &str,
        traits: &[String],
        kg_context: Option<&str>,
    ) -> Option<String> {
        #[cfg(feature = "nlu-llm")]
        if let Some(ref llm_arc) = self.llm_translator {
            let llm = std::sync::Arc::clone(llm_arc);
            let prompt = llm_translator::build_open_ended_prompt(query, persona_name, traits, kg_context);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                llm.generate(&prompt, 256)
            }));
            match result {
                Ok(Ok(text)) if !text.is_empty() => {
                    return Some(text);
                }
                Ok(Ok(_)) => {
                    tracing::debug!("LLM open-ended generation returned empty text");
                }
                Ok(Err(e)) => {
                    tracing::debug!(error = %e, "LLM open-ended generation failed");
                }
                Err(_) => {
                    tracing::warn!("LLM open-ended generation panicked");
                }
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

