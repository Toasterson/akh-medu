//! n akh semantic router: routes queries to optimal backends via LinUCB bandit.
//!
//! Query flow:
//! 1. Check semantic cache (exact hash + VSA similarity ≥ 0.95).
//! 2. Extract 8-dimensional feature vector (archetype similarities).
//! 3. LinUCB bandit selects the backend with highest UCB score.
//! 4. Caller executes the backend; on result calls `report_reward`.
//!
//! Fallback: if routing fails or the backend returns an error the pipeline
//! falls back to the full sequential cascade and the failure is penalised
//! with reward 0.0.

use std::sync::Arc;

use miette::Diagnostic;
use thiserror::Error;

use crate::vsa::encode::encode_label;
use crate::vsa::ops::VsaOps;
use crate::vsa::{Dimension, Encoding, HyperVec};

use super::NluParseResult;
use super::bandit::{D, FeatureVec, LinUcbBandit};
use super::cache::SemanticCache;

/// Identifies a routable NLU backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum BackendId {
    RuleParser,
    OnnxNer,
    CandleLlm,
    VsaDirect,
    CloudApi,
    BurnLocal,
}

impl std::fmt::Display for BackendId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RuleParser => f.write_str("rule_parser"),
            Self::OnnxNer => f.write_str("onnx_ner"),
            Self::CandleLlm => f.write_str("candle_llm"),
            Self::VsaDirect => f.write_str("vsa_direct"),
            Self::CloudApi => f.write_str("cloud_api"),
            Self::BurnLocal => f.write_str("burn_local"),
        }
    }
}

/// Errors from the semantic router.
#[derive(Debug, Error, Diagnostic)]
pub enum RouterError {
    #[error("VSA encoding failed for input: {reason}")]
    #[diagnostic(
        code(akh::router::vsa_encode_failed),
        help("Check that the input contains at least one non-whitespace token")
    )]
    VsaEncode { reason: String },
}

pub type RouterResult<T> = Result<T, RouterError>;

/// A named query archetype: a VSA centroid for a class of queries.
///
/// Archetypes are bootstrapped from seed phrases at router init and can be
/// extended by clustering over the parse ranker's exemplar memory.
pub struct QueryArchetype {
    pub name: &'static str,
    pub prototype: HyperVec,
    /// Hint for the bandit's initial prior (overridable by learned reward).
    pub preferred_backend: Option<BackendId>,
}

/// Router configuration.
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// LinUCB exploration parameter α; range [0, 2] (default 0.25).
    pub exploration_rate: f32,
    /// Semantic cache capacity in entries (default 256).
    pub cache_size: usize,
    /// Minimum bandit score to skip cascade; below this falls back (default 0.40).
    pub min_confidence: f32,
    /// Permit routing to cloud APIs (default false).
    pub enable_cloud: bool,
    /// Cache TTL in seconds (default 300 = 5 min, per plan §28c).
    pub cache_ttl_secs: u64,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            exploration_rate: 0.25,
            cache_size: 256,
            min_confidence: 0.40,
            enable_cloud: false,
            cache_ttl_secs: 300,
        }
    }
}

/// The output of a routing decision.
///
/// The caller must use `arm_idx` and `feature_vec` when calling
/// [`NakhRouter::report_reward`] after backend execution.
pub struct RouteDecision {
    /// Backend selected by the bandit.
    pub backend: BackendId,
    /// LinUCB arm index — needed by `report_reward`.
    pub arm_idx: usize,
    /// Feature vector used for this decision — needed by `report_reward`.
    pub feature_vec: FeatureVec,
}

/// The `n akh` ("intelligence of akh") semantic router.
///
/// Replaces the fixed 4-tier cascade with a contextual bandit that learns
/// which backend handles which query type best.  A semantic LRU cache
/// intercepts repeated or near-identical queries before the bandit runs.
pub struct NakhRouter {
    bandit: LinUcbBandit,
    archetypes: Vec<QueryArchetype>,
    cache: SemanticCache,
    config: RouterConfig,
    ops: Arc<VsaOps>,
}

impl NakhRouter {
    /// Create a router with the given VSA ops and configuration.
    pub fn new(ops: Arc<VsaOps>, config: RouterConfig) -> Self {
        let backends = Self::backend_list(&config);
        let bandit = LinUcbBandit::new(&backends, config.exploration_rate);
        let cache = SemanticCache::new(config.cache_size, config.cache_ttl_secs);
        let archetypes = Self::bootstrap_archetypes(&ops);

        Self {
            bandit,
            archetypes,
            cache,
            config,
            ops,
        }
    }

    /// Convenience constructor: creates VsaOps internally with production defaults.
    pub fn with_defaults() -> Self {
        let ops = Arc::new(VsaOps::new(
            crate::simd::best_kernel(),
            Dimension::DEFAULT,
            Encoding::Bipolar,
        ));
        Self::new(ops, RouterConfig::default())
    }

    fn backend_list(config: &RouterConfig) -> Vec<BackendId> {
        let mut backends = vec![
            BackendId::RuleParser,
            BackendId::OnnxNer,
            BackendId::CandleLlm,
            BackendId::VsaDirect,
            BackendId::BurnLocal,
        ];
        if config.enable_cloud {
            backends.push(BackendId::CloudApi);
        }
        backends
    }

    /// Seed archetypes from representative example phrases for the 8 canonical
    /// query types.  All 8 types map to D=8 feature dimensions via similarity.
    fn bootstrap_archetypes(ops: &VsaOps) -> Vec<QueryArchetype> {
        const SEEDS: &[(&str, &[&str], Option<BackendId>)] = &[
            (
                "greeting",
                &[
                    "hello",
                    "good morning",
                    "hi there",
                    "goodbye",
                    "farewell",
                    "greetings",
                ],
                Some(BackendId::RuleParser),
            ),
            (
                "fact_assertion",
                &[
                    "dogs are mammals",
                    "water is wet",
                    "the sky is blue",
                    "cats are felines",
                ],
                Some(BackendId::RuleParser),
            ),
            (
                "entity_query",
                &[
                    "who is Alan Turing",
                    "what is a neuron",
                    "tell me about rust",
                    "describe the concept",
                ],
                Some(BackendId::OnnxNer),
            ),
            (
                "relationship_query",
                &[
                    "how are X and Y related",
                    "what connects A to B",
                    "find path between nodes",
                    "what is the link",
                ],
                Some(BackendId::VsaDirect),
            ),
            (
                "complex_reasoning",
                &[
                    "if A causes B and B prevents C",
                    "why does X lead to Y given Z",
                    "explain the relationship between cause and effect",
                ],
                Some(BackendId::CandleLlm),
            ),
            (
                "code_request",
                &[
                    "write a function that",
                    "implement an algorithm for",
                    "generate code for the following",
                    "create a program that",
                ],
                Some(BackendId::BurnLocal),
            ),
            (
                "meta_command",
                &[
                    "show my goals",
                    "list triples",
                    "display status",
                    "what do you know about this",
                    "summarise",
                ],
                Some(BackendId::RuleParser),
            ),
            (
                "temporal_query",
                &[
                    "what happened after the event",
                    "when did X occur",
                    "before the deadline",
                    "during the period",
                ],
                Some(BackendId::RuleParser),
            ),
        ];

        let mut archetypes = Vec::with_capacity(SEEDS.len());
        for (name, examples, preferred) in SEEDS {
            let vecs: Vec<HyperVec> = examples
                .iter()
                .filter_map(|s| encode_label(ops, s).ok())
                .collect();
            if vecs.is_empty() {
                continue;
            }
            let refs: Vec<&HyperVec> = vecs.iter().collect();
            if let Ok(prototype) = ops.bundle(&refs) {
                archetypes.push(QueryArchetype {
                    name,
                    prototype,
                    preferred_backend: *preferred,
                });
            }
        }
        archetypes
    }

    /// Extract the 8-dimensional feature vector for a query.
    ///
    /// Each dimension i is the VSA similarity between the encoded query and
    /// archetype i's prototype.  This grounds the bandit context in semantic
    /// query type rather than raw text statistics.
    fn extract_features(&self, query_vec: &HyperVec) -> FeatureVec {
        let mut features = [0.0f32; D];
        for (i, archetype) in self.archetypes.iter().enumerate() {
            if i >= D {
                break;
            }
            features[i] = self
                .ops
                .similarity(query_vec, &archetype.prototype)
                .unwrap_or(0.0)
                .max(0.0);
        }
        features
    }

    /// Look up `input` in the semantic cache.
    ///
    /// Returns the cached result if it is still live (within TTL and
    /// similarity ≥ 0.95).  Returns `None` on miss.
    pub fn find_cached(&mut self, input: &str) -> Option<NluParseResult> {
        let query_vec = encode_label(&self.ops, input).ok()?;
        self.cache.find(input, &query_vec, &self.ops)
    }

    /// Ask the bandit which backend should handle `input`.
    ///
    /// Returns a [`RouteDecision`] that the caller must pass to
    /// [`report_reward`] after the backend completes.
    ///
    /// # Errors
    ///
    /// Returns [`RouterError::VsaEncode`] if the input cannot be encoded
    /// (e.g., the input is empty or all-whitespace).
    pub fn route(&mut self, input: &str) -> RouterResult<RouteDecision> {
        let query_vec = encode_label(&self.ops, input).map_err(|e| RouterError::VsaEncode {
            reason: e.to_string(),
        })?;

        let features = self.extract_features(&query_vec);
        let (arm_idx, backend) = self.bandit.select(&features);

        tracing::debug!(
            router = "bandit",
            backend = %backend,
            arm_idx,
            "n akh bandit selected backend"
        );

        Ok(RouteDecision {
            backend,
            arm_idx,
            feature_vec: features,
        })
    }

    /// Report outcome of a routing decision back to the bandit.
    ///
    /// The reward is the result's `confidence` score (in [0, 1]).  Successful
    /// results are cached so near-identical future queries skip inference.
    ///
    /// Call with `reward_override = Some(0.0)` to penalise a backend that
    /// required cascade fallback.
    pub fn report_reward(
        &mut self,
        decision: &RouteDecision,
        result: &NluParseResult,
        input: &str,
        reward_override: Option<f32>,
    ) {
        let reward = reward_override.unwrap_or(result.confidence);
        self.bandit
            .update(decision.arm_idx, &decision.feature_vec, reward);

        // Cache results with reasonable confidence.
        if result.confidence >= 0.5 {
            if let Ok(query_vec) = encode_label(&self.ops, input) {
                self.cache.insert(input, query_vec, result.clone());
            }
        }
    }

    /// Bump the exploration parameter (Phase 27 integration hook).
    ///
    /// Call when model weights are hot-swapped so the bandit re-explores
    /// the `BurnLocal` arm's updated capabilities.
    pub fn on_model_weights_updated(&mut self, delta: f32) {
        self.bandit.bump_exploration(delta);
        tracing::info!(
            alpha = self.bandit.alpha,
            delta,
            "n akh: exploration bumped after model weight hot-swap"
        );
    }

    /// Gradually decay exploration toward `base` by `step` per call.
    pub fn decay_exploration(&mut self, base: f32, step: f32) {
        self.bandit.decay_exploration(base, step);
    }

    /// Invalidate the semantic cache.
    ///
    /// Call after knowledge-graph mutations to prevent stale results.
    pub fn invalidate_cache(&mut self) {
        self.cache.invalidate_all();
    }

    /// Serialize bandit state for persistence (redb / file).
    pub fn bandit_to_bytes(&self) -> Vec<u8> {
        self.bandit.to_bytes()
    }

    /// Restore bandit state from persisted bytes, replacing current state.
    pub fn restore_bandit(&mut self, data: &[u8]) {
        if let Some(bandit) = LinUcbBandit::from_bytes(data) {
            self.bandit = bandit;
        }
    }

    /// Minimum confidence threshold below which routing falls back to cascade.
    pub fn min_confidence(&self) -> f32 {
        self.config.min_confidence
    }

    /// Number of live entries in the semantic cache.
    pub fn cache_live_count(&self) -> usize {
        self.cache.live_count()
    }

    /// Current exploration parameter α.
    pub fn alpha(&self) -> f32 {
        self.bandit.alpha
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vsa::{Dimension, Encoding};

    fn make_router() -> NakhRouter {
        let ops = Arc::new(VsaOps::new(
            crate::simd::best_kernel(),
            Dimension::TEST,
            Encoding::Bipolar,
        ));
        NakhRouter::new(ops, RouterConfig::default())
    }

    #[test]
    fn router_selects_a_backend() {
        let mut router = make_router();
        let decision = router.route("hello").expect("should route");
        assert!(
            matches!(
                decision.backend,
                BackendId::RuleParser
                    | BackendId::OnnxNer
                    | BackendId::CandleLlm
                    | BackendId::VsaDirect
                    | BackendId::BurnLocal
            ),
            "backend should be one of the enabled variants"
        );
    }

    #[test]
    fn cache_returns_none_before_insert() {
        let mut router = make_router();
        assert!(router.find_cached("novel query xyz").is_none());
    }

    #[test]
    fn cache_hit_after_report_reward() {
        use crate::grammar::abs::AbsTree;

        let mut router = make_router();
        let input = "dogs are mammals";
        let decision = router.route(input).expect("should route");
        let result = NluParseResult {
            tree: AbsTree::entity("test"),
            source_tier: 1,
            confidence: 0.85,
            exemplar_similarity: None,
            neural_concepts: vec![],
        };
        router.report_reward(&decision, &result, input, None);

        // Cache should now hold this result
        let cached = router.find_cached(input);
        assert!(
            cached.is_some(),
            "cache should hold the result after report_reward"
        );
    }

    #[test]
    fn exploration_bump_and_decay() {
        let mut router = make_router();
        let base_alpha = router.alpha();
        router.on_model_weights_updated(0.5);
        assert!(router.alpha() > base_alpha);
        router.decay_exploration(base_alpha, 10.0);
        assert_eq!(router.alpha(), base_alpha);
    }

    #[test]
    fn archetype_bootstrap_produces_d_archetypes() {
        let router = make_router();
        assert_eq!(
            router.archetypes.len(),
            D,
            "should bootstrap exactly D={D} archetypes"
        );
    }

    #[test]
    fn invalidate_clears_cache() {
        use crate::grammar::abs::AbsTree;

        let mut router = make_router();
        let input = "water is wet";
        let decision = router.route(input).expect("should route");
        let result = NluParseResult {
            tree: AbsTree::entity("water"),
            source_tier: 1,
            confidence: 0.90,
            exemplar_similarity: None,
            neural_concepts: vec![],
        };
        router.report_reward(&decision, &result, input, None);
        assert!(router.find_cached(input).is_some());

        router.invalidate_cache();
        assert!(
            router.find_cached(input).is_none(),
            "cache should be empty after invalidation"
        );
    }
}
