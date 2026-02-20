//! Interference-based inference engine.
//!
//! Uses VSA constructive/destructive interference patterns to infer new
//! knowledge from existing symbol associations. The engine combines
//! graph-guided spreading activation with VSA bind/unbind recovery
//! and similarity-based cleanup to discover implicit knowledge.

pub mod backward;
pub mod engine;
pub mod superposition;

// Re-export Phase 9 inference context for external use.
pub use engine::InferPhase9Context;

use crate::symbol::SymbolId;
use crate::vsa::HyperVec;

// Re-export unified provenance types so downstream code can use them
// through `crate::infer::DerivationKind` / `crate::infer::ProvenanceRecord`.
pub use crate::provenance::{DerivationKind, ProvenanceRecord};

/// Query for the inference engine.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InferenceQuery {
    /// Seed symbols to start inference from.
    pub seeds: Vec<SymbolId>,
    /// Maximum number of results.
    pub top_k: usize,
    /// Maximum inference depth (for multi-step reasoning).
    pub max_depth: usize,
    /// Minimum confidence threshold for activations (default: 0.1).
    pub min_confidence: f32,
    /// Minimum VSA similarity for cleanup recovery (default: 0.6).
    pub min_similarity: f32,
    /// Whether to verify inferences with the e-graph engine.
    pub verify_with_egraph: bool,
    /// Optional predicate filter — only follow edges with these predicates.
    pub predicate_filter: Option<Vec<SymbolId>>,
}

impl Default for InferenceQuery {
    fn default() -> Self {
        Self {
            seeds: Vec::new(),
            top_k: 10,
            max_depth: 1,
            min_confidence: 0.1,
            min_similarity: 0.6,
            verify_with_egraph: false,
            predicate_filter: None,
        }
    }
}

impl InferenceQuery {
    /// Set seed symbols.
    pub fn with_seeds(mut self, seeds: Vec<SymbolId>) -> Self {
        self.seeds = seeds;
        self
    }

    /// Set maximum inference depth.
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Enable e-graph verification of inferred results.
    pub fn with_egraph_verification(mut self) -> Self {
        self.verify_with_egraph = true;
        self
    }

    /// Set minimum confidence threshold.
    pub fn with_min_confidence(mut self, min_confidence: f32) -> Self {
        self.min_confidence = min_confidence;
        self
    }

    /// Set minimum VSA similarity threshold.
    pub fn with_min_similarity(mut self, min_similarity: f32) -> Self {
        self.min_similarity = min_similarity;
        self
    }

    /// Set predicate filter.
    pub fn with_predicate_filter(mut self, predicates: Vec<SymbolId>) -> Self {
        self.predicate_filter = Some(predicates);
        self
    }
}

/// Result of an inference query.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InferenceResult {
    /// Activated symbols with their confidence scores.
    pub activations: Vec<(SymbolId, f32)>,
    /// The interference pattern (combined hypervector).
    pub pattern: Option<HyperVec>,
    /// Provenance records explaining how each activation was derived.
    pub provenance: Vec<ProvenanceRecord>,
}
