//! Interference-based inference engine.
//!
//! Uses VSA constructive/destructive interference patterns to infer new
//! knowledge from existing symbol associations. Full implementation in Phase 2.

use crate::symbol::SymbolId;
use crate::vsa::HyperVec;

/// Query for the inference engine.
#[derive(Debug, Clone)]
pub struct InferenceQuery {
    /// Seed symbols to start inference from.
    pub seeds: Vec<SymbolId>,
    /// Maximum number of results.
    pub top_k: usize,
    /// Maximum inference depth (for multi-step reasoning).
    pub max_depth: usize,
}

impl Default for InferenceQuery {
    fn default() -> Self {
        Self {
            seeds: Vec::new(),
            top_k: 10,
            max_depth: 1,
        }
    }
}

/// Result of an inference query.
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// Activated symbols with their similarity scores.
    pub activations: Vec<(SymbolId, f32)>,
    /// The interference pattern (combined hypervector).
    pub pattern: Option<HyperVec>,
}
