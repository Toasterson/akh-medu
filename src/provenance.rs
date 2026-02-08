//! Provenance ledger: tracks how knowledge was derived.
//!
//! Every inference, reasoning step, or extraction produces a provenance record
//! that links the derived knowledge back to its sources.
//! Full implementation in Phase 2.

use serde::{Deserialize, Serialize};

use crate::symbol::SymbolId;

/// How a piece of knowledge was derived.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DerivationKind {
    /// Directly extracted from source material.
    Extracted,
    /// Inferred via VSA interference patterns.
    Inferred,
    /// Derived via symbolic reasoning (e-graph rewriting).
    Reasoned,
    /// Aggregated from multiple sources.
    Aggregated,
}

/// A single provenance record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceRecord {
    /// ID of the derived symbol or triple.
    pub derived_id: SymbolId,
    /// IDs of the source symbols/triples.
    pub sources: Vec<SymbolId>,
    /// How this was derived.
    pub kind: DerivationKind,
    /// Timestamp (seconds since UNIX epoch).
    pub timestamp: u64,
    /// Confidence in the derivation.
    pub confidence: f32,
}
