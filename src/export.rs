//! Export types for serializing engine state.
//!
//! These types provide human-readable, label-resolved representations
//! of symbols, triples, and provenance chains suitable for JSON export.

use serde::{Deserialize, Serialize};

/// Exported symbol with resolved label.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolExport {
    /// Numeric symbol ID.
    pub id: u64,
    /// Human-readable label.
    pub label: String,
    /// Symbol kind (Entity, Relation, Composite, Glyph).
    pub kind: String,
    /// Creation timestamp (seconds since UNIX epoch).
    pub created_at: u64,
}

/// Exported triple with resolved labels for all positions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleExport {
    /// Subject symbol ID.
    pub subject_id: u64,
    /// Subject label.
    pub subject_label: String,
    /// Predicate symbol ID.
    pub predicate_id: u64,
    /// Predicate label.
    pub predicate_label: String,
    /// Object symbol ID.
    pub object_id: u64,
    /// Object label.
    pub object_label: String,
    /// Confidence score.
    pub confidence: f32,
}

/// Exported provenance record with resolved labels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceExport {
    /// Provenance record ID.
    pub id: u64,
    /// Derived symbol ID.
    pub derived_id: u64,
    /// Derived symbol label.
    pub derived_label: String,
    /// Derivation kind description.
    pub kind: String,
    /// Confidence score.
    pub confidence: f32,
    /// Inference depth.
    pub depth: usize,
    /// Source symbol IDs.
    pub sources: Vec<u64>,
}
