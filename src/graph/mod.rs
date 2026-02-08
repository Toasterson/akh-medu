//! Knowledge Graph: dual-indexed graph with in-memory and persistent layers.
//!
//! The knowledge graph stores triples (subject, predicate, object) with metadata.
//!
//! - **In-memory layer** ([`KnowledgeGraph`]): uses `petgraph` for fast traversals
//! - **Persistent layer** ([`SparqlStore`]): uses `oxigraph` for SPARQL queries and durability
//!
//! Both layers share the same [`Triple`] data model and can be synchronized.

pub mod analytics;
pub mod index;
pub mod sparql;
pub mod traverse;

use serde::{Deserialize, Serialize};

use crate::symbol::SymbolId;

/// A triple (subject, predicate, object) in the knowledge graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Triple {
    /// The subject of the triple.
    pub subject: SymbolId,
    /// The predicate (relation) of the triple.
    pub predicate: SymbolId,
    /// The object of the triple.
    pub object: SymbolId,
    /// Confidence score in [0.0, 1.0].
    pub confidence: f32,
    /// Timestamp (seconds since UNIX epoch).
    pub timestamp: u64,
    /// Optional provenance ID linking to the provenance ledger.
    pub provenance_id: Option<u64>,
}

impl Triple {
    /// Create a new triple with full confidence and current timestamp.
    pub fn new(subject: SymbolId, predicate: SymbolId, object: SymbolId) -> Self {
        Self {
            subject,
            predicate,
            object,
            confidence: 1.0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            provenance_id: None,
        }
    }

    /// Set the confidence score.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set the provenance ID.
    pub fn with_provenance(mut self, provenance_id: u64) -> Self {
        self.provenance_id = Some(provenance_id);
        self
    }
}

/// Edge data stored on petgraph edges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeData {
    /// The predicate symbol for this edge.
    pub predicate: SymbolId,
    /// Confidence in [0.0, 1.0].
    pub confidence: f32,
    /// Provenance link.
    pub provenance_id: Option<u64>,
    /// Timestamp.
    pub timestamp: u64,
}

impl From<&Triple> for EdgeData {
    fn from(t: &Triple) -> Self {
        Self {
            predicate: t.predicate,
            confidence: t.confidence,
            provenance_id: t.provenance_id,
            timestamp: t.timestamp,
        }
    }
}
