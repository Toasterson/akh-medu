//! In-memory knowledge graph with dual-indexing.
//!
//! Uses `petgraph` for the graph structure and `DashMap` for fast lookups
//! by subject, predicate, or object.

use std::sync::RwLock;

use dashmap::DashMap;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;

use crate::error::GraphError;
use crate::symbol::SymbolId;

use super::{EdgeData, Triple};

/// Result type for graph operations.
pub type GraphResult<T> = std::result::Result<T, GraphError>;

/// In-memory knowledge graph backed by petgraph with dual-indexing.
///
/// Provides O(1) node lookups by SymbolId and fast predicate-based queries
/// through a secondary index.
pub struct KnowledgeGraph {
    /// The directed graph: nodes are SymbolIds, edges carry EdgeData.
    graph: RwLock<DiGraph<SymbolId, EdgeData>>,
    /// SymbolId → NodeIndex mapping for O(1) node lookups.
    node_index: DashMap<SymbolId, NodeIndex>,
    /// Predicate index: predicate SymbolId → list of (subject, object) pairs.
    predicate_index: DashMap<SymbolId, Vec<(SymbolId, SymbolId)>>,
    /// Triple count.
    triple_count: std::sync::atomic::AtomicUsize,
}

impl KnowledgeGraph {
    /// Create a new empty knowledge graph.
    pub fn new() -> Self {
        Self {
            graph: RwLock::new(DiGraph::new()),
            node_index: DashMap::new(),
            predicate_index: DashMap::new(),
            triple_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Ensure a node exists for the given symbol, returning its NodeIndex.
    fn ensure_node(&self, symbol: SymbolId) -> NodeIndex {
        if let Some(idx) = self.node_index.get(&symbol) {
            return *idx.value();
        }
        let mut graph = self.graph.write().expect("graph lock poisoned");
        // Double-check after acquiring write lock
        if let Some(idx) = self.node_index.get(&symbol) {
            return *idx.value();
        }
        let idx = graph.add_node(symbol);
        self.node_index.insert(symbol, idx);
        idx
    }

    /// Insert a triple into the graph.
    ///
    /// Creates nodes for subject and object if they don't exist.
    /// Adds an edge from subject → object with the predicate as edge data.
    pub fn insert_triple(&self, triple: &Triple) -> GraphResult<()> {
        let subj_idx = self.ensure_node(triple.subject);
        let obj_idx = self.ensure_node(triple.object);
        self.ensure_node(triple.predicate); // ensure predicate is also a node

        let edge_data = EdgeData::from(triple);

        {
            let mut graph = self.graph.write().expect("graph lock poisoned");
            graph.add_edge(subj_idx, obj_idx, edge_data);
        }

        // Update predicate index
        self.predicate_index
            .entry(triple.predicate)
            .or_default()
            .push((triple.subject, triple.object));

        self.triple_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(())
    }

    /// Get all objects for a given subject and predicate.
    pub fn objects_of(&self, subject: SymbolId, predicate: SymbolId) -> Vec<SymbolId> {
        let graph = self.graph.read().expect("graph lock poisoned");
        let subj_idx = match self.node_index.get(&subject) {
            Some(idx) => *idx.value(),
            None => return vec![],
        };

        graph
            .edges_directed(subj_idx, Direction::Outgoing)
            .filter(|e| e.weight().predicate == predicate)
            .filter_map(|e| {
                let target = e.target();
                graph.node_weight(target).copied()
            })
            .collect()
    }

    /// Get all subjects for a given predicate and object.
    pub fn subjects_of(&self, predicate: SymbolId, object: SymbolId) -> Vec<SymbolId> {
        let graph = self.graph.read().expect("graph lock poisoned");
        let obj_idx = match self.node_index.get(&object) {
            Some(idx) => *idx.value(),
            None => return vec![],
        };

        graph
            .edges_directed(obj_idx, Direction::Incoming)
            .filter(|e| e.weight().predicate == predicate)
            .filter_map(|e| {
                let source = e.source();
                graph.node_weight(source).copied()
            })
            .collect()
    }

    /// Get all triples involving a given predicate.
    pub fn triples_for_predicate(&self, predicate: SymbolId) -> Vec<(SymbolId, SymbolId)> {
        self.predicate_index
            .get(&predicate)
            .map(|v| v.value().clone())
            .unwrap_or_default()
    }

    /// Get all triples where the given symbol appears as subject.
    pub fn triples_from(&self, subject: SymbolId) -> Vec<Triple> {
        let graph = self.graph.read().expect("graph lock poisoned");
        let subj_idx = match self.node_index.get(&subject) {
            Some(idx) => *idx.value(),
            None => return vec![],
        };

        graph
            .edges_directed(subj_idx, Direction::Outgoing)
            .filter_map(|e| {
                let object = *graph.node_weight(e.target())?;
                let edge = e.weight();
                Some(Triple {
                    subject,
                    predicate: edge.predicate,
                    object,
                    confidence: edge.confidence,
                    timestamp: edge.timestamp,
                    provenance_id: edge.provenance_id,
                })
            })
            .collect()
    }

    /// Get all triples where the given symbol appears as object (incoming edges).
    pub fn triples_to(&self, object: SymbolId) -> Vec<Triple> {
        let graph = self.graph.read().expect("graph lock poisoned");
        let obj_idx = match self.node_index.get(&object) {
            Some(idx) => *idx.value(),
            None => return vec![],
        };

        graph
            .edges_directed(obj_idx, Direction::Incoming)
            .filter_map(|e| {
                let subject = *graph.node_weight(e.source())?;
                let edge = e.weight();
                Some(Triple {
                    subject,
                    predicate: edge.predicate,
                    object,
                    confidence: edge.confidence,
                    timestamp: edge.timestamp,
                    provenance_id: edge.provenance_id,
                })
            })
            .collect()
    }

    /// Check if a node exists.
    pub fn has_node(&self, symbol: SymbolId) -> bool {
        self.node_index.contains_key(&symbol)
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.node_index.len()
    }

    /// Number of triples (edges).
    pub fn triple_count(&self) -> usize {
        self.triple_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get all node symbols.
    pub fn nodes(&self) -> Vec<SymbolId> {
        self.node_index.iter().map(|e| *e.key()).collect()
    }

    /// Get all predicate symbols.
    pub fn predicates(&self) -> Vec<SymbolId> {
        self.predicate_index.iter().map(|e| *e.key()).collect()
    }

    /// Bulk-load triples into the graph. Used for restoring from persistent storage.
    /// Returns the number of triples successfully loaded.
    pub fn bulk_load(&self, triples: &[Triple]) -> GraphResult<usize> {
        let mut count = 0;
        for triple in triples {
            self.insert_triple(triple)?;
            count += 1;
        }
        Ok(count)
    }

    /// Get all triples in the graph.
    pub fn all_triples(&self) -> Vec<Triple> {
        let graph = self.graph.read().expect("graph lock poisoned");
        graph
            .edge_indices()
            .filter_map(|ei| {
                let (src, dst) = graph.edge_endpoints(ei)?;
                let subject = *graph.node_weight(src)?;
                let object = *graph.node_weight(dst)?;
                let edge = graph.edge_weight(ei)?;
                Some(Triple {
                    subject,
                    predicate: edge.predicate,
                    object,
                    confidence: edge.confidence,
                    timestamp: edge.timestamp,
                    provenance_id: edge.provenance_id,
                })
            })
            .collect()
    }
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for KnowledgeGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KnowledgeGraph")
            .field("nodes", &self.node_count())
            .field("triples", &self.triple_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(id: u64) -> SymbolId {
        SymbolId::new(id).unwrap()
    }

    #[test]
    fn insert_and_query() {
        let kg = KnowledgeGraph::new();
        let sun = sym(1);
        let is_a = sym(2);
        let star = sym(3);

        kg.insert_triple(&Triple::new(sun, is_a, star)).unwrap();

        assert!(kg.has_node(sun));
        assert!(kg.has_node(star));
        assert_eq!(kg.node_count(), 3); // sun, is_a, star
        assert_eq!(kg.triple_count(), 1);

        let objects = kg.objects_of(sun, is_a);
        assert_eq!(objects, vec![star]);

        let subjects = kg.subjects_of(is_a, star);
        assert_eq!(subjects, vec![sun]);
    }

    #[test]
    fn predicate_index() {
        let kg = KnowledgeGraph::new();
        let a = sym(1);
        let b = sym(2);
        let c = sym(3);
        let rel = sym(10);

        kg.insert_triple(&Triple::new(a, rel, b)).unwrap();
        kg.insert_triple(&Triple::new(a, rel, c)).unwrap();

        let pairs = kg.triples_for_predicate(rel);
        assert_eq!(pairs.len(), 2);
    }

    #[test]
    fn triples_from() {
        let kg = KnowledgeGraph::new();
        let a = sym(1);
        let b = sym(2);
        let c = sym(3);
        let r1 = sym(10);
        let r2 = sym(11);

        kg.insert_triple(&Triple::new(a, r1, b)).unwrap();
        kg.insert_triple(&Triple::new(a, r2, c)).unwrap();

        let triples = kg.triples_from(a);
        assert_eq!(triples.len(), 2);
    }

    #[test]
    fn all_triples() {
        let kg = KnowledgeGraph::new();
        let a = sym(1);
        let b = sym(2);
        let c = sym(3);
        let r = sym(10);

        kg.insert_triple(&Triple::new(a, r, b)).unwrap();
        kg.insert_triple(&Triple::new(b, r, c)).unwrap();

        let all = kg.all_triples();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn confidence() {
        let kg = KnowledgeGraph::new();
        let t = Triple::new(sym(1), sym(2), sym(3)).with_confidence(0.8);
        kg.insert_triple(&t).unwrap();

        let triples = kg.triples_from(sym(1));
        assert_eq!(triples.len(), 1);
        assert!((triples[0].confidence - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn triples_to() {
        let kg = KnowledgeGraph::new();
        let a = sym(1);
        let b = sym(2);
        let c = sym(3);
        let r1 = sym(10);
        let r2 = sym(11);

        kg.insert_triple(&Triple::new(a, r1, c)).unwrap();
        kg.insert_triple(&Triple::new(b, r2, c)).unwrap();

        let triples = kg.triples_to(c);
        assert_eq!(triples.len(), 2);
        assert!(triples.iter().all(|t| t.object == c));
    }

    #[test]
    fn empty_queries() {
        let kg = KnowledgeGraph::new();
        assert!(kg.objects_of(sym(1), sym(2)).is_empty());
        assert!(kg.subjects_of(sym(1), sym(2)).is_empty());
        assert!(kg.triples_from(sym(1)).is_empty());
    }
}
