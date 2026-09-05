//! In-memory knowledge graph with dual-indexing.
//!
//! Uses `petgraph` for the graph structure and `DashMap` for fast lookups
//! by subject, predicate, or object.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::RwLock;

use dashmap::DashMap;
use petgraph::Direction;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;

use crate::error::GraphError;
use crate::symbol::SymbolId;

use super::{EdgeData, Triple};

type TripleKey = (SymbolId, SymbolId, SymbolId);

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
    /// Provenance ID → exact graph keys carrying that ledger link. Key zero
    /// is reserved for graph edges with no provenance link.
    provenance_index: DashMap<u64, BTreeSet<TripleKey>>,
    /// Ordered confidence bits → exact graph keys at that confidence.
    /// Valid non-negative finite `f32` values preserve numeric order in their
    /// IEEE bit representation, so range lookup is logarithmic plus matches.
    confidence_index: RwLock<BTreeMap<u32, BTreeSet<TripleKey>>>,
    /// Triple count.
    triple_count: std::sync::atomic::AtomicUsize,
    #[cfg(test)]
    all_triples_scans: std::sync::atomic::AtomicUsize,
}

impl KnowledgeGraph {
    /// Create a new empty knowledge graph.
    pub fn new() -> Self {
        Self {
            graph: RwLock::new(DiGraph::new()),
            node_index: DashMap::new(),
            predicate_index: DashMap::new(),
            provenance_index: DashMap::new(),
            confidence_index: RwLock::new(BTreeMap::new()),
            triple_count: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(test)]
            all_triples_scans: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn add_metadata_indices(&self, key: TripleKey, edge: &EdgeData) {
        self.provenance_index
            .entry(edge.provenance_id.unwrap_or(0))
            .or_default()
            .insert(key);
        if edge.confidence.is_finite() && (0.0..=1.0).contains(&edge.confidence) {
            let confidence = if edge.confidence == 0.0 {
                0.0_f32.to_bits()
            } else {
                edge.confidence.to_bits()
            };
            self.confidence_index
                .write()
                .expect("confidence index lock poisoned")
                .entry(confidence)
                .or_default()
                .insert(key);
        }
    }

    fn remove_metadata_indices(&self, key: TripleKey, edge: &EdgeData) {
        let provenance_id = edge.provenance_id.unwrap_or(0);
        if let Some(mut keys) = self.provenance_index.get_mut(&provenance_id) {
            keys.remove(&key);
        }
        if !edge.confidence.is_finite() || !(0.0..=1.0).contains(&edge.confidence) {
            return;
        }
        let confidence = if edge.confidence == 0.0 {
            0.0_f32.to_bits()
        } else {
            edge.confidence.to_bits()
        };
        let mut index = self
            .confidence_index
            .write()
            .expect("confidence index lock poisoned");
        if let Some(keys) = index.get_mut(&confidence) {
            keys.remove(&key);
            if keys.is_empty() {
                index.remove(&confidence);
            }
        }
    }

    fn replace_metadata_indices(&self, key: TripleKey, old: &EdgeData, new: &EdgeData) {
        if old.provenance_id != new.provenance_id || old.confidence != new.confidence {
            self.remove_metadata_indices(key, old);
            self.add_metadata_indices(key, new);
        }
    }

    fn triples_for_keys(&self, keys: impl IntoIterator<Item = TripleKey>) -> Vec<Triple> {
        let graph = self.graph.read().expect("graph lock poisoned");
        keys.into_iter()
            .filter_map(|(subject, predicate, object)| {
                let subj_idx = *self.node_index.get(&subject)?.value();
                let obj_idx = *self.node_index.get(&object)?.value();
                let edge = graph
                    .edges_directed(subj_idx, Direction::Outgoing)
                    .find(|edge| edge.target() == obj_idx && edge.weight().predicate == predicate)?
                    .weight();
                Some(Triple {
                    subject,
                    predicate,
                    object,
                    confidence: edge.confidence,
                    timestamp: edge.timestamp,
                    provenance_id: edge.provenance_id,
                    compartment_id: edge.compartment_id.clone(),
                })
            })
            .collect()
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
    ///
    /// **Idempotent**: if an edge with the same (subject, predicate, object)
    /// already exists, updates its confidence (max of old and new) and
    /// provenance instead of creating a duplicate edge.
    pub fn insert_triple(&self, triple: &Triple) -> GraphResult<()> {
        let subj_idx = self.ensure_node(triple.subject);
        let obj_idx = self.ensure_node(triple.object);
        self.ensure_node(triple.predicate); // ensure predicate is also a node

        let edge_data = EdgeData::from(triple);

        let mut graph = self.graph.write().expect("graph lock poisoned");
        let existing = graph
            .edges_directed(subj_idx, Direction::Outgoing)
            .find(|e| e.target() == obj_idx && e.weight().predicate == triple.predicate)
            .map(|e| e.id());
        let key = (triple.subject, triple.predicate, triple.object);
        if let Some(edge_id) = existing {
            if let Some(weight) = graph.edge_weight_mut(edge_id) {
                let old = weight.clone();
                weight.confidence = weight.confidence.max(edge_data.confidence);
                if edge_data.provenance_id.is_some() {
                    weight.provenance_id = edge_data.provenance_id;
                }
                weight.timestamp = edge_data.timestamp;
                self.replace_metadata_indices(key, &old, weight);
            }
            return Ok(());
        }

        graph.add_edge(subj_idx, obj_idx, edge_data.clone());
        // Update predicate index
        self.predicate_index
            .entry(triple.predicate)
            .or_default()
            .push((triple.subject, triple.object));
        self.add_metadata_indices(key, &edge_data);

        self.triple_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(())
    }

    /// Insert or exactly replace a triple's complete edge metadata.
    ///
    /// Normal live assertions merge confidence and retain an old provenance
    /// link when the new assertion is untracked. WAL replay needs replacement
    /// semantics instead: the later assertion is authoritative, including an
    /// explicit `None` provenance value, otherwise a reasserted fact can wake
    /// falsely attributed to its pre-snapshot source.
    pub(crate) fn replace_triple_exact(&self, triple: &Triple) -> GraphResult<()> {
        let subj_idx = self.ensure_node(triple.subject);
        let obj_idx = self.ensure_node(triple.object);
        self.ensure_node(triple.predicate);
        let edge_data = EdgeData::from(triple);

        let mut graph = self.graph.write().expect("graph lock poisoned");
        let existing = graph
            .edges_directed(subj_idx, Direction::Outgoing)
            .find(|edge| edge.target() == obj_idx && edge.weight().predicate == triple.predicate)
            .map(|edge| edge.id());
        let key = (triple.subject, triple.predicate, triple.object);
        if let Some(edge_id) = existing {
            if let Some(weight) = graph.edge_weight_mut(edge_id) {
                let old = weight.clone();
                *weight = edge_data.clone();
                self.replace_metadata_indices(key, &old, weight);
            }
            return Ok(());
        }

        graph.add_edge(subj_idx, obj_idx, edge_data.clone());
        self.predicate_index
            .entry(triple.predicate)
            .or_default()
            .push((triple.subject, triple.object));
        self.add_metadata_indices(key, &edge_data);
        self.triple_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Remove a triple from the graph.
    ///
    /// Removes the edge between subject and object with the matching predicate.
    /// Returns true if a matching edge was found and removed, false otherwise.
    /// Does NOT remove nodes (they may be referenced by other edges).
    pub fn remove_triple(&self, subject: SymbolId, predicate: SymbolId, object: SymbolId) -> bool {
        let subj_idx = match self.node_index.get(&subject) {
            Some(idx) => *idx.value(),
            None => return false,
        };
        let obj_idx = match self.node_index.get(&object) {
            Some(idx) => *idx.value(),
            None => return false,
        };

        let mut graph = self.graph.write().expect("graph lock poisoned");
        let edge_to_remove = graph
            .edges_directed(subj_idx, Direction::Outgoing)
            .find(|e| e.target() == obj_idx && e.weight().predicate == predicate)
            .map(|e| e.id());
        let removed = edge_to_remove.and_then(|edge_id| graph.remove_edge(edge_id));

        if let Some(edge) = removed {
            let key = (subject, predicate, object);

            // Update predicate index
            if let Some(mut pairs) = self.predicate_index.get_mut(&predicate) {
                pairs.retain(|&(s, o)| !(s == subject && o == object));
            }
            self.remove_metadata_indices(key, &edge);

            self.triple_count
                .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Return the exact stored edge, including all volatile metadata.
    ///
    /// Live assertions merge duplicate edges, so callers that publish a WAL
    /// image must read back this value rather than serialising the incoming
    /// assertion verbatim.
    pub fn triple(
        &self,
        subject: SymbolId,
        predicate: SymbolId,
        object: SymbolId,
    ) -> Option<Triple> {
        let subj_idx = *self.node_index.get(&subject)?.value();
        let obj_idx = *self.node_index.get(&object)?.value();
        let graph = self.graph.read().expect("graph lock poisoned");
        let edge = graph
            .edges_directed(subj_idx, Direction::Outgoing)
            .find(|edge| edge.target() == obj_idx && edge.weight().predicate == predicate)?;
        let data = edge.weight();
        Some(Triple {
            subject,
            predicate,
            object,
            confidence: data.confidence,
            timestamp: data.timestamp,
            provenance_id: data.provenance_id,
            compartment_id: data.compartment_id.clone(),
        })
    }

    /// Remove all triples in a given compartment.
    ///
    /// Iterates all edges, collects those whose `compartment_id` matches, then
    /// removes them in descending index order to avoid index invalidation from
    /// petgraph's swap-remove.
    ///
    /// Returns the number of triples removed.
    pub fn remove_triples_in_compartment(&self, compartment_id: &str) -> usize {
        let mut graph = self.graph.write().expect("graph lock poisoned");
        let mut to_remove: Vec<_> = graph
            .edge_indices()
            .filter(|&edge_id| {
                graph
                    .edge_weight(edge_id)
                    .is_some_and(|edge| edge.compartment_id.as_deref() == Some(compartment_id))
            })
            .collect();
        to_remove.sort_by(|left, right| right.cmp(left));

        let mut removed = 0;
        for edge_id in to_remove {
            if let Some(edge) = graph.edge_weight(edge_id).cloned()
                && let Some((src, dst)) = graph.edge_endpoints(edge_id)
                && let (Some(subject), Some(object)) = (
                    graph.node_weight(src).copied(),
                    graph.node_weight(dst).copied(),
                )
            {
                graph.remove_edge(edge_id);
                let predicate = edge.predicate;
                if let Some(mut pairs) = self.predicate_index.get_mut(&predicate) {
                    pairs.retain(|&(s, o)| !(s == subject && o == object));
                }
                self.remove_metadata_indices((subject, predicate, object), &edge);
                removed += 1;
            }
        }

        if removed > 0 {
            self.triple_count
                .fetch_sub(removed, std::sync::atomic::Ordering::Relaxed);
        }

        removed
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

    /// Materialize exact triples for one predicate without scanning all edges.
    pub fn triples_with_predicate(&self, predicate: SymbolId) -> Vec<Triple> {
        let keys = self
            .predicate_index
            .get(&predicate)
            .map(|pairs| {
                pairs
                    .iter()
                    .map(|&(subject, object)| (subject, predicate, object))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        self.triples_for_keys(keys)
    }

    /// Materialize triples linked to any requested provenance record.
    pub fn triples_with_provenance_ids(&self, ids: &HashSet<u64>) -> Vec<Triple> {
        let mut keys = BTreeSet::new();
        for id in ids {
            if let Some(indexed) = self.provenance_index.get(id) {
                keys.extend(indexed.iter().copied());
            }
        }
        self.triples_for_keys(keys)
    }

    /// Materialize triples with no ledger link via the reserved index bucket.
    pub fn triples_without_provenance(&self) -> Vec<Triple> {
        let keys = self
            .provenance_index
            .get(&0)
            .map(|indexed| indexed.iter().copied().collect::<Vec<_>>())
            .unwrap_or_default();
        self.triples_for_keys(keys)
    }

    /// Materialize triples at or above `minimum` via the ordered confidence
    /// index. Invalid or out-of-range thresholds match nothing.
    pub fn triples_at_or_above_confidence(&self, minimum: f32) -> Vec<Triple> {
        if !minimum.is_finite() || !(0.0..=1.0).contains(&minimum) {
            return Vec::new();
        }
        let keys = {
            let index = self
                .confidence_index
                .read()
                .expect("confidence index lock poisoned");
            let minimum_bits = if minimum == 0.0 {
                0.0_f32.to_bits()
            } else {
                minimum.to_bits()
            };
            index
                .range(minimum_bits..)
                .flat_map(|(_, keys)| keys.iter().copied())
                .collect::<BTreeSet<_>>()
        };
        self.triples_for_keys(keys)
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
                    compartment_id: edge.compartment_id.clone(),
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
                    compartment_id: edge.compartment_id.clone(),
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
        self.triple_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get all node symbols.
    pub fn nodes(&self) -> Vec<SymbolId> {
        self.node_index.iter().map(|e| *e.key()).collect()
    }

    /// Get all predicate symbols.
    pub fn predicates(&self) -> Vec<SymbolId> {
        self.predicate_index.iter().map(|e| *e.key()).collect()
    }

    /// Acquire a read lock on the inner graph for analytics.
    pub fn graph(&self) -> std::sync::RwLockReadGuard<'_, DiGraph<SymbolId, EdgeData>> {
        self.graph.read().expect("graph lock poisoned")
    }

    /// Build a reverse lookup: NodeIndex -> SymbolId.
    pub fn node_index_to_symbol(&self) -> HashMap<NodeIndex, SymbolId> {
        self.node_index
            .iter()
            .map(|e| (*e.value(), *e.key()))
            .collect()
    }

    /// Bulk-load triples into the graph. Used for restoring from persistent storage.
    /// Returns the number of triples successfully loaded.
    ///
    /// Holds the write lock for the entire batch to avoid per-triple lock overhead.
    ///
    /// Idempotent: duplicate `(subject, predicate, object)` edges are skipped, so
    /// waking from a snapshot twice (or loading data that already exists in-memory)
    /// does not create parallel edges. This keeps `all_triples()` output stable and
    /// avoids unique-key violations when the snapshot writer re-persists the same
    /// fact set to a table whose primary key is the triple itself.
    pub fn bulk_load(&self, triples: &[Triple]) -> GraphResult<usize> {
        let mut graph = self.graph.write().expect("graph lock poisoned");
        let mut count = 0;
        for triple in triples {
            let subj_idx = *self
                .node_index
                .entry(triple.subject)
                .or_insert_with(|| graph.add_node(triple.subject));
            let obj_idx = *self
                .node_index
                .entry(triple.object)
                .or_insert_with(|| graph.add_node(triple.object));
            self.node_index
                .entry(triple.predicate)
                .or_insert_with(|| graph.add_node(triple.predicate));

            let already_exists = graph
                .edges_directed(subj_idx, Direction::Outgoing)
                .any(|e| e.target() == obj_idx && e.weight().predicate == triple.predicate);
            if already_exists {
                continue;
            }

            let edge = EdgeData::from(triple);
            graph.add_edge(subj_idx, obj_idx, edge.clone());
            self.predicate_index
                .entry(triple.predicate)
                .or_default()
                .push((triple.subject, triple.object));
            self.add_metadata_indices((triple.subject, triple.predicate, triple.object), &edge);
            count += 1;
        }
        self.triple_count
            .fetch_add(count, std::sync::atomic::Ordering::Relaxed);
        Ok(count)
    }

    /// Get all triples in the graph.
    pub fn all_triples(&self) -> Vec<Triple> {
        #[cfg(test)]
        self.all_triples_scans
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
                    compartment_id: edge.compartment_id.clone(),
                })
            })
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn reset_all_triples_scan_count(&self) {
        self.all_triples_scans
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(crate) fn all_triples_scan_count(&self) -> usize {
        self.all_triples_scans
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Adjust the confidence of an existing triple by `delta`, clamped to `[min_conf, max_conf]`.
    ///
    /// Unlike `insert_triple` (which always takes the max of old and new confidence),
    /// this method can both increase and decrease confidence — required for RL updates
    /// where unsuccessful actions must reduce confidence.
    ///
    /// Returns `true` if the triple was found and updated, `false` if not present.
    pub fn update_triple_confidence(
        &self,
        subject: SymbolId,
        predicate: SymbolId,
        object: SymbolId,
        delta: f32,
        min_conf: f32,
        max_conf: f32,
    ) -> bool {
        let subj_idx = match self.node_index.get(&subject) {
            Some(idx) => *idx.value(),
            None => return false,
        };
        let obj_idx = match self.node_index.get(&object) {
            Some(idx) => *idx.value(),
            None => return false,
        };

        let mut graph = self.graph.write().expect("graph lock poisoned");
        let edge_id = graph
            .edges_directed(subj_idx, Direction::Outgoing)
            .find(|e| e.target() == obj_idx && e.weight().predicate == predicate)
            .map(|e| e.id());

        if let Some(weight) = edge_id.and_then(|edge_id| graph.edge_weight_mut(edge_id)) {
            let old = weight.clone();
            weight.confidence = (weight.confidence + delta).clamp(min_conf, max_conf);
            self.replace_metadata_indices((subject, predicate, object), &old, weight);
            true
        } else {
            false
        }
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

    #[test]
    fn remove_triples_in_compartment_basic() {
        let kg = KnowledgeGraph::new();
        let a = sym(1);
        let b = sym(2);
        let c = sym(3);
        let r = sym(10);

        // Two triples in "repo-x", one global
        kg.insert_triple(&Triple::new(a, r, b).with_compartment("repo-x".into()))
            .unwrap();
        kg.insert_triple(&Triple::new(a, r, c).with_compartment("repo-x".into()))
            .unwrap();
        kg.insert_triple(&Triple::new(b, r, c)).unwrap();

        assert_eq!(kg.triple_count(), 3);

        let removed = kg.remove_triples_in_compartment("repo-x");
        assert_eq!(removed, 2);
        assert_eq!(kg.triple_count(), 1);

        // The global triple should still exist
        let remaining = kg.all_triples();
        assert_eq!(remaining.len(), 1);
        assert!(remaining[0].compartment_id.is_none());
    }

    #[test]
    fn remove_triples_in_compartment_empty() {
        let kg = KnowledgeGraph::new();
        let removed = kg.remove_triples_in_compartment("nonexistent");
        assert_eq!(removed, 0);
    }

    #[test]
    fn remove_triples_in_compartment_updates_predicate_index() {
        let kg = KnowledgeGraph::new();
        let a = sym(1);
        let b = sym(2);
        let c = sym(3);
        let r = sym(10);

        kg.insert_triple(&Triple::new(a, r, b).with_compartment("repo-y".into()))
            .unwrap();
        kg.insert_triple(&Triple::new(a, r, c)).unwrap();

        assert_eq!(kg.triples_for_predicate(r).len(), 2);

        kg.remove_triples_in_compartment("repo-y");

        // Predicate index should only have the global triple
        let pairs = kg.triples_for_predicate(r);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], (a, c));
    }

    #[test]
    fn insert_triple_is_idempotent() {
        let kg = KnowledgeGraph::new();
        let sun = sym(1);
        let is_a = sym(2);
        let star = sym(3);

        // Insert the same triple 100 times
        for _ in 0..100 {
            kg.insert_triple(&Triple::new(sun, is_a, star)).unwrap();
        }

        // Should have exactly 1 triple, not 100
        assert_eq!(kg.triple_count(), 1);
        assert_eq!(kg.objects_of(sun, is_a), vec![star]);
        assert_eq!(kg.triples_for_predicate(is_a).len(), 1);
    }

    #[test]
    fn idempotent_insert_takes_max_confidence() {
        let kg = KnowledgeGraph::new();
        let a = sym(1);
        let r = sym(2);
        let b = sym(3);

        let mut t1 = Triple::new(a, r, b);
        t1.confidence = 0.5;
        kg.insert_triple(&t1).unwrap();

        let mut t2 = Triple::new(a, r, b);
        t2.confidence = 0.9;
        kg.insert_triple(&t2).unwrap();

        // Still one triple, but confidence should be max(0.5, 0.9) = 0.9
        assert_eq!(kg.triple_count(), 1);

        let triples = kg.triples_from(a);
        assert_eq!(triples.len(), 1);
        assert!((triples[0].confidence - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn exact_replay_replaces_and_can_clear_provenance() {
        let kg = KnowledgeGraph::new();
        let a = sym(1);
        let r = sym(2);
        let b = sym(3);
        let mut tracked = Triple::new(a, r, b).with_provenance(41);
        tracked.confidence = 0.9;
        tracked.timestamp = 100;
        kg.insert_triple(&tracked).unwrap();

        let mut replayed = Triple::new(a, r, b);
        replayed.confidence = 0.4;
        replayed.timestamp = 200;
        kg.replace_triple_exact(&replayed).unwrap();

        let restored = kg.triples_from(a).pop().unwrap();
        assert_eq!(kg.triple_count(), 1);
        assert_eq!(restored.provenance_id, None);
        assert_eq!(restored.confidence, 0.4);
        assert_eq!(restored.timestamp, 200);
    }

    #[test]
    fn audit_reverse_indices_follow_metadata_updates_and_removal() {
        let kg = KnowledgeGraph::new();
        let a = sym(1);
        let relation = sym(2);
        let b = sym(3);
        let mut first = Triple::new(a, relation, b)
            .with_confidence(0.4)
            .with_provenance(41);
        first.compartment_id = Some("audit".into());
        kg.insert_triple(&first).unwrap();

        assert_eq!(
            kg.triples_with_provenance_ids(&HashSet::from([41])).len(),
            1
        );
        assert!(kg.triples_at_or_above_confidence(0.5).is_empty());
        assert!(kg.triples_without_provenance().is_empty());

        let updated = Triple::new(a, relation, b)
            .with_confidence(0.9)
            .with_provenance(42);
        kg.insert_triple(&updated).unwrap();
        assert!(
            kg.triples_with_provenance_ids(&HashSet::from([41]))
                .is_empty()
        );
        assert_eq!(
            kg.triples_with_provenance_ids(&HashSet::from([42])).len(),
            1
        );
        assert_eq!(kg.triples_at_or_above_confidence(0.8).len(), 1);

        let untracked = Triple::new(a, relation, b).with_confidence(0.2);
        kg.replace_triple_exact(&untracked).unwrap();
        assert!(
            kg.triples_with_provenance_ids(&HashSet::from([42]))
                .is_empty()
        );
        assert_eq!(kg.triples_without_provenance().len(), 1);
        assert!(kg.triples_at_or_above_confidence(0.8).is_empty());

        assert!(kg.remove_triple(a, relation, b));
        assert!(kg.triples_without_provenance().is_empty());
        assert!(kg.triples_with_predicate(relation).is_empty());

        let c = sym(4);
        let mut restored = Triple::new(a, relation, c)
            .with_confidence(-0.0)
            .with_provenance(43);
        restored.compartment_id = Some("restored".into());
        assert_eq!(kg.bulk_load(&[restored]).unwrap(), 1);
        assert_eq!(
            kg.triples_with_provenance_ids(&HashSet::from([43])).len(),
            1
        );
        assert!(kg.triples_at_or_above_confidence(0.1).is_empty());
        assert!(kg.update_triple_confidence(a, relation, c, 0.75, 0.0, 1.0));
        assert_eq!(kg.triples_at_or_above_confidence(0.7).len(), 1);
        assert_eq!(kg.remove_triples_in_compartment("restored"), 1);
        assert!(
            kg.triples_with_provenance_ids(&HashSet::from([43]))
                .is_empty()
        );
        assert!(kg.triples_at_or_above_confidence(0.0).is_empty());
    }
}
