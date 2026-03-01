//! Relationship Graph — Phase 25b.
//!
//! Models interpersonal relationships between contacts with typed edges,
//! strength decay, reinforcement, and social circle detection via VSA
//! similarity clustering.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;
use crate::vsa::encode::encode_token;
use crate::vsa::ops::VsaOps;
use crate::vsa::HyperVec;

use super::contact::{ContactError, ContactManager, ContactResult};

// ═══════════════════════════════════════════════════════════════════════
// RelationshipKind
// ═══════════════════════════════════════════════════════════════════════

/// The kind of interpersonal relationship between two contacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationshipKind {
    Family,
    Friend,
    Colleague,
    Acquaintance,
    Mentor,
    Mentee,
    Manager,
    Report,
    Neighbor,
    Service,
}

impl RelationshipKind {
    /// All variants for iteration.
    pub const ALL: &'static [Self] = &[
        Self::Family,
        Self::Friend,
        Self::Colleague,
        Self::Acquaintance,
        Self::Mentor,
        Self::Mentee,
        Self::Manager,
        Self::Report,
        Self::Neighbor,
        Self::Service,
    ];

    /// Slug for KG predicate naming.
    pub fn slug(&self) -> &'static str {
        match self {
            Self::Family => "family",
            Self::Friend => "friend",
            Self::Colleague => "colleague",
            Self::Acquaintance => "acquaintance",
            Self::Mentor => "mentor",
            Self::Mentee => "mentee",
            Self::Manager => "manager",
            Self::Report => "report",
            Self::Neighbor => "neighbor",
            Self::Service => "service",
        }
    }
}

impl fmt::Display for RelationshipKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.slug())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Relationship
// ═══════════════════════════════════════════════════════════════════════

/// A directed relationship edge between two contacts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    /// Source contact ID.
    pub from: String,
    /// Target contact ID.
    pub to: String,
    /// The kind of relationship.
    pub kind: RelationshipKind,
    /// Strength in [0.0, 1.0], decays over time.
    pub strength: f32,
    /// When the relationship was established (UNIX seconds).
    pub established_at: u64,
    /// Last reinforcement timestamp (UNIX seconds).
    pub last_reinforced: u64,
    /// KG symbol for the source contact.
    pub from_symbol: SymbolId,
    /// KG symbol for the target contact.
    pub to_symbol: SymbolId,
}

// ═══════════════════════════════════════════════════════════════════════
// SocialCircle
// ═══════════════════════════════════════════════════════════════════════

/// A cluster of contacts detected by shared relationship patterns.
#[derive(Debug, Clone)]
pub struct SocialCircle {
    /// Label (auto-generated).
    pub label: String,
    /// Contact IDs in this circle.
    pub members: Vec<String>,
}

// ═══════════════════════════════════════════════════════════════════════
// RelationshipPredicates
// ═══════════════════════════════════════════════════════════════════════

/// Well-known predicates for relationship KG triples.
pub struct RelationshipPredicates {
    /// One predicate per RelationshipKind: `contact:rel-{kind}`.
    pub kind_preds: HashMap<RelationshipKind, SymbolId>,
    /// `contact:rel-strength` — numeric strength.
    pub rel_strength: SymbolId,
    /// `contact:social-circle` — circle membership.
    pub social_circle: SymbolId,
}

impl RelationshipPredicates {
    /// Resolve or create all relationship predicates.
    pub fn init(engine: &Engine) -> ContactResult<Self> {
        let mut kind_preds = HashMap::new();
        for kind in RelationshipKind::ALL {
            let pred =
                engine.resolve_or_create_relation(&format!("contact:rel-{}", kind.slug()))?;
            kind_preds.insert(*kind, pred);
        }
        Ok(Self {
            kind_preds,
            rel_strength: engine.resolve_or_create_relation("contact:rel-strength")?,
            social_circle: engine.resolve_or_create_relation("contact:social-circle")?,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════
// RelRoleVectors — VSA role vectors for relationship encoding
// ═══════════════════════════════════════════════════════════════════════

/// Deterministic role hypervectors for encoding relationship patterns.
pub struct RelRoleVectors {
    pub partner: HyperVec,
    pub kind: HyperVec,
    pub strength: HyperVec,
    pub recency: HyperVec,
}

impl RelRoleVectors {
    /// Create role vectors via deterministic token encoding.
    pub fn new(ops: &VsaOps) -> Self {
        Self {
            partner: encode_token(ops, "rel-role:partner"),
            kind: encode_token(ops, "rel-role:kind"),
            strength: encode_token(ops, "rel-role:strength"),
            recency: encode_token(ops, "rel-role:recency"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// RelationshipGraph
// ═══════════════════════════════════════════════════════════════════════

/// Manages the social relationship graph between contacts.
#[derive(Default)]
pub struct RelationshipGraph {
    /// Edges keyed by (from, to) contact IDs.
    edges: HashMap<(String, String), Relationship>,
    /// Adjacency index: contact_id → set of neighbor contact_ids.
    adjacency: HashMap<String, Vec<String>>,
    /// Per-contact VSA relationship vector (bundled from all their relationships).
    rel_vectors: HashMap<String, HyperVec>,
    predicates: Option<RelationshipPredicates>,
    role_vectors: Option<RelRoleVectors>,
}

impl Serialize for RelationshipGraph {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.edges.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for RelationshipGraph {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let edges =
            HashMap::<(String, String), Relationship>::deserialize(deserializer)?;
        // Rebuild adjacency index.
        let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();
        for (from, to) in edges.keys() {
            adjacency
                .entry(from.clone())
                .or_default()
                .push(to.clone());
        }
        Ok(Self {
            edges,
            adjacency,
            rel_vectors: HashMap::new(),
            predicates: None,
            role_vectors: None,
        })
    }
}

impl fmt::Debug for RelationshipGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RelationshipGraph")
            .field("edge_count", &self.edges.len())
            .field("node_count", &self.adjacency.len())
            .finish()
    }
}

impl RelationshipGraph {
    /// Create a new graph, initializing predicates and role vectors.
    pub fn new(engine: &Engine) -> ContactResult<Self> {
        let predicates = RelationshipPredicates::init(engine)?;
        let role_vectors = RelRoleVectors::new(engine.ops());
        Ok(Self {
            edges: HashMap::new(),
            adjacency: HashMap::new(),
            rel_vectors: HashMap::new(),
            predicates: Some(predicates),
            role_vectors: Some(role_vectors),
        })
    }

    /// Ensure predicates and role vectors are initialized (post-deserialization).
    pub fn ensure_init(&mut self, engine: &Engine) -> ContactResult<()> {
        if self.predicates.is_none() {
            self.predicates = Some(RelationshipPredicates::init(engine)?);
        }
        if self.role_vectors.is_none() {
            self.role_vectors = Some(RelRoleVectors::new(engine.ops()));
        }
        Ok(())
    }

    /// Number of relationship edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Access predicates, returning error if not initialized.
    fn predicates(&self) -> ContactResult<&RelationshipPredicates> {
        self.predicates.as_ref().ok_or_else(|| {
            ContactError::Engine(Box::new(crate::error::AkhError::Store(
                crate::error::StoreError::Serialization {
                    message: "RelationshipGraph predicates not initialized".to_string(),
                },
            )))
        })
    }

    // ── CRUD ──────────────────────────────────────────────────────────

    /// Add a relationship between two contacts.
    ///
    /// Stores bidirectional KG triples and records provenance.
    pub fn add_relationship(
        &mut self,
        engine: &Engine,
        contact_mgr: &ContactManager,
        from: &str,
        to: &str,
        kind: RelationshipKind,
        strength: f32,
    ) -> ContactResult<()> {
        let from_contact = contact_mgr.get(from).ok_or_else(|| ContactError::NotFound {
            contact_id: from.to_string(),
        })?;
        let to_contact = contact_mgr.get(to).ok_or_else(|| ContactError::NotFound {
            contact_id: to.to_string(),
        })?;

        let preds = self.predicates()?;
        let kind_pred = preds.kind_preds[&kind];
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let strength = strength.clamp(0.0, 1.0);

        // Store KG triple: from --rel-kind--> to.
        engine.add_triple(&Triple::new(from_contact.symbol_id, kind_pred, to_contact.symbol_id))?;

        let rel = Relationship {
            from: from.to_string(),
            to: to.to_string(),
            kind,
            strength,
            established_at: now,
            last_reinforced: now,
            from_symbol: from_contact.symbol_id,
            to_symbol: to_contact.symbol_id,
        };

        self.edges
            .insert((from.to_string(), to.to_string()), rel);
        self.adjacency
            .entry(from.to_string())
            .or_default()
            .push(to.to_string());

        // Record provenance.
        let mut record = ProvenanceRecord::new(
            from_contact.symbol_id,
            DerivationKind::RelationshipRecorded {
                from: from.to_string(),
                to: to.to_string(),
                kind: kind.slug().to_string(),
            },
        );
        let _ = engine.store_provenance(&mut record);

        // Rebuild VSA vector for the from contact.
        self.rebuild_rel_vector(engine, from);

        Ok(())
    }

    /// Remove a relationship between two contacts.
    pub fn remove_relationship(&mut self, from: &str, to: &str) -> Option<Relationship> {
        let removed = self.edges.remove(&(from.to_string(), to.to_string()));
        if removed.is_some() {
            if let Some(adj) = self.adjacency.get_mut(from) {
                adj.retain(|n| n != to);
            }
        }
        removed
    }

    /// Get all relationships of a contact.
    pub fn relationships_of(&self, contact_id: &str) -> Vec<&Relationship> {
        self.adjacency
            .get(contact_id)
            .map(|neighbors| {
                neighbors
                    .iter()
                    .filter_map(|n| {
                        self.edges.get(&(contact_id.to_string(), n.clone()))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get relationships of a specific kind for a contact.
    pub fn relationships_of_kind(
        &self,
        contact_id: &str,
        kind: RelationshipKind,
    ) -> Vec<&Relationship> {
        self.relationships_of(contact_id)
            .into_iter()
            .filter(|r| r.kind == kind)
            .collect()
    }

    /// Reinforce a relationship (bumps strength toward 1.0).
    pub fn reinforce(&mut self, from: &str, to: &str, timestamp: u64) {
        if let Some(rel) = self
            .edges
            .get_mut(&(from.to_string(), to.to_string()))
        {
            // Move strength 20% closer to 1.0.
            rel.strength += (1.0 - rel.strength) * 0.2;
            rel.strength = rel.strength.clamp(0.0, 1.0);
            rel.last_reinforced = timestamp;
        }
    }

    /// Apply exponential decay to all relationship strengths.
    ///
    /// `strength *= exp(-lambda * days_since_reinforced)`
    pub fn decay_all(&mut self, now: u64, lambda: f64) {
        for rel in self.edges.values_mut() {
            let days = (now.saturating_sub(rel.last_reinforced) as f64) / 86400.0;
            let factor = (-lambda * days).exp() as f32;
            rel.strength *= factor;
            rel.strength = rel.strength.clamp(0.0, 1.0);
        }
    }

    /// Detect social circles by grouping contacts with similar VSA relationship vectors.
    ///
    /// Returns clusters of contacts whose relationship vectors have high Hamming similarity.
    pub fn detect_circles(
        &self,
        _engine: &Engine,
        min_members: usize,
    ) -> Vec<SocialCircle> {
        // Simple greedy clustering: for each unassigned contact, find all contacts
        // with Hamming similarity > 0.6 to form a circle.
        let contacts_with_vecs: Vec<(&String, &HyperVec)> = self.rel_vectors.iter().collect();
        if contacts_with_vecs.is_empty() {
            return Vec::new();
        }

        let mut assigned: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut circles = Vec::new();
        let threshold = 0.6;

        for (cid, vec) in &contacts_with_vecs {
            if assigned.contains(*cid) {
                continue;
            }
            let mut members = vec![(*cid).clone()];
            for (other_cid, other_vec) in &contacts_with_vecs {
                if *cid == *other_cid || assigned.contains(*other_cid) {
                    continue;
                }
                let sim = hamming_similarity(vec, other_vec);
                if sim > threshold {
                    members.push((*other_cid).clone());
                }
            }
            if members.len() >= min_members {
                for m in &members {
                    assigned.insert(m.clone());
                }
                circles.push(SocialCircle {
                    label: format!("circle-{}", circles.len() + 1),
                    members,
                });
            }
        }

        circles
    }

    /// Rewire relationships after a contact merge: update edges referencing
    /// the discarded contact to point to the kept contact.
    pub fn rewire_after_merge(&mut self, keep_id: &str, discard_id: &str) {
        // Collect edges to rewire.
        let edges_to_rewire: Vec<(String, String)> = self
            .edges
            .keys()
            .filter(|(f, t)| f == discard_id || t == discard_id)
            .cloned()
            .collect();

        for (from, to) in edges_to_rewire {
            if let Some(mut rel) = self.edges.remove(&(from.clone(), to.clone())) {
                let new_from = if from == discard_id {
                    keep_id.to_string()
                } else {
                    from
                };
                let new_to = if to == discard_id {
                    keep_id.to_string()
                } else {
                    to
                };
                // Skip self-loops created by merge.
                if new_from == new_to {
                    continue;
                }
                rel.from = new_from.clone();
                rel.to = new_to.clone();
                self.edges.insert((new_from.clone(), new_to.clone()), rel);
                self.adjacency
                    .entry(new_from)
                    .or_default()
                    .push(new_to);
            }
        }

        // Remove adjacency entry for discarded contact.
        self.adjacency.remove(discard_id);
        self.rel_vectors.remove(discard_id);
    }

    // ── VSA ───────────────────────────────────────────────────────────

    /// Rebuild the VSA relationship vector for a contact by bundling all their
    /// relationship partners (bound with role vectors).
    fn rebuild_rel_vector(&mut self, engine: &Engine, contact_id: &str) {
        let roles = match &self.role_vectors {
            Some(r) => r,
            None => return,
        };
        let ops = engine.ops();

        let rels = self.relationships_of(contact_id);
        if rels.is_empty() {
            self.rel_vectors.remove(contact_id);
            return;
        }

        let mut vecs = Vec::new();
        for rel in rels {
            // Encode: partner ⊗ token(to_id) + kind ⊗ token(kind_slug)
            let partner_vec = encode_token(ops, &format!("contact:{}", rel.to));
            if let Ok(bound) = ops.bind(&roles.partner, &partner_vec) {
                vecs.push(bound);
            }

            let kind_vec = encode_token(ops, &format!("rel-kind:{}", rel.kind.slug()));
            if let Ok(kind_bound) = ops.bind(&roles.kind, &kind_vec) {
                vecs.push(kind_bound);
            }
        }

        if vecs.is_empty() {
            return;
        }

        let refs: Vec<&HyperVec> = vecs.iter().collect();
        if let Ok(bundled) = ops.bundle(&refs) {
            self.rel_vectors.insert(contact_id.to_string(), bundled);
        }
    }

    // ── Persistence ───────────────────────────────────────────────────

    /// Persist to the engine's durable store.
    pub fn persist(&self, engine: &Engine) -> ContactResult<()> {
        let bytes = bincode::serialize(self).map_err(|e| {
            ContactError::Engine(Box::new(crate::error::AkhError::Store(
                crate::error::StoreError::Serialization {
                    message: format!("failed to serialize relationship graph: {e}"),
                },
            )))
        })?;
        engine
            .store()
            .put_meta(b"agent:relationship_graph", &bytes)
            .map_err(|e| ContactError::Engine(Box::new(crate::error::AkhError::Store(e))))?;
        Ok(())
    }

    /// Restore from the engine's durable store.
    pub fn restore(engine: &Engine) -> ContactResult<Self> {
        let bytes = engine
            .store()
            .get_meta(b"agent:relationship_graph")
            .map_err(|e| ContactError::Engine(Box::new(crate::error::AkhError::Store(e))))?
            .ok_or(ContactError::NotFound {
                contact_id: "<store>".to_string(),
            })?;
        let mut graph: Self = bincode::deserialize(&bytes).map_err(|e| {
            ContactError::Engine(Box::new(crate::error::AkhError::Store(
                crate::error::StoreError::Serialization {
                    message: format!("failed to deserialize relationship graph: {e}"),
                },
            )))
        })?;
        graph.ensure_init(engine)?;
        Ok(graph)
    }
}

/// Simple Hamming-similarity between two binary hypervectors.
fn hamming_similarity(a: &HyperVec, b: &HyperVec) -> f64 {
    let a_data = a.data();
    let b_data = b.data();
    let total_bits = a_data.len() * 8;
    if total_bits == 0 {
        return 0.0;
    }
    let differing: u32 = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum();
    1.0 - (differing as f64 / total_bits as f64)
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Engine;

    fn test_engine() -> Engine {
        use crate::engine::EngineConfig;
        use crate::vsa::Dimension;
        Engine::new(EngineConfig {
            dimension: Dimension(1000),
            ..EngineConfig::default()
        })
        .expect("in-memory engine")
    }

    fn setup() -> (Engine, ContactManager, RelationshipGraph) {
        let engine = test_engine();
        let mgr = ContactManager::new(&engine).unwrap();
        let graph = RelationshipGraph::new(&engine).unwrap();
        (engine, mgr, graph)
    }

    #[test]
    fn add_and_query_relationship() {
        let (engine, mut mgr, mut graph) = setup();

        let alice = mgr
            .create_contact(&engine, "Alice", &["alice@a.com".to_string()])
            .unwrap();
        let bob = mgr
            .create_contact(&engine, "Bob", &["bob@b.com".to_string()])
            .unwrap();

        graph
            .add_relationship(&engine, &mgr, &alice, &bob, RelationshipKind::Friend, 0.8)
            .unwrap();

        assert_eq!(graph.edge_count(), 1);

        let rels = graph.relationships_of(&alice);
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].to, bob);
        assert_eq!(rels[0].kind, RelationshipKind::Friend);
    }

    #[test]
    fn relationships_of_kind() {
        let (engine, mut mgr, mut graph) = setup();

        let alice = mgr
            .create_contact(&engine, "Alice", &["alice@a.com".to_string()])
            .unwrap();
        let bob = mgr
            .create_contact(&engine, "Bob", &["bob@b.com".to_string()])
            .unwrap();
        let carol = mgr
            .create_contact(&engine, "Carol", &["carol@c.com".to_string()])
            .unwrap();

        graph
            .add_relationship(&engine, &mgr, &alice, &bob, RelationshipKind::Friend, 0.8)
            .unwrap();
        graph
            .add_relationship(&engine, &mgr, &alice, &carol, RelationshipKind::Colleague, 0.6)
            .unwrap();

        let friends = graph.relationships_of_kind(&alice, RelationshipKind::Friend);
        assert_eq!(friends.len(), 1);
        let colleagues = graph.relationships_of_kind(&alice, RelationshipKind::Colleague);
        assert_eq!(colleagues.len(), 1);
    }

    #[test]
    fn reinforce_and_decay() {
        let (engine, mut mgr, mut graph) = setup();

        let alice = mgr
            .create_contact(&engine, "Alice", &["alice@a.com".to_string()])
            .unwrap();
        let bob = mgr
            .create_contact(&engine, "Bob", &["bob@b.com".to_string()])
            .unwrap();

        graph
            .add_relationship(&engine, &mgr, &alice, &bob, RelationshipKind::Friend, 0.5)
            .unwrap();

        // Reinforce.
        graph.reinforce(&alice, &bob, 100);
        let rel = graph.edges.get(&(alice.clone(), bob.clone())).unwrap();
        assert!(rel.strength > 0.5);

        // Decay: 365 days later with lambda=0.01.
        let now = rel.last_reinforced + 365 * 86400;
        graph.decay_all(now, 0.01);
        let rel = graph.edges.get(&(alice, bob)).unwrap();
        assert!(rel.strength < 0.6); // Should have decayed significantly.
    }

    #[test]
    fn rewire_after_merge() {
        let (engine, mut mgr, mut graph) = setup();

        let alice = mgr
            .create_contact(&engine, "Alice", &["alice@a.com".to_string()])
            .unwrap();
        let bob = mgr
            .create_contact(&engine, "Bob", &["bob@b.com".to_string()])
            .unwrap();
        let carol = mgr
            .create_contact(&engine, "Carol", &["carol@c.com".to_string()])
            .unwrap();

        graph
            .add_relationship(&engine, &mgr, &alice, &carol, RelationshipKind::Friend, 0.8)
            .unwrap();
        graph
            .add_relationship(&engine, &mgr, &bob, &carol, RelationshipKind::Colleague, 0.6)
            .unwrap();

        // Merge alice into bob.
        graph.rewire_after_merge(&bob, &alice);

        // Alice's edge to carol should now be bob→carol.
        assert!(graph.edges.get(&(alice.clone(), carol.clone())).is_none());
        let bob_rels = graph.relationships_of(&bob);
        assert!(bob_rels.len() >= 1);
    }

    #[test]
    fn persist_and_restore() {
        let dir = tempfile::tempdir().unwrap();
        let engine = Engine::new(crate::engine::EngineConfig {
            data_dir: Some(dir.path().to_path_buf()),
            dimension: crate::vsa::Dimension(1000),
            ..crate::engine::EngineConfig::default()
        })
        .unwrap();
        let mut mgr = ContactManager::new(&engine).unwrap();
        let mut graph = RelationshipGraph::new(&engine).unwrap();

        let alice = mgr
            .create_contact(&engine, "Alice", &["alice@a.com".to_string()])
            .unwrap();
        let bob = mgr
            .create_contact(&engine, "Bob", &["bob@b.com".to_string()])
            .unwrap();

        graph
            .add_relationship(&engine, &mgr, &alice, &bob, RelationshipKind::Friend, 0.8)
            .unwrap();

        graph.persist(&engine).unwrap();

        let restored = RelationshipGraph::restore(&engine).unwrap();
        assert_eq!(restored.edge_count(), 1);
    }

    #[test]
    fn detect_circles_empty() {
        let (engine, _mgr, graph) = setup();
        let circles = graph.detect_circles(&engine, 2);
        assert!(circles.is_empty());
    }
}
