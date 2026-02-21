//! Social knowledge graph and theory of mind (Phase 12d).
//!
//! Each person or agent the akh interacts with becomes a KG entity with
//! well-known predicates. Per-interlocutor microtheories (Phase 9a) represent
//! what the agent believes they know. VSA interest bundling enables discovery
//! of interest overlaps.

use std::collections::HashMap;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::engine::Engine;
use crate::error::AkhResult;
use crate::symbol::SymbolId;
use crate::vsa::HyperVec;

use super::channel::ChannelKind;
use super::channel_message::InterlocutorId;
use super::multi_agent::InterlocutorKind;

// ── Error types ─────────────────────────────────────────────────────────

/// Errors from the interlocutor subsystem.
#[derive(Debug, Error, Diagnostic)]
pub enum InterlocutorError {
    #[error("interlocutor not found: \"{id}\"")]
    #[diagnostic(
        code(akh::interlocutor::not_found),
        help("The interlocutor has not been registered. They will be auto-registered on first message.")
    )]
    NotFound { id: String },

    #[error("cannot demote operator trust level")]
    #[diagnostic(
        code(akh::interlocutor::operator_immutable),
        help("The operator's trust level is always Operator and cannot be changed.")
    )]
    OperatorImmutable,
}

pub type InterlocutorResult<T> = Result<T, InterlocutorError>;

// ── Well-known predicates ───────────────────────────────────────────────

/// Well-known KG predicates for interlocutor modeling.
#[derive(Debug, Clone)]
pub struct InterlocutorPredicates {
    /// `interlocutor:has-channel` — channel IDs this person uses.
    pub has_channel: SymbolId,
    /// `interlocutor:has-trust-level` — trust level string.
    pub has_trust_level: SymbolId,
    /// `interlocutor:has-interest` — interest concept.
    pub has_interest: SymbolId,
    /// `interlocutor:last-interaction` — timestamp of last interaction.
    pub last_interaction: SymbolId,
    /// `interlocutor:interaction-count` — total interaction count.
    pub interaction_count: SymbolId,
    /// `interlocutor:has-knowledge-mt` — microtheory SymbolId for theory of mind.
    pub has_knowledge_mt: SymbolId,
}

impl InterlocutorPredicates {
    /// Resolve or create all interlocutor predicates in the engine.
    pub fn init(engine: &Engine) -> AkhResult<Self> {
        Ok(Self {
            has_channel: engine.resolve_or_create_relation("interlocutor:has-channel")?,
            has_trust_level: engine.resolve_or_create_relation("interlocutor:has-trust-level")?,
            has_interest: engine.resolve_or_create_relation("interlocutor:has-interest")?,
            last_interaction: engine.resolve_or_create_relation("interlocutor:last-interaction")?,
            interaction_count: engine.resolve_or_create_relation("interlocutor:interaction-count")?,
            has_knowledge_mt: engine.resolve_or_create_relation("interlocutor:has-knowledge-mt")?,
        })
    }
}

// ── InterlocutorProfile ─────────────────────────────────────────────────

/// Profile for a person or agent the akh interacts with.
///
/// Represents the agent's model of who this person is, what they know,
/// and what they're interested in.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterlocutorProfile {
    /// The interlocutor's string identifier (same as `InterlocutorId`).
    pub interlocutor_id: String,
    /// KG entity SymbolId representing this interlocutor.
    pub symbol_id: SymbolId,
    /// Whether this interlocutor is a human or another agent (Phase 12g).
    pub kind: InterlocutorKind,
    /// Channel IDs this interlocutor uses.
    pub channel_ids: Vec<String>,
    /// Trust level determining capability preset.
    pub trust_level: ChannelKind,
    /// SymbolId of their personal microtheory (theory of mind context).
    pub knowledge_mt: Option<SymbolId>,
    /// SymbolIds of concepts this interlocutor is interested in.
    pub interests: Vec<SymbolId>,
    /// Number of interactions recorded.
    pub interaction_count: u64,
    /// Timestamp of most recent interaction.
    pub last_interaction: u64,
}

impl InterlocutorProfile {
    /// Whether this is the operator profile.
    pub fn is_operator(&self) -> bool {
        self.interlocutor_id == "operator"
    }

    /// Whether this interlocutor is another agent (not human).
    pub fn is_agent(&self) -> bool {
        self.kind == InterlocutorKind::Agent
    }
}

// ── InterlocutorRegistry ────────────────────────────────────────────────

/// Registry of known interlocutors.
///
/// Tracks all people and agents the akh has interacted with,
/// indexed by their string ID.
#[derive(Debug)]
pub struct InterlocutorRegistry {
    /// Profiles indexed by interlocutor ID.
    profiles: HashMap<String, InterlocutorProfile>,
    /// VSA interest vectors indexed by interlocutor ID.
    interest_vectors: HashMap<String, HyperVec>,
    /// Well-known predicates (None until initialized with engine).
    predicates: Option<InterlocutorPredicates>,
}

impl InterlocutorRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
            interest_vectors: HashMap::new(),
            predicates: None,
        }
    }

    /// Initialize predicates from the engine.
    pub fn init_predicates(&mut self, engine: &Engine) -> AkhResult<()> {
        self.predicates = Some(InterlocutorPredicates::init(engine)?);
        Ok(())
    }

    /// Get the well-known predicates (panics if not initialized).
    pub fn predicates(&self) -> &InterlocutorPredicates {
        self.predicates
            .as_ref()
            .expect("InterlocutorPredicates not initialized; call init_predicates first")
    }

    /// Number of known interlocutors.
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    /// Whether no interlocutors are registered.
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    /// Look up an interlocutor by ID.
    pub fn get(&self, id: &str) -> Option<&InterlocutorProfile> {
        self.profiles.get(id)
    }

    /// Look up a mutable interlocutor by ID.
    pub fn get_mut(&mut self, id: &str) -> Option<&mut InterlocutorProfile> {
        self.profiles.get_mut(id)
    }

    /// Get the interest vector for an interlocutor.
    pub fn interest_vector(&self, id: &str) -> Option<&HyperVec> {
        self.interest_vectors.get(id)
    }

    /// Iterate over all profiles.
    pub fn profiles(&self) -> impl Iterator<Item = &InterlocutorProfile> {
        self.profiles.values()
    }

    /// Register a new interlocutor or return existing.
    ///
    /// Creates a KG entity and optionally a personal microtheory.
    pub fn register(
        &mut self,
        interlocutor_id: &InterlocutorId,
        channel_id: &str,
        trust_level: ChannelKind,
        engine: &Engine,
    ) -> AkhResult<&InterlocutorProfile> {
        let id_str = interlocutor_id.as_str().to_string();

        if self.profiles.contains_key(&id_str) {
            // Update channel list and last interaction.
            let profile = self.profiles.get_mut(&id_str).unwrap();
            if !profile.channel_ids.contains(&channel_id.to_string()) {
                profile.channel_ids.push(channel_id.to_string());
            }
            profile.last_interaction = now_secs();
            profile.interaction_count += 1;
            return Ok(self.profiles.get(&id_str).unwrap());
        }

        // Create KG entity for this interlocutor.
        let entity_label = format!("interlocutor:{}", id_str);
        let sym_id = engine.resolve_or_create_entity(&entity_label)?;

        // Create personal microtheory (theory of mind context).
        let mt_label = format!("mt:{}-knowledge", id_str);
        let knowledge_mt = engine
            .create_context(
                &mt_label,
                crate::compartment::ContextDomain::Belief,
                &[],
            )
            .ok()
            .map(|mt| mt.id);

        // Store trust level in KG.
        if let Some(ref preds) = self.predicates {
            let trust_str = format!("{trust_level}");
            let trust_sym = engine.resolve_or_create_entity(&trust_str)?;
            let _ = engine.add_triple(
                &crate::graph::Triple::new(sym_id, preds.has_trust_level, trust_sym),
            );

            // Store channel association.
            let ch_sym = engine.resolve_or_create_entity(channel_id)?;
            let _ = engine.add_triple(
                &crate::graph::Triple::new(sym_id, preds.has_channel, ch_sym),
            );

            // Store microtheory link.
            if let Some(mt_id) = knowledge_mt {
                let _ = engine.add_triple(
                    &crate::graph::Triple::new(sym_id, preds.has_knowledge_mt, mt_id),
                );
            }
        }

        let profile = InterlocutorProfile {
            interlocutor_id: id_str.clone(),
            symbol_id: sym_id,
            kind: InterlocutorKind::default(),
            channel_ids: vec![channel_id.to_string()],
            trust_level,
            knowledge_mt,
            interests: Vec::new(),
            interaction_count: 1,
            last_interaction: now_secs(),
        };

        self.profiles.insert(id_str.clone(), profile);
        Ok(self.profiles.get(&id_str).unwrap())
    }

    /// Record an interest for an interlocutor.
    pub fn add_interest(
        &mut self,
        interlocutor_id: &str,
        concept: SymbolId,
        engine: &Engine,
    ) -> InterlocutorResult<()> {
        let profile = self
            .profiles
            .get_mut(interlocutor_id)
            .ok_or_else(|| InterlocutorError::NotFound {
                id: interlocutor_id.to_string(),
            })?;

        if !profile.interests.contains(&concept) {
            profile.interests.push(concept);

            // Record in KG.
            if let Some(ref preds) = self.predicates {
                let _ = engine.add_triple(
                    &crate::graph::Triple::new(profile.symbol_id, preds.has_interest, concept),
                );
            }

            // Rebuild interest vector.
            self.rebuild_interest_vector(interlocutor_id, engine);
        }

        Ok(())
    }

    /// Update the trust level for an interlocutor.
    pub fn set_trust_level(
        &mut self,
        interlocutor_id: &str,
        trust_level: ChannelKind,
    ) -> InterlocutorResult<()> {
        let profile = self
            .profiles
            .get_mut(interlocutor_id)
            .ok_or_else(|| InterlocutorError::NotFound {
                id: interlocutor_id.to_string(),
            })?;

        if profile.is_operator() {
            return Err(InterlocutorError::OperatorImmutable);
        }

        profile.trust_level = trust_level;
        Ok(())
    }

    /// Record a "knows" fact in the interlocutor's theory-of-mind microtheory.
    pub fn record_knowledge(
        &self,
        interlocutor_id: &str,
        concept: SymbolId,
        engine: &Engine,
    ) -> InterlocutorResult<()> {
        let profile = self
            .profiles
            .get(interlocutor_id)
            .ok_or_else(|| InterlocutorError::NotFound {
                id: interlocutor_id.to_string(),
            })?;

        if let Some(mt_id) = profile.knowledge_mt {
            let mt_label = format!("mt:{}-knowledge", interlocutor_id);
            let knows = engine
                .resolve_or_create_relation("interlocutor:knows")
                .unwrap_or(concept);
            let _ = engine.add_triple(
                &crate::graph::Triple::new(profile.symbol_id, knows, concept)
                    .with_compartment(mt_label),
            );
            let _ = mt_id; // used as context scope marker
        }

        Ok(())
    }

    /// Find similar interlocutors to a given one (impression formation).
    ///
    /// Uses HNSW nearest-neighbor search on interest vectors.
    pub fn find_similar(
        &self,
        interlocutor_id: &str,
        k: usize,
    ) -> Vec<(String, f32)> {
        let query = match self.interest_vectors.get(interlocutor_id) {
            Some(v) => v,
            None => return Vec::new(),
        };

        let mut results = Vec::new();
        for (id, vec) in &self.interest_vectors {
            if id == interlocutor_id {
                continue;
            }
            // Compute Hamming similarity.
            let sim = hamming_similarity(query, vec);
            results.push((id.clone(), sim));
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Rebuild the VSA interest vector for an interlocutor.
    fn rebuild_interest_vector(&mut self, interlocutor_id: &str, engine: &Engine) {
        let profile = match self.profiles.get(interlocutor_id) {
            Some(p) => p,
            None => return,
        };

        if profile.interests.is_empty() {
            self.interest_vectors.remove(interlocutor_id);
            return;
        }

        let ops = engine.ops();
        let item_mem = engine.item_memory();

        // Get or create vectors for each interest concept.
        let vecs: Vec<HyperVec> = profile
            .interests
            .iter()
            .map(|sym| item_mem.get_or_create(ops, *sym))
            .collect();

        // Bundle all interest vectors.
        let refs: Vec<&HyperVec> = vecs.iter().collect();
        if let Ok(bundled) = ops.bundle(&refs) {
            self.interest_vectors
                .insert(interlocutor_id.to_string(), bundled);
        }
    }
}

impl Default for InterlocutorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Interest overlap queries ────────────────────────────────────────────

/// Compute the interest overlap between two interlocutors.
///
/// Returns a similarity score in [0.0, 1.0] or None if either
/// interlocutor has no interest vector.
pub fn interest_overlap(registry: &InterlocutorRegistry, a: &str, b: &str) -> Option<f32> {
    let va = registry.interest_vector(a)?;
    let vb = registry.interest_vector(b)?;
    Some(hamming_similarity(va, vb))
}

/// Compute normalized Hamming similarity between two hypervectors.
fn hamming_similarity(a: &HyperVec, b: &HyperVec) -> f32 {
    if a.dim() != b.dim() {
        return 0.0;
    }
    let total_bits = a.dim().0 as f32;
    let matching: u32 = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(x, y)| (x ^ y).count_zeros())
        .sum();
    matching as f32 / total_bits
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_new_is_empty() {
        let reg = InterlocutorRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn interlocutor_profile_is_operator() {
        let profile = InterlocutorProfile {
            interlocutor_id: "operator".to_string(),
            symbol_id: SymbolId::new(1).unwrap(),
            kind: InterlocutorKind::Human,
            channel_ids: vec!["op-ch".to_string()],
            trust_level: ChannelKind::Operator,
            knowledge_mt: None,
            interests: Vec::new(),
            interaction_count: 0,
            last_interaction: 0,
        };
        assert!(profile.is_operator());
        assert!(!profile.is_agent());
    }

    #[test]
    fn interlocutor_profile_not_operator() {
        let profile = InterlocutorProfile {
            interlocutor_id: "alice".to_string(),
            symbol_id: SymbolId::new(2).unwrap(),
            kind: InterlocutorKind::Human,
            channel_ids: vec![],
            trust_level: ChannelKind::Social,
            knowledge_mt: None,
            interests: Vec::new(),
            interaction_count: 0,
            last_interaction: 0,
        };
        assert!(!profile.is_operator());
    }

    #[test]
    fn interlocutor_profile_is_agent() {
        let profile = InterlocutorProfile {
            interlocutor_id: "peer-akh".to_string(),
            symbol_id: SymbolId::new(3).unwrap(),
            kind: InterlocutorKind::Agent,
            channel_ids: vec![],
            trust_level: ChannelKind::Trusted,
            knowledge_mt: None,
            interests: Vec::new(),
            interaction_count: 0,
            last_interaction: 0,
        };
        assert!(profile.is_agent());
        assert!(!profile.is_operator());
    }

    #[test]
    fn set_trust_level_operator_immutable() {
        let mut reg = InterlocutorRegistry::new();
        reg.profiles.insert(
            "operator".to_string(),
            InterlocutorProfile {
                interlocutor_id: "operator".to_string(),
                symbol_id: SymbolId::new(1).unwrap(),
                kind: InterlocutorKind::Human,
                channel_ids: vec![],
                trust_level: ChannelKind::Operator,
                knowledge_mt: None,
                interests: Vec::new(),
                interaction_count: 0,
                last_interaction: 0,
            },
        );

        let result = reg.set_trust_level("operator", ChannelKind::Social);
        assert!(result.is_err());
    }

    #[test]
    fn set_trust_level_normal_user() {
        let mut reg = InterlocutorRegistry::new();
        reg.profiles.insert(
            "alice".to_string(),
            InterlocutorProfile {
                interlocutor_id: "alice".to_string(),
                symbol_id: SymbolId::new(2).unwrap(),
                kind: InterlocutorKind::Human,
                channel_ids: vec![],
                trust_level: ChannelKind::Social,
                knowledge_mt: None,
                interests: Vec::new(),
                interaction_count: 0,
                last_interaction: 0,
            },
        );

        reg.set_trust_level("alice", ChannelKind::Trusted).unwrap();
        assert_eq!(reg.get("alice").unwrap().trust_level, ChannelKind::Trusted);
    }

    #[test]
    fn set_trust_level_not_found() {
        let mut reg = InterlocutorRegistry::new();
        let result = reg.set_trust_level("unknown", ChannelKind::Social);
        assert!(result.is_err());
    }

    #[test]
    fn find_similar_empty() {
        let reg = InterlocutorRegistry::new();
        let results = reg.find_similar("alice", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn hamming_similarity_identical() {
        let v = HyperVec::zero(crate::vsa::Dimension(64), crate::vsa::Encoding::Bipolar);
        let sim = hamming_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn interest_overlap_no_vectors() {
        let reg = InterlocutorRegistry::new();
        assert!(interest_overlap(&reg, "alice", "bob").is_none());
    }

    #[test]
    fn add_interest_not_found() {
        use crate::engine::{Engine, EngineConfig};
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let mut reg = InterlocutorRegistry::new();
        let concept = SymbolId::new(42).unwrap();
        let result = reg.add_interest("unknown", concept, &engine);
        assert!(result.is_err());
    }
}
