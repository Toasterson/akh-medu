//! Epistemic logic & theory of mind — Phase 19.
//!
//! Models what agents know, believe, and are ignorant about. Supports
//! dynamic epistemic logic (DEL) operations for updating beliefs when
//! new information arrives, and a theory-of-mind engine for predicting
//! other agents' behavior.
//!
//! ## Sub-phases
//!
//! - **19a**: Epistemic state representation (Knows/Believes/ConsidersPossible/IgnorantAbout)
//! - **19b**: Dynamic epistemic logic operations (announcements, messages, retractions)
//! - **19c**: Theory of mind engine (recursive belief modeling, behavior prediction)

use std::collections::HashMap;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::engine::Engine;
use crate::symbol::SymbolId;

// ═══════════════════════════════════════════════════════════════════════
// Error
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Error, Diagnostic)]
pub enum EpistemicError {
    #[error("agent not registered: {id}")]
    #[diagnostic(
        code(akh::agent::epistemic::agent_not_found),
        help("Register the agent via `register_agent()` first.")
    )]
    AgentNotFound { id: u64 },

    #[error("{0}")]
    #[diagnostic(
        code(akh::agent::epistemic::engine),
        help("An engine-level error occurred during epistemic reasoning.")
    )]
    Engine(Box<crate::error::AkhError>),
}

impl From<crate::error::AkhError> for EpistemicError {
    fn from(e: crate::error::AkhError) -> Self {
        Self::Engine(Box::new(e))
    }
}

pub type EpistemicResult<T> = std::result::Result<T, EpistemicError>;

// ═══════════════════════════════════════════════════════════════════════
// 19a — Epistemic State Representation
// ═══════════════════════════════════════════════════════════════════════

/// Epistemic modality: how an agent relates to a proposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EpistemicModality {
    /// Factual, justified, true in all accessible worlds.
    Knows,
    /// May be false, true in most accessible worlds.
    Believes,
    /// True in at least one accessible world.
    ConsidersPossible,
    /// Neither knows φ nor knows ¬φ.
    IgnorantAbout,
}

impl EpistemicModality {
    pub fn as_label(&self) -> &'static str {
        match self {
            Self::Knows => "knows",
            Self::Believes => "believes",
            Self::ConsidersPossible => "considers possible",
            Self::IgnorantAbout => "ignorant about",
        }
    }

    /// Confidence level associated with each modality.
    pub fn confidence(&self) -> f32 {
        match self {
            Self::Knows => 0.95,
            Self::Believes => 0.70,
            Self::ConsidersPossible => 0.40,
            Self::IgnorantAbout => 0.0,
        }
    }
}

/// A reference to a proposition in the knowledge graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PropositionRef {
    /// A specific triple (s, p, o).
    Triple {
        subject: SymbolId,
        predicate: SymbolId,
        object: SymbolId,
    },
    /// A single entity.
    Entity(SymbolId),
}

/// An epistemic proposition: agent + modality + proposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicProposition {
    /// The agent holding this epistemic state.
    pub agent: SymbolId,
    /// The modality (knows, believes, etc.).
    pub modality: EpistemicModality,
    /// The proposition.
    pub proposition: PropositionRef,
    /// Confidence in this assessment.
    pub confidence: f32,
    /// When this was last updated.
    pub timestamp: u64,
}

/// Epistemic profile for a single agent: what they know, believe, are ignorant about.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicProfile {
    /// Agent entity.
    pub agent_id: SymbolId,
    /// Known propositions.
    pub known: Vec<PropositionRef>,
    /// Believed propositions.
    pub believed: Vec<PropositionRef>,
    /// Ignorance set.
    pub ignorance: Vec<PropositionRef>,
    /// Last updated.
    pub updated_at: u64,
}

// ═══════════════════════════════════════════════════════════════════════
// 19b — Dynamic Epistemic Logic Operations
// ═══════════════════════════════════════════════════════════════════════

/// An epistemic event that changes what agents know.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EpistemicEvent {
    /// Public announcement: all audience members learn the proposition.
    PublicAnnouncement {
        announcer: SymbolId,
        proposition: PropositionRef,
        audience: Vec<SymbolId>,
    },
    /// Private message: only recipient learns.
    PrivateMessage {
        sender: SymbolId,
        recipient: SymbolId,
        proposition: PropositionRef,
    },
    /// Direct observation: high-confidence knowledge.
    Observation {
        observer: SymbolId,
        proposition: PropositionRef,
    },
    /// Potentially deceptive announcement (from unreliable source).
    DeceptiveAnnouncement {
        announcer: SymbolId,
        claimed: PropositionRef,
        audience: Vec<SymbolId>,
        source_reliability: f32,
    },
    /// Question: reveals asker's ignorance.
    Question {
        asker: SymbolId,
        about: PropositionRef,
    },
    /// Retraction: announcer takes back a proposition.
    Retraction {
        announcer: SymbolId,
        retracted: PropositionRef,
        audience: Vec<SymbolId>,
    },
}

/// Result of applying an epistemic event.
#[derive(Debug, Clone)]
pub struct EpistemicUpdateResult {
    /// Agents whose epistemic state changed.
    pub updated_agents: Vec<SymbolId>,
    /// New propositions added.
    pub new_propositions: Vec<EpistemicProposition>,
    /// Propositions whose modality changed.
    pub revised_propositions: Vec<(PropositionRef, EpistemicModality, EpistemicModality)>,
}

// ═══════════════════════════════════════════════════════════════════════
// 19c — Theory of Mind
// ═══════════════════════════════════════════════════════════════════════

/// Theory of Mind recursion depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ToMLevel {
    /// Non-strategic: random/heuristic behavior.
    Level0,
    /// Model what the other knows/believes, predict their behavior.
    Level1,
    /// Model what the other thinks WE know.
    Level2,
}

/// A theory-of-mind model of another agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToMModel {
    /// The agent being modeled.
    pub target_agent: SymbolId,
    /// Recursion depth.
    pub level: ToMLevel,
    /// What we think they know.
    pub their_knowledge: Vec<PropositionRef>,
    /// What we think they believe.
    pub their_beliefs: Vec<PropositionRef>,
    /// What we think their goals are.
    pub their_goals: Vec<String>,
    /// Predicted actions.
    pub predicted_actions: Vec<PredictedAction>,
    /// Confidence in the model.
    pub confidence: f32,
}

/// A predicted action from a ToM model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedAction {
    pub description: String,
    pub probability: f32,
    pub rationale: String,
}

/// Information advantage assessment.
#[derive(Debug, Clone)]
pub struct InformationAdvantage {
    /// Propositions we know but they don't.
    pub our_advantages: Vec<PropositionRef>,
    /// Propositions they know but we don't.
    pub their_advantages: Vec<PropositionRef>,
    /// Strategic assessment.
    pub assessment: StrategicAssessment,
}

/// Strategic assessment of information position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrategicAssessment {
    InformationSuperior,
    InformationParity,
    InformationInferior,
    Unknown,
}

impl StrategicAssessment {
    pub fn as_label(&self) -> &'static str {
        match self {
            Self::InformationSuperior => "information superior",
            Self::InformationParity => "information parity",
            Self::InformationInferior => "information inferior",
            Self::Unknown => "unknown",
        }
    }
}

/// Communication strategy based on ToM analysis.
#[derive(Debug, Clone)]
pub struct CommunicationStrategy {
    /// Propositions to share (helps our goals).
    pub share: Vec<PropositionRef>,
    /// Propositions to withhold (sharing would hurt).
    pub withhold: Vec<PropositionRef>,
    /// Propositions to ask about (fills our gaps).
    pub inquire: Vec<PropositionRef>,
    /// Deception is always rejected — log the reason.
    pub deception_rejected_reason: Option<String>,
    /// Confidence in the strategy.
    pub confidence: f32,
}

// ═══════════════════════════════════════════════════════════════════════
// EpistemicStateManager
// ═══════════════════════════════════════════════════════════════════════

/// Manages epistemic states for all known agents.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EpistemicStateManager {
    /// Per-agent epistemic profiles.
    pub profiles: HashMap<u64, EpistemicProfile>,
    /// Event log for replay/debugging.
    pub event_count: u64,
}

impl EpistemicStateManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an agent for epistemic tracking.
    pub fn register_agent(&mut self, agent_id: SymbolId) {
        self.profiles.entry(agent_id.get()).or_insert_with(|| {
            EpistemicProfile {
                agent_id,
                known: Vec::new(),
                believed: Vec::new(),
                ignorance: Vec::new(),
                updated_at: 0,
            }
        });
    }

    /// Assert that an agent knows a proposition.
    pub fn assert_knows(&mut self, agent: SymbolId, prop: PropositionRef) {
        let profile = self.profiles.entry(agent.get()).or_insert_with(|| {
            EpistemicProfile {
                agent_id: agent,
                known: Vec::new(),
                believed: Vec::new(),
                ignorance: Vec::new(),
                updated_at: 0,
            }
        });
        if !profile.known.contains(&prop) {
            profile.known.push(prop.clone());
        }
        // Remove from ignorance if present.
        profile.ignorance.retain(|p| p != &prop);
        profile.updated_at = now();
    }

    /// Assert that an agent believes a proposition (lower confidence than Knows).
    pub fn assert_believes(&mut self, agent: SymbolId, prop: PropositionRef) {
        let profile = self.profiles.entry(agent.get()).or_insert_with(|| {
            EpistemicProfile {
                agent_id: agent,
                known: Vec::new(),
                believed: Vec::new(),
                ignorance: Vec::new(),
                updated_at: 0,
            }
        });
        if !profile.believed.contains(&prop) && !profile.known.contains(&prop) {
            profile.believed.push(prop.clone());
        }
        profile.ignorance.retain(|p| p != &prop);
        profile.updated_at = now();
    }

    /// Mark that an agent is ignorant about a proposition.
    pub fn mark_ignorant(&mut self, agent: SymbolId, prop: PropositionRef) {
        let profile = self.profiles.entry(agent.get()).or_insert_with(|| {
            EpistemicProfile {
                agent_id: agent,
                known: Vec::new(),
                believed: Vec::new(),
                ignorance: Vec::new(),
                updated_at: 0,
            }
        });
        if !profile.ignorance.contains(&prop) {
            profile.ignorance.push(prop);
        }
        profile.updated_at = now();
    }

    /// Query what modality an agent has for a proposition.
    pub fn query_modality(&self, agent: SymbolId, prop: &PropositionRef) -> EpistemicModality {
        if let Some(profile) = self.profiles.get(&agent.get()) {
            if profile.known.contains(prop) {
                return EpistemicModality::Knows;
            }
            if profile.believed.contains(prop) {
                return EpistemicModality::Believes;
            }
            if profile.ignorance.contains(prop) {
                return EpistemicModality::IgnorantAbout;
            }
        }
        EpistemicModality::IgnorantAbout
    }

    /// Get an agent's full epistemic profile.
    pub fn get_profile(&self, agent: SymbolId) -> Option<&EpistemicProfile> {
        self.profiles.get(&agent.get())
    }

    /// Compute common knowledge between two agents.
    pub fn common_knowledge(
        &self,
        agent_a: SymbolId,
        agent_b: SymbolId,
    ) -> Vec<PropositionRef> {
        let a_known = self.profiles.get(&agent_a.get()).map(|p| &p.known);
        let b_known = self.profiles.get(&agent_b.get()).map(|p| &p.known);

        match (a_known, b_known) {
            (Some(a), Some(b)) => a.iter().filter(|p| b.contains(p)).cloned().collect(),
            _ => Vec::new(),
        }
    }

    /// Find information asymmetries: propositions one agent knows but the other doesn't.
    pub fn information_asymmetry(
        &self,
        agent_a: SymbolId,
        agent_b: SymbolId,
    ) -> InformationAdvantage {
        let a_known = self
            .profiles
            .get(&agent_a.get())
            .map(|p| &p.known)
            .cloned()
            .unwrap_or_default();
        let b_known = self
            .profiles
            .get(&agent_b.get())
            .map(|p| &p.known)
            .cloned()
            .unwrap_or_default();

        let our_advantages: Vec<_> = a_known.iter().filter(|p| !b_known.contains(p)).cloned().collect();
        let their_advantages: Vec<_> = b_known.iter().filter(|p| !a_known.contains(p)).cloned().collect();

        let assessment = if our_advantages.len() > their_advantages.len() + 2 {
            StrategicAssessment::InformationSuperior
        } else if their_advantages.len() > our_advantages.len() + 2 {
            StrategicAssessment::InformationInferior
        } else if our_advantages.is_empty() && their_advantages.is_empty() {
            StrategicAssessment::InformationParity
        } else {
            StrategicAssessment::InformationParity
        };

        InformationAdvantage {
            our_advantages,
            their_advantages,
            assessment,
        }
    }

    // ─── DEL Operations (19b) ─────────────────────────────────────

    /// Apply an epistemic event, updating all affected agents.
    pub fn apply_event(&mut self, event: &EpistemicEvent) -> EpistemicUpdateResult {
        self.event_count += 1;
        match event {
            EpistemicEvent::PublicAnnouncement {
                announcer: _,
                proposition,
                audience,
            } => {
                let mut updated = Vec::new();
                for &agent in audience {
                    self.assert_knows(agent, proposition.clone());
                    updated.push(agent);
                }
                EpistemicUpdateResult {
                    updated_agents: updated,
                    new_propositions: Vec::new(),
                    revised_propositions: Vec::new(),
                }
            }
            EpistemicEvent::PrivateMessage {
                sender: _,
                recipient,
                proposition,
            } => {
                self.assert_knows(*recipient, proposition.clone());
                EpistemicUpdateResult {
                    updated_agents: vec![*recipient],
                    new_propositions: Vec::new(),
                    revised_propositions: Vec::new(),
                }
            }
            EpistemicEvent::Observation {
                observer,
                proposition,
            } => {
                self.assert_knows(*observer, proposition.clone());
                EpistemicUpdateResult {
                    updated_agents: vec![*observer],
                    new_propositions: Vec::new(),
                    revised_propositions: Vec::new(),
                }
            }
            EpistemicEvent::DeceptiveAnnouncement {
                announcer: _,
                claimed,
                audience,
                source_reliability,
            } => {
                let mut updated = Vec::new();
                for &agent in audience {
                    if *source_reliability > 0.5 {
                        self.assert_believes(agent, claimed.clone());
                    }
                    // Low reliability → mark as "considers possible" (just believed, not known)
                    updated.push(agent);
                }
                EpistemicUpdateResult {
                    updated_agents: updated,
                    new_propositions: Vec::new(),
                    revised_propositions: Vec::new(),
                }
            }
            EpistemicEvent::Question { asker, about } => {
                self.mark_ignorant(*asker, about.clone());
                EpistemicUpdateResult {
                    updated_agents: vec![*asker],
                    new_propositions: Vec::new(),
                    revised_propositions: Vec::new(),
                }
            }
            EpistemicEvent::Retraction {
                announcer: _,
                retracted,
                audience,
            } => {
                let mut updated = Vec::new();
                let mut revised = Vec::new();
                for &agent in audience {
                    let old = self.query_modality(agent, retracted);
                    if let Some(profile) = self.profiles.get_mut(&agent.get()) {
                        profile.known.retain(|p| p != retracted);
                        profile.believed.retain(|p| p != retracted);
                        if !profile.ignorance.contains(retracted) {
                            profile.ignorance.push(retracted.clone());
                        }
                        profile.updated_at = now();
                    }
                    revised.push((retracted.clone(), old, EpistemicModality::IgnorantAbout));
                    updated.push(agent);
                }
                EpistemicUpdateResult {
                    updated_agents: updated,
                    new_propositions: Vec::new(),
                    revised_propositions: revised,
                }
            }
        }
    }

    /// Number of tracked agents.
    pub fn agent_count(&self) -> usize {
        self.profiles.len()
    }

    /// Persist to durable store.
    pub fn persist(&self, engine: &Engine) -> EpistemicResult<()> {
        let bytes = bincode::serialize(self).map_err(|e| {
            EpistemicError::Engine(Box::new(crate::error::AkhError::Store(
                crate::error::StoreError::Serialization {
                    message: format!("epistemic manager serialize: {e}"),
                },
            )))
        })?;
        engine
            .store()
            .put_meta(b"agent:epistemic_manager", &bytes)
            .map_err(|e| EpistemicError::Engine(Box::new(e.into())))?;
        Ok(())
    }

    /// Restore from durable store.
    pub fn restore(engine: &Engine) -> Self {
        engine
            .store()
            .get_meta(b"agent:epistemic_manager")
            .ok()
            .flatten()
            .and_then(|bytes| bincode::deserialize(&bytes).ok())
            .unwrap_or_default()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ToMEngine (19c)
// ═══════════════════════════════════════════════════════════════════════

/// Theory of Mind engine: models other agents' beliefs and predicts behavior.
#[derive(Debug, Clone)]
pub struct ToMEngine {
    /// Maximum recursion depth.
    pub max_level: ToMLevel,
}

impl Default for ToMEngine {
    fn default() -> Self {
        Self {
            max_level: ToMLevel::Level1,
        }
    }
}

impl ToMEngine {
    pub fn new(max_level: ToMLevel) -> Self {
        Self { max_level }
    }

    /// Build a theory-of-mind model of a target agent.
    pub fn build_model(
        &self,
        target: SymbolId,
        epistemic_mgr: &EpistemicStateManager,
    ) -> ToMModel {
        let profile = epistemic_mgr.get_profile(target);

        let (their_knowledge, their_beliefs) = match profile {
            Some(p) => (p.known.clone(), p.believed.clone()),
            None => (Vec::new(), Vec::new()),
        };

        ToMModel {
            target_agent: target,
            level: self.max_level,
            their_knowledge,
            their_beliefs,
            their_goals: Vec::new(), // inferred from interaction history
            predicted_actions: Vec::new(),
            confidence: if profile.is_some() { 0.6 } else { 0.2 },
        }
    }

    /// Compute information advantage between us and another agent.
    pub fn information_advantage(
        &self,
        us: SymbolId,
        them: SymbolId,
        epistemic_mgr: &EpistemicStateManager,
    ) -> InformationAdvantage {
        epistemic_mgr.information_asymmetry(us, them)
    }

    /// Plan communication strategy based on ToM analysis.
    pub fn plan_communication(
        &self,
        model: &ToMModel,
        our_knowledge: &[PropositionRef],
    ) -> CommunicationStrategy {
        // Share: what we know that they don't (and is not sensitive).
        let share: Vec<_> = our_knowledge
            .iter()
            .filter(|p| !model.their_knowledge.contains(p))
            .take(5)
            .cloned()
            .collect();

        // Inquire: what they might know that we don't.
        let inquire: Vec<_> = model
            .their_knowledge
            .iter()
            .filter(|p| !our_knowledge.contains(p))
            .take(5)
            .cloned()
            .collect();

        CommunicationStrategy {
            share,
            withhold: Vec::new(), // conservative: don't withhold by default
            inquire,
            deception_rejected_reason: Some(
                "Deception violates integrity principle — always rejected".into(),
            ),
            confidence: model.confidence,
        }
    }
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(id: u64) -> SymbolId {
        SymbolId::new(id).unwrap()
    }

    fn prop_entity(id: u64) -> PropositionRef {
        PropositionRef::Entity(sym(id))
    }

    fn prop_triple(s: u64, p: u64, o: u64) -> PropositionRef {
        PropositionRef::Triple {
            subject: sym(s),
            predicate: sym(p),
            object: sym(o),
        }
    }

    // ── 19a: Epistemic State ──────────────────────────────────────

    #[test]
    fn modality_confidence_ordering() {
        assert!(EpistemicModality::Knows.confidence() > EpistemicModality::Believes.confidence());
        assert!(EpistemicModality::Believes.confidence() > EpistemicModality::ConsidersPossible.confidence());
        assert_eq!(EpistemicModality::IgnorantAbout.confidence(), 0.0);
    }

    #[test]
    fn assert_knows_adds_to_profile() {
        let mut mgr = EpistemicStateManager::new();
        let agent = sym(1);
        let prop = prop_entity(42);

        mgr.assert_knows(agent, prop.clone());

        let profile = mgr.get_profile(agent).unwrap();
        assert!(profile.known.contains(&prop));
    }

    #[test]
    fn assert_believes_not_duplicate_of_knows() {
        let mut mgr = EpistemicStateManager::new();
        let agent = sym(1);
        let prop = prop_entity(42);

        mgr.assert_knows(agent, prop.clone());
        mgr.assert_believes(agent, prop.clone());

        let profile = mgr.get_profile(agent).unwrap();
        // Should be in known, not believed (knows is stronger).
        assert!(profile.known.contains(&prop));
        assert!(!profile.believed.contains(&prop));
    }

    #[test]
    fn query_modality_knows() {
        let mut mgr = EpistemicStateManager::new();
        let agent = sym(1);
        let prop = prop_entity(42);
        mgr.assert_knows(agent, prop.clone());
        assert_eq!(mgr.query_modality(agent, &prop), EpistemicModality::Knows);
    }

    #[test]
    fn query_modality_ignorant_default() {
        let mgr = EpistemicStateManager::new();
        let prop = prop_entity(42);
        assert_eq!(
            mgr.query_modality(sym(99), &prop),
            EpistemicModality::IgnorantAbout
        );
    }

    #[test]
    fn common_knowledge_intersection() {
        let mut mgr = EpistemicStateManager::new();
        let a = sym(1);
        let b = sym(2);
        let p1 = prop_entity(10);
        let p2 = prop_entity(11);
        let p3 = prop_entity(12);

        mgr.assert_knows(a, p1.clone());
        mgr.assert_knows(a, p2.clone());
        mgr.assert_knows(b, p2.clone());
        mgr.assert_knows(b, p3.clone());

        let common = mgr.common_knowledge(a, b);
        assert_eq!(common.len(), 1);
        assert!(common.contains(&p2));
    }

    #[test]
    fn information_asymmetry_finds_gaps() {
        let mut mgr = EpistemicStateManager::new();
        let a = sym(1);
        let b = sym(2);

        mgr.assert_knows(a, prop_entity(10));
        mgr.assert_knows(a, prop_entity(11));
        mgr.assert_knows(b, prop_entity(12));

        let adv = mgr.information_asymmetry(a, b);
        assert_eq!(adv.our_advantages.len(), 2);
        assert_eq!(adv.their_advantages.len(), 1);
        assert_eq!(adv.assessment, StrategicAssessment::InformationParity);
    }

    // ── 19b: DEL Operations ───────────────────────────────────────

    #[test]
    fn public_announcement_updates_all() {
        let mut mgr = EpistemicStateManager::new();
        let a = sym(1);
        let b = sym(2);
        let prop = prop_entity(42);

        let event = EpistemicEvent::PublicAnnouncement {
            announcer: sym(99),
            proposition: prop.clone(),
            audience: vec![a, b],
        };

        let result = mgr.apply_event(&event);
        assert_eq!(result.updated_agents.len(), 2);
        assert_eq!(mgr.query_modality(a, &prop), EpistemicModality::Knows);
        assert_eq!(mgr.query_modality(b, &prop), EpistemicModality::Knows);
    }

    #[test]
    fn private_message_only_recipient() {
        let mut mgr = EpistemicStateManager::new();
        let sender = sym(1);
        let recipient = sym(2);
        let eavesdropper = sym(3);
        let prop = prop_entity(42);

        mgr.register_agent(eavesdropper);

        let event = EpistemicEvent::PrivateMessage {
            sender,
            recipient,
            proposition: prop.clone(),
        };

        mgr.apply_event(&event);
        assert_eq!(mgr.query_modality(recipient, &prop), EpistemicModality::Knows);
        assert_eq!(
            mgr.query_modality(eavesdropper, &prop),
            EpistemicModality::IgnorantAbout
        );
    }

    #[test]
    fn question_reveals_ignorance() {
        let mut mgr = EpistemicStateManager::new();
        let asker = sym(1);
        let prop = prop_entity(42);

        let event = EpistemicEvent::Question {
            asker,
            about: prop.clone(),
        };

        mgr.apply_event(&event);
        assert_eq!(
            mgr.query_modality(asker, &prop),
            EpistemicModality::IgnorantAbout
        );
    }

    #[test]
    fn retraction_moves_to_ignorant() {
        let mut mgr = EpistemicStateManager::new();
        let agent = sym(1);
        let prop = prop_entity(42);

        mgr.assert_knows(agent, prop.clone());
        assert_eq!(mgr.query_modality(agent, &prop), EpistemicModality::Knows);

        let event = EpistemicEvent::Retraction {
            announcer: sym(99),
            retracted: prop.clone(),
            audience: vec![agent],
        };

        let result = mgr.apply_event(&event);
        assert_eq!(result.revised_propositions.len(), 1);
        assert_eq!(
            mgr.query_modality(agent, &prop),
            EpistemicModality::IgnorantAbout
        );
    }

    #[test]
    fn deceptive_announcement_uses_reliability() {
        let mut mgr = EpistemicStateManager::new();
        let agent = sym(1);
        let prop = prop_entity(42);

        // High reliability → believed.
        let event = EpistemicEvent::DeceptiveAnnouncement {
            announcer: sym(99),
            claimed: prop.clone(),
            audience: vec![agent],
            source_reliability: 0.8,
        };
        mgr.apply_event(&event);
        assert_eq!(mgr.query_modality(agent, &prop), EpistemicModality::Believes);
    }

    // ── 19c: Theory of Mind ───────────────────────────────────────

    #[test]
    fn tom_build_model() {
        let mut mgr = EpistemicStateManager::new();
        let target = sym(10);
        mgr.assert_knows(target, prop_entity(42));
        mgr.assert_believes(target, prop_entity(43));

        let tom = ToMEngine::default();
        let model = tom.build_model(target, &mgr);

        assert_eq!(model.target_agent, target);
        assert_eq!(model.their_knowledge.len(), 1);
        assert_eq!(model.their_beliefs.len(), 1);
        assert!(model.confidence > 0.5);
    }

    #[test]
    fn tom_unknown_agent_low_confidence() {
        let mgr = EpistemicStateManager::new();
        let tom = ToMEngine::default();
        let model = tom.build_model(sym(99), &mgr);
        assert!(model.confidence < 0.5);
    }

    #[test]
    fn communication_strategy_deception_rejected() {
        let mgr = EpistemicStateManager::new();
        let tom = ToMEngine::default();
        let model = tom.build_model(sym(10), &mgr);
        let strategy = tom.plan_communication(&model, &[]);
        assert!(strategy.deception_rejected_reason.is_some());
    }

    #[test]
    fn serialization_roundtrip() {
        let mut mgr = EpistemicStateManager::new();
        mgr.assert_knows(sym(1), prop_entity(42));
        mgr.assert_believes(sym(2), prop_triple(1, 2, 3));

        let bytes = bincode::serialize(&mgr).unwrap();
        let restored: EpistemicStateManager = bincode::deserialize(&bytes).unwrap();
        assert_eq!(restored.profiles.len(), 2);
    }

    #[test]
    fn strategic_assessment_labels() {
        assert_eq!(StrategicAssessment::InformationSuperior.as_label(), "information superior");
        assert_eq!(StrategicAssessment::Unknown.as_label(), "unknown");
    }
}
