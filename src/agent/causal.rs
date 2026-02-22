//! Causal world model — Phase 15a.
//!
//! Explicit causal representation: cause-and-effect predicates, action schemas
//! with preconditions and effects, state transition prediction, and outcome
//! verification.  The causal model enables the agent to predict what will happen
//! before acting and to learn from the accuracy of its predictions.

use std::collections::HashMap;
use std::fmt;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::DerivationKind;
use crate::symbol::SymbolId;
use crate::vsa::encode::encode_token;
use crate::vsa::ops::VsaOps;
use crate::vsa::HyperVec;

// ═══════════════════════════════════════════════════════════════════════
// Error
// ═══════════════════════════════════════════════════════════════════════

/// Errors specific to the causal reasoning subsystem.
#[derive(Debug, Error, Diagnostic)]
pub enum CausalError {
    #[error("action schema not found: {name}")]
    #[diagnostic(
        code(akh::agent::causal::schema_not_found),
        help("Register an action schema with `causal_manager.register_action_schema()`.")
    )]
    SchemaNotFound { name: String },

    #[error("no applicable actions in current state")]
    #[diagnostic(
        code(akh::agent::causal::no_applicable),
        help("The current KG state does not satisfy any action preconditions.")
    )]
    NoApplicableActions,

    #[error("pattern binding failed: unresolved variable `{variable}`")]
    #[diagnostic(
        code(akh::agent::causal::binding_failed),
        help("Ensure all variables in effects appear in preconditions.")
    )]
    BindingFailed { variable: String },

    #[error("{0}")]
    #[diagnostic(
        code(akh::agent::causal::engine),
        help("An engine-level error occurred during causal reasoning.")
    )]
    Engine(Box<crate::error::AkhError>),
}

impl From<crate::error::AkhError> for CausalError {
    fn from(e: crate::error::AkhError) -> Self {
        Self::Engine(Box::new(e))
    }
}

/// Result alias for the causal subsystem.
pub type CausalResult<T> = std::result::Result<T, CausalError>;

// ═══════════════════════════════════════════════════════════════════════
// CausalRelation
// ═══════════════════════════════════════════════════════════════════════

/// Types of causal relation between entities or events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CausalRelation {
    /// A causes B (sufficient condition).
    Causes,
    /// A enables B (necessary but not sufficient).
    Enables,
    /// A prevents B (if A then not B).
    Prevents,
    /// A inhibits B (weakens but doesn't fully prevent).
    Inhibits,
}

impl CausalRelation {
    pub fn as_label(&self) -> &'static str {
        match self {
            Self::Causes => "causes",
            Self::Enables => "enables",
            Self::Prevents => "prevents",
            Self::Inhibits => "inhibits",
        }
    }

    pub fn from_label(s: &str) -> Option<Self> {
        match s {
            "causes" => Some(Self::Causes),
            "enables" => Some(Self::Enables),
            "prevents" => Some(Self::Prevents),
            "inhibits" => Some(Self::Inhibits),
            _ => None,
        }
    }
}

impl fmt::Display for CausalRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_label())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CausalPredicates
// ═══════════════════════════════════════════════════════════════════════

/// Well-known KG predicates for causal reasoning (namespace: `causal:`).
#[derive(Debug, Clone)]
pub struct CausalPredicates {
    pub causes: SymbolId,
    pub enables: SymbolId,
    pub prevents: SymbolId,
    pub inhibits: SymbolId,
    pub has_precondition: SymbolId,
    pub has_effect: SymbolId,
    pub has_expected_outcome: SymbolId,
    pub causal_strength: SymbolId,
}

impl CausalPredicates {
    /// Resolve or create all causal predicates.
    pub fn init(engine: &Engine) -> CausalResult<Self> {
        Ok(Self {
            causes: engine.resolve_or_create_relation("causal:causes")?,
            enables: engine.resolve_or_create_relation("causal:enables")?,
            prevents: engine.resolve_or_create_relation("causal:prevents")?,
            inhibits: engine.resolve_or_create_relation("causal:inhibits")?,
            has_precondition: engine.resolve_or_create_relation("causal:has-precondition")?,
            has_effect: engine.resolve_or_create_relation("causal:has-effect")?,
            has_expected_outcome: engine
                .resolve_or_create_relation("causal:has-expected-outcome")?,
            causal_strength: engine.resolve_or_create_relation("causal:causal-strength")?,
        })
    }

    /// Return the predicate SymbolId for a given relation type.
    pub fn predicate_for(&self, relation: CausalRelation) -> SymbolId {
        match relation {
            CausalRelation::Causes => self.causes,
            CausalRelation::Enables => self.enables,
            CausalRelation::Prevents => self.prevents,
            CausalRelation::Inhibits => self.inhibits,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Pattern & Effect Types
// ═══════════════════════════════════════════════════════════════════════

/// An element in a causal pattern: concrete symbol, named variable, or wildcard.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PatternElement {
    /// A specific SymbolId.
    Concrete(SymbolId),
    /// A named variable bound during matching.
    Variable(String),
    /// Match anything.
    Wildcard,
}

impl PatternElement {
    /// Resolve this element against a binding environment.
    pub fn resolve(&self, bindings: &HashMap<String, SymbolId>) -> Option<SymbolId> {
        match self {
            Self::Concrete(id) => Some(*id),
            Self::Variable(name) => bindings.get(name).copied(),
            Self::Wildcard => None,
        }
    }

    /// Check whether this element matches a concrete SymbolId,
    /// potentially adding a binding.
    pub fn matches(
        &self,
        actual: SymbolId,
        bindings: &mut HashMap<String, SymbolId>,
    ) -> bool {
        match self {
            Self::Concrete(id) => *id == actual,
            Self::Variable(name) => {
                if let Some(&bound) = bindings.get(name) {
                    bound == actual
                } else {
                    bindings.insert(name.clone(), actual);
                    true
                }
            }
            Self::Wildcard => true,
        }
    }
}

/// A pattern for matching precondition triples.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CausalPattern {
    pub subject: PatternElement,
    pub predicate: PatternElement,
    pub object: PatternElement,
    /// If true, the pattern must NOT match (negation-as-failure).
    pub negated: bool,
}

impl CausalPattern {
    /// Check whether this pattern matches any triple in the KG.
    pub fn matches_any(
        &self,
        triples: &[Triple],
        bindings: &mut HashMap<String, SymbolId>,
    ) -> bool {
        let found = triples.iter().any(|t| {
            let mut trial_bindings = bindings.clone();
            self.subject.matches(t.subject, &mut trial_bindings)
                && self.predicate.matches(t.predicate, &mut trial_bindings)
                && self.object.matches(t.object, &mut trial_bindings)
                && {
                    // Only commit bindings if all three matched.
                    *bindings = trial_bindings;
                    true
                }
        });
        if self.negated { !found } else { found }
    }
}

/// The kind of effect an action has on the KG.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EffectKind {
    /// Triple is added to the KG.
    Assert,
    /// Triple is removed from the KG.
    Retract,
    /// Confidence of an existing triple is modified by delta.
    ModifyConfidence { delta: f32 },
}

/// An effect that an action has on the KG state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CausalEffect {
    pub kind: EffectKind,
    pub subject: PatternElement,
    pub predicate: PatternElement,
    pub object: PatternElement,
    /// Expected confidence of the resulting triple.
    pub confidence: f32,
}

// ═══════════════════════════════════════════════════════════════════════
// ActionSchema
// ═══════════════════════════════════════════════════════════════════════

/// An action schema describing what a tool/action does to world state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionSchema {
    /// Symbol for the action (usually a tool-name entity).
    pub action_id: SymbolId,
    /// Human-readable name (matches tool name).
    pub name: String,
    /// Preconditions: triples that must hold for this action to be applicable.
    pub preconditions: Vec<CausalPattern>,
    /// Effects: triples that will be added/removed after execution.
    pub effects: Vec<CausalEffect>,
    /// Observed success rate (updated after each execution).
    pub success_rate: f32,
    /// Number of times this action has been executed.
    pub execution_count: u32,
}

impl ActionSchema {
    /// Check whether all preconditions are satisfied against the current KG state.
    pub fn is_applicable(&self, triples: &[Triple]) -> bool {
        let mut bindings = HashMap::new();
        self.preconditions
            .iter()
            .all(|p| p.matches_any(triples, &mut bindings))
    }

    /// Predict the state transition from applying this action.
    pub fn predict_effects(
        &self,
        triples: &[Triple],
    ) -> CausalResult<StateTransition> {
        let mut bindings = HashMap::new();
        for p in &self.preconditions {
            p.matches_any(triples, &mut bindings);
        }

        let mut assertions = Vec::new();
        let mut retractions = Vec::new();
        let mut confidence_changes = Vec::new();

        for eff in &self.effects {
            let s = eff.subject.resolve(&bindings);
            let p = eff.predicate.resolve(&bindings);
            let o = eff.object.resolve(&bindings);

            // Skip effects with unresolvable variables.
            let (Some(s), Some(p), Some(o)) = (s, p, o) else {
                continue;
            };

            match eff.kind {
                EffectKind::Assert => assertions.push((s, p, o)),
                EffectKind::Retract => retractions.push((s, p, o)),
                EffectKind::ModifyConfidence { delta } => {
                    confidence_changes.push((s, p, o, delta));
                }
            }
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Ok(StateTransition {
            action_id: self.action_id,
            assertions,
            retractions,
            confidence_changes,
            verified: None,
            timestamp: now,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════
// StateTransition
// ═══════════════════════════════════════════════════════════════════════

/// A predicted state transition: before → action → after.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    /// The action that caused this transition.
    pub action_id: SymbolId,
    /// Triples added by this action.
    pub assertions: Vec<(SymbolId, SymbolId, SymbolId)>,
    /// Triples removed by this action.
    pub retractions: Vec<(SymbolId, SymbolId, SymbolId)>,
    /// Confidence modifications.
    pub confidence_changes: Vec<(SymbolId, SymbolId, SymbolId, f32)>,
    /// Whether the prediction was verified against actual outcome.
    pub verified: Option<bool>,
    /// Timestamp.
    pub timestamp: u64,
}

// ═══════════════════════════════════════════════════════════════════════
// CausalRoleVectors
// ═══════════════════════════════════════════════════════════════════════

/// VSA role vectors for encoding causal state-action pairs.
#[derive(Debug, Clone)]
pub struct CausalRoleVectors {
    pub state: HyperVec,
    pub action: HyperVec,
    pub outcome: HyperVec,
    pub precondition: HyperVec,
    pub effect: HyperVec,
    pub strength: HyperVec,
}

impl CausalRoleVectors {
    /// Create role vectors from deterministic tokens.
    pub fn new(ops: &VsaOps) -> Self {
        Self {
            state: encode_token(ops, "causal-role:state"),
            action: encode_token(ops, "causal-role:action"),
            outcome: encode_token(ops, "causal-role:outcome"),
            precondition: encode_token(ops, "causal-role:precondition"),
            effect: encode_token(ops, "causal-role:effect"),
            strength: encode_token(ops, "causal-role:strength"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CausalManager
// ═══════════════════════════════════════════════════════════════════════

/// Manages the causal world model: action schemas, predictions, verification.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CausalManager {
    /// Registered action schemas, keyed by name.
    pub schemas: HashMap<String, ActionSchema>,
    /// Lazily initialized predicates (not serialized).
    #[serde(skip)]
    predicates: Option<CausalPredicates>,
    /// Lazily initialized role vectors (not serialized).
    #[serde(skip)]
    role_vectors: Option<CausalRoleVectors>,
}

impl CausalManager {
    /// Create a new causal manager and initialize KG predicates.
    pub fn new(engine: &Engine) -> CausalResult<Self> {
        let mut mgr = Self::default();
        mgr.ensure_init(engine)?;
        Ok(mgr)
    }

    /// Ensure predicates and role vectors are initialized.
    pub fn ensure_init(&mut self, engine: &Engine) -> CausalResult<()> {
        if self.predicates.is_none() {
            self.predicates = Some(CausalPredicates::init(engine)?);
        }
        if self.role_vectors.is_none() {
            self.role_vectors = Some(CausalRoleVectors::new(engine.ops()));
        }
        Ok(())
    }

    /// Restore from durable store.
    pub fn restore(engine: &Engine) -> CausalResult<Self> {
        let data = engine
            .store()
            .get_meta(b"agent:causal_manager")
            .map_err(|e| CausalError::Engine(Box::new(e.into())))?;
        match data {
            Some(bytes) if !bytes.is_empty() => {
                let mut mgr: Self =
                    bincode::deserialize(&bytes).map_err(|e| CausalError::Engine(Box::new(
                        crate::error::AkhError::Store(crate::error::StoreError::Serialization {
                            message: format!("causal manager deserialize: {e}"),
                        }),
                    )))?;
                mgr.ensure_init(engine)?;
                Ok(mgr)
            }
            _ => Self::new(engine),
        }
    }

    /// Persist to durable store.
    pub fn persist(&self, engine: &Engine) -> CausalResult<()> {
        let bytes = bincode::serialize(self).map_err(|e| {
            CausalError::Engine(Box::new(crate::error::AkhError::Store(
                crate::error::StoreError::Serialization {
                    message: format!("causal manager serialize: {e}"),
                },
            )))
        })?;
        engine
            .store()
            .put_meta(b"agent:causal_manager", &bytes)
            .map_err(|e| CausalError::Engine(Box::new(e.into())))?;
        Ok(())
    }

    // ─── Schema Management ─────────────────────────────────────────

    /// Register or update an action schema.
    pub fn register_action_schema(&mut self, schema: ActionSchema) {
        self.schemas.insert(schema.name.clone(), schema);
    }

    /// Get a schema by name.
    pub fn get_schema(&self, name: &str) -> Option<&ActionSchema> {
        self.schemas.get(name)
    }

    /// List all registered schemas.
    pub fn list_schemas(&self) -> Vec<&ActionSchema> {
        self.schemas.values().collect()
    }

    /// Bootstrap action schemas from the tool registry.
    ///
    /// Creates a minimal schema for each tool with no preconditions
    /// and a single generic "executed" effect.  These are refined over
    /// time as predictions are verified.
    pub fn bootstrap_schemas_from_tools(
        &mut self,
        tool_names: &[String],
        engine: &Engine,
    ) -> CausalResult<usize> {
        let mut count = 0;
        for name in tool_names {
            if self.schemas.contains_key(name) {
                continue;
            }
            let action_id = engine.resolve_or_create_entity(&format!("tool:{name}"))?;
            let schema = ActionSchema {
                action_id,
                name: name.clone(),
                preconditions: Vec::new(),
                effects: Vec::new(),
                success_rate: 0.5, // uninformed prior
                execution_count: 0,
            };
            self.schemas.insert(name.clone(), schema);
            count += 1;
        }
        Ok(count)
    }

    // ─── Prediction ────────────────────────────────────────────────

    /// Return all action schemas whose preconditions are satisfied.
    pub fn applicable_actions(&self, engine: &Engine) -> Vec<&ActionSchema> {
        let triples = engine.all_triples();
        self.schemas
            .values()
            .filter(|s| s.preconditions.is_empty() || s.is_applicable(&triples))
            .collect()
    }

    /// Predict effects of executing an action.
    pub fn predict_effects(
        &self,
        action_name: &str,
        engine: &Engine,
    ) -> CausalResult<StateTransition> {
        let schema = self
            .schemas
            .get(action_name)
            .ok_or_else(|| CausalError::SchemaNotFound {
                name: action_name.into(),
            })?;
        let triples = engine.all_triples();
        schema.predict_effects(&triples)
    }

    /// Verify a prediction against the actual state after execution.
    /// Returns true if the prediction was accurate.
    pub fn verify_prediction(
        &self,
        transition: &StateTransition,
        engine: &Engine,
    ) -> bool {
        let triples = engine.all_triples();

        // Check assertions: were the predicted triples actually added?
        let assertions_ok = transition.assertions.iter().all(|(s, p, o)| {
            triples
                .iter()
                .any(|t| t.subject == *s && t.predicate == *p && t.object == *o)
        });

        // Check retractions: were the predicted triples actually removed?
        let retractions_ok = transition.retractions.iter().all(|(s, p, o)| {
            !triples
                .iter()
                .any(|t| t.subject == *s && t.predicate == *p && t.object == *o)
        });

        assertions_ok && retractions_ok
    }

    /// Update a schema's statistics after observing an outcome.
    pub fn update_schema_from_outcome(
        &mut self,
        action_name: &str,
        success: bool,
    ) {
        if let Some(schema) = self.schemas.get_mut(action_name) {
            schema.execution_count += 1;
            // Exponential moving average for success rate.
            let alpha = 1.0 / (1.0 + schema.execution_count as f32 * 0.1);
            let outcome = if success { 1.0 } else { 0.0 };
            schema.success_rate = schema.success_rate * (1.0 - alpha) + outcome * alpha;
        }
    }

    // ─── Causal Strength ───────────────────────────────────────────

    /// Compute the causal strength between cause and effect entities.
    ///
    /// Looks for direct causal links and transitive chains (depth 2).
    pub fn causal_strength(
        &self,
        cause: SymbolId,
        effect: SymbolId,
        engine: &Engine,
    ) -> f32 {
        let preds = match &self.predicates {
            Some(p) => p,
            None => return 0.0,
        };

        // Direct link?
        let triples = engine.triples_from(cause);
        for t in &triples {
            if t.object == effect && t.predicate == preds.causes {
                return t.confidence;
            }
        }

        // Transitive (depth 2): cause → intermediate → effect
        let mut best = 0.0_f32;
        for t1 in &triples {
            if t1.predicate == preds.causes || t1.predicate == preds.enables {
                let intermediate_triples = engine.triples_from(t1.object);
                for t2 in &intermediate_triples {
                    if t2.object == effect && t2.predicate == preds.causes {
                        let transitive =
                            t1.confidence * t2.confidence * 0.8;
                        best = best.max(transitive);
                    }
                }
            }
        }
        best
    }

    // ─── VSA Encoding ──────────────────────────────────────────────

    /// Encode a state-action pair as a HyperVec for similarity lookup.
    pub fn encode_state_action(
        &self,
        ops: &VsaOps,
        state_triples: &[Triple],
        action_id: SymbolId,
    ) -> Option<HyperVec> {
        let roles = self.role_vectors.as_ref()?;

        // Encode state: bundle up to 20 most-recent triples.
        let state_vecs: Vec<HyperVec> = state_triples
            .iter()
            .rev()
            .take(20)
            .filter_map(|t| {
                let s = crate::vsa::encode::encode_symbol(ops, t.subject);
                let p = crate::vsa::encode::encode_symbol(ops, t.predicate);
                ops.bind(&s, &p).ok()
            })
            .collect();

        if state_vecs.is_empty() {
            return None;
        }
        let state_refs: Vec<&HyperVec> = state_vecs.iter().collect();
        let state_bundle = ops.bundle(&state_refs).ok()?;

        // Encode action.
        let action_vec = crate::vsa::encode::encode_symbol(ops, action_id);

        // Bind state with state-role and action with action-role, then bundle.
        let state_bound = ops.bind(&roles.state, &state_bundle).ok()?;
        let action_bound = ops.bind(&roles.action, &action_vec).ok()?;

        ops.bundle(&[&state_bound, &action_bound]).ok()
    }

    // ─── Provenance ────────────────────────────────────────────────

    /// Record provenance for a learned causal schema.
    pub fn record_schema_provenance(
        &self,
        engine: &Engine,
        action_name: &str,
        precondition_count: usize,
        effect_count: usize,
    ) -> CausalResult<()> {
        let schema = self
            .schemas
            .get(action_name)
            .ok_or_else(|| CausalError::SchemaNotFound {
                name: action_name.into(),
            })?;

        let mut record = crate::provenance::ProvenanceRecord::new(
            schema.action_id,
            DerivationKind::CausalSchemaLearned {
                action_name: action_name.to_string(),
                precondition_count: precondition_count as u32,
                effect_count: effect_count as u32,
            },
        )
        .with_confidence(schema.success_rate);
        engine
            .store_provenance(&mut record)
            .map_err(|e| CausalError::Engine(Box::new(e)))?;
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── CausalRelation ─────────────────────────────────────────────

    #[test]
    fn causal_relation_labels_roundtrip() {
        for rel in [
            CausalRelation::Causes,
            CausalRelation::Enables,
            CausalRelation::Prevents,
            CausalRelation::Inhibits,
        ] {
            let label = rel.as_label();
            assert_eq!(CausalRelation::from_label(label), Some(rel));
        }
    }

    #[test]
    fn causal_relation_display() {
        assert_eq!(format!("{}", CausalRelation::Causes), "causes");
        assert_eq!(format!("{}", CausalRelation::Prevents), "prevents");
    }

    #[test]
    fn causal_relation_from_label_unknown() {
        assert_eq!(CausalRelation::from_label("unknown"), None);
    }

    // ── PatternElement ─────────────────────────────────────────────

    #[test]
    fn pattern_element_concrete_matches_exact() {
        let id = SymbolId::new(42).unwrap();
        let elem = PatternElement::Concrete(id);
        let mut bindings = HashMap::new();
        assert!(elem.matches(id, &mut bindings));
        assert!(!elem.matches(SymbolId::new(99).unwrap(), &mut bindings));
    }

    #[test]
    fn pattern_element_variable_binds_and_checks() {
        let id = SymbolId::new(42).unwrap();
        let elem = PatternElement::Variable("x".into());
        let mut bindings = HashMap::new();

        // First match: binds.
        assert!(elem.matches(id, &mut bindings));
        assert_eq!(bindings.get("x"), Some(&id));

        // Second match with same value: succeeds.
        assert!(elem.matches(id, &mut bindings));

        // Different value: fails.
        assert!(!elem.matches(SymbolId::new(99).unwrap(), &mut bindings));
    }

    #[test]
    fn pattern_element_wildcard_matches_anything() {
        let elem = PatternElement::Wildcard;
        let mut bindings = HashMap::new();
        assert!(elem.matches(SymbolId::new(1).unwrap(), &mut bindings));
        assert!(elem.matches(SymbolId::new(999).unwrap(), &mut bindings));
        assert!(bindings.is_empty());
    }

    #[test]
    fn pattern_element_resolve_concrete() {
        let id = SymbolId::new(42).unwrap();
        let elem = PatternElement::Concrete(id);
        assert_eq!(elem.resolve(&HashMap::new()), Some(id));
    }

    #[test]
    fn pattern_element_resolve_variable() {
        let id = SymbolId::new(42).unwrap();
        let elem = PatternElement::Variable("x".into());
        let mut bindings = HashMap::new();
        assert_eq!(elem.resolve(&bindings), None);
        bindings.insert("x".into(), id);
        assert_eq!(elem.resolve(&bindings), Some(id));
    }

    // ── CausalPattern ──────────────────────────────────────────────

    #[test]
    fn causal_pattern_matches_triple() {
        let s = SymbolId::new(1).unwrap();
        let p = SymbolId::new(2).unwrap();
        let o = SymbolId::new(3).unwrap();

        let triple = Triple::new(s, p, o).with_confidence(0.9);

        let pattern = CausalPattern {
            subject: PatternElement::Concrete(s),
            predicate: PatternElement::Wildcard,
            object: PatternElement::Variable("target".into()),
            negated: false,
        };

        let mut bindings = HashMap::new();
        assert!(pattern.matches_any(&[triple], &mut bindings));
        assert_eq!(bindings.get("target"), Some(&o));
    }

    #[test]
    fn causal_pattern_negated_no_match() {
        let s = SymbolId::new(1).unwrap();
        let p = SymbolId::new(2).unwrap();
        let o = SymbolId::new(3).unwrap();

        let triple = Triple::new(s, p, o).with_confidence(0.9);

        let pattern = CausalPattern {
            subject: PatternElement::Concrete(s),
            predicate: PatternElement::Concrete(p),
            object: PatternElement::Concrete(o),
            negated: true,
        };

        let mut bindings = HashMap::new();
        // Triple exists but pattern is negated, so matches_any returns false.
        assert!(!pattern.matches_any(&[triple], &mut bindings));
    }

    // ── ActionSchema ───────────────────────────────────────────────

    #[test]
    fn action_schema_applicable_no_preconditions() {
        let schema = ActionSchema {
            action_id: SymbolId::new(10).unwrap(),
            name: "test".into(),
            preconditions: Vec::new(),
            effects: Vec::new(),
            success_rate: 0.5,
            execution_count: 0,
        };
        assert!(schema.is_applicable(&[]));
    }

    #[test]
    fn action_schema_applicable_with_preconditions() {
        let s = SymbolId::new(1).unwrap();
        let p = SymbolId::new(2).unwrap();
        let o = SymbolId::new(3).unwrap();

        let schema = ActionSchema {
            action_id: SymbolId::new(10).unwrap(),
            name: "test".into(),
            preconditions: vec![CausalPattern {
                subject: PatternElement::Concrete(s),
                predicate: PatternElement::Concrete(p),
                object: PatternElement::Wildcard,
                negated: false,
            }],
            effects: Vec::new(),
            success_rate: 0.5,
            execution_count: 0,
        };

        let triple = Triple::new(s, p, o).with_confidence(0.9);

        assert!(schema.is_applicable(&[triple]));
        assert!(!schema.is_applicable(&[]));
    }

    #[test]
    fn action_schema_predict_effects_assert() {
        let s = SymbolId::new(1).unwrap();
        let p = SymbolId::new(2).unwrap();
        let o = SymbolId::new(3).unwrap();
        let new_o = SymbolId::new(4).unwrap();

        let schema = ActionSchema {
            action_id: SymbolId::new(10).unwrap(),
            name: "test".into(),
            preconditions: vec![CausalPattern {
                subject: PatternElement::Variable("x".into()),
                predicate: PatternElement::Concrete(p),
                object: PatternElement::Wildcard,
                negated: false,
            }],
            effects: vec![CausalEffect {
                kind: EffectKind::Assert,
                subject: PatternElement::Variable("x".into()),
                predicate: PatternElement::Concrete(p),
                object: PatternElement::Concrete(new_o),
                confidence: 0.8,
            }],
            success_rate: 0.5,
            execution_count: 0,
        };

        let triple = Triple::new(s, p, o).with_confidence(0.9);

        let transition = schema.predict_effects(&[triple]).unwrap();
        assert_eq!(transition.assertions.len(), 1);
        assert_eq!(transition.assertions[0], (s, p, new_o));
        assert!(transition.retractions.is_empty());
    }

    #[test]
    fn action_schema_predict_effects_retract() {
        let s = SymbolId::new(1).unwrap();
        let p = SymbolId::new(2).unwrap();
        let o = SymbolId::new(3).unwrap();

        let schema = ActionSchema {
            action_id: SymbolId::new(10).unwrap(),
            name: "test".into(),
            preconditions: vec![CausalPattern {
                subject: PatternElement::Variable("x".into()),
                predicate: PatternElement::Concrete(p),
                object: PatternElement::Variable("y".into()),
                negated: false,
            }],
            effects: vec![CausalEffect {
                kind: EffectKind::Retract,
                subject: PatternElement::Variable("x".into()),
                predicate: PatternElement::Concrete(p),
                object: PatternElement::Variable("y".into()),
                confidence: 0.0,
            }],
            success_rate: 0.5,
            execution_count: 0,
        };

        let triple = Triple::new(s, p, o).with_confidence(0.9);

        let transition = schema.predict_effects(&[triple]).unwrap();
        assert!(transition.assertions.is_empty());
        assert_eq!(transition.retractions.len(), 1);
        assert_eq!(transition.retractions[0], (s, p, o));
    }

    // ── CausalManager ──────────────────────────────────────────────

    #[test]
    fn causal_manager_default_empty() {
        let mgr = CausalManager::default();
        assert!(mgr.schemas.is_empty());
    }

    #[test]
    fn causal_manager_register_and_get() {
        let mut mgr = CausalManager::default();
        let schema = ActionSchema {
            action_id: SymbolId::new(10).unwrap(),
            name: "test_tool".into(),
            preconditions: Vec::new(),
            effects: Vec::new(),
            success_rate: 0.5,
            execution_count: 0,
        };
        mgr.register_action_schema(schema);
        assert!(mgr.get_schema("test_tool").is_some());
        assert!(mgr.get_schema("nonexistent").is_none());
    }

    #[test]
    fn causal_manager_serialization_roundtrip() {
        let mut mgr = CausalManager::default();
        let schema = ActionSchema {
            action_id: SymbolId::new(10).unwrap(),
            name: "test_tool".into(),
            preconditions: Vec::new(),
            effects: vec![CausalEffect {
                kind: EffectKind::Assert,
                subject: PatternElement::Concrete(SymbolId::new(1).unwrap()),
                predicate: PatternElement::Concrete(SymbolId::new(2).unwrap()),
                object: PatternElement::Variable("x".into()),
                confidence: 0.8,
            }],
            success_rate: 0.75,
            execution_count: 5,
        };
        mgr.register_action_schema(schema);

        let bytes = bincode::serialize(&mgr).unwrap();
        let restored: CausalManager = bincode::deserialize(&bytes).unwrap();
        assert_eq!(restored.schemas.len(), 1);
        let s = restored.get_schema("test_tool").unwrap();
        assert_eq!(s.execution_count, 5);
        assert!((s.success_rate - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn update_schema_from_outcome_success() {
        let mut mgr = CausalManager::default();
        mgr.register_action_schema(ActionSchema {
            action_id: SymbolId::new(10).unwrap(),
            name: "tool".into(),
            preconditions: Vec::new(),
            effects: Vec::new(),
            success_rate: 0.5,
            execution_count: 0,
        });

        mgr.update_schema_from_outcome("tool", true);
        let s = mgr.get_schema("tool").unwrap();
        assert_eq!(s.execution_count, 1);
        assert!(s.success_rate > 0.5);

        mgr.update_schema_from_outcome("tool", false);
        let s = mgr.get_schema("tool").unwrap();
        assert_eq!(s.execution_count, 2);
    }

    #[test]
    fn state_transition_empty() {
        let t = StateTransition {
            action_id: SymbolId::new(1).unwrap(),
            assertions: Vec::new(),
            retractions: Vec::new(),
            confidence_changes: Vec::new(),
            verified: None,
            timestamp: 0,
        };
        assert!(t.assertions.is_empty());
        assert!(t.retractions.is_empty());
        assert!(t.confidence_changes.is_empty());
        assert!(t.verified.is_none());
    }

    // ── CausalPredicates namespace ─────────────────────────────────

    #[test]
    fn causal_predicates_label_namespace() {
        // All causal predicate labels should start with "causal:".
        let expected = [
            "causal:causes",
            "causal:enables",
            "causal:prevents",
            "causal:inhibits",
            "causal:has-precondition",
            "causal:has-effect",
            "causal:has-expected-outcome",
            "causal:causal-strength",
        ];
        for label in &expected {
            assert!(label.starts_with("causal:"));
        }
    }

    // ── Role Vectors ───────────────────────────────────────────────

    #[test]
    fn role_vectors_distinct() {
        let ops = VsaOps::new(
            crate::simd::best_kernel(),
            crate::vsa::Dimension::TEST,
            crate::vsa::Encoding::Bipolar,
        );
        let roles = CausalRoleVectors::new(&ops);

        let vecs = [
            &roles.state,
            &roles.action,
            &roles.outcome,
            &roles.precondition,
            &roles.effect,
            &roles.strength,
        ];
        for i in 0..vecs.len() {
            for j in (i + 1)..vecs.len() {
                let sim = ops.similarity(vecs[i], vecs[j]).unwrap_or(0.0);
                assert!(
                    sim < 0.7,
                    "causal role vectors {i} and {j} too similar: {sim}"
                );
            }
        }
    }

    // ── EffectKind ─────────────────────────────────────────────────

    #[test]
    fn effect_kind_serialization() {
        let kinds = [
            EffectKind::Assert,
            EffectKind::Retract,
            EffectKind::ModifyConfidence { delta: 0.1 },
        ];
        for kind in &kinds {
            let bytes = bincode::serialize(kind).unwrap();
            let decoded: EffectKind = bincode::deserialize(&bytes).unwrap();
            assert_eq!(*kind, decoded);
        }
    }
}
