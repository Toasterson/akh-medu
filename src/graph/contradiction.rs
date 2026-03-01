//! Contradiction detection: checks new triples against existing knowledge.
//!
//! Inspired by Cyc's consistency enforcement. Detects:
//!
//! - **Functional violations** — same `(subject, predicate)` with conflicting objects
//!   when the predicate is declared functional (`onto:functional`)
//! - **Disjointness violations** — new triple violates a disjointness constraint
//!   (e.g., asserting X is both a Mouse and a Moose when they're disjoint)
//! - **Temporal conflicts** — new triple contradicts a still-valid existing triple
//!   (using temporal profiles from Phase 9k)
//! - **Intra-microtheory conflicts** — contradiction within the same context
//!
//! Contradictions are *reported*, not blocked. The caller decides how to proceed
//! (add anyway, replace, abort).

use std::collections::HashSet;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::engine::Engine;
use crate::error::AkhResult;
use crate::graph::Triple;
use crate::symbol::SymbolId;
use crate::temporal::TemporalRegistry;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors specific to contradiction detection.
#[derive(Debug, Error, Diagnostic)]
pub enum ContradictionError {
    #[error("functional violation: ({subject_label}, {predicate_label}) already maps to \"{existing_label}\", cannot also map to \"{incoming_label}\"")]
    #[diagnostic(
        code(akh::contradiction::functional_violation),
        help(
            "This predicate is declared functional (one-to-one). The subject already has a \
             different object for this predicate. Either remove the existing triple first, \
             or don't declare this predicate as functional."
        )
    )]
    FunctionalViolation {
        subject_label: String,
        predicate_label: String,
        existing_label: String,
        incoming_label: String,
    },

    #[error("disjointness violation: \"{entity_label}\" cannot be both \"{type_a_label}\" and \"{type_b_label}\" (they are disjoint)")]
    #[diagnostic(
        code(akh::contradiction::disjointness_violation),
        help(
            "Two types were declared disjoint via `onto:disjoint_with`, meaning no entity \
             can be an instance of both. Check the entity's type assertions."
        )
    )]
    DisjointnessViolation {
        entity_label: String,
        type_a_label: String,
        type_b_label: String,
    },
}

/// Result type for contradiction operations.
pub type ContradictionResult<T> = std::result::Result<T, ContradictionError>;

// ---------------------------------------------------------------------------
// Well-known predicates
// ---------------------------------------------------------------------------

/// Well-known relation SymbolIds for contradiction detection (prefixed `onto:`).
#[derive(Debug, Clone)]
pub struct ContradictionPredicates {
    /// `onto:functional` — marks a relation as functional (many-to-one).
    pub functional: SymbolId,
    /// `onto:disjoint_with` — two types share no instances.
    pub disjoint_with: SymbolId,
    /// `is-a` — instance-of relation.
    pub is_a: SymbolId,
}

impl ContradictionPredicates {
    /// Resolve all contradiction predicates, creating them if needed.
    pub fn resolve(engine: &Engine) -> AkhResult<Self> {
        Ok(Self {
            functional: engine.resolve_or_create_relation("onto:functional")?,
            disjoint_with: engine.resolve_or_create_relation("onto:disjoint_with")?,
            is_a: engine.resolve_or_create_relation("is-a")?,
        })
    }
}

// ---------------------------------------------------------------------------
// Contradiction types
// ---------------------------------------------------------------------------

/// Kind of contradiction detected.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ContradictionKind {
    /// Predicate is functional but two different objects exist for the same subject.
    FunctionalViolation {
        existing_object: SymbolId,
        incoming_object: SymbolId,
    },
    /// Entity would be instance of two disjoint types.
    DisjointnessViolation {
        entity: SymbolId,
        type_a: SymbolId,
        type_b: SymbolId,
    },
    /// New triple conflicts with a still-valid (not expired) existing triple.
    TemporalConflict {
        existing_confidence: f32,
        incoming_confidence: f32,
    },
    /// Contradiction within the same microtheory/compartment.
    IntraMicrotheoryConflict {
        context: SymbolId,
    },
}

/// A detected contradiction between an existing and incoming triple.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contradiction {
    /// The existing triple in the KG.
    pub existing: Triple,
    /// The incoming triple being checked.
    pub incoming: Triple,
    /// The kind of contradiction.
    pub kind: ContradictionKind,
    /// Optional context (microtheory) where the contradiction was detected.
    pub context: Option<SymbolId>,
}

// ---------------------------------------------------------------------------
// Checking
// ---------------------------------------------------------------------------

/// Set of functional predicates.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FunctionalPredicates {
    predicates: HashSet<SymbolId>,
}

impl FunctionalPredicates {
    /// Create an empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare a predicate as functional.
    pub fn declare_functional(&mut self, predicate: SymbolId) {
        self.predicates.insert(predicate);
    }

    /// Check if a predicate is functional.
    pub fn is_functional(&self, predicate: SymbolId) -> bool {
        self.predicates.contains(&predicate)
    }

    /// Number of declared functional predicates.
    pub fn len(&self) -> usize {
        self.predicates.len()
    }

    /// Whether no functional predicates are declared.
    pub fn is_empty(&self) -> bool {
        self.predicates.is_empty()
    }
}

// ─── Multi-valued predicates ─────────────────────────────────────────────────

/// Set of predicates declared as multi-valued (one-to-many).
///
/// Multi-valued predicates naturally have multiple objects for the same
/// subject (e.g. `agent:learned` — an episode learns many things).
/// These are excluded from temporal conflict detection, which would
/// otherwise flag every new object as conflicting with existing ones.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MultiValuedPredicates {
    predicates: HashSet<SymbolId>,
}

impl MultiValuedPredicates {
    /// Create an empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare a predicate as multi-valued.
    pub fn declare_multi_valued(&mut self, predicate: SymbolId) {
        self.predicates.insert(predicate);
    }

    /// Check if a predicate is multi-valued.
    pub fn is_multi_valued(&self, predicate: SymbolId) -> bool {
        self.predicates.contains(&predicate)
    }

    /// Number of declared multi-valued predicates.
    pub fn len(&self) -> usize {
        self.predicates.len()
    }

    /// Whether no multi-valued predicates are declared.
    pub fn is_empty(&self) -> bool {
        self.predicates.is_empty()
    }
}

/// Set of disjointness constraints.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DisjointnessConstraints {
    /// Pairs of disjoint types. Stored both ways for O(1) lookup.
    pairs: HashSet<(SymbolId, SymbolId)>,
}

impl DisjointnessConstraints {
    /// Create an empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare two types as disjoint.
    pub fn declare_disjoint(&mut self, type_a: SymbolId, type_b: SymbolId) {
        self.pairs.insert((type_a, type_b));
        self.pairs.insert((type_b, type_a));
    }

    /// Check if two types are disjoint.
    pub fn are_disjoint(&self, type_a: SymbolId, type_b: SymbolId) -> bool {
        self.pairs.contains(&(type_a, type_b))
    }

    /// Number of disjointness pairs (counting each pair once).
    pub fn len(&self) -> usize {
        self.pairs.len() / 2
    }

    /// Whether no disjointness constraints are declared.
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}

/// Check a new triple for contradictions against existing knowledge.
///
/// Returns all detected contradictions (empty = no contradictions).
/// The caller decides how to handle them (add anyway, replace, abort).
pub fn check_contradictions(
    engine: &Engine,
    incoming: &Triple,
    functional_preds: &FunctionalPredicates,
    disjointness: &DisjointnessConstraints,
    temporal_registry: Option<&TemporalRegistry>,
    multi_valued: Option<&MultiValuedPredicates>,
) -> Vec<Contradiction> {
    let mut contradictions = Vec::new();

    let kg = engine.knowledge_graph();

    // 1. Functional violation check
    if functional_preds.is_functional(incoming.predicate) {
        let existing: Vec<_> = kg
            .triples_from(incoming.subject)
            .into_iter()
            .filter(|t| t.predicate == incoming.predicate && t.object != incoming.object)
            .collect();
        for e in existing {
            contradictions.push(Contradiction {
                existing: e.clone(),
                incoming: incoming.clone(),
                kind: ContradictionKind::FunctionalViolation {
                    existing_object: e.object,
                    incoming_object: incoming.object,
                },
                context: incoming
                    .compartment_id
                    .as_ref()
                    .and_then(|c| engine.lookup_symbol(c).ok()),
            });
        }
    }

    // 2. Disjointness violation check
    // Only relevant for is-a triples
    let is_a = engine.lookup_symbol("is-a").ok();
    if let Some(is_a_id) = is_a
        && incoming.predicate == is_a_id
    {
        // The incoming triple asserts: entity is-a new_type
        // Check all existing types of this entity
        let existing_types: Vec<SymbolId> = kg
            .triples_from(incoming.subject)
            .into_iter()
            .filter(|t| t.predicate == is_a_id)
            .map(|t| t.object)
            .collect();

        for existing_type in &existing_types {
            if disjointness.are_disjoint(*existing_type, incoming.object) {
                contradictions.push(Contradiction {
                    existing: Triple::new(incoming.subject, is_a_id, *existing_type),
                    incoming: incoming.clone(),
                    kind: ContradictionKind::DisjointnessViolation {
                        entity: incoming.subject,
                        type_a: *existing_type,
                        type_b: incoming.object,
                    },
                    context: incoming
                        .compartment_id
                        .as_ref()
                        .and_then(|c| engine.lookup_symbol(c).ok()),
                });
            }
        }
    }

    // 3. Temporal conflict check — skipped for multi-valued predicates.
    let is_multi = multi_valued
        .map(|mv| mv.is_multi_valued(incoming.predicate))
        .unwrap_or(false);
    if !is_multi
        && let Some(temp_reg) = temporal_registry
    {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Find existing triples with same subject+predicate but different object
        let existing: Vec<_> = kg
            .triples_from(incoming.subject)
            .into_iter()
            .filter(|t| t.predicate == incoming.predicate && t.object != incoming.object)
            .collect();

        for e in existing {
            // Only flag if the existing triple is still valid (not expired)
            if !temp_reg.is_expired(&e, now, 0.01) {
                // Skip if already caught by functional check
                if functional_preds.is_functional(incoming.predicate) {
                    continue;
                }
                contradictions.push(Contradiction {
                    existing: e.clone(),
                    incoming: incoming.clone(),
                    kind: ContradictionKind::TemporalConflict {
                        existing_confidence: temp_reg.apply_decay(&e, now),
                        incoming_confidence: incoming.confidence,
                    },
                    context: incoming
                        .compartment_id
                        .as_ref()
                        .and_then(|c| engine.lookup_symbol(c).ok()),
                });
            }
        }
    }

    // 4. Intra-microtheory conflict: if both triples are in the same compartment
    // and contradict (same subject+predicate, different object), flag it
    if let Some(ref compartment) = incoming.compartment_id {
        let existing: Vec<_> = kg
            .triples_from(incoming.subject)
            .into_iter()
            .filter(|t| {
                t.predicate == incoming.predicate
                    && t.object != incoming.object
                    && t.compartment_id.as_deref() == Some(compartment.as_str())
            })
            .collect();

        for e in existing {
            // Skip if already caught by functional or temporal checks
            let already_caught = contradictions.iter().any(|c| c.existing.object == e.object);
            if !already_caught {
                let ctx_id = engine.lookup_symbol(compartment).ok();
                if let Some(ctx) = ctx_id {
                    contradictions.push(Contradiction {
                        existing: e.clone(),
                        incoming: incoming.clone(),
                        kind: ContradictionKind::IntraMicrotheoryConflict { context: ctx },
                        context: Some(ctx),
                    });
                }
            }
        }
    }

    contradictions
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::symbol::SymbolKind;

    #[test]
    fn functional_violation_detected() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let _preds = ContradictionPredicates::resolve(&engine).unwrap();

        let alice = engine.create_symbol(SymbolKind::Entity, "Alice").unwrap();
        let bob = engine.create_symbol(SymbolKind::Entity, "Bob").unwrap();
        let charlie = engine.create_symbol(SymbolKind::Entity, "Charlie").unwrap();
        let spouse = engine
            .create_symbol(SymbolKind::Relation, "spouse")
            .unwrap();

        // Alice spouse Bob (existing)
        engine
            .add_triple(&Triple::new(alice.id, spouse.id, bob.id))
            .unwrap();

        let mut func = FunctionalPredicates::new();
        func.declare_functional(spouse.id);

        let disjoint = DisjointnessConstraints::new();

        // Try to add Alice spouse Charlie — functional violation
        let incoming = Triple::new(alice.id, spouse.id, charlie.id);
        let contradictions =
            check_contradictions(&engine, &incoming, &func, &disjoint, None, None);

        assert_eq!(contradictions.len(), 1);
        assert!(matches!(
            contradictions[0].kind,
            ContradictionKind::FunctionalViolation { .. }
        ));
    }

    #[test]
    fn functional_same_object_no_violation() {
        let engine = Engine::new(EngineConfig::default()).unwrap();

        let alice = engine.create_symbol(SymbolKind::Entity, "Alice").unwrap();
        let bob = engine.create_symbol(SymbolKind::Entity, "Bob").unwrap();
        let spouse = engine
            .create_symbol(SymbolKind::Relation, "spouse")
            .unwrap();

        engine
            .add_triple(&Triple::new(alice.id, spouse.id, bob.id))
            .unwrap();

        let mut func = FunctionalPredicates::new();
        func.declare_functional(spouse.id);

        // Re-assert same triple — no violation
        let incoming = Triple::new(alice.id, spouse.id, bob.id);
        let contradictions = check_contradictions(
            &engine,
            &incoming,
            &func,
            &DisjointnessConstraints::new(),
            None,
            None,
        );
        assert!(contradictions.is_empty());
    }

    #[test]
    fn disjointness_violation_detected() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let preds = ContradictionPredicates::resolve(&engine).unwrap();

        let mouse = engine.create_symbol(SymbolKind::Entity, "Mouse").unwrap();
        let moose = engine.create_symbol(SymbolKind::Entity, "Moose").unwrap();
        let jerry = engine.create_symbol(SymbolKind::Entity, "Jerry").unwrap();

        // Jerry is-a Mouse (existing)
        engine
            .add_triple(&Triple::new(jerry.id, preds.is_a, mouse.id))
            .unwrap();

        let mut disjoint = DisjointnessConstraints::new();
        disjoint.declare_disjoint(mouse.id, moose.id);

        // Try to assert Jerry is-a Moose — disjointness violation
        let incoming = Triple::new(jerry.id, preds.is_a, moose.id);
        let contradictions = check_contradictions(
            &engine,
            &incoming,
            &FunctionalPredicates::new(),
            &disjoint,
            None,
            None,
        );

        assert_eq!(contradictions.len(), 1);
        assert!(matches!(
            contradictions[0].kind,
            ContradictionKind::DisjointnessViolation { .. }
        ));
    }

    #[test]
    fn no_disjointness_violation_for_compatible_types() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let preds = ContradictionPredicates::resolve(&engine).unwrap();

        let animal = engine.create_symbol(SymbolKind::Entity, "Animal").unwrap();
        let pet = engine.create_symbol(SymbolKind::Entity, "Pet").unwrap();
        let fido = engine.create_symbol(SymbolKind::Entity, "Fido").unwrap();

        engine
            .add_triple(&Triple::new(fido.id, preds.is_a, animal.id))
            .unwrap();

        // Animal and Pet are NOT disjoint
        let disjoint = DisjointnessConstraints::new();

        let incoming = Triple::new(fido.id, preds.is_a, pet.id);
        let contradictions = check_contradictions(
            &engine,
            &incoming,
            &FunctionalPredicates::new(),
            &disjoint,
            None,
            None,
        );
        assert!(contradictions.is_empty());
    }

    #[test]
    fn temporal_conflict_detected() {
        let engine = Engine::new(EngineConfig::default()).unwrap();

        let alice = engine.create_symbol(SymbolKind::Entity, "Alice").unwrap();
        let loc_a = engine.create_symbol(SymbolKind::Entity, "LocA").unwrap();
        let loc_b = engine.create_symbol(SymbolKind::Entity, "LocB").unwrap();
        let located = engine
            .create_symbol(SymbolKind::Relation, "located-at")
            .unwrap();

        // Alice located-at LocA — recent timestamp
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut existing = Triple::new(alice.id, located.id, loc_a.id);
        existing.timestamp = now - 10; // 10 seconds ago
        engine.add_triple(&existing).unwrap();

        let mut temp_reg = TemporalRegistry::new();
        temp_reg
            .set_profile(
                located.id,
                crate::temporal::TemporalProfile::Ephemeral { ttl_secs: 3600 },
            )
            .unwrap();

        // Try to add Alice located-at LocB — temporal conflict
        let incoming = Triple::new(alice.id, located.id, loc_b.id);
        let contradictions = check_contradictions(
            &engine,
            &incoming,
            &FunctionalPredicates::new(),
            &DisjointnessConstraints::new(),
            Some(&temp_reg),
            None,
        );

        assert_eq!(contradictions.len(), 1);
        assert!(matches!(
            contradictions[0].kind,
            ContradictionKind::TemporalConflict { .. }
        ));
    }

    #[test]
    fn expired_temporal_no_conflict() {
        let engine = Engine::new(EngineConfig::default()).unwrap();

        let alice = engine.create_symbol(SymbolKind::Entity, "Alice").unwrap();
        let loc_a = engine.create_symbol(SymbolKind::Entity, "LocA").unwrap();
        let loc_b = engine.create_symbol(SymbolKind::Entity, "LocB").unwrap();
        let located = engine
            .create_symbol(SymbolKind::Relation, "located-at")
            .unwrap();

        // Alice located-at LocA — old timestamp (expired)
        let mut existing = Triple::new(alice.id, located.id, loc_a.id);
        existing.timestamp = 1000; // long ago
        engine.add_triple(&existing).unwrap();

        let mut temp_reg = TemporalRegistry::new();
        temp_reg
            .set_profile(
                located.id,
                crate::temporal::TemporalProfile::Ephemeral { ttl_secs: 100 },
            )
            .unwrap();

        let incoming = Triple::new(alice.id, located.id, loc_b.id);
        let contradictions = check_contradictions(
            &engine,
            &incoming,
            &FunctionalPredicates::new(),
            &DisjointnessConstraints::new(),
            Some(&temp_reg),
            None,
        );

        // The existing triple is expired, so no temporal conflict
        assert!(contradictions.is_empty());
    }

    #[test]
    fn no_contradictions_clean_triple() {
        let engine = Engine::new(EngineConfig::default()).unwrap();

        let a = engine.create_symbol(SymbolKind::Entity, "A").unwrap();
        let b = engine.create_symbol(SymbolKind::Entity, "B").unwrap();
        let r = engine.create_symbol(SymbolKind::Relation, "R").unwrap();

        let incoming = Triple::new(a.id, r.id, b.id);
        let contradictions = check_contradictions(
            &engine,
            &incoming,
            &FunctionalPredicates::new(),
            &DisjointnessConstraints::new(),
            None,
            None,
        );
        assert!(contradictions.is_empty());
    }

    #[test]
    fn functional_predicates_len() {
        let mut fp = FunctionalPredicates::new();
        assert!(fp.is_empty());
        fp.declare_functional(SymbolId::new(1).unwrap());
        assert_eq!(fp.len(), 1);
        assert!(!fp.is_empty());
    }

    #[test]
    fn multi_valued_skips_temporal_conflict() {
        let engine = Engine::new(EngineConfig::default()).unwrap();

        let ep = engine.create_symbol(SymbolKind::Entity, "episode:1").unwrap();
        let sym_a = engine.create_symbol(SymbolKind::Entity, "concept:A").unwrap();
        let sym_b = engine.create_symbol(SymbolKind::Entity, "concept:B").unwrap();
        let learned = engine
            .create_symbol(SymbolKind::Relation, "agent:learned")
            .unwrap();

        // episode:1 learned concept:A (existing)
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut existing = Triple::new(ep.id, learned.id, sym_a.id);
        existing.timestamp = now - 10;
        engine.add_triple(&existing).unwrap();

        let mut temp_reg = TemporalRegistry::new();
        temp_reg
            .set_profile(
                learned.id,
                crate::temporal::TemporalProfile::Ephemeral { ttl_secs: 3600 },
            )
            .unwrap();

        // Declare agent:learned as multi-valued.
        let mut mv = MultiValuedPredicates::new();
        mv.declare_multi_valued(learned.id);

        // Adding episode:1 learned concept:B should NOT trigger temporal conflict.
        let incoming = Triple::new(ep.id, learned.id, sym_b.id);
        let contradictions = check_contradictions(
            &engine,
            &incoming,
            &FunctionalPredicates::new(),
            &DisjointnessConstraints::new(),
            Some(&temp_reg),
            Some(&mv),
        );

        assert!(
            contradictions.is_empty(),
            "multi-valued predicate should not trigger temporal conflict"
        );

        // Without multi-valued flag, same scenario DOES trigger temporal conflict.
        let contradictions_no_mv = check_contradictions(
            &engine,
            &incoming,
            &FunctionalPredicates::new(),
            &DisjointnessConstraints::new(),
            Some(&temp_reg),
            None,
        );

        assert_eq!(contradictions_no_mv.len(), 1);
        assert!(matches!(
            contradictions_no_mv[0].kind,
            ContradictionKind::TemporalConflict { .. }
        ));
    }

    #[test]
    fn multi_valued_predicates_len() {
        let mut mv = MultiValuedPredicates::new();
        assert!(mv.is_empty());
        mv.declare_multi_valued(SymbolId::new(1).unwrap());
        assert_eq!(mv.len(), 1);
        assert!(!mv.is_empty());
        assert!(mv.is_multi_valued(SymbolId::new(1).unwrap()));
        assert!(!mv.is_multi_valued(SymbolId::new(2).unwrap()));
    }

    #[test]
    fn disjointness_symmetry() {
        let mut dc = DisjointnessConstraints::new();
        let a = SymbolId::new(1).unwrap();
        let b = SymbolId::new(2).unwrap();
        dc.declare_disjoint(a, b);
        assert!(dc.are_disjoint(a, b));
        assert!(dc.are_disjoint(b, a));
        assert_eq!(dc.len(), 1);
    }
}
