//! Arity and type constraints for relations.
//!
//! Inspired by Cyc's `#$arity`, `#$arg1Isa`, `#$arg2Isa`: every relation declares
//! how many arguments it takes and what type each argument must be. The system
//! enforces these at assertion time — you can't assert `(biologicalMother France
//! BillClinton)` because France isn't an Animal.
//!
//! Enforcement is opt-in: call `check_triple_constraints()` before `add_triple()`
//! to get diagnostic errors for violations. Skippable for bootstrap/migration.

use std::collections::HashMap;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::engine::Engine;
use crate::error::AkhResult;
use crate::graph::Triple;
use crate::symbol::SymbolId;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors from arity/type constraint checking.
#[derive(Debug, Error, Diagnostic)]
pub enum ArityError {
    #[error(
        "type violation: argument {arg_position} of relation {relation_label} \
         expected type \"{expected_label}\" but got \"{actual_label}\""
    )]
    #[diagnostic(
        code(akh::arity::type_violation),
        help(
            "The subject or object does not match the declared argument type for this relation. \
             Check that the entity has an `is-a` link to the required type, or declare the \
             correct type constraint with `declare_relation()`."
        )
    )]
    TypeViolation {
        relation_label: String,
        arg_position: usize,
        expected_label: String,
        actual_label: String,
        relation: SymbolId,
        expected: SymbolId,
        actual: SymbolId,
    },

    #[error(
        "arity violation: relation {relation_label} expects arity {expected_arity}, \
         but triple provides {actual_arity} arguments"
    )]
    #[diagnostic(
        code(akh::arity::arity_violation),
        help(
            "The number of arguments does not match the declared arity for this relation. \
             Standard triples are binary (arity 2). Check the relation's declaration."
        )
    )]
    ArityViolation {
        relation_label: String,
        expected_arity: usize,
        actual_arity: usize,
    },
}

/// Result type for arity operations.
pub type ArityResult<T> = std::result::Result<T, ArityError>;

// ---------------------------------------------------------------------------
// Well-known predicates
// ---------------------------------------------------------------------------

/// Well-known relation SymbolIds for ontological constraints (prefixed `onto:`).
#[derive(Debug, Clone)]
pub struct ArityPredicates {
    /// `onto:arity` — declared arity of a relation.
    pub arity: SymbolId,
    /// `onto:arg1type` — required type for argument 1 (subject).
    pub arg1type: SymbolId,
    /// `onto:arg2type` — required type for argument 2 (object).
    pub arg2type: SymbolId,
    /// `is-a` — instance-of relation for type checking.
    pub is_a: SymbolId,
}

impl ArityPredicates {
    /// Resolve all `onto:` constraint predicates, creating them if needed.
    pub fn resolve(engine: &Engine) -> AkhResult<Self> {
        Ok(Self {
            arity: engine.resolve_or_create_relation("onto:arity")?,
            arg1type: engine.resolve_or_create_relation("onto:arg1type")?,
            arg2type: engine.resolve_or_create_relation("onto:arg2type")?,
            is_a: engine.resolve_or_create_relation("is-a")?,
        })
    }
}

// ---------------------------------------------------------------------------
// Constraint declarations
// ---------------------------------------------------------------------------

/// Declared constraints for a relation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationConstraint {
    /// The relation this constraint applies to.
    pub relation: SymbolId,
    /// Declared arity (standard triples are binary = 2).
    pub arity: usize,
    /// Required type for argument 1 (subject). `None` means unconstrained.
    pub arg1_type: Option<SymbolId>,
    /// Required type for argument 2 (object). `None` means unconstrained.
    pub arg2_type: Option<SymbolId>,
}

/// A detected constraint violation.
#[derive(Debug, Clone)]
pub enum ConstraintViolation {
    /// Arity mismatch.
    Arity {
        relation: SymbolId,
        expected: usize,
        actual: usize,
    },
    /// Type mismatch for an argument.
    Type {
        relation: SymbolId,
        arg_position: usize,
        expected_type: SymbolId,
        actual_symbol: SymbolId,
    },
}

// ---------------------------------------------------------------------------
// Constraint registry
// ---------------------------------------------------------------------------

/// Registry of relation constraints.
///
/// Stores declared arity and argument types per relation, and checks triples
/// against them on demand.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConstraintRegistry {
    constraints: HashMap<SymbolId, RelationConstraint>,
}

impl ConstraintRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare constraints for a relation.
    pub fn declare(
        &mut self,
        relation: SymbolId,
        arity: usize,
        arg1_type: Option<SymbolId>,
        arg2_type: Option<SymbolId>,
    ) {
        self.constraints.insert(
            relation,
            RelationConstraint {
                relation,
                arity,
                arg1_type,
                arg2_type,
            },
        );
    }

    /// Get the constraint for a relation, if declared.
    pub fn get(&self, relation: SymbolId) -> Option<&RelationConstraint> {
        self.constraints.get(&relation)
    }

    /// Check a triple against declared constraints.
    ///
    /// Returns a list of violations (empty = no violations).
    /// The `engine` is needed to check `is-a` chains for type membership.
    pub fn check_triple(
        &self,
        triple: &Triple,
        engine: &Engine,
    ) -> Vec<ConstraintViolation> {
        let mut violations = Vec::new();

        let Some(constraint) = self.constraints.get(&triple.predicate) else {
            return violations; // No constraint declared — OK
        };

        // Check arity: standard triples are always binary (subject, object)
        if constraint.arity != 2 {
            violations.push(ConstraintViolation::Arity {
                relation: triple.predicate,
                expected: constraint.arity,
                actual: 2,
            });
        }

        // Check arg1 type (subject)
        if let Some(required_type) = constraint.arg1_type {
            if !is_instance_of(engine, triple.subject, required_type) {
                violations.push(ConstraintViolation::Type {
                    relation: triple.predicate,
                    arg_position: 1,
                    expected_type: required_type,
                    actual_symbol: triple.subject,
                });
            }
        }

        // Check arg2 type (object)
        if let Some(required_type) = constraint.arg2_type {
            if !is_instance_of(engine, triple.object, required_type) {
                violations.push(ConstraintViolation::Type {
                    relation: triple.predicate,
                    arg_position: 2,
                    expected_type: required_type,
                    actual_symbol: triple.object,
                });
            }
        }

        violations
    }

    /// Check a triple and return a diagnostic error if violations are found.
    pub fn check_triple_or_err(
        &self,
        triple: &Triple,
        engine: &Engine,
    ) -> ArityResult<()> {
        let violations = self.check_triple(triple, engine);
        if let Some(v) = violations.first() {
            match v {
                ConstraintViolation::Arity {
                    relation,
                    expected,
                    actual,
                } => Err(ArityError::ArityViolation {
                    relation_label: engine.resolve_label(*relation),
                    expected_arity: *expected,
                    actual_arity: *actual,
                }),
                ConstraintViolation::Type {
                    relation,
                    arg_position,
                    expected_type,
                    actual_symbol,
                } => Err(ArityError::TypeViolation {
                    relation_label: engine.resolve_label(*relation),
                    arg_position: *arg_position,
                    expected_label: engine.resolve_label(*expected_type),
                    actual_label: engine.resolve_label(*actual_symbol),
                    relation: *relation,
                    expected: *expected_type,
                    actual: *actual_symbol,
                }),
            }
        } else {
            Ok(())
        }
    }

    /// Number of declared constraints.
    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    /// Whether the registry has no constraints.
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Type checking helpers
// ---------------------------------------------------------------------------

/// Check if `entity` is an instance of `type_id` via `is-a` chain.
///
/// Walks the `is-a` edges from `entity` transitively. Returns true if
/// `type_id` is found anywhere in the chain.
fn is_instance_of(engine: &Engine, entity: SymbolId, type_id: SymbolId) -> bool {
    // Direct check: entity == type_id (an entity is trivially its own type)
    if entity == type_id {
        return true;
    }

    let kg = engine.knowledge_graph();

    // Resolve is-a predicate
    let is_a = match engine.lookup_symbol("is-a") {
        Ok(sym) => sym,
        _ => return false,
    };

    // BFS up the is-a chain
    let mut visited = std::collections::HashSet::new();
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(entity);

    while let Some(current) = queue.pop_front() {
        if !visited.insert(current) {
            continue;
        }
        // Find types of current: (current, is-a, ?)
        let types: Vec<_> = kg
            .triples_from(current)
            .into_iter()
            .filter(|t| t.predicate == is_a)
            .collect();

        for t in types {
            if t.object == type_id {
                return true;
            }
            queue.push_back(t.object);
        }
    }

    false
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::symbol::SymbolKind;

    fn setup_typed_scenario() -> (Engine, ArityPredicates) {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let preds = ArityPredicates::resolve(&engine).unwrap();

        // Create types
        let _animal = engine.create_symbol(SymbolKind::Entity, "Animal").unwrap();
        let _country = engine.create_symbol(SymbolKind::Entity, "Country").unwrap();
        let _person = engine.create_symbol(SymbolKind::Entity, "Person").unwrap();

        // Person is-a Animal
        let person = engine.lookup_symbol("Person").unwrap();
        let animal = engine.lookup_symbol("Animal").unwrap();
        engine
            .add_triple(&Triple::new(person, preds.is_a, animal))
            .unwrap();

        // Create instances
        let alice = engine.create_symbol(SymbolKind::Entity, "Alice").unwrap();
        let france = engine.create_symbol(SymbolKind::Entity, "France").unwrap();
        engine
            .add_triple(&Triple::new(alice.id, preds.is_a, person))
            .unwrap();
        engine
            .add_triple(&Triple::new(france.id, preds.is_a, engine.lookup_symbol("Country").unwrap()))
            .unwrap();

        (engine, preds)
    }

    #[test]
    fn is_instance_of_direct() {
        let (engine, preds) = setup_typed_scenario();
        let alice = engine.lookup_symbol("Alice").unwrap();
        let person = engine.lookup_symbol("Person").unwrap();

        assert!(is_instance_of(&engine, alice, person));
    }

    #[test]
    fn is_instance_of_transitive() {
        let (engine, _preds) = setup_typed_scenario();
        let alice = engine.lookup_symbol("Alice").unwrap();
        let animal = engine.lookup_symbol("Animal").unwrap();

        assert!(is_instance_of(&engine, alice, animal));
    }

    #[test]
    fn is_instance_of_fails() {
        let (engine, _preds) = setup_typed_scenario();
        let france = engine.lookup_symbol("France").unwrap();
        let animal = engine.lookup_symbol("Animal").unwrap();

        assert!(!is_instance_of(&engine, france, animal));
    }

    #[test]
    fn type_constraint_passes() {
        let (engine, _preds) = setup_typed_scenario();
        let alice = engine.lookup_symbol("Alice").unwrap();
        let animal = engine.lookup_symbol("Animal").unwrap();

        let mother = engine
            .create_symbol(SymbolKind::Relation, "biologicalMother")
            .unwrap();

        let mut registry = ConstraintRegistry::new();
        registry.declare(mother.id, 2, Some(animal), Some(animal));

        let alice2 = engine.create_symbol(SymbolKind::Entity, "Alice2").unwrap();
        let person = engine.lookup_symbol("Person").unwrap();
        engine
            .add_triple(&Triple::new(alice2.id, _preds.is_a, person))
            .unwrap();

        let triple = Triple::new(alice, mother.id, alice2.id);
        let violations = registry.check_triple(&triple, &engine);
        assert!(violations.is_empty(), "violations: {violations:?}");
    }

    #[test]
    fn type_constraint_fails_for_wrong_type() {
        let (engine, _preds) = setup_typed_scenario();
        let france = engine.lookup_symbol("France").unwrap();
        let alice = engine.lookup_symbol("Alice").unwrap();
        let animal = engine.lookup_symbol("Animal").unwrap();

        let mother = engine
            .create_symbol(SymbolKind::Relation, "biologicalMother")
            .unwrap();

        let mut registry = ConstraintRegistry::new();
        registry.declare(mother.id, 2, Some(animal), Some(animal));

        let triple = Triple::new(france, mother.id, alice);
        let violations = registry.check_triple(&triple, &engine);
        assert_eq!(violations.len(), 1);
        assert!(matches!(
            violations[0],
            ConstraintViolation::Type {
                arg_position: 1,
                ..
            }
        ));
    }

    #[test]
    fn check_triple_or_err_reports_type_violation() {
        let (engine, _preds) = setup_typed_scenario();
        let france = engine.lookup_symbol("France").unwrap();
        let alice = engine.lookup_symbol("Alice").unwrap();
        let animal = engine.lookup_symbol("Animal").unwrap();

        let mother = engine
            .create_symbol(SymbolKind::Relation, "biologicalMother")
            .unwrap();

        let mut registry = ConstraintRegistry::new();
        registry.declare(mother.id, 2, Some(animal), Some(animal));

        let triple = Triple::new(france, mother.id, alice);
        let err = registry.check_triple_or_err(&triple, &engine).unwrap_err();
        assert!(matches!(err, ArityError::TypeViolation { .. }));
        let msg = format!("{err}");
        assert!(msg.contains("France"));
    }

    #[test]
    fn no_constraint_passes() {
        let (engine, _preds) = setup_typed_scenario();
        let france = engine.lookup_symbol("France").unwrap();
        let alice = engine.lookup_symbol("Alice").unwrap();

        let unconstrained = engine
            .create_symbol(SymbolKind::Relation, "knows")
            .unwrap();

        let registry = ConstraintRegistry::new();
        let violations = registry.check_triple(
            &Triple::new(france, unconstrained.id, alice),
            &engine,
        );
        assert!(violations.is_empty());
    }

    #[test]
    fn arity_violation() {
        let (engine, _preds) = setup_typed_scenario();
        let alice = engine.lookup_symbol("Alice").unwrap();

        let ternary = engine
            .create_symbol(SymbolKind::Relation, "between")
            .unwrap();

        let mut registry = ConstraintRegistry::new();
        registry.declare(ternary.id, 3, None, None);

        let triple = Triple::new(alice, ternary.id, alice);
        let violations = registry.check_triple(&triple, &engine);
        assert_eq!(violations.len(), 1);
        assert!(matches!(
            violations[0],
            ConstraintViolation::Arity {
                expected: 3,
                actual: 2,
                ..
            }
        ));
    }

    #[test]
    fn registry_len_and_empty() {
        let mut reg = ConstraintRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);

        reg.declare(
            SymbolId::new(1).unwrap(),
            2,
            None,
            None,
        );
        assert!(!reg.is_empty());
        assert_eq!(reg.len(), 1);
    }
}
