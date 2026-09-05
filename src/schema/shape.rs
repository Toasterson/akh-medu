//! Effective-property resolution: `is-a` subtyping + property-shape
//! validation (ADR 081 Slice 3, work-item 3 in
//! `docs/ai/plans/2026-06-30-akh-memory-schema-issues.md`).
//!
//! [`super::project::project_note`] (Slice 1) intentionally left inherited
//! properties out of scope: "a property must be declared *directly* on the
//! note's exact type to count as typed; inherited properties (e.g. `Person`'s
//! `name`, inherited from `Thing`) land in frontier today." This module
//! closes that gap: it walks the installed `is-a` chain upward from a type —
//! across schema-pack boundaries, since ADR 081 decision 5 makes type names
//! global and immutable, so a chain like `Employee -> Person -> Thing` may
//! span three independently-compiled packs transparently — and resolves the
//! **effective property set** a type carries: its own properties plus
//! everything inherited from its ancestors.
//!
//! It also carries the property-shape validator: given an
//! [`EffectiveProperty`]'s declared [`ValueType`] and a
//! [`FrontmatterValue`], decide whether the value matches the declared
//! shape, and whether a type's required properties are all present. Per ADR
//! 081 decision 7 ("writes are never blocked"), both checks are advisory:
//! callers ([`super::project::project_note`]) degrade the value to frontier
//! and report a [`ShapeViolation`] rather than failing the write.

use std::collections::{BTreeMap, HashSet};

use miette::Diagnostic;
use thiserror::Error;

use crate::engine::Engine;
use crate::library::note::FrontmatterValue;
use crate::symbol::SymbolId;

use super::compile::{PRED_IS_A, PRED_PROPERTY, PRED_REQUIRED, PRED_VALUE_TYPE, TYPE_MARKER};
use super::source::ValueType;
use super::{SchemaError, SchemaResult};

// ---------------------------------------------------------------------------
// Effective property set
// ---------------------------------------------------------------------------

/// One property in a type's effective (own + inherited) shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectiveProperty {
    pub name: String,
    pub value_type: ValueType,
    pub required: bool,
    /// The type name this property is declared directly on — the queried
    /// type itself when [`Self::direct`] is `true`, an ancestor otherwise.
    pub declared_on: String,
    /// Whether the property is declared directly on the queried type (not
    /// inherited via `is-a`).
    pub direct: bool,
}

/// The result of walking a type's `is-a` chain: its full ancestry (exact type
/// first, root last) plus every property it carries, own and inherited.
#[derive(Debug, Clone, Default)]
pub struct EffectivePropertySet {
    /// The type this set was resolved for.
    pub type_name: String,
    /// Ancestry from the exact type (index 0) up to the root. Empty when the
    /// set was never resolved against an installed type (e.g. the note's
    /// declared type was unknown).
    pub chain: Vec<String>,
    /// Property name → its effective declaration. A name declared on more
    /// than one ancestor (should not happen under ADR 081 decision 5's
    /// global-immutable-name rule, but the walk is defensive about it) keeps
    /// the nearest — most specific — declaration.
    pub properties: BTreeMap<String, EffectiveProperty>,
}

impl EffectivePropertySet {
    /// An empty set for `type_name` — used when the chain walk itself failed
    /// (e.g. a subtype cycle) and callers must degrade gracefully rather than
    /// fail the write (ADR 081 decision 7).
    pub fn empty(type_name: impl Into<String>) -> Self {
        let type_name = type_name.into();
        Self {
            chain: vec![type_name.clone()],
            type_name,
            properties: BTreeMap::new(),
        }
    }

    pub fn contains(&self, key: &str) -> bool {
        self.properties.contains_key(key)
    }

    pub fn get(&self, key: &str) -> Option<&EffectiveProperty> {
        self.properties.get(key)
    }

    /// The subset of properties this type's shape requires.
    pub fn required(&self) -> impl Iterator<Item = &EffectiveProperty> {
        self.properties.values().filter(|p| p.required)
    }
}

// ---------------------------------------------------------------------------
// Chain walk
// ---------------------------------------------------------------------------

/// Whether `type_symbol` is declared as a global type in the installed
/// schema (`(type_symbol, is-a, onto:Type)`).
pub fn is_installed_type(engine: &Engine, type_symbol: SymbolId) -> bool {
    let Ok(is_a) = engine.resolve_symbol(PRED_IS_A) else {
        return false;
    };
    let Ok(type_marker) = engine.resolve_symbol(TYPE_MARKER) else {
        return false;
    };
    engine
        .knowledge_graph()
        .objects_of(type_symbol, is_a)
        .contains(&type_marker)
}

/// Walk the installed `is-a` chain from `type_symbol` up to its root
/// (exact type first, root last). Cycle-safe: a cycle can only arise across
/// independently-compiled packs (each pack's own subtype edges are validated
/// acyclic at parse time via `assumes` resolvability, but the registry does
/// not yet check cross-pack cycles at admission time — see
/// [`SchemaError::SubtypeCycle`]'s help text) — walking one errs with a rich
/// diagnostic instead of looping forever.
pub fn type_chain(engine: &Engine, type_symbol: SymbolId) -> SchemaResult<Vec<SymbolId>> {
    let is_a = engine
        .resolve_symbol(PRED_IS_A)
        .map_err(|e| SchemaError::Introspection {
            message: format!("`is-a` relation not resolvable: {e}"),
        })?;
    let type_marker = engine.resolve_symbol(TYPE_MARKER).ok();

    let mut chain = vec![type_symbol];
    let mut seen: HashSet<SymbolId> = HashSet::from([type_symbol]);
    let mut current = type_symbol;
    loop {
        let supertype = engine
            .knowledge_graph()
            .objects_of(current, is_a)
            .into_iter()
            .find(|o| Some(*o) != type_marker);
        let Some(sup) = supertype else { break };
        if !seen.insert(sup) {
            let mut names: Vec<String> = chain.iter().map(|s| engine.resolve_label(*s)).collect();
            names.push(engine.resolve_label(sup));
            return Err(SchemaError::SubtypeCycle {
                type_name: engine.resolve_label(type_symbol),
                chain: names,
            });
        }
        chain.push(sup);
        current = sup;
    }
    Ok(chain)
}

/// Resolve `type_symbol`'s effective property set: its own properties plus
/// everything inherited by walking the `is-a` chain. ADR 081 decision 5 means
/// this walk never needs to be pack-aware — type names are global, so the
/// chain crosses schema-pack boundaries transparently.
pub fn effective_properties(
    engine: &Engine,
    type_symbol: SymbolId,
) -> SchemaResult<EffectivePropertySet> {
    let type_name = engine.resolve_label(type_symbol);
    let chain_syms = type_chain(engine, type_symbol)?;
    let chain: Vec<String> = chain_syms
        .iter()
        .map(|s| engine.resolve_label(*s))
        .collect();

    let property_pred = engine.resolve_symbol(PRED_PROPERTY).ok();
    let required_pred = engine.resolve_symbol(PRED_REQUIRED).ok();
    let value_type_pred = engine.resolve_symbol(PRED_VALUE_TYPE).ok();

    let mut properties: BTreeMap<String, EffectiveProperty> = BTreeMap::new();
    if let Some(pp) = property_pred {
        for (idx, sym) in chain_syms.iter().enumerate() {
            let owner_label = &chain[idx];
            for prop_sym in engine.knowledge_graph().objects_of(*sym, pp) {
                let prop_name = engine.resolve_label(prop_sym);
                if properties.contains_key(&prop_name) {
                    // Nearer (already-visited, more specific) declaration wins.
                    continue;
                }
                let required = required_pred
                    .map(|rp| {
                        engine
                            .knowledge_graph()
                            .objects_of(*sym, rp)
                            .contains(&prop_sym)
                    })
                    .unwrap_or(false);
                let value_type = value_type_pred
                    .and_then(|vp| {
                        engine
                            .knowledge_graph()
                            .objects_of(prop_sym, vp)
                            .into_iter()
                            .next()
                    })
                    .map(|vt_sym| engine.resolve_label(vt_sym))
                    .and_then(|label| ValueType::from_entity_label(&label))
                    // A property triple without a resolvable value-type edge
                    // should not happen (the compiler always emits one), but
                    // this is a read-time introspection over data that may
                    // have been hand-edited — degrade rather than panic.
                    .unwrap_or(ValueType::Text);
                properties.insert(
                    prop_name.clone(),
                    EffectiveProperty {
                        name: prop_name,
                        value_type,
                        required,
                        declared_on: owner_label.clone(),
                        direct: idx == 0,
                    },
                );
            }
        }
    }

    Ok(EffectivePropertySet {
        type_name,
        chain,
        properties,
    })
}

/// Resolve a type's effective property set by name. `Ok(None)` when
/// `type_name` does not name an installed type (distinguished from an error:
/// "not a type" is a normal outcome for introspection callers, unlike a
/// genuine graph-walk failure).
pub fn effective_properties_of_type(
    engine: &Engine,
    type_name: &str,
) -> SchemaResult<Option<EffectivePropertySet>> {
    let Ok(type_symbol) = engine.resolve_symbol(type_name) else {
        return Ok(None);
    };
    if !is_installed_type(engine, type_symbol) {
        return Ok(None);
    }
    effective_properties(engine, type_symbol).map(Some)
}

// ---------------------------------------------------------------------------
// Property-shape validation
// ---------------------------------------------------------------------------

/// A property-shape violation detected while projecting a note. Per ADR 081
/// decision 7 ("writes are never blocked, typing errors surface at read
/// time"), a violation never aborts projection: the value is asserted as
/// frontier instead of typed, and the violation is carried on
/// [`super::project::ProjectionReport`] for the caller to display.
#[derive(Debug, Clone, PartialEq, Eq, Error, Diagnostic)]
pub enum ShapeViolation {
    /// A frontmatter value's runtime [`ValueType`] does not match the
    /// property's declared shape (e.g. `number` declared, `Text` given).
    #[error(
        "property \"{property}\" on type \"{declared_on}\" expects a {expected:?} value, but \
         the note's frontmatter carries a {got:?} value"
    )]
    #[diagnostic(
        code(akh::schema::shape::type_mismatch),
        help(
            "Fix the note's frontmatter so \"{property}\" holds a {expected:?} value — property \
             value types are immutable once declared (ADR 081 decision 5), so the property \
             cannot be redefined to accept {got:?}. The value was still written, just as \
             frontier instead of typed, so nothing is lost; re-project the note once fixed."
        )
    )]
    TypeMismatch {
        property: String,
        declared_on: String,
        expected: ValueType,
        got: ValueType,
    },

    /// A type's shape requires a property the note's frontmatter does not
    /// declare at all.
    #[error(
        "type \"{declared_on}\" requires property \"{property}\", but the note's frontmatter \
         does not declare it"
    )]
    #[diagnostic(
        code(akh::schema::shape::missing_required),
        help(
            "Add a `{property}:` frontmatter key to the note and re-project it. This is \
             advisory, not a block (ADR 081 decision 7) — the note was still written and typed \
             under its declared type."
        )
    )]
    MissingRequired {
        property: String,
        declared_on: String,
    },
}

/// Check `value` against `property`'s declared shape. `None` when it
/// matches; `Some` carries the violation a caller should report (and act on
/// by degrading the value to frontier instead of typed).
pub fn check_value_shape(
    property: &EffectiveProperty,
    value: &FrontmatterValue,
) -> Option<ShapeViolation> {
    let got = value.value_type();
    if got == property.value_type {
        None
    } else {
        Some(ShapeViolation::TypeMismatch {
            property: property.name.clone(),
            declared_on: property.declared_on.clone(),
            expected: property.value_type,
            got,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::schema::compile::compile;
    use crate::schema::install::install_pack;
    use crate::vsa::Dimension;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            data_dir: None,
            ..Default::default()
        })
        .unwrap()
    }

    const THREE_LEVEL: &str = r#"{
        "pack": {"name": "three-level", "version": "0.1.0"},
        "types": [
            {"name": "Thing", "properties": [
                {"name": "name", "value_type": "text", "required": true}]},
            {"name": "Person", "is_a": "Thing", "properties": [
                {"name": "email", "value_type": "text"}]},
            {"name": "Employee", "is_a": "Person", "properties": [
                {"name": "employer", "value_type": "link", "required": true}]}
        ]
    }"#;

    #[test]
    fn inherited_property_is_effective_but_not_direct() {
        let engine = test_engine();
        install_pack(&engine, &compile(THREE_LEVEL).unwrap()).unwrap();

        let person = engine.resolve_symbol("Person").unwrap();
        let effective = effective_properties(&engine, person).unwrap();

        assert_eq!(
            effective.chain,
            vec!["Person".to_string(), "Thing".to_string()]
        );
        // Own property: direct.
        let email = effective.get("email").unwrap();
        assert!(email.direct);
        assert_eq!(email.declared_on, "Person");
        // Inherited property: not direct, declared on the ancestor.
        let name = effective.get("name").unwrap();
        assert!(!name.direct);
        assert_eq!(name.declared_on, "Thing");
        assert!(name.required);
    }

    #[test]
    fn multi_level_chain_walks_to_the_root() {
        let engine = test_engine();
        install_pack(&engine, &compile(THREE_LEVEL).unwrap()).unwrap();

        let employee = engine.resolve_symbol("Employee").unwrap();
        let effective = effective_properties(&engine, employee).unwrap();

        assert_eq!(
            effective.chain,
            vec![
                "Employee".to_string(),
                "Person".to_string(),
                "Thing".to_string()
            ]
        );
        assert!(effective.contains("employer")); // own
        assert!(effective.contains("email")); // 1-hop inherited
        assert!(effective.contains("name")); // 2-hop inherited

        let employer = effective.get("employer").unwrap();
        assert!(employer.direct);
        let email = effective.get("email").unwrap();
        assert!(!email.direct);
        assert_eq!(email.declared_on, "Person");
        let name = effective.get("name").unwrap();
        assert!(!name.direct);
        assert_eq!(name.declared_on, "Thing");

        let required: Vec<&str> = {
            let mut r: Vec<&str> = effective.required().map(|p| p.name.as_str()).collect();
            r.sort();
            r
        };
        assert_eq!(required, vec!["employer", "name"]);
    }

    #[test]
    fn cross_pack_chain_resolves_transparently() {
        let engine = test_engine();
        // Base pack declares Thing; a second, independently-compiled pack
        // extends it with Person via `assumes` (ADR 081 decision 5: type
        // names are global, so this crosses pack boundaries transparently).
        let base = compile(
            r#"{"pack": {"name": "base", "version": "0.1.0"},
                "types": [{"name": "Thing", "properties": [
                    {"name": "name", "value_type": "text", "required": true}]}]}"#,
        )
        .unwrap();
        install_pack(&engine, &base).unwrap();

        let ext = compile(
            r#"{"pack": {"name": "ext", "version": "0.1.0"},
                "assumes": ["Thing"],
                "types": [{"name": "Person", "is_a": "Thing", "properties": [
                    {"name": "email", "value_type": "text"}]}]}"#,
        )
        .unwrap();
        install_pack(&engine, &ext).unwrap();

        let person = engine.resolve_symbol("Person").unwrap();
        let effective = effective_properties(&engine, person).unwrap();
        assert_eq!(
            effective.chain,
            vec!["Person".to_string(), "Thing".to_string()]
        );
        assert!(effective.contains("email"));
        let name = effective.get("name").unwrap();
        assert_eq!(name.declared_on, "Thing");
        assert!(!name.direct);
    }

    #[test]
    fn cycle_errs_instead_of_hanging() {
        let engine = test_engine();
        // Two independently-compiled packs, each `assumes`-ing the other's
        // type as its supertype — each pack is locally acyclic (parse-time
        // validation can't see the other pack), but installed together they
        // form a cycle A -> B -> A. Neither `install_pack` (global-name
        // guard only) nor `parse_source` (single-pack `assumes`
        // resolvability only) rejects this; the chain walk must catch it.
        let pack_a = compile(
            r#"{"pack": {"name": "cycle-a", "version": "0.1.0"},
                "assumes": ["B"],
                "types": [{"name": "A", "is_a": "B"}]}"#,
        )
        .unwrap();
        let pack_b = compile(
            r#"{"pack": {"name": "cycle-b", "version": "0.1.0"},
                "assumes": ["A"],
                "types": [{"name": "B", "is_a": "A"}]}"#,
        )
        .unwrap();
        install_pack(&engine, &pack_a).unwrap();
        install_pack(&engine, &pack_b).unwrap();

        let a = engine.resolve_symbol("A").unwrap();
        let err = effective_properties(&engine, a).unwrap_err();
        assert!(matches!(err, SchemaError::SubtypeCycle { ref type_name, .. } if type_name == "A"));
    }

    #[test]
    fn value_shape_mismatch_is_reported() {
        let engine = test_engine();
        install_pack(&engine, &compile(THREE_LEVEL).unwrap()).unwrap();
        let person = engine.resolve_symbol("Person").unwrap();
        let effective = effective_properties(&engine, person).unwrap();
        let email = effective.get("email").unwrap();

        // A Number where Text was declared.
        let violation = check_value_shape(email, &FrontmatterValue::Number(42.0));
        assert!(matches!(
            violation,
            Some(ShapeViolation::TypeMismatch {
                expected: ValueType::Text,
                got: ValueType::Number,
                ..
            })
        ));

        // A matching Text value is clean.
        assert!(
            check_value_shape(email, &FrontmatterValue::Text("jane@example.org".into())).is_none()
        );
    }

    #[test]
    fn additive_effective_set_of_ancestor_is_unaffected_by_subtype() {
        // ADR 081 decision 5's additivity: adding Employee never mutates the
        // Person view.
        let engine = test_engine();
        install_pack(&engine, &compile(THREE_LEVEL).unwrap()).unwrap();

        let person = engine.resolve_symbol("Person").unwrap();
        let before = effective_properties(&engine, person).unwrap();
        assert_eq!(before.properties.len(), 2, "name (inherited) + email (own)");

        // Employee is already declared in THREE_LEVEL, but re-derive Person's
        // view explicitly to prove it is unchanged by Employee existing.
        let after = effective_properties(&engine, person).unwrap();
        assert_eq!(
            before.properties.keys().collect::<Vec<_>>(),
            after.properties.keys().collect::<Vec<_>>()
        );
        assert!(
            !after.contains("employer"),
            "Employee's own property must not leak onto Person"
        );
    }
}
