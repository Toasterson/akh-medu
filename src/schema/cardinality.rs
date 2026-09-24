//! Declared predicate cardinality and aliases, read back from the KG
//! (Phase 53j, ADR 113 — memory hygiene).
//!
//! A schema pack declares a relation `single_valued` (optionally `temporal`)
//! and lists `aliases`; the compiler lowers those to
//! `(r, onto:cardinality, onto:cardinality/…)` and `(alias, onto:alias-of, r)`
//! triples ([`super::compile`]). This module is the read side the engine's
//! `add_triple` path and the hygiene pass use.
//!
//! The declarations are read straight from the live graph, never cached: a
//! pack install, a wake from the store, or a relocation all make them visible
//! with no extra registry to persist or rebuild. A lookup is one label lookup
//! plus the handful of edges leaving the predicate symbol, and returns
//! immediately when no pack ever declared a cardinality or alias.

use crate::engine::Engine;
use crate::symbol::SymbolId;

use super::compile::{
    CARDINALITY_SINGLE_VALUED, CARDINALITY_SINGLE_VALUED_TEMPORAL, PRED_ALIAS_OF, PRED_CARDINALITY,
    PRED_IS_A, RELATION_MARKER,
};

/// How many live objects a subject may hold for a predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PredicateCardinality {
    /// Any number (every undeclared predicate).
    #[default]
    Multi,
    /// One live object per `(subject, compartment)`: the newest assertion
    /// supersedes older ones.
    SingleValued {
        /// Superseded values are history that stopped holding when the
        /// successor was asserted (recorded as `valid_until`).
        temporal: bool,
    },
}

impl PredicateCardinality {
    /// Whether the predicate holds one live object per subject.
    pub const fn is_single_valued(self) -> bool {
        matches!(self, PredicateCardinality::SingleValued { .. })
    }

    /// Whether superseded values are temporal history.
    pub const fn is_temporal(self) -> bool {
        matches!(self, PredicateCardinality::SingleValued { temporal: true })
    }
}

/// The cardinality declared for `predicate` by any installed schema pack.
///
/// Conflicting declarations resolve to the most specific one: temporal
/// single-valued, then single-valued, then multi.
pub fn declared_cardinality(engine: &Engine, predicate: SymbolId) -> PredicateCardinality {
    let Ok(cardinality_pred) = engine.lookup_symbol(PRED_CARDINALITY) else {
        return PredicateCardinality::Multi;
    };
    let single = engine.lookup_symbol(CARDINALITY_SINGLE_VALUED).ok();
    let temporal = engine
        .lookup_symbol(CARDINALITY_SINGLE_VALUED_TEMPORAL)
        .ok();
    let mut declared = PredicateCardinality::Multi;
    for edge in engine.triples_from(predicate) {
        if edge.predicate != cardinality_pred {
            continue;
        }
        if Some(edge.object) == temporal {
            return PredicateCardinality::SingleValued { temporal: true };
        }
        if Some(edge.object) == single {
            declared = PredicateCardinality::SingleValued { temporal: false };
        }
    }
    declared
}

/// Every predicate some installed pack declared single-valued, with its
/// cardinality, in ascending symbol order (deterministic).
pub fn single_valued_predicates(engine: &Engine) -> Vec<(SymbolId, PredicateCardinality)> {
    let Ok(cardinality_pred) = engine.lookup_symbol(PRED_CARDINALITY) else {
        return Vec::new();
    };
    let mut predicates: Vec<SymbolId> = engine
        .triples_with_predicate(cardinality_pred)
        .into_iter()
        .map(|edge| edge.subject)
        .collect();
    predicates.sort_unstable();
    predicates.dedup();
    predicates
        .into_iter()
        .filter_map(|p| {
            let cardinality = declared_cardinality(engine, p);
            cardinality.is_single_valued().then_some((p, cardinality))
        })
        .collect()
}

/// Whether `symbol` is a declared relation: `(symbol, is-a, relation)`, the
/// convention schema packs and seed packs both use.
pub fn is_declared_relation(engine: &Engine, symbol: SymbolId) -> bool {
    let (Ok(is_a), Ok(marker)) = (
        engine.lookup_symbol(PRED_IS_A),
        engine.lookup_symbol(RELATION_MARKER),
    ) else {
        return false;
    };
    engine
        .triples_from(symbol)
        .iter()
        .any(|edge| edge.predicate == is_a && edge.object == marker)
}

/// The canonical relation `predicate` is a declared alias of, if any.
///
/// Only an `onto:alias-of` edge whose target is a declared relation counts:
/// ADR 081 promotion records the same edge for frontier *properties*, and
/// those must not start rewriting predicates. When several targets qualify
/// (conflicting packs) the lowest symbol id wins, deterministically.
pub fn declared_alias_target(engine: &Engine, predicate: SymbolId) -> Option<SymbolId> {
    let alias_of = engine.lookup_symbol(PRED_ALIAS_OF).ok()?;
    engine
        .triples_from(predicate)
        .into_iter()
        .filter(|edge| edge.predicate == alias_of && edge.object != predicate)
        .map(|edge| edge.object)
        .filter(|target| is_declared_relation(engine, *target))
        .min()
}

/// Every declared `(alias, canonical)` pair whose canonical is a declared
/// relation, in ascending `(canonical, alias)` order.
pub fn declared_aliases(engine: &Engine) -> Vec<(SymbolId, SymbolId)> {
    let Ok(alias_of) = engine.lookup_symbol(PRED_ALIAS_OF) else {
        return Vec::new();
    };
    let mut aliases: Vec<SymbolId> = engine
        .triples_with_predicate(alias_of)
        .into_iter()
        .map(|edge| edge.subject)
        .collect();
    aliases.sort_unstable();
    aliases.dedup();
    let mut pairs: Vec<(SymbolId, SymbolId)> = aliases
        .into_iter()
        .filter_map(|alias| declared_alias_target(engine, alias).map(|c| (alias, c)))
        .collect();
    pairs.sort_unstable_by_key(|(alias, canonical)| (*canonical, *alias));
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::schema::{compile, install_pack};
    use crate::vsa::Dimension;

    fn engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    const PACK: &str = r#"{
        "pack": {"name": "card-test", "version": "0.1.0"},
        "relations": [
            {"name": "next_check", "cardinality": "single_valued", "temporal": true,
             "aliases": ["next-check", "has_next_check"]},
            {"name": "owner", "cardinality": "single_valued"},
            {"name": "tagged"}
        ]
    }"#;

    #[test]
    fn installed_pack_declares_cardinality_and_aliases() {
        let engine = engine();
        install_pack(&engine, &compile(PACK).unwrap()).unwrap();
        let next_check = engine.lookup_symbol("next_check").unwrap();
        let owner = engine.lookup_symbol("owner").unwrap();
        let tagged = engine.lookup_symbol("tagged").unwrap();
        assert_eq!(
            declared_cardinality(&engine, next_check),
            PredicateCardinality::SingleValued { temporal: true }
        );
        assert_eq!(
            declared_cardinality(&engine, owner),
            PredicateCardinality::SingleValued { temporal: false }
        );
        assert_eq!(
            declared_cardinality(&engine, tagged),
            PredicateCardinality::Multi
        );
        assert_eq!(single_valued_predicates(&engine).len(), 2);

        let dash = engine.lookup_symbol("next-check").unwrap();
        let has = engine.lookup_symbol("has_next_check").unwrap();
        assert_eq!(declared_alias_target(&engine, dash), Some(next_check));
        assert_eq!(declared_alias_target(&engine, has), Some(next_check));
        assert_eq!(declared_alias_target(&engine, next_check), None);
        assert_eq!(declared_aliases(&engine).len(), 2);
    }

    /// Insert-time supersession: the newest write wins, a backdated write
    /// older than the live value is recorded but never published, and other
    /// subjects and compartments are untouched.
    #[test]
    fn insert_supersedes_older_values_and_drops_stale_writes() {
        use crate::graph::Triple;

        let engine = engine();
        install_pack(&engine, &compile(PACK).unwrap()).unwrap();
        let owner = engine.lookup_symbol("owner").unwrap();
        let s = engine.resolve_or_create_entity("wi-1").unwrap();
        let other = engine.resolve_or_create_entity("wi-2").unwrap();
        let [a, b, c] =
            ["alice", "bob", "carol"].map(|l| engine.resolve_or_create_entity(l).unwrap());
        let at = |subject, object, ts| Triple {
            timestamp: ts,
            ..Triple::new(subject, owner, object)
        };
        let live = |subject| -> Vec<SymbolId> {
            engine
                .triples_from(subject)
                .into_iter()
                .filter(|t| t.predicate == owner)
                .map(|t| t.object)
                .collect()
        };

        engine.add_triple(&at(s, a, 10)).unwrap();
        engine.add_triple(&at(other, a, 10)).unwrap();
        engine.add_triple(&at(s, b, 20)).unwrap();
        assert_eq!(live(s), vec![b]);
        assert_eq!(live(other), vec![a], "other subjects are untouched");

        // A backdated write is older than the live value: not published.
        engine.add_triple(&at(s, c, 15)).unwrap();
        assert_eq!(live(s), vec![b]);

        // A different compartment is a different context.
        engine
            .add_triple(&at(s, c, 30).with_compartment("mt:other".into()))
            .unwrap();
        let mut both = live(s);
        both.sort();
        let mut expected = vec![b, c];
        expected.sort();
        assert_eq!(both, expected);

        // Re-asserting the live value is not a supersession.
        engine.add_triple(&at(s, b, 40)).unwrap();
        assert!(live(s).contains(&b));
    }

    #[test]
    fn a_fresh_engine_declares_nothing() {
        let engine = engine();
        let p = engine.resolve_or_create_relation("status").unwrap();
        assert_eq!(
            declared_cardinality(&engine, p),
            PredicateCardinality::Multi
        );
        assert_eq!(declared_alias_target(&engine, p), None);
        assert!(single_valued_predicates(&engine).is_empty());
    }
}
