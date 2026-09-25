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
//!
//! ## Two tiers: an akh's own packs over the built-in default
//!
//! Declarations come from two tiers ([`DeclarationTier`]). A built-in default
//! pack (the fleet memory pack, [`super::builtin`]) writes its declarations
//! under the `onto:default-*` predicates; every other schema pack writes the
//! plain ones. Every read here consults the pack tier first and falls back to
//! the default tier only when the pack tier says nothing about the symbol:
//!
//! - **cardinality**: any `onto:cardinality` edge on the predicate (including
//!   an explicit `onto:cardinality/multi`) replaces the default-tier value;
//! - **aliases**: a pack alias wins; a default alias is ignored when a pack
//!   declares the alias spelling itself as a relation;
//! - **relations**: a default-tier relation stops counting when a pack
//!   declares that spelling as an alias of one of its own relations.
//!
//! Within one tier, conflicting cardinalities resolve to the most specific
//! (temporal single-valued, then single-valued, then multi) and conflicting
//! alias targets to the lowest symbol id, as before.

use crate::engine::Engine;
use crate::symbol::SymbolId;

use super::compile::{
    CARDINALITY_MULTI, CARDINALITY_SINGLE_VALUED, CARDINALITY_SINGLE_VALUED_TEMPORAL,
    PRED_ALIAS_OF, PRED_CARDINALITY, PRED_DEFAULT_ALIAS_OF, PRED_DEFAULT_CARDINALITY,
    PRED_DEFAULT_RELATION, PRED_IS_A, RELATION_MARKER,
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

    /// Specificity rank: conflicting declarations within one tier resolve
    /// to the highest.
    const fn specificity(self) -> u8 {
        match self {
            PredicateCardinality::Multi => 0,
            PredicateCardinality::SingleValued { temporal: false } => 1,
            PredicateCardinality::SingleValued { temporal: true } => 2,
        }
    }
}

/// Which tier a declaration comes from. Pack declarations take precedence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeclarationTier {
    /// An installed schema pack (admitted through the registry, or an akh's
    /// own pack): `is-a relation`, `onto:cardinality`, `onto:alias-of`.
    Pack,
    /// A built-in default pack compiled into the binary:
    /// `onto:default-relation`, `onto:default-cardinality`,
    /// `onto:default-alias-of`.
    BuiltinDefault,
}

impl DeclarationTier {
    /// The predicate label carrying this tier's cardinality declarations.
    pub const fn cardinality_predicate(self) -> &'static str {
        match self {
            DeclarationTier::Pack => PRED_CARDINALITY,
            DeclarationTier::BuiltinDefault => PRED_DEFAULT_CARDINALITY,
        }
    }

    /// The predicate label carrying this tier's alias declarations.
    pub const fn alias_predicate(self) -> &'static str {
        match self {
            DeclarationTier::Pack => PRED_ALIAS_OF,
            DeclarationTier::BuiltinDefault => PRED_DEFAULT_ALIAS_OF,
        }
    }
}

/// The cardinality `predicate` carries in one tier, if that tier declares
/// one (the most specific when it declares several).
fn tier_cardinality(
    engine: &Engine,
    predicate: SymbolId,
    tier: DeclarationTier,
) -> Option<PredicateCardinality> {
    let tier_predicate = engine.lookup_symbol(tier.cardinality_predicate()).ok()?;
    let lookup = |label| engine.lookup_symbol(label).ok();
    let (multi, single, temporal) = (
        lookup(CARDINALITY_MULTI),
        lookup(CARDINALITY_SINGLE_VALUED),
        lookup(CARDINALITY_SINGLE_VALUED_TEMPORAL),
    );
    engine
        .triples_from(predicate)
        .into_iter()
        .filter(|edge| edge.predicate == tier_predicate)
        .filter_map(|edge| {
            let value = Some(edge.object);
            if value == temporal {
                Some(PredicateCardinality::SingleValued { temporal: true })
            } else if value == single {
                Some(PredicateCardinality::SingleValued { temporal: false })
            } else if value == multi {
                Some(PredicateCardinality::Multi)
            } else {
                None
            }
        })
        .max_by_key(|c| c.specificity())
}

/// The cardinality declared for `predicate`: an installed pack's declaration
/// if it has one, else the built-in default pack's, else multi.
///
/// Conflicting declarations within a tier resolve to the most specific one:
/// temporal single-valued, then single-valued, then multi.
pub fn declared_cardinality(engine: &Engine, predicate: SymbolId) -> PredicateCardinality {
    tier_cardinality(engine, predicate, DeclarationTier::Pack)
        .or_else(|| tier_cardinality(engine, predicate, DeclarationTier::BuiltinDefault))
        .unwrap_or_default()
}

/// Every predicate some installed pack (either tier) declared single-valued,
/// with its effective cardinality, in ascending symbol order (deterministic).
pub fn single_valued_predicates(engine: &Engine) -> Vec<(SymbolId, PredicateCardinality)> {
    let mut predicates: Vec<SymbolId> = [DeclarationTier::Pack, DeclarationTier::BuiltinDefault]
        .into_iter()
        .filter_map(|tier| engine.lookup_symbol(tier.cardinality_predicate()).ok())
        .flat_map(|p| engine.triples_with_predicate(p))
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

/// Whether `symbol` has an edge `(symbol, predicate_label, object_label)`.
fn has_edge(engine: &Engine, symbol: SymbolId, predicate_label: &str, object_label: &str) -> bool {
    let (Ok(predicate), Ok(object)) = (
        engine.lookup_symbol(predicate_label),
        engine.lookup_symbol(object_label),
    ) else {
        return false;
    };
    engine
        .triples_from(symbol)
        .iter()
        .any(|edge| edge.predicate == predicate && edge.object == object)
}

/// `(symbol, is-a, relation)`: declared by a schema pack or a seed pack.
fn is_pack_relation(engine: &Engine, symbol: SymbolId) -> bool {
    has_edge(engine, symbol, PRED_IS_A, RELATION_MARKER)
}

/// Whether a pack declares `symbol` an alias of one of the pack tier's own
/// relations (non-recursive: the target must carry `is-a relation`).
fn is_pack_alias(engine: &Engine, symbol: SymbolId) -> bool {
    let Ok(alias_of) = engine.lookup_symbol(PRED_ALIAS_OF) else {
        return false;
    };
    engine
        .triples_from(symbol)
        .into_iter()
        .any(|edge| edge.predicate == alias_of && is_pack_relation(engine, edge.object))
}

/// Whether `symbol` is a declared relation: `(symbol, is-a, relation)`, the
/// convention schema packs and seed packs both use, or a built-in default
/// pack's `(symbol, onto:default-relation, relation)` that no pack re-declares
/// as an alias.
pub fn is_declared_relation(engine: &Engine, symbol: SymbolId) -> bool {
    is_pack_relation(engine, symbol)
        || (has_edge(engine, symbol, PRED_DEFAULT_RELATION, RELATION_MARKER)
            && !is_pack_alias(engine, symbol))
}

/// The lowest declared-relation target of `predicate`'s alias edges in one
/// tier.
fn tier_alias_target(
    engine: &Engine,
    predicate: SymbolId,
    tier: DeclarationTier,
) -> Option<SymbolId> {
    let alias_of = engine.lookup_symbol(tier.alias_predicate()).ok()?;
    engine
        .triples_from(predicate)
        .into_iter()
        .filter(|edge| edge.predicate == alias_of && edge.object != predicate)
        .map(|edge| edge.object)
        .filter(|target| is_declared_relation(engine, *target))
        .min()
}

/// The canonical relation `predicate` is a declared alias of, if any.
///
/// Only an alias edge whose target is a declared relation counts: ADR 081
/// promotion records the same edge for frontier *properties*, and those must
/// not start rewriting predicates. When several targets qualify
/// (conflicting packs) the lowest symbol id wins, deterministically. A pack
/// alias beats a built-in default alias, and a default alias does not apply
/// to a spelling a pack declares as a relation of its own.
pub fn declared_alias_target(engine: &Engine, predicate: SymbolId) -> Option<SymbolId> {
    tier_alias_target(engine, predicate, DeclarationTier::Pack).or_else(|| {
        let target = tier_alias_target(engine, predicate, DeclarationTier::BuiltinDefault)?;
        (!is_pack_relation(engine, predicate)).then_some(target)
    })
}

/// Every declared `(alias, canonical)` pair whose canonical is a declared
/// relation (either tier, after precedence), in ascending
/// `(canonical, alias)` order.
pub fn declared_aliases(engine: &Engine) -> Vec<(SymbolId, SymbolId)> {
    let mut aliases: Vec<SymbolId> = [DeclarationTier::Pack, DeclarationTier::BuiltinDefault]
        .into_iter()
        .filter_map(|tier| engine.lookup_symbol(tier.alias_predicate()).ok())
        .flat_map(|p| engine.triples_with_predicate(p))
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
