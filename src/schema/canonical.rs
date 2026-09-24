//! Structural predicate canonicalization (Phase 53j, ADR 113).
//!
//! Models spell one predicate several ways: `next_check`, `next-check`,
//! `next check`. The [`PredicateKey`] of a label is a purely structural
//! normal form: Unicode NFKC, Unicode-aware lowercasing, and every run of
//! separators (whitespace, `_`, and hyphen/dash punctuation) collapsed to one
//! `_`. Nothing here knows any word of any language (CLAUDE.md: no hardcoded
//! natural-language constants). Namespace colons are kept, so `agent:x` and
//! `x` never collide.
//!
//! Two predicates *collide* when their keys are equal. [`collisions`] groups
//! the live predicates of an engine into such families and picks the
//! canonical spelling deterministically ([`CanonicalChoice`]):
//!
//! 1. a relation declared by a schema or seed pack (`(r, is-a, relation)`),
//! 2. an engine-reserved name (`is-a`, `part-of`, …),
//! 3. the spelling with the most live triples,
//! 4. the lexicographically smallest label.
//!
//! A family with more than one *declared* spelling is [`FamilyVerdict::Ambiguous`]:
//! two packs deliberately declared both, so hygiene reports it instead of
//! merging it.

use std::collections::{BTreeMap, HashMap};

use serde::Serialize;
use unicode_normalization::UnicodeNormalization;

use crate::engine::Engine;
use crate::symbol::SymbolId;

use super::cardinality::is_declared_relation;
use super::source::is_reserved;

/// The structural canonical key of a predicate label.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct PredicateKey(String);

impl PredicateKey {
    /// Compute the key of `label`.
    pub fn of(label: &str) -> Self {
        let mut key = String::with_capacity(label.len());
        let mut pending_separator = false;
        for c in label.nfkc() {
            if is_separator(c) {
                pending_separator = !key.is_empty();
                continue;
            }
            if pending_separator {
                key.push('_');
                pending_separator = false;
            }
            key.extend(c.to_lowercase());
        }
        PredicateKey(key)
    }

    /// The key as a string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for PredicateKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Word-separator characters, by Unicode class: whitespace, the low line,
/// and hyphen/dash punctuation that survives NFKC (U+2010–U+2015, U+2212
/// MINUS SIGN, U+2E3A/U+2E3B, U+301C, U+FE58). A character class, not a word
/// list.
fn is_separator(c: char) -> bool {
    c.is_whitespace()
        || matches!(
            c,
            '_' | '-' | '\u{2010}'
                ..='\u{2015}' | '\u{2212}' | '\u{2E3A}' | '\u{2E3B}' | '\u{301C}' | '\u{FE58}'
        )
}

/// Why a family's canonical spelling was chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalChoice {
    /// The only spelling declared as a relation by a schema/seed pack.
    Declared,
    /// An engine-reserved name.
    Reserved,
    /// The spelling with the most live triples.
    MostFrequent,
    /// Tied on live triples; the lexicographically smallest label.
    LexicographicTie,
}

/// One spelling in a collision family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PredicateSpelling {
    /// The predicate symbol.
    #[serde(serialize_with = "serialize_symbol")]
    pub symbol: SymbolId,
    /// Its label.
    pub label: String,
    /// Live triples using it.
    pub live_triples: usize,
    /// Declared as a relation by a schema/seed pack.
    pub declared: bool,
    /// The relation this spelling is a declared alias of, if any.
    #[serde(serialize_with = "serialize_opt_symbol")]
    pub alias_of: Option<SymbolId>,
}

fn serialize_opt_symbol<S: serde::Serializer>(
    id: &Option<SymbolId>,
    s: S,
) -> Result<S::Ok, S::Error> {
    match id {
        Some(id) => s.serialize_some(&id.get()),
        None => s.serialize_none(),
    }
}

fn serialize_symbol<S: serde::Serializer>(id: &SymbolId, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_u64(id.get())
}

/// What hygiene may do with a family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum FamilyVerdict {
    /// Exact key collision with one determinable canonical spelling: safe to
    /// auto-merge.
    Merge {
        /// Index into [`PredicateFamily::members`] of the canonical spelling.
        canonical: usize,
        /// Why that spelling won.
        choice: CanonicalChoice,
    },
    /// More than one spelling is a declared relation, or a declared alias
    /// points outside the family's declared relation: report only.
    Ambiguous,
    /// Every spelling is a declared alias of a relation outside the family:
    /// report only (merge them with `kg_merge_predicates mode=declared`).
    DeclaredAliases,
}

/// Predicates whose labels share one [`PredicateKey`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PredicateFamily {
    /// The shared key.
    pub key: PredicateKey,
    /// Every spelling, sorted by label.
    pub members: Vec<PredicateSpelling>,
    /// Merge or report.
    pub verdict: FamilyVerdict,
}

impl PredicateFamily {
    /// The canonical spelling, when the family is mergeable.
    pub fn canonical(&self) -> Option<&PredicateSpelling> {
        match self.verdict {
            FamilyVerdict::Merge { canonical, .. } => self.members.get(canonical),
            FamilyVerdict::Ambiguous | FamilyVerdict::DeclaredAliases => None,
        }
    }

    /// The non-canonical spellings that still carry live triples, when the
    /// family is mergeable.
    pub fn aliases_with_triples(&self) -> Vec<&PredicateSpelling> {
        match self.verdict {
            FamilyVerdict::Merge { canonical, .. } => self
                .members
                .iter()
                .enumerate()
                .filter(|(i, m)| *i != canonical && m.live_triples > 0)
                .map(|(_, m)| m)
                .collect(),
            FamilyVerdict::Ambiguous | FamilyVerdict::DeclaredAliases => Vec::new(),
        }
    }

    /// Live triples not yet under the canonical spelling.
    pub fn pending_triples(&self) -> usize {
        self.aliases_with_triples()
            .iter()
            .map(|m| m.live_triples)
            .sum()
    }
}

/// Group `(predicate, label, live triple count, declared)` rows into
/// collision families (two or more spellings) and decide each family's
/// verdict. Pure, so the choice rules are unit-testable.
pub fn families_from(spellings: Vec<PredicateSpelling>) -> Vec<PredicateFamily> {
    let mut groups: BTreeMap<PredicateKey, Vec<PredicateSpelling>> = BTreeMap::new();
    for spelling in spellings {
        groups
            .entry(PredicateKey::of(&spelling.label))
            .or_default()
            .push(spelling);
    }
    groups
        .into_iter()
        .filter(|(_, members)| members.len() > 1)
        .map(|(key, mut members)| {
            members.sort_by(|a, b| a.label.cmp(&b.label).then(a.symbol.cmp(&b.symbol)));
            let verdict = verdict_for(&members);
            PredicateFamily {
                key,
                members,
                verdict,
            }
        })
        .collect()
}

fn verdict_for(members: &[PredicateSpelling]) -> FamilyVerdict {
    let declared: Vec<usize> = (0..members.len())
        .filter(|&i| members[i].declared)
        .collect();
    match declared.as_slice() {
        [only] => {
            // A declared alias in the family must point at this relation;
            // one that belongs elsewhere makes the family ambiguous.
            let target = members[*only].symbol;
            if members
                .iter()
                .any(|m| m.alias_of.is_some_and(|t| t != target))
            {
                return FamilyVerdict::Ambiguous;
            }
            return FamilyVerdict::Merge {
                canonical: *only,
                choice: CanonicalChoice::Declared,
            };
        }
        [] => {}
        _ => return FamilyVerdict::Ambiguous,
    }
    // Spellings a pack declared as aliases of a relation outside this
    // family are an operator decision, not a spelling collision.
    if members.iter().any(|m| m.alias_of.is_some()) {
        return FamilyVerdict::DeclaredAliases;
    }
    let reserved: Vec<usize> = (0..members.len())
        .filter(|&i| is_reserved(&members[i].label))
        .collect();
    match reserved.as_slice() {
        [only] => {
            return FamilyVerdict::Merge {
                canonical: *only,
                choice: CanonicalChoice::Reserved,
            };
        }
        [] => {}
        _ => return FamilyVerdict::Ambiguous,
    }
    // Most live triples; ties go to the smallest label. `members` is sorted
    // by label, so the first index with the maximum count is that label.
    let max = members.iter().map(|m| m.live_triples).max().unwrap_or(0);
    let winners: Vec<usize> = (0..members.len())
        .filter(|&i| members[i].live_triples == max)
        .collect();
    FamilyVerdict::Merge {
        canonical: winners[0],
        choice: if winners.len() == 1 {
            CanonicalChoice::MostFrequent
        } else {
            CanonicalChoice::LexicographicTie
        },
    }
}

/// Every collision family among the predicates of `engine`'s live triples
/// plus its declared relations (a declared relation with no triples yet
/// still anchors its family as the canonical spelling).
pub fn collisions(engine: &Engine) -> Vec<PredicateFamily> {
    let mut counts: HashMap<SymbolId, usize> = HashMap::new();
    for triple in engine.all_triples() {
        *counts.entry(triple.predicate).or_default() += 1;
    }
    // Declared relations without live triples still join their family.
    if let (Ok(is_a), Ok(marker)) = (
        engine.lookup_symbol(super::compile::PRED_IS_A),
        engine.lookup_symbol(super::compile::RELATION_MARKER),
    ) {
        for edge in engine.triples_to(marker) {
            if edge.predicate == is_a {
                counts.entry(edge.subject).or_default();
            }
        }
    }
    let spellings = counts
        .into_iter()
        .map(|(symbol, live_triples)| PredicateSpelling {
            symbol,
            label: engine.resolve_label(symbol),
            live_triples,
            declared: is_declared_relation(engine, symbol),
            alias_of: super::cardinality::declared_alias_target(engine, symbol),
        })
        .collect();
    families_from(spellings)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(s: &str) -> String {
        PredicateKey::of(s).as_str().to_string()
    }

    fn spelling(id: u64, label: &str, live: usize, declared: bool) -> PredicateSpelling {
        PredicateSpelling {
            symbol: SymbolId::new(id).unwrap(),
            label: label.into(),
            live_triples: live,
            declared,
            alias_of: None,
        }
    }

    fn alias(id: u64, label: &str, live: usize, target: u64) -> PredicateSpelling {
        PredicateSpelling {
            alias_of: SymbolId::new(target),
            ..spelling(id, label, live, false)
        }
    }

    #[test]
    fn separators_and_case_collapse_structurally() {
        assert_eq!(key("next_check"), "next_check");
        assert_eq!(key("next-check"), "next_check");
        assert_eq!(key("next check"), "next_check");
        assert_eq!(key("Next  Check"), "next_check");
        assert_eq!(key("  next__-check  "), "next_check");
        // Unicode: dashes and full-width forms (NFKC) and non-ASCII case.
        assert_eq!(key("next\u{2010}check"), "next_check");
        assert_eq!(key("ｎｅｘｔ－ｃｈｅｃｋ"), "next_check");
        assert_eq!(key("NÄCHSTE Prüfung"), "nächste_prüfung");
        assert_eq!(key("Следующая-Проверка"), key("следующая проверка"));
        // Namespaces stay distinct; other punctuation is kept.
        assert_ne!(key("agent:next_check"), key("next_check"));
        assert_ne!(key("next.check"), key("next_check"));
        assert_ne!(key("nextcheck"), key("next_check"));
    }

    #[test]
    fn most_frequent_spelling_wins_then_lexicographic() {
        let families = families_from(vec![
            spelling(1, "next_check", 206, false),
            spelling(2, "next-check", 139, false),
            spelling(3, "next check", 131, false),
            spelling(4, "status", 44, false),
        ]);
        assert_eq!(families.len(), 1);
        let family = &families[0];
        assert_eq!(family.canonical().unwrap().label, "next_check");
        assert!(matches!(
            family.verdict,
            FamilyVerdict::Merge {
                choice: CanonicalChoice::MostFrequent,
                ..
            }
        ));
        assert_eq!(family.pending_triples(), 139 + 131);

        let tie = families_from(vec![
            spelling(5, "b-state", 3, false),
            spelling(6, "b_state", 3, false),
        ]);
        assert_eq!(tie[0].canonical().unwrap().label, "b-state");
        assert!(matches!(
            tie[0].verdict,
            FamilyVerdict::Merge {
                choice: CanonicalChoice::LexicographicTie,
                ..
            }
        ));
    }

    #[test]
    fn declared_and_reserved_spellings_anchor_the_family() {
        let declared = families_from(vec![
            spelling(1, "next check", 300, false),
            spelling(2, "next_check", 0, true),
        ]);
        assert_eq!(declared[0].canonical().unwrap().label, "next_check");

        let reserved = families_from(vec![
            spelling(1, "is_a", 9, false),
            spelling(2, "is-a", 2, false),
        ]);
        assert_eq!(reserved[0].canonical().unwrap().label, "is-a");

        let ambiguous = families_from(vec![
            spelling(1, "owner", 1, true),
            spelling(2, "Owner", 1, true),
        ]);
        assert_eq!(ambiguous[0].verdict, FamilyVerdict::Ambiguous);
        assert!(ambiguous[0].aliases_with_triples().is_empty());

        // Aliases of the family's own declared relation merge into it.
        let own = families_from(vec![
            spelling(1, "next_check", 5, true),
            alias(2, "next-check", 9, 1),
        ]);
        assert_eq!(own[0].canonical().unwrap().label, "next_check");

        // Aliases of a relation outside the family: report, never shuffle
        // triples between alias spellings.
        let outside = families_from(vec![
            alias(2, "has_next_check", 1, 9),
            alias(3, "has next check", 4, 9),
        ]);
        assert_eq!(outside[0].verdict, FamilyVerdict::DeclaredAliases);
        assert!(outside[0].aliases_with_triples().is_empty());

        // A family alias belonging to another relation makes it ambiguous.
        let foreign = families_from(vec![spelling(1, "owner", 1, true), alias(2, "Owner", 1, 7)]);
        assert_eq!(foreign[0].verdict, FamilyVerdict::Ambiguous);
    }
}
