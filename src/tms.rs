//! Truth Maintenance System: support tracking and retraction cascades.
//!
//! Inspired by Cyc's TMS. Every deduction records its full list of supporting
//! premises (a `SupportSet`). A triple may have multiple support sets (alternative
//! justifications). When a support is retracted, the TMS automatically:
//!
//! 1. Retracts all conclusions that depended solely on that support
//! 2. Re-evaluates conclusions that had alternative justifications
//! 3. Cascades through the entire dependency graph
//!
//! This makes the KB self-healing: removing a false premise automatically
//! cleans up all downstream inferences.

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::provenance::DerivationKind;
use crate::symbol::SymbolId;

// ---------------------------------------------------------------------------
// Support set
// ---------------------------------------------------------------------------

/// A set of premises that justify a derived conclusion.
///
/// A triple may have multiple support sets — alternative justifications.
/// If any one support set is fully satisfied, the triple remains valid.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SupportSet {
    /// The premise symbol IDs (triples or axioms) that justify this conclusion.
    pub premises: Vec<SymbolId>,
    /// How this conclusion was derived.
    pub derivation: DerivationKind,
    /// Confidence contributed by this support set.
    pub confidence: f64,
}

impl SupportSet {
    /// Create a new support set.
    pub fn new(premises: Vec<SymbolId>, derivation: DerivationKind, confidence: f64) -> Self {
        Self {
            premises,
            derivation,
            confidence,
        }
    }

    /// Check if this support set depends on the given symbol.
    pub fn depends_on(&self, symbol: SymbolId) -> bool {
        self.premises.contains(&symbol)
    }

    /// Check if all premises in this support set are still present in the live set.
    pub fn is_satisfied(&self, live_symbols: &HashSet<SymbolId>) -> bool {
        self.premises.iter().all(|p| live_symbols.contains(p))
    }
}

// ---------------------------------------------------------------------------
// Retraction result
// ---------------------------------------------------------------------------

/// Result of a retraction cascade through the TMS.
#[derive(Debug, Clone)]
pub struct RetractionResult {
    /// Symbols that were fully retracted (no remaining support sets).
    pub retracted: Vec<SymbolId>,
    /// Symbols that were re-evaluated with surviving support sets.
    /// Each entry is `(symbol, new_confidence)`.
    pub re_evaluated: Vec<(SymbolId, f64)>,
    /// Maximum cascade depth reached.
    pub cascade_depth: usize,
}

// ---------------------------------------------------------------------------
// Truth Maintenance System
// ---------------------------------------------------------------------------

/// Truth Maintenance System tracking support sets and retraction cascades.
///
/// Each derived symbol can have one or more support sets. When a premise is
/// retracted, the TMS identifies all affected conclusions and either retracts
/// them (if no alternative justifications remain) or re-evaluates their
/// confidence (if alternative justifications survive).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TruthMaintenanceSystem {
    /// Derived symbol → list of support sets (alternative justifications).
    supports: HashMap<SymbolId, Vec<SupportSet>>,
    /// Reverse index: premise symbol → set of derived symbols that depend on it.
    dependents: HashMap<SymbolId, HashSet<SymbolId>>,
}

impl TruthMaintenanceSystem {
    /// Create a new empty TMS.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a support set for a derived symbol.
    ///
    /// A symbol can have multiple support sets (alternative justifications).
    /// The symbol remains valid as long as at least one support set is fully satisfied.
    pub fn add_support(&mut self, derived: SymbolId, support: SupportSet) {
        // Update reverse index
        for &premise in &support.premises {
            self.dependents.entry(premise).or_default().insert(derived);
        }

        // Add to support sets
        self.supports.entry(derived).or_default().push(support);
    }

    /// Get all support sets for a derived symbol.
    pub fn supports_for(&self, derived: SymbolId) -> &[SupportSet] {
        self.supports
            .get(&derived)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get all symbols that directly depend on the given premise.
    pub fn direct_dependents(&self, premise: SymbolId) -> HashSet<SymbolId> {
        self.dependents
            .get(&premise)
            .cloned()
            .unwrap_or_default()
    }

    /// Compute the effective confidence for a symbol from its surviving support sets.
    ///
    /// Uses the maximum confidence across all remaining support sets.
    pub fn effective_confidence(&self, derived: SymbolId) -> f64 {
        self.supports
            .get(&derived)
            .map(|sets| {
                sets.iter()
                    .map(|s| s.confidence)
                    .fold(0.0_f64, f64::max)
            })
            .unwrap_or(0.0)
    }

    /// Retract a symbol and cascade through all dependents.
    ///
    /// This is the core TMS operation. When a premise is retracted:
    /// 1. Find all symbols whose support sets include the retracted premise
    /// 2. Remove those support sets
    /// 3. If a symbol has no remaining support sets, retract it (recursive cascade)
    /// 4. If a symbol has remaining support sets, re-evaluate its confidence
    ///
    /// Returns the full retraction result for the caller to apply to the KG.
    pub fn retract(&mut self, symbol: SymbolId) -> RetractionResult {
        let mut retracted = Vec::new();
        let mut re_evaluated = Vec::new();
        let mut max_depth = 0;

        // BFS cascade
        let mut queue: VecDeque<(SymbolId, usize)> = VecDeque::new();
        queue.push_back((symbol, 0));
        let mut visited = HashSet::new();
        visited.insert(symbol);

        while let Some((current, depth)) = queue.pop_front() {
            max_depth = max_depth.max(depth);

            // Remove support sets for the retracted symbol itself
            self.supports.remove(&current);

            // Find all symbols that depend on `current`
            let dependents = self
                .dependents
                .remove(&current)
                .unwrap_or_default();

            for dependent in dependents {
                // Remove support sets that contain the retracted premise
                if let Some(support_sets) = self.supports.get_mut(&dependent) {
                    support_sets.retain(|s| !s.depends_on(current));

                    if support_sets.is_empty() {
                        // No remaining justifications → retract this symbol too
                        self.supports.remove(&dependent);
                        retracted.push(dependent);

                        if visited.insert(dependent) {
                            queue.push_back((dependent, depth + 1));
                        }
                    } else {
                        // Has alternative justifications → re-evaluate confidence
                        let new_confidence = support_sets
                            .iter()
                            .map(|s| s.confidence)
                            .fold(0.0_f64, f64::max);
                        re_evaluated.push((dependent, new_confidence));
                    }
                }
            }
        }

        // The initially retracted symbol
        retracted.insert(0, symbol);

        // Clean up reverse index entries pointing to retracted symbols
        for &r in &retracted {
            for support_sets in self.supports.values() {
                for support in support_sets {
                    for &premise in &support.premises {
                        if let Some(deps) = self.dependents.get_mut(&premise) {
                            deps.remove(&r);
                        }
                    }
                }
            }
        }

        RetractionResult {
            retracted,
            re_evaluated,
            cascade_depth: max_depth,
        }
    }

    /// Check if a symbol has any support (is justified).
    pub fn is_supported(&self, symbol: SymbolId) -> bool {
        self.supports
            .get(&symbol)
            .is_some_and(|sets| !sets.is_empty())
    }

    /// Get the total number of tracked symbols.
    pub fn tracked_count(&self) -> usize {
        self.supports.len()
    }

    /// Get the total number of support sets across all symbols.
    pub fn support_set_count(&self) -> usize {
        self.supports.values().map(|v| v.len()).sum()
    }

    /// Remove all tracking data.
    pub fn clear(&mut self) {
        self.supports.clear();
        self.dependents.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(id: u64) -> SymbolId {
        SymbolId::new(id).unwrap()
    }

    #[test]
    fn add_and_query_support() {
        let mut tms = TruthMaintenanceSystem::new();

        let derived = sym(10);
        let premise_a = sym(1);
        let premise_b = sym(2);

        tms.add_support(
            derived,
            SupportSet::new(
                vec![premise_a, premise_b],
                DerivationKind::Reasoned,
                0.9,
            ),
        );

        assert!(tms.is_supported(derived));
        assert_eq!(tms.supports_for(derived).len(), 1);
        assert_eq!(tms.effective_confidence(derived), 0.9);
    }

    #[test]
    fn multiple_support_sets() {
        let mut tms = TruthMaintenanceSystem::new();

        let derived = sym(10);

        // Two alternative justifications
        tms.add_support(
            derived,
            SupportSet::new(vec![sym(1), sym(2)], DerivationKind::Reasoned, 0.8),
        );
        tms.add_support(
            derived,
            SupportSet::new(vec![sym(3)], DerivationKind::Extracted, 0.95),
        );

        assert_eq!(tms.supports_for(derived).len(), 2);
        assert_eq!(tms.effective_confidence(derived), 0.95); // max of 0.8 and 0.95
    }

    #[test]
    fn simple_retraction() {
        let mut tms = TruthMaintenanceSystem::new();

        let axiom = sym(1);
        let derived = sym(10);

        tms.add_support(
            derived,
            SupportSet::new(vec![axiom], DerivationKind::Reasoned, 0.9),
        );

        let result = tms.retract(axiom);

        assert!(result.retracted.contains(&axiom));
        assert!(result.retracted.contains(&derived));
        assert!(!tms.is_supported(derived));
    }

    #[test]
    fn retraction_with_alternative_justification() {
        let mut tms = TruthMaintenanceSystem::new();

        let axiom_a = sym(1);
        let axiom_b = sym(2);
        let derived = sym(10);

        // Two alternative justifications
        tms.add_support(
            derived,
            SupportSet::new(vec![axiom_a], DerivationKind::Reasoned, 0.8),
        );
        tms.add_support(
            derived,
            SupportSet::new(vec![axiom_b], DerivationKind::Extracted, 0.95),
        );

        // Retract axiom_a — derived should survive with axiom_b's support
        let result = tms.retract(axiom_a);

        assert!(result.retracted.contains(&axiom_a));
        assert!(!result.retracted.contains(&derived)); // NOT retracted
        assert_eq!(result.re_evaluated.len(), 1);
        assert_eq!(result.re_evaluated[0], (derived, 0.95));
        assert!(tms.is_supported(derived));
        assert_eq!(tms.supports_for(derived).len(), 1);
    }

    #[test]
    fn cascade_retraction() {
        let mut tms = TruthMaintenanceSystem::new();

        // Chain: axiom → A → B → C
        let axiom = sym(1);
        let a = sym(10);
        let b = sym(20);
        let c = sym(30);

        tms.add_support(
            a,
            SupportSet::new(vec![axiom], DerivationKind::Reasoned, 0.9),
        );
        tms.add_support(
            b,
            SupportSet::new(vec![a], DerivationKind::Reasoned, 0.8),
        );
        tms.add_support(
            c,
            SupportSet::new(vec![b], DerivationKind::Reasoned, 0.7),
        );

        let result = tms.retract(axiom);

        // All should be retracted
        assert!(result.retracted.contains(&axiom));
        assert!(result.retracted.contains(&a));
        assert!(result.retracted.contains(&b));
        assert!(result.retracted.contains(&c));
        assert_eq!(result.cascade_depth, 3);
    }

    #[test]
    fn partial_cascade_with_alternative() {
        let mut tms = TruthMaintenanceSystem::new();

        // axiom_1 → A → B
        // axiom_2 → B (alternative justification for B)
        let axiom_1 = sym(1);
        let axiom_2 = sym(2);
        let a = sym(10);
        let b = sym(20);

        tms.add_support(
            a,
            SupportSet::new(vec![axiom_1], DerivationKind::Reasoned, 0.9),
        );
        tms.add_support(
            b,
            SupportSet::new(vec![a], DerivationKind::Reasoned, 0.8),
        );
        tms.add_support(
            b,
            SupportSet::new(vec![axiom_2], DerivationKind::Extracted, 0.7),
        );

        // Retract axiom_1 — A should be retracted, B should survive via axiom_2
        let result = tms.retract(axiom_1);

        assert!(result.retracted.contains(&axiom_1));
        assert!(result.retracted.contains(&a));
        assert!(!result.retracted.contains(&b)); // B survives

        // B is re-evaluated (first via A cascade, then surviving via axiom_2)
        let b_re_eval = result.re_evaluated.iter().find(|(s, _)| *s == b);
        assert!(b_re_eval.is_some());
        assert_eq!(b_re_eval.unwrap().1, 0.7); // axiom_2's confidence
    }

    #[test]
    fn diamond_dependency() {
        let mut tms = TruthMaintenanceSystem::new();

        // Diamond: axiom → A, axiom → B, A + B → C
        let axiom = sym(1);
        let a = sym(10);
        let b = sym(20);
        let c = sym(30);

        tms.add_support(
            a,
            SupportSet::new(vec![axiom], DerivationKind::Reasoned, 0.9),
        );
        tms.add_support(
            b,
            SupportSet::new(vec![axiom], DerivationKind::Reasoned, 0.8),
        );
        tms.add_support(
            c,
            SupportSet::new(vec![a, b], DerivationKind::Reasoned, 0.7),
        );

        let result = tms.retract(axiom);

        // All should be retracted (C depends on both A and B, both retracted)
        assert!(result.retracted.contains(&axiom));
        assert!(result.retracted.contains(&a));
        assert!(result.retracted.contains(&b));
        assert!(result.retracted.contains(&c));
    }

    #[test]
    fn retract_unsupported_symbol() {
        let mut tms = TruthMaintenanceSystem::new();

        // Retracting a symbol with no dependents is a no-op
        let result = tms.retract(sym(999));
        assert_eq!(result.retracted.len(), 1); // just the symbol itself
        assert!(result.re_evaluated.is_empty());
    }

    #[test]
    fn direct_dependents() {
        let mut tms = TruthMaintenanceSystem::new();

        let axiom = sym(1);
        let a = sym(10);
        let b = sym(20);

        tms.add_support(
            a,
            SupportSet::new(vec![axiom], DerivationKind::Reasoned, 0.9),
        );
        tms.add_support(
            b,
            SupportSet::new(vec![axiom], DerivationKind::Reasoned, 0.8),
        );

        let deps = tms.direct_dependents(axiom);
        assert_eq!(deps.len(), 2);
        assert!(deps.contains(&a));
        assert!(deps.contains(&b));
    }

    #[test]
    fn support_set_depends_on() {
        let support = SupportSet::new(vec![sym(1), sym(2)], DerivationKind::Reasoned, 0.9);
        assert!(support.depends_on(sym(1)));
        assert!(support.depends_on(sym(2)));
        assert!(!support.depends_on(sym(3)));
    }

    #[test]
    fn support_set_satisfaction() {
        let support = SupportSet::new(vec![sym(1), sym(2)], DerivationKind::Reasoned, 0.9);
        let mut live = HashSet::new();
        live.insert(sym(1));
        live.insert(sym(2));
        live.insert(sym(3));

        assert!(support.is_satisfied(&live));

        live.remove(&sym(2));
        assert!(!support.is_satisfied(&live));
    }
}
