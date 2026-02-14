//! Forward-chaining rule engine with pattern matching.
//!
//! Runs inference rules to a fixpoint or max iteration cap,
//! producing derived triples with full provenance.

use std::collections::{HashMap, HashSet};

use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;

use super::error::AutonomousResult;
use super::rules::{InferenceRule, RuleSet, RuleTerm, TriplePattern};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the forward-chaining rule engine.
#[derive(Debug, Clone)]
pub struct RuleEngineConfig {
    /// Maximum forward-chaining iterations (default: 10).
    pub max_iterations: usize,
    /// Minimum confidence for derived triples (default: 0.1).
    pub min_confidence: f32,
    /// Whether to add derived triples to the KG immediately (default: true).
    pub auto_commit: bool,
    /// Hard cap on total new triples (default: 1000).
    pub max_new_triples: usize,
}

impl Default for RuleEngineConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            min_confidence: 0.1,
            auto_commit: true,
            max_new_triples: 1000,
        }
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// A single derived triple with provenance.
#[derive(Debug, Clone)]
pub struct DerivedTriple {
    pub triple: Triple,
    pub rule_name: String,
    pub antecedent_triples: Vec<Triple>,
    pub confidence: f32,
    pub iteration: usize,
}

/// Result of running the rule engine.
#[derive(Debug, Clone)]
pub struct RuleEngineResult {
    pub derived: Vec<DerivedTriple>,
    pub iterations: usize,
    pub reached_fixpoint: bool,
    /// Per-rule derivation counts.
    pub rule_stats: HashMap<String, usize>,
}

// ---------------------------------------------------------------------------
// Rule engine
// ---------------------------------------------------------------------------

/// Forward-chaining rule engine.
pub struct RuleEngine {
    config: RuleEngineConfig,
    rule_sets: Vec<RuleSet>,
}

impl RuleEngine {
    pub fn new(config: RuleEngineConfig) -> Self {
        Self {
            config,
            rule_sets: Vec::new(),
        }
    }

    pub fn with_rules(mut self, rules: RuleSet) -> Self {
        self.rule_sets.push(rules);
        self
    }

    /// Run forward-chaining on the given engine's knowledge graph.
    pub fn run(&self, engine: &Engine) -> AutonomousResult<RuleEngineResult> {
        let mut derived: Vec<DerivedTriple> = Vec::new();
        let mut rule_stats: HashMap<String, usize> = HashMap::new();
        // Track derived triples as (s, p, o) to avoid duplicates.
        let mut derived_set: HashSet<(u64, u64, u64)> = HashSet::new();

        let all_rules: Vec<&InferenceRule> = self
            .rule_sets
            .iter()
            .flat_map(|rs| rs.rules.iter())
            .filter(|r| r.enabled)
            .collect();

        let mut iteration = 0;
        let mut reached_fixpoint = false;

        for iter in 0..self.config.max_iterations {
            iteration = iter + 1;
            let mut new_this_round: Vec<DerivedTriple> = Vec::new();

            for rule in &all_rules {
                let bindings_list = self.match_antecedents(rule, engine)?;

                let mut rule_derived_count = 0;
                for (bindings, matched_triples) in &bindings_list {
                    if rule_derived_count >= rule.max_derivations_per_iteration {
                        break;
                    }

                    for consequent in &rule.consequents {
                        if let Some(triple) =
                            self.instantiate_pattern(consequent, bindings, engine)?
                        {
                            let key = (
                                triple.subject.get(),
                                triple.predicate.get(),
                                triple.object.get(),
                            );

                            // Skip if already in KG or already derived.
                            if engine.has_triple(triple.subject, triple.predicate, triple.object)
                                || derived_set.contains(&key)
                            {
                                continue;
                            }

                            let confidence =
                                rule.confidence_factor * avg_confidence(matched_triples);
                            if confidence < self.config.min_confidence {
                                continue;
                            }

                            let derived_triple = DerivedTriple {
                                triple: triple.with_confidence(confidence),
                                rule_name: rule.name.clone(),
                                antecedent_triples: matched_triples.clone(),
                                confidence,
                                iteration,
                            };

                            derived_set.insert(key);
                            new_this_round.push(derived_triple);
                            rule_derived_count += 1;

                            if derived.len() + new_this_round.len() >= self.config.max_new_triples {
                                break;
                            }
                        }
                    }

                    if derived.len() + new_this_round.len() >= self.config.max_new_triples {
                        break;
                    }
                }

                *rule_stats.entry(rule.name.clone()).or_insert(0) += rule_derived_count;

                if derived.len() + new_this_round.len() >= self.config.max_new_triples {
                    break;
                }
            }

            if new_this_round.is_empty() {
                reached_fixpoint = true;
                break;
            }

            // Commit new triples to the KG and store provenance.
            for dt in &new_this_round {
                if self.config.auto_commit {
                    // Ignore duplicate-triple errors.
                    let _ = engine.add_triple(&dt.triple);
                }

                // Store provenance.
                let antecedent_ids: Vec<SymbolId> =
                    dt.antecedent_triples.iter().map(|t| t.subject).collect();
                let mut record = ProvenanceRecord::new(
                    dt.triple.subject,
                    DerivationKind::RuleInference {
                        rule_name: dt.rule_name.clone(),
                        antecedents: antecedent_ids.clone(),
                    },
                )
                .with_sources(antecedent_ids)
                .with_confidence(dt.confidence)
                .with_depth(dt.iteration);
                let _ = engine.store_provenance(&mut record);
            }

            derived.extend(new_this_round);

            if derived.len() >= self.config.max_new_triples {
                break;
            }
        }

        Ok(RuleEngineResult {
            derived,
            iterations: iteration,
            reached_fixpoint,
            rule_stats,
        })
    }

    // -----------------------------------------------------------------------
    // Pattern matching
    // -----------------------------------------------------------------------

    /// Match all antecedents of a rule against the KG.
    /// Returns a list of (variable bindings, matched triples) tuples.
    fn match_antecedents(
        &self,
        rule: &InferenceRule,
        engine: &Engine,
    ) -> AutonomousResult<Vec<(HashMap<String, SymbolId>, Vec<Triple>)>> {
        if rule.antecedents.is_empty() {
            return Ok(Vec::new());
        }

        // Start with the first antecedent.
        let mut results: Vec<(HashMap<String, SymbolId>, Vec<Triple>)> = self
            .match_pattern(&rule.antecedents[0], &HashMap::new(), engine)?
            .into_iter()
            .map(|(b, t)| (b, vec![t]))
            .collect();

        // For each subsequent antecedent, extend existing bindings.
        for pattern in rule.antecedents.iter().skip(1) {
            let mut new_results = Vec::new();
            for (bindings, matched) in &results {
                let extensions = self.match_pattern(pattern, bindings, engine)?;
                for (ext_bindings, ext_triple) in extensions {
                    let mut combined = bindings.clone();
                    combined.extend(ext_bindings);
                    let mut combined_triples = matched.clone();
                    combined_triples.push(ext_triple);
                    new_results.push((combined, combined_triples));
                }
            }
            results = new_results;

            if results.is_empty() {
                break;
            }
        }

        Ok(results)
    }

    /// Match a single triple pattern against the KG, given existing bindings.
    fn match_pattern(
        &self,
        pattern: &TriplePattern,
        bindings: &HashMap<String, SymbolId>,
        engine: &Engine,
    ) -> AutonomousResult<Vec<(HashMap<String, SymbolId>, Triple)>> {
        let mut results = Vec::new();

        // Resolve the predicate — this is the fast path using predicate index.
        let pred_resolved = self.resolve_term(&pattern.predicate, bindings, engine);

        let candidate_triples: Vec<Triple> = if let Some(pred_id) = pred_resolved {
            // Fast path: filter all_triples by predicate to preserve confidence.
            engine
                .all_triples()
                .into_iter()
                .filter(|t| t.predicate == pred_id)
                .collect()
        } else {
            // Slow path: scan all triples.
            engine.all_triples()
        };

        for triple in &candidate_triples {
            let mut new_bindings = HashMap::new();
            let matched =
                self.match_term(
                    &pattern.subject,
                    triple.subject,
                    bindings,
                    &mut new_bindings,
                ) && self.match_term(
                    &pattern.predicate,
                    triple.predicate,
                    bindings,
                    &mut new_bindings,
                ) && self.match_term(&pattern.object, triple.object, bindings, &mut new_bindings);

            if matched {
                results.push((new_bindings, triple.clone()));
            }
        }

        Ok(results)
    }

    /// Try to match a single term against a concrete symbol ID.
    fn match_term(
        &self,
        term: &RuleTerm,
        value: SymbolId,
        existing_bindings: &HashMap<String, SymbolId>,
        new_bindings: &mut HashMap<String, SymbolId>,
    ) -> bool {
        match term {
            RuleTerm::Concrete(id) => *id == value,
            RuleTerm::Variable(name) => {
                // Check if already bound in existing or new bindings.
                if let Some(&existing) = existing_bindings.get(name) {
                    existing == value
                } else if let Some(&already_new) = new_bindings.get(name) {
                    already_new == value
                } else {
                    new_bindings.insert(name.clone(), value);
                    true
                }
            }
            RuleTerm::Label(_) => {
                // Labels are not resolved during matching — they're resolved
                // during pattern compilation. At match time, they should have
                // been pre-resolved or handled as a predicate-index fast-path.
                // Fall back to true (permissive).
                true
            }
        }
    }

    /// Resolve a rule term to a concrete SymbolId if possible.
    fn resolve_term(
        &self,
        term: &RuleTerm,
        bindings: &HashMap<String, SymbolId>,
        engine: &Engine,
    ) -> Option<SymbolId> {
        match term {
            RuleTerm::Concrete(id) => Some(*id),
            RuleTerm::Variable(name) => bindings.get(name).copied(),
            RuleTerm::Label(label) => engine.lookup_symbol(label).ok(),
        }
    }

    /// Instantiate a consequent pattern with the given bindings.
    fn instantiate_pattern(
        &self,
        pattern: &TriplePattern,
        bindings: &HashMap<String, SymbolId>,
        engine: &Engine,
    ) -> AutonomousResult<Option<Triple>> {
        let s = self.instantiate_term(&pattern.subject, bindings, engine)?;
        let p = self.instantiate_term(&pattern.predicate, bindings, engine)?;
        let o = self.instantiate_term(&pattern.object, bindings, engine)?;

        match (s, p, o) {
            (Some(s), Some(p), Some(o)) => Ok(Some(Triple::new(s, p, o))),
            _ => Ok(None),
        }
    }

    /// Instantiate a single term: resolve variables from bindings, labels from engine.
    fn instantiate_term(
        &self,
        term: &RuleTerm,
        bindings: &HashMap<String, SymbolId>,
        engine: &Engine,
    ) -> AutonomousResult<Option<SymbolId>> {
        match term {
            RuleTerm::Concrete(id) => Ok(Some(*id)),
            RuleTerm::Variable(name) => Ok(bindings.get(name).copied()),
            RuleTerm::Label(label) => {
                // Auto-create relation for predicates (labels in consequent
                // predicate position often need to be created).
                match engine.resolve_or_create_relation(label) {
                    Ok(id) => Ok(Some(id)),
                    Err(_) => Ok(None),
                }
            }
        }
    }
}

/// Compute the average confidence of a set of triples.
fn avg_confidence(triples: &[Triple]) -> f32 {
    if triples.is_empty() {
        return 1.0;
    }
    triples.iter().map(|t| t.confidence).sum::<f32>() / triples.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::vsa::Dimension;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn transitive_closure_derives_new_triple() {
        let engine = test_engine();
        // A is-a B, B is-a C => A is-a C
        engine
            .ingest_label_triples(&[
                ("A".into(), "is-a".into(), "B".into(), 1.0),
                ("B".into(), "is-a".into(), "C".into(), 1.0),
            ])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig::default()).with_rules(RuleSet::builtin());
        let result = re.run(&engine).unwrap();

        assert!(!result.derived.is_empty());
        let a = engine.lookup_symbol("A").unwrap();
        let c = engine.lookup_symbol("C").unwrap();
        let isa = engine.lookup_symbol("is-a").unwrap();
        assert!(engine.has_triple(a, isa, c));
    }

    #[test]
    fn symmetric_rule_derives_inverse() {
        let engine = test_engine();
        engine
            .ingest_label_triples(&[("X".into(), "similar-to".into(), "Y".into(), 1.0)])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig::default()).with_rules(RuleSet::builtin());
        let result = re.run(&engine).unwrap();

        let y = engine.lookup_symbol("Y").unwrap();
        let x = engine.lookup_symbol("X").unwrap();
        let sim = engine.lookup_symbol("similar-to").unwrap();
        assert!(engine.has_triple(y, sim, x));
        assert!(result.rule_stats.get("similar-to-symmetric").is_some());
    }

    #[test]
    fn inverse_relation_parent_child() {
        let engine = test_engine();
        engine
            .ingest_label_triples(&[("Dad".into(), "parent-of".into(), "Kid".into(), 1.0)])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig::default()).with_rules(RuleSet::builtin());
        re.run(&engine).unwrap();

        let kid = engine.lookup_symbol("Kid").unwrap();
        let dad = engine.lookup_symbol("Dad").unwrap();
        let child_of = engine.lookup_symbol("child-of").unwrap();
        assert!(engine.has_triple(kid, child_of, dad));
    }

    #[test]
    fn fixpoint_reached_with_no_new_derivations() {
        let engine = test_engine();
        // Only one triple, transitive rules need chains.
        engine
            .ingest_label_triples(&[("A".into(), "has-a".into(), "B".into(), 1.0)])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig {
            max_iterations: 5,
            ..Default::default()
        })
        .with_rules(RuleSet::builtin());
        let result = re.run(&engine).unwrap();

        // Fixpoint reached (no chains to close except symmetric "similar-to" which isn't present).
        assert!(result.reached_fixpoint);
    }

    #[test]
    fn max_iterations_respected() {
        let engine = test_engine();
        // Create a long chain so that each iteration derives one more hop.
        engine
            .ingest_label_triples(&[
                ("A".into(), "is-a".into(), "B".into(), 1.0),
                ("B".into(), "is-a".into(), "C".into(), 1.0),
                ("C".into(), "is-a".into(), "D".into(), 1.0),
                ("D".into(), "is-a".into(), "E".into(), 1.0),
            ])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig {
            max_iterations: 1,
            ..Default::default()
        })
        .with_rules(RuleSet::builtin());
        let result = re.run(&engine).unwrap();

        // After 1 iteration, only direct transitive hops are derived.
        assert_eq!(result.iterations, 1);
        // A->C, B->D, C->E derived in iteration 1; A->D, A->E, B->E need more iterations.
        assert!(!result.reached_fixpoint || result.derived.is_empty());
    }

    #[test]
    fn min_confidence_filter() {
        let engine = test_engine();
        engine
            .ingest_label_triples(&[
                ("A".into(), "is-a".into(), "B".into(), 0.3),
                ("B".into(), "is-a".into(), "C".into(), 0.3),
            ])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig {
            min_confidence: 0.5,
            ..Default::default()
        })
        .with_rules(RuleSet::builtin());
        let result = re.run(&engine).unwrap();

        // Derived confidence = 0.95 * avg(0.3, 0.3) = 0.95 * 0.3 = 0.285 < 0.5
        assert!(result.derived.is_empty());
    }

    #[test]
    fn no_duplicate_derivations() {
        let engine = test_engine();
        engine
            .ingest_label_triples(&[("X".into(), "similar-to".into(), "Y".into(), 1.0)])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig {
            max_iterations: 5,
            ..Default::default()
        })
        .with_rules(RuleSet::builtin());
        let result = re.run(&engine).unwrap();

        // Y similar-to X should be derived exactly once.
        let y_sim_x: Vec<_> = result
            .derived
            .iter()
            .filter(|dt| dt.rule_name == "similar-to-symmetric")
            .collect();
        assert_eq!(y_sim_x.len(), 1);
    }

    #[test]
    fn confidence_propagation() {
        let engine = test_engine();
        engine
            .ingest_label_triples(&[
                ("A".into(), "is-a".into(), "B".into(), 0.8),
                ("B".into(), "is-a".into(), "C".into(), 0.9),
            ])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig::default()).with_rules(RuleSet::builtin());
        let result = re.run(&engine).unwrap();

        let a_c = result
            .derived
            .iter()
            .find(|dt| {
                let a = engine.lookup_symbol("A").unwrap();
                let c = engine.lookup_symbol("C").unwrap();
                dt.triple.subject == a && dt.triple.object == c
            })
            .unwrap();

        // Confidence = 0.95 * avg(0.8, 0.9) = 0.95 * 0.85 = 0.8075
        assert!((a_c.confidence - 0.8075).abs() < 0.01);
    }

    #[test]
    fn max_new_triples_cap() {
        let engine = test_engine();
        // Create enough to generate many derivations.
        engine
            .ingest_label_triples(&[
                ("A".into(), "is-a".into(), "B".into(), 1.0),
                ("B".into(), "is-a".into(), "C".into(), 1.0),
                ("C".into(), "is-a".into(), "D".into(), 1.0),
                ("D".into(), "is-a".into(), "E".into(), 1.0),
                ("E".into(), "is-a".into(), "F".into(), 1.0),
            ])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig {
            max_new_triples: 2,
            ..Default::default()
        })
        .with_rules(RuleSet::builtin());
        let result = re.run(&engine).unwrap();

        assert!(result.derived.len() <= 2);
    }
}
