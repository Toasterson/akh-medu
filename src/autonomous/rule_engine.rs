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
// Noise entity filter
// ---------------------------------------------------------------------------

/// Returns `true` if an entity name matches known noise/artifact patterns from
/// agent learning and scaffolding. Entities matching these patterns should be
/// excluded from rule inference to prevent nonsense transitive closures (e.g.
/// `Tecton → is-a → Stor01` through intermediate tool-output entities).
pub fn is_noise_entity(name: &str) -> bool {
    // Agent learning artifacts
    name.starts_with("episode:")
    // Agent goal records
    || name.starts_with("goal:")
    // Agent criteria patterns
    || name.starts_with("criteria:")
    // Agent scaffolding markers
    || name.starts_with("expand:")
    || name.starts_with("desc:")
    // Numeric rate/certainty markers like "rate:0.8000"
    || name.starts_with("rate:")
    // Bare numeric strings like "0.9910"
    || is_numeric_string(name)
}

/// Returns `true` if the string is purely numeric (digits and at most one decimal point).
fn is_numeric_string(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let mut dot_count = 0;
    for ch in s.chars() {
        if ch == '.' {
            dot_count += 1;
            if dot_count > 1 {
                return false;
            }
        } else if !ch.is_ascii_digit() {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Semantic type-compatibility guard
// ---------------------------------------------------------------------------

/// Returns `true` if the label looks like a snake_case tool or function name
/// (e.g. `infer_rules`, `kg_query`, `code_ingest`).  These identifiers are
/// internal artefacts and must not appear as nodes in a type hierarchy.
fn is_snake_case_tool_name(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.contains('_')
        && !s.contains('-')
        && !s.contains(' ')
        && s.chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

/// Returns `true` if the label matches known CI/CD or storage infrastructure
/// entity patterns.  Such labels must not become type-hierarchy parents of
/// agent identity entities (e.g. `Ptah is-a ForgejoActions` is semantically
/// invalid).
fn is_infrastructure_entity(s: &str) -> bool {
    let lower = s.to_ascii_lowercase();
    // Forgejo CI/CD platform (ForgejoActions, Forgejo, forgejo-runner, …)
    lower.contains("forgejo")
    // Generic CI runner labels
    || lower == "ci runner"
    || lower == "ci-runner"
    || lower.starts_with("ci runner ")
    || lower.starts_with("ci-runner")
    // Numbered CI runner instances like "runner-01", "runner_prod" (not standalone "runner")
    || lower.starts_with("runner-")
    || lower.starts_with("runner_")
    // Storage node labels: "Storage", "Storage-01", "Stor01", "StorageNode"
    || lower == "storage"
    || lower.starts_with("storage ")
    || lower.starts_with("storage-")
    || lower.starts_with("storage_")
    || lower.starts_with("storagenode")
    // Numbered storage nodes like "Stor01", "Stor-1" (prefix + digit or dash)
    || (lower.starts_with("stor")
        && lower.len() > 4
        && lower[4..].starts_with(|c: char| c.is_ascii_digit() || c == '-'))
}

/// Returns `true` if the label is a standard UUID (8-4-4-4-12 hex groups).
fn is_uuid(s: &str) -> bool {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 5 {
        return false;
    }
    let expected = [8usize, 4, 4, 4, 12];
    parts
        .iter()
        .zip(expected.iter())
        .all(|(part, &len)| part.len() == len && part.chars().all(|c| c.is_ascii_hexdigit()))
}

/// Returns `true` when `entity_label` must not participate in a type-hierarchy
/// assertion (`is-a` or `child-of` predicate).
///
/// Blocks cross-domain contamination: agent identity nodes must not gain
/// spurious `is-a` / `child-of` links to CI runners, storage nodes, tool-name
/// stubs, UUIDs, or other operational artefacts that have no place in the
/// concept taxonomy.  Apply this to **both** the subject and the object of
/// any candidate derived triple.
pub fn is_invalid_type_hierarchy_participant(pred_label: &str, entity_label: &str) -> bool {
    if !matches!(pred_label, "is-a" | "child-of") {
        return false;
    }
    is_noise_entity(entity_label)
        || is_snake_case_tool_name(entity_label)
        || is_infrastructure_entity(entity_label)
        || is_uuid(entity_label)
}

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
                            // Semantic type-compatibility guard: reject derivations
                            // that would link type-hierarchy predicates to infrastructure
                            // entities, tool-name stubs, UUIDs, or other non-taxonomic
                            // artefacts (fixes KG contamination from the P2 infer_rules bug).
                            let pred_label = engine.resolve_label(triple.predicate);
                            let subj_label = engine.resolve_label(triple.subject);
                            let obj_label = engine.resolve_label(triple.object);
                            if is_invalid_type_hierarchy_participant(&pred_label, &subj_label)
                                || is_invalid_type_hierarchy_participant(&pred_label, &obj_label)
                            {
                                continue;
                            }

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
    #[allow(clippy::type_complexity)]
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
            // Skip triples involving noise/artifact entities (agent scaffolding,
            // numeric noise, etc.) to prevent nonsensical transitive closures.
            let subj_label = engine.resolve_label(triple.subject);
            let obj_label = engine.resolve_label(triple.object);
            if is_noise_entity(&subj_label) || is_noise_entity(&obj_label) {
                continue;
            }

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

    #[test]
    fn noise_entities_do_not_participate_in_inference() {
        let engine = test_engine();
        // Chain: A -> goal:X -> B. The goal:X entity should be skipped by
        // the noise filter, so transitive closure should NOT derive A is-a B.
        engine
            .ingest_label_triples(&[
                ("A".into(), "is-a".into(), "goal:xyz".into(), 1.0),
                ("goal:xyz".into(), "is-a".into(), "B".into(), 1.0),
            ])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig::default()).with_rules(RuleSet::builtin());
        let result = re.run(&engine).unwrap();

        // A is-a B should NOT be derived because the intermediate entity
        // "goal:xyz" is a noise entity.
        let a = engine.lookup_symbol("A").unwrap();
        let b = engine.lookup_symbol("B").unwrap();
        let isa = engine.lookup_symbol("is-a").unwrap();
        assert!(!engine.has_triple(a, isa, b));

        // is-a-transitive should have produced zero derivations.
        let transitive_derived = result
            .rule_stats
            .get("is-a-transitive")
            .copied()
            .unwrap_or(0);
        assert_eq!(transitive_derived, 0);
    }

    #[test]
    fn noise_entity_detection_patterns() {
        // These should all be recognized as noise.
        assert!(is_noise_entity("episode:42"));
        assert!(is_noise_entity("goal:learn_rust"));
        assert!(is_noise_entity("criteria:quality_score"));
        assert!(is_noise_entity("expand:some_concept"));
        assert!(is_noise_entity("desc:a_description"));
        assert!(is_noise_entity("rate:0.8000"));
        assert!(is_noise_entity("0.9910"));

        // These should NOT be recognized as noise.
        assert!(!is_noise_entity("Tecton"));
        assert!(!is_noise_entity("MacMini"));
        assert!(!is_noise_entity("Stor01"));
        assert!(!is_noise_entity("is-a"));
        assert!(!is_noise_entity("part-of"));
    }

    #[test]
    fn infrastructure_entity_detection() {
        // CI/CD and storage patterns that must be detected.
        assert!(is_infrastructure_entity("ForgejoActions"));
        assert!(is_infrastructure_entity("forgejo-runner"));
        assert!(is_infrastructure_entity("CI runner"));
        assert!(is_infrastructure_entity("CI-runner"));
        assert!(is_infrastructure_entity("runner-01"));
        assert!(is_infrastructure_entity("runner_prod"));
        assert!(is_infrastructure_entity("Storage"));
        assert!(is_infrastructure_entity("StorageNode"));
        assert!(is_infrastructure_entity("Stor01"));
        assert!(is_infrastructure_entity("Stor-1"));

        // Legitimate entity labels that must NOT be blocked.
        assert!(!is_infrastructure_entity("Ptah"));
        assert!(!is_infrastructure_entity("Khnum"));
        assert!(!is_infrastructure_entity("System"));
        assert!(!is_infrastructure_entity("Service"));
        assert!(!is_infrastructure_entity("Process"));
        assert!(!is_infrastructure_entity("runner")); // standalone too broad — not blocked
        assert!(!is_infrastructure_entity("Marathon runner"));
    }

    #[test]
    fn snake_case_tool_name_detection() {
        assert!(is_snake_case_tool_name("infer_rules"));
        assert!(is_snake_case_tool_name("kg_query"));
        assert!(is_snake_case_tool_name("code_ingest"));
        assert!(is_snake_case_tool_name("memory_recall"));

        // Predicates with hyphens must not be flagged.
        assert!(!is_snake_case_tool_name("is-a"));
        assert!(!is_snake_case_tool_name("child-of"));
        assert!(!is_snake_case_tool_name("part-of"));
        // Uppercase entity names must not be flagged.
        assert!(!is_snake_case_tool_name("Ptah"));
        assert!(!is_snake_case_tool_name("System"));
        // Labels with spaces must not be flagged.
        assert!(!is_snake_case_tool_name("CI runner"));
    }

    #[test]
    fn uuid_detection() {
        assert!(is_uuid("550e8400-e29b-41d4-a716-446655440000"));
        assert!(!is_uuid("Ptah"));
        assert!(!is_uuid("episode:42"));
        assert!(!is_uuid("550e8400-e29b-41d4-a716")); // too short
    }

    #[test]
    fn invalid_type_hierarchy_participant_guard() {
        // is-a with infrastructure objects must be blocked.
        assert!(is_invalid_type_hierarchy_participant(
            "is-a",
            "ForgejoActions"
        ));
        assert!(is_invalid_type_hierarchy_participant("is-a", "CI runner"));
        assert!(is_invalid_type_hierarchy_participant("is-a", "Stor01"));
        assert!(is_invalid_type_hierarchy_participant("is-a", "runner-01"));
        assert!(is_invalid_type_hierarchy_participant("is-a", "infer_rules"));
        assert!(is_invalid_type_hierarchy_participant("is-a", "goal:xyz"));
        assert!(is_invalid_type_hierarchy_participant(
            "is-a",
            "550e8400-e29b-41d4-a716-446655440000"
        ));

        // child-of with the same patterns must be blocked.
        assert!(is_invalid_type_hierarchy_participant(
            "child-of",
            "episode:42"
        ));
        assert!(is_invalid_type_hierarchy_participant(
            "child-of", "kg_query"
        ));
        assert!(is_invalid_type_hierarchy_participant(
            "child-of",
            "ForgejoActions"
        ));

        // Valid type-hierarchy participants must NOT be blocked.
        assert!(!is_invalid_type_hierarchy_participant("is-a", "Ptah"));
        assert!(!is_invalid_type_hierarchy_participant("is-a", "System"));
        assert!(!is_invalid_type_hierarchy_participant("is-a", "Concept"));
        assert!(!is_invalid_type_hierarchy_participant("child-of", "Khnum"));

        // Non-taxonomic predicates must always pass (guard is a no-op).
        assert!(!is_invalid_type_hierarchy_participant(
            "part-of",
            "ForgejoActions"
        ));
        assert!(!is_invalid_type_hierarchy_participant(
            "similar-to",
            "runner"
        ));
    }

    #[test]
    fn infrastructure_object_blocked_from_is_a_derivation() {
        let engine = test_engine();
        // Ptah is-a ForgejoActions should not be derived even if a
        // chain X is-a ForgejoActions, Ptah is-a X exists.
        engine
            .ingest_label_triples(&[
                ("Ptah".into(), "is-a".into(), "AgentSystem".into(), 1.0),
                (
                    "AgentSystem".into(),
                    "is-a".into(),
                    "ForgejoActions".into(),
                    1.0,
                ),
            ])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig::default()).with_rules(RuleSet::builtin());
        let result = re.run(&engine).unwrap();

        let ptah = engine.lookup_symbol("Ptah").unwrap();
        let forgejo = engine.lookup_symbol("ForgejoActions").unwrap();
        let isa = engine.lookup_symbol("is-a").unwrap();

        // Ptah is-a ForgejoActions must NOT be derived.
        assert!(!engine.has_triple(ptah, isa, forgejo));
        // The rule produced zero valid derivations.
        let count = result
            .rule_stats
            .get("is-a-transitive")
            .copied()
            .unwrap_or(0);
        assert_eq!(
            count, 0,
            "is-a-transitive should derive nothing through infrastructure"
        );
    }

    #[test]
    fn tool_name_blocked_from_child_of_derivation() {
        let engine = test_engine();
        // "Ptah parent-of infer_rules" → would produce "infer_rules child-of Ptah"
        // but infer_rules is a tool name and must be blocked.
        engine
            .ingest_label_triples(&[("Ptah".into(), "parent-of".into(), "infer_rules".into(), 1.0)])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig::default()).with_rules(RuleSet::builtin());
        re.run(&engine).unwrap();

        let infer_rules_sym = engine.lookup_symbol("infer_rules").unwrap();
        let ptah = engine.lookup_symbol("Ptah").unwrap();
        let child_of = engine.lookup_symbol("child-of").unwrap();

        assert!(!engine.has_triple(infer_rules_sym, child_of, ptah));
    }
}
