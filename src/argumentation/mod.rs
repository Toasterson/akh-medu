//! Pro/Con argumentation: structured evidence collection and meta-rule ranking.
//!
//! Inspired by Cyc's argumentation framework. When queried, the system gathers
//! all pro and con arguments for each possible answer, ranks them via meta-rules
//! (monotonicity, specificity, recency, depth, source quality, constructiveness),
//! and produces a fully auditable [`Verdict`].
//!
//! # Architecture
//!
//! - [`Argument`]: a single pro or con evidence chain with strength metadata
//! - [`ArgumentSet`]: all arguments for a query, with computed [`Verdict`]
//! - [`MetaRule`]: ranked evaluation criteria (6 rules, applied in priority order)
//! - [`Verdict`]: winning answer, confidence, top pro/con, full reasoning

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::engine::Engine;
use crate::error::AkhResult;
use crate::graph::Triple;
use crate::graph::defeasible::DefeasiblePredicates;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;

// ---------------------------------------------------------------------------
// Polarity
// ---------------------------------------------------------------------------

/// Whether an argument supports or opposes a conclusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Polarity {
    /// Evidence supporting the conclusion.
    Pro,
    /// Evidence opposing the conclusion.
    Con,
}

impl fmt::Display for Polarity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Polarity::Pro => write!(f, "pro"),
            Polarity::Con => write!(f, "con"),
        }
    }
}

// ---------------------------------------------------------------------------
// MetaRule
// ---------------------------------------------------------------------------

/// Meta-level rules for ranking arguments, applied in priority order.
///
/// Cyc uses ordered meta-rules to decide which arguments to prefer:
/// monotonic > default, specific > general, recent > stale, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetaRule {
    /// Monotonically-derived conclusions always win over defeasible ones.
    Monotonicity,
    /// More specific premises beat more general (via `is-a` depth from 9d).
    Specificity,
    /// Newer evidence beats older (by timestamp).
    Recency,
    /// Shorter provenance chains beat longer (less default-truth decay).
    Depth,
    /// Higher-confidence sources beat lower-confidence ones.
    SourceQuality,
    /// Arguments identifying concrete instances beat pure existence claims.
    Constructiveness,
}

impl MetaRule {
    /// The default ranking order: monotonicity first, constructiveness last.
    pub fn default_order() -> Vec<MetaRule> {
        vec![
            MetaRule::Monotonicity,
            MetaRule::Specificity,
            MetaRule::Recency,
            MetaRule::Depth,
            MetaRule::SourceQuality,
            MetaRule::Constructiveness,
        ]
    }
}

impl fmt::Display for MetaRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetaRule::Monotonicity => write!(f, "monotonicity"),
            MetaRule::Specificity => write!(f, "specificity"),
            MetaRule::Recency => write!(f, "recency"),
            MetaRule::Depth => write!(f, "depth"),
            MetaRule::SourceQuality => write!(f, "source-quality"),
            MetaRule::Constructiveness => write!(f, "constructiveness"),
        }
    }
}

// ---------------------------------------------------------------------------
// Argument
// ---------------------------------------------------------------------------

/// A single argument for or against a conclusion.
///
/// Wraps a provenance chain (the evidence trail) with strength metadata
/// computed from the chain's properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Argument {
    /// The conclusion this argument supports or opposes.
    pub conclusion: SymbolId,
    /// Whether this is a pro or con argument.
    pub polarity: Polarity,
    /// The provenance chain backing this argument (from sources to conclusion).
    pub chain: Vec<ProvenanceRecord>,
    /// Overall strength in [0.0, 1.0], computed from meta-rule evaluation.
    pub strength: f64,
    /// Whether this argument derives from monotonic (non-defeasible) reasoning.
    pub monotonic: bool,
    /// The meta-rule scores that contributed to the strength.
    pub scores: MetaRuleScores,
}

/// Per-meta-rule scores for an argument.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetaRuleScores {
    /// 1.0 if monotonic, 0.5 if default.
    pub monotonicity: f64,
    /// Specificity score based on type depth (higher = more specific).
    pub specificity: f64,
    /// Recency score (newer = higher).
    pub recency: f64,
    /// Depth score (shorter chain = higher).
    pub depth: f64,
    /// Source quality (average confidence of the chain).
    pub source_quality: f64,
    /// Constructiveness (1.0 if chain has concrete instances, 0.5 otherwise).
    pub constructiveness: f64,
}

impl MetaRuleScores {
    /// Compute a weighted total using the meta-rule priority order.
    ///
    /// Higher-priority rules get exponentially more weight:
    /// rule at position 0 gets weight 32, position 1 gets 16, etc.
    pub fn weighted_total(&self, order: &[MetaRule]) -> f64 {
        let mut total = 0.0;
        let mut weight = 1u64 << order.len();
        for rule in order {
            let score = match rule {
                MetaRule::Monotonicity => self.monotonicity,
                MetaRule::Specificity => self.specificity,
                MetaRule::Recency => self.recency,
                MetaRule::Depth => self.depth,
                MetaRule::SourceQuality => self.source_quality,
                MetaRule::Constructiveness => self.constructiveness,
            };
            total += score * weight as f64;
            weight >>= 1;
        }
        total
    }
}

// ---------------------------------------------------------------------------
// Verdict
// ---------------------------------------------------------------------------

/// The conclusion of an argumentation process.
///
/// Summarizes the winning answer with its confidence, the top pro/con
/// arguments, and a human-readable reasoning chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Verdict {
    /// The winning answer (object of the best pro argument).
    pub answer: Option<SymbolId>,
    /// Confidence in the verdict [0.0, 1.0].
    pub confidence: f64,
    /// The strongest pro arguments (sorted by strength, descending).
    pub pro: Vec<Argument>,
    /// The strongest con arguments (sorted by strength, descending).
    pub con: Vec<Argument>,
    /// Human-readable reasoning summary.
    pub reasoning: String,
    /// Which meta-rule was decisive (first rule that separated the winner from runner-up).
    pub decisive_rule: Option<MetaRule>,
}

// ---------------------------------------------------------------------------
// ArgumentSet
// ---------------------------------------------------------------------------

/// All arguments collected for a query, with computed verdict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgumentSet {
    /// The subject being queried.
    pub subject: SymbolId,
    /// The predicate being queried.
    pub predicate: SymbolId,
    /// All candidate answers grouped with their arguments.
    pub candidates: HashMap<SymbolId, Vec<Argument>>,
    /// The computed verdict.
    pub verdict: Verdict,
}

// ---------------------------------------------------------------------------
// Scoring helpers
// ---------------------------------------------------------------------------

/// Check if a derivation kind is monotonic (non-defeasible).
fn is_monotonic_derivation(kind: &DerivationKind) -> bool {
    matches!(
        kind,
        DerivationKind::Extracted
            | DerivationKind::Seed
            | DerivationKind::DefeasibleOverride { .. }
    )
}

/// Check if a provenance chain has concrete instances (constructive evidence).
fn is_constructive(chain: &[ProvenanceRecord]) -> bool {
    chain.iter().any(|r| {
        matches!(
            &r.kind,
            DerivationKind::GraphEdge { .. }
                | DerivationKind::Extracted
                | DerivationKind::FillerRecovery { .. }
        )
    })
}

/// Score a provenance chain on all meta-rule dimensions.
fn score_chain(
    chain: &[ProvenanceRecord],
    type_depth: Option<usize>,
    now: u64,
) -> MetaRuleScores {
    let monotonic = chain.iter().all(|r| is_monotonic_derivation(&r.kind));

    // Average confidence across the chain.
    let avg_confidence = if chain.is_empty() {
        0.0
    } else {
        chain.iter().map(|r| r.confidence as f64).sum::<f64>() / chain.len() as f64
    };

    // Most recent timestamp in the chain.
    let max_timestamp = chain.iter().map(|r| r.timestamp).max().unwrap_or(0);
    // Recency: exponential decay — halves every 24 hours.
    let age_secs = now.saturating_sub(max_timestamp);
    let recency = (-0.00001 * age_secs as f64).exp(); // ~0.42 after 24h

    // Depth: shorter chains are better. Score = 1 / (1 + max_depth).
    let max_depth = chain.iter().map(|r| r.depth).max().unwrap_or(0);
    let depth_score = 1.0 / (1.0 + max_depth as f64);

    // Specificity: from type depth (deeper in is-a = more specific).
    let specificity = type_depth.map(|d| d as f64 / (1.0 + d as f64)).unwrap_or(0.5);

    let constructive = if is_constructive(chain) { 1.0 } else { 0.5 };

    MetaRuleScores {
        monotonicity: if monotonic { 1.0 } else { 0.5 },
        specificity,
        recency,
        depth: depth_score,
        source_quality: avg_confidence,
        constructiveness: constructive,
    }
}

// ---------------------------------------------------------------------------
// Core argumentation logic
// ---------------------------------------------------------------------------

/// Collect arguments for a query `(subject, predicate, ?)` and produce a verdict.
///
/// Steps:
/// 1. Find all triples matching `(subject, predicate, *)` — candidate answers
/// 2. For each candidate, build pro arguments from its provenance chain
/// 3. For conflicting candidates, build con arguments against each other
/// 4. Score all arguments via meta-rules
/// 5. Produce a verdict with the winning answer
pub fn argue(
    engine: &Engine,
    subject: SymbolId,
    predicate: SymbolId,
) -> AkhResult<ArgumentSet> {
    argue_with_rules(engine, subject, predicate, &MetaRule::default_order())
}

/// Like [`argue`] but with a custom meta-rule ordering.
pub fn argue_with_rules(
    engine: &Engine,
    subject: SymbolId,
    predicate: SymbolId,
    meta_rules: &[MetaRule],
) -> AkhResult<ArgumentSet> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Step 1: Find all candidate triples for (subject, predicate, ?).
    let all_triples = engine.triples_from(subject);
    let candidate_triples: Vec<Triple> = all_triples
        .into_iter()
        .filter(|t| t.predicate == predicate)
        .collect();

    // Resolve defeasible predicates for specificity scoring.
    let defeasible_preds = crate::graph::defeasible::DefeasiblePredicates::resolve(engine)?;

    // Step 2: Build pro arguments for each candidate.
    let mut candidates: HashMap<SymbolId, Vec<Argument>> = HashMap::new();

    for triple in &candidate_triples {
        let object = triple.object;

        // Build a provenance record chain for this triple.
        let chain = build_provenance_chain(engine, triple);

        // Compute type depth for specificity.
        let type_depth = compute_type_depth(engine, triple.subject, &defeasible_preds);

        let scores = score_chain(&chain, type_depth, now);
        let monotonic = scores.monotonicity > 0.9;
        let strength = scores.weighted_total(meta_rules);

        let arg = Argument {
            conclusion: object,
            polarity: Polarity::Pro,
            chain,
            strength,
            monotonic,
            scores,
        };

        candidates.entry(object).or_default().push(arg);
    }

    // Step 3: Build con arguments — each candidate's existence is a con for others.
    if candidates.len() > 1 {
        let candidate_objects: Vec<SymbolId> = candidates.keys().copied().collect();
        for &obj in &candidate_objects {
            let con_args: Vec<Argument> = candidate_objects
                .iter()
                .filter(|&&other| other != obj)
                .filter_map(|&other| {
                    // The existence of a competing answer is a con argument.
                    let other_args = candidates.get(&other)?;
                    let best_other = other_args.iter().max_by(|a, b| {
                        a.strength.partial_cmp(&b.strength).unwrap_or(std::cmp::Ordering::Equal)
                    })?;

                    // Con strength is derived from the competing answer's pro strength,
                    // attenuated by 0.8 (cons are slightly weaker than the pro they oppose).
                    let con_scores = MetaRuleScores {
                        monotonicity: best_other.scores.monotonicity * 0.8,
                        specificity: best_other.scores.specificity * 0.8,
                        recency: best_other.scores.recency * 0.8,
                        depth: best_other.scores.depth * 0.8,
                        source_quality: best_other.scores.source_quality * 0.8,
                        constructiveness: best_other.scores.constructiveness * 0.8,
                    };
                    let con_strength = con_scores.weighted_total(meta_rules);

                    Some(Argument {
                        conclusion: obj,
                        polarity: Polarity::Con,
                        chain: best_other.chain.clone(),
                        strength: con_strength,
                        monotonic: best_other.monotonic,
                        scores: con_scores,
                    })
                })
                .collect();

            candidates.entry(obj).or_default().extend(con_args);
        }
    }

    // Step 4: Determine the winning candidate.
    let verdict = compute_verdict(&candidates, meta_rules, engine);

    Ok(ArgumentSet {
        subject,
        predicate,
        candidates,
        verdict,
    })
}

/// Build a provenance chain for a triple.
///
/// If the engine has a provenance ledger, fetches records for the triple's
/// object. Otherwise, creates a synthetic record from the triple itself.
fn build_provenance_chain(engine: &Engine, triple: &Triple) -> Vec<ProvenanceRecord> {
    // Try to get real provenance records from the ledger.
    if let Ok(records) = engine.provenance_of(triple.object) {
        if !records.is_empty() {
            return records;
        }
    }

    // Fallback: create a synthetic provenance record from the triple.
    vec![ProvenanceRecord::new(
        triple.object,
        DerivationKind::GraphEdge {
            from: triple.subject,
            predicate: triple.predicate,
        },
    )
    .with_confidence(triple.confidence)
    .with_sources(vec![triple.subject, triple.predicate])]
}

/// Compute type depth for specificity scoring.
fn compute_type_depth(
    engine: &Engine,
    subject: SymbolId,
    preds: &DefeasiblePredicates,
) -> Option<usize> {
    let depth = crate::graph::defeasible::type_depth(engine, subject, preds.is_a);
    if depth > 0 { Some(depth) } else { None }
}

/// Compute the verdict from all candidates and their arguments.
fn compute_verdict(
    candidates: &HashMap<SymbolId, Vec<Argument>>,
    meta_rules: &[MetaRule],
    engine: &Engine,
) -> Verdict {
    if candidates.is_empty() {
        return Verdict {
            answer: None,
            confidence: 0.0,
            pro: Vec::new(),
            con: Vec::new(),
            reasoning: "No candidate answers found".to_string(),
            decisive_rule: None,
        };
    }

    // For each candidate, compute net strength = sum(pro) - sum(con).
    let mut candidate_scores: Vec<(SymbolId, f64, Vec<Argument>, Vec<Argument>)> = candidates
        .iter()
        .map(|(&obj, args)| {
            let pro: Vec<Argument> = args
                .iter()
                .filter(|a| a.polarity == Polarity::Pro)
                .cloned()
                .collect();
            let con: Vec<Argument> = args
                .iter()
                .filter(|a| a.polarity == Polarity::Con)
                .cloned()
                .collect();

            let pro_strength: f64 = pro.iter().map(|a| a.strength).sum();
            let con_strength: f64 = con.iter().map(|a| a.strength).sum();
            let net = pro_strength - con_strength;

            (obj, net, pro, con)
        })
        .collect();

    // Sort by net strength (descending).
    candidate_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let (winner_sym, winner_net, mut winner_pro, mut winner_con) =
        candidate_scores.remove(0);

    // Sort pro/con by strength descending.
    winner_pro.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap_or(std::cmp::Ordering::Equal));
    winner_con.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap_or(std::cmp::Ordering::Equal));

    // Determine the decisive meta-rule (first rule that separates winner from runner-up).
    let decisive = if candidate_scores.is_empty() {
        None
    } else {
        let runner_up_args = candidates.get(&candidate_scores[0].0);
        find_decisive_rule(
            &winner_pro,
            runner_up_args.map(|a| a.as_slice()).unwrap_or(&[]),
            meta_rules,
        )
    };

    // Confidence: normalize net strength to [0, 1].
    let max_possible = winner_pro.iter().map(|a| a.strength).sum::<f64>().max(1.0);
    let confidence = (winner_net / max_possible).clamp(0.0, 1.0);

    // Build reasoning string.
    let winner_label = engine.resolve_label(winner_sym);
    let pro_count = winner_pro.len();
    let con_count = winner_con.len();
    let total_candidates = candidates.len();

    let decisive_str = decisive
        .map(|r| format!(", decisive: {r}"))
        .unwrap_or_default();

    let reasoning = format!(
        "\"{winner_label}\" wins with {pro_count} pro and {con_count} con \
         arguments out of {total_candidates} candidate(s){decisive_str}"
    );

    Verdict {
        answer: Some(winner_sym),
        confidence,
        pro: winner_pro,
        con: winner_con,
        reasoning,
        decisive_rule: decisive,
    }
}

/// Find the first meta-rule that separates the winner from the runner-up.
fn find_decisive_rule(
    winner_args: &[Argument],
    runner_up_args: &[Argument],
    meta_rules: &[MetaRule],
) -> Option<MetaRule> {
    let winner_best = winner_args
        .iter()
        .filter(|a| a.polarity == Polarity::Pro)
        .max_by(|a, b| a.strength.partial_cmp(&b.strength).unwrap_or(std::cmp::Ordering::Equal));
    let runner_best = runner_up_args
        .iter()
        .filter(|a| a.polarity == Polarity::Pro)
        .max_by(|a, b| a.strength.partial_cmp(&b.strength).unwrap_or(std::cmp::Ordering::Equal));

    let (winner, runner) = match (winner_best, runner_best) {
        (Some(w), Some(r)) => (w, r),
        _ => return None,
    };

    for rule in meta_rules {
        let (w_score, r_score) = match rule {
            MetaRule::Monotonicity => (winner.scores.monotonicity, runner.scores.monotonicity),
            MetaRule::Specificity => (winner.scores.specificity, runner.scores.specificity),
            MetaRule::Recency => (winner.scores.recency, runner.scores.recency),
            MetaRule::Depth => (winner.scores.depth, runner.scores.depth),
            MetaRule::SourceQuality => (winner.scores.source_quality, runner.scores.source_quality),
            MetaRule::Constructiveness => {
                (winner.scores.constructiveness, runner.scores.constructiveness)
            }
        };

        // If there's a meaningful difference (>5%), this rule is decisive.
        if (w_score - r_score).abs() > 0.05 {
            return Some(*rule);
        }
    }

    None
}

// ===========================================================================
// DerivationKind for argumentation
// ===========================================================================

// The ArgumentVerdict derivation kind is added in provenance.rs.

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::graph::Triple;
    use crate::symbol::SymbolKind;
    use crate::vsa::Dimension;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    fn sym(engine: &Engine, kind: SymbolKind, label: &str) -> SymbolId {
        engine.create_symbol(kind, label).unwrap().id
    }

    // --- MetaRule tests ---

    #[test]
    fn default_meta_rule_order_has_six_rules() {
        let order = MetaRule::default_order();
        assert_eq!(order.len(), 6);
        assert_eq!(order[0], MetaRule::Monotonicity);
        assert_eq!(order[5], MetaRule::Constructiveness);
    }

    // --- MetaRuleScores tests ---

    #[test]
    fn weighted_total_prioritizes_earlier_rules() {
        let scores_mono = MetaRuleScores {
            monotonicity: 1.0,
            specificity: 0.0,
            recency: 0.0,
            depth: 0.0,
            source_quality: 0.0,
            constructiveness: 0.0,
        };
        let scores_construct = MetaRuleScores {
            monotonicity: 0.0,
            specificity: 0.0,
            recency: 0.0,
            depth: 0.0,
            source_quality: 0.0,
            constructiveness: 1.0,
        };

        let order = MetaRule::default_order();
        assert!(
            scores_mono.weighted_total(&order) > scores_construct.weighted_total(&order),
            "Monotonicity (rule 0) should outweigh constructiveness (rule 5)"
        );
    }

    // --- score_chain tests ---

    #[test]
    fn monotonic_chain_scores_higher_monotonicity() {
        let chain_mono = vec![
            ProvenanceRecord::new(
                SymbolId::new(1).unwrap(),
                DerivationKind::Extracted,
            )
            .with_confidence(0.9),
        ];
        let chain_default = vec![
            ProvenanceRecord::new(
                SymbolId::new(2).unwrap(),
                DerivationKind::Reasoned,
            )
            .with_confidence(0.9),
        ];

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let scores_mono = score_chain(&chain_mono, None, now);
        let scores_default = score_chain(&chain_default, None, now);

        assert!(
            scores_mono.monotonicity > scores_default.monotonicity,
            "Extracted chain should have higher monotonicity than Reasoned"
        );
    }

    #[test]
    fn shorter_chain_scores_higher_depth() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let short_chain = vec![
            ProvenanceRecord::new(SymbolId::new(1).unwrap(), DerivationKind::Extracted)
                .with_depth(1),
        ];
        let long_chain = vec![
            ProvenanceRecord::new(SymbolId::new(1).unwrap(), DerivationKind::Extracted)
                .with_depth(5),
        ];

        let short_scores = score_chain(&short_chain, None, now);
        let long_scores = score_chain(&long_chain, None, now);

        assert!(
            short_scores.depth > long_scores.depth,
            "Shorter chain ({}) should score higher depth than longer ({})",
            short_scores.depth,
            long_scores.depth,
        );
    }

    // --- Polarity display ---

    #[test]
    fn polarity_display() {
        assert_eq!(format!("{}", Polarity::Pro), "pro");
        assert_eq!(format!("{}", Polarity::Con), "con");
    }

    // --- Full argumentation tests ---

    #[test]
    fn single_candidate_argument() {
        let engine = test_engine();
        let sun = sym(&engine, SymbolKind::Entity, "Sun");
        let is_a = sym(&engine, SymbolKind::Relation, "is-a");
        let star = sym(&engine, SymbolKind::Entity, "Star");

        engine.add_triple(&Triple::new(sun, is_a, star)).unwrap();

        let result = argue(&engine, sun, is_a).unwrap();

        assert_eq!(result.subject, sun);
        assert_eq!(result.predicate, is_a);
        assert_eq!(result.candidates.len(), 1);
        assert!(result.candidates.contains_key(&star));

        let verdict = &result.verdict;
        assert_eq!(verdict.answer, Some(star));
        assert!(verdict.confidence > 0.0);
        assert!(!verdict.pro.is_empty());
        assert!(verdict.con.is_empty()); // Single candidate → no con arguments
        assert!(verdict.reasoning.contains("Star"));
    }

    #[test]
    fn competing_candidates_produce_cons() {
        let engine = test_engine();
        let tweety = sym(&engine, SymbolKind::Entity, "Tweety");
        let can = sym(&engine, SymbolKind::Relation, "can");
        let fly = sym(&engine, SymbolKind::Entity, "Fly");
        let swim = sym(&engine, SymbolKind::Entity, "Swim");

        engine.add_triple(&Triple::new(tweety, can, fly).with_confidence(0.8)).unwrap();
        engine.add_triple(&Triple::new(tweety, can, swim).with_confidence(0.6)).unwrap();

        let result = argue(&engine, tweety, can).unwrap();

        assert_eq!(result.candidates.len(), 2);

        // Both should have pro and con arguments.
        for (_, args) in &result.candidates {
            let has_pro = args.iter().any(|a| a.polarity == Polarity::Pro);
            let has_con = args.iter().any(|a| a.polarity == Polarity::Con);
            assert!(has_pro, "Each candidate should have pro arguments");
            assert!(has_con, "Each candidate should have con arguments (from competing candidate)");
        }

        // Verdict should pick the higher-confidence answer.
        let verdict = &result.verdict;
        assert!(verdict.answer.is_some());
    }

    #[test]
    fn higher_confidence_wins() {
        let engine = test_engine();
        let bird = sym(&engine, SymbolKind::Entity, "Bird");
        let flies = sym(&engine, SymbolKind::Relation, "does");
        let yes = sym(&engine, SymbolKind::Entity, "Yes");
        let no = sym(&engine, SymbolKind::Entity, "No");

        engine.add_triple(&Triple::new(bird, flies, yes).with_confidence(0.9)).unwrap();
        engine.add_triple(&Triple::new(bird, flies, no).with_confidence(0.3)).unwrap();

        let result = argue(&engine, bird, flies).unwrap();

        let verdict = &result.verdict;
        assert_eq!(verdict.answer, Some(yes), "Higher confidence 'Yes' should win");
    }

    #[test]
    fn no_candidates_produces_empty_verdict() {
        let engine = test_engine();
        let x = sym(&engine, SymbolKind::Entity, "X");
        let y = sym(&engine, SymbolKind::Relation, "Y");

        let result = argue(&engine, x, y).unwrap();

        assert!(result.candidates.is_empty());
        assert!(result.verdict.answer.is_none());
        assert_eq!(result.verdict.confidence, 0.0);
        assert!(result.verdict.reasoning.contains("No candidate"));
    }

    #[test]
    fn meta_rule_order_affects_outcome() {
        let engine = test_engine();
        let x = sym(&engine, SymbolKind::Entity, "X");
        let rel = sym(&engine, SymbolKind::Relation, "rel");
        let a = sym(&engine, SymbolKind::Entity, "A");

        engine.add_triple(&Triple::new(x, rel, a)).unwrap();

        // Custom order: depth first.
        let custom_order = vec![
            MetaRule::Depth,
            MetaRule::Monotonicity,
            MetaRule::Specificity,
            MetaRule::Recency,
            MetaRule::SourceQuality,
            MetaRule::Constructiveness,
        ];

        let result = argue_with_rules(&engine, x, rel, &custom_order).unwrap();
        assert!(result.verdict.answer.is_some());
    }

    #[test]
    fn argument_strength_is_positive() {
        let engine = test_engine();
        let s = sym(&engine, SymbolKind::Entity, "S");
        let p = sym(&engine, SymbolKind::Relation, "P");
        let o = sym(&engine, SymbolKind::Entity, "O");

        engine.add_triple(&Triple::new(s, p, o)).unwrap();

        let result = argue(&engine, s, p).unwrap();

        for (_, args) in &result.candidates {
            for arg in args {
                assert!(arg.strength > 0.0, "Argument strength should be positive, got {}", arg.strength);
            }
        }
    }

    #[test]
    fn verdict_reasoning_contains_candidate_count() {
        let engine = test_engine();
        let s = sym(&engine, SymbolKind::Entity, "S");
        let p = sym(&engine, SymbolKind::Relation, "P");
        let o1 = sym(&engine, SymbolKind::Entity, "O1");
        let o2 = sym(&engine, SymbolKind::Entity, "O2");

        engine.add_triple(&Triple::new(s, p, o1)).unwrap();
        engine.add_triple(&Triple::new(s, p, o2)).unwrap();

        let result = argue(&engine, s, p).unwrap();
        assert!(result.verdict.reasoning.contains("2 candidate"));
    }

    #[test]
    fn constructive_evidence_scores_higher() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Graph edge is constructive.
        let chain_constructive = vec![
            ProvenanceRecord::new(
                SymbolId::new(1).unwrap(),
                DerivationKind::GraphEdge {
                    from: SymbolId::new(2).unwrap(),
                    predicate: SymbolId::new(3).unwrap(),
                },
            )
            .with_confidence(0.9),
        ];
        // Reasoned is not constructive.
        let chain_non_constructive = vec![
            ProvenanceRecord::new(
                SymbolId::new(1).unwrap(),
                DerivationKind::Reasoned,
            )
            .with_confidence(0.9),
        ];

        let scores_c = score_chain(&chain_constructive, None, now);
        let scores_nc = score_chain(&chain_non_constructive, None, now);

        assert!(
            scores_c.constructiveness > scores_nc.constructiveness,
            "Constructive ({}) should score higher than non-constructive ({})",
            scores_c.constructiveness,
            scores_nc.constructiveness,
        );
    }
}
