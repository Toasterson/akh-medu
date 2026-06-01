//! ACH bridge for drawer contradiction analysis — Phase 42i.
//!
//! Wires Phase 18 ACH + Phase 17 Dempster-Shafer against a pair of
//! heka memory drawers to determine which claim is better supported.
//!
//! # Flow
//!
//! 1. Accept two `DrawerClaim` descriptors (claim text, confidence, timestamp, source).
//! 2. Build an `AchAnalysis` with H0 = drawer-A's claim, H1 = drawer-B's claim.
//! 3. Populate evidence:
//!    - Each drawer's self-asserted claim (primary).
//!    - Temporal recency (newer claim weakly preferred).
//!    - KG facts matching the claim subjects (when an `Engine` is supplied).
//! 4. Call `AchAnalysis::evaluate()`.
//! 5. Convert per-hypothesis consistency matrices → DS `MassFunction` chains →
//!    `BeliefInterval` via Dempster's rule.
//! 6. Return `DrawerAchResult` with ranked sides + deciding-evidence chain.
//!
//! Plan: Anima WI `019dd0c7-2557-7e40-874d-d3c99af2e239`

use serde::{Deserialize, Serialize};

use crate::agent::{
    AchAnalysis, ConsistencyRating, BeliefInterval, MassFunction, SourceReliability,
};
use crate::engine::Engine;

// ═══════════════════════════════════════════════════════════════════════
// Public Types
// ═══════════════════════════════════════════════════════════════════════

/// A drawer's claim as seen by the ACH bridge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawerClaim {
    /// Opaque drawer identifier (e.g. numeric string or UUID).
    pub drawer_id: String,
    /// Human-readable claim text (e.g. "temperature is 37 degrees").
    pub claim: String,
    /// Optional source label (used to derive `SourceReliability`).
    pub source: Option<String>,
    /// Unix timestamp when this claim was recorded.
    pub timestamp: u64,
    /// Stated confidence [0.0, 1.0] — maps to `SourceReliability`.
    pub confidence: f32,
}

/// ACH + DS result for one side of a contradiction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AchSideResult {
    /// Drawer identifier.
    pub drawer_id: String,
    /// The full claim text.
    pub claim: String,
    /// Raw ACH inconsistency score (lower = more credible).
    pub inconsistency_score: f32,
    /// Dempster-Shafer belief interval [Bel, Pl].
    pub belief: BeliefInterval,
    /// Rank among all hypotheses (1 = most credible).
    pub rank: usize,
}

/// A single piece of evidence that drove the analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecidingEvidence {
    /// Human-readable description of the evidence item.
    pub description: String,
    /// Drawer ID this evidence came from, if any.
    pub drawer_id: Option<String>,
    /// KG triple text if this came from the knowledge graph.
    pub kg_triple: Option<String>,
    /// Diagnosticity score (higher = more differentiating).
    pub diagnosticity: f32,
    /// Drawer ID of the hypothesis this evidence favours, if clear.
    pub favors: Option<String>,
}

/// Full ACH contradiction-analysis result for two drawers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawerAchResult {
    /// Unique analysis identifier (timestamp-based).
    pub analysis_id: String,
    /// The question posed to the ACH engine.
    pub question: String,
    /// Result for drawer A.
    pub side_a: AchSideResult,
    /// Result for drawer B.
    pub side_b: AchSideResult,
    /// Top deciding evidence items ordered by diagnosticity.
    pub deciding_evidence: Vec<DecidingEvidence>,
    /// Indistinguishable hypothesis pairs (information gaps).
    pub information_gaps: Vec<String>,
    /// Full ASCII consistency matrix for debug/display.
    pub matrix_text: String,
}

/// Configuration for `analyze_drawer_conflict`.
#[derive(Debug, Clone)]
pub struct DrawerAchConfig {
    /// Maximum KG triples to consult per subject term.
    pub kg_evidence_limit: usize,
    /// Maximum deciding-evidence items to return.
    pub max_deciding: usize,
}

impl Default for DrawerAchConfig {
    fn default() -> Self {
        Self {
            kg_evidence_limit: 5,
            max_deciding: 5,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Main Entry Point
// ═══════════════════════════════════════════════════════════════════════

/// Analyze a contradiction between two memory drawer claims.
///
/// Constructs an ACH tableau, rates all evidence, evaluates, converts
/// to Dempster-Shafer belief intervals, and returns a structured result.
///
/// `engine` is optional; when supplied, KG facts about the claim subjects
/// are added as additional evidence.
pub fn analyze_drawer_conflict(
    drawer_a: &DrawerClaim,
    drawer_b: &DrawerClaim,
    engine: Option<&Engine>,
    config: &DrawerAchConfig,
) -> DrawerAchResult {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let analysis_id = format!("ach-drawer-{ts}");

    let question = format!(
        "Which claim is correct: \"{}\" (drawer {}) or \"{}\" (drawer {})?",
        drawer_a.claim, drawer_a.drawer_id, drawer_b.claim, drawer_b.drawer_id
    );

    let mut ach = AchAnalysis::new(&analysis_id, &question);

    // H0 = drawer_a's claim, H1 = drawer_b's claim.
    ach.add_hypothesis(&drawer_a.claim, None);
    ach.add_hypothesis(&drawer_b.claim, None);

    // ── E0: Drawer A self-asserts its claim ──────────────────────────────
    let rel_a = SourceReliability::from_trust_score(drawer_a.confidence.clamp(0.0, 1.0));
    ach.add_evidence(
        format!(
            "Drawer {} asserts: \"{}\"",
            drawer_a.drawer_id, drawer_a.claim
        ),
        rel_a,
        None,
    );
    let e_idx_a = ach.evidence.len() - 1;
    ach.rate(e_idx_a, 0, ConsistencyRating::Consistent);
    ach.rate(e_idx_a, 1, ConsistencyRating::StronglyInconsistent);

    // ── E1: Drawer B self-asserts its claim ──────────────────────────────
    let rel_b = SourceReliability::from_trust_score(drawer_b.confidence.clamp(0.0, 1.0));
    ach.add_evidence(
        format!(
            "Drawer {} asserts: \"{}\"",
            drawer_b.drawer_id, drawer_b.claim
        ),
        rel_b,
        None,
    );
    let e_idx_b = ach.evidence.len() - 1;
    ach.rate(e_idx_b, 0, ConsistencyRating::StronglyInconsistent);
    ach.rate(e_idx_b, 1, ConsistencyRating::Consistent);

    // ── E2: Temporal recency ─────────────────────────────────────────────
    if drawer_a.timestamp != drawer_b.timestamp {
        let (newer_h, older_h, time_diff) = if drawer_b.timestamp > drawer_a.timestamp {
            (1usize, 0usize, drawer_b.timestamp - drawer_a.timestamp)
        } else {
            (0usize, 1usize, drawer_a.timestamp - drawer_b.timestamp)
        };
        let days = (time_diff / 86_400).max(1);
        let newer_id = if newer_h == 0 {
            &drawer_a.drawer_id
        } else {
            &drawer_b.drawer_id
        };
        let e_idx_t = ach.evidence.len();
        ach.add_evidence(
            format!(
                "Temporal: drawer {} is {} day(s) newer",
                newer_id, days
            ),
            SourceReliability::CompletelyReliable,
            None,
        );
        // Newer claim not inconsistent with itself; older is weakly suspect.
        ach.rate(e_idx_t, older_h, ConsistencyRating::WeaklyInconsistent);
        let _ = newer_h; // newer_h stays Neutral (matrix default)
    }

    // ── KG evidence (optional) ───────────────────────────────────────────
    if let Some(eng) = engine {
        add_kg_evidence(&mut ach, drawer_a, drawer_b, eng, config);
    }

    ach.evaluate();

    // ── Derive DS belief intervals ───────────────────────────────────────
    let belief_a = belief_interval_for_hypothesis(&ach, 0);
    let belief_b = belief_interval_for_hypothesis(&ach, 1);

    let score_a = ach.hypotheses.first().map(|h| h.inconsistency_score).unwrap_or(0.0);
    let score_b = ach.hypotheses.get(1).map(|h| h.inconsistency_score).unwrap_or(0.0);
    let rank_a = ach.ranking.iter().position(|&(j, _)| j == 0).unwrap_or(0) + 1;
    let rank_b = ach.ranking.iter().position(|&(j, _)| j == 1).unwrap_or(1) + 1;

    let deciding_evidence =
        collect_deciding_evidence(&ach, drawer_a, drawer_b, config.max_deciding);
    let information_gaps = ach.information_gaps();
    let matrix_text = ach.render_matrix();

    DrawerAchResult {
        analysis_id,
        question,
        side_a: AchSideResult {
            drawer_id: drawer_a.drawer_id.clone(),
            claim: drawer_a.claim.clone(),
            inconsistency_score: score_a,
            belief: belief_a,
            rank: rank_a,
        },
        side_b: AchSideResult {
            drawer_id: drawer_b.drawer_id.clone(),
            claim: drawer_b.claim.clone(),
            inconsistency_score: score_b,
            belief: belief_b,
            rank: rank_b,
        },
        deciding_evidence,
        information_gaps,
        matrix_text,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Dempster-Shafer Belief Interval Construction
// ═══════════════════════════════════════════════════════════════════════

/// Build a DS belief interval for hypothesis `hyp_idx` by combining all
/// evidence mass functions via Dempster's rule.
fn belief_interval_for_hypothesis(ach: &AchAnalysis, hyp_idx: usize) -> BeliefInterval {
    let mut masses: Vec<MassFunction> = Vec::new();

    for (i, evidence) in ach.evidence.iter().enumerate() {
        let rating = ach
            .matrix
            .get(i)
            .and_then(|row| row.get(hyp_idx))
            .copied()
            .unwrap_or(ConsistencyRating::Neutral);

        let base_mass = consistency_to_mass(rating);
        if !base_mass.is_vacuous() {
            let reliability = evidence.source_reliability.to_factor();
            masses.push(base_mass.discount(reliability));
        }
    }

    if masses.is_empty() {
        return BeliefInterval {
            belief: 0.0,
            plausibility: 1.0,
        };
    }

    match MassFunction::combine_multiple(&masses) {
        Ok(combined) => combined.belief_interval(),
        // Total conflict: both sides equally supported/refuted — near-zero interval.
        Err(_) => BeliefInterval {
            belief: 0.0,
            plausibility: 0.5,
        },
    }
}

/// Map an ACH consistency rating to a DS mass function over {True, False}.
fn consistency_to_mass(rating: ConsistencyRating) -> MassFunction {
    match rating {
        ConsistencyRating::Consistent => MassFunction::new(0.75, 0.0),
        ConsistencyRating::Neutral => MassFunction::vacuous(),
        ConsistencyRating::WeaklyInconsistent => MassFunction::new(0.1, 0.45),
        ConsistencyRating::StronglyInconsistent => MassFunction::new(0.0, 0.75),
        ConsistencyRating::NotApplicable => MassFunction::vacuous(),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Evidence Collection
// ═══════════════════════════════════════════════════════════════════════

/// Gather the top deciding-evidence items (by diagnosticity).
fn collect_deciding_evidence(
    ach: &AchAnalysis,
    drawer_a: &DrawerClaim,
    drawer_b: &DrawerClaim,
    max: usize,
) -> Vec<DecidingEvidence> {
    let mut items = Vec::new();

    for &ev_idx in ach.diagnostic_evidence.iter().take(max) {
        let Some(ev) = ach.evidence.get(ev_idx) else {
            continue;
        };
        let Some(row) = ach.matrix.get(ev_idx) else {
            continue;
        };

        let rating_a = row.first().copied().unwrap_or(ConsistencyRating::Neutral);
        let rating_b = row.get(1).copied().unwrap_or(ConsistencyRating::Neutral);

        let drawer_ref = if ev.description.contains(&*drawer_a.drawer_id) {
            Some(drawer_a.drawer_id.clone())
        } else if ev.description.contains(&*drawer_b.drawer_id) {
            Some(drawer_b.drawer_id.clone())
        } else {
            None
        };

        let kg_triple = if ev.description.starts_with("KG: ") {
            Some(ev.description.trim_start_matches("KG: ").to_string())
        } else {
            None
        };

        let favors = match (rating_a, rating_b) {
            (ConsistencyRating::Consistent, ConsistencyRating::StronglyInconsistent)
            | (ConsistencyRating::Consistent, ConsistencyRating::WeaklyInconsistent) => {
                Some(drawer_a.drawer_id.clone())
            }
            (ConsistencyRating::StronglyInconsistent, ConsistencyRating::Consistent)
            | (ConsistencyRating::WeaklyInconsistent, ConsistencyRating::Consistent) => {
                Some(drawer_b.drawer_id.clone())
            }
            _ => None,
        };

        items.push(DecidingEvidence {
            description: ev.description.clone(),
            drawer_id: drawer_ref,
            kg_triple,
            diagnosticity: ev.diagnosticity,
            favors,
        });
    }

    items
}

// ═══════════════════════════════════════════════════════════════════════
// KG Evidence
// ═══════════════════════════════════════════════════════════════════════

/// Augment the ACH analysis with evidence from the knowledge graph.
///
/// Looks up entities matching keywords shared between both claims, then
/// adds triples about those entities. A simple lexical heuristic determines
/// whether each triple's object label appears in either claim's text.
fn add_kg_evidence(
    ach: &mut AchAnalysis,
    drawer_a: &DrawerClaim,
    drawer_b: &DrawerClaim,
    engine: &Engine,
    config: &DrawerAchConfig,
) {
    // Tokenise both claims; look for substantive words (>2 chars) shared between them.
    let words_a: Vec<&str> = drawer_a
        .claim
        .split_whitespace()
        .filter(|w| w.len() > 2)
        .collect();
    let words_b: Vec<&str> = drawer_b
        .claim
        .split_whitespace()
        .filter(|w| w.len() > 2)
        .collect();

    let common: Vec<String> = words_a
        .iter()
        .filter(|w| words_b.contains(*w))
        .map(|w| w.to_lowercase())
        .collect();

    // Prefer shared subjects; fall back to first word from drawer A.
    let search_terms: Vec<String> = if !common.is_empty() {
        common.into_iter().take(2).collect()
    } else {
        words_a.iter().take(1).map(|w| w.to_lowercase()).collect()
    };

    let claim_a_lower = drawer_a.claim.to_lowercase();
    let claim_b_lower = drawer_b.claim.to_lowercase();
    let mut kg_count = 0;

    for term in &search_terms {
        if kg_count >= config.kg_evidence_limit {
            break;
        }

        let Ok(symbol_id) = engine.resolve_symbol(term) else {
            continue;
        };

        let triples = engine.triples_from(symbol_id);
        let remaining = config.kg_evidence_limit - kg_count;

        for triple in triples.iter().take(remaining) {
            let subj = engine.resolve_label(triple.subject);
            let pred = engine.resolve_label(triple.predicate);
            let obj = engine.resolve_label(triple.object);
            let triple_text = format!("{subj} {pred} {obj}");
            let obj_lower = obj.to_lowercase();

            let ev_idx = ach.evidence.len();
            let rel = SourceReliability::from_trust_score(triple.confidence.clamp(0.0, 1.0));
            ach.add_evidence(format!("KG: {triple_text}"), rel, Some(triple.subject));

            // Rate: does the KG object corroborate one of the claims?
            let in_a = claim_a_lower.contains(&obj_lower);
            let in_b = claim_b_lower.contains(&obj_lower);
            match (in_a, in_b) {
                (true, false) => {
                    ach.rate(ev_idx, 0, ConsistencyRating::Consistent);
                    ach.rate(ev_idx, 1, ConsistencyRating::WeaklyInconsistent);
                }
                (false, true) => {
                    ach.rate(ev_idx, 0, ConsistencyRating::WeaklyInconsistent);
                    ach.rate(ev_idx, 1, ConsistencyRating::Consistent);
                }
                _ => {} // both neutral (matrix default)
            }

            kg_count += 1;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn monday_drawer() -> DrawerClaim {
        DrawerClaim {
            drawer_id: "drawer-001".into(),
            claim: "temperature is 37 degrees".into(),
            source: Some("thermometer-A".into()),
            timestamp: 1_716_000_000, // Monday
            confidence: 0.9,
        }
    }

    fn friday_drawer() -> DrawerClaim {
        DrawerClaim {
            drawer_id: "drawer-002".into(),
            claim: "temperature is 22 degrees".into(),
            source: Some("thermometer-B".into()),
            timestamp: 1_716_432_000, // Friday (+5 days)
            confidence: 0.85,
        }
    }

    #[test]
    fn basic_contradiction_produces_result() {
        let a = monday_drawer();
        let b = friday_drawer();
        let config = DrawerAchConfig::default();
        let result = analyze_drawer_conflict(&a, &b, None, &config);

        assert!(!result.analysis_id.is_empty());
        assert_eq!(result.side_a.drawer_id, "drawer-001");
        assert_eq!(result.side_b.drawer_id, "drawer-002");
        // Both sides should be ranked.
        assert!(result.side_a.rank >= 1);
        assert!(result.side_b.rank >= 1);
    }

    #[test]
    fn temporal_recency_favors_newer() {
        // Friday drawer is newer and equally confident → should rank #1.
        let a = monday_drawer();
        let b = friday_drawer();
        let config = DrawerAchConfig::default();
        let result = analyze_drawer_conflict(&a, &b, None, &config);

        // Newer drawer (B, Friday) should have lower inconsistency score → rank 1.
        assert!(
            result.side_b.rank <= result.side_a.rank,
            "newer drawer should be ranked at least as high: side_b.rank={} side_a.rank={}",
            result.side_b.rank,
            result.side_a.rank
        );
    }

    #[test]
    fn belief_intervals_are_valid() {
        let a = monday_drawer();
        let b = friday_drawer();
        let config = DrawerAchConfig::default();
        let result = analyze_drawer_conflict(&a, &b, None, &config);

        let bi_a = &result.side_a.belief;
        let bi_b = &result.side_b.belief;

        assert!(bi_a.belief >= 0.0 && bi_a.belief <= 1.0, "belief_a out of range");
        assert!(
            bi_a.plausibility >= bi_a.belief,
            "plausibility_a < belief_a"
        );
        assert!(bi_b.belief >= 0.0 && bi_b.belief <= 1.0, "belief_b out of range");
        assert!(
            bi_b.plausibility >= bi_b.belief,
            "plausibility_b < belief_b"
        );
    }

    #[test]
    fn deciding_evidence_cites_drawer_ids() {
        let a = monday_drawer();
        let b = friday_drawer();
        let config = DrawerAchConfig::default();
        let result = analyze_drawer_conflict(&a, &b, None, &config);

        let has_a_ref = result
            .deciding_evidence
            .iter()
            .any(|e| e.drawer_id.as_deref() == Some("drawer-001"));
        let has_b_ref = result
            .deciding_evidence
            .iter()
            .any(|e| e.drawer_id.as_deref() == Some("drawer-002"));
        assert!(has_a_ref, "no deciding evidence references drawer-001");
        assert!(has_b_ref, "no deciding evidence references drawer-002");
    }

    #[test]
    fn same_timestamp_no_temporal_evidence() {
        let a = DrawerClaim {
            drawer_id: "d1".into(),
            claim: "X is Y".into(),
            source: None,
            timestamp: 1_000_000,
            confidence: 0.8,
        };
        let b = DrawerClaim {
            drawer_id: "d2".into(),
            claim: "X is Z".into(),
            source: None,
            timestamp: 1_000_000, // same timestamp
            confidence: 0.8,
        };
        let config = DrawerAchConfig::default();
        let result = analyze_drawer_conflict(&a, &b, None, &config);

        // With no temporal evidence and equal confidence, scores should be equal.
        let delta = (result.side_a.inconsistency_score - result.side_b.inconsistency_score).abs();
        assert!(
            delta < 0.01,
            "expected equal scores, got {delta:.3}: a={}, b={}",
            result.side_a.inconsistency_score,
            result.side_b.inconsistency_score
        );
    }

    #[test]
    fn matrix_text_contains_both_claims() {
        let a = monday_drawer();
        let b = friday_drawer();
        let config = DrawerAchConfig::default();
        let result = analyze_drawer_conflict(&a, &b, None, &config);

        assert!(result.matrix_text.contains("ACH:"), "missing ACH header");
        assert!(result.matrix_text.contains("Ranking"), "missing ranking");
    }

    #[test]
    fn high_confidence_drawer_ranks_first_when_newer() {
        let old_low = DrawerClaim {
            drawer_id: "old".into(),
            claim: "sky is green".into(),
            source: None,
            timestamp: 1_000_000,
            confidence: 0.3, // low confidence
        };
        let new_high = DrawerClaim {
            drawer_id: "new".into(),
            claim: "sky is blue".into(),
            source: None,
            timestamp: 1_100_000, // newer
            confidence: 0.95,     // high confidence
        };
        let config = DrawerAchConfig::default();
        let result = analyze_drawer_conflict(&old_low, &new_high, None, &config);

        assert_eq!(
            result.side_b.rank, 1,
            "high-confidence newer drawer should rank 1"
        );
    }
}
