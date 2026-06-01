//! Skill discovery from reasoning: detect latent skills in background learning.
//!
//! Phase 31d — during the background consolidation cycle (sleep/dream phase),
//! identify clusters of frequently co-activated symbols that form a coherent
//! domain. When a cluster meets the novelty and coherence thresholds, it is
//! surfaced as a `SkillCandidate` for user review and optional promotion to a
//! full skillpack.

use serde::{Deserialize, Serialize};

use crate::graph::Triple;
use crate::symbol::SymbolId;

/// A cluster of co-activated symbols discovered during background reasoning.
///
/// This is the raw signal from spreading-activation tracking; further analysis
/// may refine it into a `SkillCandidate`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoActivationCluster {
    /// Symbol IDs that were frequently activated together.
    pub symbols: Vec<SymbolId>,
    /// Estimated co-activation frequency (higher = stronger cluster).
    pub frequency: f32,
    /// Triples induced by these symbols in the knowledge graph.
    pub induced_triples: Vec<Triple>,
}

/// Configuration for the skill candidate detection pass.
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    /// Minimum number of triples in the induced subgraph for a cluster to be
    /// considered a skill candidate.
    pub min_triples: usize,
    /// Minimum number of distinct predicates required (tests conceptual diversity).
    pub min_distinct_predicates: usize,
    /// Minimum intra-cluster prototype similarity for coherence check (0.0–1.0).
    pub min_coherence: f32,
    /// Minimum co-activation frequency threshold to consider a cluster.
    pub min_frequency: f32,
    /// Prototype similarity threshold against existing skills: if any existing
    /// skill prototype is more similar than this, the cluster is not novel.
    pub novelty_threshold: f32,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            min_triples: 5,
            min_distinct_predicates: 2,
            min_coherence: 0.55,
            min_frequency: 3.0,
            novelty_threshold: 0.70,
        }
    }
}

/// A candidate skill surfaced from background reasoning.
///
/// Candidates are stored in the KG and presented to the user for review.
/// A user-approved candidate can be packaged into a full skillpack via the
/// Phase 30c authoring pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillCandidate {
    /// LLM-suggested name for the potential skill (pending user review).
    pub suggested_name: String,
    /// LLM-suggested description.
    pub suggested_description: String,
    /// Domain tags inferred from the cluster.
    pub domain_tags: Vec<String>,
    /// Core triples that define the candidate skill's content.
    pub core_triples: Vec<Triple>,
    /// Egg rewrite rules that fired within this cluster's subgraph.
    pub related_rules: Vec<String>,
    /// Confidence that this is a genuinely useful, novel skill (0.0–1.0).
    pub confidence: f32,
    /// Opaque provenance token (e.g., a SymbolId or UUID string) linking back
    /// to the consolidation event that surfaced this candidate.
    pub provenance_token: String,
}

/// Filter: check whether a co-activation cluster meets the minimum quality
/// thresholds from `config` to be considered for skill candidate promotion.
pub fn meets_quality_threshold(cluster: &CoActivationCluster, config: &DiscoveryConfig) -> bool {
    if cluster.frequency < config.min_frequency {
        return false;
    }
    if cluster.induced_triples.len() < config.min_triples {
        return false;
    }
    let distinct_predicates: std::collections::HashSet<SymbolId> = cluster
        .induced_triples
        .iter()
        .map(|t| t.predicate)
        .collect();
    if distinct_predicates.len() < config.min_distinct_predicates {
        return false;
    }
    true
}

/// Build a `SkillCandidate` from a cluster, using heuristic name and domain
/// inference.
///
/// In production, the `suggested_name` and `suggested_description` fields would
/// be populated by an LLM call (Phase 30c SkillAuthor). Here, we fill in
/// placeholders so the pipeline can function without an LLM dependency.
pub fn build_candidate_from_cluster(
    cluster: &CoActivationCluster,
    provenance_token: String,
) -> SkillCandidate {
    // Heuristic: use the first predicate string as a domain hint.
    let domain_hint = format!(
        "domain-{}",
        cluster.symbols.first().map(|s| s.get()).unwrap_or(0)
    );

    SkillCandidate {
        suggested_name: format!(
            "auto-skill-{}",
            provenance_token.chars().take(8).collect::<String>()
        ),
        suggested_description: format!(
            "Auto-discovered skill with {} triples and {} symbols.",
            cluster.induced_triples.len(),
            cluster.symbols.len()
        ),
        domain_tags: vec![domain_hint],
        core_triples: cluster.induced_triples.clone(),
        related_rules: Vec::new(),
        confidence: (cluster.frequency / 10.0).min(1.0),
        provenance_token,
    }
}

/// Run one discovery pass over a set of co-activation clusters.
///
/// Filters by quality thresholds, then builds `SkillCandidate` entries for
/// each cluster that passes. The caller is responsible for novelty-checking
/// against existing skill prototypes (Phase 31b) and presenting survivors to
/// the user.
pub fn discover_from_clusters(
    clusters: &[CoActivationCluster],
    config: &DiscoveryConfig,
) -> Vec<SkillCandidate> {
    clusters
        .iter()
        .enumerate()
        .filter(|(_, cluster)| meets_quality_threshold(cluster, config))
        .map(|(i, cluster)| build_candidate_from_cluster(cluster, format!("cluster-{i}")))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Triple;
    use crate::symbol::SymbolId;

    fn sym(n: u64) -> SymbolId {
        SymbolId::new(n).unwrap()
    }

    fn make_triple(s: u64, p: u64, o: u64) -> Triple {
        Triple::new(sym(s), sym(p), sym(o))
    }

    fn make_cluster(n_triples: usize, n_predicates: usize, freq: f32) -> CoActivationCluster {
        let mut triples = Vec::new();
        for i in 0..n_triples {
            triples.push(make_triple(
                100 + i as u64,
                (i % n_predicates) as u64 + 1,
                200 + i as u64,
            ));
        }
        CoActivationCluster {
            symbols: vec![sym(1), sym(2), sym(3)],
            frequency: freq,
            induced_triples: triples,
        }
    }

    #[test]
    fn quality_threshold_passes() {
        let cluster = make_cluster(10, 3, 5.0);
        let config = DiscoveryConfig::default();
        assert!(meets_quality_threshold(&cluster, &config));
    }

    #[test]
    fn quality_threshold_fails_too_few_triples() {
        let cluster = make_cluster(3, 3, 5.0);
        let config = DiscoveryConfig::default();
        assert!(!meets_quality_threshold(&cluster, &config));
    }

    #[test]
    fn quality_threshold_fails_low_frequency() {
        let cluster = make_cluster(10, 3, 1.0);
        let config = DiscoveryConfig::default();
        assert!(!meets_quality_threshold(&cluster, &config));
    }

    #[test]
    fn discover_from_clusters_filters_correctly() {
        let good = make_cluster(10, 3, 5.0);
        let bad = make_cluster(2, 1, 1.0);
        let config = DiscoveryConfig::default();

        let candidates = discover_from_clusters(&[good, bad], &config);
        assert_eq!(candidates.len(), 1);
    }

    #[test]
    fn build_candidate_has_correct_triple_count() {
        let cluster = make_cluster(7, 2, 4.0);
        let candidate = build_candidate_from_cluster(&cluster, "test-token".to_string());
        assert_eq!(candidate.core_triples.len(), 7);
        assert_eq!(candidate.provenance_token, "test-token");
    }
}
