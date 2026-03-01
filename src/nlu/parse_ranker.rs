//! VSA Parse Ranker: exemplar-based parse ranking for NLU Tier 4.
//!
//! The ranker accumulates successful parse exemplars over time. When a new
//! input arrives that Tiers 1-3 cannot handle, the ranker checks if a
//! structurally similar input was previously parsed, and returns the
//! corresponding parse tree as a candidate.
//!
//! This creates a self-improving feedback loop: the more inputs the system
//! successfully parses, the better it handles novel but similar inputs.

use serde::{Deserialize, Serialize};

use crate::grammar::abs::AbsTree;

/// A stored exemplar of a successful parse.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseExemplar {
    /// The original input text.
    pub input: String,
    /// The normalized (lowercased, trimmed) input for matching.
    normalized: String,
    /// The resulting abstract syntax tree.
    pub tree: AbsTree,
    /// Which NLU tier produced this parse.
    pub source_tier: u8,
    /// Confidence of the original parse.
    pub confidence: f32,
}

/// A ranked parse result from the ranker.
#[derive(Debug, Clone)]
pub struct RankedParse {
    /// The parse tree from the best matching exemplar.
    pub tree: AbsTree,
    /// Similarity to the matched exemplar (0.0 - 1.0).
    pub similarity: f32,
    /// Confidence inherited from the exemplar, discounted by similarity.
    pub confidence: f32,
}

/// The VSA Parse Ranker.
///
/// Stores parse exemplars and retrieves the best match for new inputs
/// using normalized text similarity. A future version will use VSA
/// encoding for structural matching.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParseRanker {
    /// Accumulated exemplars from successful parses.
    exemplars: Vec<ParseExemplar>,
    /// Maximum number of exemplars to store.
    max_exemplars: usize,
}

impl ParseRanker {
    /// Create a new empty ranker.
    pub fn new() -> Self {
        Self {
            exemplars: Vec::new(),
            max_exemplars: 1000,
        }
    }

    /// Record a successful parse as an exemplar.
    pub fn record_success(&mut self, input: &str, tree: &AbsTree, source_tier: u8, confidence: f32) {
        let normalized = normalize_input(input);

        // Don't store duplicates
        if self.exemplars.iter().any(|e| e.normalized == normalized) {
            return;
        }

        let exemplar = ParseExemplar {
            input: input.to_string(),
            normalized,
            tree: tree.clone(),
            source_tier,
            confidence,
        };

        if self.exemplars.len() >= self.max_exemplars {
            // Remove the oldest exemplar with the lowest confidence
            if let Some(min_idx) = self
                .exemplars
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.confidence
                        .partial_cmp(&b.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                self.exemplars.swap_remove(min_idx);
            }
        }

        self.exemplars.push(exemplar);
    }

    /// Find the most similar exemplar to the given input.
    ///
    /// Returns `Some(RankedParse)` if a sufficiently similar exemplar exists
    /// (similarity > 0.6), `None` otherwise.
    pub fn find_similar(&self, input: &str) -> Option<RankedParse> {
        if self.exemplars.is_empty() {
            return None;
        }

        let normalized = normalize_input(input);
        let input_words: Vec<&str> = normalized.split_whitespace().collect();

        let mut best_sim = 0.0_f32;
        let mut best_exemplar: Option<&ParseExemplar> = None;

        for exemplar in &self.exemplars {
            let exemplar_words: Vec<&str> = exemplar.normalized.split_whitespace().collect();
            let sim = jaccard_similarity(&input_words, &exemplar_words);
            if sim > best_sim {
                best_sim = sim;
                best_exemplar = Some(exemplar);
            }
        }

        let exemplar = best_exemplar?;
        if best_sim < 0.6 {
            return None;
        }

        Some(RankedParse {
            tree: exemplar.tree.clone(),
            similarity: best_sim,
            confidence: exemplar.confidence * best_sim,
        })
    }

    /// Check if a similar exemplar exists (quick check without returning the tree).
    pub fn has_similar_exemplar(&self, input: &str) -> bool {
        self.find_similar(input).is_some()
    }

    /// Number of stored exemplars.
    pub fn exemplar_count(&self) -> usize {
        self.exemplars.len()
    }

    /// Serialize the ranker state to bytes for persistence.
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    /// Deserialize the ranker state from bytes.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        bincode::deserialize(data).ok()
    }
}

/// Normalize input for comparison: lowercase, trim, collapse whitespace.
fn normalize_input(input: &str) -> String {
    input
        .trim()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Jaccard similarity between two word sets.
fn jaccard_similarity(a: &[&str], b: &[&str]) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let set_a: std::collections::HashSet<&str> = a.iter().copied().collect();
    let set_b: std::collections::HashSet<&str> = b.iter().copied().collect();

    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_and_retrieve() {
        let mut ranker = ParseRanker::new();
        let tree = AbsTree::entity("test");
        ranker.record_success("dogs are mammals", &tree, 1, 0.9);

        assert_eq!(ranker.exemplar_count(), 1);

        // Exact match should find it
        let result = ranker.find_similar("dogs are mammals");
        assert!(result.is_some());
        let ranked = result.unwrap();
        assert!((ranked.similarity - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn similar_input_matches() {
        let mut ranker = ParseRanker::new();
        let tree = AbsTree::triple(
            AbsTree::entity("dogs"),
            AbsTree::relation("are"),
            AbsTree::entity("mammals"),
        );
        ranker.record_success("all dogs are furry mammals", &tree, 1, 0.9);

        // Similar input — shares 4 of 5 words (Jaccard = 4/6 ≈ 0.67)
        let result = ranker.find_similar("all cats are furry mammals");
        assert!(result.is_some());
        let ranked = result.unwrap();
        assert!(ranked.similarity > 0.6);
    }

    #[test]
    fn dissimilar_input_no_match() {
        let mut ranker = ParseRanker::new();
        let tree = AbsTree::entity("test");
        ranker.record_success("dogs are mammals", &tree, 1, 0.9);

        // Very different input
        let result = ranker.find_similar("the weather is nice today");
        assert!(result.is_none());
    }

    #[test]
    fn no_duplicates() {
        let mut ranker = ParseRanker::new();
        let tree = AbsTree::entity("test");
        ranker.record_success("dogs are mammals", &tree, 1, 0.9);
        ranker.record_success("dogs are mammals", &tree, 1, 0.9);

        assert_eq!(ranker.exemplar_count(), 1);
    }

    #[test]
    fn persistence_roundtrip() {
        let mut ranker = ParseRanker::new();
        let tree = AbsTree::entity("test");
        ranker.record_success("dogs are mammals", &tree, 1, 0.9);

        let bytes = ranker.to_bytes();
        let restored = ParseRanker::from_bytes(&bytes).unwrap();

        assert_eq!(restored.exemplar_count(), 1);
        assert!(restored.has_similar_exemplar("dogs are mammals"));
    }

    #[test]
    fn empty_ranker_returns_none() {
        let ranker = ParseRanker::new();
        assert!(ranker.find_similar("anything").is_none());
    }

    #[test]
    fn case_insensitive_matching() {
        let mut ranker = ParseRanker::new();
        let tree = AbsTree::entity("test");
        ranker.record_success("Dogs Are Mammals", &tree, 1, 0.9);

        let result = ranker.find_similar("dogs are mammals");
        assert!(result.is_some());
    }

    #[test]
    fn confidence_discounted_by_similarity() {
        let mut ranker = ParseRanker::new();
        let tree = AbsTree::entity("test");
        ranker.record_success("dogs are mammals", &tree, 1, 0.9);

        let result = ranker.find_similar("dogs are mammals").unwrap();
        // Exact match: confidence = 0.9 * 1.0 = 0.9
        assert!((result.confidence - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn max_exemplars_eviction() {
        let mut ranker = ParseRanker {
            exemplars: Vec::new(),
            max_exemplars: 3,
        };
        let tree = AbsTree::entity("test");

        ranker.record_success("input one", &tree, 1, 0.5);
        ranker.record_success("input two", &tree, 1, 0.9);
        ranker.record_success("input three", &tree, 1, 0.8);
        assert_eq!(ranker.exemplar_count(), 3);

        // Adding a 4th should evict the lowest-confidence one
        ranker.record_success("input four", &tree, 1, 0.7);
        assert_eq!(ranker.exemplar_count(), 3);

        // "input one" (confidence 0.5) should have been evicted
        assert!(!ranker.has_similar_exemplar("input one"));
    }
}
