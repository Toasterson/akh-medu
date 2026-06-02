//! Analysis of Competing Hypotheses (ACH) — Phase 18b.
//!
//! Heuer's structured analytic technique for evaluating multiple hypotheses
//! against available evidence. Focuses on **disconfirmation**: the best
//! hypothesis is the one least inconsistent with the evidence, not the one
//! with the most supporting evidence.
//!
//! Key concepts:
//! - **Consistency matrix**: evidence × hypothesis → consistency rating
//! - **Diagnosticity**: how much an evidence item differentiates hypotheses
//! - **Ranking**: hypotheses ordered by ascending inconsistency score

use serde::{Deserialize, Serialize};

use crate::symbol::SymbolId;

use super::source_reliability::SourceReliability;

// ═══════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════

/// Consistency rating between an evidence item and a hypothesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsistencyRating {
    /// Evidence strongly supports the hypothesis.
    Consistent,
    /// Evidence neither supports nor contradicts.
    Neutral,
    /// Evidence is unexpected under this hypothesis but not impossible.
    WeaklyInconsistent,
    /// Evidence contradicts this hypothesis.
    StronglyInconsistent,
    /// Evidence is irrelevant to this hypothesis.
    NotApplicable,
}

impl ConsistencyRating {
    /// Numeric weight for inconsistency scoring.
    /// Higher = more inconsistent.
    pub fn inconsistency_weight(&self) -> f32 {
        match self {
            Self::Consistent => 0.0,
            Self::Neutral => 0.0,
            Self::WeaklyInconsistent => 1.0,
            Self::StronglyInconsistent => 2.0,
            Self::NotApplicable => 0.0,
        }
    }

    pub fn as_label(&self) -> &'static str {
        match self {
            Self::Consistent => "++",
            Self::Neutral => "--",
            Self::WeaklyInconsistent => "-",
            Self::StronglyInconsistent => "XX",
            Self::NotApplicable => "NA",
        }
    }
}

/// A hypothesis in an ACH analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AchHypothesis {
    /// Human-readable description.
    pub description: String,
    /// Optional KG entity link.
    pub entity: Option<SymbolId>,
    /// Computed inconsistency score (lower = more likely).
    pub inconsistency_score: f32,
}

/// An evidence item in an ACH analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AchEvidence {
    /// Human-readable description.
    pub description: String,
    /// Source reliability rating.
    pub source_reliability: SourceReliability,
    /// Diagnosticity: how much this differentiates hypotheses (computed).
    pub diagnosticity: f32,
    /// Optional source entity link.
    pub source: Option<SymbolId>,
}

/// A complete ACH analysis session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AchAnalysis {
    /// Analysis identifier.
    pub id: String,
    /// The question being analyzed.
    pub question: String,
    /// Competing hypotheses.
    pub hypotheses: Vec<AchHypothesis>,
    /// Evidence items.
    pub evidence: Vec<AchEvidence>,
    /// Consistency matrix: evidence[i] × hypothesis[j].
    pub matrix: Vec<Vec<ConsistencyRating>>,
    /// Ranked hypotheses (index, score) — ascending by inconsistency.
    pub ranking: Vec<(usize, f32)>,
    /// Indices of the most diagnostic evidence items.
    pub diagnostic_evidence: Vec<usize>,
    /// Timestamp of creation.
    pub created_at: u64,
}

impl AchAnalysis {
    /// Create a new analysis session.
    pub fn new(id: impl Into<String>, question: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            question: question.into(),
            hypotheses: Vec::new(),
            evidence: Vec::new(),
            matrix: Vec::new(),
            ranking: Vec::new(),
            diagnostic_evidence: Vec::new(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Add a hypothesis.
    pub fn add_hypothesis(&mut self, description: impl Into<String>, entity: Option<SymbolId>) {
        self.hypotheses.push(AchHypothesis {
            description: description.into(),
            entity,
            inconsistency_score: 0.0,
        });
        // Extend existing matrix rows with Neutral for the new hypothesis.
        for row in &mut self.matrix {
            row.push(ConsistencyRating::Neutral);
        }
    }

    /// Add an evidence item.
    pub fn add_evidence(
        &mut self,
        description: impl Into<String>,
        source_reliability: SourceReliability,
        source: Option<SymbolId>,
    ) {
        self.evidence.push(AchEvidence {
            description: description.into(),
            source_reliability,
            diagnosticity: 0.0,
            source,
        });
        // Add a new matrix row with Neutral for all hypotheses.
        self.matrix
            .push(vec![ConsistencyRating::Neutral; self.hypotheses.len()]);
    }

    /// Set a consistency rating in the matrix.
    pub fn rate(
        &mut self,
        evidence_idx: usize,
        hypothesis_idx: usize,
        rating: ConsistencyRating,
    ) {
        if let Some(row) = self.matrix.get_mut(evidence_idx) {
            if let Some(cell) = row.get_mut(hypothesis_idx) {
                *cell = rating;
            }
        }
    }

    /// Evaluate the analysis: compute inconsistency scores, rankings, diagnosticity.
    pub fn evaluate(&mut self) {
        let h_count = self.hypotheses.len();
        let e_count = self.evidence.len();

        if h_count == 0 || e_count == 0 {
            return;
        }

        // Compute per-hypothesis inconsistency scores.
        for (j, hyp) in self.hypotheses.iter_mut().enumerate() {
            let mut score = 0.0_f32;
            for (i, row) in self.matrix.iter().enumerate() {
                if let Some(&rating) = row.get(j) {
                    // Weight by source reliability.
                    let reliability_weight = self.evidence[i].source_reliability.to_factor();
                    score += rating.inconsistency_weight() * reliability_weight;
                }
            }
            hyp.inconsistency_score = score;
        }

        // Rank hypotheses by ascending inconsistency (least inconsistent first).
        self.ranking = self
            .hypotheses
            .iter()
            .enumerate()
            .map(|(j, h)| (j, h.inconsistency_score))
            .collect();
        self.ranking
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Compute diagnosticity per evidence item.
        // High diagnosticity = high variance in consistency ratings across hypotheses.
        for (i, ev) in self.evidence.iter_mut().enumerate() {
            if let Some(row) = self.matrix.get(i) {
                let weights: Vec<f32> = row.iter().map(|r| r.inconsistency_weight()).collect();
                let mean = weights.iter().sum::<f32>() / weights.len() as f32;
                let variance = weights.iter().map(|w| (w - mean).powi(2)).sum::<f32>()
                    / weights.len() as f32;
                ev.diagnosticity = variance.sqrt(); // std dev as diagnosticity measure
            }
        }

        // Top diagnostic evidence indices.
        let mut diag_indices: Vec<(usize, f32)> = self
            .evidence
            .iter()
            .enumerate()
            .map(|(i, e)| (i, e.diagnosticity))
            .collect();
        diag_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        self.diagnostic_evidence = diag_indices.iter().map(|(i, _)| *i).collect();
    }

    /// Render the consistency matrix as a text table.
    pub fn render_matrix(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("ACH: {}", self.question));
        lines.push(String::new());

        // Header row.
        let mut header = String::from("Evidence \\ Hypothesis");
        for (j, h) in self.hypotheses.iter().enumerate() {
            let short = if h.description.len() > 15 {
                format!("H{}", j + 1)
            } else {
                h.description.clone()
            };
            header.push_str(&format!(" | {short:>6}"));
        }
        header.push_str(" | Diag");
        lines.push(header);
        lines.push("-".repeat(60));

        // Evidence rows.
        for (i, ev) in self.evidence.iter().enumerate() {
            let label = if ev.description.len() > 20 {
                format!("E{}", i + 1)
            } else {
                ev.description.clone()
            };
            let mut row = format!("{label:<20}");
            if let Some(matrix_row) = self.matrix.get(i) {
                for rating in matrix_row {
                    row.push_str(&format!(" | {:>6}", rating.as_label()));
                }
            }
            row.push_str(&format!(" | {:.2}", ev.diagnosticity));
            lines.push(row);
        }

        lines.push(String::new());

        // Scores row.
        let mut scores = String::from("Inconsistency score ");
        for h in &self.hypotheses {
            scores.push_str(&format!(" | {:>6.2}", h.inconsistency_score));
        }
        lines.push(scores);

        // Ranking.
        if !self.ranking.is_empty() {
            lines.push(String::new());
            lines.push("Ranking (least inconsistent first):".into());
            for (rank, &(j, score)) in self.ranking.iter().enumerate() {
                lines.push(format!(
                    "  {}. {} (score: {:.2})",
                    rank + 1,
                    self.hypotheses[j].description,
                    score
                ));
            }
        }

        lines.join("\n")
    }

    /// Identify information gaps: hypotheses that are hard to distinguish.
    pub fn information_gaps(&self) -> Vec<String> {
        let mut gaps = Vec::new();

        // Find hypothesis pairs with similar scores.
        for i in 0..self.ranking.len() {
            for j in (i + 1)..self.ranking.len() {
                let (hi, si) = self.ranking[i];
                let (hj, sj) = self.ranking[j];
                if (si - sj).abs() < 0.5 {
                    gaps.push(format!(
                        "Cannot distinguish \"{}\" from \"{}\" (scores {:.2} vs {:.2}) — seek differentiating evidence",
                        self.hypotheses[hi].description,
                        self.hypotheses[hj].description,
                        si, sj
                    ));
                }
            }
        }

        gaps
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_analysis() -> AchAnalysis {
        let mut a = AchAnalysis::new("test-1", "Why did the server crash?");
        a.add_hypothesis("Hardware failure", None);
        a.add_hypothesis("Software bug", None);
        a.add_hypothesis("Operator error", None);

        a.add_evidence("Error logs show OOM", SourceReliability::UsuallyReliable, None);
        a.add_evidence("Hardware diagnostics clean", SourceReliability::CompletelyReliable, None);
        a.add_evidence("Recent code deploy", SourceReliability::FairlyReliable, None);
        a
    }

    #[test]
    fn ach_analysis_creation() {
        let a = make_analysis();
        assert_eq!(a.hypotheses.len(), 3);
        assert_eq!(a.evidence.len(), 3);
        assert_eq!(a.matrix.len(), 3); // 3 evidence rows
        assert_eq!(a.matrix[0].len(), 3); // 3 hypothesis columns
    }

    #[test]
    fn consistency_rating_weights() {
        assert_eq!(ConsistencyRating::Consistent.inconsistency_weight(), 0.0);
        assert_eq!(ConsistencyRating::StronglyInconsistent.inconsistency_weight(), 2.0);
        assert_eq!(ConsistencyRating::NotApplicable.inconsistency_weight(), 0.0);
    }

    #[test]
    fn evaluate_ranks_by_inconsistency() {
        let mut a = make_analysis();

        // E0 (OOM) → consistent with software bug, inconsistent with hardware
        a.rate(0, 0, ConsistencyRating::WeaklyInconsistent); // hardware
        a.rate(0, 1, ConsistencyRating::Consistent);          // software
        a.rate(0, 2, ConsistencyRating::Neutral);              // operator

        // E1 (hardware clean) → strongly inconsistent with hardware
        a.rate(1, 0, ConsistencyRating::StronglyInconsistent); // hardware
        a.rate(1, 1, ConsistencyRating::Neutral);              // software
        a.rate(1, 2, ConsistencyRating::Neutral);              // operator

        // E2 (recent deploy) → consistent with software bug
        a.rate(2, 0, ConsistencyRating::Neutral);              // hardware
        a.rate(2, 1, ConsistencyRating::Consistent);           // software
        a.rate(2, 2, ConsistencyRating::WeaklyInconsistent);   // operator

        a.evaluate();

        // Hardware should rank worst (most inconsistent).
        assert!(a.hypotheses[0].inconsistency_score > a.hypotheses[1].inconsistency_score);
        // Software bug should rank best (least inconsistent).
        assert_eq!(a.ranking[0].0, 1, "software bug should be top-ranked");
    }

    #[test]
    fn diagnosticity_high_for_differentiating() {
        let mut a = make_analysis();
        // E0: consistent with H0, strongly inconsistent with H1, neutral with H2
        a.rate(0, 0, ConsistencyRating::Consistent);
        a.rate(0, 1, ConsistencyRating::StronglyInconsistent);
        a.rate(0, 2, ConsistencyRating::Neutral);

        // E1: neutral across all → low diagnosticity
        a.rate(1, 0, ConsistencyRating::Neutral);
        a.rate(1, 1, ConsistencyRating::Neutral);
        a.rate(1, 2, ConsistencyRating::Neutral);

        a.evaluate();

        assert!(
            a.evidence[0].diagnosticity > a.evidence[1].diagnosticity,
            "E0 should be more diagnostic: {} vs {}",
            a.evidence[0].diagnosticity,
            a.evidence[1].diagnosticity
        );
    }

    #[test]
    fn render_matrix_readable() {
        let mut a = make_analysis();
        a.rate(0, 1, ConsistencyRating::Consistent);
        a.rate(1, 0, ConsistencyRating::StronglyInconsistent);
        a.evaluate();

        let rendered = a.render_matrix();
        assert!(rendered.contains("ACH:"));
        assert!(rendered.contains("Ranking"));
    }

    #[test]
    fn information_gaps_similar_scores() {
        let mut a = AchAnalysis::new("test", "Q");
        a.add_hypothesis("H1", None);
        a.add_hypothesis("H2", None);
        a.add_evidence("E1", SourceReliability::FairlyReliable, None);
        // Both neutral → same score → gap
        a.evaluate();

        let gaps = a.information_gaps();
        assert!(!gaps.is_empty(), "should detect indistinguishable hypotheses");
    }

    #[test]
    fn serialization_roundtrip() {
        let a = make_analysis();
        let bytes = bincode::serialize(&a).unwrap();
        let restored: AchAnalysis = bincode::deserialize(&bytes).unwrap();
        assert_eq!(restored.hypotheses.len(), 3);
        assert_eq!(restored.evidence.len(), 3);
    }

    #[test]
    fn empty_analysis_evaluate_no_panic() {
        let mut a = AchAnalysis::new("empty", "Q");
        a.evaluate(); // should not panic
        assert!(a.ranking.is_empty());
    }
}
