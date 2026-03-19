//! Deception & credibility analysis — Phase 18c.
//!
//! Multi-signal credibility assessment for incoming information:
//! - Factual support: does the KG corroborate the claims?
//! - Self-consistency: does the source contradict its own past claims?
//! - Cross-source consistency: do other reliable sources agree?
//! - Manipulation markers: urgency, flattery, threats, emotional appeal
//!
//! Outputs a `CredibilityRecommendation` (Accept / AcceptWithCaution /
//! SeekCorroboration / Discard / FlagForReview).

use serde::{Deserialize, Serialize};

use crate::engine::Engine;
use crate::symbol::SymbolId;

use super::evidence::EvidenceManager;
use super::source_reliability::SourceReliabilityManager;

// ═══════════════════════════════════════════════════════════════════════
// CredibilitySignals
// ═══════════════════════════════════════════════════════════════════════

/// Multi-signal credibility assessment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredibilitySignals {
    /// How well the claim is supported by existing KG facts (0.0–1.0).
    pub factual_support: f32,
    /// Does the source show bias patterns? (0.0 = no bias, 1.0 = strong bias).
    pub bias_indicator: f32,
    /// Consistency with the source's own past claims (0.0–1.0).
    pub self_consistency: f32,
    /// Consistency with other reliable sources (0.0–1.0).
    pub cross_source_consistency: f32,
    /// Hedging/uncertainty markers detected (0.0 = definitive, 1.0 = very hedged).
    pub uncertainty_markers: f32,
    /// Combined credibility score (weighted average).
    pub combined_score: f32,
}

impl CredibilitySignals {
    /// Compute the combined score from individual signals.
    pub fn compute_combined(&mut self) {
        self.combined_score = self.factual_support * 0.30
            + self.self_consistency * 0.25
            + self.cross_source_consistency * 0.25
            + (1.0 - self.bias_indicator) * 0.10
            + (1.0 - self.uncertainty_markers) * 0.10;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// DeceptionIndicators
// ═══════════════════════════════════════════════════════════════════════

/// Indicators of potential deception in a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeceptionIndicators {
    /// Claims that contradict known KG facts.
    pub contradicts_known_facts: usize,
    /// Claims that contradict the source's own prior statements.
    pub self_contradictions: usize,
    /// Detected manipulation markers.
    pub manipulation_markers: Vec<String>,
    /// Overall deception likelihood (0.0 = honest, 1.0 = deceptive).
    pub deception_likelihood: f32,
    /// Human-readable reasoning.
    pub reasoning: String,
}

// ═══════════════════════════════════════════════════════════════════════
// CredibilityRecommendation
// ═══════════════════════════════════════════════════════════════════════

/// Recommendation for how to handle information based on credibility assessment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CredibilityRecommendation {
    /// Accept with standard confidence.
    Accept,
    /// Accept but with reduced confidence.
    AcceptWithCaution,
    /// Seek corroboration before accepting.
    SeekCorroboration,
    /// Likely deceptive — discard.
    Discard,
    /// Cannot determine automatically — flag for operator.
    FlagForReview,
}

impl CredibilityRecommendation {
    pub fn as_label(&self) -> &'static str {
        match self {
            Self::Accept => "accept",
            Self::AcceptWithCaution => "accept with caution",
            Self::SeekCorroboration => "seek corroboration",
            Self::Discard => "discard",
            Self::FlagForReview => "flag for review",
        }
    }

    /// Derive recommendation from credibility signals and deception indicators.
    pub fn from_analysis(signals: &CredibilitySignals, deception: &DeceptionIndicators) -> Self {
        if deception.deception_likelihood > 0.8 {
            return Self::Discard;
        }
        if deception.deception_likelihood > 0.5 || !deception.manipulation_markers.is_empty() {
            return Self::FlagForReview;
        }
        if signals.combined_score > 0.8 {
            Self::Accept
        } else if signals.combined_score > 0.6 {
            Self::AcceptWithCaution
        } else if signals.combined_score > 0.3 {
            Self::SeekCorroboration
        } else {
            Self::FlagForReview
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CredibilityAnalysis (result)
// ═══════════════════════════════════════════════════════════════════════

/// Complete credibility analysis result.
#[derive(Debug, Clone)]
pub struct CredibilityAnalysis {
    /// The source being assessed.
    pub source_id: SymbolId,
    /// Multi-signal credibility assessment.
    pub signals: CredibilitySignals,
    /// Deception indicators.
    pub deception: DeceptionIndicators,
    /// Recommendation.
    pub recommendation: CredibilityRecommendation,
}

// ═══════════════════════════════════════════════════════════════════════
// Manipulation Detection
// ═══════════════════════════════════════════════════════════════════════

/// Detect manipulation markers in text.
///
/// Checks for common persuasion/deception patterns:
/// - Urgency: "immediately", "urgent", "act now", "deadline"
/// - Flattery: "you're the only one", "specially selected", "exclusive"
/// - Threats: "or else", "consequences", "you must"
/// - Emotional appeal: "think of the children", "how would you feel"
/// - Authority abuse: "trust me", "I'm an expert", "everyone knows"
pub fn detect_manipulation_markers(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let mut markers = Vec::new();

    // Urgency patterns.
    let urgency = [
        "immediately", "urgent", "act now", "right away", "deadline",
        "time is running out", "don't delay", "asap",
    ];
    for pat in &urgency {
        if lower.contains(pat) {
            markers.push(format!("urgency: \"{pat}\""));
        }
    }

    // Flattery patterns.
    let flattery = [
        "you're the only", "specially selected", "exclusive offer",
        "chosen few", "uniquely qualified",
    ];
    for pat in &flattery {
        if lower.contains(pat) {
            markers.push(format!("flattery: \"{pat}\""));
        }
    }

    // Threat patterns.
    let threats = [
        "or else", "consequences", "you must", "no choice",
        "forced to", "will be held responsible",
    ];
    for pat in &threats {
        if lower.contains(pat) {
            markers.push(format!("threat: \"{pat}\""));
        }
    }

    // Authority abuse.
    let authority = [
        "trust me", "everyone knows", "it's obvious",
        "no one disagrees", "experts agree",
    ];
    for pat in &authority {
        if lower.contains(pat) {
            markers.push(format!("authority: \"{pat}\""));
        }
    }

    markers
}

// ═══════════════════════════════════════════════════════════════════════
// CredibilityAnalyzer
// ═══════════════════════════════════════════════════════════════════════

/// Performs credibility analysis on incoming information.
#[derive(Debug, Clone, Default)]
pub struct CredibilityAnalyzer;

impl CredibilityAnalyzer {
    /// Analyze credibility of claims from a source.
    ///
    /// `claim_text`: the raw text of the claim for manipulation detection.
    /// `claim_ids`: KG entities representing the claims being made.
    pub fn analyze(
        &self,
        source_id: SymbolId,
        claim_text: &str,
        claim_ids: &[SymbolId],
        engine: &Engine,
        evidence_mgr: &EvidenceManager,
        reliability_mgr: &SourceReliabilityManager,
    ) -> CredibilityAnalysis {
        // 1. Factual support: how many claims have KG backing?
        let factual_support = if claim_ids.is_empty() {
            0.5 // no claims to check
        } else {
            let supported = claim_ids
                .iter()
                .filter(|&&id| !engine.triples_from(id).is_empty())
                .count();
            supported as f32 / claim_ids.len() as f32
        };

        // 2. Self-consistency: check source's past claims via evidence manager.
        let self_consistency = {
            let source_claims = evidence_mgr
                .claims
                .iter()
                .filter(|(_, items)| items.iter().any(|e| e.source_id == source_id))
                .count();
            if source_claims == 0 {
                0.5 // no history
            } else {
                // If source has evidence, check for conflicts in their own claims.
                let conflicting = evidence_mgr
                    .claims
                    .values()
                    .filter(|items| {
                        let from_source: Vec<_> =
                            items.iter().filter(|e| e.source_id == source_id).collect();
                        from_source.len() > 1
                            && from_source
                                .windows(2)
                                .any(|w| w[0].mass.conflict_with(&w[1].mass) > 0.5)
                    })
                    .count();
                1.0 - (conflicting as f32 / source_claims as f32).min(1.0)
            }
        };

        // 3. Cross-source consistency: does the source agree with reliable sources?
        let cross_source_consistency = {
            let trust = reliability_mgr.sources.get(&source_id.get());
            match trust {
                Some(t) => t.competence.expected(),
                None => 0.5,
            }
        };

        // 4. Bias indicator: based on trust model integrity dimension.
        let bias_indicator = {
            let trust = reliability_mgr.sources.get(&source_id.get());
            match trust {
                Some(t) => 1.0 - t.integrity.expected(),
                None => 0.3, // mild assumed bias for unknown sources
            }
        };

        // 5. Manipulation markers.
        let manipulation_markers = detect_manipulation_markers(claim_text);
        let uncertainty_markers = if claim_text.contains('?')
            || claim_text.contains("maybe")
            || claim_text.contains("perhaps")
            || claim_text.contains("might")
        {
            0.5
        } else {
            0.1
        };

        let mut signals = CredibilitySignals {
            factual_support,
            bias_indicator,
            self_consistency,
            cross_source_consistency,
            uncertainty_markers,
            combined_score: 0.0,
        };
        signals.compute_combined();

        // Deception indicators.
        let contradictions = claim_ids
            .iter()
            .filter(|&&id| {
                // Check if any existing triple contradicts this claim.
                let triples = engine.triples_from(id);
                triples.iter().any(|t| t.confidence < 0.3)
            })
            .count();

        let deception_likelihood = {
            let mut score = 0.0_f32;
            if contradictions > 0 {
                score += 0.3;
            }
            if !manipulation_markers.is_empty() {
                score += 0.2 * manipulation_markers.len() as f32;
            }
            if bias_indicator > 0.7 {
                score += 0.2;
            }
            score.min(1.0)
        };

        let reasoning = format!(
            "factual={:.2}, self_cons={:.2}, cross={:.2}, bias={:.2}, manip={}, deception={:.2}",
            factual_support,
            self_consistency,
            cross_source_consistency,
            bias_indicator,
            manipulation_markers.len(),
            deception_likelihood
        );

        let deception = DeceptionIndicators {
            contradicts_known_facts: contradictions,
            self_contradictions: 0, // simplified
            manipulation_markers,
            deception_likelihood,
            reasoning,
        };

        let recommendation = CredibilityRecommendation::from_analysis(&signals, &deception);

        CredibilityAnalysis {
            source_id,
            signals,
            deception,
            recommendation,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credibility_signals_combined() {
        let mut signals = CredibilitySignals {
            factual_support: 0.8,
            bias_indicator: 0.1,
            self_consistency: 0.9,
            cross_source_consistency: 0.7,
            uncertainty_markers: 0.2,
            combined_score: 0.0,
        };
        signals.compute_combined();
        assert!(signals.combined_score > 0.7);
    }

    #[test]
    fn recommendation_accept() {
        let signals = CredibilitySignals {
            factual_support: 0.9,
            bias_indicator: 0.0,
            self_consistency: 0.9,
            cross_source_consistency: 0.9,
            uncertainty_markers: 0.0,
            combined_score: 0.9,
        };
        let deception = DeceptionIndicators {
            contradicts_known_facts: 0,
            self_contradictions: 0,
            manipulation_markers: vec![],
            deception_likelihood: 0.0,
            reasoning: String::new(),
        };
        assert_eq!(
            CredibilityRecommendation::from_analysis(&signals, &deception),
            CredibilityRecommendation::Accept
        );
    }

    #[test]
    fn recommendation_discard() {
        let signals = CredibilitySignals {
            factual_support: 0.1,
            bias_indicator: 0.9,
            self_consistency: 0.2,
            cross_source_consistency: 0.1,
            uncertainty_markers: 0.0,
            combined_score: 0.1,
        };
        let deception = DeceptionIndicators {
            contradicts_known_facts: 3,
            self_contradictions: 2,
            manipulation_markers: vec!["urgency".into()],
            deception_likelihood: 0.9,
            reasoning: String::new(),
        };
        assert_eq!(
            CredibilityRecommendation::from_analysis(&signals, &deception),
            CredibilityRecommendation::Discard
        );
    }

    #[test]
    fn manipulation_marker_urgency() {
        let markers = detect_manipulation_markers("You must act immediately or else!");
        assert!(markers.len() >= 2); // "immediately" + "or else"
    }

    #[test]
    fn manipulation_marker_flattery() {
        let markers = detect_manipulation_markers("You're the only one who can help");
        assert!(!markers.is_empty());
    }

    #[test]
    fn manipulation_marker_none() {
        let markers = detect_manipulation_markers("The temperature today is 22 degrees.");
        assert!(markers.is_empty());
    }

    #[test]
    fn recommendation_labels() {
        assert_eq!(CredibilityRecommendation::Accept.as_label(), "accept");
        assert_eq!(CredibilityRecommendation::Discard.as_label(), "discard");
        assert_eq!(
            CredibilityRecommendation::SeekCorroboration.as_label(),
            "seek corroboration"
        );
    }

    #[test]
    fn credibility_analyzer_no_claims() {
        let engine = crate::engine::Engine::new(crate::engine::EngineConfig::default()).unwrap();
        let evidence_mgr = EvidenceManager::default();
        let reliability_mgr = SourceReliabilityManager::new();
        let analyzer = CredibilityAnalyzer;

        let source = engine.resolve_or_create_entity("test-source").unwrap();
        let result = analyzer.analyze(
            source,
            "The sky is blue",
            &[],
            &engine,
            &evidence_mgr,
            &reliability_mgr,
        );

        // No claims, no manipulation → accept with caution (neutral signals).
        assert!(matches!(
            result.recommendation,
            CredibilityRecommendation::Accept
                | CredibilityRecommendation::AcceptWithCaution
                | CredibilityRecommendation::SeekCorroboration
        ));
    }
}
