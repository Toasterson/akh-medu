//! Contradiction sub-verifier: wraps Phase 9l contradiction detection
//! (`Engine::check_contradictions`) over the top hypothesis' materialisation.

use crate::engine::Engine;
use crate::graph::Triple;
use crate::graph::contradiction::ContradictionKind;

use super::super::ThinkingPredicates;
use super::super::error::ThinkingResult;
use super::super::state::{ThoughtState, TripleRef};
use super::{ThoughtVerifier, VerifierReason, VerifierVerdict};

/// Materialises the top hypothesis as a candidate triple
/// `(query subject, query relation, hypothesis)` and asks the engine's
/// contradiction registries (functional predicates, disjointness, temporal,
/// multi-valued) whether asserting it would contradict existing knowledge.
///
/// Neutral (score 1.0, no reasons) when the goal has no `(subject, relation)`
/// query structure or no hypothesis has resolved to a symbol yet — there is
/// nothing to materialise, hence nothing to contradict.
pub struct ContradictionVerifier {
    preds: ThinkingPredicates,
}

impl ContradictionVerifier {
    pub fn new(preds: ThinkingPredicates) -> Self {
        Self { preds }
    }

    fn kind_label(kind: &ContradictionKind) -> &'static str {
        match kind {
            ContradictionKind::FunctionalViolation { .. } => "functional-violation",
            ContradictionKind::DisjointnessViolation { .. } => "disjointness-violation",
            ContradictionKind::TemporalConflict { .. } => "temporal-conflict",
            ContradictionKind::IntraMicrotheoryConflict { .. } => "intra-microtheory-conflict",
        }
    }
}

impl ThoughtVerifier for ContradictionVerifier {
    fn score(&mut self, engine: &Engine, state: &ThoughtState) -> ThinkingResult<VerifierVerdict> {
        // Query structure: (S, R) from the goal's thinking predicates.
        let goal_triples = engine.triples_from(state.origin.goal);
        let subject = goal_triples
            .iter()
            .find(|t| t.predicate == self.preds.seeks)
            .map(|t| t.object)
            .unwrap_or(state.origin.goal);
        let relation = goal_triples
            .iter()
            .find(|t| t.predicate == self.preds.relation)
            .map(|t| t.object);

        let (Some(relation), Some(hypothesis)) = (
            relation,
            state.top_hypothesis().and_then(|h| h.symbol),
        ) else {
            return Ok(VerifierVerdict {
                score: 1.0,
                sufficient: false,
                reasons: Vec::new(),
            });
        };

        let candidate = Triple::new(subject, relation, hypothesis);
        let contradictions = engine.check_contradictions(&candidate);

        if contradictions.is_empty() {
            return Ok(VerifierVerdict {
                score: 1.0,
                sufficient: false,
                reasons: Vec::new(),
            });
        }

        let reasons: Vec<VerifierReason> = contradictions
            .iter()
            .map(|c| VerifierReason::ContradictionDetected {
                kind: Self::kind_label(&c.kind).to_string(),
                locus: TripleRef {
                    subject: c.existing.subject,
                    predicate: c.existing.predicate,
                    object: c.existing.object,
                },
            })
            .collect();

        // Each contradiction halves the score.
        let score = 0.5_f32.powi(reasons.len() as i32);

        Ok(VerifierVerdict {
            score,
            sufficient: false,
            reasons,
        })
    }

    fn name(&self) -> &'static str {
        "contradiction"
    }
}
