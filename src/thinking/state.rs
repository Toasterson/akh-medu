//! `ThoughtState` and the faithful reasoning trace (Phase 41a).
//!
//! A [`ThoughtState`] is a unit of cognitive focus during deliberation:
//! a HyperVec focus (the "residual stream"), hypotheses held in superposition
//! with Dempster-Shafer belief, open goals, the monotonically-grown trace,
//! and the remaining budget.
//!
//! Held in memory during a thinking session; every emitted [`ThoughtStep`] is
//! offered to the [`super::persistence::ThoughtPersistence`] seam
//! (Phase 41d will route it to `akhomed.thinking_traces` via the Phase 38 WAL).

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::agent::evidence::{BeliefInterval, MassFunction};
use crate::symbol::SymbolId;
use crate::vsa::HyperVec;

use super::budget::{BudgetResource, ThinkingBudget};
use super::verifier::VerifierReason;

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

/// Process-unique identifier for one deliberation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ThoughtId(pub u64);

static NEXT_THOUGHT_ID: AtomicU64 = AtomicU64::new(1);

impl ThoughtId {
    /// Allocate the next process-unique thought id.
    pub fn next() -> Self {
        Self(NEXT_THOUGHT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl std::fmt::Display for ThoughtId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "thought:{}", self.0)
    }
}

/// Provenance bookkeeping for a deliberation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtOrigin {
    /// The goal symbol the deliberation was opened for.
    pub goal: SymbolId,
    /// Parent thought (cross-thought dependencies; see plan open question 3).
    pub parent: Option<ThoughtId>,
    /// Unix seconds when deliberation started.
    pub started_at: u64,
}

// ---------------------------------------------------------------------------
// Trace vocabulary
// ---------------------------------------------------------------------------

/// A lightweight, serializable reference to a triple (for trace payloads).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TripleRef {
    pub subject: SymbolId,
    pub predicate: SymbolId,
    pub object: SymbolId,
}

/// One primitive step in the reasoning trace.
///
/// The trace *is* the computation: every variant records what the primitive
/// actually did, not a verbal rationalisation of it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThoughtStep {
    /// One frontier expansion of spreading activation (or one VSA analogy
    /// recall, which is the same retrieval primitive over item memory).
    SpreadingActivation {
        frontier_size: u32,
        new_concepts: Vec<SymbolId>,
        novelty: f32,
    },
    /// One equality-saturation pass over the goal expression.
    EgraphSaturate {
        iterations: u32,
        eclasses_added: u32,
        best_cost_delta: f32,
    },
    /// One (small) batch of MCTS rollouts.
    MctsRollout {
        trajectory_len: u32,
        return_value: f32,
        action: Option<SymbolId>,
    },
    /// One pro/con argumentation round on the query topic.
    Argumentation {
        topic: SymbolId,
        answer: Option<SymbolId>,
        confidence: f32,
    },
    /// One counterfactual fork (None intervention = no applicable actions).
    Counterfactual {
        intervention: Option<TripleRef>,
        divergence: f32,
    },
    /// One Dempster-Shafer combination pass over hypothesis evidence.
    EvidenceCombine {
        conflict: f32,
        narrowed_interval: BeliefInterval,
    },
    /// Superposition maintenance: merge near-duplicates, prune low belief.
    SuperpositionMerge {
        hypotheses_in: u32,
        hypotheses_out: u32,
        mean_similarity: f32,
    },
    /// A structural-verifier evaluation (free of step budget; bookkeeping).
    VerifierScore {
        score: f32,
        sufficient: bool,
        reasons: Vec<VerifierReason>,
    },
}

impl ThoughtStep {
    /// Short kind label (stable; used for CSV emission and 41d payload rows).
    pub fn kind(&self) -> &'static str {
        match self {
            Self::SpreadingActivation { .. } => "spreading-activation",
            Self::EgraphSaturate { .. } => "egraph-saturate",
            Self::MctsRollout { .. } => "mcts-rollout",
            Self::Argumentation { .. } => "argumentation",
            Self::Counterfactual { .. } => "counterfactual",
            Self::EvidenceCombine { .. } => "evidence-combine",
            Self::SuperpositionMerge { .. } => "superposition-merge",
            Self::VerifierScore { .. } => "verifier-score",
        }
    }
}

// ---------------------------------------------------------------------------
// Hypotheses
// ---------------------------------------------------------------------------

/// A candidate answer held in superposition.
#[derive(Debug, Clone)]
pub struct Hypothesis {
    /// Resolved symbol, if the hypothesis vector cleaned up to one.
    pub symbol: Option<SymbolId>,
    /// VSA encoding of the candidate answer.
    pub vec: HyperVec,
    /// Dempster-Shafer mass over {true, false} for this candidate.
    pub mass: MassFunction,
    /// Sources whose evidence contributed (for the Phase 18 verifier).
    pub sources: Vec<SymbolId>,
}

impl Hypothesis {
    /// The belief bound `[Bel, Pl]` on this hypothesis.
    pub fn interval(&self) -> BeliefInterval {
        self.mass.belief_interval()
    }

    /// Lower belief bound (used for ranking and pruning).
    pub fn belief(&self) -> f32 {
        self.mass.m_true
    }
}

// ---------------------------------------------------------------------------
// ThoughtState
// ---------------------------------------------------------------------------

/// A unit of cognitive focus during deliberation.
#[derive(Debug, Clone)]
pub struct ThoughtState {
    /// Identity of this deliberation.
    pub id: ThoughtId,
    /// HyperVec representation of current cognitive focus.
    /// Accumulates as the loop proceeds via VSA bundle (majority-vote
    /// superposition) of the goal and recently activated concepts.
    pub focus_vec: HyperVec,
    /// Multiple hypotheses held simultaneously in superposition.
    pub hypotheses: Vec<Hypothesis>,
    /// Goals that opened during deliberation.
    pub open_goals: Vec<SymbolId>,
    /// The reasoning trace, monotonically grown.
    pub trace: Vec<ThoughtStep>,
    /// Remaining budget. Charged on every primitive step.
    pub budget: ThinkingBudget,
    /// Provenance bookkeeping.
    pub origin: ThoughtOrigin,

    // -- Working memory (loop-internal; rebuilt on resume, not persisted) --
    /// Symbols awaiting frontier expansion, ordered by activation.
    pub(crate) frontier: VecDeque<SymbolId>,
    /// Symbols already expanded by spreading activation.
    pub(crate) expanded: HashSet<SymbolId>,
    /// Activation confidence per symbol (product of edge confidences).
    pub(crate) activation: HashMap<SymbolId, f32>,
    /// Symbols that can never be answers: the goal itself, its seeds, and
    /// the objects of its `thinking:*` metadata triples (query structure is
    /// not knowledge).
    pub(crate) excluded: HashSet<SymbolId>,
    /// Recent focus_vec similarity deltas (for the convergence verifier).
    pub(crate) focus_deltas: VecDeque<f32>,
    /// Whether the one-shot analogy recall has already run.
    pub(crate) analogy_done: bool,
    /// Latest composite verifier score.
    pub(crate) last_verifier_score: f32,
}

impl ThoughtState {
    /// Count of budget-charged primitive steps in the trace
    /// (verifier evaluations are bookkeeping and excluded).
    pub fn primitive_steps(&self) -> usize {
        self.trace
            .iter()
            .filter(|s| !matches!(s, ThoughtStep::VerifierScore { .. }))
            .count()
    }

    /// The hypothesis with the highest belief, if any.
    pub fn top_hypothesis(&self) -> Option<&Hypothesis> {
        self.hypotheses
            .iter()
            .max_by(|a, b| a.belief().total_cmp(&b.belief()))
    }

    /// Anytime read access: current best answer even mid-deliberation
    /// (plan open question 4).
    pub fn current_best(&self) -> Option<(Option<SymbolId>, BeliefInterval)> {
        self.top_hypothesis().map(|h| (h.symbol, h.interval()))
    }
}

// ---------------------------------------------------------------------------
// Conclusion
// ---------------------------------------------------------------------------

/// Why the loop stopped.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HaltReason {
    /// Focus stabilised across the convergence window.
    Converged { delta: f32 },
    /// A budget axis ran out.
    BudgetExhausted { resource: BudgetResource },
    /// The structural verifier declared the answer sufficient.
    VerifierSufficient { score: f32 },
    /// Phase 41e: marginal value of another step fell below threshold.
    DiminishingReturns { marginal_value: f32 },
    /// Aborted (e.g. fatal contradiction or caller cancellation).
    Aborted { reason: String },
}

impl HaltReason {
    /// Stable kebab-case label for CSV emission.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Converged { .. } => "converged",
            Self::BudgetExhausted { .. } => "budget-exhausted",
            Self::VerifierSufficient { .. } => "verifier-sufficient",
            Self::DiminishingReturns { .. } => "diminishing-returns",
            Self::Aborted { .. } => "aborted",
        }
    }
}

/// The result of a deliberation.
#[derive(Debug, Clone)]
pub struct Conclusion {
    /// Identity of the deliberation that produced this conclusion.
    pub thought_id: ThoughtId,
    /// VSA encoding of the answer (top hypothesis, or the focus vector when
    /// no hypothesis formed).
    pub answer: HyperVec,
    /// The answer resolved to a symbol via item memory, when possible.
    pub answer_symbol: Option<SymbolId>,
    /// The faithful reasoning trace.
    pub trace: Vec<ThoughtStep>,
    /// Dempster-Shafer belief bound on the answer.
    pub confidence: BeliefInterval,
    /// Budget-charged primitive steps consumed.
    pub steps_used: u32,
    /// Wall-clock consumed (milliseconds).
    pub wall_ms: u64,
    /// Phase 11g VOC credits consumed.
    pub voc_used: f32,
    /// Why deliberation stopped.
    pub halt_reason: HaltReason,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thought_ids_are_unique() {
        let a = ThoughtId::next();
        let b = ThoughtId::next();
        assert_ne!(a, b);
    }

    #[test]
    fn thought_step_serializes_to_json() {
        let step = ThoughtStep::EvidenceCombine {
            conflict: 0.1,
            narrowed_interval: BeliefInterval {
                belief: 0.6,
                plausibility: 0.9,
            },
        };
        let json = serde_json::to_string(&step).expect("serialize");
        assert!(json.contains("EvidenceCombine"));
        let back: ThoughtStep = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.kind(), "evidence-combine");
    }

    #[test]
    fn halt_reason_labels_are_stable() {
        assert_eq!(
            HaltReason::BudgetExhausted {
                resource: BudgetResource::Steps
            }
            .as_str(),
            "budget-exhausted"
        );
        assert_eq!(HaltReason::Converged { delta: 0.0 }.as_str(), "converged");
    }
}
