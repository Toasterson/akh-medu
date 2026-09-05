//! Primitive-rotation scheduler (Phase 41b).
//!
//! Decides which reasoning primitive the loop runs next. Three
//! implementations, per the Phase 41 plan:
//!
//! - [`RoundRobinScheduler`] — static rotation in plan order (41a default).
//! - [`WeightedScheduler`] — deterministic stride scheduling over a weight
//!   table (no RNG, so deliberations are reproducible).
//! - [`GoalShapeScheduler`] — classifies the goal once against the KG and
//!   picks a hard-coded weight table: factual queries get heavier spreading
//!   activation + argumentation, planning queries heavier MCTS, analogy
//!   queries heavier retrieval/merge.
//!
//! Learned rotation policies are explicitly out of scope (plan: deferred to a
//! possible Phase 41h).

use serde::{Deserialize, Serialize};

use crate::engine::Engine;
use crate::symbol::SymbolId;

use super::ThinkingPredicates;
use super::error::{ThinkingError, ThinkingResult};
use super::state::ThoughtState;

// ---------------------------------------------------------------------------
// Primitive vocabulary
// ---------------------------------------------------------------------------

/// The reasoning primitives the loop can drive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrimitiveKind {
    SpreadingActivation,
    EgraphSaturate,
    MctsRollout,
    Argumentation,
    Counterfactual,
    EvidenceCombine,
    SuperpositionMerge,
}

impl PrimitiveKind {
    /// The full rotation, in plan order.
    pub fn all() -> [PrimitiveKind; 7] {
        [
            Self::SpreadingActivation,
            Self::EgraphSaturate,
            Self::MctsRollout,
            Self::Argumentation,
            Self::Counterfactual,
            Self::EvidenceCombine,
            Self::SuperpositionMerge,
        ]
    }

    /// Base VOC cost of one step of this primitive (Phase 11g currency).
    pub fn voc_cost(&self) -> f32 {
        match self {
            Self::SpreadingActivation => 1.0,
            Self::EgraphSaturate => 2.0,
            Self::MctsRollout => 2.0,
            Self::Argumentation => 1.5,
            Self::Counterfactual => 1.5,
            Self::EvidenceCombine => 0.5,
            Self::SuperpositionMerge => 0.5,
        }
    }
}

impl std::fmt::Display for PrimitiveKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::SpreadingActivation => "spreading-activation",
            Self::EgraphSaturate => "egraph-saturate",
            Self::MctsRollout => "mcts-rollout",
            Self::Argumentation => "argumentation",
            Self::Counterfactual => "counterfactual",
            Self::EvidenceCombine => "evidence-combine",
            Self::SuperpositionMerge => "superposition-merge",
        };
        f.write_str(s)
    }
}

// ---------------------------------------------------------------------------
// Scheduler trait
// ---------------------------------------------------------------------------

/// Chooses the next primitive given the current thought state.
pub trait Scheduler: Send {
    /// Pick the next primitive to run.
    fn next(&mut self, state: &ThoughtState) -> ThinkingResult<PrimitiveKind>;

    /// Human-readable scheduler name (for traces and benchmarks).
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// Round-robin
// ---------------------------------------------------------------------------

/// Static rotation in a fixed order.
#[derive(Debug, Clone)]
pub struct RoundRobinScheduler {
    rotation: Vec<PrimitiveKind>,
    idx: usize,
}

impl RoundRobinScheduler {
    /// The full plan-order rotation.
    pub fn full() -> Self {
        Self {
            rotation: PrimitiveKind::all().to_vec(),
            idx: 0,
        }
    }

    /// A custom rotation (must be non-empty).
    pub fn new(rotation: Vec<PrimitiveKind>) -> ThinkingResult<Self> {
        if rotation.is_empty() {
            return Err(ThinkingError::EmptyRotation);
        }
        Ok(Self { rotation, idx: 0 })
    }
}

impl Scheduler for RoundRobinScheduler {
    fn next(&mut self, _state: &ThoughtState) -> ThinkingResult<PrimitiveKind> {
        let p = self.rotation[self.idx % self.rotation.len()];
        self.idx = self.idx.wrapping_add(1);
        Ok(p)
    }

    fn name(&self) -> &'static str {
        "round-robin"
    }
}

// ---------------------------------------------------------------------------
// Weighted (deterministic stride scheduling)
// ---------------------------------------------------------------------------

/// Deterministic weighted interleaving.
///
/// Classic stride scheduling: each `next()` adds every primitive's weight to
/// its accumulator, picks the largest accumulator, and subtracts the weight
/// total from the winner. Over N picks each primitive runs in proportion to
/// its weight, with no randomness.
#[derive(Debug, Clone)]
pub struct WeightedScheduler {
    entries: Vec<(PrimitiveKind, f32)>,
    acc: Vec<f32>,
    total: f32,
}

impl WeightedScheduler {
    /// Build from a weight table. Zero-weight entries are dropped.
    pub fn new(weights: &[(PrimitiveKind, f32)]) -> ThinkingResult<Self> {
        let entries: Vec<(PrimitiveKind, f32)> =
            weights.iter().copied().filter(|(_, w)| *w > 0.0).collect();
        if entries.is_empty() {
            return Err(ThinkingError::BadWeights {
                reason: "no strictly positive weights".into(),
            });
        }
        if entries.iter().any(|(_, w)| !w.is_finite()) {
            return Err(ThinkingError::BadWeights {
                reason: "weights must be finite".into(),
            });
        }
        let total = entries.iter().map(|(_, w)| w).sum();
        let acc = vec![0.0; entries.len()];
        Ok(Self {
            entries,
            acc,
            total,
        })
    }
}

impl Scheduler for WeightedScheduler {
    fn next(&mut self, _state: &ThoughtState) -> ThinkingResult<PrimitiveKind> {
        let mut best = 0usize;
        for i in 0..self.entries.len() {
            self.acc[i] += self.entries[i].1;
            if self.acc[i] > self.acc[best] {
                best = i;
            }
        }
        self.acc[best] -= self.total;
        Ok(self.entries[best].0)
    }

    fn name(&self) -> &'static str {
        "weighted"
    }
}

// ---------------------------------------------------------------------------
// Goal shape
// ---------------------------------------------------------------------------

/// Canonical goal shapes recognised by the [`GoalShapeScheduler`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoalShape {
    /// A `(subject, relation, ?)` query: spreading + argumentation + evidence.
    Factual,
    /// An `A : B :: C : ?` query: retrieval + superposition maintenance.
    Analogy,
    /// A "what should be done" goal: MCTS + counterfactual heavy.
    Planning,
    /// No recognised structure: explore the neighbourhood.
    Exploratory,
}

impl GoalShape {
    /// Hard-coded weight table per shape (plan 41b: mapping is small and
    /// hard-coded; learned policies are deferred).
    pub fn weights(&self) -> Vec<(PrimitiveKind, f32)> {
        use PrimitiveKind as P;
        match self {
            Self::Factual => vec![
                (P::SpreadingActivation, 3.0),
                (P::Argumentation, 3.0),
                (P::EvidenceCombine, 2.0),
                (P::SuperpositionMerge, 1.0),
            ],
            Self::Analogy => vec![
                (P::SpreadingActivation, 4.0),
                (P::SuperpositionMerge, 2.0),
                (P::EvidenceCombine, 1.0),
            ],
            Self::Planning => vec![
                (P::MctsRollout, 4.0),
                (P::SpreadingActivation, 2.0),
                (P::Counterfactual, 1.0),
                (P::EvidenceCombine, 1.0),
                (P::SuperpositionMerge, 1.0),
            ],
            Self::Exploratory => vec![
                (P::SpreadingActivation, 4.0),
                (P::Argumentation, 1.0),
                (P::EvidenceCombine, 1.0),
                (P::SuperpositionMerge, 1.0),
            ],
        }
    }

    /// Classify a goal symbol against the KG.
    ///
    /// - analogy markers (`thinking:analogy-a`) → [`GoalShape::Analogy`]
    /// - a query relation (`thinking:relation`) → [`GoalShape::Factual`]
    /// - a plan marker (`thinking:plan`) or a `plan:`-prefixed label →
    ///   [`GoalShape::Planning`]
    /// - otherwise → [`GoalShape::Exploratory`]
    pub fn classify(engine: &Engine, goal: SymbolId, preds: &ThinkingPredicates) -> Self {
        let triples = engine.triples_from(goal);
        let has = |p: SymbolId| triples.iter().any(|t| t.predicate == p);
        if has(preds.analogy_a) {
            return Self::Analogy;
        }
        if has(preds.plan_marker) || engine.resolve_label(goal).starts_with("plan:") {
            return Self::Planning;
        }
        if has(preds.relation) {
            return Self::Factual;
        }
        Self::Exploratory
    }
}

/// Scheduler that biases the rotation toward primitives matching the goal's
/// shape (classified once, at construction).
#[derive(Debug, Clone)]
pub struct GoalShapeScheduler {
    shape: GoalShape,
    inner: WeightedScheduler,
}

impl GoalShapeScheduler {
    /// Classify `goal` and build the shape-specific weighted rotation.
    pub fn for_goal(
        engine: &Engine,
        goal: SymbolId,
        preds: &ThinkingPredicates,
    ) -> ThinkingResult<Self> {
        let shape = GoalShape::classify(engine, goal, preds);
        let inner = WeightedScheduler::new(&shape.weights())?;
        Ok(Self { shape, inner })
    }

    /// The classified shape.
    pub fn shape(&self) -> GoalShape {
        self.shape
    }
}

impl Scheduler for GoalShapeScheduler {
    fn next(&mut self, state: &ThoughtState) -> ThinkingResult<PrimitiveKind> {
        self.inner.next(state)
    }

    fn name(&self) -> &'static str {
        "goal-shape"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // A ThoughtState is only passed through; build a minimal one.
    fn dummy_state() -> ThoughtState {
        use crate::vsa::{Dimension, Encoding, HyperVec};
        ThoughtState {
            id: super::super::state::ThoughtId::next(),
            focus_vec: HyperVec::zero(Dimension::TEST, Encoding::Bipolar),
            hypotheses: Vec::new(),
            open_goals: Vec::new(),
            trace: Vec::new(),
            budget: super::super::budget::ThinkingBudget::steps(10),
            origin: super::super::state::ThoughtOrigin {
                goal: SymbolId::new(1).unwrap(),
                parent: None,
                started_at: 0,
            },
            frontier: Default::default(),
            expanded: Default::default(),
            activation: Default::default(),
            excluded: Default::default(),
            focus_deltas: Default::default(),
            analogy_done: false,
            last_verifier_score: 0.0,
        }
    }

    #[test]
    fn round_robin_cycles_in_order() {
        let mut s = RoundRobinScheduler::full();
        let st = dummy_state();
        let first: Vec<PrimitiveKind> = (0..7).map(|_| s.next(&st).unwrap()).collect();
        assert_eq!(first, PrimitiveKind::all().to_vec());
        // Wraps around.
        assert_eq!(s.next(&st).unwrap(), PrimitiveKind::SpreadingActivation);
    }

    #[test]
    fn empty_rotation_rejected() {
        assert!(matches!(
            RoundRobinScheduler::new(vec![]),
            Err(ThinkingError::EmptyRotation)
        ));
    }

    #[test]
    fn weighted_honours_configured_weights() {
        let mut s = WeightedScheduler::new(&[
            (PrimitiveKind::SpreadingActivation, 3.0),
            (PrimitiveKind::EvidenceCombine, 1.0),
        ])
        .unwrap();
        let st = dummy_state();
        let mut counts: HashMap<PrimitiveKind, usize> = HashMap::new();
        for _ in 0..400 {
            *counts.entry(s.next(&st).unwrap()).or_default() += 1;
        }
        let spread = counts[&PrimitiveKind::SpreadingActivation];
        let evid = counts[&PrimitiveKind::EvidenceCombine];
        assert_eq!(spread + evid, 400);
        assert_eq!(spread, 300, "3:1 weights should give exactly 3/4 of picks");
        assert_eq!(evid, 100);
    }

    #[test]
    fn weighted_rejects_zero_weights() {
        assert!(matches!(
            WeightedScheduler::new(&[(PrimitiveKind::MctsRollout, 0.0)]),
            Err(ThinkingError::BadWeights { .. })
        ));
    }

    #[test]
    fn weighted_is_deterministic() {
        let table = [
            (PrimitiveKind::SpreadingActivation, 2.0),
            (PrimitiveKind::Argumentation, 1.5),
            (PrimitiveKind::SuperpositionMerge, 0.5),
        ];
        let st = dummy_state();
        let mut a = WeightedScheduler::new(&table).unwrap();
        let mut b = WeightedScheduler::new(&table).unwrap();
        for _ in 0..100 {
            assert_eq!(a.next(&st).unwrap(), b.next(&st).unwrap());
        }
    }

    #[test]
    fn shape_weight_tables_are_valid() {
        for shape in [
            GoalShape::Factual,
            GoalShape::Analogy,
            GoalShape::Planning,
            GoalShape::Exploratory,
        ] {
            let w = shape.weights();
            assert!(!w.is_empty());
            assert!(WeightedScheduler::new(&w).is_ok());
        }
    }

    #[test]
    fn planning_shape_favours_mcts() {
        let mut s = WeightedScheduler::new(&GoalShape::Planning.weights()).unwrap();
        let st = dummy_state();
        let mut counts: HashMap<PrimitiveKind, usize> = HashMap::new();
        for _ in 0..90 {
            *counts.entry(s.next(&st).unwrap()).or_default() += 1;
        }
        let mcts = counts
            .get(&PrimitiveKind::MctsRollout)
            .copied()
            .unwrap_or(0);
        for (kind, n) in &counts {
            if *kind != PrimitiveKind::MctsRollout {
                assert!(
                    mcts > *n,
                    "MCTS should dominate the planning rotation (mcts={mcts}, {kind}={n})"
                );
            }
        }
    }
}
