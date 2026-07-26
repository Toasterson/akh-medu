//! Phase 41 — The Thinking Engine: a unified reasoning loop with
//! test-time-compute scaling.
//!
//! Composes the existing reasoning primitives (spreading activation, e-graph
//! saturation, MCTS, argumentation, counterfactual fork, Dempster-Shafer
//! evidence fusion, superposition merging) into a single [`ThinkingLoop`]
//! with a measurable [`ThinkingBudget`] and a faithful [`ThoughtStep`] trace.
//!
//! - Plan: `docs/ai/plans/2026-04-27-phase41-thinking-engine.md`
//! - ADR: `docs/ai/decisions/052-unified-thinking-loop.md`
//! - Findings: `docs/ai/decisions/085-thinking-engine-findings.md`
//! - Anima ticket: `019dd0c5-9f4b-74e3-ad5c-3bc0a22e0ef9`
//!
//! # Quick start
//!
//! ```no_run
//! use akh_medu::engine::{Engine, EngineConfig};
//! use akh_medu::thinking::ThinkingBudget;
//!
//! let engine = Engine::new(EngineConfig::default()).unwrap();
//! let goal = engine.resolve_or_create_entity("why-is-the-sky-blue").unwrap();
//! let conclusion = engine.deliberate(goal, ThinkingBudget::steps(300)).unwrap();
//! println!("answer: {:?} ({})", conclusion.answer_symbol, conclusion.halt_reason.as_str());
//! ```
//!
//! # Query structure
//!
//! Deliberation goals are regular Entity symbols. Optional well-known
//! `thinking:*` relations sharpen the query (mirroring the `agent:*`
//! predicate convention):
//!
//! | predicate               | meaning                                    |
//! |-------------------------|--------------------------------------------|
//! | `thinking:seeks`        | the query subject (spreading seed)          |
//! | `thinking:relation`     | the query relation (`(S, R, ?)` shape)      |
//! | `thinking:answer-type`  | answers must be `is_a` this type            |
//! | `thinking:analogy-a/b/c`| `A : B :: C : ?` analogy terms              |
//! | `thinking:plan`         | marks a planning-shaped goal                |
//! | `thinking:asserted-by`  | evidence source attribution (Phase 18)      |

pub mod bench;
pub mod budget;
pub mod error;
pub mod loop_runner;
pub mod persistence;
pub mod scheduler;
pub mod state;
pub mod verifier;
pub mod voc_budget;

pub use budget::{BudgetResource, ThinkingBudget};
pub use error::{ThinkingError, ThinkingResult};
pub use loop_runner::{SchedulerChoice, ThinkingConfig, ThinkingLoop};
pub use persistence::{InMemoryThoughtPersistence, NullThoughtPersistence, ThoughtPersistence};
pub use scheduler::{
    GoalShape, GoalShapeScheduler, PrimitiveKind, RoundRobinScheduler, Scheduler,
    WeightedScheduler,
};
pub use state::{
    Conclusion, HaltReason, Hypothesis, ThoughtId, ThoughtOrigin, ThoughtState, ThoughtStep,
    TripleRef,
};
pub use verifier::{CompositeVerifier, ThoughtVerifier, VerifierReason, VerifierVerdict};
pub use voc_budget::{VocBudgetController, VocDecision};

use crate::engine::Engine;
use crate::error::AkhResult;
use crate::symbol::SymbolId;

/// Well-known `thinking:*` relation symbols, resolved once per loop
/// (mirrors the `AgentPredicates` pattern — no new `SymbolKind` variants).
#[derive(Debug, Clone)]
pub struct ThinkingPredicates {
    /// `thinking:seeks` — the query subject.
    pub seeks: SymbolId,
    /// `thinking:relation` — the query relation.
    pub relation: SymbolId,
    /// `thinking:answer-type` — answers must be `is_a` this type.
    pub answer_type: SymbolId,
    /// `thinking:analogy-a` — the A term of `A : B :: C : ?`.
    pub analogy_a: SymbolId,
    /// `thinking:analogy-b`.
    pub analogy_b: SymbolId,
    /// `thinking:analogy-c`.
    pub analogy_c: SymbolId,
    /// `thinking:plan` — marks planning-shaped goals.
    pub plan_marker: SymbolId,
    /// `thinking:intervention` — counterfactual trace attribution.
    pub intervention: SymbolId,
    /// `thinking:asserted-by` — evidence source attribution.
    pub asserted_by: SymbolId,
    /// `is_a` — the type predicate used for answer-type checks.
    pub is_a: SymbolId,
}

impl ThinkingPredicates {
    /// Resolve (or create) the well-known relations on this engine.
    pub fn resolve(engine: &Engine) -> ThinkingResult<Self> {
        let rel = |label: &str| -> ThinkingResult<SymbolId> {
            Ok(engine.resolve_or_create_relation(label)?)
        };
        Ok(Self {
            seeks: rel("thinking:seeks")?,
            relation: rel("thinking:relation")?,
            answer_type: rel("thinking:answer-type")?,
            analogy_a: rel("thinking:analogy-a")?,
            analogy_b: rel("thinking:analogy-b")?,
            analogy_c: rel("thinking:analogy-c")?,
            plan_marker: rel("thinking:plan")?,
            intervention: rel("thinking:intervention")?,
            asserted_by: rel("thinking:asserted-by")?,
            is_a: rel("is_a")?,
        })
    }
}

impl Engine {
    /// Deliberate on `goal` with the default thinking loop (goal-shape
    /// scheduler, composite verifier, VOC-adaptive budget, in-memory
    /// persistence).
    ///
    /// This is the "think harder" dial: a bigger budget buys more primitive
    /// steps. For persistent traces or custom scheduling build a
    /// [`ThinkingLoop`] directly.
    pub fn deliberate(&self, goal: SymbolId, budget: ThinkingBudget) -> AkhResult<Conclusion> {
        let mut thinking_loop = ThinkingLoop::new(self)?;
        Ok(thinking_loop.deliberate(goal, budget)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::graph::Triple;
    use crate::vsa::Dimension;
    use std::sync::Arc;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    /// Build a `(S, R, ?)` factual query goal with one strong and two weak
    /// candidate answers.
    fn factual_goal(engine: &Engine) -> (SymbolId, SymbolId) {
        let preds = ThinkingPredicates::resolve(engine).unwrap();
        let s = engine.resolve_or_create_entity("paris").unwrap();
        let r = engine.resolve_or_create_relation("capital-of").unwrap();
        let good = engine.resolve_or_create_entity("france").unwrap();
        let weak1 = engine.resolve_or_create_entity("texas").unwrap();
        let weak2 = engine.resolve_or_create_entity("atlantis").unwrap();
        let country = engine.resolve_or_create_entity("country").unwrap();

        for (o, conf) in [(good, 0.9f32), (weak1, 0.3), (weak2, 0.2)] {
            engine
                .add_triple(&Triple::new(s, r, o).with_confidence(conf))
                .unwrap();
            engine
                .add_triple(&Triple::new(o, preds.is_a, country))
                .unwrap();
        }

        let goal = engine.resolve_or_create_entity("query:capital").unwrap();
        engine.add_triple(&Triple::new(goal, preds.seeks, s)).unwrap();
        engine
            .add_triple(&Triple::new(goal, preds.relation, r))
            .unwrap();
        engine
            .add_triple(&Triple::new(goal, preds.answer_type, country))
            .unwrap();
        (goal, good)
    }

    #[test]
    fn deliberate_answers_factual_query() {
        let engine = test_engine();
        let (goal, expected) = factual_goal(&engine);
        let conclusion = engine
            .deliberate(goal, ThinkingBudget::steps(60))
            .unwrap();
        assert_eq!(
            conclusion.answer_symbol,
            Some(expected),
            "halt={:?} trace_kinds={:?}",
            conclusion.halt_reason,
            conclusion
                .trace
                .iter()
                .map(|s| s.kind())
                .collect::<Vec<_>>()
        );
        assert!(conclusion.confidence.belief > 0.4);
        assert!(!conclusion.trace.is_empty());
        assert!(conclusion.steps_used > 0);
    }

    #[test]
    fn budget_exhaustion_halts_loop() {
        let engine = test_engine();
        let (goal, _) = factual_goal(&engine);
        let mut tl = ThinkingLoop::new(&engine).unwrap();
        // Disable the adaptive budget + verifier sufficiency so only the
        // step budget can stop the loop.
        tl.config.adaptive_budget = false;
        tl.verifier_mut().config.sufficiency_threshold = 2.0; // unreachable
        let conclusion = tl.deliberate(goal, ThinkingBudget::steps(5)).unwrap();
        assert_eq!(conclusion.steps_used, 5);
        assert!(matches!(
            conclusion.halt_reason,
            HaltReason::BudgetExhausted {
                resource: BudgetResource::Steps
            }
        ));
    }

    #[test]
    fn ungrounded_goal_is_rejected() {
        let engine = test_engine();
        let mut tl = ThinkingLoop::new(&engine).unwrap();
        let bogus = SymbolId::new(999_999_999).unwrap();
        let err = tl.deliberate(bogus, ThinkingBudget::steps(10));
        assert!(matches!(err, Err(ThinkingError::UngroundedGoal { .. })));
    }

    #[test]
    fn invalid_budget_is_rejected() {
        let engine = test_engine();
        let (goal, _) = factual_goal(&engine);
        let mut tl = ThinkingLoop::new(&engine).unwrap();
        let err = tl.deliberate(goal, ThinkingBudget::steps(0));
        assert!(matches!(err, Err(ThinkingError::InvalidBudget { .. })));
    }

    #[test]
    fn trace_is_persisted_through_the_seam() {
        let engine = test_engine();
        let (goal, _) = factual_goal(&engine);
        let persistence = Arc::new(InMemoryThoughtPersistence::new());
        let mut tl = ThinkingLoop::new(&engine)
            .unwrap()
            .with_persistence(persistence.clone());
        let conclusion = tl.deliberate(goal, ThinkingBudget::steps(20)).unwrap();

        let trace = persistence.trace(conclusion.thought_id).expect("persisted");
        assert_eq!(trace.steps.len(), conclusion.trace.len());
        assert_eq!(trace.halt_reason.as_ref(), Some(&conclusion.halt_reason));
        assert_eq!(trace.origin.goal, goal);
    }

    #[test]
    fn multi_hop_chain_is_found_with_enough_budget() {
        let engine = test_engine();
        let preds = ThinkingPredicates::resolve(&engine).unwrap();
        let r = engine.resolve_or_create_relation("path").unwrap();
        let target_type = engine.resolve_or_create_entity("target").unwrap();

        // chain: n0 -path-> n1 -path-> n2 -path-> n3 (answer-typed)
        let nodes: Vec<SymbolId> = (0..4)
            .map(|i| engine.resolve_or_create_entity(&format!("hop{i}")).unwrap())
            .collect();
        for w in nodes.windows(2) {
            engine
                .add_triple(&Triple::new(w[0], r, w[1]).with_confidence(0.95))
                .unwrap();
        }
        engine
            .add_triple(&Triple::new(nodes[3], preds.is_a, target_type))
            .unwrap();

        let goal = engine.resolve_or_create_entity("query:chain").unwrap();
        engine
            .add_triple(&Triple::new(goal, preds.seeks, nodes[0]))
            .unwrap();
        engine
            .add_triple(&Triple::new(goal, preds.relation, r))
            .unwrap();
        engine
            .add_triple(&Triple::new(goal, preds.answer_type, target_type))
            .unwrap();

        let conclusion = engine
            .deliberate(goal, ThinkingBudget::steps(100))
            .unwrap();
        assert_eq!(
            conclusion.answer_symbol,
            Some(nodes[3]),
            "halt={:?}",
            conclusion.halt_reason
        );
    }

    #[test]
    fn goal_shape_classification_matches_structure() {
        let engine = test_engine();
        let preds = ThinkingPredicates::resolve(&engine).unwrap();
        let (factual, _) = factual_goal(&engine);
        assert_eq!(
            GoalShape::classify(&engine, factual, &preds),
            GoalShape::Factual
        );

        let a = engine.resolve_or_create_entity("a").unwrap();
        let analogy = engine.resolve_or_create_entity("query:analogy").unwrap();
        engine
            .add_triple(&Triple::new(analogy, preds.analogy_a, a))
            .unwrap();
        assert_eq!(
            GoalShape::classify(&engine, analogy, &preds),
            GoalShape::Analogy
        );

        let bare = engine.resolve_or_create_entity("just-a-thing").unwrap();
        assert_eq!(
            GoalShape::classify(&engine, bare, &preds),
            GoalShape::Exploratory
        );
    }

    #[test]
    fn analogy_goal_recovers_structured_binding() {
        let engine = test_engine();
        let preds = ThinkingPredicates::resolve(&engine).unwrap();
        let ops = engine.ops();

        // Structured item memory: B = A ⊗ R and D = C ⊗ R, so C ⊗ (A ⊗ B)
        // lands exactly on D.
        let a = engine.resolve_or_create_entity("king").unwrap();
        let b = engine.resolve_or_create_entity("queen").unwrap();
        let c = engine.resolve_or_create_entity("man").unwrap();
        let d = engine.resolve_or_create_entity("woman").unwrap();
        let mem = engine.item_memory();
        let va = mem.get_or_create(ops, a);
        let vc = mem.get_or_create(ops, c);
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(41);
        let r = ops.random(&mut rng);
        mem.insert(b, ops.bind(&va, &r).unwrap());
        mem.insert(d, ops.bind(&vc, &r).unwrap());
        // Distractors so the search is non-trivial.
        for i in 0..8 {
            let s = engine
                .resolve_or_create_entity(&format!("noise{i}"))
                .unwrap();
            mem.get_or_create(ops, s);
        }
        mem.rebuild_hnsw();

        let goal = engine.resolve_or_create_entity("query:king-queen").unwrap();
        engine.add_triple(&Triple::new(goal, preds.analogy_a, a)).unwrap();
        engine.add_triple(&Triple::new(goal, preds.analogy_b, b)).unwrap();
        engine.add_triple(&Triple::new(goal, preds.analogy_c, c)).unwrap();

        let conclusion = engine.deliberate(goal, ThinkingBudget::steps(40)).unwrap();
        assert_eq!(
            conclusion.answer_symbol,
            Some(d),
            "halt={:?}",
            conclusion.halt_reason
        );
    }
}
