//! The unified `ThinkingLoop` (Phase 41a): drives the existing reasoning
//! primitives in a scheduled rotation over a persistent [`ThoughtState`].
//!
//! No new reasoning algorithms live here — every primitive delegates to a
//! shipped subsystem:
//!
//! | primitive             | delegates to                                         |
//! |-----------------------|------------------------------------------------------|
//! | spreading activation  | `Engine::infer` (one frontier ring per step) and     |
//! |                       | `Engine::infer_analogy` for analogy-shaped goals      |
//! | e-graph saturate      | egg `Runner` over `Engine::all_rules`                 |
//! | MCTS rollout          | `agent::mcts::MctsPlanner` (Phase 16a)                |
//! | argumentation         | `argumentation::argue` (Cyc-style pro/con)            |
//! | counterfactual        | `agent::counterfactual::counterfactual_query` (15c)   |
//! | evidence combine      | `agent::evidence::MassFunction` Dempster rule (17)    |
//! | superposition merge   | `VsaOps::bundle` majority-vote superposition          |
//!
//! The loop halts on convergence, budget exhaustion, verifier sufficiency
//! (Phase 41c), or diminishing VOC returns (Phase 41e).

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use egg::{AstSize, Extractor, Runner};

use crate::agent::causal::CausalManager;
use crate::agent::counterfactual::{CounterfactualQuery, counterfactual_query};
use crate::agent::evidence::MassFunction;
use crate::agent::mcts::{MctsConfig, MctsPlanner};
use crate::agent::state_value::{AgentState, ValueFunction};
use crate::engine::Engine;
use crate::infer::InferenceQuery;
use crate::symbol::{SymbolId, SymbolKind};
use crate::vsa::HyperVec;

use super::ThinkingPredicates;
use super::budget::ThinkingBudget;
use super::error::{ThinkingError, ThinkingResult};
use super::persistence::{InMemoryThoughtPersistence, ThoughtPersistence};
use super::scheduler::{
    GoalShapeScheduler, PrimitiveKind, RoundRobinScheduler, Scheduler, WeightedScheduler,
};
use super::state::{
    Conclusion, HaltReason, Hypothesis, ThoughtId, ThoughtOrigin, ThoughtState, ThoughtStep,
    TripleRef,
};
use super::verifier::{CompositeVerifier, ThoughtVerifier};
use super::voc_budget::{VocBudgetController, VocDecision};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Which scheduler the loop builds for each deliberation.
#[derive(Debug, Clone)]
pub enum SchedulerChoice {
    /// Full rotation in plan order (41a baseline).
    RoundRobin,
    /// Goal-shape-biased weighted rotation (41b default).
    GoalShape,
    /// Explicit weight table.
    Weighted(Vec<(PrimitiveKind, f32)>),
}

/// Tunables for the thinking loop.
#[derive(Debug, Clone)]
pub struct ThinkingConfig {
    /// Frontier symbols expanded per spreading-activation step.
    pub beam_width: usize,
    /// Per-seed activation retrieval cap.
    pub activation_top_k: usize,
    /// Minimum propagated confidence for spreading activation.
    pub min_confidence: f32,
    /// Run the full composite verifier every N primitive steps.
    pub verify_every: u32,
    /// Superposition capacity before pruning (~7 fillers in 10K bits).
    pub hypothesis_capacity: usize,
    /// Prune hypotheses whose belief falls below this (plan open question 2).
    pub prune_belief_floor: f32,
    /// Merge hypotheses whose vectors are at least this similar.
    pub merge_similarity: f32,
    /// How many focus deltas to retain for the convergence verifier.
    pub focus_window: usize,
    /// MCTS iterations per rollout step (kept tiny: one step = one probe).
    pub mcts_iterations: usize,
    /// Analogy retrieval candidates.
    pub analogy_top_k: usize,
    /// Consult the VOC controller (Phase 41e). Disable for fixed-budget
    /// scaling measurements.
    pub adaptive_budget: bool,
}

impl Default for ThinkingConfig {
    fn default() -> Self {
        Self {
            beam_width: 4,
            activation_top_k: 32,
            min_confidence: 0.02,
            verify_every: 4,
            hypothesis_capacity: 7,
            prune_belief_floor: 0.05,
            merge_similarity: 0.95,
            focus_window: 16,
            mcts_iterations: 2,
            analogy_top_k: 8,
            adaptive_budget: true,
        }
    }
}

// ---------------------------------------------------------------------------
// The loop
// ---------------------------------------------------------------------------

/// The unified thinking loop. Holds no KG state of its own — everything it
/// knows lives in the [`Engine`] or the per-deliberation [`ThoughtState`].
pub struct ThinkingLoop<'e> {
    engine: &'e Engine,
    preds: ThinkingPredicates,
    scheduler_choice: SchedulerChoice,
    verifier: CompositeVerifier,
    persistence: Arc<dyn ThoughtPersistence>,
    pub config: ThinkingConfig,
    /// Lazily-built causal manager for MCTS/counterfactual probes.
    /// `None` = not yet attempted; `Some(None)` = attempted, unavailable.
    causal: Option<Option<CausalManager>>,
    value_fn: ValueFunction,
}

impl<'e> ThinkingLoop<'e> {
    /// Build a loop with the 41b goal-shape scheduler, the 41c composite
    /// verifier, and in-memory persistence.
    pub fn new(engine: &'e Engine) -> ThinkingResult<Self> {
        let preds = ThinkingPredicates::resolve(engine)?;
        let verifier = CompositeVerifier::new(engine, &preds);
        Ok(Self {
            engine,
            preds,
            scheduler_choice: SchedulerChoice::GoalShape,
            verifier,
            persistence: Arc::new(InMemoryThoughtPersistence::new()),
            config: ThinkingConfig::default(),
            causal: None,
            value_fn: ValueFunction::default(),
        })
    }

    /// Override the scheduler choice.
    pub fn with_scheduler(mut self, choice: SchedulerChoice) -> Self {
        self.scheduler_choice = choice;
        self
    }

    /// Override the persistence backend (Phase 41d seam).
    pub fn with_persistence(mut self, persistence: Arc<dyn ThoughtPersistence>) -> Self {
        self.persistence = persistence;
        self
    }

    /// Override the config.
    pub fn with_config(mut self, config: ThinkingConfig) -> Self {
        self.config = config;
        self
    }

    /// Access the composite verifier (tests tune thresholds through this).
    pub fn verifier_mut(&mut self) -> &mut CompositeVerifier {
        &mut self.verifier
    }

    // -----------------------------------------------------------------------
    // Entry point
    // -----------------------------------------------------------------------

    /// Deliberate on `goal` under `budget` until a halt condition fires.
    pub fn deliberate(
        &mut self,
        goal: SymbolId,
        budget: ThinkingBudget,
    ) -> ThinkingResult<Conclusion> {
        budget.validate()?;
        if self.engine.get_symbol_meta(goal).is_err() {
            return Err(ThinkingError::UngroundedGoal { goal: goal.get() });
        }

        let started = Instant::now();
        let mut state = self.initial_state(goal, budget)?;
        let mut scheduler = self.build_scheduler(goal)?;
        let mut voc = self
            .config
            .adaptive_budget
            .then(|| VocBudgetController::for_budget(&state.budget));

        self.persistence.begin(state.id, &state.origin)?;
        let mut flushed: u32 = 0;

        loop {
            // -- Halting: budget --
            if let Some(resource) = state.budget.exhausted() {
                return self.conclude(state, started, HaltReason::BudgetExhausted { resource });
            }

            // -- One primitive step --
            let primitive = scheduler.next(&state)?;
            let step_start = Instant::now();
            let step = self.run_primitive(primitive, &mut state)?;
            state
                .budget
                .charge(primitive.voc_cost(), step_start.elapsed().as_millis() as u64);
            state.trace.push(step.clone());
            self.persistence.flush_step(state.id, flushed, &step)?;
            flushed += 1;

            let steps = state.budget.used_steps;

            // -- Halting: convergence (cheap check, every step) --
            if let Some(delta) = self.verifier.convergence.stable(&state)
                && steps as usize > self.config.focus_window
            {
                return self.conclude(state, started, HaltReason::Converged { delta });
            }

            // -- Full structural verification every K steps --
            if steps % self.config.verify_every == 0 {
                let verdict = self.verifier.score(self.engine, &state)?;
                state.last_verifier_score = verdict.score;
                let vstep = ThoughtStep::VerifierScore {
                    score: verdict.score,
                    sufficient: verdict.sufficient,
                    reasons: verdict.reasons.clone(),
                };
                state.trace.push(vstep.clone());
                self.persistence.flush_step(state.id, flushed, &vstep)?;
                flushed += 1;

                if let Some(ctl) = voc.as_mut() {
                    ctl.observe_score(goal, verdict.score);
                }

                if verdict.sufficient {
                    return self.conclude(
                        state,
                        started,
                        HaltReason::VerifierSufficient {
                            score: verdict.score,
                        },
                    );
                }
            }

            // -- VOC-driven adaptive budget (Phase 41e) --
            if let Some(ctl) = voc.as_ref()
                && steps % ctl.check_every == 0
            {
                let top_belief = state.top_hypothesis().map(|h| h.belief()).unwrap_or(0.0);
                match ctl.decide(goal, &state.budget, top_belief) {
                    VocDecision::Continue => {}
                    VocDecision::Halt { marginal_value } => {
                        return self.conclude(
                            state,
                            started,
                            HaltReason::DiminishingReturns { marginal_value },
                        );
                    }
                    VocDecision::Extend { extra, .. } => {
                        let cap = ctl.global_step_cap;
                        state.budget.extend_steps(extra, cap);
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // State setup / teardown
    // -----------------------------------------------------------------------

    fn initial_state(
        &self,
        goal: SymbolId,
        budget: ThinkingBudget,
    ) -> ThinkingResult<ThoughtState> {
        let focus_vec = self
            .engine
            .item_memory()
            .get_or_create(self.engine.ops(), goal);

        // Seeds: explicit `thinking:seeks` objects, else the goal itself.
        let goal_triples = self.engine.triples_from(goal);
        let mut seeds: Vec<SymbolId> = goal_triples
            .iter()
            .filter(|t| t.predicate == self.preds.seeks)
            .map(|t| t.object)
            .collect();
        if seeds.is_empty() {
            seeds.push(goal);
        }
        seeds.sort();
        seeds.dedup();

        let mut activation = HashMap::new();
        let mut frontier = VecDeque::new();
        for &s in &seeds {
            activation.insert(s, 1.0f32);
            frontier.push_back(s);
        }

        // Query structure is not knowledge: the goal, its seeds, and the
        // objects of its `thinking:*` metadata triples can never be answers.
        let meta_preds = [
            self.preds.seeks,
            self.preds.relation,
            self.preds.answer_type,
            self.preds.analogy_a,
            self.preds.analogy_b,
            self.preds.analogy_c,
            self.preds.plan_marker,
            self.preds.intervention,
            self.preds.asserted_by,
        ];
        let mut excluded: HashSet<SymbolId> = goal_triples
            .iter()
            .filter(|t| meta_preds.contains(&t.predicate))
            .map(|t| t.object)
            .collect();
        excluded.insert(goal);
        excluded.extend(seeds.iter().copied());

        let started_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Ok(ThoughtState {
            id: ThoughtId::next(),
            focus_vec,
            hypotheses: Vec::new(),
            open_goals: vec![goal],
            trace: Vec::new(),
            budget,
            origin: ThoughtOrigin {
                goal,
                parent: None,
                started_at,
            },
            frontier,
            expanded: HashSet::new(),
            activation,
            excluded,
            focus_deltas: VecDeque::new(),
            analogy_done: false,
            last_verifier_score: 0.0,
        })
    }

    fn build_scheduler(&self, goal: SymbolId) -> ThinkingResult<Box<dyn Scheduler>> {
        Ok(match &self.scheduler_choice {
            SchedulerChoice::RoundRobin => Box::new(RoundRobinScheduler::full()),
            SchedulerChoice::GoalShape => Box::new(GoalShapeScheduler::for_goal(
                self.engine,
                goal,
                &self.preds,
            )?),
            SchedulerChoice::Weighted(table) => Box::new(WeightedScheduler::new(table)?),
        })
    }

    fn conclude(
        &self,
        state: ThoughtState,
        started: Instant,
        halt_reason: HaltReason,
    ) -> ThinkingResult<Conclusion> {
        let (answer, answer_symbol, confidence) = match state.top_hypothesis() {
            Some(top) => {
                let symbol = match top.symbol {
                    Some(s) => Some(s),
                    // Clean up the hypothesis vector through item memory.
                    None => self
                        .engine
                        .item_memory()
                        .search(&top.vec, 1)
                        .ok()
                        .and_then(|r| r.first().map(|n| n.symbol_id)),
                };
                (top.vec.clone(), symbol, top.interval())
            }
            None => (
                state.focus_vec.clone(),
                None,
                MassFunction::vacuous().belief_interval(),
            ),
        };

        let conclusion = Conclusion {
            thought_id: state.id,
            answer,
            answer_symbol,
            trace: state.trace,
            confidence,
            steps_used: state.budget.used_steps,
            wall_ms: started.elapsed().as_millis() as u64,
            voc_used: state.budget.used_voc,
            halt_reason,
        };
        self.persistence.conclude(conclusion.thought_id, &conclusion)?;
        Ok(conclusion)
    }

    // -----------------------------------------------------------------------
    // Primitive dispatch
    // -----------------------------------------------------------------------

    fn run_primitive(
        &mut self,
        primitive: PrimitiveKind,
        state: &mut ThoughtState,
    ) -> ThinkingResult<ThoughtStep> {
        match primitive {
            PrimitiveKind::SpreadingActivation => self.step_spreading(state),
            PrimitiveKind::EgraphSaturate => self.step_egraph(state),
            PrimitiveKind::MctsRollout => self.step_mcts(state),
            PrimitiveKind::Argumentation => self.step_argumentation(state),
            PrimitiveKind::Counterfactual => self.step_counterfactual(state),
            PrimitiveKind::EvidenceCombine => self.step_evidence(state),
            PrimitiveKind::SuperpositionMerge => self.step_merge(state),
        }
    }

    // -- Query-structure helpers -------------------------------------------

    fn goal_object(&self, state: &ThoughtState, predicate: SymbolId) -> Option<SymbolId> {
        self.engine
            .triples_from(state.origin.goal)
            .into_iter()
            .find(|t| t.predicate == predicate)
            .map(|t| t.object)
    }

    fn query_subject(&self, state: &ThoughtState) -> SymbolId {
        self.goal_object(state, self.preds.seeks)
            .unwrap_or(state.origin.goal)
    }

    /// Does `candidate` qualify as an answer for this goal?
    ///
    /// With an `answer-type` constraint: must be `is_a` that type.
    /// Without: any Entity symbol qualifies (relations never do).
    fn is_answer_candidate(&self, state: &ThoughtState, candidate: SymbolId) -> bool {
        if state.excluded.contains(&candidate) {
            return false;
        }
        match self.goal_object(state, self.preds.answer_type) {
            Some(ty) => self.engine.has_triple(candidate, self.preds.is_a, ty),
            None => self
                .engine
                .get_symbol_meta(candidate)
                .map(|m| matches!(m.kind, SymbolKind::Entity))
                .unwrap_or(false),
        }
    }

    /// Insert or strengthen a hypothesis for `symbol`.
    ///
    /// Idempotent policy: the stored mass is the max-belief of old and new —
    /// repeating the same primitive on unchanged knowledge must not inflate
    /// belief (only `EvidenceCombine` performs real Dempster fusion).
    fn upsert_hypothesis(
        &self,
        state: &mut ThoughtState,
        symbol: SymbolId,
        mass: MassFunction,
        sources: &[SymbolId],
    ) {
        let vec = self
            .engine
            .item_memory()
            .get_or_create(self.engine.ops(), symbol);
        if let Some(h) = state
            .hypotheses
            .iter_mut()
            .find(|h| h.symbol == Some(symbol))
        {
            if mass.m_true > h.mass.m_true {
                h.mass = mass;
            }
            for &s in sources {
                if !h.sources.contains(&s) {
                    h.sources.push(s);
                }
            }
        } else {
            state.hypotheses.push(Hypothesis {
                symbol: Some(symbol),
                vec,
                mass,
                sources: sources.to_vec(),
            });
        }
    }

    /// Fold freshly activated concept vectors into the focus vector
    /// (majority-vote bundle) and record the similarity delta.
    fn update_focus(&self, state: &mut ThoughtState, new_vecs: &[HyperVec]) {
        let ops = self.engine.ops();
        let delta = if new_vecs.is_empty() {
            0.0
        } else {
            let mut refs: Vec<&HyperVec> = vec![&state.focus_vec];
            refs.extend(new_vecs.iter());
            match ops.bundle(&refs) {
                Ok(next) => {
                    let sim = ops.similarity(&state.focus_vec, &next).unwrap_or(1.0);
                    state.focus_vec = next;
                    (1.0 - sim).max(0.0)
                }
                Err(_) => 0.0,
            }
        };
        state.focus_deltas.push_back(delta);
        while state.focus_deltas.len() > self.config.focus_window {
            state.focus_deltas.pop_front();
        }
    }

    fn causal_manager(&mut self) -> Option<&CausalManager> {
        if self.causal.is_none() {
            self.causal = Some(CausalManager::new(self.engine).ok());
        }
        self.causal.as_ref().and_then(|c| c.as_ref())
    }

    // -- Primitives ---------------------------------------------------------

    /// One frontier ring of spreading activation (`Engine::infer`, depth 1),
    /// or the one-shot VSA analogy recall for analogy-shaped goals.
    fn step_spreading(&mut self, state: &mut ThoughtState) -> ThinkingResult<ThoughtStep> {
        // Analogy-shaped goal: A : B :: C : ? via existing VSA arithmetic.
        if !state.analogy_done {
            let a = self.goal_object(state, self.preds.analogy_a);
            let b = self.goal_object(state, self.preds.analogy_b);
            let c = self.goal_object(state, self.preds.analogy_c);
            if let (Some(a), Some(b), Some(c)) = (a, b, c) {
                state.analogy_done = true;
                let candidates = self
                    .engine
                    .infer_analogy(a, b, c, self.config.analogy_top_k)?;
                let mut new_concepts = Vec::new();
                let mut new_vecs = Vec::new();
                for (sym, sim) in &candidates {
                    // The analogy terms, the goal, and other query metadata
                    // are never answers.
                    if state.excluded.contains(sym) {
                        continue;
                    }
                    let conf = (sim.clamp(0.0, 1.0)) * 0.85;
                    self.upsert_hypothesis(state, *sym, MassFunction::from_confidence(conf), &[]);
                    new_concepts.push(*sym);
                    new_vecs.push(
                        self.engine
                            .item_memory()
                            .get_or_create(self.engine.ops(), *sym),
                    );
                }
                new_vecs.truncate(3);
                self.update_focus(state, &new_vecs);
                let novelty = if candidates.is_empty() { 0.0 } else { 1.0 };
                return Ok(ThoughtStep::SpreadingActivation {
                    frontier_size: 1,
                    new_concepts,
                    novelty,
                });
            }
        }

        // Regular frontier expansion: up to beam_width unexpanded symbols.
        let mut batch = Vec::with_capacity(self.config.beam_width);
        while batch.len() < self.config.beam_width {
            let Some(sym) = state.frontier.pop_front() else {
                break;
            };
            if state.expanded.contains(&sym) {
                continue;
            }
            batch.push(sym);
        }

        if batch.is_empty() {
            self.update_focus(state, &[]);
            return Ok(ThoughtStep::SpreadingActivation {
                frontier_size: 0,
                new_concepts: Vec::new(),
                novelty: 0.0,
            });
        }

        let predicate_filter = self
            .goal_object(state, self.preds.relation)
            .map(|r| vec![r]);

        let mut discovered: Vec<(SymbolId, f32)> = Vec::new();
        let mut seen = 0usize;
        for &seed in &batch {
            state.expanded.insert(seed);
            let parent_act = state.activation.get(&seed).copied().unwrap_or(1.0);

            let mut query = InferenceQuery::default()
                .with_seeds(vec![seed])
                .with_max_depth(1)
                .with_min_confidence(self.config.min_confidence);
            query.top_k = self.config.activation_top_k;
            if let Some(filter) = predicate_filter.clone() {
                query = query.with_predicate_filter(filter);
            }

            let result = self.engine.infer(&query)?;
            for (sym, conf) in result.activations {
                if sym == seed {
                    continue;
                }
                let act = parent_act * conf;
                if act < self.config.min_confidence {
                    continue;
                }
                match state.activation.get(&sym) {
                    Some(&prev) if prev >= act => {
                        seen += 1;
                    }
                    _ => {
                        state.activation.insert(sym, act);
                        discovered.push((sym, act));
                    }
                }
            }
        }

        // Deterministic frontier order: strongest activation first, symbol id
        // as tie-break (engine map iteration order is not deterministic).
        discovered.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        discovered.dedup_by_key(|(s, _)| *s);

        let mut new_concepts = Vec::with_capacity(discovered.len());
        let mut new_vecs = Vec::new();
        for &(sym, act) in &discovered {
            state.frontier.push_back(sym);
            new_concepts.push(sym);
            if self.is_answer_candidate(state, sym) {
                self.upsert_hypothesis(
                    state,
                    sym,
                    MassFunction::from_confidence(act * 0.9),
                    &[],
                );
            }
            if new_vecs.len() < 4 {
                new_vecs.push(
                    self.engine
                        .item_memory()
                        .get_or_create(self.engine.ops(), sym),
                );
            }
        }

        let novelty = if new_concepts.is_empty() && seen == 0 {
            0.0
        } else {
            new_concepts.len() as f32 / (new_concepts.len() + seen) as f32
        };
        self.update_focus(state, &new_vecs);

        Ok(ThoughtStep::SpreadingActivation {
            frontier_size: batch.len() as u32,
            new_concepts,
            novelty,
        })
    }

    /// One equality-saturation pass when the goal label parses as an
    /// expression; a recorded no-op otherwise.
    fn step_egraph(&mut self, state: &mut ThoughtState) -> ThinkingResult<ThoughtStep> {
        let label = self.engine.resolve_label(state.origin.goal);
        if !label.starts_with('(') {
            return Ok(ThoughtStep::EgraphSaturate {
                iterations: 0,
                eclasses_added: 0,
                best_cost_delta: 0.0,
            });
        }
        let Ok(parsed) = label.parse::<egg::RecExpr<crate::reason::AkhLang>>() else {
            return Ok(ThoughtStep::EgraphSaturate {
                iterations: 0,
                eclasses_added: 0,
                best_cost_delta: 0.0,
            });
        };

        let rules = self.engine.all_rules();
        let initial_size = parsed.as_ref().len() as f32;
        let runner = Runner::default().with_expr(&parsed).run(&rules);
        let extractor = Extractor::new(&runner.egraph, AstSize);
        let (best_cost, best) = extractor.find_best(runner.roots[0]);

        // A simplified form that names a known symbol becomes a hypothesis.
        if let Ok(sym) = self.engine.lookup_symbol(&best.to_string()) {
            self.upsert_hypothesis(state, sym, MassFunction::from_confidence(0.7), &[]);
        }

        Ok(ThoughtStep::EgraphSaturate {
            iterations: runner.iterations.len() as u32,
            eclasses_added: runner.egraph.number_of_classes() as u32,
            best_cost_delta: initial_size - best_cost as f32,
        })
    }

    /// A tiny MCTS probe (Phase 16a planner, a few iterations).
    fn step_mcts(&mut self, state: &mut ThoughtState) -> ThinkingResult<ThoughtStep> {
        let goal = state.origin.goal;
        let top_belief = state.top_hypothesis().map(|h| h.belief()).unwrap_or(0.0);
        let engine = self.engine;
        let fluent_count = engine.knowledge_graph().triple_count();
        let mcts_iterations = self.config.mcts_iterations;
        let value_fn = self.value_fn.clone();

        let Some(causal) = self.causal_manager() else {
            return Ok(ThoughtStep::MctsRollout {
                trajectory_len: 0,
                return_value: 0.0,
                action: None,
            });
        };

        let root = AgentState {
            goal_states: vec![(goal, top_belief)],
            fluent_count,
            state_vec: None,
            timestamp: state.origin.started_at,
        };
        let planner = MctsPlanner::new(MctsConfig {
            max_iterations: mcts_iterations,
            max_depth: 3,
            rollout_depth: 2,
            ..Default::default()
        });
        let result = planner.plan(root, causal, &value_fn, engine, None);

        let action = result.best_plan.first().copied();
        if let Some(act) = action
            && self.is_answer_candidate(state, act)
        {
            let conf = 0.5 + 0.3 * result.expected_value.tanh();
            self.upsert_hypothesis(state, act, MassFunction::from_confidence(conf), &[]);
        }

        Ok(ThoughtStep::MctsRollout {
            trajectory_len: result.best_plan.len() as u32,
            return_value: result.expected_value,
            action,
        })
    }

    /// One pro/con argumentation round on the query topic (Cyc meta-rules).
    fn step_argumentation(&mut self, state: &mut ThoughtState) -> ThinkingResult<ThoughtStep> {
        let subject = self.query_subject(state);
        let relation = match self.goal_object(state, self.preds.relation) {
            Some(r) => Some(r),
            // Without an explicit query relation, argue over the subject's
            // most common outgoing predicate (deterministic tie-break).
            None => {
                let mut counts: HashMap<SymbolId, usize> = HashMap::new();
                for t in self.engine.triples_from(subject) {
                    *counts.entry(t.predicate).or_default() += 1;
                }
                counts
                    .into_iter()
                    .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))
                    .map(|(p, _)| p)
            }
        };

        let Some(relation) = relation else {
            return Ok(ThoughtStep::Argumentation {
                topic: subject,
                answer: None,
                confidence: 0.0,
            });
        };

        let set = self.engine.argue(subject, relation)?;
        let answer = set.verdict.answer;
        let confidence = set.verdict.confidence as f32;

        if let Some(ans) = answer
            && self.is_answer_candidate(state, ans)
        {
            self.upsert_hypothesis(
                state,
                ans,
                MassFunction::from_confidence(confidence.clamp(0.0, 1.0) * 0.9),
                &[],
            );
        }

        Ok(ThoughtStep::Argumentation {
            topic: subject,
            answer,
            confidence,
        })
    }

    /// One counterfactual fork over the causal world model (Phase 15c), when
    /// at least two applicable actions exist; a recorded no-op otherwise.
    fn step_counterfactual(&mut self, state: &mut ThoughtState) -> ThinkingResult<ThoughtStep> {
        let started_at = state.origin.started_at;
        let goal = state.origin.goal;
        let intervention_pred = self.preds.intervention;
        let engine = self.engine;

        let Some(causal) = self.causal_manager() else {
            return Ok(ThoughtStep::Counterfactual {
                intervention: None,
                divergence: 0.0,
            });
        };

        let actions = causal.applicable_actions(engine);
        if actions.len() < 2 {
            return Ok(ThoughtStep::Counterfactual {
                intervention: None,
                divergence: 0.0,
            });
        }
        let actual = actions[0].action_id;
        let hypothetical = actions[1].action_id;

        let query = CounterfactualQuery {
            actual_action: actual,
            hypothetical_action: hypothetical,
            timestamp: started_at,
        };
        let divergence = match counterfactual_query(&query, causal, engine) {
            Ok(result) => result.divergent_fluents.len() as f32,
            Err(_) => 0.0,
        };

        Ok(ThoughtStep::Counterfactual {
            intervention: Some(TripleRef {
                subject: goal,
                predicate: intervention_pred,
                object: hypothetical,
            }),
            divergence,
        })
    }

    /// One Dempster-Shafer fusion pass (Phase 17) over the hypotheses'
    /// knowledge-graph evidence.
    fn step_evidence(&mut self, state: &mut ThoughtState) -> ThinkingResult<ThoughtStep> {
        let subject = self.query_subject(state);
        let relation = self.goal_object(state, self.preds.relation);

        // Candidate evidence from the KG: (subject, relation, o) triples.
        let mut kg_evidence: HashMap<SymbolId, f32> = HashMap::new();
        if let Some(rel) = relation {
            for t in self.engine.triples_from(subject) {
                if t.predicate == rel {
                    let e = kg_evidence.entry(t.object).or_insert(0.0);
                    *e = e.max(t.confidence);
                }
            }
            // Every KG candidate that qualifies becomes a hypothesis.
            let mut candidates: Vec<(SymbolId, f32)> =
                kg_evidence.iter().map(|(&s, &c)| (s, c)).collect();
            candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            for (sym, conf) in candidates {
                if self.is_answer_candidate(state, sym) {
                    self.upsert_hypothesis(
                        state,
                        sym,
                        MassFunction::from_confidence(conf * 0.5),
                        &[],
                    );
                }
            }
        }

        let best_conf = kg_evidence.values().copied().fold(0.0f32, f32::max);

        // Fuse per hypothesis: direct support vs. strongest competitor.
        let mut updates: Vec<(usize, MassFunction, Vec<SymbolId>)> = Vec::new();
        for (i, h) in state.hypotheses.iter().enumerate() {
            let Some(sym) = h.symbol else { continue };
            if state.excluded.contains(&sym) {
                continue;
            }
            let kg_support = kg_evidence.get(&sym).copied();
            let support = kg_support
                .unwrap_or_else(|| state.activation.get(&sym).copied().unwrap_or(0.0));
            if support <= 0.0 {
                continue;
            }
            // Direct KG candidates compete with each other: the strongest
            // rival counts as attenuated counter-evidence. Hypotheses that
            // arrived via activation (multi-hop discoveries) are not direct
            // `(S, R, ?)` candidates and have no rivals in this frame.
            let against = match kg_support {
                Some(own) if own < best_conf => best_conf * 0.5,
                Some(_) => {
                    kg_evidence
                        .iter()
                        .filter(|&(&s, _)| s != sym)
                        .map(|(_, &c)| c)
                        .fold(0.0f32, f32::max)
                        * 0.5
                }
                None => 0.0,
            };
            let observed = MassFunction::new(support * 0.9, against.min(1.0 - support * 0.9));
            let fused = h.mass.combine(&observed).unwrap_or(observed);

            // Track declared assertion sources for the Phase 18 verifier.
            let sources: Vec<SymbolId> = self
                .engine
                .triples_from(sym)
                .into_iter()
                .filter(|t| t.predicate == self.preds.asserted_by)
                .map(|t| t.object)
                .collect();
            updates.push((i, fused, sources));
        }
        for (i, mass, sources) in updates {
            state.hypotheses[i].mass = mass;
            for s in sources {
                if !state.hypotheses[i].sources.contains(&s) {
                    state.hypotheses[i].sources.push(s);
                }
            }
        }

        // Conflict between the top two hypotheses.
        let mut ranked: Vec<&Hypothesis> = state.hypotheses.iter().collect();
        ranked.sort_by(|a, b| b.belief().total_cmp(&a.belief()));
        let conflict = if ranked.len() >= 2 {
            ranked[0].mass.conflict_with(&ranked[1].mass)
        } else {
            0.0
        };
        let narrowed_interval = ranked
            .first()
            .map(|h| h.interval())
            .unwrap_or_else(|| MassFunction::vacuous().belief_interval());

        Ok(ThoughtStep::EvidenceCombine {
            conflict,
            narrowed_interval,
        })
    }

    /// Superposition maintenance: merge near-duplicate hypotheses, prune
    /// low-belief ones, refresh the focus bundle.
    fn step_merge(&mut self, state: &mut ThoughtState) -> ThinkingResult<ThoughtStep> {
        let ops = self.engine.ops();
        let hypotheses_in = state.hypotheses.len() as u32;

        // Sort by belief so merges keep the stronger representative.
        state
            .hypotheses
            .sort_by(|a, b| b.belief().total_cmp(&a.belief()));

        let mut merged: Vec<Hypothesis> = Vec::with_capacity(state.hypotheses.len());
        let mut sims: Vec<f32> = Vec::new();
        for h in state.hypotheses.drain(..) {
            let mut absorbed = false;
            for kept in merged.iter_mut() {
                let same_symbol = kept.symbol.is_some() && kept.symbol == h.symbol;
                let sim = ops.similarity(&kept.vec, &h.vec).unwrap_or(0.0);
                sims.push(sim);
                if same_symbol || sim >= self.config.merge_similarity {
                    // Keep the higher-belief mass; union sources.
                    if h.mass.m_true > kept.mass.m_true {
                        kept.mass = h.mass;
                    }
                    for s in &h.sources {
                        if !kept.sources.contains(s) {
                            kept.sources.push(*s);
                        }
                    }
                    absorbed = true;
                    break;
                }
            }
            if !absorbed {
                merged.push(h);
            }
        }

        // Prune: capacity first, then the belief floor (never drop the last).
        merged.truncate(self.config.hypothesis_capacity);
        if merged.len() > 1 {
            let floor = self.config.prune_belief_floor;
            let strongest = merged[0].belief();
            merged.retain(|h| h.belief() >= floor || h.belief() >= strongest);
        }
        state.hypotheses = merged;

        // Refresh focus: goal vector + top hypothesis vectors.
        let goal_vec = self
            .engine
            .item_memory()
            .get_or_create(ops, state.origin.goal);
        let mut new_vecs: Vec<HyperVec> = vec![goal_vec];
        new_vecs.extend(state.hypotheses.iter().take(3).map(|h| h.vec.clone()));
        self.update_focus(state, &new_vecs);

        let mean_similarity = if sims.is_empty() {
            1.0
        } else {
            sims.iter().sum::<f32>() / sims.len() as f32
        };

        Ok(ThoughtStep::SuperpositionMerge {
            hypotheses_in,
            hypotheses_out: state.hypotheses.len() as u32,
            mean_similarity,
        })
    }
}
