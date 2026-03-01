//! Competitive reasoner dispatch: bid-based strategy selection.
//!
//! Inspired by Cyc's 1,100+ Heuristic Level modules, each of which "bids"
//! on sub-problems and the cheapest applicable one wins. Resource budgets
//! enforce time limits; slow reasoners get interrupted.
//!
//! # Architecture
//!
//! - [`Reasoner`] trait: `can_handle(&Problem) -> Option<Bid>`, `solve(&Problem, Duration) -> DispatchResult`
//! - [`ReasonerRegistry`]: holds registered reasoners, sorts bids, executes cheapest, falls back on failure
//! - Built-in wrappers: `SpreadingActivationReasoner`, `BackwardChainingReasoner`,
//!   `SuperpositionReasoner`, `EGraphReasoner`
//! - Specialized reasoners: `TransitiveClosureReasoner`, `TypeHierarchyReasoner`,
//!   `PredicateHierarchyReasoner`

use std::collections::{HashSet, VecDeque};
use std::fmt;
use std::time::{Duration, Instant};

use miette::Diagnostic;
use thiserror::Error;

use crate::engine::Engine;
use crate::error::InferError;
use crate::graph::index::KnowledgeGraph;
use crate::infer::{InferenceQuery, InferenceResult};
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors specific to the dispatch subsystem.
#[derive(Debug, Error, Diagnostic)]
pub enum DispatchError {
    #[error("no reasoner can handle this problem: {problem_kind}")]
    #[diagnostic(
        code(akh::dispatch::no_reasoner),
        help(
            "None of the registered reasoners bid on this problem type. \
             Register a reasoner that handles '{problem_kind}', or reformulate \
             the query as a different problem kind."
        )
    )]
    NoReasonerAvailable { problem_kind: String },

    #[error("reasoner '{reasoner}' exceeded time budget of {budget_ms}ms")]
    #[diagnostic(
        code(akh::dispatch::timeout),
        help(
            "The reasoner took too long. Increase the time budget, \
             simplify the problem, or try a different reasoner."
        )
    )]
    Timeout { reasoner: String, budget_ms: u64 },

    #[error("all {tried} reasoner(s) failed for problem '{problem_kind}'")]
    #[diagnostic(
        code(akh::dispatch::all_failed),
        help(
            "Every reasoner that bid on this problem failed. \
             Check the inner errors for details, or simplify the query."
        )
    )]
    AllFailed {
        problem_kind: String,
        tried: usize,
    },

    #[error(transparent)]
    #[diagnostic(transparent)]
    Infer(#[from] InferError),

    #[error("engine error during dispatch: {0}")]
    #[diagnostic(
        code(akh::dispatch::engine),
        help("An engine-level error occurred during reasoner execution.")
    )]
    Engine(#[source] Box<crate::error::AkhError>),
}

/// Result type for dispatch operations.
pub type DispatchResult<T> = std::result::Result<T, DispatchError>;

// ---------------------------------------------------------------------------
// Problem description
// ---------------------------------------------------------------------------

/// A problem submitted to the dispatch system.
///
/// Different problem kinds route to different reasoners. The dispatcher
/// collects bids from all registered reasoners and runs the cheapest one.
#[derive(Debug, Clone)]
pub enum Problem {
    /// Forward inference: spread activation from seeds.
    ForwardInference(InferenceQuery),

    /// Backward chaining: find support for a goal symbol.
    BackwardChaining {
        goal: SymbolId,
        max_depth: usize,
        min_confidence: f32,
    },

    /// Superposition: multi-hypothesis reasoning from seeds.
    Superposition {
        seeds: Vec<SymbolId>,
        max_depth: usize,
    },

    /// E-graph simplification / verification.
    EGraphSimplify {
        expression: String,
    },

    /// Transitive closure query: is `subject` transitively related to `object`
    /// via `predicate`? (e.g., is Dog transitively is-a Animal?)
    TransitiveClosure {
        subject: SymbolId,
        predicate: SymbolId,
        object: SymbolId,
    },

    /// Type hierarchy query: is `instance` a (transitive) instance of `type_`?
    TypeCheck {
        instance: SymbolId,
        type_: SymbolId,
    },

    /// Predicate subsumption: does `specific` predicate specialize `general`?
    PredicateSubsumption {
        specific: SymbolId,
        general: SymbolId,
    },
}

impl Problem {
    /// Human-readable kind label for diagnostics.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::ForwardInference(_) => "forward-inference",
            Self::BackwardChaining { .. } => "backward-chaining",
            Self::Superposition { .. } => "superposition",
            Self::EGraphSimplify { .. } => "egraph-simplify",
            Self::TransitiveClosure { .. } => "transitive-closure",
            Self::TypeCheck { .. } => "type-check",
            Self::PredicateSubsumption { .. } => "predicate-subsumption",
        }
    }
}

// ---------------------------------------------------------------------------
// Bid
// ---------------------------------------------------------------------------

/// A reasoner's bid on a problem: estimated cost and confidence.
#[derive(Debug, Clone)]
pub struct Bid {
    /// Estimated wall-clock cost to solve the problem.
    pub estimated_cost: Duration,
    /// Reasoner's confidence that it can produce a useful answer (0.0–1.0).
    pub confidence: f64,
}

impl Bid {
    /// Create a new bid.
    pub fn new(estimated_cost: Duration, confidence: f64) -> Self {
        Self {
            estimated_cost,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Score used for sorting: lower is better.
    /// Combines cost (prefer fast) with confidence (prefer certain).
    /// score = cost_ms / (confidence + 0.01)
    fn score(&self) -> f64 {
        let cost_ms = self.estimated_cost.as_secs_f64() * 1000.0;
        cost_ms / (self.confidence + 0.01)
    }
}

// ---------------------------------------------------------------------------
// Reasoner outcome
// ---------------------------------------------------------------------------

/// The output of a successful reasoner solve.
#[derive(Debug)]
pub enum ReasonerOutput {
    /// Inference activations with provenance.
    Inference(InferenceResult),

    /// Boolean answer (e.g., transitive closure, type check).
    BooleanAnswer {
        answer: bool,
        /// The chain of intermediate symbols that supports the answer.
        chain: Vec<SymbolId>,
        confidence: f32,
    },

    /// Simplified expression from e-graph.
    Simplified {
        original: String,
        simplified: String,
        cost: usize,
    },
}

/// Metadata about a dispatch execution.
#[derive(Debug)]
pub struct DispatchTrace {
    /// Which reasoner solved the problem.
    pub reasoner_name: String,
    /// All bids received (reasoner name, bid).
    pub bids: Vec<(String, Bid)>,
    /// Wall-clock time spent.
    pub elapsed: Duration,
    /// Number of reasoners that were tried (including failures).
    pub attempts: usize,
}

// ---------------------------------------------------------------------------
// Reasoner trait
// ---------------------------------------------------------------------------

/// A reasoning strategy that can bid on and solve problems.
///
/// Reasoners are registered with [`ReasonerRegistry`] and compete via
/// [`Bid`]s on each problem. The cheapest applicable reasoner wins.
pub trait Reasoner: Send + Sync {
    /// Human-readable name for diagnostics and tracing.
    fn name(&self) -> &str;

    /// Inspect a problem and optionally bid on it.
    ///
    /// Return `None` if this reasoner cannot handle the problem.
    /// Return `Some(Bid)` with estimated cost and confidence if it can.
    fn can_handle(&self, problem: &Problem) -> Option<Bid>;

    /// Solve the problem within the given time budget.
    ///
    /// The reasoner should check elapsed time periodically and return
    /// `Err(DispatchError::Timeout)` if the budget is exceeded.
    fn solve(
        &self,
        problem: &Problem,
        engine: &Engine,
        budget: Duration,
    ) -> DispatchResult<ReasonerOutput>;
}

impl fmt::Debug for dyn Reasoner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Reasoner({})", self.name())
    }
}

// ---------------------------------------------------------------------------
// Registry + Dispatcher
// ---------------------------------------------------------------------------

/// Registry of reasoning strategies with bid-based dispatch.
pub struct ReasonerRegistry {
    reasoners: Vec<Box<dyn Reasoner>>,
    /// Default time budget for a single reasoner invocation.
    default_budget: Duration,
}

impl fmt::Debug for ReasonerRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&str> = self.reasoners.iter().map(|r| r.name()).collect();
        f.debug_struct("ReasonerRegistry")
            .field("reasoners", &names)
            .field("default_budget", &self.default_budget)
            .finish()
    }
}

impl ReasonerRegistry {
    /// Create an empty registry with the given default budget.
    pub fn new(default_budget: Duration) -> Self {
        Self {
            reasoners: Vec::new(),
            default_budget,
        }
    }

    /// Create a registry pre-populated with all built-in reasoners.
    pub fn with_builtins(default_budget: Duration) -> Self {
        let mut reg = Self::new(default_budget);
        reg.register(Box::new(TransitiveClosureReasoner));
        reg.register(Box::new(TypeHierarchyReasoner));
        reg.register(Box::new(PredicateHierarchyReasoner));
        reg.register(Box::new(SpreadingActivationReasoner));
        reg.register(Box::new(BackwardChainingReasoner));
        reg.register(Box::new(SuperpositionReasoner));
        reg.register(Box::new(EGraphReasoner));
        reg
    }

    /// Register a new reasoner.
    pub fn register(&mut self, reasoner: Box<dyn Reasoner>) {
        self.reasoners.push(reasoner);
    }

    /// Number of registered reasoners.
    pub fn len(&self) -> usize {
        self.reasoners.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.reasoners.is_empty()
    }

    /// List registered reasoner names.
    pub fn reasoner_names(&self) -> Vec<&str> {
        self.reasoners.iter().map(|r| r.name()).collect()
    }

    /// Dispatch a problem: collect bids, run cheapest, fall back on failure.
    ///
    /// Returns the output together with a trace of the dispatch process.
    pub fn dispatch(
        &self,
        problem: &Problem,
        engine: &Engine,
    ) -> DispatchResult<(ReasonerOutput, DispatchTrace)> {
        self.dispatch_with_budget(problem, engine, self.default_budget)
    }

    /// Dispatch with a specific time budget per reasoner.
    pub fn dispatch_with_budget(
        &self,
        problem: &Problem,
        engine: &Engine,
        budget: Duration,
    ) -> DispatchResult<(ReasonerOutput, DispatchTrace)> {
        // Collect bids
        let mut bids: Vec<(usize, Bid)> = Vec::new();
        for (idx, reasoner) in self.reasoners.iter().enumerate() {
            if let Some(bid) = reasoner.can_handle(problem) {
                bids.push((idx, bid));
            }
        }

        if bids.is_empty() {
            return Err(DispatchError::NoReasonerAvailable {
                problem_kind: problem.kind().to_string(),
            });
        }

        // Sort by score (lower is better)
        bids.sort_by(|a, b| a.1.score().partial_cmp(&b.1.score()).unwrap_or(std::cmp::Ordering::Equal));

        let all_bids: Vec<(String, Bid)> = bids
            .iter()
            .map(|(idx, bid)| (self.reasoners[*idx].name().to_string(), bid.clone()))
            .collect();

        // Try each reasoner in bid order
        let mut attempts = 0;
        for (idx, _bid) in &bids {
            attempts += 1;
            let reasoner = &self.reasoners[*idx];
            let start = Instant::now();

            match reasoner.solve(problem, engine, budget) {
                Ok(output) => {
                    return Ok((
                        output,
                        DispatchTrace {
                            reasoner_name: reasoner.name().to_string(),
                            bids: all_bids,
                            elapsed: start.elapsed(),
                            attempts,
                        },
                    ));
                }
                Err(_) => continue, // Try next reasoner
            }
        }

        Err(DispatchError::AllFailed {
            problem_kind: problem.kind().to_string(),
            tried: attempts,
        })
    }
}

// ===========================================================================
// Built-in reasoner wrappers
// ===========================================================================

// ---------------------------------------------------------------------------
// Spreading Activation
// ---------------------------------------------------------------------------

/// Wraps the existing `InferEngine` spreading-activation strategy.
pub struct SpreadingActivationReasoner;

impl Reasoner for SpreadingActivationReasoner {
    fn name(&self) -> &str {
        "spreading-activation"
    }

    fn can_handle(&self, problem: &Problem) -> Option<Bid> {
        match problem {
            Problem::ForwardInference(query) => {
                // Cost scales with seeds × depth
                let factor = (query.seeds.len() * query.max_depth).max(1) as f64;
                Some(Bid::new(
                    Duration::from_micros((factor * 500.0) as u64),
                    0.8,
                ))
            }
            _ => None,
        }
    }

    fn solve(
        &self,
        problem: &Problem,
        engine: &Engine,
        _budget: Duration,
    ) -> DispatchResult<ReasonerOutput> {
        let Problem::ForwardInference(query) = problem else {
            return Err(DispatchError::NoReasonerAvailable {
                problem_kind: problem.kind().to_string(),
            });
        };

        let result = engine.infer(query).map_err(|e| DispatchError::Engine(Box::new(e)))?;
        Ok(ReasonerOutput::Inference(result))
    }
}

// ---------------------------------------------------------------------------
// Backward Chaining
// ---------------------------------------------------------------------------

/// Wraps the existing `infer_backward` backward-chaining strategy.
pub struct BackwardChainingReasoner;

impl Reasoner for BackwardChainingReasoner {
    fn name(&self) -> &str {
        "backward-chaining"
    }

    fn can_handle(&self, problem: &Problem) -> Option<Bid> {
        match problem {
            Problem::BackwardChaining { max_depth, .. } => {
                // Backward chaining can be expensive at depth
                let factor = (*max_depth).max(1) as f64;
                Some(Bid::new(
                    Duration::from_micros((factor * 1000.0) as u64),
                    0.75,
                ))
            }
            _ => None,
        }
    }

    fn solve(
        &self,
        problem: &Problem,
        engine: &Engine,
        _budget: Duration,
    ) -> DispatchResult<ReasonerOutput> {
        let Problem::BackwardChaining {
            goal,
            max_depth,
            min_confidence,
        } = problem
        else {
            return Err(DispatchError::NoReasonerAvailable {
                problem_kind: problem.kind().to_string(),
            });
        };

        let config = crate::infer::backward::BackwardConfig {
            max_depth: *max_depth,
            min_confidence: *min_confidence,
            vsa_verify: true,
        };

        let chains =
            crate::infer::backward::infer_backward(engine, *goal, &config)
                .map_err(DispatchError::Infer)?;

        // Convert backward chains into InferenceResult format
        let mut activations: Vec<(SymbolId, f32)> = Vec::new();
        let mut provenance: Vec<ProvenanceRecord> = Vec::new();
        let mut seen = HashSet::new();

        for chain in &chains {
            for triple in &chain.supporting_triples {
                if seen.insert(triple.subject) {
                    activations.push((triple.subject, chain.confidence));
                    provenance.push(
                        ProvenanceRecord::new(
                            triple.subject,
                            DerivationKind::GraphEdge {
                                from: triple.object,
                                predicate: triple.predicate,
                            },
                        )
                        .with_confidence(chain.confidence),
                    );
                }
            }
        }

        Ok(ReasonerOutput::Inference(InferenceResult {
            activations,
            pattern: None,
            provenance,
        }))
    }
}

// ---------------------------------------------------------------------------
// Superposition
// ---------------------------------------------------------------------------

/// Wraps the existing `infer_with_superposition` strategy.
pub struct SuperpositionReasoner;

impl Reasoner for SuperpositionReasoner {
    fn name(&self) -> &str {
        "superposition"
    }

    fn can_handle(&self, problem: &Problem) -> Option<Bid> {
        match problem {
            Problem::Superposition { seeds, max_depth } => {
                // Superposition is expensive but handles ambiguity well
                let factor = (seeds.len() * *max_depth).max(1) as f64;
                Some(Bid::new(
                    Duration::from_micros((factor * 2000.0) as u64),
                    0.85,
                ))
            }
            _ => None,
        }
    }

    fn solve(
        &self,
        problem: &Problem,
        engine: &Engine,
        _budget: Duration,
    ) -> DispatchResult<ReasonerOutput> {
        let Problem::Superposition { seeds, max_depth } = problem else {
            return Err(DispatchError::NoReasonerAvailable {
                problem_kind: problem.kind().to_string(),
            });
        };

        let config = crate::infer::superposition::SuperpositionConfig {
            max_depth: *max_depth,
            ..Default::default()
        };

        let result = crate::infer::superposition::infer_with_superposition(seeds, engine, &config)
            .map_err(DispatchError::Infer)?;

        // Convert SuperpositionResult into InferenceResult
        let (activations, provenance) = match result.dominant {
            Some(hyp) => (hyp.activated, hyp.provenance),
            None => (Vec::new(), Vec::new()),
        };

        Ok(ReasonerOutput::Inference(InferenceResult {
            activations,
            pattern: None,
            provenance,
        }))
    }
}

// ---------------------------------------------------------------------------
// E-Graph
// ---------------------------------------------------------------------------

/// Wraps the `egg` e-graph rewrite/simplification engine.
pub struct EGraphReasoner;

impl Reasoner for EGraphReasoner {
    fn name(&self) -> &str {
        "egraph"
    }

    fn can_handle(&self, problem: &Problem) -> Option<Bid> {
        match problem {
            Problem::EGraphSimplify { .. } => {
                Some(Bid::new(Duration::from_micros(200), 0.9))
            }
            _ => None,
        }
    }

    fn solve(
        &self,
        problem: &Problem,
        _engine: &Engine,
        _budget: Duration,
    ) -> DispatchResult<ReasonerOutput> {
        let Problem::EGraphSimplify { expression } = problem else {
            return Err(DispatchError::NoReasonerAvailable {
                problem_kind: problem.kind().to_string(),
            });
        };

        use egg::{AstSize, Extractor, Runner};

        let expr: egg::RecExpr<crate::reason::AkhLang> = expression
            .parse()
            .map_err(|e: egg::RecExprParseError<_>| DispatchError::Infer(
                InferError::Reason(crate::error::ReasonError::ParseError {
                    message: e.to_string(),
                }),
            ))?;

        let rules = crate::reason::builtin_rules();
        let runner = Runner::default().with_expr(&expr).run(&rules);
        let extractor = Extractor::new(&runner.egraph, AstSize);
        let (cost, best) = extractor.find_best(runner.roots[0]);

        Ok(ReasonerOutput::Simplified {
            original: expression.clone(),
            simplified: best.to_string(),
            cost,
        })
    }
}

// ===========================================================================
// Specialized reasoners
// ===========================================================================

// ---------------------------------------------------------------------------
// Transitive Closure
// ---------------------------------------------------------------------------

/// Fast transitive closure query for any predicate.
///
/// Checks whether `subject` is transitively related to `object` via
/// `predicate` by doing a BFS over the knowledge graph. Caching is left
/// to higher layers (the engine can cache frequent queries).
pub struct TransitiveClosureReasoner;

impl TransitiveClosureReasoner {
    /// BFS from `start` following edges with the given predicate.
    /// Returns `Some(chain)` if `target` is reachable, `None` otherwise.
    fn bfs_chain(
        kg: &KnowledgeGraph,
        start: SymbolId,
        predicate: SymbolId,
        target: SymbolId,
        max_depth: usize,
    ) -> Option<Vec<SymbolId>> {
        let mut visited = HashSet::new();
        let mut queue: VecDeque<(SymbolId, Vec<SymbolId>)> = VecDeque::new();
        queue.push_back((start, vec![start]));
        visited.insert(start);

        while let Some((current, path)) = queue.pop_front() {
            if path.len() > max_depth + 1 {
                continue;
            }
            if current == target && path.len() > 1 {
                return Some(path);
            }

            let triples = kg.triples_from(current);
            for triple in &triples {
                if triple.predicate == predicate && !visited.contains(&triple.object) {
                    visited.insert(triple.object);
                    let mut new_path = path.clone();
                    new_path.push(triple.object);
                    if triple.object == target {
                        return Some(new_path);
                    }
                    queue.push_back((triple.object, new_path));
                }
            }
        }

        None
    }
}

impl Reasoner for TransitiveClosureReasoner {
    fn name(&self) -> &str {
        "transitive-closure"
    }

    fn can_handle(&self, problem: &Problem) -> Option<Bid> {
        match problem {
            Problem::TransitiveClosure { .. } => {
                // Very fast for cached/small graphs
                Some(Bid::new(Duration::from_micros(50), 0.95))
            }
            Problem::TypeCheck { .. } => {
                // Can also handle type checks via is-a transitive closure
                Some(Bid::new(Duration::from_micros(100), 0.85))
            }
            _ => None,
        }
    }

    fn solve(
        &self,
        problem: &Problem,
        engine: &Engine,
        _budget: Duration,
    ) -> DispatchResult<ReasonerOutput> {
        match problem {
            Problem::TransitiveClosure {
                subject,
                predicate,
                object,
            } => {
                let kg = engine.knowledge_graph();
                match Self::bfs_chain(kg, *subject, *predicate, *object, 50) {
                    Some(chain) => {
                        let confidence = 1.0 / chain.len() as f32; // Decay with distance
                        Ok(ReasonerOutput::BooleanAnswer {
                            answer: true,
                            chain,
                            confidence: confidence.max(0.1),
                        })
                    }
                    None => Ok(ReasonerOutput::BooleanAnswer {
                        answer: false,
                        chain: Vec::new(),
                        confidence: 1.0,
                    }),
                }
            }
            Problem::TypeCheck { instance, type_ } => {
                // Use is-a predicate for type checking
                let is_a = engine.resolve_or_create_relation("is-a")
                    .map_err(|e| DispatchError::Engine(Box::new(e)))?;

                let kg = engine.knowledge_graph();
                match Self::bfs_chain(kg, *instance, is_a, *type_, 50) {
                    Some(chain) => {
                        let confidence = 1.0 / chain.len() as f32;
                        Ok(ReasonerOutput::BooleanAnswer {
                            answer: true,
                            chain,
                            confidence: confidence.max(0.1),
                        })
                    }
                    None => Ok(ReasonerOutput::BooleanAnswer {
                        answer: false,
                        chain: Vec::new(),
                        confidence: 1.0,
                    }),
                }
            }
            _ => Err(DispatchError::NoReasonerAvailable {
                problem_kind: problem.kind().to_string(),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Type Hierarchy
// ---------------------------------------------------------------------------

/// Fast type-checking via cached ancestry chains.
///
/// Specifically handles `TypeCheck` problems using `is-a` transitivity.
/// Compared to `TransitiveClosureReasoner`, this builds a full ancestor
/// set for the instance and checks membership — better for repeated
/// queries against the same instance.
pub struct TypeHierarchyReasoner;

impl TypeHierarchyReasoner {
    /// Collect all transitive ancestors of `start` via `predicate`.
    fn ancestors(
        kg: &KnowledgeGraph,
        start: SymbolId,
        predicate: SymbolId,
    ) -> Vec<SymbolId> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited.insert(start);

        while let Some(current) = queue.pop_front() {
            let triples = kg.triples_from(current);
            for triple in &triples {
                if triple.predicate == predicate && !visited.contains(&triple.object) {
                    visited.insert(triple.object);
                    result.push(triple.object);
                    queue.push_back(triple.object);
                }
            }
        }

        result
    }
}

impl Reasoner for TypeHierarchyReasoner {
    fn name(&self) -> &str {
        "type-hierarchy"
    }

    fn can_handle(&self, problem: &Problem) -> Option<Bid> {
        match problem {
            Problem::TypeCheck { .. } => {
                // Specialized for type checking — bids aggressively
                Some(Bid::new(Duration::from_micros(30), 0.95))
            }
            _ => None,
        }
    }

    fn solve(
        &self,
        problem: &Problem,
        engine: &Engine,
        _budget: Duration,
    ) -> DispatchResult<ReasonerOutput> {
        let Problem::TypeCheck { instance, type_ } = problem else {
            return Err(DispatchError::NoReasonerAvailable {
                problem_kind: problem.kind().to_string(),
            });
        };

        let is_a = engine.resolve_or_create_relation("is-a")
            .map_err(|e| DispatchError::Infer(InferError::Graph(
                crate::error::GraphError::Analytics {
                    message: format!("failed to resolve is-a: {e}"),
                },
            )))?;

        let kg = engine.knowledge_graph();
        let ancestors = Self::ancestors(kg, *instance, is_a);

        if ancestors.contains(type_) {
            // Build a minimal chain
            let chain = TransitiveClosureReasoner::bfs_chain(kg, *instance, is_a, *type_, 50)
                .unwrap_or_else(|| vec![*instance, *type_]);

            let confidence = 1.0 / chain.len() as f32;
            Ok(ReasonerOutput::BooleanAnswer {
                answer: true,
                chain,
                confidence: confidence.max(0.1),
            })
        } else {
            Ok(ReasonerOutput::BooleanAnswer {
                answer: false,
                chain: Vec::new(),
                confidence: 1.0,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Predicate Hierarchy
// ---------------------------------------------------------------------------

/// Fast predicate subsumption check using the Phase 9b predicate hierarchy.
///
/// Checks whether `specific` is a specialization of `general` via
/// `rel:generalizes` chains.
pub struct PredicateHierarchyReasoner;

impl Reasoner for PredicateHierarchyReasoner {
    fn name(&self) -> &str {
        "predicate-hierarchy"
    }

    fn can_handle(&self, problem: &Problem) -> Option<Bid> {
        match problem {
            Problem::PredicateSubsumption { .. } => {
                Some(Bid::new(Duration::from_micros(40), 0.95))
            }
            _ => None,
        }
    }

    fn solve(
        &self,
        problem: &Problem,
        engine: &Engine,
        _budget: Duration,
    ) -> DispatchResult<ReasonerOutput> {
        let Problem::PredicateSubsumption { specific, general } = problem else {
            return Err(DispatchError::NoReasonerAvailable {
                problem_kind: problem.kind().to_string(),
            });
        };

        let rel_generalizes = engine.resolve_or_create_relation("rel:generalizes")
            .map_err(|e| DispatchError::Engine(Box::new(e)))?;

        let kg = engine.knowledge_graph();

        // BFS from specific → general via rel:generalizes
        match TransitiveClosureReasoner::bfs_chain(kg, *specific, rel_generalizes, *general, 20) {
            Some(chain) => {
                let confidence = 1.0 / chain.len() as f32;
                Ok(ReasonerOutput::BooleanAnswer {
                    answer: true,
                    chain,
                    confidence: confidence.max(0.1),
                })
            }
            None => Ok(ReasonerOutput::BooleanAnswer {
                answer: false,
                chain: Vec::new(),
                confidence: 1.0,
            }),
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::graph::Triple;
    use crate::symbol::SymbolKind;
    use crate::vsa::Dimension;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    fn sym(engine: &Engine, kind: SymbolKind, label: &str) -> SymbolId {
        engine.create_symbol(kind, label).unwrap().id
    }

    // --- Registry tests ---

    #[test]
    fn registry_with_builtins_has_all_reasoners() {
        let reg = ReasonerRegistry::with_builtins(Duration::from_secs(5));
        let names = reg.reasoner_names();
        assert!(names.contains(&"spreading-activation"));
        assert!(names.contains(&"backward-chaining"));
        assert!(names.contains(&"superposition"));
        assert!(names.contains(&"egraph"));
        assert!(names.contains(&"transitive-closure"));
        assert!(names.contains(&"type-hierarchy"));
        assert!(names.contains(&"predicate-hierarchy"));
        assert_eq!(reg.len(), 7);
    }

    #[test]
    fn no_reasoner_for_unhandled_problem() {
        let reg = ReasonerRegistry::new(Duration::from_secs(1));
        // Empty registry → no one can handle anything
        let engine = test_engine();
        let problem = Problem::EGraphSimplify {
            expression: "(not (not x))".to_string(),
        };
        let result = reg.dispatch(&problem, &engine);
        assert!(matches!(result, Err(DispatchError::NoReasonerAvailable { .. })));
    }

    // --- Bid scoring ---

    #[test]
    fn bid_score_prefers_fast_and_confident() {
        let fast_confident = Bid::new(Duration::from_micros(10), 0.95);
        let slow_uncertain = Bid::new(Duration::from_micros(5000), 0.3);
        assert!(
            fast_confident.score() < slow_uncertain.score(),
            "fast+confident ({}) should score lower than slow+uncertain ({})",
            fast_confident.score(),
            slow_uncertain.score(),
        );
    }

    // --- E-Graph reasoner ---

    #[test]
    fn egraph_simplifies_double_negation() {
        let engine = test_engine();
        let reg = ReasonerRegistry::with_builtins(Duration::from_secs(5));
        let problem = Problem::EGraphSimplify {
            expression: "(not (not x))".to_string(),
        };
        let (output, trace) = reg.dispatch(&problem, &engine).unwrap();
        assert_eq!(trace.reasoner_name, "egraph");
        match output {
            ReasonerOutput::Simplified { simplified, cost, .. } => {
                assert_eq!(simplified, "x");
                assert_eq!(cost, 1);
            }
            other => panic!("expected Simplified, got {other:?}"),
        }
    }

    #[test]
    fn egraph_simplifies_bind_self_inverse() {
        let engine = test_engine();
        let reg = ReasonerRegistry::with_builtins(Duration::from_secs(5));
        let problem = Problem::EGraphSimplify {
            expression: "(bind a (bind a b))".to_string(),
        };
        let (output, _) = reg.dispatch(&problem, &engine).unwrap();
        match output {
            ReasonerOutput::Simplified { simplified, .. } => {
                assert_eq!(simplified, "b");
            }
            other => panic!("expected Simplified, got {other:?}"),
        }
    }

    // --- Forward inference ---

    #[test]
    fn forward_inference_dispatches_to_spreading_activation() {
        let engine = test_engine();
        let sun = sym(&engine, SymbolKind::Entity, "Sun");
        let is_a = sym(&engine, SymbolKind::Relation, "is-a");
        let star = sym(&engine, SymbolKind::Entity, "Star");

        engine.add_triple(&Triple::new(sun, is_a, star)).unwrap();

        let reg = ReasonerRegistry::with_builtins(Duration::from_secs(5));
        let problem = Problem::ForwardInference(
            InferenceQuery::default().with_seeds(vec![sun]).with_max_depth(1),
        );
        let (output, trace) = reg.dispatch(&problem, &engine).unwrap();
        assert_eq!(trace.reasoner_name, "spreading-activation");

        match output {
            ReasonerOutput::Inference(result) => {
                let syms: Vec<SymbolId> = result.activations.iter().map(|(s, _)| *s).collect();
                assert!(syms.contains(&star), "Should find Star via spreading activation");
            }
            other => panic!("expected Inference, got {other:?}"),
        }
    }

    // --- Transitive closure ---

    #[test]
    fn transitive_closure_finds_chain() {
        let engine = test_engine();
        let dog = sym(&engine, SymbolKind::Entity, "Dog");
        let mammal = sym(&engine, SymbolKind::Entity, "Mammal");
        let animal = sym(&engine, SymbolKind::Entity, "Animal");
        let is_a = sym(&engine, SymbolKind::Relation, "is-a");

        engine.add_triple(&Triple::new(dog, is_a, mammal)).unwrap();
        engine.add_triple(&Triple::new(mammal, is_a, animal)).unwrap();

        let reg = ReasonerRegistry::with_builtins(Duration::from_secs(5));
        let problem = Problem::TransitiveClosure {
            subject: dog,
            predicate: is_a,
            object: animal,
        };
        let (output, trace) = reg.dispatch(&problem, &engine).unwrap();
        assert_eq!(trace.reasoner_name, "transitive-closure");

        match output {
            ReasonerOutput::BooleanAnswer { answer, chain, .. } => {
                assert!(answer, "Dog should be transitively is-a Animal");
                assert_eq!(chain.len(), 3, "Chain should be [Dog, Mammal, Animal]");
                assert_eq!(chain[0], dog);
                assert_eq!(chain[1], mammal);
                assert_eq!(chain[2], animal);
            }
            other => panic!("expected BooleanAnswer, got {other:?}"),
        }
    }

    #[test]
    fn transitive_closure_returns_false_for_unrelated() {
        let engine = test_engine();
        let dog = sym(&engine, SymbolKind::Entity, "Dog");
        let plant = sym(&engine, SymbolKind::Entity, "Plant");
        let is_a = sym(&engine, SymbolKind::Relation, "is-a");

        let reg = ReasonerRegistry::with_builtins(Duration::from_secs(5));
        let problem = Problem::TransitiveClosure {
            subject: dog,
            predicate: is_a,
            object: plant,
        };
        let (output, _) = reg.dispatch(&problem, &engine).unwrap();

        match output {
            ReasonerOutput::BooleanAnswer { answer, chain, .. } => {
                assert!(!answer, "Dog should NOT be transitively is-a Plant");
                assert!(chain.is_empty());
            }
            other => panic!("expected BooleanAnswer, got {other:?}"),
        }
    }

    // --- Type hierarchy ---

    #[test]
    fn type_hierarchy_finds_transitive_type() {
        let engine = test_engine();
        let fido = sym(&engine, SymbolKind::Entity, "Fido");
        let dog = sym(&engine, SymbolKind::Entity, "Dog");
        let mammal = sym(&engine, SymbolKind::Entity, "Mammal");
        let is_a = sym(&engine, SymbolKind::Relation, "is-a");

        engine.add_triple(&Triple::new(fido, is_a, dog)).unwrap();
        engine.add_triple(&Triple::new(dog, is_a, mammal)).unwrap();

        let reg = ReasonerRegistry::with_builtins(Duration::from_secs(5));
        let problem = Problem::TypeCheck {
            instance: fido,
            type_: mammal,
        };
        let (output, trace) = reg.dispatch(&problem, &engine).unwrap();
        // TypeHierarchyReasoner bids lower cost → should win
        assert_eq!(trace.reasoner_name, "type-hierarchy");

        match output {
            ReasonerOutput::BooleanAnswer { answer, chain, .. } => {
                assert!(answer, "Fido should be a Mammal transitively");
                assert!(chain.len() >= 2, "Chain should have at least 2 nodes");
            }
            other => panic!("expected BooleanAnswer, got {other:?}"),
        }
    }

    // --- Predicate hierarchy ---

    #[test]
    fn predicate_hierarchy_finds_subsumption() {
        let engine = test_engine();
        let bio_mother = sym(&engine, SymbolKind::Relation, "biologicalMother");
        let parent = sym(&engine, SymbolKind::Relation, "parent");
        let rel_gen = sym(&engine, SymbolKind::Relation, "rel:generalizes");

        // biologicalMother generalizes-to parent
        engine
            .add_triple(&Triple::new(bio_mother, rel_gen, parent))
            .unwrap();

        let reg = ReasonerRegistry::with_builtins(Duration::from_secs(5));
        let problem = Problem::PredicateSubsumption {
            specific: bio_mother,
            general: parent,
        };
        let (output, trace) = reg.dispatch(&problem, &engine).unwrap();
        assert_eq!(trace.reasoner_name, "predicate-hierarchy");

        match output {
            ReasonerOutput::BooleanAnswer { answer, chain, .. } => {
                assert!(answer, "biologicalMother should subsume parent");
                assert!(chain.len() >= 2);
            }
            other => panic!("expected BooleanAnswer, got {other:?}"),
        }
    }

    // --- Dispatch fallback ---

    #[test]
    fn dispatch_falls_back_on_failure() {
        // Create a custom reasoner that always fails
        struct FailReasoner;
        impl Reasoner for FailReasoner {
            fn name(&self) -> &str { "always-fail" }
            fn can_handle(&self, problem: &Problem) -> Option<Bid> {
                match problem {
                    Problem::EGraphSimplify { .. } => {
                        Some(Bid::new(Duration::from_micros(1), 1.0)) // Bids cheapest
                    }
                    _ => None,
                }
            }
            fn solve(&self, _: &Problem, _: &Engine, _: Duration) -> DispatchResult<ReasonerOutput> {
                Err(DispatchError::AllFailed {
                    problem_kind: "test".to_string(),
                    tried: 1,
                })
            }
        }

        let engine = test_engine();
        let mut reg = ReasonerRegistry::new(Duration::from_secs(5));
        reg.register(Box::new(FailReasoner));
        reg.register(Box::new(EGraphReasoner));

        let problem = Problem::EGraphSimplify {
            expression: "(not (not x))".to_string(),
        };
        let (output, trace) = reg.dispatch(&problem, &engine).unwrap();
        // FailReasoner bids cheaper but fails → falls back to EGraphReasoner
        assert_eq!(trace.reasoner_name, "egraph");
        assert_eq!(trace.attempts, 2);
        match output {
            ReasonerOutput::Simplified { simplified, .. } => assert_eq!(simplified, "x"),
            other => panic!("expected Simplified, got {other:?}"),
        }
    }

    // --- Backward chaining via dispatch ---

    #[test]
    fn backward_chaining_dispatches() {
        let engine = test_engine();
        let dog = sym(&engine, SymbolKind::Entity, "Dog");
        let mammal = sym(&engine, SymbolKind::Entity, "Mammal");
        let is_a = sym(&engine, SymbolKind::Relation, "is-a");

        engine.add_triple(&Triple::new(dog, is_a, mammal)).unwrap();

        let reg = ReasonerRegistry::with_builtins(Duration::from_secs(5));
        let problem = Problem::BackwardChaining {
            goal: mammal,
            max_depth: 3,
            min_confidence: 0.1,
        };
        let (output, trace) = reg.dispatch(&problem, &engine).unwrap();
        assert_eq!(trace.reasoner_name, "backward-chaining");

        match output {
            ReasonerOutput::Inference(result) => {
                assert!(
                    !result.activations.is_empty(),
                    "Backward chaining should find support for Mammal"
                );
            }
            other => panic!("expected Inference, got {other:?}"),
        }
    }

    // --- Trace includes all bids ---

    #[test]
    fn trace_contains_all_bids() {
        let engine = test_engine();
        let fido = sym(&engine, SymbolKind::Entity, "Fido");
        let mammal = sym(&engine, SymbolKind::Entity, "Mammal");
        let is_a = sym(&engine, SymbolKind::Relation, "is-a");
        engine.add_triple(&Triple::new(fido, is_a, mammal)).unwrap();

        let reg = ReasonerRegistry::with_builtins(Duration::from_secs(5));
        let problem = Problem::TypeCheck {
            instance: fido,
            type_: mammal,
        };
        let (_, trace) = reg.dispatch(&problem, &engine).unwrap();

        // TypeCheck should have bids from both type-hierarchy and transitive-closure
        let bid_names: Vec<&str> = trace.bids.iter().map(|(n, _)| n.as_str()).collect();
        assert!(bid_names.contains(&"type-hierarchy"), "type-hierarchy should bid");
        assert!(bid_names.contains(&"transitive-closure"), "transitive-closure should bid");
    }
}
