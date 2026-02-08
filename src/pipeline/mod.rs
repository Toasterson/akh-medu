//! Processing pipeline: linear multi-stage execution.
//!
//! Pipelines chain together processing stages (extract, infer, reason, retrieve)
//! in a linear sequence. Each stage consumes the output of the previous stage
//! and produces data for the next.

use std::sync::Arc;

use egg::{AstSize, Extractor, Rewrite, Runner};

use crate::error::PipelineError;
use crate::graph::index::KnowledgeGraph;
use crate::graph::traverse::{traverse_bfs, TraversalConfig, TraversalResult};
use crate::graph::Triple;
use crate::infer::engine::InferEngine;
use crate::infer::{InferenceQuery, InferenceResult};
use crate::reason::AkhLang;
use crate::symbol::SymbolId;
use crate::vsa::item_memory::ItemMemory;
use crate::vsa::ops::VsaOps;

/// Result type for pipeline operations.
pub type PipelineResult<T> = std::result::Result<T, PipelineError>;

// ---------------------------------------------------------------------------
// Data types flowing between stages
// ---------------------------------------------------------------------------

/// Data flowing through the pipeline between stages.
#[derive(Debug, Clone)]
pub enum PipelineData {
    /// A set of seed symbols.
    Seeds(Vec<SymbolId>),
    /// A set of triples.
    Triples(Vec<Triple>),
    /// Result of a graph traversal.
    Traversal(TraversalResult),
    /// Result of spreading-activation inference.
    Inference(InferenceResult),
    /// Result of e-graph reasoning.
    Reasoning(ReasoningResult),
}

impl PipelineData {
    /// Human-readable name for the data variant (for error messages).
    fn variant_name(&self) -> &'static str {
        match self {
            Self::Seeds(_) => "Seeds",
            Self::Triples(_) => "Triples",
            Self::Traversal(_) => "Traversal",
            Self::Inference(_) => "Inference",
            Self::Reasoning(_) => "Reasoning",
        }
    }
}

/// Result of e-graph reasoning.
#[derive(Debug, Clone)]
pub struct ReasoningResult {
    /// Simplified expression as a string.
    pub simplified_expr: String,
    /// Cost of the best expression.
    pub cost: usize,
    /// Whether the e-graph reached saturation.
    pub saturated: bool,
}

// ---------------------------------------------------------------------------
// Stage configuration
// ---------------------------------------------------------------------------

/// Configuration for a specific stage kind.
#[derive(Debug, Clone)]
pub enum StageConfig {
    /// Configuration for the Retrieve stage.
    Retrieve { traversal: TraversalConfig },
    /// Configuration for the Infer stage.
    Infer { query_template: InferenceQuery },
    /// Configuration for the Reason stage.
    Reason {
        max_iterations: usize,
        node_limit: usize,
    },
    /// Configuration for the ExtractTriples stage.
    ExtractTriples { min_confidence: f32 },
    /// Default (no extra configuration).
    Default,
}

/// Built-in pipeline stage types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageKind {
    /// Extract triples from input.
    ExtractTriples,
    /// Run VSA interference inference.
    Infer,
    /// Run e-graph reasoning.
    Reason,
    /// Retrieve from knowledge graph.
    Retrieve,
}

/// Named processing stage in a pipeline.
#[derive(Debug, Clone)]
pub struct PipelineStage {
    /// Stage name.
    pub name: String,
    /// Stage kind.
    pub kind: StageKind,
    /// Stage-specific configuration.
    pub config: StageConfig,
}

// ---------------------------------------------------------------------------
// Pipeline context (shared resources)
// ---------------------------------------------------------------------------

/// Shared resources available to all pipeline stages.
pub struct PipelineContext {
    pub ops: Arc<VsaOps>,
    pub item_memory: Arc<ItemMemory>,
    pub knowledge_graph: Arc<KnowledgeGraph>,
    pub rules: Vec<Rewrite<AkhLang, ()>>,
}

impl std::fmt::Debug for PipelineContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineContext")
            .field("rules_count", &self.rules.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Pipeline output
// ---------------------------------------------------------------------------

/// Output of a pipeline run.
#[derive(Debug, Clone)]
pub struct PipelineOutput {
    /// Final result data.
    pub result: PipelineData,
    /// Intermediate results from each stage: (stage_name, data).
    pub stage_results: Vec<(String, PipelineData)>,
    /// Number of stages that were executed.
    pub stages_executed: usize,
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// A linear processing pipeline.
#[derive(Debug, Clone)]
pub struct Pipeline {
    pub name: String,
    pub stages: Vec<PipelineStage>,
}

impl Pipeline {
    /// Run the pipeline with the given context and initial data.
    pub fn run(&self, ctx: &PipelineContext, initial: PipelineData) -> PipelineResult<PipelineOutput> {
        if self.stages.is_empty() {
            return Err(PipelineError::EmptyPipeline {
                name: self.name.clone(),
            });
        }

        let mut current = initial;
        let mut stage_results = Vec::with_capacity(self.stages.len());

        for (i, stage) in self.stages.iter().enumerate() {
            let output = execute_stage(ctx, &current, stage).map_err(|e| {
                PipelineError::StageFailure {
                    pipeline_name: self.name.clone(),
                    stage_name: stage.name.clone(),
                    stage_index: i,
                    source: Box::new(e),
                }
            })?;
            stage_results.push((stage.name.clone(), output.clone()));
            current = output;
        }

        Ok(PipelineOutput {
            result: current,
            stages_executed: self.stages.len(),
            stage_results,
        })
    }

    /// Built-in query pipeline: Retrieve → Infer → Reason.
    pub fn query_pipeline() -> Self {
        Self {
            name: "query".into(),
            stages: vec![
                PipelineStage {
                    name: "retrieve".into(),
                    kind: StageKind::Retrieve,
                    config: StageConfig::Retrieve {
                        traversal: TraversalConfig::default(),
                    },
                },
                PipelineStage {
                    name: "infer".into(),
                    kind: StageKind::Infer,
                    config: StageConfig::Infer {
                        query_template: InferenceQuery::default(),
                    },
                },
                PipelineStage {
                    name: "reason".into(),
                    kind: StageKind::Reason,
                    config: StageConfig::Reason {
                        max_iterations: 100,
                        node_limit: 10_000,
                    },
                },
            ],
        }
    }

    /// Built-in ingest pipeline: ExtractTriples (single stage).
    pub fn ingest_pipeline() -> Self {
        Self {
            name: "ingest".into(),
            stages: vec![PipelineStage {
                name: "extract_triples".into(),
                kind: StageKind::ExtractTriples,
                config: StageConfig::ExtractTriples {
                    min_confidence: 0.0,
                },
            }],
        }
    }
}

// ---------------------------------------------------------------------------
// Stage execution
// ---------------------------------------------------------------------------

fn execute_stage(
    ctx: &PipelineContext,
    input: &PipelineData,
    stage: &PipelineStage,
) -> PipelineResult<PipelineData> {
    match stage.kind {
        StageKind::Retrieve => execute_retrieve(ctx, input, &stage.config, &stage.name),
        StageKind::Infer => execute_infer(ctx, input, &stage.config, &stage.name),
        StageKind::Reason => execute_reason(ctx, input, &stage.config, &stage.name),
        StageKind::ExtractTriples => execute_extract_triples(input, &stage.config, &stage.name),
    }
}

/// Extract seeds from any pipeline data variant.
fn extract_seeds(data: &PipelineData) -> PipelineResult<Vec<SymbolId>> {
    match data {
        PipelineData::Seeds(seeds) => {
            if seeds.is_empty() {
                Err(PipelineError::NoSeeds)
            } else {
                Ok(seeds.clone())
            }
        }
        PipelineData::Triples(triples) => {
            let mut seeds: Vec<SymbolId> = triples
                .iter()
                .flat_map(|t| [t.subject, t.predicate, t.object])
                .collect();
            seeds.sort();
            seeds.dedup();
            if seeds.is_empty() {
                Err(PipelineError::NoSeeds)
            } else {
                Ok(seeds)
            }
        }
        PipelineData::Traversal(result) => {
            let seeds: Vec<SymbolId> = result.visited.iter().copied().collect();
            if seeds.is_empty() {
                Err(PipelineError::NoSeeds)
            } else {
                Ok(seeds)
            }
        }
        PipelineData::Inference(result) => {
            let seeds: Vec<SymbolId> = result.activations.iter().map(|(s, _)| *s).collect();
            if seeds.is_empty() {
                Err(PipelineError::NoSeeds)
            } else {
                Ok(seeds)
            }
        }
        PipelineData::Reasoning(_) => Err(PipelineError::IncompatibleData {
            stage_name: "extract_seeds".into(),
            expected: "Seeds, Triples, Traversal, or Inference".into(),
            actual: "Reasoning".into(),
        }),
    }
}

fn execute_retrieve(
    ctx: &PipelineContext,
    input: &PipelineData,
    config: &StageConfig,
    stage_name: &str,
) -> PipelineResult<PipelineData> {
    let seeds = extract_seeds(input)?;
    let traversal_config = match config {
        StageConfig::Retrieve { traversal } => traversal.clone(),
        _ => TraversalConfig::default(),
    };

    let result = traverse_bfs(&ctx.knowledge_graph, &seeds, &traversal_config).map_err(|e| {
        PipelineError::StageExecution {
            stage_name: stage_name.into(),
            message: format!("traversal failed: {e}"),
        }
    })?;

    Ok(PipelineData::Traversal(result))
}

fn execute_infer(
    ctx: &PipelineContext,
    input: &PipelineData,
    config: &StageConfig,
    stage_name: &str,
) -> PipelineResult<PipelineData> {
    let seeds = extract_seeds(input)?;

    let query = match config {
        StageConfig::Infer { query_template } => {
            let mut q = query_template.clone();
            q.seeds = seeds;
            q
        }
        _ => InferenceQuery::default().with_seeds(seeds),
    };

    let engine = InferEngine::new(
        Arc::clone(&ctx.ops),
        Arc::clone(&ctx.item_memory),
        Arc::clone(&ctx.knowledge_graph),
    );

    let result = engine.infer_with_rules(&query, &ctx.rules).map_err(|e| {
        PipelineError::StageExecution {
            stage_name: stage_name.into(),
            message: format!("inference failed: {e}"),
        }
    })?;

    Ok(PipelineData::Inference(result))
}

fn execute_reason(
    ctx: &PipelineContext,
    input: &PipelineData,
    config: &StageConfig,
    stage_name: &str,
) -> PipelineResult<PipelineData> {
    let (max_iterations, node_limit) = match config {
        StageConfig::Reason {
            max_iterations,
            node_limit,
        } => (*max_iterations, *node_limit),
        _ => (100, 10_000),
    };

    // Build an s-expression from inference activations or seeds.
    let expr_str = match input {
        PipelineData::Inference(result) => {
            if result.activations.is_empty() {
                return Ok(PipelineData::Reasoning(ReasoningResult {
                    simplified_expr: String::new(),
                    cost: 0,
                    saturated: true,
                }));
            }
            // Build a nested bundle of the top activations.
            let ids: Vec<String> = result
                .activations
                .iter()
                .take(10)
                .map(|(s, _)| s.get().to_string())
                .collect();
            if ids.len() == 1 {
                ids[0].clone()
            } else {
                ids.iter().skip(1).fold(ids[0].clone(), |acc, id| {
                    format!("(bundle {} {})", acc, id)
                })
            }
        }
        PipelineData::Seeds(seeds) => {
            if seeds.is_empty() {
                return Err(PipelineError::NoSeeds);
            }
            let ids: Vec<String> = seeds.iter().map(|s| s.get().to_string()).collect();
            if ids.len() == 1 {
                ids[0].clone()
            } else {
                ids.iter().skip(1).fold(ids[0].clone(), |acc, id| {
                    format!("(bundle {} {})", acc, id)
                })
            }
        }
        other => {
            return Err(PipelineError::IncompatibleData {
                stage_name: stage_name.into(),
                expected: "Inference or Seeds".into(),
                actual: other.variant_name().into(),
            });
        }
    };

    let expr = expr_str
        .parse::<egg::RecExpr<AkhLang>>()
        .map_err(|e| PipelineError::StageExecution {
            stage_name: stage_name.into(),
            message: format!("expression parse failed: {e}"),
        })?;

    let runner = Runner::default()
        .with_iter_limit(max_iterations)
        .with_node_limit(node_limit)
        .with_expr(&expr)
        .run(&ctx.rules);

    let saturated = runner.stop_reason.as_ref().is_some_and(|r| {
        matches!(r, egg::StopReason::Saturated)
    });

    let extractor = Extractor::new(&runner.egraph, AstSize);
    let (cost, best) = extractor.find_best(runner.roots[0]);

    Ok(PipelineData::Reasoning(ReasoningResult {
        simplified_expr: best.to_string(),
        cost,
        saturated,
    }))
}

fn execute_extract_triples(
    input: &PipelineData,
    config: &StageConfig,
    stage_name: &str,
) -> PipelineResult<PipelineData> {
    let min_confidence = match config {
        StageConfig::ExtractTriples { min_confidence } => *min_confidence,
        _ => 0.0,
    };

    match input {
        PipelineData::Triples(triples) => {
            let filtered: Vec<Triple> = triples
                .iter()
                .filter(|t| t.confidence >= min_confidence)
                .cloned()
                .collect();
            Ok(PipelineData::Triples(filtered))
        }
        PipelineData::Traversal(result) => {
            let filtered: Vec<Triple> = result
                .triples
                .iter()
                .filter(|t| t.confidence >= min_confidence)
                .cloned()
                .collect();
            Ok(PipelineData::Triples(filtered))
        }
        PipelineData::Inference(result) => {
            // Convert activations to synthetic triples (activation → inferred_as → self)
            // This is a simplified extraction; real use would have more domain logic.
            let triples: Vec<Triple> = result
                .activations
                .iter()
                .filter(|(_, conf)| *conf >= min_confidence)
                .map(|(sym, conf)| Triple::new(*sym, *sym, *sym).with_confidence(*conf))
                .collect();
            Ok(PipelineData::Triples(triples))
        }
        other => Err(PipelineError::IncompatibleData {
            stage_name: stage_name.into(),
            expected: "Triples, Traversal, or Inference".into(),
            actual: other.variant_name().into(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Triple;
    use crate::reason::builtin_rules;
    use crate::simd;
    use crate::vsa::{Dimension, Encoding};

    fn test_context() -> PipelineContext {
        let ops = Arc::new(VsaOps::new(
            simd::best_kernel(),
            Dimension::TEST,
            Encoding::Bipolar,
        ));
        let item_memory = Arc::new(ItemMemory::new(Dimension::TEST, Encoding::Bipolar, 1000));
        let knowledge_graph = Arc::new(KnowledgeGraph::new());
        PipelineContext {
            ops,
            item_memory,
            knowledge_graph,
            rules: builtin_rules(),
        }
    }

    fn sym(id: u64) -> SymbolId {
        SymbolId::new(id).unwrap()
    }

    #[test]
    fn empty_pipeline_error() {
        let ctx = test_context();
        let pipeline = Pipeline {
            name: "empty".into(),
            stages: vec![],
        };
        let result = pipeline.run(&ctx, PipelineData::Seeds(vec![sym(1)]));
        assert!(matches!(
            result,
            Err(PipelineError::EmptyPipeline { .. })
        ));
    }

    #[test]
    fn retrieve_stage() {
        let ctx = test_context();

        // Insert a triple into the knowledge graph.
        let a = sym(1);
        let r = sym(10);
        let b = sym(2);
        ctx.knowledge_graph
            .insert_triple(&Triple::new(a, r, b))
            .unwrap();

        let pipeline = Pipeline {
            name: "retrieve-only".into(),
            stages: vec![PipelineStage {
                name: "retrieve".into(),
                kind: StageKind::Retrieve,
                config: StageConfig::Retrieve {
                    traversal: TraversalConfig {
                        max_depth: 2,
                        ..Default::default()
                    },
                },
            }],
        };

        let output = pipeline
            .run(&ctx, PipelineData::Seeds(vec![a]))
            .unwrap();
        assert_eq!(output.stages_executed, 1);
        match &output.result {
            PipelineData::Traversal(tr) => {
                assert!(!tr.triples.is_empty());
                assert!(tr.visited.contains(&b));
            }
            other => panic!("expected Traversal, got {}", other.variant_name()),
        }
    }

    #[test]
    fn infer_stage() {
        let ctx = test_context();

        let sun = sym(1);
        let is_a = sym(2);
        let star = sym(3);

        ctx.item_memory.get_or_create(&ctx.ops, sun);
        ctx.item_memory.get_or_create(&ctx.ops, is_a);
        ctx.item_memory.get_or_create(&ctx.ops, star);
        ctx.knowledge_graph
            .insert_triple(&Triple::new(sun, is_a, star))
            .unwrap();

        let pipeline = Pipeline {
            name: "infer-only".into(),
            stages: vec![PipelineStage {
                name: "infer".into(),
                kind: StageKind::Infer,
                config: StageConfig::Infer {
                    query_template: InferenceQuery::default().with_max_depth(1),
                },
            }],
        };

        let output = pipeline
            .run(&ctx, PipelineData::Seeds(vec![sun]))
            .unwrap();
        match &output.result {
            PipelineData::Inference(result) => {
                let syms: Vec<SymbolId> = result.activations.iter().map(|(s, _)| *s).collect();
                assert!(syms.contains(&star), "Star should be inferred from Sun");
            }
            other => panic!("expected Inference, got {}", other.variant_name()),
        }
    }

    #[test]
    fn query_pipeline_end_to_end() {
        let ctx = test_context();

        let a = sym(1);
        let r = sym(10);
        let b = sym(2);

        ctx.item_memory.get_or_create(&ctx.ops, a);
        ctx.item_memory.get_or_create(&ctx.ops, r);
        ctx.item_memory.get_or_create(&ctx.ops, b);
        ctx.knowledge_graph
            .insert_triple(&Triple::new(a, r, b))
            .unwrap();

        let pipeline = Pipeline::query_pipeline();
        let output = pipeline
            .run(&ctx, PipelineData::Seeds(vec![a]))
            .unwrap();

        assert_eq!(output.stages_executed, 3);
        assert!(matches!(output.result, PipelineData::Reasoning(_)));
    }

    #[test]
    fn incompatible_data_error() {
        let ctx = test_context();
        let reasoning = PipelineData::Reasoning(ReasoningResult {
            simplified_expr: "x".into(),
            cost: 1,
            saturated: true,
        });

        let pipeline = Pipeline {
            name: "bad".into(),
            stages: vec![PipelineStage {
                name: "extract".into(),
                kind: StageKind::ExtractTriples,
                config: StageConfig::ExtractTriples {
                    min_confidence: 0.0,
                },
            }],
        };

        let result = pipeline.run(&ctx, reasoning);
        assert!(result.is_err());
    }

    #[test]
    fn extract_seeds_from_all_variants() {
        let seeds_data = PipelineData::Seeds(vec![sym(1), sym(2)]);
        assert_eq!(extract_seeds(&seeds_data).unwrap().len(), 2);

        let triples_data = PipelineData::Triples(vec![Triple::new(sym(1), sym(2), sym(3))]);
        let from_triples = extract_seeds(&triples_data).unwrap();
        assert_eq!(from_triples.len(), 3);

        let traversal_data = PipelineData::Traversal(TraversalResult {
            triples: vec![],
            visited: [sym(1), sym(2)].into_iter().collect(),
            depth_reached: 0,
        });
        assert_eq!(extract_seeds(&traversal_data).unwrap().len(), 2);

        let inference_data = PipelineData::Inference(InferenceResult {
            activations: vec![(sym(1), 0.9), (sym(2), 0.8)],
            pattern: None,
            provenance: vec![],
        });
        assert_eq!(extract_seeds(&inference_data).unwrap().len(), 2);

        let reasoning_data = PipelineData::Reasoning(ReasoningResult {
            simplified_expr: "x".into(),
            cost: 1,
            saturated: true,
        });
        assert!(extract_seeds(&reasoning_data).is_err());
    }
}
