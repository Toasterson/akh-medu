//! Processing pipeline: DAG-based multi-stage execution.
//!
//! Pipelines chain together processing stages (extract, infer, reason, retrieve)
//! in a directed acyclic graph. Full implementation in Phase 2.

/// Named processing stage in a pipeline.
#[derive(Debug, Clone)]
pub struct PipelineStage {
    /// Stage name.
    pub name: String,
    /// Stage kind.
    pub kind: StageKind,
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
