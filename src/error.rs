//! Rich diagnostic error types for the akh-medu engine.
//!
//! Each subsystem defines its own error type with miette `#[diagnostic]` derives,
//! providing error codes, help text, and source chains so users know exactly what
//! went wrong and how to fix it.

use miette::Diagnostic;
use thiserror::Error;

/// Top-level error type for the akh-medu engine.
///
/// Each variant wraps a subsystem-specific error, preserving the full diagnostic
/// chain (error codes, help text, source spans) through to the user.
#[derive(Debug, Error, Diagnostic)]
pub enum AkhError {
    #[error(transparent)]
    #[diagnostic(transparent)]
    Vsa(#[from] VsaError),

    #[error(transparent)]
    #[diagnostic(transparent)]
    Store(#[from] StoreError),

    #[error(transparent)]
    #[diagnostic(transparent)]
    Graph(#[from] GraphError),

    #[error(transparent)]
    #[diagnostic(transparent)]
    Symbol(#[from] SymbolError),

    #[error(transparent)]
    #[diagnostic(transparent)]
    Infer(#[from] InferError),

    #[error(transparent)]
    #[diagnostic(transparent)]
    Reason(#[from] ReasonError),

    #[error(transparent)]
    #[diagnostic(transparent)]
    Skill(#[from] SkillError),

    #[error(transparent)]
    #[diagnostic(transparent)]
    Provenance(#[from] ProvenanceError),

    #[error(transparent)]
    #[diagnostic(transparent)]
    Pipeline(#[from] PipelineError),

    #[error(transparent)]
    #[diagnostic(transparent)]
    Engine(#[from] EngineError),
}

// ---------------------------------------------------------------------------
// VSA errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error, Diagnostic)]
pub enum VsaError {
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    #[diagnostic(
        code(akh::vsa::dim_mismatch),
        help(
            "All hypervectors in an operation must share the same dimension. \
             Check that you created them with the same Dimension parameter, \
             or re-encode the mismatched vector."
        )
    )]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("hypervector not found for symbol {symbol_id}")]
    #[diagnostic(
        code(akh::vsa::not_found),
        help(
            "The symbol has no associated hypervector in item memory. \
             Either insert it first with `item_memory.get_or_create()`, \
             or check that the symbol ID is correct."
        )
    )]
    HypervectorNotFound { symbol_id: u64 },

    #[error("empty bundle: cannot bundle zero hypervectors")]
    #[diagnostic(
        code(akh::vsa::empty_bundle),
        help("Provide at least one hypervector to the bundle operation.")
    )]
    EmptyBundle,

    #[error("encoding {encoding} is not supported for this operation")]
    #[diagnostic(
        code(akh::vsa::unsupported_encoding),
        help(
            "This operation requires a different encoding. \
             Bipolar supports bind/bundle/permute. \
             FHRR and SSP will be available in a future release."
        )
    )]
    UnsupportedEncoding { encoding: String },

    #[error("HNSW index error: {message}")]
    #[diagnostic(
        code(akh::vsa::hnsw_error),
        help("The HNSW approximate nearest-neighbor index encountered an internal error.")
    )]
    HnswError { message: String },
}

// ---------------------------------------------------------------------------
// Store errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error, Diagnostic)]
pub enum StoreError {
    #[error("I/O error: {source}")]
    #[diagnostic(
        code(akh::store::io),
        help(
            "A filesystem operation failed. Check that the data directory exists, \
             has correct permissions, and that the disk is not full."
        )
    )]
    Io {
        #[source]
        source: std::io::Error,
    },

    #[error("redb transaction error: {message}")]
    #[diagnostic(
        code(akh::store::redb),
        help(
            "The embedded database encountered a transaction error. \
             This may indicate corruption — try running with a fresh data directory. \
             If the problem persists, file a bug report."
        )
    )]
    Redb { message: String },

    #[error("serialization error: {message}")]
    #[diagnostic(
        code(akh::store::serde),
        help(
            "Failed to serialize or deserialize data. \
             This usually means the stored data format has changed between versions. \
             Try re-ingesting your data."
        )
    )]
    Serialization { message: String },

    #[error("key not found: {key}")]
    #[diagnostic(
        code(akh::store::not_found),
        help("The requested key does not exist in the store. Verify the key is correct.")
    )]
    NotFound { key: String },

    #[error("memory map error: {message}")]
    #[diagnostic(
        code(akh::store::mmap),
        help(
            "Failed to create or access a memory-mapped file. \
             Check available virtual memory and file permissions."
        )
    )]
    Mmap { message: String },
}

// ---------------------------------------------------------------------------
// Graph errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error, Diagnostic)]
pub enum GraphError {
    #[error("node not found: symbol {symbol_id}")]
    #[diagnostic(
        code(akh::graph::node_not_found),
        help(
            "The symbol has no corresponding node in the knowledge graph. \
             Insert a triple containing this symbol first."
        )
    )]
    NodeNotFound { symbol_id: u64 },

    #[error("duplicate triple: ({subject}, {predicate}, {object})")]
    #[diagnostic(
        code(akh::graph::duplicate_triple),
        help("This exact triple already exists in the graph. No action needed.")
    )]
    DuplicateTriple {
        subject: u64,
        predicate: u64,
        object: u64,
    },

    #[error("SPARQL query error: {message}")]
    #[diagnostic(
        code(akh::graph::sparql),
        help(
            "The SPARQL query failed. Check the query syntax and ensure \
             the oxigraph store is initialized."
        )
    )]
    Sparql { message: String },

    #[error("traversal depth exceeded maximum of {max_depth}")]
    #[diagnostic(
        code(akh::graph::depth_exceeded),
        help(
            "The graph traversal reached the maximum allowed depth. \
             Increase `max_depth` if deeper traversal is needed, \
             or check for cycles in your graph."
        )
    )]
    DepthExceeded { max_depth: usize },
}

// ---------------------------------------------------------------------------
// Symbol errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error, Diagnostic)]
pub enum SymbolError {
    #[error("symbol allocator exhausted: cannot allocate more than u64::MAX symbols")]
    #[diagnostic(
        code(akh::symbol::exhausted),
        help(
            "The symbol ID space is exhausted. This is extremely unlikely \
             in practice (requires 2^64 allocations). If you see this error, \
             something is very wrong — check for ID allocation loops."
        )
    )]
    AllocatorExhausted,

    #[error("invalid symbol kind: {kind}")]
    #[diagnostic(
        code(akh::symbol::invalid_kind),
        help("Valid symbol kinds are: Entity, Relation, Composite, Glyph.")
    )]
    InvalidKind { kind: String },

    #[error("symbol not found: id {symbol_id}")]
    #[diagnostic(
        code(akh::symbol::not_found),
        help("No symbol exists with this ID. Use `all_symbols()` to list available symbols.")
    )]
    NotFound { symbol_id: u64 },

    #[error("duplicate label: \"{label}\" already registered as sym:{existing_id}")]
    #[diagnostic(
        code(akh::symbol::duplicate_label),
        help(
            "A symbol with this label already exists. Use `lookup_symbol(\"{label}\")` to retrieve it, \
             or choose a different label."
        )
    )]
    DuplicateLabel { label: String, existing_id: u64 },

    #[error("symbol not found by label: \"{label}\"")]
    #[diagnostic(
        code(akh::symbol::label_not_found),
        help(
            "No symbol with this label exists. Lookup is case-insensitive. \
             Create the symbol first with `create_symbol()`."
        )
    )]
    LabelNotFound { label: String },
}

// ---------------------------------------------------------------------------
// Inference errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error, Diagnostic)]
pub enum InferError {
    #[error("no seeds provided for inference query")]
    #[diagnostic(
        code(akh::infer::no_seeds),
        help("An inference query requires at least one seed symbol to start from.")
    )]
    NoSeeds,

    #[error("inference depth exceeded maximum of {max_depth}")]
    #[diagnostic(
        code(akh::infer::depth_exceeded),
        help(
            "Multi-step inference reached the depth limit. \
             Increase `max_depth` in InferenceQuery or simplify the query."
        )
    )]
    DepthExceeded { max_depth: usize },

    #[error("no activations produced from {seed_count} seed(s) at depth {depth}")]
    #[diagnostic(
        code(akh::infer::no_activations),
        help(
            "The seed symbols had no outgoing edges in the knowledge graph. \
             Add triples connecting these symbols, or increase `max_depth` to \
             explore further."
        )
    )]
    NoActivations { seed_count: usize, depth: usize },

    #[error("analogy requires exactly 3 distinct symbols, got {count}")]
    #[diagnostic(
        code(akh::infer::invalid_analogy),
        help(
            "Analogy inference computes A:B :: C:? — provide exactly three \
             distinct SymbolIds (a, b, c) to find the fourth."
        )
    )]
    InvalidAnalogy { count: usize },

    #[error("activation for {symbol_id} below threshold: {confidence:.4} < {threshold:.4}")]
    #[diagnostic(
        code(akh::infer::below_threshold),
        help(
            "The recovered symbol's confidence is too low to be considered a \
             valid inference. Lower `min_confidence` or `min_similarity` in the \
             query, or add more supporting triples to the knowledge graph."
        )
    )]
    BelowThreshold {
        symbol_id: u64,
        confidence: f32,
        threshold: f32,
    },

    #[error(transparent)]
    #[diagnostic(transparent)]
    Vsa(#[from] VsaError),

    #[error(transparent)]
    #[diagnostic(transparent)]
    Graph(#[from] GraphError),

    #[error(transparent)]
    #[diagnostic(transparent)]
    Reason(#[from] ReasonError),
}

// ---------------------------------------------------------------------------
// Reasoning errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error, Diagnostic)]
pub enum ReasonError {
    #[error("e-graph saturation did not converge within {iterations} iterations")]
    #[diagnostic(
        code(akh::reason::no_convergence),
        help(
            "The equality saturation process did not reach a fixed point. \
             Try increasing the iteration limit, or review your rewrite rules \
             for non-terminating patterns."
        )
    )]
    NoConvergence { iterations: usize },

    #[error("invalid rewrite rule: {rule}")]
    #[diagnostic(
        code(akh::reason::invalid_rule),
        help("Check the rule syntax. Rules must be valid s-expressions for AkhLang.")
    )]
    InvalidRule { rule: String },

    #[error("expression parse error: {message}")]
    #[diagnostic(
        code(akh::reason::parse_error),
        help("The expression could not be parsed. Check for balanced parentheses and valid operators.")
    )]
    ParseError { message: String },
}

// ---------------------------------------------------------------------------
// Skill errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error, Diagnostic)]
pub enum SkillError {
    #[error("skillpack not found: {name}")]
    #[diagnostic(
        code(akh::skill::not_found),
        help(
            "No skillpack with this name is registered. \
             Check available skillpacks with `akh-medu skill list`."
        )
    )]
    NotFound { name: String },

    #[error("skillpack load error: {message}")]
    #[diagnostic(
        code(akh::skill::load_error),
        help(
            "Failed to load the skillpack from disk. \
             Verify the skillpack directory exists and contains a valid manifest."
        )
    )]
    LoadError { message: String },

    #[error("memory budget exceeded: need {required_mb} MB, available {available_mb} MB")]
    #[diagnostic(
        code(akh::skill::memory_budget),
        help(
            "Not enough memory to load this skillpack. \
             Either increase the memory budget in EngineConfig, \
             or evict other loaded skillpacks first."
        )
    )]
    MemoryBudgetExceeded {
        required_mb: usize,
        available_mb: usize,
    },

    #[error("invalid manifest at {path}: {message}")]
    #[diagnostic(
        code(akh::skill::invalid_manifest),
        help(
            "The skill.json file is malformed or missing required fields. \
             Ensure it contains id, name, version, and domains."
        )
    )]
    InvalidManifest { path: String, message: String },

    #[error("invalid rule in skillpack {skill_id}: {message}")]
    #[diagnostic(
        code(akh::skill::invalid_rule),
        help(
            "A rewrite rule could not be parsed. Rules must use the format: \
             <lhs-pattern> => <rhs-pattern>, e.g. '(bind ?a (bind ?a ?b)) => ?b'."
        )
    )]
    InvalidRule { skill_id: String, message: String },

    #[error("invalid state transition for skill {skill_id}: {from} -> {to}")]
    #[diagnostic(
        code(akh::skill::invalid_transition),
        help(
            "Skillpacks follow the lifecycle Cold → Warm → Hot. \
             You cannot skip states or go backwards except via deactivate."
        )
    )]
    InvalidTransition {
        skill_id: String,
        from: String,
        to: String,
    },

    #[error("skills directory not found: {path}")]
    #[diagnostic(
        code(akh::skill::no_skills_dir),
        help(
            "The skills directory does not exist. \
             Create it at the expected path or configure a different data directory."
        )
    )]
    NoSkillsDir { path: String },

    #[error("I/O error for skillpack {skill_id}: {source}")]
    #[diagnostic(
        code(akh::skill::io),
        help("A filesystem operation failed while accessing the skillpack directory.")
    )]
    Io {
        skill_id: String,
        #[source]
        source: std::io::Error,
    },
}

// ---------------------------------------------------------------------------
// Provenance errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error, Diagnostic)]
pub enum ProvenanceError {
    #[error("provenance record not found: id {id}")]
    #[diagnostic(
        code(akh::provenance::not_found),
        help(
            "No provenance record exists with this ID. \
             Check that the ID is correct, or query by derived symbol instead."
        )
    )]
    NotFound { id: u64 },

    #[error("provenance ledger requires persistence — no durable store configured")]
    #[diagnostic(
        code(akh::provenance::no_persistence),
        help(
            "The provenance ledger needs a data directory for storage. \
             Pass --data-dir to enable persistence, or use Engine::new() \
             with a configured data_dir."
        )
    )]
    NoPersistence,

    #[error(transparent)]
    #[diagnostic(transparent)]
    Store(#[from] StoreError),
}

// ---------------------------------------------------------------------------
// Pipeline errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error, Diagnostic)]
pub enum PipelineError {
    #[error("pipeline {name} has no stages")]
    #[diagnostic(
        code(akh::pipeline::empty),
        help("A pipeline must contain at least one stage. Add stages before running.")
    )]
    EmptyPipeline { name: String },

    #[error("stage {stage_name} execution failed: {message}")]
    #[diagnostic(
        code(akh::pipeline::stage_execution),
        help("An individual pipeline stage encountered an error during execution.")
    )]
    StageExecution { stage_name: String, message: String },

    #[error("pipeline {pipeline_name} failed at stage {stage_index} ({stage_name})")]
    #[diagnostic(
        code(akh::pipeline::stage_failure),
        help(
            "A pipeline stage failed. Check the inner error for details. \
             The pipeline was aborted at this stage."
        )
    )]
    StageFailure {
        pipeline_name: String,
        stage_name: String,
        stage_index: usize,
        #[source]
        source: Box<PipelineError>,
    },

    #[error("stage {stage_name} expected {expected} data, got {actual}")]
    #[diagnostic(
        code(akh::pipeline::incompatible_data),
        help(
            "The output of the previous stage is not compatible with this stage's input. \
             Check pipeline stage ordering — e.g. Retrieve produces Traversal data, \
             not Seeds."
        )
    )]
    IncompatibleData {
        stage_name: String,
        expected: String,
        actual: String,
    },

    #[error("no seed symbols available for pipeline execution")]
    #[diagnostic(
        code(akh::pipeline::no_seeds),
        help("The pipeline needs seed symbols to start. Provide initial Seeds data.")
    )]
    NoSeeds,
}

// ---------------------------------------------------------------------------
// Engine errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error, Diagnostic)]
pub enum EngineError {
    #[error("invalid configuration: {message}")]
    #[diagnostic(
        code(akh::engine::invalid_config),
        help("Check the EngineConfig fields. {message}")
    )]
    InvalidConfig { message: String },

    #[error("data directory error: {path}")]
    #[diagnostic(
        code(akh::engine::data_dir),
        help(
            "The data directory could not be accessed. \
             Ensure the path exists and has read/write permissions."
        )
    )]
    DataDir { path: String },

    #[error("ingest format error: {message}")]
    #[diagnostic(
        code(akh::engine::ingest_format),
        help(
            "Triple data must be in one of these formats:\n  \
             Label-based: {{\"subject\": \"Sun\", \"predicate\": \"is-a\", \"object\": \"Star\"}}\n  \
             Numeric: {{\"s\": 1, \"p\": 2, \"o\": 3}}"
        )
    )]
    IngestFormat { message: String },
}

/// Convenience alias for functions returning akh-medu results.
pub type AkhResult<T> = std::result::Result<T, AkhError>;

/// Result type for provenance operations.
pub type ProvenanceResult<T> = std::result::Result<T, ProvenanceError>;

/// Result type for pipeline operations.
pub type PipelineResult<T> = std::result::Result<T, PipelineError>;

/// Result type for skill operations.
pub type SkillResult<T> = std::result::Result<T, SkillError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vsa_error_converts_to_akh_error() {
        let err = VsaError::DimensionMismatch {
            expected: 10_000,
            actual: 5_000,
        };
        let akh: AkhError = err.into();
        assert!(matches!(akh, AkhError::Vsa(VsaError::DimensionMismatch { .. })));
    }

    #[test]
    fn store_error_converts_to_akh_error() {
        let err = StoreError::NotFound {
            key: "test".into(),
        };
        let akh: AkhError = err.into();
        assert!(matches!(akh, AkhError::Store(StoreError::NotFound { .. })));
    }

    #[test]
    fn infer_error_wraps_vsa_error() {
        let vsa_err = VsaError::EmptyBundle;
        let infer_err: InferError = vsa_err.into();
        assert!(matches!(infer_err, InferError::Vsa(VsaError::EmptyBundle)));
    }

    #[test]
    fn error_display_messages_are_descriptive() {
        let err = VsaError::DimensionMismatch {
            expected: 10_000,
            actual: 5_000,
        };
        let msg = format!("{err}");
        assert!(msg.contains("10000"));
        assert!(msg.contains("5000"));
    }
}
