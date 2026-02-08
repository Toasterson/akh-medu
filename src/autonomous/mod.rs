//! Autonomous KG building via neuro-symbolic reasoning.
//!
//! This module enables the agent to autonomously grow its knowledge graph
//! through logical deduction: transitive reasoning, inverse relations,
//! knowledge gap identification, multi-path confidence fusion, and schema
//! discovery.

pub mod error;
pub mod fusion;
pub mod gap;
pub mod rule_engine;
pub mod rules;
pub mod schema;

pub use error::{AutonomousError, AutonomousResult};
pub use fusion::{FusedConfidence, FusionConfig, InferencePath};
pub use gap::{GapAnalysisConfig, GapAnalysisResult, GapKind, KnowledgeGap};
pub use rule_engine::{DerivedTriple, RuleEngine, RuleEngineConfig, RuleEngineResult};
pub use rules::{InferenceRule, RuleKind, RuleSet, RuleTerm, TriplePattern};
pub use schema::{
    DiscoveredType, PredicatePattern, RelationHierarchy, SchemaDiscoveryConfig,
    SchemaDiscoveryResult,
};

/// Trait for external pattern recognizers (future: transformer models).
/// Unused in Phase 9 — establishes the integration boundary.
pub trait PatternRecognizer: Send + Sync {
    /// Given an entity and its KG context, predict likely predicates.
    fn predict_predicates(
        &self,
        entity: crate::symbol::SymbolId,
        context: &[crate::graph::Triple],
    ) -> Vec<(crate::symbol::SymbolId, f32)>;

    /// Extract (subject, predicate, object, confidence) tuples from text.
    fn extract_triples(&self, text: &str) -> Vec<(String, String, String, f32)>;

    /// Predict likely type classifications for an entity.
    fn predict_types(
        &self,
        entity: crate::symbol::SymbolId,
        context: &[crate::graph::Triple],
    ) -> Vec<(crate::symbol::SymbolId, f32)>;

    /// Display name for this recognizer.
    fn name(&self) -> &str;
}
