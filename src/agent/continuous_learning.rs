//! Continuous learning idle task: wires directed curiosity into the bootstrap pipeline.
//!
//! On a long timer (~2 hours), chains:
//! 1. Directed curiosity → identifies top information gaps (proximal zone)
//! 2. Builds a synthetic `ExpansionResult` from gap concept labels
//! 3. Prerequisite analysis → orders the concepts
//! 4. Resource discovery → finds learning material (conservative API budget)
//! 5. Curriculum ingestion → pulls resources into KG
//! 6. Competence assessment → measures progress
//!
//! This closes the curiosity→learning loop so the akh autonomously fills its
//! own knowledge gaps during idle time.

use std::sync::Arc;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::bootstrap::competence::{CompetenceAssessor, CompetenceConfig};
use crate::bootstrap::expand::ExpansionResult;
use crate::bootstrap::ingest::{CurriculumIngestor, IngestionConfig};
use crate::bootstrap::prerequisite::{PrerequisiteAnalyzer, PrerequisiteConfig, ZpdZone};
use crate::bootstrap::purpose::{DreyfusLevel, PurposeModel};
use crate::bootstrap::resources::{
    self, ResourceDiscoverer, ResourceDiscoveryConfig,
};
use crate::engine::Engine;
use crate::provenance::{DerivationKind, ProvenanceId, ProvenanceRecord};

use super::curiosity::{compute_directed_curiosity, DirectedCuriosityConfig};

// ── Errors ──────────────────────────────────────────────────────────────

/// Errors from the continuous learning idle task.
#[derive(Debug, Error, Diagnostic)]
pub enum ContinuousLearningError {
    #[error("no curiosity targets found in proximal zone")]
    #[diagnostic(
        code(akh::agent::continuous_learning::no_targets),
        help(
            "Directed curiosity found no concepts in the proximal zone. \
             The knowledge graph may be too sparse or too complete for gap detection."
        )
    )]
    NoCuriosityTargets,

    #[error("no expansion concepts could be resolved from curiosity targets")]
    #[diagnostic(
        code(akh::agent::continuous_learning::no_expansion),
        help(
            "Curiosity targets were found but their labels could not be \
             resolved into an expansion result for the bootstrap pipeline."
        )
    )]
    NoExpansionConcepts,

    #[error("stage \"{stage}\" failed: {reason}")]
    #[diagnostic(
        code(akh::agent::continuous_learning::stage_failed),
        help(
            "A continuous learning sub-stage encountered an error. \
             The next idle cycle will retry automatically."
        )
    )]
    StageFailed {
        stage: &'static str,
        reason: String,
    },

    #[error("{0}")]
    #[diagnostic(
        code(akh::agent::continuous_learning::engine),
        help("An engine-level error occurred during continuous learning.")
    )]
    Engine(Box<crate::error::AkhError>),
}

impl From<crate::error::AkhError> for ContinuousLearningError {
    fn from(e: crate::error::AkhError) -> Self {
        Self::Engine(Box::new(e))
    }
}

/// Convenience alias.
pub type ContinuousLearningResult<T> = std::result::Result<T, ContinuousLearningError>;

// ── Configuration ───────────────────────────────────────────────────────

/// Configuration for the continuous learning idle task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuousLearningConfig {
    /// Maximum curiosity targets to explore per cycle (default: 5).
    pub max_targets: usize,
    /// Maximum API calls for resource discovery (default: 5).
    pub max_api_calls: usize,
    /// Minimum gap score to accept a curiosity target (default: 0.3).
    pub min_gap_score: f32,
}

impl Default for ContinuousLearningConfig {
    fn default() -> Self {
        Self {
            max_targets: 5,
            max_api_calls: 5,
            min_gap_score: 0.3,
        }
    }
}

// ── Result ──────────────────────────────────────────────────────────────

/// Summary of a continuous learning cycle.
#[derive(Debug, Clone)]
pub struct ContinuousLearningRunResult {
    /// Number of curiosity targets found in the proximal zone.
    pub targets_found: usize,
    /// Number of learning resources discovered.
    pub resources_discovered: usize,
    /// Number of concepts successfully ingested into the KG.
    pub concepts_ingested: usize,
    /// Competence score after ingestion (if assessment ran).
    pub competence_score: f32,
    /// Dreyfus level string after assessment (if ran).
    pub dreyfus_level: String,
    /// Provenance IDs created during this cycle.
    pub provenance_ids: Vec<ProvenanceId>,
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Build a synthetic `ExpansionResult` from curiosity target labels.
fn build_synthetic_expansion(labels: &[String]) -> ExpansionResult {
    ExpansionResult {
        concept_count: labels.len(),
        relation_count: 0,
        rejected_count: 0,
        api_calls: 0,
        domain_prototype_id: None,
        microtheory_id: None,
        provenance_ids: Vec::new(),
        accepted_labels: labels.to_vec(),
        boundary_rejects: Vec::new(),
    }
}

/// Try to load the bootstrap session's purpose model, falling back to a
/// minimal synthetic one built from concept labels.
fn resolve_purpose_model(engine: &Engine, labels: &[String]) -> PurposeModel {
    // Try to load from a persisted bootstrap session.
    let purpose_from_session = engine
        .store()
        .get_meta(b"bootstrap:session")
        .ok()
        .flatten()
        .and_then(|bytes| {
            bincode::deserialize::<crate::bootstrap::orchestrator::BootstrapSession>(&bytes).ok()
        })
        .and_then(|session| session.intent)
        .map(|intent| intent.purpose);

    if let Some(purpose) = purpose_from_session {
        return purpose;
    }

    // Fallback: minimal synthetic purpose.
    PurposeModel {
        domain: labels.first().cloned().unwrap_or_default(),
        competence_level: DreyfusLevel::Competent,
        seed_concepts: labels.to_vec(),
        description: "continuous learning".to_string(),
    }
}

// ── Core function ───────────────────────────────────────────────────────

/// Run a single continuous learning cycle.
///
/// Chains directed curiosity → prerequisite analysis → resource discovery →
/// curriculum ingestion → competence assessment. Each sub-stage failure is
/// handled gracefully: the function either falls back or returns an early
/// summary rather than propagating errors that would kill the idle scheduler.
pub fn run_continuous_learning(
    engine: &Arc<Engine>,
    curiosity_config: &DirectedCuriosityConfig,
    config: &ContinuousLearningConfig,
) -> ContinuousLearningResult<ContinuousLearningRunResult> {
    // 1. Directed curiosity — identify proximal zone targets.
    let mut curiosity_cfg = curiosity_config.clone();
    curiosity_cfg.max_targets = config.max_targets;

    let report = compute_directed_curiosity(engine, &curiosity_cfg)
        .map_err(|e| ContinuousLearningError::StageFailed {
            stage: "curiosity",
            reason: e.to_string(),
        })?;

    // 2. Filter to proximal targets above min_gap_score.
    let proximal_targets: Vec<_> = report
        .targets
        .iter()
        .filter(|t| t.zpd_zone == ZpdZone::Proximal && t.gap_score >= config.min_gap_score)
        .collect();

    if proximal_targets.is_empty() {
        return Err(ContinuousLearningError::NoCuriosityTargets);
    }

    let targets_found = proximal_targets.len();
    let gap_score_avg = proximal_targets.iter().map(|t| t.gap_score).sum::<f32>()
        / targets_found as f32;

    // 3. Resolve labels and build synthetic expansion.
    let labels: Vec<String> = proximal_targets
        .iter()
        .map(|t| engine.resolve_label(t.concept))
        .collect();

    if labels.is_empty() {
        return Err(ContinuousLearningError::NoExpansionConcepts);
    }

    let expansion = build_synthetic_expansion(&labels);

    // 4. Prerequisite analysis (fall back to synthetic curriculum on failure).
    let prereq_config = PrerequisiteConfig::default();
    let prereq_result = match PrerequisiteAnalyzer::new(engine, prereq_config) {
        Ok(analyzer) => match analyzer.analyze(&expansion, engine) {
            Ok(result) => result,
            Err(_) => resources::synthetic_curriculum_from_expansion(&expansion, engine),
        },
        Err(_) => resources::synthetic_curriculum_from_expansion(&expansion, engine),
    };

    // 5. Resource discovery (conservative API budget).
    let resource_config = ResourceDiscoveryConfig {
        max_api_calls: config.max_api_calls,
        ..ResourceDiscoveryConfig::default()
    };

    let resource_result = match ResourceDiscoverer::new(engine, resource_config) {
        Ok(mut discoverer) => {
            match discoverer.discover(&prereq_result, &expansion, &labels, engine) {
                Ok(result) => result,
                Err(e) => {
                    // No resources found — return early with partial summary.
                    let mut result = ContinuousLearningRunResult {
                        targets_found,
                        resources_discovered: 0,
                        concepts_ingested: 0,
                        competence_score: 0.0,
                        dreyfus_level: "unknown".to_string(),
                        provenance_ids: Vec::new(),
                    };

                    // Still store provenance for the curiosity stage.
                    if let Ok(prov_id) = store_provenance(
                        engine,
                        &result,
                        gap_score_avg,
                    ) {
                        result.provenance_ids.push(prov_id);
                    }

                    tracing::info!(
                        "continuous_learning: resource discovery failed: {e}, returning partial"
                    );
                    return Ok(result);
                }
            }
        }
        Err(e) => {
            tracing::info!("continuous_learning: resource discoverer init failed: {e}");
            let mut result = ContinuousLearningRunResult {
                targets_found,
                resources_discovered: 0,
                concepts_ingested: 0,
                competence_score: 0.0,
                dreyfus_level: "unknown".to_string(),
                provenance_ids: Vec::new(),
            };
            if let Ok(prov_id) = store_provenance(engine, &result, gap_score_avg) {
                result.provenance_ids.push(prov_id);
            }
            return Ok(result);
        }
    };

    let resources_discovered = resource_result.resources.len();

    // 6. Curriculum ingestion.
    let ingest_config = IngestionConfig::default();
    let concepts_ingested = match CurriculumIngestor::new(engine, ingest_config) {
        Ok(mut ingestor) => match ingestor.ingest(&prereq_result, &resource_result, engine) {
            Ok(run) => run.concepts_ingested,
            Err(e) => {
                tracing::info!("continuous_learning: ingestion failed: {e}");
                0
            }
        },
        Err(e) => {
            tracing::info!("continuous_learning: ingestor init failed: {e}");
            0
        }
    };

    // 7. Competence assessment (optional — skip on failure).
    let purpose_model = resolve_purpose_model(engine, &labels);
    let competence_config = CompetenceConfig::default();
    let (competence_score, dreyfus_level) = match CompetenceAssessor::new(engine, competence_config)
    {
        Ok(assessor) => match assessor.assess(&prereq_result, &purpose_model, engine) {
            Ok(report) => (report.overall_score, report.overall_dreyfus.to_string()),
            Err(e) => {
                tracing::info!("continuous_learning: assessment failed: {e}");
                (0.0, "unknown".to_string())
            }
        },
        Err(e) => {
            tracing::info!("continuous_learning: assessor init failed: {e}");
            (0.0, "unknown".to_string())
        }
    };

    // 8. Store provenance (tag 70).
    let mut result = ContinuousLearningRunResult {
        targets_found,
        resources_discovered,
        concepts_ingested,
        competence_score,
        dreyfus_level,
        provenance_ids: Vec::new(),
    };

    if let Ok(prov_id) = store_provenance(engine, &result, gap_score_avg) {
        result.provenance_ids.push(prov_id);
    }

    Ok(result)
}

/// Store a `ContinuousLearning` provenance record (tag 70).
fn store_provenance(
    engine: &Engine,
    result: &ContinuousLearningRunResult,
    gap_score_avg: f32,
) -> ContinuousLearningResult<ProvenanceId> {
    // Use SymbolId(1) as sentinel — matches orchestrator pattern.
    let derived_id = crate::symbol::SymbolId::new(1).expect("1 is non-zero");
    let mut record = ProvenanceRecord::new(
        derived_id,
        DerivationKind::ContinuousLearning {
            targets_explored: result.targets_found,
            resources_found: result.resources_discovered,
            concepts_ingested: result.concepts_ingested,
            dreyfus_level: result.dreyfus_level.clone(),
            gap_score_avg,
        },
    )
    .with_confidence(gap_score_avg);

    engine
        .store_provenance(&mut record)
        .map_err(|e| ContinuousLearningError::Engine(Box::new(e)))
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig};
    use crate::vsa::Dimension;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension(1000),
            ..EngineConfig::default()
        })
        .unwrap()
    }

    #[test]
    fn config_defaults() {
        let config = ContinuousLearningConfig::default();
        assert_eq!(config.max_targets, 5);
        assert_eq!(config.max_api_calls, 5);
        assert!((config.min_gap_score - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn error_display_no_targets() {
        let err = ContinuousLearningError::NoCuriosityTargets;
        let msg = err.to_string();
        assert!(msg.contains("no curiosity targets"));
    }

    #[test]
    fn error_display_no_expansion() {
        let err = ContinuousLearningError::NoExpansionConcepts;
        let msg = err.to_string();
        assert!(msg.contains("no expansion concepts"));
    }

    #[test]
    fn error_display_stage_failed() {
        let err = ContinuousLearningError::StageFailed {
            stage: "curiosity",
            reason: "test failure".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("curiosity"));
        assert!(msg.contains("test failure"));
    }

    #[test]
    fn error_from_akh_error() {
        let akh_err = crate::error::AkhError::Store(crate::error::StoreError::Redb {
            message: "test".to_string(),
        });
        let err = ContinuousLearningError::from(akh_err);
        assert!(matches!(err, ContinuousLearningError::Engine(_)));
    }

    #[test]
    fn result_construction() {
        let result = ContinuousLearningRunResult {
            targets_found: 3,
            resources_discovered: 10,
            concepts_ingested: 5,
            competence_score: 0.65,
            dreyfus_level: "competent".to_string(),
            provenance_ids: Vec::new(),
        };
        assert_eq!(result.targets_found, 3);
        assert_eq!(result.resources_discovered, 10);
        assert_eq!(result.concepts_ingested, 5);
        assert!((result.competence_score - 0.65).abs() < f32::EPSILON);
        assert_eq!(result.dreyfus_level, "competent");
    }

    #[test]
    fn build_synthetic_expansion_from_labels() {
        let labels = vec!["algebra".to_string(), "calculus".to_string()];
        let expansion = build_synthetic_expansion(&labels);
        assert_eq!(expansion.accepted_labels.len(), 2);
        assert_eq!(expansion.concept_count, 2);
        assert_eq!(expansion.relation_count, 0);
        assert!(expansion.provenance_ids.is_empty());
    }

    #[test]
    fn empty_kg_returns_no_curiosity_targets() {
        let engine = Arc::new(test_engine());
        let curiosity_cfg = DirectedCuriosityConfig::default();
        let config = ContinuousLearningConfig::default();

        let result = run_continuous_learning(&engine, &curiosity_cfg, &config);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ContinuousLearningError::NoCuriosityTargets
        ));
    }

    #[test]
    fn config_serialization_roundtrip() {
        let config = ContinuousLearningConfig::default();
        let encoded = bincode::serialize(&config).unwrap();
        let decoded: ContinuousLearningConfig = bincode::deserialize(&encoded).unwrap();
        assert_eq!(decoded.max_targets, config.max_targets);
        assert_eq!(decoded.max_api_calls, config.max_api_calls);
        assert!((decoded.min_gap_score - config.min_gap_score).abs() < f32::EPSILON);
    }

    #[test]
    fn provenance_tag_70_variant() {
        let kind = DerivationKind::ContinuousLearning {
            targets_explored: 5,
            resources_found: 10,
            concepts_ingested: 3,
            dreyfus_level: "competent".to_string(),
            gap_score_avg: 0.72,
        };
        assert_eq!(kind.tag(), 70);
    }
}
