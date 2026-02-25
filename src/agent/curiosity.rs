//! Directed curiosity: information-gap-theory-based exploration targeting (Phase 11j).
//!
//! Enhances the existing Curiosity drive with ZPD-aware targeting so the agent
//! pursues concepts it's *almost* ready to learn (proximal zone sweet spot),
//! not random exploration.
//!
//! Uses a Gaussian-like curve peaking at a configurable fill ratio (~0.7) to
//! model the "curiosity sweet spot" from Loewenstein's information gap theory.

use serde::{Deserialize, Serialize};

use crate::bootstrap::prerequisite::ZpdZone;
use crate::engine::Engine;
use crate::symbol::{SymbolId, SymbolKind};

use super::error::AgentResult;
use super::goal::GoalSource;
use super::goal_generation::GoalProposal;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for directed curiosity targeting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectedCuriosityConfig {
    /// Maximum number of curiosity targets to track (default: 10).
    pub max_targets: usize,
    /// Lower bound of fill ratio for the "proximal" zone (default: 0.5).
    pub proximal_low: f32,
    /// Upper bound of fill ratio for the "proximal" zone (default: 0.9).
    pub proximal_high: f32,
    /// Fill ratio at which curiosity peaks — the information gap sweet spot (default: 0.7).
    pub peak_fill_ratio: f32,
    /// Multiplicative factor for "beyond" zone items (default: 0.2).
    pub beyond_curiosity_factor: f32,
    /// Multiplicative factor for "known" zone items (default: 0.0 — no curiosity).
    pub known_curiosity_factor: f32,
}

impl Default for DirectedCuriosityConfig {
    fn default() -> Self {
        Self {
            max_targets: 10,
            proximal_low: 0.5,
            proximal_high: 0.9,
            peak_fill_ratio: 0.7,
            beyond_curiosity_factor: 0.2,
            known_curiosity_factor: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A single curiosity target: a concept the agent is drawn to explore.
#[derive(Debug, Clone)]
pub struct CuriosityTarget {
    /// The concept symbol this target represents.
    pub concept: SymbolId,
    /// Information gap score (0.0–1.0), peaks at ~0.7 fill ratio.
    pub gap_score: f32,
    /// Expected number of relations for this concept's type.
    pub expected_relations: usize,
    /// Actual number of outgoing relations this concept has.
    pub filled_relations: usize,
    /// Ratio of filled / expected.
    pub fill_ratio: f32,
    /// ZPD zone classification based on fill ratio.
    pub zpd_zone: ZpdZone,
}

/// Result of directed curiosity analysis.
#[derive(Debug, Clone)]
pub struct CuriosityReport {
    /// Ranked curiosity targets (highest gap_score first).
    pub targets: Vec<CuriosityTarget>,
    /// Overall curiosity strength: max gap_score among proximal-zone targets.
    pub strongest_drive: f32,
    /// Goal proposals generated from the top targets.
    pub proposals: Vec<GoalProposal>,
}

// ---------------------------------------------------------------------------
// Core functions
// ---------------------------------------------------------------------------

/// Compute the information gap score for a given fill ratio.
///
/// Uses a Gaussian-like curve peaking at `config.peak_fill_ratio`:
///   score = exp(-((fill_ratio - peak) / 0.2)^2)
///
/// The score is then modulated by the ZPD zone:
/// - Known → `known_curiosity_factor * score` (typically 0.0)
/// - Proximal → `score` (full curiosity)
/// - Beyond → `beyond_curiosity_factor * score` (reduced curiosity)
pub fn compute_gap_score(fill_ratio: f32, config: &DirectedCuriosityConfig) -> f32 {
    let peak = config.peak_fill_ratio;
    let diff = fill_ratio - peak;
    let sigma = 0.2;
    let raw_score = (-((diff / sigma).powi(2))).exp();

    let zone = classify_zpd_zone(fill_ratio, config);
    let factor = match zone {
        ZpdZone::Known => config.known_curiosity_factor,
        ZpdZone::Proximal => 1.0,
        ZpdZone::Beyond => config.beyond_curiosity_factor,
    };

    (raw_score * factor).clamp(0.0, 1.0)
}

/// Classify a fill ratio into a Vygotsky ZPD zone.
///
/// - fill_ratio >= proximal_high → Known
/// - fill_ratio >= proximal_low → Proximal
/// - fill_ratio < proximal_low → Beyond
pub fn classify_zpd_zone(fill_ratio: f32, config: &DirectedCuriosityConfig) -> ZpdZone {
    if fill_ratio >= config.proximal_high {
        ZpdZone::Known
    } else if fill_ratio >= config.proximal_low {
        ZpdZone::Proximal
    } else {
        ZpdZone::Beyond
    }
}

/// Discover curiosity targets by analyzing entity completeness.
///
/// For each entity symbol:
/// 1. Count its outgoing relations.
/// 2. Compare to the average outgoing relation count across all entities (schema average).
/// 3. Compute fill_ratio = actual / expected.
/// 4. Compute gap_score via the Gaussian curve.
/// 5. Sort by gap_score descending, take top `max_targets`.
pub fn discover_curiosity_targets(
    engine: &Engine,
    config: &DirectedCuriosityConfig,
) -> AgentResult<Vec<CuriosityTarget>> {
    let symbols = engine.all_symbols();
    let entities: Vec<_> = symbols
        .iter()
        .filter(|s| s.kind == SymbolKind::Entity)
        .collect();

    if entities.is_empty() {
        return Ok(Vec::new());
    }

    // Compute the average outgoing relation count across all entities.
    let total_outgoing: usize = entities
        .iter()
        .map(|e| engine.triples_from(e.id).len())
        .sum();
    let avg_relations = if entities.is_empty() {
        1
    } else {
        (total_outgoing / entities.len()).max(1)
    };

    let mut targets: Vec<CuriosityTarget> = entities
        .iter()
        .map(|e| {
            let filled = engine.triples_from(e.id).len();
            let expected = avg_relations;
            let fill_ratio = (filled as f32 / expected as f32).clamp(0.0, 2.0);
            let gap_score = compute_gap_score(fill_ratio, config);
            let zpd_zone = classify_zpd_zone(fill_ratio, config);

            CuriosityTarget {
                concept: e.id,
                gap_score,
                expected_relations: expected,
                filled_relations: filled,
                fill_ratio,
                zpd_zone,
            }
        })
        .collect();

    // Sort by gap_score descending.
    targets.sort_by(|a, b| {
        b.gap_score
            .partial_cmp(&a.gap_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    targets.truncate(config.max_targets);

    Ok(targets)
}

/// Generate goal proposals from curiosity targets.
///
/// Only proximal-zone targets produce proposals.
pub fn generate_curiosity_proposals(
    targets: &[CuriosityTarget],
    engine: &Engine,
) -> AgentResult<Vec<GoalProposal>> {
    let proposals: Vec<GoalProposal> = targets
        .iter()
        .filter(|t| t.zpd_zone == ZpdZone::Proximal)
        .map(|t| {
            let label = engine.resolve_label(t.concept);
            GoalProposal {
                description: format!(
                    "Learn more about '{}': {} of {} expected relations filled",
                    label, t.filled_relations, t.expected_relations
                ),
                rationale: format!(
                    "Information gap score {:.2} (fill ratio {:.2}) in proximal zone",
                    t.gap_score, t.fill_ratio
                ),
                source: GoalSource::DriveExceeded {
                    drive: "curiosity".to_string(),
                    strength: t.gap_score,
                },
                priority_suggestion: (t.gap_score * 128.0) as u8,
                success_criteria: format!(
                    "'{}' has at least {} outgoing relations",
                    label, t.expected_relations
                ),
                conflicts_with: Vec::new(),
                feasibility: 0.8,
            }
        })
        .collect();

    Ok(proposals)
}

/// Run the full directed curiosity pipeline.
///
/// Returns a `CuriosityReport` with targets, overall drive strength, and proposals.
pub fn compute_directed_curiosity(
    engine: &Engine,
    config: &DirectedCuriosityConfig,
) -> AgentResult<CuriosityReport> {
    let targets = discover_curiosity_targets(engine, config)?;

    let proposals = generate_curiosity_proposals(&targets, engine)?;

    let strongest_drive = targets
        .iter()
        .filter(|t| t.zpd_zone == ZpdZone::Proximal)
        .map(|t| t.gap_score)
        .fold(0.0f32, f32::max);

    Ok(CuriosityReport {
        targets,
        strongest_drive,
        proposals,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
        let config = DirectedCuriosityConfig::default();
        assert_eq!(config.max_targets, 10);
        assert!((config.proximal_low - 0.5).abs() < f32::EPSILON);
        assert!((config.proximal_high - 0.9).abs() < f32::EPSILON);
        assert!((config.peak_fill_ratio - 0.7).abs() < f32::EPSILON);
        assert!((config.beyond_curiosity_factor - 0.2).abs() < f32::EPSILON);
        assert!((config.known_curiosity_factor - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn gap_score_peaks_at_configured_ratio() {
        let config = DirectedCuriosityConfig {
            peak_fill_ratio: 0.7,
            proximal_low: 0.3,
            proximal_high: 0.9,
            ..DirectedCuriosityConfig::default()
        };

        let at_peak = compute_gap_score(0.7, &config);
        let below_peak = compute_gap_score(0.5, &config);
        let above_peak = compute_gap_score(0.85, &config);

        // Score at peak should be highest (close to 1.0).
        assert!(
            at_peak > below_peak,
            "peak={at_peak} should exceed below={below_peak}"
        );
        assert!(
            at_peak > above_peak,
            "peak={at_peak} should exceed above={above_peak}"
        );
        assert!(at_peak > 0.9, "peak score should be near 1.0, got {at_peak}");
    }

    #[test]
    fn gap_score_known_zone_suppressed() {
        let config = DirectedCuriosityConfig::default();

        // fill_ratio >= 0.9 → Known → suppressed to 0.0
        let score = compute_gap_score(0.95, &config);
        assert!(
            score < f32::EPSILON,
            "Known zone should be suppressed, got {score}"
        );
    }

    #[test]
    fn gap_score_beyond_zone_reduced() {
        let config = DirectedCuriosityConfig::default();

        // fill_ratio < 0.5 → Beyond → reduced by beyond_curiosity_factor (0.2)
        let beyond_score = compute_gap_score(0.3, &config);
        let proximal_score = compute_gap_score(0.7, &config);

        assert!(
            beyond_score < proximal_score,
            "Beyond score {beyond_score} should be less than proximal {proximal_score}"
        );
        assert!(
            beyond_score <= 0.2 + f32::EPSILON,
            "Beyond score {beyond_score} should be <= beyond_curiosity_factor"
        );
    }

    #[test]
    fn classify_zpd_zone_thresholds() {
        let config = DirectedCuriosityConfig::default();

        assert_eq!(classify_zpd_zone(0.95, &config), ZpdZone::Known);
        assert_eq!(classify_zpd_zone(0.9, &config), ZpdZone::Known);
        assert_eq!(classify_zpd_zone(0.7, &config), ZpdZone::Proximal);
        assert_eq!(classify_zpd_zone(0.5, &config), ZpdZone::Proximal);
        assert_eq!(classify_zpd_zone(0.49, &config), ZpdZone::Beyond);
        assert_eq!(classify_zpd_zone(0.0, &config), ZpdZone::Beyond);
    }

    #[test]
    fn discover_targets_empty_kg() {
        let engine = test_engine();
        let config = DirectedCuriosityConfig::default();

        let targets = discover_curiosity_targets(&engine, &config).unwrap();
        assert!(targets.is_empty());
    }

    #[test]
    fn discover_targets_finds_incomplete_entities() {
        let engine = test_engine();

        // Create entities with varying degrees of completeness.
        let well_connected = engine.resolve_or_create_entity("well_connected").unwrap();
        let sparse = engine.resolve_or_create_entity("sparse").unwrap();
        let _isolated = engine.resolve_or_create_entity("isolated").unwrap();

        let rel = engine.resolve_or_create_relation("test:rel").unwrap();
        let obj1 = engine.resolve_or_create_entity("obj1").unwrap();
        let obj2 = engine.resolve_or_create_entity("obj2").unwrap();
        let obj3 = engine.resolve_or_create_entity("obj3").unwrap();

        // Well connected: 3 outgoing.
        engine.add_triple(&crate::graph::Triple::new(well_connected, rel, obj1)).unwrap();
        engine.add_triple(&crate::graph::Triple::new(well_connected, rel, obj2)).unwrap();
        engine.add_triple(&crate::graph::Triple::new(well_connected, rel, obj3)).unwrap();

        // Sparse: 1 outgoing.
        engine.add_triple(&crate::graph::Triple::new(sparse, rel, obj1)).unwrap();

        let config = DirectedCuriosityConfig {
            max_targets: 10,
            ..DirectedCuriosityConfig::default()
        };

        let targets = discover_curiosity_targets(&engine, &config).unwrap();
        assert!(!targets.is_empty());

        // All targets should have valid fields.
        for t in &targets {
            assert!(t.gap_score >= 0.0 && t.gap_score <= 1.0);
            assert!(t.fill_ratio >= 0.0);
        }
    }

    #[test]
    fn generate_proposals_only_for_proximal() {
        let engine = test_engine();

        let targets = vec![
            CuriosityTarget {
                concept: engine.resolve_or_create_entity("proximal_concept").unwrap(),
                gap_score: 0.9,
                expected_relations: 5,
                filled_relations: 3,
                fill_ratio: 0.6,
                zpd_zone: ZpdZone::Proximal,
            },
            CuriosityTarget {
                concept: engine.resolve_or_create_entity("known_concept").unwrap(),
                gap_score: 0.1,
                expected_relations: 5,
                filled_relations: 5,
                fill_ratio: 1.0,
                zpd_zone: ZpdZone::Known,
            },
            CuriosityTarget {
                concept: engine.resolve_or_create_entity("beyond_concept").unwrap(),
                gap_score: 0.2,
                expected_relations: 5,
                filled_relations: 1,
                fill_ratio: 0.2,
                zpd_zone: ZpdZone::Beyond,
            },
        ];

        let proposals = generate_curiosity_proposals(&targets, &engine).unwrap();
        assert_eq!(proposals.len(), 1, "Only proximal targets should produce proposals");
        assert!(proposals[0].description.contains("proximal_concept"));
    }

    #[test]
    fn compute_directed_curiosity_full_pipeline() {
        let engine = test_engine();
        let config = DirectedCuriosityConfig::default();

        let report = compute_directed_curiosity(&engine, &config).unwrap();
        // Empty KG → no targets, zero drive.
        assert!(report.targets.is_empty());
        assert!(report.strongest_drive < f32::EPSILON);
        assert!(report.proposals.is_empty());
    }

    #[test]
    fn config_serialization_roundtrip() {
        let config = DirectedCuriosityConfig::default();
        let encoded = bincode::serialize(&config).unwrap();
        let decoded: DirectedCuriosityConfig = bincode::deserialize(&encoded).unwrap();
        assert_eq!(decoded.max_targets, config.max_targets);
        assert!((decoded.peak_fill_ratio - config.peak_fill_ratio).abs() < f32::EPSILON);
    }
}
