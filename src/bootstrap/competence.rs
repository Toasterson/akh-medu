//! Competence assessment: evaluates how well the akh knows its domain (Phase 14g).
//!
//! After curriculum ingestion (Phase 14f), this module assesses the populated KG
//! using gap analysis, schema discovery, graph analytics, and VSA structural analysis.
//! Produces a unified `CompetenceReport` with per-knowledge-area Dreyfus level
//! estimation and a readiness recommendation.
//!
//! Pipeline: `14f IngestionRunResult + Engine → CompetenceAssessor → CompetenceReport → 14h Orchestrator`

use std::collections::{HashMap, HashSet};

use miette::Diagnostic;
use thiserror::Error;

use crate::autonomous::gap::{self, GapAnalysisConfig};
use crate::autonomous::schema::{self, SchemaDiscoveryConfig};
use crate::bootstrap::prerequisite::{CurriculumEntry, PrereqAnalysisResult};
use crate::bootstrap::purpose::{DreyfusLevel, PurposeModel};
use crate::engine::Engine;
use crate::graph::Triple;
use crate::graph::analytics::shortest_path;
use crate::provenance::{DerivationKind, ProvenanceId, ProvenanceRecord};
use crate::symbol::SymbolId;

// ── Errors ──────────────────────────────────────────────────────────────

/// Errors specific to competence assessment.
#[derive(Debug, Error, Diagnostic)]
pub enum CompetenceError {
    #[error("no curriculum concepts to assess — the curriculum is empty")]
    #[diagnostic(
        code(akh::bootstrap::competence::no_concepts),
        help(
            "Run prerequisite analysis first with `akh awaken prerequisite --seeds ...` \
             to generate a curriculum with concepts to assess."
        )
    )]
    NoConcepts,

    #[error("knowledge graph too sparse for meaningful assessment ({triple_count} triple(s), need at least {min_required})")]
    #[diagnostic(
        code(akh::bootstrap::competence::insufficient_triples),
        help(
            "Run curriculum ingestion first with `akh awaken ingest --seeds ...` \
             to populate the knowledge graph before assessment."
        )
    )]
    InsufficientTriples { triple_count: usize, min_required: usize },

    #[error("{0}")]
    #[diagnostic(
        code(akh::bootstrap::competence::engine),
        help("An engine-level error occurred during competence assessment.")
    )]
    Engine(Box<crate::error::AkhError>),
}

impl From<crate::error::AkhError> for CompetenceError {
    fn from(e: crate::error::AkhError) -> Self {
        Self::Engine(Box::new(e))
    }
}

/// Convenience alias.
pub type CompetenceResult<T> = std::result::Result<T, CompetenceError>;

// ── Configuration ───────────────────────────────────────────────────────

/// Tunable parameters for competence assessment.
#[derive(Debug, Clone)]
pub struct CompetenceConfig {
    /// Minimum triples per concept to consider it "known" (default: 3).
    pub min_triples_per_concept: usize,
    /// Maximum Bloom depth to evaluate (1–4, default: 4).
    pub bloom_max_depth: usize,
    /// Weight for coverage score in Dreyfus calculation.
    pub coverage_weight: f32,
    /// Weight for connectivity (1 - dead_end_ratio) in Dreyfus calculation.
    pub connectivity_weight: f32,
    /// Weight for type diversity in Dreyfus calculation.
    pub type_diversity_weight: f32,
    /// Weight for relation density in Dreyfus calculation.
    pub relation_density_weight: f32,
    /// Weight for cross-domain connections in Dreyfus calculation.
    pub cross_domain_weight: f32,
}

impl Default for CompetenceConfig {
    fn default() -> Self {
        Self {
            min_triples_per_concept: 3,
            bloom_max_depth: 4,
            coverage_weight: 0.30,
            connectivity_weight: 0.20,
            type_diversity_weight: 0.20,
            relation_density_weight: 0.15,
            cross_domain_weight: 0.15,
        }
    }
}

// ── Bloom's Taxonomy ────────────────────────────────────────────────────

/// Bloom's taxonomy level for competency questions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BloomLevel {
    /// "Does {concept} exist?"
    Remember = 0,
    /// "How does {concept} relate to {peer}?"
    Understand = 1,
    /// "What does {concept} connect to?" (>= 2 outgoing)
    Apply = 2,
    /// "Why does {concept} require {prereq}?" (all prereqs Known)
    Analyze = 3,
}

impl BloomLevel {
    /// All levels in ascending order.
    pub fn all() -> &'static [BloomLevel] {
        &[
            BloomLevel::Remember,
            BloomLevel::Understand,
            BloomLevel::Apply,
            BloomLevel::Analyze,
        ]
    }

    /// Human-readable label.
    pub fn as_label(&self) -> &'static str {
        match self {
            Self::Remember => "remember",
            Self::Understand => "understand",
            Self::Apply => "apply",
            Self::Analyze => "analyze",
        }
    }
}

impl std::fmt::Display for BloomLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_label())
    }
}

// ── Types ───────────────────────────────────────────────────────────────

/// A single competency question evaluated against the KG.
#[derive(Debug, Clone)]
struct CompetencyQuestion {
    _bloom_level: BloomLevel,
    _concept: SymbolId,
    _related_concept: Option<SymbolId>,
    answered: bool,
}

/// Assessment of a single knowledge area (group of concepts at similar tiers).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KnowledgeAreaAssessment {
    /// Name of this knowledge area (derived from concept labels).
    pub name: String,
    /// Estimated Dreyfus level.
    pub dreyfus_level: DreyfusLevel,
    /// Numeric score [0.0, 1.0].
    pub score: f32,
    /// Total triples for concepts in this area.
    pub triple_count: usize,
    /// Number of competency questions answered.
    pub cq_answered: usize,
    /// Total competency questions generated.
    pub cq_total: usize,
    /// Number of knowledge gaps found.
    pub gap_count: usize,
    /// Average relation density (triples per concept).
    pub relation_density: f32,
    /// Score breakdown for verbose output.
    pub score_components: ScoreComponents,
}

/// Individual score components for transparency.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScoreComponents {
    pub coverage: f32,
    pub connectivity: f32,
    pub type_diversity: f32,
    pub relation_density: f32,
    pub cross_domain: f32,
}

/// What should the akh do after assessment?
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum BootstrapRecommendation {
    /// Knowledge is sufficient for the target Dreyfus level.
    Ready,
    /// More learning needed — focus on specific areas.
    ContinueLearning {
        estimated_cycles: usize,
        focus_areas: Vec<String>,
    },
    /// Cannot assess — operator input needed.
    NeedsOperatorInput { question: String },
}

impl std::fmt::Display for BootstrapRecommendation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ready => write!(f, "ready"),
            Self::ContinueLearning {
                estimated_cycles,
                focus_areas,
            } => write!(
                f,
                "continue learning (~{} cycles, focus: {})",
                estimated_cycles,
                focus_areas.join(", ")
            ),
            Self::NeedsOperatorInput { question } => {
                write!(f, "needs operator input: {}", question)
            }
        }
    }
}

/// Full competence assessment report.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompetenceReport {
    /// Overall estimated Dreyfus level.
    pub overall_dreyfus: DreyfusLevel,
    /// Overall numeric score [0.0, 1.0].
    pub overall_score: f32,
    /// Per-knowledge-area breakdown.
    pub knowledge_areas: Vec<KnowledgeAreaAssessment>,
    /// Remaining knowledge gaps.
    pub remaining_gaps: Vec<String>,
    /// Recommended next action.
    pub recommendation: BootstrapRecommendation,
    /// Provenance IDs for stored assessment triples.
    pub provenance_ids: Vec<ProvenanceId>,
}

// ── Predicates ──────────────────────────────────────────────────────────

/// Well-known relation symbols for assessment triples.
struct CompetencePredicates {
    dreyfus_level: SymbolId,
    competence_score: SymbolId,
    assessed_at: SymbolId,
}

impl CompetencePredicates {
    fn init(engine: &Engine) -> CompetenceResult<Self> {
        Ok(Self {
            dreyfus_level: engine.resolve_or_create_relation("assess:dreyfus_level")?,
            competence_score: engine.resolve_or_create_relation("assess:competence_score")?,
            assessed_at: engine.resolve_or_create_relation("assess:assessed_at")?,
        })
    }
}

// ── Core Assessor ───────────────────────────────────────────────────────

/// Performs competence assessment over the populated KG.
pub struct CompetenceAssessor {
    config: CompetenceConfig,
    predicates: CompetencePredicates,
}

impl CompetenceAssessor {
    /// Create a new assessor, resolving well-known predicates.
    pub fn new(engine: &Engine, config: CompetenceConfig) -> CompetenceResult<Self> {
        let predicates = CompetencePredicates::init(engine)?;
        Ok(Self { config, predicates })
    }

    /// Run the full competence assessment.
    ///
    /// Takes the prerequisite analysis result (curriculum) and purpose model,
    /// evaluates the KG, and returns a `CompetenceReport`.
    pub fn assess(
        &self,
        prereq_result: &PrereqAnalysisResult,
        purpose_model: &PurposeModel,
        engine: &Engine,
    ) -> CompetenceResult<CompetenceReport> {
        // Precondition: curriculum non-empty.
        if prereq_result.curriculum.is_empty() {
            return Err(CompetenceError::NoConcepts);
        }

        // Precondition: KG has triples.
        let all_triples = engine.all_triples();
        let min_required = self.config.min_triples_per_concept;
        if all_triples.len() < min_required {
            return Err(CompetenceError::InsufficientTriples {
                triple_count: all_triples.len(),
                min_required,
            });
        }

        // Group curriculum entries by knowledge area (tier buckets).
        let knowledge_areas = self.group_by_knowledge_area(&prereq_result.curriculum);

        // Assess each knowledge area.
        let mut ka_assessments = Vec::new();
        let mut total_gaps = Vec::new();

        for (ka_name, entries) in &knowledge_areas {
            let assessment = self.assess_knowledge_area(ka_name, entries, engine)?;
            total_gaps.extend(
                std::iter::repeat_n(ka_name.clone(), assessment.gap_count)
                    .take(assessment.gap_count),
            );
            ka_assessments.push(assessment);
        }

        // Aggregate overall score (weighted average by concept count per KA).
        let total_concepts: usize = ka_assessments
            .iter()
            .map(|ka| ka.cq_total.max(1))
            .sum();
        let overall_score = if total_concepts > 0 {
            ka_assessments
                .iter()
                .map(|ka| ka.score * ka.cq_total.max(1) as f32)
                .sum::<f32>()
                / total_concepts as f32
        } else {
            0.0
        };
        let overall_dreyfus = score_to_dreyfus(overall_score);

        // Generate recommendation.
        let target_score = dreyfus_to_min_score(&purpose_model.competence_level);
        let recommendation = self.generate_recommendation(
            overall_score,
            target_score,
            &ka_assessments,
            &total_gaps,
        );

        // Collect remaining gaps as descriptive strings.
        let remaining_gaps: Vec<String> = ka_assessments
            .iter()
            .filter(|ka| ka.score < target_score)
            .map(|ka| format!("{} (score: {:.2}, level: {})", ka.name, ka.score, ka.dreyfus_level))
            .collect();

        // Store assessment provenance.
        let provenance_ids = self.store_provenance(
            overall_score,
            overall_dreyfus,
            &ka_assessments,
            &recommendation,
            engine,
        )?;

        Ok(CompetenceReport {
            overall_dreyfus,
            overall_score,
            knowledge_areas: ka_assessments,
            remaining_gaps,
            recommendation,
            provenance_ids,
        })
    }

    /// Group curriculum entries into knowledge areas by tier bucket.
    ///
    /// Tier 0 = "foundational", tier 1-2 = "intermediate", tier 3+ = "advanced".
    fn group_by_knowledge_area<'a>(
        &self,
        curriculum: &'a [CurriculumEntry],
    ) -> Vec<(String, Vec<&'a CurriculumEntry>)> {
        let mut groups: HashMap<String, Vec<&CurriculumEntry>> = HashMap::new();

        for entry in curriculum {
            let area_name = if entry.tier == 0 {
                "foundational".to_string()
            } else if entry.tier <= 2 {
                "intermediate".to_string()
            } else {
                "advanced".to_string()
            };
            groups.entry(area_name).or_default().push(entry);
        }

        // Sort by tier level for deterministic ordering.
        let mut sorted: Vec<(String, Vec<&CurriculumEntry>)> = groups.into_iter().collect();
        sorted.sort_by(|a, b| {
            let order = |name: &str| match name {
                "foundational" => 0,
                "intermediate" => 1,
                "advanced" => 2,
                _ => 3,
            };
            order(&a.0).cmp(&order(&b.0))
        });
        sorted
    }

    /// Assess a single knowledge area.
    fn assess_knowledge_area(
        &self,
        ka_name: &str,
        entries: &[&CurriculumEntry],
        engine: &Engine,
    ) -> CompetenceResult<KnowledgeAreaAssessment> {
        let concept_count = entries.len();
        let concept_ids: Vec<SymbolId> = entries.iter().map(|e| e.concept).collect();
        let concept_set: HashSet<SymbolId> = concept_ids.iter().copied().collect();

        // a) Count triples per concept.
        let mut total_triples = 0usize;
        for &concept in &concept_ids {
            total_triples += engine.triples_from(concept).len();
        }

        // b) Gap analysis for coverage + dead-end ratio.
        let gap_config = GapAnalysisConfig {
            min_degree: 2,
            cluster_size: 3,
            max_gaps: 50,
            ..Default::default()
        };
        let (coverage_score, dead_end_ratio, gap_count) =
            match gap::analyze_gaps(engine, &concept_ids, &gap_config) {
                Ok(result) => {
                    let de_ratio = if result.entities_analyzed > 0 {
                        result.dead_ends as f32 / result.entities_analyzed as f32
                    } else {
                        1.0
                    };
                    (result.coverage_score, de_ratio, result.gaps.len())
                }
                // If gap analysis fails (e.g. no goals), use fallback values.
                Err(_) => (0.0, 1.0, 0),
            };

        // c) Schema discovery for type diversity.
        let schema_config = SchemaDiscoveryConfig {
            min_type_members: 2,
            ..Default::default()
        };
        let type_count = match schema::discover_schema(engine, &schema_config) {
            Ok(result) => result.types.len(),
            Err(_) => 0,
        };
        let expected_types = (concept_count / 3).max(1);
        let type_diversity = (type_count as f32 / expected_types as f32).min(1.0);

        // d) Evaluate competency questions (Bloom's 4 levels).
        let cqs = self.evaluate_competency_questions(entries, engine);
        let cq_answered = cqs.iter().filter(|q| q.answered).count();
        let cq_total = cqs.len();

        // e) Compute relation density.
        let relation_density_raw = total_triples as f32 / concept_count.max(1) as f32;
        let relation_density_norm = (relation_density_raw / 5.0).min(1.0);

        // f) Compute cross-domain score.
        let cross_domain = self.compute_cross_domain(&concept_ids, &concept_set, engine);

        // g) Weighted Dreyfus score.
        let connectivity = 1.0 - dead_end_ratio;
        let dreyfus_score = self.config.coverage_weight * coverage_score
            + self.config.connectivity_weight * connectivity
            + self.config.type_diversity_weight * type_diversity
            + self.config.relation_density_weight * relation_density_norm
            + self.config.cross_domain_weight * cross_domain;

        let dreyfus_level = score_to_dreyfus(dreyfus_score);

        Ok(KnowledgeAreaAssessment {
            name: ka_name.to_string(),
            dreyfus_level,
            score: dreyfus_score,
            triple_count: total_triples,
            cq_answered,
            cq_total,
            gap_count,
            relation_density: relation_density_raw,
            score_components: ScoreComponents {
                coverage: coverage_score,
                connectivity,
                type_diversity,
                relation_density: relation_density_norm,
                cross_domain,
            },
        })
    }

    /// Evaluate competency questions at Bloom's taxonomy levels.
    fn evaluate_competency_questions(
        &self,
        entries: &[&CurriculumEntry],
        engine: &Engine,
    ) -> Vec<CompetencyQuestion> {
        let mut questions = Vec::new();
        let bloom_depth = self.config.bloom_max_depth.min(4);

        for entry in entries {
            let concept = entry.concept;

            // Remember: does concept entity exist?
            if bloom_depth >= 1 {
                let answered = engine.lookup_symbol(&entry.label).is_ok();
                questions.push(CompetencyQuestion {
                    _bloom_level: BloomLevel::Remember,
                    _concept: concept,
                    _related_concept: None,
                    answered,
                });
            }

            // Understand: path exists between concept and a peer?
            if bloom_depth >= 2 {
                let peer = self.find_peer(entry, entries);
                let answered = if let Some(peer_id) = peer {
                    let kg = engine.knowledge_graph();
                    shortest_path(kg, concept, peer_id)
                        .ok()
                        .flatten()
                        .is_some()
                } else {
                    false
                };
                questions.push(CompetencyQuestion {
                    _bloom_level: BloomLevel::Understand,
                    _concept: concept,
                    _related_concept: peer,
                    answered,
                });
            }

            // Apply: concept has >= 2 outgoing relation triples?
            if bloom_depth >= 3 {
                let outgoing = engine.triples_from(concept);
                let answered = outgoing.len() >= 2;
                questions.push(CompetencyQuestion {
                    _bloom_level: BloomLevel::Apply,
                    _concept: concept,
                    _related_concept: None,
                    answered,
                });
            }

            // Analyze: all prerequisites are in Known zone?
            if bloom_depth >= 4 {
                let answered = if entry.prerequisites.is_empty() {
                    // No prerequisites → trivially passes.
                    true
                } else {
                    entry.prerequisites.iter().all(|prereq| {
                        // Check if each prereq concept has enough triples.
                        engine.triples_from(*prereq).len() >= self.config.min_triples_per_concept
                    })
                };
                questions.push(CompetencyQuestion {
                    _bloom_level: BloomLevel::Analyze,
                    _concept: concept,
                    _related_concept: None,
                    answered,
                });
            }
        }

        questions
    }

    /// Find a peer concept for the Understand CQ — pick another entry at the same tier.
    fn find_peer(
        &self,
        entry: &CurriculumEntry,
        entries: &[&CurriculumEntry],
    ) -> Option<SymbolId> {
        entries
            .iter()
            .find(|other| other.concept != entry.concept && other.tier == entry.tier)
            .map(|other| other.concept)
    }

    /// Compute fraction of concepts with edges to concepts outside the current KA.
    fn compute_cross_domain(
        &self,
        concept_ids: &[SymbolId],
        concept_set: &HashSet<SymbolId>,
        engine: &Engine,
    ) -> f32 {
        if concept_ids.is_empty() {
            return 0.0;
        }

        let mut cross_domain_count = 0usize;
        for &concept in concept_ids {
            let outgoing = engine.triples_from(concept);
            let incoming = engine.triples_to(concept);
            let has_external = outgoing
                .iter()
                .any(|t| !concept_set.contains(&t.object))
                || incoming
                    .iter()
                    .any(|t| !concept_set.contains(&t.subject));
            if has_external {
                cross_domain_count += 1;
            }
        }

        cross_domain_count as f32 / concept_ids.len() as f32
    }

    /// Generate the bootstrap recommendation.
    fn generate_recommendation(
        &self,
        overall_score: f32,
        target_score: f32,
        ka_assessments: &[KnowledgeAreaAssessment],
        total_gaps: &[String],
    ) -> BootstrapRecommendation {
        if overall_score >= target_score {
            return BootstrapRecommendation::Ready;
        }

        // If too many gaps and very low score, ask operator.
        if overall_score < 0.1 && total_gaps.len() > 10 {
            return BootstrapRecommendation::NeedsOperatorInput {
                question: "Assessment found very sparse knowledge and many gaps. \
                           Are the seed concepts correct? Should different sources be tried?"
                    .to_string(),
            };
        }

        // Find focus areas (KAs below threshold).
        let focus_areas: Vec<String> = ka_assessments
            .iter()
            .filter(|ka| ka.score < target_score)
            .map(|ka| ka.name.clone())
            .collect();

        // Estimate cycles: rough heuristic based on gap between current and target.
        let gap = (target_score - overall_score).max(0.0);
        let estimated_cycles = (gap * 100.0) as usize + focus_areas.len() * 10;

        BootstrapRecommendation::ContinueLearning {
            estimated_cycles,
            focus_areas,
        }
    }

    /// Store assessment triples and provenance.
    fn store_provenance(
        &self,
        overall_score: f32,
        overall_dreyfus: DreyfusLevel,
        ka_assessments: &[KnowledgeAreaAssessment],
        recommendation: &BootstrapRecommendation,
        engine: &Engine,
    ) -> CompetenceResult<Vec<ProvenanceId>> {
        let mut provenance_ids = Vec::new();

        // Create an assessment entity.
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let assess_label = format!("assess:competence_{ts}");
        let assess_id = engine.resolve_or_create_entity(&assess_label)?;

        // Store overall score triple.
        let score_label = format!("{overall_score:.4}");
        let score_obj = engine.resolve_or_create_entity(&score_label)?;
        let score_triple = Triple::new(assess_id, self.predicates.competence_score, score_obj);
        let _ = engine.add_triple(&score_triple);

        // Store Dreyfus level triple.
        let level_obj = engine.resolve_or_create_entity(overall_dreyfus.as_label())?;
        let level_triple = Triple::new(assess_id, self.predicates.dreyfus_level, level_obj);
        let _ = engine.add_triple(&level_triple);

        // Store timestamp triple.
        let ts_label = format!("{ts}");
        let ts_obj = engine.resolve_or_create_entity(&ts_label)?;
        let ts_triple = Triple::new(assess_id, self.predicates.assessed_at, ts_obj);
        let _ = engine.add_triple(&ts_triple);

        // Compute CQ totals.
        let cq_answered: u32 = ka_assessments.iter().map(|ka| ka.cq_answered as u32).sum();
        let cq_total: u32 = ka_assessments.iter().map(|ka| ka.cq_total as u32).sum();

        // Store provenance record.
        let mut record = ProvenanceRecord::new(
            assess_id,
            DerivationKind::CompetenceAssessment {
                overall_score,
                overall_dreyfus: overall_dreyfus.as_label().to_string(),
                knowledge_area_count: ka_assessments.len() as u32,
                cq_answered,
                cq_total,
                recommendation: recommendation.to_string(),
            },
        )
        .with_confidence(overall_score);

        if let Ok(id) = engine.store_provenance(&mut record) {
            provenance_ids.push(id);
        }

        Ok(provenance_ids)
    }
}

// ── Score Mapping ───────────────────────────────────────────────────────

/// Map a numeric score [0.0, 1.0] to a Dreyfus level.
pub fn score_to_dreyfus(score: f32) -> DreyfusLevel {
    if score >= 0.8 {
        DreyfusLevel::Expert
    } else if score >= 0.6 {
        DreyfusLevel::Proficient
    } else if score >= 0.4 {
        DreyfusLevel::Competent
    } else if score >= 0.2 {
        DreyfusLevel::AdvancedBeginner
    } else {
        DreyfusLevel::Novice
    }
}

/// Map a target Dreyfus level to its minimum score threshold.
pub fn dreyfus_to_min_score(level: &DreyfusLevel) -> f32 {
    match level {
        DreyfusLevel::Novice => 0.0,
        DreyfusLevel::AdvancedBeginner => 0.2,
        DreyfusLevel::Competent => 0.4,
        DreyfusLevel::Proficient => 0.6,
        DreyfusLevel::Expert => 0.8,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Config defaults
    #[test]
    fn config_defaults() {
        let cfg = CompetenceConfig::default();
        assert_eq!(cfg.min_triples_per_concept, 3);
        assert_eq!(cfg.bloom_max_depth, 4);
        assert!((cfg.coverage_weight - 0.30).abs() < 0.001);
        assert!((cfg.connectivity_weight - 0.20).abs() < 0.001);
        assert!((cfg.type_diversity_weight - 0.20).abs() < 0.001);
        assert!((cfg.relation_density_weight - 0.15).abs() < 0.001);
        assert!((cfg.cross_domain_weight - 0.15).abs() < 0.001);
        // Weights should sum to 1.0
        let total = cfg.coverage_weight
            + cfg.connectivity_weight
            + cfg.type_diversity_weight
            + cfg.relation_density_weight
            + cfg.cross_domain_weight;
        assert!((total - 1.0).abs() < 0.001);
    }

    // 2. Error enum formatting
    #[test]
    fn error_formatting() {
        let e1 = CompetenceError::NoConcepts;
        assert!(format!("{e1}").contains("no curriculum"));

        let e2 = CompetenceError::InsufficientTriples {
            triple_count: 2,
            min_required: 3,
        };
        let msg = format!("{e2}");
        assert!(msg.contains("2"));
        assert!(msg.contains("3"));
    }

    // 3. BloomLevel ordering
    #[test]
    fn bloom_level_ordering() {
        assert!(BloomLevel::Remember < BloomLevel::Understand);
        assert!(BloomLevel::Understand < BloomLevel::Apply);
        assert!(BloomLevel::Apply < BloomLevel::Analyze);
    }

    // 4. Dreyfus score threshold mapping (5 boundaries)
    #[test]
    fn dreyfus_score_boundaries() {
        assert_eq!(score_to_dreyfus(0.0), DreyfusLevel::Novice);
        assert_eq!(score_to_dreyfus(0.19), DreyfusLevel::Novice);
        assert_eq!(score_to_dreyfus(0.2), DreyfusLevel::AdvancedBeginner);
        assert_eq!(score_to_dreyfus(0.39), DreyfusLevel::AdvancedBeginner);
        assert_eq!(score_to_dreyfus(0.4), DreyfusLevel::Competent);
        assert_eq!(score_to_dreyfus(0.59), DreyfusLevel::Competent);
        assert_eq!(score_to_dreyfus(0.6), DreyfusLevel::Proficient);
        assert_eq!(score_to_dreyfus(0.79), DreyfusLevel::Proficient);
        assert_eq!(score_to_dreyfus(0.8), DreyfusLevel::Expert);
        assert_eq!(score_to_dreyfus(1.0), DreyfusLevel::Expert);
    }

    // 5. Dreyfus score weighted formula correctness
    #[test]
    fn dreyfus_weighted_formula() {
        let cfg = CompetenceConfig::default();
        // All components at 1.0 should give 1.0.
        let score = cfg.coverage_weight * 1.0
            + cfg.connectivity_weight * 1.0
            + cfg.type_diversity_weight * 1.0
            + cfg.relation_density_weight * 1.0
            + cfg.cross_domain_weight * 1.0;
        assert!((score - 1.0).abs() < 0.001);

        // All components at 0.5 should give 0.5.
        let score_half = cfg.coverage_weight * 0.5
            + cfg.connectivity_weight * 0.5
            + cfg.type_diversity_weight * 0.5
            + cfg.relation_density_weight * 0.5
            + cfg.cross_domain_weight * 0.5;
        assert!((score_half - 0.5).abs() < 0.001);
    }

    // 6. BloomLevel display
    #[test]
    fn bloom_level_display() {
        assert_eq!(BloomLevel::Remember.to_string(), "remember");
        assert_eq!(BloomLevel::Understand.to_string(), "understand");
        assert_eq!(BloomLevel::Apply.to_string(), "apply");
        assert_eq!(BloomLevel::Analyze.to_string(), "analyze");
    }

    // 7. BloomLevel::all() returns 4 levels in order
    #[test]
    fn bloom_level_all() {
        let all = BloomLevel::all();
        assert_eq!(all.len(), 4);
        assert_eq!(all[0], BloomLevel::Remember);
        assert_eq!(all[3], BloomLevel::Analyze);
    }

    // 8. Type diversity capped at 1.0
    #[test]
    fn type_diversity_capped() {
        // 10 types found, 3 concepts → expected types = 1 → ratio = 10.0, capped at 1.0
        let type_count = 10;
        let concept_count = 3;
        let expected = (concept_count / 3).max(1);
        let diversity = (type_count as f32 / expected as f32).min(1.0);
        assert!((diversity - 1.0).abs() < 0.001);
    }

    // 9. Relation density normalized correctly
    #[test]
    fn relation_density_normalized() {
        // 15 triples / 3 concepts = 5.0 → 5.0 / 5.0 = 1.0
        let raw: f32 = 15.0 / 3.0;
        let norm = (raw / 5.0).min(1.0);
        assert!((norm - 1.0).abs() < 0.001);

        // 3 triples / 3 concepts = 1.0 → 1.0 / 5.0 = 0.2
        let raw2: f32 = 3.0 / 3.0;
        let norm2 = (raw2 / 5.0).min(1.0);
        assert!((norm2 - 0.2).abs() < 0.001);

        // 30 triples / 3 concepts = 10.0 → 10.0 / 5.0 = 2.0 → capped at 1.0
        let raw3: f32 = 30.0 / 3.0;
        let norm3 = (raw3 / 5.0).min(1.0);
        assert!((norm3 - 1.0).abs() < 0.001);
    }

    // 10. BootstrapRecommendation::Ready when score >= target
    #[test]
    fn recommendation_ready() {
        let rec = BootstrapRecommendation::Ready;
        assert_eq!(format!("{rec}"), "ready");
    }

    // 11. BootstrapRecommendation::ContinueLearning with focus_areas
    #[test]
    fn recommendation_continue_learning() {
        let rec = BootstrapRecommendation::ContinueLearning {
            estimated_cycles: 50,
            focus_areas: vec!["advanced".to_string(), "intermediate".to_string()],
        };
        let msg = format!("{rec}");
        assert!(msg.contains("50 cycles"));
        assert!(msg.contains("advanced"));
        assert!(msg.contains("intermediate"));
    }

    // 12. BootstrapRecommendation equality
    #[test]
    fn recommendation_equality() {
        assert_eq!(BootstrapRecommendation::Ready, BootstrapRecommendation::Ready);
        assert_ne!(
            BootstrapRecommendation::Ready,
            BootstrapRecommendation::NeedsOperatorInput {
                question: "test".to_string(),
            }
        );
    }

    // 13. dreyfus_to_min_score roundtrip
    #[test]
    fn dreyfus_min_score_roundtrip() {
        assert!((dreyfus_to_min_score(&DreyfusLevel::Novice) - 0.0).abs() < 0.001);
        assert!((dreyfus_to_min_score(&DreyfusLevel::AdvancedBeginner) - 0.2).abs() < 0.001);
        assert!((dreyfus_to_min_score(&DreyfusLevel::Competent) - 0.4).abs() < 0.001);
        assert!((dreyfus_to_min_score(&DreyfusLevel::Proficient) - 0.6).abs() < 0.001);
        assert!((dreyfus_to_min_score(&DreyfusLevel::Expert) - 0.8).abs() < 0.001);

        // Score at boundary maps back to correct level.
        for level in [
            DreyfusLevel::Novice,
            DreyfusLevel::AdvancedBeginner,
            DreyfusLevel::Competent,
            DreyfusLevel::Proficient,
            DreyfusLevel::Expert,
        ] {
            let min_score = dreyfus_to_min_score(&level);
            assert_eq!(score_to_dreyfus(min_score), level);
        }
    }

    // 14. NeedsOperatorInput display
    #[test]
    fn needs_operator_input_display() {
        let rec = BootstrapRecommendation::NeedsOperatorInput {
            question: "Are seeds correct?".to_string(),
        };
        let msg = format!("{rec}");
        assert!(msg.contains("needs operator input"));
        assert!(msg.contains("Are seeds correct?"));
    }

    // 15. ScoreComponents can be constructed
    #[test]
    fn score_components_construction() {
        let sc = ScoreComponents {
            coverage: 0.8,
            connectivity: 0.9,
            type_diversity: 0.5,
            relation_density: 0.7,
            cross_domain: 0.3,
        };
        assert!((sc.coverage - 0.8).abs() < 0.001);
        assert!((sc.cross_domain - 0.3).abs() < 0.001);
    }
}
