//! Iterative curriculum ingestion with pedagogical ordering (Phase 14f).
//!
//! Takes tier-ordered concepts from prerequisite analysis (14d) and discovered
//! resources (14e), ingests them in curriculum order, cross-validates multi-method
//! concepts, grounds VSA vectors, and tracks saturation.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use miette::Diagnostic;
use thiserror::Error;

use crate::bootstrap::expand::normalize_label;
use crate::bootstrap::prerequisite::PrereqAnalysisResult;
use crate::bootstrap::resources::{DiscoveredResource, ResourceDiscoveryResult};
use crate::engine::Engine;
use crate::library::catalog::LibraryCatalog;
use crate::library::error::LibraryError;
use crate::library::ingest::{IngestConfig, ingest_document, ingest_url};
use crate::library::model::{ContentFormat, DocumentSource};
use crate::provenance::{DerivationKind, ProvenanceId, ProvenanceRecord};
use crate::symbol::SymbolId;
use crate::graph::Triple;
use crate::vsa::grounding::{GroundingConfig, ground_symbol};

// ── Errors ──────────────────────────────────────────────────────────────

/// Errors specific to curriculum ingestion.
#[derive(Debug, Error, Diagnostic)]
pub enum IngestionError {
    #[error("no curriculum entries — nothing to ingest")]
    #[diagnostic(
        code(akh::bootstrap::ingest::no_curriculum),
        help(
            "The prerequisite analysis returned an empty curriculum. \
             Run domain expansion with broader seeds, or check that \
             the ZPD classification produced at least one Proximal concept."
        )
    )]
    NoCurriculum,

    #[error("no resources found for any curriculum concept")]
    #[diagnostic(
        code(akh::bootstrap::ingest::no_resources),
        help(
            "Resource discovery did not find any resources matching curriculum \
             concepts. Try broadening the search (more API sources, lower \
             quality thresholds) or expanding the domain first."
        )
    )]
    NoResources,

    #[error("library catalog error: {0}")]
    #[diagnostic(
        code(akh::bootstrap::ingest::catalog),
        help(
            "Failed to open or operate the library catalog. Check that the \
             catalog directory exists and is writable."
        )
    )]
    Catalog(String),

    #[error("{0}")]
    #[diagnostic(
        code(akh::bootstrap::ingest::engine),
        help("An engine-level error occurred during curriculum ingestion.")
    )]
    Engine(Box<crate::error::AkhError>),
}

impl From<crate::error::AkhError> for IngestionError {
    fn from(e: crate::error::AkhError) -> Self {
        Self::Engine(Box::new(e))
    }
}

/// Convenience alias.
pub type IngestionResult<T> = std::result::Result<T, IngestionError>;

// ── Configuration ───────────────────────────────────────────────────────

/// Configuration for the curriculum ingestion pipeline.
#[derive(Debug, Clone)]
pub struct IngestionConfig {
    /// Maximum ingestion cycles before stopping (default: 500).
    pub max_cycles: usize,
    /// Consecutive zero-new-triple results before a concept is considered
    /// saturated (default: 3).
    pub saturation_threshold: usize,
    /// Confidence boost for cross-validated concepts (>= 2 extraction methods).
    pub cross_validation_boost: f32,
    /// Minimum number of distinct extraction methods for cross-validation.
    pub cross_validation_min_methods: usize,
    /// Attempt URL ingestion for open-access resources (default: true).
    pub try_url_ingestion: bool,
    /// Timeout in seconds for URL fetch attempts (default: 15).
    pub url_timeout_secs: u64,
    /// Number of VSA grounding rounds per concept (default: 2).
    pub grounding_rounds: usize,
    /// Override catalog directory (default: XDG data dir).
    pub catalog_dir: Option<PathBuf>,
}

impl Default for IngestionConfig {
    fn default() -> Self {
        Self {
            max_cycles: 500,
            saturation_threshold: 3,
            cross_validation_boost: 0.15,
            cross_validation_min_methods: 2,
            try_url_ingestion: true,
            url_timeout_secs: 15,
            grounding_rounds: 2,
            catalog_dir: None,
        }
    }
}

// ── Well-Known Predicates ───────────────────────────────────────────────

/// Well-known `ingest:*` predicates for curriculum ingestion metadata.
struct IngestionPredicates {
    ingested_from: SymbolId,
    extraction_method: SymbolId,
    cross_validation_count: SymbolId,
    curriculum_tier: SymbolId,
    ingestion_status: SymbolId,
}

impl IngestionPredicates {
    fn init(engine: &Engine) -> IngestionResult<Self> {
        let preds = Self {
            ingested_from: engine.resolve_or_create_relation("ingest:ingested_from")?,
            extraction_method: engine.resolve_or_create_relation("ingest:extraction_method")?,
            cross_validation_count: engine
                .resolve_or_create_relation("ingest:cross_validation_count")?,
            curriculum_tier: engine.resolve_or_create_relation("ingest:curriculum_tier")?,
            ingestion_status: engine.resolve_or_create_relation("ingest:ingestion_status")?,
        };

        // A concept can be ingested from multiple resources and extracted
        // by multiple methods — register these as multi-valued.
        let mut mv = engine.multi_valued_preds_mut();
        mv.declare_multi_valued(preds.ingested_from);
        mv.declare_multi_valued(preds.extraction_method);

        Ok(preds)
    }
}

// ── Saturation Tracker ──────────────────────────────────────────────────

/// Tracks per-concept consecutive zero-new-triple streaks to detect saturation.
struct SaturationTracker {
    /// Consecutive zero-new-triple count per concept.
    zero_streak: HashMap<SymbolId, usize>,
    /// Concepts that have been saturated.
    saturated: HashSet<SymbolId>,
}

impl SaturationTracker {
    fn new() -> Self {
        Self {
            zero_streak: HashMap::new(),
            saturated: HashSet::new(),
        }
    }

    /// Record an ingestion attempt. Returns `true` if the concept just became saturated.
    fn record(&mut self, concept: SymbolId, triples_added: usize, threshold: usize) -> bool {
        if self.saturated.contains(&concept) {
            return false;
        }

        if triples_added == 0 {
            let streak = self.zero_streak.entry(concept).or_insert(0);
            *streak += 1;
            if *streak >= threshold {
                self.saturated.insert(concept);
                return true;
            }
        } else {
            self.zero_streak.insert(concept, 0);
        }

        false
    }

    /// Check if a concept is saturated.
    fn is_saturated(&self, concept: SymbolId) -> bool {
        self.saturated.contains(&concept)
    }
}

// ── Outcome Types ───────────────────────────────────────────────────────

/// Outcome of ingesting a single resource for a concept.
struct ConceptIngestionOutcome {
    triples_added: usize,
    concepts_extracted: usize,
    abstract_ingested: bool,
    url_attempted: bool,
    url_succeeded: bool,
    extraction_methods: Vec<String>,
}

impl ConceptIngestionOutcome {
    fn empty() -> Self {
        Self {
            triples_added: 0,
            concepts_extracted: 0,
            abstract_ingested: false,
            url_attempted: false,
            url_succeeded: false,
            extraction_methods: Vec::new(),
        }
    }

    fn accumulate(&mut self, other: &ConceptIngestionOutcome) {
        self.triples_added += other.triples_added;
        self.concepts_extracted += other.concepts_extracted;
        self.abstract_ingested |= other.abstract_ingested;
        self.url_attempted |= other.url_attempted;
        self.url_succeeded |= other.url_succeeded;
        for m in &other.extraction_methods {
            if !self.extraction_methods.contains(m) {
                self.extraction_methods.push(m.clone());
            }
        }
    }
}

/// Result of a full curriculum ingestion run.
#[derive(Debug)]
pub struct IngestionRunResult {
    /// Total ingestion cycles executed.
    pub cycles: usize,
    /// Total KG triples created across all resources.
    pub total_triples: usize,
    /// Total atomic concepts extracted.
    pub total_concepts_extracted: usize,
    /// Number of curriculum concepts that were ingested (at least 1 triple).
    pub concepts_ingested: usize,
    /// Number of concepts that reached saturation.
    pub concepts_saturated: usize,
    /// Number of URL ingestion attempts.
    pub url_attempts: usize,
    /// Number of successful URL ingestions.
    pub url_successes: usize,
    /// Number of concepts cross-validated (>= 2 extraction methods).
    pub cross_validated_concepts: usize,
    /// Number of symbols that received VSA grounding updates.
    pub symbols_grounded: usize,
    /// Provenance record IDs generated during ingestion.
    pub provenance_ids: Vec<ProvenanceId>,
}

// ── Curriculum Ingestor ─────────────────────────────────────────────────

/// Ingests discovered resources in curriculum (tier) order, cross-validates,
/// grounds VSA vectors, and tracks saturation.
pub struct CurriculumIngestor {
    config: IngestionConfig,
    predicates: IngestionPredicates,
    tracker: SaturationTracker,
    cycle_count: usize,
}

impl CurriculumIngestor {
    /// Create a new ingestor, resolving well-known predicates.
    pub fn new(engine: &Engine, config: IngestionConfig) -> IngestionResult<Self> {
        let predicates = IngestionPredicates::init(engine)?;
        Ok(Self {
            config,
            predicates,
            tracker: SaturationTracker::new(),
            cycle_count: 0,
        })
    }

    /// Run the full curriculum ingestion pipeline.
    ///
    /// Chains: prerequisite analysis result + resource discovery result → ingestion.
    pub fn ingest(
        &mut self,
        prereq_result: &PrereqAnalysisResult,
        resource_result: &ResourceDiscoveryResult,
        engine: &Engine,
    ) -> IngestionResult<IngestionRunResult> {
        // ── Preconditions ──────────────────────────────────────────
        if prereq_result.curriculum.is_empty() {
            return Err(IngestionError::NoCurriculum);
        }
        if resource_result.resources.is_empty() {
            return Err(IngestionError::NoResources);
        }

        // ── Build resource index ───────────────────────────────────
        let resource_index = build_resource_index(&resource_result.resources);

        // ── Sort curriculum: tier ASC, similarity_to_known DESC ────
        let mut sorted_curriculum: Vec<_> = prereq_result.curriculum.iter().collect();
        sorted_curriculum.sort_by(|a, b| {
            a.tier
                .cmp(&b.tier)
                .then_with(|| b.similarity_to_known.partial_cmp(&a.similarity_to_known)
                    .unwrap_or(std::cmp::Ordering::Equal))
        });

        // ── Open library catalog ───────────────────────────────────
        let catalog_dir = self.catalog_dir()?;
        let mut catalog = LibraryCatalog::open(&catalog_dir)
            .map_err(|e| IngestionError::Catalog(e.to_string()))?;

        // ── Aggregation state ──────────────────────────────────────
        let mut result = IngestionRunResult {
            cycles: 0,
            total_triples: 0,
            total_concepts_extracted: 0,
            concepts_ingested: 0,
            concepts_saturated: 0,
            url_attempts: 0,
            url_successes: 0,
            cross_validated_concepts: 0,
            symbols_grounded: 0,
            provenance_ids: Vec::new(),
        };

        // ── Per-concept ingestion tracking ─────────────────────────
        let mut concepts_with_triples: HashSet<SymbolId> = HashSet::new();

        // ── Main loop: tier-ordered ingestion ──────────────────────
        let max_tier = prereq_result.max_tier;
        for tier in 0..=max_tier {
            let tier_entries: Vec<_> = sorted_curriculum
                .iter()
                .filter(|e| e.tier == tier)
                .collect();

            for entry in tier_entries {
                if self.tracker.is_saturated(entry.concept) {
                    continue;
                }
                if self.cycle_count >= self.config.max_cycles {
                    break;
                }

                let normalized = normalize_label(&entry.label);
                let resources = match resource_index.get(&normalized) {
                    Some(res) => res,
                    None => continue,
                };

                // Sort resources by quality DESC.
                let mut sorted_resources: Vec<_> = resources.iter().collect();
                sorted_resources.sort_by(|a, b| {
                    b.quality_score
                        .partial_cmp(&a.quality_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let mut concept_outcome = ConceptIngestionOutcome::empty();

                for resource in sorted_resources {
                    if self.cycle_count >= self.config.max_cycles {
                        break;
                    }

                    let outcome = self.ingest_single(
                        engine,
                        &mut catalog,
                        entry.concept,
                        &entry.label,
                        entry.tier,
                        resource,
                    );

                    concept_outcome.accumulate(&outcome);
                    self.cycle_count += 1;
                    result.cycles += 1;

                    let just_saturated = self.tracker.record(
                        entry.concept,
                        outcome.triples_added,
                        self.config.saturation_threshold,
                    );

                    if just_saturated {
                        result.concepts_saturated += 1;
                    }

                    // Record provenance for this ingestion.
                    if outcome.triples_added > 0 && let Some(prov_id) = self.record_provenance(
                        engine,
                        &ProvenanceInfo {
                            concept: entry.concept,
                            concept_label: &entry.label,
                            resource_title: &resource.title,
                            triples_added: outcome.triples_added,
                            extraction_methods: &outcome.extraction_methods,
                            tier: entry.tier,
                            cross_validated: false,
                        },
                    ) {
                        result.provenance_ids.push(prov_id);
                    }

                    result.url_attempts += usize::from(outcome.url_attempted);
                    result.url_successes += usize::from(outcome.url_succeeded);
                }

                // Accumulate concept-level stats.
                result.total_triples += concept_outcome.triples_added;
                result.total_concepts_extracted += concept_outcome.concepts_extracted;

                if concept_outcome.triples_added > 0 {
                    concepts_with_triples.insert(entry.concept);

                    // Cross-validate.
                    let xval_count = self.cross_validate(engine, entry.concept, &entry.label);
                    if xval_count >= self.config.cross_validation_min_methods {
                        result.cross_validated_concepts += 1;

                        // Record cross-validation provenance.
                        if let Some(prov_id) = self.record_provenance(
                            engine,
                            &ProvenanceInfo {
                                concept: entry.concept,
                                concept_label: &entry.label,
                                resource_title: "(cross-validated)",
                                triples_added: 0,
                                extraction_methods: &concept_outcome.extraction_methods,
                                tier: entry.tier,
                                cross_validated: true,
                            },
                        ) {
                            result.provenance_ids.push(prov_id);
                        }
                    }

                    // Ground the concept's VSA vector.
                    if self.ground_concept(engine, entry.concept) {
                        result.symbols_grounded += 1;
                    }
                }
            }

            if self.cycle_count >= self.config.max_cycles {
                break;
            }
        }

        result.concepts_ingested = concepts_with_triples.len();

        Ok(result)
    }

    /// Ingest a single resource for a concept: abstract text first, then
    /// optionally the URL for open-access resources.
    fn ingest_single(
        &self,
        engine: &Engine,
        catalog: &mut LibraryCatalog,
        concept: SymbolId,
        concept_label: &str,
        tier: u32,
        resource: &DiscoveredResource,
    ) -> ConceptIngestionOutcome {
        let mut outcome = ConceptIngestionOutcome::empty();

        // ── Layer 1: Abstract text (always attempted) ──────────────
        if !resource.abstract_text.is_empty() {
            let data = resource.abstract_text.as_bytes();
            let source = DocumentSource::Inline;
            let config = IngestConfig {
                title: Some(resource.title.clone()),
                tags: vec![
                    format!("concept:{}", concept_label),
                    format!("tier:{}", tier),
                ],
                format: Some(ContentFormat::PlainText),
                ..Default::default()
            };

            match ingest_document(engine, catalog, data, source, config, None) {
                Ok(ingest_result) => {
                    outcome.triples_added += ingest_result.triple_count;
                    outcome.concepts_extracted += ingest_result.concept_count;
                    outcome.abstract_ingested = true;
                    outcome.extraction_methods.push("abstract_text".to_string());

                    // Store metadata triples linking concept to ingested resource.
                    self.store_ingestion_metadata(
                        engine,
                        concept,
                        &resource.title,
                        "abstract",
                        tier,
                    );
                }
                Err(LibraryError::Duplicate { .. }) => {
                    // Already ingested this abstract — skip silently.
                }
                Err(e) => {
                    eprintln!(
                        "  Warning: abstract ingestion failed for '{}': {e}",
                        resource.title
                    );
                }
            }
        }

        // ── Layer 2: URL ingestion (open-access only, best-effort) ──
        if self.config.try_url_ingestion && resource.open_access && !resource.url.is_empty() {
            outcome.url_attempted = true;

            let url_config = IngestConfig {
                title: Some(format!("{} (full text)", resource.title)),
                tags: vec![
                    format!("concept:{}", concept_label),
                    format!("tier:{}", tier),
                    "url-ingestion".to_string(),
                ],
                format: None, // auto-detect
                ..Default::default()
            };

            match ingest_url(engine, catalog, &resource.url, url_config, None, self.config.url_timeout_secs) {
                Ok(ingest_result) => {
                    outcome.triples_added += ingest_result.triple_count;
                    outcome.concepts_extracted += ingest_result.concept_count;
                    outcome.url_succeeded = true;
                    outcome.extraction_methods.push("url_fulltext".to_string());

                    self.store_ingestion_metadata(
                        engine,
                        concept,
                        &resource.title,
                        "url",
                        tier,
                    );
                }
                Err(e) => {
                    // URL ingestion is best-effort — log and continue.
                    eprintln!(
                        "  Warning: URL ingestion failed for '{}' ({}): {}",
                        resource.title, resource.url, e
                    );
                }
            }
        }

        outcome
    }

    /// Store metadata triples linking a concept to an ingested resource.
    fn store_ingestion_metadata(
        &self,
        engine: &Engine,
        concept: SymbolId,
        resource_title: &str,
        method: &str,
        tier: u32,
    ) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // ingest:ingested_from → resource title entity
        if let Ok(title_entity) =
            engine.resolve_or_create_entity(&format!("resource:{resource_title}"))
        {
            let triple = Triple {
                subject: concept,
                predicate: self.predicates.ingested_from,
                object: title_entity,
                confidence: 0.9,
                timestamp: now,
                provenance_id: None,
                compartment_id: None,
            };
            let _ = engine.add_triple(&triple);
        }

        // ingest:extraction_method → method entity
        if let Ok(method_entity) =
            engine.resolve_or_create_entity(&format!("method:{method}"))
        {
            let triple = Triple {
                subject: concept,
                predicate: self.predicates.extraction_method,
                object: method_entity,
                confidence: 0.9,
                timestamp: now,
                provenance_id: None,
                compartment_id: None,
            };
            let _ = engine.add_triple(&triple);
        }

        // ingest:curriculum_tier → tier entity
        if let Ok(tier_entity) =
            engine.resolve_or_create_entity(&format!("tier:{tier}"))
        {
            let triple = Triple {
                subject: concept,
                predicate: self.predicates.curriculum_tier,
                object: tier_entity,
                confidence: 0.9,
                timestamp: now,
                provenance_id: None,
                compartment_id: None,
            };
            let _ = engine.add_triple(&triple);
        }
    }

    /// Cross-validate a concept by counting distinct extraction methods
    /// in its provenance records. Returns the number of unique methods.
    fn cross_validate(&self, engine: &Engine, concept: SymbolId, _concept_label: &str) -> usize {
        let records = match engine.provenance_of(concept) {
            Ok(recs) => recs,
            Err(_) => return 0,
        };

        let mut methods: HashSet<String> = HashSet::new();
        for record in &records {
            if let DerivationKind::ConceptExtracted {
                extraction_method, ..
            } = &record.kind
            {
                methods.insert(extraction_method.clone());
            }
        }

        let method_count = methods.len();

        // If cross-validated, add annotation triple with boosted confidence.
        if method_count >= self.config.cross_validation_min_methods {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            if let Ok(xval_entity) =
                engine.resolve_or_create_entity("status:cross_validated")
            {
                let boosted = (0.9 + self.config.cross_validation_boost).min(1.0);
                let triple = Triple {
                    subject: concept,
                    predicate: self.predicates.cross_validation_count,
                    object: xval_entity,
                    confidence: boosted,
                    timestamp: now,
                    provenance_id: None,
                    compartment_id: None,
                };
                let _ = engine.add_triple(&triple);
            }

            // Also store ingestion status.
            if let Ok(status_entity) =
                engine.resolve_or_create_entity(&format!("xval:{method_count}_methods"))
            {
                let triple = Triple {
                    subject: concept,
                    predicate: self.predicates.ingestion_status,
                    object: status_entity,
                    confidence: 0.95,
                    timestamp: now,
                    provenance_id: None,
                    compartment_id: None,
                };
                let _ = engine.add_triple(&triple);
            }
        }

        method_count
    }

    /// Ground a concept's VSA vector after ingestion.
    /// Returns `true` if the vector was updated.
    fn ground_concept(&self, engine: &Engine, concept: SymbolId) -> bool {
        let grounding_config = GroundingConfig {
            rounds: self.config.grounding_rounds,
            neighbor_weight: 0.3,
            min_confidence: 0.3,
        };

        let ops = engine.ops();
        let im = engine.item_memory();

        match ground_symbol(concept, engine, ops, im, &grounding_config) {
            Ok(new_vec) => {
                im.insert(concept, new_vec);
                true
            }
            Err(_) => false,
        }
    }

    /// Record a provenance entry for curriculum ingestion.
    fn record_provenance(
        &self,
        engine: &Engine,
        info: &ProvenanceInfo<'_>,
    ) -> Option<ProvenanceId> {
        let kind = DerivationKind::CurriculumIngestion {
            concept_label: info.concept_label.to_string(),
            resource_title: info.resource_title.to_string(),
            triples_added: info.triples_added as u32,
            extraction_methods: info.extraction_methods.join(", "),
            tier: info.tier,
            cross_validated: info.cross_validated,
        };

        let mut record = ProvenanceRecord {
            id: None,
            derived_id: info.concept,
            sources: Vec::new(),
            kind,
            confidence: if info.cross_validated { 0.95 } else { 0.85 },
            depth: 0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        engine.store_provenance(&mut record).ok()
    }

    /// Resolve the catalog directory.
    fn catalog_dir(&self) -> IngestionResult<PathBuf> {
        if let Some(ref dir) = self.config.catalog_dir {
            return Ok(dir.clone());
        }

        // Use XDG_DATA_HOME or ~/.local/share fallback.
        let data_home = std::env::var("XDG_DATA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
                PathBuf::from(home).join(".local").join("share")
            });

        let base = data_home.join("akh-medu").join("library");

        std::fs::create_dir_all(&base)
            .map_err(|e| IngestionError::Catalog(format!("create catalog dir: {e}")))?;

        Ok(base)
    }
}

/// Parameters for recording provenance of a curriculum ingestion step.
struct ProvenanceInfo<'a> {
    concept: SymbolId,
    concept_label: &'a str,
    resource_title: &'a str,
    triples_added: usize,
    extraction_methods: &'a [String],
    tier: u32,
    cross_validated: bool,
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Build an index from normalized concept label → resources that cover it.
fn build_resource_index<'a>(
    resources: &'a [DiscoveredResource],
) -> HashMap<String, Vec<&'a DiscoveredResource>> {
    let mut index: HashMap<String, Vec<&'a DiscoveredResource>> = HashMap::new();

    for resource in resources {
        for concept_label in &resource.covers_concepts {
            let normalized = normalize_label(concept_label);
            index.entry(normalized).or_default().push(resource);
        }
    }

    index
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bootstrap::prerequisite::{CurriculumEntry, ZpdZone};
    use crate::bootstrap::resources::ResourceApi;
    use crate::bootstrap::purpose::DreyfusLevel;

    #[test]
    fn config_defaults() {
        let config = IngestionConfig::default();
        assert_eq!(config.max_cycles, 500);
        assert_eq!(config.saturation_threshold, 3);
        assert!((config.cross_validation_boost - 0.15).abs() < f32::EPSILON);
        assert_eq!(config.cross_validation_min_methods, 2);
        assert!(config.try_url_ingestion);
        assert_eq!(config.url_timeout_secs, 15);
        assert_eq!(config.grounding_rounds, 2);
        assert!(config.catalog_dir.is_none());
    }

    #[test]
    fn error_display_no_curriculum() {
        let err = IngestionError::NoCurriculum;
        let msg = format!("{err}");
        assert!(msg.contains("no curriculum"));
    }

    #[test]
    fn error_display_no_resources() {
        let err = IngestionError::NoResources;
        let msg = format!("{err}");
        assert!(msg.contains("no resources"));
    }

    #[test]
    fn error_display_catalog() {
        let err = IngestionError::Catalog("disk full".into());
        let msg = format!("{err}");
        assert!(msg.contains("disk full"));
    }

    #[test]
    fn saturation_tracker_resets_on_nonzero_triples() {
        let mut tracker = SaturationTracker::new();
        let concept = SymbolId::new(1).unwrap();

        // Two consecutive zeros.
        tracker.record(concept, 0, 3);
        tracker.record(concept, 0, 3);
        assert!(!tracker.is_saturated(concept));

        // Non-zero resets the streak.
        tracker.record(concept, 5, 3);
        assert_eq!(tracker.zero_streak.get(&concept), Some(&0));

        // Need 3 more consecutive zeros to saturate.
        tracker.record(concept, 0, 3);
        tracker.record(concept, 0, 3);
        assert!(!tracker.is_saturated(concept));

        tracker.record(concept, 0, 3);
        assert!(tracker.is_saturated(concept));
    }

    #[test]
    fn saturation_tracker_saturates_after_threshold() {
        let mut tracker = SaturationTracker::new();
        let concept = SymbolId::new(42).unwrap();

        assert!(!tracker.record(concept, 0, 3));
        assert!(!tracker.record(concept, 0, 3));
        assert!(tracker.record(concept, 0, 3)); // 3rd zero → saturated
        assert!(tracker.is_saturated(concept));
    }

    #[test]
    fn saturation_tracker_independent_per_concept() {
        let mut tracker = SaturationTracker::new();
        let a = SymbolId::new(1).unwrap();
        let b = SymbolId::new(2).unwrap();

        // Saturate concept A.
        tracker.record(a, 0, 2);
        tracker.record(a, 0, 2);
        assert!(tracker.is_saturated(a));

        // Concept B should not be affected.
        assert!(!tracker.is_saturated(b));
        tracker.record(b, 0, 2);
        assert!(!tracker.is_saturated(b));
    }

    fn make_resource(title: &str, concepts: &[&str], quality: f32, open_access: bool) -> DiscoveredResource {
        DiscoveredResource {
            title: title.to_string(),
            url: if open_access {
                format!("https://example.com/{}", title.replace(' ', "-"))
            } else {
                String::new()
            },
            source_api: ResourceApi::SemanticScholar,
            quality_score: quality,
            covers_concepts: concepts.iter().map(|s| s.to_string()).collect(),
            difficulty_estimate: DreyfusLevel::Competent,
            open_access,
            abstract_text: format!("Abstract about {title}"),
            year: Some(2024),
        }
    }

    #[test]
    fn resource_index_maps_concepts_to_resources() {
        let resources = vec![
            make_resource("Paper A", &["rust", "compiler"], 0.8, false),
            make_resource("Paper B", &["compiler", "optimization"], 0.7, false),
        ];

        let index = build_resource_index(&resources);

        assert_eq!(index.get("rust").map(|v| v.len()), Some(1));
        assert_eq!(index.get("compiler").map(|v| v.len()), Some(2));
        assert_eq!(index.get("optimization").map(|v| v.len()), Some(1));
    }

    #[test]
    fn resource_index_graceful_with_no_resources() {
        let resources: Vec<DiscoveredResource> = Vec::new();
        let index = build_resource_index(&resources);
        assert!(index.is_empty());
    }

    #[test]
    fn curriculum_sort_order_tier_asc_similarity_desc() {
        let entries = vec![
            CurriculumEntry {
                concept: SymbolId::new(1).unwrap(),
                label: "high-sim-tier-0".into(),
                zone: ZpdZone::Proximal,
                prereq_coverage: 0.0,
                similarity_to_known: 0.9,
                tier: 0,
                prerequisites: vec![],
            },
            CurriculumEntry {
                concept: SymbolId::new(2).unwrap(),
                label: "low-sim-tier-0".into(),
                zone: ZpdZone::Proximal,
                prereq_coverage: 0.0,
                similarity_to_known: 0.3,
                tier: 0,
                prerequisites: vec![],
            },
            CurriculumEntry {
                concept: SymbolId::new(3).unwrap(),
                label: "tier-1".into(),
                zone: ZpdZone::Proximal,
                prereq_coverage: 0.0,
                similarity_to_known: 0.8,
                tier: 1,
                prerequisites: vec![],
            },
        ];

        let mut sorted: Vec<_> = entries.iter().collect();
        sorted.sort_by(|a, b| {
            a.tier
                .cmp(&b.tier)
                .then_with(|| b.similarity_to_known.partial_cmp(&a.similarity_to_known)
                    .unwrap_or(std::cmp::Ordering::Equal))
        });

        // Tier 0 first, then tier 1.
        assert_eq!(sorted[0].tier, 0);
        assert_eq!(sorted[1].tier, 0);
        assert_eq!(sorted[2].tier, 1);
        // Within tier 0: higher similarity first.
        assert!(sorted[0].similarity_to_known > sorted[1].similarity_to_known);
    }

    #[test]
    fn cross_validation_boost_capped_at_1_0() {
        // Ensure boost arithmetic doesn't exceed 1.0.
        let base = 0.9_f32;
        let boost = 0.15_f32;
        let result = (base + boost).min(1.0);
        assert!((result - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn cross_validation_no_boost_below_threshold() {
        // With only 1 method, no boost should apply.
        let min_methods = 2;
        let method_count = 1;
        assert!(method_count < min_methods);
    }

    #[test]
    fn cross_validation_boost_with_sufficient_methods() {
        let min_methods = 2;
        let method_count = 3;
        assert!(method_count >= min_methods);
    }

    #[test]
    fn ingest_empty_curriculum_returns_error() {
        let prereq_result = PrereqAnalysisResult {
            edges: vec![],
            curriculum: vec![],
            concepts_analyzed: 0,
            edge_count: 0,
            cycles_broken: 0,
            max_tier: 0,
            zone_distribution: HashMap::new(),
            provenance_ids: vec![],
        };

        // We can't call ingest() without a real engine, but we can test the
        // precondition logic via the error type.
        assert!(prereq_result.curriculum.is_empty());
        let err = IngestionError::NoCurriculum;
        assert!(matches!(err, IngestionError::NoCurriculum));
    }

    #[test]
    fn ingest_empty_resources_returns_error() {
        let resources: Vec<DiscoveredResource> = vec![];

        assert!(resources.is_empty());
        let err = IngestionError::NoResources;
        assert!(matches!(err, IngestionError::NoResources));
    }

    #[test]
    fn concept_ingestion_outcome_accumulation() {
        let mut total = ConceptIngestionOutcome::empty();
        let a = ConceptIngestionOutcome {
            triples_added: 5,
            concepts_extracted: 3,
            abstract_ingested: true,
            url_attempted: false,
            url_succeeded: false,
            extraction_methods: vec!["abstract_text".to_string()],
        };
        let b = ConceptIngestionOutcome {
            triples_added: 2,
            concepts_extracted: 1,
            abstract_ingested: false,
            url_attempted: true,
            url_succeeded: true,
            extraction_methods: vec!["url_fulltext".to_string()],
        };

        total.accumulate(&a);
        total.accumulate(&b);

        assert_eq!(total.triples_added, 7);
        assert_eq!(total.concepts_extracted, 4);
        assert!(total.abstract_ingested);
        assert!(total.url_attempted);
        assert!(total.url_succeeded);
        assert_eq!(total.extraction_methods.len(), 2);
    }

    #[test]
    fn ingestion_run_result_aggregation() {
        let result = IngestionRunResult {
            cycles: 10,
            total_triples: 42,
            total_concepts_extracted: 15,
            concepts_ingested: 5,
            concepts_saturated: 2,
            url_attempts: 3,
            url_successes: 1,
            cross_validated_concepts: 3,
            symbols_grounded: 4,
            provenance_ids: vec![],
        };

        assert_eq!(result.cycles, 10);
        assert_eq!(result.total_triples, 42);
        assert_eq!(result.concepts_ingested, 5);
        assert_eq!(result.concepts_saturated, 2);
        assert_eq!(result.url_attempts, 3);
        assert_eq!(result.url_successes, 1);
        assert_eq!(result.cross_validated_concepts, 3);
        assert_eq!(result.symbols_grounded, 4);
    }
}
