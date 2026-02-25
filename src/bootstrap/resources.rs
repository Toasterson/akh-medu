//! Resource discovery for ZPD-proximal concepts (Phase 14e).
//!
//! Searches Semantic Scholar, OpenAlex, and Open Library for learning resources
//! matching concepts in the Proximal zone, scores them for quality, deduplicates
//! by VSA similarity, and stores as KG entities with provenance.

use std::collections::HashMap;
use std::sync::Arc;

use miette::Diagnostic;
use serde_json::Value;
use thiserror::Error;

use crate::bootstrap::expand::{normalize_label, ExpansionResult};
use crate::bootstrap::prerequisite::{CurriculumEntry, PrereqAnalysisResult, ZpdZone};
use crate::bootstrap::purpose::DreyfusLevel;
use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceId, ProvenanceRecord};
use crate::symbol::SymbolId;
use crate::vsa::encode::encode_label;

// ── Errors ──────────────────────────────────────────────────────────────

/// Errors specific to resource discovery.
#[derive(Debug, Error, Diagnostic)]
pub enum ResourceDiscoveryError {
    #[error("no proximal concepts in curriculum")]
    #[diagnostic(
        code(akh::bootstrap::resources::no_proximal_concepts),
        help(
            "The prerequisite analysis did not find any concepts in the ZPD \
             Proximal zone. Run domain expansion with broader seeds, or \
             lower the ZPD similarity bounds."
        )
    )]
    NoProximalConcepts,

    #[error("{0}")]
    #[diagnostic(
        code(akh::bootstrap::resources::engine),
        help("An engine-level error occurred during resource discovery.")
    )]
    Engine(Box<crate::error::AkhError>),
}

impl From<crate::error::AkhError> for ResourceDiscoveryError {
    fn from(e: crate::error::AkhError) -> Self {
        Self::Engine(Box::new(e))
    }
}

/// Convenience alias.
pub type ResourceResult<T> = std::result::Result<T, ResourceDiscoveryError>;

// ── Types ───────────────────────────────────────────────────────────────

/// Which open API sourced a resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceApi {
    SemanticScholar,
    OpenAlex,
    OpenLibrary,
}

impl std::fmt::Display for ResourceApi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SemanticScholar => f.write_str("Semantic Scholar"),
            Self::OpenAlex => f.write_str("OpenAlex"),
            Self::OpenLibrary => f.write_str("Open Library"),
        }
    }
}

/// Configuration for resource discovery.
#[derive(Debug, Clone)]
pub struct ResourceDiscoveryConfig {
    /// Maximum total API calls across all sources.
    pub max_api_calls: usize,
    /// Delay between API calls in milliseconds.
    pub delay_ms: u64,
    /// Maximum resources to keep per concept.
    pub max_per_concept: usize,
    /// Minimum quality score to accept a resource.
    pub min_quality: f32,
    /// Minimum VSA similarity to domain prototype to accept a resource.
    /// Resources below this threshold are considered off-topic noise.
    pub min_vsa_similarity: f32,
    /// VSA similarity threshold for deduplication (0.0–1.0).
    pub dedup_threshold: f32,
    /// Enable Semantic Scholar API.
    pub use_semantic_scholar: bool,
    /// Enable OpenAlex API.
    pub use_openalex: bool,
    /// Enable Open Library API.
    pub use_open_library: bool,
}

impl Default for ResourceDiscoveryConfig {
    fn default() -> Self {
        Self {
            max_api_calls: 60,
            delay_ms: 300,
            max_per_concept: 3,
            min_quality: 0.3,
            min_vsa_similarity: 0.3,
            dedup_threshold: 0.85,
            use_semantic_scholar: true,
            use_openalex: true,
            use_open_library: true,
        }
    }
}

/// Well-known predicates for resource entities in the KG.
struct ResourcePredicates {
    title: SymbolId,
    url: SymbolId,
    source_api: SymbolId,
    quality_score: SymbolId,
    covers_concept: SymbolId,
    difficulty: SymbolId,
    open_access: SymbolId,
    abstract_text: SymbolId,
    year: SymbolId,
}

impl ResourcePredicates {
    fn init(engine: &Engine) -> ResourceResult<Self> {
        Ok(Self {
            title: engine.resolve_or_create_relation("resource:title")?,
            url: engine.resolve_or_create_relation("resource:url")?,
            source_api: engine.resolve_or_create_relation("resource:source_api")?,
            quality_score: engine.resolve_or_create_relation("resource:quality_score")?,
            covers_concept: engine.resolve_or_create_relation("resource:covers_concept")?,
            difficulty: engine.resolve_or_create_relation("resource:difficulty")?,
            open_access: engine.resolve_or_create_relation("resource:open_access")?,
            abstract_text: engine.resolve_or_create_relation("resource:abstract_text")?,
            year: engine.resolve_or_create_relation("resource:year")?,
        })
    }
}

/// Intermediate pre-scoring resource from API JSON.
struct RawResource {
    title: String,
    url: String,
    source_api: ResourceApi,
    abstract_text: String,
    year: Option<u32>,
    citation_count: u32,
    is_open_access: bool,
    is_book: bool,
    concept_label: String,
}

/// A scored, accepted learning resource.
#[derive(Debug, Clone)]
pub struct DiscoveredResource {
    pub title: String,
    pub url: String,
    pub source_api: ResourceApi,
    pub quality_score: f32,
    pub covers_concepts: Vec<String>,
    pub difficulty_estimate: DreyfusLevel,
    pub open_access: bool,
    pub abstract_text: String,
    pub year: Option<u32>,
}

/// Result of a resource discovery run.
#[derive(Debug)]
pub struct ResourceDiscoveryResult {
    pub resources: Vec<DiscoveredResource>,
    pub api_calls_made: usize,
    pub concepts_searched: usize,
    pub provenance_ids: Vec<ProvenanceId>,
}

// ── ResourceDiscoverer ──────────────────────────────────────────────────

/// Discovers learning resources for ZPD-proximal concepts.
pub struct ResourceDiscoverer {
    config: ResourceDiscoveryConfig,
    predicates: ResourcePredicates,
    api_call_count: usize,
}

impl ResourceDiscoverer {
    /// Create a new discoverer, resolving well-known predicates.
    pub fn new(engine: &Engine, config: ResourceDiscoveryConfig) -> ResourceResult<Self> {
        let predicates = ResourcePredicates::init(engine)?;
        Ok(Self {
            config,
            predicates,
            api_call_count: 0,
        })
    }

    /// Run resource discovery for all proximal concepts in the curriculum.
    ///
    /// `seed_labels` are the original seed concepts (e.g. `["rust", "compiler"]`)
    /// used for query disambiguation. These should be the primary domain terms,
    /// not expanded concepts.
    pub fn discover(
        &mut self,
        prereq_result: &PrereqAnalysisResult,
        expansion_result: &ExpansionResult,
        seed_labels: &[String],
        engine: &Arc<Engine>,
    ) -> ResourceResult<ResourceDiscoveryResult> {
        let ops = engine.ops();

        // Collect proximal concepts.
        let proximal: Vec<&CurriculumEntry> = prereq_result
            .curriculum
            .iter()
            .filter(|e| e.zone == ZpdZone::Proximal)
            .collect();

        if proximal.is_empty() {
            return Err(ResourceDiscoveryError::NoProximalConcepts);
        }

        // Use seed labels as query disambiguation context.
        // Seeds like "rust", "compiler" disambiguate searches much better than
        // expanded concepts like "type system", "garbage collector".
        let domain_labels: Vec<&str> = seed_labels.iter().map(String::as_str).collect();

        // Build domain prototype HV for VSA similarity scoring.
        // Try to retrieve from item memory (stored by expand phase), else build from seeds.
        let domain_prototype = expansion_result
            .domain_prototype_id
            .and_then(|id| engine.item_memory().get(id))
            .or_else(|| {
                // Fallback: bundle encoded seed labels (same as expand.rs does).
                let vecs: Vec<_> = seed_labels
                    .iter()
                    .filter_map(|s| encode_label(ops, s).ok())
                    .collect();
                if vecs.is_empty() {
                    return None;
                }
                let refs: Vec<&crate::vsa::HyperVec> = vecs.iter().collect();
                ops.bundle(&refs).ok()
            });

        let mut all_resources: Vec<DiscoveredResource> = Vec::new();
        let concepts_searched = proximal.len();

        for entry in &proximal {
            if self.api_call_count >= self.config.max_api_calls {
                break;
            }

            let query = build_search_query(&entry.label, &domain_labels);
            let zone_similarity = entry.similarity_to_known;

            let mut raws: Vec<RawResource> = Vec::new();

            if self.config.use_semantic_scholar {
                raws.extend(self.search_semantic_scholar(&query, &entry.label));
            }
            if self.config.use_openalex {
                raws.extend(self.search_openalex(&query, &entry.label));
            }
            if self.config.use_open_library {
                raws.extend(self.search_open_library(&query, &entry.label));
            }

            // Score and filter.
            for raw in raws {
                let vsa_sim = domain_prototype
                    .as_ref()
                    .and_then(|proto| {
                        let text = format!("{} {}", raw.title, raw.abstract_text);
                        encode_label(ops, &text)
                            .ok()
                            .and_then(|vec| ops.similarity(&vec, proto).ok())
                    })
                    .unwrap_or(0.5);

                // Reject clearly off-topic resources by VSA similarity floor.
                if vsa_sim < self.config.min_vsa_similarity {
                    continue;
                }

                let quality = score_resource(&raw, vsa_sim);
                if quality < self.config.min_quality {
                    continue;
                }

                let difficulty = estimate_difficulty(zone_similarity);

                all_resources.push(DiscoveredResource {
                    title: raw.title,
                    url: raw.url,
                    source_api: raw.source_api,
                    quality_score: quality,
                    covers_concepts: vec![raw.concept_label],
                    difficulty_estimate: difficulty,
                    open_access: raw.is_open_access,
                    abstract_text: raw.abstract_text,
                    year: raw.year,
                });
            }
        }

        // Deduplicate.
        let deduped = deduplicate_resources(all_resources, self.config.dedup_threshold, ops);

        // Limit per concept.
        let limited = limit_per_concept(deduped, self.config.max_per_concept);

        // Store in KG.
        let provenance_ids = self.store_resources(&limited, engine)?;

        Ok(ResourceDiscoveryResult {
            resources: limited,
            api_calls_made: self.api_call_count,
            concepts_searched,
            provenance_ids,
        })
    }

    // ── API search methods ──────────────────────────────────────────

    fn search_semantic_scholar(
        &mut self,
        query: &str,
        concept_label: &str,
    ) -> Vec<RawResource> {
        let encoded = url_encode(query);
        let url = format!(
            "https://api.semanticscholar.org/graph/v1/paper/search?query={encoded}&fields=title,url,abstract,year,citationCount,openAccessPdf,isOpenAccess&limit=5"
        );

        let Some(json) = self.api_call(&url) else {
            return Vec::new();
        };

        let mut results = Vec::new();
        if let Some(data) = json.get("data").and_then(|d| d.as_array()) {
            for item in data {
                let title = item
                    .get("title")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if title.is_empty() {
                    continue;
                }

                let url_val = item
                    .get("url")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let abstract_text = item
                    .get("abstract")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let year = item
                    .get("year")
                    .and_then(|v| v.as_u64())
                    .map(|y| y as u32);

                let citation_count = item
                    .get("citationCount")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32;

                let is_open_access = item
                    .get("isOpenAccess")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                results.push(RawResource {
                    title,
                    url: url_val,
                    source_api: ResourceApi::SemanticScholar,
                    abstract_text,
                    year,
                    citation_count,
                    is_open_access,
                    is_book: false,
                    concept_label: concept_label.to_string(),
                });
            }
        }
        results
    }

    fn search_openalex(&mut self, query: &str, concept_label: &str) -> Vec<RawResource> {
        let encoded = url_encode(query);
        let url = format!(
            "https://api.openalex.org/works?search={encoded}&sort=cited_by_count:desc&per-page=5"
        );

        let Some(json) = self.api_call(&url) else {
            return Vec::new();
        };

        let mut results = Vec::new();
        if let Some(data) = json.get("results").and_then(|d| d.as_array()) {
            for item in data {
                let title = item
                    .get("title")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if title.is_empty() {
                    continue;
                }

                let url_val = item
                    .get("doi")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                // OpenAlex stores abstracts as inverted index.
                let abstract_text = item
                    .get("abstract_inverted_index")
                    .map(reconstruct_openalex_abstract)
                    .unwrap_or_default();

                let year = item
                    .get("publication_year")
                    .and_then(|v| v.as_u64())
                    .map(|y| y as u32);

                let citation_count = item
                    .get("cited_by_count")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32;

                let is_open_access = item
                    .get("open_access")
                    .and_then(|oa| oa.get("is_oa"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                results.push(RawResource {
                    title,
                    url: url_val,
                    source_api: ResourceApi::OpenAlex,
                    abstract_text,
                    year,
                    citation_count,
                    is_open_access,
                    is_book: false,
                    concept_label: concept_label.to_string(),
                });
            }
        }
        results
    }

    fn search_open_library(&mut self, query: &str, concept_label: &str) -> Vec<RawResource> {
        let encoded = url_encode(query);
        let url = format!("https://openlibrary.org/search.json?q={encoded}&limit=5");

        let Some(json) = self.api_call(&url) else {
            return Vec::new();
        };

        let mut results = Vec::new();
        if let Some(docs) = json.get("docs").and_then(|d| d.as_array()) {
            for item in docs {
                let title = item
                    .get("title")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if title.is_empty() {
                    continue;
                }

                let key = item
                    .get("key")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let url_val = format!("https://openlibrary.org{key}");

                let year = item
                    .get("first_publish_year")
                    .and_then(|v| v.as_u64())
                    .map(|y| y as u32);

                results.push(RawResource {
                    title,
                    url: url_val,
                    source_api: ResourceApi::OpenLibrary,
                    abstract_text: String::new(),
                    year,
                    citation_count: 0,
                    is_open_access: true, // Open Library is open access.
                    is_book: true,
                    concept_label: concept_label.to_string(),
                });
            }
        }
        results
    }

    fn api_call(&mut self, url: &str) -> Option<Value> {
        if self.api_call_count >= self.config.max_api_calls {
            return None;
        }
        self.api_call_count += 1;

        if self.config.delay_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(self.config.delay_ms));
        }

        let body: Value = ureq::get(url)
            .timeout(std::time::Duration::from_secs(10))
            .call()
            .ok()?
            .into_json()
            .ok()?;
        Some(body)
    }

    // ── KG storage ──────────────────────────────────────────────────

    fn store_resources(
        &self,
        resources: &[DiscoveredResource],
        engine: &Arc<Engine>,
    ) -> ResourceResult<Vec<ProvenanceId>> {
        let mut provenance_ids = Vec::new();
        let preds = &self.predicates;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        for resource in resources {
            let label = format!("resource:{}", normalize_label(&resource.title));
            let entity_id = engine.resolve_or_create_entity(&label)?;

            // Title triple.
            let title_obj = engine.resolve_or_create_entity(&resource.title)?;
            engine.add_triple(&Triple {
                subject: entity_id,
                predicate: preds.title,
                object: title_obj,
                confidence: resource.quality_score,
                timestamp: now,
                provenance_id: None,
                compartment_id: None,
            })?;

            // URL triple.
            if !resource.url.is_empty() {
                let url_obj = engine.resolve_or_create_entity(&resource.url)?;
                engine.add_triple(&Triple {
                    subject: entity_id,
                    predicate: preds.url,
                    object: url_obj,
                    confidence: 1.0,
                    timestamp: now,
                    provenance_id: None,
                    compartment_id: None,
                })?;
            }

            // Source API triple.
            let api_label = resource.source_api.to_string();
            let api_obj = engine.resolve_or_create_entity(&api_label)?;
            engine.add_triple(&Triple {
                subject: entity_id,
                predicate: preds.source_api,
                object: api_obj,
                confidence: 1.0,
                timestamp: now,
                provenance_id: None,
                compartment_id: None,
            })?;

            // Quality score triple.
            let score_label = format!("quality:{:.2}", resource.quality_score);
            let score_obj = engine.resolve_or_create_entity(&score_label)?;
            engine.add_triple(&Triple {
                subject: entity_id,
                predicate: preds.quality_score,
                object: score_obj,
                confidence: resource.quality_score,
                timestamp: now,
                provenance_id: None,
                compartment_id: None,
            })?;

            // Covers concept triples.
            for concept in &resource.covers_concepts {
                if let Ok(concept_id) = engine.resolve_or_create_entity(concept) {
                    engine.add_triple(&Triple {
                        subject: entity_id,
                        predicate: preds.covers_concept,
                        object: concept_id,
                        confidence: resource.quality_score,
                        timestamp: now,
                        provenance_id: None,
                        compartment_id: None,
                    })?;
                }
            }

            // Difficulty triple.
            let diff_label = resource.difficulty_estimate.as_label();
            let diff_obj = engine.resolve_or_create_entity(diff_label)?;
            engine.add_triple(&Triple {
                subject: entity_id,
                predicate: preds.difficulty,
                object: diff_obj,
                confidence: 1.0,
                timestamp: now,
                provenance_id: None,
                compartment_id: None,
            })?;

            // Open access triple.
            let oa_label = if resource.open_access { "true" } else { "false" };
            let oa_obj = engine.resolve_or_create_entity(oa_label)?;
            engine.add_triple(&Triple {
                subject: entity_id,
                predicate: preds.open_access,
                object: oa_obj,
                confidence: 1.0,
                timestamp: now,
                provenance_id: None,
                compartment_id: None,
            })?;

            // Year triple (if present).
            if let Some(year) = resource.year {
                let year_label = format!("year:{year}");
                let year_obj = engine.resolve_or_create_entity(&year_label)?;
                engine.add_triple(&Triple {
                    subject: entity_id,
                    predicate: preds.year,
                    object: year_obj,
                    confidence: 1.0,
                    timestamp: now,
                    provenance_id: None,
                    compartment_id: None,
                })?;
            }

            // Abstract text triple (if non-empty).
            if !resource.abstract_text.is_empty() {
                let abs_obj = engine.resolve_or_create_entity(&resource.abstract_text)?;
                engine.add_triple(&Triple {
                    subject: entity_id,
                    predicate: preds.abstract_text,
                    object: abs_obj,
                    confidence: 1.0,
                    timestamp: now,
                    provenance_id: None,
                    compartment_id: None,
                })?;
            }

            // Provenance record.
            let sources_str = resource.source_api.to_string();
            let mut record = ProvenanceRecord::new(
                entity_id,
                DerivationKind::ResourceDiscovery {
                    concept_label: resource
                        .covers_concepts
                        .first()
                        .cloned()
                        .unwrap_or_default(),
                    resource_count: 1,
                    sources: sources_str,
                    top_title: resource.title.clone(),
                },
            )
            .with_confidence(resource.quality_score);

            if let Ok(id) = engine.store_provenance(&mut record) {
                provenance_ids.push(id);
            }
        }

        Ok(provenance_ids)
    }
}

// ── Free functions ──────────────────────────────────────────────────────

/// Build a search query from a concept label and domain context.
///
/// Concatenates the concept label with up to 3 domain labels, truncated to 120 chars.
fn build_search_query(label: &str, domain_labels: &[&str]) -> String {
    let mut query = normalize_label(label);
    for dl in domain_labels.iter().take(3) {
        let candidate = format!("{query} {}", normalize_label(dl));
        if candidate.len() > 120 {
            break;
        }
        query = candidate;
    }
    if query.len() > 120 {
        query.truncate(120);
        // Avoid truncating mid-word.
        if let Some(pos) = query.rfind(' ') {
            query.truncate(pos);
        }
    }
    query
}

/// Build a synthetic `PrereqAnalysisResult` from expansion labels when prerequisite
/// analysis fails (e.g. too few concepts for edge discovery).
///
/// All expanded concepts are treated as ZPD Proximal with default similarity 0.5,
/// giving resource discovery a reasonable set of concepts to search for.
pub fn synthetic_curriculum_from_expansion(
    expansion_result: &ExpansionResult,
    engine: &Engine,
) -> PrereqAnalysisResult {
    let curriculum: Vec<CurriculumEntry> = expansion_result
        .accepted_labels
        .iter()
        .filter_map(|label| {
            let concept = engine.resolve_or_create_entity(label).ok()?;
            Some(CurriculumEntry {
                concept,
                label: label.clone(),
                zone: ZpdZone::Proximal,
                prereq_coverage: 0.0,
                similarity_to_known: 0.5,
                tier: 0,
                prerequisites: Vec::new(),
            })
        })
        .collect();

    let mut zone_distribution = HashMap::new();
    zone_distribution.insert(ZpdZone::Proximal, curriculum.len());

    PrereqAnalysisResult {
        edges: Vec::new(),
        concepts_analyzed: curriculum.len(),
        edge_count: 0,
        cycles_broken: 0,
        max_tier: 0,
        zone_distribution,
        provenance_ids: Vec::new(),
        curriculum,
    }
}

/// Score a raw resource based on multiple quality signals.
fn score_resource(raw: &RawResource, vsa_similarity: f32) -> f32 {
    let current_year = 2026u32;
    let age = raw
        .year
        .map(|y| current_year.saturating_sub(y))
        .unwrap_or(10);

    // Citation impact: normalized by age. Median ~10 citations/year for a good paper.
    let citation_impact = if age == 0 {
        0.5 // New publication, no citation data.
    } else {
        let citations_per_year = raw.citation_count as f32 / age as f32;
        (citations_per_year / 10.0).min(1.0)
    };

    // Recency bonus.
    let recency = (1.0 - (age as f32 / 20.0)).clamp(0.0, 1.0);

    // Open access.
    let open_access = if raw.is_open_access { 1.0 } else { 0.0 };

    // Source type.
    let source_type = if raw.is_book { 1.0 } else { 0.5 };

    // Weighted sum.
    citation_impact * 0.30
        + recency * 0.15
        + open_access * 0.20
        + vsa_similarity * 0.25
        + source_type * 0.10
}

/// Estimate learning difficulty from ZPD similarity to known concepts.
fn estimate_difficulty(similarity_to_known: f32) -> DreyfusLevel {
    if similarity_to_known < 0.40 {
        DreyfusLevel::Novice
    } else if similarity_to_known < 0.50 {
        DreyfusLevel::AdvancedBeginner
    } else if similarity_to_known < 0.60 {
        DreyfusLevel::Competent
    } else if similarity_to_known < 0.70 {
        DreyfusLevel::Proficient
    } else {
        DreyfusLevel::Expert
    }
}

/// Deduplicate resources by pairwise VSA similarity of their title + abstract.
///
/// O(n²) but n is small (~45 max).
fn deduplicate_resources(
    resources: Vec<DiscoveredResource>,
    threshold: f32,
    ops: &crate::vsa::ops::VsaOps,
) -> Vec<DiscoveredResource> {
    if resources.len() <= 1 {
        return resources;
    }

    // Pre-encode all resource HVs.
    let hvs: Vec<Option<crate::vsa::HyperVec>> = resources
        .iter()
        .map(|r| {
            let text = format!("{} {}", r.title, r.abstract_text);
            encode_label(ops, &text).ok()
        })
        .collect();

    let mut keep = vec![true; resources.len()];

    for i in 0..resources.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..resources.len() {
            if !keep[j] {
                continue;
            }
            if let (Some(a), Some(b)) = (&hvs[i], &hvs[j])
                && let Ok(sim) = ops.similarity(a, b)
                && sim >= threshold
            {
                // Keep the higher-scoring one.
                if resources[i].quality_score >= resources[j].quality_score {
                    keep[j] = false;
                } else {
                    keep[i] = false;
                    break; // i is eliminated, move on.
                }
            }
        }
    }

    resources
        .into_iter()
        .zip(keep)
        .filter_map(|(r, k)| if k { Some(r) } else { None })
        .collect()
}

/// Limit resources to max_per_concept, keeping highest-scored per concept.
fn limit_per_concept(
    mut resources: Vec<DiscoveredResource>,
    max_per_concept: usize,
) -> Vec<DiscoveredResource> {
    // Sort by quality descending.
    resources.sort_by(|a, b| b.quality_score.partial_cmp(&a.quality_score).unwrap_or(std::cmp::Ordering::Equal));

    let mut counts: HashMap<String, usize> = HashMap::new();
    resources
        .into_iter()
        .filter(|r| {
            let concept = r.covers_concepts.first().cloned().unwrap_or_default();
            let count = counts.entry(concept).or_insert(0);
            if *count < max_per_concept {
                *count += 1;
                true
            } else {
                false
            }
        })
        .collect()
}

/// Reconstruct an abstract from OpenAlex's inverted index format.
///
/// OpenAlex stores abstracts as `{"word": [pos0, pos1, ...], ...}`.
/// This function rebuilds the original text.
pub fn reconstruct_openalex_abstract(inverted: &Value) -> String {
    let Some(obj) = inverted.as_object() else {
        return String::new();
    };

    let mut positions: Vec<(usize, String)> = Vec::new();
    for (word, pos_array) in obj {
        if let Some(arr) = pos_array.as_array() {
            for pos_val in arr {
                if let Some(pos) = pos_val.as_u64() {
                    positions.push((pos as usize, word.clone()));
                }
            }
        }
    }

    positions.sort_by_key(|(pos, _)| *pos);

    let words: Vec<&str> = positions.iter().map(|(_, w)| w.as_str()).collect();
    words.join(" ")
}

/// Percent-encode a string for use in URL query parameters.
fn url_encode(input: &str) -> String {
    let mut encoded = String::with_capacity(input.len() * 2);
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char);
            }
            b' ' => encoded.push('+'),
            _ => {
                encoded.push('%');
                encoded.push(char::from(HEX_CHARS[(byte >> 4) as usize]));
                encoded.push(char::from(HEX_CHARS[(byte & 0x0f) as usize]));
            }
        }
    }
    encoded
}

const HEX_CHARS: [u8; 16] = *b"0123456789ABCDEF";

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Config defaults ─────────────────────────────────────────────

    #[test]
    fn config_defaults() {
        let config = ResourceDiscoveryConfig::default();
        assert_eq!(config.max_api_calls, 60);
        assert_eq!(config.delay_ms, 300);
        assert_eq!(config.max_per_concept, 3);
        assert!((config.min_quality - 0.3).abs() < f32::EPSILON);
        assert!((config.min_vsa_similarity - 0.3).abs() < f32::EPSILON);
        assert!((config.dedup_threshold - 0.85).abs() < f32::EPSILON);
        assert!(config.use_semantic_scholar);
        assert!(config.use_openalex);
        assert!(config.use_open_library);
    }

    // ── ResourceApi Display ─────────────────────────────────────────

    #[test]
    fn resource_api_display() {
        assert_eq!(ResourceApi::SemanticScholar.to_string(), "Semantic Scholar");
        assert_eq!(ResourceApi::OpenAlex.to_string(), "OpenAlex");
        assert_eq!(ResourceApi::OpenLibrary.to_string(), "Open Library");
    }

    // ── Error formatting ────────────────────────────────────────────

    #[test]
    fn error_no_proximal_display() {
        let err = ResourceDiscoveryError::NoProximalConcepts;
        let msg = format!("{err}");
        assert!(msg.contains("no proximal concepts"));
    }

    // ── build_search_query ──────────────────────────────────────────

    #[test]
    fn build_query_basic() {
        let q = build_search_query("compiler", &["systems", "optimization"]);
        assert_eq!(q, "compiler systems optimization");
    }

    #[test]
    fn build_query_no_domain() {
        let q = build_search_query("machine learning", &[]);
        assert_eq!(q, "machine learning");
    }

    #[test]
    fn build_query_truncation() {
        // Create a long label that exceeds 120 chars with domain context.
        let long_label = "a".repeat(100);
        let domains = ["domain1", "domain2", "domain3"];
        let q = build_search_query(&long_label, &domains);
        assert!(q.len() <= 120);
    }

    #[test]
    fn build_query_normalizes() {
        let q = build_search_query("Machine-Learning_Pipeline", &[]);
        assert_eq!(q, "machine learning pipeline");
    }

    // ── estimate_difficulty ─────────────────────────────────────────

    #[test]
    fn difficulty_novice() {
        assert_eq!(estimate_difficulty(0.35), DreyfusLevel::Novice);
    }

    #[test]
    fn difficulty_advanced_beginner() {
        assert_eq!(estimate_difficulty(0.45), DreyfusLevel::AdvancedBeginner);
    }

    #[test]
    fn difficulty_competent() {
        assert_eq!(estimate_difficulty(0.55), DreyfusLevel::Competent);
    }

    #[test]
    fn difficulty_proficient() {
        assert_eq!(estimate_difficulty(0.65), DreyfusLevel::Proficient);
    }

    #[test]
    fn difficulty_expert() {
        assert_eq!(estimate_difficulty(0.75), DreyfusLevel::Expert);
    }

    #[test]
    fn difficulty_boundaries() {
        assert_eq!(estimate_difficulty(0.0), DreyfusLevel::Novice);
        assert_eq!(estimate_difficulty(0.40), DreyfusLevel::AdvancedBeginner);
        assert_eq!(estimate_difficulty(0.50), DreyfusLevel::Competent);
        assert_eq!(estimate_difficulty(0.60), DreyfusLevel::Proficient);
        assert_eq!(estimate_difficulty(0.70), DreyfusLevel::Expert);
        assert_eq!(estimate_difficulty(1.0), DreyfusLevel::Expert);
    }

    // ── score_resource ──────────────────────────────────────────────

    #[test]
    fn score_open_access_bonus() {
        let open = RawResource {
            title: "Test".to_string(),
            url: String::new(),
            source_api: ResourceApi::SemanticScholar,
            abstract_text: String::new(),
            year: Some(2024),
            citation_count: 50,
            is_open_access: true,
            is_book: false,
            concept_label: "test".to_string(),
        };
        let closed = RawResource {
            is_open_access: false,
            ..RawResource {
                title: "Test".to_string(),
                url: String::new(),
                source_api: ResourceApi::SemanticScholar,
                abstract_text: String::new(),
                year: Some(2024),
                citation_count: 50,
                is_open_access: false,
                is_book: false,
                concept_label: "test".to_string(),
            }
        };

        let open_score = score_resource(&open, 0.5);
        let closed_score = score_resource(&closed, 0.5);

        // Open access should add 0.20 * 1.0 = 0.20 to the score.
        assert!(open_score > closed_score);
        assert!((open_score - closed_score - 0.20).abs() < 0.01);
    }

    #[test]
    fn score_citation_impact() {
        let high_cite = RawResource {
            title: "Test".to_string(),
            url: String::new(),
            source_api: ResourceApi::SemanticScholar,
            abstract_text: String::new(),
            year: Some(2024),
            citation_count: 100,
            is_open_access: false,
            is_book: false,
            concept_label: "test".to_string(),
        };
        let low_cite = RawResource {
            citation_count: 1,
            ..RawResource {
                title: "Test".to_string(),
                url: String::new(),
                source_api: ResourceApi::SemanticScholar,
                abstract_text: String::new(),
                year: Some(2024),
                citation_count: 1,
                is_open_access: false,
                is_book: false,
                concept_label: "test".to_string(),
            }
        };

        let high_score = score_resource(&high_cite, 0.5);
        let low_score = score_resource(&low_cite, 0.5);
        assert!(high_score > low_score);
    }

    // ── reconstruct_openalex_abstract ───────────────────────────────

    #[test]
    fn reconstruct_abstract_basic() {
        let json: Value = serde_json::json!({
            "This": [0],
            "is": [1],
            "a": [2],
            "test": [3]
        });
        let text = reconstruct_openalex_abstract(&json);
        assert_eq!(text, "This is a test");
    }

    #[test]
    fn reconstruct_abstract_repeated_word() {
        let json: Value = serde_json::json!({
            "the": [0, 3],
            "cat": [1],
            "chased": [2],
            "mouse": [4]
        });
        let text = reconstruct_openalex_abstract(&json);
        assert_eq!(text, "the cat chased the mouse");
    }

    #[test]
    fn reconstruct_abstract_empty() {
        let json: Value = serde_json::json!(null);
        let text = reconstruct_openalex_abstract(&json);
        assert!(text.is_empty());
    }

    // ── url_encode ──────────────────────────────────────────────────

    #[test]
    fn url_encode_spaces() {
        assert_eq!(url_encode("hello world"), "hello+world");
    }

    #[test]
    fn url_encode_special_chars() {
        let encoded = url_encode("a&b=c");
        assert!(encoded.contains("%26"));
        assert!(encoded.contains("%3D"));
    }

    // ── discover with empty curriculum ──────────────────────────────

    #[test]
    fn discover_no_proximal_concepts() {
        // An empty curriculum should produce NoProximalConcepts error.
        let prereq = PrereqAnalysisResult {
            edges: Vec::new(),
            curriculum: Vec::new(),
            concepts_analyzed: 0,
            edge_count: 0,
            cycles_broken: 0,
            max_tier: 0,
            zone_distribution: HashMap::new(),
            provenance_ids: Vec::new(),
        };
        let _expansion = ExpansionResult {
            concept_count: 0,
            relation_count: 0,
            rejected_count: 0,
            api_calls: 0,
            domain_prototype_id: None,
            microtheory_id: None,
            provenance_ids: Vec::new(),
            accepted_labels: Vec::new(),
            boundary_rejects: Vec::new(),
        };

        // We cannot construct a ResourceDiscoverer without an engine,
        // so test the filter logic directly.
        let proximal: Vec<&CurriculumEntry> = prereq
            .curriculum
            .iter()
            .filter(|e| e.zone == ZpdZone::Proximal)
            .collect();
        assert!(proximal.is_empty());
    }

    // ── limit_per_concept ───────────────────────────────────────────

    #[test]
    fn limit_per_concept_caps() {
        let resources = vec![
            DiscoveredResource {
                title: "A".to_string(),
                url: String::new(),
                source_api: ResourceApi::SemanticScholar,
                quality_score: 0.9,
                covers_concepts: vec!["rust".to_string()],
                difficulty_estimate: DreyfusLevel::Competent,
                open_access: true,
                abstract_text: String::new(),
                year: Some(2024),
            },
            DiscoveredResource {
                title: "B".to_string(),
                url: String::new(),
                source_api: ResourceApi::OpenAlex,
                quality_score: 0.8,
                covers_concepts: vec!["rust".to_string()],
                difficulty_estimate: DreyfusLevel::Competent,
                open_access: true,
                abstract_text: String::new(),
                year: Some(2023),
            },
            DiscoveredResource {
                title: "C".to_string(),
                url: String::new(),
                source_api: ResourceApi::OpenLibrary,
                quality_score: 0.7,
                covers_concepts: vec!["rust".to_string()],
                difficulty_estimate: DreyfusLevel::Competent,
                open_access: true,
                abstract_text: String::new(),
                year: Some(2022),
            },
            DiscoveredResource {
                title: "D".to_string(),
                url: String::new(),
                source_api: ResourceApi::SemanticScholar,
                quality_score: 0.6,
                covers_concepts: vec!["rust".to_string()],
                difficulty_estimate: DreyfusLevel::Competent,
                open_access: true,
                abstract_text: String::new(),
                year: Some(2021),
            },
        ];

        let limited = limit_per_concept(resources, 2);
        assert_eq!(limited.len(), 2);
        assert_eq!(limited[0].title, "A");
        assert_eq!(limited[1].title, "B");
    }
}
