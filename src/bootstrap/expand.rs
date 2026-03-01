//! Domain expansion: skeleton ontology from seed concepts (Phase 14c).
//!
//! Takes seed concepts from a `PurposeModel` and expands them into a skeleton
//! ontology of concept nodes in the KG by querying external knowledge sources
//! (Wikidata, Wikipedia, ConceptNet) and filtering candidates by VSA similarity
//! to a domain prototype vector.

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use miette::Diagnostic;
use thiserror::Error;

use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceId, ProvenanceRecord};
use crate::symbol::SymbolId;
use crate::vsa::encode::{encode_label, encode_token};
use crate::vsa::ops::VsaOps;
use crate::vsa::HyperVec;

use super::purpose::PurposeModel;

// ── Errors ──────────────────────────────────────────────────────────────

/// Errors from domain expansion.
#[derive(Debug, Error, Diagnostic)]
pub enum DomainExpandError {
    #[error("no seed concepts provided for domain expansion")]
    #[diagnostic(
        code(akh::bootstrap::expand::no_seeds),
        help(
            "Provide at least one seed concept via --seeds or a purpose statement with --purpose."
        )
    )]
    NoSeeds,

    #[error("all {candidate_count} candidate concepts fell below similarity threshold {threshold:.2}")]
    #[diagnostic(
        code(akh::bootstrap::expand::empty_expansion),
        help(
            "Lower the similarity threshold with --threshold (default 0.6) \
             or provide more specific seed concepts."
        )
    )]
    EmptyExpansion { threshold: f32, candidate_count: usize },

    #[error("API call budget exhausted after {calls} calls")]
    #[diagnostic(
        code(akh::bootstrap::expand::rate_limit),
        help(
            "Increase max_api_calls in ExpansionConfig or reduce the number of seed concepts."
        )
    )]
    RateLimitReached { calls: usize },

    #[error("{0}")]
    #[diagnostic(
        code(akh::bootstrap::expand::engine),
        help("An engine-level error occurred during domain expansion.")
    )]
    Engine(Box<crate::error::AkhError>),
}

impl From<crate::error::AkhError> for DomainExpandError {
    fn from(e: crate::error::AkhError) -> Self {
        Self::Engine(Box::new(e))
    }
}

/// Convenience alias for domain expansion operations.
pub type ExpandResult<T> = std::result::Result<T, DomainExpandError>;

// ── Configuration ───────────────────────────────────────────────────────

/// Configuration for domain expansion.
#[derive(Debug, Clone)]
pub struct ExpansionConfig {
    /// Minimum VSA similarity for a candidate to be accepted.
    pub similarity_threshold: f32,
    /// Maximum BFS depth for expansion.
    pub max_depth: usize,
    /// Maximum number of concept nodes to create.
    pub max_concepts: usize,
    /// Maximum number of external API calls.
    pub max_api_calls: usize,
    /// Delay between API calls in milliseconds.
    pub inter_call_delay_ms: u64,
    /// Whether to query ConceptNet.
    pub use_conceptnet: bool,
}

impl Default for ExpansionConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.6,
            max_depth: 3,
            max_concepts: 200,
            max_api_calls: 50,
            inter_call_delay_ms: 200,
            use_conceptnet: true,
        }
    }
}

// ── Predicates ──────────────────────────────────────────────────────────

/// Well-known relation predicates for domain expansion (`expand:` namespace).
pub struct ExpansionPredicates {
    pub expanded_from: SymbolId,
    pub instance_of: SymbolId,
    pub subclass_of: SymbolId,
    pub part_of: SymbolId,
    pub has_part: SymbolId,
    pub related_to: SymbolId,
    pub has_prerequisite: SymbolId,
    pub used_for: SymbolId,
    pub domain_prototype: SymbolId,
}

impl ExpansionPredicates {
    /// Initialize predicates by resolving or creating relation symbols.
    pub fn init(engine: &Engine) -> ExpandResult<Self> {
        Ok(Self {
            expanded_from: engine.resolve_or_create_relation("expand:expanded_from")?,
            instance_of: engine.resolve_or_create_relation("expand:instance_of")?,
            subclass_of: engine.resolve_or_create_relation("expand:subclass_of")?,
            part_of: engine.resolve_or_create_relation("expand:part_of")?,
            has_part: engine.resolve_or_create_relation("expand:has_part")?,
            related_to: engine.resolve_or_create_relation("expand:related_to")?,
            has_prerequisite: engine.resolve_or_create_relation("expand:has_prerequisite")?,
            used_for: engine.resolve_or_create_relation("expand:used_for")?,
            domain_prototype: engine.resolve_or_create_relation("expand:domain_prototype")?,
        })
    }

    /// Map a ConceptNet relation string to the corresponding predicate symbol.
    pub fn conceptnet_relation(&self, rel: &str) -> Option<SymbolId> {
        match rel {
            "IsA" => Some(self.instance_of),
            "PartOf" => Some(self.part_of),
            "HasA" => Some(self.has_part),
            "RelatedTo" => Some(self.related_to),
            "HasPrerequisite" => Some(self.has_prerequisite),
            "UsedFor" => Some(self.used_for),
            _ => None,
        }
    }
}

// ── Role Vectors ────────────────────────────────────────────────────────

/// VSA role vectors for domain expansion.
pub struct ExpansionRoleVectors {
    pub concept: HyperVec,
    pub relation: HyperVec,
    pub source: HyperVec,
    pub depth: HyperVec,
}

impl ExpansionRoleVectors {
    /// Initialize role vectors.
    pub fn init(ops: &VsaOps) -> Self {
        Self {
            concept: encode_token(ops, "expand-role:concept"),
            relation: encode_token(ops, "expand-role:relation"),
            source: encode_token(ops, "expand-role:source"),
            depth: encode_token(ops, "expand-role:depth"),
        }
    }
}

// ── Data Types ──────────────────────────────────────────────────────────

/// Source of a discovered concept.
#[derive(Debug, Clone, PartialEq)]
pub enum ConceptSource {
    Seed,
    Wikidata,
    Wikipedia,
    ConceptNet,
}

impl fmt::Display for ConceptSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Seed => write!(f, "seed"),
            Self::Wikidata => write!(f, "wikidata"),
            Self::Wikipedia => write!(f, "wikipedia"),
            Self::ConceptNet => write!(f, "conceptnet"),
        }
    }
}

/// A candidate concept discovered during expansion.
#[derive(Debug, Clone)]
pub struct CandidateConcept {
    pub label: String,
    pub source: ConceptSource,
    pub depth: usize,
    pub similarity: f32,
    pub accepted: bool,
}

/// A discovered relation between two concepts.
#[derive(Debug, Clone)]
pub struct DiscoveredRelation {
    pub subject_label: String,
    pub predicate_label: String,
    pub object_label: String,
    pub source: ConceptSource,
    pub confidence: f32,
}

/// Internal result from KG insertion.
type InsertionResult = (usize, usize, Vec<ProvenanceId>, Option<SymbolId>, Option<SymbolId>);

/// Result of a domain expansion run.
#[derive(Debug, Clone)]
pub struct ExpansionResult {
    pub concept_count: usize,
    pub relation_count: usize,
    pub rejected_count: usize,
    pub api_calls: usize,
    pub domain_prototype_id: Option<SymbolId>,
    pub microtheory_id: Option<SymbolId>,
    pub provenance_ids: Vec<ProvenanceId>,
    pub accepted_labels: Vec<String>,
    pub boundary_rejects: Vec<String>,
}

// ── DomainExpander ──────────────────────────────────────────────────────

/// Performs a single domain expansion run, creating a skeleton ontology in the KG.
///
/// Not persisted as long-lived state — created per expansion run. The KG triples
/// and provenance records it produces are the durable artifacts.
pub struct DomainExpander {
    config: ExpansionConfig,
    predicates: ExpansionPredicates,
    #[allow(dead_code)] // reserved for deeper VSA integration in Phase 14d+
    role_vectors: ExpansionRoleVectors,
    api_call_count: usize,
}

impl DomainExpander {
    /// Create a new expander for a single expansion run.
    pub fn new(engine: &Engine, config: ExpansionConfig) -> ExpandResult<Self> {
        let predicates = ExpansionPredicates::init(engine)?;
        let role_vectors = ExpansionRoleVectors::init(engine.ops());
        Ok(Self {
            config,
            predicates,
            role_vectors,
            api_call_count: 0,
        })
    }

    /// Run domain expansion from a purpose model.
    pub fn expand(
        &mut self,
        purpose: &PurposeModel,
        engine: &Arc<Engine>,
    ) -> ExpandResult<ExpansionResult> {
        let seeds = &purpose.seed_concepts;
        if seeds.is_empty() {
            return Err(DomainExpandError::NoSeeds);
        }

        let ops = engine.ops();

        // 1. Build domain prototype: bundle of encoded seed labels.
        let domain_prototype = self.build_domain_prototype(ops, seeds)?;

        // 2. Collect candidates and relations from external sources.
        let mut candidates: Vec<CandidateConcept> = Vec::new();
        let mut relations: Vec<DiscoveredRelation> = Vec::new();

        // Add seeds themselves as candidates.
        for seed in seeds {
            candidates.push(CandidateConcept {
                label: seed.clone(),
                source: ConceptSource::Seed,
                depth: 0,
                similarity: 1.0,
                accepted: true,
            });
        }

        // Query external sources for each seed.
        for seed in seeds {
            self.query_wikidata(seed, &mut candidates, &mut relations);
            self.query_wikipedia(seed, &mut candidates, &mut relations);
            if self.config.use_conceptnet {
                self.query_conceptnet(seed, &mut candidates, &mut relations);
            }
        }

        // 3. Deduplicate candidates by normalized label.
        let mut seen: HashSet<String> = HashSet::new();
        candidates.retain(|c| seen.insert(normalize_label(&c.label)));

        // 4. Score and filter by VSA similarity.
        let mut total_candidates = 0;
        for candidate in &mut candidates {
            if candidate.source == ConceptSource::Seed {
                continue; // seeds always accepted
            }
            total_candidates += 1;
            if let Ok(vec) = encode_label(ops, &candidate.label) {
                candidate.similarity = ops.similarity(&vec, &domain_prototype).unwrap_or(0.5);
            }
            candidate.accepted = candidate.similarity >= self.config.similarity_threshold;
        }

        // 5. Cap at max_concepts.
        // Sort accepted by similarity descending, take top max_concepts.
        let mut accepted: Vec<_> = candidates.iter().filter(|c| c.accepted).cloned().collect();
        accepted.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        accepted.truncate(self.config.max_concepts);

        let accepted_labels: Vec<String> = accepted.iter().map(|c| c.label.clone()).collect();
        let accepted_set: HashSet<String> = accepted_labels.iter().map(|l| normalize_label(l)).collect();

        let rejected: Vec<_> = candidates
            .iter()
            .filter(|c| !c.accepted)
            .map(|c| c.label.clone())
            .collect();

        if accepted.is_empty() {
            return Err(DomainExpandError::EmptyExpansion {
                threshold: self.config.similarity_threshold,
                candidate_count: total_candidates,
            });
        }

        // 6. Insert into KG.
        let (concept_count, relation_count, provenance_ids, domain_proto_id, microtheory_id) =
            self.insert_into_kg(engine, &accepted, &relations, &accepted_set, seeds)?;

        // 7. Store the domain prototype HV in item memory so downstream phases
        //    (e.g. resource discovery) can retrieve it for similarity scoring.
        if let Some(proto_id) = domain_proto_id {
            engine.item_memory().insert(proto_id, domain_prototype);
        }

        Ok(ExpansionResult {
            concept_count,
            relation_count,
            rejected_count: rejected.len(),
            api_calls: self.api_call_count,
            domain_prototype_id: domain_proto_id,
            microtheory_id,
            provenance_ids,
            accepted_labels,
            boundary_rejects: rejected,
        })
    }

    // ── VSA Boundary ────────────────────────────────────────────────────

    /// Build the domain prototype vector by bundling encoded seed labels.
    fn build_domain_prototype(
        &self,
        ops: &VsaOps,
        seeds: &[String],
    ) -> ExpandResult<HyperVec> {
        let vecs: Vec<HyperVec> = seeds
            .iter()
            .filter_map(|s| encode_label(ops, s).ok())
            .collect();

        if vecs.is_empty() {
            return Err(DomainExpandError::NoSeeds);
        }

        if vecs.len() == 1 {
            return Ok(vecs.into_iter().next().unwrap());
        }

        let refs: Vec<&HyperVec> = vecs.iter().collect();
        ops.bundle(&refs).map_err(|e| {
            DomainExpandError::Engine(Box::new(crate::error::AkhError::Vsa(e)))
        })
    }

    /// Check if a candidate is within the domain boundary.
    pub fn is_within_boundary(
        ops: &VsaOps,
        candidate_label: &str,
        domain_prototype: &HyperVec,
        threshold: f32,
    ) -> bool {
        encode_label(ops, candidate_label)
            .ok()
            .and_then(|vec| ops.similarity(&vec, domain_prototype).ok())
            .is_some_and(|sim| sim >= threshold)
    }

    // ── External API Queries ────────────────────────────────────────────

    /// Make an API call with rate limiting. Returns `None` if budget exhausted.
    fn api_call(&mut self, url: &str) -> Option<serde_json::Value> {
        if self.api_call_count >= self.config.max_api_calls {
            return None;
        }
        self.api_call_count += 1;

        if self.config.inter_call_delay_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(
                self.config.inter_call_delay_ms,
            ));
        }

        let body: serde_json::Value = ureq::get(url)
            .timeout(std::time::Duration::from_secs(10))
            .call()
            .ok()?
            .into_json()
            .ok()?;
        Some(body)
    }

    /// Query Wikidata for entity search + claims.
    fn query_wikidata(
        &mut self,
        seed: &str,
        candidates: &mut Vec<CandidateConcept>,
        relations: &mut Vec<DiscoveredRelation>,
    ) {
        let encoded = simple_url_encode(seed);

        // 1. Search for entities.
        let search_url = format!(
            "https://www.wikidata.org/w/api.php?action=wbsearchentities&search={encoded}&language=en&limit=3&format=json"
        );
        let Some(search_data) = self.api_call(&search_url) else {
            return;
        };

        let qids: Vec<String> = parse_wikidata_search(&search_data);
        if qids.is_empty() {
            return;
        }

        // 2. Fetch entity claims.
        let ids_param = qids.join("|");
        let entity_url = format!(
            "https://www.wikidata.org/w/api.php?action=wbgetentities&ids={ids_param}&props=claims|labels&languages=en&format=json"
        );
        let Some(entity_data) = self.api_call(&entity_url) else {
            return;
        };

        parse_wikidata_entities(&entity_data, seed, candidates, relations);
    }

    /// Query Wikipedia for categories and category members.
    fn query_wikipedia(
        &mut self,
        seed: &str,
        candidates: &mut Vec<CandidateConcept>,
        relations: &mut Vec<DiscoveredRelation>,
    ) {
        let encoded = simple_url_encode(seed);

        // 1. Get categories for the article.
        let cat_url = format!(
            "https://en.wikipedia.org/w/api.php?action=query&titles={encoded}&prop=categories&cllimit=20&format=json"
        );
        let Some(cat_data) = self.api_call(&cat_url) else {
            return;
        };

        let categories = parse_wikipedia_categories(&cat_data);

        // 2. For the first few useful categories, get members.
        let useful_cats: Vec<_> = categories
            .iter()
            .filter(|c| !is_wikipedia_meta_category(c))
            .take(3)
            .cloned()
            .collect();

        for cat in &useful_cats {
            let cat_encoded = simple_url_encode(cat);
            let members_url = format!(
                "https://en.wikipedia.org/w/api.php?action=query&list=categorymembers&cmtitle=Category:{cat_encoded}&cmlimit=20&cmtype=page&format=json"
            );
            let Some(members_data) = self.api_call(&members_url) else {
                continue;
            };

            let members = parse_wikipedia_categorymembers(&members_data);
            for member in members {
                candidates.push(CandidateConcept {
                    label: member.clone(),
                    source: ConceptSource::Wikipedia,
                    depth: 1,
                    similarity: 0.0,
                    accepted: false,
                });
                relations.push(DiscoveredRelation {
                    subject_label: member,
                    predicate_label: "subclass_of".to_string(),
                    object_label: cat.clone(),
                    source: ConceptSource::Wikipedia,
                    confidence: 0.6,
                });
            }
        }
    }

    /// Query ConceptNet for edges.
    fn query_conceptnet(
        &mut self,
        seed: &str,
        candidates: &mut Vec<CandidateConcept>,
        relations: &mut Vec<DiscoveredRelation>,
    ) {
        let encoded = seed.to_lowercase().replace(' ', "_");
        let url = format!("http://api.conceptnet.io/c/en/{encoded}?limit=20");
        let Some(data) = self.api_call(&url) else {
            return;
        };

        parse_conceptnet_edges(&data, seed, candidates, relations);
    }

    // ── KG Insertion ────────────────────────────────────────────────────

    /// Insert accepted concepts and relations into the knowledge graph.
    fn insert_into_kg(
        &self,
        engine: &Arc<Engine>,
        accepted: &[CandidateConcept],
        relations: &[DiscoveredRelation],
        accepted_set: &HashSet<String>,
        seeds: &[String],
    ) -> ExpandResult<InsertionResult> {
        let mut provenance_ids = Vec::new();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Create a microtheory context for this expansion.
        let microtheory_label = format!(
            "expand:microtheory:{}",
            seeds.first().map(|s| s.as_str()).unwrap_or("unknown")
        );
        let microtheory_id = engine.resolve_or_create_entity(&microtheory_label)?;

        // Create the domain prototype entity.
        let prototype_label = format!(
            "expand:prototype:{}",
            seeds.first().map(|s| s.as_str()).unwrap_or("unknown")
        );
        let prototype_id = engine.resolve_or_create_entity(&prototype_label)?;

        // Create entity symbols for each accepted concept.
        let mut concept_count = 0;
        for concept in accepted {
            let concept_id = engine.resolve_or_create_entity(&concept.label)?;

            // Link to domain prototype.
            let triple = Triple {
                subject: concept_id,
                predicate: self.predicates.expanded_from,
                object: prototype_id,
                confidence: concept.similarity,
                timestamp: now,
                provenance_id: None,
                compartment_id: None,
            };
            engine.add_triple(&triple)?;
            concept_count += 1;
        }

        // Insert discovered relations (only for accepted concepts).
        let mut relation_count = 0;
        for rel in relations {
            let subj_norm = normalize_label(&rel.subject_label);
            let obj_norm = normalize_label(&rel.object_label);
            if !accepted_set.contains(&subj_norm) && !accepted_set.contains(&obj_norm) {
                continue; // skip if neither end is accepted
            }

            let subj_id = engine.resolve_or_create_entity(&rel.subject_label)?;
            let pred_id = self.map_predicate_label(&rel.predicate_label, engine)?;
            let obj_id = engine.resolve_or_create_entity(&rel.object_label)?;

            let triple = Triple {
                subject: subj_id,
                predicate: pred_id,
                object: obj_id,
                confidence: rel.confidence,
                timestamp: now,
                provenance_id: None,
                compartment_id: None,
            };
            engine.add_triple(&triple)?;
            relation_count += 1;
        }

        // Store provenance: one record per seed.
        for seed in seeds {
            let mut record = ProvenanceRecord::new(
                prototype_id,
                DerivationKind::DomainExpansion {
                    seed_label: seed.clone(),
                    concept_count: concept_count as u32,
                    relation_count: relation_count as u32,
                    source: "wikidata+wikipedia+conceptnet".to_string(),
                },
            )
            .with_confidence(0.8);

            if let Ok(prov_id) = engine.store_provenance(&mut record) {
                provenance_ids.push(prov_id);
            }
        }

        Ok((
            concept_count,
            relation_count,
            provenance_ids,
            Some(prototype_id),
            Some(microtheory_id),
        ))
    }

    /// Map a predicate label string to the appropriate predicate SymbolId.
    fn map_predicate_label(
        &self,
        label: &str,
        engine: &Engine,
    ) -> ExpandResult<SymbolId> {
        match label {
            "instance_of" => Ok(self.predicates.instance_of),
            "subclass_of" => Ok(self.predicates.subclass_of),
            "part_of" => Ok(self.predicates.part_of),
            "has_part" => Ok(self.predicates.has_part),
            "related_to" => Ok(self.predicates.related_to),
            "has_prerequisite" => Ok(self.predicates.has_prerequisite),
            "used_for" => Ok(self.predicates.used_for),
            "expanded_from" => Ok(self.predicates.expanded_from),
            _ => Ok(engine.resolve_or_create_relation(&format!("expand:{label}"))?),
        }
    }
}

// ── JSON Parsing Helpers ────────────────────────────────────────────────

/// Parse Wikidata search results, returning QIDs.
pub fn parse_wikidata_search(data: &serde_json::Value) -> Vec<String> {
    data.get("search")
        .and_then(|s| s.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| item.get("id").and_then(|id| id.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// Parse Wikidata entity claims for instance-of (P31), subclass-of (P279),
/// part-of (P361), and has-part (P527). Adds candidates and relations.
pub fn parse_wikidata_entities(
    data: &serde_json::Value,
    seed: &str,
    candidates: &mut Vec<CandidateConcept>,
    relations: &mut Vec<DiscoveredRelation>,
) {
    let Some(entities) = data.get("entities").and_then(|e| e.as_object()) else {
        return;
    };

    // Map Wikidata property IDs to relation labels.
    let property_map = [
        ("P31", "instance_of"),
        ("P279", "subclass_of"),
        ("P361", "part_of"),
        ("P527", "has_part"),
    ];

    for (_qid, entity) in entities {
        // Get the entity label.
        let entity_label = entity
            .pointer("/labels/en/value")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if !entity_label.is_empty() && entity_label.to_lowercase() != seed.to_lowercase() {
            candidates.push(CandidateConcept {
                label: entity_label.to_string(),
                source: ConceptSource::Wikidata,
                depth: 1,
                similarity: 0.0,
                accepted: false,
            });
        }

        let Some(claims) = entity.get("claims").and_then(|c| c.as_object()) else {
            continue;
        };

        for (prop_id, rel_label) in &property_map {
            let Some(prop_claims) = claims.get(*prop_id).and_then(|p| p.as_array()) else {
                continue;
            };

            for claim in prop_claims {
                let target_label = claim
                    .pointer("/mainsnak/datavalue/value/id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                if target_label.is_empty() {
                    continue;
                }

                // The target is a QID; we use it as a placeholder label.
                // In a full implementation we'd resolve the QID to its label.
                let label = format!("wd:{target_label}");
                candidates.push(CandidateConcept {
                    label: label.clone(),
                    source: ConceptSource::Wikidata,
                    depth: 1,
                    similarity: 0.0,
                    accepted: false,
                });

                relations.push(DiscoveredRelation {
                    subject_label: if entity_label.is_empty() {
                        seed.to_string()
                    } else {
                        entity_label.to_string()
                    },
                    predicate_label: rel_label.to_string(),
                    object_label: label,
                    source: ConceptSource::Wikidata,
                    confidence: 0.7,
                });
            }
        }
    }
}

/// Parse Wikipedia category list from API response.
pub fn parse_wikipedia_categories(data: &serde_json::Value) -> Vec<String> {
    let Some(pages) = data.pointer("/query/pages").and_then(|p| p.as_object()) else {
        return Vec::new();
    };

    let mut categories = Vec::new();
    for (_page_id, page) in pages {
        if let Some(cats) = page.get("categories").and_then(|c| c.as_array()) {
            for cat in cats {
                if let Some(title) = cat.get("title").and_then(|t| t.as_str()) {
                    // Strip "Category:" prefix.
                    let name = title.strip_prefix("Category:").unwrap_or(title);
                    categories.push(name.to_string());
                }
            }
        }
    }
    categories
}

/// Parse Wikipedia category members from API response.
pub fn parse_wikipedia_categorymembers(data: &serde_json::Value) -> Vec<String> {
    data.pointer("/query/categorymembers")
        .and_then(|cm| cm.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| item.get("title").and_then(|t| t.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// Parse ConceptNet edges and add candidates/relations.
pub fn parse_conceptnet_edges(
    data: &serde_json::Value,
    seed: &str,
    candidates: &mut Vec<CandidateConcept>,
    relations: &mut Vec<DiscoveredRelation>,
) {
    let Some(edges) = data.get("edges").and_then(|e| e.as_array()) else {
        return;
    };

    for edge in edges {
        let rel = edge
            .pointer("/rel/label")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let start_label = edge
            .pointer("/start/label")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let end_label = edge
            .pointer("/end/label")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let weight = edge
            .get("weight")
            .and_then(|w| w.as_f64())
            .unwrap_or(1.0) as f32;

        // Only handle known relations.
        let predicate_label = match rel {
            "IsA" => "instance_of",
            "PartOf" => "part_of",
            "HasA" => "has_part",
            "RelatedTo" => "related_to",
            "HasPrerequisite" => "has_prerequisite",
            "UsedFor" => "used_for",
            _ => continue,
        };

        // Determine which end is the new concept.
        let (new_label, subj, obj) = if start_label.to_lowercase() == seed.to_lowercase() {
            (end_label, start_label, end_label)
        } else if end_label.to_lowercase() == seed.to_lowercase() {
            (start_label, start_label, end_label)
        } else {
            // Neither end matches the seed; skip.
            continue;
        };

        if !new_label.is_empty() {
            candidates.push(CandidateConcept {
                label: new_label.to_string(),
                source: ConceptSource::ConceptNet,
                depth: 1,
                similarity: 0.0,
                accepted: false,
            });

            relations.push(DiscoveredRelation {
                subject_label: subj.to_string(),
                predicate_label: predicate_label.to_string(),
                object_label: obj.to_string(),
                source: ConceptSource::ConceptNet,
                confidence: (weight / 10.0).clamp(0.1, 1.0),
            });
        }
    }
}

// ── Utility Helpers ─────────────────────────────────────────────────────

/// Normalize a label for deduplication: lowercase, trim, hyphens→spaces.
pub fn normalize_label(label: &str) -> String {
    let mut s = label.trim().to_lowercase();
    // Replace both hyphens and underscores with spaces in a single pass.
    // SAFETY: replacing ASCII bytes with ASCII bytes preserves UTF-8 validity.
    unsafe {
        for byte in s.as_bytes_mut() {
            if *byte == b'-' || *byte == b'_' {
                *byte = b' ';
            }
        }
    }
    s
}

/// Simple percent-encoding for URL query parameters.
fn simple_url_encode(input: &str) -> String {
    let mut encoded = String::with_capacity(input.len() * 2);
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char);
            }
            b' ' => encoded.push_str("%20"),
            _ => {
                encoded.push_str(&format!("%{:02X}", byte));
            }
        }
    }
    encoded
}

/// Check if a Wikipedia category is a meta/maintenance category.
fn is_wikipedia_meta_category(cat: &str) -> bool {
    let lower = cat.to_lowercase();
    lower.contains("articles")
        || lower.contains("pages")
        || lower.contains("stubs")
        || lower.contains("wikidata")
        || lower.contains("webarchive")
        || lower.contains("cs1")
        || lower.contains("wikipedia")
        || lower.contains("accuracy disputes")
        || lower.contains("short description")
        || lower.contains("use dmy dates")
        || lower.contains("use mdy dates")
        || lower.contains("all articles")
        || lower.contains("commons category")
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Config ──────────────────────────────────────────────────────

    #[test]
    fn expansion_config_default_values() {
        let config = ExpansionConfig::default();
        assert!((config.similarity_threshold - 0.6).abs() < f32::EPSILON);
        assert_eq!(config.max_depth, 3);
        assert_eq!(config.max_concepts, 200);
        assert_eq!(config.max_api_calls, 50);
        assert_eq!(config.inter_call_delay_ms, 200);
        assert!(config.use_conceptnet);
    }

    // ── ConceptSource display ───────────────────────────────────────

    #[test]
    fn concept_source_display() {
        assert_eq!(format!("{}", ConceptSource::Seed), "seed");
        assert_eq!(format!("{}", ConceptSource::Wikidata), "wikidata");
        assert_eq!(format!("{}", ConceptSource::Wikipedia), "wikipedia");
        assert_eq!(format!("{}", ConceptSource::ConceptNet), "conceptnet");
    }

    // ── Normalize label ─────────────────────────────────────────────

    #[test]
    fn normalize_label_cases() {
        assert_eq!(normalize_label("Hello World"), "hello world");
        assert_eq!(normalize_label("  Compiler  "), "compiler");
        assert_eq!(normalize_label("type-system"), "type system");
        assert_eq!(normalize_label("hello_world"), "hello world");
        assert_eq!(normalize_label("UPPER-CASE_mixed"), "upper case mixed");
    }

    // ── Deduplication ───────────────────────────────────────────────

    #[test]
    fn deduplication_by_normalized_label() {
        let mut candidates = vec![
            CandidateConcept {
                label: "Compiler".to_string(),
                source: ConceptSource::Wikidata,
                depth: 1,
                similarity: 0.8,
                accepted: true,
            },
            CandidateConcept {
                label: "compiler".to_string(),
                source: ConceptSource::ConceptNet,
                depth: 1,
                similarity: 0.7,
                accepted: true,
            },
            CandidateConcept {
                label: "Optimization".to_string(),
                source: ConceptSource::Wikipedia,
                depth: 1,
                similarity: 0.9,
                accepted: true,
            },
        ];

        let mut seen: HashSet<String> = HashSet::new();
        candidates.retain(|c| seen.insert(normalize_label(&c.label)));
        assert_eq!(candidates.len(), 2);
    }

    // ── Wikidata parsing ────────────────────────────────────────────

    #[test]
    fn parse_wikidata_search_response() {
        let data: serde_json::Value = serde_json::json!({
            "search": [
                {"id": "Q183", "label": "Germany"},
                {"id": "Q7889", "label": "compiler"}
            ]
        });
        let qids = parse_wikidata_search(&data);
        assert_eq!(qids, vec!["Q183", "Q7889"]);
    }

    #[test]
    fn parse_wikidata_search_empty() {
        let data: serde_json::Value = serde_json::json!({"search": []});
        let qids = parse_wikidata_search(&data);
        assert!(qids.is_empty());
    }

    #[test]
    fn parse_wikidata_entity_claims() {
        let data: serde_json::Value = serde_json::json!({
            "entities": {
                "Q7889": {
                    "labels": {"en": {"value": "compiler"}},
                    "claims": {
                        "P31": [{
                            "mainsnak": {"datavalue": {"value": {"id": "Q21198342"}}}
                        }],
                        "P279": [{
                            "mainsnak": {"datavalue": {"value": {"id": "Q166142"}}}
                        }]
                    }
                }
            }
        });

        let mut candidates = Vec::new();
        let mut relations = Vec::new();
        parse_wikidata_entities(&data, "compiler", &mut candidates, &mut relations);

        // Should have added candidates for P31 and P279 targets.
        assert!(candidates.iter().any(|c| c.label == "wd:Q21198342"));
        assert!(candidates.iter().any(|c| c.label == "wd:Q166142"));

        // Should have relations.
        assert!(relations.iter().any(|r| r.predicate_label == "instance_of"));
        assert!(relations.iter().any(|r| r.predicate_label == "subclass_of"));
    }

    // ── Wikipedia parsing ───────────────────────────────────────────

    #[test]
    fn parse_wikipedia_categories_response() {
        let data: serde_json::Value = serde_json::json!({
            "query": {
                "pages": {
                    "12345": {
                        "categories": [
                            {"title": "Category:Computer science"},
                            {"title": "Category:Articles with short description"}
                        ]
                    }
                }
            }
        });
        let cats = parse_wikipedia_categories(&data);
        assert_eq!(cats, vec!["Computer science", "Articles with short description"]);
    }

    #[test]
    fn parse_wikipedia_categorymembers_response() {
        let data: serde_json::Value = serde_json::json!({
            "query": {
                "categorymembers": [
                    {"title": "Parser"},
                    {"title": "Lexer"},
                    {"title": "Abstract syntax tree"}
                ]
            }
        });
        let members = parse_wikipedia_categorymembers(&data);
        assert_eq!(members, vec!["Parser", "Lexer", "Abstract syntax tree"]);
    }

    // ── ConceptNet parsing ──────────────────────────────────────────

    #[test]
    fn parse_conceptnet_edges_response() {
        let data: serde_json::Value = serde_json::json!({
            "edges": [
                {
                    "rel": {"label": "IsA"},
                    "start": {"label": "compiler"},
                    "end": {"label": "software"},
                    "weight": 4.5
                },
                {
                    "rel": {"label": "RelatedTo"},
                    "start": {"label": "compiler"},
                    "end": {"label": "programming"},
                    "weight": 3.2
                },
                {
                    "rel": {"label": "Synonym"},
                    "start": {"label": "compiler"},
                    "end": {"label": "translator"},
                    "weight": 1.0
                }
            ]
        });

        let mut candidates = Vec::new();
        let mut relations = Vec::new();
        parse_conceptnet_edges(&data, "compiler", &mut candidates, &mut relations);

        // Synonym should be skipped.
        assert_eq!(candidates.len(), 2);
        assert!(candidates.iter().any(|c| c.label == "software"));
        assert!(candidates.iter().any(|c| c.label == "programming"));
        assert_eq!(relations.len(), 2);
    }

    // ── ConceptNet relation mapping ─────────────────────────────────

    #[test]
    fn conceptnet_relation_mapping() {
        // We can't fully test this without an engine, but we can test
        // the logic of the predicate_label matching in parse_conceptnet_edges.
        let data: serde_json::Value = serde_json::json!({
            "edges": [
                {"rel": {"label": "IsA"}, "start": {"label": "dog"}, "end": {"label": "animal"}, "weight": 5.0},
                {"rel": {"label": "PartOf"}, "start": {"label": "engine"}, "end": {"label": "car"}, "weight": 3.0},
                {"rel": {"label": "HasA"}, "start": {"label": "dog"}, "end": {"label": "tail"}, "weight": 4.0},
                {"rel": {"label": "HasPrerequisite"}, "start": {"label": "dog"}, "end": {"label": "food"}, "weight": 2.0},
                {"rel": {"label": "UsedFor"}, "start": {"label": "dog"}, "end": {"label": "guarding"}, "weight": 2.5}
            ]
        });

        let mut candidates = Vec::new();
        let mut relations = Vec::new();
        parse_conceptnet_edges(&data, "dog", &mut candidates, &mut relations);

        let pred_labels: Vec<_> = relations.iter().map(|r| r.predicate_label.as_str()).collect();
        assert!(pred_labels.contains(&"instance_of"));
        assert!(pred_labels.contains(&"has_part"));
        assert!(pred_labels.contains(&"has_prerequisite"));
        assert!(pred_labels.contains(&"used_for"));
    }

    // ── Rate limiting ───────────────────────────────────────────────

    #[test]
    fn rate_limit_enforcement() {
        // DomainExpander needs an engine, but we can test the counter logic
        // by checking the configuration constraint directly.
        let config = ExpansionConfig {
            max_api_calls: 3,
            inter_call_delay_ms: 0,
            ..ExpansionConfig::default()
        };
        assert_eq!(config.max_api_calls, 3);
        // The actual enforcement happens in api_call() which requires network.
    }

    // ── Empty seeds error ───────────────────────────────────────────

    #[test]
    fn empty_seeds_error() {
        let err = DomainExpandError::NoSeeds;
        let msg = format!("{err}");
        assert!(msg.contains("no seed"));
    }

    // ── ExpansionResult ─────────────────────────────────────────────

    #[test]
    fn expansion_result_serialization() {
        let result = ExpansionResult {
            concept_count: 42,
            relation_count: 15,
            rejected_count: 8,
            api_calls: 12,
            domain_prototype_id: None,
            microtheory_id: None,
            provenance_ids: Vec::new(),
            accepted_labels: vec!["compiler".to_string(), "parser".to_string()],
            boundary_rejects: vec!["unrelated".to_string()],
        };
        assert_eq!(result.concept_count, 42);
        assert_eq!(result.relation_count, 15);
        assert_eq!(result.rejected_count, 8);
        assert_eq!(result.accepted_labels.len(), 2);
    }

    // ── Wikipedia meta-category filter ──────────────────────────────

    #[test]
    fn meta_category_filter() {
        assert!(is_wikipedia_meta_category("Articles with short description"));
        assert!(is_wikipedia_meta_category("All stub articles"));
        assert!(is_wikipedia_meta_category("CS1 maint: archived copy"));
        assert!(!is_wikipedia_meta_category("Computer science"));
        assert!(!is_wikipedia_meta_category("Compilers"));
    }

    // ── Candidate boundary ──────────────────────────────────────────

    #[test]
    fn candidate_boundary_accept() {
        let mut c = CandidateConcept {
            label: "parser".to_string(),
            source: ConceptSource::Wikidata,
            depth: 1,
            similarity: 0.75,
            accepted: false,
        };
        c.accepted = c.similarity >= 0.6;
        assert!(c.accepted);
    }

    #[test]
    fn candidate_boundary_reject() {
        let mut c = CandidateConcept {
            label: "unrelated".to_string(),
            source: ConceptSource::Wikidata,
            depth: 1,
            similarity: 0.45,
            accepted: false,
        };
        c.accepted = c.similarity >= 0.6;
        assert!(!c.accepted);
    }

    // ── URL encoding ────────────────────────────────────────────────

    #[test]
    fn url_encode_basic() {
        assert_eq!(simple_url_encode("hello world"), "hello%20world");
        assert_eq!(simple_url_encode("compiler"), "compiler");
        assert_eq!(simple_url_encode("a&b=c"), "a%26b%3Dc");
    }
}
