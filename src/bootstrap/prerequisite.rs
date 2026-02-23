//! Prerequisite discovery & Vygotsky ZPD classification (Phase 14d).
//!
//! Takes a skeleton ontology produced by domain expansion (Phase 14c) and:
//! 1. Discovers prerequisite relationships between concepts (building a DAG).
//! 2. Classifies each concept into Vygotsky ZPD zones (Known, Proximal, Beyond).
//! 3. Generates a curriculum ordering via topological sort.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::Arc;

use miette::Diagnostic;
use thiserror::Error;

use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceId, ProvenanceRecord};
use crate::symbol::SymbolId;
use crate::vsa::encode::{encode_label, encode_token};
use crate::vsa::HyperVec;

use super::expand::ExpansionResult;

// ── Errors ──────────────────────────────────────────────────────────────

/// Errors from prerequisite discovery and ZPD classification.
#[derive(Debug, Error, Diagnostic)]
pub enum PrerequisiteError {
    #[error("no concepts found in the knowledge graph for prerequisite analysis")]
    #[diagnostic(
        code(akh::bootstrap::prerequisite::no_concepts),
        help(
            "Run domain expansion first with `akh awaken expand --seeds ...` \
             to populate the knowledge graph with concepts."
        )
    )]
    NoConcepts,

    #[error("cycle detected in prerequisite graph ({cycle_size} concepts involved)")]
    #[diagnostic(
        code(akh::bootstrap::prerequisite::cycle_detected),
        help(
            "The prerequisite graph contains cycles. Cycle-breaking removed the \
             lowest-confidence edges, but {cycle_size} concepts were involved."
        )
    )]
    CycleDetected { cycle_size: usize },

    #[error("no valid prerequisite edges found among {analyzed} concepts")]
    #[diagnostic(
        code(akh::bootstrap::prerequisite::empty_curriculum),
        help(
            "Lower min_edge_confidence or provide more specific seed concepts \
             with richer prerequisite relationships."
        )
    )]
    EmptyCurriculum { analyzed: usize },

    #[error("{0}")]
    #[diagnostic(
        code(akh::bootstrap::prerequisite::engine),
        help("An engine-level error occurred during prerequisite analysis.")
    )]
    Engine(Box<crate::error::AkhError>),
}

impl From<crate::error::AkhError> for PrerequisiteError {
    fn from(e: crate::error::AkhError) -> Self {
        Self::Engine(Box::new(e))
    }
}

/// Convenience alias for prerequisite operations.
pub type PrerequisiteResult<T> = std::result::Result<T, PrerequisiteError>;

// ── Configuration ───────────────────────────────────────────────────────

/// Configuration for prerequisite analysis and ZPD classification.
#[derive(Debug, Clone)]
pub struct PrerequisiteConfig {
    /// Minimum triple count for a concept to be considered "Known".
    pub known_min_triples: usize,
    /// Minimum VSA similarity to known bundle for "Known" classification.
    pub known_similarity_threshold: f32,
    /// Minimum prerequisite coverage fraction for "Proximal" classification.
    pub proximal_min_prereq_coverage: f32,
    /// Lower similarity bound for "Proximal" zone.
    pub proximal_similarity_low: f32,
    /// Upper similarity bound for "Proximal" zone (above → "Known" candidate).
    pub proximal_similarity_high: f32,
    /// Minimum confidence for a prerequisite edge to be retained.
    pub min_edge_confidence: f32,
    /// Weight for ConceptNet-imported prerequisite edges.
    pub weight_conceptnet: f32,
    /// Weight for structural heuristic prerequisite edges.
    pub weight_structural: f32,
    /// Weight for VSA asymmetric similarity prerequisite edges.
    pub weight_vsa: f32,
    /// Maximum number of prerequisites per concept.
    pub max_prereqs_per_concept: usize,
}

impl Default for PrerequisiteConfig {
    fn default() -> Self {
        Self {
            known_min_triples: 5,
            known_similarity_threshold: 0.7,
            proximal_min_prereq_coverage: 0.5,
            proximal_similarity_low: 0.3,
            proximal_similarity_high: 0.7,
            min_edge_confidence: 0.3,
            weight_conceptnet: 0.5,
            weight_structural: 0.3,
            weight_vsa: 0.2,
            max_prereqs_per_concept: 10,
        }
    }
}

// ── Predicates ──────────────────────────────────────────────────────────

/// Well-known relation predicates for prerequisite discovery (`prereq:` namespace).
pub struct PrerequisitePredicates {
    pub prerequisite_of: SymbolId,
    pub zpd_zone: SymbolId,
    pub curriculum_tier: SymbolId,
    pub prereq_coverage: SymbolId,
    pub similarity_to_known: SymbolId,
}

impl PrerequisitePredicates {
    /// Initialize predicates by resolving or creating relation symbols.
    fn init(engine: &Engine) -> PrerequisiteResult<Self> {
        Ok(Self {
            prerequisite_of: engine.resolve_or_create_relation("prereq:prerequisite_of")?,
            zpd_zone: engine.resolve_or_create_relation("prereq:zpd_zone")?,
            curriculum_tier: engine.resolve_or_create_relation("prereq:curriculum_tier")?,
            prereq_coverage: engine.resolve_or_create_relation("prereq:prereq_coverage")?,
            similarity_to_known: engine.resolve_or_create_relation("prereq:similarity_to_known")?,
        })
    }
}

// ── Data Types ──────────────────────────────────────────────────────────

/// Vygotsky ZPD zone classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ZpdZone {
    /// The concept is already well-known (rich in triples, high similarity).
    Known,
    /// The concept is within the Zone of Proximal Development — learnable next.
    Proximal,
    /// The concept is beyond current reach — prerequisites not met.
    Beyond,
}

impl ZpdZone {
    /// Convert to a stable string label.
    pub fn as_label(&self) -> &'static str {
        match self {
            Self::Known => "known",
            Self::Proximal => "proximal",
            Self::Beyond => "beyond",
        }
    }

    /// Parse from a label string.
    pub fn from_label(s: &str) -> Option<Self> {
        match s {
            "known" => Some(Self::Known),
            "proximal" => Some(Self::Proximal),
            "beyond" => Some(Self::Beyond),
            _ => None,
        }
    }
}

impl fmt::Display for ZpdZone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_label())
    }
}

/// Source strategy for a prerequisite edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrerequisiteSource {
    /// Imported from ConceptNet HasPrerequisite relations.
    ConceptNet,
    /// Inferred from structural relations (subclass_of, part_of, instance_of).
    Structural,
    /// Discovered via VSA asymmetric similarity.
    VsaSimilarity,
}

impl fmt::Display for PrerequisiteSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConceptNet => write!(f, "conceptnet"),
            Self::Structural => write!(f, "structural"),
            Self::VsaSimilarity => write!(f, "vsa"),
        }
    }
}

/// A prerequisite edge: `from` is prerequisite of `to`.
#[derive(Debug, Clone)]
pub struct PrerequisiteEdge {
    /// The prerequisite concept (learn this first).
    pub from: SymbolId,
    /// The dependent concept (learn this after).
    pub to: SymbolId,
    /// Combined confidence from all strategies.
    pub confidence: f32,
    /// Which strategies contributed to this edge.
    pub sources: Vec<PrerequisiteSource>,
}

/// A single entry in the generated curriculum.
#[derive(Debug, Clone)]
pub struct CurriculumEntry {
    /// The concept symbol.
    pub concept: SymbolId,
    /// Human-readable label.
    pub label: String,
    /// ZPD classification.
    pub zone: ZpdZone,
    /// Fraction of prerequisites that are Known.
    pub prereq_coverage: f32,
    /// VSA similarity to the "known" bundle.
    pub similarity_to_known: f32,
    /// Topological tier (0 = no prerequisites).
    pub tier: u32,
    /// Direct prerequisites of this concept.
    pub prerequisites: Vec<SymbolId>,
}

/// Result of a full prerequisite analysis run.
#[derive(Debug, Clone)]
pub struct PrereqAnalysisResult {
    /// All discovered prerequisite edges (after cycle-breaking).
    pub edges: Vec<PrerequisiteEdge>,
    /// Generated curriculum in learning order.
    pub curriculum: Vec<CurriculumEntry>,
    /// Number of domain concepts analyzed.
    pub concepts_analyzed: usize,
    /// Number of prerequisite edges discovered.
    pub edge_count: usize,
    /// Number of cycles broken during DAG construction.
    pub cycles_broken: usize,
    /// Maximum tier in the curriculum.
    pub max_tier: u32,
    /// Count of concepts per ZPD zone.
    pub zone_distribution: HashMap<ZpdZone, usize>,
    /// Provenance records created.
    pub provenance_ids: Vec<ProvenanceId>,
}

// ── PrerequisiteAnalyzer ────────────────────────────────────────────────

/// Performs prerequisite discovery and ZPD classification on expanded domain concepts.
///
/// Ephemeral: created per analysis run. The KG triples and provenance records
/// it produces are the durable artifacts.
pub struct PrerequisiteAnalyzer {
    config: PrerequisiteConfig,
    predicates: PrerequisitePredicates,
    /// Reads Phase 14c `expand:has_prerequisite` edges.
    expand_has_prerequisite: SymbolId,
    /// Reads Phase 14c `expand:subclass_of` edges.
    expand_subclass_of: SymbolId,
    /// Reads Phase 14c `expand:part_of` edges.
    expand_part_of: SymbolId,
    /// Reads Phase 14c `expand:instance_of` edges.
    expand_instance_of: SymbolId,
    /// Reads Phase 14c `expand:expanded_from` edges.
    #[allow(dead_code)] // reserved for Phase 14e+ curriculum ingestion
    expand_expanded_from: SymbolId,
    /// VSA role vector for asymmetric prerequisite binding.
    prereq_role: HyperVec,
}

impl PrerequisiteAnalyzer {
    /// Create a new analyzer for a single prerequisite analysis run.
    pub fn new(engine: &Engine, config: PrerequisiteConfig) -> PrerequisiteResult<Self> {
        let predicates = PrerequisitePredicates::init(engine)?;
        let prereq_role = encode_token(engine.ops(), "prereq-role:prerequisite");

        Ok(Self {
            config,
            predicates,
            expand_has_prerequisite: engine
                .resolve_or_create_relation("expand:has_prerequisite")?,
            expand_subclass_of: engine.resolve_or_create_relation("expand:subclass_of")?,
            expand_part_of: engine.resolve_or_create_relation("expand:part_of")?,
            expand_instance_of: engine.resolve_or_create_relation("expand:instance_of")?,
            expand_expanded_from: engine.resolve_or_create_relation("expand:expanded_from")?,
            prereq_role,
        })
    }

    /// Run the full prerequisite analysis pipeline.
    ///
    /// Takes an expansion result (from Phase 14c) and the engine, discovers
    /// prerequisite relationships, breaks cycles, classifies ZPD zones, and
    /// generates a curriculum ordering.
    pub fn analyze(
        &self,
        expansion_result: &ExpansionResult,
        engine: &Arc<Engine>,
    ) -> PrerequisiteResult<PrereqAnalysisResult> {
        // 1. Collect domain concepts from expansion result.
        let domain_concepts = self.collect_domain_concepts(expansion_result, engine)?;
        if domain_concepts.is_empty() {
            return Err(PrerequisiteError::NoConcepts);
        }
        let concept_set: HashSet<SymbolId> = domain_concepts.keys().copied().collect();

        // 2. Run 3 prerequisite discovery strategies.
        let mut raw_edges: Vec<(SymbolId, SymbolId, f32, PrerequisiteSource)> = Vec::new();
        self.discover_conceptnet_prereqs(engine, &concept_set, &mut raw_edges);
        self.discover_structural_prereqs(engine, &concept_set, &mut raw_edges);
        self.discover_vsa_prereqs(engine, &domain_concepts, &mut raw_edges);

        // 3. Merge edges: deduplicate (from,to) pairs, combine confidences.
        let mut merged = self.merge_edges(&raw_edges);

        // 4. Break cycles: DFS, remove lowest-confidence edge per cycle.
        let cycles_broken = self.break_cycles(&mut merged);

        // 5. Build known bundle: VSA bundle of high-triple-count concepts.
        let known_bundle = self.build_known_bundle(engine, &domain_concepts);

        // 6. Topological sort: Kahn's algorithm → tier numbers.
        let tiers = self.topological_sort(&merged, &concept_set);

        // 7. Classify ZPD zones.
        let classifications = self.classify_zpd(
            engine,
            &domain_concepts,
            &merged,
            &tiers,
            known_bundle.as_ref(),
        );

        // 8. Generate curriculum: sort by (tier ASC, zone ASC, similarity DESC).
        let curriculum = self.generate_curriculum(engine, &classifications, &merged, &tiers);

        if merged.is_empty() && domain_concepts.len() > 1 {
            return Err(PrerequisiteError::EmptyCurriculum {
                analyzed: domain_concepts.len(),
            });
        }

        // 9. Persist to KG.
        let provenance_ids = self.persist_to_kg(engine, &merged, &classifications, &tiers)?;

        // 10. Build zone distribution.
        let mut zone_distribution = HashMap::new();
        for entry in &curriculum {
            *zone_distribution.entry(entry.zone).or_insert(0) += 1;
        }

        let max_tier = curriculum.iter().map(|e| e.tier).max().unwrap_or(0);

        Ok(PrereqAnalysisResult {
            edge_count: merged.len(),
            edges: merged,
            curriculum,
            concepts_analyzed: domain_concepts.len(),
            cycles_broken,
            max_tier,
            zone_distribution,
            provenance_ids,
        })
    }

    // ── Step 1: Collect domain concepts ─────────────────────────────────

    /// Collect domain concepts from the expansion result.
    ///
    /// Returns a map of SymbolId → label for all concepts found in the expansion.
    fn collect_domain_concepts(
        &self,
        expansion_result: &ExpansionResult,
        engine: &Arc<Engine>,
    ) -> PrerequisiteResult<HashMap<SymbolId, String>> {
        let mut concepts = HashMap::new();
        for label in &expansion_result.accepted_labels {
            if let Some(id) = engine.registry().lookup(label) {
                concepts.insert(id, label.clone());
            }
        }
        Ok(concepts)
    }

    // ── Step 2a: ConceptNet prerequisite import ─────────────────────────

    /// Strategy 1: Read `expand:has_prerequisite` triples from Phase 14c.
    ///
    /// ConceptNet `(A, HasPrerequisite, B)` means B is prerequisite of A,
    /// so edge direction is `from: B, to: A`.
    fn discover_conceptnet_prereqs(
        &self,
        engine: &Arc<Engine>,
        concept_set: &HashSet<SymbolId>,
        edges: &mut Vec<(SymbolId, SymbolId, f32, PrerequisiteSource)>,
    ) {
        let triples = engine
            .knowledge_graph()
            .triples_for_predicate(self.expand_has_prerequisite);
        for (subject, object) in triples {
            // subject has_prerequisite object → object is prereq of subject
            if concept_set.contains(&subject) && concept_set.contains(&object) {
                edges.push((
                    object,
                    subject,
                    self.config.weight_conceptnet,
                    PrerequisiteSource::ConceptNet,
                ));
            }
        }
    }

    // ── Step 2b: Structural heuristic prerequisites ─────────────────────

    /// Strategy 2: Infer prerequisites from structural relations.
    ///
    /// - `A subclass_of B` → B prereq of A (general before specific), confidence 0.6
    /// - `A part_of B` → A prereq of B (part before whole), confidence 0.5
    /// - `A instance_of B` → B prereq of A (type before instance), confidence 0.4
    fn discover_structural_prereqs(
        &self,
        engine: &Arc<Engine>,
        concept_set: &HashSet<SymbolId>,
        edges: &mut Vec<(SymbolId, SymbolId, f32, PrerequisiteSource)>,
    ) {
        // subclass_of: A subclass_of B → B is prereq of A
        let subclass_triples = engine
            .knowledge_graph()
            .triples_for_predicate(self.expand_subclass_of);
        for (subject, object) in subclass_triples {
            if concept_set.contains(&subject) && concept_set.contains(&object) {
                edges.push((
                    object,
                    subject,
                    0.6 * self.config.weight_structural,
                    PrerequisiteSource::Structural,
                ));
            }
        }

        // part_of: A part_of B → A is prereq of B
        let part_of_triples = engine
            .knowledge_graph()
            .triples_for_predicate(self.expand_part_of);
        for (subject, object) in part_of_triples {
            if concept_set.contains(&subject) && concept_set.contains(&object) {
                edges.push((
                    subject,
                    object,
                    0.5 * self.config.weight_structural,
                    PrerequisiteSource::Structural,
                ));
            }
        }

        // instance_of: A instance_of B → B is prereq of A
        let instance_triples = engine
            .knowledge_graph()
            .triples_for_predicate(self.expand_instance_of);
        for (subject, object) in instance_triples {
            if concept_set.contains(&subject) && concept_set.contains(&object) {
                edges.push((
                    object,
                    subject,
                    0.4 * self.config.weight_structural,
                    PrerequisiteSource::Structural,
                ));
            }
        }
    }

    // ── Step 2c: VSA asymmetric similarity ──────────────────────────────

    /// Strategy 3: Discover prerequisites via VSA asymmetric similarity.
    ///
    /// For each pair (A, B): if `similarity(bind(encode(A), prereq_role), encode(B)) > 0.55`
    /// then A is prerequisite of B.
    fn discover_vsa_prereqs(
        &self,
        engine: &Arc<Engine>,
        domain_concepts: &HashMap<SymbolId, String>,
        edges: &mut Vec<(SymbolId, SymbolId, f32, PrerequisiteSource)>,
    ) {
        let ops = engine.ops();

        // Pre-encode all concepts.
        let encoded: Vec<(SymbolId, HyperVec)> = domain_concepts
            .iter()
            .filter_map(|(&id, label)| encode_label(ops, label).ok().map(|v| (id, v)))
            .collect();

        // O(n²) comparison: acceptable for 50-200 concepts.
        for (i, (id_a, vec_a)) in encoded.iter().enumerate() {
            // bind(encode(A), prereq_role)
            let bound = match ops.bind(vec_a, &self.prereq_role) {
                Ok(b) => b,
                Err(_) => continue,
            };

            for (j, (id_b, vec_b)) in encoded.iter().enumerate() {
                if i == j {
                    continue;
                }
                let sim = match ops.similarity(&bound, vec_b) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                if sim > 0.55 {
                    edges.push((
                        *id_a,
                        *id_b,
                        sim * self.config.weight_vsa,
                        PrerequisiteSource::VsaSimilarity,
                    ));
                }
            }
        }
    }

    // ── Step 3: Merge edges ─────────────────────────────────────────────

    /// Merge raw edges: deduplicate (from, to) pairs, combine confidences and sources.
    fn merge_edges(
        &self,
        raw_edges: &[(SymbolId, SymbolId, f32, PrerequisiteSource)],
    ) -> Vec<PrerequisiteEdge> {
        let mut edge_map: HashMap<(SymbolId, SymbolId), (f32, Vec<PrerequisiteSource>)> =
            HashMap::new();

        for &(from, to, confidence, source) in raw_edges {
            if from == to {
                continue; // skip self-loops
            }
            let entry = edge_map.entry((from, to)).or_insert((0.0, Vec::new()));
            entry.0 += confidence;
            if !entry.1.contains(&source) {
                entry.1.push(source);
            }
        }

        let mut edges: Vec<PrerequisiteEdge> = edge_map
            .into_iter()
            .filter(|(_, (conf, _))| *conf >= self.config.min_edge_confidence)
            .map(|((from, to), (confidence, sources))| PrerequisiteEdge {
                from,
                to,
                confidence: confidence.min(1.0),
                sources,
            })
            .collect();

        // Limit prerequisites per concept.
        let mut per_concept: HashMap<SymbolId, Vec<usize>> = HashMap::new();
        for (i, edge) in edges.iter().enumerate() {
            per_concept.entry(edge.to).or_default().push(i);
        }
        let mut remove_set = HashSet::new();
        for (_concept, mut indices) in per_concept {
            if indices.len() > self.config.max_prereqs_per_concept {
                indices.sort_by(|a, b| {
                    edges[*b]
                        .confidence
                        .partial_cmp(&edges[*a].confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                for &idx in &indices[self.config.max_prereqs_per_concept..] {
                    remove_set.insert(idx);
                }
            }
        }
        if !remove_set.is_empty() {
            let mut kept = Vec::new();
            for (i, edge) in edges.into_iter().enumerate() {
                if !remove_set.contains(&i) {
                    kept.push(edge);
                }
            }
            edges = kept;
        }

        edges
    }

    // ── Step 4: Break cycles ────────────────────────────────────────────

    /// Break cycles in the prerequisite graph by removing lowest-confidence edges.
    ///
    /// Uses iterative DFS cycle detection. Returns the number of cycles broken.
    fn break_cycles(&self, edges: &mut Vec<PrerequisiteEdge>) -> usize {
        let mut cycles_broken = 0;

        loop {
            // Build adjacency from current edges.
            let mut adj: HashMap<SymbolId, Vec<(usize, SymbolId)>> = HashMap::new();
            for (i, edge) in edges.iter().enumerate() {
                adj.entry(edge.from).or_default().push((i, edge.to));
            }

            // DFS to find a cycle.
            let nodes: Vec<SymbolId> = adj.keys().copied().collect();
            let cycle_edge = find_cycle_edge(&nodes, &adj, edges);

            match cycle_edge {
                Some(edge_idx) => {
                    edges.remove(edge_idx);
                    cycles_broken += 1;
                }
                None => break,
            }
        }

        cycles_broken
    }

    // ── Step 5: Build known bundle ──────────────────────────────────────

    /// Build a VSA bundle representing "known" knowledge.
    ///
    /// Concepts with many triples are considered well-known and bundled together
    /// to form a reference vector for similarity comparison.
    fn build_known_bundle(
        &self,
        engine: &Arc<Engine>,
        domain_concepts: &HashMap<SymbolId, String>,
    ) -> Option<HyperVec> {
        let ops = engine.ops();
        let mut known_vecs: Vec<HyperVec> = Vec::new();

        for (&id, label) in domain_concepts {
            let triple_count = engine.triples_from(id).len();
            if triple_count >= self.config.known_min_triples
                && let Ok(vec) = encode_label(ops, label)
            {
                known_vecs.push(vec);
            }
        }

        if known_vecs.is_empty() {
            return None;
        }

        let refs: Vec<&HyperVec> = known_vecs.iter().collect();
        ops.bundle(&refs).ok()
    }

    // ── Step 6: Topological sort ────────────────────────────────────────

    /// Kahn's algorithm for topological sorting. Returns a map of concept → tier.
    ///
    /// Tier 0 = no prerequisites (foundational concepts).
    fn topological_sort(
        &self,
        edges: &[PrerequisiteEdge],
        concept_set: &HashSet<SymbolId>,
    ) -> HashMap<SymbolId, u32> {
        // Build in-degree map and adjacency.
        let mut in_degree: HashMap<SymbolId, usize> = HashMap::new();
        let mut adj: HashMap<SymbolId, Vec<SymbolId>> = HashMap::new();

        for &id in concept_set {
            in_degree.entry(id).or_insert(0);
        }
        for edge in edges {
            *in_degree.entry(edge.to).or_insert(0) += 1;
            in_degree.entry(edge.from).or_insert(0);
            adj.entry(edge.from).or_default().push(edge.to);
        }

        // Kahn's algorithm with tier tracking.
        let mut queue: VecDeque<SymbolId> = in_degree
            .iter()
            .filter(|(_, deg)| **deg == 0)
            .map(|(&id, _)| id)
            .collect();
        let mut tiers: HashMap<SymbolId, u32> = HashMap::new();

        for &id in &queue {
            tiers.insert(id, 0);
        }

        while let Some(node) = queue.pop_front() {
            let node_tier = tiers[&node];
            if let Some(neighbors) = adj.get(&node) {
                for &neighbor in neighbors {
                    let deg = in_degree.get_mut(&neighbor).unwrap();
                    *deg -= 1;
                    let neighbor_tier = tiers.entry(neighbor).or_insert(0);
                    *neighbor_tier = (*neighbor_tier).max(node_tier + 1);
                    if *deg == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        // Any nodes not in tiers (shouldn't happen after cycle-breaking) get max tier.
        let max_tier = tiers.values().copied().max().unwrap_or(0);
        for &id in concept_set {
            tiers.entry(id).or_insert(max_tier + 1);
        }

        tiers
    }

    // ── Step 7: Classify ZPD ────────────────────────────────────────────

    /// Classify each concept into a ZPD zone based on:
    /// - triple count + similarity to known bundle → Known
    /// - prerequisite coverage + mid-range similarity → Proximal
    /// - otherwise → Beyond
    fn classify_zpd(
        &self,
        engine: &Arc<Engine>,
        domain_concepts: &HashMap<SymbolId, String>,
        edges: &[PrerequisiteEdge],
        tiers: &HashMap<SymbolId, u32>,
        known_bundle: Option<&HyperVec>,
    ) -> HashMap<SymbolId, (ZpdZone, f32, f32)> {
        let ops = engine.ops();

        // Build prereq map: concept → list of prerequisite concept IDs.
        let mut prereqs_of: HashMap<SymbolId, Vec<SymbolId>> = HashMap::new();
        for edge in edges {
            prereqs_of.entry(edge.to).or_default().push(edge.from);
        }

        let mut result: HashMap<SymbolId, (ZpdZone, f32, f32)> = HashMap::new();

        // Process in tier order (bottom-up) so prereq zone info propagates.
        let mut sorted_concepts: Vec<(SymbolId, u32)> = domain_concepts
            .keys()
            .map(|&id| (id, *tiers.get(&id).unwrap_or(&0)))
            .collect();
        sorted_concepts.sort_by_key(|(_, tier)| *tier);

        for (concept_id, _tier) in &sorted_concepts {
            let label = &domain_concepts[concept_id];
            let triple_count = engine.triples_from(*concept_id).len();

            // Compute similarity to known bundle.
            let similarity = known_bundle
                .and_then(|kb| {
                    encode_label(ops, label)
                        .ok()
                        .and_then(|vec| ops.similarity(&vec, kb).ok())
                })
                .unwrap_or(0.0);

            // Compute prerequisite coverage: fraction of prereqs that are Known.
            let prereq_coverage = if let Some(prereqs) = prereqs_of.get(concept_id) {
                if prereqs.is_empty() {
                    1.0
                } else {
                    let known_count = prereqs
                        .iter()
                        .filter(|p| {
                            result
                                .get(p)
                                .is_some_and(|(zone, _, _)| *zone == ZpdZone::Known)
                        })
                        .count();
                    known_count as f32 / prereqs.len() as f32
                }
            } else {
                1.0 // no prerequisites → fully covered
            };

            // Classify.
            let zone = if triple_count >= self.config.known_min_triples
                && similarity >= self.config.known_similarity_threshold
            {
                ZpdZone::Known
            } else if prereq_coverage >= self.config.proximal_min_prereq_coverage
                && similarity >= self.config.proximal_similarity_low
                && similarity < self.config.proximal_similarity_high
            {
                ZpdZone::Proximal
            } else if prereq_coverage >= self.config.proximal_min_prereq_coverage
                && similarity >= self.config.proximal_similarity_high
            {
                // High coverage + high similarity but not enough triples → still Proximal
                ZpdZone::Proximal
            } else {
                ZpdZone::Beyond
            };

            result.insert(*concept_id, (zone, prereq_coverage, similarity));
        }

        result
    }

    // ── Step 8: Generate curriculum ─────────────────────────────────────

    /// Generate a curriculum ordering sorted by (tier ASC, zone ASC, similarity DESC).
    fn generate_curriculum(
        &self,
        engine: &Arc<Engine>,
        classifications: &HashMap<SymbolId, (ZpdZone, f32, f32)>,
        edges: &[PrerequisiteEdge],
        tiers: &HashMap<SymbolId, u32>,
    ) -> Vec<CurriculumEntry> {
        // Build prereq map.
        let mut prereqs_of: HashMap<SymbolId, Vec<SymbolId>> = HashMap::new();
        for edge in edges {
            prereqs_of.entry(edge.to).or_default().push(edge.from);
        }

        let mut curriculum: Vec<CurriculumEntry> = classifications
            .iter()
            .map(|(&concept, &(zone, prereq_coverage, similarity))| {
                let tier = *tiers.get(&concept).unwrap_or(&0);
                let prerequisites = prereqs_of.get(&concept).cloned().unwrap_or_default();
                CurriculumEntry {
                    concept,
                    label: engine.resolve_label(concept),
                    zone,
                    prereq_coverage,
                    similarity_to_known: similarity,
                    tier,
                    prerequisites,
                }
            })
            .collect();

        // Sort: tier ASC, zone ASC (Known < Proximal < Beyond), similarity DESC.
        curriculum.sort_by(|a, b| {
            a.tier
                .cmp(&b.tier)
                .then(a.zone.cmp(&b.zone))
                .then(
                    b.similarity_to_known
                        .partial_cmp(&a.similarity_to_known)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        });

        curriculum
    }

    // ── Step 9: Persist to KG ───────────────────────────────────────────

    /// Persist prerequisite edges, ZPD classifications, and provenance to the KG.
    fn persist_to_kg(
        &self,
        engine: &Arc<Engine>,
        edges: &[PrerequisiteEdge],
        classifications: &HashMap<SymbolId, (ZpdZone, f32, f32)>,
        tiers: &HashMap<SymbolId, u32>,
    ) -> PrerequisiteResult<Vec<ProvenanceId>> {
        let mut provenance_ids = Vec::new();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Persist prerequisite edges.
        for edge in edges {
            let triple = Triple {
                subject: edge.from,
                predicate: self.predicates.prerequisite_of,
                object: edge.to,
                confidence: edge.confidence,
                timestamp: now,
                provenance_id: None,
                compartment_id: None,
            };
            engine.add_triple(&triple)?;

            let from_label = engine.resolve_label(edge.from);
            let to_label = engine.resolve_label(edge.to);
            let sources_str = edge
                .sources
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("+");

            let mut record = ProvenanceRecord::new(
                edge.to,
                DerivationKind::PrerequisiteDiscovered {
                    from_label,
                    to_label,
                    edge_count: edges.len() as u32,
                    strategy_sources: sources_str,
                },
            )
            .with_confidence(edge.confidence)
            .with_sources(vec![edge.from, edge.to]);
            if let Ok(id) = engine.store_provenance(&mut record) {
                provenance_ids.push(id);
            }
        }

        // Persist ZPD classifications.
        for (&concept, &(zone, prereq_coverage, similarity)) in classifications {
            let tier = *tiers.get(&concept).unwrap_or(&0);

            // Store ZPD zone as a triple: concept prereq:zpd_zone "zone_label"
            let zone_entity = engine.resolve_or_create_entity(zone.as_label())?;
            let zone_triple = Triple {
                subject: concept,
                predicate: self.predicates.zpd_zone,
                object: zone_entity,
                confidence: 1.0,
                timestamp: now,
                provenance_id: None,
                compartment_id: None,
            };
            engine.add_triple(&zone_triple)?;

            // Store curriculum tier.
            let tier_label = format!("tier:{tier}");
            let tier_entity = engine.resolve_or_create_entity(&tier_label)?;
            let tier_triple = Triple {
                subject: concept,
                predicate: self.predicates.curriculum_tier,
                object: tier_entity,
                confidence: 1.0,
                timestamp: now,
                provenance_id: None,
                compartment_id: None,
            };
            engine.add_triple(&tier_triple)?;

            // Provenance.
            let concept_label = engine.resolve_label(concept);
            let mut record = ProvenanceRecord::new(
                concept,
                DerivationKind::ZpdClassification {
                    concept_label,
                    zone: zone.as_label().to_string(),
                    prereq_coverage,
                    similarity_to_known: similarity,
                    curriculum_tier: tier,
                },
            )
            .with_confidence(1.0);
            if let Ok(id) = engine.store_provenance(&mut record) {
                provenance_ids.push(id);
            }
        }

        Ok(provenance_ids)
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Find a cycle in the directed graph and return the index of the lowest-confidence
/// edge in that cycle. Returns `None` if no cycle exists.
fn find_cycle_edge(
    nodes: &[SymbolId],
    adj: &HashMap<SymbolId, Vec<(usize, SymbolId)>>,
    edges: &[PrerequisiteEdge],
) -> Option<usize> {
    let mut visited = HashSet::new();
    let mut in_stack = HashSet::new();
    let mut stack_edges: Vec<usize> = Vec::new();

    for &start in nodes {
        if visited.contains(&start) {
            continue;
        }
        if let Some(edge_idx) =
            dfs_find_cycle(start, adj, edges, &mut visited, &mut in_stack, &mut stack_edges)
        {
            return Some(edge_idx);
        }
    }
    None
}

/// DFS helper that returns the index of the lowest-confidence edge in a detected cycle.
fn dfs_find_cycle(
    node: SymbolId,
    adj: &HashMap<SymbolId, Vec<(usize, SymbolId)>>,
    edges: &[PrerequisiteEdge],
    visited: &mut HashSet<SymbolId>,
    in_stack: &mut HashSet<SymbolId>,
    stack_edges: &mut Vec<usize>,
) -> Option<usize> {
    visited.insert(node);
    in_stack.insert(node);

    if let Some(neighbors) = adj.get(&node) {
        for &(edge_idx, neighbor) in neighbors {
            if edge_idx >= edges.len() {
                continue;
            }
            if !visited.contains(&neighbor) {
                stack_edges.push(edge_idx);
                if let Some(result) =
                    dfs_find_cycle(neighbor, adj, edges, visited, in_stack, stack_edges)
                {
                    return Some(result);
                }
                stack_edges.pop();
            } else if in_stack.contains(&neighbor) {
                // Cycle found! Find the lowest-confidence edge.
                stack_edges.push(edge_idx);
                let min_edge = stack_edges
                    .iter()
                    .copied()
                    .filter(|&idx| idx < edges.len())
                    .min_by(|&a, &b| {
                        edges[a]
                            .confidence
                            .partial_cmp(&edges[b].confidence)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                stack_edges.pop();
                return min_edge;
            }
        }
    }

    in_stack.remove(&node);
    None
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prerequisite_config_default_values() {
        let config = PrerequisiteConfig::default();
        assert_eq!(config.known_min_triples, 5);
        assert!((config.known_similarity_threshold - 0.7).abs() < f32::EPSILON);
        assert!((config.proximal_min_prereq_coverage - 0.5).abs() < f32::EPSILON);
        assert!((config.proximal_similarity_low - 0.3).abs() < f32::EPSILON);
        assert!((config.proximal_similarity_high - 0.7).abs() < f32::EPSILON);
        assert!((config.min_edge_confidence - 0.3).abs() < f32::EPSILON);
        assert!((config.weight_conceptnet - 0.5).abs() < f32::EPSILON);
        assert!((config.weight_structural - 0.3).abs() < f32::EPSILON);
        assert!((config.weight_vsa - 0.2).abs() < f32::EPSILON);
        assert_eq!(config.max_prereqs_per_concept, 10);
    }

    #[test]
    fn zpd_zone_display_roundtrip() {
        for zone in [ZpdZone::Known, ZpdZone::Proximal, ZpdZone::Beyond] {
            let label = zone.as_label();
            let parsed = ZpdZone::from_label(label).unwrap();
            assert_eq!(parsed, zone);
            assert_eq!(zone.to_string(), label);
        }
    }

    #[test]
    fn zpd_zone_from_label_invalid() {
        assert_eq!(ZpdZone::from_label("invalid"), None);
        assert_eq!(ZpdZone::from_label(""), None);
    }

    #[test]
    fn prerequisite_source_display() {
        assert_eq!(PrerequisiteSource::ConceptNet.to_string(), "conceptnet");
        assert_eq!(PrerequisiteSource::Structural.to_string(), "structural");
        assert_eq!(PrerequisiteSource::VsaSimilarity.to_string(), "vsa");
    }

    #[test]
    fn error_no_concepts_message() {
        let err = PrerequisiteError::NoConcepts;
        assert!(err.to_string().contains("no concepts"));
    }

    #[test]
    fn error_cycle_detected_message() {
        let err = PrerequisiteError::CycleDetected { cycle_size: 3 };
        assert!(err.to_string().contains("cycle"));
        assert!(err.to_string().contains("3"));
    }

    #[test]
    fn error_empty_curriculum_message() {
        let err = PrerequisiteError::EmptyCurriculum { analyzed: 10 };
        assert!(err.to_string().contains("10"));
    }

    #[test]
    fn merge_edges_same_pair() {
        let config = PrerequisiteConfig::default();
        let analyzer_config = config;
        // Build a minimal analyzer-like context for merge_edges.
        let id_a = SymbolId::new(1).unwrap();
        let id_b = SymbolId::new(2).unwrap();

        let raw = vec![
            (id_a, id_b, 0.3, PrerequisiteSource::ConceptNet),
            (id_a, id_b, 0.2, PrerequisiteSource::Structural),
        ];

        // Manually create a merge like the analyzer does.
        let mut edge_map: HashMap<(SymbolId, SymbolId), (f32, Vec<PrerequisiteSource>)> =
            HashMap::new();
        for &(from, to, confidence, source) in &raw {
            let entry = edge_map.entry((from, to)).or_insert((0.0, Vec::new()));
            entry.0 += confidence;
            if !entry.1.contains(&source) {
                entry.1.push(source);
            }
        }

        let edges: Vec<PrerequisiteEdge> = edge_map
            .into_iter()
            .filter(|(_, (conf, _))| *conf >= analyzer_config.min_edge_confidence)
            .map(|((from, to), (confidence, sources))| PrerequisiteEdge {
                from,
                to,
                confidence: confidence.min(1.0),
                sources,
            })
            .collect();

        assert_eq!(edges.len(), 1);
        assert!((edges[0].confidence - 0.5).abs() < f32::EPSILON);
        assert_eq!(edges[0].sources.len(), 2);
    }

    #[test]
    fn merge_edges_different_pairs() {
        let id_a = SymbolId::new(1).unwrap();
        let id_b = SymbolId::new(2).unwrap();
        let id_c = SymbolId::new(3).unwrap();

        let raw = vec![
            (id_a, id_b, 0.5, PrerequisiteSource::ConceptNet),
            (id_b, id_c, 0.4, PrerequisiteSource::Structural),
        ];

        let mut edge_map: HashMap<(SymbolId, SymbolId), (f32, Vec<PrerequisiteSource>)> =
            HashMap::new();
        for &(from, to, confidence, source) in &raw {
            let entry = edge_map.entry((from, to)).or_insert((0.0, Vec::new()));
            entry.0 += confidence;
            if !entry.1.contains(&source) {
                entry.1.push(source);
            }
        }

        let edges: Vec<PrerequisiteEdge> = edge_map
            .into_iter()
            .filter(|(_, (conf, _))| *conf >= 0.3)
            .map(|((from, to), (confidence, sources))| PrerequisiteEdge {
                from,
                to,
                confidence: confidence.min(1.0),
                sources,
            })
            .collect();

        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn break_cycles_simple_triangle() {
        let id_a = SymbolId::new(1).unwrap();
        let id_b = SymbolId::new(2).unwrap();
        let id_c = SymbolId::new(3).unwrap();

        let mut edges = vec![
            PrerequisiteEdge {
                from: id_a,
                to: id_b,
                confidence: 0.8,
                sources: vec![PrerequisiteSource::ConceptNet],
            },
            PrerequisiteEdge {
                from: id_b,
                to: id_c,
                confidence: 0.6,
                sources: vec![PrerequisiteSource::Structural],
            },
            PrerequisiteEdge {
                from: id_c,
                to: id_a,
                confidence: 0.3, // lowest — should be removed
                sources: vec![PrerequisiteSource::VsaSimilarity],
            },
        ];

        let config = PrerequisiteConfig::default();
        // Simulate break_cycles logic.
        let mut cycles_broken = 0;
        loop {
            let mut adj: HashMap<SymbolId, Vec<(usize, SymbolId)>> = HashMap::new();
            for (i, edge) in edges.iter().enumerate() {
                adj.entry(edge.from).or_default().push((i, edge.to));
            }
            let nodes: Vec<SymbolId> = adj.keys().copied().collect();
            let cycle_edge = find_cycle_edge(&nodes, &adj, &edges);
            match cycle_edge {
                Some(edge_idx) => {
                    edges.remove(edge_idx);
                    cycles_broken += 1;
                }
                None => break,
            }
        }

        assert!(cycles_broken >= 1);
        assert_eq!(edges.len(), 2);
        // The remaining edges should form a DAG.
        let _ = config; // suppress warning
    }

    #[test]
    fn break_cycles_no_cycle() {
        let id_a = SymbolId::new(1).unwrap();
        let id_b = SymbolId::new(2).unwrap();
        let id_c = SymbolId::new(3).unwrap();

        let mut edges = vec![
            PrerequisiteEdge {
                from: id_a,
                to: id_b,
                confidence: 0.8,
                sources: vec![PrerequisiteSource::ConceptNet],
            },
            PrerequisiteEdge {
                from: id_a,
                to: id_c,
                confidence: 0.6,
                sources: vec![PrerequisiteSource::Structural],
            },
        ];

        let original_len = edges.len();
        loop {
            let mut adj: HashMap<SymbolId, Vec<(usize, SymbolId)>> = HashMap::new();
            for (i, edge) in edges.iter().enumerate() {
                adj.entry(edge.from).or_default().push((i, edge.to));
            }
            let nodes: Vec<SymbolId> = adj.keys().copied().collect();
            let cycle_edge = find_cycle_edge(&nodes, &adj, &edges);
            match cycle_edge {
                Some(edge_idx) => {
                    edges.remove(edge_idx);
                }
                None => break,
            }
        }

        assert_eq!(edges.len(), original_len);
    }

    #[test]
    fn topological_sort_linear_chain() {
        // A → B → C
        let id_a = SymbolId::new(1).unwrap();
        let id_b = SymbolId::new(2).unwrap();
        let id_c = SymbolId::new(3).unwrap();

        let edges = vec![
            PrerequisiteEdge {
                from: id_a,
                to: id_b,
                confidence: 0.8,
                sources: vec![],
            },
            PrerequisiteEdge {
                from: id_b,
                to: id_c,
                confidence: 0.7,
                sources: vec![],
            },
        ];

        let concept_set: HashSet<SymbolId> = [id_a, id_b, id_c].into_iter().collect();

        let config = PrerequisiteConfig::default();
        // Simulate topological_sort.
        let mut in_degree: HashMap<SymbolId, usize> = HashMap::new();
        let mut adj: HashMap<SymbolId, Vec<SymbolId>> = HashMap::new();
        for &id in &concept_set {
            in_degree.entry(id).or_insert(0);
        }
        for edge in &edges {
            *in_degree.entry(edge.to).or_insert(0) += 1;
            in_degree.entry(edge.from).or_insert(0);
            adj.entry(edge.from).or_default().push(edge.to);
        }

        let mut queue: VecDeque<SymbolId> = in_degree
            .iter()
            .filter(|(_, deg)| **deg == 0)
            .map(|(&id, _)| id)
            .collect();
        let mut tiers: HashMap<SymbolId, u32> = HashMap::new();
        for &id in &queue {
            tiers.insert(id, 0);
        }
        while let Some(node) = queue.pop_front() {
            let node_tier = tiers[&node];
            if let Some(neighbors) = adj.get(&node) {
                for &neighbor in neighbors {
                    let deg = in_degree.get_mut(&neighbor).unwrap();
                    *deg -= 1;
                    let neighbor_tier = tiers.entry(neighbor).or_insert(0);
                    *neighbor_tier = (*neighbor_tier).max(node_tier + 1);
                    if *deg == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        assert_eq!(tiers[&id_a], 0);
        assert_eq!(tiers[&id_b], 1);
        assert_eq!(tiers[&id_c], 2);
        let _ = config;
    }

    #[test]
    fn topological_sort_diamond() {
        //   A
        //  / \
        // B   C
        //  \ /
        //   D
        let id_a = SymbolId::new(1).unwrap();
        let id_b = SymbolId::new(2).unwrap();
        let id_c = SymbolId::new(3).unwrap();
        let id_d = SymbolId::new(4).unwrap();

        let edges = vec![
            PrerequisiteEdge {
                from: id_a,
                to: id_b,
                confidence: 0.8,
                sources: vec![],
            },
            PrerequisiteEdge {
                from: id_a,
                to: id_c,
                confidence: 0.7,
                sources: vec![],
            },
            PrerequisiteEdge {
                from: id_b,
                to: id_d,
                confidence: 0.6,
                sources: vec![],
            },
            PrerequisiteEdge {
                from: id_c,
                to: id_d,
                confidence: 0.5,
                sources: vec![],
            },
        ];

        let concept_set: HashSet<SymbolId> = [id_a, id_b, id_c, id_d].into_iter().collect();

        let mut in_degree: HashMap<SymbolId, usize> = HashMap::new();
        let mut adj: HashMap<SymbolId, Vec<SymbolId>> = HashMap::new();
        for &id in &concept_set {
            in_degree.entry(id).or_insert(0);
        }
        for edge in &edges {
            *in_degree.entry(edge.to).or_insert(0) += 1;
            in_degree.entry(edge.from).or_insert(0);
            adj.entry(edge.from).or_default().push(edge.to);
        }

        let mut queue: VecDeque<SymbolId> = in_degree
            .iter()
            .filter(|(_, deg)| **deg == 0)
            .map(|(&id, _)| id)
            .collect();
        let mut tiers: HashMap<SymbolId, u32> = HashMap::new();
        for &id in &queue {
            tiers.insert(id, 0);
        }
        while let Some(node) = queue.pop_front() {
            let node_tier = tiers[&node];
            if let Some(neighbors) = adj.get(&node) {
                for &neighbor in neighbors {
                    let deg = in_degree.get_mut(&neighbor).unwrap();
                    *deg -= 1;
                    let neighbor_tier = tiers.entry(neighbor).or_insert(0);
                    *neighbor_tier = (*neighbor_tier).max(node_tier + 1);
                    if *deg == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        assert_eq!(tiers[&id_a], 0);
        assert_eq!(tiers[&id_b], 1);
        assert_eq!(tiers[&id_c], 1);
        assert_eq!(tiers[&id_d], 2);
    }

    #[test]
    fn topological_sort_no_edges() {
        let id_a = SymbolId::new(1).unwrap();
        let id_b = SymbolId::new(2).unwrap();

        let edges: Vec<PrerequisiteEdge> = vec![];
        let concept_set: HashSet<SymbolId> = [id_a, id_b].into_iter().collect();

        let mut in_degree: HashMap<SymbolId, usize> = HashMap::new();
        for &id in &concept_set {
            in_degree.entry(id).or_insert(0);
        }

        let queue: VecDeque<SymbolId> = in_degree
            .iter()
            .filter(|(_, deg)| **deg == 0)
            .map(|(&id, _)| id)
            .collect();
        let mut tiers: HashMap<SymbolId, u32> = HashMap::new();
        for &id in &queue {
            tiers.insert(id, 0);
        }

        assert_eq!(tiers[&id_a], 0);
        assert_eq!(tiers[&id_b], 0);
        let _ = edges;
    }

    #[test]
    fn classify_zpd_high_triples_high_sim() {
        // A concept with >= known_min_triples and high similarity → Known
        let config = PrerequisiteConfig::default();
        let triple_count = config.known_min_triples + 1;
        let similarity = 0.8; // above known_similarity_threshold (0.7)

        let zone = if triple_count >= config.known_min_triples
            && similarity >= config.known_similarity_threshold
        {
            ZpdZone::Known
        } else {
            ZpdZone::Beyond
        };
        assert_eq!(zone, ZpdZone::Known);
    }

    #[test]
    fn classify_zpd_prereqs_met_mid_sim() {
        // Prereqs mostly known + mid similarity → Proximal
        let config = PrerequisiteConfig::default();
        let prereq_coverage = 0.8; // above 0.5
        let similarity = 0.5; // between 0.3 and 0.7
        let triple_count = 2; // below known_min_triples

        let zone =
            if triple_count >= config.known_min_triples
                && similarity >= config.known_similarity_threshold
            {
                ZpdZone::Known
            } else if prereq_coverage >= config.proximal_min_prereq_coverage
                && similarity >= config.proximal_similarity_low
                && similarity < config.proximal_similarity_high
            {
                ZpdZone::Proximal
            } else {
                ZpdZone::Beyond
            };
        assert_eq!(zone, ZpdZone::Proximal);
    }

    #[test]
    fn classify_zpd_prereqs_not_met() {
        // Low prereq coverage + low similarity → Beyond
        let config = PrerequisiteConfig::default();
        let prereq_coverage = 0.2; // below 0.5
        let similarity = 0.1; // below 0.3
        let triple_count = 1;

        let zone =
            if triple_count >= config.known_min_triples
                && similarity >= config.known_similarity_threshold
            {
                ZpdZone::Known
            } else if prereq_coverage >= config.proximal_min_prereq_coverage
                && similarity >= config.proximal_similarity_low
                && similarity < config.proximal_similarity_high
            {
                ZpdZone::Proximal
            } else {
                ZpdZone::Beyond
            };
        assert_eq!(zone, ZpdZone::Beyond);
    }

    #[test]
    fn curriculum_order_tiers_then_zones() {
        // Entries should sort by tier, then zone, then similarity desc.
        let mut entries = vec![
            CurriculumEntry {
                concept: SymbolId::new(1).unwrap(),
                label: "c1".into(),
                zone: ZpdZone::Beyond,
                prereq_coverage: 0.0,
                similarity_to_known: 0.1,
                tier: 0,
                prerequisites: vec![],
            },
            CurriculumEntry {
                concept: SymbolId::new(2).unwrap(),
                label: "c2".into(),
                zone: ZpdZone::Known,
                prereq_coverage: 1.0,
                similarity_to_known: 0.9,
                tier: 0,
                prerequisites: vec![],
            },
            CurriculumEntry {
                concept: SymbolId::new(3).unwrap(),
                label: "c3".into(),
                zone: ZpdZone::Proximal,
                prereq_coverage: 0.5,
                similarity_to_known: 0.5,
                tier: 1,
                prerequisites: vec![],
            },
        ];

        entries.sort_by(|a, b| {
            a.tier
                .cmp(&b.tier)
                .then(a.zone.cmp(&b.zone))
                .then(
                    b.similarity_to_known
                        .partial_cmp(&a.similarity_to_known)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        });

        // Tier 0 Known first, then tier 0 Beyond, then tier 1 Proximal
        assert_eq!(entries[0].label, "c2"); // tier 0, Known
        assert_eq!(entries[1].label, "c1"); // tier 0, Beyond
        assert_eq!(entries[2].label, "c3"); // tier 1, Proximal
    }

    #[test]
    fn structural_heuristic_subclass_implies_prereq() {
        // If A subclass_of B, then B should be prerequisite of A.
        // This is a logical test — B (general) comes before A (specific).
        let id_general = SymbolId::new(1).unwrap();
        let id_specific = SymbolId::new(2).unwrap();

        // Simulate: specific subclass_of general → general is prereq of specific
        let from = id_general; // prerequisite
        let to = id_specific; // depends on prerequisite

        let edge = PrerequisiteEdge {
            from,
            to,
            confidence: 0.6 * 0.3, // 0.6 * weight_structural
            sources: vec![PrerequisiteSource::Structural],
        };

        assert_eq!(edge.from, id_general);
        assert_eq!(edge.to, id_specific);
    }

    #[test]
    fn zpd_zone_ordering() {
        // Known < Proximal < Beyond (for sorting purposes)
        assert!(ZpdZone::Known < ZpdZone::Proximal);
        assert!(ZpdZone::Proximal < ZpdZone::Beyond);
    }
}
