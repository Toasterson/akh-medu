//! Cross-language entity resolution with dynamic learning.
//!
//! Detects when the same entity appears in different languages and unifies
//! them under a canonical label. Uses a 4-tier strategy:
//!
//! 1. **Runtime aliases**: O(1) lookup in a hot in-memory alias table
//! 2. **Learned equivalences**: Persisted cross-lingual mappings discovered by
//!    structural (KG), distributional (VSA), or co-occurrence strategies
//! 3. **Static equivalence table**: ~120 hand-curated entries for high-frequency terms
//! 4. **Fallback**: Return the surface form unchanged

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::equivalences::lookup_equivalence;
use super::preprocess::{ExtractedEntity, PreProcessorOutput};
use crate::graph::index::KnowledgeGraph;
use crate::registry::SymbolRegistry;
use crate::store::{StoreResult, TieredStore};
use crate::vsa::encode::encode_label;
use crate::vsa::item_memory::ItemMemory;
use crate::vsa::ops::VsaOps;

/// Prefix used for persisting learned equivalences in the durable store.
const EQUIV_PREFIX: &[u8] = b"equiv:";

/// A dynamically learned equivalence mapping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedEquivalence {
    /// The canonical (English) label.
    pub canonical: String,
    /// The surface form that maps to `canonical`.
    pub surface: String,
    /// BCP 47 language code of the surface form, if known.
    pub source_language: String,
    /// Confidence score for this mapping (0.0–1.0).
    pub confidence: f32,
    /// How this equivalence was discovered.
    pub source: EquivalenceSource,
}

/// How an equivalence was discovered.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum EquivalenceSource {
    /// From the compiled-in static table.
    Static,
    /// Strategy 1: KG structural fingerprint matching.
    KgStructural,
    /// Strategy 2: VSA distributional similarity.
    VsaSimilarity,
    /// Strategy 3: Parallel chunk co-occurrence.
    CoOccurrence,
    /// User-added via CLI or API.
    Manual,
}

impl std::fmt::Display for EquivalenceSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Static => write!(f, "static"),
            Self::KgStructural => write!(f, "kg-structural"),
            Self::VsaSimilarity => write!(f, "vsa-similarity"),
            Self::CoOccurrence => write!(f, "co-occurrence"),
            Self::Manual => write!(f, "manual"),
        }
    }
}

/// Aggregate statistics about the equivalence table.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EquivalenceStats {
    pub runtime_aliases: usize,
    pub learned_total: usize,
    pub kg_structural: usize,
    pub vsa_similarity: usize,
    pub co_occurrence: usize,
    pub manual: usize,
}

/// Entity resolver with alias tracking, learned equivalences, and cross-lingual matching.
pub struct EntityResolver {
    /// Runtime alias table: surface_lowercase → canonical label.
    aliases: HashMap<String, String>,
    /// Learned equivalences: surface_lowercase → LearnedEquivalence.
    learned: HashMap<String, LearnedEquivalence>,
}

/// Result of entity resolution.
#[derive(Debug, Clone)]
pub struct ResolutionResult {
    /// The canonical label (English by convention).
    pub canonical: String,
    /// Whether this was resolved (true) or left as-is (false).
    pub resolved: bool,
    /// Aliases that map to this canonical label.
    pub aliases: Vec<String>,
}

impl EntityResolver {
    /// Create a new empty resolver.
    pub fn new() -> Self {
        Self {
            aliases: HashMap::new(),
            learned: HashMap::new(),
        }
    }

    /// Load persisted learned equivalences from the durable store.
    pub fn load_from_store(store: &TieredStore) -> Self {
        let mut resolver = Self::new();
        if let Ok(entries) = store.scan_prefix(EQUIV_PREFIX) {
            for (_key, value) in entries {
                if let Ok(equiv) = bincode::deserialize::<LearnedEquivalence>(&value) {
                    let surface_lower = equiv.surface.to_lowercase();
                    resolver.learned.insert(surface_lower, equiv);
                }
            }
        }
        resolver
    }

    /// Persist all learned equivalences to the durable store.
    pub fn persist_to_store(&self, store: &TieredStore) -> StoreResult<()> {
        for equiv in self.learned.values() {
            let key_str = format!("equiv:{}", equiv.surface.to_lowercase());
            let value = bincode::serialize(equiv).map_err(|e| {
                crate::error::StoreError::Serialization {
                    message: format!("failed to serialize learned equivalence: {e}"),
                }
            })?;
            store.put_meta(key_str.as_bytes(), &value)?;
        }
        Ok(())
    }

    /// Register a runtime alias: `surface` maps to `canonical`.
    pub fn add_alias(&mut self, surface: impl Into<String>, canonical: impl Into<String>) {
        let surface = surface.into();
        let canonical = canonical.into();
        if surface.to_lowercase() != canonical.to_lowercase() {
            self.aliases.insert(surface.to_lowercase(), canonical);
        }
    }

    /// Add a learned equivalence directly.
    pub fn add_learned(&mut self, equiv: LearnedEquivalence) {
        let key = equiv.surface.to_lowercase();
        // Only insert if higher confidence than existing, or if new
        if let Some(existing) = self.learned.get(&key) {
            if equiv.confidence <= existing.confidence {
                return;
            }
        }
        self.learned.insert(key, equiv);
    }

    /// Resolve a surface form to its canonical label.
    ///
    /// Tries in order:
    /// 1. Runtime alias table (exact match, case-insensitive)
    /// 2. Learned equivalences (persisted, loaded from store)
    /// 3. Static equivalence table (compiled-in)
    /// 4. Returns the original surface form if no match
    pub fn resolve(&self, surface: &str) -> ResolutionResult {
        let lower = surface.to_lowercase();

        // Tier 1: Runtime alias table
        if let Some(canonical) = self.aliases.get(&lower) {
            return ResolutionResult {
                canonical: canonical.clone(),
                resolved: true,
                aliases: vec![surface.to_string()],
            };
        }

        // Tier 2: Learned equivalences
        if let Some(equiv) = self.learned.get(&lower) {
            return ResolutionResult {
                canonical: equiv.canonical.clone(),
                resolved: true,
                aliases: vec![surface.to_string()],
            };
        }

        // Tier 3: Static equivalence table
        if let Some(canonical) = lookup_equivalence(surface) {
            return ResolutionResult {
                canonical: canonical.to_string(),
                resolved: true,
                aliases: vec![surface.to_string()],
            };
        }

        // Tier 4: No match — return as-is
        ResolutionResult {
            canonical: surface.to_string(),
            resolved: false,
            aliases: vec![],
        }
    }

    /// Resolve an extracted entity, updating its canonical name and aliases.
    pub fn resolve_entity(&self, entity: &mut ExtractedEntity) {
        let result = self.resolve(&entity.name);
        if result.resolved {
            // Keep original name as alias if canonical is different
            if result.canonical.to_lowercase() != entity.name.to_lowercase() {
                entity.aliases.push(entity.name.clone());
            }
            entity.canonical_name = result.canonical;
        }
    }

    /// Resolve all entities in a collection, also deduplicating by canonical name.
    pub fn resolve_entities(&self, entities: &mut Vec<ExtractedEntity>) {
        for entity in entities.iter_mut() {
            self.resolve_entity(entity);
        }

        // Merge entities with the same canonical name
        let mut seen: HashMap<String, usize> = HashMap::new();
        let mut merged = Vec::new();

        for entity in entities.drain(..) {
            let key = entity.canonical_name.to_lowercase();
            if let Some(&idx) = seen.get(&key) {
                // Merge aliases
                let existing = &mut merged[idx];
                let existing: &mut ExtractedEntity = existing;
                for alias in &entity.aliases {
                    if !existing.aliases.contains(alias) {
                        existing.aliases.push(alias.clone());
                    }
                }
                // Keep higher confidence
                if entity.confidence > existing.confidence {
                    existing.confidence = entity.confidence;
                }
            } else {
                seen.insert(key, merged.len());
                merged.push(entity);
            }
        }

        *entities = merged;
    }

    // -----------------------------------------------------------------------
    // Learning strategies
    // -----------------------------------------------------------------------

    /// Strategy 1: Learn equivalences from KG structural fingerprints.
    ///
    /// For each unresolved entity in the KG, collects its relational fingerprint
    /// (the set of `(predicate_label, resolved_object_label)` tuples). If another
    /// already-resolved entity shares at least one fingerprint tuple, proposes
    /// the unresolved entity as an alias for the resolved one.
    ///
    /// Returns the number of new equivalences discovered.
    pub fn learn_from_kg(
        &mut self,
        kg: &KnowledgeGraph,
        registry: &SymbolRegistry,
    ) -> usize {
        let all_nodes = kg.nodes();
        let mut discovered = 0usize;

        // Build fingerprints for all nodes
        type Fingerprint = Vec<(String, String)>;
        let mut node_fingerprints: HashMap<String, Fingerprint> = HashMap::new();

        for &node_id in &all_nodes {
            let label = match registry.get(node_id) {
                Some(meta) => meta.label,
                None => continue,
            };

            let triples = kg.triples_from(node_id);
            let mut fingerprint = Vec::new();

            for triple in &triples {
                let pred_label = registry.get(triple.predicate)
                    .map(|m| m.label)
                    .unwrap_or_default();
                let obj_label = registry.get(triple.object)
                    .map(|m| m.label)
                    .unwrap_or_default();

                // Resolve the object to its canonical form
                let obj_resolved = self.resolve(&obj_label);
                fingerprint.push((pred_label, obj_resolved.canonical));
            }

            if !fingerprint.is_empty() {
                fingerprint.sort();
                node_fingerprints.insert(label, fingerprint);
            }
        }

        // Separate nodes into resolved and unresolved
        let mut resolved_fps: Vec<(String, Fingerprint)> = Vec::new();
        let mut unresolved_fps: Vec<(String, Fingerprint)> = Vec::new();

        for (label, fp) in &node_fingerprints {
            let res = self.resolve(label);
            if res.resolved {
                resolved_fps.push((res.canonical.clone(), fp.clone()));
            } else {
                unresolved_fps.push((label.clone(), fp.clone()));
            }
        }

        // Compare unresolved against resolved
        for (unresolved_label, unresolved_fp) in &unresolved_fps {
            for (canonical, resolved_fp) in &resolved_fps {
                if unresolved_label.to_lowercase() == canonical.to_lowercase() {
                    continue;
                }

                // Count shared fingerprint tuples
                let shared = unresolved_fp.iter()
                    .filter(|t| resolved_fp.contains(t))
                    .count();

                if shared >= 1 {
                    let overlap = shared as f32
                        / unresolved_fp.len().max(resolved_fp.len()) as f32;
                    let confidence = overlap.min(0.95);

                    self.add_learned(LearnedEquivalence {
                        canonical: canonical.clone(),
                        surface: unresolved_label.clone(),
                        source_language: String::new(),
                        confidence,
                        source: EquivalenceSource::KgStructural,
                    });
                    discovered += 1;
                }
            }
        }

        discovered
    }

    /// Strategy 2: Learn equivalences from VSA label similarity.
    ///
    /// For each unresolved entity in the KG, encodes its label as a hypervector
    /// and searches item memory for nearest neighbors. If a neighbor's label
    /// resolves to a known canonical and the similarity exceeds the threshold,
    /// proposes the equivalence.
    ///
    /// Returns the number of new equivalences discovered.
    pub fn learn_from_vsa(
        &mut self,
        ops: &VsaOps,
        item_memory: &ItemMemory,
        registry: &SymbolRegistry,
        similarity_threshold: f32,
    ) -> usize {
        let all_symbols = registry.all();
        let mut discovered = 0usize;

        // Collect unresolved labels
        let unresolved: Vec<_> = all_symbols.iter()
            .filter(|meta| !self.resolve(&meta.label).resolved)
            .collect();

        for meta in &unresolved {
            let encoded = match encode_label(ops, &meta.label) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let neighbors = match item_memory.search(&encoded, 5) {
                Ok(n) => n,
                Err(_) => continue,
            };

            for neighbor in &neighbors {
                if neighbor.symbol_id == meta.id {
                    continue;
                }
                if neighbor.similarity < similarity_threshold {
                    continue;
                }

                let neighbor_label = match registry.get(neighbor.symbol_id) {
                    Some(m) => m.label,
                    None => continue,
                };

                let neighbor_resolved = self.resolve(&neighbor_label);
                if neighbor_resolved.resolved {
                    self.add_learned(LearnedEquivalence {
                        canonical: neighbor_resolved.canonical,
                        surface: meta.label.clone(),
                        source_language: String::new(),
                        confidence: neighbor.similarity,
                        source: EquivalenceSource::VsaSimilarity,
                    });
                    discovered += 1;
                    break; // Take the best match only
                }
            }
        }

        discovered
    }

    /// Strategy 3: Learn from parallel chunk co-occurrence.
    ///
    /// Groups preprocessor outputs by shared `chunk_id` prefix (e.g., `"doc1_en"`
    /// and `"doc1_ru"` share prefix `"doc1"`). For chunks in different languages
    /// within the same group, aligns entities by extraction order and proposes
    /// positional equivalences.
    ///
    /// Returns the number of new equivalences discovered.
    pub fn learn_from_parallel_chunks(
        &mut self,
        outputs: &[PreProcessorOutput],
    ) -> usize {
        if outputs.len() < 2 {
            return 0;
        }

        let mut discovered = 0usize;

        // Group outputs by chunk_id prefix (everything before the last '_')
        let mut groups: HashMap<String, Vec<&PreProcessorOutput>> = HashMap::new();

        for output in outputs {
            let chunk_id = match &output.chunk_id {
                Some(id) => id.clone(),
                None => continue,
            };

            // Extract prefix: everything before the last '_'
            let prefix = match chunk_id.rfind('_') {
                Some(pos) => &chunk_id[..pos],
                None => continue,
            };

            groups.entry(prefix.to_string()).or_default().push(output);
        }

        // For each group with multiple languages, align entities by position
        for (_prefix, group) in &groups {
            if group.len() < 2 {
                continue;
            }

            // Check for different languages in the group
            let languages: Vec<&str> = group.iter()
                .map(|o| o.source_language.as_str())
                .collect();

            let has_multiple_langs = {
                let mut unique = languages.clone();
                unique.sort();
                unique.dedup();
                unique.len() > 1
            };

            if !has_multiple_langs {
                continue;
            }

            // Use the first output as reference, align others
            let reference = group[0];
            for other in &group[1..] {
                if other.source_language == reference.source_language {
                    continue;
                }

                let pair_count = reference.entities.len().min(other.entities.len());
                for i in 0..pair_count {
                    let ref_entity = &reference.entities[i];
                    let other_entity = &other.entities[i];

                    // Skip if same canonical already
                    if ref_entity.canonical_name.to_lowercase()
                        == other_entity.canonical_name.to_lowercase()
                    {
                        continue;
                    }

                    // Skip if both are unresolved (no canonical anchor)
                    let ref_resolved = self.resolve(&ref_entity.canonical_name);
                    let other_resolved = self.resolve(&other_entity.canonical_name);

                    if ref_resolved.resolved && !other_resolved.resolved {
                        let confidence = 0.7 * (pair_count as f32 / group.len() as f32).min(1.0);
                        self.add_learned(LearnedEquivalence {
                            canonical: ref_resolved.canonical,
                            surface: other_entity.canonical_name.clone(),
                            source_language: other.source_language.clone(),
                            confidence,
                            source: EquivalenceSource::CoOccurrence,
                        });
                        discovered += 1;
                    } else if other_resolved.resolved && !ref_resolved.resolved {
                        let confidence = 0.7 * (pair_count as f32 / group.len() as f32).min(1.0);
                        self.add_learned(LearnedEquivalence {
                            canonical: other_resolved.canonical,
                            surface: ref_entity.canonical_name.clone(),
                            source_language: reference.source_language.clone(),
                            confidence,
                            source: EquivalenceSource::CoOccurrence,
                        });
                        discovered += 1;
                    }
                }
            }
        }

        discovered
    }

    /// Run all three learning strategies. Returns total new equivalences discovered.
    pub fn learn_all(
        &mut self,
        kg: &KnowledgeGraph,
        registry: &SymbolRegistry,
        ops: &VsaOps,
        item_memory: &ItemMemory,
        outputs: &[PreProcessorOutput],
    ) -> usize {
        let mut total = 0;
        total += self.learn_from_kg(kg, registry);
        total += self.learn_from_vsa(ops, item_memory, registry, 0.65);
        total += self.learn_from_parallel_chunks(outputs);
        total
    }

    // -----------------------------------------------------------------------
    // Export / Import
    // -----------------------------------------------------------------------

    /// Export all learned equivalences as a serializable list.
    pub fn export_learned(&self) -> Vec<LearnedEquivalence> {
        self.learned.values().cloned().collect()
    }

    /// Import equivalences from an external list (manual curation workflow).
    pub fn import_equivalences(&mut self, equivs: &[LearnedEquivalence]) {
        for equiv in equivs {
            self.add_learned(equiv.clone());
        }
    }

    /// Statistics: how many equivalences by source.
    pub fn stats(&self) -> EquivalenceStats {
        let mut stats = EquivalenceStats {
            runtime_aliases: self.aliases.len(),
            learned_total: self.learned.len(),
            ..Default::default()
        };
        for equiv in self.learned.values() {
            match equiv.source {
                EquivalenceSource::Static => {}
                EquivalenceSource::KgStructural => stats.kg_structural += 1,
                EquivalenceSource::VsaSimilarity => stats.vsa_similarity += 1,
                EquivalenceSource::CoOccurrence => stats.co_occurrence += 1,
                EquivalenceSource::Manual => stats.manual += 1,
            }
        }
        stats
    }

    /// Number of learned equivalences.
    pub fn learned_count(&self) -> usize {
        self.learned.len()
    }
}

impl Default for EntityResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_via_equivalence_table() {
        let resolver = EntityResolver::new();
        let result = resolver.resolve("Москва");
        assert!(result.resolved);
        assert_eq!(result.canonical, "Moscow");
    }

    #[test]
    fn resolve_french_via_equivalence() {
        let resolver = EntityResolver::new();
        let result = resolver.resolve("Moscou");
        assert!(result.resolved);
        assert_eq!(result.canonical, "Moscow");
    }

    #[test]
    fn resolve_via_runtime_alias() {
        let mut resolver = EntityResolver::new();
        resolver.add_alias("собака", "Dog");

        let result = resolver.resolve("собака");
        assert!(result.resolved);
        assert_eq!(result.canonical, "Dog");
    }

    #[test]
    fn resolve_unknown_returns_original() {
        let resolver = EntityResolver::new();
        let result = resolver.resolve("Foobarqux");
        assert!(!result.resolved);
        assert_eq!(result.canonical, "Foobarqux");
    }

    #[test]
    fn resolve_entity_updates_fields() {
        let resolver = EntityResolver::new();
        let mut entity = ExtractedEntity {
            name: "Москва".to_string(),
            entity_type: "PLACE".to_string(),
            canonical_name: "Москва".to_string(),
            confidence: 0.90,
            aliases: vec![],
            source_language: "ru".to_string(),
        };

        resolver.resolve_entity(&mut entity);
        assert_eq!(entity.canonical_name, "Moscow");
        assert!(entity.aliases.contains(&"Москва".to_string()));
    }

    #[test]
    fn resolve_entities_merges_duplicates() {
        let resolver = EntityResolver::new();
        let mut entities = vec![
            ExtractedEntity {
                name: "Moscow".to_string(),
                entity_type: "PLACE".to_string(),
                canonical_name: "Moscow".to_string(),
                confidence: 0.85,
                aliases: vec![],
                source_language: "en".to_string(),
            },
            ExtractedEntity {
                name: "Москва".to_string(),
                entity_type: "PLACE".to_string(),
                canonical_name: "Москва".to_string(),
                confidence: 0.90,
                aliases: vec![],
                source_language: "ru".to_string(),
            },
        ];

        resolver.resolve_entities(&mut entities);
        assert_eq!(entities.len(), 1, "should merge Moscow and Москва");
        assert_eq!(entities[0].canonical_name, "Moscow");
        assert!(entities[0].aliases.contains(&"Москва".to_string()));
        assert_eq!(entities[0].confidence, 0.90, "should keep higher confidence");
    }

    #[test]
    fn three_languages_same_entity() {
        let resolver = EntityResolver::new();

        let moscow_en = resolver.resolve("Moscow");
        let moscow_ru = resolver.resolve("Москва");
        let moscow_fr = resolver.resolve("Moscou");

        assert_eq!(moscow_en.canonical, "Moscow");
        assert_eq!(moscow_ru.canonical, "Moscow");
        assert_eq!(moscow_fr.canonical, "Moscow");
    }

    #[test]
    fn resolution_order_learned_before_static() {
        let mut resolver = EntityResolver::new();

        // "Москва" normally resolves to "Moscow" via static table.
        // Override with a learned equivalence.
        resolver.add_learned(LearnedEquivalence {
            canonical: "Moskau".to_string(),
            surface: "Москва".to_string(),
            source_language: "de".to_string(),
            confidence: 0.99,
            source: EquivalenceSource::Manual,
        });

        let result = resolver.resolve("Москва");
        assert!(result.resolved);
        assert_eq!(result.canonical, "Moskau", "learned should override static");
    }

    #[test]
    fn export_import_roundtrip() {
        let mut resolver = EntityResolver::new();
        resolver.add_learned(LearnedEquivalence {
            canonical: "Dog".to_string(),
            surface: "собака".to_string(),
            source_language: "ru".to_string(),
            confidence: 0.85,
            source: EquivalenceSource::KgStructural,
        });
        resolver.add_learned(LearnedEquivalence {
            canonical: "Cat".to_string(),
            surface: "кошка".to_string(),
            source_language: "ru".to_string(),
            confidence: 0.90,
            source: EquivalenceSource::VsaSimilarity,
        });

        let exported = resolver.export_learned();
        assert_eq!(exported.len(), 2);

        // Import into fresh resolver
        let mut resolver2 = EntityResolver::new();
        resolver2.import_equivalences(&exported);
        assert_eq!(resolver2.learned_count(), 2);

        let result = resolver2.resolve("собака");
        assert!(result.resolved);
        assert_eq!(result.canonical, "Dog");
    }

    #[test]
    fn stats_counts_by_source() {
        let mut resolver = EntityResolver::new();
        resolver.add_alias("собака", "Dog");
        resolver.add_alias("кошка", "Cat");

        resolver.add_learned(LearnedEquivalence {
            canonical: "Dog".to_string(),
            surface: "hund".to_string(),
            source_language: "de".to_string(),
            confidence: 0.8,
            source: EquivalenceSource::KgStructural,
        });
        resolver.add_learned(LearnedEquivalence {
            canonical: "Cat".to_string(),
            surface: "gato".to_string(),
            source_language: "es".to_string(),
            confidence: 0.7,
            source: EquivalenceSource::VsaSimilarity,
        });
        resolver.add_learned(LearnedEquivalence {
            canonical: "Bird".to_string(),
            surface: "oiseau".to_string(),
            source_language: "fr".to_string(),
            confidence: 0.6,
            source: EquivalenceSource::CoOccurrence,
        });

        let stats = resolver.stats();
        assert_eq!(stats.runtime_aliases, 2);
        assert_eq!(stats.learned_total, 3);
        assert_eq!(stats.kg_structural, 1);
        assert_eq!(stats.vsa_similarity, 1);
        assert_eq!(stats.co_occurrence, 1);
        assert_eq!(stats.manual, 0);
    }

    #[test]
    fn learn_from_parallel_chunks_position_correlation() {
        let mut resolver = EntityResolver::new();
        // Use a learned equivalence to make one entity "resolved".
        // "архетип_en" → "Archetype" so the EN side is resolved, RU side is not.
        resolver.add_learned(LearnedEquivalence {
            canonical: "Archetype".to_string(),
            surface: "archetype-concept".to_string(),
            source_language: "en".to_string(),
            confidence: 0.95,
            source: EquivalenceSource::Manual,
        });

        let outputs = vec![
            PreProcessorOutput {
                chunk_id: Some("doc1_en".to_string()),
                source_language: "en".to_string(),
                detected_language_confidence: 0.9,
                entities: vec![
                    ExtractedEntity {
                        name: "archetype-concept".to_string(),
                        entity_type: "CONCEPT".to_string(),
                        canonical_name: "archetype-concept".to_string(),
                        confidence: 0.9,
                        aliases: vec![],
                        source_language: "en".to_string(),
                    },
                ],
                claims: vec![],
                abs_trees: vec![],
            },
            PreProcessorOutput {
                chunk_id: Some("doc1_ru".to_string()),
                source_language: "ru".to_string(),
                detected_language_confidence: 0.9,
                entities: vec![
                    ExtractedEntity {
                        name: "архетип".to_string(),
                        entity_type: "CONCEPT".to_string(),
                        canonical_name: "архетип".to_string(),
                        confidence: 0.9,
                        aliases: vec![],
                        source_language: "ru".to_string(),
                    },
                ],
                claims: vec![],
                abs_trees: vec![],
            },
        ];

        let count = resolver.learn_from_parallel_chunks(&outputs);
        // "archetype-concept" resolves (via learned equiv), "архетип" does not
        assert!(count >= 1, "should discover at least one co-occurrence equivalence, got {count}");

        let result = resolver.resolve("архетип");
        assert!(result.resolved);
        assert_eq!(result.canonical, "Archetype");
    }

    #[test]
    fn persist_and_load_roundtrip() {
        // Test with a real durable store (memory-only has no durable backend).
        let dir = tempfile::TempDir::new().unwrap();
        let store = TieredStore::with_persistence(dir.path(), "test_equiv").unwrap();

        let mut resolver = EntityResolver::new();
        resolver.add_learned(LearnedEquivalence {
            canonical: "Dog".to_string(),
            surface: "собака".to_string(),
            source_language: "ru".to_string(),
            confidence: 0.85,
            source: EquivalenceSource::KgStructural,
        });
        resolver.add_learned(LearnedEquivalence {
            canonical: "Cat".to_string(),
            surface: "кошка".to_string(),
            source_language: "ru".to_string(),
            confidence: 0.90,
            source: EquivalenceSource::Manual,
        });

        resolver.persist_to_store(&store).unwrap();

        // Load into a fresh resolver
        let resolver2 = EntityResolver::load_from_store(&store);
        assert_eq!(resolver2.learned_count(), 2);

        let result = resolver2.resolve("собака");
        assert!(result.resolved);
        assert_eq!(result.canonical, "Dog");

        let result = resolver2.resolve("кошка");
        assert!(result.resolved);
        assert_eq!(result.canonical, "Cat");
    }

    #[test]
    fn add_learned_keeps_higher_confidence() {
        let mut resolver = EntityResolver::new();

        resolver.add_learned(LearnedEquivalence {
            canonical: "Dog".to_string(),
            surface: "hund".to_string(),
            source_language: "de".to_string(),
            confidence: 0.5,
            source: EquivalenceSource::VsaSimilarity,
        });

        // Lower confidence should not replace
        resolver.add_learned(LearnedEquivalence {
            canonical: "Canine".to_string(),
            surface: "hund".to_string(),
            source_language: "de".to_string(),
            confidence: 0.3,
            source: EquivalenceSource::KgStructural,
        });

        let result = resolver.resolve("hund");
        assert_eq!(result.canonical, "Dog", "should keep higher-confidence mapping");

        // Higher confidence should replace
        resolver.add_learned(LearnedEquivalence {
            canonical: "Canine".to_string(),
            surface: "hund".to_string(),
            source_language: "de".to_string(),
            confidence: 0.9,
            source: EquivalenceSource::Manual,
        });

        let result = resolver.resolve("hund");
        assert_eq!(result.canonical, "Canine", "should update to higher-confidence mapping");
    }
}
