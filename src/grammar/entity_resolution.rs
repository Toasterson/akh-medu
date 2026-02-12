//! Cross-language entity resolution.
//!
//! Detects when the same entity appears in different languages and unifies
//! them under a canonical label. Uses a 3-tier strategy:
//!
//! 1. **Exact alias match**: O(1) lookup in a runtime alias table
//! 2. **Static equivalence table**: ~200 entries for high-frequency terms
//! 3. **VSA similarity**: Fallback for novel entities (Latin-script near-matches)

use std::collections::HashMap;

use super::equivalences::lookup_equivalence;
use super::preprocess::ExtractedEntity;

/// Entity resolver with alias tracking and cross-lingual matching.
pub struct EntityResolver {
    /// Runtime alias table: surface form → canonical label.
    aliases: HashMap<String, String>,
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
        }
    }

    /// Register an alias: `surface` maps to `canonical`.
    pub fn add_alias(&mut self, surface: impl Into<String>, canonical: impl Into<String>) {
        let surface = surface.into();
        let canonical = canonical.into();
        if surface.to_lowercase() != canonical.to_lowercase() {
            self.aliases.insert(surface.to_lowercase(), canonical);
        }
    }

    /// Resolve a surface form to its canonical label.
    ///
    /// Tries in order:
    /// 1. Runtime alias table (exact match, case-insensitive)
    /// 2. Static equivalence table
    /// 3. Returns the original surface form if no match
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

        // Tier 2: Static equivalence table
        if let Some(canonical) = lookup_equivalence(surface) {
            return ResolutionResult {
                canonical: canonical.to_string(),
                resolved: true,
                aliases: vec![surface.to_string()],
            };
        }

        // Tier 3: No match — return as-is
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
}
