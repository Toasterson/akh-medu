//! Community recipe sharing — Phase 14i.
//!
//! Purpose recipes are TOML-based descriptions of a bootstrap configuration
//! that can be shared, imported, and used to cold-start new instances without
//! repeating the full discovery/ingestion process.
//!
//! A recipe captures:
//! - **Purpose**: domain, competence level, seed concepts
//! - **Identity** (optional): cultural reference, archetype, traits
//! - **Seeds**: required and optional seed packs
//! - **Prerequisites**: ordered dependency graph between topics
//! - **Resources**: URLs and skillpacks for ingestion
//! - **Validation**: expected knowledge coverage thresholds
//!
//! Recipes are structured as TOML for human readability and version control.
//! They share **structure** (syllabus), not copyrighted content.

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::purpose::DreyfusLevel;

// ═══════════════════════════════════════════════════════════════════════
// Error
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Error, Diagnostic)]
pub enum RecipeError {
    #[error("failed to parse recipe: {message}")]
    #[diagnostic(
        code(akh::bootstrap::recipe::parse),
        help("Check the TOML syntax of the recipe file.")
    )]
    Parse { message: String },

    #[error("recipe validation failed: {message}")]
    #[diagnostic(
        code(akh::bootstrap::recipe::validation),
        help("Ensure the recipe meets minimum requirements (seeds, domain).")
    )]
    Validation { message: String },

    #[error("dependency cycle detected: {cycle}")]
    #[diagnostic(
        code(akh::bootstrap::recipe::cycle),
        help("Remove the circular dependency in the prerequisite graph.")
    )]
    DependencyCycle { cycle: String },

    #[error("failed to serialize recipe: {message}")]
    #[diagnostic(
        code(akh::bootstrap::recipe::serialize),
        help("Internal error during recipe serialization.")
    )]
    Serialize { message: String },
}

pub type RecipeResult<T> = std::result::Result<T, RecipeError>;

// ═══════════════════════════════════════════════════════════════════════
// PurposeRecipe
// ═══════════════════════════════════════════════════════════════════════

/// A shareable purpose recipe: describes how to bootstrap a domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurposeRecipe {
    /// Recipe metadata.
    pub purpose: RecipePurpose,
    /// Seed packs to apply.
    #[serde(default)]
    pub seeds: RecipeSeeds,
    /// Prerequisite ordering between topics.
    #[serde(default)]
    pub prerequisites: std::collections::HashMap<String, PrerequisiteEntry>,
    /// Resources for ingestion.
    #[serde(default)]
    pub resources: RecipeResources,
    /// Validation thresholds.
    #[serde(default)]
    pub validation: RecipeValidation,
    /// Optional identity configuration.
    pub identity: Option<RecipeIdentity>,
}

/// Recipe purpose metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecipePurpose {
    /// Unique recipe identifier (e.g., "gcc-compiler-expert").
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Semantic version.
    #[serde(default = "default_version")]
    pub version: String,
    /// Primary domain.
    pub domain: String,
    /// Target competence level.
    #[serde(default)]
    pub target_dreyfus: DreyfusLevel,
    /// Recipe author (e.g., "akh://operator@instance").
    #[serde(default)]
    pub author: String,
    /// Brief description.
    #[serde(default)]
    pub description: String,
}

fn default_version() -> String {
    "1.0.0".into()
}

/// Seed pack requirements.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecipeSeeds {
    /// Required seed packs (must be available).
    #[serde(default)]
    pub required: Vec<String>,
    /// Optional seed packs (apply if available).
    #[serde(default)]
    pub optional: Vec<String>,
}

/// A prerequisite ordering entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrerequisiteEntry {
    /// Topics that must come before this one.
    #[serde(default)]
    pub before: Vec<String>,
}

/// Resources for ingestion.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecipeResources {
    /// URLs to ingest.
    #[serde(default)]
    pub urls: Vec<String>,
    /// Skillpack names to install.
    #[serde(default)]
    pub skillpacks: Vec<String>,
}

/// Validation thresholds for recipe completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecipeValidation {
    /// Minimum triples expected after bootstrap.
    pub min_triples: usize,
    /// Minimum coverage of seed concepts (0.0–1.0).
    pub min_coverage: f32,
    /// Required entity types that must exist.
    #[serde(default)]
    pub required_types: Vec<String>,
}

impl Default for RecipeValidation {
    fn default() -> Self {
        Self {
            min_triples: 100,
            min_coverage: 0.5,
            required_types: Vec::new(),
        }
    }
}

/// Optional identity section for the recipe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecipeIdentity {
    /// Cultural/historical/fictional reference name.
    pub reference: String,
    /// Optional Wikidata QID.
    #[serde(default)]
    pub reference_qid: String,
    /// Jungian archetype.
    #[serde(default)]
    pub archetype: String,
    /// Cultural origin.
    #[serde(default)]
    pub culture: String,
    /// Suggested persona names.
    #[serde(default)]
    pub suggested_names: Vec<String>,
    /// Personality traits.
    #[serde(default)]
    pub traits: Vec<String>,
}

// ═══════════════════════════════════════════════════════════════════════
// Recipe Operations
// ═══════════════════════════════════════════════════════════════════════

impl PurposeRecipe {
    /// Parse a recipe from a TOML string.
    pub fn from_toml(toml_str: &str) -> RecipeResult<Self> {
        toml::from_str(toml_str).map_err(|e| RecipeError::Parse {
            message: e.to_string(),
        })
    }

    /// Serialize to a TOML string.
    pub fn to_toml(&self) -> RecipeResult<String> {
        toml::to_string_pretty(self).map_err(|e| RecipeError::Serialize {
            message: e.to_string(),
        })
    }

    /// Validate the recipe for completeness.
    pub fn validate(&self) -> RecipeResult<()> {
        if self.purpose.id.is_empty() {
            return Err(RecipeError::Validation {
                message: "recipe id is required".into(),
            });
        }
        if self.purpose.domain.is_empty() {
            return Err(RecipeError::Validation {
                message: "domain is required".into(),
            });
        }
        if self.seeds.required.is_empty() && self.resources.urls.is_empty() {
            return Err(RecipeError::Validation {
                message: "at least one required seed or resource URL is needed".into(),
            });
        }
        // Check for dependency cycles in prerequisites.
        self.check_cycles()?;
        Ok(())
    }

    /// Check for cycles in the prerequisite graph.
    fn check_cycles(&self) -> RecipeResult<()> {
        // Simple DFS cycle detection.
        let mut visited = std::collections::HashSet::new();
        let mut stack = std::collections::HashSet::new();

        for topic in self.prerequisites.keys() {
            if !visited.contains(topic) {
                self.dfs_cycle(topic, &mut visited, &mut stack)?;
            }
        }
        Ok(())
    }

    fn dfs_cycle(
        &self,
        node: &str,
        visited: &mut std::collections::HashSet<String>,
        stack: &mut std::collections::HashSet<String>,
    ) -> RecipeResult<()> {
        visited.insert(node.to_string());
        stack.insert(node.to_string());

        if let Some(entry) = self.prerequisites.get(node) {
            for dep in &entry.before {
                if stack.contains(dep.as_str()) {
                    return Err(RecipeError::DependencyCycle {
                        cycle: format!("{node} → {dep}"),
                    });
                }
                if !visited.contains(dep.as_str()) {
                    self.dfs_cycle(dep, visited, stack)?;
                }
            }
        }

        stack.remove(node);
        Ok(())
    }

    /// Get topics in dependency-respecting order (topological sort).
    pub fn topic_order(&self) -> Vec<String> {
        let mut in_degree: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let mut graph: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();

        // "topic" = { before = ["a", "b"] } means topic must come before a and b.
        // So topic → a, topic → b. In-degree of a and b increases.
        for (topic, entry) in &self.prerequisites {
            in_degree.entry(topic.clone()).or_insert(0);
            for successor in &entry.before {
                in_degree.entry(successor.clone()).or_insert(0);
                graph.entry(topic.clone()).or_default().push(successor.clone());
                *in_degree.entry(successor.clone()).or_insert(0) += 1;
            }
        }

        // Kahn's algorithm with deterministic ordering.
        let mut queue: std::collections::BTreeSet<String> = in_degree
            .iter()
            .filter(|(_, deg)| **deg == 0)
            .map(|(k, _)| k.clone())
            .collect();

        let mut result = Vec::new();
        while let Some(node) = queue.iter().next().cloned() {
            queue.remove(&node);
            result.push(node.clone());
            if let Some(successors) = graph.get(&node) {
                for succ in successors {
                    if let Some(deg) = in_degree.get_mut(succ) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.insert(succ.clone());
                        }
                    }
                }
            }
        }

        result
    }

    /// Generate a recipe from a completed bootstrap session.
    pub fn generate(
        id: impl Into<String>,
        name: impl Into<String>,
        domain: impl Into<String>,
        target_dreyfus: DreyfusLevel,
        seed_concepts: &[String],
        prerequisite_pairs: &[(String, String)],
    ) -> Self {
        let mut prerequisites = std::collections::HashMap::new();
        for (topic, dep) in prerequisite_pairs {
            prerequisites
                .entry(topic.clone())
                .or_insert_with(|| PrerequisiteEntry {
                    before: Vec::new(),
                })
                .before
                .push(dep.clone());
        }

        PurposeRecipe {
            purpose: RecipePurpose {
                id: id.into(),
                name: name.into(),
                version: "1.0.0".into(),
                domain: domain.into(),
                target_dreyfus,
                author: String::new(),
                description: String::new(),
            },
            seeds: RecipeSeeds {
                required: seed_concepts.to_vec(),
                optional: Vec::new(),
            },
            prerequisites,
            resources: RecipeResources::default(),
            validation: RecipeValidation::default(),
            identity: None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RECIPE: &str = r#"
[purpose]
id = "gcc-compiler-expert"
name = "GCC Compiler Expert"
version = "1.0.0"
domain = "compiler-engineering"
target_dreyfus = "Expert"
author = "akh://test@localhost"
description = "Bootstrap for compiler engineering expertise"

[seeds]
required = ["compiler-theory", "c-language"]
optional = ["assembly-x86"]

[prerequisites]
"formal-languages" = { before = ["lexical-analysis", "parsing"] }
"lexical-analysis" = { before = ["compiler-front-end"] }

[resources]
urls = ["https://gcc.gnu.org/onlinedocs/gccint/"]
skillpacks = ["gcc-internals"]

[validation]
min_triples = 500
min_coverage = 0.75
required_types = ["optimization-pass", "intermediate-representation"]

[identity]
reference = "Ptah"
reference_qid = "Q146321"
archetype = "creator"
culture = "egyptian"
suggested_names = ["Ptah-medu", "Ptah-maat"]
traits = ["meticulous", "creative", "systematic"]
"#;

    #[test]
    fn parse_full_recipe() {
        let recipe = PurposeRecipe::from_toml(SAMPLE_RECIPE).unwrap();
        assert_eq!(recipe.purpose.id, "gcc-compiler-expert");
        assert_eq!(recipe.purpose.domain, "compiler-engineering");
        assert_eq!(recipe.purpose.target_dreyfus, DreyfusLevel::Expert);
        assert_eq!(recipe.seeds.required.len(), 2);
        assert_eq!(recipe.seeds.optional.len(), 1);
        assert_eq!(recipe.prerequisites.len(), 2);
        assert_eq!(recipe.resources.urls.len(), 1);
        assert_eq!(recipe.validation.min_triples, 500);
        assert!(recipe.identity.is_some());
        assert_eq!(recipe.identity.unwrap().reference, "Ptah");
    }

    #[test]
    fn parse_minimal_recipe() {
        let toml = r#"
[purpose]
id = "test"
name = "Test"
domain = "testing"

[seeds]
required = ["basics"]
"#;
        let recipe = PurposeRecipe::from_toml(toml).unwrap();
        assert_eq!(recipe.purpose.id, "test");
        assert!(recipe.identity.is_none());
        assert!(recipe.resources.urls.is_empty());
    }

    #[test]
    fn roundtrip_toml() {
        let recipe = PurposeRecipe::from_toml(SAMPLE_RECIPE).unwrap();
        let serialized = recipe.to_toml().unwrap();
        let restored = PurposeRecipe::from_toml(&serialized).unwrap();
        assert_eq!(restored.purpose.id, recipe.purpose.id);
        assert_eq!(restored.seeds.required.len(), recipe.seeds.required.len());
    }

    #[test]
    fn validate_valid_recipe() {
        let recipe = PurposeRecipe::from_toml(SAMPLE_RECIPE).unwrap();
        assert!(recipe.validate().is_ok());
    }

    #[test]
    fn validate_missing_id() {
        let toml = r#"
[purpose]
id = ""
name = "Test"
domain = "testing"

[seeds]
required = ["basics"]
"#;
        let recipe = PurposeRecipe::from_toml(toml).unwrap();
        assert!(recipe.validate().is_err());
    }

    #[test]
    fn validate_no_seeds_or_urls() {
        let toml = r#"
[purpose]
id = "test"
name = "Test"
domain = "testing"
"#;
        let recipe = PurposeRecipe::from_toml(toml).unwrap();
        assert!(recipe.validate().is_err());
    }

    #[test]
    fn topological_sort() {
        let recipe = PurposeRecipe::from_toml(SAMPLE_RECIPE).unwrap();
        let order = recipe.topic_order();
        // formal-languages should come before lexical-analysis and parsing.
        let fl_pos = order.iter().position(|t| t == "formal-languages");
        let la_pos = order.iter().position(|t| t == "lexical-analysis");
        assert!(fl_pos.is_some());
        if let (Some(fl), Some(la)) = (fl_pos, la_pos) {
            assert!(fl < la, "formal-languages should precede lexical-analysis");
        }
    }

    #[test]
    fn cycle_detection() {
        let toml = r#"
[purpose]
id = "cycle"
name = "Cycle"
domain = "test"

[seeds]
required = ["a"]

[prerequisites]
"a" = { before = ["b"] }
"b" = { before = ["a"] }
"#;
        let recipe = PurposeRecipe::from_toml(toml).unwrap();
        assert!(recipe.validate().is_err());
    }

    #[test]
    fn generate_recipe() {
        let recipe = PurposeRecipe::generate(
            "test-recipe",
            "Test Recipe",
            "testing",
            DreyfusLevel::Competent,
            &["unit-testing".into(), "integration-testing".into()],
            &[("integration-testing".into(), "unit-testing".into())],
        );
        assert_eq!(recipe.purpose.id, "test-recipe");
        assert_eq!(recipe.seeds.required.len(), 2);
        assert_eq!(recipe.prerequisites.len(), 1);
    }

    #[test]
    fn recipe_default_validation() {
        let v = RecipeValidation::default();
        assert_eq!(v.min_triples, 100);
        assert!((v.min_coverage - 0.5).abs() < f32::EPSILON);
    }
}
