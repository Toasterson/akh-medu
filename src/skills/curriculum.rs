//! Skill curriculum: learning progression for bootstrapping.
//!
//! Phase 31h — extends skills with a Bloom-taxonomy-based curriculum that
//! orders concept acquisition, specifies mastery targets, and provides
//! assessment queries. The bootstrap pipeline (Phase 14) uses curricula to
//! structure domain learning.

use serde::{Deserialize, Serialize};

use crate::error::{SkillError, SkillResult};

/// Bloom's taxonomy mastery target for a curriculum concept.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BloomLevel {
    /// Recall facts and basic concepts.
    Remember,
    /// Explain ideas or concepts.
    Understand,
    /// Use information in new situations.
    Apply,
    /// Draw connections among ideas.
    Analyze,
    /// Justify a decision or course of action.
    Evaluate,
    /// Produce new or original work.
    Create,
}

impl std::fmt::Display for BloomLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Remember => "remember",
            Self::Understand => "understand",
            Self::Apply => "apply",
            Self::Analyze => "analyze",
            Self::Evaluate => "evaluate",
            Self::Create => "create",
        };
        write!(f, "{s}")
    }
}

impl std::str::FromStr for BloomLevel {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "remember" => Ok(Self::Remember),
            "understand" => Ok(Self::Understand),
            "apply" => Ok(Self::Apply),
            "analyze" | "analyse" => Ok(Self::Analyze),
            "evaluate" => Ok(Self::Evaluate),
            "create" => Ok(Self::Create),
            other => Err(format!("unknown Bloom level: {other}")),
        }
    }
}

/// A single concept within a skill's curriculum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumConcept {
    /// The concept's label (maps to a symbol in the KG).
    pub name: String,
    /// Target mastery level for this concept.
    pub bloom_target: BloomLevel,
    /// Concept IDs that must be understood before this one.
    #[serde(default)]
    pub prerequisites: Vec<String>,
}

/// An assessment query for testing understanding of a skill's concepts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assessment {
    /// The question or reasoning task (natural language).
    pub query: String,
    /// Concept names the correct answer should reference.
    pub expected_concepts: Vec<String>,
    /// The Bloom level this assessment targets.
    pub bloom_level: BloomLevel,
}

/// A learning progression for a skillpack.
///
/// Curricula are stored in `curriculum.json` alongside `skill.json` and
/// are loaded on demand.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillCurriculum {
    /// All concepts covered by the skill.
    pub concepts: Vec<CurriculumConcept>,
    /// Topological layers: each inner `Vec` contains concept names that can be
    /// learned in parallel once the previous layer is complete.
    #[serde(default)]
    pub layers: Vec<Vec<String>>,
    /// Assessment queries ordered from foundational to advanced.
    #[serde(default)]
    pub assessments: Vec<Assessment>,
}

impl SkillCurriculum {
    /// Load a curriculum from a JSON file.
    pub fn load(path: &std::path::Path) -> SkillResult<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| SkillError::Io {
            skill_id: path.display().to_string(),
            source: e,
        })?;
        serde_json::from_str(&content).map_err(|e| SkillError::InvalidManifest {
            path: path.display().to_string(),
            message: format!("curriculum parse error: {e}"),
        })
    }

    /// Compute topological layers from concept prerequisites using a simple
    /// Kahn's algorithm.
    ///
    /// Returns layers ordered from "learn first" to last. Concepts not
    /// reachable in the prerequisite graph appear in the final layer.
    pub fn compute_layers(&self) -> Vec<Vec<String>> {
        use std::collections::{HashMap, HashSet, VecDeque};

        let concept_names: HashSet<&str> = self.concepts.iter().map(|c| c.name.as_str()).collect();

        // in-degree: number of prerequisites still unsatisfied
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut dependents: HashMap<&str, Vec<&str>> = HashMap::new();

        for c in &self.concepts {
            let entry = in_degree.entry(c.name.as_str()).or_insert(0);
            for pre in &c.prerequisites {
                if concept_names.contains(pre.as_str()) {
                    *entry += 1;
                    dependents
                        .entry(pre.as_str())
                        .or_default()
                        .push(c.name.as_str());
                }
            }
        }

        let mut queue: VecDeque<&str> = in_degree
            .iter()
            .filter(|(_, d)| **d == 0)
            .map(|(&name, _)| name)
            .collect();

        let mut layers: Vec<Vec<String>> = Vec::new();

        while !queue.is_empty() {
            let layer_size = queue.len();
            let mut layer: Vec<String> = Vec::with_capacity(layer_size);
            for _ in 0..layer_size {
                let name = queue.pop_front().unwrap();
                layer.push(name.to_string());
                if let Some(deps) = dependents.get(name) {
                    for &dep in deps {
                        let d = in_degree.get_mut(dep).unwrap();
                        *d -= 1;
                        if *d == 0 {
                            queue.push_back(dep);
                        }
                    }
                }
            }
            layers.push(layer);
        }

        layers
    }

    /// Return all concepts at or below the given Bloom level that have no
    /// unsatisfied prerequisites (given a set of already-mastered concepts).
    pub fn next_concepts<'a>(
        &'a self,
        mastered: &std::collections::HashSet<String>,
        max_level: BloomLevel,
    ) -> Vec<&'a CurriculumConcept> {
        self.concepts
            .iter()
            .filter(|c| {
                c.bloom_target <= max_level
                    && !mastered.contains(&c.name)
                    && c.prerequisites.iter().all(|p| mastered.contains(p))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn concept(name: &str, bloom: BloomLevel, prereqs: &[&str]) -> CurriculumConcept {
        CurriculumConcept {
            name: name.to_string(),
            bloom_target: bloom,
            prerequisites: prereqs.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn bloom_ordering() {
        assert!(BloomLevel::Remember < BloomLevel::Create);
        assert!(BloomLevel::Apply < BloomLevel::Evaluate);
    }

    #[test]
    fn bloom_roundtrip() {
        use std::str::FromStr;
        assert_eq!(
            BloomLevel::from_str("remember").unwrap(),
            BloomLevel::Remember
        );
        assert_eq!(BloomLevel::from_str("CREATE").unwrap(), BloomLevel::Create);
    }

    #[test]
    fn compute_layers_simple() {
        let curriculum = SkillCurriculum {
            concepts: vec![
                concept("star", BloomLevel::Remember, &[]),
                concept("galaxy", BloomLevel::Understand, &["star"]),
                concept("cosmology", BloomLevel::Analyze, &["galaxy"]),
            ],
            layers: Vec::new(),
            assessments: Vec::new(),
        };

        let layers = curriculum.compute_layers();
        assert_eq!(layers.len(), 3);
        assert!(layers[0].contains(&"star".to_string()));
        assert!(layers[1].contains(&"galaxy".to_string()));
        assert!(layers[2].contains(&"cosmology".to_string()));
    }

    #[test]
    fn next_concepts_filters_by_bloom_and_prerequisites() {
        let curriculum = SkillCurriculum {
            concepts: vec![
                concept("star", BloomLevel::Remember, &[]),
                concept("galaxy", BloomLevel::Understand, &["star"]),
                concept("cosmology", BloomLevel::Analyze, &["galaxy"]),
            ],
            layers: Vec::new(),
            assessments: Vec::new(),
        };

        let mut mastered = std::collections::HashSet::new();
        // Nothing mastered: only `star` (Remember) is available.
        let next = curriculum.next_concepts(&mastered, BloomLevel::Understand);
        assert_eq!(next.len(), 1);
        assert_eq!(next[0].name, "star");

        mastered.insert("star".to_string());
        // After mastering star, `galaxy` (Understand) becomes available.
        let next = curriculum.next_concepts(&mastered, BloomLevel::Understand);
        assert_eq!(next.len(), 1);
        assert_eq!(next[0].name, "galaxy");
    }

    #[test]
    fn curriculum_roundtrip_json() {
        let curriculum = SkillCurriculum {
            concepts: vec![concept("star", BloomLevel::Remember, &[])],
            layers: vec![vec!["star".to_string()]],
            assessments: vec![Assessment {
                query: "What is a star?".to_string(),
                expected_concepts: vec!["star".to_string()],
                bloom_level: BloomLevel::Remember,
            }],
        };

        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), serde_json::to_string(&curriculum).unwrap()).unwrap();

        let loaded = SkillCurriculum::load(tmp.path()).unwrap();
        assert_eq!(loaded.concepts.len(), 1);
        assert_eq!(loaded.assessments.len(), 1);
    }
}
