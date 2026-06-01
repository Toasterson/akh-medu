//! Skill manager: discover, warm, activate, and deactivate skillpacks.
//!
//! Scans a skills directory for subdirectories containing `skill.json` manifests.
//! Each skill follows the lifecycle: Cold → Warm → Hot.
//!
//! Extended in Phase 31:
//! - 31a: dependency-graph-aware activation (via `graph.rs`)
//! - 31b: per-skill neural prototype storage and query matching
//! - 31c: last-used tracking for idle-deactivation routing

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use egg::{Pattern, Rewrite};

use crate::error::{SkillError, SkillResult};
use crate::graph::Triple;
use crate::graph::index::KnowledgeGraph;
use crate::reason::AkhLang;
use crate::symbol::SymbolId;
use crate::vsa::HyperVec;
use crate::vsa::encode::encode_symbol;
use crate::vsa::ops::VsaOps;

use super::{LoadedSkill, SkillActivation, SkillInfo, SkillManifest, SkillState};

/// Manages the lifecycle of skillpacks.
pub struct SkillManager {
    skills_dir: PathBuf,
    skills: RwLock<HashMap<String, LoadedSkill>>,
    loaded_bytes: AtomicUsize,
    max_bytes: usize,
    /// Stored rule text per skill: skill_id -> Vec<(name, lhs, rhs)>.
    #[allow(clippy::type_complexity)]
    rule_sources: RwLock<HashMap<String, Vec<(String, String, String)>>>,
    /// Triples loaded from each active skill (skill_id → triples).
    /// Used by Phase 31b prototype computation.
    skill_triples: RwLock<HashMap<String, Vec<Triple>>>,
    /// Neural prototype per skill (skill_id → serialised HyperVec bytes).
    /// Populated at activation time via `compute_and_store_prototype`.
    prototypes: DashMap<String, Vec<u8>>,
    /// Last-access timestamp per Hot skill (Phase 31c idle-deactivation).
    last_used: DashMap<String, Instant>,
}

impl std::fmt::Debug for SkillManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SkillManager")
            .field("skills_dir", &self.skills_dir)
            .field("loaded_bytes", &self.loaded_bytes.load(Ordering::Relaxed))
            .field("max_bytes", &self.max_bytes)
            .finish()
    }
}

impl SkillManager {
    /// Create a new skill manager.
    ///
    /// `max_memory_mb` is the total memory budget for loaded skillpacks.
    pub fn new(skills_dir: PathBuf, max_memory_mb: usize) -> Self {
        Self {
            skills_dir,
            skills: RwLock::new(HashMap::new()),
            loaded_bytes: AtomicUsize::new(0),
            max_bytes: max_memory_mb * 1024 * 1024,
            rule_sources: RwLock::new(HashMap::new()),
            skill_triples: RwLock::new(HashMap::new()),
            prototypes: DashMap::new(),
            last_used: DashMap::new(),
        }
    }

    /// Get the skills directory path.
    pub fn skills_dir(&self) -> &std::path::Path {
        &self.skills_dir
    }

    /// Scan the skills directory for subdirectories containing `skill.json`.
    ///
    /// Registers each found skill as Cold. Returns the number of skills discovered.
    pub fn discover(&self) -> SkillResult<usize> {
        if !self.skills_dir.exists() {
            return Err(SkillError::NoSkillsDir {
                path: self.skills_dir.display().to_string(),
            });
        }

        let entries = std::fs::read_dir(&self.skills_dir).map_err(|e| SkillError::Io {
            skill_id: "<discovery>".into(),
            source: e,
        })?;

        let mut count = 0;
        let mut skills = self.skills.write().expect("skills lock poisoned");

        for entry in entries {
            let entry = entry.map_err(|e| SkillError::Io {
                skill_id: "<discovery>".into(),
                source: e,
            })?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            let manifest_path = path.join("skill.json");
            if !manifest_path.exists() {
                continue;
            }

            // Use the directory name as the skill ID for discovery.
            let skill_id = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();

            if !skills.contains_key(&skill_id) {
                skills.insert(
                    skill_id.clone(),
                    LoadedSkill {
                        manifest: SkillManifest {
                            id: skill_id,
                            name: String::new(),
                            version: String::new(),
                            description: String::new(),
                            domains: Vec::new(),
                            weight_size_bytes: 0,
                            triples_file: None,
                            rules_file: None,
                            cli_tools: Vec::new(),
                            wasm_tools: Vec::new(),
                            tool_config: std::collections::HashMap::new(),
                            depends: Vec::new(),
                            conflicts: Vec::new(),
                            provides: Vec::new(),
                            requires: Vec::new(),
                            remote_url: None,
                            curriculum_file: None,
                        },
                        state: SkillState::Cold,
                        triple_count: 0,
                        rule_count: 0,
                        memory_bytes: 0,
                        last_used_at: None,
                    },
                );
                count += 1;
            }
        }

        Ok(count)
    }

    /// Parse the manifest for a Cold skill, transitioning it to Warm.
    pub fn warm(&self, skill_id: &str) -> SkillResult<SkillManifest> {
        let mut skills = self.skills.write().expect("skills lock poisoned");
        let skill = skills
            .get_mut(skill_id)
            .ok_or_else(|| SkillError::NotFound {
                name: skill_id.into(),
            })?;

        if skill.state != SkillState::Cold {
            return Err(SkillError::InvalidTransition {
                skill_id: skill_id.into(),
                from: skill.state.to_string(),
                to: "Warm".into(),
            });
        }

        let manifest_path = self.skills_dir.join(skill_id).join("skill.json");
        let content = std::fs::read_to_string(&manifest_path).map_err(|e| SkillError::Io {
            skill_id: skill_id.into(),
            source: e,
        })?;

        let manifest: SkillManifest =
            serde_json::from_str(&content).map_err(|e| SkillError::InvalidManifest {
                path: manifest_path.display().to_string(),
                message: e.to_string(),
            })?;

        skill.manifest = manifest.clone();
        skill.state = SkillState::Warm;

        Ok(manifest)
    }

    /// Activate a Warm skill: load triples into the knowledge graph
    /// and parse rules. Transitions the skill to Hot.
    ///
    /// Phase 31a: checks for conflicts with currently active skills before
    /// loading. Phase 31c: records the activation timestamp.
    pub fn activate(
        &self,
        skill_id: &str,
        knowledge_graph: &KnowledgeGraph,
    ) -> SkillResult<SkillActivation> {
        // Phase 31a: conflict check against currently active skills.
        let active_ids = self.active_skill_ids();
        {
            let skills = self.skills.read().expect("skills lock poisoned");
            if let Some(skill) = skills.get(skill_id) {
                if let Some(conflict) = super::graph::SkillDependencyGraph::check_conflicts(
                    &skill.manifest,
                    &active_ids,
                ) {
                    return Err(SkillError::SkillConflict {
                        skill_id: skill_id.into(),
                        conflict,
                    });
                }
            }
        }

        let mut skills = self.skills.write().expect("skills lock poisoned");
        let skill = skills
            .get_mut(skill_id)
            .ok_or_else(|| SkillError::NotFound {
                name: skill_id.into(),
            })?;

        if skill.state != SkillState::Warm {
            return Err(SkillError::InvalidTransition {
                skill_id: skill_id.into(),
                from: skill.state.to_string(),
                to: "Hot".into(),
            });
        }

        let skill_dir = self.skills_dir.join(skill_id);
        let mut triples_loaded = 0usize;
        let mut rules_loaded = 0usize;
        let mut memory_estimate = 0usize;
        let mut loaded_triples: Vec<Triple> = Vec::new();

        // Load triples if available.
        let triples_file = skill
            .manifest
            .triples_file
            .clone()
            .unwrap_or_else(|| "triples.json".into());
        let triples_path = skill_dir.join(&triples_file);
        if triples_path.exists() {
            let content = std::fs::read_to_string(&triples_path).map_err(|e| SkillError::Io {
                skill_id: skill_id.into(),
                source: e,
            })?;

            let raw_triples: Vec<serde_json::Value> =
                serde_json::from_str(&content).map_err(|e| SkillError::InvalidManifest {
                    path: triples_path.display().to_string(),
                    message: format!("triples parse error: {e}"),
                })?;

            for val in &raw_triples {
                let s = val["s"].as_u64().unwrap_or(0);
                let p = val["p"].as_u64().unwrap_or(0);
                let o = val["o"].as_u64().unwrap_or(0);

                if let (Some(s), Some(p), Some(o)) =
                    (SymbolId::new(s), SymbolId::new(p), SymbolId::new(o))
                {
                    let triple = Triple::new(s, p, o);
                    // Ignore duplicate triple errors.
                    let _ = knowledge_graph.insert_triple(&triple);
                    loaded_triples.push(triple);
                    triples_loaded += 1;
                }
            }

            memory_estimate += content.len();
        }

        // Load rules if available.
        let rules_file = skill
            .manifest
            .rules_file
            .clone()
            .unwrap_or_else(|| "rules.txt".into());
        let rules_path = skill_dir.join(&rules_file);
        let mut rule_tuples: Vec<(String, String, String)> = Vec::new();
        if rules_path.exists() {
            let content = std::fs::read_to_string(&rules_path).map_err(|e| SkillError::Io {
                skill_id: skill_id.into(),
                source: e,
            })?;

            for (i, line) in content.lines().enumerate() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                if let Some((lhs, rhs)) = line.split_once("=>") {
                    let name = format!("{}-rule-{}", skill_id, i);
                    rule_tuples.push((name, lhs.trim().to_string(), rhs.trim().to_string()));
                    rules_loaded += 1;
                } else {
                    return Err(SkillError::InvalidRule {
                        skill_id: skill_id.into(),
                        message: format!("line {}: missing '=>' separator", i + 1),
                    });
                }
            }

            memory_estimate += content.len();
        }

        // Check memory budget.
        let current = self.loaded_bytes.load(Ordering::Relaxed);
        if current + memory_estimate > self.max_bytes {
            return Err(SkillError::MemoryBudgetExceeded {
                required_mb: memory_estimate / (1024 * 1024) + 1,
                available_mb: (self.max_bytes.saturating_sub(current)) / (1024 * 1024),
            });
        }

        // Commit.
        self.loaded_bytes
            .fetch_add(memory_estimate, Ordering::Relaxed);
        skill.state = SkillState::Hot;
        skill.triple_count = triples_loaded;
        skill.rule_count = rules_loaded;
        skill.memory_bytes = memory_estimate;
        // Phase 31c: record activation time.
        skill.last_used_at = Some(Instant::now());

        if !rule_tuples.is_empty() {
            self.rule_sources
                .write()
                .expect("rule_sources lock poisoned")
                .insert(skill_id.to_string(), rule_tuples);
        }

        // Phase 31b: store triples for prototype computation.
        if !loaded_triples.is_empty() {
            self.skill_triples
                .write()
                .expect("skill_triples lock poisoned")
                .insert(skill_id.to_string(), loaded_triples);
        }

        // Phase 31c: record last-used timestamp.
        self.last_used.insert(skill_id.to_string(), Instant::now());

        Ok(SkillActivation {
            skill_id: skill_id.into(),
            triples_loaded,
            rules_loaded,
            memory_bytes: memory_estimate,
        })
    }

    /// Deactivate a Hot skill: mark inactive, free budget.
    /// Triples already in the KG persist until engine restart.
    pub fn deactivate(&self, skill_id: &str) -> SkillResult<()> {
        let mut skills = self.skills.write().expect("skills lock poisoned");
        let skill = skills
            .get_mut(skill_id)
            .ok_or_else(|| SkillError::NotFound {
                name: skill_id.into(),
            })?;

        if skill.state != SkillState::Hot {
            return Err(SkillError::InvalidTransition {
                skill_id: skill_id.into(),
                from: skill.state.to_string(),
                to: "Warm (deactivate)".into(),
            });
        }

        self.loaded_bytes
            .fetch_sub(skill.memory_bytes, Ordering::Relaxed);
        skill.state = SkillState::Warm;
        skill.memory_bytes = 0;
        skill.last_used_at = None;

        self.rule_sources
            .write()
            .expect("rule_sources lock poisoned")
            .remove(skill_id);

        // Phase 31b/c: clear per-skill data.
        self.skill_triples
            .write()
            .expect("skill_triples lock poisoned")
            .remove(skill_id);
        self.prototypes.remove(skill_id);
        self.last_used.remove(skill_id);

        Ok(())
    }

    /// Get the current state of a skill, or `None` if not discovered.
    pub fn skill_state(&self, skill_id: &str) -> Option<SkillState> {
        let skills = self.skills.read().expect("skills lock poisoned");
        skills.get(skill_id).map(|s| s.state)
    }

    /// Build an activation summary for an already-Hot skill (idempotent reload).
    pub fn hot_activation(&self, skill_id: &str) -> Option<SkillActivation> {
        let skills = self.skills.read().expect("skills lock poisoned");
        let skill = skills.get(skill_id)?;
        if skill.state != SkillState::Hot {
            return None;
        }
        Some(SkillActivation {
            skill_id: skill_id.to_string(),
            triples_loaded: skill.triple_count,
            rules_loaded: skill.rule_count,
            memory_bytes: skill.memory_bytes,
        })
    }

    /// Add extra triples to a Hot skill's count (e.g. label triples loaded externally).
    pub fn add_triple_count(&self, skill_id: &str, extra: usize) {
        let mut skills = self.skills.write().expect("skills lock poisoned");
        if let Some(skill) = skills.get_mut(skill_id) {
            skill.triple_count += extra;
        }
    }

    /// List all known skills with their current state.
    pub fn list(&self) -> Vec<SkillInfo> {
        let skills = self.skills.read().expect("skills lock poisoned");
        skills
            .values()
            .map(|s| SkillInfo {
                id: s.manifest.id.clone(),
                name: s.manifest.name.clone(),
                version: s.manifest.version.clone(),
                description: s.manifest.description.clone(),
                state: s.state,
                domains: s.manifest.domains.clone(),
                triple_count: s.triple_count,
                rule_count: s.rule_count,
                provides: s.manifest.provides.clone(),
                requires: s.manifest.requires.clone(),
            })
            .collect()
    }

    /// Build rewrite rules from all active (Hot) skills.
    pub fn active_rules(&self) -> Vec<Rewrite<AkhLang, ()>> {
        let rule_sources = self
            .rule_sources
            .read()
            .expect("rule_sources lock poisoned");
        let mut rules = Vec::new();

        for (skill_id, tuples) in rule_sources.iter() {
            for (name, lhs, rhs) in tuples {
                // egg::rewrite! is a macro; we must use the lower-level API.
                let lhs_result: Result<Pattern<AkhLang>, _> = lhs.parse();
                let rhs_result: Result<Pattern<AkhLang>, _> = rhs.parse();
                if let (Ok(lhs_pat), Ok(rhs_pat)) = (lhs_result, rhs_result) {
                    match Rewrite::new(name.clone(), lhs_pat, rhs_pat) {
                        Ok(rule) => rules.push(rule),
                        Err(e) => {
                            tracing::warn!(
                                skill_id,
                                rule = name.as_str(),
                                error = %e,
                                "failed to compile skill rule, skipping"
                            );
                        }
                    }
                } else {
                    tracing::warn!(
                        skill_id,
                        rule = name.as_str(),
                        "failed to parse skill rule patterns, skipping"
                    );
                }
            }
        }

        rules
    }

    /// Get info for a specific skill.
    pub fn get_info(&self, skill_id: &str) -> SkillResult<SkillInfo> {
        let skills = self.skills.read().expect("skills lock poisoned");
        let skill = skills.get(skill_id).ok_or_else(|| SkillError::NotFound {
            name: skill_id.into(),
        })?;
        Ok(SkillInfo {
            id: skill.manifest.id.clone(),
            name: skill.manifest.name.clone(),
            version: skill.manifest.version.clone(),
            description: skill.manifest.description.clone(),
            state: skill.state,
            domains: skill.manifest.domains.clone(),
            triple_count: skill.triple_count,
            rule_count: skill.rule_count,
            provides: skill.manifest.provides.clone(),
            requires: skill.manifest.requires.clone(),
        })
    }

    // ── Phase 31b: Neural skill embeddings ───────────────────────────────────

    /// Return the triples loaded from a specific skill (Phase 31b).
    ///
    /// Only populated for currently-Hot skills loaded via the numeric triple
    /// path. Label-based triples loaded through `engine.load_skill_with_labels`
    /// are not tracked here.
    pub fn triples_of_skill(&self, skill_id: &str) -> Vec<Triple> {
        self.skill_triples
            .read()
            .expect("skill_triples lock poisoned")
            .get(skill_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Compute and store a neural prototype HyperVec for a skill (Phase 31b).
    ///
    /// The prototype is the bundle of all symbol vectors for the skill's
    /// triples (subject, predicate, object). Stored as serialised bytes under
    /// the skill's ID; retrieve with `prototype_bytes`.
    ///
    /// Call this after `activate` for skills where semantic matching is needed.
    /// Returns `None` if the skill has no triples to bundle.
    pub fn compute_and_store_prototype(&self, skill_id: &str, ops: &VsaOps) -> Option<()> {
        let triples = self.triples_of_skill(skill_id);
        if triples.is_empty() {
            return None;
        }

        // Encode each triple's symbols.
        let vecs: Vec<HyperVec> = triples
            .iter()
            .flat_map(|t| {
                [
                    encode_symbol(ops, t.subject),
                    encode_symbol(ops, t.predicate),
                    encode_symbol(ops, t.object),
                ]
            })
            .collect();

        let refs: Vec<&HyperVec> = vecs.iter().collect();
        let prototype = ops.bundle(&refs).ok()?;

        // Serialise to bytes for compact storage.
        let bytes = bincode::serialize(&prototype).ok()?;
        self.prototypes.insert(skill_id.to_string(), bytes);
        Some(())
    }

    /// Retrieve the stored prototype bytes for a skill (Phase 31b).
    ///
    /// Returns `None` if no prototype has been computed for this skill.
    pub fn prototype_bytes(&self, skill_id: &str) -> Option<Vec<u8>> {
        self.prototypes.get(skill_id).map(|b| b.value().clone())
    }

    /// Deserialise and return the prototype HyperVec for a skill (Phase 31b).
    ///
    /// Returns `None` if no prototype is stored for this skill.
    pub fn prototype(&self, skill_id: &str) -> Option<HyperVec> {
        let bytes = self.prototype_bytes(skill_id)?;
        bincode::deserialize(&bytes).ok()
    }

    /// Find the `k` most similar skills to a query HyperVec (Phase 31b/c).
    ///
    /// Returns a list of `(skill_id, similarity)` pairs sorted by descending
    /// similarity. Only considers skills that have a stored prototype.
    pub fn match_skills_to_query(
        &self,
        query: &HyperVec,
        ops: &VsaOps,
        k: usize,
        threshold: f32,
    ) -> Vec<(String, f32)> {
        let mut matches: Vec<(String, f32)> = self
            .prototypes
            .iter()
            .filter_map(|entry| {
                let skill_id = entry.key().clone();
                let proto: HyperVec = bincode::deserialize(entry.value()).ok()?;
                let sim = ops.similarity(query, &proto).ok()?;
                if sim >= threshold {
                    Some((skill_id, sim))
                } else {
                    None
                }
            })
            .collect();

        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        matches.truncate(k);
        matches
    }

    // ── Phase 31c: Skill-aware routing and idle deactivation ─────────────────

    /// Record that a skill was just used (Phase 31c).
    ///
    /// Updates both the `last_used` map and the `LoadedSkill.last_used_at` field.
    pub fn touch_skill(&self, skill_id: &str) {
        self.last_used.insert(skill_id.to_string(), Instant::now());
        let mut skills = self.skills.write().expect("skills lock poisoned");
        if let Some(skill) = skills.get_mut(skill_id) {
            skill.last_used_at = Some(Instant::now());
        }
    }

    /// Return the IDs of skills that have been idle longer than `cooldown` (Phase 31c).
    ///
    /// These are candidates for auto-deactivation to free memory.
    pub fn collect_idle_skills(&self, cooldown: Duration) -> Vec<String> {
        let now = Instant::now();
        self.last_used
            .iter()
            .filter(|entry| now.duration_since(*entry.value()) > cooldown)
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Return the IDs of all currently-Hot skills.
    pub fn active_skill_ids(&self) -> Vec<String> {
        let skills = self.skills.read().expect("skills lock poisoned");
        skills
            .iter()
            .filter(|(_, s)| s.state == SkillState::Hot)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Return all known skill manifests (for graph construction).
    pub fn all_manifests(&self) -> HashMap<String, SkillManifest> {
        let skills = self.skills.read().expect("skills lock poisoned");
        skills
            .iter()
            .map(|(id, s)| (id.clone(), s.manifest.clone()))
            .collect()
    }

    /// Compute the activation plan for a target skill using the dependency graph.
    ///
    /// Returns an ordered list of skill IDs to activate (dependencies first),
    /// excluding already-active skills. Returns an error for circular dependencies
    /// or conflicts.
    pub fn activation_plan(&self, target_skill_id: &str) -> SkillResult<Vec<String>> {
        let all_manifests = self.all_manifests();
        let active_ids = self.active_skill_ids();
        super::graph::compute_activation_plan(target_skill_id, &all_manifests, &active_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Create a temp skills directory with a test skillpack.
    fn setup_skill_dir() -> (TempDir, PathBuf) {
        let dir = TempDir::new().unwrap();
        let skills_dir = dir.path().join("skills");
        std::fs::create_dir(&skills_dir).unwrap();

        let skill_dir = skills_dir.join("astronomy");
        std::fs::create_dir(&skill_dir).unwrap();

        // Write manifest.
        let manifest = serde_json::json!({
            "id": "astronomy",
            "name": "Astronomy Pack",
            "version": "1.0.0",
            "description": "Basic astronomical knowledge",
            "domains": ["astronomy", "science"],
            "weight_size_bytes": 0,
            "triples_file": "triples.json",
            "rules_file": "rules.txt",
            "depends": [],
            "conflicts": [],
            "provides": ["celestial_bodies"],
            "requires": []
        });
        std::fs::write(
            skill_dir.join("skill.json"),
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();

        // Write triples.
        let triples = serde_json::json!([
            {"s": 1, "p": 2, "o": 3},
            {"s": 3, "p": 2, "o": 4}
        ]);
        std::fs::write(
            skill_dir.join("triples.json"),
            serde_json::to_string(&triples).unwrap(),
        )
        .unwrap();

        // Write rules.
        std::fs::write(
            skill_dir.join("rules.txt"),
            "# Astronomy rules\n(similar ?x ?y) => (similar ?y ?x)\n",
        )
        .unwrap();

        (dir, skills_dir)
    }

    #[test]
    fn discover_skills() {
        let (_dir, skills_dir) = setup_skill_dir();
        let mgr = SkillManager::new(skills_dir, 100);
        let count = mgr.discover().unwrap();
        assert_eq!(count, 1);

        let list = mgr.list();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, "astronomy");
        assert_eq!(list[0].state, SkillState::Cold);
    }

    #[test]
    fn warm_parses_manifest() {
        let (_dir, skills_dir) = setup_skill_dir();
        let mgr = SkillManager::new(skills_dir, 100);
        mgr.discover().unwrap();

        let manifest = mgr.warm("astronomy").unwrap();
        assert_eq!(manifest.name, "Astronomy Pack");
        assert_eq!(manifest.version, "1.0.0");
        assert_eq!(manifest.domains, vec!["astronomy", "science"]);
        assert_eq!(manifest.provides, vec!["celestial_bodies"]);
    }

    #[test]
    fn activate_loads_triples() {
        let (_dir, skills_dir) = setup_skill_dir();
        let mgr = SkillManager::new(skills_dir, 100);
        let kg = KnowledgeGraph::new();

        mgr.discover().unwrap();
        mgr.warm("astronomy").unwrap();
        let activation = mgr.activate("astronomy", &kg).unwrap();

        assert_eq!(activation.triples_loaded, 2);
        assert!(kg.triple_count() >= 2);
    }

    #[test]
    fn activate_loads_rules() {
        let (_dir, skills_dir) = setup_skill_dir();
        let mgr = SkillManager::new(skills_dir, 100);
        let kg = KnowledgeGraph::new();

        mgr.discover().unwrap();
        mgr.warm("astronomy").unwrap();
        let activation = mgr.activate("astronomy", &kg).unwrap();
        assert_eq!(activation.rules_loaded, 1);

        let rules = mgr.active_rules();
        assert_eq!(rules.len(), 1);
    }

    #[test]
    fn budget_enforcement() {
        let (_dir, skills_dir) = setup_skill_dir();
        // Extremely small budget: 0 MB.
        let mgr = SkillManager::new(skills_dir, 0);
        let kg = KnowledgeGraph::new();

        mgr.discover().unwrap();
        mgr.warm("astronomy").unwrap();
        let result = mgr.activate("astronomy", &kg);
        assert!(matches!(
            result,
            Err(SkillError::MemoryBudgetExceeded { .. })
        ));
    }

    #[test]
    fn deactivate_frees_budget() {
        let (_dir, skills_dir) = setup_skill_dir();
        let mgr = SkillManager::new(skills_dir, 100);
        let kg = KnowledgeGraph::new();

        mgr.discover().unwrap();
        mgr.warm("astronomy").unwrap();
        let activation = mgr.activate("astronomy", &kg).unwrap();
        assert!(activation.memory_bytes > 0);

        mgr.deactivate("astronomy").unwrap();

        let info = mgr.get_info("astronomy").unwrap();
        assert_eq!(info.state, SkillState::Warm);

        // Rules should be cleared.
        let rules = mgr.active_rules();
        assert!(rules.is_empty());
    }

    #[test]
    fn triples_of_skill_populated_after_activate() {
        let (_dir, skills_dir) = setup_skill_dir();
        let mgr = SkillManager::new(skills_dir, 100);
        let kg = KnowledgeGraph::new();

        mgr.discover().unwrap();
        mgr.warm("astronomy").unwrap();
        mgr.activate("astronomy", &kg).unwrap();

        let triples = mgr.triples_of_skill("astronomy");
        assert_eq!(triples.len(), 2);
    }

    #[test]
    fn triples_cleared_after_deactivate() {
        let (_dir, skills_dir) = setup_skill_dir();
        let mgr = SkillManager::new(skills_dir, 100);
        let kg = KnowledgeGraph::new();

        mgr.discover().unwrap();
        mgr.warm("astronomy").unwrap();
        mgr.activate("astronomy", &kg).unwrap();
        mgr.deactivate("astronomy").unwrap();

        let triples = mgr.triples_of_skill("astronomy");
        assert!(triples.is_empty());
    }

    #[test]
    fn idle_skill_detection() {
        let (_dir, skills_dir) = setup_skill_dir();
        let mgr = SkillManager::new(skills_dir, 100);
        let kg = KnowledgeGraph::new();

        mgr.discover().unwrap();
        mgr.warm("astronomy").unwrap();
        mgr.activate("astronomy", &kg).unwrap();

        // Immediately: no skills are "idle" with a 1-hour cooldown.
        let idle = mgr.collect_idle_skills(Duration::from_secs(3600));
        assert!(idle.is_empty());

        // With a zero cooldown everything is idle.
        let idle = mgr.collect_idle_skills(Duration::from_secs(0));
        assert_eq!(idle.len(), 1);
        assert_eq!(idle[0], "astronomy");
    }

    #[test]
    fn touch_skill_updates_timestamp() {
        let (_dir, skills_dir) = setup_skill_dir();
        let mgr = SkillManager::new(skills_dir, 100);
        let kg = KnowledgeGraph::new();

        mgr.discover().unwrap();
        mgr.warm("astronomy").unwrap();
        mgr.activate("astronomy", &kg).unwrap();

        // After touch, idle detection with 0-second cooldown still returns it
        // (touch just resets the timestamp; detecting idle needs elapsed time).
        mgr.touch_skill("astronomy");
        let idle_long = mgr.collect_idle_skills(Duration::from_secs(3600));
        assert!(idle_long.is_empty());
    }

    #[test]
    fn conflict_prevents_activation() {
        let dir = TempDir::new().unwrap();
        let skills_dir = dir.path().join("skills");
        std::fs::create_dir(&skills_dir).unwrap();

        for (name, conflicts) in [("base", vec![]), ("conflicting", vec!["base"])] {
            let skill_dir = skills_dir.join(name);
            std::fs::create_dir(&skill_dir).unwrap();
            let manifest = serde_json::json!({
                "id": name,
                "name": name,
                "version": "1.0.0",
                "domains": [],
                "depends": [],
                "conflicts": conflicts,
                "provides": [],
                "requires": []
            });
            std::fs::write(
                skill_dir.join("skill.json"),
                serde_json::to_string_pretty(&manifest).unwrap(),
            )
            .unwrap();
        }

        let mgr = SkillManager::new(skills_dir, 100);
        let kg = KnowledgeGraph::new();
        mgr.discover().unwrap();

        // Activate "base".
        mgr.warm("base").unwrap();
        mgr.activate("base", &kg).unwrap();

        // Trying to activate "conflicting" should fail.
        mgr.warm("conflicting").unwrap();
        let result = mgr.activate("conflicting", &kg);
        assert!(matches!(result, Err(SkillError::SkillConflict { .. })));
    }
}
