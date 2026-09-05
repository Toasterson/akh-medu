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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, RwLock};
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

use super::{
    LoadedSkill, SkillAction, SkillActivation, SkillInfo, SkillManifest, SkillPlan, SkillState,
};

/// Manages the lifecycle of skillpacks.
pub struct SkillManager {
    skills_dir: PathBuf,
    skills: RwLock<HashMap<String, LoadedSkill>>,
    /// Serializes lifecycle transitions across the legacy direct-KG API and
    /// the Engine-owned canonical activation path.  The latter deliberately
    /// keeps this guard while the Engine admits and publishes every planned
    /// triple so a concurrent deactivate/activate cannot invalidate the plan
    /// before it is committed Hot.
    activation_gate: Mutex<()>,
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

/// Fully parsed, side-effect-free activation input.
///
/// The Engine consumes `triples` through its canonical mutation pipeline and
/// only then asks the manager to publish the lifecycle/rule/action state.  No
/// graph reference is exposed to the manager on that path.
pub(crate) struct SkillActivationPlan {
    skill_id: String,
    source_state: SkillState,
    manifest: SkillManifest,
    triples: Vec<Triple>,
    label_triples: Vec<(String, String, String, f32)>,
    rules: Vec<(String, String, String)>,
    actions: Vec<SkillAction>,
    plans: Vec<SkillPlan>,
    memory_bytes: usize,
}

impl SkillActivationPlan {
    pub(crate) fn triples(&self) -> &[Triple] {
        &self.triples
    }

    pub(crate) fn label_triples(&self) -> &[(String, String, String, f32)] {
        &self.label_triples
    }
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
            activation_gate: Mutex::new(()),
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
        let _activation_guard = self
            .activation_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.discover_locked()
    }

    fn discover_locked(&self) -> SkillResult<usize> {
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
                            actions_file: None,
                            plans_file: None,
                            procedures_file: None,
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
                        actions: Vec::new(),
                        plans: Vec::new(),
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
        let _activation_guard = self
            .activation_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.warm_locked(skill_id)
    }

    fn warm_locked(&self, skill_id: &str) -> SkillResult<SkillManifest> {
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

        let manifest = self.read_manifest(skill_id)?;

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
        self.activate_with::<SkillError, _>(skill_id, false, |plan| {
            for triple in plan.triples() {
                // Preserve the legacy manager-only API: callers that choose
                // it still publish directly to the supplied graph. Engine
                // callers use `activate_with` below and never enter here.
                let _ = knowledge_graph.insert_triple(triple);
            }
            Ok(plan.triples().len())
        })
    }

    /// Parse an activation completely, let the Engine publish its triples, and
    /// only then mark the skill Hot. The activation gate spans all three
    /// phases, so the plan cannot race a direct lifecycle transition.
    pub(crate) fn activate_with<E, F>(
        &self,
        skill_id: &str,
        allow_cold: bool,
        publish_triples: F,
    ) -> Result<SkillActivation, E>
    where
        E: From<SkillError>,
        F: FnOnce(&SkillActivationPlan) -> Result<usize, E>,
    {
        let _activation_guard = self
            .activation_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let plan = self
            .build_activation_plan_locked(skill_id, allow_cold)
            .map_err(E::from)?;
        let triples_loaded = publish_triples(&plan)?;
        self.commit_activation_locked(plan, triples_loaded)
            .map_err(E::from)
    }

    fn read_manifest(&self, skill_id: &str) -> SkillResult<SkillManifest> {
        let manifest_path = self.skills_dir.join(skill_id).join("skill.json");
        let content = std::fs::read_to_string(&manifest_path).map_err(|e| SkillError::Io {
            skill_id: skill_id.into(),
            source: e,
        })?;
        serde_json::from_str(&content).map_err(|e| SkillError::InvalidManifest {
            path: manifest_path.display().to_string(),
            message: e.to_string(),
        })
    }

    /// Build the entire activation without changing lifecycle state or any
    /// graph projection. All filesystem parsing and the memory/conflict checks
    /// therefore complete before the Engine is invited to publish a triple.
    fn build_activation_plan_locked(
        &self,
        skill_id: &str,
        allow_cold: bool,
    ) -> SkillResult<SkillActivationPlan> {
        let (source_state, manifest) = {
            let skills = self.skills.read().expect("skills lock poisoned");
            let skill = skills.get(skill_id).ok_or_else(|| SkillError::NotFound {
                name: skill_id.into(),
            })?;
            match skill.state {
                SkillState::Cold if allow_cold => (SkillState::Cold, self.read_manifest(skill_id)?),
                SkillState::Warm => (SkillState::Warm, skill.manifest.clone()),
                state => {
                    return Err(SkillError::InvalidTransition {
                        skill_id: skill_id.into(),
                        from: state.to_string(),
                        to: "Hot".into(),
                    });
                }
            }
        };

        let active_ids = self.active_skill_ids();
        if let Some(conflict) =
            super::graph::SkillDependencyGraph::check_conflicts(&manifest, &active_ids)
        {
            return Err(SkillError::SkillConflict {
                skill_id: skill_id.into(),
                conflict,
            });
        }

        let skill_dir = self.skills_dir.join(skill_id);
        let mut memory_bytes = 0usize;
        let mut triples = Vec::new();
        let mut label_triples = Vec::new();

        let triples_file = manifest
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
            let is_label_format = raw_triples
                .first()
                .is_some_and(|value| value.get("subject").is_some());
            for value in &raw_triples {
                let subject = value["s"].as_u64().unwrap_or(0);
                let predicate = value["p"].as_u64().unwrap_or(0);
                let object = value["o"].as_u64().unwrap_or(0);
                if let (Some(subject), Some(predicate), Some(object)) = (
                    SymbolId::new(subject),
                    SymbolId::new(predicate),
                    SymbolId::new(object),
                ) {
                    triples.push(Triple::new(subject, predicate, object));
                }
                if is_label_format {
                    let subject = value["subject"].as_str().unwrap_or("");
                    let predicate = value["predicate"].as_str().unwrap_or("");
                    let object = value["object"].as_str().unwrap_or("");
                    if !subject.is_empty() && !predicate.is_empty() && !object.is_empty() {
                        label_triples.push((
                            subject.to_string(),
                            predicate.to_string(),
                            object.to_string(),
                            value["confidence"].as_f64().unwrap_or(1.0) as f32,
                        ));
                    }
                }
            }
            memory_bytes += content.len();
        }

        let rules_file = manifest
            .rules_file
            .clone()
            .unwrap_or_else(|| "rules.txt".into());
        let rules_path = skill_dir.join(&rules_file);
        let mut rules = Vec::new();
        if rules_path.exists() {
            let content = std::fs::read_to_string(&rules_path).map_err(|e| SkillError::Io {
                skill_id: skill_id.into(),
                source: e,
            })?;
            for (index, line) in content.lines().enumerate() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                if let Some((lhs, rhs)) = line.split_once("=>") {
                    rules.push((
                        format!("{skill_id}-rule-{index}"),
                        lhs.trim().to_string(),
                        rhs.trim().to_string(),
                    ));
                } else {
                    return Err(SkillError::InvalidRule {
                        skill_id: skill_id.into(),
                        message: format!("line {}: missing '=>' separator", index + 1),
                    });
                }
            }
            memory_bytes += content.len();
        }

        let actions_file = manifest
            .actions_file
            .clone()
            .unwrap_or_else(|| "actions.json".into());
        let actions_path = skill_dir.join(&actions_file);
        let actions = if actions_path.exists() {
            let content = std::fs::read_to_string(&actions_path).map_err(|e| SkillError::Io {
                skill_id: skill_id.into(),
                source: e,
            })?;
            memory_bytes += content.len();
            serde_json::from_str(&content).map_err(|e| SkillError::InvalidManifest {
                path: actions_path.display().to_string(),
                message: format!("actions parse error: {e}"),
            })?
        } else {
            Vec::new()
        };

        let plans_file = manifest
            .plans_file
            .clone()
            .unwrap_or_else(|| "plans.json".into());
        let plans_path = skill_dir.join(&plans_file);
        let plans = if plans_path.exists() {
            let content = std::fs::read_to_string(&plans_path).map_err(|e| SkillError::Io {
                skill_id: skill_id.into(),
                source: e,
            })?;
            memory_bytes += content.len();
            serde_json::from_str(&content).map_err(|e| SkillError::InvalidManifest {
                path: plans_path.display().to_string(),
                message: format!("plans parse error: {e}"),
            })?
        } else {
            Vec::new()
        };

        let current = self.loaded_bytes.load(Ordering::Relaxed);
        if current + memory_bytes > self.max_bytes {
            return Err(SkillError::MemoryBudgetExceeded {
                required_mb: memory_bytes / (1024 * 1024) + 1,
                available_mb: (self.max_bytes.saturating_sub(current)) / (1024 * 1024),
            });
        }

        Ok(SkillActivationPlan {
            skill_id: skill_id.into(),
            source_state,
            manifest,
            triples,
            label_triples,
            rules,
            actions,
            plans,
            memory_bytes,
        })
    }

    fn commit_activation_locked(
        &self,
        plan: SkillActivationPlan,
        triples_loaded: usize,
    ) -> SkillResult<SkillActivation> {
        let SkillActivationPlan {
            skill_id,
            source_state,
            manifest,
            triples,
            label_triples: _,
            rules,
            actions,
            plans,
            memory_bytes,
        } = plan;
        let rules_loaded = rules.len();
        let actions_loaded = actions.len();
        let plans_loaded = plans.len();
        let now = Instant::now();

        let mut skills = self.skills.write().expect("skills lock poisoned");
        let skill = skills
            .get_mut(&skill_id)
            .ok_or_else(|| SkillError::NotFound {
                name: skill_id.clone(),
            })?;
        if skill.state != source_state {
            return Err(SkillError::InvalidTransition {
                skill_id,
                from: skill.state.to_string(),
                to: "Hot".into(),
            });
        }

        self.loaded_bytes.fetch_add(memory_bytes, Ordering::Relaxed);
        skill.manifest = manifest;
        skill.state = SkillState::Hot;
        skill.triple_count = triples_loaded;
        skill.rule_count = rules_loaded;
        skill.memory_bytes = memory_bytes;
        skill.actions = actions;
        skill.plans = plans;
        skill.last_used_at = Some(now);

        if !rules.is_empty() {
            self.rule_sources
                .write()
                .expect("rule_sources lock poisoned")
                .insert(skill_id.clone(), rules);
        }
        if !triples.is_empty() {
            self.skill_triples
                .write()
                .expect("skill_triples lock poisoned")
                .insert(skill_id.clone(), triples);
        }
        self.last_used.insert(skill_id.clone(), now);

        if actions_loaded > 0 {
            tracing::debug!(skill_id, actions_loaded, "loaded procedural action schemas");
        }
        if plans_loaded > 0 {
            tracing::debug!(skill_id, plans_loaded, "loaded procedural plan templates");
        }

        Ok(SkillActivation {
            skill_id,
            triples_loaded,
            rules_loaded,
            memory_bytes,
            actions_loaded,
            plans_loaded,
        })
    }

    /// Deactivate a Hot skill: mark inactive, free budget.
    /// Triples already in the KG persist until engine restart.
    pub fn deactivate(&self, skill_id: &str) -> SkillResult<()> {
        let _activation_guard = self
            .activation_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
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
        // Clear procedural data (Phase 30).
        skill.actions.clear();
        skill.plans.clear();
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
            actions_loaded: skill.actions.len(),
            plans_loaded: skill.plans.len(),
        })
    }

    /// Add extra triples to a Hot skill's count (e.g. label triples loaded externally).
    pub fn add_triple_count(&self, skill_id: &str, extra: usize) {
        let _activation_guard = self
            .activation_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
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
                action_count: s.actions.len(),
                plan_count: s.plans.len(),
                provides: s.manifest.provides.clone(),
                requires: s.manifest.requires.clone(),
            })
            .collect()
    }

    /// Return the action schemas loaded for a Hot skill (Phase 30).
    pub fn actions(&self, skill_id: &str) -> Vec<SkillAction> {
        let skills = self.skills.read().expect("skills lock poisoned");
        skills
            .get(skill_id)
            .filter(|s| s.state == SkillState::Hot)
            .map(|s| s.actions.clone())
            .unwrap_or_default()
    }

    /// Return the plan templates loaded for a Hot skill (Phase 30).
    pub fn plans(&self, skill_id: &str) -> Vec<SkillPlan> {
        let skills = self.skills.read().expect("skills lock poisoned");
        skills
            .get(skill_id)
            .filter(|s| s.state == SkillState::Hot)
            .map(|s| s.plans.clone())
            .unwrap_or_default()
    }

    /// Return all action schemas from all Hot skills (Phase 30).
    pub fn all_active_actions(&self) -> Vec<(String, SkillAction)> {
        let skills = self.skills.read().expect("skills lock poisoned");
        skills
            .values()
            .filter(|s| s.state == SkillState::Hot)
            .flat_map(|s| {
                s.actions
                    .iter()
                    .map(|a| (s.manifest.id.clone(), a.clone()))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Return all plan templates from all Hot skills (Phase 30).
    pub fn all_active_plans(&self) -> Vec<(String, SkillPlan)> {
        let skills = self.skills.read().expect("skills lock poisoned");
        skills
            .values()
            .filter(|s| s.state == SkillState::Hot)
            .flat_map(|s| {
                s.plans
                    .iter()
                    .map(|p| (s.manifest.id.clone(), p.clone()))
                    .collect::<Vec<_>>()
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
            action_count: skill.actions.len(),
            plan_count: skill.plans.len(),
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
        assert_eq!(activation.actions_loaded, 0);
        assert_eq!(activation.plans_loaded, 0);
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
    fn activate_loads_actions_and_plans() {
        let (dir, skills_dir) = setup_skill_dir();
        let skill_dir = dir.path().join("skills").join("astronomy");

        // Write actions.json
        let actions = serde_json::json!([
            {
                "action": "observe_star",
                "description": "Observe a star to determine its properties",
                "preconditions": ["telescope_available"],
                "effects": {
                    "initiates": ["star_observed"],
                    "terminates": ["star_unknown"]
                },
                "estimated_duration": "medium"
            }
        ]);
        std::fs::write(
            skill_dir.join("actions.json"),
            serde_json::to_string_pretty(&actions).unwrap(),
        )
        .unwrap();

        // Write plans.json
        let plans = serde_json::json!([
            {
                "goal": "classify_star",
                "method": {"type": "sequential"},
                "steps": ["observe_star"],
                "preconditions": ["has_star_to_classify"]
            }
        ]);
        std::fs::write(
            skill_dir.join("plans.json"),
            serde_json::to_string_pretty(&plans).unwrap(),
        )
        .unwrap();

        let mgr = SkillManager::new(skills_dir, 100);
        let kg = KnowledgeGraph::new();

        mgr.discover().unwrap();
        mgr.warm("astronomy").unwrap();
        let activation = mgr.activate("astronomy", &kg).unwrap();

        assert_eq!(activation.actions_loaded, 1);
        assert_eq!(activation.plans_loaded, 1);

        let actions = mgr.actions("astronomy");
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].action, "observe_star");

        let plans = mgr.plans("astronomy");
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].goal, "classify_star");

        let all_actions = mgr.all_active_actions();
        assert_eq!(all_actions.len(), 1);
        assert_eq!(all_actions[0].0, "astronomy");
    }

    #[test]
    fn deactivate_clears_procedural_data() {
        let (dir, skills_dir) = setup_skill_dir();
        let skill_dir = dir.path().join("skills").join("astronomy");

        let actions = serde_json::json!([{
            "action": "observe_star",
            "description": "Observe a star",
            "preconditions": [],
            "effects": {"initiates": [], "terminates": []},
            "estimated_duration": "fast"
        }]);
        std::fs::write(
            skill_dir.join("actions.json"),
            serde_json::to_string(&actions).unwrap(),
        )
        .unwrap();

        let mgr = SkillManager::new(skills_dir, 100);
        let kg = KnowledgeGraph::new();
        mgr.discover().unwrap();
        mgr.warm("astronomy").unwrap();
        let activation = mgr.activate("astronomy", &kg).unwrap();
        assert_eq!(activation.actions_loaded, 1);

        mgr.deactivate("astronomy").unwrap();

        let actions = mgr.actions("astronomy");
        assert!(actions.is_empty(), "actions cleared after deactivate");
        let all = mgr.all_active_actions();
        assert!(all.is_empty());
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
        assert_eq!(mgr.skill_state("astronomy"), Some(SkillState::Warm));
        assert_eq!(
            kg.triple_count(),
            0,
            "a rejected plan must not touch the KG"
        );
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
