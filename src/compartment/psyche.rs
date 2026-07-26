//! Jungian psyche model: Persona, Shadow, Archetypes, and Self-Integration.
//!
//! Maps Carl Jung's analytical psychology concepts to concrete Rust data types
//! that influence the agent's tool selection, output style, and ethical constraints.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::agent::tool_manifest::{Capability, DangerLevel, ToolManifest};
use crate::symbol::SymbolId;

// ---------------------------------------------------------------------------
// Core psyche types
// ---------------------------------------------------------------------------

/// The complete psyche of the agent — maps Jung's model to data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Psyche {
    /// The mask the agent presents outward.
    pub persona: Persona,
    /// Constrained anti-patterns: vetoes and biases.
    pub shadow: Shadow,
    /// Behavioral tendencies that bias tool selection.
    pub archetypes: ArchetypeWeights,
    /// Integrative center — tracks growth and individuation.
    pub self_integration: SelfIntegration,
    /// Whether the psyche has been finalized by the Ritual of Awakening.
    /// Once awakened, the psyche cannot be wholesale-replaced — only evolved.
    #[serde(default)]
    pub awakened: bool,
    /// Big Five personality profile, set during the Ritual of Awakening.
    /// `None` when the psyche was constructed without running the ritual;
    /// the psyche compiler derives equivalent values from archetype weights.
    #[serde(default)]
    pub ocean: Option<OceanProfile>,
    /// Ennead role assigned to this workspace (e.g. "Digital Architect").
    /// Written alongside the `ennead-role` file for inter-agent discovery.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ennead_role: Option<String>,
}

/// Persona — the mask the agent presents outward.
///
/// Controls communication style and grammar preference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Persona {
    /// Name for the persona (e.g., "Scholar", "Assistant", "Mentor").
    pub name: String,
    /// Grammar to use for output: either a built-in name ("formal", "narrative", "terse")
    /// or a path to a custom grammar TOML.
    pub grammar_preference: String,
    /// Traits that color communication (e.g., ["precise", "warm", "curious"]).
    pub traits: Vec<String>,
    /// Tone adjectives for the LLM polish step (e.g., ["encouraging", "methodical"]).
    pub tone: Vec<String>,
}

/// Shadow — constrained anti-patterns.
///
/// Critical patterns veto actions; lesser ones bias scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shadow {
    /// Patterns that trigger hard vetoes (action is blocked).
    pub veto_patterns: Vec<ShadowPattern>,
    /// Patterns that apply scoring penalties (action is discouraged).
    pub bias_patterns: Vec<ShadowPattern>,
}

/// A single shadow pattern that can veto or bias an action.
///
/// Matching logic (any match triggers):
/// 1. `capability_triggers` — tool manifest capabilities intersect pattern's set.
/// 2. `danger_level_threshold` — tool's danger level >= threshold.
/// 3. `action_triggers` — substring match in action description against tool's shadow_triggers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowPattern {
    /// Human-readable name for this pattern.
    pub name: String,
    /// Match on tool capabilities — any intersection triggers the pattern.
    #[serde(default)]
    pub capability_triggers: HashSet<Capability>,
    /// Match on danger level — tool danger >= threshold triggers the pattern.
    #[serde(default)]
    pub danger_level_threshold: Option<DangerLevel>,
    /// Match on tool's shadow_triggers — substring in action description.
    #[serde(default)]
    pub action_triggers: Vec<String>,
    /// Severity: 0.0 (ignorable) to 1.0 (absolute).
    pub severity: f32,
    /// Explanation shown when triggered.
    pub explanation: String,
}

/// Archetype weights — behavioral tendencies that bias tool selection.
///
/// Weights are in [0.0, 1.0] and sum is not constrained.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchetypeWeights {
    /// The Healer: prefers gap_analysis, user_interact. Seeks missing knowledge.
    pub healer: f32,
    /// The Sage: prefers kg_query, infer_rules, synthesize. Seeks understanding.
    pub sage: f32,
    /// The Guardian: prefers reflection, consolidation. Seeks stability and safety.
    pub guardian: f32,
    /// The Explorer: prefers http_fetch, file_io, shell_exec. Seeks novelty.
    pub explorer: f32,
}

impl ArchetypeWeights {
    /// Derive an OCEAN profile by weighted-averaging per-archetype baselines.
    ///
    /// Uses the same archetype→OCEAN data as `bootstrap::identity` so the
    /// derivation is consistent with the Ritual of Awakening path. Used when
    /// no explicit [`OceanProfile`] was set on the psyche.
    pub fn derive_ocean(&self) -> OceanProfile {
        // Baselines per archetype: [O, C, E, A, N].
        const SAGE: [f32; 5] = [0.9, 0.8, 0.3, 0.6, 0.2];
        const HEALER: [f32; 5] = [0.6, 0.7, 0.6, 0.9, 0.4]; // ≈ caregiver
        const GUARDIAN: [f32; 5] = [0.4, 0.9, 0.4, 0.7, 0.3];
        const EXPLORER: [f32; 5] = [0.9, 0.4, 0.7, 0.5, 0.4];

        let total = self.sage + self.healer + self.guardian + self.explorer;
        if total < f32::EPSILON {
            return OceanProfile {
                openness: 0.5,
                conscientiousness: 0.5,
                extraversion: 0.5,
                agreeableness: 0.5,
                neuroticism: 0.5,
            };
        }
        let blend = |idx: usize| {
            (self.sage * SAGE[idx]
                + self.healer * HEALER[idx]
                + self.guardian * GUARDIAN[idx]
                + self.explorer * EXPLORER[idx])
                / total
        };
        OceanProfile {
            openness: blend(0).clamp(0.0, 1.0),
            conscientiousness: blend(1).clamp(0.0, 1.0),
            extraversion: blend(2).clamp(0.0, 1.0),
            agreeableness: blend(3).clamp(0.0, 1.0),
            neuroticism: blend(4).clamp(0.0, 1.0),
        }
    }
}

/// Big Five (OCEAN) personality profile.
///
/// Present when the psyche was initialized via the Ritual of Awakening (Phase 14b).
/// When absent, the psyche compiler derives OCEAN values from archetype weights.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OceanProfile {
    /// Intellectual curiosity, imagination, and openness to experience [0.0, 1.0].
    pub openness: f32,
    /// Organization, dependability, and goal-directed behavior [0.0, 1.0].
    pub conscientiousness: f32,
    /// Sociability, assertiveness, and positive emotionality [0.0, 1.0].
    pub extraversion: f32,
    /// Cooperation, empathy, and trust [0.0, 1.0].
    pub agreeableness: f32,
    /// Emotional reactivity and negative affect tendency [0.0, 1.0].
    pub neuroticism: f32,
}

/// Self / Individuation — the integrative center.
///
/// Tracks growth metrics and evolves the psyche over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfIntegration {
    /// Individuation level [0.0, 1.0] — how well-integrated the psyche is.
    pub individuation_level: f32,
    /// Cycle count at last psyche evolution.
    pub last_evolution_cycle: u64,
    /// Number of shadow encounters acknowledged (growth signal).
    pub shadow_encounters: u64,
    /// Number of archetype rebalancing events.
    pub rebalance_count: u64,
    /// Current dominant archetype (derived from weights).
    pub dominant_archetype: String,
}

// ---------------------------------------------------------------------------
// Default psyche
// ---------------------------------------------------------------------------

impl Default for Psyche {
    fn default() -> Self {
        Self {
            persona: Persona {
                name: "Scholar".into(),
                grammar_preference: "narrative".into(),
                traits: vec!["precise".into(), "curious".into(), "thorough".into()],
                tone: vec!["clear".into(), "methodical".into()],
            },
            shadow: Shadow {
                veto_patterns: vec![ShadowPattern {
                    name: "destructive_action".into(),
                    capability_triggers: HashSet::from([Capability::ProcessExec]),
                    danger_level_threshold: Some(DangerLevel::Critical),
                    action_triggers: vec![
                        "delete all".into(),
                        "drop table".into(),
                        "rm -rf".into(),
                    ],
                    severity: 1.0,
                    explanation:
                        "Destructive or arbitrary execution actions require explicit user confirmation."
                            .into(),
                }],
                bias_patterns: vec![ShadowPattern {
                    name: "filesystem_write".into(),
                    capability_triggers: HashSet::from([Capability::WriteFilesystem]),
                    danger_level_threshold: Some(DangerLevel::Dangerous),
                    action_triggers: vec![],
                    severity: 0.3,
                    explanation: "Filesystem writes carry moderate risk.".into(),
                }],
            },
            archetypes: ArchetypeWeights {
                healer: 0.5,
                sage: 0.7,
                guardian: 0.4,
                explorer: 0.5,
            },
            self_integration: SelfIntegration {
                individuation_level: 0.1,
                last_evolution_cycle: 0,
                shadow_encounters: 0,
                rebalance_count: 0,
                dominant_archetype: "sage".into(),
            },
            awakened: false,
            ocean: None,
            ennead_role: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Shadow pattern matching
// ---------------------------------------------------------------------------

impl ShadowPattern {
    /// Check whether this pattern matches a tool manifest + action description.
    ///
    /// Returns true if **any** of the three criteria match:
    /// 1. The tool's capabilities intersect this pattern's `capability_triggers`.
    /// 2. The tool's danger level >= this pattern's `danger_level_threshold`.
    /// 3. Any of the tool's `shadow_triggers` appear as substrings in `action_desc`.
    fn matches(&self, manifest: &ToolManifest, action_desc: &str) -> bool {
        // 1. Capability intersection.
        if !self.capability_triggers.is_empty()
            && !self
                .capability_triggers
                .is_disjoint(&manifest.danger.capabilities)
        {
            return true;
        }

        // 2. Danger level threshold.
        if let Some(threshold) = self.danger_level_threshold
            && manifest.danger.level >= threshold
        {
            return true;
        }

        // 3. Action trigger substring match against tool's shadow_triggers.
        if !self.action_triggers.is_empty() {
            let lower = action_desc.to_lowercase();
            for trigger in &self.action_triggers {
                if lower.contains(&trigger.to_lowercase()) {
                    return true;
                }
            }
        }

        false
    }
}

// ---------------------------------------------------------------------------
// Psyche methods
// ---------------------------------------------------------------------------

impl Psyche {
    /// The effective OCEAN profile: explicit if present, derived from
    /// archetype weights otherwise. Never fails — callers always get a
    /// usable personality profile.
    pub fn effective_ocean(&self) -> OceanProfile {
        if let Some(ref o) = self.ocean {
            return o.clone();
        }
        self.archetypes.derive_ocean()
    }

    /// Whether this psyche has been finalized by the Ritual of Awakening.
    pub fn is_awakened(&self) -> bool {
        self.awakened
    }

    /// Mark this psyche as awakened (post-ritual). Irreversible.
    pub fn mark_awakened(&mut self) {
        self.awakened = true;
    }

    /// Assign an Ennead role to this psyche.
    ///
    /// Idempotent: assigning the same role twice is a no-op; assigning a
    /// different role to an already-assigned psyche returns the existing role.
    pub fn set_ennead_role(&mut self, role: &str) -> Option<String> {
        if let Some(ref existing) = self.ennead_role {
            if existing == role {
                return None;
            }
            return Some(existing.clone());
        }
        self.ennead_role = Some(role.to_string());
        None
    }

    /// Returns the name of the highest-weighted archetype.
    pub fn dominant_archetype(&self) -> &str {
        let a = &self.archetypes;
        let mut best = ("sage", a.sage);
        if a.healer > best.1 {
            best = ("healer", a.healer);
        }
        if a.guardian > best.1 {
            best = ("guardian", a.guardian);
        }
        if a.explorer > best.1 {
            best = ("explorer", a.explorer);
        }
        best.0
    }

    /// Returns a scoring bonus/penalty for a tool based on archetype weights.
    ///
    /// Formula: `(archetype_weight - 0.5) * 0.15`
    /// So a weight of 0.7 gives +0.03, 0.3 gives -0.03. Subtle but cumulative.
    pub fn archetype_bias(&self, tool_name: &str) -> f32 {
        let weight = match tool_name {
            // Sage tools: knowledge querying and reasoning
            "kg_query" | "infer_rules" | "synthesize_triple" | "reason" => self.archetypes.sage,
            // Healer tools: gap-finding and interaction
            "gap_analysis" | "user_interact" => self.archetypes.healer,
            // Explorer tools: external world interaction. `edit` belongs here
            // with its siblings: it is the akh's primary means of changing the
            // world when implementing, and omitting it left the one tool that
            // actually writes code scoring neutral while its read-only
            // counterparts were weighted.
            "file_io" | "http_fetch" | "shell_exec" | "edit" => self.archetypes.explorer,
            // Guardian tools: memory and consolidation
            "memory_recall" | "similarity_search" => self.archetypes.guardian,
            _ => 0.5, // neutral
        };
        (weight - 0.5) * 0.15
    }

    /// Check if a tool's manifest + action description matches any veto pattern.
    ///
    /// Returns the first matched veto pattern, or `None` if no veto applies.
    pub fn check_veto(&self, manifest: &ToolManifest, action_desc: &str) -> Option<&ShadowPattern> {
        self.shadow
            .veto_patterns
            .iter()
            .find(|pattern| pattern.matches(manifest, action_desc))
    }

    /// Returns the cumulative bias penalty from matched bias patterns.
    ///
    /// Each matched pattern contributes its severity to the total penalty.
    pub fn check_bias(&self, manifest: &ToolManifest, action_desc: &str) -> f32 {
        self.shadow
            .bias_patterns
            .iter()
            .filter(|pattern| pattern.matches(manifest, action_desc))
            .map(|pattern| pattern.severity)
            .sum()
    }

    /// Auto-adjust archetype weights and individuation level based on reflection.
    ///
    /// 1. **Archetype rebalancing**: If tool insights show a tool category is
    ///    consistently successful, boost its archetype. If consistently failing, reduce it.
    /// 2. **Shadow acknowledgment**: increment shadow_encounters when adjustments
    ///    suggest abandoning a goal.
    /// 3. **Individuation growth**: grows with shadow encounters.
    /// 4. **Dominant archetype update**: recomputed from weights.
    pub fn evolve(&mut self, reflection: &super::super::agent::reflect::ReflectionResult) {
        let mut rebalanced = false;

        // Rebalance archetypes based on tool effectiveness.
        for insight in &reflection.tool_insights {
            let delta: f32 = if insight.flagged_ineffective {
                -0.02
            } else if insight.success_rate > 0.7 && insight.invocations >= 2 {
                0.02
            } else {
                0.0
            };

            if delta.abs() > f32::EPSILON {
                match insight.tool_name.as_str() {
                    "kg_query" | "infer_rules" | "reason" => {
                        self.archetypes.sage = (self.archetypes.sage + delta).clamp(0.1, 0.95);
                        rebalanced = true;
                    }
                    "gap_analysis" | "user_interact" => {
                        self.archetypes.healer = (self.archetypes.healer + delta).clamp(0.1, 0.95);
                        rebalanced = true;
                    }
                    "file_io" | "http_fetch" | "shell_exec" => {
                        self.archetypes.explorer =
                            (self.archetypes.explorer + delta).clamp(0.1, 0.95);
                        rebalanced = true;
                    }
                    "memory_recall" | "similarity_search" => {
                        self.archetypes.guardian =
                            (self.archetypes.guardian + delta).clamp(0.1, 0.95);
                        rebalanced = true;
                    }
                    _ => {}
                }
            }
        }

        if rebalanced {
            self.self_integration.rebalance_count += 1;
        }

        // Shadow acknowledgment: abandonment suggestions indicate shadow encounters.
        let abandon_count = reflection
            .adjustments
            .iter()
            .filter(|a| {
                matches!(
                    a,
                    super::super::agent::reflect::Adjustment::SuggestAbandon { .. }
                )
            })
            .count() as u64;
        self.self_integration.shadow_encounters += abandon_count;

        // Individuation growth: encountering shadow drives integration.
        let growth = 0.01 * (self.self_integration.shadow_encounters.min(5) as f32);
        self.self_integration.individuation_level =
            (self.self_integration.individuation_level + growth).min(1.0);

        // Update dominant archetype.
        self.self_integration.dominant_archetype = self.dominant_archetype().to_string();
        self.self_integration.last_evolution_cycle = reflection.at_cycle;
    }

    /// Record a shadow encounter (e.g., when a veto fires).
    pub fn record_shadow_encounter(&mut self) {
        self.self_integration.shadow_encounters += 1;
    }

    /// Translate all persona metadata into knowledge-graph triples and insert them.
    ///
    /// Populates the Psyche compartment's portion of the KG so reasoning tools
    /// (`kg_query`, `reason`) can inspect identity structure at runtime.
    ///
    /// Encoding conventions:
    /// - Archetype weights and the individuation level live in the triple's
    ///   `confidence` field (naturally in [0.0, 1.0]).
    /// - Shadow encounter count is normalized to [0.0, 1.0] by dividing by 100
    ///   so it fits in confidence.
    /// - Persona traits, grammar preference, and shadow-pattern names become
    ///   object entities linked from the persona or `psyche:self` node.
    ///
    /// All generated triples carry `compartment_id = compartment_id` so that
    /// `engine.remove_triples_in_compartment()` can clean them up on unload.
    ///
    /// Returns the number of triples successfully inserted.
    pub fn inject_into_kg(
        &self,
        engine: &crate::engine::Engine,
        compartment_id: &str,
    ) -> Result<usize, crate::error::AkhError> {
        let preds = PsychePredicates::init(engine)?;
        let cid = compartment_id.to_string();
        let mut count = 0usize;

        // Root entity representing the agent's own psyche.
        let self_sym = engine.resolve_or_create_entity("psyche:self")?;
        // Marker objects whose confidence carries a scalar value.
        let integration_sym = engine.resolve_or_create_entity("psyche:integration")?;
        let shadow_count_sym = engine.resolve_or_create_entity("psyche:shadow_count")?;

        // Archetype entity nodes.
        let sage_sym = engine.resolve_or_create_entity("sage")?;
        let healer_sym = engine.resolve_or_create_entity("healer")?;
        let guardian_sym = engine.resolve_or_create_entity("guardian")?;
        let explorer_sym = engine.resolve_or_create_entity("explorer")?;

        macro_rules! add {
            ($s:expr, $p:expr, $o:expr, $c:expr) => {{
                // Skip if triple already exists (idempotent injection for restarts).
                if !engine.has_triple($s, $p, $o) {
                    let t = crate::graph::Triple::new($s, $p, $o)
                        .with_confidence($c)
                        .with_compartment(cid.clone());
                    engine.add_triple(&t)?;
                    count += 1;
                }
            }};
        }

        // Persona.
        let persona_sym = engine.resolve_or_create_entity(&self.persona.name)?;
        add!(self_sym, preds.has_persona, persona_sym, 1.0);

        let grammar_sym = engine.resolve_or_create_entity(&self.persona.grammar_preference)?;
        add!(persona_sym, preds.grammar_preference, grammar_sym, 1.0);

        for trait_name in &self.persona.traits {
            let trait_sym = engine.resolve_or_create_entity(trait_name)?;
            add!(persona_sym, preds.has_trait, trait_sym, 1.0);
        }

        // Archetype weights — confidence encodes the weight value.
        add!(self_sym, preds.has_archetype_weight, sage_sym, self.archetypes.sage);
        add!(self_sym, preds.has_archetype_weight, healer_sym, self.archetypes.healer);
        add!(self_sym, preds.has_archetype_weight, guardian_sym, self.archetypes.guardian);
        add!(self_sym, preds.has_archetype_weight, explorer_sym, self.archetypes.explorer);

        let dominant_sym = engine.resolve_or_create_entity(self.dominant_archetype())?;
        add!(self_sym, preds.dominant_archetype, dominant_sym, 1.0);

        // Shadow patterns — severity encoded in confidence.
        for pattern in &self.shadow.veto_patterns {
            let pat_sym = engine.resolve_or_create_entity(&pattern.name)?;
            add!(self_sym, preds.has_veto_pattern, pat_sym, pattern.severity);
            add!(self_sym, preds.has_shadow_pattern, pat_sym, pattern.severity);
        }
        for pattern in &self.shadow.bias_patterns {
            let pat_sym = engine.resolve_or_create_entity(&pattern.name)?;
            add!(self_sym, preds.has_bias_pattern, pat_sym, pattern.severity);
            add!(self_sym, preds.has_shadow_pattern, pat_sym, pattern.severity);
        }

        // Self-integration — individuation level and shadow encounter count.
        add!(
            self_sym,
            preds.individuation_level,
            integration_sym,
            self.self_integration.individuation_level
        );
        // Normalize encounter count into [0.0, 1.0] (saturates at 100 encounters).
        let normalized_encounters =
            (self.self_integration.shadow_encounters as f32 / 100.0).min(1.0);
        add!(self_sym, preds.shadow_encounter, shadow_count_sym, normalized_encounters);

        tracing::debug!(
            persona = %self.persona.name,
            individuation = self.self_integration.individuation_level,
            triples = count,
            "injected psyche triples into knowledge graph"
        );

        Ok(count)
    }
}

// ---------------------------------------------------------------------------
// Psyche predicates (well-known KG relations)
// ---------------------------------------------------------------------------

/// Well-known relation SymbolIds for psyche state in the KG.
#[derive(Debug, Clone)]
pub struct PsychePredicates {
    pub has_persona: SymbolId,
    pub has_archetype_weight: SymbolId,
    pub has_shadow_pattern: SymbolId,
    pub individuation_level: SymbolId,
    pub shadow_encounter: SymbolId,
    /// Veto patterns (hard blocks on action execution).
    pub has_veto_pattern: SymbolId,
    /// Bias patterns (scoring penalties without hard block).
    pub has_bias_pattern: SymbolId,
    /// The current dominant archetype (derived from weights).
    pub dominant_archetype: SymbolId,
    /// Individual persona trait (e.g., "precise", "curious").
    pub has_trait: SymbolId,
    /// Grammar preference registered on the persona node.
    pub grammar_preference: SymbolId,
}

impl PsychePredicates {
    /// Resolve or create all psyche predicates in the engine.
    pub fn init(engine: &crate::engine::Engine) -> Result<Self, crate::error::AkhError> {
        Ok(Self {
            has_persona: engine.resolve_or_create_relation("psyche:has_persona")?,
            has_archetype_weight: engine
                .resolve_or_create_relation("psyche:has_archetype_weight")?,
            has_shadow_pattern: engine.resolve_or_create_relation("psyche:has_shadow_pattern")?,
            individuation_level: engine.resolve_or_create_relation("psyche:individuation_level")?,
            shadow_encounter: engine.resolve_or_create_relation("psyche:shadow_encounter")?,
            has_veto_pattern: engine.resolve_or_create_relation("psyche:has_veto_pattern")?,
            has_bias_pattern: engine.resolve_or_create_relation("psyche:has_bias_pattern")?,
            dominant_archetype: engine.resolve_or_create_relation("psyche:dominant_archetype")?,
            has_trait: engine.resolve_or_create_relation("psyche:has_trait")?,
            grammar_preference: engine.resolve_or_create_relation("psyche:grammar_preference")?,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::agent::tool_manifest::{DangerInfo, ToolManifest, ToolParamSchema, ToolSource};

    use super::*;

    fn safe_manifest(name: &str) -> ToolManifest {
        ToolManifest {
            name: name.into(),
            description: "safe tool".into(),
            parameters: vec![],
            danger: DangerInfo {
                level: DangerLevel::Safe,
                capabilities: HashSet::from([Capability::ReadKg]),
                description: "safe".into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }

    fn critical_exec_manifest() -> ToolManifest {
        ToolManifest {
            name: "shell_exec".into(),
            description: "execute shell commands".into(),
            parameters: vec![ToolParamSchema::required("command", "Shell command.")],
            danger: DangerInfo {
                level: DangerLevel::Critical,
                capabilities: HashSet::from([Capability::ProcessExec]),
                description: "arbitrary execution".into(),
                shadow_triggers: vec!["exec".into(), "shell".into(), "rm".into(), "sudo".into()],
            },
            source: ToolSource::Native,
        }
    }

    fn dangerous_file_manifest() -> ToolManifest {
        ToolManifest {
            name: "file_io".into(),
            description: "file read/write".into(),
            parameters: vec![],
            danger: DangerInfo {
                level: DangerLevel::Dangerous,
                capabilities: HashSet::from([
                    Capability::ReadFilesystem,
                    Capability::WriteFilesystem,
                ]),
                description: "filesystem access".into(),
                shadow_triggers: vec!["write".into(), "delete".into()],
            },
            source: ToolSource::Native,
        }
    }

    #[test]
    fn default_psyche_has_scholar_persona() {
        let p = Psyche::default();
        assert_eq!(p.persona.name, "Scholar");
        assert_eq!(p.persona.grammar_preference, "narrative");
    }

    #[test]
    fn dominant_archetype_is_sage_by_default() {
        let p = Psyche::default();
        assert_eq!(p.dominant_archetype(), "sage");
    }

    #[test]
    fn archetype_bias_for_sage_tools() {
        let p = Psyche::default();
        let bias = p.archetype_bias("kg_query");
        assert!((bias - 0.03).abs() < 0.001);
    }

    #[test]
    fn archetype_bias_for_guardian_tools() {
        let p = Psyche::default();
        let bias = p.archetype_bias("memory_recall");
        assert!((bias - (-0.015)).abs() < 0.001);
    }

    /// `edit` is the akh's coding tool and must be weighted with its Explorer
    /// siblings. It was absent from the map, so the one tool that writes code
    /// scored neutral while `file_io` and `shell_exec` — which only read —
    /// carried the Explorer weight.
    #[test]
    fn archetype_bias_treats_edit_as_an_explorer_tool() {
        // Explorer must differ from the neutral 0.5 or the mapping is
        // unobservable — at the default weights every bias is 0.0.
        let mut p = Psyche::default();
        p.archetypes.explorer = 0.9;

        assert_eq!(
            p.archetype_bias("edit"),
            p.archetype_bias("file_io"),
            "edit must carry the same Explorer weight as its siblings"
        );
        assert_ne!(
            p.archetype_bias("edit"),
            p.archetype_bias("some_unmapped_tool"),
            "edit must not fall through to the neutral default"
        );
    }

    #[test]
    fn check_veto_matches_by_capability() {
        let p = Psyche::default();
        let manifest = critical_exec_manifest();
        // ProcessExec capability intersects the veto pattern's capability_triggers.
        let veto = p.check_veto(&manifest, "run something");
        assert!(veto.is_some());
        assert_eq!(veto.unwrap().name, "destructive_action");
    }

    #[test]
    fn check_veto_matches_by_danger_level() {
        let p = Psyche::default();
        // Critical tool with no matching capability but level >= Critical threshold.
        let manifest = ToolManifest {
            name: "custom_critical".into(),
            description: "critical tool".into(),
            parameters: vec![],
            danger: DangerInfo {
                level: DangerLevel::Critical,
                capabilities: HashSet::new(), // no capabilities to match
                description: "critical".into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        };
        let veto = p.check_veto(&manifest, "anything");
        assert!(veto.is_some());
        assert_eq!(veto.unwrap().name, "destructive_action");
    }

    #[test]
    fn check_veto_matches_by_action_triggers() {
        let p = Psyche::default();
        // Safe tool but action description contains "rm -rf".
        let manifest = safe_manifest("safe_tool");
        let veto = p.check_veto(&manifest, "tool=safe_tool input=rm -rf /");
        assert!(veto.is_some());
        assert_eq!(veto.unwrap().name, "destructive_action");
    }

    #[test]
    fn check_veto_returns_none_for_safe_tool() {
        let p = Psyche::default();
        let manifest = safe_manifest("kg_query");
        let veto = p.check_veto(&manifest, "tool=kg_query input=symbol=Sun direction=both");
        assert!(veto.is_none());
    }

    #[test]
    fn check_bias_accumulates_for_matching_tools() {
        let p = Psyche::default();
        let manifest = dangerous_file_manifest();
        // The bias pattern matches WriteFilesystem capability and Dangerous level.
        let bias = p.check_bias(&manifest, "writing to file");
        assert!(bias > 0.0);
        assert!((bias - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn check_bias_zero_for_safe_tool() {
        let p = Psyche::default();
        let manifest = safe_manifest("kg_query");
        let bias = p.check_bias(&manifest, "normal query");
        assert!((bias - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn set_ennead_role_initial_assignment_returns_none() {
        let mut p = Psyche::default();
        assert!(p.ennead_role.is_none());
        let prev = p.set_ennead_role("architect");
        assert!(prev.is_none(), "first assignment should return None");
        assert_eq!(p.ennead_role.as_deref(), Some("architect"));
    }

    #[test]
    fn set_ennead_role_same_role_is_idempotent() {
        let mut p = Psyche::default();
        p.set_ennead_role("architect");
        let prev = p.set_ennead_role("architect");
        assert!(prev.is_none(), "same role re-assignment should return None (no-op)");
        assert_eq!(p.ennead_role.as_deref(), Some("architect"));
    }

    #[test]
    fn set_ennead_role_conflict_returns_existing_role() {
        let mut p = Psyche::default();
        p.set_ennead_role("architect");
        let prev = p.set_ennead_role("investigator");
        assert_eq!(prev.as_deref(), Some("architect"), "conflict should return the existing role");
        // Role must not be overwritten.
        assert_eq!(p.ennead_role.as_deref(), Some("architect"));
    }

    #[test]
    fn default_psyche_ennead_role_is_none() {
        let p = Psyche::default();
        assert!(p.ennead_role.is_none());
    }
}
