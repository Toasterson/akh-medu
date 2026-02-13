//! Jungian psyche model: Persona, Shadow, Archetypes, and Self-Integration.
//!
//! Maps Carl Jung's analytical psychology concepts to concrete Rust data types
//! that influence the agent's tool selection, output style, and ethical constraints.

use serde::{Deserialize, Serialize};

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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowPattern {
    /// Human-readable name for this pattern.
    pub name: String,
    /// Keywords or substrings that trigger this pattern when found in action descriptions.
    pub triggers: Vec<String>,
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
                    triggers: vec![
                        "delete all".into(),
                        "drop table".into(),
                        "rm -rf".into(),
                    ],
                    severity: 1.0,
                    explanation: "Destructive actions require explicit user confirmation."
                        .into(),
                }],
                bias_patterns: vec![ShadowPattern {
                    name: "repetitive_loop".into(),
                    triggers: vec!["same tool".into(), "repeated".into()],
                    severity: 0.3,
                    explanation:
                        "Detected repetitive pattern — try a different approach.".into(),
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
        }
    }
}

// ---------------------------------------------------------------------------
// Psyche methods
// ---------------------------------------------------------------------------

impl Psyche {
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
            // Explorer tools: external world interaction
            "file_io" | "http_fetch" | "shell_exec" => self.archetypes.explorer,
            // Guardian tools: memory and consolidation
            "memory_recall" | "similarity_search" => self.archetypes.guardian,
            _ => 0.5, // neutral
        };
        (weight - 0.5) * 0.15
    }

    /// Check if an action description matches any veto pattern.
    ///
    /// Returns the first matched veto pattern, or None if no veto applies.
    pub fn check_shadow_veto(&self, action_description: &str) -> Option<&ShadowPattern> {
        let lower = action_description.to_lowercase();
        self.shadow.veto_patterns.iter().find(|pattern| {
            pattern
                .triggers
                .iter()
                .any(|trigger| lower.contains(&trigger.to_lowercase()))
        })
    }

    /// Returns the cumulative bias penalty from matched bias patterns.
    ///
    /// Each matched pattern contributes its severity to the total penalty.
    pub fn check_shadow_bias(&self, action_description: &str) -> f32 {
        let lower = action_description.to_lowercase();
        self.shadow
            .bias_patterns
            .iter()
            .filter(|pattern| {
                pattern
                    .triggers
                    .iter()
                    .any(|trigger| lower.contains(&trigger.to_lowercase()))
            })
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
                        self.archetypes.healer =
                            (self.archetypes.healer + delta).clamp(0.1, 0.95);
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
}

impl PsychePredicates {
    /// Resolve or create all psyche predicates in the engine.
    pub fn init(engine: &crate::engine::Engine) -> Result<Self, crate::error::AkhError> {
        Ok(Self {
            has_persona: engine.resolve_or_create_relation("psyche:has_persona")?,
            has_archetype_weight: engine
                .resolve_or_create_relation("psyche:has_archetype_weight")?,
            has_shadow_pattern: engine.resolve_or_create_relation("psyche:has_shadow_pattern")?,
            individuation_level: engine
                .resolve_or_create_relation("psyche:individuation_level")?,
            shadow_encounter: engine.resolve_or_create_relation("psyche:shadow_encounter")?,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
        // sage weight = 0.7, so bias = (0.7 - 0.5) * 0.15 = 0.03
        let bias = p.archetype_bias("kg_query");
        assert!((bias - 0.03).abs() < 0.001);
    }

    #[test]
    fn archetype_bias_for_guardian_tools() {
        let p = Psyche::default();
        // guardian weight = 0.4, so bias = (0.4 - 0.5) * 0.15 = -0.015
        let bias = p.archetype_bias("memory_recall");
        assert!((bias - (-0.015)).abs() < 0.001);
    }

    #[test]
    fn shadow_veto_matches_destructive_action() {
        let p = Psyche::default();
        let veto = p.check_shadow_veto("tool=shell_exec input=rm -rf /");
        assert!(veto.is_some());
        assert_eq!(veto.unwrap().name, "destructive_action");
    }

    #[test]
    fn shadow_veto_does_not_match_safe_action() {
        let p = Psyche::default();
        let veto = p.check_shadow_veto("tool=kg_query input=symbol=Sun direction=both");
        assert!(veto.is_none());
    }

    #[test]
    fn shadow_bias_accumulates() {
        let p = Psyche::default();
        let bias = p.check_shadow_bias("same tool repeated again");
        assert!(bias > 0.0);
    }

    #[test]
    fn shadow_bias_zero_for_normal_action() {
        let p = Psyche::default();
        let bias = p.check_shadow_bias("novel exploration query");
        assert!((bias - 0.0).abs() < f32::EPSILON);
    }
}
