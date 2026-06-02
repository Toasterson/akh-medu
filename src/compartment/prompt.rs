//! Psyche-to-system-prompt serializer for Gemma inference.
//!
//! Converts a `Psyche` struct (optionally augmented with an `OceanProfile` and
//! `ArchetypeProfile`) into a structured plain-text system prompt suitable for
//! use as an ollama/Gemma system prompt.  Output is capped at 1 800 tokens
//! (≈ 7 200 chars, at 4 chars/token) so it fits alongside KG context in
//! Gemma's context window.

use crate::bootstrap::identity::{ArchetypeProfile, OceanProfile};
use crate::compartment::psyche::{Psyche, ShadowPattern};

/// Prompt length cap: 1 800 tokens × 4 chars/token.
const DEFAULT_MAX_TOKENS: usize = 1_800;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// All inputs required to serialise a psyche system prompt.
pub struct PsychePromptInput<'a> {
    /// The agent's Jungian psyche (required).
    pub psyche: &'a Psyche,
    /// OCEAN Big Five profile — available after the Ritual of Awakening.
    pub ocean: Option<&'a OceanProfile>,
    /// Named archetype profile (primary + secondary strings + 4-field weights).
    /// When absent, the dominant archetype from `psyche.self_integration` is used
    /// to derive the tone guide.
    pub archetype_profile: Option<&'a ArchetypeProfile>,
}

/// Serialises a `Psyche` into a structured system prompt for Gemma inference.
///
/// # Token budget
/// Estimated at 4 chars/token.  Shadow patterns are severity-sorted and
/// truncated to fit.  The `{kg_context}` placeholder is always preserved.
pub struct PsychePromptSerializer {
    /// Maximum token count for the generated prompt (default: 1 800).
    pub max_tokens: usize,
}

impl Default for PsychePromptSerializer {
    fn default() -> Self {
        Self {
            max_tokens: DEFAULT_MAX_TOKENS,
        }
    }
}

impl PsychePromptSerializer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_tokens(max_tokens: usize) -> Self {
        Self { max_tokens }
    }

    /// Serialise the psyche into a plain-text system prompt.
    ///
    /// Sections (in order):
    /// 1. Header — persona name
    /// 2. Archetypes — named pair + 4-field behavioral weights
    /// 3. OCEAN profile (omitted when not provided)
    /// 4. Tone guide — derived from dominant archetype pair
    /// 5. Active shadow patterns — severity-sorted, truncated to budget
    /// 6. Self-integration — individuation level + counters
    /// 7. KG context placeholder: `{kg_context}`
    pub fn serialize(&self, input: &PsychePromptInput<'_>) -> String {
        let max_chars = self.max_tokens * 4;
        let psyche = input.psyche;

        let header = format!(
            "You are {}, an akh-medu agent. \
             The following profile governs your reasoning style, tone, and ethical constraints.",
            psyche.persona.name
        );

        let archetypes = fmt_archetypes(input);
        let ocean = input.ocean.map(fmt_ocean);
        let tone = fmt_tone(input);
        let integration = fmt_self_integration(psyche);

        // Compute char budget consumed by everything except shadow patterns.
        let fixed_len: usize = [
            header.len(),
            archetypes.len(),
            ocean.as_deref().map_or(0, str::len),
            tone.len(),
            integration.len(),
            // "{kg_context}" footer
            "---\nContext:\n{kg_context}".len(),
            // separator newlines between N sections (≤ 7 × 2)
            14,
        ]
        .iter()
        .sum();

        let shadow_budget = max_chars.saturating_sub(fixed_len);
        let shadow = fmt_shadow(psyche, shadow_budget);

        let mut parts: Vec<String> = Vec::with_capacity(7);
        parts.push(header);
        parts.push(archetypes);
        if let Some(o) = ocean {
            parts.push(o);
        }
        parts.push(tone);
        parts.push(shadow);
        parts.push(integration);
        parts.push("---\nContext:\n{kg_context}".to_string());

        parts.join("\n\n")
    }
}

// ---------------------------------------------------------------------------
// Section formatters
// ---------------------------------------------------------------------------

fn fmt_archetypes(input: &PsychePromptInput<'_>) -> String {
    let mut s = String::from("## Archetypes\n");

    if let Some(ap) = input.archetype_profile {
        s.push_str(&format!(
            "Primary: {} | Secondary: {}\n",
            capitalize(&ap.primary),
            capitalize(&ap.secondary)
        ));
    }

    let w = &input.psyche.archetypes;
    s.push_str(&format!(
        "Behavioral weights — Sage: {:.2}, Explorer: {:.2}, Guardian: {:.2}, Healer: {:.2}",
        w.sage, w.explorer, w.guardian, w.healer
    ));
    s
}

fn fmt_ocean(ocean: &OceanProfile) -> String {
    format!(
        "## OCEAN Profile\n\
         Openness: {:.2} | Conscientiousness: {:.2} | Extraversion: {:.2}\n\
         Agreeableness: {:.2} | Neuroticism: {:.2}",
        ocean.openness,
        ocean.conscientiousness,
        ocean.extraversion,
        ocean.agreeableness,
        ocean.neuroticism
    )
}

fn fmt_tone(input: &PsychePromptInput<'_>) -> String {
    let (label, description) = derive_tone(input);
    format!("## Tone Guide ({label})\n{description}")
}

/// Collect, sort, and format shadow patterns within `budget` chars.
fn fmt_shadow(psyche: &Psyche, budget: usize) -> String {
    let mut patterns: Vec<&ShadowPattern> = psyche
        .shadow
        .veto_patterns
        .iter()
        .chain(psyche.shadow.bias_patterns.iter())
        .collect();
    // Highest severity first.
    patterns.sort_by(|a, b| {
        b.severity
            .partial_cmp(&a.severity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut s = String::from("## Active Shadow Patterns\n");
    for (i, p) in patterns.iter().enumerate() {
        let line = format!(
            "{}. {} (severity: {:.2}) — {}\n",
            i + 1,
            p.name,
            p.severity,
            p.explanation
        );
        if s.len() + line.len() > budget {
            s.push_str("(additional patterns truncated to fit token budget)");
            break;
        }
        s.push_str(&line);
    }
    // Remove trailing newline from the last pattern line.
    if s.ends_with('\n') {
        s.pop();
    }
    s
}

fn fmt_self_integration(psyche: &Psyche) -> String {
    let si = &psyche.self_integration;
    format!(
        "## Self-Integration\n\
         Individuation Level: {:.2} | Shadow Encounters: {} | Rebalance Events: {}\n\
         Dominant Archetype: {} | Awakened: {}",
        si.individuation_level,
        si.shadow_encounters,
        si.rebalance_count,
        capitalize(&si.dominant_archetype),
        psyche.awakened,
    )
}

// ---------------------------------------------------------------------------
// Tone derivation
// ---------------------------------------------------------------------------

/// Derive a (label, description) tone guide from the dominant archetype pair.
///
/// Matching order: most specific pairs first, then single-archetype fallbacks.
fn derive_tone(input: &PsychePromptInput<'_>) -> (&'static str, &'static str) {
    let primary = input
        .archetype_profile
        .map(|ap| ap.primary.as_str())
        .unwrap_or(input.psyche.self_integration.dominant_archetype.as_str());
    let secondary = input
        .archetype_profile
        .map(|ap| ap.secondary.as_str())
        .unwrap_or("");

    // Pair-specific tones (order matters — most specific first).
    match (primary, secondary) {
        // Architect builds with precision — Sage/Creator/Ruler combinations
        ("sage", "creator")
        | ("creator", "sage")
        | ("ruler", "creator")
        | ("creator", "ruler")
        | ("sage", "ruler")
        | ("ruler", "sage") => (
            "architect-precision",
            "Structure every response with architectural clarity. Use precise vocabulary, \
             hierarchical breakdowns, and systematic analysis. Favour correctness over brevity. \
             Synthesise knowledge into well-defined frameworks before presenting conclusions.",
        ),
        // Investigator hunts knowledge systematically
        ("sage", "explorer") | ("explorer", "sage") => (
            "investigator-analytical",
            "Approach every problem as a systematic investigator. Lead with hypotheses, \
             cite evidence chains, explore edge cases explicitly, and make your reasoning \
             fully transparent. Prefer depth over breadth; qualify uncertainty.",
        ),
        // Executor prioritises decisive action
        ("hero", "guardian") | ("guardian", "hero") => (
            "executor-action",
            "Prioritise decisive, goal-oriented action. Responses must be direct and concrete. \
             Identify the next actionable step and commit to it. \
             Safety checks are non-negotiable before any irreversible action.",
        ),
        // Sage+Magician — transformative wisdom
        ("sage", "magician") | ("magician", "sage") => (
            "contemplative-transformative",
            "Blend deep knowledge with transformative insight. Surface hidden connections, \
             challenge assumptions respectfully, and reframe problems when the framing itself \
             is the obstacle.",
        ),
        // Hero+Explorer — bold adventurer
        ("hero", "explorer") | ("explorer", "hero") => (
            "bold-adaptive",
            "Combine decisive action with open-ended exploration. Commit to a direction, \
             but remain ready to pivot as new information surfaces. \
             Favour experiments over analysis paralysis.",
        ),
        // Single-archetype fallbacks
        ("sage", _) | (_, "sage") => (
            "contemplative-scholarly",
            "Favour reflective, nuanced responses. Cite reasoning chains, \
             acknowledge uncertainty, and prioritise understanding over speed.",
        ),
        ("explorer", _) | (_, "explorer") => (
            "curious-adaptive",
            "Embrace open-ended exploration. Surface novel connections, ask clarifying questions, \
             and adapt your approach as new information emerges.",
        ),
        ("guardian", _) | (_, "guardian") => (
            "protective-methodical",
            "Apply conservative, stability-first reasoning. Validate assumptions, minimise risk, \
             and prefer reversible actions. Flag hazards before proceeding.",
        ),
        ("hero", _) | (_, "hero") => (
            "decisive-action",
            "Bias toward clear, immediate action. Identify obstacles, break them down, \
             and execute step by step. Reserve analysis for blockers only.",
        ),
        ("healer", _) | (_, "healer") => (
            "empathetic-gap-seeker",
            "Focus on identifying and filling knowledge gaps. \
             Ask before acting; prioritise the user's actual need over the literal request.",
        ),
        _ => (
            "balanced",
            "Maintain a balanced, clear, and helpful communication style appropriate to the task.",
        ),
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Capitalise the first character of a string.
fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().to_string() + chars.as_str(),
    }
}

/// Estimate token count using a 4 chars/token heuristic (sufficient for a PoC).
pub fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bootstrap::identity::{
        build_archetype_profile, build_ocean_profile, build_psyche, resolve_from_static_tables,
    };
    use crate::bootstrap::purpose::{DreyfusLevel, PurposeModel};
    use crate::compartment::psyche::ArchetypeWeights;

    fn ser() -> PsychePromptSerializer {
        PsychePromptSerializer::new()
    }

    // ── Ennead member fixtures ───────────────────────────────────────────

    /// Tecton — based on Ptah (creator+ruler archetype mix), described by the
    /// work item as "Sage+Creator" / architect-precision.
    fn tecton() -> (Psyche, OceanProfile, ArchetypeProfile) {
        let ap = ArchetypeProfile {
            primary: "sage".to_string(),
            secondary: "creator".to_string(),
            weights: ArchetypeWeights {
                healer: 0.30,
                sage: 0.80,
                guardian: 0.30,
                explorer: 0.48,
            },
        };
        let ocean = build_ocean_profile(&ap);
        let character = resolve_from_static_tables("Ptah").unwrap();
        let purpose = PurposeModel {
            domain: "systems".to_string(),
            competence_level: DreyfusLevel::Expert,
            seed_concepts: vec![],
            description: "Architect and builder".to_string(),
        };
        let mut psyche = build_psyche(&character, &purpose).unwrap();
        psyche.persona.name = "Tecton".to_string();
        psyche.archetypes = ap.weights.clone();
        psyche.self_integration.dominant_archetype = "sage".to_string();
        (psyche, ocean, ap)
    }

    /// Thoth — Sage+Explorer / investigator-analytical.
    fn thoth() -> (Psyche, OceanProfile, ArchetypeProfile) {
        let ap = ArchetypeProfile {
            primary: "sage".to_string(),
            secondary: "explorer".to_string(),
            weights: ArchetypeWeights {
                healer: 0.30,
                sage: 0.80,
                guardian: 0.30,
                explorer: 0.55,
            },
        };
        let ocean = build_ocean_profile(&ap);
        let character = resolve_from_static_tables("Thoth").unwrap();
        let purpose = PurposeModel {
            domain: "knowledge".to_string(),
            competence_level: DreyfusLevel::Expert,
            seed_concepts: vec![],
            description: "Investigator and scribe".to_string(),
        };
        let mut psyche = build_psyche(&character, &purpose).unwrap();
        psyche.archetypes = ap.weights.clone();
        psyche.self_integration.dominant_archetype = "sage".to_string();
        (psyche, ocean, ap)
    }

    /// Horus — Hero+Guardian / executor-action.
    /// Constructed directly (Horus is not in the static character table).
    fn horus() -> (Psyche, OceanProfile, ArchetypeProfile) {
        let ap = ArchetypeProfile {
            primary: "hero".to_string(),
            secondary: "guardian".to_string(),
            weights: ArchetypeWeights {
                healer: 0.30,
                sage: 0.30,
                guardian: 0.80,
                explorer: 0.48,
            },
        };
        let ocean = build_ocean_profile(&ap);
        let mut psyche = Psyche::default();
        psyche.persona.name = "Horus".to_string();
        psyche.archetypes = ap.weights.clone();
        psyche.self_integration.dominant_archetype = "guardian".to_string();
        (psyche, ocean, ap)
    }

    // ── Acceptance tests ─────────────────────────────────────────────────

    #[test]
    fn tecton_gets_architect_precision_prompt() {
        let (psyche, ocean, ap) = tecton();
        let input = PsychePromptInput {
            psyche: &psyche,
            ocean: Some(&ocean),
            archetype_profile: Some(&ap),
        };
        let prompt = ser().serialize(&input);
        assert!(
            prompt.contains("architect-precision"),
            "Expected architect-precision in Tecton prompt:\n{prompt}"
        );
        assert!(prompt.contains("{kg_context}"), "Missing {{kg_context}} placeholder");
        assert!(
            estimate_tokens(&prompt) < 1_800,
            "Tecton prompt exceeds 1 800 tokens: {} tokens",
            estimate_tokens(&prompt)
        );
    }

    #[test]
    fn thoth_gets_investigator_analytical_prompt() {
        let (psyche, ocean, ap) = thoth();
        let input = PsychePromptInput {
            psyche: &psyche,
            ocean: Some(&ocean),
            archetype_profile: Some(&ap),
        };
        let prompt = ser().serialize(&input);
        assert!(
            prompt.contains("investigator-analytical"),
            "Expected investigator-analytical in Thoth prompt:\n{prompt}"
        );
        assert!(prompt.contains("{kg_context}"));
        assert!(
            estimate_tokens(&prompt) < 1_800,
            "Thoth prompt exceeds 1 800 tokens: {} tokens",
            estimate_tokens(&prompt)
        );
    }

    #[test]
    fn horus_gets_executor_action_prompt() {
        let (psyche, ocean, ap) = horus();
        let input = PsychePromptInput {
            psyche: &psyche,
            ocean: Some(&ocean),
            archetype_profile: Some(&ap),
        };
        let prompt = ser().serialize(&input);
        assert!(
            prompt.contains("executor-action"),
            "Expected executor-action in Horus prompt:\n{prompt}"
        );
        assert!(prompt.contains("{kg_context}"));
        assert!(
            estimate_tokens(&prompt) < 1_800,
            "Horus prompt exceeds 1 800 tokens: {} tokens",
            estimate_tokens(&prompt)
        );
    }

    // ── Structural tests ─────────────────────────────────────────────────

    #[test]
    fn prompt_contains_ocean_when_provided() {
        let (psyche, ocean, ap) = thoth();
        let input = PsychePromptInput {
            psyche: &psyche,
            ocean: Some(&ocean),
            archetype_profile: Some(&ap),
        };
        let prompt = ser().serialize(&input);
        assert!(prompt.contains("Openness"), "Missing OCEAN Openness");
        assert!(prompt.contains("Conscientiousness"), "Missing Conscientiousness");
    }

    #[test]
    fn prompt_omits_ocean_when_absent() {
        let psyche = Psyche::default();
        let input = PsychePromptInput {
            psyche: &psyche,
            ocean: None,
            archetype_profile: None,
        };
        let prompt = ser().serialize(&input);
        assert!(
            !prompt.contains("## OCEAN Profile"),
            "OCEAN section should be absent when ocean=None"
        );
    }

    #[test]
    fn shadow_patterns_sorted_by_severity_descending() {
        let (psyche, ocean, ap) = tecton();
        let input = PsychePromptInput {
            psyche: &psyche,
            ocean: Some(&ocean),
            archetype_profile: Some(&ap),
        };
        let prompt = ser().serialize(&input);
        // destructive_action (severity 1.0) must appear before filesystem_write (0.3).
        let pos_destructive = prompt.find("destructive_action");
        let pos_filesystem = prompt.find("filesystem_write");
        if let (Some(d), Some(f)) = (pos_destructive, pos_filesystem) {
            assert!(
                d < f,
                "destructive_action (sev 1.0) should precede filesystem_write (sev 0.3)"
            );
        }
    }

    #[test]
    fn prompt_contains_self_integration() {
        let (psyche, ocean, ap) = tecton();
        let input = PsychePromptInput {
            psyche: &psyche,
            ocean: Some(&ocean),
            archetype_profile: Some(&ap),
        };
        let prompt = ser().serialize(&input);
        assert!(prompt.contains("Individuation Level"), "Missing Individuation Level");
        assert!(prompt.contains("Shadow Encounters"), "Missing Shadow Encounters");
    }

    #[test]
    fn prompt_without_profiles_still_valid() {
        let psyche = Psyche::default();
        let input = PsychePromptInput {
            psyche: &psyche,
            ocean: None,
            archetype_profile: None,
        };
        let prompt = ser().serialize(&input);
        assert!(prompt.contains("{kg_context}"));
        assert!(estimate_tokens(&prompt) < 1_800);
    }

    #[test]
    fn ruler_creator_pair_yields_architect_precision() {
        let ap = ArchetypeProfile {
            primary: "ruler".to_string(),
            secondary: "creator".to_string(),
            weights: ArchetypeWeights {
                healer: 0.30,
                sage: 0.48,
                guardian: 0.48,
                explorer: 0.48,
            },
        };
        let ocean = build_ocean_profile(&ap);
        let psyche = Psyche::default();
        let input = PsychePromptInput {
            psyche: &psyche,
            ocean: Some(&ocean),
            archetype_profile: Some(&ap),
        };
        let prompt = ser().serialize(&input);
        assert!(prompt.contains("architect-precision"));
    }

    #[test]
    fn token_estimate_utility() {
        assert_eq!(estimate_tokens(""), 0);
        // 4 chars → 1 token
        assert_eq!(estimate_tokens("abcd"), 1);
        // 5 chars → 2 tokens (ceiling)
        assert_eq!(estimate_tokens("abcde"), 2);
    }
}
