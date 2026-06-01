//! Cross-skill inference: bridge rules for overlapping skill concepts.
//!
//! Phase 31g — when two skills are co-active and share overlapping concepts
//! (detected via neural prototype similarity), bridge rules are generated to
//! disambiguate polysemy or integrate complementary knowledge.
//!
//! Bridge rules are egg rewrite-rule strings tagged with multi-skill
//! provenance so reasoning chains that cross skill boundaries record which
//! skills contributed.

use serde::{Deserialize, Serialize};

use crate::symbol::SymbolId;

/// A bridge rule connecting concepts from two co-active skills.
///
/// Bridge rules are expressed in egg's rewrite syntax (same format as the
/// `rules.txt` file) and are generated when overlapping concepts are detected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeRule {
    /// Name of the rule (unique within the manager).
    pub name: String,
    /// Egg rewrite LHS pattern (e.g. `"(is_a ?x star)"`).
    pub lhs: String,
    /// Egg rewrite RHS pattern (e.g. `"(ambiguous ?x star)"`).
    pub rhs: String,
    /// Skill IDs whose overlap motivated this rule.
    pub source_skills: [String; 2],
    /// Human-readable explanation of what this bridge handles.
    pub rationale: String,
}

/// Provenance tag for inferences that used bridge rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeProvenance {
    /// Skills whose bridge rule was applied.
    pub source_skills: Vec<String>,
    /// Rule name that fired.
    pub rule_name: String,
}

/// Detected overlap between two co-active skills.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillOverlap {
    /// First skill ID.
    pub skill_a: String,
    /// Second skill ID.
    pub skill_b: String,
    /// Prototype similarity score (0.0–1.0).
    pub similarity: f32,
    /// Symbol IDs present in both skills' triple sets.
    pub shared_symbols: Vec<SymbolId>,
}

/// Threshold above which two skill prototypes are considered overlapping.
pub const OVERLAP_THRESHOLD: f32 = 0.55;

/// Detect overlaps between all pairs of active skills.
///
/// `skill_ids` is the list of currently-active skill IDs.
/// `prototype_similarity` is a callback that returns the similarity between
/// two skills' prototype vectors (provided by the caller using Phase 31b data).
///
/// Returns only pairs where similarity exceeds `OVERLAP_THRESHOLD`.
pub fn detect_overlaps<F>(skill_ids: &[String], mut prototype_similarity: F) -> Vec<SkillOverlap>
where
    F: FnMut(&str, &str) -> Option<(f32, Vec<SymbolId>)>,
{
    let mut overlaps = Vec::new();
    for (i, a) in skill_ids.iter().enumerate() {
        for b in skill_ids.iter().skip(i + 1) {
            if let Some((sim, shared)) = prototype_similarity(a, b) {
                if sim > OVERLAP_THRESHOLD {
                    overlaps.push(SkillOverlap {
                        skill_a: a.clone(),
                        skill_b: b.clone(),
                        similarity: sim,
                        shared_symbols: shared,
                    });
                }
            }
        }
    }
    overlaps
}

/// Generate a disambiguation bridge rule for a shared polysemous symbol.
///
/// When `symbol_label` appears in both skill A and skill B with conflicting
/// predicate contexts, the bridge rule flags it as ambiguous:
/// `(is_a ?x <label>) => (ambiguous ?x <label>)`.
pub fn generate_disambiguation_rule(
    overlap: &SkillOverlap,
    symbol_label: &str,
    rule_index: usize,
) -> BridgeRule {
    BridgeRule {
        name: format!(
            "bridge-{}-{}-disambig-{}",
            overlap.skill_a, overlap.skill_b, rule_index
        ),
        lhs: format!("(is_a ?x {symbol_label})"),
        rhs: format!("(ambiguous ?x {symbol_label})"),
        source_skills: [overlap.skill_a.clone(), overlap.skill_b.clone()],
        rationale: format!(
            "Symbol '{symbol_label}' appears in both '{}' and '{}' with potentially \
             different meanings; flagged as ambiguous.",
            overlap.skill_a, overlap.skill_b
        ),
    }
}

/// Generate an integration bridge rule for a complementary shared symbol.
///
/// When `symbol_label` represents the same concept in both skills and the
/// skills' knowledge is complementary, the rule propagates facts:
/// `(concept_in_a ?x) => (concept_in_b ?x)`.
pub fn generate_integration_rule(
    overlap: &SkillOverlap,
    symbol_label: &str,
    predicate_a: &str,
    predicate_b: &str,
    rule_index: usize,
) -> BridgeRule {
    BridgeRule {
        name: format!(
            "bridge-{}-{}-integrate-{}",
            overlap.skill_a, overlap.skill_b, rule_index
        ),
        lhs: format!("({predicate_a} ?x {symbol_label})"),
        rhs: format!("({predicate_b} ?x {symbol_label})"),
        source_skills: [overlap.skill_a.clone(), overlap.skill_b.clone()],
        rationale: format!(
            "Bridges '{predicate_a}' from '{}' with '{predicate_b}' from '{}' \
             via shared concept '{symbol_label}'.",
            overlap.skill_a, overlap.skill_b
        ),
    }
}

/// Summarise a set of bridge rules as a string block suitable for inclusion
/// in a `rules.txt` file.
pub fn rules_to_text(rules: &[BridgeRule]) -> String {
    let mut out = String::new();
    for rule in rules {
        out.push_str(&format!("# {}\n", rule.rationale));
        out.push_str(&format!("{} => {}\n", rule.lhs, rule.rhs));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::SymbolId;

    fn sym(n: u64) -> SymbolId {
        SymbolId::new(n).unwrap()
    }

    fn make_overlap(a: &str, b: &str, sim: f32, shared_n: usize) -> SkillOverlap {
        SkillOverlap {
            skill_a: a.to_string(),
            skill_b: b.to_string(),
            similarity: sim,
            shared_symbols: (1..=shared_n as u64).map(sym).collect(),
        }
    }

    #[test]
    fn detect_overlaps_above_threshold() {
        let skills = vec![
            "astronomy".to_string(),
            "mythology".to_string(),
            "physics".to_string(),
        ];

        let mut sim_map = std::collections::HashMap::new();
        sim_map.insert(("astronomy", "mythology"), (0.65f32, vec![sym(42)]));
        sim_map.insert(("astronomy", "physics"), (0.30, vec![]));
        sim_map.insert(("mythology", "physics"), (0.20, vec![]));

        let overlaps = detect_overlaps(&skills, |a, b| {
            sim_map
                .get(&(a, b))
                .or_else(|| sim_map.get(&(b, a)))
                .cloned()
        });

        assert_eq!(overlaps.len(), 1);
        assert_eq!(overlaps[0].skill_a, "astronomy");
        assert_eq!(overlaps[0].skill_b, "mythology");
    }

    #[test]
    fn disambiguation_rule_format() {
        let overlap = make_overlap("astronomy", "mythology", 0.65, 1);
        let rule = generate_disambiguation_rule(&overlap, "star", 0);
        assert_eq!(rule.lhs, "(is_a ?x star)");
        assert_eq!(rule.rhs, "(ambiguous ?x star)");
        assert_eq!(rule.source_skills[0], "astronomy");
        assert_eq!(rule.source_skills[1], "mythology");
    }

    #[test]
    fn integration_rule_format() {
        let overlap = make_overlap("physics", "sports", 0.60, 2);
        let rule = generate_integration_rule(&overlap, "velocity", "has_velocity", "speed_of", 0);
        assert!(rule.lhs.contains("velocity"));
        assert!(rule.rhs.contains("velocity"));
    }

    #[test]
    fn rules_to_text_produces_valid_lines() {
        let overlap = make_overlap("astronomy", "mythology", 0.65, 1);
        let rule = generate_disambiguation_rule(&overlap, "star", 0);
        let text = rules_to_text(&[rule]);
        assert!(text.contains("# "));
        assert!(text.contains("=>"));
    }

    #[test]
    fn no_overlap_below_threshold() {
        let skills = vec!["a".to_string(), "b".to_string()];
        let overlaps = detect_overlaps(&skills, |_, _| Some((0.30, vec![])));
        assert!(overlaps.is_empty());
    }
}
