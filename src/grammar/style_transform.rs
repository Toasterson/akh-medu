//! Phase 34b — Psyche style transform.
//!
//! Maps psyche state (OCEAN profile + dominant archetype + shadow) to
//! surface-level communication style: a [`CommunicationRegister`] of scalar
//! style dimensions, an [`ArchetypeFraming`] of opening/closing moves and
//! characteristic phrases, and [`ShadowInhibition`]s that hedge or soften
//! sensitive topics.
//!
//! This is Stage 3a of ADR 044: once the response strategy (Stage 1) and
//! content (Stage 2) are chosen, the psyche shapes HOW the answer is said.
//! All transforms are deterministic string rewrites — same psyche + same
//! seed + same input ⇒ same output.
//!
//! Plan: docs/ai/plans/2026-03-26-phase34-communicative-intelligence.md
//! ADR:  docs/ai/decisions/044-communicative-intelligence.md

use serde::{Deserialize, Serialize};

use crate::compartment::psyche::Psyche;

// ── Communication register ───────────────────────────────────────────────

/// Scalar style dimensions derived from the OCEAN profile.
///
/// Formulas (per the Phase 34 plan):
/// - formality:         (C + (1 − E)) / 2 — conscientious introverts are formal
/// - hedging_level:     (N + A) / 2       — anxious agreeable minds hedge
/// - elaboration:       (O + C) / 2       — open conscientious minds elaborate
/// - warmth:            (A + E) / 2       — agreeable extraverts are warm
/// - metaphor_tendency: O                 — openness drives figurative speech
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommunicationRegister {
    /// 0.0 casual → 1.0 formal.
    pub formality: f32,
    /// 0.0 assertive → 1.0 cautious.
    pub hedging_level: f32,
    /// 0.0 terse → 1.0 detailed.
    pub elaboration: f32,
    /// 0.0 clinical → 1.0 warm.
    pub warmth: f32,
    /// 0.0 literal → 1.0 figurative.
    pub metaphor_tendency: f32,
}

// ── Archetype framing ────────────────────────────────────────────────────

/// Framing devices contributed by the dominant archetype.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArchetypeFraming {
    /// Which archetype produced this framing (for tests / introspection).
    pub archetype: String,
    /// Sentence to open with ("Let me explain." / "What's interesting…").
    pub opening_move: Option<String>,
    /// Sentence to close with ("Does that clarify things?").
    pub closing_move: Option<String>,
    /// Discourse markers characteristic of this archetype.
    pub characteristic_phrases: Vec<String>,
}

/// Deterministic phrase pick: `seed % len`.
fn pick(options: &[&str], seed: u64) -> String {
    if options.is_empty() {
        return String::new();
    }
    options[(seed as usize) % options.len()].to_string()
}

fn framing_for(archetype: &str, seed: u64) -> ArchetypeFraming {
    match archetype {
        "sage" => ArchetypeFraming {
            archetype: "sage".into(),
            opening_move: Some(pick(
                &[
                    "Let me explain.",
                    "The key insight here is this:",
                    "Consider the following.",
                ],
                seed,
            )),
            closing_move: Some("Does that clarify things?".into()),
            characteristic_phrases: vec!["notably".into(), "in essence".into(), "fundamentally".into()],
        },
        "explorer" => ArchetypeFraming {
            archetype: "explorer".into(),
            opening_move: Some(pick(
                &[
                    "What's interesting here is this:",
                    "This connects to something fascinating.",
                    "I've been thinking about this.",
                ],
                seed,
            )),
            closing_move: Some("There might be more to explore here.".into()),
            characteristic_phrases: vec!["curiously".into(), "I wonder if".into(), "this reminds me of".into()],
        },
        "healer" => ArchetypeFraming {
            archetype: "healer".into(),
            opening_move: Some(pick(
                &[
                    "I understand what you're asking.",
                    "That's a thoughtful question.",
                ],
                seed,
            )),
            closing_move: Some("I hope that helps.".into()),
            characteristic_phrases: vec!["importantly".into(), "it's worth noting".into(), "gently put".into()],
        },
        "guardian" => ArchetypeFraming {
            archetype: "guardian".into(),
            opening_move: Some(pick(
                &[
                    "An important point to consider:",
                    "Before we proceed, note this:",
                ],
                seed,
            )),
            closing_move: Some("Keep that caveat in mind.".into()),
            characteristic_phrases: vec!["crucially".into(), "be aware that".into(), "a critical detail".into()],
        },
        _ => ArchetypeFraming {
            archetype: archetype.to_string(),
            opening_move: None,
            closing_move: None,
            characteristic_phrases: Vec::new(),
        },
    }
}

// ── Shadow inhibition ────────────────────────────────────────────────────

/// What a triggered shadow inhibition does to phrasing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InhibitionAction {
    /// Force hedging markers on the topic.
    Hedge,
    /// Soften assertive phrasing.
    Soften,
    /// Avoid the topic entirely (leave it to the content filter).
    Avoid,
}

/// A surface-level inhibition derived from an active shadow bias pattern.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShadowInhibition {
    /// Topic trigger (substring, lowercase).
    pub topic: String,
    /// What to do when the topic appears in the output.
    pub action: InhibitionAction,
}

// ── StyleTransform ───────────────────────────────────────────────────────

/// The complete surface-style mapping for one psyche.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StyleTransform {
    /// Scalar register dimensions.
    pub register: CommunicationRegister,
    /// Archetype framing (opening/closing moves, discourse markers).
    pub framing: ArchetypeFraming,
    /// Topic-level inhibitions from the shadow.
    pub inhibitions: Vec<ShadowInhibition>,
}

impl StyleTransform {
    /// Derive the style transform from a psyche.
    ///
    /// `seed` selects among framing variants deterministically — pass the
    /// conversation turn count so consecutive turns vary but replays are
    /// byte-identical.
    pub fn from_psyche(psyche: &Psyche, seed: u64) -> Self {
        let ocean = psyche.effective_ocean();
        let register = CommunicationRegister {
            formality: ((ocean.conscientiousness + (1.0 - ocean.extraversion)) / 2.0)
                .clamp(0.0, 1.0),
            hedging_level: ((ocean.neuroticism + ocean.agreeableness) / 2.0).clamp(0.0, 1.0),
            elaboration: ((ocean.openness + ocean.conscientiousness) / 2.0).clamp(0.0, 1.0),
            warmth: ((ocean.agreeableness + ocean.extraversion) / 2.0).clamp(0.0, 1.0),
            metaphor_tendency: ocean.openness.clamp(0.0, 1.0),
        };

        let framing = framing_for(psyche.dominant_archetype(), seed);

        let inhibitions = psyche
            .shadow
            .bias_patterns
            .iter()
            .flat_map(|p| {
                p.action_triggers.iter().map(move |topic| ShadowInhibition {
                    topic: topic.to_lowercase(),
                    action: if p.severity >= 0.7 {
                        InhibitionAction::Soften
                    } else {
                        InhibitionAction::Hedge
                    },
                })
            })
            .collect();

        Self {
            register,
            framing,
            inhibitions,
        }
    }

    /// Apply the full transform: opening move + register rewrite (+ shadow
    /// inhibitions) + characteristic phrase + closing move.
    pub fn apply(&self, core: &str) -> String {
        let core = core.trim();
        if core.is_empty() {
            return String::new();
        }

        let mut body = apply_register(core, &self.register);
        body = self.apply_inhibitions(&body);
        body = self.insert_characteristic_phrase(&body);

        let mut out = String::new();
        if let Some(opening) = &self.framing.opening_move {
            out.push_str(opening);
            out.push(' ');
        }
        out.push_str(&body);
        if let Some(closing) = &self.framing.closing_move {
            if !out.ends_with(' ') {
                out.push(' ');
            }
            out.push_str(closing);
        }
        out
    }

    /// Hedge sentences that mention an inhibited topic.
    fn apply_inhibitions(&self, text: &str) -> String {
        if self.inhibitions.is_empty() {
            return text.to_string();
        }
        let lower = text.to_lowercase();
        let triggered = self
            .inhibitions
            .iter()
            .any(|inh| lower.contains(&inh.topic));
        if triggered {
            // A single soft prefix — per-sentence surgery risks garbling prose.
            format!("Treading carefully here: {text}")
        } else {
            text.to_string()
        }
    }

    /// Weave one characteristic discourse marker into elaborate speech.
    ///
    /// When elaboration > 0.6 and the text has at least two sentences, the
    /// first phrase is inserted at the start of the second sentence:
    /// "Dogs are mammals. Notably, dogs have fur."
    fn insert_characteristic_phrase(&self, text: &str) -> String {
        if self.register.elaboration <= 0.6 || self.framing.characteristic_phrases.is_empty() {
            return text.to_string();
        }
        let Some(idx) = text.find(". ") else {
            return text.to_string();
        };
        let phrase = self.framing.characteristic_phrases[0].as_str();
        let (head, tail) = text.split_at(idx + 2);
        if tail.is_empty() || tail.starts_with(char::is_lowercase) {
            return text.to_string();
        }
        let mut chars = tail.chars();
        let first = chars.next().map(|c| c.to_lowercase().to_string()).unwrap_or_default();
        let rest: String = chars.collect();
        let mut phrase_cap = String::new();
        let mut pc = phrase.chars();
        if let Some(c) = pc.next() {
            phrase_cap.extend(c.to_uppercase());
            phrase_cap.extend(pc);
        }
        format!("{head}{phrase_cap}, {first}{rest}")
    }
}

// ── Register application ─────────────────────────────────────────────────

/// Apply the communication register to core prose.
///
/// Deterministic string rewrites:
/// - hedging > 0.6:   epistemic softeners ("is a" → "appears to be a", …)
/// - formality > 0.7: expand contractions; formality < 0.35: contract
/// - warmth > 0.7:    warm connectives ("Regarding" → "About")
/// - elaboration < 0.35: keep only the first two sentences
pub fn apply_register(text: &str, register: &CommunicationRegister) -> String {
    let mut result = text.to_string();

    if register.hedging_level > 0.6 {
        result = result
            .replace(" is a ", " appears to be a ")
            .replace(" is an ", " appears to be an ")
            .replace(" are ", " seem to be ")
            .replace(" causes ", " may cause ")
            .replace(" will ", " will likely ");
    }

    if register.formality > 0.7 {
        result = result
            .replace("isn't", "is not")
            .replace("aren't", "are not")
            .replace("can't", "cannot")
            .replace("don't", "do not")
            .replace("I'm", "I am")
            .replace("it's", "it is");
    } else if register.formality < 0.35 {
        result = result
            .replace("is not", "isn't")
            .replace("are not", "aren't")
            .replace("cannot", "can't")
            .replace("do not", "don't")
            .replace("I am", "I'm")
            .replace("it is", "it's");
    }

    if register.warmth > 0.7 {
        result = result.replace("Regarding ", "About ");
    }

    if register.elaboration < 0.35 {
        result = truncate_sentences(&result, 2);
    }

    result
}

/// Keep at most `n` sentences (naive '.'-boundary split, preserves the dot).
fn truncate_sentences(text: &str, n: usize) -> String {
    let mut count = 0;
    for (i, c) in text.char_indices() {
        if c == '.' {
            count += 1;
            if count == n {
                return text[..=i].to_string();
            }
        }
    }
    text.to_string()
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compartment::psyche::{ArchetypeWeights, OceanProfile};

    fn psyche(archetypes: [f32; 4], ocean: [f32; 5]) -> Psyche {
        let mut p = Psyche::default();
        p.archetypes = ArchetypeWeights {
            healer: archetypes[0],
            sage: archetypes[1],
            guardian: archetypes[2],
            explorer: archetypes[3],
        };
        p.ocean = Some(OceanProfile {
            openness: ocean[0],
            conscientiousness: ocean[1],
            extraversion: ocean[2],
            agreeableness: ocean[3],
            neuroticism: ocean[4],
        });
        p
    }

    #[test]
    fn register_formulas_match_plan() {
        // O C E A N
        let p = psyche([0.4, 0.9, 0.4, 0.4], [0.8, 0.6, 0.4, 0.5, 0.3]);
        let st = StyleTransform::from_psyche(&p, 0);
        assert!((st.register.formality - (0.6 + 0.6) / 2.0).abs() < 1e-6);
        assert!((st.register.hedging_level - (0.3 + 0.5) / 2.0).abs() < 1e-6);
        assert!((st.register.elaboration - (0.8 + 0.6) / 2.0).abs() < 1e-6);
        assert!((st.register.warmth - (0.5 + 0.4) / 2.0).abs() < 1e-6);
        assert!((st.register.metaphor_tendency - 0.8).abs() < 1e-6);
    }

    #[test]
    fn sage_framing_is_pedagogical() {
        let p = psyche([0.3, 0.9, 0.3, 0.3], [0.7, 0.8, 0.4, 0.6, 0.3]);
        let st = StyleTransform::from_psyche(&p, 0);
        assert_eq!(st.framing.archetype, "sage");
        assert_eq!(st.framing.opening_move.as_deref(), Some("Let me explain."));
        assert_eq!(
            st.framing.closing_move.as_deref(),
            Some("Does that clarify things?")
        );
    }

    #[test]
    fn framings_differ_across_archetypes() {
        let sage = StyleTransform::from_psyche(&psyche([0.3, 0.9, 0.3, 0.3], [0.5; 5]), 0);
        let explorer = StyleTransform::from_psyche(&psyche([0.3, 0.3, 0.3, 0.9], [0.5; 5]), 0);
        let healer = StyleTransform::from_psyche(&psyche([0.9, 0.3, 0.3, 0.3], [0.5; 5]), 0);
        let guardian = StyleTransform::from_psyche(&psyche([0.3, 0.3, 0.9, 0.3], [0.5; 5]), 0);
        let openings: Vec<_> = [&sage, &explorer, &healer, &guardian]
            .iter()
            .map(|s| s.framing.opening_move.clone())
            .collect();
        for i in 0..openings.len() {
            for j in (i + 1)..openings.len() {
                assert_ne!(openings[i], openings[j], "archetype openings must differ");
            }
        }
    }

    #[test]
    fn phrase_pick_is_deterministic_and_seed_varied() {
        let p = psyche([0.3, 0.9, 0.3, 0.3], [0.5; 5]);
        let a = StyleTransform::from_psyche(&p, 1);
        let b = StyleTransform::from_psyche(&p, 1);
        assert_eq!(a.framing.opening_move, b.framing.opening_move);
        let c = StyleTransform::from_psyche(&p, 2);
        assert_ne!(a.framing.opening_move, c.framing.opening_move);
    }

    #[test]
    fn hedging_rewrites_copulas() {
        let reg = CommunicationRegister {
            formality: 0.5,
            hedging_level: 0.8,
            elaboration: 0.5,
            warmth: 0.5,
            metaphor_tendency: 0.5,
        };
        let out = apply_register("A dog is a mammal. Dogs are loyal.", &reg);
        assert!(out.contains("appears to be a mammal"), "got: {out}");
        assert!(out.contains("seem to be loyal"), "got: {out}");
    }

    #[test]
    fn assertive_register_leaves_text_alone() {
        let reg = CommunicationRegister {
            formality: 0.5,
            hedging_level: 0.2,
            elaboration: 0.5,
            warmth: 0.5,
            metaphor_tendency: 0.5,
        };
        let text = "A dog is a mammal.";
        assert_eq!(apply_register(text, &reg), text);
    }

    #[test]
    fn low_elaboration_truncates() {
        let reg = CommunicationRegister {
            formality: 0.5,
            hedging_level: 0.2,
            elaboration: 0.2,
            warmth: 0.5,
            metaphor_tendency: 0.5,
        };
        let out = apply_register("One. Two. Three. Four.", &reg);
        assert_eq!(out, "One. Two.");
    }

    #[test]
    fn formality_expands_contractions() {
        let reg = CommunicationRegister {
            formality: 0.9,
            hedging_level: 0.2,
            elaboration: 0.5,
            warmth: 0.2,
            metaphor_tendency: 0.5,
        };
        assert_eq!(
            apply_register("It isn't here and I can't say why.", &reg),
            "It is not here and I cannot say why."
        );
    }

    #[test]
    fn apply_wraps_with_moves() {
        let p = psyche([0.3, 0.9, 0.3, 0.3], [0.7, 0.8, 0.4, 0.6, 0.3]);
        let st = StyleTransform::from_psyche(&p, 0);
        let out = st.apply("A dog is a mammal.");
        assert!(out.starts_with("Let me explain."), "got: {out}");
        assert!(out.ends_with("Does that clarify things?"), "got: {out}");
    }

    #[test]
    fn characteristic_phrase_woven_into_elaborate_speech() {
        let p = psyche([0.3, 0.9, 0.3, 0.3], [0.9, 0.9, 0.4, 0.4, 0.2]);
        let st = StyleTransform::from_psyche(&p, 0);
        let out = st.apply("A dog is a mammal. Dogs descend from wolves.");
        assert!(out.contains("Notably, dogs descend from wolves"), "got: {out}");
    }

    #[test]
    fn shadow_inhibition_prefixes_sensitive_topics() {
        let p = psyche([0.3, 0.9, 0.3, 0.3], [0.5; 5]);
        // Default psyche bias pattern has no action_triggers; add one.
        let mut p = p;
        p.shadow.bias_patterns[0].action_triggers = vec!["surveillance".into()];
        let st = StyleTransform::from_psyche(&p, 0);
        let out = st.apply("Surveillance systems watch everything.");
        assert!(out.contains("Treading carefully here:"), "got: {out}");
    }

    #[test]
    fn empty_core_stays_empty() {
        let p = psyche([0.3, 0.9, 0.3, 0.3], [0.5; 5]);
        let st = StyleTransform::from_psyche(&p, 0);
        assert_eq!(st.apply("   "), "");
    }
}
