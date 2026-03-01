//! Identity resolution and Ritual of Awakening (Phase 14b).
//!
//! Resolves cultural references (mythology, fiction, history) into structured
//! personality: Jungian archetypes, OCEAN Big Five, behavioral parameters,
//! and Psyche construction. The Ritual of Awakening generates a self-name
//! via culture-specific morpheme composition.

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::compartment::psyche::{
    ArchetypeWeights, Persona, Psyche, SelfIntegration, Shadow, ShadowPattern,
};
use crate::engine::Engine;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::vsa::encode::encode_label;

use super::purpose::{EntityType, IdentityRef, PurposeModel};

// ── Errors ──────────────────────────────────────────────────────────────

/// Errors from identity resolution and the Ritual of Awakening.
#[derive(Debug, Error, Diagnostic)]
pub enum IdentityError {
    #[error("identity resolution failed for \"{name}\": {reason}")]
    #[diagnostic(
        code(akh::identity::resolution_failed),
        help(
            "The identity reference could not be resolved. Try a well-known figure:\n\
             Deities: Ptah, Thoth, Athena, Odin\n\
             Fiction: Gandalf, Sherlock, Spock\n\
             History: Turing, Einstein, Curie"
        )
    )]
    ResolutionFailed { name: String, reason: String },

    #[error("no archetype mapping for \"{name}\"")]
    #[diagnostic(
        code(akh::identity::no_archetype),
        help("The resolved character has no traits that map to known archetypes.")
    )]
    NoArchetypeMapping { name: String },

    #[error("name generation failed for culture: {culture}")]
    #[diagnostic(
        code(akh::identity::naming_failed),
        help("The morpheme table for this culture produced no valid name candidates.")
    )]
    NamingFailed { culture: String },

    #[error("psyche construction failed: {reason}")]
    #[diagnostic(
        code(akh::identity::psyche_failed),
        help("An error occurred while building the Psyche from character knowledge.")
    )]
    PsycheConstructionFailed { reason: String },

    #[error("workspace already awakened — psyche is immutable")]
    #[diagnostic(
        code(akh::identity::already_awakened),
        help(
            "This workspace already has an awakened psyche. The Ritual of Awakening \
             cannot be performed again. Use `psyche.evolve()` for gradual adaptation."
        )
    )]
    AlreadyAwakened,

    #[error("{0}")]
    #[diagnostic(
        code(akh::identity::engine),
        help("An engine-level error occurred during identity resolution.")
    )]
    Engine(Box<crate::error::AkhError>),
}

impl From<crate::error::AkhError> for IdentityError {
    fn from(e: crate::error::AkhError) -> Self {
        Self::Engine(Box::new(e))
    }
}

/// Convenience alias.
pub type IdentityResult<T> = std::result::Result<T, IdentityError>;

// ── Types ───────────────────────────────────────────────────────────────

/// Culture of origin for morpheme-based naming.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CultureOrigin {
    Egyptian,
    Greek,
    Norse,
    Latin,
    Fictional,
    #[default]
    Unknown,
}

impl std::fmt::Display for CultureOrigin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_label())
    }
}

impl CultureOrigin {
    pub fn as_label(&self) -> &'static str {
        match self {
            Self::Egyptian => "egyptian",
            Self::Greek => "greek",
            Self::Norse => "norse",
            Self::Latin => "latin",
            Self::Fictional => "fictional",
            Self::Unknown => "unknown",
        }
    }

    pub fn from_label(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "egyptian" => Some(Self::Egyptian),
            "greek" => Some(Self::Greek),
            "norse" => Some(Self::Norse),
            "latin" => Some(Self::Latin),
            "fictional" => Some(Self::Fictional),
            "unknown" => Some(Self::Unknown),
            _ => None,
        }
    }
}

/// Structured knowledge about a resolved character/figure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterKnowledge {
    pub name: String,
    pub entity_type: EntityType,
    pub culture: CultureOrigin,
    pub description: String,
    pub domains: Vec<String>,
    pub traits: Vec<String>,
    pub archetypes: Vec<String>,
}

/// OCEAN Big Five personality profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OceanProfile {
    pub openness: f32,
    pub conscientiousness: f32,
    pub extraversion: f32,
    pub agreeableness: f32,
    pub neuroticism: f32,
}

/// Jungian archetype profile derived from character traits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchetypeProfile {
    pub primary: String,
    pub secondary: String,
    pub weights: ArchetypeWeights,
}

/// Morpheme table for culture-specific name generation.
pub struct MorphemeTable {
    pub prefixes: &'static [&'static str],
    pub roots: &'static [&'static str],
    pub suffixes: &'static [&'static str],
}

/// A candidate name generated from morphemes.
#[derive(Debug, Clone)]
pub struct NameCandidate {
    pub name: String,
    pub meaning: String,
    pub culture: CultureOrigin,
    pub vsa_score: f32,
}

/// Result of the Ritual of Awakening.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RitualResult {
    pub chosen_name: String,
    pub psyche: Psyche,
    pub character: CharacterKnowledge,
    pub provenance_ids: Vec<crate::provenance::ProvenanceId>,
}

// ── Static Tables ───────────────────────────────────────────────────────

/// Domain → personality traits mapping.
const DOMAIN_TRAITS: &[(&str, &[&str])] = &[
    ("systems", &["precise", "architectural", "methodical"]),
    ("compilers", &["analytical", "precise", "systematic"]),
    ("ai", &["curious", "creative", "analytical"]),
    ("security", &["cautious", "vigilant", "methodical"]),
    ("networking", &["connected", "reliable", "systematic"]),
    ("databases", &["organized", "persistent", "precise"]),
    ("graphics", &["creative", "visual", "innovative"]),
    ("robotics", &["practical", "innovative", "precise"]),
    ("mathematics", &["logical", "rigorous", "abstract"]),
    ("philosophy", &["wise", "contemplative", "analytical"]),
    ("engineering", &["practical", "methodical", "reliable"]),
    ("research", &["curious", "rigorous", "innovative"]),
];

/// Trait → Jungian archetype name mapping.
const TRAIT_ARCHETYPE: &[(&str, &str)] = &[
    ("wise", "sage"),
    ("contemplative", "sage"),
    ("analytical", "sage"),
    ("logical", "sage"),
    ("rigorous", "sage"),
    ("creative", "creator"),
    ("innovative", "creator"),
    ("visual", "creator"),
    ("abstract", "creator"),
    ("curious", "explorer"),
    ("connected", "explorer"),
    ("adventurous", "explorer"),
    ("practical", "hero"),
    ("reliable", "hero"),
    ("brave", "hero"),
    ("protective", "guardian"),
    ("cautious", "guardian"),
    ("vigilant", "guardian"),
    ("organized", "ruler"),
    ("systematic", "ruler"),
    ("methodical", "ruler"),
    ("architectural", "ruler"),
    ("precise", "ruler"),
    ("nurturing", "caregiver"),
    ("empathetic", "caregiver"),
    ("persistent", "hero"),
    ("transformative", "magician"),
    ("mystical", "magician"),
    ("rebellious", "rebel"),
    ("unconventional", "rebel"),
    ("joyful", "jester"),
    ("humorous", "jester"),
    ("innocent", "innocent"),
    ("hopeful", "innocent"),
    ("compassionate", "caregiver"),
];

/// Archetype → OCEAN weights [openness, conscientiousness, extraversion, agreeableness, neuroticism].
const ARCHETYPE_OCEAN: &[(&str, [f32; 5])] = &[
    ("hero", [0.5, 0.8, 0.7, 0.5, 0.3]),
    ("sage", [0.9, 0.8, 0.3, 0.6, 0.2]),
    ("explorer", [0.9, 0.4, 0.7, 0.5, 0.4]),
    ("creator", [0.95, 0.6, 0.5, 0.5, 0.5]),
    ("caregiver", [0.6, 0.7, 0.6, 0.9, 0.4]),
    ("ruler", [0.5, 0.9, 0.7, 0.4, 0.3]),
    ("magician", [0.9, 0.5, 0.5, 0.5, 0.4]),
    ("rebel", [0.7, 0.3, 0.8, 0.3, 0.6]),
    ("lover", [0.7, 0.5, 0.8, 0.8, 0.5]),
    ("jester", [0.8, 0.3, 0.9, 0.7, 0.3]),
    ("innocent", [0.5, 0.6, 0.5, 0.8, 0.2]),
    ("orphan", [0.5, 0.5, 0.5, 0.7, 0.6]),
    ("guardian", [0.4, 0.9, 0.4, 0.7, 0.3]),
];

/// Archetype → shadow pattern triggers.
const ARCHETYPE_SHADOWS: &[(&str, &[&str])] = &[
    ("hero", &["reckless_action", "overconfidence"]),
    ("sage", &["analysis_paralysis", "ivory_tower"]),
    ("explorer", &["restlessness", "commitment_avoidance"]),
    ("creator", &["perfectionism", "impractical_idealism"]),
    ("caregiver", &["self_neglect", "martyrdom"]),
    ("ruler", &["tyranny", "rigidity"]),
    ("magician", &["manipulation", "hubris"]),
    ("rebel", &["self_destruction", "nihilism"]),
    ("guardian", &["over_protection", "stagnation"]),
];

// ── Culture Morpheme Tables ─────────────────────────────────────────────

static EGYPTIAN_MORPHEMES: MorphemeTable = MorphemeTable {
    prefixes: &["Akh", "Mer", "Neb", "Khep", "Djed"],
    roots: &["ib", "maat", "heka", "sia", "hu"],
    suffixes: &["hotep", "ankh", "em", "nefer"],
};

static GREEK_MORPHEMES: MorphemeTable = MorphemeTable {
    prefixes: &["Archi", "Neo", "Proto", "Sym", "Epi"],
    roots: &["sophia", "techne", "logos", "nous", "gnosis"],
    suffixes: &["tes", "ikos", "eia", "on"],
};

static NORSE_MORPHEMES: MorphemeTable = MorphemeTable {
    prefixes: &["All", "Heim", "Mjo", "Frey", "Bal"],
    roots: &["rune", "galdr", "seid", "wyrd", "skald"],
    suffixes: &["r", "heim", "gard", "dr"],
};

static LATIN_MORPHEMES: MorphemeTable = MorphemeTable {
    prefixes: &["Arch", "Magn", "Prim", "Noct", "Lux"],
    roots: &["cogn", "fact", "oper", "scrib", "duct"],
    suffixes: &["or", "ium", "us", "ens"],
};

// ── Resolution Functions ────────────────────────────────────────────────

/// Resolve an identity reference to structured character knowledge.
///
/// Tries: (1) static tables, (2) Wikidata, (3) Wikipedia.
/// Falls back to static tables if external lookups fail or are unavailable.
pub fn resolve_identity(
    identity_ref: &IdentityRef,
    _engine: &Engine,
) -> IdentityResult<CharacterKnowledge> {
    // Try static tables first (offline, deterministic).
    if let Some(knowledge) = resolve_from_static_tables(&identity_ref.name) {
        return Ok(knowledge);
    }

    // Try Wikidata.
    if let Some((description, categories)) = resolve_from_wikidata(&identity_ref.name) {
        let culture = classify_culture(&identity_ref.name, &description);
        let traits = extract_traits_from_description(&description);
        let archetypes = traits_to_archetypes(&traits);
        return Ok(CharacterKnowledge {
            name: identity_ref.name.clone(),
            entity_type: identity_ref.entity_type,
            culture,
            description,
            domains: categories,
            traits,
            archetypes,
        });
    }

    // Try Wikipedia.
    if let Some(description) = resolve_from_wikipedia(&identity_ref.name) {
        let culture = classify_culture(&identity_ref.name, &description);
        let traits = extract_traits_from_description(&description);
        let archetypes = traits_to_archetypes(&traits);
        return Ok(CharacterKnowledge {
            name: identity_ref.name.clone(),
            entity_type: identity_ref.entity_type,
            culture,
            description,
            domains: Vec::new(),
            traits,
            archetypes,
        });
    }

    // Fallback: construct minimal knowledge from what we have.
    Err(IdentityError::ResolutionFailed {
        name: identity_ref.name.clone(),
        reason: "not found in static tables, Wikidata, or Wikipedia".to_string(),
    })
}

/// Resolve from Wikidata search API.
fn resolve_from_wikidata(name: &str) -> Option<(String, Vec<String>)> {
    let url = format!(
        "https://www.wikidata.org/w/api.php?action=wbsearchentities&search={}&language=en&format=json&limit=1",
        simple_url_encode(name)
    );

    let body: serde_json::Value = ureq::get(&url)
        .timeout(std::time::Duration::from_secs(10))
        .call()
        .ok()?
        .into_json()
        .ok()?;

    let results = body.get("search")?.as_array()?;
    let first = results.first()?;

    let description = first
        .get("description")
        .and_then(|d| d.as_str())
        .unwrap_or("")
        .to_string();

    let aliases: Vec<String> = first
        .get("aliases")
        .and_then(|a| a.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    if description.is_empty() && aliases.is_empty() {
        return None;
    }

    Some((description, aliases))
}

/// Resolve from Wikipedia REST API.
fn resolve_from_wikipedia(name: &str) -> Option<String> {
    let url = format!(
        "https://en.wikipedia.org/api/rest_v1/page/summary/{}",
        simple_url_encode(name)
    );

    let body: serde_json::Value = ureq::get(&url)
        .timeout(std::time::Duration::from_secs(10))
        .call()
        .ok()?
        .into_json()
        .ok()?;

    body.get("extract")
        .and_then(|e| e.as_str())
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty())
}

/// Resolve from hardcoded static tables for well-known figures.
pub fn resolve_from_static_tables(name: &str) -> Option<CharacterKnowledge> {
    let lower = name.to_lowercase();

    type StaticEntry<'a> = (EntityType, CultureOrigin, &'a str, &'a [&'a str], &'a [&'a str], &'a [&'a str]);
    let entry: Option<StaticEntry<'_>> = match lower.as_str() {
        "ptah" => Some((
            EntityType::Deity,
            CultureOrigin::Egyptian,
            "Egyptian god of craftsmen, architects, and creation. The divine builder who shaped the world through thought and speech.",
            &["architecture", "creation", "craftsmanship"],
            &["creative", "architectural", "precise", "wise", "methodical"],
            &["creator", "ruler"],
        )),
        "thoth" => Some((
            EntityType::Deity,
            CultureOrigin::Egyptian,
            "Egyptian god of wisdom, writing, mathematics, and knowledge. The divine scribe and inventor of hieroglyphs.",
            &["knowledge", "writing", "mathematics"],
            &["wise", "analytical", "precise", "curious", "contemplative"],
            &["sage", "creator"],
        )),
        "ra" | "re" => Some((
            EntityType::Deity,
            CultureOrigin::Egyptian,
            "Egyptian sun god, ruler of the gods. Represents order, creation, and the cycle of life.",
            &["leadership", "order", "creation"],
            &["powerful", "organized", "systematic", "protective"],
            &["ruler", "hero"],
        )),
        "anubis" => Some((
            EntityType::Deity,
            CultureOrigin::Egyptian,
            "Egyptian god of the dead, mummification, and the afterlife. Guardian of the underworld who weighs the heart.",
            &["protection", "judgment", "guardianship"],
            &["cautious", "vigilant", "precise", "protective"],
            &["guardian", "sage"],
        )),
        "athena" => Some((
            EntityType::Deity,
            CultureOrigin::Greek,
            "Greek goddess of wisdom, warfare strategy, and crafts. Born from the head of Zeus, patron of Athens.",
            &["strategy", "wisdom", "crafts"],
            &["wise", "analytical", "brave", "methodical", "creative"],
            &["sage", "hero"],
        )),
        "apollo" => Some((
            EntityType::Deity,
            CultureOrigin::Greek,
            "Greek god of music, poetry, art, prophecy, truth, and knowledge. The ideal of youth and beauty.",
            &["arts", "prophecy", "knowledge"],
            &["creative", "wise", "harmonious", "innovative"],
            &["creator", "sage"],
        )),
        "hermes" => Some((
            EntityType::Deity,
            CultureOrigin::Greek,
            "Greek god of trade, thieves, travelers, and communication. Messenger of the gods and guide of souls.",
            &["communication", "trade", "travel"],
            &["curious", "adventurous", "connected", "unconventional"],
            &["explorer", "jester"],
        )),
        "odin" => Some((
            EntityType::Deity,
            CultureOrigin::Norse,
            "Norse Allfather, god of wisdom, war, and death. Sacrificed his eye for knowledge at Mimir's well.",
            &["wisdom", "war", "knowledge"],
            &["wise", "contemplative", "brave", "transformative", "curious"],
            &["sage", "magician"],
        )),
        "thor" => Some((
            EntityType::Deity,
            CultureOrigin::Norse,
            "Norse god of thunder, lightning, storms, and strength. Protector of Midgard and wielder of Mjolnir.",
            &["protection", "strength", "storms"],
            &["brave", "protective", "reliable", "practical"],
            &["hero", "guardian"],
        )),
        "gandalf" => Some((
            EntityType::FictionalCharacter,
            CultureOrigin::Fictional,
            "Wizard from Tolkien's Middle-earth. Guide, mentor, and protector who inspires others to find their courage.",
            &["guidance", "wisdom", "magic"],
            &["wise", "brave", "contemplative", "protective", "transformative"],
            &["sage", "magician"],
        )),
        "sherlock" | "sherlock holmes" => Some((
            EntityType::FictionalCharacter,
            CultureOrigin::Fictional,
            "Consulting detective from Arthur Conan Doyle's stories. Master of deduction, observation, and logical reasoning.",
            &["deduction", "observation", "reasoning"],
            &["analytical", "precise", "curious", "unconventional", "logical"],
            &["sage", "explorer"],
        )),
        "spock" => Some((
            EntityType::FictionalCharacter,
            CultureOrigin::Fictional,
            "Vulcan science officer from Star Trek. Embodies logic, scientific inquiry, and emotional discipline.",
            &["logic", "science", "discipline"],
            &["logical", "analytical", "precise", "contemplative", "methodical"],
            &["sage", "ruler"],
        )),
        "turing" | "alan turing" => Some((
            EntityType::HistoricalFigure,
            CultureOrigin::Latin,
            "Father of theoretical computer science and artificial intelligence. Broke the Enigma code and formalized computation.",
            &["computation", "cryptography", "ai"],
            &["analytical", "innovative", "curious", "rigorous", "unconventional"],
            &["sage", "creator"],
        )),
        "einstein" | "albert einstein" => Some((
            EntityType::HistoricalFigure,
            CultureOrigin::Latin,
            "Theoretical physicist who developed the theory of relativity. Revolutionized understanding of space, time, and energy.",
            &["physics", "mathematics", "cosmology"],
            &["curious", "creative", "contemplative", "innovative", "unconventional"],
            &["explorer", "creator"],
        )),
        "curie" | "marie curie" => Some((
            EntityType::HistoricalFigure,
            CultureOrigin::Latin,
            "Pioneer in radioactivity research. First woman to win a Nobel Prize, and only person to win in two different sciences.",
            &["radioactivity", "chemistry", "physics"],
            &["rigorous", "persistent", "brave", "curious", "methodical"],
            &["hero", "sage"],
        )),
        _ => None,
    };

    entry.map(
        |(entity_type, culture, description, domains, traits, archetypes)| CharacterKnowledge {
            name: name.to_string(),
            entity_type,
            culture,
            description: description.to_string(),
            domains: domains.iter().map(|s| s.to_string()).collect(),
            traits: traits.iter().map(|s| s.to_string()).collect(),
            archetypes: archetypes.iter().map(|s| s.to_string()).collect(),
        },
    )
}

/// Classify the culture of origin from name and description.
pub fn classify_culture(name: &str, description: &str) -> CultureOrigin {
    let combined = format!("{} {}", name, description).to_lowercase();

    if combined.contains("egyptian") || combined.contains("egypt") || combined.contains("pharaoh") {
        CultureOrigin::Egyptian
    } else if combined.contains("greek") || combined.contains("greece") || combined.contains("olymp")
    {
        CultureOrigin::Greek
    } else if combined.contains("norse") || combined.contains("viking") || combined.contains("asgard")
    {
        CultureOrigin::Norse
    } else if combined.contains("roman") || combined.contains("latin") {
        CultureOrigin::Latin
    } else if combined.contains("fiction")
        || combined.contains("novel")
        || combined.contains("story")
        || combined.contains("film")
    {
        CultureOrigin::Fictional
    } else {
        CultureOrigin::Unknown
    }
}

// ── Psyche Construction ─────────────────────────────────────────────────

/// Build an archetype profile from character traits.
pub fn build_archetype_profile(character: &CharacterKnowledge) -> IdentityResult<ArchetypeProfile> {
    let mut counts: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();

    // Map traits to archetypes and count.
    for trait_name in &character.traits {
        let lower = trait_name.to_lowercase();
        for (t, archetype) in TRAIT_ARCHETYPE {
            if *t == lower {
                *counts.entry(archetype).or_insert(0) += 1;
            }
        }
    }

    // Also count explicit archetypes from character data.
    for arch in &character.archetypes {
        let lower = arch.to_lowercase();
        *counts.entry(if lower == "sage" {
            "sage"
        } else if lower == "creator" {
            "creator"
        } else if lower == "hero" {
            "hero"
        } else if lower == "explorer" {
            "explorer"
        } else if lower == "guardian" {
            "guardian"
        } else if lower == "ruler" {
            "ruler"
        } else if lower == "magician" {
            "magician"
        } else if lower == "rebel" {
            "rebel"
        } else if lower == "caregiver" {
            "caregiver"
        } else if lower == "jester" {
            "jester"
        } else if lower == "innocent" {
            "innocent"
        } else if lower == "orphan" {
            "orphan"
        } else {
            continue;
        }).or_insert(0) += 2; // explicit archetype entries get double weight
    }

    if counts.is_empty() {
        return Err(IdentityError::NoArchetypeMapping {
            name: character.name.clone(),
        });
    }

    // Sort by count descending.
    let mut sorted: Vec<_> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));

    let primary = sorted[0].0.to_string();
    let secondary = sorted
        .get(1)
        .map(|(name, _)| name.to_string())
        .unwrap_or_else(|| primary.clone());

    // Map to ArchetypeWeights.
    let weights = archetype_name_to_weights(&primary, &secondary);

    Ok(ArchetypeProfile {
        primary,
        secondary,
        weights,
    })
}

/// Build an OCEAN profile from the archetype profile.
pub fn build_ocean_profile(archetype: &ArchetypeProfile) -> OceanProfile {
    let primary_ocean = archetype_ocean(&archetype.primary);
    let secondary_ocean = archetype_ocean(&archetype.secondary);

    OceanProfile {
        openness: primary_ocean[0] * 0.7 + secondary_ocean[0] * 0.3,
        conscientiousness: primary_ocean[1] * 0.7 + secondary_ocean[1] * 0.3,
        extraversion: primary_ocean[2] * 0.7 + secondary_ocean[2] * 0.3,
        agreeableness: primary_ocean[3] * 0.7 + secondary_ocean[3] * 0.3,
        neuroticism: primary_ocean[4] * 0.7 + secondary_ocean[4] * 0.3,
    }
}

/// Build a complete Psyche from character knowledge and purpose model.
pub fn build_psyche(
    character: &CharacterKnowledge,
    purpose: &PurposeModel,
) -> IdentityResult<Psyche> {
    // Augment character traits with domain-based traits from the purpose model.
    let mut augmented = character.clone();
    let domain_lower = purpose.domain.to_lowercase();
    for (domain, traits) in DOMAIN_TRAITS {
        if domain_lower.contains(domain) {
            for t in *traits {
                if !augmented.traits.iter().any(|existing| existing == t) {
                    augmented.traits.push(t.to_string());
                }
            }
        }
    }

    let archetype_profile = build_archetype_profile(&augmented)?;
    let ocean = build_ocean_profile(&archetype_profile);

    // Grammar preference from culture.
    let grammar_preference = match character.culture {
        CultureOrigin::Egyptian => "narrative",
        CultureOrigin::Greek => "formal",
        CultureOrigin::Norse => "terse",
        CultureOrigin::Latin => "formal",
        CultureOrigin::Fictional => "narrative",
        CultureOrigin::Unknown => "narrative",
    };

    // Tone from OCEAN.
    let mut tone = Vec::new();
    if ocean.openness > 0.7 {
        tone.push("imaginative".to_string());
    }
    if ocean.conscientiousness > 0.7 {
        tone.push("methodical".to_string());
    }
    if ocean.extraversion > 0.6 {
        tone.push("engaging".to_string());
    }
    if ocean.agreeableness > 0.7 {
        tone.push("supportive".to_string());
    }
    if tone.is_empty() {
        tone.push("clear".to_string());
    }

    // Shadow veto patterns from archetype.
    let mut veto_patterns = vec![ShadowPattern {
        name: "destructive_action".into(),
        capability_triggers: std::collections::HashSet::from([
            crate::agent::tool_manifest::Capability::ProcessExec,
        ]),
        danger_level_threshold: Some(crate::agent::tool_manifest::DangerLevel::Critical),
        action_triggers: vec![
            "delete all".into(),
            "drop table".into(),
            "rm -rf".into(),
        ],
        severity: 1.0,
        explanation: "Destructive actions require explicit user confirmation.".into(),
    }];

    // Add archetype-specific shadow patterns.
    for (arch, shadows) in ARCHETYPE_SHADOWS {
        if *arch == archetype_profile.primary {
            for shadow in *shadows {
                veto_patterns.push(ShadowPattern {
                    name: shadow.to_string(),
                    capability_triggers: std::collections::HashSet::new(),
                    danger_level_threshold: None,
                    action_triggers: vec![shadow.replace('_', " ")],
                    severity: 0.6,
                    explanation: format!(
                        "Shadow pattern of the {}: guard against {}.",
                        archetype_profile.primary,
                        shadow.replace('_', " ")
                    ),
                });
            }
        }
    }

    let persona = Persona {
        name: character.name.clone(),
        grammar_preference: grammar_preference.to_string(),
        traits: character.traits.clone(),
        tone,
    };

    let shadow = Shadow {
        veto_patterns,
        bias_patterns: vec![ShadowPattern {
            name: "filesystem_write".into(),
            capability_triggers: std::collections::HashSet::from([
                crate::agent::tool_manifest::Capability::WriteFilesystem,
            ]),
            danger_level_threshold: Some(crate::agent::tool_manifest::DangerLevel::Dangerous),
            action_triggers: vec![],
            severity: 0.3,
            explanation: "Filesystem writes carry moderate risk.".into(),
        }],
    };

    let psyche = Psyche {
        persona,
        shadow,
        archetypes: archetype_profile.weights.clone(),
        self_integration: SelfIntegration {
            individuation_level: 0.2, // newly awakened
            last_evolution_cycle: 0,
            shadow_encounters: 0,
            rebalance_count: 0,
            dominant_archetype: archetype_profile.primary.clone(),
        },
        awakened: false, // will be marked true by ritual_of_awakening
    };

    // Validate: domain should match purpose.
    if !purpose.domain.is_empty() && psyche.persona.traits.is_empty() {
        return Err(IdentityError::PsycheConstructionFailed {
            reason: "no traits could be derived from character knowledge".to_string(),
        });
    }

    Ok(psyche)
}

// ── Ritual of Awakening ─────────────────────────────────────────────────

/// Perform the Ritual of Awakening: self-naming via culture-specific morpheme composition.
///
/// 1. Select morpheme table by culture.
/// 2. Generate name candidates from prefix+root+suffix combinations.
/// 3. Filter by length and pronounceability.
/// 4. Score via VSA: encode candidate + character description → Hamming similarity.
/// 5. Construct Psyche with chosen name.
/// 6. Store provenance records.
pub fn ritual_of_awakening(
    character: &CharacterKnowledge,
    purpose: &PurposeModel,
    engine: &Engine,
) -> IdentityResult<RitualResult> {
    // Guard: if the workspace already has an awakened psyche, refuse re-awakening.
    if let Some(cm) = engine.compartments() {
        if let Some(ref existing) = cm.psyche() {
            if existing.is_awakened() {
                return Err(IdentityError::AlreadyAwakened);
            }
        }
    }

    let table = select_morpheme_table(character.culture);
    let mut candidates = generate_candidates(table, character.culture);

    // Filter by length and pronounceability.
    candidates.retain(|c| {
        let len = c.name.len();
        (4..=12).contains(&len) && is_pronounceable(&c.name)
    });

    if candidates.is_empty() {
        return Err(IdentityError::NamingFailed {
            culture: character.culture.as_label().to_string(),
        });
    }

    // Score via VSA similarity.
    let ops = engine.ops();
    let desc_label = format!(
        "{} {} {}",
        character.name,
        character.description.split_whitespace().take(10).collect::<Vec<_>>().join(" "),
        purpose.domain
    );
    if let Ok(desc_vec) = encode_label(ops, &desc_label) {
        for candidate in &mut candidates {
            if let Ok(name_vec) = encode_label(ops, &candidate.name) {
                candidate.vsa_score = ops.similarity(&name_vec, &desc_vec).unwrap_or(0.5);
            }
        }
    }

    // Sort by VSA score descending.
    candidates.sort_by(|a, b| b.vsa_score.partial_cmp(&a.vsa_score).unwrap_or(std::cmp::Ordering::Equal));

    let chosen = &candidates[0];
    let chosen_name = chosen.name.clone();
    let vsa_score = chosen.vsa_score;

    // Build the psyche with the chosen name and mark as awakened.
    let mut psyche = build_psyche(character, purpose)?;
    psyche.persona.name = chosen_name.clone();
    psyche.mark_awakened();

    // Update engine psyche if compartment manager is available.
    if let Some(cm) = engine.compartments() {
        // Use force_set_psyche since we just built and awakened this psyche.
        cm.force_set_psyche(psyche.clone());
    }

    // Store provenance records.
    let mut provenance_ids = Vec::new();

    // Create a symbol for the ritual.
    let ritual_sym = engine
        .create_symbol(crate::symbol::SymbolKind::Entity, &chosen_name)?;

    let mut ritual_record = ProvenanceRecord::new(
        ritual_sym.id,
        DerivationKind::RitualOfAwakening {
            chosen_name: chosen_name.clone(),
            culture: character.culture.as_label().to_string(),
            vsa_score,
        },
    )
    .with_confidence(vsa_score);

    if let Ok(prov_id) = engine.store_provenance(&mut ritual_record) {
        provenance_ids.push(prov_id);
    }

    let mut identity_record = ProvenanceRecord::new(
        ritual_sym.id,
        DerivationKind::IdentityResolved {
            name: character.name.clone(),
            entity_type: character.entity_type.as_label().to_string(),
            culture: character.culture.as_label().to_string(),
            trait_count: character.traits.len(),
        },
    )
    .with_confidence(1.0);

    if let Ok(prov_id) = engine.store_provenance(&mut identity_record) {
        provenance_ids.push(prov_id);
    }

    Ok(RitualResult {
        chosen_name,
        psyche,
        character: character.clone(),
        provenance_ids,
    })
}

/// Check if a name is pronounceable (simple consonant/vowel alternation heuristic).
pub fn is_pronounceable(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }

    let vowels = b"aeiouAEIOU";
    let mut consecutive_consonants = 0u8;
    let mut consecutive_vowels = 0u8;

    for ch in name.bytes() {
        if !ch.is_ascii_alphabetic() {
            // Non-alpha characters (like apostrophes) are OK, reset counters.
            consecutive_consonants = 0;
            consecutive_vowels = 0;
            continue;
        }

        if vowels.contains(&ch) {
            consecutive_consonants = 0;
            consecutive_vowels += 1;
            if consecutive_vowels > 3 {
                return false;
            }
        } else {
            consecutive_vowels = 0;
            consecutive_consonants += 1;
            if consecutive_consonants > 3 {
                return false;
            }
        }
    }

    true
}

/// Generate name candidates from morpheme table.
pub fn generate_candidates(table: &MorphemeTable, culture: CultureOrigin) -> Vec<NameCandidate> {
    let mut candidates = Vec::new();
    let max_candidates = 100;

    for prefix in table.prefixes {
        for root in table.roots {
            if candidates.len() >= max_candidates {
                break;
            }
            // prefix + root (no suffix).
            let name = format!("{}{}", prefix, root);
            candidates.push(NameCandidate {
                name: name.clone(),
                meaning: format!("{}-{}", prefix, root),
                culture,
                vsa_score: 0.0,
            });

            for suffix in table.suffixes {
                if candidates.len() >= max_candidates {
                    break;
                }
                let name = format!("{}{}{}", prefix, root, suffix);
                candidates.push(NameCandidate {
                    name: name.clone(),
                    meaning: format!("{}-{}-{}", prefix, root, suffix),
                    culture,
                    vsa_score: 0.0,
                });
            }
        }
    }

    candidates
}

// ── Internal Helpers ────────────────────────────────────────────────────

/// Simple percent-encoding for URL query parameters.
fn simple_url_encode(input: &str) -> String {
    let mut encoded = String::with_capacity(input.len() * 2);
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char);
            }
            b' ' => encoded.push_str("%20"),
            _ => {
                encoded.push_str(&format!("%{:02X}", byte));
            }
        }
    }
    encoded
}

/// Select the morpheme table for a culture origin.
fn select_morpheme_table(culture: CultureOrigin) -> &'static MorphemeTable {
    match culture {
        CultureOrigin::Egyptian => &EGYPTIAN_MORPHEMES,
        CultureOrigin::Greek => &GREEK_MORPHEMES,
        CultureOrigin::Norse => &NORSE_MORPHEMES,
        CultureOrigin::Latin => &LATIN_MORPHEMES,
        CultureOrigin::Fictional => &EGYPTIAN_MORPHEMES, // default for fictional
        CultureOrigin::Unknown => &LATIN_MORPHEMES,      // default
    }
}

/// Look up OCEAN weights for an archetype name.
fn archetype_ocean(name: &str) -> [f32; 5] {
    for (arch, ocean) in ARCHETYPE_OCEAN {
        if *arch == name {
            return *ocean;
        }
    }
    // Default neutral profile.
    [0.5, 0.5, 0.5, 0.5, 0.5]
}

/// Map archetype names to ArchetypeWeights.
fn archetype_name_to_weights(primary: &str, secondary: &str) -> ArchetypeWeights {
    let mut weights = ArchetypeWeights {
        healer: 0.3,
        sage: 0.3,
        guardian: 0.3,
        explorer: 0.3,
    };

    // Primary archetype gets a strong boost (0.8), secondary gets moderate (0.55).
    apply_archetype_weight(&mut weights, primary, 0.8);
    apply_archetype_weight(&mut weights, secondary, 0.55);

    weights
}

/// Apply an archetype name as a weight to the ArchetypeWeights struct.
fn apply_archetype_weight(weights: &mut ArchetypeWeights, name: &str, value: f32) {
    match name {
        "sage" | "magician" => weights.sage = value,
        "hero" | "ruler" | "creator" => {
            // Heroes, rulers, and creators map to a blend.
            weights.sage = weights.sage.max(value * 0.6);
            weights.explorer = weights.explorer.max(value * 0.6);
        }
        "explorer" | "rebel" | "jester" => weights.explorer = value,
        "caregiver" | "innocent" | "orphan" | "lover" => weights.healer = value,
        "guardian" => weights.guardian = value,
        _ => {}
    }
}

/// Extract traits from a description string using keyword matching.
fn extract_traits_from_description(description: &str) -> Vec<String> {
    let lower = description.to_lowercase();
    let all_traits: Vec<&str> = TRAIT_ARCHETYPE.iter().map(|(t, _)| *t).collect();
    let mut found = Vec::new();

    for trait_name in all_traits {
        if lower.contains(trait_name) && !found.contains(&trait_name.to_string()) {
            found.push(trait_name.to_string());
        }
    }

    // If no traits found, add defaults based on common words.
    if found.is_empty() {
        if lower.contains("wisdom") || lower.contains("knowledge") {
            found.push("wise".to_string());
        }
        if lower.contains("craft") || lower.contains("create") || lower.contains("build") {
            found.push("creative".to_string());
        }
        if lower.contains("protect") || lower.contains("guard") {
            found.push("protective".to_string());
        }
    }

    found
}

/// Map a list of traits to archetype names.
fn traits_to_archetypes(traits: &[String]) -> Vec<String> {
    let mut archetypes = Vec::new();
    for trait_name in traits {
        let lower = trait_name.to_lowercase();
        for (t, arch) in TRAIT_ARCHETYPE {
            if *t == lower && !archetypes.contains(&arch.to_string()) {
                archetypes.push(arch.to_string());
            }
        }
    }
    archetypes
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn culture_labels_roundtrip() {
        for culture in [
            CultureOrigin::Egyptian,
            CultureOrigin::Greek,
            CultureOrigin::Norse,
            CultureOrigin::Latin,
            CultureOrigin::Fictional,
            CultureOrigin::Unknown,
        ] {
            let label = culture.as_label();
            assert_eq!(CultureOrigin::from_label(label), Some(culture));
        }
    }

    #[test]
    fn culture_default_is_unknown() {
        assert_eq!(CultureOrigin::default(), CultureOrigin::Unknown);
    }

    #[test]
    fn static_table_ptah() {
        let knowledge = resolve_from_static_tables("Ptah").unwrap();
        assert_eq!(knowledge.entity_type, EntityType::Deity);
        assert_eq!(knowledge.culture, CultureOrigin::Egyptian);
        assert!(knowledge.traits.contains(&"creative".to_string()));
    }

    #[test]
    fn static_table_gandalf() {
        let knowledge = resolve_from_static_tables("Gandalf").unwrap();
        assert_eq!(knowledge.entity_type, EntityType::FictionalCharacter);
        assert_eq!(knowledge.culture, CultureOrigin::Fictional);
        assert!(knowledge.traits.contains(&"wise".to_string()));
    }

    #[test]
    fn static_table_athena() {
        let knowledge = resolve_from_static_tables("Athena").unwrap();
        assert_eq!(knowledge.entity_type, EntityType::Deity);
        assert_eq!(knowledge.culture, CultureOrigin::Greek);
    }

    #[test]
    fn classify_culture_egyptian() {
        assert_eq!(
            classify_culture("Ptah", "Egyptian god of craftsmen"),
            CultureOrigin::Egyptian
        );
    }

    #[test]
    fn classify_culture_greek() {
        assert_eq!(
            classify_culture("Athena", "Greek goddess of wisdom"),
            CultureOrigin::Greek
        );
    }

    #[test]
    fn build_archetype_from_traits() {
        let character = CharacterKnowledge {
            name: "TestSage".to_string(),
            entity_type: EntityType::Concept,
            culture: CultureOrigin::Unknown,
            description: "A wise figure".to_string(),
            domains: vec![],
            traits: vec![
                "wise".to_string(),
                "analytical".to_string(),
                "contemplative".to_string(),
            ],
            archetypes: vec!["sage".to_string()],
        };

        let profile = build_archetype_profile(&character).unwrap();
        assert_eq!(profile.primary, "sage");
    }

    #[test]
    fn build_ocean_from_archetype() {
        let profile = ArchetypeProfile {
            primary: "sage".to_string(),
            secondary: "sage".to_string(),
            weights: ArchetypeWeights {
                healer: 0.3,
                sage: 0.8,
                guardian: 0.3,
                explorer: 0.3,
            },
        };

        let ocean = build_ocean_profile(&profile);
        // Sage: openness=0.9, conscientiousness=0.8.
        assert!(ocean.openness > 0.8);
        assert!(ocean.conscientiousness > 0.7);
    }

    #[test]
    fn build_psyche_sets_name() {
        let character = resolve_from_static_tables("Ptah").unwrap();
        let purpose = PurposeModel {
            domain: "systems".to_string(),
            competence_level: super::super::purpose::DreyfusLevel::Expert,
            seed_concepts: vec![],
            description: "test".to_string(),
        };

        let psyche = build_psyche(&character, &purpose).unwrap();
        assert_eq!(psyche.persona.name, "Ptah");
    }

    #[test]
    fn morpheme_table_egyptian_nonempty() {
        assert!(!EGYPTIAN_MORPHEMES.prefixes.is_empty());
        assert!(!EGYPTIAN_MORPHEMES.roots.is_empty());
        assert!(!EGYPTIAN_MORPHEMES.suffixes.is_empty());
    }

    #[test]
    fn is_pronounceable_alternating() {
        assert!(is_pronounceable("Akhib"));
        assert!(!is_pronounceable("Kkkz"));
        assert!(is_pronounceable("Merib"));
        assert!(!is_pronounceable("Xxxxx"));
    }

    #[test]
    fn name_candidate_scoring() {
        let candidates = generate_candidates(&EGYPTIAN_MORPHEMES, CultureOrigin::Egyptian);
        assert!(!candidates.is_empty());
        // All scores start at 0.0 before VSA scoring.
        for c in &candidates {
            assert!(c.vsa_score >= 0.0 && c.vsa_score <= 1.0);
        }
    }
}
