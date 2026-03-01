//! Purpose & identity parser for operator bootstrap statements (Phase 14a).
//!
//! Parses operator declarations like "You are the Architect of the System based on Ptah"
//! into structured `BootstrapIntent` with domain, competence level, seed concepts,
//! and an optional identity reference to a cultural/historical/fictional figure.

use std::sync::LazyLock;

use miette::Diagnostic;
use regex::Regex;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── Errors ──────────────────────────────────────────────────────────────

/// Errors from the bootstrap purpose parser.
#[derive(Debug, Error, Diagnostic)]
pub enum BootstrapError {
    #[error("empty input — no purpose statement provided")]
    #[diagnostic(
        code(akh::bootstrap::empty_input),
        help("Provide a statement like: \"You are the Architect based on Ptah, expert in systems\"")
    )]
    EmptyInput,

    #[error("no purpose could be extracted from the input")]
    #[diagnostic(
        code(akh::bootstrap::no_purpose),
        help(
            "Include a domain or role description. Examples:\n\
             - \"expert in compilers\"\n\
             - \"You are a knowledge engineer\"\n\
             - \"master of distributed systems based on Athena\""
        )
    )]
    NoPurpose,

    #[error("invalid competence level: \"{level}\"")]
    #[diagnostic(
        code(akh::bootstrap::invalid_competence),
        help("Valid levels: novice, advanced-beginner, competent, proficient, expert")
    )]
    InvalidCompetence { level: String },

    #[error("{0}")]
    #[diagnostic(
        code(akh::bootstrap::engine),
        help("An engine-level error occurred during bootstrap parsing.")
    )]
    Engine(Box<crate::error::AkhError>),
}

impl From<crate::error::AkhError> for BootstrapError {
    fn from(e: crate::error::AkhError) -> Self {
        Self::Engine(Box::new(e))
    }
}

/// Convenience alias.
pub type BootstrapResult<T> = std::result::Result<T, BootstrapError>;

// ── Types ───────────────────────────────────────────────────────────────

/// Dreyfus model of skill acquisition — competence level.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DreyfusLevel {
    #[default]
    Novice,
    AdvancedBeginner,
    Competent,
    Proficient,
    Expert,
}

impl std::fmt::Display for DreyfusLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_label())
    }
}

impl DreyfusLevel {
    pub fn as_label(&self) -> &'static str {
        match self {
            Self::Novice => "novice",
            Self::AdvancedBeginner => "advanced-beginner",
            Self::Competent => "competent",
            Self::Proficient => "proficient",
            Self::Expert => "expert",
        }
    }

    pub fn from_label(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "novice" => Some(Self::Novice),
            "advanced-beginner" | "advanced beginner" => Some(Self::AdvancedBeginner),
            "competent" => Some(Self::Competent),
            "proficient" => Some(Self::Proficient),
            "expert" => Some(Self::Expert),
            _ => None,
        }
    }
}

/// Type of the referenced cultural/historical/fictional entity.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityType {
    Deity,
    FictionalCharacter,
    HistoricalFigure,
    Concept,
    #[default]
    Unknown,
}

impl std::fmt::Display for EntityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_label())
    }
}

impl EntityType {
    pub fn as_label(&self) -> &'static str {
        match self {
            Self::Deity => "deity",
            Self::FictionalCharacter => "fictional-character",
            Self::HistoricalFigure => "historical-figure",
            Self::Concept => "concept",
            Self::Unknown => "unknown",
        }
    }

    pub fn from_label(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "deity" => Some(Self::Deity),
            "fictional-character" | "fictional character" => Some(Self::FictionalCharacter),
            "historical-figure" | "historical figure" => Some(Self::HistoricalFigure),
            "concept" => Some(Self::Concept),
            "unknown" => Some(Self::Unknown),
            _ => None,
        }
    }
}

/// A reference to a cultural, historical, or fictional identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityRef {
    /// Name of the referenced figure (e.g., "Ptah", "Gandalf").
    pub name: String,
    /// Classified entity type.
    pub entity_type: EntityType,
    /// The source phrase that contained this reference.
    pub source_phrase: String,
}

/// Extracted purpose model from the operator statement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurposeModel {
    /// Primary domain of competence (e.g., "systems", "compilers").
    pub domain: String,
    /// Competence level on the Dreyfus scale.
    pub competence_level: DreyfusLevel,
    /// Seed concepts extracted from the statement.
    pub seed_concepts: Vec<String>,
    /// Original or synthesized description of the purpose.
    pub description: String,
}

/// The full parsed bootstrap intent from an operator statement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapIntent {
    /// The extracted purpose model.
    pub purpose: PurposeModel,
    /// Optional identity reference (cultural/historical/fictional figure).
    pub identity: Option<IdentityRef>,
}

// ── Regex Patterns ──────────────────────────────────────────────────────

static RE_BASED_ON: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)based on (\w[\w\s]*\w|\w+)").unwrap());

static RE_LIKE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(?:be |act )like (\w[\w\s]*\w|\w+)").unwrap());

static RE_INSPIRED: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(?:inspired by|in the spirit of) (\w[\w\s]*\w|\w+)").unwrap());

static RE_AS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)you are (?:the |an? )?(\w[\w\s]*\w|\w+)").unwrap());

static RE_DOMAIN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:expert in|specialist in|master of|focused on) (\w[\w\s]*\w|\w+)").unwrap()
});

// ── Static Entity Sets ──────────────────────────────────────────────────

const DEITIES: &[&str] = &[
    "ptah", "ra", "thoth", "anubis", "osiris", "isis", "horus", "sekhmet",
    "bastet", "sobek", "khnum", "atum", "maat", "hathor", "set",
    "athena", "apollo", "hermes", "hephaestus", "zeus", "hera", "ares",
    "artemis", "poseidon", "demeter", "dionysus", "prometheus",
    "odin", "thor", "loki", "freya", "frigg", "tyr", "baldr", "heimdall",
    "minerva", "mars", "jupiter", "neptune", "vulcan",
];

const FICTIONAL: &[&str] = &[
    "gandalf", "dumbledore", "sherlock", "spock", "yoda", "merlin",
    "morpheus", "neo", "cortana", "jarvis", "hal", "data", "r2d2",
    "samwise", "aragorn", "legolas", "hermione", "dr strange",
    "tony stark", "batman", "superman",
];

const HISTORICAL: &[&str] = &[
    "einstein", "turing", "curie", "davinci", "da vinci", "leonardo",
    "newton", "tesla", "lovelace", "babbage", "dijkstra", "knuth",
    "feynman", "hawking", "aristotle", "plato", "socrates",
    "archimedes", "euclid", "gauss", "euler",
];

// ── Parser Functions ────────────────────────────────────────────────────

/// Parse a purpose/identity statement from the operator.
///
/// Extracts domain, competence level, seed concepts, and optional identity
/// reference from natural language declarations like:
/// - "You are the Architect of the System based on Ptah"
/// - "Be like Gandalf — a GCC compiler expert"
/// - "expert in distributed systems"
pub fn parse_purpose(input: &str) -> BootstrapResult<BootstrapIntent> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(BootstrapError::EmptyInput);
    }

    // 1. Extract identity reference via regex cascade (first match wins).
    let identity = extract_identity_ref(trimmed);

    // 2. Extract domain from RE_DOMAIN or infer from statement.
    let domain = extract_domain(trimmed);

    // 3. Extract competence level from keywords.
    let competence_level = extract_competence(trimmed);

    // 4. Extract seed concepts.
    let seed_concepts = extract_seed_concepts(trimmed);

    // Ensure we got at least a domain or identity.
    if domain.is_empty() && identity.is_none() {
        return Err(BootstrapError::NoPurpose);
    }

    let description = trimmed.to_string();

    Ok(BootstrapIntent {
        purpose: PurposeModel {
            domain,
            competence_level,
            seed_concepts,
            description,
        },
        identity,
    })
}

/// Classify entity type from a name by checking static sets.
pub fn classify_entity_type(name: &str) -> EntityType {
    let lower = name.to_lowercase();

    // Check multi-word matches and single-word matches.
    if DEITIES.iter().any(|d| lower == *d) {
        return EntityType::Deity;
    }
    if FICTIONAL.iter().any(|f| lower == *f) {
        return EntityType::FictionalCharacter;
    }
    if HISTORICAL.iter().any(|h| lower == *h) {
        return EntityType::HistoricalFigure;
    }

    EntityType::Unknown
}

/// Extract competence level from keywords in the input.
pub fn extract_competence(input: &str) -> DreyfusLevel {
    let lower = input.to_lowercase();

    if lower.contains("expert") || lower.contains("master") {
        DreyfusLevel::Expert
    } else if lower.contains("proficient") || lower.contains("advanced") {
        DreyfusLevel::Proficient
    } else if lower.contains("competent") || lower.contains("skilled") {
        DreyfusLevel::Competent
    } else if lower.contains("beginner") || lower.contains("novice") || lower.contains("learning")
    {
        DreyfusLevel::Novice
    } else {
        // Default: if a domain is mentioned, assume Competent; otherwise Novice.
        if RE_DOMAIN.is_match(&lower) {
            DreyfusLevel::Competent
        } else {
            DreyfusLevel::Novice
        }
    }
}

// ── Internal Helpers ────────────────────────────────────────────────────

/// Try to extract an identity reference from the input.
fn extract_identity_ref(input: &str) -> Option<IdentityRef> {
    // Try patterns in priority order: based_on > like > inspired > as.
    let (name, source_phrase) = if let Some(caps) = RE_BASED_ON.captures(input) {
        let name = caps[1].trim().to_string();
        let phrase = caps[0].to_string();
        (name, phrase)
    } else if let Some(caps) = RE_LIKE.captures(input) {
        let name = caps[1].trim().to_string();
        let phrase = caps[0].to_string();
        (name, phrase)
    } else if let Some(caps) = RE_INSPIRED.captures(input) {
        let name = caps[1].trim().to_string();
        let phrase = caps[0].to_string();
        (name, phrase)
    } else if let Some(caps) = RE_AS.captures(input) {
        // Only use "you are X" if X looks like a proper noun (starts uppercase).
        let name = caps[1].trim().to_string();
        if name.chars().next().is_some_and(|c| c.is_uppercase())
            && classify_entity_type(&name) != EntityType::Unknown
        {
            let phrase = caps[0].to_string();
            (name, phrase)
        } else {
            return None;
        }
    } else {
        return None;
    };

    let entity_type = classify_entity_type(&name);

    Some(IdentityRef {
        name,
        entity_type,
        source_phrase,
    })
}

/// Extract the domain from the input using RE_DOMAIN or heuristic fallback.
fn extract_domain(input: &str) -> String {
    if let Some(caps) = RE_DOMAIN.captures(input) {
        return caps[1].trim().to_string();
    }

    // Heuristic: look for "of X" or "in X" after role words.
    let lower = input.to_lowercase();
    for prefix in &["architect of ", "builder of ", "engineer of "] {
        if let Some(idx) = lower.find(prefix) {
            let rest = &input[idx + prefix.len()..];
            let domain = rest
                .split([',', '.', ';'])
                .next()
                .unwrap_or("")
                .trim();
            if !domain.is_empty() {
                return domain.to_string();
            }
        }
    }

    String::new()
}

/// Extract seed concepts from the input: capitalized words and domain nouns.
fn extract_seed_concepts(input: &str) -> Vec<String> {
    let mut concepts = Vec::new();

    // Collect capitalized words (skip sentence-initial position).
    let words: Vec<&str> = input.split_whitespace().collect();
    for (i, word) in words.iter().enumerate() {
        if i == 0 {
            continue;
        }
        let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
        if clean.len() >= 2
            && clean.chars().next().is_some_and(|c| c.is_uppercase())
            && !is_stopword(clean)
        {
            // Skip identity-reference names (they're tracked separately).
            concepts.push(clean.to_string());
        }
    }

    // Add domain keywords from RE_DOMAIN.
    if let Some(caps) = RE_DOMAIN.captures(input) {
        let domain = caps[1].trim();
        for word in domain.split_whitespace() {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
            if clean.len() >= 2 && !concepts.iter().any(|c| c.eq_ignore_ascii_case(clean)) {
                concepts.push(clean.to_string());
            }
        }
    }

    concepts
}

/// Check if a word is a common stopword (articles, prepositions, etc.).
fn is_stopword(word: &str) -> bool {
    matches!(
        word.to_lowercase().as_str(),
        "the" | "a" | "an" | "of" | "in" | "on" | "at" | "to" | "for"
            | "is" | "are" | "was" | "be" | "been"
            | "and" | "or" | "not" | "but"
            | "you" | "your" | "like" | "based"
            | "with" | "from" | "by" | "as"
    )
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_empty_input_error() {
        let result = parse_purpose("");
        assert!(matches!(result, Err(BootstrapError::EmptyInput)));
    }

    #[test]
    fn parse_based_on_deity() {
        let intent = parse_purpose("You are the Architect based on Ptah, expert in systems")
            .unwrap();
        let id = intent.identity.unwrap();
        assert_eq!(id.name, "Ptah");
        assert_eq!(id.entity_type, EntityType::Deity);
    }

    #[test]
    fn parse_like_fictional() {
        let intent = parse_purpose("Be like Gandalf, a compiler expert").unwrap();
        let id = intent.identity.unwrap();
        assert_eq!(id.name, "Gandalf");
        assert_eq!(id.entity_type, EntityType::FictionalCharacter);
    }

    #[test]
    fn parse_domain_extraction() {
        let intent = parse_purpose("expert in compilers and type systems").unwrap();
        assert!(intent.purpose.domain.contains("compilers"));
    }

    #[test]
    fn parse_competence_level() {
        assert_eq!(extract_competence("a beginner in Rust"), DreyfusLevel::Novice);
        assert_eq!(extract_competence("expert in systems"), DreyfusLevel::Expert);
        assert_eq!(extract_competence("master of compilers"), DreyfusLevel::Expert);
        assert_eq!(extract_competence("proficient developer"), DreyfusLevel::Proficient);
    }

    #[test]
    fn parse_seed_concepts() {
        let concepts = extract_seed_concepts("Study the System Architecture based on Egyptian methods");
        assert!(concepts.iter().any(|c| c == "System"));
        assert!(concepts.iter().any(|c| c == "Architecture"));
        assert!(concepts.iter().any(|c| c == "Egyptian"));
    }

    #[test]
    fn parse_combined_statement() {
        let intent =
            parse_purpose("You are the Architect based on Ptah, expert in systems").unwrap();
        assert!(intent.identity.is_some());
        assert_eq!(intent.purpose.competence_level, DreyfusLevel::Expert);
        assert!(intent.purpose.domain.contains("systems"));
    }

    #[test]
    fn dreyfus_labels_roundtrip() {
        for level in [
            DreyfusLevel::Novice,
            DreyfusLevel::AdvancedBeginner,
            DreyfusLevel::Competent,
            DreyfusLevel::Proficient,
            DreyfusLevel::Expert,
        ] {
            let label = level.as_label();
            assert_eq!(DreyfusLevel::from_label(label), Some(level));
        }
    }

    #[test]
    fn entity_type_labels_roundtrip() {
        for et in [
            EntityType::Deity,
            EntityType::FictionalCharacter,
            EntityType::HistoricalFigure,
            EntityType::Concept,
            EntityType::Unknown,
        ] {
            let label = et.as_label();
            assert_eq!(EntityType::from_label(label), Some(et));
        }
    }

    #[test]
    fn classify_deity_names() {
        assert_eq!(classify_entity_type("Ptah"), EntityType::Deity);
        assert_eq!(classify_entity_type("Thoth"), EntityType::Deity);
        assert_eq!(classify_entity_type("Athena"), EntityType::Deity);
        assert_eq!(classify_entity_type("Odin"), EntityType::Deity);
    }

    #[test]
    fn classify_fictional_names() {
        assert_eq!(classify_entity_type("Gandalf"), EntityType::FictionalCharacter);
        assert_eq!(classify_entity_type("Sherlock"), EntityType::FictionalCharacter);
        assert_eq!(classify_entity_type("Spock"), EntityType::FictionalCharacter);
    }

    #[test]
    fn parse_no_identity_just_purpose() {
        let intent = parse_purpose("expert in AI and machine learning").unwrap();
        assert!(intent.identity.is_none());
        assert!(!intent.purpose.domain.is_empty());
    }
}
