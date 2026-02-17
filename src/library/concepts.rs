//! Atomic concept extraction from document text.
//!
//! Provides two-phase concept extraction:
//! - **Phase A (relational):** Head noun phrase extraction trims sentence-fragment
//!   spans from relational patterns ("X is-a Y") down to atomic concepts.
//! - **Phase B (standalone):** Scans for capitalized proper nouns, technical
//!   compounds, and repeated terms not captured by relational patterns.

use std::collections::{HashMap, HashSet};

/// An extracted atomic concept from document text.
#[derive(Debug, Clone)]
pub struct ExtractedConcept {
    pub label: String,
    pub confidence: f32,
    pub source: ConceptSource,
}

/// How a concept was discovered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConceptSource {
    /// Head noun from an "X is-a Y" relational pattern.
    RelationalHead,
    /// Mid-sentence capitalized word sequence (e.g., "Carl Jung").
    CapitalizedTerm,
    /// Hyphenated compound (e.g., "self-knowledge").
    TechnicalTerm,
    /// Non-stopword appearing 3+ times in chunk.
    RepeatedTerm,
}

impl ConceptSource {
    /// Provenance extraction method string.
    pub fn method_str(&self) -> &'static str {
        match self {
            Self::RelationalHead => "relational_head",
            Self::CapitalizedTerm => "capitalized",
            Self::TechnicalTerm => "technical",
            Self::RepeatedTerm => "repeated",
        }
    }
}

// ---------------------------------------------------------------------------
// Stopword / function-word sets
// ---------------------------------------------------------------------------

const DETERMINERS: &[&str] = &[
    "the", "a", "an", "this", "that", "these", "those", "some", "any",
];

const PRONOUNS: &[&str] = &[
    "it", "he", "she", "they", "we", "one", "you", "its", "his", "her", "their",
];

const FILLER_ADVERBS: &[&str] = &[
    "very", "quite", "rather", "amply", "clearly", "obviously", "also", "indeed",
    "simply", "just",
];

const CLAUSE_BOUNDARIES: &[&str] = &[
    "that", "which", "who", "whom", "where", "when", "while", "because",
    "although", "since", "if", "unless",
];

const TRAILING_AUX_VERBS: &[&str] = &[
    "is", "are", "was", "were", "has", "have", "had", "been", "being", "does", "do",
];

const TRAILING_PARTICIPLES: &[&str] = &[
    "shown", "said", "called", "known", "considered", "given",
];

/// Common verbs that indicate structure rather than content when scanning
/// right-to-left for the rightmost noun cluster.
const VERB_STOPS: &[&str] = &[
    "is", "are", "was", "were", "has", "have", "had", "been", "being",
    "does", "do", "shown", "said", "called", "known", "considered", "given",
    "agree", "resemble", "oppose", "complement", "influence", "produce",
    "require", "enable", "prevent", "seldom", "rarely", "often", "usually",
    "sometimes", "occasionally", "always", "never", "frequently", "invariably",
];

/// Preposition/conjunction connectors allowed inside noun clusters.
const NOUN_CONNECTORS: &[&str] = &["of", "and", "in", "for"];

const STOPWORDS: &[&str] = &[
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "this", "that", "these", "those",
    "it", "its", "he", "she", "they", "we", "you", "his", "her", "their",
    "my", "your", "our", "not", "no", "so", "if", "as", "up", "out", "about",
    "into", "over", "after", "than", "then", "just", "also", "very", "quite",
    "rather", "some", "any", "each", "all", "more", "most", "such", "only",
    "same", "other", "own",
];

// ---------------------------------------------------------------------------
// Head noun phrase extraction
// ---------------------------------------------------------------------------

/// Trim a sentence-fragment word span down to its core concept phrase.
///
/// Preserves adjectives (they carry semantic meaning) but strips determiners,
/// pronouns, filler adverbs, clause tails, and trailing auxiliary verbs.
pub fn extract_head_noun_phrase(words: &[&str]) -> Option<String> {
    if words.is_empty() {
        return None;
    }

    // Step 1: Truncate at first clause boundary — take only words before.
    let clause_end = words.iter().position(|w| {
        let low = w.to_lowercase();
        CLAUSE_BOUNDARIES.contains(&low.as_str())
    });
    let words = match clause_end {
        Some(0) => return None,
        Some(pos) => &words[..pos],
        None => words,
    };

    // Step 2: Strip leading determiners, pronouns, filler adverbs.
    let start = leading_skip_count(words);
    if start >= words.len() {
        return None;
    }
    let words = &words[start..];

    // Step 3: Strip trailing auxiliary/linking verbs.
    let words = strip_trailing(words, TRAILING_AUX_VERBS);
    if words.is_empty() {
        return None;
    }

    // Step 4: Strip trailing past participles used as passives.
    let words = strip_trailing(words, TRAILING_PARTICIPLES);
    if words.is_empty() {
        return None;
    }

    // Step 5: Count content words.
    let content_count = words
        .iter()
        .filter(|w| {
            let low = w.to_lowercase();
            !NOUN_CONNECTORS.contains(&low.as_str())
        })
        .count();

    if content_count == 0 {
        return None;
    }

    // Step 6: If >5 content words remain, take rightmost noun cluster.
    let result_words = if content_count > 5 {
        rightmost_noun_cluster(words)
    } else {
        words.to_vec()
    };

    if result_words.is_empty() {
        return None;
    }

    // Capitalize first letter.
    let joined = result_words.join(" ");
    let capitalized = capitalize_first(&joined);
    Some(capitalized)
}

/// Count leading words that are determiners, pronouns, or filler adverbs.
fn leading_skip_count(words: &[&str]) -> usize {
    let mut count = 0;
    for w in words {
        let low = w.to_lowercase();
        if DETERMINERS.contains(&low.as_str())
            || PRONOUNS.contains(&low.as_str())
            || FILLER_ADVERBS.contains(&low.as_str())
        {
            count += 1;
        } else {
            break;
        }
    }
    count
}

/// Strip trailing words that match a given set (case-insensitive).
fn strip_trailing<'a, 'b>(words: &'a [&'b str], set: &[&str]) -> &'a [&'b str] {
    let mut end = words.len();
    while end > 0 {
        let low = words[end - 1].to_lowercase();
        if set.contains(&low.as_str()) {
            end -= 1;
        } else {
            break;
        }
    }
    &words[..end]
}

/// Extract the rightmost noun cluster from a word span.
///
/// Scans right-to-left, keeping content words + connectors ("of", "and", "in"),
/// stopping at verbs. Caps result at 5 words.
fn rightmost_noun_cluster<'a>(words: &[&'a str]) -> Vec<&'a str> {
    let mut cluster: Vec<&'a str> = Vec::new();
    for &w in words.iter().rev() {
        let low = w.to_lowercase();
        if VERB_STOPS.contains(&low.as_str()) {
            break;
        }
        cluster.push(w);
        // Cap at 5 words.
        if cluster.len() >= 5 {
            break;
        }
    }
    cluster.reverse();
    cluster
}

fn capitalize_first(s: &str) -> String {
    let s = s.trim();
    if s.is_empty() {
        return String::new();
    }
    let mut chars = s.chars();
    let first = chars.next().unwrap().to_uppercase().to_string();
    first + chars.as_str()
}

// ---------------------------------------------------------------------------
// Richer triple extraction
// ---------------------------------------------------------------------------

/// A triple extracted with richer semantics.
pub type ExtractedTriple = (String, String, String, f32);

/// Frequency/intensity adverb → confidence weight.
fn adverb_weight(adverb: &str) -> Option<f32> {
    match adverb.to_lowercase().as_str() {
        "always" | "invariably" => Some(1.0),
        "often" | "frequently" | "usually" => Some(0.75),
        "sometimes" | "occasionally" => Some(0.45),
        "seldom" | "rarely" => Some(0.2),
        "never" => Some(0.05),
        _ => None,
    }
}

/// Bare relational verbs beyond is-a/has-a.
const BARE_VERBS: &[&str] = &[
    "agree", "resemble", "oppose", "complement", "influence",
    "produce", "require", "enable", "prevent",
];

/// Check if a word matches a bare verb (base or 3rd-person-singular form).
fn matches_bare_verb(word: &str, verb: &str) -> bool {
    if word == verb {
        return true;
    }
    // Match 3rd-person forms: "agrees" for "agree", "influences" for "influence",
    // "produces" for "produce", "resembles" for "resemble", "opposes" for "oppose".
    if let Some(stem) = word.strip_suffix('s') {
        if stem == verb {
            return true;
        }
        // "es" suffix: "produces" → strip "es" → "produc" != "produce", but
        // strip "s" → "produce" == "produce" ✓ (already handled above).
        // Handle verbs ending in 'e': "influences" → strip "s" → "influence" ✓.
    }
    false
}

/// Extended triple extraction with conjunction splitting, frequency modulation,
/// and bare verb patterns.
pub fn extract_richer_triples(sentence: &str) -> Vec<ExtractedTriple> {
    let mut results = Vec::new();
    let s = sentence
        .trim()
        .trim_end_matches(|c: char| c == '.' || c == '!' || c == '?');
    let words: Vec<&str> = s.split_whitespace().collect();
    if words.len() < 3 {
        return results;
    }
    let lower: Vec<String> = words.iter().map(|w| w.to_lowercase()).collect();

    // --- Bare verb patterns ---
    for verb in BARE_VERBS {
        for (i, lw) in lower.iter().enumerate() {
            if matches_bare_verb(lw, verb) && i > 0 && i + 1 < words.len() {
                let base_conf = 0.7f32;

                // Check for adverb immediately before the verb.
                let (subj_end, conf) = if i >= 2 {
                    if let Some(weight) = adverb_weight(&lower[i - 1]) {
                        (i - 1, base_conf * weight)
                    } else {
                        (i, base_conf)
                    }
                } else {
                    (i, base_conf)
                };

                // Check for adverb immediately after the verb.
                let (obj_start, conf) = if i + 2 < words.len() {
                    if let Some(weight) = adverb_weight(&lower[i + 1]) {
                        (i + 2, base_conf * weight)
                    } else {
                        (i + 1, conf)
                    }
                } else {
                    (i + 1, conf)
                };

                let subj_words = &words[..subj_end];
                let obj_words = &words[obj_start..];

                if subj_words.is_empty() || obj_words.is_empty() {
                    continue;
                }

                // Try conjunction splitting on subject.
                let subjects = split_conjunction(subj_words);
                for subj_span in &subjects {
                    if let Some(subj) = extract_head_noun_phrase(subj_span) {
                        if let Some(obj) = extract_head_noun_phrase(obj_words) {
                            results.push((subj, verb.to_string(), obj, conf));
                        }
                    }
                }
                break; // Only match first occurrence of this verb.
            }
        }
    }

    // --- Conjunction splitting for existing relational patterns ---
    // Detect "X and Y is/are/has Z" patterns.
    extract_conjunction_relational(&words, &lower, &mut results);

    results
}

/// Split a word span at "and"/"or" to produce multiple sub-spans.
fn split_conjunction<'a>(words: &[&'a str]) -> Vec<Vec<&'a str>> {
    let mut groups: Vec<Vec<&'a str>> = Vec::new();
    let mut current: Vec<&str> = Vec::new();

    for &w in words {
        let low = w.to_lowercase();
        if low == "and" || low == "or" {
            if !current.is_empty() {
                groups.push(current.clone());
                current.clear();
            }
        } else {
            current.push(w);
        }
    }
    if !current.is_empty() {
        groups.push(current);
    }
    if groups.is_empty() {
        groups.push(words.to_vec());
    }
    groups
}

/// Detect "X and Y is/are/has [a/an] Z" and produce split triples.
fn extract_conjunction_relational(
    words: &[&str],
    lower: &[String],
    results: &mut Vec<ExtractedTriple>,
) {
    // Find "and"/"or" in subject position (before a linking verb).
    let conj_patterns: &[(&[&str], &str, f32)] = &[
        (&["is", "a"], "is-a", 0.9),
        (&["is", "an"], "is-a", 0.9),
        (&["are"], "is-a", 0.85),
        (&["has", "a"], "has-a", 0.85),
        (&["has", "an"], "has-a", 0.85),
        (&["have", "a"], "has-a", 0.85),
        (&["have", "an"], "has-a", 0.85),
        (&["have"], "has-a", 0.85),
    ];

    for &(pattern, predicate, base_conf) in conj_patterns {
        let plen = pattern.len();
        if lower.len() < plen + 2 {
            continue;
        }
        for i in 1..lower.len().saturating_sub(plen) {
            let matches = pattern.iter().enumerate().all(|(j, p)| {
                i + j < lower.len() && lower[i + j] == *p
            });
            if !matches {
                continue;
            }

            let subj_words = &words[..i];
            let obj_start = i + plen;
            if obj_start >= words.len() {
                continue;
            }
            let obj_words = &words[obj_start..];

            // Only process if subject contains "and"/"or".
            let has_conj = subj_words
                .iter()
                .any(|w| w.eq_ignore_ascii_case("and") || w.eq_ignore_ascii_case("or"));
            if !has_conj {
                continue;
            }

            let subjects = split_conjunction(subj_words);
            if subjects.len() < 2 {
                continue;
            }

            // Check for adverb modulation.
            let conf = if i >= 2 {
                if let Some(weight) = adverb_weight(&lower[i - 1]) {
                    // The adverb is part of subject, adjust conf but don't re-strip.
                    base_conf * weight
                } else {
                    base_conf
                }
            } else {
                base_conf
            };

            for subj_span in &subjects {
                if let Some(subj) = extract_head_noun_phrase(subj_span) {
                    if let Some(obj) = extract_head_noun_phrase(obj_words) {
                        results.push((subj, predicate.to_string(), obj, conf));
                    }
                }
            }
            return; // Found a pattern, done.
        }
    }
}

// ---------------------------------------------------------------------------
// Standalone concept extraction
// ---------------------------------------------------------------------------

/// Find concepts NOT already captured by relational extraction.
///
/// Returns capitalized proper nouns, technical compounds (hyphenated), and
/// repeated significant terms.
pub fn extract_concepts_from_chunk(
    text: &str,
    relational_labels: &HashSet<String>,
) -> Vec<ExtractedConcept> {
    let mut concepts = Vec::new();
    let normalized_relational: HashSet<String> = relational_labels
        .iter()
        .map(|l| l.to_lowercase())
        .collect();

    // --- Capitalized sequences (confidence 0.8) ---
    for sentence in text.split(|c: char| c == '.' || c == '!' || c == '?') {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        if words.len() < 2 {
            continue;
        }
        // Skip first word (sentence-initial capitalization).
        let mut i = 1;
        while i < words.len() {
            let w = words[i].trim_matches(|c: char| !c.is_alphanumeric());
            if !w.is_empty() && w.chars().next().map_or(false, |c| c.is_uppercase())
                && w.len() > 1
                && !is_stopword(w)
            {
                // Gather consecutive capitalized words (max 3).
                let start = i;
                let mut end = i + 1;
                while end < words.len() && end - start < 3 {
                    let next = words[end].trim_matches(|c: char| !c.is_alphanumeric());
                    if !next.is_empty()
                        && next.chars().next().map_or(false, |c| c.is_uppercase())
                    {
                        end += 1;
                    } else {
                        break;
                    }
                }
                let label: String = words[start..end]
                    .iter()
                    .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
                    .collect::<Vec<_>>()
                    .join(" ");
                if !label.is_empty() && !normalized_relational.contains(&label.to_lowercase()) {
                    concepts.push(ExtractedConcept {
                        label,
                        confidence: 0.8,
                        source: ConceptSource::CapitalizedTerm,
                    });
                }
                i = end;
            } else {
                i += 1;
            }
        }
    }

    // --- Technical compounds (confidence 0.75) ---
    for word in text.split_whitespace() {
        let cleaned = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '-');
        if let Some(dash_pos) = cleaned.find('-') {
            let left = &cleaned[..dash_pos];
            let right = &cleaned[dash_pos + 1..];
            if left.len() >= 3
                && right.len() >= 3
                && left.chars().all(|c| c.is_alphabetic())
                && right.chars().all(|c| c.is_alphabetic())
            {
                let label = capitalize_first(cleaned);
                if !normalized_relational.contains(&label.to_lowercase()) {
                    concepts.push(ExtractedConcept {
                        label,
                        confidence: 0.75,
                        source: ConceptSource::TechnicalTerm,
                    });
                }
            }
        }
    }

    // --- Repeated terms (confidence 0.7) ---
    let mut freq: HashMap<String, usize> = HashMap::new();
    for word in text.split_whitespace() {
        let cleaned = word
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase();
        if cleaned.len() >= 3 && !is_stopword(&cleaned) {
            *freq.entry(cleaned).or_insert(0) += 1;
        }
    }
    for (term, count) in &freq {
        if *count >= 3 && !normalized_relational.contains(term) {
            let label = capitalize_first(term);
            concepts.push(ExtractedConcept {
                label,
                confidence: 0.7,
                source: ConceptSource::RepeatedTerm,
            });
        }
    }

    // --- Dedup by normalized label ---
    let mut seen: HashSet<String> = HashSet::new();
    concepts.retain(|c| seen.insert(c.label.to_lowercase()));

    concepts
}

fn is_stopword(word: &str) -> bool {
    STOPWORDS.contains(&word.to_lowercase().as_str())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- extract_head_noun_phrase --

    #[test]
    fn head_noun_strips_determiner() {
        let words = vec!["The", "collective", "unconscious"];
        assert_eq!(
            extract_head_noun_phrase(&words),
            Some("Collective unconscious".into())
        );
    }

    #[test]
    fn head_noun_preserves_adjective() {
        let words = vec!["dark", "aspect", "of", "personality"];
        assert_eq!(
            extract_head_noun_phrase(&words),
            Some("Dark aspect of personality".into())
        );
    }

    #[test]
    fn head_noun_clause_truncation() {
        let words = vec![
            "Amply", "shown", "that", "the", "conscious", "and", "the",
            "unconscious", "seldom", "agree",
        ];
        // Truncate at "that" → ["Amply", "shown"]
        // Strip "Amply" (filler) → ["shown"]
        // Strip "shown" (participle) → empty → None
        assert_eq!(extract_head_noun_phrase(&words), None);
    }

    #[test]
    fn head_noun_process_phrase() {
        let words = vec!["A", "process", "of", "psychological", "differentiation"];
        assert_eq!(
            extract_head_noun_phrase(&words),
            Some("Process of psychological differentiation".into())
        );
    }

    #[test]
    fn head_noun_single_word() {
        let words = vec!["Shadow"];
        assert_eq!(
            extract_head_noun_phrase(&words),
            Some("Shadow".into())
        );
    }

    #[test]
    fn head_noun_empty_input() {
        let words: Vec<&str> = vec![];
        assert_eq!(extract_head_noun_phrase(&words), None);
    }

    #[test]
    fn head_noun_all_stopwords() {
        let words = vec!["the", "a", "an"];
        assert_eq!(extract_head_noun_phrase(&words), None);
    }

    #[test]
    fn head_noun_strips_trailing_verb() {
        let words = vec!["the", "shadow", "is"];
        assert_eq!(
            extract_head_noun_phrase(&words),
            Some("Shadow".into())
        );
    }

    #[test]
    fn head_noun_long_span_takes_rightmost_cluster() {
        // >5 content words: should take rightmost noun cluster.
        let words = vec![
            "very", "complex", "internal", "structure", "of", "the",
            "deeply", "buried", "psychological", "archetype", "pattern",
            "of", "individuation",
        ];
        let result = extract_head_noun_phrase(&words);
        assert!(result.is_some());
        // After stripping "very" (filler), 12 words remain with >5 content words.
        // Rightmost cluster should be capped at 5 words.
        let r = result.unwrap();
        assert!(r.split_whitespace().count() <= 5);
    }

    // -- extract_richer_triples --

    #[test]
    fn richer_bare_verb_extraction() {
        let triples = extract_richer_triples("Shadow influences behavior");
        assert!(!triples.is_empty());
        let (subj, pred, obj, conf) = &triples[0];
        assert_eq!(subj, "Shadow");
        assert_eq!(pred, "influence");
        assert_eq!(obj, "Behavior");
        assert!(*conf > 0.6 && *conf <= 0.7);
    }

    #[test]
    fn richer_adverb_modulation() {
        let triples = extract_richer_triples("Conscious seldom agree Unconscious");
        assert!(!triples.is_empty());
        let (_, _, _, conf) = &triples[0];
        // 0.7 * 0.2 = 0.14
        assert!(*conf < 0.2);
    }

    #[test]
    fn richer_conjunction_splitting() {
        let triples =
            extract_richer_triples("Anima and Animus are contrasexual archetypes");
        // Should produce 2 triples: one for Anima, one for Animus.
        assert!(triples.len() >= 2);
        let labels: Vec<&str> = triples.iter().map(|(s, _, _, _)| s.as_str()).collect();
        assert!(labels.contains(&"Anima"));
        assert!(labels.contains(&"Animus"));
    }

    // -- extract_concepts_from_chunk --

    #[test]
    fn concepts_capitalized_proper_noun() {
        let text = "According to Carl Jung, the psyche has structure.";
        let relational = HashSet::new();
        let concepts = extract_concepts_from_chunk(text, &relational);
        let labels: Vec<&str> = concepts.iter().map(|c| c.label.as_str()).collect();
        assert!(labels.iter().any(|l| l.contains("Carl") || l.contains("Jung")));
    }

    #[test]
    fn concepts_technical_compound() {
        let text = "The idea of self-knowledge is central to depth psychology.";
        let relational = HashSet::new();
        let concepts = extract_concepts_from_chunk(text, &relational);
        let labels: Vec<&str> = concepts.iter().map(|c| c.label.as_str()).collect();
        assert!(labels.iter().any(|l| l.to_lowercase().contains("self-knowledge")));
    }

    #[test]
    fn concepts_repeated_term() {
        let text = "Shadow appears when shadow is projected. The shadow concept is vital.";
        let relational = HashSet::new();
        let concepts = extract_concepts_from_chunk(text, &relational);
        let labels: Vec<&str> = concepts.iter().map(|c| c.label.as_str()).collect();
        assert!(labels.iter().any(|l| l.to_lowercase() == "shadow"));
    }

    #[test]
    fn concepts_dedup_with_relational() {
        let text = "Shadow appears when shadow is projected. The shadow concept is vital.";
        let mut relational = HashSet::new();
        relational.insert("Shadow".to_string());
        let concepts = extract_concepts_from_chunk(text, &relational);
        let labels: Vec<&str> = concepts.iter().map(|c| c.label.as_str()).collect();
        // "Shadow" should be excluded since it's already in relational set.
        assert!(!labels.iter().any(|l| l.to_lowercase() == "shadow"));
    }

    #[test]
    fn concepts_empty_input() {
        let concepts = extract_concepts_from_chunk("", &HashSet::new());
        assert!(concepts.is_empty());
    }

    #[test]
    fn concepts_all_caps_acronym() {
        let text = "The SSH tunnel provides secure access. Use SSH for remote connections.";
        let relational = HashSet::new();
        let concepts = extract_concepts_from_chunk(text, &relational);
        let labels: Vec<&str> = concepts.iter().map(|c| c.label.as_str()).collect();
        assert!(labels.iter().any(|l| l.contains("SSH")));
    }
}
