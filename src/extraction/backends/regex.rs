//! Regex-based triple extraction fallback.
//!
//! Always available, no model required. Extracts basic patterns like
//! "X is a Y", "X is part of Y", "X has Y", "X causes Y".

use crate::extraction::RawTriple;

/// Extract triples from text using regex patterns.
pub fn extract_regex(text: &str) -> Vec<RawTriple> {
    let mut triples = Vec::new();

    for sentence in split_sentences(text) {
        let sentence = sentence.trim();
        if sentence.len() < 5 {
            continue;
        }

        // "X is a Y" / "X is an Y"
        if let Some(triple) = match_is_a(sentence) {
            triples.push(triple);
        }

        // "X is part of Y"
        if let Some(triple) = match_part_of(sentence) {
            triples.push(triple);
        }

        // "X has Y" / "X has a Y"
        if let Some(triple) = match_has(sentence) {
            triples.push(triple);
        }

        // "X causes Y" / "X leads to Y"
        if let Some(triple) = match_causes(sentence) {
            triples.push(triple);
        }

        // "X contains Y"
        if let Some(triple) = match_contains(sentence) {
            triples.push(triple);
        }

        // "X supports Y"
        if let Some(triple) = match_supports(sentence) {
            triples.push(triple);
        }
    }

    triples
}

fn split_sentences(text: &str) -> Vec<&str> {
    text.split(['.', '!', '?'])
        .filter(|s| !s.trim().is_empty())
        .collect()
}

fn match_is_a(sentence: &str) -> Option<RawTriple> {
    // "The X is a Y" / "X is an Y" / "X are Y"
    let patterns = [" is a ", " is an ", " are "];
    for pat in &patterns {
        if let Some(pos) = sentence.to_lowercase().find(pat) {
            let subject = clean_noun_phrase(&sentence[..pos]);
            let object = clean_noun_phrase(&sentence[pos + pat.len()..]);
            if !subject.is_empty() && !object.is_empty() {
                return Some(RawTriple {
                    subject,
                    predicate: "is-a".to_string(),
                    object,
                    confidence: 0.7,
                });
            }
        }
    }
    None
}

fn match_part_of(sentence: &str) -> Option<RawTriple> {
    let lower = sentence.to_lowercase();
    let pat = " is part of ";
    if let Some(pos) = lower.find(pat) {
        let subject = clean_noun_phrase(&sentence[..pos]);
        let object = clean_noun_phrase(&sentence[pos + pat.len()..]);
        if !subject.is_empty() && !object.is_empty() {
            return Some(RawTriple {
                subject,
                predicate: "part-of".to_string(),
                object,
                confidence: 0.7,
            });
        }
    }
    None
}

fn match_has(sentence: &str) -> Option<RawTriple> {
    let lower = sentence.to_lowercase();
    for pat in [" has a ", " has an ", " has "] {
        if let Some(pos) = lower.find(pat) {
            let subject = clean_noun_phrase(&sentence[..pos]);
            let object = clean_noun_phrase(&sentence[pos + pat.len()..]);
            if !subject.is_empty() && !object.is_empty() {
                return Some(RawTriple {
                    subject,
                    predicate: "has-a".to_string(),
                    object,
                    confidence: 0.6,
                });
            }
        }
    }
    None
}

fn match_causes(sentence: &str) -> Option<RawTriple> {
    let lower = sentence.to_lowercase();
    for pat in [" causes ", " leads to ", " results in "] {
        if let Some(pos) = lower.find(pat) {
            let subject = clean_noun_phrase(&sentence[..pos]);
            let object = clean_noun_phrase(&sentence[pos + pat.len()..]);
            if !subject.is_empty() && !object.is_empty() {
                return Some(RawTriple {
                    subject,
                    predicate: "causes".to_string(),
                    object,
                    confidence: 0.6,
                });
            }
        }
    }
    None
}

fn match_contains(sentence: &str) -> Option<RawTriple> {
    let lower = sentence.to_lowercase();
    if let Some(pos) = lower.find(" contains ") {
        let subject = clean_noun_phrase(&sentence[..pos]);
        let object = clean_noun_phrase(&sentence[pos + " contains ".len()..]);
        if !subject.is_empty() && !object.is_empty() {
            return Some(RawTriple {
                subject,
                predicate: "contains".to_string(),
                object,
                confidence: 0.6,
            });
        }
    }
    None
}

fn match_supports(sentence: &str) -> Option<RawTriple> {
    let lower = sentence.to_lowercase();
    if let Some(pos) = lower.find(" supports ") {
        let subject = clean_noun_phrase(&sentence[..pos]);
        let object = clean_noun_phrase(&sentence[pos + " supports ".len()..]);
        if !subject.is_empty() && !object.is_empty() {
            return Some(RawTriple {
                subject,
                predicate: "supports".to_string(),
                object,
                confidence: 0.6,
            });
        }
    }
    None
}

/// Clean a noun phrase: strip articles, trim whitespace.
fn clean_noun_phrase(raw: &str) -> String {
    let trimmed = raw.trim();
    let lower = trimmed.to_lowercase();

    // Strip leading articles
    let stripped = if lower.starts_with("the ") {
        &trimmed[4..]
    } else if lower.starts_with("a ") {
        &trimmed[2..]
    } else if lower.starts_with("an ") {
        &trimmed[3..]
    } else {
        trimmed
    };

    stripped.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_is_a() {
        let triples = extract_regex("The pyramid is a structure. The temple is a building.");
        assert!(triples.iter().any(|t| t.subject == "pyramid" && t.object == "structure"));
        assert!(triples.iter().any(|t| t.subject == "temple" && t.object == "building"));
    }

    #[test]
    fn extract_part_of() {
        let triples = extract_regex("The foundation is part of the building.");
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].predicate, "part-of");
    }

    #[test]
    fn extract_causes() {
        let triples = extract_regex("Removing the wall causes structural failure.");
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].predicate, "causes");
    }
}
