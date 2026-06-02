//! ELIZA-style KG-grounded Q&A handler.
//!
//! Matches factual-question patterns against the input, extracts the subject
//! entity, and looks it up in the knowledge graph. Falls back to a static
//! template when the KG has no matching triple.
//!
//! Latency target: < 1 ms (pure in-memory regex + map lookup).

use regex::Regex;

// ---------------------------------------------------------------------------
// KgBackend trait
// ---------------------------------------------------------------------------

/// Read-only view into a knowledge graph, used by [`ElizaHandler`].
///
/// Implementors bridge from the caller's KG engine to the ELIZA handler.
/// All methods must be non-blocking and return quickly (< 1 ms each).
pub trait KgBackend: Send + Sync {
    /// Return the direct object for `(subject, predicate)`, or `None`.
    fn fact(&self, subject: &str, predicate: &str) -> Option<String>;

    /// Return up to `limit` `(predicate, object)` pairs related to `subject`.
    fn related(&self, subject: &str, limit: usize) -> Vec<(String, String)>;

    /// Search entities by label, returning up to `limit` `(label, summary)` pairs.
    fn search(&self, query: &str, limit: usize) -> Vec<(String, String)>;
}

// ---------------------------------------------------------------------------
// NullKg — no-op backend for tests and stub configurations
// ---------------------------------------------------------------------------

/// A KG backend that always returns empty results.
///
/// Useful in unit tests and when no engine is attached to the routing layer.
pub struct NullKg;

impl KgBackend for NullKg {
    fn fact(&self, _subject: &str, _predicate: &str) -> Option<String> {
        None
    }
    fn related(&self, _subject: &str, _limit: usize) -> Vec<(String, String)> {
        vec![]
    }
    fn search(&self, _query: &str, _limit: usize) -> Vec<(String, String)> {
        vec![]
    }
}

// ---------------------------------------------------------------------------
// Internal pattern representation
// ---------------------------------------------------------------------------

struct ElizaPattern {
    trigger: Regex,
    /// Index of the capture group that holds the subject entity (1-based).
    subject_group: Option<usize>,
    /// Static response template; `{entity}` is substituted with the matched subject.
    response_template: &'static str,
    /// KG predicate to look up for the entity, if any.
    kg_predicate: Option<&'static str>,
}

// ---------------------------------------------------------------------------
// ElizaHandler
// ---------------------------------------------------------------------------

/// ELIZA-style handler: answers factual questions from the KG.
///
/// Patterns are matched in declaration order; the first match wins.
/// When the KG has a direct answer for the extracted entity, it is embedded
/// in the response. When it does not, the static template is used.
///
/// Returns `None` from [`Self::respond`] when no pattern matches, signalling
/// the caller to escalate to the Gemma or Hybrid path.
pub struct ElizaHandler {
    patterns: Vec<ElizaPattern>,
}

impl Default for ElizaHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl ElizaHandler {
    /// Construct a new handler with the built-in pattern set.
    pub fn new() -> Self {
        Self {
            patterns: build_patterns(),
        }
    }

    /// Attempt to answer `input` via KG-grounded pattern matching.
    ///
    /// Returns `Some(response)` on a pattern match, `None` otherwise.
    pub fn respond(&self, input: &str, kg: &dyn KgBackend) -> Option<String> {
        for pat in &self.patterns {
            let Some(caps) = pat.trigger.captures(input) else {
                continue;
            };

            let entity = pat
                .subject_group
                .and_then(|g| caps.get(g))
                .map(|m| m.as_str().trim().to_string())
                .unwrap_or_default();

            if !entity.is_empty() {
                // Preferred: direct KG fact
                if let Some(pred) = pat.kg_predicate
                    && let Some(answer) = kg.fact(&entity, pred)
                {
                    return Some(format!("{entity} — {answer}"));
                }

                // Fallback: any related facts from the KG
                let related = kg.related(&entity, 4);
                if !related.is_empty() {
                    let facts = related
                        .iter()
                        .map(|(p, o)| format!("{p}: {o}"))
                        .collect::<Vec<_>>()
                        .join("; ");
                    return Some(format!("{entity} — {facts}"));
                }

                // Last resort: search by label
                let hits = kg.search(&entity, 2);
                if !hits.is_empty() {
                    let summaries = hits
                        .iter()
                        .map(|(lbl, summary)| format!("{lbl}: {summary}"))
                        .collect::<Vec<_>>()
                        .join("; ");
                    return Some(summaries);
                }
            }

            // No KG data — use the static template (only if it doesn't reference an
            // empty entity substitution)
            let resp = pat.response_template.replace("{entity}", &entity);
            if !entity.is_empty() || !resp.contains("{entity}") {
                return Some(resp);
            }
        }

        None
    }
}

// ---------------------------------------------------------------------------
// Default pattern set
// ---------------------------------------------------------------------------

fn build_patterns() -> Vec<ElizaPattern> {
    vec![
        ElizaPattern {
            trigger: Regex::new(
                r"(?i)what\s+is\s+(?:the\s+)?([A-Za-z][A-Za-z0-9 '\-]{1,48}?)\s*\??\s*$",
            )
            .unwrap(),
            subject_group: Some(1),
            response_template: "I know about {entity}. Let me check the knowledge graph.",
            kg_predicate: Some("is-a"),
        },
        ElizaPattern {
            trigger: Regex::new(
                r"(?i)what\s+are\s+(?:the\s+)?([A-Za-z][A-Za-z0-9 '\-]{1,48}?)\s*\??\s*$",
            )
            .unwrap(),
            subject_group: Some(1),
            response_template: "Here is what I know about {entity}.",
            kg_predicate: None,
        },
        ElizaPattern {
            trigger: Regex::new(
                r"(?i)who\s+is\s+([A-Za-z][A-Za-z0-9 '\-]{1,48}?)\s*\??\s*$",
            )
            .unwrap(),
            subject_group: Some(1),
            response_template: "{entity} is an entity I can look up.",
            kg_predicate: Some("instance-of"),
        },
        ElizaPattern {
            trigger: Regex::new(
                r"(?i)(?:define|explain)\s+(?:the\s+)?([A-Za-z][A-Za-z0-9 '\-]{1,48}?)\s*\.?\s*$",
            )
            .unwrap(),
            subject_group: Some(1),
            response_template: "Definition of {entity}.",
            kg_predicate: Some("definition"),
        },
        ElizaPattern {
            trigger: Regex::new(
                r"(?i)describe\s+(?:the\s+)?([A-Za-z][A-Za-z0-9 '\-]{1,48}?)\s*\.?\s*$",
            )
            .unwrap(),
            subject_group: Some(1),
            response_template: "Here is a description of {entity}.",
            kg_predicate: Some("description"),
        },
        ElizaPattern {
            trigger: Regex::new(
                r"(?i)(?:list|tell\s+me\s+about)\s+(?:the\s+)?([A-Za-z][A-Za-z0-9 '\-]{1,48}?)\s*\.?\s*$",
            )
            .unwrap(),
            subject_group: Some(1),
            response_template: "Here are facts about {entity}.",
            kg_predicate: None,
        },
        ElizaPattern {
            trigger: Regex::new(r"(?i)how\s+many\s+(.+?)\s+(?:are|exist|does)\b").unwrap(),
            subject_group: Some(1),
            response_template: "Let me check the count for {entity}.",
            kg_predicate: Some("count"),
        },
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct MockKg {
        facts: Vec<(&'static str, &'static str, &'static str)>,
    }

    impl KgBackend for MockKg {
        fn fact(&self, subject: &str, predicate: &str) -> Option<String> {
            self.facts
                .iter()
                .find(|(s, p, _)| s.eq_ignore_ascii_case(subject) && *p == predicate)
                .map(|(_, _, o)| o.to_string())
        }
        fn related(&self, subject: &str, limit: usize) -> Vec<(String, String)> {
            self.facts
                .iter()
                .filter(|(s, _, _)| s.eq_ignore_ascii_case(subject))
                .take(limit)
                .map(|(_, p, o)| (p.to_string(), o.to_string()))
                .collect()
        }
        fn search(&self, _query: &str, _limit: usize) -> Vec<(String, String)> {
            vec![]
        }
    }

    #[test]
    fn responds_with_kg_fact() {
        let kg = MockKg {
            facts: vec![("Ennead", "is-a", "group of nine Egyptian deities")],
        };
        let handler = ElizaHandler::new();
        let resp = handler.respond("What is the Ennead?", &kg);
        assert!(resp.is_some(), "expected a response");
        let text = resp.unwrap();
        assert!(text.contains("Ennead"), "response should mention the entity");
        assert!(
            text.contains("nine Egyptian deities"),
            "response should include the KG fact"
        );
    }

    #[test]
    fn falls_back_to_related_when_no_direct_fact() {
        let kg = MockKg {
            facts: vec![("Ra", "role", "sun god"), ("Ra", "pantheon", "Egyptian")],
        };
        let handler = ElizaHandler::new();
        let resp = handler.respond("Who is Ra?", &kg);
        assert!(resp.is_some());
        let text = resp.unwrap();
        assert!(text.contains("Ra"));
    }

    #[test]
    fn returns_none_when_no_pattern_matches() {
        let handler = ElizaHandler::new();
        // Creative directive — no ELIZA pattern for this
        let resp = handler.respond("Write a poem about the Ennead", &NullKg);
        assert!(resp.is_none());
    }

    #[test]
    fn static_template_used_when_kg_empty() {
        let handler = ElizaHandler::new();
        let resp = handler.respond("What is the Ennead?", &NullKg);
        // Pattern matches but KG has nothing — should return static template
        assert!(resp.is_some());
        assert!(resp.unwrap().contains("Ennead"));
    }
}
