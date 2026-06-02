//! ELIZA script format: TOML deserialization.
//!
//! A script is a `.toml` file with a `[script]` header and one or more `[[rules]]`.
//! Each rule lists a set of keyword synonyms, optional per-rule similarity threshold,
//! and a list of response templates that are cycled round-robin.
//!
//! ## Example
//!
//! ```toml
//! [script]
//! name = "Doctor"
//! default_response = "Please go on."
//! initial_greeting = "How do you do. Please tell me your problem."
//!
//! [[rules]]
//! # All keywords are synonym variants for the same concept.
//! keywords = ["mother", "mom", "mama", "parent", "father", "dad"]
//! similarity_threshold = 0.70
//! responses = [
//!     "Tell me more about your family.",
//!     "Who else in your family feels that way?",
//! ]
//!
//! [[rules]]
//! keywords = ["hello", "hi", "hey", "greetings"]
//! responses = ["How do you do. Please state your problem."]
//! ```

use serde::{Deserialize, Serialize};

use super::error::{ElizaError, ElizaResult};

/// Top-level ELIZA script loaded from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Script {
    pub script: ScriptMeta,
    #[serde(default)]
    pub rules: Vec<Rule>,
}

/// Script-level metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptMeta {
    pub name: String,
    /// Response when no rule matches.
    #[serde(default = "default_response_str")]
    pub default_response: String,
    /// Optional first message the agent sends.
    #[serde(default)]
    pub initial_greeting: Option<String>,
}

/// A single keyword/response rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rule {
    /// Synonym variants for this concept.
    ///
    /// At load time all keywords are encoded as hypervectors and bundled into a
    /// single rule vector. Input words are matched against individual keyword
    /// vectors via exact string comparison or VSA cosine similarity.
    pub keywords: Vec<String>,

    /// Minimum Hamming similarity for a VSA fuzzy match (0.0–1.0).
    ///
    /// With hash-based token encoding the threshold filters random noise (~0.5).
    /// Only tokens that share the same hash (i.e. exact same string) exceed
    /// this threshold reliably out of the box; explicit synonym listing handles
    /// the rest via the exact-match fast path.
    #[serde(default = "default_threshold")]
    pub similarity_threshold: f32,

    /// Responses cycled round-robin when this rule fires.
    pub responses: Vec<String>,

    /// Higher priority rules are checked before lower ones. Default: 0.
    #[serde(default)]
    pub priority: i32,
}

fn default_response_str() -> String {
    "Please go on.".to_string()
}

fn default_threshold() -> f32 {
    0.70
}

impl Script {
    /// Parse a TOML string into a [`Script`] and validate it.
    pub fn from_toml(source: &str) -> ElizaResult<Self> {
        let script: Script = toml::from_str(source).map_err(|e| ElizaError::ScriptParse {
            reason: e.to_string(),
        })?;
        script.validate()?;
        Ok(script)
    }

    /// Check structural invariants (non-empty keywords and responses per rule).
    pub fn validate(&self) -> ElizaResult<()> {
        for (i, rule) in self.rules.iter().enumerate() {
            if rule.keywords.is_empty() {
                return Err(ElizaError::NoKeywords { index: i });
            }
            if rule.responses.is_empty() {
                return Err(ElizaError::NoResponses { index: i });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
[script]
name = "Test"
default_response = "Go on."
initial_greeting = "Hello."

[[rules]]
keywords = ["mother", "mom", "mama", "parent"]
similarity_threshold = 0.70
responses = ["Tell me about your family.", "Who else?"]

[[rules]]
keywords = ["hello", "hi"]
responses = ["Hi there."]
priority = 10
"#;

    #[test]
    fn parse_sample_script() {
        let s = Script::from_toml(SAMPLE).unwrap();
        assert_eq!(s.script.name, "Test");
        assert_eq!(s.rules.len(), 2);
        assert_eq!(s.rules[0].keywords, ["mother", "mom", "mama", "parent"]);
        assert!((s.rules[0].similarity_threshold - 0.70).abs() < 0.001);
        assert_eq!(s.rules[1].priority, 10);
    }

    #[test]
    fn validate_rejects_empty_keywords() {
        let toml = r#"
[script]
name = "X"

[[rules]]
keywords = []
responses = ["hi"]
"#;
        let err = Script::from_toml(toml).unwrap_err();
        assert!(matches!(err, ElizaError::NoKeywords { index: 0 }));
    }

    #[test]
    fn validate_rejects_empty_responses() {
        let toml = r#"
[script]
name = "X"

[[rules]]
keywords = ["hi"]
responses = []
"#;
        let err = Script::from_toml(toml).unwrap_err();
        assert!(matches!(err, ElizaError::NoResponses { index: 0 }));
    }

    #[test]
    fn default_threshold_is_0_70() {
        let toml = r#"
[script]
name = "X"

[[rules]]
keywords = ["hello"]
responses = ["Hi."]
"#;
        let s = Script::from_toml(toml).unwrap();
        assert!((s.rules[0].similarity_threshold - 0.70).abs() < 0.001);
    }
}
