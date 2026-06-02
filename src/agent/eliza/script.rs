//! TOML script parser for the ELIZA pattern-matching engine.
//!
//! Scripts define keywords with priority ranks, decomposition patterns, and
//! reassembly templates. Synonym groups can be referenced in patterns with the
//! `@group` syntax.
//!
//! ## TOML format
//!
//! ```toml
//! [script]
//! name = "Doctor"
//! initial = "How do you do? Please tell me your problem."
//! fallback = ["I'm not sure I understand.", "Please go on."]
//!
//! [synonyms]
//! belief = ["believe", "feel", "think", "wish"]
//!
//! [[keywords]]
//! key = "i am"
//! rank = 10
//! [[keywords.rules]]
//! decomposition = "* i am *"
//! reassembly = ["How long have you been {2}?", "Do you enjoy being {2}?"]
//! ```

use miette::Diagnostic;
use serde::Deserialize;
use std::collections::HashMap;
use thiserror::Error;

/// Errors produced during ELIZA script parsing and validation.
#[derive(Debug, Error, Diagnostic)]
pub enum ScriptError {
    #[error("TOML parse error: {message}")]
    #[diagnostic(
        code(akh::eliza::script_parse),
        help(
            "Verify the TOML syntax in your ELIZA script file. \
             Common issues: missing quotes around strings, incorrect array \
             syntax for reassembly lists."
        )
    )]
    TomlParse { message: String },

    #[error("keyword '{key}' has no decomposition rules")]
    #[diagnostic(
        code(akh::eliza::script_empty_rules),
        help(
            "Every keyword must have at least one [[keywords.rules]] entry \
             with a decomposition pattern and a non-empty reassembly list."
        )
    )]
    EmptyRules { key: String },

    #[error("decomposition rule for keyword '{key}' has an empty reassembly list")]
    #[diagnostic(
        code(akh::eliza::script_empty_reassembly),
        help(
            "Each [[keywords.rules]] entry must include at least one \
             reassembly string. Add responses to the reassembly array."
        )
    )]
    EmptyReassembly { key: String },
}

pub type ScriptResult<T> = Result<T, ScriptError>;

/// A single decomposition rule: a wildcard pattern and its response templates.
///
/// Pattern syntax:
/// - `*` — matches any sequence of words (0 or more), captured for substitution.
/// - `@group` — matches any synonym from the named synonym group (not captured).
/// - Literal words — matched case-insensitively.
///
/// Reassembly syntax: use `{N}` (1-indexed) to substitute the N-th wildcard
/// capture from the decomposition match.
#[derive(Debug, Clone, Deserialize)]
pub struct Rule {
    /// Decomposition pattern. Example: `"* i am *"`.
    pub decomposition: String,

    /// Ordered response templates, cycled through in order.
    /// Example: `["How long have you been {2}?", "Do you enjoy being {2}?"]`
    pub reassembly: Vec<String>,
}

/// A keyword entry with its decomposition rules.
#[derive(Debug, Clone, Deserialize)]
pub struct Keyword {
    /// Keyword phrase to match in user input (case-insensitive substring).
    /// Examples: `"i am"`, `"computer"`, `"sorry"`.
    pub key: String,

    /// Priority rank — higher values are checked before lower ones.
    /// Ties are broken by order in the script.
    #[serde(default)]
    pub rank: i32,

    /// When `true`, matched responses are pushed onto the memory stack instead
    /// of being returned immediately. They are recalled when no other keyword
    /// matches and the memory stack is non-empty.
    #[serde(default)]
    pub memory: bool,

    /// Decomposition rules for this keyword, tried top-to-bottom.
    pub rules: Vec<Rule>,
}

/// A fully parsed ELIZA script ready to be handed to `ElizaEngine`.
///
/// Keywords are sorted by rank (descending) after parsing so that
/// `ElizaEngine` can iterate them in priority order.
#[derive(Debug, Clone, Deserialize)]
pub struct Script {
    /// Human-readable script name.
    pub name: String,

    /// Greeting sent at the start of a conversation.
    pub initial: String,

    /// Fallback responses cycled through when no keyword matches and the
    /// memory stack is empty.
    #[serde(default)]
    pub fallback: Vec<String>,

    /// Synonym groups. The key is referenced as `@key` in decomposition
    /// patterns and expands to a regex alternation over the words.
    #[serde(default)]
    pub synonyms: HashMap<String, Vec<String>>,

    /// All keyword entries; sorted by rank descending after parsing.
    #[serde(default)]
    pub keywords: Vec<Keyword>,
}

impl Script {
    /// Parse and validate an ELIZA script from a TOML source string.
    ///
    /// # Errors
    ///
    /// - [`ScriptError::TomlParse`] — invalid TOML syntax.
    /// - [`ScriptError::EmptyRules`] — a keyword has no rules.
    /// - [`ScriptError::EmptyReassembly`] — a rule has an empty reassembly list.
    pub fn from_toml(source: &str) -> ScriptResult<Self> {
        let mut script: Script = toml::from_str(source)
            .map_err(|e| ScriptError::TomlParse { message: e.to_string() })?;

        for kw in &script.keywords {
            if kw.rules.is_empty() {
                return Err(ScriptError::EmptyRules { key: kw.key.clone() });
            }
            for rule in &kw.rules {
                if rule.reassembly.is_empty() {
                    return Err(ScriptError::EmptyReassembly { key: kw.key.clone() });
                }
            }
        }

        if script.fallback.is_empty() {
            script.fallback = vec![
                "I'm not sure I understand you fully.".to_string(),
                "Please go on.".to_string(),
                "What does that suggest to you?".to_string(),
            ];
        }

        // Priority order: highest rank first; stable sort preserves script order on ties.
        script.keywords.sort_by(|a, b| b.rank.cmp(&a.rank));

        Ok(script)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_TOML: &str = r#"
name = "Test"
initial = "Hello."
fallback = ["I see.", "Go on."]

[[keywords]]
key = "hello"
rank = 5

[[keywords.rules]]
decomposition = "*"
reassembly = ["Hi there!"]
"#;

    const SYNONYM_TOML: &str = r#"
name = "Test"
initial = "Hello."
fallback = ["I see."]

[synonyms]
belief = ["believe", "feel", "think"]
family = ["mother", "father", "sister", "brother"]

[[keywords]]
key = "believe"
rank = 0

[[keywords.rules]]
decomposition = "* @belief *"
reassembly = ["Do you really {1}?", "Why do you {1}?"]
"#;

    #[test]
    fn test_parse_minimal_script() {
        let script = Script::from_toml(MINIMAL_TOML).expect("should parse");
        assert_eq!(script.name, "Test");
        assert_eq!(script.initial, "Hello.");
        assert_eq!(script.fallback, &["I see.", "Go on."]);
        assert_eq!(script.keywords.len(), 1);
        assert_eq!(script.keywords[0].key, "hello");
        assert_eq!(script.keywords[0].rank, 5);
        assert!(!script.keywords[0].memory);
        assert_eq!(script.keywords[0].rules[0].decomposition, "*");
        assert_eq!(script.keywords[0].rules[0].reassembly, &["Hi there!"]);
    }

    #[test]
    fn test_parse_synonyms() {
        let script = Script::from_toml(SYNONYM_TOML).expect("should parse");
        assert!(script.synonyms.contains_key("belief"));
        assert_eq!(script.synonyms["belief"], &["believe", "feel", "think"]);
        assert!(script.synonyms.contains_key("family"));
    }

    #[test]
    fn test_keywords_sorted_by_rank_descending() {
        let toml = r#"
name = "T"
initial = "Hi."
fallback = ["Ok."]

[[keywords]]
key = "low"
rank = 1

[[keywords.rules]]
decomposition = "*"
reassembly = ["low"]

[[keywords]]
key = "high"
rank = 99

[[keywords.rules]]
decomposition = "*"
reassembly = ["high"]

[[keywords]]
key = "mid"
rank = 50

[[keywords.rules]]
decomposition = "*"
reassembly = ["mid"]
"#;
        let script = Script::from_toml(toml).expect("should parse");
        let ranks: Vec<i32> = script.keywords.iter().map(|k| k.rank).collect();
        assert_eq!(ranks, vec![99, 50, 1], "keywords should be sorted high→low");
    }

    #[test]
    fn test_error_empty_rules() {
        let toml = r#"
name = "T"
initial = "Hi."
fallback = ["Ok."]

[[keywords]]
key = "broken"
rank = 0
rules = []
"#;
        let err = Script::from_toml(toml).unwrap_err();
        assert!(matches!(err, ScriptError::EmptyRules { .. }));
    }

    #[test]
    fn test_error_empty_reassembly() {
        let toml = r#"
name = "T"
initial = "Hi."
fallback = ["Ok."]

[[keywords]]
key = "broken"
rank = 0

[[keywords.rules]]
decomposition = "*"
reassembly = []
"#;
        let err = Script::from_toml(toml).unwrap_err();
        assert!(matches!(err, ScriptError::EmptyReassembly { .. }));
    }

    #[test]
    fn test_default_fallback_when_omitted() {
        let toml = r#"
name = "T"
initial = "Hi."

[[keywords]]
key = "x"
rank = 0

[[keywords.rules]]
decomposition = "*"
reassembly = ["r"]
"#;
        // No fallback field — should get default responses.
        let script = Script::from_toml(toml).expect("should parse");
        assert!(!script.fallback.is_empty(), "default fallback should be injected");
    }

    #[test]
    fn test_memory_flag_parsed() {
        let toml = r#"
name = "T"
initial = "Hi."
fallback = ["Ok."]

[[keywords]]
key = "my"
rank = 0
memory = true

[[keywords.rules]]
decomposition = "* my *"
reassembly = ["Tell me more about your {2}."]
"#;
        let script = Script::from_toml(toml).expect("should parse");
        assert!(script.keywords[0].memory);
    }
}
