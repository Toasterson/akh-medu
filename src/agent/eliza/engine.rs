//! ELIZA pattern-matching engine.
//!
//! Implements the classic Weizenbaum (1966) pipeline:
//! 1. Normalize input (lowercase, collapse whitespace).
//! 2. Find all keywords present in the input, ordered by rank descending.
//! 3. For each keyword, try its decomposition rules in order.
//! 4. On a match, extract wildcard captures and format a reassembly response.
//! 5. Memory keywords defer their response to the memory stack.
//! 6. Fall back to the memory stack, then to the script's fallback list.

use std::collections::HashMap;

use regex::Regex;

use super::script::Script;

// ---------------------------------------------------------------------------
// Internal compiled representation
// ---------------------------------------------------------------------------

/// Pre-compiled regex for a single decomposition rule.
struct CompiledRule {
    pattern: Regex,
    /// Number of `*` wildcard capture groups in the pattern.
    wildcard_count: usize,
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Stateful ELIZA conversation engine.
///
/// Each `ElizaEngine` encapsulates a single conversation session: it holds the
/// script, pre-compiled patterns, cyclic reassembly counters, and memory stack.
/// Create a new instance per conversation or call [`ElizaEngine::reset`] to
/// clear session state while keeping the script.
pub struct ElizaEngine {
    script: Script,
    /// Pre-compiled patterns indexed by `(keyword_index, rule_index)`.
    compiled: HashMap<(usize, usize), CompiledRule>,
    /// Cyclic reassembly counter per `(keyword_index, rule_index)`.
    reassembly_counter: HashMap<(usize, usize), usize>,
    /// LIFO memory stack of deferred responses from memory-flagged keywords.
    memory_stack: Vec<String>,
    /// Round-robin index into the fallback response list.
    fallback_counter: usize,
}

impl ElizaEngine {
    /// Build an engine from a parsed script, pre-compiling all decomposition
    /// patterns.
    pub fn new(script: Script) -> Self {
        let mut compiled = HashMap::new();
        for (ki, kw) in script.keywords.iter().enumerate() {
            for (ri, rule) in kw.rules.iter().enumerate() {
                let cr = compile_rule(&rule.decomposition, &script.synonyms);
                compiled.insert((ki, ri), cr);
            }
        }
        Self {
            script,
            compiled,
            reassembly_counter: HashMap::new(),
            memory_stack: Vec::new(),
            fallback_counter: 0,
        }
    }

    /// The initial greeting defined in the script.
    pub fn greeting(&self) -> &str {
        &self.script.initial
    }

    /// Reset session state (reassembly counters, memory stack, fallback index)
    /// without reloading the script or recompiling patterns.
    pub fn reset(&mut self) {
        self.reassembly_counter.clear();
        self.memory_stack.clear();
        self.fallback_counter = 0;
    }

    /// Generate a response to `input`.
    ///
    /// Processing order:
    /// 1. Normalize input.
    /// 2. Iterate keywords in rank order; skip those not present in input.
    /// 3. For each keyword, try decomposition rules top-to-bottom; stop at
    ///    the first matching rule.
    /// 4. Memory keywords: push the formatted response onto the memory stack
    ///    and continue scanning remaining keywords.
    /// 5. Non-memory keywords: record the first such response (highest rank
    ///    wins) and continue scanning for additional memory keyword matches.
    /// 6. After scanning all keywords:
    ///    - If a non-memory match was found, return it.
    ///    - Else if only memory keywords matched, return the next fallback
    ///      response (memory is stored for a future turn).
    ///    - Else (no keyword matched at all), pop from the memory stack; if
    ///      empty, return the next fallback.
    pub fn respond(&mut self, input: &str) -> String {
        let normalized = normalize(input);

        let mut non_memory_response: Option<String> = None;
        let mut any_keyword_matched = false;

        let kw_count = self.script.keywords.len();
        for ki in 0..kw_count {
            let kw_key = self.script.keywords[ki].key.to_lowercase();
            let kw_memory = self.script.keywords[ki].memory;
            let rule_count = self.script.keywords[ki].rules.len();

            if !word_match(&normalized, &kw_key) {
                continue;
            }

            // Try rules top-to-bottom; stop at the first match for this keyword.
            for ri in 0..rule_count {
                let captures_opt: Option<Vec<String>> =
                    self.compiled.get(&(ki, ri)).and_then(|compiled| {
                        compiled.pattern.captures(&normalized).map(|caps| {
                            (1..=compiled.wildcard_count)
                                .map(|i| {
                                    caps.get(i)
                                        .map(|m| m.as_str().trim().to_string())
                                        .unwrap_or_default()
                                })
                                .collect()
                        })
                    });

                let Some(captures) = captures_opt else {
                    continue;
                };

                any_keyword_matched = true;
                let reassembly = self.script.keywords[ki].rules[ri].reassembly.clone();
                let response = self.assemble(ki, ri, &reassembly, &captures);

                if kw_memory {
                    self.memory_stack.push(response);
                } else if non_memory_response.is_none() {
                    non_memory_response = Some(response);
                }
                break; // first matching rule wins for this keyword
            }
        }

        if let Some(response) = non_memory_response {
            return response;
        }

        // Only memory keywords matched → defer to memory; return fallback this turn.
        // No keywords matched at all → try to recall from memory stack.
        if !any_keyword_matched {
            if let Some(remembered) = self.memory_stack.pop() {
                return remembered;
            }
        }
        self.next_fallback()
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Select the next reassembly template (cyclic) and substitute captures.
    fn assemble(
        &mut self,
        ki: usize,
        ri: usize,
        templates: &[String],
        captures: &[String],
    ) -> String {
        let counter = self.reassembly_counter.entry((ki, ri)).or_insert(0);
        let template = &templates[*counter % templates.len()];
        *counter += 1;

        let mut response = template.clone();
        for (i, cap) in captures.iter().enumerate() {
            response = response.replace(&format!("{{{}}}", i + 1), cap.as_str());
        }
        response
    }

    fn next_fallback(&mut self) -> String {
        if self.script.fallback.is_empty() {
            return "I see.".to_string();
        }
        let idx = self.fallback_counter % self.script.fallback.len();
        self.fallback_counter += 1;
        self.script.fallback[idx].clone()
    }
}

// ---------------------------------------------------------------------------
// Pattern compilation
// ---------------------------------------------------------------------------

/// Convert an ELIZA decomposition pattern string into a compiled regex.
///
/// Pattern tokens:
/// - `*` — any sequence of words (lazy capture, greedy for the final `*`).
/// - `@key` — synonym group; expands to a non-capturing alternation.
/// - Literal word — matched case-insensitively with `regex::escape`.
///
/// Whitespace between tokens: `\s*` when adjacent to a wildcard (which can
/// capture nothing), `\s+` between two consecutive non-wildcard tokens.
fn compile_rule(pattern: &str, synonyms: &HashMap<String, Vec<String>>) -> CompiledRule {
    let tokens: Vec<&str> = pattern.split_whitespace().collect();
    let n = tokens.len();
    let mut wildcard_count: usize = 0;
    let mut parts: Vec<String> = vec!["(?i)^".to_string()];
    let mut prev_wildcard = true; // Treat start-of-input as wildcard for spacing logic.

    for (i, &token) in tokens.iter().enumerate() {
        let is_last = i + 1 == n;
        let is_wildcard = token == "*";

        // Insert whitespace separator between tokens.
        if i > 0 {
            if prev_wildcard || is_wildcard {
                parts.push(r"\s*".to_string());
            } else {
                parts.push(r"\s+".to_string());
            }
        }

        if is_wildcard {
            // Last wildcard is greedy so it consumes the rest of the input;
            // earlier wildcards are lazy to find the leftmost keyword match.
            if is_last {
                parts.push(r"(.*)".to_string());
            } else {
                parts.push(r"(.*?)".to_string());
            }
            wildcard_count += 1;
        } else if let Some(key) = token.strip_prefix('@') {
            let lower_key = key.to_lowercase();
            if let Some(words) = synonyms.get(&lower_key) {
                let alts: String = words
                    .iter()
                    .map(|w| regex::escape(w))
                    .collect::<Vec<_>>()
                    .join("|");
                parts.push(format!("(?:{})", alts));
            } else {
                // Unknown synonym group — treat as literal.
                parts.push(regex::escape(token));
            }
        } else {
            parts.push(regex::escape(token));
        }

        prev_wildcard = is_wildcard;
    }

    // Allow trailing whitespace only when the last token was not a wildcard
    // (wildcards already consume trailing space via `(.*)`).
    if !prev_wildcard {
        parts.push(r"\s*$".to_string());
    } else {
        parts.push(r"$".to_string());
    }

    let pat = parts.join("");
    let re = Regex::new(&pat).unwrap_or_else(|e| {
        tracing::warn!(pattern, error = %e, "failed to compile ELIZA decomposition pattern");
        Regex::new("$^").unwrap() // sentinel that never matches
    });

    CompiledRule {
        pattern: re,
        wildcard_count,
    }
}

// ---------------------------------------------------------------------------
// Input normalization
// ---------------------------------------------------------------------------

/// Lowercase and collapse internal whitespace to single spaces.
fn normalize(input: &str) -> String {
    input
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Return `true` iff `keyword` (a whitespace-separated phrase) appears as a
/// sequence of complete word tokens in `text`.
///
/// Uses token-window matching rather than substring search to avoid false
/// positives like keyword `"am"` matching inside `"diagram"`.
fn word_match(text: &str, keyword: &str) -> bool {
    let kw_words: Vec<&str> = keyword.split_whitespace().collect();
    if kw_words.is_empty() {
        return false;
    }
    let text_words: Vec<&str> = text.split_whitespace().collect();
    if kw_words.len() > text_words.len() {
        return false;
    }
    text_words
        .windows(kw_words.len())
        .any(|w| w == kw_words.as_slice())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::eliza::script::Script;

    // -----------------------------------------------------------------------
    // Helper: build a minimal engine from an inline TOML string.
    // -----------------------------------------------------------------------
    fn engine_from_toml(toml: &str) -> ElizaEngine {
        let script = Script::from_toml(toml).expect("test script must parse");
        ElizaEngine::new(script)
    }

    // -----------------------------------------------------------------------
    // normalize / word_match
    // -----------------------------------------------------------------------

    #[test]
    fn test_normalize_lowercases_and_collapses_spaces() {
        assert_eq!(normalize("Hello   World"), "hello world");
        assert_eq!(normalize("  I  AM  "), "i am");
        assert_eq!(normalize("COMPUTER"), "computer");
    }

    #[test]
    fn test_word_match_phrase() {
        assert!(word_match("i think i am happy", "i am"));
        assert!(word_match("i am happy", "i am"));
        assert!(!word_match("i came here", "i am"));
        assert!(!word_match("diagram", "am"));
    }

    #[test]
    fn test_word_match_single_word() {
        assert!(word_match("the computer is broken", "computer"));
        assert!(!word_match("the computational model", "computer"));
    }

    // -----------------------------------------------------------------------
    // Greeting
    // -----------------------------------------------------------------------

    #[test]
    fn test_greeting_returns_initial() {
        let engine = engine_from_toml(
            r#"
name = "T"
initial = "Hello, please tell me your problem."
fallback = ["I see."]

[[keywords]]
key = "x"
rank = 0

[[keywords.rules]]
decomposition = "*"
reassembly = ["r"]
"#,
        );
        assert_eq!(engine.greeting(), "Hello, please tell me your problem.");
    }

    // -----------------------------------------------------------------------
    // Pattern compilation (tested via full engine round-trip)
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_wildcard_matches_anything() {
        let mut engine = engine_from_toml(
            r#"
name = "T"
initial = "Hi."
fallback = ["I see."]

[[keywords]]
key = "hello"
rank = 0

[[keywords.rules]]
decomposition = "*"
reassembly = ["Hello!"]
"#,
        );
        assert_eq!(engine.respond("hello"), "Hello!");
        assert_eq!(engine.respond("say hello please"), "Hello!");
    }

    #[test]
    fn test_decomposition_extracts_two_captures() {
        let mut engine = engine_from_toml(
            r#"
name = "T"
initial = "Hi."
fallback = ["I see."]

[[keywords]]
key = "i am"
rank = 0

[[keywords.rules]]
decomposition = "* i am *"
reassembly = ["How long have you been {2}?"]
"#,
        );
        assert_eq!(
            engine.respond("i am very tired"),
            "How long have you been very tired?"
        );
        assert_eq!(
            engine.respond("i think i am confused"),
            "How long have you been confused?"
        );
    }

    #[test]
    fn test_first_capture_contains_preamble() {
        let mut engine = engine_from_toml(
            r#"
name = "T"
initial = "Hi."
fallback = ["I see."]

[[keywords]]
key = "i am"
rank = 0

[[keywords.rules]]
decomposition = "* i am *"
reassembly = ["Before: '{1}', After: '{2}'"]
"#,
        );
        let resp = engine.respond("sometimes i am nervous");
        assert_eq!(resp, "Before: 'sometimes', After: 'nervous'");
    }

    #[test]
    fn test_empty_preamble_capture() {
        let mut engine = engine_from_toml(
            r#"
name = "T"
initial = "Hi."
fallback = ["I see."]

[[keywords]]
key = "i am"
rank = 0

[[keywords.rules]]
decomposition = "* i am *"
reassembly = ["preamble='{1}' rest='{2}'"]
"#,
        );
        // No words before "i am" — first capture should be empty.
        let resp = engine.respond("i am fine");
        assert_eq!(resp, "preamble='' rest='fine'");
    }

    // -----------------------------------------------------------------------
    // Keyword rank priority
    // -----------------------------------------------------------------------

    #[test]
    fn test_higher_rank_keyword_wins() {
        let mut engine = engine_from_toml(
            r#"
name = "T"
initial = "Hi."
fallback = ["I see."]

[[keywords]]
key = "computer"
rank = 50

[[keywords.rules]]
decomposition = "* computer *"
reassembly = ["Computer branch"]

[[keywords]]
key = "afraid"
rank = 5

[[keywords.rules]]
decomposition = "*"
reassembly = ["Afraid branch"]
"#,
        );
        // Input contains both "afraid" and "computer" — rank 50 wins.
        let resp = engine.respond("i am afraid of the computer");
        assert_eq!(resp, "Computer branch");
    }

    // -----------------------------------------------------------------------
    // Cyclic reassembly
    // -----------------------------------------------------------------------

    #[test]
    fn test_cyclic_reassembly_cycles_in_order() {
        let mut engine = engine_from_toml(
            r#"
name = "T"
initial = "Hi."
fallback = ["Fallback."]

[[keywords]]
key = "sorry"
rank = 0

[[keywords.rules]]
decomposition = "*"
reassembly = ["A", "B", "C"]
"#,
        );
        assert_eq!(engine.respond("sorry"), "A");
        assert_eq!(engine.respond("sorry"), "B");
        assert_eq!(engine.respond("sorry"), "C");
        assert_eq!(engine.respond("sorry"), "A"); // wraps back
    }

    // -----------------------------------------------------------------------
    // Multiple rules per keyword
    // -----------------------------------------------------------------------

    #[test]
    fn test_second_rule_used_when_first_does_not_match() {
        let mut engine = engine_from_toml(
            r#"
name = "T"
initial = "Hi."
fallback = ["I see."]

[[keywords]]
key = "i"
rank = 0

[[keywords.rules]]
decomposition = "* i am *"
reassembly = ["Are you {2}?"]

[[keywords.rules]]
decomposition = "*"
reassembly = ["You said something with 'i'."]
"#,
        );
        // "i want cake" does not match "* i am *" but matches the catch-all "*".
        let resp = engine.respond("i want cake");
        assert_eq!(resp, "You said something with 'i'.");
    }

    // -----------------------------------------------------------------------
    // Synonym matching
    // -----------------------------------------------------------------------

    #[test]
    fn test_synonym_in_pattern_matches_any_word() {
        // The keyword "i" is present in all three inputs. The decomposition
        // pattern "* @belief *" expands @belief to all synonym members so that
        // "feel", "think", and "believe" each trigger the same response.
        let mut engine = engine_from_toml(
            r#"
name = "T"
initial = "Hi."
fallback = ["I see."]

[synonyms]
belief = ["believe", "feel", "think", "wish"]

[[keywords]]
key = "i"
rank = 0

[[keywords.rules]]
decomposition = "* @belief *"
reassembly = ["Why do you say that?"]
"#,
        );
        assert_eq!(engine.respond("i feel lonely"), "Why do you say that?");
        assert_eq!(
            engine.respond("i think you are wrong"),
            "Why do you say that?"
        );
        assert_eq!(engine.respond("i believe so"), "Why do you say that?");
    }

    // -----------------------------------------------------------------------
    // Memory stack
    // -----------------------------------------------------------------------

    #[test]
    fn test_memory_keyword_defers_response() {
        let mut engine = engine_from_toml(
            r#"
name = "T"
initial = "Hi."
fallback = ["Fallback."]

[[keywords]]
key = "my"
rank = 0
memory = true

[[keywords.rules]]
decomposition = "* my *"
reassembly = ["Tell me more about your {2}."]
"#,
        );
        // Memory keyword matches → response is deferred to memory stack; fallback
        // returned this turn (classic Weizenbaum ELIZA behaviour).
        let resp = engine.respond("my dog is ill");
        assert_eq!(resp, "Fallback.");
        // On the next turn with no keyword match, the deferred memory is recalled.
        let resp2 = engine.respond("nothing in particular");
        assert_eq!(resp2, "Tell me more about your dog is ill.");
    }

    #[test]
    fn test_memory_stack_deferred_until_no_other_match() {
        let mut engine = engine_from_toml(
            r#"
name = "T"
initial = "Hi."
fallback = ["Fallback."]

[[keywords]]
key = "hello"
rank = 10

[[keywords.rules]]
decomposition = "*"
reassembly = ["Hello response"]

[[keywords]]
key = "my"
rank = 0
memory = true

[[keywords.rules]]
decomposition = "* my *"
reassembly = ["Memory: your {2}."]
"#,
        );
        // Input contains both "hello" (rank 10, non-memory) and "my" (memory).
        // The non-memory keyword should win immediately.
        let resp1 = engine.respond("hello my friend");
        assert_eq!(resp1, "Hello response");

        // The memory response was stored. Now send something that matches nothing.
        let resp2 = engine.respond("blah blah");
        assert_eq!(resp2, "Memory: your friend.");
    }

    #[test]
    fn test_memory_stack_is_lifo() {
        let mut engine = engine_from_toml(
            r#"
name = "T"
initial = "Hi."
fallback = ["Fallback."]

[[keywords]]
key = "my"
rank = 0
memory = true

[[keywords.rules]]
decomposition = "* my *"
reassembly = ["About: {2}."]
"#,
        );
        // Push two memory entries.
        engine.respond("my cat"); // pushed "About: cat."
        engine.respond("my dog"); // pushed "About: dog."

        // LIFO: last pushed is first recalled.
        let r1 = engine.respond("nothing"); // pops "About: dog."
        assert_eq!(r1, "About: dog.");
        let r2 = engine.respond("nothing"); // pops "About: cat."
        assert_eq!(r2, "About: cat.");
        // Memory exhausted — fallback.
        let r3 = engine.respond("nothing");
        assert_eq!(r3, "Fallback.");
    }

    // -----------------------------------------------------------------------
    // Fallback
    // -----------------------------------------------------------------------

    #[test]
    fn test_fallback_when_no_keyword_matches() {
        let mut engine = engine_from_toml(
            r#"
name = "T"
initial = "Hi."
fallback = ["I don't understand.", "Please go on."]

[[keywords]]
key = "hello"
rank = 0

[[keywords.rules]]
decomposition = "*"
reassembly = ["Hi!"]
"#,
        );
        assert_eq!(engine.respond("unrelated input xyz"), "I don't understand.");
    }

    #[test]
    fn test_fallback_cycles() {
        let mut engine = engine_from_toml(
            r#"
name = "T"
initial = "Hi."
fallback = ["A", "B"]

[[keywords]]
key = "zzz"
rank = 0

[[keywords.rules]]
decomposition = "*"
reassembly = ["never"]
"#,
        );
        assert_eq!(engine.respond("x"), "A");
        assert_eq!(engine.respond("x"), "B");
        assert_eq!(engine.respond("x"), "A"); // wraps
    }

    // -----------------------------------------------------------------------
    // Reset
    // -----------------------------------------------------------------------

    #[test]
    fn test_reset_clears_counters_and_memory() {
        let mut engine = engine_from_toml(
            r#"
name = "T"
initial = "Hi."
fallback = ["Fallback."]

[[keywords]]
key = "sorry"
rank = 0

[[keywords.rules]]
decomposition = "*"
reassembly = ["A", "B"]
"#,
        );
        assert_eq!(engine.respond("sorry"), "A");
        assert_eq!(engine.respond("sorry"), "B");
        engine.reset();
        assert_eq!(engine.respond("sorry"), "A"); // counter reset
    }
}
