//! Intent classification for natural language user input.
//!
//! Regex-based classification that works without an LLM. Identifies user intent
//! from common patterns like "What is X?", "Find X", "X is a Y", etc.

/// The question word that opens a query, used for discourse focus classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionWord {
    Who,
    What,
    How,
    Why,
    Where,
    When,
    Which,
    /// Yes/no questions: "Is X a Y?", "Does X have Y?", "Can X do Y?"
    YesNo,
}

/// Classified user intent from natural language input.
#[derive(Debug, Clone)]
pub enum UserIntent {
    /// "What/Who/Where/How ... X?" — look up information.
    Query {
        subject: String,
        original_input: String,
        question_word: Option<QuestionWord>,
    },
    /// "X is a Y" / "X has Y" — assert a fact.
    Assert { text: String },
    /// "Find/Learn/Discover ..." — set an agent goal.
    SetGoal { description: String },
    /// "Run/Cycle ..." — execute OODA cycles.
    RunAgent { cycles: Option<usize> },
    /// "Status/Goals" — show current state.
    ShowStatus,
    /// "Show/Render/Graph ..." — render hieroglyphic notation.
    RenderHiero { entity: Option<String> },
    /// "Help" — show help information.
    Help,
    /// Unrecognized input — pass through.
    Freeform { text: String },
}

/// Classify user intent from natural language input using regex patterns.
///
/// Patterns are tried in priority order; first match wins.
pub fn classify_intent(input: &str) -> UserIntent {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return UserIntent::Freeform {
            text: String::new(),
        };
    }

    let lower = trimmed.to_lowercase();

    // Help.
    if lower == "help" || lower == "?" || lower.starts_with("help ") {
        return UserIntent::Help;
    }

    // Status.
    if lower == "status"
        || lower == "goals"
        || lower == "show status"
        || lower == "show goals"
        || lower.starts_with("list goals")
    {
        return UserIntent::ShowStatus;
    }

    // Render hieroglyphic notation.
    if lower.starts_with("show ") || lower.starts_with("render ") || lower.starts_with("graph ") {
        let rest = if lower.starts_with("show ") {
            trimmed[5..].trim()
        } else if lower.starts_with("render ") {
            trimmed[7..].trim()
        } else {
            trimmed[6..].trim()
        };
        let entity = if rest.is_empty() {
            None
        } else {
            Some(rest.to_string())
        };
        return UserIntent::RenderHiero { entity };
    }

    // Run agent.
    if lower.starts_with("run") || lower.starts_with("cycle") {
        let cycles = extract_number(&lower);
        return UserIntent::RunAgent { cycles };
    }

    // Query: starts with question word or ends with '?'.
    let question_word = extract_question_word(&lower);
    if question_word.is_some()
        || lower.starts_with("is ")
        || lower.starts_with("does ")
        || lower.starts_with("do ")
        || lower.starts_with("can ")
        || trimmed.ends_with('?')
    {
        let subject = extract_subject_from_question(trimmed);
        let qw = question_word.or_else(|| {
            // Yes/no starters: is, does, do, can
            if lower.starts_with("is ")
                || lower.starts_with("does ")
                || lower.starts_with("do ")
                || lower.starts_with("can ")
            {
                Some(QuestionWord::YesNo)
            } else {
                None
            }
        });
        return UserIntent::Query {
            subject,
            original_input: trimmed.to_string(),
            question_word: qw,
        };
    }

    // Set goal: starts with action verbs.
    if lower.starts_with("find ")
        || lower.starts_with("learn ")
        || lower.starts_with("discover ")
        || lower.starts_with("explore ")
        || lower.starts_with("search ")
        || lower.starts_with("analyze ")
        || lower.starts_with("investigate ")
        || lower.starts_with("determine ")
        || lower.starts_with("classify ")
        || lower.starts_with("identify ")
    {
        return UserIntent::SetGoal {
            description: trimmed.to_string(),
        };
    }

    // Assert: contains relational patterns.
    if contains_assertion_pattern(&lower) {
        return UserIntent::Assert {
            text: trimmed.to_string(),
        };
    }

    // Freeform fallback.
    UserIntent::Freeform {
        text: trimmed.to_string(),
    }
}

/// Check if the input contains an assertion pattern.
fn contains_assertion_pattern(lower: &str) -> bool {
    // "X is a Y" / "X is an Y"
    if lower.contains(" is a ") || lower.contains(" is an ") {
        return true;
    }
    // "X are Y" / "X are a Y"
    if lower.contains(" are ") {
        return true;
    }
    // "X has Y" / "X has a Y"
    if lower.contains(" has ") || lower.contains(" have ") {
        return true;
    }
    // "X contains Y"
    if lower.contains(" contains ") {
        return true;
    }
    // "X is part of Y"
    if lower.contains(" is part of ") {
        return true;
    }
    // "X is located in Y"
    if lower.contains(" is located in ") {
        return true;
    }
    // "X causes Y"
    if lower.contains(" causes ") {
        return true;
    }
    // "X is similar to Y"
    if lower.contains(" is similar to ") {
        return true;
    }
    // "X is made of Y"
    if lower.contains(" is made of ") {
        return true;
    }
    false
}

/// Extract a number from the input (e.g., "run 5 cycles" → Some(5)).
fn extract_number(input: &str) -> Option<usize> {
    for word in input.split_whitespace() {
        if let Ok(n) = word.parse::<usize>() {
            return Some(n);
        }
    }
    None
}

/// Extract the question word from the beginning of a lowercase query.
fn extract_question_word(lower: &str) -> Option<QuestionWord> {
    if lower.starts_with("what ") {
        Some(QuestionWord::What)
    } else if lower.starts_with("who ") {
        Some(QuestionWord::Who)
    } else if lower.starts_with("where ") {
        Some(QuestionWord::Where)
    } else if lower.starts_with("when ") {
        Some(QuestionWord::When)
    } else if lower.starts_with("how ") {
        Some(QuestionWord::How)
    } else if lower.starts_with("why ") {
        Some(QuestionWord::Why)
    } else if lower.starts_with("which ") {
        Some(QuestionWord::Which)
    } else {
        None
    }
}

/// Extract the subject from a question.
fn extract_subject_from_question(input: &str) -> String {
    let s = input.trim().trim_end_matches('?').trim();

    // Remove leading question words.
    let words: Vec<&str> = s.split_whitespace().collect();
    if words.len() < 2 {
        return s.to_string();
    }

    let first_lower = words[0].to_lowercase();
    let skip = match first_lower.as_str() {
        "what" | "who" | "where" | "when" | "how" | "why" | "which" => {
            // Also skip "is", "are", "do", "does", "can" after question word.
            if words.len() > 1 {
                let second_lower = words[1].to_lowercase();
                if ["is", "are", "do", "does", "can", "was", "were", "about"]
                    .contains(&second_lower.as_str())
                {
                    2
                } else {
                    1
                }
            } else {
                1
            }
        }
        "is" | "does" | "do" | "can" => 1,
        _ => 0,
    };

    let mut final_words: Vec<&str> = words[skip..].to_vec();

    // Strip trailing auxiliary verbs left over from modal patterns.
    // "What can you do?" → skip 2 → ["you", "do"] → strip "do" → ["you"]
    // "What does the cat eat?" → skip 2 → ["the", "cat", "eat"] → strip "eat"?
    // Only strip trailing "do" — it's the generic auxiliary, not a content verb.
    if final_words.len() > 1 {
        let last = final_words.last().unwrap().to_lowercase();
        if last == "do" {
            final_words.pop();
        }
    }

    if final_words.is_empty() {
        return s.to_string();
    }

    // Remove leading articles "a", "an", "the".
    let first_remaining = final_words[0].to_lowercase();
    if ["a", "an", "the"].contains(&first_remaining.as_str()) && final_words.len() > 1 {
        final_words[1..].join(" ")
    } else {
        final_words.join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_query() {
        let intent = classify_intent("What is a dog?");
        assert!(matches!(intent, UserIntent::Query { .. }));
        if let UserIntent::Query {
            question_word,
            original_input,
            ..
        } = &intent
        {
            assert_eq!(*question_word, Some(QuestionWord::What));
            assert_eq!(original_input, "What is a dog?");
        }
    }

    #[test]
    fn classify_query_who() {
        let intent = classify_intent("Who are you?");
        assert!(matches!(intent, UserIntent::Query { .. }));
        if let UserIntent::Query {
            subject,
            question_word,
            ..
        } = &intent
        {
            assert_eq!(*question_word, Some(QuestionWord::Who));
            assert_eq!(subject, "you");
        }
    }

    #[test]
    fn classify_assert() {
        let intent = classify_intent("Dogs are mammals");
        assert!(matches!(intent, UserIntent::Assert { .. }));
    }

    #[test]
    fn classify_goal() {
        let intent = classify_intent("Find similar animals to Dog");
        assert!(matches!(intent, UserIntent::SetGoal { .. }));
    }

    #[test]
    fn classify_status() {
        let intent = classify_intent("status");
        assert!(matches!(intent, UserIntent::ShowStatus));
    }

    #[test]
    fn classify_render() {
        let intent = classify_intent("show Dog");
        assert!(matches!(intent, UserIntent::RenderHiero { .. }));
        if let UserIntent::RenderHiero { entity } = classify_intent("show Dog") {
            assert_eq!(entity.as_deref(), Some("Dog"));
        }
    }

    #[test]
    fn classify_run() {
        let intent = classify_intent("run 5 cycles");
        assert!(matches!(intent, UserIntent::RunAgent { cycles: Some(5) }));
    }

    #[test]
    fn classify_help() {
        let intent = classify_intent("help");
        assert!(matches!(intent, UserIntent::Help));
    }

    #[test]
    fn classify_freeform() {
        let intent = classify_intent("hello world");
        assert!(matches!(intent, UserIntent::Freeform { .. }));
    }

    #[test]
    fn extract_subject_from_question_basic() {
        let subject = extract_subject_from_question("What is a dog?");
        assert_eq!(subject, "dog");
    }

    #[test]
    fn extract_subject_capability_question() {
        // "What can you do?" should extract "you", not "you do".
        let subject = extract_subject_from_question("What can you do?");
        assert_eq!(subject, "you");
    }

    #[test]
    fn extract_subject_who_are_you() {
        let subject = extract_subject_from_question("Who are you?");
        assert_eq!(subject, "you");
    }
}
