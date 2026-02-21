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
        /// Whether the question's auxiliary signals capability ("can", "peut", etc.).
        capability_signal: bool,
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
    /// "Set detail concise/normal/full" — change response detail level.
    SetDetail { level: String },
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

    // Set detail level.
    if lower.starts_with("set detail ") || lower.starts_with("detail ") {
        let rest = if lower.starts_with("set detail ") {
            trimmed[11..].trim()
        } else {
            trimmed[7..].trim()
        };
        return UserIntent::SetDetail {
            level: rest.to_lowercase(),
        };
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

    // Query: starts with question word, auxiliary verb, or ends with '?'.
    let lexicon = crate::grammar::lexer::Lexicon::for_language(
        crate::grammar::lexer::Language::default(),
    );
    let question_word = extract_question_word(&lower);
    let first_word = lower.split_whitespace().next().unwrap_or("");
    if question_word.is_some()
        || lexicon.is_auxiliary_verb(first_word)
        || trimmed.ends_with('?')
    {
        let frame = lexicon.parse_question_frame(trimmed);
        let subject = frame.subject_tokens.join(" ");
        let qw = question_word.or_else(|| {
            if frame.auxiliary.is_some() && frame.question_word.is_none() {
                Some(QuestionWord::YesNo)
            } else {
                None
            }
        });
        return UserIntent::Query {
            subject,
            original_input: trimmed.to_string(),
            question_word: qw,
            capability_signal: frame.signals_capability,
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
    fn classify_set_detail() {
        let intent = classify_intent("set detail concise");
        assert!(matches!(intent, UserIntent::SetDetail { level } if level == "concise"));

        let intent = classify_intent("detail full");
        assert!(matches!(intent, UserIntent::SetDetail { level } if level == "full"));

        let intent = classify_intent("set detail Normal");
        assert!(matches!(intent, UserIntent::SetDetail { level } if level == "normal"));
    }

    #[test]
    fn question_frame_basic() {
        let lexicon = crate::grammar::lexer::Lexicon::default_english();
        let frame = lexicon.parse_question_frame("What is a dog?");
        assert_eq!(frame.subject_tokens.join(" "), "dog");
    }

    #[test]
    fn question_frame_capability() {
        // "What can you do?" should extract "you", not "you do".
        let lexicon = crate::grammar::lexer::Lexicon::default_english();
        let frame = lexicon.parse_question_frame("What can you do?");
        assert_eq!(frame.subject_tokens.join(" "), "you");
        assert!(frame.signals_capability);
        assert!(frame.trailing_stripped);
    }

    #[test]
    fn question_frame_who_are_you() {
        let lexicon = crate::grammar::lexer::Lexicon::default_english();
        let frame = lexicon.parse_question_frame("Who are you?");
        assert_eq!(frame.subject_tokens.join(" "), "you");
        assert!(!frame.signals_capability);
    }
}
