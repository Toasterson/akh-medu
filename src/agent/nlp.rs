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
    /// "Why X?" / "How confident?" / "What do you know about X?" / "Explain X"
    Explain {
        query: super::explain::ExplanationQuery,
    },
    /// A structured agent-to-agent protocol message (Phase 12g).
    ///
    /// These bypass the NLP classifier entirely — they arrive as
    /// `MessageContent::AgentMessage` and are dispatched to the
    /// multi-agent handler for capability validation and processing.
    AgentProtocol {
        message: super::multi_agent::AgentProtocolMessage,
    },
    /// PIM command (Phase 13e): "pim inbox", "pim next", etc.
    PimCommand { subcommand: String, args: String },
    /// Calendar command (Phase 13f): "cal today", "cal add ...", etc.
    CalCommand { subcommand: String, args: String },
    /// Preference command (Phase 13g): "pref status", "pref train ...", etc.
    PrefCommand { subcommand: String, args: String },
    /// Causal reasoning command (Phase 15a): "causal schemas", "causal predict ...", etc.
    CausalQuery { subcommand: String, args: String },
    /// Awaken command (Phase 14a+14b): "awaken parse ...", "awaken resolve ...", etc.
    AwakenCommand { subcommand: String, args: String },
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

    // Awaken commands (Phase 14a+14b).
    if lower == "awaken" || lower.starts_with("awaken ") {
        let rest = if lower == "awaken" {
            ""
        } else {
            trimmed[7..].trim()
        };
        let mut parts = rest.splitn(2, char::is_whitespace);
        let subcommand = parts.next().unwrap_or("").to_string();
        let args = parts.next().unwrap_or("").to_string();
        return UserIntent::AwakenCommand { subcommand, args };
    }

    // Causal reasoning commands (Phase 15a).
    if lower == "causal" || lower.starts_with("causal ") {
        let rest = if lower == "causal" {
            ""
        } else {
            trimmed[7..].trim()
        };
        let mut parts = rest.splitn(2, char::is_whitespace);
        let subcommand = parts.next().unwrap_or("").to_string();
        let args = parts.next().unwrap_or("").to_string();
        return UserIntent::CausalQuery { subcommand, args };
    }

    // Preference commands (Phase 13g) — checked before PIM.
    if lower == "pref" || lower.starts_with("pref ") {
        let rest = if lower == "pref" {
            ""
        } else {
            trimmed[5..].trim()
        };
        let mut parts = rest.splitn(2, char::is_whitespace);
        let subcommand = parts.next().unwrap_or("").to_string();
        let args = parts.next().unwrap_or("").to_string();
        return UserIntent::PrefCommand { subcommand, args };
    }

    // PIM commands (Phase 13e).
    if lower == "pim" || lower.starts_with("pim ") {
        let rest = if lower == "pim" {
            ""
        } else {
            trimmed[4..].trim()
        };
        let mut parts = rest.splitn(2, char::is_whitespace);
        let subcommand = parts.next().unwrap_or("").to_string();
        let args = parts.next().unwrap_or("").to_string();
        return UserIntent::PimCommand { subcommand, args };
    }

    // Calendar commands (Phase 13f).
    if lower == "cal" || lower.starts_with("cal ") {
        let rest = if lower == "cal" {
            ""
        } else {
            trimmed[4..].trim()
        };
        let mut parts = rest.splitn(2, char::is_whitespace);
        let subcommand = parts.next().unwrap_or("").to_string();
        let args = parts.next().unwrap_or("").to_string();
        return UserIntent::CalCommand { subcommand, args };
    }

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

    // Explanation queries (Phase 12f): "why X?", "explain X", "how confident", etc.
    if let Some(query) = super::explain::ExplanationQuery::parse(trimmed) {
        return UserIntent::Explain { query };
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

/// Classification of conversational (non-query, non-command) user input.
///
/// Used to distinguish greetings, follow-ups, and acknowledgments from
/// freeform input that should escalate to an autonomous goal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversationalKind {
    /// "hello", "hi", "hey", "good morning", etc.
    Greeting,
    /// "tell me more", "go on", "what else", "elaborate", etc.
    FollowUp,
    /// "thanks", "ok", "got it", "interesting", etc.
    Acknowledgment,
    /// "what can you do", "describe yourself", "who are you" (sans question mark).
    MetaQuestion,
    /// Fallback — still escalate to goal.
    Unrecognized,
}

/// Classify freeform text into a conversational kind using keyword-feature scoring.
///
/// Uses the language-aware [`Lexicon`] for all word/phrase lookups, so the same
/// scoring logic works for English, Russian, Arabic, French, Spanish, etc.
///
/// Extracts features (keyword presence, position, co-occurrence) and scores each
/// category. The highest-scoring category wins if it meets the threshold.
pub fn classify_conversational(
    text: &str,
    lexicon: &crate::grammar::lexer::Lexicon,
) -> ConversationalKind {
    let lower = text.trim().to_lowercase();
    if lower.is_empty() {
        return ConversationalKind::Unrecognized;
    }

    let words: Vec<&str> = lower.split_whitespace().collect();
    if words.is_empty() {
        return ConversationalKind::Unrecognized;
    }

    let mut scores = [
        (ConversationalKind::Greeting, 0i32),
        (ConversationalKind::FollowUp, 0),
        (ConversationalKind::Acknowledgment, 0),
        (ConversationalKind::MetaQuestion, 0),
    ];

    // ── Greeting features ────────────────────────────────────────────
    let has_greeting_word = words.iter().any(|w| {
        let stripped = w.trim_matches(|c: char| c.is_ascii_punctuation());
        lexicon.is_greeting_word(stripped)
    });
    let has_greeting_phrase = lexicon.has_greeting_phrase(&lower);
    let greeting_at_start = words.first().is_some_and(|w| {
        let stripped = w.trim_matches(|c: char| c.is_ascii_punctuation());
        lexicon.is_greeting_word(stripped)
    }) || has_greeting_phrase;

    if greeting_at_start {
        scores[0].1 += 3;
    } else if has_greeting_word {
        scores[0].1 += 1;
    }

    // ── Follow-up features ───────────────────────────────────────────
    let has_followup_phrase = lexicon.has_followup_phrase(&lower);
    let has_followup_cue = words.iter().any(|w| {
        let stripped = w.trim_matches(|c: char| c.is_ascii_punctuation());
        lexicon.is_followup_cue(stripped)
    });
    let short_followup = has_followup_cue && words.len() <= 4;

    if has_followup_phrase {
        scores[1].1 += 3;
    } else if short_followup {
        scores[1].1 += 2;
    }

    // ── Acknowledgment features ──────────────────────────────────────
    let has_ack_phrase = lexicon.has_ack_phrase(&lower);
    let has_ack_word = words.iter().any(|w| {
        let stripped = w.trim_matches(|c: char| c.is_ascii_punctuation());
        lexicon.is_ack_word(stripped)
    });
    let terse_ack = has_ack_word && words.len() <= 3;

    if has_ack_phrase {
        scores[2].1 += 3;
    } else if terse_ack {
        scores[2].1 += 3;
    } else if has_ack_word {
        scores[2].1 += 1;
    }

    // ── Meta-question features ───────────────────────────────────────
    let has_meta_phrase = lexicon.has_meta_phrase(&lower);
    let has_self_ref = words.iter().any(|w| {
        let stripped = w.trim_matches(|c: char| c.is_ascii_punctuation());
        lexicon.is_meta_self_word(stripped)
    });
    let has_capability_ref = words.iter().any(|w| {
        let stripped = w.trim_matches(|c: char| c.is_ascii_punctuation());
        lexicon.is_meta_capability_word(stripped)
    });

    if has_meta_phrase {
        scores[3].1 += 3;
    } else if has_self_ref && has_capability_ref {
        scores[3].1 += 2;
    }

    // ── Pick winner ──────────────────────────────────────────────────
    const THRESHOLD: i32 = 2;

    scores.sort_by(|a, b| b.1.cmp(&a.1));
    if scores[0].1 >= THRESHOLD {
        scores[0].0
    } else {
        ConversationalKind::Unrecognized
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

    // ── classify_conversational tests ────────────────────────────────────

    fn en() -> crate::grammar::lexer::Lexicon {
        crate::grammar::lexer::Lexicon::default_english()
    }

    #[test]
    fn conversational_greeting_basic() {
        let lex = en();
        assert_eq!(classify_conversational("hello", &lex), ConversationalKind::Greeting);
        assert_eq!(classify_conversational("Hi", &lex), ConversationalKind::Greeting);
        assert_eq!(classify_conversational("good morning", &lex), ConversationalKind::Greeting);
        assert_eq!(classify_conversational("Hey there", &lex), ConversationalKind::Greeting);
    }

    #[test]
    fn conversational_greeting_natural() {
        let lex = en();
        assert_eq!(classify_conversational("hello!", &lex), ConversationalKind::Greeting);
        assert_eq!(classify_conversational("hey, how's it going", &lex), ConversationalKind::Greeting);
        assert_eq!(classify_conversational("Good afternoon", &lex), ConversationalKind::Greeting);
    }

    #[test]
    fn conversational_follow_up_basic() {
        let lex = en();
        assert_eq!(classify_conversational("tell me more", &lex), ConversationalKind::FollowUp);
        assert_eq!(classify_conversational("go on", &lex), ConversationalKind::FollowUp);
        assert_eq!(classify_conversational("what else", &lex), ConversationalKind::FollowUp);
    }

    #[test]
    fn conversational_follow_up_natural() {
        let lex = en();
        assert_eq!(classify_conversational("elaborate please", &lex), ConversationalKind::FollowUp);
        assert_eq!(classify_conversational("can you expand on that", &lex), ConversationalKind::FollowUp);
        assert_eq!(classify_conversational("more details", &lex), ConversationalKind::FollowUp);
    }

    #[test]
    fn conversational_acknowledgment_basic() {
        let lex = en();
        assert_eq!(classify_conversational("thanks", &lex), ConversationalKind::Acknowledgment);
        assert_eq!(classify_conversational("ok", &lex), ConversationalKind::Acknowledgment);
        assert_eq!(classify_conversational("got it", &lex), ConversationalKind::Acknowledgment);
        assert_eq!(classify_conversational("interesting", &lex), ConversationalKind::Acknowledgment);
    }

    #[test]
    fn conversational_acknowledgment_natural() {
        let lex = en();
        assert_eq!(classify_conversational("thank you!", &lex), ConversationalKind::Acknowledgment);
        assert_eq!(classify_conversational("cool, thanks", &lex), ConversationalKind::Acknowledgment);
        assert_eq!(classify_conversational("makes sense", &lex), ConversationalKind::Acknowledgment);
        assert_eq!(classify_conversational("sounds good", &lex), ConversationalKind::Acknowledgment);
    }

    #[test]
    fn conversational_meta_question_basic() {
        let lex = en();
        assert_eq!(classify_conversational("what can you do", &lex), ConversationalKind::MetaQuestion);
        assert_eq!(classify_conversational("describe yourself", &lex), ConversationalKind::MetaQuestion);
        assert_eq!(classify_conversational("who are you", &lex), ConversationalKind::MetaQuestion);
    }

    #[test]
    fn conversational_meta_question_natural() {
        let lex = en();
        assert_eq!(classify_conversational("tell me about yourself", &lex), ConversationalKind::MetaQuestion);
        assert_eq!(classify_conversational("what are you", &lex), ConversationalKind::MetaQuestion);
        assert_eq!(classify_conversational("your capabilities", &lex), ConversationalKind::MetaQuestion);
    }

    #[test]
    fn conversational_unrecognized() {
        let lex = en();
        assert_eq!(classify_conversational("the quick brown fox", &lex), ConversationalKind::Unrecognized);
        assert_eq!(classify_conversational("", &lex), ConversationalKind::Unrecognized);
        assert_eq!(classify_conversational("photosynthesis in plants", &lex), ConversationalKind::Unrecognized);
    }

    // ── Multilingual conversational tests ────────────────────────────

    #[test]
    fn conversational_russian_greeting() {
        let lex = crate::grammar::lexer::Lexicon::default_russian();
        assert_eq!(classify_conversational("привет", &lex), ConversationalKind::Greeting);
        assert_eq!(classify_conversational("доброе утро", &lex), ConversationalKind::Greeting);
    }

    #[test]
    fn conversational_russian_ack() {
        let lex = crate::grammar::lexer::Lexicon::default_russian();
        assert_eq!(classify_conversational("спасибо", &lex), ConversationalKind::Acknowledgment);
    }

    #[test]
    fn conversational_french_greeting() {
        let lex = crate::grammar::lexer::Lexicon::default_french();
        assert_eq!(classify_conversational("bonjour", &lex), ConversationalKind::Greeting);
        assert_eq!(classify_conversational("salut", &lex), ConversationalKind::Greeting);
    }

    #[test]
    fn conversational_french_meta() {
        let lex = crate::grammar::lexer::Lexicon::default_french();
        assert_eq!(classify_conversational("qui es-tu", &lex), ConversationalKind::MetaQuestion);
    }

    #[test]
    fn conversational_spanish_greeting() {
        let lex = crate::grammar::lexer::Lexicon::default_spanish();
        assert_eq!(classify_conversational("hola", &lex), ConversationalKind::Greeting);
        assert_eq!(classify_conversational("buenos días", &lex), ConversationalKind::Greeting);
    }

    #[test]
    fn conversational_arabic_greeting() {
        let lex = crate::grammar::lexer::Lexicon::default_arabic();
        assert_eq!(classify_conversational("مرحبا", &lex), ConversationalKind::Greeting);
        assert_eq!(classify_conversational("السلام عليكم", &lex), ConversationalKind::Greeting);
    }
}
