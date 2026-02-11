//! Recursive descent parser: prose → abstract syntax trees.
//!
//! Parses natural language input into [`AbsTree`] nodes using a priority
//! cascade: commands → questions → compound sentences → declarative facts → freeform.
//!
//! The parser is hand-rolled (no external parser combinator dependency) for
//! full control over error messages, recovery, and the small fixed grammar.

use super::abs::AbsTree;
use super::concrete::ParseContext;
use super::error::{GrammarError, GrammarResult};
use super::lexer::{self, CommandKind, Lexicon, Token};

/// The result of parsing prose input.
#[derive(Debug, Clone)]
pub enum ParseResult {
    /// One or more facts extracted as abstract syntax trees.
    Facts(Vec<AbsTree>),
    /// A query with a subject hint.
    Query {
        subject: String,
        tree: AbsTree,
    },
    /// A system command.
    Command(CommandKind),
    /// A goal-setting directive.
    Goal { description: String },
    /// Free-form text that didn't parse into a structured pattern.
    Freeform {
        text: String,
        /// Best-effort partial parses, if any.
        partial: Vec<AbsTree>,
    },
}

/// Parse prose input into a [`ParseResult`].
///
/// This is the top-level entry point, trying each strategy in priority order.
pub fn parse_prose(input: &str, ctx: &ParseContext) -> ParseResult {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return ParseResult::Freeform {
            text: String::new(),
            partial: vec![],
        };
    }

    let lexicon = Lexicon::default_english();

    // 1. Commands (highest priority)
    if let Some(cmd) = lexicon.match_command(trimmed) {
        return ParseResult::Command(cmd);
    }

    // 2. Goal-setting verbs
    if let Some(desc) = lexicon.match_goal(trimmed) {
        return ParseResult::Goal { description: desc };
    }

    // 3. Questions
    if is_question(trimmed, &lexicon) {
        let subject = extract_question_subject(trimmed);
        let tree = AbsTree::entity(&subject);
        return ParseResult::Query {
            subject,
            tree,
        };
    }

    // Tokenize for pattern matching
    let tokens = lexer::tokenize(
        trimmed,
        ctx.registry,
        ctx.ops,
        ctx.item_memory,
        &lexicon,
    );

    // 4. Compound sentences (split on "and"/"or")
    if let Some(result) = try_compound(&tokens, &lexicon) {
        return result;
    }

    // 5. Declarative facts (subject-predicate-object)
    if let Some(tree) = try_fact(&tokens, &lexicon) {
        return ParseResult::Facts(vec![tree]);
    }

    // 6. Freeform fallback
    ParseResult::Freeform {
        text: trimmed.to_string(),
        partial: vec![],
    }
}

/// Parse prose using the universal parser (shared by all grammars).
///
/// This is called by concrete grammars that don't override `parse()`.
pub fn parse_universal(input: &str, ctx: &ParseContext) -> GrammarResult<AbsTree> {
    let result = parse_prose(input, ctx);
    match result {
        ParseResult::Facts(facts) => {
            if facts.len() == 1 {
                Ok(facts.into_iter().next().unwrap())
            } else {
                Ok(AbsTree::and(facts))
            }
        }
        ParseResult::Query { subject, .. } => Ok(AbsTree::entity(subject)),
        ParseResult::Command(_) => Err(GrammarError::ParseFailed {
            input: input.to_string(),
        }),
        ParseResult::Goal { description } => Ok(AbsTree::Freeform(description)),
        ParseResult::Freeform { text, .. } => Ok(AbsTree::Freeform(text)),
    }
}

/// Check if input looks like a question.
fn is_question(input: &str, lexicon: &Lexicon) -> bool {
    if input.trim_end().ends_with('?') {
        return true;
    }
    let first_word = input
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_lowercase();
    lexicon.is_question_word(&first_word)
}

/// Extract the subject from a question, stripping question words and auxiliaries.
fn extract_question_subject(input: &str) -> String {
    let s = input.trim().trim_end_matches('?').trim();
    let words: Vec<&str> = s.split_whitespace().collect();
    if words.len() < 2 {
        return s.to_string();
    }

    let first_lower = words[0].to_lowercase();
    let skip = match first_lower.as_str() {
        "what" | "who" | "where" | "when" | "how" | "why" | "which" => {
            if words.len() > 1 {
                let second = words[1].to_lowercase();
                if ["is", "are", "do", "does", "can", "was", "were", "about"]
                    .contains(&second.as_str())
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

    let remaining: String = words[skip..].join(" ");
    let final_words: Vec<&str> = remaining.split_whitespace().collect();
    if final_words.is_empty() {
        return s.to_string();
    }

    let first_remaining = final_words[0].to_lowercase();
    if ["a", "an", "the"].contains(&first_remaining.as_str()) && final_words.len() > 1 {
        final_words[1..].join(" ")
    } else {
        remaining
    }
}

/// Try to parse a compound sentence (split on "and"/"or").
fn try_compound(tokens: &[Token], lexicon: &Lexicon) -> Option<ParseResult> {
    // Find coordination points
    let mut split_points = Vec::new();
    let mut coord_is_and = true;

    for (i, token) in tokens.iter().enumerate() {
        match token.normalized.as_str() {
            "and" => {
                split_points.push(i);
                coord_is_and = true;
            }
            "or" => {
                split_points.push(i);
                coord_is_and = false;
            }
            _ => {}
        }
    }

    if split_points.is_empty() {
        return None;
    }

    // Split token stream at coordination points
    let clauses = split_at_coords(tokens, &split_points);
    if clauses.len() < 2 {
        return None;
    }

    // Try to parse each clause as a fact
    let mut facts = Vec::new();
    for clause_tokens in &clauses {
        if let Some(tree) = try_fact(clause_tokens, lexicon) {
            facts.push(tree);
        }
    }

    if facts.len() < 2 {
        return None;
    }

    let tree = if coord_is_and {
        AbsTree::and(facts.clone())
    } else {
        AbsTree::or(facts.clone())
    };

    Some(ParseResult::Facts(vec![tree]))
}

/// Split tokens at coordination points into separate clause token slices.
fn split_at_coords<'a>(tokens: &'a [Token], split_points: &[usize]) -> Vec<&'a [Token]> {
    let mut clauses = Vec::new();
    let mut start = 0;

    for &point in split_points {
        if point > start {
            clauses.push(&tokens[start..point]);
        }
        start = point + 1; // skip the "and"/"or" token
    }

    if start < tokens.len() {
        clauses.push(&tokens[start..]);
    }

    clauses
}

/// Try to parse a token stream as a declarative fact (subject-predicate-object).
fn try_fact(tokens: &[Token], lexicon: &Lexicon) -> Option<AbsTree> {
    if tokens.is_empty() {
        return None;
    }

    // Try each relational pattern (longest patterns tried first due to lexicon ordering)
    for pattern in lexicon.relational_patterns() {
        if let Some((subj_end, obj_start)) = lexer::find_relational_pattern(tokens, pattern) {
            let subject = tokens_to_entity(&tokens[..subj_end]);
            let object = tokens_to_entity(&tokens[obj_start..]);

            if subject.is_none() || object.is_none() {
                continue;
            }

            let subject = subject.unwrap();
            let predicate = AbsTree::relation(&pattern.predicate_label);
            let object = object.unwrap();

            let confidence = compute_confidence(&subject, &object, pattern);

            if (confidence - 1.0).abs() < f32::EPSILON {
                return Some(AbsTree::triple(subject, predicate, object));
            } else {
                return Some(AbsTree::triple_with_confidence(
                    subject, predicate, object, confidence,
                ));
            }
        }
    }

    // Fallback: if there are exactly 3 content tokens, treat as S-P-O
    let content: Vec<&Token> = tokens.iter().filter(|t| !t.semantically_void).collect();
    if content.len() == 3 {
        let subject = token_to_entity(content[0]);
        let predicate = AbsTree::relation(&content[1].normalized);
        let object = token_to_entity(content[2]);
        return Some(AbsTree::WithConfidence {
            inner: Box::new(AbsTree::triple(subject, predicate, object)),
            confidence: 0.7,
        });
    }

    None
}

/// Convert a slice of tokens to an entity reference.
fn tokens_to_entity(tokens: &[Token]) -> Option<AbsTree> {
    let content: Vec<&Token> = tokens.iter().filter(|t| !t.semantically_void).collect();

    if content.is_empty() {
        return None;
    }

    if content.len() == 1 {
        return Some(token_to_entity(content[0]));
    }

    // Multi-word entity: join labels
    let label: String = content
        .iter()
        .map(|t| t.surface.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    // Use the resolution from the first token that has one
    let symbol_id = content.iter().find_map(|t| match &t.resolution {
        lexer::Resolution::Exact(id) | lexer::Resolution::Compound { symbol_id: id, .. } => {
            Some(*id)
        }
        lexer::Resolution::Fuzzy { symbol_id, .. } => Some(*symbol_id),
        lexer::Resolution::Unresolved => None,
    });

    Some(AbsTree::EntityRef { label, symbol_id })
}

/// Convert a single token to an entity reference.
fn token_to_entity(token: &Token) -> AbsTree {
    let symbol_id = match &token.resolution {
        lexer::Resolution::Exact(id) | lexer::Resolution::Compound { symbol_id: id, .. } => {
            Some(*id)
        }
        lexer::Resolution::Fuzzy { symbol_id, .. } => Some(*symbol_id),
        lexer::Resolution::Unresolved => None,
    };

    AbsTree::EntityRef {
        label: token.surface.clone(),
        symbol_id,
    }
}

/// Compute confidence from resolution quality and pattern base confidence.
fn compute_confidence(subject: &AbsTree, object: &AbsTree, pattern: &lexer::RelationalPattern) -> f32 {
    let base = pattern.default_confidence;
    let subj_factor = resolution_confidence(subject);
    let obj_factor = resolution_confidence(object);
    // Geometric mean
    (base * subj_factor * obj_factor).cbrt().min(1.0)
}

fn resolution_confidence(tree: &AbsTree) -> f32 {
    match tree {
        AbsTree::EntityRef {
            symbol_id: Some(_), ..
        } => 1.0,
        AbsTree::EntityRef {
            symbol_id: None, ..
        } => 0.8,
        _ => 0.8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(input: &str) -> ParseResult {
        let ctx = ParseContext::default();
        parse_prose(input, &ctx)
    }

    #[test]
    fn parse_is_a() {
        let result = parse("Dogs are mammals");
        assert!(matches!(result, ParseResult::Facts(_)));
        if let ParseResult::Facts(facts) = result {
            assert_eq!(facts.len(), 1);
            // Without a registry, tokens are unresolved, so confidence < 1.0
            // and the triple is wrapped in WithConfidence.
            let cat = facts[0].cat();
            assert!(
                cat == super::super::cat::Cat::Statement
                    || cat == super::super::cat::Cat::Confidence,
                "expected Statement or Confidence, got {cat:?}"
            );
        }
    }

    #[test]
    fn parse_is_a_article() {
        let result = parse("A dog is a mammal");
        assert!(matches!(result, ParseResult::Facts(_)));
    }

    #[test]
    fn parse_has_a() {
        let result = parse("Dog has a tail");
        assert!(matches!(result, ParseResult::Facts(_)));
    }

    #[test]
    fn parse_contains() {
        let result = parse("Module contains function");
        assert!(matches!(result, ParseResult::Facts(_)));
    }

    #[test]
    fn parse_part_of() {
        let result = parse("Wheel is part of car");
        assert!(matches!(result, ParseResult::Facts(_)));
    }

    #[test]
    fn parse_question() {
        let result = parse("What is a dog?");
        assert!(matches!(result, ParseResult::Query { .. }));
        if let ParseResult::Query { subject, .. } = result {
            assert_eq!(subject, "dog");
        }
    }

    #[test]
    fn parse_question_trailing() {
        let result = parse("Dogs are mammals?");
        // Ends with ? so treated as question
        assert!(matches!(result, ParseResult::Query { .. }));
    }

    #[test]
    fn parse_command_help() {
        let result = parse("help");
        assert!(matches!(result, ParseResult::Command(CommandKind::Help)));
    }

    #[test]
    fn parse_command_status() {
        let result = parse("status");
        assert!(matches!(
            result,
            ParseResult::Command(CommandKind::ShowStatus)
        ));
    }

    #[test]
    fn parse_command_run() {
        let result = parse("run 5");
        assert!(matches!(
            result,
            ParseResult::Command(CommandKind::RunAgent { cycles: Some(5) })
        ));
    }

    #[test]
    fn parse_goal() {
        let result = parse("find similar animals to Dog");
        assert!(matches!(result, ParseResult::Goal { .. }));
    }

    #[test]
    fn parse_compound_and() {
        let result = parse("Dogs are mammals and cats are mammals");
        assert!(matches!(result, ParseResult::Facts(_)));
        if let ParseResult::Facts(facts) = result {
            assert_eq!(facts.len(), 1);
            // Should be a conjunction
            assert_eq!(facts[0].cat(), super::super::cat::Cat::Conjunction);
        }
    }

    #[test]
    fn parse_freeform() {
        let result = parse("hello world");
        assert!(matches!(result, ParseResult::Freeform { .. }));
    }

    #[test]
    fn parse_empty() {
        let result = parse("");
        assert!(matches!(result, ParseResult::Freeform { .. }));
    }

    #[test]
    fn parse_complex_assertion() {
        let result = parse("The vsa module is located in the engine");
        assert!(matches!(result, ParseResult::Facts(_)));
    }
}
