//! Recursive descent parser: prose → abstract syntax trees.
//!
//! Parses natural language input into [`AbsTree`] nodes using a priority
//! cascade: commands → questions → compound sentences → declarative facts → freeform.
//!
//! The parser is hand-rolled (no external parser combinator dependency) for
//! full control over error messages, recovery, and the small fixed grammar.

use super::abs::{AbsTree, CompareOrd, TemporalExpr};
use super::concrete::ParseContext;
use super::error::{GrammarError, GrammarResult};
use super::lexer::{self, CommandKind, Language, Lexicon, Token};

/// The result of parsing prose input.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ParseResult {
    /// One or more facts extracted as abstract syntax trees.
    Facts(Vec<AbsTree>),
    /// A query with a subject hint.
    Query { subject: String, tree: AbsTree },
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

    // Select lexicon: explicit override > language-based > auto-detect
    let effective_lang = if ctx.language == Language::Auto {
        super::detect::detect_language(trimmed).language
    } else {
        ctx.language
    };
    let lexicon = ctx
        .lexicon
        .clone()
        .unwrap_or_else(|| Lexicon::for_language(effective_lang));

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
        let frame = lexicon.parse_question_frame(trimmed);
        let subject = frame.subject_tokens.join(" ");
        let tree = AbsTree::entity(&subject);
        return ParseResult::Query { subject, tree };
    }

    // Tokenize for pattern matching
    let tokens = lexer::tokenize(trimmed, ctx.registry, ctx.ops, ctx.item_memory, &lexicon);

    // 3a. Conditionals ("if X then Y")
    if let Some(tree) = try_conditional(trimmed, &tokens, &lexicon) {
        return ParseResult::Facts(vec![tree]);
    }

    // 3b. Modals ("want to X", "must Y")
    if let Some(tree) = try_modal(trimmed, &tokens, &lexicon) {
        return ParseResult::Facts(vec![tree]);
    }

    // 3c. Temporals ("tomorrow X", "now X")
    if let Some(tree) = try_temporal(trimmed, &tokens, &lexicon) {
        return ParseResult::Facts(vec![tree]);
    }

    // 3d. Negation ("not X", "X is not Y")
    if let Some(tree) = try_negation(trimmed, &tokens, &lexicon) {
        return ParseResult::Facts(vec![tree]);
    }

    // 3e. Quantified ("all X are Y")
    if let Some(tree) = try_quantified(trimmed, &tokens, &lexicon) {
        return ParseResult::Facts(vec![tree]);
    }

    // 3f. Comparative ("X is bigger than Y")
    if let Some(tree) = try_comparative(trimmed, &tokens, &lexicon) {
        return ParseResult::Facts(vec![tree]);
    }

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
    let first_word = input.split_whitespace().next().unwrap_or("").to_lowercase();
    lexicon.is_question_word(&first_word)
}

// ── NLU recognizer functions ──────────────────────────────────────────

/// Try to parse a conditional: "if X then Y" / "если X то Y" / "si X entonces Y".
///
/// Strategy: find a conditional trigger word, split at comma/"then", parse halves.
fn try_conditional(input: &str, tokens: &[Token], lexicon: &Lexicon) -> Option<AbsTree> {
    // Check if first content token is a conditional trigger
    let first_content = tokens.iter().find(|t| !t.semantically_void)?;
    if !lexicon.is_conditional_trigger(&first_content.normalized) {
        return None;
    }

    // Remove the trigger word from the input
    let after_trigger = input
        .trim()
        .get(first_content.span.end..)?
        .trim()
        .trim_start_matches(',')
        .trim();

    if after_trigger.is_empty() {
        return None;
    }

    // Split at "then"/comma — find the split point
    let (cond_str, cons_str) = split_conditional(after_trigger, lexicon)?;

    let condition = AbsTree::Freeform(cond_str.trim().to_string());
    let consequent = AbsTree::Freeform(cons_str.trim().to_string());

    Some(AbsTree::conditional(condition, consequent))
}

/// Split conditional body into condition and consequent parts.
fn split_conditional<'a>(input: &'a str, _lexicon: &Lexicon) -> Option<(&'a str, &'a str)> {
    // Try splitting at known delimiters: "then", "то", "entonces", "alors", comma
    let delimiters = [", then ", " then ", ",then ", ", то ", " то ", ", entonces ", " entonces ", ", alors ", " alors "];
    for delim in &delimiters {
        if let Some(pos) = input.to_lowercase().find(delim) {
            let cond = &input[..pos];
            let cons = &input[pos + delim.len()..];
            if !cond.is_empty() && !cons.is_empty() {
                return Some((cond, cons));
            }
        }
    }
    // Try splitting at comma
    if let Some(pos) = input.find(',') {
        let cond = &input[..pos];
        let cons = &input[pos + 1..];
        if !cond.trim().is_empty() && !cons.trim().is_empty() {
            return Some((cond, cons.trim_start()));
        }
    }
    None
}

/// Try to parse a modal: "want to X" / "хочу X" / "quiero X".
///
/// Strategy: check first 1-2 tokens for a modal verb, parse remainder as scope.
fn try_modal(input: &str, tokens: &[Token], lexicon: &Lexicon) -> Option<AbsTree> {
    let content_tokens: Vec<&Token> = tokens.iter().filter(|t| !t.semantically_void).collect();
    if content_tokens.is_empty() {
        return None;
    }

    // Check first content token for modal verb
    let first = &content_tokens[0];
    let modality = lexicon.modality_for(&first.normalized)?;

    // Get the rest of the input after the modal verb
    let after_modal = input.trim().get(first.span.end..)?.trim();

    // Strip common infinitive markers: "to", "que", "de"
    let scope_str = after_modal
        .strip_prefix("to ")
        .or_else(|| after_modal.strip_prefix("que "))
        .or_else(|| after_modal.strip_prefix("de "))
        .or_else(|| after_modal.strip_prefix("أن "))
        .unwrap_or(after_modal)
        .trim();

    if scope_str.is_empty() {
        return None;
    }

    let inner = AbsTree::Freeform(scope_str.to_string());
    Some(AbsTree::modal(modality, inner))
}

/// Try to parse a temporal expression: "tomorrow X" / "сейчас X" / "الآن X".
///
/// Strategy: find a temporal word (typically at start or end), extract TemporalExpr::Named.
fn try_temporal(input: &str, tokens: &[Token], lexicon: &Lexicon) -> Option<AbsTree> {
    let content_tokens: Vec<&Token> = tokens.iter().filter(|t| !t.semantically_void).collect();
    if content_tokens.len() < 2 {
        return None;
    }

    // Check if first content token is a temporal word
    let first = &content_tokens[0];
    if lexicon.is_temporal_word(&first.normalized) {
        let after = input.trim().get(first.span.end..)?.trim();
        if after.is_empty() {
            return None;
        }
        let time_expr = TemporalExpr::Named(first.surface.to_string());
        let inner = AbsTree::Freeform(after.to_string());
        return Some(AbsTree::temporal(time_expr, inner));
    }

    // Check if last content token is a temporal word
    let last = content_tokens.last()?;
    if lexicon.is_temporal_word(&last.normalized) {
        let before = input.trim().get(..last.span.start)?.trim();
        if before.is_empty() {
            return None;
        }
        let time_expr = TemporalExpr::Named(last.surface.to_string());
        let inner = AbsTree::Freeform(before.to_string());
        return Some(AbsTree::temporal(time_expr, inner));
    }

    None
}

/// Try to parse negation: "not X" / "dogs are not cats" / "не X".
///
/// Strategy: find a negation word, remove it, parse remainder, wrap in Negation.
fn try_negation(input: &str, tokens: &[Token], lexicon: &Lexicon) -> Option<AbsTree> {
    // Find negation word in the token stream
    let neg_pos = tokens
        .iter()
        .position(|t| lexicon.is_negation_word(&t.normalized))?;

    let neg_token = &tokens[neg_pos];

    // Build the remainder text without the negation word
    let before = input.get(..neg_token.span.start).unwrap_or("").trim();
    let after = input.get(neg_token.span.end..).unwrap_or("").trim();

    let remainder = if before.is_empty() {
        after.to_string()
    } else if after.is_empty() {
        before.to_string()
    } else {
        format!("{before} {after}")
    };

    if remainder.is_empty() {
        return None;
    }

    let inner = AbsTree::Freeform(remainder);
    Some(AbsTree::negation(inner))
}

/// Try to parse quantified statement: "all dogs are mammals" / "все собаки — млекопитающие".
///
/// Strategy: check first token for quantifier, map to Quantifier, parse scope.
fn try_quantified(input: &str, tokens: &[Token], lexicon: &Lexicon) -> Option<AbsTree> {
    let content_tokens: Vec<&Token> = tokens.iter().filter(|t| !t.semantically_void).collect();
    if content_tokens.len() < 2 {
        return None;
    }

    let first = &content_tokens[0];
    let quantifier = lexicon.quantifier_for(&first.normalized)?;

    let after = input.trim().get(first.span.end..)?.trim();
    if after.is_empty() {
        return None;
    }

    let scope = AbsTree::Freeform(after.to_string());
    Some(AbsTree::quantified(quantifier, scope))
}

/// Try to parse a comparative: "X is bigger than Y" / "X больше чем Y".
///
/// Strategy: scan for comparative word + "than"/"чем", extract entities & property.
fn try_comparative(input: &str, tokens: &[Token], lexicon: &Lexicon) -> Option<AbsTree> {
    // Find the comparative word
    let comp_pos = tokens
        .iter()
        .position(|t| lexicon.is_comparative_word(&t.normalized))?;

    let comp_token = &tokens[comp_pos];
    let property = comp_token.surface.clone();

    // Find "than"/"чем"/"que"/"de" after the comparative
    let than_words = ["than", "чем", "que", "de", "من"];
    let than_pos = tokens[comp_pos + 1..]
        .iter()
        .position(|t| than_words.contains(&t.normalized.as_str()))
        .map(|p| p + comp_pos + 1)?;

    let than_token = &tokens[than_pos];

    // Entity A: everything before the comparative (skip copulas like "is")
    let before_comp = input.get(..comp_token.span.start).unwrap_or("").trim();
    // Strip trailing copulas
    let entity_a_str = before_comp
        .trim_end_matches(" is")
        .trim_end_matches(" are")
        .trim_end_matches(" es")
        .trim_end_matches(" est")
        .trim();

    // Entity B: everything after "than"
    let entity_b_str = input.get(than_token.span.end..).unwrap_or("").trim();

    if entity_a_str.is_empty() || entity_b_str.is_empty() {
        return None;
    }

    let entity_a = AbsTree::entity(entity_a_str);
    let entity_b = AbsTree::entity(entity_b_str);

    // Default to GreaterThan for comparatives like "bigger", "more"
    // Could be refined with word-level semantics later
    let ordering = CompareOrd::GreaterThan;

    Some(AbsTree::comparison(entity_a, entity_b, property, ordering))
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
fn compute_confidence(
    subject: &AbsTree,
    object: &AbsTree,
    pattern: &lexer::RelationalPattern,
) -> f32 {
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

    #[test]
    fn parse_capability_question() {
        let result = parse("What can you do?");
        assert!(matches!(result, ParseResult::Query { .. }));
        if let ParseResult::Query { subject, .. } = result {
            assert_eq!(subject, "you");
        }
    }

    // ── NLU recognizer tests ────────────────────────────────────

    use super::super::cat::Cat;

    fn parse_facts_cat(input: &str) -> Option<Cat> {
        match parse(input) {
            ParseResult::Facts(facts) if !facts.is_empty() => Some(facts[0].cat()),
            _ => None,
        }
    }

    // Conditional tests
    #[test]
    fn parse_conditional_if_then() {
        let cat = parse_facts_cat("if it rains, cancel the meeting");
        assert_eq!(cat, Some(Cat::Conditional));
    }

    #[test]
    fn parse_conditional_if_then_explicit() {
        let cat = parse_facts_cat("if it rains then stay inside");
        assert_eq!(cat, Some(Cat::Conditional));
    }

    // Modal tests
    #[test]
    fn parse_modal_want() {
        let cat = parse_facts_cat("want to learn rust");
        assert_eq!(cat, Some(Cat::Modal));
    }

    #[test]
    fn parse_modal_must() {
        let cat = parse_facts_cat("must finish the report");
        assert_eq!(cat, Some(Cat::Modal));
    }

    #[test]
    fn parse_modal_could() {
        let cat = parse_facts_cat("could learn faster");
        assert_eq!(cat, Some(Cat::Modal));
    }

    // Temporal tests
    #[test]
    fn parse_temporal_tomorrow() {
        let cat = parse_facts_cat("tomorrow finish the report");
        assert_eq!(cat, Some(Cat::Temporal));
    }

    #[test]
    fn parse_temporal_now() {
        let cat = parse_facts_cat("now start the meeting");
        assert_eq!(cat, Some(Cat::Temporal));
    }

    // Negation tests
    #[test]
    fn parse_negation_not() {
        let cat = parse_facts_cat("not a mammal");
        assert_eq!(cat, Some(Cat::Negation));
    }

    #[test]
    fn parse_negation_never() {
        let cat = parse_facts_cat("never eat fish");
        assert_eq!(cat, Some(Cat::Negation));
    }

    // Quantified tests
    #[test]
    fn parse_quantified_all() {
        let cat = parse_facts_cat("all dogs are mammals");
        assert_eq!(cat, Some(Cat::Quantifier));
    }

    #[test]
    fn parse_quantified_every() {
        let cat = parse_facts_cat("every student passed");
        assert_eq!(cat, Some(Cat::Quantifier));
    }

    #[test]
    fn parse_quantified_some() {
        let cat = parse_facts_cat("some birds can fly");
        assert_eq!(cat, Some(Cat::Quantifier));
    }

    // Comparative tests
    #[test]
    fn parse_comparative_bigger() {
        let cat = parse_facts_cat("elephants are bigger than mice");
        assert_eq!(cat, Some(Cat::Comparison));
    }

    #[test]
    fn parse_comparative_more() {
        let cat = parse_facts_cat("gold is more valuable than silver");
        assert_eq!(cat, Some(Cat::Comparison));
    }

    // Russian NLU tests
    #[test]
    fn parse_negation_russian() {
        let cat = parse_facts_cat("не млекопитающее");
        assert_eq!(cat, Some(Cat::Negation));
    }

    #[test]
    fn parse_modal_russian() {
        let cat = parse_facts_cat("хочу учить русский");
        assert_eq!(cat, Some(Cat::Modal));
    }

    #[test]
    fn parse_conditional_russian() {
        let cat = parse_facts_cat("если идёт дождь, останься дома");
        assert_eq!(cat, Some(Cat::Conditional));
    }

    // French NLU tests
    #[test]
    fn parse_negation_french() {
        // "pas" is a negation word
        let cat = parse_facts_cat("pas un mammifère");
        assert_eq!(cat, Some(Cat::Negation));
    }

    #[test]
    fn parse_modal_french() {
        let cat = parse_facts_cat("veux apprendre le français");
        assert_eq!(cat, Some(Cat::Modal));
    }

    // Additional negation tests
    #[test]
    fn parse_negation_dogs_not_fish() {
        let cat = parse_facts_cat("not a valid solution");
        assert_eq!(cat, Some(Cat::Negation));
    }

    #[test]
    fn parse_negation_cannot() {
        let cat = parse_facts_cat("cannot proceed further");
        assert_eq!(cat, Some(Cat::Negation));
    }

    // Cascade priority tests
    #[test]
    fn parse_conditional_before_negation() {
        // "if" conditional trigger should take priority over "not" negation
        let cat = parse_facts_cat("if not ready, then wait");
        assert_eq!(cat, Some(Cat::Conditional));
    }

    #[test]
    fn parse_temporal_before_negation() {
        // Temporal word at start takes priority
        let cat = parse_facts_cat("tomorrow not available");
        assert_eq!(cat, Some(Cat::Temporal));
    }
}
