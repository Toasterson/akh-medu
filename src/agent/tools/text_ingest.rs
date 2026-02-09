//! Text ingest tool: extract triples from natural language using regex patterns.
//!
//! Built-in patterns recognize common relational constructs like "X is a Y",
//! "X has Y", "X contains Y", etc. No LLM required for core extraction.

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::engine::Engine;
use crate::symbol::SymbolId;

/// Extract triples from natural language text and add them to the KG.
pub struct TextIngestTool;

impl Tool for TextIngestTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "text_ingest".into(),
            description: "Extract triples from natural language text using pattern matching \
                          and add them to the knowledge graph."
                .into(),
            parameters: vec![
                ToolParam {
                    name: "text".into(),
                    description: "The text to extract triples from. Can be inline text \
                                  or a file path (prefixed with 'file:')."
                        .into(),
                    required: true,
                },
                ToolParam {
                    name: "max_sentences".into(),
                    description: "Maximum number of sentences to process. Default: 100.".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let text_param = input.require("text", "text_ingest")?;
        let max_sentences: usize = input
            .get("max_sentences")
            .and_then(|s| s.parse().ok())
            .unwrap_or(100);

        let text = if let Some(path) = text_param.strip_prefix("file:") {
            match std::fs::read_to_string(path.trim()) {
                Ok(content) => content,
                Err(e) => {
                    return Ok(ToolOutput::err(format!(
                        "Failed to read \"{path}\": {e}"
                    )));
                }
            }
        } else {
            text_param.to_string()
        };

        let sentences = split_sentences(&text);
        let mut total_extracted = 0usize;
        let mut symbols = Vec::new();

        for sentence in sentences.iter().take(max_sentences) {
            let extracted = extract_triples(sentence);
            for (subject, predicate, object, confidence) in &extracted {
                match ingest_extracted(engine, subject, predicate, object, *confidence) {
                    Ok(syms) => {
                        symbols.extend(syms);
                        total_extracted += 1;
                    }
                    Err(_) => {}
                }
            }
        }

        let msg = format!(
            "Text ingest: processed {} sentence(s), extracted {} triple(s).",
            sentences.len().min(max_sentences),
            total_extracted,
        );
        Ok(ToolOutput::ok_with_symbols(msg, symbols))
    }
}

/// Split text into sentences using basic punctuation rules.
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if ch == '.' || ch == '!' || ch == '?' || ch == '\n' {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() && trimmed.len() > 1 {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }

    // Handle text without sentence-ending punctuation.
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() && trimmed.len() > 1 {
        sentences.push(trimmed);
    }

    sentences
}

/// A single extracted triple from text.
type ExtractedTriple = (String, String, String, f32);

/// Extract triples from a single sentence using regex-like pattern matching.
///
/// Patterns recognized:
/// - "X is a Y" / "X is an Y" → (X, is-a, Y)
/// - "X are Y" / "X are a Y" → (X, is-a, Y)
/// - "X has Y" / "X has a Y" → (X, has-a, Y)
/// - "X contains Y" → (X, contains, Y)
/// - "X is part of Y" → (X, part-of, Y)
/// - "X causes Y" → (X, causes, Y)
/// - "X is located in Y" → (X, located-in, Y)
/// - "X is similar to Y" → (X, similar-to, Y)
/// - "X, which is a Y" → (X, is-a, Y)
/// - "X is made of Y" → (X, composed-of, Y)
fn extract_triples(sentence: &str) -> Vec<ExtractedTriple> {
    let mut results = Vec::new();
    let s = sentence.trim().trim_end_matches(|c: char| c == '.' || c == '!' || c == '?');

    // Normalize whitespace.
    let words: Vec<&str> = s.split_whitespace().collect();
    if words.len() < 3 {
        return results;
    }

    let lower: Vec<String> = words.iter().map(|w| w.to_lowercase()).collect();

    // Try each pattern in order. First match wins per sentence,
    // but some sentences may yield multiple triples.
    try_is_a_pattern(&words, &lower, &mut results);
    try_are_pattern(&words, &lower, &mut results);
    try_has_pattern(&words, &lower, &mut results);
    try_contains_pattern(&words, &lower, &mut results);
    try_part_of_pattern(&words, &lower, &mut results);
    try_causes_pattern(&words, &lower, &mut results);
    try_located_in_pattern(&words, &lower, &mut results);
    try_similar_to_pattern(&words, &lower, &mut results);
    try_which_is_pattern(&words, &lower, &mut results);
    try_made_of_pattern(&words, &lower, &mut results);

    results
}

/// "X is a Y" / "X is an Y"
fn try_is_a_pattern(words: &[&str], lower: &[String], results: &mut Vec<ExtractedTriple>) {
    for i in 0..lower.len().saturating_sub(3) {
        if lower[i + 1] == "is" && (lower[i + 2] == "a" || lower[i + 2] == "an") {
            let subject = capitalize(words[..=i].join(" ").trim_matches(|c: char| !c.is_alphanumeric()));
            let object = capitalize(words[i + 3..].join(" ").trim_matches(|c: char| !c.is_alphanumeric()));
            if !subject.is_empty() && !object.is_empty() {
                results.push((subject, "is-a".into(), object, 0.9));
                return;
            }
        }
    }
}

/// "X are Y" / "X are a Y"
fn try_are_pattern(words: &[&str], lower: &[String], results: &mut Vec<ExtractedTriple>) {
    for i in 0..lower.len().saturating_sub(2) {
        if lower[i + 1] == "are" {
            let subject = capitalize(words[..=i].join(" ").trim_matches(|c: char| !c.is_alphanumeric()));
            let rest_start = if i + 2 < lower.len() && (lower[i + 2] == "a" || lower[i + 2] == "an") {
                i + 3
            } else {
                i + 2
            };
            if rest_start >= words.len() {
                continue;
            }
            let object = capitalize(words[rest_start..].join(" ").trim_matches(|c: char| !c.is_alphanumeric()));
            if !subject.is_empty() && !object.is_empty() {
                results.push((subject, "is-a".into(), object, 0.85));
                return;
            }
        }
    }
}

/// "X has Y" / "X has a Y"
fn try_has_pattern(words: &[&str], lower: &[String], results: &mut Vec<ExtractedTriple>) {
    for i in 0..lower.len().saturating_sub(2) {
        if lower[i + 1] == "has" {
            let subject = capitalize(words[..=i].join(" ").trim_matches(|c: char| !c.is_alphanumeric()));
            let rest_start = if i + 2 < lower.len() && (lower[i + 2] == "a" || lower[i + 2] == "an") {
                i + 3
            } else {
                i + 2
            };
            if rest_start >= words.len() {
                continue;
            }
            let object = capitalize(words[rest_start..].join(" ").trim_matches(|c: char| !c.is_alphanumeric()));
            if !subject.is_empty() && !object.is_empty() {
                results.push((subject, "has-a".into(), object, 0.85));
                return;
            }
        }
    }
}

/// "X contains Y"
fn try_contains_pattern(words: &[&str], lower: &[String], results: &mut Vec<ExtractedTriple>) {
    for i in 0..lower.len().saturating_sub(2) {
        if lower[i + 1] == "contains" {
            let subject = capitalize(words[..=i].join(" ").trim_matches(|c: char| !c.is_alphanumeric()));
            let object = capitalize(words[i + 2..].join(" ").trim_matches(|c: char| !c.is_alphanumeric()));
            if !subject.is_empty() && !object.is_empty() {
                results.push((subject, "contains".into(), object, 0.85));
                return;
            }
        }
    }
}

/// "X is part of Y"
fn try_part_of_pattern(words: &[&str], lower: &[String], results: &mut Vec<ExtractedTriple>) {
    for i in 0..lower.len().saturating_sub(4) {
        if lower[i + 1] == "is" && lower[i + 2] == "part" && lower[i + 3] == "of" {
            let subject = capitalize(words[..=i].join(" ").trim_matches(|c: char| !c.is_alphanumeric()));
            let object = capitalize(words[i + 4..].join(" ").trim_matches(|c: char| !c.is_alphanumeric()));
            if !subject.is_empty() && !object.is_empty() {
                results.push((subject, "part-of".into(), object, 0.9));
                return;
            }
        }
    }
}

/// "X causes Y"
fn try_causes_pattern(words: &[&str], lower: &[String], results: &mut Vec<ExtractedTriple>) {
    for i in 0..lower.len().saturating_sub(2) {
        if lower[i + 1] == "causes" {
            let subject = capitalize(words[..=i].join(" ").trim_matches(|c: char| !c.is_alphanumeric()));
            let object = capitalize(words[i + 2..].join(" ").trim_matches(|c: char| !c.is_alphanumeric()));
            if !subject.is_empty() && !object.is_empty() {
                results.push((subject, "causes".into(), object, 0.85));
                return;
            }
        }
    }
}

/// "X is located in Y"
fn try_located_in_pattern(words: &[&str], lower: &[String], results: &mut Vec<ExtractedTriple>) {
    for i in 0..lower.len().saturating_sub(4) {
        if lower[i + 1] == "is" && lower[i + 2] == "located" && lower[i + 3] == "in" {
            let subject = capitalize(words[..=i].join(" ").trim_matches(|c: char| !c.is_alphanumeric()));
            let object = capitalize(words[i + 4..].join(" ").trim_matches(|c: char| !c.is_alphanumeric()));
            if !subject.is_empty() && !object.is_empty() {
                results.push((subject, "located-in".into(), object, 0.9));
                return;
            }
        }
    }
}

/// "X is similar to Y"
fn try_similar_to_pattern(words: &[&str], lower: &[String], results: &mut Vec<ExtractedTriple>) {
    for i in 0..lower.len().saturating_sub(4) {
        if lower[i + 1] == "is" && lower[i + 2] == "similar" && lower[i + 3] == "to" {
            let subject = capitalize(words[..=i].join(" ").trim_matches(|c: char| !c.is_alphanumeric()));
            let object = capitalize(words[i + 4..].join(" ").trim_matches(|c: char| !c.is_alphanumeric()));
            if !subject.is_empty() && !object.is_empty() {
                results.push((subject, "similar-to".into(), object, 0.8));
                return;
            }
        }
    }
}

/// "X, which is a Y" or "X which is a Y"
fn try_which_is_pattern(words: &[&str], lower: &[String], results: &mut Vec<ExtractedTriple>) {
    for i in 0..lower.len().saturating_sub(4) {
        let w = lower[i + 1].trim_matches(',');
        if w == "which" && lower[i + 2] == "is" && (lower[i + 3] == "a" || lower[i + 3] == "an") {
            let subject = capitalize(words[..=i].join(" ").trim_matches(|c: char| !c.is_alphanumeric()));
            let object = capitalize(words[i + 4..].join(" ").trim_matches(|c: char| !c.is_alphanumeric()));
            if !subject.is_empty() && !object.is_empty() {
                results.push((subject, "is-a".into(), object, 0.85));
                return;
            }
        }
    }
}

/// "X is made of Y"
fn try_made_of_pattern(words: &[&str], lower: &[String], results: &mut Vec<ExtractedTriple>) {
    for i in 0..lower.len().saturating_sub(4) {
        if lower[i + 1] == "is" && lower[i + 2] == "made" && lower[i + 3] == "of" {
            let subject = capitalize(words[..=i].join(" ").trim_matches(|c: char| !c.is_alphanumeric()));
            let object = capitalize(words[i + 4..].join(" ").trim_matches(|c: char| !c.is_alphanumeric()));
            if !subject.is_empty() && !object.is_empty() {
                results.push((subject, "composed-of".into(), object, 0.85));
                return;
            }
        }
    }
}

/// Capitalize the first letter of a string, preserving the rest.
fn capitalize(s: &str) -> String {
    let s = s.trim();
    if s.is_empty() {
        return String::new();
    }
    let mut chars = s.chars();
    let first = chars.next().unwrap().to_uppercase().to_string();
    first + chars.as_str()
}

/// Resolve labels and add a triple to the engine.
fn ingest_extracted(
    engine: &Engine,
    subject: &str,
    predicate: &str,
    object: &str,
    confidence: f32,
) -> Result<Vec<SymbolId>, crate::error::AkhError> {
    let s = engine.resolve_or_create_entity(subject)?;
    let p = engine.resolve_or_create_relation(predicate)?;
    let o = engine.resolve_or_create_entity(object)?;

    let triple = crate::graph::Triple::new(s, p, o).with_confidence(confidence);
    let _ = engine.add_triple(&triple);

    Ok(vec![s, p, o])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::vsa::Dimension;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn extract_is_a() {
        let triples = extract_triples("A dog is a mammal.");
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].1, "is-a");
    }

    #[test]
    fn extract_are() {
        let triples = extract_triples("Dogs are mammals.");
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].1, "is-a");
    }

    #[test]
    fn extract_has() {
        let triples = extract_triples("A bird has a beak.");
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].1, "has-a");
    }

    #[test]
    fn extract_part_of() {
        let triples = extract_triples("The wheel is part of the car.");
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].1, "part-of");
    }

    #[test]
    fn extract_located_in() {
        let triples = extract_triples("Paris is located in France.");
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].1, "located-in");
    }

    #[test]
    fn extract_causes() {
        let triples = extract_triples("Smoking causes cancer.");
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].1, "causes");
    }

    #[test]
    fn extract_made_of() {
        let triples = extract_triples("Water is made of hydrogen and oxygen.");
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].1, "composed-of");
    }

    #[test]
    fn split_sentences_basic() {
        let sentences = split_sentences("Dogs are mammals. Cats are mammals. Birds can fly.");
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn text_tool_execute() {
        let engine = test_engine();
        let input = ToolInput::new()
            .with_param("text", "Dogs are mammals. Paris is located in France.");

        let tool = TextIngestTool;
        let result = tool.execute(&engine, input).unwrap();
        assert!(result.success);
        assert!(result.result.contains("extracted 2 triple(s)"));
    }

    #[test]
    fn text_tool_file_input() {
        let engine = test_engine();
        let dir = tempfile::TempDir::new().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "The eagle is a bird. Gold is a metal.").unwrap();

        let input = ToolInput::new()
            .with_param("text", &format!("file:{}", file_path.display()));

        let tool = TextIngestTool;
        let result = tool.execute(&engine, input).unwrap();
        assert!(result.success);
        assert!(result.result.contains("extracted 2 triple(s)"));
    }
}
