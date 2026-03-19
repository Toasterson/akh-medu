//! Prompt templates for LLM-based triple extraction.

/// Build the system prompt for triple extraction.
pub fn system_prompt() -> &'static str {
    "You are a knowledge extraction system. Extract factual triples from text.\n\
     Output a JSON array of objects with keys: \"s\" (subject), \"p\" (predicate), \"o\" (object), \"c\" (confidence 0.0-1.0).\n\
     Rules:\n\
     - Each triple must be a factual statement, not opinion\n\
     - Subject and object are noun phrases (capitalize proper nouns)\n\
     - Predicate is a relation (is-a, has-a, part-of, located-in, causes, contains, supports, designed, built-by, etc.)\n\
     - Confidence: 1.0 for explicit statements, 0.7 for implied, 0.5 for uncertain\n\
     - Output ONLY the JSON array, no explanation\n\
     - Maximum {max} triples per text"
}

/// Build the few-shot example for the prompt.
pub fn few_shot_example() -> &'static str {
    "Extract triples from: \"The Great Pyramid of Giza was built for Pharaoh Khufu. \
     It is the largest of the three pyramids on the Giza plateau. The pyramid was \
     constructed using approximately 2.3 million limestone blocks.\"\n\n\
     [{\"s\":\"Great Pyramid of Giza\",\"p\":\"built-for\",\"o\":\"Pharaoh Khufu\",\"c\":1.0},\
     {\"s\":\"Great Pyramid of Giza\",\"p\":\"is-a\",\"o\":\"pyramid\",\"c\":1.0},\
     {\"s\":\"Great Pyramid of Giza\",\"p\":\"located-in\",\"o\":\"Giza plateau\",\"c\":1.0},\
     {\"s\":\"Great Pyramid of Giza\",\"p\":\"constructed-with\",\"o\":\"limestone blocks\",\"c\":0.9}]"
}

/// Build the user message for a specific text chunk.
pub fn user_message(text: &str) -> String {
    format!("Extract triples from: \"{text}\"")
}

/// Build a complete prompt for the local LLM (ChatML format).
pub fn chatml_prompt(text: &str, max_triples: usize) -> String {
    let sys = system_prompt().replace("{max}", &max_triples.to_string());
    let few_shot = few_shot_example();
    let user = user_message(text);

    format!(
        "<|im_start|>system\n{sys}<|im_end|>\n\
         <|im_start|>user\n{few_shot}<|im_end|>\n\
         <|im_start|>assistant\n[{{\"s\":\"Great Pyramid of Giza\",\"p\":\"built-for\",\"o\":\"Pharaoh Khufu\",\"c\":1.0}},{{\"s\":\"Great Pyramid of Giza\",\"p\":\"is-a\",\"o\":\"pyramid\",\"c\":1.0}},{{\"s\":\"Great Pyramid of Giza\",\"p\":\"located-in\",\"o\":\"Giza plateau\",\"c\":1.0}},{{\"s\":\"Great Pyramid of Giza\",\"p\":\"constructed-with\",\"o\":\"limestone blocks\",\"c\":0.9}}]<|im_end|>\n\
         <|im_start|>user\n{user}<|im_end|>\n\
         <|im_start|>assistant\n["
    )
}

/// Build OpenAI-compatible messages array for the external API.
pub fn openai_messages(
    text: &str,
    max_triples: usize,
) -> Vec<serde_json::Value> {
    let sys = system_prompt().replace("{max}", &max_triples.to_string());
    let few_shot = few_shot_example();

    vec![
        serde_json::json!({
            "role": "system",
            "content": sys
        }),
        serde_json::json!({
            "role": "user",
            "content": few_shot
        }),
        serde_json::json!({
            "role": "assistant",
            "content": "[{\"s\":\"Great Pyramid of Giza\",\"p\":\"built-for\",\"o\":\"Pharaoh Khufu\",\"c\":1.0},{\"s\":\"Great Pyramid of Giza\",\"p\":\"is-a\",\"o\":\"pyramid\",\"c\":1.0},{\"s\":\"Great Pyramid of Giza\",\"p\":\"located-in\",\"o\":\"Giza plateau\",\"c\":1.0},{\"s\":\"Great Pyramid of Giza\",\"p\":\"constructed-with\",\"o\":\"limestone blocks\",\"c\":0.9}]"
        }),
        serde_json::json!({
            "role": "user",
            "content": user_message(text)
        }),
    ]
}

/// Parse LLM output into RawTriples.
///
/// Handles both clean JSON arrays and messy LLM output (extra text before/after JSON).
pub fn parse_triples(raw: &str) -> Vec<super::RawTriple> {
    // Try to find JSON array in the output.
    let json_str = extract_json_array(raw);

    let parsed: Vec<serde_json::Value> = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    parsed
        .into_iter()
        .filter_map(|v| {
            let s = v.get("s")?.as_str()?.trim().to_string();
            let p = v.get("p")?.as_str()?.trim().to_string();
            let o = v.get("o")?.as_str()?.trim().to_string();
            let c = v
                .get("c")
                .and_then(|c| c.as_f64())
                .unwrap_or(0.7) as f32;

            if s.is_empty() || p.is_empty() || o.is_empty() {
                return None;
            }

            Some(super::RawTriple {
                subject: s,
                predicate: p,
                object: o,
                confidence: c.clamp(0.0, 1.0),
            })
        })
        .collect()
}

/// Extract a JSON array from potentially messy LLM output.
fn extract_json_array(raw: &str) -> String {
    let trimmed = raw.trim();

    // If it starts with [, try to find matching ]
    if trimmed.starts_with('[') {
        if let Some(end) = find_matching_bracket(trimmed) {
            return trimmed[..=end].to_string();
        }
        // Unmatched bracket — try adding ]
        return format!("{trimmed}]");
    }

    // Look for [ anywhere in the output
    if let Some(start) = trimmed.find('[') {
        let rest = &trimmed[start..];
        if let Some(end) = find_matching_bracket(rest) {
            return rest[..=end].to_string();
        }
        return format!("{rest}]");
    }

    // No array found — return empty array
    "[]".to_string()
}

/// Find the index of the closing ] that matches the opening [.
fn find_matching_bracket(s: &str) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, ch) in s.char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if ch == '\\' && in_string {
            escape_next = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        if ch == '[' {
            depth += 1;
        } else if ch == ']' {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_clean_json() {
        let raw = r#"[{"s":"Sun","p":"is-a","o":"Star","c":0.95}]"#;
        let triples = parse_triples(raw);
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].subject, "Sun");
        assert_eq!(triples[0].predicate, "is-a");
        assert_eq!(triples[0].object, "Star");
        assert!((triples[0].confidence - 0.95).abs() < 0.01);
    }

    #[test]
    fn parse_messy_output_with_preamble() {
        let raw = "Here are the triples:\n[{\"s\":\"A\",\"p\":\"is-a\",\"o\":\"B\",\"c\":1.0}]";
        let triples = parse_triples(raw);
        assert_eq!(triples.len(), 1);
    }

    #[test]
    fn parse_truncated_json() {
        let raw = r#"[{"s":"A","p":"rel","o":"B","c":0.8},{"s":"C","p":"rel","o":"D","c":0.9}"#;
        let triples = parse_triples(raw);
        // Should recover at least the valid entries
        assert!(triples.len() >= 1);
    }

    #[test]
    fn parse_empty_fields_filtered() {
        let raw = r#"[{"s":"","p":"is-a","o":"Star","c":0.9}]"#;
        let triples = parse_triples(raw);
        assert!(triples.is_empty());
    }

    #[test]
    fn chatml_prompt_contains_text() {
        let prompt = chatml_prompt("test text here", 10);
        assert!(prompt.contains("test text here"));
        assert!(prompt.contains("<|im_start|>system"));
    }

    #[test]
    fn openai_messages_structure() {
        let msgs = openai_messages("test", 10);
        assert_eq!(msgs.len(), 4); // system + few-shot user + few-shot assistant + user
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[3]["role"], "user");
    }
}
