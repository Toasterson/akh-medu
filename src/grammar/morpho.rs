//! English morphology helpers for natural-sounding prose generation.
//!
//! These are lightweight, rule-based transformations — not a full NLP
//! morphology engine. They cover the common cases needed for linearizing
//! knowledge graph triples into readable sentences.

/// Choose "a" or "an" based on the following word.
///
/// Uses a simplified heuristic: "an" before vowel sounds.
pub fn article(word: &str) -> &'static str {
    let first = word.trim().to_lowercase();
    let first_char = first.chars().next().unwrap_or('x');

    // Special cases: "honest", "hour", "heir" start with silent 'h'
    if first.starts_with("honest")
        || first.starts_with("hour")
        || first.starts_with("heir")
        || first.starts_with("honor")
    {
        return "an";
    }

    // Special cases: "uni-", "use", "user" start with a consonant 'y' sound
    if first.starts_with("uni")
        || first.starts_with("use")
        || first.starts_with("user")
        || first.starts_with("util")
    {
        return "a";
    }

    match first_char {
        'a' | 'e' | 'i' | 'o' | 'u' => "an",
        _ => "a",
    }
}

/// Humanize a predicate label into readable English.
///
/// Converts kebab-case predicate names to natural phrases:
/// - "is-a" → "is a"
/// - "part-of" → "is part of"
/// - "contains-mod" → "contains"
/// - "defines-fn" → "defines"
pub fn humanize_predicate(predicate: &str) -> String {
    match predicate {
        "is-a" | "is_a" => "is a".to_string(),
        "has-a" | "has_a" => "has".to_string(),
        "part-of" | "part_of" => "is part of".to_string(),
        "contains" => "contains".to_string(),
        "located-in" | "located_in" => "is located in".to_string(),
        "causes" => "causes".to_string(),
        "similar-to" | "similar_to" => "is similar to".to_string(),
        "composed-of" | "composed_of" => "is composed of".to_string(),
        "depends-on" | "depends_on" => "depends on".to_string(),
        "implements" => "implements".to_string(),
        "defines-fn" | "defines_fn" => "defines function".to_string(),
        "defines-struct" | "defines_struct" => "defines struct".to_string(),
        "defines-enum" | "defines_enum" => "defines enum".to_string(),
        "defines-type" | "defines_type" => "defines type".to_string(),
        "defines-mod" | "defines_mod" => "defines module".to_string(),
        "contains-mod" | "contains_mod" => "contains module".to_string(),
        "defined-in" | "defined_in" => "is defined in".to_string(),
        "has-method" | "has_method" => "has method".to_string(),
        "has-variant" | "has_variant" => "has variant".to_string(),
        // Code predicates with namespace prefix
        p if p.starts_with("code:") => humanize_predicate(&p[5..]),
        // Generic kebab-to-space fallback
        other => other.replace('-', " ").replace('_', " "),
    }
}

/// Capitalize the first letter of a string.
pub fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => {
            let upper: String = c.to_uppercase().collect();
            upper + chars.as_str()
        }
    }
}

/// Simple plural heuristic for English nouns.
pub fn pluralize(word: &str) -> String {
    if word.is_empty() {
        return String::new();
    }

    let lower = word.to_lowercase();

    // Irregular plurals
    match lower.as_str() {
        "child" => return format_case(word, "children"),
        "person" => return format_case(word, "people"),
        "mouse" => return format_case(word, "mice"),
        "datum" => return format_case(word, "data"),
        "index" => return format_case(word, "indices"),
        "vertex" => return format_case(word, "vertices"),
        "matrix" => return format_case(word, "matrices"),
        _ => {}
    }

    // Already plural (simple heuristic)
    if lower.ends_with('s') && !lower.ends_with("ss") {
        return word.to_string();
    }

    // -y → -ies (consonant + y)
    if lower.ends_with('y') {
        let before_y = lower.chars().rev().nth(1).unwrap_or('a');
        if !matches!(before_y, 'a' | 'e' | 'i' | 'o' | 'u') {
            return format!("{}ies", &word[..word.len() - 1]);
        }
    }

    // -s, -x, -z, -ch, -sh → -es
    if lower.ends_with('s')
        || lower.ends_with('x')
        || lower.ends_with('z')
        || lower.ends_with("ch")
        || lower.ends_with("sh")
    {
        return format!("{word}es");
    }

    format!("{word}s")
}

/// Preserve the case style of the original word in the replacement.
fn format_case(original: &str, replacement: &str) -> String {
    if original.chars().next().map_or(false, |c| c.is_uppercase()) {
        capitalize(replacement)
    } else {
        replacement.to_string()
    }
}

/// Wrap a label in backticks for code-like presentation.
pub fn code_quote(label: &str) -> String {
    format!("`{label}`")
}

/// Join items in a list with commas and a final conjunction.
///
/// - 0 items → ""
/// - 1 item → "A"
/// - 2 items → "A and B"
/// - 3+ items → "A, B, and C" (Oxford comma)
pub fn join_list(items: &[String], conjunction: &str) -> String {
    match items.len() {
        0 => String::new(),
        1 => items[0].clone(),
        2 => format!("{} {conjunction} {}", items[0], items[1]),
        _ => {
            let (last, rest) = items.split_last().unwrap();
            format!("{}, {conjunction} {last}", rest.join(", "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn article_selection() {
        assert_eq!(article("apple"), "an");
        assert_eq!(article("banana"), "a");
        assert_eq!(article("elephant"), "an");
        assert_eq!(article("user"), "a");
        assert_eq!(article("hour"), "an");
        assert_eq!(article("utility"), "a");
    }

    #[test]
    fn humanize_common_predicates() {
        assert_eq!(humanize_predicate("is-a"), "is a");
        assert_eq!(humanize_predicate("part-of"), "is part of");
        assert_eq!(humanize_predicate("contains"), "contains");
        assert_eq!(humanize_predicate("code:defines-fn"), "defines function");
        assert_eq!(humanize_predicate("custom-relation"), "custom relation");
    }

    #[test]
    fn capitalize_works() {
        assert_eq!(capitalize("hello"), "Hello");
        assert_eq!(capitalize(""), "");
        assert_eq!(capitalize("Hello"), "Hello");
    }

    #[test]
    fn pluralize_regular() {
        assert_eq!(pluralize("dog"), "dogs");
        assert_eq!(pluralize("entity"), "entities");
        assert_eq!(pluralize("box"), "boxes");
        assert_eq!(pluralize("index"), "indices");
    }

    #[test]
    fn join_list_variations() {
        assert_eq!(join_list(&[], "and"), "");
        assert_eq!(join_list(&["A".into()], "and"), "A");
        assert_eq!(join_list(&["A".into(), "B".into()], "and"), "A and B");
        assert_eq!(
            join_list(&["A".into(), "B".into(), "C".into()], "and"),
            "A, B, and C"
        );
    }
}
