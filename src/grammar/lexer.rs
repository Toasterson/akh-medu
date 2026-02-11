//! Lexer: tokenization, symbol resolution, and relational pattern matching.
//!
//! The lexer performs three passes over the input:
//! 1. **Tokenize**: whitespace split with punctuation handling and span tracking
//! 2. **Compound resolution**: greedy longest-match against the SymbolRegistry
//! 3. **Individual resolution**: exact registry lookup → VSA fuzzy match → Unresolved
//!
//! The `Lexicon` maps function words to grammatical roles and relational
//! patterns to canonical predicate labels.

use crate::registry::SymbolRegistry;
use crate::symbol::SymbolId;
use crate::vsa::item_memory::ItemMemory;
use crate::vsa::ops::VsaOps;

/// Byte-level source span for error reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

/// How a token was resolved against the symbol system.
#[derive(Debug, Clone)]
pub enum Resolution {
    /// Exact match in SymbolRegistry (case-insensitive).
    Exact(SymbolId),
    /// Fuzzy match via VSA ItemMemory similarity search.
    Fuzzy { symbol_id: SymbolId, similarity: f32 },
    /// Multi-word compound resolved as a single symbol.
    Compound { symbol_id: SymbolId, word_count: usize },
    /// No match found — treated as a new/unknown entity.
    Unresolved,
}

/// A single lexical token.
#[derive(Debug, Clone)]
pub struct Token {
    /// The original surface text.
    pub surface: String,
    /// Normalized form (lowercase, trimmed).
    pub normalized: String,
    /// Source position in input.
    pub span: Span,
    /// How this token was resolved.
    pub resolution: Resolution,
    /// Whether this token is a function word (article, conjunction, etc.)
    /// that carries no semantic content.
    pub semantically_void: bool,
}

/// A multi-word relational pattern like "is a" → "is-a".
#[derive(Debug, Clone)]
pub struct RelationalPattern {
    /// Ordered surface words (lowercase).
    pub words: Vec<String>,
    /// The canonical predicate label.
    pub predicate_label: String,
    /// Default confidence assigned to triples using this pattern.
    pub default_confidence: f32,
}

/// The lexicon: maps surface forms to grammatical roles.
pub struct Lexicon {
    /// Semantically void words (articles, determiners).
    void_words: Vec<String>,
    /// Multi-word relational patterns (sorted longest first).
    relational_patterns: Vec<RelationalPattern>,
    /// Question words that trigger query parsing.
    question_words: Vec<String>,
    /// Goal-setting verbs.
    goal_verbs: Vec<String>,
    /// Command patterns.
    commands: Vec<(String, CommandKind)>,
}

/// Non-declarative commands recognized by the lexer.
#[derive(Debug, Clone)]
pub enum CommandKind {
    Help,
    ShowStatus,
    RunAgent { cycles: Option<usize> },
    RenderHiero { entity: Option<String> },
    SetGoal { description: String },
}

impl Lexicon {
    /// Build the default English lexicon with all patterns from the existing
    /// `nlp.rs` and `text_ingest.rs` modules.
    pub fn default_english() -> Self {
        let void_words = vec![
            "a".into(), "an".into(), "the".into(),
        ];

        // Relational patterns sorted longest-first for greedy matching
        let relational_patterns = vec![
            // 4-word patterns
            rel("is similar to", "similar-to", 0.85),
            rel("is located in", "located-in", 0.90),
            rel("is composed of", "composed-of", 0.85),
            rel("is part of", "part-of", 0.90),
            rel("is made of", "composed-of", 0.85),
            // 3-word patterns
            rel("depends on", "depends-on", 0.85),
            rel("belongs to", "part-of", 0.85),
            // 2-word patterns
            rel("is a", "is-a", 0.90),
            rel("is an", "is-a", 0.90),
            rel("are a", "is-a", 0.85),
            rel("are an", "is-a", 0.85),
            rel("has a", "has-a", 0.85),
            rel("has an", "has-a", 0.85),
            rel("have a", "has-a", 0.85),
            // 1-word patterns (must be after multi-word)
            rel("are", "is-a", 0.85),
            rel("has", "has-a", 0.85),
            rel("have", "has-a", 0.85),
            rel("contains", "contains", 0.85),
            rel("causes", "causes", 0.85),
            rel("implements", "implements", 0.85),
            rel("defines", "defines", 0.85),
        ];

        let question_words = vec![
            "what".into(), "who".into(), "where".into(), "when".into(),
            "how".into(), "why".into(), "which".into(),
            "is".into(), "does".into(), "do".into(), "can".into(),
        ];

        let goal_verbs = vec![
            "find".into(), "learn".into(), "discover".into(), "explore".into(),
            "search".into(), "analyze".into(), "investigate".into(),
            "determine".into(), "classify".into(), "identify".into(),
        ];

        let commands = vec![
            ("help".into(), CommandKind::Help),
            ("?".into(), CommandKind::Help),
            ("status".into(), CommandKind::ShowStatus),
            ("goals".into(), CommandKind::ShowStatus),
            ("show status".into(), CommandKind::ShowStatus),
            ("show goals".into(), CommandKind::ShowStatus),
            ("list goals".into(), CommandKind::ShowStatus),
        ];

        Self {
            void_words,
            relational_patterns,
            question_words,
            goal_verbs,
            commands,
        }
    }

    /// Whether a word is semantically void (article/determiner).
    pub fn is_void(&self, word: &str) -> bool {
        let lower = word.to_lowercase();
        self.void_words.iter().any(|v| *v == lower)
    }

    /// Get the relational patterns (longest first).
    pub fn relational_patterns(&self) -> &[RelationalPattern] {
        &self.relational_patterns
    }

    /// Whether a word is a question word.
    pub fn is_question_word(&self, word: &str) -> bool {
        let lower = word.to_lowercase();
        self.question_words.iter().any(|q| *q == lower)
    }

    /// Whether a word is a goal-setting verb.
    pub fn is_goal_verb(&self, word: &str) -> bool {
        let lower = word.to_lowercase();
        self.goal_verbs.iter().any(|g| *g == lower)
    }

    /// Try to match a command from the input.
    pub fn match_command(&self, input: &str) -> Option<CommandKind> {
        let lower = input.trim().to_lowercase();

        for (pattern, kind) in &self.commands {
            if lower == *pattern || lower.starts_with(&format!("{pattern} ")) {
                return Some(kind.clone());
            }
        }

        // Dynamic commands: "run N", "cycle N", "show X", "render X"
        if lower.starts_with("run") || lower.starts_with("cycle") {
            let cycles = extract_number(&lower);
            return Some(CommandKind::RunAgent { cycles });
        }

        if lower.starts_with("show ") || lower.starts_with("render ") || lower.starts_with("graph ") {
            let rest = if lower.starts_with("show ") {
                input.trim()[5..].trim()
            } else if lower.starts_with("render ") {
                input.trim()[7..].trim()
            } else {
                input.trim()[6..].trim()
            };
            // Check if this is a status command first
            if rest.eq_ignore_ascii_case("status") || rest.eq_ignore_ascii_case("goals") {
                return Some(CommandKind::ShowStatus);
            }
            let entity = if rest.is_empty() {
                None
            } else {
                Some(rest.to_string())
            };
            return Some(CommandKind::RenderHiero { entity });
        }

        None
    }

    /// Try to match a goal-setting pattern.
    pub fn match_goal(&self, input: &str) -> Option<String> {
        let lower = input.trim().to_lowercase();
        for verb in &self.goal_verbs {
            if lower.starts_with(&format!("{verb} ")) {
                return Some(input.trim().to_string());
            }
        }
        None
    }

    /// Get the preferred surface form for a canonical predicate.
    pub fn surface_form(&self, canonical: &str) -> Option<String> {
        self.relational_patterns
            .iter()
            .find(|p| p.predicate_label == canonical)
            .map(|p| p.words.join(" "))
    }
}

fn rel(pattern: &str, label: &str, confidence: f32) -> RelationalPattern {
    RelationalPattern {
        words: pattern.split_whitespace().map(String::from).collect(),
        predicate_label: label.to_string(),
        default_confidence: confidence,
    }
}

fn extract_number(input: &str) -> Option<usize> {
    input
        .split_whitespace()
        .find_map(|word| word.parse::<usize>().ok())
}

/// Tokenize input text into tokens with resolution against the symbol system.
///
/// If `registry`, `ops`, and `item_memory` are provided, performs symbol
/// resolution. Otherwise, all tokens are `Unresolved`.
pub fn tokenize(
    input: &str,
    registry: Option<&SymbolRegistry>,
    ops: Option<&VsaOps>,
    item_memory: Option<&ItemMemory>,
    lexicon: &Lexicon,
) -> Vec<Token> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    // Pass 1: basic whitespace tokenization with span tracking
    let mut raw_tokens = Vec::new();
    let mut pos = 0;

    for word in trimmed.split_whitespace() {
        // Find the actual position in the original string
        let start = trimmed[pos..].find(word).map(|i| i + pos).unwrap_or(pos);
        let end = start + word.len();

        // Strip trailing punctuation (but preserve it for later)
        let clean = word.trim_end_matches(|c: char| c == '.' || c == ',' || c == '!' || c == ';');

        raw_tokens.push(Token {
            surface: clean.to_string(),
            normalized: clean.to_lowercase(),
            span: Span { start, end },
            resolution: Resolution::Unresolved,
            semantically_void: lexicon.is_void(clean),
        });

        pos = end;
    }

    // Pass 2: compound resolution (if registry available)
    if let Some(reg) = registry {
        resolve_compounds(&mut raw_tokens, reg);
    }

    // Pass 3: individual resolution
    if let Some(reg) = registry {
        for token in &mut raw_tokens {
            if matches!(token.resolution, Resolution::Unresolved) && !token.semantically_void {
                // Try exact match first
                if let Some(id) = reg.lookup(&token.normalized) {
                    token.resolution = Resolution::Exact(id);
                } else if let (Some(vsa_ops), Some(im)) = (ops, item_memory) {
                    // Try VSA fuzzy match
                    resolve_fuzzy(token, vsa_ops, im);
                }
            }
        }
    }

    raw_tokens
}

/// Greedy longest-match compound resolution.
///
/// Slides a window from length 4 down to 2 over the token stream,
/// checking if concatenated words form a registered symbol.
fn resolve_compounds(tokens: &mut Vec<Token>, registry: &SymbolRegistry) {
    let max_window = 4.min(tokens.len());

    for window_size in (2..=max_window).rev() {
        let mut i = 0;
        while i + window_size <= tokens.len() {
            let compound: String = tokens[i..i + window_size]
                .iter()
                .map(|t| t.normalized.as_str())
                .collect::<Vec<_>>()
                .join(" ");

            if let Some(id) = registry.lookup(&compound) {
                // Merge tokens: replace first with compound, mark rest for removal
                let start = tokens[i].span.start;
                let end = tokens[i + window_size - 1].span.end;

                tokens[i] = Token {
                    surface: tokens[i..i + window_size]
                        .iter()
                        .map(|t| t.surface.as_str())
                        .collect::<Vec<_>>()
                        .join(" "),
                    normalized: compound,
                    span: Span { start, end },
                    resolution: Resolution::Compound {
                        symbol_id: id,
                        word_count: window_size,
                    },
                    semantically_void: false,
                };

                // Remove the merged tokens
                for _ in 1..window_size {
                    tokens.remove(i + 1);
                }
            }
            i += 1;
        }
    }
}

/// Minimum similarity threshold for fuzzy resolution to accept a match.
const DEFAULT_FUZZY_THRESHOLD: f32 = 0.6;

/// Try VSA-based fuzzy resolution for an unresolved token.
///
/// Encodes the token's normalized text into a hypervector via
/// [`encode_token`](crate::vsa::encode::encode_token), searches the
/// item memory for the `k=3` most similar symbols, and accepts the
/// best match if it exceeds [`DEFAULT_FUZZY_THRESHOLD`].
fn resolve_fuzzy(token: &mut Token, ops: &VsaOps, item_memory: &ItemMemory) {
    // Skip very short tokens — single characters are too ambiguous
    if token.normalized.len() < 2 {
        return;
    }

    let query_vec = crate::vsa::encode::encode_token(ops, &token.normalized);

    let results = match item_memory.search(&query_vec, 3) {
        Ok(r) => r,
        Err(_) => return, // Silently fall through on search errors
    };

    if let Some(best) = results.first() {
        if best.similarity > DEFAULT_FUZZY_THRESHOLD {
            token.resolution = Resolution::Fuzzy {
                symbol_id: best.symbol_id,
                similarity: best.similarity,
            };
        }
    }
}

/// Find a relational pattern in a token stream and return the split points.
///
/// Returns `Some((subject_end_idx, object_start_idx))` if found.
pub fn find_relational_pattern(
    tokens: &[Token],
    pattern: &RelationalPattern,
) -> Option<(usize, usize)> {
    let pattern_len = pattern.words.len();
    if tokens.len() < pattern_len + 2 {
        // Need at least 1 subject token + pattern + 1 object token
        return None;
    }

    // Scan for the pattern anywhere in the token stream (not at the very start or end)
    for i in 1..=tokens.len().saturating_sub(pattern_len + 1) {
        let matches = tokens[i..i + pattern_len]
            .iter()
            .zip(&pattern.words)
            .all(|(token, word)| token.normalized == *word);

        if matches {
            return Some((i, i + pattern_len));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd;
    use crate::vsa::{Dimension, Encoding};

    #[test]
    fn tokenize_simple() {
        let lexicon = Lexicon::default_english();
        let tokens = tokenize("Dogs are mammals", None, None, None, &lexicon);
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].normalized, "dogs");
        assert_eq!(tokens[1].normalized, "are");
        assert_eq!(tokens[2].normalized, "mammals");
    }

    #[test]
    fn articles_are_void() {
        let lexicon = Lexicon::default_english();
        let tokens = tokenize("A dog is an animal", None, None, None, &lexicon);
        assert!(tokens[0].semantically_void); // "a"
        assert!(!tokens[1].semantically_void); // "dog"
        assert!(tokens[3].semantically_void); // "an"
    }

    #[test]
    fn trailing_punctuation_stripped() {
        let lexicon = Lexicon::default_english();
        let tokens = tokenize("Dogs are mammals.", None, None, None, &lexicon);
        assert_eq!(tokens[2].normalized, "mammals");
    }

    #[test]
    fn find_pattern_is_a() {
        let lexicon = Lexicon::default_english();
        let tokens = tokenize("Dog is a Mammal", None, None, None, &lexicon);
        let pattern = &lexicon.relational_patterns()
            .iter()
            .find(|p| p.predicate_label == "is-a" && p.words == ["is", "a"])
            .unwrap();
        let result = find_relational_pattern(&tokens, pattern);
        assert!(result.is_some());
        let (subj_end, obj_start) = result.unwrap();
        assert_eq!(subj_end, 1); // "Dog" ends at index 1
        assert_eq!(obj_start, 3); // "Mammal" starts at index 3
    }

    #[test]
    fn find_pattern_contains() {
        let lexicon = Lexicon::default_english();
        let tokens = tokenize("Module contains function", None, None, None, &lexicon);
        let pattern = lexicon.relational_patterns()
            .iter()
            .find(|p| p.predicate_label == "contains")
            .unwrap();
        let result = find_relational_pattern(&tokens, pattern);
        assert!(result.is_some());
    }

    #[test]
    fn command_matching() {
        let lexicon = Lexicon::default_english();
        assert!(matches!(lexicon.match_command("help"), Some(CommandKind::Help)));
        assert!(matches!(lexicon.match_command("status"), Some(CommandKind::ShowStatus)));
        assert!(matches!(lexicon.match_command("run 5"), Some(CommandKind::RunAgent { cycles: Some(5) })));
        assert!(matches!(lexicon.match_command("show Dog"), Some(CommandKind::RenderHiero { .. })));
    }

    #[test]
    fn goal_matching() {
        let lexicon = Lexicon::default_english();
        assert!(lexicon.match_goal("find similar animals").is_some());
        assert!(lexicon.match_goal("explore the knowledge graph").is_some());
        assert!(lexicon.match_goal("Dogs are mammals").is_none());
    }

    #[test]
    fn surface_form_lookup() {
        let lexicon = Lexicon::default_english();
        assert_eq!(lexicon.surface_form("is-a"), Some("is a".to_string()));
        assert_eq!(lexicon.surface_form("contains"), Some("contains".to_string()));
        assert_eq!(lexicon.surface_form("nonexistent"), None);
    }

    // ── resolve_fuzzy tests ─────────────────────────────────────────────

    fn test_ops() -> VsaOps {
        VsaOps::new(simd::best_kernel(), Dimension::TEST, Encoding::Bipolar)
    }

    #[test]
    fn fuzzy_resolves_with_populated_item_memory() {
        use crate::vsa::item_memory::ItemMemory;
        let ops = test_ops();
        let im = ItemMemory::new(Dimension::TEST, Encoding::Bipolar, 100);

        // Insert a known symbol
        let sym = crate::symbol::SymbolId::new(42).unwrap();
        let vec = crate::vsa::encode::encode_token(&ops, "hello");
        im.insert(sym, vec);

        // Search for the same token — should find it with high similarity
        let mut token = Token {
            surface: "hello".into(),
            normalized: "hello".into(),
            span: Span { start: 0, end: 5 },
            resolution: Resolution::Unresolved,
            semantically_void: false,
        };
        resolve_fuzzy(&mut token, &ops, &im);

        match &token.resolution {
            Resolution::Fuzzy { symbol_id, similarity } => {
                assert_eq!(*symbol_id, sym);
                assert!(*similarity > 0.9, "similarity={similarity}");
            }
            other => panic!("expected Fuzzy, got {other:?}"),
        }
    }

    #[test]
    fn fuzzy_short_tokens_skipped() {
        use crate::vsa::item_memory::ItemMemory;
        let ops = test_ops();
        let im = ItemMemory::new(Dimension::TEST, Encoding::Bipolar, 100);

        let mut token = Token {
            surface: "a".into(),
            normalized: "a".into(),
            span: Span { start: 0, end: 1 },
            resolution: Resolution::Unresolved,
            semantically_void: false,
        };
        resolve_fuzzy(&mut token, &ops, &im);
        assert!(matches!(token.resolution, Resolution::Unresolved));
    }

    #[test]
    fn fuzzy_empty_memory_no_crash() {
        use crate::vsa::item_memory::ItemMemory;
        let ops = test_ops();
        let im = ItemMemory::new(Dimension::TEST, Encoding::Bipolar, 100);

        let mut token = Token {
            surface: "hello".into(),
            normalized: "hello".into(),
            span: Span { start: 0, end: 5 },
            resolution: Resolution::Unresolved,
            semantically_void: false,
        };
        resolve_fuzzy(&mut token, &ops, &im);
        // Should stay unresolved — no crash
        assert!(matches!(token.resolution, Resolution::Unresolved));
    }

    #[test]
    fn fuzzy_resolves_to_correct_symbol_not_random() {
        use crate::vsa::item_memory::ItemMemory;
        let ops = test_ops();
        let im = ItemMemory::new(Dimension::TEST, Encoding::Bipolar, 100);

        // Insert two symbols via encode_token with known labels
        let sym_hello = crate::symbol::SymbolId::new(42).unwrap();
        let sym_world = crate::symbol::SymbolId::new(43).unwrap();
        let vec_hello = crate::vsa::encode::encode_token(&ops, "hello");
        let vec_world = crate::vsa::encode::encode_token(&ops, "world");
        im.insert(sym_hello, vec_hello);
        im.insert(sym_world, vec_world);

        // Searching for "hello" should find sym_hello, not sym_world
        let mut token = Token {
            surface: "hello".into(),
            normalized: "hello".into(),
            span: Span { start: 0, end: 5 },
            resolution: Resolution::Unresolved,
            semantically_void: false,
        };
        resolve_fuzzy(&mut token, &ops, &im);

        match &token.resolution {
            Resolution::Fuzzy { symbol_id, .. } => {
                assert_eq!(*symbol_id, sym_hello, "should resolve to the correct symbol");
            }
            other => panic!("expected Fuzzy, got {other:?}"),
        }
    }
}
