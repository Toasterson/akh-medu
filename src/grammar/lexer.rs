//! Lexer: tokenization, symbol resolution, and relational pattern matching.
//!
//! The lexer performs three passes over the input:
//! 1. **Tokenize**: whitespace split with punctuation handling and span tracking
//! 2. **Compound resolution**: greedy longest-match against the SymbolRegistry
//! 3. **Individual resolution**: exact registry lookup → VSA fuzzy match → Unresolved
//!
//! The `Lexicon` maps function words to grammatical roles and relational
//! patterns to canonical predicate labels.

use serde::{Deserialize, Serialize};

use crate::registry::SymbolRegistry;
use crate::symbol::SymbolId;
use crate::vsa::item_memory::ItemMemory;
use crate::vsa::ops::VsaOps;

/// Supported languages for the grammar system.
///
/// `Auto` means detect from text (Phase 10c). Until detection is wired,
/// `Auto` falls back to English.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum Language {
    English,
    Russian,
    Arabic,
    French,
    Spanish,
    #[default]
    Auto,
}

impl Language {
    /// BCP 47 language tag.
    pub fn bcp47(&self) -> &'static str {
        match self {
            Language::English => "en",
            Language::Russian => "ru",
            Language::Arabic => "ar",
            Language::French => "fr",
            Language::Spanish => "es",
            Language::Auto => "auto",
        }
    }

    /// Parse from a BCP 47 tag or common name.
    pub fn from_code(code: &str) -> Option<Self> {
        match code.to_lowercase().as_str() {
            "en" | "english" => Some(Language::English),
            "ru" | "russian" => Some(Language::Russian),
            "ar" | "arabic" => Some(Language::Arabic),
            "fr" | "french" => Some(Language::French),
            "es" | "spanish" => Some(Language::Spanish),
            "auto" => Some(Language::Auto),
            _ => None,
        }
    }
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Language::English => write!(f, "English"),
            Language::Russian => write!(f, "Russian"),
            Language::Arabic => write!(f, "Arabic"),
            Language::French => write!(f, "French"),
            Language::Spanish => write!(f, "Spanish"),
            Language::Auto => write!(f, "Auto"),
        }
    }
}

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
    Fuzzy {
        symbol_id: SymbolId,
        similarity: f32,
    },
    /// Multi-word compound resolved as a single symbol.
    Compound {
        symbol_id: SymbolId,
        word_count: usize,
    },
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

/// Structural decomposition of a question's grammatical frame.
#[derive(Debug, Clone)]
pub struct QuestionFrame {
    /// Opening question word (e.g., "what", "qui").
    pub question_word: Option<String>,
    /// Auxiliary/modal verb after question word (e.g., "can", "est").
    pub auxiliary: Option<String>,
    /// Content subject tokens (between aux and trailing aux, articles stripped).
    pub subject_tokens: Vec<String>,
    /// Whether a trailing auxiliary was stripped.
    pub trailing_stripped: bool,
    /// Whether the auxiliary signals capability ("can", "peut", "может").
    pub signals_capability: bool,
}

/// The lexicon: maps surface forms to grammatical roles.
#[derive(Clone)]
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
    /// Auxiliary/modal verbs that follow question words (e.g., "is", "can", "does").
    auxiliary_verbs: Vec<String>,
    /// Trailing auxiliaries void at end of questions (e.g., "do" in "What can you do?").
    trailing_auxiliaries: Vec<String>,
    /// Modal verbs that signal capability/ability (subset of auxiliary_verbs).
    capability_modals: Vec<String>,
}

/// Non-declarative commands recognized by the lexer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommandKind {
    Help,
    ShowStatus,
    RunAgent { cycles: Option<usize> },
    RenderHiero { entity: Option<String> },
    SetGoal { description: String },
}

impl Lexicon {
    /// Get the lexicon for the given language.
    ///
    /// Returns the language-specific lexicon with relational patterns
    /// that map to the same canonical predicates across all languages.
    pub fn for_language(lang: Language) -> Self {
        match lang {
            Language::English | Language::Auto => Self::default_english(),
            Language::Russian => Self::default_russian(),
            Language::Arabic => Self::default_arabic(),
            Language::French => Self::default_french(),
            Language::Spanish => Self::default_spanish(),
        }
    }

    /// Build the default English lexicon with all patterns from the existing
    /// `nlp.rs` and `text_ingest.rs` modules.
    pub fn default_english() -> Self {
        let void_words = vec!["a".into(), "an".into(), "the".into()];

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
            "what".into(),
            "who".into(),
            "where".into(),
            "when".into(),
            "how".into(),
            "why".into(),
            "which".into(),
            "is".into(),
            "does".into(),
            "do".into(),
            "can".into(),
        ];

        let goal_verbs = vec![
            "find".into(),
            "learn".into(),
            "discover".into(),
            "explore".into(),
            "search".into(),
            "analyze".into(),
            "investigate".into(),
            "determine".into(),
            "classify".into(),
            "identify".into(),
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

        let auxiliary_verbs = vec![
            "is".into(), "are".into(), "was".into(), "were".into(),
            "do".into(), "does".into(), "did".into(),
            "can".into(), "could".into(),
            "will".into(), "would".into(),
            "shall".into(), "should".into(),
            "may".into(), "might".into(), "must".into(),
            "about".into(),
        ];
        let trailing_auxiliaries = vec!["do".into(), "does".into()];
        let capability_modals = vec!["can".into(), "could".into()];

        Self {
            void_words,
            relational_patterns,
            question_words,
            goal_verbs,
            commands,
            auxiliary_verbs,
            trailing_auxiliaries,
            capability_modals,
        }
    }

    /// Build the Russian lexicon.
    ///
    /// Relational patterns map to the same canonical predicates as English.
    /// Commands stay English (CLI is English).
    pub fn default_russian() -> Self {
        let void_words = Vec::new(); // Russian has no articles

        let relational_patterns = vec![
            // Multi-word patterns (longest first)
            rel("является частью", "part-of", 0.90),
            rel("находится в", "located-in", 0.90),
            rel("состоит из", "composed-of", 0.85),
            rel("зависит от", "depends-on", 0.85),
            rel("похож на", "similar-to", 0.85),
            rel("содержит в себе", "contains", 0.85),
            // Single/shorter patterns
            rel("является", "is-a", 0.90),
            rel("имеет", "has-a", 0.85),
            rel("содержит", "contains", 0.85),
            rel("вызывает", "causes", 0.85),
            rel("определяет", "defines", 0.85),
            rel("реализует", "implements", 0.85),
            // "это" used as copula
            rel("это", "is-a", 0.80),
        ];

        let question_words = vec![
            "что".into(),
            "кто".into(),
            "где".into(),
            "когда".into(),
            "как".into(),
            "почему".into(),
            "какой".into(),
            "какая".into(),
            "какие".into(),
        ];

        let goal_verbs = vec![
            "найти".into(),
            "изучить".into(),
            "обнаружить".into(),
            "исследовать".into(),
            "определить".into(),
            "классифицировать".into(),
        ];

        // Commands stay English
        let commands = vec![
            ("help".into(), CommandKind::Help),
            ("?".into(), CommandKind::Help),
            ("status".into(), CommandKind::ShowStatus),
        ];

        let auxiliary_verbs = vec![
            "это".into(), "является".into(), "может".into(), "ли".into(),
        ];
        let trailing_auxiliaries = vec!["делать".into()];
        let capability_modals = vec!["может".into()];

        Self {
            void_words,
            relational_patterns,
            question_words,
            goal_verbs,
            commands,
            auxiliary_verbs,
            trailing_auxiliaries,
            capability_modals,
        }
    }

    /// Build the Arabic lexicon.
    ///
    /// Relational patterns map to the same canonical predicates as English.
    pub fn default_arabic() -> Self {
        let void_words = vec![
            "ال".into(), // definite article prefix (when tokenized separately)
        ];

        let relational_patterns = vec![
            // Multi-word patterns (longest first)
            rel("يحتوي على", "contains", 0.85),
            rel("يقع في", "located-in", 0.90),
            rel("جزء من", "part-of", 0.90),
            rel("يتكون من", "composed-of", 0.85),
            rel("يعتمد على", "depends-on", 0.85),
            // Shorter patterns
            rel("هو", "is-a", 0.90),
            rel("هي", "is-a", 0.90),
            rel("لديه", "has-a", 0.85),
            rel("لديها", "has-a", 0.85),
            rel("يسبب", "causes", 0.85),
            rel("يشبه", "similar-to", 0.85),
        ];

        let question_words = vec![
            "ما".into(),
            "من".into(),
            "أين".into(),
            "متى".into(),
            "كيف".into(),
            "لماذا".into(),
            "هل".into(),
        ];

        let goal_verbs = vec![
            "ابحث".into(),
            "اكتشف".into(),
            "حلل".into(),
            "حدد".into(),
            "صنف".into(),
        ];

        let commands = vec![
            ("help".into(), CommandKind::Help),
            ("?".into(), CommandKind::Help),
            ("status".into(), CommandKind::ShowStatus),
        ];

        let auxiliary_verbs = vec![
            "هل".into(), "هو".into(), "هي".into(), "يمكن".into(),
        ];
        let trailing_auxiliaries = vec!["تفعل".into(), "يفعل".into()];
        let capability_modals = vec!["يمكن".into(), "يستطيع".into()];

        Self {
            void_words,
            relational_patterns,
            question_words,
            goal_verbs,
            commands,
            auxiliary_verbs,
            trailing_auxiliaries,
            capability_modals,
        }
    }

    /// Build the French lexicon.
    ///
    /// Relational patterns map to the same canonical predicates as English.
    pub fn default_french() -> Self {
        let void_words = vec![
            "le".into(),
            "la".into(),
            "les".into(),
            "l'".into(),
            "un".into(),
            "une".into(),
            "des".into(),
            "du".into(),
            "de".into(),
            "d'".into(),
        ];

        let relational_patterns = vec![
            // Multi-word patterns (longest first)
            rel("est similaire à", "similar-to", 0.85),
            rel("est similaire a", "similar-to", 0.85), // without accent
            rel("se trouve dans", "located-in", 0.90),
            rel("est composé de", "composed-of", 0.85),
            rel("est compose de", "composed-of", 0.85), // without accent
            rel("fait partie de", "part-of", 0.90),
            rel("dépend de", "depends-on", 0.85),
            rel("depend de", "depends-on", 0.85), // without accent
            // 2-word patterns
            rel("est un", "is-a", 0.90),
            rel("est une", "is-a", 0.90),
            rel("a un", "has-a", 0.85),
            rel("a une", "has-a", 0.85),
            // Single patterns
            rel("contient", "contains", 0.85),
            rel("cause", "causes", 0.85),
            rel("définit", "defines", 0.85),
            rel("definit", "defines", 0.85), // without accent
        ];

        let question_words = vec![
            "que".into(),
            "qui".into(),
            "où".into(),
            "quand".into(),
            "comment".into(),
            "pourquoi".into(),
            "quel".into(),
            "quelle".into(),
            "quels".into(),
            "quelles".into(),
            "est-ce".into(),
        ];

        let goal_verbs = vec![
            "trouver".into(),
            "découvrir".into(),
            "explorer".into(),
            "analyser".into(),
            "déterminer".into(),
            "identifier".into(),
            "classifier".into(),
        ];

        let commands = vec![
            ("help".into(), CommandKind::Help),
            ("?".into(), CommandKind::Help),
            ("status".into(), CommandKind::ShowStatus),
        ];

        let auxiliary_verbs = vec![
            "est".into(), "sont".into(),
            "peut".into(), "peux".into(), "pouvez".into(),
            "fait".into(),
        ];
        let trailing_auxiliaries = vec!["faire".into()];
        let capability_modals = vec!["peut".into(), "peux".into(), "pouvez".into()];

        Self {
            void_words,
            relational_patterns,
            question_words,
            goal_verbs,
            commands,
            auxiliary_verbs,
            trailing_auxiliaries,
            capability_modals,
        }
    }

    /// Build the Spanish lexicon.
    ///
    /// Relational patterns map to the same canonical predicates as English.
    pub fn default_spanish() -> Self {
        let void_words = vec![
            "el".into(),
            "la".into(),
            "los".into(),
            "las".into(),
            "un".into(),
            "una".into(),
            "unos".into(),
            "unas".into(),
            "del".into(),
            "de".into(),
            "al".into(),
        ];

        let relational_patterns = vec![
            // Multi-word patterns (longest first)
            rel("es similar a", "similar-to", 0.85),
            rel("se encuentra en", "located-in", 0.90),
            rel("está compuesto de", "composed-of", 0.85),
            rel("esta compuesto de", "composed-of", 0.85), // without accent
            rel("es parte de", "part-of", 0.90),
            rel("depende de", "depends-on", 0.85),
            // 2-word patterns
            rel("es un", "is-a", 0.90),
            rel("es una", "is-a", 0.90),
            rel("tiene un", "has-a", 0.85),
            rel("tiene una", "has-a", 0.85),
            // Single patterns
            rel("contiene", "contains", 0.85),
            rel("causa", "causes", 0.85),
            rel("tiene", "has-a", 0.80),
            rel("define", "defines", 0.85),
        ];

        let question_words = vec![
            "qué".into(),
            "que".into(),
            "quién".into(),
            "quien".into(),
            "dónde".into(),
            "donde".into(),
            "cuándo".into(),
            "cuando".into(),
            "cómo".into(),
            "como".into(),
            "por qué".into(),
        ];

        let goal_verbs = vec![
            "encontrar".into(),
            "descubrir".into(),
            "explorar".into(),
            "analizar".into(),
            "determinar".into(),
            "identificar".into(),
            "clasificar".into(),
        ];

        let commands = vec![
            ("help".into(), CommandKind::Help),
            ("?".into(), CommandKind::Help),
            ("status".into(), CommandKind::ShowStatus),
        ];

        let auxiliary_verbs = vec![
            "es".into(), "son".into(),
            "puede".into(), "puedes".into(),
            "hace".into(), "hacen".into(),
        ];
        let trailing_auxiliaries = vec!["hacer".into()];
        let capability_modals = vec!["puede".into(), "puedes".into(), "pueden".into()];

        Self {
            void_words,
            relational_patterns,
            question_words,
            goal_verbs,
            commands,
            auxiliary_verbs,
            trailing_auxiliaries,
            capability_modals,
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

    /// Whether a word is an auxiliary/modal verb.
    pub fn is_auxiliary_verb(&self, word: &str) -> bool {
        let lower = word.to_lowercase();
        self.auxiliary_verbs.iter().any(|a| *a == lower)
    }

    /// Whether a word is a trailing auxiliary (strippable at end of questions).
    pub fn is_trailing_auxiliary(&self, word: &str) -> bool {
        let lower = word.to_lowercase();
        self.trailing_auxiliaries.iter().any(|a| *a == lower)
    }

    /// Whether a word is a capability/ability modal verb.
    pub fn is_capability_modal(&self, word: &str) -> bool {
        let lower = word.to_lowercase();
        self.capability_modals.iter().any(|m| *m == lower)
    }

    /// Parse a question into its grammatical frame.
    ///
    /// Decomposes the question into question word, auxiliary verb,
    /// content subject tokens, and capability signal — all language-aware
    /// via the lexicon's word lists.
    pub fn parse_question_frame(&self, input: &str) -> QuestionFrame {
        let s = input.trim().trim_end_matches('?').trim();
        let words: Vec<&str> = s.split_whitespace().collect();

        if words.is_empty() {
            return QuestionFrame {
                question_word: None,
                auxiliary: None,
                subject_tokens: Vec::new(),
                trailing_stripped: false,
                signals_capability: false,
            };
        }

        let mut pos = 0;
        let mut question_word = None;
        let mut auxiliary = None;

        // Check if first word is a question word.
        if self.is_question_word(words[0]) {
            question_word = Some(words[0].to_lowercase());
            pos = 1;

            // Check if second word is an auxiliary verb.
            if pos < words.len() && self.is_auxiliary_verb(words[pos]) {
                auxiliary = Some(words[pos].to_lowercase());
                pos += 1;
            }
        } else if self.is_auxiliary_verb(words[0]) {
            // First word IS an auxiliary (e.g., "Can you help?").
            auxiliary = Some(words[0].to_lowercase());
            pos = 1;
        }

        // Remaining words are content.
        let mut content: Vec<String> = words[pos..]
            .iter()
            .map(|w| w.to_string())
            .collect();

        // Strip trailing auxiliary if more than one content word remains.
        let mut trailing_stripped = false;
        if content.len() > 1 {
            if let Some(last) = content.last() {
                if self.is_trailing_auxiliary(last) {
                    content.pop();
                    trailing_stripped = true;
                }
            }
        }

        // Strip leading void words (articles) from content.
        while !content.is_empty() && self.is_void(&content[0]) && content.len() > 1 {
            content.remove(0);
        }

        let signals_capability = auxiliary
            .as_deref()
            .is_some_and(|a| self.is_capability_modal(a))
            || question_word
                .as_deref()
                .is_some_and(|q| self.is_capability_modal(q));

        QuestionFrame {
            question_word,
            auxiliary,
            subject_tokens: content,
            trailing_stripped,
            signals_capability,
        }
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

        if lower.starts_with("show ") || lower.starts_with("render ") || lower.starts_with("graph ")
        {
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
/// Check if a character is a punctuation mark that should be stripped from tokens.
///
/// Handles ASCII punctuation as well as Arabic, CJK, and typographic punctuation.
fn is_strippable_punctuation(c: char) -> bool {
    matches!(
        c,
        '.' | ',' | '!' | ';' | ':' | '?' |
        // Arabic punctuation
        '\u{061F}' |  // ؟ Arabic question mark
        '\u{061B}' |  // ؛ Arabic semicolon
        '\u{06D4}' |  // ۔ Arabic full stop
        '\u{060C}' |  // ، Arabic comma
        // Guillemets and smart quotes
        '\u{00AB}' |  // « left guillemet
        '\u{00BB}' |  // » right guillemet
        '\u{201C}' |  // " left double quote
        '\u{201D}' |  // " right double quote
        '\u{2018}' |  // ' left single quote
        '\u{2019}' |  // ' right single quote
        // CJK punctuation
        '\u{FF0C}' |  // ， fullwidth comma
        '\u{3002}' |  // 。 ideographic full stop
        '\u{FF01}' |  // ！ fullwidth exclamation
        '\u{FF1F}' |  // ？ fullwidth question mark
        // Spanish inverted punctuation
        '\u{00A1}' |  // ¡ inverted exclamation
        '\u{00BF}' // ¿ inverted question mark
    )
}

pub fn tokenize(
    input: &str,
    registry: Option<&SymbolRegistry>,
    ops: Option<&VsaOps>,
    item_memory: Option<&ItemMemory>,
    lexicon: &Lexicon,
) -> Vec<Token> {
    use unicode_normalization::UnicodeNormalization;
    let trimmed: String = input.trim().nfc().collect();
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

        // Strip trailing punctuation (Unicode-aware)
        let clean = word.trim_end_matches(is_strippable_punctuation);
        // Also strip leading inverted punctuation (Spanish ¡¿)
        let clean = clean.trim_start_matches(is_strippable_punctuation);

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
        let pattern = &lexicon
            .relational_patterns()
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
        let pattern = lexicon
            .relational_patterns()
            .iter()
            .find(|p| p.predicate_label == "contains")
            .unwrap();
        let result = find_relational_pattern(&tokens, pattern);
        assert!(result.is_some());
    }

    #[test]
    fn command_matching() {
        let lexicon = Lexicon::default_english();
        assert!(matches!(
            lexicon.match_command("help"),
            Some(CommandKind::Help)
        ));
        assert!(matches!(
            lexicon.match_command("status"),
            Some(CommandKind::ShowStatus)
        ));
        assert!(matches!(
            lexicon.match_command("run 5"),
            Some(CommandKind::RunAgent { cycles: Some(5) })
        ));
        assert!(matches!(
            lexicon.match_command("show Dog"),
            Some(CommandKind::RenderHiero { .. })
        ));
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
        assert_eq!(
            lexicon.surface_form("contains"),
            Some("contains".to_string())
        );
        assert_eq!(lexicon.surface_form("nonexistent"), None);
    }

    // ── QuestionFrame tests ─────────────────────────────────────────────

    #[test]
    fn question_frame_what_can_you_do() {
        let lexicon = Lexicon::default_english();
        let frame = lexicon.parse_question_frame("What can you do?");
        assert_eq!(frame.question_word.as_deref(), Some("what"));
        assert_eq!(frame.auxiliary.as_deref(), Some("can"));
        assert_eq!(frame.subject_tokens, vec!["you"]);
        assert!(frame.signals_capability);
        assert!(frame.trailing_stripped);
    }

    #[test]
    fn question_frame_what_is_a_dog() {
        let lexicon = Lexicon::default_english();
        let frame = lexicon.parse_question_frame("What is a dog?");
        assert_eq!(frame.question_word.as_deref(), Some("what"));
        assert_eq!(frame.auxiliary.as_deref(), Some("is"));
        assert_eq!(frame.subject_tokens, vec!["dog"]);
        assert!(!frame.signals_capability);
        assert!(!frame.trailing_stripped);
    }

    #[test]
    fn question_frame_who_are_you() {
        let lexicon = Lexicon::default_english();
        let frame = lexicon.parse_question_frame("Who are you?");
        assert_eq!(frame.question_word.as_deref(), Some("who"));
        assert_eq!(frame.auxiliary.as_deref(), Some("are"));
        assert_eq!(frame.subject_tokens, vec!["you"]);
        assert!(!frame.signals_capability);
    }

    #[test]
    fn question_frame_can_you_help() {
        // "can" is in both question_words and auxiliary_verbs.
        // parse_question_frame checks question_word first, so "can" is the question_word.
        // signals_capability is true because "can" is a capability modal.
        let lexicon = Lexicon::default_english();
        let frame = lexicon.parse_question_frame("Can you help?");
        assert_eq!(frame.question_word.as_deref(), Some("can"));
        assert_eq!(frame.auxiliary, None);
        assert_eq!(frame.subject_tokens, vec!["you", "help"]);
        assert!(frame.signals_capability);
    }

    #[test]
    fn question_frame_auxiliary_accessors() {
        let lexicon = Lexicon::default_english();
        assert!(lexicon.is_auxiliary_verb("can"));
        assert!(lexicon.is_auxiliary_verb("Is"));
        assert!(!lexicon.is_auxiliary_verb("dog"));
        assert!(lexicon.is_trailing_auxiliary("do"));
        assert!(lexicon.is_trailing_auxiliary("does"));
        assert!(!lexicon.is_trailing_auxiliary("can"));
        assert!(lexicon.is_capability_modal("can"));
        assert!(lexicon.is_capability_modal("could"));
        assert!(!lexicon.is_capability_modal("is"));
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
            Resolution::Fuzzy {
                symbol_id,
                similarity,
            } => {
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
                assert_eq!(
                    *symbol_id, sym_hello,
                    "should resolve to the correct symbol"
                );
            }
            other => panic!("expected Fuzzy, got {other:?}"),
        }
    }
}
