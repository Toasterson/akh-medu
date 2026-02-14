//! Language detection via script analysis and word frequency heuristics.
//!
//! Provides zero-dependency language detection for the supported languages:
//! English, Russian, Arabic, French, Spanish. Uses a two-stage approach:
//!
//! 1. **Script analysis**: Unicode codepoint ranges identify Cyrillic (→Russian)
//!    and Arabic (→Arabic) with high confidence.
//! 2. **Latin disambiguation**: Word frequency heuristics + diacritical markers
//!    distinguish English, French, and Spanish.

use super::lexer::Language;

/// Result of language detection for a text fragment.
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// The detected language.
    pub language: Language,
    /// Confidence score (0.0–1.0).
    pub confidence: f32,
}

/// Detect the language of a text fragment.
///
/// Returns the most likely language with a confidence score.
/// For short texts (< 5 words), confidence will be lower.
pub fn detect_language(text: &str) -> DetectionResult {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return DetectionResult {
            language: Language::English,
            confidence: 0.0,
        };
    }

    // Stage 1: Script analysis — count codepoints by Unicode block
    let mut cyrillic = 0u32;
    let mut arabic = 0u32;
    let mut latin = 0u32;
    let mut total_alpha = 0u32;

    for c in trimmed.chars() {
        if !c.is_alphabetic() {
            continue;
        }
        total_alpha += 1;
        match c {
            // Cyrillic
            '\u{0400}'..='\u{052F}' | '\u{2DE0}'..='\u{2DFF}' | '\u{A640}'..='\u{A69F}' => {
                cyrillic += 1;
            }
            // Arabic + Arabic Supplement + Arabic Extended-A
            '\u{0600}'..='\u{06FF}'
            | '\u{0750}'..='\u{077F}'
            | '\u{08A0}'..='\u{08FF}'
            | '\u{FB50}'..='\u{FDFF}'
            | '\u{FE70}'..='\u{FEFF}' => {
                arabic += 1;
            }
            // Latin (Basic + Extended-A + Extended-B)
            '\u{0041}'..='\u{024F}' | '\u{1E00}'..='\u{1EFF}' => {
                latin += 1;
            }
            _ => {}
        }
    }

    if total_alpha == 0 {
        return DetectionResult {
            language: Language::English,
            confidence: 0.1,
        };
    }

    let cyrillic_ratio = cyrillic as f32 / total_alpha as f32;
    let arabic_ratio = arabic as f32 / total_alpha as f32;
    let latin_ratio = latin as f32 / total_alpha as f32;

    // Cyrillic-dominant → Russian
    if cyrillic_ratio > 0.5 {
        return DetectionResult {
            language: Language::Russian,
            confidence: (0.70 + cyrillic_ratio * 0.25).min(0.95),
        };
    }

    // Arabic-dominant → Arabic
    if arabic_ratio > 0.5 {
        return DetectionResult {
            language: Language::Arabic,
            confidence: (0.70 + arabic_ratio * 0.25).min(0.95),
        };
    }

    // Latin-dominant → disambiguate with word frequency heuristics
    if latin_ratio > 0.5 {
        return detect_latin_language(trimmed);
    }

    // Mixed or unclear — default to English with low confidence
    DetectionResult {
        language: Language::English,
        confidence: 0.3,
    }
}

/// Disambiguate among Latin-script languages using word frequency
/// and diacritical markers.
fn detect_latin_language(text: &str) -> DetectionResult {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();

    // Count language markers
    let mut english_score: f32 = 0.0;
    let mut french_score: f32 = 0.0;
    let mut spanish_score: f32 = 0.0;

    // Word-based markers
    const ENGLISH_MARKERS: &[&str] = &[
        "the", "is", "are", "was", "were", "with", "from", "this", "that", "and", "for", "not",
        "but", "have", "has", "had", "will", "would", "can", "could", "should", "it", "they", "we",
        "you", "he", "she",
    ];
    const FRENCH_MARKERS: &[&str] = &[
        "le", "la", "les", "des", "est", "dans", "avec", "une", "sur", "pour", "pas", "qui", "que",
        "sont", "ont", "fait", "plus", "mais", "aussi", "cette", "ces", "nous", "vous", "ils",
        "elles",
    ];
    const SPANISH_MARKERS: &[&str] = &[
        "el", "los", "las", "está", "esta", "tiene", "tiene", "por", "para", "pero", "también",
        "tambien", "como", "más", "mas", "son", "hay", "ser", "estar", "muy", "todo", "puede",
        "sobre", "nos", "ese", "esa", "estos",
    ];

    for word in &words {
        let w = word.trim_matches(|c: char| !c.is_alphanumeric());
        if ENGLISH_MARKERS.contains(&w) {
            english_score += 1.0;
        }
        if FRENCH_MARKERS.contains(&w) {
            french_score += 1.0;
        }
        if SPANISH_MARKERS.contains(&w) {
            spanish_score += 1.0;
        }
    }

    // Diacritical markers
    let has_french_diacritics = lower.contains('é')
        || lower.contains('è')
        || lower.contains('ê')
        || lower.contains('ë')
        || lower.contains('ç')
        || lower.contains('à')
        || lower.contains('ù')
        || lower.contains('î')
        || lower.contains('ô')
        || lower.contains('œ');

    let has_spanish_diacritics = lower.contains('ñ')
        || lower.contains('á')
        || lower.contains('í')
        || lower.contains('ó')
        || lower.contains('ú')
        || lower.contains('ü');

    let has_inverted_punctuation = text.contains('¿') || text.contains('¡');

    if has_french_diacritics {
        french_score += 2.0;
    }
    if has_spanish_diacritics {
        spanish_score += 2.0;
    }
    if has_inverted_punctuation {
        spanish_score += 3.0;
    }

    // Normalize by word count so longer texts don't inflate any one language
    let word_count = words.len().max(1) as f32;
    let en_norm = english_score / word_count;
    let fr_norm = french_score / word_count;
    let es_norm = spanish_score / word_count;

    let max_score = en_norm.max(fr_norm).max(es_norm);

    if max_score < 0.01 {
        // No markers found — default English with low confidence
        return DetectionResult {
            language: Language::English,
            confidence: 0.4,
        };
    }

    // Pick winner. If scores are very close, confidence is lower.
    let (language, raw_confidence) = if en_norm >= fr_norm && en_norm >= es_norm {
        (
            Language::English,
            0.60 + (en_norm - fr_norm.max(es_norm)).min(0.20),
        )
    } else if fr_norm >= en_norm && fr_norm >= es_norm {
        (
            Language::French,
            0.60 + (fr_norm - en_norm.max(es_norm)).min(0.20),
        )
    } else {
        (
            Language::Spanish,
            0.60 + (es_norm - en_norm.max(fr_norm)).min(0.20),
        )
    };

    DetectionResult {
        language,
        confidence: raw_confidence.min(0.85),
    }
}

/// Detect language per sentence, supporting mixed-language corpora.
///
/// Splits text on sentence boundaries (`.`, `!`, `?`, and their Unicode
/// equivalents) and detects the language of each sentence independently.
pub fn detect_per_sentence(text: &str) -> Vec<(String, DetectionResult)> {
    let sentences = split_sentences(text);
    sentences
        .into_iter()
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            let result = detect_language(&s);
            (s, result)
        })
        .collect()
}

/// Split text into sentences at standard sentence-ending punctuation.
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for c in text.chars() {
        current.push(c);
        if is_sentence_end(c) {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }

    // Remaining text (no trailing punctuation)
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    sentences
}

fn is_sentence_end(c: char) -> bool {
    matches!(
        c,
        '.' | '!' | '?' |
        '\u{061F}' |  // ؟ Arabic question mark
        '\u{06D4}' |  // ۔ Arabic full stop
        '\u{3002}' |  // 。 CJK full stop
        '\u{FF01}' |  // ！ fullwidth exclamation
        '\u{FF1F}' // ？ fullwidth question mark
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_cyrillic_as_russian() {
        let result = detect_language("Собака является млекопитающим");
        assert_eq!(result.language, Language::Russian);
        assert!(
            result.confidence >= 0.90,
            "confidence={}",
            result.confidence
        );
    }

    #[test]
    fn detect_arabic_text() {
        let result = detect_language("القطة حيوان أليف");
        assert_eq!(result.language, Language::Arabic);
        assert!(
            result.confidence >= 0.90,
            "confidence={}",
            result.confidence
        );
    }

    #[test]
    fn detect_french() {
        let result = detect_language("Le chien est un mammifère");
        assert_eq!(result.language, Language::French);
    }

    #[test]
    fn detect_english() {
        let result = detect_language("Dogs are mammals");
        assert_eq!(result.language, Language::English);
    }

    #[test]
    fn detect_spanish() {
        let result = detect_language("El perro es un mamífero");
        assert_eq!(result.language, Language::Spanish);
    }

    #[test]
    fn mixed_corpus_per_sentence() {
        let text = "Dogs are mammals. Собака является млекопитающим.";
        let results = detect_per_sentence(text);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].1.language, Language::English);
        assert_eq!(results[1].1.language, Language::Russian);
    }

    #[test]
    fn empty_text_defaults() {
        let result = detect_language("");
        assert_eq!(result.language, Language::English);
    }
}
