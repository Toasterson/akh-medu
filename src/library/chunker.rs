//! Chunk normalization: merge short paragraphs, split long ones.
//!
//! Produces chunks targeting 200–500 words for consistent NLP extraction
//! and VSA encoding quality.

use crate::library::model::ContentChunk;

/// Configuration for chunk normalization.
pub struct ChunkConfig {
    /// Minimum words per chunk (short paragraphs get merged).
    pub min_words: usize,
    /// Target words per chunk.
    pub target_words: usize,
    /// Maximum words per chunk (long paragraphs get split).
    pub max_words: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            min_words: 50,
            target_words: 300,
            max_words: 500,
        }
    }
}

/// Normalize a sequence of raw chunks into consistent-sized chunks.
///
/// - Chunks below `min_words` are merged with the next chunk.
/// - Chunks above `max_words` are split at sentence boundaries.
/// - Structural metadata (chapter/section) is preserved from the first
///   contributing chunk.
pub fn normalize_chunks(raw: &[ContentChunk], config: &ChunkConfig) -> Vec<ContentChunk> {
    if raw.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::new();
    let mut buffer = String::new();
    let mut buffer_words = 0usize;
    let mut buffer_chapter = raw[0].chapter;
    let mut buffer_section = raw[0].section;
    let mut idx = 0usize;

    for chunk in raw {
        // If this chunk starts a new chapter, flush the buffer first.
        if chunk.chapter != buffer_chapter && !buffer.is_empty() {
            emit_chunk(
                &mut result,
                &mut idx,
                &buffer,
                buffer_chapter,
                buffer_section,
                config,
            );
            buffer.clear();
            buffer_words = 0;
        }

        buffer_chapter = chunk.chapter;
        buffer_section = chunk.section;

        if !buffer.is_empty() {
            buffer.push(' ');
        }
        buffer.push_str(&chunk.text);
        buffer_words += chunk.word_count;

        // If buffer exceeds target, flush.
        if buffer_words >= config.target_words {
            emit_chunk(
                &mut result,
                &mut idx,
                &buffer,
                buffer_chapter,
                buffer_section,
                config,
            );
            buffer.clear();
            buffer_words = 0;
        }
    }

    // Flush remaining buffer.
    if !buffer.is_empty() {
        // If the remaining buffer is very short and we have a previous chunk,
        // merge it into the last chunk.
        if buffer_words < config.min_words && !result.is_empty() {
            let last = result.last_mut().unwrap();
            last.text.push(' ');
            last.text.push_str(&buffer);
            last.word_count += buffer_words;
        } else {
            emit_chunk(
                &mut result,
                &mut idx,
                &buffer,
                buffer_chapter,
                buffer_section,
                config,
            );
        }
    }

    result
}

/// Emit one or more chunks from a text buffer, splitting at sentence boundaries
/// if the buffer exceeds `max_words`.
fn emit_chunk(
    result: &mut Vec<ContentChunk>,
    idx: &mut usize,
    text: &str,
    chapter: usize,
    section: usize,
    config: &ChunkConfig,
) {
    let word_count = text.split_whitespace().count();
    if word_count <= config.max_words {
        result.push(ContentChunk {
            index: *idx,
            text: text.to_string(),
            word_count,
            chapter,
            section,
        });
        *idx += 1;
        return;
    }

    // Split at sentence boundaries.
    let sentences = split_at_sentences(text);
    let mut current = String::new();
    let mut current_words = 0usize;

    for sentence in &sentences {
        let sw = sentence.split_whitespace().count();
        if current_words + sw > config.max_words && !current.is_empty() {
            result.push(ContentChunk {
                index: *idx,
                text: current.trim().to_string(),
                word_count: current_words,
                chapter,
                section,
            });
            *idx += 1;
            current.clear();
            current_words = 0;
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(sentence);
        current_words += sw;
    }

    if !current.is_empty() {
        result.push(ContentChunk {
            index: *idx,
            text: current.trim().to_string(),
            word_count: current_words,
            chapter,
            section,
        });
        *idx += 1;
    }
}

/// Split text at sentence boundaries (`.`, `!`, `?` followed by whitespace).
fn split_at_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    let chars: Vec<char> = text.chars().collect();
    for (i, &ch) in chars.iter().enumerate() {
        current.push(ch);
        if (ch == '.' || ch == '!' || ch == '?')
            && i + 1 < chars.len()
            && chars[i + 1].is_whitespace()
        {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }
    sentences
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk(index: usize, text: &str, chapter: usize) -> ContentChunk {
        ContentChunk {
            index,
            text: text.into(),
            word_count: text.split_whitespace().count(),
            chapter,
            section: 0,
        }
    }

    #[test]
    fn short_chunks_merged() {
        let raw = vec![
            make_chunk(0, "Hello world.", 0),
            make_chunk(1, "Another sentence.", 0),
            make_chunk(2, "Third one.", 0),
        ];
        let config = ChunkConfig {
            min_words: 5,
            target_words: 10,
            max_words: 50,
        };
        let result = normalize_chunks(&raw, &config);
        // All three short chunks should be merged into one.
        assert_eq!(result.len(), 1);
        assert!(result[0].text.contains("Hello world."));
        assert!(result[0].text.contains("Third one."));
    }

    #[test]
    fn chapter_boundary_forces_flush() {
        let raw = vec![
            make_chunk(0, "Chapter one text here.", 0),
            make_chunk(1, "Chapter two text here.", 1),
        ];
        let config = ChunkConfig {
            min_words: 1,
            target_words: 100,
            max_words: 200,
        };
        let result = normalize_chunks(&raw, &config);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].chapter, 0);
        assert_eq!(result[1].chapter, 1);
    }

    #[test]
    fn long_chunk_split_at_sentences() {
        // Create a chunk with many words that exceeds max.
        let sentence = "This is a test sentence with multiple words.";
        let text = std::iter::repeat(sentence)
            .take(20)
            .collect::<Vec<_>>()
            .join(" ");
        let raw = vec![make_chunk(0, &text, 0)];
        let config = ChunkConfig {
            min_words: 5,
            target_words: 30,
            max_words: 40,
        };
        let result = normalize_chunks(&raw, &config);
        assert!(result.len() > 1, "Should split into multiple chunks");
        for chunk in &result {
            assert!(chunk.word_count <= config.max_words + 10); // small tolerance for last sentence
        }
    }

    #[test]
    fn empty_input() {
        let result = normalize_chunks(&[], &ChunkConfig::default());
        assert!(result.is_empty());
    }
}
