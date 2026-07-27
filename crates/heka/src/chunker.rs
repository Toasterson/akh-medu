//! Paragraph-aware chunking for drawer text.
//!
//! Ported from `crates/seshat/src/ingest.rs` (Phase 37). The chunking
//! strategy is **paragraph-aware greedy packing**:
//!
//! 1. The body is split on blank lines into paragraphs.
//! 2. For markdown-formatted input, a leading `# … ######` heading is
//!    pulled off the start of each paragraph and propagated onto the
//!    resulting chunk (and every chunk that follows, until the next
//!    heading is seen).
//! 3. Paragraphs are packed into chunks up to `target_chars` characters.
//!    A paragraph that won't fit gets flushed into a new chunk; one that
//!    is *itself* too large falls back to sentence-boundary splitting,
//!    and finally to hard-character splits for anything sentence-splitting
//!    can't rescue (e.g. giant URLs or minified JSON).
//!
//! The default budget of 1500 chars keeps each chunk under
//! `all-MiniLM-L6-v2`'s 512-WordPiece-token cap with healthy headroom
//! for English prose (~0.25–0.3 tokens per char → ~400 tokens).

/// Default target chunk size in characters.
pub const DEFAULT_TARGET_CHUNK_CHARS: usize = 1500;

/// Minimum chunk-size budget. Guards against pathologically small
/// inputs (`target_chars = 0`) that would otherwise loop.
const MIN_TARGET_CHARS: usize = 200;

/// A chunk-to-be, before it's assigned an ID and persisted.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkDraft {
    pub text: String,
    pub word_count: i32,
    /// Markdown heading propagated from a preceding heading paragraph.
    pub heading: Option<String>,
}

/// Paragraph-aware greedy chunker.
///
/// See the module docstring for the full strategy. `format` is used
/// only to decide whether to pull markdown-style headings off the front
/// of paragraphs; passing `"markdown"` or `"md"` enables heading
/// extraction, anything else treats the input as opaque prose.
pub fn chunk_paragraph_aware(
    body: &str,
    target_chars: usize,
    format: &str,
) -> Vec<ChunkDraft> {
    let target = target_chars.max(MIN_TARGET_CHARS);
    let is_markdown = matches!(format, "markdown" | "md");

    let mut chunks: Vec<ChunkDraft> = Vec::new();
    let mut current = String::new();
    let mut current_heading: Option<String> = None;

    for para in split_paragraphs(body) {
        // If the first line of this paragraph is a markdown heading,
        // split it off and use it as the propagated heading.
        let (heading_update, content) = if is_markdown {
            match split_leading_heading(&para) {
                Some((h, rest)) => (Some(h), rest),
                None => (None, Some(para)),
            }
        } else {
            (None, Some(para))
        };

        if let Some(h) = heading_update {
            flush(&mut chunks, &mut current, current_heading.clone());
            current_heading = Some(h);
        }

        let content = match content {
            Some(c) if !c.trim().is_empty() => c,
            _ => continue,
        };

        if content.chars().count() > target {
            flush(&mut chunks, &mut current, current_heading.clone());
            for sub in split_long_paragraph(&content, target) {
                let wc = word_count(&sub);
                chunks.push(ChunkDraft {
                    text: sub,
                    word_count: wc,
                    heading: current_heading.clone(),
                });
            }
            continue;
        }

        // Will this paragraph push us over the budget?
        let sep = if current.is_empty() { 0 } else { 2 };
        let would_overflow = !current.is_empty()
            && current.chars().count() + sep + content.chars().count() > target;
        if would_overflow {
            flush(&mut chunks, &mut current, current_heading.clone());
            current = content;
        } else if current.is_empty() {
            current = content;
        } else {
            current.push_str("\n\n");
            current.push_str(&content);
        }
    }

    flush(&mut chunks, &mut current, current_heading);
    chunks
}

fn flush(chunks: &mut Vec<ChunkDraft>, current: &mut String, heading: Option<String>) {
    if current.is_empty() {
        return;
    }
    let text = std::mem::take(current);
    let wc = word_count(&text);
    chunks.push(ChunkDraft {
        text,
        word_count: wc,
        heading,
    });
}

/// Split `body` on blank lines, trim each paragraph, drop empties.
/// `\r\n` is normalized to `\n`.
fn split_paragraphs(body: &str) -> Vec<String> {
    let normalized = body.replace("\r\n", "\n");
    let mut out = Vec::new();
    let mut current = String::new();
    for line in normalized.split('\n') {
        if line.trim().is_empty() {
            if !current.trim().is_empty() {
                out.push(current.trim().to_string());
            }
            current.clear();
        } else {
            if !current.is_empty() {
                current.push('\n');
            }
            current.push_str(line);
        }
    }
    if !current.trim().is_empty() {
        out.push(current.trim().to_string());
    }
    out
}

/// If the first line of `para` is a markdown ATX heading (`#` through
/// `######` followed by whitespace), return `(heading_text, remaining_body)`.
/// `remaining_body` is `None` if the heading was the only line.
fn split_leading_heading(para: &str) -> Option<(String, Option<String>)> {
    let mut lines = para.lines();
    let first = lines.next()?;
    let trimmed = first.trim_start();
    let mut chars = trimmed.chars();
    let mut hashes = 0;
    while let Some('#') = chars.clone().next() {
        chars.next();
        hashes += 1;
        if hashes > 6 {
            return None;
        }
    }
    if hashes == 0 {
        return None;
    }
    match chars.next() {
        Some(c) if c.is_whitespace() => {
            let heading = chars.as_str().trim().trim_end_matches('#').trim().to_string();
            let rest: String = lines.collect::<Vec<_>>().join("\n");
            let rest = if rest.trim().is_empty() {
                None
            } else {
                Some(rest.trim().to_string())
            };
            Some((heading, rest))
        }
        _ => None,
    }
}

fn word_count(s: &str) -> i32 {
    s.split_whitespace().count() as i32
}

/// Split an over-budget paragraph on sentence boundaries and greedily
/// repack. Sentences that are themselves too long fall through to
/// `hard_split` as a last resort.
fn split_long_paragraph(para: &str, target: usize) -> Vec<String> {
    let sentences = split_sentences(para);
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();

    for sent in sentences {
        if sent.chars().count() > target {
            if !current.is_empty() {
                out.push(std::mem::take(&mut current));
            }
            out.extend(hard_split(&sent, target));
            continue;
        }
        let sep = if current.is_empty() { 0 } else { 1 };
        if !current.is_empty() && current.chars().count() + sep + sent.chars().count() > target {
            out.push(std::mem::take(&mut current));
            current = sent;
        } else if current.is_empty() {
            current = sent;
        } else {
            current.push(' ');
            current.push_str(&sent);
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// Simple prose sentence splitter: split on `". "`, `"? "`, `"! "`
/// while keeping the terminator with the preceding sentence.
fn split_sentences(para: &str) -> Vec<String> {
    let chars: Vec<char> = para.chars().collect();
    let mut out: Vec<String> = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        let is_term = matches!(c, '.' | '?' | '!');
        let next_ws = i + 1 < chars.len() && chars[i + 1].is_whitespace();
        if is_term && next_ws {
            let piece: String = chars[start..=i].iter().collect();
            let piece = piece.trim().to_string();
            if !piece.is_empty() {
                out.push(piece);
            }
            start = i + 2;
            i = start;
            continue;
        }
        i += 1;
    }
    if start < chars.len() {
        let tail: String = chars[start..].iter().collect();
        let tail = tail.trim().to_string();
        if !tail.is_empty() {
            out.push(tail);
        }
    }
    if out.is_empty() {
        out.push(para.trim().to_string());
    }
    out
}

/// Last-resort character-boundary split for sentences that don't fit.
fn hard_split(s: &str, target: usize) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        let end = (i + target).min(chars.len());
        out.push(chars[i..end].iter().collect::<String>());
        i = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_small_paragraphs_into_one_chunk() {
        let body = "Para one.\n\nPara two.\n\nPara three.";
        let chunks = chunk_paragraph_aware(body, 1500, "text");
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("Para one."));
        assert!(chunks[0].text.contains("Para three."));
        assert!(chunks[0].heading.is_none());
    }

    #[test]
    fn splits_when_budget_exceeded() {
        let para = "Lorem ipsum dolor sit amet consectetur adipiscing elit.";
        let body = std::iter::repeat(para)
            .take(9)
            .collect::<Vec<_>>()
            .join("\n\n");
        let chunks = chunk_paragraph_aware(&body, 200, "text");
        assert!(chunks.len() >= 2);
        for c in &chunks {
            assert!(
                c.text.chars().count() <= 200,
                "chunk {} chars",
                c.text.chars().count()
            );
        }
    }

    #[test]
    fn splits_long_paragraph_on_sentence_boundaries() {
        let long = "One short sentence. ".repeat(50); // ~1000 chars
        let chunks = chunk_paragraph_aware(&long, 300, "text");
        assert!(chunks.len() >= 3);
        for c in &chunks {
            assert!(c.text.chars().count() <= 300);
        }
    }

    #[test]
    fn hard_splits_ridiculously_long_non_sentence() {
        let body = "a".repeat(5000);
        let chunks = chunk_paragraph_aware(&body, 1000, "text");
        assert!(chunks.len() >= 5);
        for c in &chunks {
            assert!(c.text.chars().count() <= 1000);
        }
    }

    #[test]
    fn ignores_tiny_budget_by_enforcing_floor() {
        let body = "para one.\n\npara two.";
        let chunks = chunk_paragraph_aware(body, 10, "text");
        assert_eq!(chunks.len(), 1);
    }
}