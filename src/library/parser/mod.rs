//! Document parser trait and format detection.
//!
//! Each supported format (HTML, PDF, EPUB, plain text) implements `ContentParser`.
//! The `parser_for()` factory returns the correct parser for a given format.

pub mod epub;
pub mod html;
pub mod pdf;

use crate::library::error::LibraryResult;
use crate::library::model::{ContentFormat, ParsedDocument};

/// Trait for format-specific document parsers.
pub trait ContentParser {
    /// Parse raw bytes into a structured document.
    fn parse(&self, data: &[u8]) -> LibraryResult<ParsedDocument>;

    /// The format this parser handles.
    fn format(&self) -> ContentFormat;
}

/// Get the appropriate parser for a content format.
pub fn parser_for(format: ContentFormat) -> LibraryResult<Box<dyn ContentParser>> {
    match format {
        ContentFormat::Html => Ok(Box::new(html::HtmlParser)),
        ContentFormat::Pdf => Ok(Box::new(pdf::PdfParser)),
        ContentFormat::Epub => Ok(Box::new(epub::EpubParser)),
        ContentFormat::PlainText => Ok(Box::new(PlainTextParser)),
    }
}

/// Detect the content format from a file extension.
pub fn detect_format(path: &str) -> Option<ContentFormat> {
    let lower = path.to_lowercase();
    if lower.ends_with(".html") || lower.ends_with(".htm") || lower.ends_with(".xhtml") {
        Some(ContentFormat::Html)
    } else if lower.ends_with(".pdf") {
        Some(ContentFormat::Pdf)
    } else if lower.ends_with(".epub") {
        Some(ContentFormat::Epub)
    } else if lower.ends_with(".txt") || lower.ends_with(".md") || lower.ends_with(".text") {
        Some(ContentFormat::PlainText)
    } else {
        None
    }
}

/// Detect format from an HTTP Content-Type header value.
pub fn detect_format_from_content_type(content_type: &str) -> Option<ContentFormat> {
    let ct = content_type.to_lowercase();
    if ct.contains("text/html") || ct.contains("application/xhtml") {
        Some(ContentFormat::Html)
    } else if ct.contains("application/pdf") {
        Some(ContentFormat::Pdf)
    } else if ct.contains("application/epub") {
        Some(ContentFormat::Epub)
    } else if ct.contains("text/plain") {
        Some(ContentFormat::PlainText)
    } else {
        None
    }
}

/// Simple plain-text parser: splits on double newlines for paragraphs.
struct PlainTextParser;

impl ContentParser for PlainTextParser {
    fn format(&self) -> ContentFormat {
        ContentFormat::PlainText
    }

    fn parse(&self, data: &[u8]) -> LibraryResult<ParsedDocument> {
        use crate::library::model::*;

        let text = String::from_utf8_lossy(data);
        let mut elements = Vec::new();
        let mut raw_chunks = Vec::new();

        // Document-level element.
        elements.push(DocumentElement {
            kind: ElementKind::Document,
            heading: String::new(),
            text: String::new(),
        });

        // Single implicit chapter.
        elements.push(DocumentElement {
            kind: ElementKind::Chapter { ordinal: 0 },
            heading: "Document".into(),
            text: String::new(),
        });

        // Split on double newlines for paragraphs.
        let paragraphs: Vec<&str> = text
            .split("\n\n")
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .collect();

        for (i, para) in paragraphs.iter().enumerate() {
            let word_count = para.split_whitespace().count();
            if word_count == 0 {
                continue;
            }

            elements.push(DocumentElement {
                kind: ElementKind::Paragraph { chunk_index: i },
                heading: format!("para:{i}"),
                text: para.to_string(),
            });

            raw_chunks.push(ContentChunk {
                index: i,
                text: para.to_string(),
                word_count,
                chapter: 0,
                section: 0,
            });
        }

        // Try to extract a title from the first line.
        let first_line = text.lines().next().unwrap_or("").trim();
        let title = if !first_line.is_empty() && first_line.len() < 200 {
            Some(first_line.to_string())
        } else {
            None
        };

        Ok(ParsedDocument {
            metadata: DocumentMetadata {
                title,
                ..Default::default()
            },
            elements,
            raw_chunks,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_html() {
        assert_eq!(detect_format("test.html"), Some(ContentFormat::Html));
        assert_eq!(detect_format("test.HTM"), Some(ContentFormat::Html));
    }

    #[test]
    fn detect_pdf() {
        assert_eq!(detect_format("book.pdf"), Some(ContentFormat::Pdf));
    }

    #[test]
    fn detect_epub() {
        assert_eq!(detect_format("novel.epub"), Some(ContentFormat::Epub));
    }

    #[test]
    fn detect_text() {
        assert_eq!(detect_format("notes.txt"), Some(ContentFormat::PlainText));
        assert_eq!(detect_format("readme.md"), Some(ContentFormat::PlainText));
    }

    #[test]
    fn detect_unknown() {
        assert_eq!(detect_format("image.png"), None);
    }

    #[test]
    fn detect_from_content_type() {
        assert_eq!(
            detect_format_from_content_type("text/html; charset=utf-8"),
            Some(ContentFormat::Html)
        );
        assert_eq!(
            detect_format_from_content_type("application/pdf"),
            Some(ContentFormat::Pdf)
        );
    }

    #[test]
    fn plain_text_parser() {
        let text = b"Title Line\n\nFirst paragraph here.\n\nSecond paragraph here.";
        let parser = PlainTextParser;
        let doc = parser.parse(text).unwrap();
        assert_eq!(doc.metadata.title, Some("Title Line".into()));
        assert_eq!(doc.raw_chunks.len(), 3); // Title line + 2 paragraphs
    }
}
