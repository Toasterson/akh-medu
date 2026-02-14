//! PDF parser using the `pdf-extract` crate.
//!
//! Extracts text from PDF bytes. Since `pdf-extract` returns all pages
//! as a single string, we use page-break heuristics (form feeds) to split
//! into pages, grouping every N pages as a chapter.

use crate::library::error::{LibraryError, LibraryResult};
use crate::library::model::*;
use crate::library::parser::ContentParser;

/// PDF document parser backed by `pdf-extract`.
pub struct PdfParser;

/// Number of pages per chapter grouping.
const PAGES_PER_CHAPTER: usize = 10;

impl ContentParser for PdfParser {
    fn format(&self) -> ContentFormat {
        ContentFormat::Pdf
    }

    fn parse(&self, data: &[u8]) -> LibraryResult<ParsedDocument> {
        let text =
            pdf_extract::extract_text_from_mem(data).map_err(|e| LibraryError::ParseError {
                format: "pdf".into(),
                message: e.to_string(),
            })?;

        if text.trim().is_empty() {
            return Err(LibraryError::EmptyDocument {
                origin: "(pdf)".into(),
            });
        }

        // Split on form feed characters (\x0C) that pdf-extract inserts between pages.
        // If no form feeds, treat double newlines as page breaks.
        let pages: Vec<&str> = if text.contains('\x0C') {
            text.split('\x0C').collect()
        } else {
            // Fallback: split on triple newlines (common in extracted text).
            text.split("\n\n\n").collect()
        };

        let mut elements = Vec::new();
        let mut raw_chunks = Vec::new();
        let mut chunk_index = 0usize;

        // Document root.
        elements.push(DocumentElement {
            kind: ElementKind::Document,
            heading: String::new(),
            text: String::new(),
        });

        // Group pages into chapters.
        let mut chapter_ordinal = 0usize;

        for (page_idx, page) in pages.iter().enumerate() {
            let page_text = page.trim();
            if page_text.is_empty() {
                continue;
            }

            // Start a new chapter every PAGES_PER_CHAPTER pages.
            if page_idx % PAGES_PER_CHAPTER == 0 {
                chapter_ordinal += 1;
                let start_page = page_idx + 1;
                let end_page = (page_idx + PAGES_PER_CHAPTER).min(pages.len());
                elements.push(DocumentElement {
                    kind: ElementKind::Chapter {
                        ordinal: chapter_ordinal,
                    },
                    heading: format!("Pages {start_page}–{end_page}"),
                    text: String::new(),
                });
            }

            // Split each page into paragraphs on double newlines.
            let paragraphs: Vec<&str> = page_text
                .split("\n\n")
                .map(|p| p.trim())
                .filter(|p| !p.is_empty())
                .collect();

            for para in paragraphs {
                // Normalize internal whitespace (PDF often has weird line breaks).
                let normalized: String = para
                    .lines()
                    .map(|l| l.trim())
                    .filter(|l| !l.is_empty())
                    .collect::<Vec<_>>()
                    .join(" ");

                let word_count = normalized.split_whitespace().count();
                if word_count == 0 {
                    continue;
                }

                elements.push(DocumentElement {
                    kind: ElementKind::Paragraph {
                        chunk_index,
                    },
                    heading: format!("para:{chunk_index}"),
                    text: normalized.clone(),
                });

                raw_chunks.push(ContentChunk {
                    index: chunk_index,
                    text: normalized,
                    word_count,
                    chapter: chapter_ordinal,
                    section: 0,
                });

                chunk_index += 1;
            }
        }

        // Try to extract a title from the first non-empty line.
        let title = text
            .lines()
            .map(|l| l.trim())
            .find(|l| !l.is_empty() && l.len() < 200)
            .map(|s| s.to_string());

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
    fn parse_simple_text_as_pdf() {
        // pdf-extract needs actual PDF bytes, so we can only test the error path
        // with non-PDF data.
        let parser = PdfParser;
        let result = parser.parse(b"This is not a PDF");
        assert!(result.is_err());
    }
}
