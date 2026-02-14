//! EPUB parser using the `epub` crate.
//!
//! Spine items become chapters. Inner XHTML content is parsed with `scraper`
//! to extract headings, paragraphs, and metadata.

use std::io::Cursor;

use scraper::{Html, Selector};

use crate::library::error::{LibraryError, LibraryResult};
use crate::library::model::*;
use crate::library::parser::ContentParser;

/// EPUB document parser backed by the `epub` crate + `scraper` for HTML content.
pub struct EpubParser;

impl ContentParser for EpubParser {
    fn format(&self) -> ContentFormat {
        ContentFormat::Epub
    }

    fn parse(&self, data: &[u8]) -> LibraryResult<ParsedDocument> {
        let cursor = Cursor::new(data.to_vec());
        let mut doc =
            epub::doc::EpubDoc::from_reader(cursor).map_err(|e| LibraryError::ParseError {
                format: "epub".into(),
                message: e.to_string(),
            })?;

        let metadata = extract_epub_metadata(&doc);

        let mut elements = Vec::new();
        let mut raw_chunks = Vec::new();
        let mut chunk_index = 0usize;

        // Document root.
        elements.push(DocumentElement {
            kind: ElementKind::Document,
            heading: String::new(),
            text: String::new(),
        });

        let num_chapters = doc.get_num_chapters();

        for chapter_idx in 0..num_chapters {
            doc.set_current_chapter(chapter_idx);

            let chapter_content = match doc.get_current_str() {
                Some((content, _mime)) => content,
                None => continue,
            };

            if chapter_content.trim().is_empty() {
                continue;
            }

            let chapter_ordinal = chapter_idx + 1;

            // Parse the XHTML content of this spine item.
            let html = Html::parse_document(&chapter_content);

            // Try to get a chapter heading from the first h1/h2/h3.
            let chapter_heading = extract_first_heading(&html)
                .unwrap_or_else(|| format!("Chapter {chapter_ordinal}"));

            elements.push(DocumentElement {
                kind: ElementKind::Chapter {
                    ordinal: chapter_ordinal,
                },
                heading: chapter_heading,
                text: String::new(),
            });

            // Extract sections and paragraphs from the HTML.
            let mut section_ordinal = 0usize;

            let content_sel =
                Selector::parse("h2, h3, h4, h5, h6, p, li, blockquote, pre, div.text, div.body")
                    .expect("static selector");

            for el in html.select(&content_sel) {
                let tag = el.value().name();
                let text = el.text().collect::<String>();
                let text = text.trim().to_string();

                if text.is_empty() {
                    continue;
                }

                match tag {
                    "h2" | "h3" | "h4" | "h5" | "h6" => {
                        section_ordinal += 1;
                        elements.push(DocumentElement {
                            kind: ElementKind::Section {
                                chapter: chapter_ordinal,
                                ordinal: section_ordinal,
                            },
                            heading: text,
                            text: String::new(),
                        });
                    }
                    _ => {
                        let word_count = text.split_whitespace().count();
                        if word_count == 0 {
                            continue;
                        }

                        elements.push(DocumentElement {
                            kind: ElementKind::Paragraph { chunk_index },
                            heading: format!("para:{chunk_index}"),
                            text: text.clone(),
                        });

                        raw_chunks.push(ContentChunk {
                            index: chunk_index,
                            text,
                            word_count,
                            chapter: chapter_ordinal,
                            section: section_ordinal,
                        });

                        chunk_index += 1;
                    }
                }
            }
        }

        if raw_chunks.is_empty() {
            return Err(LibraryError::EmptyDocument {
                origin: "(epub)".into(),
            });
        }

        Ok(ParsedDocument {
            metadata,
            elements,
            raw_chunks,
        })
    }
}

/// Extract metadata from EPUB document.
fn extract_epub_metadata(doc: &epub::doc::EpubDoc<Cursor<Vec<u8>>>) -> DocumentMetadata {
    let title = doc.mdata("title").map(|m| m.value.clone());
    let author = doc.mdata("creator").map(|m| m.value.clone());
    let description = doc.mdata("description").map(|m| m.value.clone());
    let language = doc.mdata("language").map(|m| m.value.clone());

    DocumentMetadata {
        title,
        author,
        description,
        language,
        keywords: Vec::new(),
    }
}

/// Extract the first heading (h1, h2, or h3) from an HTML fragment.
fn extract_first_heading(html: &Html) -> Option<String> {
    let sel = Selector::parse("h1, h2, h3").ok()?;
    html.select(&sel)
        .next()
        .map(|el| el.text().collect::<String>().trim().to_string())
        .filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_invalid_epub_returns_error() {
        let parser = EpubParser;
        let result = parser.parse(b"This is not an EPUB");
        assert!(result.is_err());
    }
}
