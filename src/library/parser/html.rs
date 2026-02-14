//! HTML parser using the `scraper` crate.
//!
//! Extracts `<title>`, heading hierarchy (`<h1>`–`<h6>`), `<p>` paragraphs,
//! and `<meta>` metadata into a `ParsedDocument`.

use scraper::{Html, Selector};

use crate::library::error::LibraryResult;
use crate::library::model::*;
use crate::library::parser::ContentParser;

/// HTML document parser backed by `scraper` (servo's html5ever).
pub struct HtmlParser;

impl ContentParser for HtmlParser {
    fn format(&self) -> ContentFormat {
        ContentFormat::Html
    }

    fn parse(&self, data: &[u8]) -> LibraryResult<ParsedDocument> {
        let text = String::from_utf8_lossy(data);
        let document = Html::parse_document(&text);

        let metadata = extract_metadata(&document);
        let (elements, raw_chunks) = extract_structure(&document);

        Ok(ParsedDocument {
            metadata,
            elements,
            raw_chunks,
        })
    }
}

/// Extract metadata from `<title>` and `<meta>` tags.
fn extract_metadata(document: &Html) -> DocumentMetadata {
    let mut meta = DocumentMetadata::default();

    // <title> tag
    if let Ok(sel) = Selector::parse("title") {
        if let Some(el) = document.select(&sel).next() {
            let title = el.text().collect::<String>().trim().to_string();
            if !title.is_empty() {
                meta.title = Some(title);
            }
        }
    }

    // <meta> tags
    if let Ok(sel) = Selector::parse("meta") {
        for el in document.select(&sel) {
            let name = el.value().attr("name").unwrap_or("").to_lowercase();
            let content = el.value().attr("content").unwrap_or("").trim();
            if content.is_empty() {
                continue;
            }
            match name.as_str() {
                "author" => meta.author = Some(content.to_string()),
                "description" => meta.description = Some(content.to_string()),
                "keywords" => {
                    meta.keywords = content.split(',').map(|k| k.trim().to_string()).collect();
                }
                "language" | "lang" => meta.language = Some(content.to_string()),
                _ => {}
            }
        }
    }

    // <html lang="..."> attribute
    if meta.language.is_none() {
        if let Ok(sel) = Selector::parse("html") {
            if let Some(el) = document.select(&sel).next() {
                if let Some(lang) = el.value().attr("lang") {
                    meta.language = Some(lang.to_string());
                }
            }
        }
    }

    meta
}

/// Walk the HTML DOM and extract structural elements + text chunks.
///
/// Strategy:
/// - `<h1>` tags start new chapters.
/// - `<h2>`–`<h6>` tags start new sections within the current chapter.
/// - `<p>`, `<li>`, `<td>`, `<blockquote>` collect text as paragraphs.
/// - If there are no `<h1>` tags, the entire document is one chapter.
fn extract_structure(document: &Html) -> (Vec<DocumentElement>, Vec<ContentChunk>) {
    let mut elements = Vec::new();
    let mut chunks = Vec::new();

    // Document root element.
    elements.push(DocumentElement {
        kind: ElementKind::Document,
        heading: String::new(),
        text: String::new(),
    });

    let mut chapter_ordinal = 0usize;
    let mut section_ordinal = 0usize;
    let mut chunk_index = 0usize;
    let mut has_h1 = false;

    // Check if there are any h1 elements to determine chapter structure.
    if let Ok(h1_sel) = Selector::parse("h1") {
        has_h1 = document.select(&h1_sel).next().is_some();
    }

    // If no h1 headings, create a default chapter.
    if !has_h1 {
        elements.push(DocumentElement {
            kind: ElementKind::Chapter { ordinal: 0 },
            heading: "Document".into(),
            text: String::new(),
        });
    }

    // Select all content-bearing elements in document order.
    // We use a combined selector and check the tag name.
    let content_selector = Selector::parse("h1, h2, h3, h4, h5, h6, p, li, blockquote, pre")
        .expect("static selector must parse");

    for el in document.select(&content_selector) {
        let tag = el.value().name();
        let text = el.text().collect::<String>();
        let text = text.trim().to_string();

        if text.is_empty() {
            continue;
        }

        match tag {
            "h1" => {
                chapter_ordinal += 1;
                section_ordinal = 0;
                elements.push(DocumentElement {
                    kind: ElementKind::Chapter {
                        ordinal: chapter_ordinal,
                    },
                    heading: text,
                    text: String::new(),
                });
            }
            "h2" | "h3" | "h4" | "h5" | "h6" => {
                section_ordinal += 1;
                let ch = if has_h1 { chapter_ordinal } else { 0 };
                elements.push(DocumentElement {
                    kind: ElementKind::Section {
                        chapter: ch,
                        ordinal: section_ordinal,
                    },
                    heading: text,
                    text: String::new(),
                });
            }
            _ => {
                // Paragraph-level content: p, li, blockquote, pre.
                let word_count = text.split_whitespace().count();
                if word_count == 0 {
                    continue;
                }

                let ch = if has_h1 { chapter_ordinal } else { 0 };

                elements.push(DocumentElement {
                    kind: ElementKind::Paragraph { chunk_index },
                    heading: format!("para:{chunk_index}"),
                    text: text.clone(),
                });

                chunks.push(ContentChunk {
                    index: chunk_index,
                    text,
                    word_count,
                    chapter: ch,
                    section: section_ordinal,
                });

                chunk_index += 1;
            }
        }
    }

    (elements, chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_html() {
        let html = r#"
        <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Chapter One</h1>
            <p>First paragraph of chapter one.</p>
            <p>Second paragraph of chapter one.</p>
            <h1>Chapter Two</h1>
            <p>First paragraph of chapter two.</p>
        </body>
        </html>"#;

        let parser = HtmlParser;
        let doc = parser.parse(html.as_bytes()).unwrap();

        assert_eq!(doc.metadata.title.as_deref(), Some("Test Page"));
        assert_eq!(doc.raw_chunks.len(), 3);
        assert!(
            doc.raw_chunks[0]
                .text
                .contains("First paragraph of chapter one")
        );
        assert!(doc.raw_chunks[2].text.contains("chapter two"));
    }

    #[test]
    fn parse_metadata() {
        let html = r#"
        <html lang="en">
        <head>
            <title>Meta Test</title>
            <meta name="author" content="Jane Doe">
            <meta name="description" content="A test document">
            <meta name="keywords" content="rust, testing, html">
        </head>
        <body><p>Content.</p></body>
        </html>"#;

        let parser = HtmlParser;
        let doc = parser.parse(html.as_bytes()).unwrap();

        assert_eq!(doc.metadata.title.as_deref(), Some("Meta Test"));
        assert_eq!(doc.metadata.author.as_deref(), Some("Jane Doe"));
        assert_eq!(doc.metadata.description.as_deref(), Some("A test document"));
        assert_eq!(doc.metadata.language.as_deref(), Some("en"));
        assert_eq!(doc.metadata.keywords, vec!["rust", "testing", "html"]);
    }

    #[test]
    fn parse_no_headings() {
        let html = r#"
        <html>
        <body>
            <p>First paragraph.</p>
            <p>Second paragraph.</p>
        </body>
        </html>"#;

        let parser = HtmlParser;
        let doc = parser.parse(html.as_bytes()).unwrap();
        assert_eq!(doc.raw_chunks.len(), 2);
        // All chunks in chapter 0.
        assert!(doc.raw_chunks.iter().all(|c| c.chapter == 0));
    }

    #[test]
    fn parse_sections() {
        let html = r#"
        <html>
        <body>
            <h1>Main</h1>
            <h2>Sub One</h2>
            <p>Text under sub one.</p>
            <h2>Sub Two</h2>
            <p>Text under sub two.</p>
        </body>
        </html>"#;

        let parser = HtmlParser;
        let doc = parser.parse(html.as_bytes()).unwrap();

        // Should have chapters and sections.
        let chapters: Vec<_> = doc
            .elements
            .iter()
            .filter(|e| matches!(e.kind, ElementKind::Chapter { .. }))
            .collect();
        assert_eq!(chapters.len(), 1);

        let sections: Vec<_> = doc
            .elements
            .iter()
            .filter(|e| matches!(e.kind, ElementKind::Section { .. }))
            .collect();
        assert_eq!(sections.len(), 2);
    }
}
