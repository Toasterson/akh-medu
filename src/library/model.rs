//! Core data types for the content library.
//!
//! Documents are decomposed into structural elements (chapters, sections,
//! paragraphs) and metadata. Each element becomes a KG symbol with
//! well-known structural triples.

use serde::{Deserialize, Serialize};

/// Supported document formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContentFormat {
    Html,
    Pdf,
    Epub,
    PlainText,
}

impl ContentFormat {
    /// Human-readable name for diagnostics.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Html => "html",
            Self::Pdf => "pdf",
            Self::Epub => "epub",
            Self::PlainText => "text",
        }
    }
}

impl std::fmt::Display for ContentFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Where the document came from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocumentSource {
    /// Local filesystem path.
    File(String),
    /// Remote URL.
    Url(String),
    /// Inline text (e.g., pasted into chat).
    Inline,
}

impl std::fmt::Display for DocumentSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::File(path) => write!(f, "file:{path}"),
            Self::Url(url) => write!(f, "{url}"),
            Self::Inline => write!(f, "(inline)"),
        }
    }
}

/// Persistent record for a document in the catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentRecord {
    /// URL-safe slug identifier (e.g., "rust-programming-language").
    pub id: String,
    /// Human-readable title.
    pub title: String,
    /// Where it came from.
    pub source: DocumentSource,
    /// Detected or specified format.
    pub format: ContentFormat,
    /// User-assigned tags for categorization.
    pub tags: Vec<String>,
    /// Number of content chunks after normalization.
    pub chunk_count: usize,
    /// Number of triples generated.
    pub triple_count: usize,
    /// Timestamp of ingestion (seconds since UNIX epoch).
    pub ingested_at: u64,
}

/// Metadata extracted from the document during parsing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub title: Option<String>,
    pub author: Option<String>,
    pub description: Option<String>,
    pub language: Option<String>,
    pub keywords: Vec<String>,
}

/// The kind of structural element within a document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ElementKind {
    /// Top-level document container.
    Document,
    /// Chapter-level division (e.g., <h1> section, PDF page group, EPUB spine item).
    Chapter { ordinal: usize },
    /// Section within a chapter (e.g., <h2>/<h3> section).
    Section { chapter: usize, ordinal: usize },
    /// A paragraph or text block.
    Paragraph { chunk_index: usize },
}

/// A single structural element extracted from a document.
#[derive(Debug, Clone)]
pub struct DocumentElement {
    /// What kind of element this is.
    pub kind: ElementKind,
    /// Heading or label for this element.
    pub heading: String,
    /// Raw text content (for paragraphs) or empty for structural containers.
    pub text: String,
}

/// A normalized text chunk ready for NLP extraction and VSA encoding.
#[derive(Debug, Clone)]
pub struct ContentChunk {
    /// Sequential index within the document.
    pub index: usize,
    /// The text content of this chunk.
    pub text: String,
    /// Word count.
    pub word_count: usize,
    /// Which chapter this chunk belongs to (0-indexed).
    pub chapter: usize,
    /// Which section within the chapter (0-indexed, 0 = no subsection).
    pub section: usize,
}

/// Result of parsing a document: structural elements + metadata + raw chunks.
#[derive(Debug)]
pub struct ParsedDocument {
    /// Document-level metadata.
    pub metadata: DocumentMetadata,
    /// Structural elements (document, chapters, sections, paragraphs).
    pub elements: Vec<DocumentElement>,
    /// Pre-chunked text blocks (before normalization).
    pub raw_chunks: Vec<ContentChunk>,
}
