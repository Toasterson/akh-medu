//! Shared content library for ingesting books, websites, and documents.
//!
//! Documents are parsed into structural elements (chapters, sections, paragraphs),
//! stored as KG symbols with well-known `doc:*` predicates, enriched with NLP
//! extraction, and embedded via VSA for semantic search.
//!
//! Each document gets its own compartment (`library:{slug}`) that can be
//! mounted by any workspace.

pub mod catalog;
pub mod chunker;
pub mod error;
pub mod inbox;
pub mod ingest;
pub mod model;
pub mod parser;
pub mod predicates;

pub use catalog::LibraryCatalog;
pub use error::{LibraryError, LibraryResult};
pub use ingest::{ingest_document, ingest_file, ingest_url, IngestConfig, IngestResult};
pub use model::{ContentFormat, DocumentRecord, DocumentSource};
pub use predicates::LibraryPredicates;
