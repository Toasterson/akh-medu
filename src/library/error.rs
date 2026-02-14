//! Rich diagnostic error types for the content library.

use miette::Diagnostic;
use thiserror::Error;

/// Errors from content library operations.
#[derive(Debug, Error, Diagnostic)]
pub enum LibraryError {
    #[error("document not found: \"{id}\"")]
    #[diagnostic(
        code(akh::library::not_found),
        help(
            "No document with this ID exists in the catalog. \
             List available documents with `akh-medu library list`."
        )
    )]
    DocumentNotFound { id: String },

    #[error("unsupported content format: \"{format}\"")]
    #[diagnostic(
        code(akh::library::unsupported_format),
        help(
            "Supported formats are: html, pdf, epub, and plain text. \
             If your file uses a different extension, specify the format explicitly \
             with --format."
        )
    )]
    UnsupportedFormat { format: String },

    #[error("parse error in {format} document: {message}")]
    #[diagnostic(
        code(akh::library::parse_error),
        help(
            "The document could not be parsed. Verify the file is valid {format} \
             and not corrupted."
        )
    )]
    ParseError { format: String, message: String },

    #[error("catalog I/O error: {message}")]
    #[diagnostic(
        code(akh::library::catalog_io),
        help(
            "Failed to read or write the library catalog. Check that the library \
             directory exists and has correct permissions."
        )
    )]
    CatalogIo { message: String },

    #[error("ingestion failed for \"{document}\": {message}")]
    #[diagnostic(
        code(akh::library::ingest_failed),
        help(
            "The document ingestion pipeline encountered an error. \
             Check the inner error for details."
        )
    )]
    IngestFailed { document: String, message: String },

    #[error("duplicate document: \"{id}\" already exists in the catalog")]
    #[diagnostic(
        code(akh::library::duplicate),
        help(
            "A document with this slug already exists. Use `library remove {id}` first, \
             or choose a different title."
        )
    )]
    Duplicate { id: String },

    #[error("fetch error for URL \"{url}\": {message}")]
    #[diagnostic(
        code(akh::library::fetch_error),
        help(
            "Failed to download the URL. Check that the URL is reachable \
             and the network is available."
        )
    )]
    FetchError { url: String, message: String },

    #[error("empty document: no content extracted from \"{origin}\"")]
    #[diagnostic(
        code(akh::library::empty_document),
        help(
            "The parser could not extract any content from the source. \
             The file may be empty or contain only non-text elements."
        )
    )]
    EmptyDocument { origin: String },

    #[error(transparent)]
    #[diagnostic(transparent)]
    Engine(#[from] crate::error::EngineError),

    #[error("I/O error: {source}")]
    #[diagnostic(
        code(akh::library::io),
        help("A filesystem operation failed. Check file paths and permissions.")
    )]
    Io {
        #[source]
        source: std::io::Error,
    },
}

/// Convenience alias for library operation results.
pub type LibraryResult<T> = std::result::Result<T, LibraryError>;
