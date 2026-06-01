//! Heka error types.

use miette::Diagnostic;
use thiserror::Error;

#[derive(Debug, Error, Diagnostic)]
pub enum HekaError {
    #[error("drawer not found: {id}")]
    #[diagnostic(
        code(heka::drawer::not_found),
        help("Use `heka_remember` to create a drawer first, then pass its ID.")
    )]
    DrawerNotFound { id: String },

    #[error("drawer store error: {message}")]
    #[diagnostic(
        code(heka::drawer::store),
        help("Check that the Heka data directory is writable.")
    )]
    Store { message: String },

    #[error("serialization error: {message}")]
    #[diagnostic(
        code(heka::drawer::serde),
        help("Drawer data may be corrupted. Re-create the affected drawer.")
    )]
    Serialization { message: String },

    #[error("engine error: {message}")]
    #[diagnostic(
        code(heka::engine),
        help("Ensure the akh-medu workspace path is correct and accessible.")
    )]
    Engine { message: String },
}

impl From<redb::DatabaseError> for HekaError {
    fn from(e: redb::DatabaseError) -> Self {
        Self::Store { message: e.to_string() }
    }
}

impl From<redb::TransactionError> for HekaError {
    fn from(e: redb::TransactionError) -> Self {
        Self::Store { message: e.to_string() }
    }
}

impl From<redb::TableError> for HekaError {
    fn from(e: redb::TableError) -> Self {
        Self::Store { message: e.to_string() }
    }
}

impl From<redb::StorageError> for HekaError {
    fn from(e: redb::StorageError) -> Self {
        Self::Store { message: e.to_string() }
    }
}

impl From<redb::CommitError> for HekaError {
    fn from(e: redb::CommitError) -> Self {
        Self::Store { message: e.to_string() }
    }
}

impl From<bincode::Error> for HekaError {
    fn from(e: bincode::Error) -> Self {
        Self::Serialization { message: e.to_string() }
    }
}

pub type HekaResult<T> = std::result::Result<T, HekaError>;
