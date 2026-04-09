//! Seshat's Archive — shared corpus library with RAG + pgvector.
//!
//! Named after Seshat, the Egyptian goddess of writing, wisdom, and knowledge.
//!
//! This crate provides a dedicated MCP service (`seshd`) that mediates access
//! to a pgvector-backed PostgreSQL corpus for multiple akh-medu engine instances.
//! Documents are ingested via the `seshat-ingest` CLI (or any tool that writes
//! to the `pa_*` tables), embedded via ONNX sentence transformers, and queried
//! via MCP tools.

pub mod config;
#[cfg(feature = "embedding")]
pub mod embed_worker;
#[cfg(feature = "embedding")]
pub mod embedder;
pub mod entity;
pub mod error;
pub mod mcp;
pub mod migration;
pub mod query;

pub use config::SeshatConfig;
pub use error::{SeshatError, SeshatResult};
