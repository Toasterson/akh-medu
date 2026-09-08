//! Seshat's Archive — shared corpus library with RAG + pgvector.
//!
//! Named after Seshat, the Egyptian goddess of writing, wisdom, and knowledge.
//!
//! This crate provides a dedicated MCP service (`seshd`) that mediates access
//! to a pgvector-backed PostgreSQL corpus for multiple akh-medu engine instances.
//! Documents are ingested via the `seshat-ingest` CLI (or any tool that writes
//! to the `pa_*` tables), embedded via ONNX sentence transformers, and queried
//! via MCP tools.

pub mod auth;
pub mod config;
#[cfg(feature = "embedding")]
pub mod embed_worker;
#[cfg(feature = "embedding")]
pub mod embedder;
pub mod entity;
pub mod error;
pub mod ingest;
pub mod mcp;
pub mod migration;
/// Phase 38j.1: typed metric-instrument registry for seshd.
/// Available only when the `tracing-otel` feature is enabled (the OTel
/// `Meter` type is required).
#[cfg(feature = "tracing-otel")]
pub mod metrics;
pub mod query;
pub mod scripts;
pub mod transcript;
/// Phase 38i: OTLP tracing skeleton for seshd.
pub mod observability;

pub use config::SeshatConfig;
pub use error::{SeshatError, SeshatResult};
