//! Heka — Postgres+pgvector evaluation store for akh-medu long-term memory.
//!
//! Named after Heka, the Egyptian god of magic and writing — the personification
//! of the creative power of words. This crate provides the foundational schema
//! (5 tables + HNSW index) and a `hekad migrate` CLI that creates the schema on
//! a target PostgreSQL database.
//!
//! This crate mirrors the Phase 37 Seshat pattern: SeaORM 1.x entities, raw-SQL
//! migrations for pgvector types that SeaORM's `ColumnDef` cannot express, and a
//! miette-diagnostic error enum.
//!
//! # Tables
//!
//! | Table                     | Purpose                                          |
//! |---------------------------|--------------------------------------------------|
//! | `heka_akhs`               | Registered akh instances (per-workspace identity)|
//! | `heka_project_bindings`   | Akh ↔ project associations                        |
//! | `heka_drawers`            | Top-level memory entries (a "drawer" of memories) |
//! | `heka_drawer_chunks`      | Text chunks within a drawer                        |
//! | `heka_drawer_embeddings`  | 384-dim pgvector embeddings for chunks (HNSW idx) |
//!
//! See `migrations/20260427_000001_init.sql` for the full DDL.

pub mod chunker;
pub mod config;
#[cfg(feature = "embedding")]
pub mod embed_worker;
#[cfg(feature = "embedding")]
pub mod embedder;
pub mod entity;
pub mod error;
pub mod eval;
pub mod ingest;
pub mod migration;

pub use config::HekaConfig;
pub use error::{HekaError, HekaResult};