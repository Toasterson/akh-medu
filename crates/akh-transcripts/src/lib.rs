//! Agent-session transcript harvest (ADR 108).
//!
//! Design: `docs/wayfinder/transcript-harvest/` (map, tickets, `CONTEXT.md`
//! glossary). Plan: `docs/ai/plans/2026-09-19-transcript-harvest.md`.
//!
//! Layers, in data-flow order:
//! - [`adapter`]  — read the three sources into verbatim records (no database)
//! - [`entity`] / [`migration`] / [`pool`] — the `harvest-pg` store
//! - [`normalize`] / [`render`] / [`redact`] / [`chunk`] — derivation
//!
//! Binaries `akh-harvest`, `akh-derive` and `akh-transcripts-mcp` compose these.

pub mod error;
pub mod model;

pub mod adapter;
pub mod entity;
pub mod harvest;
pub mod migration;
pub mod pool;

pub mod normalize;
pub mod render;
pub mod redact;
pub mod chunk;

pub mod auth;
pub mod mcp;

pub use error::{TranscriptError, TranscriptResult};
