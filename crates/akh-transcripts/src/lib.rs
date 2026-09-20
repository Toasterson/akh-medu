//! Agent-session transcript harvest (ADR 108).
//!
//! Design: `docs/wayfinder/transcript-harvest/` (map, tickets, `CONTEXT.md`
//! glossary). Plan: `docs/ai/plans/2026-09-19-transcript-harvest.md`.
//!
//! Layers, in data-flow order:
//! - [`adapter`]  — read the three sources into verbatim records (no database)
//! - [`entity`] / [`migration`] / [`pool`] — the `harvest-pg` store
//! - [`normalize`] / [`render`] / [`redact`] / [`chunk`] — derivation logic
//! - [`derive`] — the `akh-derive` worker loop that composes the derivation
//!   layer above into passes over `harvest-pg` (anti-join discovery, session
//!   rollups, embedding)
//!
//! Binaries `akh-harvest`, `akh-derive` and `akh-transcripts-mcp` compose these.

pub mod error;
pub mod model;

pub mod adapter;
pub mod entity;
pub mod migration;
pub mod pool;

pub mod chunk;
pub mod normalize;
pub mod redact;
pub mod render;

pub mod derive;

pub use error::{TranscriptError, TranscriptResult};
