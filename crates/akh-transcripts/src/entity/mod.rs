//! SeaORM entities for the `harvest-pg` store (ADR 108).
//!
//! Vocabulary per `docs/wayfinder/transcript-harvest/CONTEXT.md`. Table names
//! are flat and prefixed `th_` (transcript-harvest), matching the project's
//! `akhomed_*` / `pa_*` convention. The DDL that creates these tables lives in
//! [`crate::migration`].
//!
//! - [`raw_record`] — `th_raw_records`, the verbatim system of record.
//! - [`event`] — `th_events`, the normalized cross-tool layer.
//! - [`session`] — `th_sessions`, derived per-session metadata.
//! - [`chunk`] — `th_chunks`, embedding units (`prompt`/`reply` only).
//! - [`drift_counter`] — `th_drift_counters`, unknown/degraded record counts.

pub mod chunk;
pub mod drift_counter;
pub mod event;
pub mod raw_record;
pub mod session;
