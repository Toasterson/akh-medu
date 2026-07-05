//! Phase 50 (W7): benchmark results schema, storage, leaderboard, and
//! self-contained HTML site generator.
//!
//! See `docs/ai/decisions/086-feature-test-vm-harness-benchmarks.md` §4 for
//! the "results discipline" this crate implements, and
//! `docs/ai/plans/2026-07-05-phase50-feature-test-benchmarks.md` (W7) for the
//! work-item scope.
//!
//! Module map:
//! - [`schema`] — the [`schema::RunRecord`] / [`schema::RunLine`] types.
//! - [`hash`] — deterministic task-set content hashing.
//! - [`store`] — `benchmarks/results/<date>-<harness>-<model>/` read/write.
//! - [`leaderboard`] — `benchmarks/LEADERBOARD.md` markdown generation.
//! - [`site`] — self-contained single-page HTML site generation.
//! - [`error`] — the crate's miette `Diagnostic` error type.

pub mod error;
pub mod hash;
pub mod leaderboard;
pub mod schema;
pub mod site;
pub mod store;

pub use error::{HarnessError, HarnessResult};
