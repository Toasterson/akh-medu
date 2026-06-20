//! Phase 44 — working-context assembly runtime (ADR 065).
//!
//! The **assembly layer**: deciding what, of everything available, enters the
//! model's context window, at what fidelity, within what budget. It sits between
//! the agentic loop (ADR 062, which *acts*) and retrieval/memory (which *stores
//! and recalls*).
//!
//! Landed here (44a + 44f):
//! - [`ContextManager`] — the seam `run_agentic_turn` calls to build each prompt.
//! - [`TokenCounter`] — backend-agnostic token counting ([`HeuristicTokenCounter`]
//!   universal; `CandleTokenCounter` exact, feature-gated).
//! - [`PassThroughManager`] — the byte-identical default (zero behaviour change).
//! - [`budget::BudgetAllocator`] + [`budget::record_assembly_telemetry`] — one
//!   ordered budget division + per-turn instrumentation.
//! - [`Retention`] + [`ProvenanceRetentionClassifier`] + [`SafetyEnforced`] — a
//!   guard that refuses to evict safety-critical spans (complements the ADR-060
//!   *execution* gate, which guards dangerous tool *calls*).
//!
//! Specified but gated on the measure-first questions (see the Phase 44 plan):
//! the default `CompactingManager` (budget allocation, graduated fidelity, delta
//! KG, discourse selection, VSA-ranked recall), the `PassiveManager` /
//! `VirtualContextManager` strategies, and the A/B harness.

pub mod budget;
pub mod compacting;
pub mod error;
pub mod manager;
pub mod retention;
pub mod safety;
pub mod types;

pub use compacting::CompactingManager;
pub use error::{ContextError, ContextResult};
pub use manager::{ContextManager, HeuristicTokenCounter, PassThroughManager, TokenCounter};
pub use retention::{ProvenanceRetentionClassifier, Retention};
pub use safety::SafetyEnforced;
pub use types::{
    AssembledContext, ContextSection, DroppedSpan, EpisodeState, StrategyId, TokenBudget,
    TurnOutcome, DEFAULT_CONTEXT_WINDOW, DEFAULT_RESERVED_OUTPUT,
};

#[cfg(feature = "candle-backend")]
pub use manager::CandleTokenCounter;
