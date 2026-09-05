//! Phase 44 — working-context assembly runtime (ADR 065).
//!
//! The **assembly layer**: deciding what, of everything available, enters the
//! model's context window, at what fidelity, within what budget. It sits between
//! the agentic loop (ADR 062, which *acts*) and retrieval/memory (which *stores
//! and recalls*).
//!
//! Foundation (44a + 44f):
//! - [`ContextManager`] — the seam `run_agentic_turn` calls to build each prompt.
//! - [`TokenCounter`] — backend-agnostic token counting ([`HeuristicTokenCounter`]
//!   universal; `CandleTokenCounter` exact, feature-gated).
//! - [`PassThroughManager`] — the byte-identical default (zero behaviour change).
//! - [`budget::BudgetAllocator`] + [`budget::record_assembly_telemetry`] — one
//!   ordered budget division (incl. all-or-nothing atomic sections) + telemetry.
//! - [`Retention`] + [`ProvenanceRetentionClassifier`] + [`SafetyEnforced`] — a
//!   guard that refuses to evict safety-critical spans (complements the ADR-060
//!   *execution* gate, which guards dangerous tool *calls*).
//!
//! Strategies (selectable per akh via `AKH_CONTEXT_STRATEGY`; daemon default
//! `compacting`):
//! - [`CompactingManager`] — history + KG + recalled + seed within the model's
//!   real budget, with allocator-driven section + tool degradation and 44e
//!   discourse selection (recency / graph-proximity). The default.
//! - [`PassiveManager`] (44h) — lean context; the longer thread is carried by
//!   background consolidation + recall, not verbatim history.
//! - [`VirtualContextManager`] (44i) — MemGPT tiers (core/recall/archival) with
//!   the `core_memory_*` self-edit tools. The daemon exposes the read-only
//!   `archival_search` page-in under every strategy.
//!
//! Also: [`discourse`] (44e selection), [`core_memory`] (44i store),
//! [`bench_phase44`](crate::bench_phase44) (44j A/B harness), and the deferred
//! [`policy`] seams (44k KV-cache, 44l retrieval swap — recorded, not built).

pub mod budget;
pub mod compacting;
pub mod core_memory;
pub mod discourse;
pub mod error;
pub mod manager;
pub mod passive;
pub mod policy;
pub mod retention;
pub mod safety;
pub mod types;
pub mod virtual_ctx;

pub use compacting::CompactingManager;
pub use error::{ContextError, ContextResult};
pub use manager::{ContextManager, HeuristicTokenCounter, PassThroughManager, TokenCounter};
pub use passive::PassiveManager;
pub use policy::{
    BackendManagedKv, KgOnlyRetrieval, KvCachePolicy, RetrievalPolicy, RetrievalSource,
};
pub use retention::{ProvenanceRetentionClassifier, Retention};
pub use safety::SafetyEnforced;
pub use types::{
    AssembledContext, ContextSection, ContextWindowStats, DEFAULT_CONTEXT_WINDOW,
    DEFAULT_RESERVED_OUTPUT, DeferredReason, DeferredToolGroup, DropReason, DroppedSpan,
    EpisodeState, SectionStats, StrategyId, TokenBudget, TurnOutcome,
};
pub use virtual_ctx::VirtualContextManager;

#[cfg(feature = "candle-backend")]
pub use manager::CandleTokenCounter;
