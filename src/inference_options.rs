//! Runtime-tunable inference options (Phase 47 / ADR 075).
//!
//! A small, lock-free bundle of per-akh inference toggles shared between the
//! [`Engine`](crate::engine::Engine) — so a tool holding only `&Engine` can flip an option at
//! runtime — and the Candle backend, which reads it once per generation turn. Held behind an
//! `Arc` by both so a change is observed without a restart and without locking the (busy) model
//! mutex. Seeded from the akh's persisted [`AkhConfig`](crate::akh_config::AkhConfig) at startup
//! and mutated live by the `inference_config` agent tool.

use std::sync::atomic::{AtomicBool, Ordering};

/// Live, lock-free inference toggles for one akh.
///
/// Every akh's [`Engine`](crate::engine::Engine) owns one of these behind an `Arc`; the Candle
/// chat backend (when attached) holds a clone of the *same* `Arc`, so a flip on one side is seen
/// by the other on the next turn. Reads/writes are `Relaxed` — these are coarse user-facing
/// switches, not a synchronization primitive.
#[derive(Debug, Default)]
pub struct InferenceOptions {
    /// Speculative (prompt-lookup) **greedy** decode — gemma4 only (Phase 47 / ADR 075).
    ///
    /// When on, greedy gemma4 turns draft a continuation from the token history and verify the
    /// span in one forward (≈2× on echo-heavy output: agentic / RAG / structured), lossless vs
    /// plain greedy. Off ⇒ the plain decode loop. Has no effect under temperature sampling or on
    /// non-gemma4 families (the eligibility gate also requires those).
    spec_decode: AtomicBool,
}

impl InferenceOptions {
    /// Whether speculative decode is currently enabled.
    pub fn spec_decode(&self) -> bool {
        self.spec_decode.load(Ordering::Relaxed)
    }

    /// Enable or disable speculative decode at runtime. Returns the **previous** value, so a
    /// caller can report whether the setting actually changed.
    pub fn set_spec_decode(&self, on: bool) -> bool {
        self.spec_decode.swap(on, Ordering::Relaxed)
    }
}
