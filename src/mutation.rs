//! Mutation hooks for the write-ahead log (Phase 38b).
//!
//! The [`Engine`](crate::engine::Engine) is synchronous and the OODA loop must
//! never block on durability. So the engine only knows about a tiny
//! [`MutationSink`] trait: after a successful assert/retract it hands the sink a
//! [`MutationRecord`] and moves on. The actual WAL — buffering, batching, and
//! the async flush to Postgres — lives behind that trait in
//! [`crate::wal`](crate::wal) (compiled only under the `postgres` feature).
//!
//! Keeping the trait here, in the core, means the engine carries **no**
//! dependency on SeaORM/sqlx: the lean `akh` client builds without any of it.

/// A single knowledge mutation captured at the engine boundary.
#[derive(Debug, Clone)]
pub struct MutationRecord {
    /// What kind of mutation this is.
    pub kind: MutationKind,
    /// Kind-specific payload, serialised to the `mutation_log.payload` JSONB
    /// column verbatim. For asserts/retracts this carries the symbol ids.
    pub payload: serde_json::Value,
}

impl MutationRecord {
    /// Construct a record from a kind and payload.
    pub fn new(kind: MutationKind, payload: serde_json::Value) -> Self {
        Self { kind, payload }
    }
}

/// Taxonomy of mutations recorded in `akhomed.mutation_log`.
///
/// `as_str` is the exact string stored in the `kind` column; replay (38c)
/// dispatches on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationKind {
    /// A triple was asserted.
    Assert,
    /// A triple was retracted.
    Retract,
    /// A microtheory was activated in the workspace.
    MicrotheoryLoad,
    /// A microtheory was deactivated.
    MicrotheoryUnload,
    /// A compartment was created.
    CompartmentCreate,
    /// A compartment was destroyed.
    CompartmentDestroy,
    /// A completed agentic conversation turn (ADR 091: transcript→episodic
    /// capture). Raw staging, not a KG mutation — never replayed through
    /// `Engine::add_triple` on wake (see the `"turn"` skip arm in the main
    /// crate's `snapshot.rs::apply_mutation`); the KG facts derived from a
    /// turn (if any) get their own `assert` rows. A dropped turn is silent
    /// memory loss, so it is protected on overflow just like `Assert`.
    Turn,
}

impl MutationKind {
    /// The canonical `mutation_log.kind` string for this variant.
    pub fn as_str(self) -> &'static str {
        match self {
            MutationKind::Assert => "assert",
            MutationKind::Retract => "retract",
            MutationKind::MicrotheoryLoad => "microtheory_load",
            MutationKind::MicrotheoryUnload => "microtheory_unload",
            MutationKind::CompartmentCreate => "compartment_create",
            MutationKind::CompartmentDestroy => "compartment_destroy",
            MutationKind::Turn => "turn",
        }
    }

    /// Whether this kind carries knowledge that should be preserved over the
    /// buffer-overflow drop policy. Asserts and turns are protected;
    /// everything else is reconstructable or less critical and is dropped
    /// first under pressure. A dropped turn is silent memory loss (ADR 091),
    /// same severity class as a dropped assert.
    pub fn is_protected_on_overflow(self) -> bool {
        matches!(self, MutationKind::Assert | MutationKind::Turn)
    }
}

/// Receives mutations as they happen at the engine boundary.
///
/// Implementations **must not block** and must not surface errors to the
/// caller: durability is best-effort from the engine's point of view. A WAL
/// that cannot reach Postgres buffers in memory and retries; it never stalls
/// the OODA loop.
pub trait MutationSink: Send + Sync {
    /// Record a mutation. Returns immediately.
    fn record(&self, mutation: MutationRecord);
}

/// Convenience: a sink that counts records, for tests and as a no-Postgres stub.
#[cfg(test)]
#[derive(Default)]
pub struct CountingSink {
    pub count: std::sync::atomic::AtomicUsize,
}

#[cfg(test)]
impl MutationSink for CountingSink {
    fn record(&self, _mutation: MutationRecord) {
        self.count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn kind_strings_are_stable() {
        assert_eq!(MutationKind::Assert.as_str(), "assert");
        assert_eq!(MutationKind::Retract.as_str(), "retract");
        assert_eq!(MutationKind::MicrotheoryLoad.as_str(), "microtheory_load");
        assert_eq!(MutationKind::CompartmentDestroy.as_str(), "compartment_destroy");
        assert_eq!(MutationKind::Turn.as_str(), "turn");
    }

    #[test]
    fn assert_and_turn_are_protected() {
        assert!(MutationKind::Assert.is_protected_on_overflow());
        assert!(MutationKind::Turn.is_protected_on_overflow());
        assert!(!MutationKind::Retract.is_protected_on_overflow());
        assert!(!MutationKind::MicrotheoryLoad.is_protected_on_overflow());
    }

    #[test]
    fn turn_kind_round_trips_through_as_str() {
        // `as_str` is the exact string stored in `mutation_log.kind`; the
        // sleep-cycle harvest phase and wake-replay skip both match on it
        // literally ("turn"), so a rename here is a breaking schema change.
        assert_eq!(MutationKind::Turn.as_str(), "turn");
    }

    #[test]
    fn sink_receives_records() {
        let sink = CountingSink::default();
        sink.record(MutationRecord::new(MutationKind::Assert, serde_json::json!({})));
        sink.record(MutationRecord::new(MutationKind::Retract, serde_json::json!({})));
        assert_eq!(sink.count.load(Ordering::Relaxed), 2);
    }
}
