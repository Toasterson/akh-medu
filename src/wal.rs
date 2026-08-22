//! Phase 38b: the mutation write-ahead log.
//!
//! A [`MutationLog`] is installed per workspace as the engine's
//! [`MutationSink`](crate::mutation::MutationSink). Asserts/retracts append to
//! an in-memory buffer synchronously (the OODA loop never blocks); a background
//! task flushes batches to `akhomed.mutation_log` every `flush_interval`.
//!
//! Durability contract:
//! - **Bounded loss on crash.** A `kill -9` loses only the un-flushed buffer,
//!   bounded by `flush_interval × write_rate`.
//! - **No loss while Postgres is down.** Failed flushes leave the batch in the
//!   buffer (re-inserted at the front) and retry on the next tick.
//! - **Bounded memory.** The buffer is capped at `max_buffer_bytes`; on overflow
//!   the oldest unprotected entries are dropped first. Asserts, compound
//!   replacements, and turns are dropped only as a last resort, with a counter
//!   for observability.
//!
//! Replay (38c) re-applies mutations through `Engine::add_triple`, which is
//! idempotent, so the WAL only needs at-least-once delivery.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use akhomed_store::uuid::Uuid;
use akhomed_store::{DatabaseConnection, MutationRow};

use crate::mutation::{MutationKind, MutationRecord, MutationSink};

/// Namespace for deriving stable per-workspace UUIDs (UUIDv5). Fixed constant
/// so the same workspace name always maps to the same Akh/workspace id across
/// restarts and hosts.
const AKH_NAMESPACE: Uuid = Uuid::from_u128(0x616b_685f_6d65_6475_5f76_3500_0000_0001);

/// Stable workspace UUID for a workspace name.
pub fn workspace_id_for(name: &str) -> Uuid {
    Uuid::new_v5(&AKH_NAMESPACE, format!("workspace:{name}").as_bytes())
}

/// Stable Akh UUID for a workspace name (one Akh per workspace, for now).
pub fn akh_id_for(name: &str) -> Uuid {
    Uuid::new_v5(&AKH_NAMESPACE, format!("akh:{name}").as_bytes())
}

/// Tunables for the WAL buffer + flusher.
#[derive(Debug, Clone)]
pub struct WalConfig {
    /// Interval between flush attempts.
    pub flush_interval: Duration,
    /// Maximum rows per flush transaction.
    pub batch_size: usize,
    /// Maximum buffered bytes before the overflow drop policy kicks in.
    pub max_buffer_bytes: usize,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            flush_interval: Duration::from_secs(2),
            batch_size: 64,
            max_buffer_bytes: 16 * 1024 * 1024,
        }
    }
}

impl WalConfig {
    /// Construct from raw config values.
    pub fn new(flush_interval: Duration, batch_size: usize, max_buffer_bytes: usize) -> Self {
        Self {
            flush_interval,
            // A zero batch size would stall the flusher; clamp to at least 1.
            batch_size: batch_size.max(1),
            max_buffer_bytes,
        }
    }
}

/// A buffered mutation awaiting flush.
struct Pending {
    kind: MutationKind,
    payload: serde_json::Value,
    /// Approximate in-memory size, used for the byte-bounded overflow policy.
    bytes: usize,
}

struct BufferState {
    queue: VecDeque<Pending>,
    bytes: usize,
}

/// The in-memory WAL buffer with a byte-bounded overflow policy.
///
/// Pure and synchronous — holds no database handle — so its push / take /
/// reinsert / overflow behaviour is unit-testable without Postgres.
pub struct WalBuffer {
    inner: Mutex<BufferState>,
    max_bytes: usize,
    dropped_on_overflow: AtomicU64,
}

impl WalBuffer {
    /// Create an empty buffer capped at `max_bytes`.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(BufferState {
                queue: VecDeque::new(),
                bytes: 0,
            }),
            max_bytes,
            dropped_on_overflow: AtomicU64::new(0),
        }
    }

    /// Append a mutation, enforcing the byte cap afterwards.
    pub fn push(&self, kind: MutationKind, payload: serde_json::Value) {
        // Approximate cost: serialised payload + per-row overhead estimate.
        let bytes = payload.to_string().len() + 64;
        let mut st = self.inner.lock().unwrap();
        st.queue.push_back(Pending { kind, payload, bytes });
        st.bytes += bytes;
        self.enforce_cap(&mut st);
    }

    /// Drop oldest non-`assert` entries first, then oldest entries as a last
    /// resort, until under the byte cap.
    fn enforce_cap(&self, st: &mut BufferState) {
        while st.bytes > self.max_bytes && !st.queue.is_empty() {
            let victim = st
                .queue
                .iter()
                .position(|p| !p.kind.is_protected_on_overflow())
                .unwrap_or(0);
            if let Some(dropped) = st.queue.remove(victim) {
                st.bytes -= dropped.bytes;
                self.dropped_on_overflow.fetch_add(1, Ordering::Relaxed);
            } else {
                break;
            }
        }
    }

    /// Remove up to `n` entries from the front for flushing.
    fn take_front(&self, n: usize) -> Vec<Pending> {
        let mut st = self.inner.lock().unwrap();
        let take = st.queue.len().min(n);
        let mut out = Vec::with_capacity(take);
        for _ in 0..take {
            if let Some(p) = st.queue.pop_front() {
                st.bytes -= p.bytes;
                out.push(p);
            }
        }
        out
    }

    /// Return entries to the front, preserving their original order. Used when a
    /// flush fails so the batch is retried next tick.
    fn reinsert_front(&self, items: Vec<Pending>) {
        let mut st = self.inner.lock().unwrap();
        for p in items.into_iter().rev() {
            st.bytes += p.bytes;
            st.queue.push_front(p);
        }
    }

    /// Number of buffered entries.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().queue.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Approximate buffered bytes.
    pub fn bytes(&self) -> usize {
        self.inner.lock().unwrap().bytes
    }

    /// Total entries dropped due to overflow.
    pub fn dropped_on_overflow(&self) -> u64 {
        self.dropped_on_overflow.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn drain_records_for_test(&self) -> Vec<MutationRecord> {
        self.take_front(usize::MAX)
            .into_iter()
            .map(|pending| MutationRecord::new(pending.kind, pending.payload))
            .collect()
    }
}

/// Flush-related observability counters.
#[derive(Debug, Default)]
pub struct WalMetrics {
    /// Rows written by the most recent successful flush.
    pub last_flush_size: AtomicUsize,
    /// Wall-clock latency of the most recent flush attempt, in milliseconds.
    pub last_flush_latency_ms: AtomicU64,
    /// Number of flush attempts that failed (Postgres unreachable, etc.).
    pub flush_failures: AtomicU64,
}

/// Per-workspace write-ahead log: a [`WalBuffer`] plus the async flush path to
/// `akhomed.mutation_log`.
pub struct MutationLog {
    db: DatabaseConnection,
    akh_id: Uuid,
    workspace_id: Uuid,
    config: WalConfig,
    buffer: WalBuffer,
    metrics: WalMetrics,
}

impl MutationLog {
    /// Build a WAL for `(akh_id, workspace_id)` writing to `db`.
    pub fn new(
        db: DatabaseConnection,
        akh_id: Uuid,
        workspace_id: Uuid,
        config: WalConfig,
    ) -> Self {
        let buffer = WalBuffer::new(config.max_buffer_bytes);
        Self {
            db,
            akh_id,
            workspace_id,
            config,
            buffer,
            metrics: WalMetrics::default(),
        }
    }

    /// Observability counters.
    pub fn metrics(&self) -> &WalMetrics {
        &self.metrics
    }

    /// Number of buffered (not-yet-flushed) mutations.
    pub fn pending(&self) -> usize {
        self.buffer.len()
    }

    /// Entries dropped under buffer pressure.
    pub fn dropped_on_overflow(&self) -> u64 {
        self.buffer.dropped_on_overflow()
    }

    /// Flush up to one batch. Returns the number of rows committed.
    ///
    /// The batch is removed from the buffer **before** the await so a concurrent
    /// overflow-drop cannot delete an in-flight entry; on failure it is
    /// re-inserted at the front and retried next tick.
    pub async fn flush_once(&self) -> usize {
        let batch = self.buffer.take_front(self.config.batch_size);
        if batch.is_empty() {
            return 0;
        }
        let rows: Vec<MutationRow> = batch
            .iter()
            .map(|p| MutationRow {
                kind: p.kind.as_str().to_owned(),
                payload: p.payload.clone(),
            })
            .collect();
        let n = rows.len();

        let start = std::time::Instant::now();
        match akhomed_store::append_mutations(&self.db, self.akh_id, self.workspace_id, &rows).await
        {
            Ok(_) => {
                self.metrics.last_flush_size.store(n, Ordering::Relaxed);
                self.metrics
                    .last_flush_latency_ms
                    .store(start.elapsed().as_millis() as u64, Ordering::Relaxed);
                n
            }
            Err(e) => {
                self.buffer.reinsert_front(batch);
                self.metrics.flush_failures.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    error = %e,
                    pending = self.pending(),
                    "WAL flush failed; batch retained, will retry"
                );
                0
            }
        }
    }

    /// Spawn the background flusher. Performs a full drain on shutdown signal.
    pub fn spawn_flusher(
        self: std::sync::Arc<Self>,
        mut shutdown: tokio::sync::watch::Receiver<bool>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(self.config.flush_interval);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tokio::select! {
                    _ = tick.tick() => {
                        let _ = self.flush_once().await;
                    }
                    _ = shutdown.changed() => {
                        // Final drain: flush until the buffer is empty (or a flush
                        // fails, in which case we stop to avoid spinning).
                        while !self.buffer.is_empty() {
                            if self.flush_once().await == 0 {
                                break;
                            }
                        }
                        break;
                    }
                }
            }
        })
    }
}

impl MutationSink for MutationLog {
    fn record(&self, mutation: MutationRecord) {
        self.buffer.push(mutation.kind, mutation.payload);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn stable_ids_are_deterministic_and_distinct() {
        assert_eq!(akh_id_for("tecton"), akh_id_for("tecton"));
        assert_eq!(workspace_id_for("tecton"), workspace_id_for("tecton"));
        assert_ne!(akh_id_for("tecton"), akh_id_for("akh-medu"));
        // Akh id and workspace id for the same name must not collide.
        assert_ne!(
            akh_id_for("tecton").as_u128(),
            workspace_id_for("tecton").as_u128()
        );
    }

    #[test]
    fn buffer_push_take_reinsert_preserves_order() {
        let buf = WalBuffer::new(1 << 20);
        for i in 0..5 {
            buf.push(MutationKind::Assert, json!({ "i": i }));
        }
        assert_eq!(buf.len(), 5);
        let batch = buf.take_front(3);
        assert_eq!(batch.len(), 3);
        assert_eq!(buf.len(), 2);
        // Re-insert the failed batch; order must be restored to [0,1,2,3,4].
        buf.reinsert_front(batch);
        assert_eq!(buf.len(), 5);
        let all = buf.take_front(10);
        let seen: Vec<i64> = all.iter().map(|p| p.payload["i"].as_i64().unwrap()).collect();
        assert_eq!(seen, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn overflow_drops_non_assert_first() {
        // Cap that holds ~2 small entries; each push is ~64+payload bytes.
        let buf = WalBuffer::new(200);
        buf.push(MutationKind::Assert, json!({ "n": 1 })); // keep
        buf.push(MutationKind::Retract, json!({ "n": 2 })); // droppable
        buf.push(MutationKind::Assert, json!({ "n": 3 })); // keep
        buf.push(MutationKind::Assert, json!({ "n": 4 })); // forces overflow
        // The retract (only non-protected entry) should have been dropped first.
        assert!(buf.dropped_on_overflow() >= 1);
        let remaining = buf.take_front(10);
        assert!(
            remaining.iter().all(|p| p.kind == MutationKind::Assert),
            "non-assert entries should be dropped before asserts"
        );
    }

    #[test]
    fn overflow_keeps_compound_replace_as_one_protected_entry() {
        let replacement = json!({
            "v": crate::mutation::REPLACE_MUTATION_VERSION,
            "retracts": [{"subject": 1, "predicate": 2, "object": 3}],
            "assertion": {"v": crate::mutation::ASSERT_MUTATION_VERSION},
        });
        let replacement_bytes = replacement.to_string().len() + 64;
        let buf = WalBuffer::new(replacement_bytes + 8);
        buf.push(MutationKind::Replace, replacement.clone());
        buf.push(
            MutationKind::Retract,
            json!({"padding": "this unprotected record must be dropped whole"}),
        );

        assert_eq!(buf.dropped_on_overflow(), 1);
        let remaining = buf.drain_records_for_test();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].kind, MutationKind::Replace);
        assert_eq!(remaining[0].payload, replacement);
    }

    #[test]
    fn overflow_drops_assert_as_last_resort() {
        let buf = WalBuffer::new(120);
        // Only asserts: with the cap exceeded, the oldest assert is dropped.
        buf.push(MutationKind::Assert, json!({ "n": 1 }));
        buf.push(MutationKind::Assert, json!({ "n": 2 }));
        buf.push(MutationKind::Assert, json!({ "n": 3 }));
        assert!(buf.dropped_on_overflow() >= 1);
        // The oldest (n:1) is the one dropped first.
        let remaining = buf.take_front(10);
        let ns: Vec<i64> = remaining.iter().map(|p| p.payload["n"].as_i64().unwrap()).collect();
        assert!(!ns.contains(&1), "oldest assert should be dropped first: {ns:?}");
    }

    #[test]
    fn wal_config_clamps_zero_batch() {
        let cfg = WalConfig::new(Duration::from_secs(2), 0, 1024);
        assert_eq!(cfg.batch_size, 1);
    }
}
