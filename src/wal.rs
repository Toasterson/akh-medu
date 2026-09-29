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
//!   replacements, replay-complete cascade retractions, and turns are dropped
//!   only as a last resort, with a counter for observability.
//!
//! Replay (38c) validates each versioned envelope and applies its exact stored
//! metadata idempotently; compound replace/retract records are indivisible.

use std::collections::VecDeque;
#[cfg(test)]
use std::future::Future;
#[cfg(test)]
use std::pin::Pin;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use akhomed_store::MutationRow;
use akhomed_store::uuid::Uuid;

use crate::mutation::{MutationKind, MutationRecord, MutationSink};

/// Stable workspace UUID for a workspace name (UUIDv5 in the fixed
/// [`akhomed_store::layout::AKH_NAMESPACE`], so the same name maps to the
/// same id across restarts and hosts). The store owns the derivation because
/// the per-akh layout (ADR 120) names files after it.
pub fn workspace_id_for(name: &str) -> Uuid {
    akhomed_store::workspace_id_for(name)
}

/// Stable Akh UUID for a workspace name (one Akh per workspace, for now).
pub fn akh_id_for(name: &str) -> Uuid {
    akhomed_store::akh_id_for(name)
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
        st.queue.push_back(Pending {
            kind,
            payload,
            bytes,
        });
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
    residency: Option<std::sync::Arc<akhomed_store::StablePgSession>>,
    akh_id: Uuid,
    workspace_id: Uuid,
    residency_epoch: i64,
    config: WalConfig,
    buffer: WalBuffer,
    metrics: WalMetrics,
    /// Serializes the complete take -> append -> commit/reinsert interval.
    /// Without this gate, a second flusher can commit later rows while an
    /// earlier failed batch is waiting to be reinserted at the front.
    flush_gate: tokio::sync::Mutex<()>,
    #[cfg(test)]
    append_override: Option<TestAppender>,
}

#[cfg(test)]
type TestAppender = std::sync::Arc<
    dyn Fn(
            Vec<MutationRow>,
        ) -> Pin<Box<dyn Future<Output = akhomed_store::StoreResult<u64>> + Send>>
        + Send
        + Sync,
>;

impl MutationLog {
    /// Build a WAL for `(akh_id, workspace_id)` through the pinned residency
    /// session and fencing epoch owned by this daemon instance.
    pub fn new(
        residency: std::sync::Arc<akhomed_store::StablePgSession>,
        akh_id: Uuid,
        workspace_id: Uuid,
        residency_epoch: i64,
        config: WalConfig,
    ) -> Self {
        let buffer = WalBuffer::new(config.max_buffer_bytes);
        Self {
            residency: Some(residency),
            akh_id,
            workspace_id,
            residency_epoch,
            config,
            buffer,
            metrics: WalMetrics::default(),
            flush_gate: tokio::sync::Mutex::new(()),
            #[cfg(test)]
            append_override: None,
        }
    }

    /// In-memory-only construction for unit tests that inspect the bounded
    /// buffer without ever flushing to PostgreSQL.
    #[cfg(test)]
    pub(crate) fn new_buffered_for_test(
        akh_id: Uuid,
        workspace_id: Uuid,
        config: WalConfig,
    ) -> Self {
        let buffer = WalBuffer::new(config.max_buffer_bytes);
        Self {
            residency: None,
            akh_id,
            workspace_id,
            residency_epoch: 0,
            config,
            buffer,
            metrics: WalMetrics::default(),
            flush_gate: tokio::sync::Mutex::new(()),
            append_override: None,
        }
    }

    #[cfg(test)]
    fn new_with_appender_for_test(
        akh_id: Uuid,
        workspace_id: Uuid,
        config: WalConfig,
        appender: TestAppender,
    ) -> Self {
        let mut log = Self::new_buffered_for_test(akh_id, workspace_id, config);
        log.append_override = Some(appender);
        log
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

    #[cfg(test)]
    pub(crate) fn drain_records_for_test(&self) -> Vec<MutationRecord> {
        self.buffer.drain_records_for_test()
    }

    /// Flush up to one batch. Returns the number of rows committed, or the
    /// storage/fencing failure that left the complete batch queued.
    ///
    /// One async gate spans take, append, and commit/reinsert. This both keeps
    /// an in-flight entry out of the overflow policy and prevents another
    /// caller from overtaking a failed earlier batch.
    pub async fn flush_once_result(&self) -> akhomed_store::StoreResult<usize> {
        let _flush_guard = self.flush_gate.lock().await;
        self.flush_once_locked().await
    }

    async fn flush_once_locked(&self) -> akhomed_store::StoreResult<usize> {
        let batch = self.buffer.take_front(self.config.batch_size);
        if batch.is_empty() {
            return Ok(0);
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
        #[cfg(test)]
        let result = if let Some(appender) = &self.append_override {
            appender(rows.clone()).await
        } else {
            self.append_rows(&rows).await
        };
        #[cfg(not(test))]
        let result = self.append_rows(&rows).await;
        match result {
            Ok(_) => {
                self.metrics.last_flush_size.store(n, Ordering::Relaxed);
                self.metrics
                    .last_flush_latency_ms
                    .store(start.elapsed().as_millis() as u64, Ordering::Relaxed);
                Ok(n)
            }
            Err(error) => {
                self.buffer.reinsert_front(batch);
                self.metrics.flush_failures.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    error = %error,
                    pending = self.pending(),
                    "WAL flush failed; batch retained, will retry"
                );
                Err(error)
            }
        }
    }

    async fn append_rows(&self, rows: &[MutationRow]) -> akhomed_store::StoreResult<u64> {
        match self.residency.as_ref() {
            Some(residency) => {
                akhomed_store::append_mutations_fenced(
                    residency,
                    self.akh_id,
                    self.workspace_id,
                    self.residency_epoch,
                    &rows,
                )
                .await
            }
            None => Err(akhomed_store::StoreError::Conflict {
                message: "buffer-only mutation log cannot flush".into(),
            }),
        }
    }

    /// Compatibility wrapper for callers that only need best-effort progress.
    /// Lifecycle/handoff paths must use the Result-bearing methods below.
    pub async fn flush_once(&self) -> usize {
        self.flush_once_result().await.unwrap_or(0)
    }

    /// Drain every row present until the buffer is empty, preserving strict
    /// order and returning the first failure with its batch still queued.
    pub async fn flush_all(&self) -> akhomed_store::StoreResult<usize> {
        let _flush_guard = self.flush_gate.lock().await;
        let mut total = 0usize;
        loop {
            let flushed = self.flush_once_locked().await?;
            total += flushed;
            if flushed == 0 {
                return Ok(total);
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
                        let _ = self.flush_once_result().await;
                    }
                    _ = shutdown.changed() => {
                        if let Err(error) = self.flush_all().await {
                            tracing::warn!(
                                error = %error,
                                pending = self.pending(),
                                "WAL shutdown drain failed; rows remain buffered"
                            );
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
    use std::sync::Arc;

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
        let seen: Vec<i64> = all
            .iter()
            .map(|p| p.payload["i"].as_i64().unwrap())
            .collect();
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
    fn overflow_keeps_tms_retract_batch_as_one_protected_entry() {
        let cascade = json!({
            "v": crate::mutation::RETRACT_BATCH_MUTATION_VERSION,
            "retracts": [
                {"subject": 1, "predicate": 2, "object": 3},
                {"subject": 4, "predicate": 2, "object": 5},
            ],
            "symbols": [],
        });
        let cascade_bytes = cascade.to_string().len() + 64;
        let buf = WalBuffer::new(cascade_bytes + 8);
        buf.push(MutationKind::RetractBatch, cascade.clone());
        buf.push(
            MutationKind::Retract,
            json!({"padding": "drop this legacy unprotected retract"}),
        );

        assert_eq!(buf.dropped_on_overflow(), 1);
        let remaining = buf.drain_records_for_test();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].kind, MutationKind::RetractBatch);
        assert_eq!(remaining[0].payload, cascade);
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
        let ns: Vec<i64> = remaining
            .iter()
            .map(|p| p.payload["n"].as_i64().unwrap())
            .collect();
        assert!(
            !ns.contains(&1),
            "oldest assert should be dropped first: {ns:?}"
        );
    }

    #[test]
    fn wal_config_clamps_zero_batch() {
        let cfg = WalConfig::new(Duration::from_secs(2), 0, 1024);
        assert_eq!(cfg.batch_size, 1);
    }

    #[tokio::test]
    async fn concurrent_flush_failure_is_reinserted_before_any_later_batch_can_commit() {
        let attempts = Arc::new(Mutex::new(Vec::<i64>::new()));
        let call = Arc::new(AtomicUsize::new(0));
        let first_entered = Arc::new(tokio::sync::Notify::new());
        let release_first = Arc::new(tokio::sync::Notify::new());
        let appender: TestAppender = Arc::new({
            let attempts = Arc::clone(&attempts);
            let call = Arc::clone(&call);
            let first_entered = Arc::clone(&first_entered);
            let release_first = Arc::clone(&release_first);
            move |rows| {
                let attempts = Arc::clone(&attempts);
                let call = Arc::clone(&call);
                let first_entered = Arc::clone(&first_entered);
                let release_first = Arc::clone(&release_first);
                Box::pin(async move {
                    let value = rows[0].payload["i"].as_i64().unwrap();
                    attempts.lock().unwrap().push(value);
                    if call.fetch_add(1, Ordering::SeqCst) == 0 {
                        first_entered.notify_one();
                        release_first.notified().await;
                        return Err(akhomed_store::StoreError::Conflict {
                            message: "injected first append failure".into(),
                        });
                    }
                    Ok(rows.len() as u64)
                })
            }
        });
        let log = Arc::new(MutationLog::new_with_appender_for_test(
            Uuid::new_v4(),
            Uuid::new_v4(),
            WalConfig::new(Duration::from_secs(60), 1, 1 << 20),
            appender,
        ));
        log.record(MutationRecord::new(MutationKind::Assert, json!({"i": 0})));
        log.record(MutationRecord::new(MutationKind::Assert, json!({"i": 1})));

        let first = tokio::spawn({
            let log = Arc::clone(&log);
            async move { log.flush_once_result().await }
        });
        first_entered.notified().await;
        let second = tokio::spawn({
            let log = Arc::clone(&log);
            async move { log.flush_once_result().await }
        });
        tokio::task::yield_now().await;
        assert_eq!(
            *attempts.lock().unwrap(),
            vec![0],
            "a concurrent flusher must not take or append the later row"
        );

        release_first.notify_one();
        assert!(first.await.unwrap().is_err());
        assert_eq!(second.await.unwrap().unwrap(), 1);
        assert_eq!(
            *attempts.lock().unwrap(),
            vec![0, 0],
            "the failed front row must retry before the later row"
        );
        assert_eq!(log.pending(), 1);
        assert_eq!(log.flush_all().await.unwrap(), 1);
        assert_eq!(*attempts.lock().unwrap(), vec![0, 0, 1]);
        assert_eq!(log.pending(), 0);
    }
}
