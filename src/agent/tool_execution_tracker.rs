//! Tool execution tracking for ACP session updates.
//!
//! Emits structured `ToolCallStarted` and `ToolCallCompleted` events over the
//! `MessageSink` so ACP clients can display tool activity in their UI. High-volume
//! search operations (seshat lookups, KG queries) are coalesced into burst
//! summaries rather than emitting one event per query.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use crate::message::{AkhMessage, MessageSink, ToolKind};

/// Global monotonic counter for generating unique tool_ids.
/// Uses AtomicU64::fetch_add for lock-free, collision-free IDs.
static TOOL_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Coalescing window for search operations (in milliseconds).
const SEARCH_COALESCE_WINDOW_MS: u64 = 500;

/// Tracks active tool executions and emits ToolCall events.
pub struct ToolExecutionTracker {
    /// Sink to emit ToolCall messages.
    ///
    /// Held as a [`Weak`] so this tracker — which is `Clone` and which
    /// outlives the turn worker via in-flight [`ToolCallGuard`]s — never
    /// keeps the underlying [`MessageSink`] (typically a per-turn
    /// `ChannelSink` over an `mpsc::UnboundedSender`) alive past the turn.
    /// The turn worker owns the only strong `Arc`; once it drops, `upgrade`
    /// returns `None` and `emit` quietly becomes a no-op. This was the
    /// "leaked `ChannelSink` clone keeps the drain channel open" half of
    /// work item `019fba6a-dba7-7d01-bd43-fc51a62587b9`: a strong copy here
    /// held the sender open after the worker joined, so the drain loop
    /// waited a full idle-timeout for a channel close that never came. The
    /// `select!` on `done_rx` is the authoritative fix; this `Weak` is the
    /// defence-in-depth that keeps any future clone from reintroducing the
    /// same wedge.
    sink: Weak<dyn MessageSink>,
    /// Active search burst being coalesced (tool_id -> SearchBurst).
    search_bursts: Arc<Mutex<HashMap<String, SearchBurst>>>,
}

/// A coalesced burst of search operations.
struct SearchBurst {
    /// The unique tool_id for this burst.
    tool_id: String,
    /// When the burst started (for window timeout).
    started_at: Instant,
    /// Queries accumulated in this burst.
    queries: Vec<String>,
    /// Top results (for summary).
    results: Vec<String>,
}

impl ToolExecutionTracker {
    /// Create a new tracker bound to the given sink.
    ///
    /// The sink is downgraded to a [`Weak`] internally (see [`Self::sink`]),
    /// so the caller's strong `Arc` remains the sole owner and the tracker
    /// cannot keep the sink alive past the turn.
    pub fn new(sink: Arc<dyn MessageSink>) -> Self {
        Self {
            sink: Arc::downgrade(&sink),
            search_bursts: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Emit a message through the bound sink, upgrading the [`Weak`] each
    /// call. After the turn worker drops its strong `Arc`, this becomes a
    /// no-op — late tool events (e.g. a `Drop` firing after the worker
    /// returned) are silently dropped, matching the `MessageSink::emit`
    /// infallible contract that already permits dropping on a closed
    /// channel.
    fn emit(&self, msg: &AkhMessage) {
        if let Some(sink) = self.sink.upgrade() {
            sink.emit(msg);
        }
    }

    /// Generate a unique tool_id.
    /// Uses AtomicU64::fetch_add to avoid collisions (fixing the bug from review round 2).
    fn generate_tool_id(tool_name: &str) -> String {
        let id = TOOL_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("{}_{}", tool_name, id)
    }

    /// Determine the ToolKind for a given tool name based on capability.
    fn classify_tool(tool_name: &str) -> ToolKind {
        match tool_name {
            "shell_exec" | "file_io" | "mcp__" | "send_mail" => ToolKind::Execute,
            "edit" => ToolKind::Edit,
            "web_fetch" | "mcp_call" => ToolKind::Fetch,
            "seshat_search" | "seshat_wish" | "search_library" | "kg_query" | "query_symbols" => {
                ToolKind::Search
            }
            _ if tool_name.starts_with("mcp__") => ToolKind::Execute,
            _ => ToolKind::Other,
        }
    }

    /// Start tracking a tool execution.
    ///
    /// For search tools, this may coalesce into an existing burst if one is active
    /// within the coalescing window. For other tools, emits a `ToolCallStarted` event.
    ///
    /// Returns a `ToolCallGuard` that emits the completion event on drop.
    pub fn start_tool(
        &self,
        tool_name: &str,
        title: String,
        locations: Vec<String>,
    ) -> ToolCallGuard {
        let kind = Self::classify_tool(tool_name);

        if kind == ToolKind::Search {
            // Coalesce search operations.
            let mut bursts = self.search_bursts.lock().unwrap();

            // Check if there's an active burst for this tool within the window.
            let burst_key = tool_name.to_string();
            let now = Instant::now();

            let should_create_new = bursts
                .get(&burst_key)
                .map(|b| now.duration_since(b.started_at) > Duration::from_millis(SEARCH_COALESCE_WINDOW_MS))
                .unwrap_or(true);

            if should_create_new {
                // Flush any existing burst.
                if let Some(old_burst) = bursts.remove(&burst_key) {
                    self.flush_search_burst(old_burst);
                }

                // Start a new burst.
                let tool_id = Self::generate_tool_id(tool_name);
                let burst = SearchBurst {
                    tool_id: tool_id.clone(),
                    started_at: now,
                    queries: vec![title.clone()],
                    results: Vec::new(),
                };
                bursts.insert(burst_key.clone(), burst);

                ToolCallGuard {
                    tracker: self.clone(),
                    tool_id,
                    tool_name: tool_name.to_string(),
                    kind,
                    is_search_burst: true,
                    emitted_start: false,
                }
            } else {
                // Add to existing burst.
                if let Some(burst) = bursts.get_mut(&burst_key) {
                    burst.queries.push(title);
                    let tool_id = burst.tool_id.clone();

                    ToolCallGuard {
                        tracker: self.clone(),
                        tool_id,
                        tool_name: tool_name.to_string(),
                        kind,
                        is_search_burst: true,
                        emitted_start: false,
                    }
                } else {
                    // Race: burst was flushed between check and lock.
                    // Fall through to create new burst.
                    let tool_id = Self::generate_tool_id(tool_name);
                    let burst = SearchBurst {
                        tool_id: tool_id.clone(),
                        started_at: now,
                        queries: vec![title],
                        results: Vec::new(),
                    };
                    bursts.insert(burst_key.clone(), burst);

                    ToolCallGuard {
                        tracker: self.clone(),
                        tool_id,
                        tool_name: tool_name.to_string(),
                        kind,
                        is_search_burst: true,
                        emitted_start: false,
                    }
                }
            }
        } else {
            // Non-search: emit ToolCallStarted immediately.
            let tool_id = Self::generate_tool_id(tool_name);
            self.emit(&AkhMessage::ToolCallStarted {
                tool_id: tool_id.clone(),
                kind,
                title,
                locations,
            });

            ToolCallGuard {
                tracker: self.clone(),
                tool_id,
                tool_name: tool_name.to_string(),
                kind,
                is_search_burst: false,
                emitted_start: true,
            }
        }
    }

    /// Flush all active search bursts (called at turn end).
    pub fn flush_all_search_bursts(&self) {
        // Lock the bursts map and drain it. If empty, this is a no-op.
        let mut bursts = self.search_bursts.lock().unwrap_or_else(|e| e.into_inner());
        for (_, burst) in bursts.drain() {
            self.flush_search_burst(burst);
        }
    }

    /// Emit a coalesced search burst as a single ToolCallStarted + ToolCallCompleted pair.
    fn flush_search_burst(&self, burst: SearchBurst) {
        let query_count = burst.queries.len();
        let title = if query_count == 1 {
            burst.queries[0].clone()
        } else {
            format!("{} queries ({})", query_count, burst.queries.join(", "))
        };

        self.emit(&AkhMessage::ToolCallStarted {
            tool_id: burst.tool_id.clone(),
            kind: ToolKind::Search,
            title,
            locations: Vec::new(),
        });

        let output = if burst.results.is_empty() {
            format!("{} queries", query_count)
        } else {
            let preview = burst.results.iter().take(3).map(|s| s.as_str()).collect::<Vec<_>>().join(", ");
            format!("{} queries, top results: {}", query_count, preview)
        };

        self.emit(&AkhMessage::ToolCallCompleted {
            tool_id: burst.tool_id,
            success: true,
            output: Some(output),
            diff: None,
        });
    }

    /// Record a result for the current search burst (if any).
    pub fn record_search_result(&self, tool_name: &str, result: String) {
        let mut bursts = self.search_bursts.lock().unwrap();
        if let Some(burst) = bursts.get_mut(tool_name) {
            burst.results.push(result);
        }
    }
}

impl Clone for ToolExecutionTracker {
    fn clone(&self) -> Self {
        Self {
            sink: self.sink.clone(),
            search_bursts: Arc::clone(&self.search_bursts),
        }
    }
}

/// RAII guard for a tool execution.
///
/// Emits `ToolCallCompleted` on drop (success or failure).
pub struct ToolCallGuard {
    tracker: ToolExecutionTracker,
    tool_id: String,
    tool_name: String,
    kind: ToolKind,
    is_search_burst: bool,
    emitted_start: bool,
}

impl ToolCallGuard {
    /// Mark this tool call as completed with output.
    pub fn complete(self, success: bool, output: Option<String>, diff: Option<String>) {
        // For search bursts, results are accumulated and flushed at turn end.
        // For other tools, emit the completion event immediately.
        if !self.is_search_burst {
            self.tracker.emit(&AkhMessage::ToolCallCompleted {
                tool_id: self.tool_id.clone(),
                success,
                output,
                diff,
            });
        } else if let Some(result) = output {
            // Record search result for coalescing.
            self.tracker.record_search_result(&self.tool_name, result);
        }

        // Don't run Drop (which would emit a duplicate completion).
        std::mem::forget(self);
    }
}

impl Drop for ToolCallGuard {
    fn drop(&mut self) {
        // Emit a completion event if one wasn't already emitted via `complete()`.
        // This handles the case where the guard is dropped without calling `complete()`.
        if !self.is_search_burst && self.emitted_start {
            self.tracker.emit(&AkhMessage::ToolCallCompleted {
                tool_id: self.tool_id.clone(),
                success: false,
                output: Some("tool execution was interrupted".to_string()),
                diff: None,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::VecSink;
    use std::thread;

    #[test]
    fn unique_tool_ids() {
        let sink = Arc::new(VecSink::new());
        let tracker = ToolExecutionTracker::new(sink.clone());

        let guard1 = tracker.start_tool("shell_exec", "ls".to_string(), vec![]);
        let guard2 = tracker.start_tool("shell_exec", "pwd".to_string(), vec![]);

        // IDs must be different (save them before calling complete()).
        let id1 = guard1.tool_id.clone();
        let id2 = guard2.tool_id.clone();
        assert_ne!(id1, id2);

        guard1.complete(true, Some("output1".to_string()), None);
        guard2.complete(true, Some("output2".to_string()), None);
    }

    #[test]
    fn search_coalescing() {
        let sink = Arc::new(VecSink::new());
        let tracker = ToolExecutionTracker::new(sink.clone());

        // Start three seshat_search calls within the coalescing window.
        let guard1 = tracker.start_tool("seshat_search", "query1".to_string(), vec![]);
        let guard2 = tracker.start_tool("seshat_search", "query2".to_string(), vec![]);
        let guard3 = tracker.start_tool("seshat_search", "query3".to_string(), vec![]);

        // All should have the same tool_id (coalesced into one burst).
        let id1 = guard1.tool_id.clone();
        let id2 = guard2.tool_id.clone();
        let id3 = guard3.tool_id.clone();
        assert_eq!(id1, id2);
        assert_eq!(id2, id3);

        guard1.complete(true, Some("result1".to_string()), None);
        guard2.complete(true, Some("result2".to_string()), None);
        guard3.complete(true, Some("result3".to_string()), None);

        // Flush the burst.
        tracker.flush_all_search_bursts();

        // Should have emitted one ToolCallStarted and one ToolCallCompleted.
        let messages = sink.messages();
        let started = messages.iter().filter(|m| matches!(m, AkhMessage::ToolCallStarted { .. })).count();
        let completed = messages.iter().filter(|m| matches!(m, AkhMessage::ToolCallCompleted { .. })).count();

        assert_eq!(started, 1);
        assert_eq!(completed, 1);
    }

    #[test]
    fn search_window_expiry() {
        let sink = Arc::new(VecSink::new());
        let tracker = ToolExecutionTracker::new(sink.clone());

        let guard1 = tracker.start_tool("seshat_search", "query1".to_string(), vec![]);
        let id1 = guard1.tool_id.clone();
        guard1.complete(true, Some("result1".to_string()), None);

        // Wait beyond the coalescing window.
        thread::sleep(Duration::from_millis(SEARCH_COALESCE_WINDOW_MS + 100));

        let guard2 = tracker.start_tool("seshat_search", "query2".to_string(), vec![]);
        let id2 = guard2.tool_id.clone();

        // Should have different tool_ids (new burst).
        assert_ne!(id1, id2);

        guard2.complete(true, Some("result2".to_string()), None);
        tracker.flush_all_search_bursts();
    }

    /// Regression for the "leaked `ChannelSink` clone keeps the drain channel
    /// open" half of work item `019fba6a-dba7-7d01-bd43-fc51a62587b9`.
    ///
    /// The tracker must hold the sink as a `Weak`, not a strong `Arc`, so that
    /// an in-flight `ToolCallGuard` (which clones the tracker) cannot keep the
    /// sink alive past the turn. This test constructs the failure shape
    /// directly: build a tracker bound to a sink, clone the tracker the way a
    /// `ToolCallGuard` does, then drop the *original* strong `Arc` (the one
    /// the turn worker would own). With a `Weak`, the surviving tracker
    /// clone's `sink` field can no longer keep the `Arc` alive —
    /// `Weak::upgrade()` returns `None`, so the late emits are no-ops. With
    /// the old strong `Arc` here, the tracker clone would keep the `Arc`'s
    /// strong count at 1, the `UnboundedSender` inside a real `ChannelSink`
    /// would stay open, and the drain loop would wait a full idle-timeout
    /// for a close that never came.
    #[test]
    fn tracker_does_not_keep_the_sink_alive_past_its_owner() {
        let sink = Arc::new(VecSink::new());
        // Hold a Weak to the *same* allocation so we can observe strong_count
        // without owning a strong ref ourselves.
        let probe = Arc::downgrade(&sink);
        // tracker holds a Weak to this Arc (downgraded inside `new`).
        let tracker = ToolExecutionTracker::new(sink.clone());
        // Strong count is 2: our `sink` + the test's `sink.clone()` we passed
        // to `new`... but `new` drops its parameter after downgrading, so the
        // count is actually 1 (just our `sink`). The tracker's Weak does not
        // contribute to strong_count.
        assert_eq!(
            Arc::strong_count(&sink),
            1,
            "tracker must not hold a strong Arc to the sink; only the caller's \
             strong ref should keep it alive"
        );

        // Emit while the strong Arc is live — normal in-turn path. The
        // ToolCallStarted reaches the sink.
        tracker.start_tool("shell_exec", "ls".to_string(), vec![]);
        assert!(
            !sink.messages().is_empty(),
            "expected the ToolCallStarted to reach the sink while the strong Arc is live, \
             got {} messages",
            sink.messages().len()
        );

        // Simulate the worker dropping its strong Arc while a ToolCallGuard
        // clone (and thus a tracker clone) is still in flight.
        let tracker_clone = tracker.clone();
        drop(sink);

        // The probe Weak must now fail to upgrade: no strong Arc remains. If
        // the tracker held a strong Arc, the probe would still resolve here.
        assert!(
            probe.upgrade().is_none(),
            "dropping the worker's strong Arc must release the sink; a tracker \
             clone (standing in for an in-flight ToolCallGuard) must not keep \
             it alive. This is the leak that wedged the drain loop."
        );

        // The surviving tracker clone emits a late completion. With a Weak,
        // `upgrade()` returns `None` and this is a no-op — no use-after-free,
        // no silent delivery to a kept-alive sink. (Cannot read `sink` here
        // because it is dropped; the structural assertion is that the call
        // returns without touching the freed allocation.)
        tracker_clone.start_tool("shell_exec", "late".to_string(), vec![]);
        tracker.start_tool("shell_exec", "also-late".to_string(), vec![]);
    }
}
