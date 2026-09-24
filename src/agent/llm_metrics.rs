//! Per-request LLM metrics instrumentation.
//!
//! Tracks what's happening inside long-running LLM requests to diagnose "turns
//! die quietly" pathologies — separating "receiving bytes but producing nothing"
//! from "genuinely slow generation" from "retry loop" from "thinking that never
//! converges" (work item "Turns die quietly", asks 1–4).

use std::cell::RefCell;
use std::marker::PhantomData;
use std::time::Instant;

// ── Usage metering (Phase 53b, ADR 113 decision 6) ─────────────────────────

/// Model usage accumulated by a [`UsageScope`]: how many provider requests
/// finished and how many tokens they billed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LlmUsage {
    /// Provider requests that reached a model and returned (completed or
    /// cancelled mid-stream). A request that failed before any response
    /// arrived is not counted, because nothing was billed.
    pub requests: u64,
    /// Prompt (input) tokens the provider reported.
    pub prompt_tokens: u64,
    /// Completion (output) tokens the provider reported.
    pub completion_tokens: u64,
}

impl LlmUsage {
    /// Prompt plus completion tokens.
    pub fn total_tokens(&self) -> u64 {
        self.prompt_tokens.saturating_add(self.completion_tokens)
    }

    /// Whether any model request was observed.
    pub fn reached_model(&self) -> bool {
        self.requests > 0
    }

    /// Add one request's worth of usage.
    fn add(&mut self, prompt_tokens: u64, completion_tokens: u64) {
        self.requests = self.requests.saturating_add(1);
        self.prompt_tokens = self.prompt_tokens.saturating_add(prompt_tokens);
        self.completion_tokens = self.completion_tokens.saturating_add(completion_tokens);
    }
}

thread_local! {
    /// Stack of open [`UsageScope`] accumulators on this thread. Every
    /// recorded request is added to every open scope, so an outer scope
    /// always includes the usage of scopes nested inside it.
    static USAGE_SCOPES: RefCell<Vec<LlmUsage>> = const { RefCell::new(Vec::new()) };
}

/// A thread-local meter for LLM usage. Every request logged on this thread
/// through [`LlmRequestMetrics::log`] (or reported with [`record_usage`])
/// while the scope is open is added to it.
///
/// The daemon opens one around each LLM-cost background task so the autonomy
/// budget can charge the task's real token spend. The scope is `!Send` on
/// purpose. A thread-local cannot follow a future across an `.await` onto
/// another worker thread, so the type system keeps it out of async code.
///
/// Scopes nest. Dropping a scope without [`UsageScope::finish`] discards its
/// total but still closes it. Scopes must be closed in LIFO order, which
/// holds automatically for lexically nested scopes.
#[must_use = "a usage scope meters only while it is alive; call `finish` to read the total"]
pub struct UsageScope {
    depth: usize,
    closed: bool,
    _not_send: PhantomData<*const ()>,
}

impl UsageScope {
    /// Open a new metering scope on the current thread.
    pub fn open() -> Self {
        let depth = USAGE_SCOPES.with(|scopes| {
            let mut scopes = scopes.borrow_mut();
            scopes.push(LlmUsage::default());
            scopes.len()
        });
        Self {
            depth,
            closed: false,
            _not_send: PhantomData,
        }
    }

    /// Usage accumulated so far, without closing the scope.
    pub fn usage(&self) -> LlmUsage {
        USAGE_SCOPES.with(|scopes| {
            scopes
                .borrow()
                .get(self.depth - 1)
                .copied()
                .unwrap_or_default()
        })
    }

    /// Close the scope and return its total.
    pub fn finish(mut self) -> LlmUsage {
        let usage = self.usage();
        self.close();
        usage
    }

    fn close(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        USAGE_SCOPES.with(|scopes| scopes.borrow_mut().truncate(self.depth - 1));
    }
}

impl Drop for UsageScope {
    fn drop(&mut self) {
        self.close();
    }
}

/// Report one finished provider request to every open [`UsageScope`] on this
/// thread. A no-op when no scope is open.
///
/// [`LlmRequestMetrics::log`] calls this for the chat paths. Call it directly
/// from LLM call sites that do not use `LlmRequestMetrics` (one-shot
/// `generate` calls, the external extraction API) so the autonomy budget
/// sees their spend too.
pub fn record_usage(prompt_tokens: u64, completion_tokens: u64) {
    USAGE_SCOPES.with(|scopes| {
        for usage in scopes.borrow_mut().iter_mut() {
            usage.add(prompt_tokens, completion_tokens);
        }
    });
}

/// Per-LLM-request metrics, accumulated across the lifetime of one `chat` /
/// `chat_stream` call. Cheap counters only (no string allocations per chunk).
#[derive(Debug, Clone, Default)]
pub struct LlmRequestMetrics {
    /// Wall-clock start of this request.
    started: Option<Instant>,
    /// Total chunks received from the provider (NDJSON lines for Ollama streaming).
    pub chunks_received: usize,
    /// Total bytes received across all chunks.
    pub bytes_received: usize,
    /// Content deltas (`message.content`) that yielded text.
    pub content_deltas: usize,
    /// Thinking deltas (`message.thinking`) that yielded text.
    pub thinking_deltas: usize,
    /// Tool-call chunks (accumulated payloads that don't stream per-delta).
    pub tool_call_chunks: usize,
    /// Keepalive-only chunks (a chunk was received but yielded no visible event).
    pub keepalive_chunks: usize,
    /// Prompt (input) tokens from the provider's final message (`prompt_eval_count`).
    pub prompt_tokens: u64,
    /// Completion (output) tokens from the provider's final message (`eval_count`).
    pub completion_tokens: u64,
    /// Request sequence number within the turn (0 = first, increments on retry).
    pub request_seq: usize,
    /// The per-request `acp.llm_request` child span, created alongside this
    /// metrics handle at each of the five LLM call sites (WI-325). The span
    /// is entered by the call site around the request; accumulated fields are
    /// recorded onto it at `log()` time so span fields exactly match the
    /// struct — no recomputed counters, no double-counting.
    span: Option<tracing::Span>,
}

impl LlmRequestMetrics {
    /// Start tracking a new request at sequence `seq`.
    pub fn new(seq: usize) -> Self {
        Self {
            started: Some(Instant::now()),
            request_seq: seq,
            ..Default::default()
        }
    }

    /// Record one chunk received (increments both `chunks_received` and `bytes_received`).
    pub fn note_chunk(&mut self, bytes: usize) {
        self.chunks_received += 1;
        self.bytes_received += bytes;
    }

    /// Record a content delta.
    pub fn note_content_delta(&mut self) {
        self.content_deltas += 1;
    }

    /// Record a thinking delta.
    pub fn note_thinking_delta(&mut self) {
        self.thinking_deltas += 1;
    }

    /// Record a tool-call chunk.
    pub fn note_tool_call_chunk(&mut self) {
        self.tool_call_chunks += 1;
    }

    /// Record a keepalive-only chunk.
    pub fn note_keepalive(&mut self) {
        self.keepalive_chunks += 1;
    }

    /// Record the provider's final token counts.
    pub fn set_tokens(&mut self, prompt_tokens: u64, completion_tokens: u64) {
        self.prompt_tokens = prompt_tokens;
        self.completion_tokens = completion_tokens;
    }

    /// Wall duration of this request.
    pub fn duration_secs(&self) -> f64 {
        self.started
            .map(|s| s.elapsed().as_secs_f64())
            .unwrap_or(0.0)
    }

    /// Create the `acp.llm_request` child span for this request.
    ///
    /// The span parents onto the ambient span at creation time (the enclosing
    /// `acp.turn` parent entered by `run_turn`'s `spawn_blocking` closure).
    /// All fields the request will accumulate are declared `Empty` up front so
    /// tracing-opentelemetry doesn't drop late-recorded fields; `record_on_span`
    /// fills them at `log()` time from this same struct's values.
    pub fn request_span(model: &str, request_seq: usize) -> tracing::Span {
        tracing::info_span!(
            "acp.llm_request",
            model = model,
            request_seq = request_seq,
            outcome = tracing::field::Empty,
            duration_secs = tracing::field::Empty,
            chunks_received = tracing::field::Empty,
            bytes_received = tracing::field::Empty,
            content_deltas = tracing::field::Empty,
            thinking_deltas = tracing::field::Empty,
            tool_call_chunks = tracing::field::Empty,
            keepalive_chunks = tracing::field::Empty,
            prompt_tokens = tracing::field::Empty,
            completion_tokens = tracing::field::Empty,
        )
    }

    /// Attach the request span created by [`Self::request_span`]. Call this at
    /// the LLM call site immediately after [`Self::new`], then enter the span
    /// for the request's duration.
    pub fn attach_span(&mut self, span: tracing::Span) {
        self.span = Some(span);
    }

    /// Fill the span fields from this struct's accumulated values. Called
    /// once at `log()` time (exactly once per request on every exit path),
    /// which preserves exactly-once recording: the span fields are the same
    /// values the log line emits, never recomputed or counted twice.
    fn record_on_span(&self, outcome: &str, duration_secs: f64) {
        if let Some(span) = &self.span {
            span.record("outcome", outcome);
            span.record("duration_secs", duration_secs);
            span.record("chunks_received", self.chunks_received as u64);
            span.record("bytes_received", self.bytes_received as u64);
            span.record("content_deltas", self.content_deltas as u64);
            span.record("thinking_deltas", self.thinking_deltas as u64);
            span.record("tool_call_chunks", self.tool_call_chunks as u64);
            span.record("keepalive_chunks", self.keepalive_chunks as u64);
            span.record("prompt_tokens", self.prompt_tokens);
            span.record("completion_tokens", self.completion_tokens);
        }
    }

    /// Structured log line for this request (one line per request, on completion
    /// or cancellation). The `outcome` string is `"completed"` or `"cancelled"`.
    pub fn log(&self, model: &str, outcome: &str) {
        let duration_secs = self.duration_secs();
        self.record_on_span(outcome, duration_secs);
        record_usage(self.prompt_tokens, self.completion_tokens);
        tracing::info!(
            model,
            outcome,
            request_seq = self.request_seq,
            duration_secs = duration_secs,
            chunks_received = self.chunks_received,
            bytes_received = self.bytes_received,
            content_deltas = self.content_deltas,
            thinking_deltas = self.thinking_deltas,
            tool_call_chunks = self.tool_call_chunks,
            keepalive_chunks = self.keepalive_chunks,
            prompt_tokens = self.prompt_tokens,
            completion_tokens = self.completion_tokens,
            "llm request metrics"
        );
    }
}

/// Per-turn metrics: aggregates multiple requests (retry attempts) within one
/// agentic turn. Tracks the total prompt size sent (for pathological-latency
/// diagnosis) plus the per-request counter.
#[derive(Debug, Default)]
pub struct TurnMetrics {
    /// Total prompt size (characters) sent across all requests this turn.
    pub total_prompt_chars: usize,
    /// Request sequence counter (increments on each `chat` / `chat_stream` call).
    request_seq: usize,
}

impl TurnMetrics {
    /// Create a new turn tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Note that a prompt of `chars` characters was sent, returning the current
    /// request sequence number (0-indexed) and incrementing for the next call.
    pub fn note_prompt(&mut self, chars: usize) -> usize {
        self.total_prompt_chars += chars;
        let seq = self.request_seq;
        self.request_seq += 1;
        seq
    }

    /// Log the turn-level summary (prompt size, total requests issued).
    pub fn log(&self) {
        tracing::info!(
            total_prompt_chars = self.total_prompt_chars,
            total_requests = self.request_seq,
            "llm turn summary"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_scope_meters_logged_requests() {
        let scope = UsageScope::open();
        let mut request = LlmRequestMetrics::new(0);
        request.set_tokens(120, 30);
        request.log("test-model", "completed");
        record_usage(10, 5);
        let usage = scope.finish();
        assert_eq!(usage.requests, 2);
        assert_eq!(usage.prompt_tokens, 130);
        assert_eq!(usage.completion_tokens, 35);
        assert_eq!(usage.total_tokens(), 165);
        assert!(usage.reached_model());
    }

    #[test]
    fn nested_scopes_roll_up_into_the_outer_scope() {
        let outer = UsageScope::open();
        record_usage(1, 1);
        let inner = UsageScope::open();
        record_usage(2, 3);
        assert_eq!(inner.finish().total_tokens(), 5);
        record_usage(4, 0);
        let total = outer.finish();
        assert_eq!(total.requests, 3);
        assert_eq!(total.total_tokens(), 11);
    }

    #[test]
    fn recording_without_a_scope_is_a_noop_and_dropped_scopes_close() {
        record_usage(99, 99);
        {
            let _dropped = UsageScope::open();
            record_usage(1, 1);
        }
        let fresh = UsageScope::open();
        assert_eq!(fresh.usage(), LlmUsage::default());
        assert!(!fresh.finish().reached_model());
    }
}
