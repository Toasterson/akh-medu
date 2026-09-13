//! Per-request LLM metrics instrumentation.
//!
//! Tracks what's happening inside long-running LLM requests to diagnose "turns
//! die quietly" pathologies — separating "receiving bytes but producing nothing"
//! from "genuinely slow generation" from "retry loop" from "thinking that never
//! converges" (work item "Turns die quietly", asks 1–4).

use std::time::Instant;

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
