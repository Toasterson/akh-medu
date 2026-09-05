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

    /// Structured log line for this request (one line per request, on completion
    /// or cancellation). The `outcome` string is `"completed"` or `"cancelled"`.
    pub fn log(&self, model: &str, outcome: &str) {
        tracing::info!(
            model,
            outcome,
            request_seq = self.request_seq,
            duration_secs = self.duration_secs(),
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
