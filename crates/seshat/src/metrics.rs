//! Phase 38j.1 — typed metric-instrument registry for seshd (Seshat's
//! Archive).
//!
//! This module declares the instruments and bounded label sets specified by
//! the §5.4 catalog of the OTLP observability research doc
//! (`docs/ai/research/2026-04-26-otlp-observability.md`) — the Seshat half of
//! the WI-34 decomposition. It does **not** wire runtime call sites; later
//! slices will import `SeshatMetrics` and record values.
//!
//! # Naming rules
//!
//! - All counters end in `_total` (the two `wish` cache gauges/counters named
//!   in the catalog follow the same convention).
//! - All durations are `f64` histograms named `*_duration_seconds` with unit
//!   `"s"`.
//! - All sizes/counts are `i64` asynchronous gauges or `u64` counters.
//!
//! # Bounded labels only
//!
//! Allowed label keys are static enums/strings. The following are **never**
//! used as label values: document IDs, chunk IDs, queries, goals, raw query
//! text, or anything derived from user input. seshd is a shared multi-tenant
//! corpus, so no `workspace` label is carried either — the `service.name`
//! resource attribute already distinguishes seshd from akhomed.
//!
//! The only bounded label in the Seshat catalog is `tool` (the MCP tool
//! name, a static string per handler) and `error_kind`.

use opentelemetry::metrics::{Counter, Histogram, Meter, ObservableGauge};

/// Root metric registry for seshd.
///
/// Construct once at startup via `SeshatMetrics::new(meter)` and share an
/// `Arc` with the components that will emit in later slices.
pub struct SeshatMetrics {
    meter: Meter,

    // ── MCP request surface ───────────────────────────────────────────────
    /// `seshat_mcp_requests_total` — every `#[tool]` handler invocation.
    ///
    /// Type: `u64` counter. Unit: `"request"`. Labels: `tool`, `status`
    /// (`status` ∈ `labels::status::{OK, ERROR}`; WI-326's
    /// `seshat_requests_total{tool,status}` maps onto this catalog counter).
    pub mcp_requests_total: Counter<u64>,

    /// `seshat_mcp_request_duration_seconds` — latency of an MCP tool call.
    ///
    /// Type: `f64` histogram. Unit: `"s"`. Labels: `tool`.
    pub mcp_request_duration_seconds: Histogram<f64>,

    /// `seshat_mcp_request_errors_total` — tool calls that returned
    /// `McpError`.
    ///
    /// Type: `u64` counter. Unit: `"error"`. Labels: `tool`, `error_kind`.
    pub mcp_request_errors_total: Counter<u64>,

    /// `seshat_search_results` — result count of a `seshat_search` call.
    ///
    /// Type: `u64` histogram. Unit: `"result"`. Labels: `tool`.
    pub search_results: Histogram<f64>,

    // ── Wish / microtheory cache ─────────────────────────────────────────
    /// `seshat_wish_cache_hits_total` — `check_cache` returned `Some`.
    ///
    /// Type: `u64` counter. Unit: `"hit"`. Labels: `result` (always
    /// `labels::result::HIT`).
    pub wish_cache_hits_total: Counter<u64>,

    /// `seshat_wish_cache_misses_total` — `check_cache` returned `None`.
    ///
    /// Type: `u64` counter. Unit: `"miss"`. Labels: `result` (always
    /// `labels::result::MISS`).
    pub wish_cache_misses_total: Counter<u64>,

    // ── Corpus size gauges ────────────────────────────────────────────────
    /// `seshat_documents_total` — documents in the archive.
    ///
    /// Type: `i64` observable gauge. Unit: `"document"`. Labels: none.
    pub documents_total: ObservableGauge<i64>,

    /// `seshat_chunks_total` — chunks in the archive.
    ///
    /// Type: `i64` observable gauge. Unit: `"chunk"`. Labels: none.
    pub chunks_total: ObservableGauge<i64>,

    /// `seshat_embedded_chunks_total` — chunks with a live embedding.
    ///
    /// Type: `i64` observable gauge. Unit: `"chunk"`. Labels: none.
    pub embedded_chunks_total: ObservableGauge<i64>,

    /// `seshat_pending_embedding_chunks` — chunks awaiting the embed worker
    /// — **the key backlog metric**.
    ///
    /// Type: `i64` observable gauge. Unit: `"chunk"`. Labels: none.
    pub pending_embedding_chunks: ObservableGauge<i64>,

    /// `seshat_microtheory_cache_size` — cached microtheory rows.
    ///
    /// Type: `i64` observable gauge. Unit: `"entry"`. Labels: none.
    pub microtheory_cache_size: ObservableGauge<i64>,

    // ── Embed worker ─────────────────────────────────────────────────────
    /// `seshat_embed_batches_total` — embed-worker batch iterations.
    ///
    /// Type: `u64` counter. Unit: `"batch"`. Labels: none.
    pub embed_batches_total: Counter<u64>,

    /// `seshat_embed_batch_duration_seconds` — `embed_pending_batch` latency.
    ///
    /// Type: `f64` histogram. Unit: `"s"`. Labels: none.
    pub embed_batch_duration_seconds: Histogram<f64>,

    /// `seshat_embed_chunks_processed_total` — chunks embedded in total.
    ///
    /// Type: `u64` counter. Unit: `"chunk"`. Labels: none.
    pub embed_chunks_processed_total: Counter<u64>,

    // ── Query / ingest pipeline ───────────────────────────────────────────
    /// `seshat_pgvector_search_duration_seconds` — pgvector ANN search
    /// latency.
    ///
    /// Type: `f64` histogram. Unit: `"s"`. Labels: none.
    pub pgvector_search_duration_seconds: Histogram<f64>,

    /// `seshat_ingest_documents_total` — documents ingested via
    /// `ingest_document`.
    ///
    /// Type: `u64` counter. Unit: `"document"`. Labels: none.
    pub ingest_documents_total: Counter<u64>,

    /// `seshat_ingest_chunk_count` — chunk count of one ingested document.
    ///
    /// Type: `f64` histogram. Unit: `"chunk"`. Labels: none.
    pub ingest_chunk_count: Histogram<f64>,

    /// `seshat_db_pool_active` — active SeaORM pool connections.
    ///
    /// Type: `i64` observable gauge. Unit: `"connection"`. Labels: none.
    pub db_pool_active: ObservableGauge<i64>,
}

impl SeshatMetrics {
    /// Build the registry from the supplied `Meter`.
    pub fn new(meter: Meter) -> Self {
        Self {
            meter: meter.clone(),

            mcp_requests_total: meter
                .u64_counter("seshat_mcp_requests_total")
                .with_description("MCP tool handler invocations")
                .with_unit("request")
                .build(),
            mcp_request_duration_seconds: meter
                .f64_histogram("seshat_mcp_request_duration_seconds")
                .with_description("Latency of an MCP tool call")
                .with_unit("s")
                .build(),
            mcp_request_errors_total: meter
                .u64_counter("seshat_mcp_request_errors_total")
                .with_description("MCP tool calls that returned an error")
                .with_unit("error")
                .build(),
            search_results: meter
                .f64_histogram("seshat_search_results")
                .with_description("Result count of a seshat_search call")
                .with_unit("result")
                .build(),

            wish_cache_hits_total: meter
                .u64_counter("seshat_wish_cache_hits_total")
                .with_description("Wish/microtheory cache hits")
                .with_unit("hit")
                .build(),
            wish_cache_misses_total: meter
                .u64_counter("seshat_wish_cache_misses_total")
                .with_description("Wish/microtheory cache misses")
                .with_unit("miss")
                .build(),

            documents_total: meter
                .i64_observable_gauge("seshat_documents_total")
                .with_description("Documents in the archive")
                .with_unit("document")
                .build(),
            chunks_total: meter
                .i64_observable_gauge("seshat_chunks_total")
                .with_description("Chunks in the archive")
                .with_unit("chunk")
                .build(),
            embedded_chunks_total: meter
                .i64_observable_gauge("seshat_embedded_chunks_total")
                .with_description("Chunks with a live embedding")
                .with_unit("chunk")
                .build(),
            pending_embedding_chunks: meter
                .i64_observable_gauge("seshat_pending_embedding_chunks")
                .with_description("Chunks awaiting the embed worker (backlog)")
                .with_unit("chunk")
                .build(),
            microtheory_cache_size: meter
                .i64_observable_gauge("seshat_microtheory_cache_size")
                .with_description("Cached microtheory rows")
                .with_unit("entry")
                .build(),

            embed_batches_total: meter
                .u64_counter("seshat_embed_batches_total")
                .with_description("Embed-worker batch iterations")
                .with_unit("batch")
                .build(),
            embed_batch_duration_seconds: meter
                .f64_histogram("seshat_embed_batch_duration_seconds")
                .with_description("Latency of one embed batch")
                .with_unit("s")
                .build(),
            embed_chunks_processed_total: meter
                .u64_counter("seshat_embed_chunks_processed_total")
                .with_description("Chunks embedded by the worker")
                .with_unit("chunk")
                .build(),

            pgvector_search_duration_seconds: meter
                .f64_histogram("seshat_pgvector_search_duration_seconds")
                .with_description("pgvector ANN search latency")
                .with_unit("s")
                .build(),
            ingest_documents_total: meter
                .u64_counter("seshat_ingest_documents_total")
                .with_description("Documents ingested via ingest_document")
                .with_unit("document")
                .build(),
            ingest_chunk_count: meter
                .f64_histogram("seshat_ingest_chunk_count")
                .with_description("Chunk count of one ingested document")
                .with_unit("chunk")
                .build(),
            db_pool_active: meter
                .i64_observable_gauge("seshat_db_pool_active")
                .with_description("Active SeaORM pool connections")
                .with_unit("connection")
                .build(),
        }
    }

    /// Return the underlying meter for ad-hoc instruments in later slices.
    pub fn meter(&self) -> &Meter {
        &self.meter
    }

    // ── WI-326 emission helpers ─────────────────────────────────────────
    //
    // All label values flowing through these helpers are `&'static str`
    // consts from `labels::*` — document/chunk/query IDs and raw query text
    // can never become metric labels because these signatures accept no
    // user-derived strings.

    /// Record one MCP tool-call outcome:
    /// `seshat_mcp_requests_total{tool,status}` (WI-326's
    /// `seshat_requests_total{tool,status}` maps onto this catalog counter),
    /// `seshat_mcp_request_duration_seconds{tool}`, and on `status == ERROR`
    /// `seshat_mcp_request_errors_total{tool,error_kind}`.
    pub fn record_tool_call(
        &self,
        tool: &'static str,
        status: &'static str,
        duration_seconds: f64,
        error_kind: Option<&'static str>,
    ) {
        use opentelemetry::KeyValue;
        self.mcp_requests_total.add(
            1,
            &[
                KeyValue::new("tool", tool),
                KeyValue::new("status", status),
            ],
        );
        self.mcp_request_duration_seconds
            .record(duration_seconds, &[KeyValue::new("tool", tool)]);
        if let Some(kind) = error_kind {
            self.mcp_request_errors_total.add(
                1,
                &[
                    KeyValue::new("tool", tool),
                    KeyValue::new("error_kind", kind),
                ],
            );
        }
    }

    /// Record one microtheory cache lookup. `result` is `labels::result::HIT`
    /// or `MISS` — the WI's bounded `result=hit|miss` vocabulary, carried on
    /// the §5.4 catalog's hit/miss counters (no user input in the labels).
    pub fn record_cache_lookup(&self, result: &'static str) {
        use opentelemetry::KeyValue;
        let attrs = [KeyValue::new("result", result)];
        match result {
            r if r == labels::result::HIT => self.wish_cache_hits_total.add(1, &attrs),
            _ => self.wish_cache_misses_total.add(1, &attrs),
        }
    }

    /// Record one embed-worker batch tick: batch counter, duration, and the
    /// chunks processed on success (pass 0 on error legs).
    pub fn record_embed_batch(&self, duration_seconds: f64, chunks_processed: u64) {
        self.embed_batches_total.add(1, &[]);
        self.embed_batch_duration_seconds
            .record(duration_seconds, &[]);
        if chunks_processed > 0 {
            self.embed_chunks_processed_total.add(chunks_processed, &[]);
        }
    }

    /// Record one pgvector similarity-search latency (label-free).
    pub fn record_pgvector_search(&self, duration_seconds: f64) {
        self.pgvector_search_duration_seconds
            .record(duration_seconds, &[]);
    }
}

/// Convenience constructors for bounded label values. These are intentionally
/// narrow so callers cannot accidentally use raw IDs, queries, or text as
/// labels.
pub mod labels {
    /// Allowed values for the `tool` label (MCP tool name — a static string
    /// per `#[tool]` handler). One const per handler currently exposed by
    /// `mcp/mod.rs`; the vocabulary is closed.
    pub mod tool {
        pub const SEARCH: &str = "seshat_search";
        pub const WISH: &str = "seshat_wish";
        pub const STATUS: &str = "seshat_status";
        pub const DOCUMENTS: &str = "seshat_documents";
        pub const INGEST: &str = "seshat_ingest";
        pub const EMBED_NOW: &str = "seshat_embed_now";
    }

    /// Allowed values for the `status` label on `seshat_mcp_requests_total`.
    pub mod status {
        pub const OK: &str = "ok";
        pub const ERROR: &str = "error";
    }

    /// Allowed values for the `result` label on cache lookups
    /// (`wish_cache_hits_total` / `wish_cache_misses_total`).
    pub mod result {
        pub const HIT: &str = "hit";
        pub const MISS: &str = "miss";
    }

    /// Allowed error-kind values.
    pub mod error_kind {
        pub const INTERNAL: &str = "internal";
        pub const TIMEOUT: &str = "timeout";
        pub const VALIDATION: &str = "validation";
        pub const DATABASE: &str = "database";
        pub const NOT_FOUND: &str = "not_found";
    }
}

#[cfg(all(test, feature = "tracing-otel"))]
mod tests {
    use super::*;
    use opentelemetry::metrics::MeterProvider;
    use opentelemetry_sdk::metrics::{ManualReader, SdkMeterProvider};

    #[test]
    fn registry_constructs_all_instruments() {
        let reader = ManualReader::builder().build();
        let provider = SdkMeterProvider::builder().with_reader(reader).build();
        let meter = provider.meter("seshd-test");
        let reg = SeshatMetrics::new(meter);

        // Touch every instrument to ensure names/types/units compile.
        let _ = &reg.mcp_requests_total;
        let _ = &reg.mcp_request_duration_seconds;
        let _ = &reg.mcp_request_errors_total;
        let _ = &reg.search_results;
        let _ = &reg.wish_cache_hits_total;
        let _ = &reg.wish_cache_misses_total;
        let _ = &reg.documents_total;
        let _ = &reg.chunks_total;
        let _ = &reg.embedded_chunks_total;
        let _ = &reg.pending_embedding_chunks;
        let _ = &reg.microtheory_cache_size;
        let _ = &reg.embed_batches_total;
        let _ = &reg.embed_batch_duration_seconds;
        let _ = &reg.embed_chunks_processed_total;
        let _ = &reg.pgvector_search_duration_seconds;
        let _ = &reg.ingest_documents_total;
        let _ = &reg.ingest_chunk_count;
        let _ = &reg.db_pool_active;

        drop(provider);
    }
}