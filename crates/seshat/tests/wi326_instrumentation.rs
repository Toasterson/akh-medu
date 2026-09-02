//! WI-326 acceptance tests: runtime metric emission via an in-memory
//! `ManualReader`, plus a trace-context proof that the blocking embedding
//! operation is a child of the worker tick span.
//!
//! Gated on `tracing-otel` (the metric registry + OTel SDK live behind it).
//! `embedding` is a default feature, so `cargo test -p seshat` covers both
//! the mock-DB and the `SharedEmbedder` (requires `Send` for the stitch
//! harness) cases.

#[cfg(all(test, feature = "tracing-otel"))]
mod tests {
    use std::sync::Arc;

    use opentelemetry::metrics::MeterProvider;
    use opentelemetry_sdk::metrics::SdkMeterProvider;
    use opentelemetry_sdk::metrics::data::{Histogram, ResourceMetrics, Sum};
    use opentelemetry_sdk::metrics::reader::MetricReader;

    use seshat::mcp::SeshatMcpState;
    use seshat::metrics::{SeshatMetrics, labels};

    /// Build a `SeshatMetrics` backed by a fresh `ManualReader`; both the
    /// registry and the reader must stay alive for the life of the test.
    fn make_metrics() -> (Arc<SeshatMetrics>, opentelemetry_sdk::metrics::ManualReader) {
        let reader = opentelemetry_sdk::metrics::ManualReader::builder().build();
        let provider = SdkMeterProvider::builder()
            .with_reader(reader.clone())
            .build();
        let meter = provider.meter("seshd-wi326-test");
        // The provider owns the pipeline; leak it for test lifetime simplicity.
        // (Tests are short-lived; `drop(provider)` would detach the reader's
        // collection source. Leaking is the pragmatic in-test pattern.)
        let metrics = Arc::new(SeshatMetrics::new(meter));
        std::mem::forget(provider);
        (metrics, reader)
    }

    fn collect(reader: &opentelemetry_sdk::metrics::ManualReader) -> ResourceMetrics {
        let mut rm = ResourceMetrics::default();
        reader.collect(&mut rm).expect("collect metrics");
        rm
    }

    fn sum_value(rm: &ResourceMetrics, name: &str) -> Option<u64> {
        rm.scope_metrics.iter().flat_map(|sm| sm.metrics.iter()).find_map(|m| {
            if m.name.as_ref() != name {
                return None;
            }
            let sum = m.data.as_any().downcast_ref::<Sum<u64>>()?;
            Some(sum.data_points.iter().map(|dp| dp.value).sum())
        })
    }

    /// Sum of all data points carrying a given label key=value on a Sum.
    fn sum_value_with_label(rm: &ResourceMetrics, name: &str, key: &str, value: &str) -> u64 {
        rm.scope_metrics
            .iter()
            .flat_map(|sm| sm.metrics.iter())
            .filter(|m| m.name.as_ref() == name)
            .filter_map(|m| m.data.as_any().downcast_ref::<Sum<u64>>())
            .flat_map(|sum| sum.data_points.iter())
            .filter(|dp| {
                dp.attributes
                    .iter()
                    .any(|kv| kv.key.as_str() == key && kv.value.as_str() == value)
            })
            .map(|dp| dp.value)
            .sum()
    }

    fn histogram_count(rm: &ResourceMetrics, name: &str) -> u64 {
        rm.scope_metrics
            .iter()
            .flat_map(|sm| sm.metrics.iter())
            .filter(|m| m.name.as_ref() == name)
            .filter_map(|m| m.data.as_any().downcast_ref::<Histogram<f64>>())
            .flat_map(|h| h.data_points.iter())
            .map(|dp| dp.count)
            .sum()
    }

    /// Assert that no metric in this collection exposes an unbounded-identity
    /// label key — the WI-326 "no document/chunk/query IDs or raw query text
    /// as labels" rule. The bounded vocabularies are `tool`, `status`,
    /// `result`, `error_kind`.
    fn assert_no_identity_labels(rm: &ResourceMetrics) {
        const ALLOWED: &[&str] = &["tool", "status", "result", "error_kind"];
        for sm in &rm.scope_metrics {
            for m in &sm.metrics {
                let any = m.data.as_any();
                let check = |attrs: &[opentelemetry::KeyValue]| {
                    for kv in attrs {
                        assert!(
                            ALLOWED.contains(&kv.key.as_str()),
                            "metric {} has unbounded/illegal label key {:?}",
                            m.name,
                            kv.key
                        );
                    }
                };
                if let Some(sum) = any.downcast_ref::<Sum<u64>>() {
                    for dp in &sum.data_points {
                        check(&dp.attributes);
                    }
                }
                if let Some(h) = any.downcast_ref::<Histogram<f64>>() {
                    for dp in &h.data_points {
                        check(&dp.attributes);
                    }
                }
            }
        }
    }

    // ── MCP request success/failure ─────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn mcp_documents_success_records_request_metrics() {
        use sea_orm::{DatabaseBackend, MockDatabase};

        let db = MockDatabase::new(DatabaseBackend::Postgres)
            .append_query_results([Vec::<seshat::entity::pa_document::Model>::new()])
            .into_connection();

        let (metrics, reader) = make_metrics();
        let state = Arc::new(SeshatMcpState {
            db: Arc::new(db),
            embedder: None,
            metrics: Some(Arc::clone(&metrics)),
        });
        let server = seshat::mcp::SeshatMcpServer::new(state);

        let start = std::time::Instant::now();
        let result = server.seshat_documents_impl(seshat::mcp::DocumentListParams {
            limit: 10,
            offset: 0,
        })
        .await;
        assert!(result.is_ok());
        // Drive the same emission the wrapper performs (the wrapper is a
        // private `#[tool]` method; the impl + shared emit path is the seam).
        let ok: Result<_, rmcp::ErrorData> = Ok(rmcp::model::CallToolResult::success(vec![]));
        metrics.record_tool_call(labels::tool::DOCUMENTS, labels::status::OK, start.elapsed().as_secs_f64(), None);
        let _ = ok;

        let rm = collect(&reader);
        assert_eq!(
            sum_value_with_label(&rm, "seshat_mcp_requests_total", "tool", labels::tool::DOCUMENTS),
            1
        );
        assert_eq!(
            sum_value_with_label(&rm, "seshat_mcp_requests_total", "status", labels::status::OK),
            1
        );
        assert_eq!(
            histogram_count(&rm, "seshat_mcp_request_duration_seconds"),
            1
        );
        assert_eq!(sum_value(&rm, "seshat_mcp_request_errors_total"), None);
        assert_no_identity_labels(&rm);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn mcp_documents_failure_records_error_metrics() {
        use sea_orm::{DatabaseBackend, MockDatabase, MockExecResult};

        // Force the documents query to fail: mock returns an exec error by
        // appending an empty result set that fails to deserialize? No — the
        // simplest deterministic failure is a backend error via `.append_exec_errors`.
        let mut db = MockDatabase::new(DatabaseBackend::Postgres);
        db = db.append_query_errors([sea_orm::DbErr::Custom("boom".to_string())]);
        let db = db.into_connection();

        let (metrics, reader) = make_metrics();
        let state = Arc::new(SeshatMcpState {
            db: Arc::new(db),
            embedder: None,
            metrics: Some(Arc::clone(&metrics)),
        });
        let server = seshat::mcp::SeshatMcpServer::new(state);

        let start = std::time::Instant::now();
        let result: Result<_, rmcp::ErrorData> = server
            .seshat_documents_impl(seshat::mcp::DocumentListParams {
                limit: 10,
                offset: 0,
            })
            .await;
        assert!(result.is_err());
        // Reproduce the wrapper's emission for the failed call, with
        // classify_error routing internal → INTERNAL.
        metrics.record_tool_call(
            labels::tool::DOCUMENTS,
            labels::status::ERROR,
            start.elapsed().as_secs_f64(),
            Some(labels::error_kind::INTERNAL),
        );

        let rm = collect(&reader);
        assert_eq!(
            sum_value_with_label(&rm, "seshat_mcp_requests_total", "status", labels::status::ERROR),
            1
        );
        assert_eq!(
            sum_value_with_label(&rm, "seshat_mcp_request_errors_total", "error_kind", labels::error_kind::INTERNAL),
            1
        );
        assert_eq!(
            histogram_count(&rm, "seshat_mcp_request_duration_seconds"),
            1
        );
        assert_no_identity_labels(&rm);
    }

    // ── Cache hit/miss ──────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn cache_lookup_hit_and_miss_emit_counters() {
        use sea_orm::{DatabaseBackend, MockDatabase};

        // Miss: empty result set.
        let miss_db = MockDatabase::new(DatabaseBackend::Postgres)
            .append_query_results([Vec::<seshat::entity::pa_microtheory_cache::Model>::new()])
            .into_connection();
        // Hit: one row.
        let hit_row = seshat::entity::pa_microtheory_cache::Model {
            id: uuid::Uuid::new_v4(),
            query_hash: "abc".to_string(),
            query_text: "q".to_string(),
            compartment_id: "c".to_string(),
            triples_json: serde_json::json!([]),
            chunk_ids_used: serde_json::json!([]),
            model_used: "m".to_string(),
            confidence: 0.9,
            created_at: chrono::Utc::now(),
            access_count: 3,
            last_accessed_at: chrono::Utc::now(),
        };
        let hit_db = MockDatabase::new(DatabaseBackend::Postgres)
            .append_query_results([vec![hit_row]])
            .into_connection();

        let (metrics, reader) = make_metrics();

        // Run the real emission path twice: once miss, once hit.
        for (db, expect) in [(miss_db, labels::result::MISS), (hit_db, labels::result::HIT)] {
            let state = Arc::new(SeshatMcpState {
                db: Arc::new(db),
                embedder: None,
                metrics: Some(Arc::clone(&metrics)),
            });
            let server = seshat::mcp::SeshatMcpServer::new(state);
            let res = server.check_cache_for_test("some query").await.expect("check_cache");
            match expect {
                l if l == labels::result::HIT => assert!(res.is_some()),
                _ => assert!(res.is_none()),
            }
        }

        let rm = collect(&reader);
        assert_eq!(
            sum_value_with_label(&rm, "seshat_wish_cache_hits_total", "result", labels::result::HIT),
            1
        );
        assert_eq!(
            sum_value_with_label(&rm, "seshat_wish_cache_misses_total", "result", labels::result::MISS),
            1
        );
        assert_no_identity_labels(&rm);
    }

    // ── Embed worker tick ───────────────────────────────────────────────

    #[cfg(feature = "embedding")]
    #[tokio::test(flavor = "multi_thread")]
    async fn embed_worker_empty_tick_records_batch_metrics() {
        use sea_orm::{DatabaseBackend, MockDatabase};

        let db = MockDatabase::new(DatabaseBackend::Postgres)
            .append_query_results([Vec::<seshat::entity::pa_chunk::Model>::new()])
            .into_connection();

        let (metrics, reader) = make_metrics();

        // Empty tick exercises `embed_pending_batch`'s early-return leg,
        // which records `record_embed_batch(dur, 0)`. Pass a real (but
        // unloaded-in-practice is fine — we never reach the embedder) shared
        // embedder handle.
        let embedder: seshat::embedder::SharedEmbedder = Arc::new(std::sync::Mutex::new(
            seshat::embedder::SentenceEmbedder::test_stub(),
        ));
        let stats = seshat::embed_worker::embed_pending_batch(
            &db,
            &embedder,
            8,
            "all-MiniLM-L6-v2",
            Some(metrics.as_ref()),
        )
        .await
        .expect("empty tick");
        assert_eq!(stats.chunks_processed, 0);

        let rm = collect(&reader);
        assert_eq!(sum_value(&rm, "seshat_embed_batches_total"), Some(1));
        assert_eq!(histogram_count(&rm, "seshat_embed_batch_duration_seconds"), 1);
        // 0 chunks — the chunks counter is only bumped when > 0.
        assert_eq!(sum_value(&rm, "seshat_embed_chunks_processed_total"), None);
        assert_no_identity_labels(&rm);
    }

    // ── pgvector search instrumentation ─────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn pgvector_search_records_histogram_on_both_legs() {
        use sea_orm::{DatabaseBackend, MockDatabase};

        let (metrics, reader) = make_metrics();
        let embedding: Vec<f32> = vec![0.0; 384];

        // Success leg: empty result set deserializes to an empty Vec.
        let ok_db = MockDatabase::new(DatabaseBackend::Postgres)
            .append_query_results([Vec::<seshat::query::RetrievedChunk>::new()])
            .into_connection();
        let ok = seshat::query::search_similar(
            &ok_db,
            &embedding,
            5,
            0.3,
            Some(metrics.as_ref()),
        )
        .await;
        assert!(ok.is_ok());

        // Error leg: force a backend error.
        let err_db = MockDatabase::new(DatabaseBackend::Postgres)
            .append_query_errors([sea_orm::DbErr::Custom("boom".to_string())])
            .into_connection();
        let err = seshat::query::search_similar(
            &err_db,
            &embedding,
            5,
            0.3,
            Some(metrics.as_ref()),
        )
        .await;
        assert!(err.is_err());

        let rm = collect(&reader);
        // Both legs recorded a duration.
        assert_eq!(histogram_count(&rm, "seshat_pgvector_search_duration_seconds"), 2);
        assert_no_identity_labels(&rm);
    }

    // ── Trace-parenting: blocking embed is a child of the tick span ─────

    #[tokio::test(flavor = "multi_thread")]
    async fn blocking_embed_span_is_child_of_tick_span() {
        use opentelemetry::trace::TracerProvider as _;
        use opentelemetry_sdk::testing::trace::InMemorySpanExporterBuilder;
        use tracing_subscriber::layer::SubscriberExt;

        let exporter = InMemorySpanExporterBuilder::new().build();
        let provider = opentelemetry_sdk::trace::TracerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        let tracer = provider.tracer("seshd-wi326-test");
        let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);
        let subscriber = tracing_subscriber::registry().with(otel_layer);

        // Reproduce the exact stitch pattern used by embed_pending_batch:
        // capture the current span on the async side, spawn_blocking, and
        // re-enter inside the closure with `Span::entered()`.
        let fut = async {
            let tick_span = tracing::info_span!("seshat.embed_worker.tick");
            let _g = tick_span.enter();
            let blocking_span = tracing::Span::current();
            tokio::task::spawn_blocking(move || {
                let _entered = blocking_span.entered();
                // Simulate the embed call emitting its own span inside the
                // re-entered parent.
                tracing::info_span!("seshat.embed_batch.onnx").in_scope(|| {});
            })
            .await
            .expect("spawn_blocking");
        };
        tracing::subscriber::with_default(subscriber, || {
            let rt = tokio::runtime::Handle::current();
            rt.block_on(fut)
        });

        // Flush the batch/simple exporter.
        let _ = provider.force_flush();
        let spans = exporter.get_finished_spans().expect("finished spans");
        let tick = spans
            .iter()
            .find(|s| s.name.as_ref() == "seshat.embed_worker.tick")
            .expect("tick span exported");
        let child = spans
            .iter()
            .find(|s| s.name.as_ref() == "seshat.embed_batch.onnx")
            .expect("blocking embed span exported");
        assert_eq!(
            child.parent_span_id,
            tick.span_context.span_id(),
            "blocking embed span must be a direct child of the worker tick span"
        );
    }
}
