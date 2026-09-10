// This module is included by embed_worker so tests execute its private
// inference-injection seam without widening the production API.
#![cfg(all(feature = "embedding", feature = "tracing-otel"))]

use super::*;
use crate::metrics::SeshatMetrics;
#[path = "metrics.rs"]
mod metrics;
use metrics::{MetricsHarness, histogram_count, sum_value};
use sea_orm::{MockDatabase, MockExecResult};

fn chunk() -> pa_chunk::Model {
    pa_chunk::Model {
        id: Uuid::new_v4(),
        document_id: Uuid::new_v4(),
        chunk_index: 0,
        text: "private chunk text".into(),
        word_count: 3,
        chapter: None,
        section: None,
        heading: None,
        embedded_at: None,
    }
}

async fn batches_measure_empty_query_failure_inference_failure_and_write_failure() {
    let metrics = MetricsHarness::new();
    let empty = MockDatabase::new(DatabaseBackend::Postgres)
        .append_query_results([Vec::<pa_chunk::Model>::new()])
        .into_connection();
    assert_eq!(
        embed_pending_batch_with(&empty, 8, "test", Some(&metrics.metrics), |_| panic!(
            "empty batch must not run inference"
        ))
        .await
        .unwrap()
        .chunks_processed,
        0
    );
    let query_error = MockDatabase::new(DatabaseBackend::Postgres)
        .append_query_errors([sea_orm::DbErr::Custom("query failure".into())])
        .into_connection();
    assert!(
        embed_pending_batch_with(&query_error, 8, "test", Some(&metrics.metrics), |_| panic!(
            "failed lookup must not run inference"
        ))
        .await
        .is_err()
    );
    let inference_error = MockDatabase::new(DatabaseBackend::Postgres)
        .append_query_results([vec![chunk()]])
        .into_connection();
    assert!(
        embed_pending_batch_with(
            &inference_error,
            8,
            "test",
            Some(&metrics.metrics),
            |_| Err(crate::SeshatError::Embedding("inference failure".into()))
        )
        .await
        .is_err()
    );
    let write_error = MockDatabase::new(DatabaseBackend::Postgres)
        .append_query_results([vec![chunk()]])
        .append_exec_errors([sea_orm::DbErr::Custom("write failure".into())])
        .into_connection();
    assert!(
        embed_pending_batch_with(&write_error, 8, "test", Some(&metrics.metrics), |_| Ok(
            vec![vec![0.1; 384]]
        ))
        .await
        .is_err()
    );
    let rm = metrics.collect();
    assert_eq!(sum_value(&rm, "seshat_embed_batches_total", &[]), 4);
    assert_eq!(
        histogram_count(&rm, "seshat_embed_batch_duration_seconds"),
        4
    );
    assert_eq!(
        sum_value(&rm, "seshat_embed_chunks_processed_total", &[]),
        0
    );
    metrics::assert_bounded_labels(&rm);
}

type CapturedSpan = (&'static str, tracing::Id, Option<tracing::Id>);

#[derive(Clone, Default)]
struct CapturedSpans(Arc<std::sync::Mutex<Vec<CapturedSpan>>>);

impl<S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>>
    tracing_subscriber::Layer<S> for CapturedSpans
{
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        id: &tracing::Id,
        context: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let parent = attrs.parent().cloned().or_else(|| {
            if attrs.is_contextual() {
                context.current_span().id().cloned()
            } else {
                None
            }
        });
        self.0
            .lock()
            .unwrap()
            .push((attrs.metadata().name(), id.clone(), parent));
    }
}

async fn populated_batch_records_chunks_and_parents_actual_blocking_inference() {
    use tracing::Instrument;
    use tracing::instrument::WithSubscriber;
    use tracing_subscriber::prelude::*;
    let metrics = MetricsHarness::new();
    let row = chunk();
    let db = MockDatabase::new(DatabaseBackend::Postgres)
        .append_query_results([vec![row.clone()], vec![row]])
        .append_exec_results([MockExecResult {
            last_insert_id: 0,
            rows_affected: 1,
        }])
        .into_connection();
    let captured = CapturedSpans::default();
    let subscriber = tracing_subscriber::registry().with(captured.clone());
    let stats = async {
        embed_pending_batch_with(&db, 8, "test", Some(&metrics.metrics), |texts| {
            assert_eq!(texts, ["private chunk text"]);
            assert_eq!(
                tracing::Span::current().metadata().unwrap().name(),
                "seshat.embed_batch.onnx"
            );
            Ok(vec![vec![0.1; 384]])
        })
        .instrument(tracing::info_span!("seshat.embed_worker.tick"))
        .await
    }
    .with_subscriber(subscriber)
    .await
    .unwrap();
    assert_eq!(stats.chunks_processed, 1);
    assert_eq!(stats.embeddings_created, 1);
    let spans = captured.0.lock().unwrap();
    let tick = spans
        .iter()
        .find(|span| span.0 == "seshat.embed_worker.tick")
        .unwrap();
    let inference = spans
        .iter()
        .find(|span| span.0 == "seshat.embed_batch.onnx")
        .unwrap();
    assert_eq!(inference.2.as_ref(), Some(&tick.1));
    let rm = metrics.collect();
    assert_eq!(sum_value(&rm, "seshat_embed_batches_total", &[]), 1);
    assert_eq!(
        sum_value(&rm, "seshat_embed_chunks_processed_total", &[]),
        1
    );
    assert_eq!(
        histogram_count(&rm, "seshat_embed_batch_duration_seconds"),
        1
    );
}

fn pending_count(count: i64) -> std::collections::BTreeMap<&'static str, sea_orm::Value> {
    std::collections::BTreeMap::from([("num_items", count.into())])
}

async fn backlog_observes_total_after_error_and_zero_after_drain() {
    let metrics = MetricsHarness::new();
    let failed = MockDatabase::new(DatabaseBackend::Postgres)
        .append_query_results([vec![chunk()]])
        .append_query_results([[pending_count(12)]])
        .into_connection();
    assert!(
        embed_pending_batch_with(&failed, 1, "test", Some(&metrics.metrics), |_| Err(
            crate::SeshatError::Embedding("inference failure".into())
        ))
        .await
        .is_err()
    );
    assert_eq!(
        metrics::gauge_value(&metrics.collect(), "seshat_pending_embedding_chunks"),
        Some(12)
    );
    let transaction_log = failed.into_transaction_log();
    let count_sql = &transaction_log[1].statements()[0].sql;
    assert!(count_sql.contains("COUNT(*)"));
    assert!(!count_sql.contains("LIMIT"));
    let row = chunk();
    let drained = MockDatabase::new(DatabaseBackend::Postgres)
        .append_query_results([vec![row.clone()], vec![row]])
        .append_query_results([[pending_count(0)]])
        .append_exec_results([MockExecResult {
            last_insert_id: 0,
            rows_affected: 1,
        }])
        .into_connection();
    embed_pending_batch_with(&drained, 1, "test", Some(&metrics.metrics), |_| {
        Ok(vec![vec![0.1; 384]])
    })
    .await
    .unwrap();
    assert_eq!(
        metrics::gauge_value(&metrics.collect(), "seshat_pending_embedding_chunks"),
        Some(0)
    );
    metrics::assert_bounded_labels(&metrics.collect());
}

// The tracing callsite is shared by all inference cases. Keep those cases
// together so tests with and without a scoped subscriber cannot race while
// that callsite is first registered.
#[tokio::test]
async fn worker_instrumentation_covers_outcomes_backlog_and_trace_context() {
    batches_measure_empty_query_failure_inference_failure_and_write_failure().await;
    backlog_observes_total_after_error_and_zero_after_drain().await;
    populated_batch_records_chunks_and_parents_actual_blocking_inference().await;
}
