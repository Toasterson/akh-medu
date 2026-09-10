//! Exercise the HTTP MCP dispatcher and actual database queries; metrics
//! are collected from the SDK, never emitted by the tests themselves.
#![cfg(feature = "tracing-otel")]

#[path = "support/metrics.rs"]
mod metrics;

use axum::{body::Body, http::Request};
use metrics::{MetricsHarness, assert_bounded_labels, histogram_count, sum_value};
use rmcp::transport::streamable_http_server::{
    StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
};
use sea_orm::{DatabaseBackend, DatabaseConnection, MockDatabase};
use serde_json::{Value, json};
use seshat::mcp::{SeshatMcpServer, SeshatMcpState};
use seshat::metrics::SeshatMetrics;
use std::{collections::BTreeMap, sync::Arc};
use tower::ServiceExt;

async fn call_tool(
    db: DatabaseConnection,
    metrics: &MetricsHarness,
    tool: &str,
    arguments: Value,
) -> Value {
    let state = Arc::new(SeshatMcpState {
        db: Arc::new(db),
        embedder: None,
        metrics: Some(Arc::clone(&metrics.metrics)),
    });
    let server = SeshatMcpServer::new(state);
    let service = StreamableHttpService::new(
        move || Ok(server.clone()),
        Arc::new(LocalSessionManager::default()),
        StreamableHttpServerConfig {
            stateful_mode: false,
            json_response: true,
            ..Default::default()
        },
    );
    let request = Request::builder().method("POST").uri("/mcp")
        .header("content-type", "application/json")
        .header("accept", "application/json, text/event-stream")
        .body(Body::from(json!({"jsonrpc":"2.0", "id":1, "method":"tools/call", "params":{"name":tool,"arguments":arguments}}).to_string())).unwrap();
    let response = axum::Router::new()
        .nest_service("/mcp", service)
        .oneshot(request)
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    serde_json::from_slice(&body).unwrap()
}

#[tokio::test]
async fn mcp_documents_dispatch_records_success_and_failure() {
    let metrics = MetricsHarness::new();
    let ok_db = MockDatabase::new(DatabaseBackend::Postgres)
        .append_query_results([Vec::<seshat::entity::pa_document::Model>::new()])
        .into_connection();
    let ok = call_tool(
        ok_db,
        &metrics,
        "seshat_documents",
        json!({"limit":10,"offset":0}),
    )
    .await;
    assert_eq!(ok["result"]["content"][0]["text"], "[]");
    let err_db = MockDatabase::new(DatabaseBackend::Postgres)
        .append_query_errors([sea_orm::DbErr::Custom("private database failure".into())])
        .into_connection();
    let err = call_tool(
        err_db,
        &metrics,
        "seshat_documents",
        json!({"limit":10,"offset":0}),
    )
    .await;
    assert_eq!(err["error"]["code"], -32603);
    let rm = metrics.collect();
    for status in ["ok", "error"] {
        assert_eq!(
            sum_value(
                &rm,
                "seshat_mcp_requests_total",
                &[("tool", "seshat_documents"), ("status", status)]
            ),
            1
        );
    }
    assert_eq!(
        sum_value(
            &rm,
            "seshat_mcp_request_errors_total",
            &[("tool", "seshat_documents"), ("error_kind", "internal")]
        ),
        1
    );
    assert_eq!(
        histogram_count(&rm, "seshat_mcp_request_duration_seconds"),
        2
    );
    assert_bounded_labels(&rm);
}

#[tokio::test]
async fn wish_dispatch_records_cache_hit_miss_and_lookup_failure() {
    let metrics = MetricsHarness::new();
    let hit_row = seshat::entity::pa_microtheory_cache::Model {
        id: uuid::Uuid::new_v4(),
        query_hash: "hash".into(),
        query_text: "private query".into(),
        compartment_id: "private compartment".into(),
        triples_json: json!([]),
        chunk_ids_used: json!([]),
        model_used: "model".into(),
        confidence: 0.9,
        created_at: chrono::Utc::now(),
        access_count: 3,
        last_accessed_at: chrono::Utc::now(),
    };
    let hit_db = MockDatabase::new(DatabaseBackend::Postgres)
        .append_query_results([vec![hit_row.clone()], vec![hit_row]])
        .into_connection();
    let hit = call_tool(
        hit_db,
        &metrics,
        "seshat_wish",
        json!({"query":"private query"}),
    )
    .await;
    let content: Value = serde_json::from_str(
        hit["result"]["content"][0]["text"]
            .as_str()
            .expect("cached result"),
    )
    .unwrap();
    assert_eq!(content["cache_hit"], true);
    let miss_db = MockDatabase::new(DatabaseBackend::Postgres)
        .append_query_results([Vec::<seshat::entity::pa_microtheory_cache::Model>::new()])
        .into_connection();
    // A miss proceeds to embedding, which is deliberately unconfigured.
    let miss = call_tool(
        miss_db,
        &metrics,
        "seshat_wish",
        json!({"query":"another private query"}),
    )
    .await;
    assert_eq!(miss["error"]["code"], -32603);
    let err_db = MockDatabase::new(DatabaseBackend::Postgres)
        .append_query_errors([sea_orm::DbErr::Custom("cache unavailable".into())])
        .into_connection();
    let err = call_tool(
        err_db,
        &metrics,
        "seshat_wish",
        json!({"query":"third query"}),
    )
    .await;
    assert_eq!(err["error"]["code"], -32603);
    let rm = metrics.collect();
    assert_eq!(
        sum_value(&rm, "seshat_wish_cache_hits_total", &[("result", "hit")]),
        1
    );
    assert_eq!(
        sum_value(&rm, "seshat_wish_cache_misses_total", &[("result", "miss")]),
        1
    );
    assert_eq!(
        sum_value(&rm, "seshat_mcp_requests_total", &[("tool", "seshat_wish")]),
        3
    );
    assert_eq!(
        sum_value(
            &rm,
            "seshat_mcp_request_errors_total",
            &[("tool", "seshat_wish")]
        ),
        2
    );
    assert_bounded_labels(&rm);
}

#[tokio::test]
async fn filtered_search_keeps_parameters_provenance_and_measures_both_outcomes() {
    let metrics = MetricsHarness::new();
    let metadata = json!({"work_item_id":"WI-341","executor":"codex","outcome":"completed"});
    let row = BTreeMap::<&str, sea_orm::Value>::from([
        ("chunk_id", uuid::Uuid::new_v4().into()),
        ("document_id", uuid::Uuid::new_v4().into()),
        ("document_title", "title".into()),
        ("text", "private content".into()),
        ("chunk_index", 0i32.into()),
        ("similarity", 0.9f32.into()),
        ("heading", Option::<String>::None.into()),
        ("document_format", "agent_session_transcript".into()),
        ("document_tags", json!(["WI-341"]).into()),
        ("document_metadata", metadata.clone().into()),
    ]);
    let db = MockDatabase::new(DatabaseBackend::Postgres)
        .append_query_results([[row]])
        .into_connection();
    let filters = seshat::query::SearchFilters {
        work_item_id: Some("WI-341".into()),
        executor: Some("codex'); DROP TABLE pa_documents; --".into()),
        outcome: Some("completed".into()),
    };
    let hits = seshat::query::search_similar_filtered(
        &db,
        &[0.1, 0.2],
        5,
        0.3,
        &filters,
        Some(&metrics.metrics),
    )
    .await
    .unwrap();
    assert_eq!(hits[0].document_metadata, metadata);
    assert_eq!(hits[0].document_tags, json!(["WI-341"]));
    let transactions = db.into_transaction_log();
    let statement = &transactions[0].statements()[0];
    assert!(!statement.sql.contains("DROP TABLE"));
    assert!(statement.sql.contains("d.metadata ->> 'work_item_id' = $3"));
    assert!(
        statement
            .sql
            .contains("LOWER(d.metadata ->> 'executor') = LOWER($4)")
    );
    assert_eq!(statement.values.as_ref().unwrap().0.len(), 6);
    let err_db = MockDatabase::new(DatabaseBackend::Postgres)
        .append_query_errors([sea_orm::DbErr::Custom("vector database unavailable".into())])
        .into_connection();
    assert!(
        seshat::query::search_similar_filtered(
            &err_db,
            &[0.1, 0.2],
            5,
            0.3,
            &filters,
            Some(&metrics.metrics)
        )
        .await
        .is_err()
    );
    let generic_db = MockDatabase::new(DatabaseBackend::Postgres)
        .append_query_results([Vec::<BTreeMap<String, sea_orm::Value>>::new()])
        .into_connection();
    assert!(
        seshat::query::search_similar(&generic_db, &[0.1, 0.2], 5, 0.3, Some(&metrics.metrics))
            .await
            .unwrap()
            .is_empty()
    );
    let rm = metrics.collect();
    assert_eq!(
        histogram_count(&rm, "seshat_pgvector_search_duration_seconds"),
        3
    );
    assert_bounded_labels(&rm);
}
