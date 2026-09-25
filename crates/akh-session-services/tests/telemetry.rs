use axum::{Router, body::Bytes, routing::post};
use opentelemetry_proto::tonic::collector::{
    logs::v1::ExportLogsServiceRequest, metrics::v1::ExportMetricsServiceRequest,
    trace::v1::ExportTraceServiceRequest,
};
use prost::Message;
use std::sync::{Arc, Mutex};

#[test]
fn exports_correlated_otlp_logs_spans_and_metrics() {
    // This integration test has its own process and sets environment before
    // creating threads; no other tests race the global OTel subscriber.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    unsafe {
        std::env::set_var(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            format!("http://{}", listener.local_addr().unwrap()),
        );
        std::env::remove_var("AKH_OTEL_CAPTURE_CONTENT");
    }
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let captured = Arc::new(Mutex::new(Vec::<(String, Vec<u8>)>::new()));
        let mut app = Router::new();
        for path in ["/v1/traces", "/v1/logs", "/v1/metrics"] {
            let captured = captured.clone();
            app = app.route(
                path,
                post(move |body: Bytes| {
                    captured.lock().unwrap().push((path.into(), body.to_vec()));
                    async {
                        (
                            [("content-type", "application/x-protobuf")],
                            Vec::<u8>::new(),
                        )
                    }
                }),
            );
        }
        let server = tokio::spawn(async {
            axum::serve(tokio::net::TcpListener::from_std(listener).unwrap(), app)
                .await
                .unwrap()
        });
        let telemetry = akh_session_services::telemetry::init("akhrtd", "fixture").unwrap();
        tokio::task::spawn_blocking(|| {
            let turn = tracing::info_span!(
                "runtime.turn",
                session.id = "test-session",
                turn.id = "test-turn"
            );
            let _turn = turn.enter();
            let tool = tracing::info_span!("runtime.tool", tool.name = "read");
            let _tool = tool.enter();
            let mut operation = akh_session_services::telemetry::Operation::new("tool");
            akh_session_services::telemetry::delta("reasoning", "PRIVATE-CONTENT", 3, "paragraph");
            akh_session_services::telemetry::usage(
                &serde_json::json!({"prompt_tokens":17,"completion_tokens":3}),
            );
            operation.success();
        })
        .await
        .unwrap();
        tokio::task::spawn_blocking(move || drop(telemetry))
            .await
            .unwrap();
        let captured = captured.lock().unwrap();
        let mut spans = Vec::new();
        let mut logs = Vec::new();
        let mut metrics = Vec::new();
        for (path, body) in captured.iter() {
            match path.as_str() {
                "/v1/traces" => {
                    for r in ExportTraceServiceRequest::decode(body.as_slice())
                        .unwrap()
                        .resource_spans
                    {
                        for s in r.scope_spans {
                            spans.extend(s.spans);
                        }
                    }
                }
                "/v1/logs" => {
                    for r in ExportLogsServiceRequest::decode(body.as_slice())
                        .unwrap()
                        .resource_logs
                    {
                        for s in r.scope_logs {
                            logs.extend(s.log_records);
                        }
                    }
                }
                "/v1/metrics" => {
                    for r in ExportMetricsServiceRequest::decode(body.as_slice())
                        .unwrap()
                        .resource_metrics
                    {
                        for s in r.scope_metrics {
                            metrics.extend(s.metrics);
                        }
                    }
                }
                _ => unreachable!(),
            }
        }
        let turn = spans.iter().find(|s| s.name == "runtime.turn").unwrap();
        let tool = spans.iter().find(|s| s.name == "runtime.tool").unwrap();
        assert_eq!(tool.parent_span_id, turn.span_id);
        assert_eq!(tool.trace_id, turn.trace_id);
        let delta = logs
            .iter()
            .find(|l| l.attributes.iter().any(|a| a.key == "delta.kind"))
            .unwrap();
        assert_eq!(delta.trace_id, tool.trace_id);
        assert_eq!(delta.span_id, tool.span_id);
        assert!(delta.attributes.iter().any(|a| a.key == "session.id"));
        assert!(delta.attributes.iter().any(|a| a.key == "turn.id"));
        assert!(!format!("{logs:?}").contains("PRIVATE-CONTENT"));
        assert!(
            metrics
                .iter()
                .any(|m| m.name == "akh.runtime.operation.duration")
        );
        assert!(metrics.iter().any(|m| m.name == "akh.runtime.tokens"));
        for name in [
            "akh.runtime.delta.bytes",
            "akh.runtime.delta.chunks",
            "akh.runtime.delta.coalesced",
        ] {
            assert!(metrics.iter().any(|m| m.name == name), "{name}");
        }
        assert!(delta.attributes.iter().any(|a| a.key == "delta.coalesced"));
        assert!(!format!("{metrics:?}").contains("test-session"));
        server.abort();
    });
}
