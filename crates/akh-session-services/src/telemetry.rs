//! OTLP/HTTP export for the independently deployed session services.
//! Session identity belongs on traces/logs, never on metric labels.
use opentelemetry::{KeyValue, trace::TracerProvider as _};
use opentelemetry_otlp::{Protocol, WithExportConfig};
use opentelemetry_sdk::{Resource, runtime::Tokio};
use std::time::Instant;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

pub struct Telemetry {
    traces: Option<opentelemetry_sdk::trace::TracerProvider>,
    logs: Option<opentelemetry_sdk::logs::LoggerProvider>,
    metrics: Option<opentelemetry_sdk::metrics::SdkMeterProvider>,
}

impl Drop for Telemetry {
    fn drop(&mut self) {
        // Called on spawn_blocking at normal shutdown so Tokio can drain the
        // batch workers. Drop is also a best-effort fallback on startup errors.
        let traces = self.traces.take();
        let logs = self.logs.take();
        let metrics = self.metrics.take();
        if let Some(p) = traces {
            let _ = p.shutdown();
        }
        if let Some(p) = logs {
            let _ = p.shutdown();
        }
        if let Some(p) = metrics {
            let _ = p.shutdown();
        }
    }
}

pub fn init(service: &str, instance: &str) -> Result<Telemetry, Box<dyn std::error::Error>> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "info,opentelemetry=warn,hyper=warn,reqwest=warn".into());
    let base = tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr));
    if std::env::var_os("OTEL_EXPORTER_OTLP_ENDPOINT").is_none() {
        base.try_init()?;
        return Ok(Telemetry {
            traces: None,
            logs: None,
            metrics: None,
        });
    }
    // Resource::default includes OTEL_RESOURCE_ATTRIBUTES (site, deployment,
    // etc). Service instance can differ while service.name remains bounded.
    let resource = Resource::default().merge(&Resource::new(vec![
        KeyValue::new("service.name", service.to_owned()),
        KeyValue::new("service.instance.id", instance.to_owned()),
        KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
    ]));
    let traces = opentelemetry_sdk::trace::TracerProvider::builder()
        .with_max_events_per_span(4096)
        .with_resource(resource.clone())
        .with_batch_exporter(
            opentelemetry_otlp::SpanExporter::builder()
                .with_http()
                .with_protocol(Protocol::HttpBinary)
                .build()?,
            Tokio,
        )
        .build();
    let logs = opentelemetry_sdk::logs::LoggerProvider::builder()
        .with_resource(resource.clone())
        .with_batch_exporter(
            opentelemetry_otlp::LogExporter::builder()
                .with_http()
                .with_protocol(Protocol::HttpBinary)
                .build()?,
            Tokio,
        )
        .build();
    let metrics = opentelemetry_sdk::metrics::SdkMeterProvider::builder()
        .with_resource(resource)
        .with_reader(
            opentelemetry_sdk::metrics::PeriodicReader::builder(
                opentelemetry_otlp::MetricExporter::builder()
                    .with_http()
                    .with_protocol(Protocol::HttpBinary)
                    .build()?,
                Tokio,
            )
            .with_interval(std::time::Duration::from_secs(15))
            .build(),
        )
        .build();
    base.with(tracing_opentelemetry::layer().with_tracer(traces.tracer("akh.runtime")))
        .with(CorrelatedLogs::new(&logs))
        .try_init()?;
    opentelemetry::global::set_meter_provider(metrics.clone());
    Ok(Telemetry {
        traces: Some(traces),
        logs: Some(logs),
        metrics: Some(metrics),
    })
}

/// RAII latency measurement also records failures and early returns.
pub struct Operation {
    started: Instant,
    kind: &'static str,
    success: bool,
}
impl Operation {
    pub fn new(kind: &'static str) -> Self {
        Self {
            started: Instant::now(),
            kind,
            success: false,
        }
    }
    pub fn success(&mut self) {
        self.success = true;
    }
}
impl Drop for Operation {
    fn drop(&mut self) {
        let seconds = self.started.elapsed().as_secs_f64();
        tracing::info!(
            event.name = "runtime.operation.completed",
            operation = self.kind,
            success = self.success,
            duration_seconds = seconds,
            "runtime operation completed"
        );
        let meter = opentelemetry::global::meter("akh.runtime");
        let labels = [
            KeyValue::new("operation", self.kind),
            KeyValue::new("success", self.success),
        ];
        meter
            .f64_histogram("akh.runtime.operation.duration")
            .with_unit("s")
            .build()
            .record(seconds, &labels);
        meter
            .u64_counter("akh.runtime.operations")
            .build()
            .add(1, &labels);
    }
}

pub fn delta(kind: &'static str, text: &str) {
    // Opt-in content is capped per event. Tool arguments, credentials and raw
    // provider bodies are never exported here.
    let content = if std::env::var("AKH_OTEL_CAPTURE_CONTENT").as_deref() == Ok("true") {
        text.chars().take(2048).collect::<String>()
    } else {
        String::new()
    };
    tracing::info!(
        event.name = "runtime.delta",
        delta.kind = kind,
        delta.bytes = text.len(),
        delta.content = content,
        "runtime stream delta"
    );
    opentelemetry::global::meter("akh.runtime")
        .u64_counter("akh.runtime.delta.bytes")
        .build()
        .add(text.len() as u64, &[KeyValue::new("kind", kind)]);
}

pub fn usage(value: &serde_json::Value) {
    for (field, kind) in [("prompt_tokens", "input"), ("completion_tokens", "output")] {
        if let Some(count) = value[field].as_u64() {
            tracing::info!(
                event.name = "runtime.tokens",
                token.kind = kind,
                token.count = count,
                "provider token usage"
            );
            opentelemetry::global::meter("akh.runtime")
                .u64_counter("akh.runtime.tokens")
                .build()
                .add(count, &[KeyValue::new("kind", kind)]);
        }
    }
    if let Some(count) = value["completion_tokens_details"]["reasoning_tokens"].as_u64() {
        tracing::info!(
            event.name = "runtime.tokens",
            token.kind = "reasoning",
            token.count = count,
            "provider reasoning usage"
        );
        opentelemetry::global::meter("akh.runtime")
            .u64_counter("akh.runtime.tokens")
            .build()
            .add(count, &[KeyValue::new("kind", "reasoning")]);
    }
}

// The stock tracing log bridge does not carry tracing span fields and only
// reads the thread-local OTel context. Explicitly bridge both so log searches
// by session/turn and Tempo links work even inside spawn_blocking workers.
#[derive(Default, Clone)]
struct Fields(std::collections::BTreeMap<String, opentelemetry::logs::AnyValue>);
impl tracing::field::Visit for Fields {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0
            .insert(field.name().into(), format!("{value:?}").into());
    }
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.0.insert(field.name().into(), value.to_owned().into());
    }
    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.0.insert(
            field.name().into(),
            (value.min(i64::MAX as u64) as i64).into(),
        );
    }
    fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
        self.0.insert(field.name().into(), value.into());
    }
    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        self.0.insert(field.name().into(), value.into());
    }
}
struct CorrelatedLogs(opentelemetry_sdk::logs::Logger);
impl CorrelatedLogs {
    fn new(provider: &opentelemetry_sdk::logs::LoggerProvider) -> Self {
        use opentelemetry::logs::LoggerProvider;
        Self(provider.logger("akh.runtime"))
    }
}
impl<S> tracing_subscriber::Layer<S> for CorrelatedLogs
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        id: &tracing::Id,
        ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut fields = Fields::default();
        attrs.record(&mut fields);
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(fields);
        }
    }
    fn on_record(
        &self,
        id: &tracing::Id,
        values: &tracing::span::Record<'_>,
        ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        if let Some(span) = ctx.span(id) {
            if let Some(fields) = span.extensions_mut().get_mut::<Fields>() {
                values.record(fields);
            }
        }
    }
    fn on_event(&self, event: &tracing::Event<'_>, ctx: tracing_subscriber::layer::Context<'_, S>) {
        use opentelemetry::{
            logs::{LogRecord, Logger, Severity},
            trace::TraceContextExt,
        };
        use tracing_opentelemetry::OpenTelemetrySpanExt;
        // Never export the exporter's own diagnostics back into itself.
        if ["opentelemetry", "hyper", "reqwest"]
            .iter()
            .any(|p| event.metadata().target().starts_with(p))
        {
            return;
        }
        let mut fields = Fields::default();
        if let Some(scope) = ctx.event_scope(event) {
            for span in scope.from_root() {
                if let Some(parent) = span.extensions().get::<Fields>() {
                    fields.0.extend(parent.0.clone());
                }
            }
        }
        event.record(&mut fields);
        let mut record = self.0.create_log_record();
        record.set_target(event.metadata().target().to_owned());
        record.set_severity_text(event.metadata().level().as_str());
        record.set_severity_number(match *event.metadata().level() {
            tracing::Level::ERROR => Severity::Error,
            tracing::Level::WARN => Severity::Warn,
            tracing::Level::INFO => Severity::Info,
            tracing::Level::DEBUG => Severity::Debug,
            _ => Severity::Trace,
        });
        if let Some(body) = fields.0.remove("message") {
            record.set_body(body);
        }
        record.add_attributes(fields.0);
        let context = tracing::Span::current().context();
        let span = context.span();
        let sc = span.span_context();
        if sc.is_valid() {
            record.set_trace_context(sc.trace_id(), sc.span_id(), Some(sc.trace_flags()));
        }
        self.0.emit(record);
    }
}
