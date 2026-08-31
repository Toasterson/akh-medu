//! Phase 38i/38j.1 — OTLP observability bootstrap (tracing + metrics).
//!
//! `init_observability(service_name)` installs a `tracing_subscriber` stack and,
//! when `OTEL_EXPORTER_OTLP_ENDPOINT` is set, an OTLP metrics pipeline in
//! addition to the existing `tracing-opentelemetry` layer. When the env var
//! is unset, it falls back to plain `fmt().init()` behaviour — keeping
//! local-dev offline-friendly.
//!
//! Sampling: head sampling 10 % via `parentbased_traceidratio` on akhomed,
//! accept-parent on seshd (seshd is a child of akhomed's trace context —
//! cross-process propagation arrives in Phase 38k). The sampler is
//! env-driven (`OTEL_TRACES_SAMPLER` / `OTEL_TRACES_SAMPLER_ARG`), defaulting
//! to `ParentBased(TraceIdRatioBased(0.1))`.
//!
//! The returned `ObservabilityShutdown` guard flushes both the batch span
//! processor and the periodic metric reader on drop. Callers must hold it alive
//! until process exit.
//!
//! OTLP push only; no Prometheus exporter is installed.

use std::env;
use std::sync::Arc;

use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

/// Combined observability shutdown guard. Drop flushes, in order, the tracer
/// provider then the meter provider so both in-flight spans and the last
/// periodic metrics push reach the collector before the process exits.
///
/// Under the `tracing-otel` feature, the inner providers are held in `Arc`s
/// shared with the global providers; on drop we shut them down. Without the
/// feature, this is a zero-sized unit — no OTel deps in the link.
pub struct ObservabilityShutdown {
    #[cfg(feature = "tracing-otel")]
    inner: Option<InnerShutdown>,
}

#[cfg(feature = "tracing-otel")]
struct InnerShutdown {
    /// Shared handle to the tracer provider we installed globally. Cloned
    /// *before* `set_tracer_provider` so the guard flushes the *correct*
    /// provider, not whatever no-op was there before (review round 1 fix).
    tracer: Arc<opentelemetry_sdk::trace::TracerProvider>,
    /// Shared handle to the meter provider we installed globally. Cloned
    /// before `set_meter_provider` for the same reason.
    meter: Arc<opentelemetry_sdk::metrics::SdkMeterProvider>,
}

impl ObservabilityShutdown {
    /// Construct a no-op shutdown (used when OTLP is not configured).
    #[inline]
    pub fn empty() -> Self {
        Self {
            #[cfg(feature = "tracing-otel")]
            inner: None,
        }
    }
}

impl Drop for ObservabilityShutdown {
    fn drop(&mut self) {
        #[cfg(feature = "tracing-otel")]
        if let Some(inner) = self.inner.take() {
            // Flush traces first, then metrics, so span data is exported
            // before the final metric shutdown.
            let _ = inner.tracer.shutdown();
            let _ = inner.meter.shutdown();
        }
    }
}

/// Initialise the observability subscriber and, when configured, the OTLP
/// metrics pipeline. Returns an `ObservabilityShutdown` guard that the caller
/// must hold alive until process exit.
///
/// - If `OTEL_EXPORTER_OTLP_ENDPOINT` is set → install OTLP span exporter,
///   OTLP metric exporter, W3C propagator, and `tracing-opentelemetry` layer
///   on top of the existing `fmt()` layer.
/// - If unset → fall back to today's plain `fmt()` behaviour.
pub fn init_observability(service_name: &'static str) -> ObservabilityShutdown {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new("info,egg=warn,hnsw_rs=warn")
    });

    #[cfg(not(feature = "tracing-otel"))]
    {
        let _ = service_name;
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .init();
        return ObservabilityShutdown::empty();
    }

    #[cfg(feature = "tracing-otel")]
    {
        let endpoint = env::var("OTEL_EXPORTER_OTLP_ENDPOINT");
        if endpoint.is_err() {
            // No collector configured → offline-friendly fallback.
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .init();
            return ObservabilityShutdown::empty();
        }

        match init_otel(service_name, endpoint.unwrap(), filter) {
            Ok(guard) => guard,
            Err(e) => {
                // OTel init failed — fall back to plain fmt rather than crash.
                tracing::warn!(error = %e, "OTLP init failed; falling back to fmt-only");
                tracing_subscriber::fmt()
                    .with_env_filter(
                        EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                            EnvFilter::new("info,egg=warn,hnsw_rs=warn")
                        }),
                    )
                    .init();
                ObservabilityShutdown::empty()
            }
        }
    }
}

/// Backwards-compatible alias for the tracing-only era. Prefer
/// `init_observability` in new call sites; this alias is deprecated and may be
/// removed once downstream branches have migrated.
#[deprecated(since = "0.7.2", note = "use init_observability instead")]
pub fn init_tracing(service_name: &'static str) -> ObservabilityShutdown {
    init_observability(service_name)
}

#[cfg(feature = "tracing-otel")]
fn init_otel(
    service_name: &'static str,
    endpoint: String,
    filter: EnvFilter,
) -> Result<ObservabilityShutdown, Box<dyn std::error::Error + Send + Sync>> {
    use opentelemetry::{KeyValue, trace::TracerProvider as _};
    use opentelemetry_sdk::{
        Resource,
        propagation::TraceContextPropagator,
        runtime::Tokio,
        trace::TracerProvider,
    };
    use opentelemetry_otlp::{MetricExporter, Protocol, SpanExporter, WithExportConfig};

    // Resource: service.name, service.namespace, deployment.environment,
    // service.instance.id (from env — set via downward API in Helm).
    let mut resource_kv = vec![KeyValue::new(
        opentelemetry_semantic_conventions::resource::SERVICE_NAME,
        service_name.to_string(),
    )];
    if let Ok(ns) = env::var("OTEL_RESOURCE_ATTRIBUTES") {
        for pair in ns.split(',') {
            let mut it = pair.splitn(2, '=');
            if let (Some(k), Some(v)) = (it.next(), it.next()) {
                resource_kv.push(KeyValue::new(k.trim().to_string(), v.trim().to_string()));
            }
        }
    }
    if let Ok(inst) = env::var("OTEL_SERVICE_INSTANCE_ID") {
        resource_kv.push(KeyValue::new(
            opentelemetry_semantic_conventions::resource::SERVICE_INSTANCE_ID,
            inst,
        ));
    }
    let resource = Resource::new(resource_kv);

    // OTLP gRPC span exporter. The endpoint env var is plumbed into the exporter
    // (review round 1 fix — previously used only as a boolean gate and the
    // exporter targeted localhost:4317).
    let span_exporter = SpanExporter::builder()
        .with_tonic()
        .with_endpoint(&endpoint)
        .with_protocol(Protocol::Grpc)
        .build()?;

    // Env-driven sampler. Defaults to ParentBased(TraceIdRatioBased(0.1)).
    let sampler = build_sampler();

    let tracer_provider = TracerProvider::builder()
        .with_batch_exporter(span_exporter, Tokio)
        .with_sampler(sampler)
        .with_resource(resource.clone())
        .build();

    // OTLP gRPC metric exporter using the same endpoint and resource.
    let metric_exporter = MetricExporter::builder()
        .with_tonic()
        .with_endpoint(&endpoint)
        .with_protocol(Protocol::Grpc)
        .build()?;

    let reader = opentelemetry_sdk::metrics::PeriodicReader::builder(metric_exporter, Tokio)
        .with_interval(std::time::Duration::from_secs(60))
        .build();

    let meter_provider = opentelemetry_sdk::metrics::SdkMeterProvider::builder()
        .with_reader(reader)
        .with_resource(resource)
        .build();

    // Install W3C TraceContext propagator for cross-process propagation (38k).
    opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());

    // Get the tracer from the provider before wrapping in Arc.
    let tracer = tracer_provider.tracer(service_name);

    // Install the global providers.
    opentelemetry::global::set_tracer_provider(tracer_provider.clone());
    opentelemetry::global::set_meter_provider(meter_provider.clone());

    // Install the OpenTelemetry layer on top of the fmt layer.
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer())
        .with(otel_layer)
        .init();

    // Keep Arcs to the providers for the shutdown guard.
    let tracer_arc = Arc::new(tracer_provider);
    let meter_arc = Arc::new(meter_provider);

    Ok(ObservabilityShutdown {
        inner: Some(InnerShutdown {
            tracer: tracer_arc,
            meter: meter_arc,
        }),
    })
}

#[cfg(feature = "tracing-otel")]
fn build_sampler() -> opentelemetry_sdk::trace::Sampler {
    use opentelemetry_sdk::trace::Sampler;

    let name = env::var("OTEL_TRACES_SAMPLER").unwrap_or_else(|_| {
        "parentbased_traceidratio".to_string()
    });
    let arg = env::var("OTEL_TRACES_SAMPLER_ARG").unwrap_or_else(|_| "0.1".to_string());
    let ratio: f64 = arg.parse().unwrap_or(0.1);

    match name.as_str() {
        "parentbased_traceidratio" | "parentbased" => {
            Sampler::ParentBased(Box::new(Sampler::TraceIdRatioBased(ratio)))
        }
        "traceidratio" => Sampler::TraceIdRatioBased(ratio),
        "always_on" => Sampler::AlwaysOn,
        "always_off" => Sampler::AlwaysOff,
        _ => Sampler::ParentBased(Box::new(Sampler::TraceIdRatioBased(ratio))),
    }
}

#[cfg(all(test, feature = "tracing-otel"))]
mod tests {
    use super::*;
    use opentelemetry::metrics::MeterProvider;

    /// The offline path must return an empty guard and not panic.
    #[test]
    fn fallback_when_endpoint_unset() {
        let prev = env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok();
        unsafe {
            env::remove_var("OTEL_EXPORTER_OTLP_ENDPOINT");
        }
        // Cannot init a tracing subscriber twice in the same process, but we
        // can still exercise the endpoint gate if a subscriber is already set.
        // This test is therefore a smoke test of env handling: no endpoint
        // means no OTLP inner.
        let _guard = init_observability("akh-test-fallback");
        // No panic and guard is empty.
        assert!(_guard.inner.is_none());
        if let Some(v) = prev {
            unsafe {
                env::set_var("OTEL_EXPORTER_OTLP_ENDPOINT", v);
            }
        }
    }

    /// Resource attributes from the env are parsed into the expected keys.
    #[test]
    fn resource_parsing() {
        let attrs = "deployment.environment=test,service.namespace=unit";
        let mut resource_kv = vec![KeyValue::new(
            opentelemetry_semantic_conventions::resource::SERVICE_NAME,
            "akh-test",
        )];
        for pair in attrs.split(',') {
            let mut it = pair.splitn(2, '=');
            if let (Some(k), Some(v)) = (it.next(), it.next()) {
                resource_kv.push(KeyValue::new(k.trim().to_string(), v.trim().to_string()));
            }
        }
        let resource = opentelemetry_sdk::Resource::new(resource_kv);
        let got: Vec<_> = resource.iter().map(|(k, v)| (k.as_ref().to_string(), v.to_string())).collect();
        assert!(got.contains(&("service.name".to_string(), "akh-test".to_string())));
        assert!(got.contains(&("deployment.environment".to_string(), "test".to_string())));
        assert!(got.contains(&("service.namespace".to_string(), "unit".to_string())));
    }

    /// Instruments can be constructed from a no-op ManualReader provider
    /// without a live collector.
    #[test]
    fn metrics_registry_constructs_offline() {
        use opentelemetry_sdk::metrics::{ManualReader, SdkMeterProvider};

        let reader = ManualReader::builder().build();
        let provider = SdkMeterProvider::builder().with_reader(reader).build();
        let meter = provider.meter("akh-test");

        let _counter = meter.u64_counter("akh_ooda_cycles_total")
            .with_description("OODA cycle count")
            .with_unit("cycle")
            .build();
        let _histogram = meter.f64_histogram("akh_ooda_cycle_duration_seconds")
            .with_description("OODA phase duration")
            .with_unit("s")
            .build();

        // No collector required; just confirm construction succeeds.
        drop(provider);
    }

    use opentelemetry::KeyValue;
}
