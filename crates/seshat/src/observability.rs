//! Phase 38i/38j.1 — OTLP observability bootstrap (tracing + metrics) for seshd.
//!
//! seshd is a separate crate with no dep on the root `akh-medu` crate, so it
//! carries its own copy of `init_observability`. Behaviour mirrors the root
//! module; see `src/observability.rs` for full design notes.
//!
//! OTLP push only; no Prometheus exporter is installed.

use std::env;
use std::sync::Arc;

use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

/// Combined observability shutdown guard. Drop flushes, in order, the tracer
/// provider then the meter provider.
pub struct ObservabilityShutdown {
    #[cfg(feature = "tracing-otel")]
    inner: Option<InnerShutdown>,
}

#[cfg(feature = "tracing-otel")]
struct InnerShutdown {
    tracer: Arc<opentelemetry_sdk::trace::TracerProvider>,
    meter: Arc<opentelemetry_sdk::metrics::SdkMeterProvider>,
}

impl ObservabilityShutdown {
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
            let _ = inner.tracer.shutdown();
            let _ = inner.meter.shutdown();
        }
    }
}

/// Initialise tracing and, when configured, the OTLP metrics pipeline.
/// OTLP layer installed only when `OTEL_EXPORTER_OTLP_ENDPOINT` is set;
/// otherwise plain `fmt()`.
pub fn init_observability(service_name: &'static str) -> ObservabilityShutdown {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "seshat=info".parse().unwrap());

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
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .init();
            return ObservabilityShutdown::empty();
        }

        match init_otel(service_name, endpoint.unwrap(), filter) {
            Ok(guard) => guard,
            Err(e) => {
                tracing::warn!(error = %e, "OTLP init failed; falling back to fmt-only");
                tracing_subscriber::fmt()
                    .with_env_filter(
                        EnvFilter::try_from_default_env()
                            .unwrap_or_else(|_| "seshat=info".parse().unwrap()),
                    )
                    .init();
                ObservabilityShutdown::empty()
            }
        }
    }
}

/// Backwards-compatible alias for the tracing-only era.
#[deprecated(since = "0.1.1", note = "use init_observability instead")]
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

    let mut resource_kv = vec![KeyValue::new(
        opentelemetry_semantic_conventions::resource::SERVICE_NAME,
        service_name.to_string(),
    )];
    if let Ok(attrs) = env::var("OTEL_RESOURCE_ATTRIBUTES") {
        for pair in attrs.split(',') {
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

    let span_exporter = SpanExporter::builder()
        .with_tonic()
        .with_endpoint(&endpoint)
        .with_protocol(Protocol::Grpc)
        .build()?;

    // seshd uses accept-parent sampling (research §8) — default ratio 0.1.
    let sampler = build_sampler();

    let tracer_provider = TracerProvider::builder()
        .with_batch_exporter(span_exporter, Tokio)
        .with_sampler(sampler)
        .with_resource(resource.clone())
        .build();

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

    opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());

    let tracer = tracer_provider.tracer(service_name);

    opentelemetry::global::set_tracer_provider(tracer_provider.clone());
    opentelemetry::global::set_meter_provider(meter_provider.clone());

    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer())
        .with(otel_layer)
        .init();

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

    /// The offline path returns an empty guard.
    #[test]
    fn fallback_when_endpoint_unset() {
        let prev = env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok();
        unsafe {
            env::remove_var("OTEL_EXPORTER_OTLP_ENDPOINT");
        }
        let _guard = init_observability("seshd-test-fallback");
        assert!(_guard.inner.is_none());
        if let Some(v) = prev {
            unsafe {
                env::set_var("OTEL_EXPORTER_OTLP_ENDPOINT", v);
            }
        }
    }

    /// Seshat instruments can be built against a ManualReader provider.
    #[test]
    fn metrics_registry_constructs_offline() {
        use opentelemetry_sdk::metrics::{ManualReader, SdkMeterProvider};

        let reader = ManualReader::builder().build();
        let provider = SdkMeterProvider::builder().with_reader(reader).build();
        let meter = provider.meter("seshd-test");

        let _counter = meter.u64_counter("seshat_requests_total")
            .with_description("Total archive requests")
            .with_unit("request")
            .build();
        let _histogram = meter.f64_histogram("seshat_request_duration_seconds")
            .with_description("Archive request latency")
            .with_unit("s")
            .build();

        drop(provider);
    }
}
