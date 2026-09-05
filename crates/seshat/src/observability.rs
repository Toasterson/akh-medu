//! Phase 38i — OTLP tracing skeleton (seshd copy).
//!
//! seshd is a separate crate with no dep on the root `akh-medu` crate, so it
//! carries its own copy of `init_tracing`. Behaviour mirrors the root module;
//! see `src/observability.rs` for the full design notes.

use std::env;
use std::sync::Arc;

use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

/// Tracer shutdown guard. Drop flushes the batch span processor.
pub struct TracerShutdown {
    #[cfg(feature = "tracing-otel")]
    inner: Option<InnerShutdown>,
}

#[cfg(feature = "tracing-otel")]
struct InnerShutdown {
    provider: Arc<opentelemetry_sdk::trace::TracerProvider>,
}

impl TracerShutdown {
    #[inline]
    pub fn empty() -> Self {
        Self {
            #[cfg(feature = "tracing-otel")]
            inner: None,
        }
    }
}

impl Drop for TracerShutdown {
    fn drop(&mut self) {
        #[cfg(feature = "tracing-otel")]
        if let Some(inner) = self.inner.take() {
            let _ = inner.provider.shutdown();
        }
    }
}

/// Initialise tracing. OTLP layer installed only when
/// `OTEL_EXPORTER_OTLP_ENDPOINT` is set; otherwise plain `fmt()`.
pub fn init_tracing(service_name: &'static str) -> TracerShutdown {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| "seshat=info".parse().unwrap());

    #[cfg(not(feature = "tracing-otel"))]
    {
        let _ = service_name;
        tracing_subscriber::fmt().with_env_filter(filter).init();
        return TracerShutdown::empty();
    }

    #[cfg(feature = "tracing-otel")]
    {
        let endpoint = env::var("OTEL_EXPORTER_OTLP_ENDPOINT");
        if endpoint.is_err() {
            tracing_subscriber::fmt().with_env_filter(filter).init();
            return TracerShutdown::empty();
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
                TracerShutdown::empty()
            }
        }
    }
}

#[cfg(feature = "tracing-otel")]
fn init_otel(
    service_name: &'static str,
    endpoint: String,
    filter: EnvFilter,
) -> Result<TracerShutdown, Box<dyn std::error::Error + Send + Sync>> {
    use opentelemetry::{KeyValue, trace::TracerProvider as _};
    use opentelemetry_otlp::{Protocol, SpanExporter, WithExportConfig};
    use opentelemetry_sdk::{
        Resource,
        propagation::TraceContextPropagator,
        runtime::Tokio,
        trace::{Sampler, TracerProvider},
    };

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

    let exporter = SpanExporter::builder()
        .with_tonic()
        .with_endpoint(&endpoint)
        .with_protocol(Protocol::Grpc)
        .build()?;

    // seshd uses accept-parent sampling (research §8) — default ratio 0.1.
    let sampler = build_sampler();

    let provider = TracerProvider::builder()
        .with_batch_exporter(exporter, Tokio)
        .with_sampler(sampler)
        .with_resource(resource)
        .build();

    opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());

    // Get the tracer from the provider before wrapping in Arc.
    let tracer = provider.tracer(service_name);

    // Set the global provider.
    opentelemetry::global::set_tracer_provider(provider.clone());

    // Install the OpenTelemetry layer on top of the fmt layer.
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer())
        .with(otel_layer)
        .init();

    // Keep an Arc to the provider for the shutdown guard.
    let provider_arc = Arc::new(provider);

    Ok(TracerShutdown {
        inner: Some(InnerShutdown {
            provider: provider_arc,
        }),
    })
}

#[cfg(feature = "tracing-otel")]
fn build_sampler() -> opentelemetry_sdk::trace::Sampler {
    use opentelemetry_sdk::trace::Sampler;

    let name =
        env::var("OTEL_TRACES_SAMPLER").unwrap_or_else(|_| "parentbased_traceidratio".to_string());
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
