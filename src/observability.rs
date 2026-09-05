//! Phase 38i — OTLP tracing skeleton.
//!
//! `init_tracing(service_name)` installs a `tracing_subscriber` stack that,
//! when `OTEL_EXPORTER_OTLP_ENDPOINT` is set, layers a
//! `tracing-opentelemetry` OTLP exporter on top of the existing `fmt()`
//! layer. When the env var is unset, it falls back to plain `fmt().init()`
//! behaviour — keeping local-dev offline-friendly (acceptance: "no
//! behavioural change vs today" when unset).
//!
//! Sampling: head sampling 10 % via `parentbased_traceidratio` on akhomed,
//! accept-parent on seshd (seshd is a child of akhomed's trace context —
//! cross-process propagation arrives in Phase 38k). The sampler is
//! env-driven (`OTEL_TRACES_SAMPLER` / `OTEL_TRACES_SAMPLER_ARG`), defaulting
//! to `ParentBased(TraceIdRatioBased(0.1))`.
//!
//! The returned `TracerShutdown` guard flushes the batch span processor on
//! drop. Callers must hold it alive until process exit.

use std::env;
use std::sync::Arc;

use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

/// Tracer shutdown guard. Drop flushes the batch span processor so in-flight
/// spans reach the collector before the process exits.
///
/// Under the `tracing-otel` feature, the inner provider is held in an `Arc`
/// shared with the global provider; on drop we shut it down. Without the
/// feature, this is a zero-sized unit — no OTel deps in the link.
pub struct TracerShutdown {
    #[cfg(feature = "tracing-otel")]
    inner: Option<InnerShutdown>,
}

#[cfg(feature = "tracing-otel")]
struct InnerShutdown {
    /// Shared handle to the tracer provider we installed globally. Cloned
    /// *before* `set_tracer_provider` so the guard flushes the *correct*
    /// provider, not whatever no-op was there before (review round 1 fix).
    provider: Arc<opentelemetry_sdk::trace::TracerProvider>,
}

impl TracerShutdown {
    /// Construct a no-op shutdown (used when OTLP is not configured).
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
            // Shut down the provider to flush the batch span processor so
            // in-flight spans reach the collector before the process exits.
            let _ = inner.provider.shutdown();
        }
    }
}

/// Initialise the tracing subscriber. Returns a `TracerShutdown` guard that
/// the caller must hold alive until process exit.
///
/// - If `OTEL_EXPORTER_OTLP_ENDPOINT` is set → install OTLP exporter +
///   `parentbased_traceidratio(0.1)` sampler + `tracing-opentelemetry` layer
///   on top of the existing `fmt()` layer.
/// - If unset → fall back to today's plain `fmt()` behaviour.
pub fn init_tracing(service_name: &'static str) -> TracerShutdown {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,egg=warn,hnsw_rs=warn"));

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
            // No collector configured → offline-friendly fallback.
            tracing_subscriber::fmt().with_env_filter(filter).init();
            return TracerShutdown::empty();
        }

        match init_otel(service_name, endpoint.unwrap(), filter) {
            Ok(guard) => guard,
            Err(e) => {
                // OTel init failed — fall back to plain fmt rather than crash.
                tracing::warn!(error = %e, "OTLP init failed; falling back to fmt-only");
                tracing_subscriber::fmt()
                    .with_env_filter(
                        EnvFilter::try_from_default_env()
                            .unwrap_or_else(|_| EnvFilter::new("info,egg=warn,hnsw_rs=warn")),
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

    // OTLP gRPC exporter. The endpoint env var is plumbed into the exporter
    // (review round 1 fix — previously used only as a boolean gate and the
    // exporter targeted localhost:4317).
    let exporter = SpanExporter::builder()
        .with_tonic()
        .with_endpoint(&endpoint)
        .with_protocol(Protocol::Grpc)
        .build()?;

    // Env-driven sampler. Defaults to ParentBased(TraceIdRatioBased(0.1)).
    let sampler = build_sampler();

    let provider = TracerProvider::builder()
        .with_batch_exporter(exporter, Tokio)
        .with_sampler(sampler)
        .with_resource(resource)
        .build();

    // Install W3C TraceContext propagator for cross-process propagation (38k).
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
