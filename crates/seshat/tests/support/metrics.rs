#![allow(dead_code)]

use std::sync::{Arc, Weak};

use super::SeshatMetrics;
use opentelemetry::metrics::MeterProvider;
use opentelemetry_sdk::metrics::{
    InstrumentKind, ManualReader, MetricResult, SdkMeterProvider, Temporality,
    data::{Gauge, Histogram, ResourceMetrics, Sum},
    reader::MetricReader,
};

#[derive(Clone, Debug)]
struct SharedReader(Arc<ManualReader>);

impl MetricReader for SharedReader {
    fn register_pipeline(&self, pipeline: Weak<opentelemetry_sdk::metrics::Pipeline>) {
        self.0.register_pipeline(pipeline);
    }
    fn collect(&self, metrics: &mut ResourceMetrics) -> MetricResult<()> {
        self.0.collect(metrics)
    }
    fn force_flush(&self) -> MetricResult<()> {
        self.0.force_flush()
    }
    fn shutdown(&self) -> MetricResult<()> {
        self.0.shutdown()
    }
    fn temporality(&self, kind: InstrumentKind) -> Temporality {
        self.0.temporality(kind)
    }
}

pub struct MetricsHarness {
    pub metrics: Arc<SeshatMetrics>,
    reader: SharedReader,
    _provider: SdkMeterProvider,
}

impl MetricsHarness {
    pub fn new() -> Self {
        let reader = SharedReader(Arc::new(ManualReader::builder().build()));
        let provider = SdkMeterProvider::builder()
            .with_reader(reader.clone())
            .build();
        let metrics = Arc::new(SeshatMetrics::new(provider.meter("wi326-test")));
        Self {
            metrics,
            reader,
            _provider: provider,
        }
    }

    pub fn collect(&self) -> ResourceMetrics {
        let mut metrics = ResourceMetrics {
            resource: opentelemetry_sdk::Resource::default(),
            scope_metrics: Vec::new(),
        };
        self.reader.collect(&mut metrics).expect("collect metrics");
        metrics
    }
}

pub fn sum_value(metrics: &ResourceMetrics, name: &str, labels: &[(&str, &str)]) -> u64 {
    metrics
        .scope_metrics
        .iter()
        .flat_map(|scope| &scope.metrics)
        .filter(|metric| metric.name.as_ref() == name)
        .filter_map(|metric| metric.data.as_any().downcast_ref::<Sum<u64>>())
        .flat_map(|sum| &sum.data_points)
        .filter(|point| {
            labels.iter().all(|(key, value)| {
                point
                    .attributes
                    .iter()
                    .any(|attr| attr.key.as_str() == *key && attr.value.as_str() == *value)
            })
        })
        .map(|point| point.value)
        .sum()
}

pub fn histogram_count(metrics: &ResourceMetrics, name: &str) -> u64 {
    metrics
        .scope_metrics
        .iter()
        .flat_map(|scope| &scope.metrics)
        .filter(|metric| metric.name.as_ref() == name)
        .filter_map(|metric| metric.data.as_any().downcast_ref::<Histogram<f64>>())
        .flat_map(|histogram| &histogram.data_points)
        .map(|point| point.count)
        .sum()
}

pub fn assert_bounded_labels(metrics: &ResourceMetrics) {
    for metric in metrics
        .scope_metrics
        .iter()
        .flat_map(|scope| &scope.metrics)
    {
        let check = |attributes: &[opentelemetry::KeyValue]| {
            for attr in attributes {
                let allowed: &[&str] = match attr.key.as_str() {
                    "tool" => &[
                        "seshat_search",
                        "seshat_wish",
                        "seshat_status",
                        "seshat_documents",
                        "seshat_ingest",
                        "seshat_embed_now",
                    ],
                    "status" => &["ok", "error"],
                    "result" => &["hit", "miss"],
                    "error_kind" => &["validation", "not_found", "internal"],
                    _ => panic!("unbounded metric label: {attr:?}"),
                };
                assert!(
                    allowed.contains(&attr.value.as_str().as_ref()),
                    "unbounded metric value: {attr:?}"
                );
            }
        };
        if let Some(sum) = metric.data.as_any().downcast_ref::<Sum<u64>>() {
            for point in &sum.data_points {
                check(&point.attributes);
            }
        }
        if let Some(gauge) = metric.data.as_any().downcast_ref::<Gauge<i64>>() {
            for point in &gauge.data_points {
                check(&point.attributes);
            }
        }
        if let Some(histogram) = metric.data.as_any().downcast_ref::<Histogram<f64>>() {
            for point in &histogram.data_points {
                check(&point.attributes);
            }
        }
    }
}

pub fn gauge_value(metrics: &ResourceMetrics, name: &str) -> Option<i64> {
    metrics
        .scope_metrics
        .iter()
        .flat_map(|scope| &scope.metrics)
        .find(|metric| metric.name.as_ref() == name)
        .and_then(|metric| metric.data.as_any().downcast_ref::<Gauge<i64>>())
        .and_then(|gauge| gauge.data_points.first())
        .map(|point| point.value)
}
