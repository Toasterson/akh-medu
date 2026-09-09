//! WI-325 — turn-level telemetry for the ACP agent-turn path.
//!
//! One parent span (`acp.turn`) per agent turn, plus bounded turn
//! started/completed counters and a wall-duration histogram recorded against
//! the 38j.1 `AkhomedMetrics` instruments. Identity (workspace, session,
//! work item) goes on span **fields only** — never onto metric labels
//! (bounded-label rule in `crate::metrics`).
//!
//! `TurnTelemetry::finish` consumes `self`, so each reachable exit path of
//! the ACP prompt handler closes the turn exactly once, at compile time —
//! matched by every turn recording its duration exactly once.

use std::time::Instant;

/// Bounded stop-reason buckets for turn metrics (WI-325 acceptance).
///
/// The `phase` label on `acp_turns_total` / `acp_turn_duration_seconds`
/// carries these values; `Started` is the begin-side counterpart recorded
/// when the turn opens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnStopReason {
    /// The turn yielded control back to the client cleanly.
    EndTurn,
    /// A `session/cancel` notification stopped the turn at a round boundary.
    Cancelled,
    /// The turn was refused (completion-contract, MCP discovery, watchdog stall).
    Refusal,
    /// Explicit bounded fallback for any outcome outside the three above.
    Other,
}

impl TurnStopReason {
    /// Bounded label value for the `phase` metric label and the span field.
    pub fn as_label(self) -> &'static str {
        match self {
            Self::EndTurn => "end_turn",
            Self::Cancelled => "cancelled",
            Self::Refusal => "refusal",
            Self::Other => "other",
        }
    }
}

/// RAII handle for one ACP turn's telemetry.
///
/// `begin` opens the parent span and records `acp_turns_total{phase=started}`;
/// `finish` records the stop reason + wall duration onto the span and the
/// terminal counter/histogram. `finish(self)` takes ownership, making
/// double-close a compile error — every turn closes exactly one parent span
/// and records duration once.
pub struct TurnTelemetry {
    span: tracing::Span,
    start: Instant,
    /// Workspace alias, retained so the [`Drop`] backstop can record the
    /// fallback outcome against the correct label without the caller
    /// re-supplying it.
    workspace: String,
    /// Whether [`TurnTelemetry::finish`] ran. Guards the drop backstop so
    /// duration is recorded exactly once even under unwind.
    finished: bool,
}

impl TurnTelemetry {
    /// Open a new `acp.turn` parent span.
    ///
    /// * `workspace` — the workspace's short alias (same value the ACP layer
    ///   uses as `agent_id`), never a UUID.
    /// * `session_id` — the ACP session id. Span field only; never a label.
    /// * `is_dispatch` — whether this turn is an autonomous dispatch
    ///   (`mode` = `dispatch|interactive`).
    pub fn begin(workspace: &str, session_id: &str, is_dispatch: bool) -> Self {
        let span = tracing::info_span!(
            "acp.turn",
            workspace = workspace,
            session_id = session_id,
            mode = if is_dispatch { "dispatch" } else { "interactive" },
            stop_reason = tracing::field::Empty,
            duration_secs = tracing::field::Empty,
        );
        record_started(workspace);
        Self {
            span,
            start: Instant::now(),
            workspace: workspace.to_string(),
            finished: false,
        }
    }

    /// A clone of the parent span, for thread-hopping instrumentation: the
    /// ACP worker enters it inside its `spawn_blocking` closure so LLM
    /// request and tool execution spans parent to the turn.
    pub fn span(&self) -> tracing::Span {
        self.span.clone()
    }

    /// Close the turn: record `stop_reason` + `duration_secs` on the span
    /// and emit the terminal counter/duration recordings, exactly once.
    pub fn finish(mut self, workspace: &str, reason: TurnStopReason) {
        self.finish_with(reason);
        let _ = workspace; // workspace is carried on `self`; kept in the signature for call-site clarity.
    }

    /// Shared finish body (does not consume, so [`Drop`] can reuse the path).
    fn finish_with(&mut self, reason: TurnStopReason) {
        if self.finished {
            return;
        }
        self.finished = true;
        let duration = self.start.elapsed().as_secs_f64();
        self.span.record("stop_reason", reason.as_label());
        self.span.record("duration_secs", duration);
        record_completed(&self.workspace, reason, duration);
    }
}

impl Drop for TurnTelemetry {
    /// Backstop: if an exit path forgets (or a panic unwinds past) `finish`,
    /// record the bounded [`TurnStopReason::Other`] outcome + duration so a
    /// turn can never leak an unclosed metric. `finish` still records exactly
    /// once in the normal case via the `finished` guard.
    fn drop(&mut self) {
        if !self.finished {
            self.finish_with(TurnStopReason::Other);
        }
    }
}

/// Core of [`record_started`], split out so tests can drive a
/// `ManualReader`-backed registry directly instead of the `OnceLock` global.
#[cfg(feature = "tracing-otel")]
fn record_started_with(m: &crate::metrics::AkhomedMetrics, workspace: &str) {
    use opentelemetry::KeyValue;
    m.acp_turns_total.add(
        1,
        &[
            KeyValue::new("workspace", workspace.to_string()),
            KeyValue::new("phase", "started"),
        ],
    );
}

/// Record `acp_turns_total{workspace, phase=started}` — no-op when the
/// global registry was never installed (e.g. offline fmt fallback, tests
/// without the `tracing-otel` feature).
#[cfg(feature = "tracing-otel")]
fn record_started(workspace: &str) {
    if let Some(m) = crate::metrics::global() {
        record_started_with(&m, workspace);
    }
}

/// Core of [`record_completed`], split out so tests can drive a
/// `ManualReader`-backed registry and assert the exact label set (workspace
/// alias + bounded phase; never session/work-item identity).
#[cfg(feature = "tracing-otel")]
fn record_completed_with(
    m: &crate::metrics::AkhomedMetrics,
    workspace: &str,
    reason: TurnStopReason,
    duration_secs: f64,
) {
    use opentelemetry::KeyValue;
    m.acp_turns_total.add(
        1,
        &[
            KeyValue::new("workspace", workspace.to_string()),
            KeyValue::new("phase", reason.as_label()),
        ],
    );
    m.acp_turn_duration_seconds.record(
        duration_secs,
        &[
            KeyValue::new("workspace", workspace.to_string()),
            KeyValue::new("phase", reason.as_label()),
        ],
    );
}

/// Record the terminal counter + wall-duration histogram once per turn.
/// `phase` carries the bounded stop reason; identity never becomes a label.
#[cfg(feature = "tracing-otel")]
fn record_completed(workspace: &str, reason: TurnStopReason, duration_secs: f64) {
    if let Some(m) = crate::metrics::global() {
        record_completed_with(&m, workspace, reason, duration_secs);
    }
}

#[cfg(not(feature = "tracing-otel"))]
fn record_started(_workspace: &str) {}

#[cfg(not(feature = "tracing-otel"))]
fn record_completed(_workspace: &str, _reason: TurnStopReason, _duration_secs: f64) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_reason_labels_are_bounded() {
        assert_eq!(TurnStopReason::EndTurn.as_label(), "end_turn");
        assert_eq!(TurnStopReason::Cancelled.as_label(), "cancelled");
        assert_eq!(TurnStopReason::Refusal.as_label(), "refusal");
        assert_eq!(TurnStopReason::Other.as_label(), "other");
    }

    /// A `tracing_subscriber::Layer` that captures span lifecycle for
    /// assertions: created spans (name + recorded fields) and closed spans
    /// (name), keyed by span id.
    mod capture {
        use std::collections::HashMap;
        use std::sync::{Arc, Mutex};
        use tracing::field::{Field, Visit};
        use tracing::span::{Attributes, Id, Record};
        use tracing::Subscriber;
        use tracing_subscriber::layer::{Context, Layer};
        use tracing_subscriber::registry::LookupSpan;

        #[derive(Default, Clone)]
        pub struct Captured {
            /// span_id -> (name, fields)
            pub created: Arc<Mutex<HashMap<u64, (String, HashMap<String, String>)>>>,
            /// span names in close order
            pub closed: Arc<Mutex<Vec<String>>>,
        }

        #[derive(Default)]
        struct FieldVisitor(HashMap<String, String>);
        impl Visit for FieldVisitor {
            fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
                self.0
                    .insert(field.name().to_string(), format!("{value:?}"));
            }
            fn record_str(&mut self, field: &Field, value: &str) {
                self.0.insert(field.name().to_string(), value.to_string());
            }
            fn record_f64(&mut self, field: &Field, value: f64) {
                self.0.insert(field.name().to_string(), value.to_string());
            }
            fn record_i64(&mut self, field: &Field, value: i64) {
                self.0.insert(field.name().to_string(), value.to_string());
            }
            fn record_bool(&mut self, field: &Field, value: bool) {
                self.0.insert(field.name().to_string(), value.to_string());
            }
        }

        impl<S> Layer<S> for Captured
        where
            S: Subscriber + for<'a> LookupSpan<'a>,
        {
            fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, _ctx: Context<'_, S>) {
                let mut v = FieldVisitor::default();
                attrs.record(&mut v);
                self.created
                    .lock()
                    .unwrap()
                    .insert(id.into_u64(), (attrs.metadata().name().to_string(), v.0));
            }
            fn on_record(&self, id: &Id, values: &Record<'_>, _ctx: Context<'_, S>) {
                let mut v = FieldVisitor::default();
                values.record(&mut v);
                if let Some((_, fields)) = self.created.lock().unwrap().get_mut(&id.into_u64()) {
                    fields.extend(v.0);
                }
            }
            fn on_close(&self, id: Id, ctx: Context<'_, S>) {
                if let Some(span) = ctx.span(&id) {
                    self.closed
                        .lock()
                        .unwrap()
                        .push(span.name().to_string());
                }
            }
        }

        impl Captured {
            pub fn span_named(&self, name: &str) -> Vec<HashMap<String, String>> {
                self.created
                    .lock()
                    .unwrap()
                    .values()
                    .filter(|(n, _)| n == name)
                    .map(|(_, f)| f.clone())
                    .collect()
            }
            pub fn closed_named(&self, name: &str) -> usize {
                self.closed
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|n| *n == name)
                    .count()
            }
        }

        pub fn with_capture<F: FnOnce()>(cap: &Captured, f: F) {
            use tracing_subscriber::layer::SubscriberExt;
            use tracing_subscriber::util::SubscriberInitExt;
            let reg = tracing_subscriber::registry().with(cap.clone());
            let _guard = reg.set_default();
            f();
        }
    }

    #[test]
    fn turn_span_opens_and_closes_exactly_once_per_finish() {
        let cap = capture::Captured::default();
        capture::with_capture(&cap, || {
            let telemetry = TurnTelemetry::begin("sekhmet", "sess-123", false);
            // Enter the span so on_close can resolve the name via registry.
            let span = telemetry.span();
            {
                let _g = span.enter();
                // an LLM child span would be created here in production
            }
            telemetry.finish("sekhmet", TurnStopReason::EndTurn);
        });
        // Exactly one acp.turn span created and closed.
        let turns = cap.span_named("acp.turn");
        assert_eq!(turns.len(), 1, "exactly one parent acp.turn span");
        assert_eq!(cap.closed_named("acp.turn"), 1, "acp.turn closed once");
        let fields = &turns[0];
        assert_eq!(fields.get("workspace").map(String::as_str), Some("sekhmet"));
        assert_eq!(fields.get("session_id").map(String::as_str), Some("sess-123"));
        assert_eq!(fields.get("mode").map(String::as_str), Some("interactive"));
        // stop_reason + duration_secs recorded via on_record before close.
        assert_eq!(
            fields.get("stop_reason").map(String::as_str),
            Some("end_turn")
        );
        assert!(fields.contains_key("duration_secs"), "duration recorded once");
    }

    #[test]
    fn drop_without_finish_records_other_fallback() {
        let cap = capture::Captured::default();
        capture::with_capture(&cap, || {
            let telemetry = TurnTelemetry::begin("sekhmet", "sess-9", true);
            let span = telemetry.span();
            let _g = span.enter();
            drop(telemetry); // never finish — unwind path
        });
        let turns = cap.span_named("acp.turn");
        assert_eq!(turns.len(), 1);
        assert_eq!(cap.closed_named("acp.turn"), 1, "closed exactly once");
        // Fallback stop reason recorded by the Drop backstop.
        assert_eq!(
            turns[0].get("stop_reason").map(String::as_str),
            Some("other")
        );
    }

    #[cfg(feature = "tracing-otel")]
    #[test]
    fn metric_labels_carry_only_workspace_alias_and_bounded_phase() {
        // Drive a ManualReader-backed registry and assert the only attribute
        // keys on the WI-325 instruments are `workspace` + bounded `phase` —
        // never session/work-item identity (bounded-label rule). The `mode`
        // value is on the span, not the metric.
        use opentelemetry::metrics::MeterProvider;
        use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData, ResourceMetrics};
        use opentelemetry_sdk::metrics::{ManualReader, SdkMeterProvider};

        let reader = ManualReader::builder().build();
        let provider = SdkMeterProvider::builder()
            .with_reader(reader.clone())
            .build();
        let meter = provider.meter("akh-test");
        let reg = crate::metrics::AkhomedMetrics::new(meter);

        record_started_with(&reg, "sekhmet");
        record_completed_with(&reg, "sekhmet", TurnStopReason::Refusal, 1.25);

        let mut buf = ResourceMetrics::default();
        reader.collect(&mut buf).expect("collect metrics");

        let mut turns_label_keys = Vec::new();
        let mut duration_label_keys = Vec::new();
        for scope in &buf.scope_metrics {
            for metric in &scope.metrics {
                match (&metric.name, &metric.data) {
                    ("akh_acp_turns_total", AggregatedMetrics::U64(MetricData::Sum(sum))) => {
                        turns_label_keys.extend(
                            sum.data_points
                                .iter()
                                .flat_map(|dp| dp.attributes.iter().map(|kv| kv.key.to_string())),
                        );
                    }
                    (
                        "akh_acp_turn_duration_seconds",
                        AggregatedMetrics::F64(MetricData::Histogram(h)),
                    ) => {
                        duration_label_keys.extend(
                            h.data_points
                                .iter()
                                .flat_map(|dp| dp.attributes.iter().map(|kv| kv.key.to_string())),
                        );
                    }
                    _ => {}
                }
            }
        }
        assert!(!turns_label_keys.is_empty(), "acp_turns_total has datapoints");
        assert!(
            !duration_label_keys.is_empty(),
            "duration histogram has datapoints"
        );
        for key in turns_label_keys.iter().chain(duration_label_keys.iter()) {
            assert!(
                key == "workspace" || key == "phase",
                "identity label {key:?} must never reach a metric"
            );
        }
        // `workspace` carries the alias, never a UUID/session/work-item id.
        assert!(turns_label_keys.iter().any(|k| k == "workspace"));
    }
