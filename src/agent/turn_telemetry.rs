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

/// Bounded outcome label for a tool-execution span.
pub(crate) fn tool_outcome_label(success: bool) -> &'static str {
    if success {
        "success"
    } else {
        "failure"
    }
}

/// RAII handle for one tool execution's `acp.tool_execution` child span.
///
/// Created alongside the existing [`ToolExecutionTracker`](crate::agent::tool_execution_tracker::ToolExecutionTracker)
/// guard inside [`ToolRegistry::execute`](crate::agent::tool::ToolRegistry::execute) —
/// the single shared execution hook the structured tool-activity follow-up
/// (`019fbe82-fe37`) already fires from. Creating the span *there* means
/// structured activity and tracing spans share one hook instead of two
/// competing instrumentation points (WI-325 scope).
///
/// The span is entered by the caller around the synchronous `tool.execute(...)`
/// body, parenting it to the ambient `acp.turn` span that `run_turn` enters.
/// `finish(success)` records the bounded `outcome` + wall `duration_secs`;
/// a [`Drop`] backstop records `outcome = "interrupted"` so an unwind past
/// `finish` can never leak an unclosed/unattributed span.
pub(crate) struct ToolExecutionSpan {
    span: tracing::Span,
    start: Instant,
    /// Whether [`ToolExecutionSpan::finish`] ran; guards the drop backstop so
    /// outcome + duration record exactly once.
    finished: bool,
}

impl ToolExecutionSpan {
    /// Open the `acp.tool_execution` span with `tool` set and `outcome` /
    /// `duration_secs` declared `Empty` up front (so tracing-opentelemetry
    /// keeps the late-recorded fields). The caller then enters
    /// [`Self::span`] for the execution body.
    pub(crate) fn begin(tool: &str) -> Self {
        let span = tracing::info_span!(
            "acp.tool_execution",
            tool = tool,
            outcome = tracing::field::Empty,
            duration_secs = tracing::field::Empty,
        );
        Self {
            span,
            start: Instant::now(),
            finished: false,
        }
    }

    /// A clone of the child span, entered by the caller around execution.
    pub(crate) fn span(&self) -> tracing::Span {
        self.span.clone()
    }

    /// Record the final `outcome` and `duration_secs` exactly once.
    pub(crate) fn finish(mut self, success: bool) {
        self.finish_with(tool_outcome_label(success));
    }

    /// Shared finish body (does not consume, so [`Drop`] can reuse the path).
    fn finish_with(&mut self, outcome: &'static str) {
        if self.finished {
            return;
        }
        self.finished = true;
        let duration = self.start.elapsed().as_secs_f64();
        self.span.record("outcome", outcome);
        self.span.record("duration_secs", duration);
    }
}

impl Drop for ToolExecutionSpan {
    /// Backstop: an unwind (or a forgotten `finish`) still records a bounded
    /// `interrupted` outcome + duration, never leaking an unclosed field set.
    fn drop(&mut self) {
        if !self.finished {
            self.finish_with("interrupted");
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
        use opentelemetry_sdk::metrics::data::{Histogram, Sum, ResourceMetrics};
        use opentelemetry_sdk::metrics::reader::MetricReader;
        use opentelemetry_sdk::metrics::{Pipeline, InstrumentKind, Temporality, MetricResult};
        use opentelemetry_sdk::metrics::{ManualReader, SdkMeterProvider};

        #[derive(Debug, Clone)]
        struct SharedReader(std::sync::Arc<ManualReader>);
        impl MetricReader for SharedReader {
            fn register_pipeline(&self, pipeline:std::sync::Weak<Pipeline>){self.0.register_pipeline(pipeline)}
            fn collect(&self, metrics:&mut ResourceMetrics)->MetricResult<()>{self.0.collect(metrics)}
            fn force_flush(&self)->MetricResult<()>{self.0.force_flush()}
            fn shutdown(&self)->MetricResult<()>{self.0.shutdown()}
            fn temporality(&self, kind:InstrumentKind)->Temporality{self.0.temporality(kind)}
        }
        let reader = SharedReader(std::sync::Arc::new(ManualReader::builder().build()));
        let provider = SdkMeterProvider::builder()
            .with_reader(reader.clone())
            .build();
        let meter = provider.meter("akh-test");
        let reg = crate::metrics::AkhomedMetrics::new(meter);

        record_started_with(&reg, "sekhmet");
        record_completed_with(&reg, "sekhmet", TurnStopReason::Refusal, 1.25);

        let mut buf = ResourceMetrics { resource: opentelemetry_sdk::Resource::empty(), scope_metrics: vec![] };
        reader.collect(&mut buf).expect("collect metrics");

        let mut turns_label_keys = Vec::new();
        let mut duration_label_keys = Vec::new();
        for scope in &buf.scope_metrics {
            for metric in &scope.metrics {
                if metric.name == "akh_acp_turns_total" {
                    let sum=metric.data.as_any().downcast_ref::<Sum<u64>>().expect("u64 sum");
                    turns_label_keys.extend(sum.data_points.iter().flat_map(|dp|dp.attributes.iter().map(|kv|kv.key.to_string())));
                } else if metric.name == "akh_acp_turn_duration_seconds" {
                    let histogram=metric.data.as_any().downcast_ref::<Histogram<f64>>().expect("f64 histogram");
                    duration_label_keys.extend(histogram.data_points.iter().flat_map(|dp|dp.attributes.iter().map(|kv|kv.key.to_string())));
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

    #[test]
    fn cancelled_turn_records_cancelled_stop_reason_once() {
        let cap = capture::Captured::default();
        capture::with_capture(&cap, || {
            let telemetry = TurnTelemetry::begin("sekhmet", "sess-cancel", false);
            let span = telemetry.span();
            {
                let _g = span.enter();
            }
            telemetry.finish("sekhmet", TurnStopReason::Cancelled);
        });
        let turns = cap.span_named("acp.turn");
        assert_eq!(turns.len(), 1, "exactly one parent acp.turn span");
        assert_eq!(cap.closed_named("acp.turn"), 1, "acp.turn closed once");
        assert_eq!(
            turns[0].get("stop_reason").map(String::as_str),
            Some("cancelled"),
            "cancellation path records the bounded cancelled stop reason"
        );
        assert!(
            turns[0].contains_key("duration_secs"),
            "duration recorded exactly once"
        );
    }

    #[test]
    fn refusal_turn_records_refusal_stop_reason_once() {
        let cap = capture::Captured::default();
        capture::with_capture(&cap, || {
            let telemetry = TurnTelemetry::begin("sekhmet", "sess-refuse", false);
            let span = telemetry.span();
            {
                let _g = span.enter();
            }
            telemetry.finish("sekhmet", TurnStopReason::Refusal);
        });
        let turns = cap.span_named("acp.turn");
        assert_eq!(turns.len(), 1, "exactly one parent acp.turn span");
        assert_eq!(cap.closed_named("acp.turn"), 1, "acp.turn closed once");
        assert_eq!(
            turns[0].get("stop_reason").map(String::as_str),
            Some("refusal"),
            "refusal path records the bounded refusal stop reason"
        );
        assert!(
            turns[0].contains_key("duration_secs"),
            "duration recorded exactly once"
        );
    }

    /// Multi-request turn: two sequential `acp.llm_request` child spans under
    /// one `acp.turn` parent, with the struct's own accumulated `llm_metrics`
    /// values recorded onto the matching span — and the parent closed exactly
    /// once (never re-closed per request).
    #[test]
    fn multi_request_turn_closes_one_parent_and_records_requests() {
        use crate::agent::llm_metrics::LlmRequestMetrics;
        let cap = capture::Captured::default();
        capture::with_capture(&cap, || {
            let telemetry = TurnTelemetry::begin("sekhmet", "sess-multi", false);
            let turn_span = telemetry.span();
            {
                let _g = turn_span.enter();
                // Request 0 and request 1 (a retry) — the multi-request turn.
                for seq in 0..2usize {
                    let mut m = LlmRequestMetrics::new(seq);
                    let req_span = LlmRequestMetrics::request_span("test-model", seq);
                    m.attach_span(req_span.clone());
                    {
                        let _rg = req_span.enter();
                        m.note_chunk(128);
                        m.note_content_delta();
                        m.set_tokens(10 + seq as u64, 5 + seq as u64);
                    }
                    m.log("test-model", "completed");
                }
            }
            telemetry.finish("sekhmet", TurnStopReason::EndTurn);
        });

        // Exactly one acp.turn, closed once even with two child requests.
        assert_eq!(
            cap.closed_named("acp.turn"),
            1,
            "one parent span closes exactly once across the whole multi-request turn"
        );
        // Two child acp.llm_request spans, each closed once.
        let reqs = cap.span_named("acp.llm_request");
        assert_eq!(reqs.len(), 2, "one child span per LLM request");
        assert_eq!(cap.closed_named("acp.llm_request"), 2);
        // request_seq distinguishes the two requests (0 then 1).
        let mut seqs: Vec<String> = reqs
            .iter()
            .filter_map(|f| f.get("request_seq").cloned())
            .collect();
        seqs.sort();
        assert_eq!(seqs, vec!["0".to_string(), "1".to_string()]);
        // The struct-computed values landed on the matching span (not recomputed).
        for f in &reqs {
            let seq: u64 = f["request_seq"].parse().unwrap();
            assert_eq!(f.get("chunks_received").map(String::as_str), Some("1"));
            assert_eq!(f.get("content_deltas").map(String::as_str), Some("1"));
            assert_eq!(
                f.get("prompt_tokens").map(String::as_str),
                Some((10 + seq).to_string().as_str())
            );
            assert_eq!(f.get("outcome").map(String::as_str), Some("completed"));
            assert!(f.contains_key("duration_secs"));
        }
    }

    /// A tool execution creates exactly one `acp.tool_execution` child span
    /// with `tool`, `outcome`, and `duration_secs`, closed once.
    #[test]
    fn tool_execution_span_records_name_outcome_and_duration_once() {
        let cap = capture::Captured::default();
        capture::with_capture(&cap, || {
            let telemetry = TurnTelemetry::begin("sekhmet", "sess-tool", false);
            let turn_span = telemetry.span();
            {
                let _g = turn_span.enter();
                let exec = ToolExecutionSpan::begin("shell_exec");
                {
                    let _eg = exec.span().entered();
                    // ... tool.execute body would run here ...
                }
                exec.finish(true);
            }
            telemetry.finish("sekhmet", TurnStopReason::EndTurn);
        });

        let tools = cap.span_named("acp.tool_execution");
        assert_eq!(tools.len(), 1, "exactly one acp.tool_execution span");
        assert_eq!(
            cap.closed_named("acp.tool_execution"),
            1,
            "tool execution span closed once"
        );
        let fields = &tools[0];
        assert_eq!(fields.get("tool").map(String::as_str), Some("shell_exec"));
        assert_eq!(fields.get("outcome").map(String::as_str), Some("success"));
        assert!(
            fields.contains_key("duration_secs"),
            "tool execution records duration exactly once"
        );
        // The parent turn still closes exactly once.
        assert_eq!(cap.closed_named("acp.turn"), 1);
    }

    /// Failure outcome path: a failing tool records `outcome = failure`.
    #[test]
    fn tool_execution_span_failure_records_failure_outcome() {
        let cap = capture::Captured::default();
        capture::with_capture(&cap, || {
            let exec = ToolExecutionSpan::begin("edit");
            {
                let _eg = exec.span().entered();
            }
            exec.finish(false);
        });
        let tools = cap.span_named("acp.tool_execution");
        assert_eq!(tools.len(), 1);
        assert_eq!(
            tools[0].get("outcome").map(String::as_str),
            Some("failure")
        );
    }

    /// Unwind backstop: dropping the span without `finish` still records a
    /// bounded `interrupted` outcome and duration exactly once.
    #[test]
    fn tool_execution_span_drop_backstop_records_interrupted() {
        let cap = capture::Captured::default();
        capture::with_capture(&cap, || {
            let exec = ToolExecutionSpan::begin("grep");
            {
                let _eg = exec.span().entered();
                drop(_eg);
            }
            drop(exec); // never finish — unwind path
        });
        let tools = cap.span_named("acp.tool_execution");
        assert_eq!(tools.len(), 1);
        assert_eq!(cap.closed_named("acp.tool_execution"), 1);
        assert_eq!(
            tools[0].get("outcome").map(String::as_str),
            Some("interrupted")
        );
        assert!(tools[0].contains_key("duration_secs"));
    }
}
