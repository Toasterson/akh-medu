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
    pub fn finish(self, workspace: &str, reason: TurnStopReason) {
        let duration = self.start.elapsed().as_secs_f64();
        self.span.record("stop_reason", reason.as_label());
        self.span.record("duration_secs", duration);
        record_completed(workspace, reason, duration);
    }
}

/// Record `acp_turns_total{workspace, phase=started}` — no-op when the
/// global registry was never installed (e.g. offline fmt fallback, tests
/// without the `tracing-otel` feature).
#[cfg(feature = "tracing-otel")]
fn record_started(workspace: &str) {
    if let Some(m) = crate::metrics::global() {
        use opentelemetry::KeyValue;
        m.acp_turns_total.add(
            1,
            &[
                KeyValue::new("workspace", workspace.to_string()),
                KeyValue::new("phase", "started"),
            ],
        );
    }
}

/// Record the terminal counter + wall-duration histogram once per turn.
/// `phase` carries the bounded stop reason; identity never becomes a label.
#[cfg(feature = "tracing-otel")]
fn record_completed(workspace: &str, reason: TurnStopReason, duration_secs: f64) {
    if let Some(m) = crate::metrics::global() {
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
}
