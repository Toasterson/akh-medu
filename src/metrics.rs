//! Phase 38j.1 — typed metric-instrument registry for the akhomed daemon.
//!
//! This module declares the instruments and bounded label sets expected by
//! WI-34 and the 2026-08-01 ACP turn addendum. It does **not** wire runtime
//! call sites; later slices will import `AkhomedMetrics` and record values.
//!
//! # Naming rules
//!
//! - All counters end in `_total`.
//! - All durations are `f64` histograms named `*_duration_seconds` with unit
//!   `"s"`.
//! - All sizes/counts are `i64` asynchronous gauges or `u64` counters.
//!
//! # Bounded labels only
//!
//! Allowed label keys are static enums/strings. The following are **never**
//! used as label values: workspace IDs, goal IDs, raw queries, raw goals,
//! unbounded text, or anything derived from user input. Workspace *name* is
//! allowed only as the caller-supplied short workspace alias, never as a UUID.

use opentelemetry::metrics::{Counter, Histogram, Meter, ObservableGauge};

/// Root metric registry for the akhomed daemon.
///
/// Construct once at startup via `AkhomedMetrics::new(meter)` and share an
/// `Arc` with the components that will emit in later slices.
pub struct AkhomedMetrics {
    meter: Meter,

    // ── OODA / executive loop ─────────────────────────────────────────────
    /// `akh_ooda_cycles_total` — number of completed OODA cycles.
    ///
    /// Type: `u64` counter. Unit: `"cycle"`. Labels: `workspace`, `outcome`.
    pub ooda_cycles_total: Counter<u64>,

    /// `akh_ooda_cycle_duration_seconds` — wall-clock duration of a full OODA
    /// cycle.
    ///
    /// Type: `f64` histogram. Unit: `"s"`. Labels: `workspace`, `outcome`.
    pub ooda_cycle_duration_seconds: Histogram<f64>,

    /// `akh_active_goals` — number of goals currently in working memory.
    ///
    /// Type: `i64` observable gauge. Unit: `"goal"`. Labels: `workspace`.
    pub active_goals: ObservableGauge<i64>,

    /// `akh_working_memory_size` — number of distinct symbols/entities held in
    /// working memory.
    ///
    /// Type: `i64` observable gauge. Unit: `"symbol"`. Labels: `workspace`.
    pub working_memory_size: ObservableGauge<i64>,

    // ── Tooling / organ passes ────────────────────────────────────────────
    /// `akh_tool_invocations_total` — tool calls dispatched by the daemon.
    ///
    /// Type: `u64` counter. Unit: `"invocation"`. Labels: `workspace`, `tool`.
    pub tool_invocations_total: Counter<u64>,

    /// `akh_tool_duration_seconds` — wall-clock time spent inside a tool.
    ///
    /// Type: `f64` histogram. Unit: `"s"`. Labels: `workspace`, `tool`.
    pub tool_duration_seconds: Histogram<f64>,

    /// `akh_tool_failures_total` — tool calls that returned an error or
    /// violated their contract.
    ///
    /// Type: `u64` counter. Unit: `"failure"`. Labels: `workspace`, `tool`,
    /// `error_kind`.
    pub tool_failures_total: Counter<u64>,

    // ── Goal lifecycle ──────────────────────────────────────────────────────
    /// `akh_goals_activated_total` — goals moved into active pursuit.
    ///
    /// Type: `u64` counter. Unit: `"goal"`. Labels: `workspace`, `kind`.
    pub goals_activated_total: Counter<u64>,

    /// `akh_goals_completed_total` — goals that reached a terminal success
    /// state.
    ///
    /// Type: `u64` counter. Unit: `"goal"`. Labels: `workspace`, `kind`,
    /// `outcome`.
    pub goals_completed_total: Counter<u64>,

    /// `akh_goals_failed_total` — goals that reached a terminal failure state.
    ///
    /// Type: `u64` counter. Unit: `"goal"`. Labels: `workspace`, `kind`,
    /// `error_kind`.
    pub goals_failed_total: Counter<u64>,

    // ── KG / symbolic operations ───────────────────────────────────────────
    /// `akh_kg_entities_total` — number of entities in the workspace KG.
    ///
    /// Type: `i64` observable gauge. Unit: `"entity"`. Labels: `workspace`.
    pub kg_entities_total: ObservableGauge<i64>,

    /// `akh_kg_triples_total` — number of triples in the workspace KG.
    ///
    /// Type: `i64` observable gauge. Unit: `"triple"`. Labels: `workspace`.
    pub kg_triples_total: ObservableGauge<i64>,

    /// `akh_kg_query_duration_seconds` — SPARQL/similarity query latency.
    ///
    /// Type: `f64` histogram. Unit: `"s"`. Labels: `workspace`, `kind`.
    pub kg_query_duration_seconds: Histogram<f64>,

    /// `akh_triples_created_total` — triples asserted by reflection or
    /// mutation.
    ///
    /// Type: `u64` counter. Unit: `"triple"`. Labels: `workspace`, `kind`.
    pub triples_created_total: Counter<u64>,

    // ── Reflection / consolidation ────────────────────────────────────────
    /// `akh_reflection_cycles_total` — completed reflection passes.
    ///
    /// Type: `u64` counter. Unit: `"cycle"`. Labels: `workspace`, `outcome`.
    pub reflection_cycles_total: Counter<u64>,

    /// `akh_reflection_duration_seconds` — reflection pass latency.
    ///
    /// Type: `f64` histogram. Unit: `"s"`. Labels: `workspace`, `outcome`.
    pub reflection_duration_seconds: Histogram<f64>,

    /// `akh_consolidation_cycles_total` — completed consolidation passes.
    ///
    /// Type: `u64` counter. Unit: `"cycle"`. Labels: `workspace`, `outcome`.
    pub consolidation_cycles_total: Counter<u64>,

    /// `akh_consolidation_duration_seconds` — consolidation pass latency.
    ///
    /// Type: `f64` histogram. Unit: `"s"`. Labels: `workspace`, `outcome`.
    pub consolidation_duration_seconds: Histogram<f64>,

    // ── ACP gateway / turn addendum (2026-08-01) ───────────────────────────
    /// `akh_acp_turns_total` — number of ACP request/response turns handled.
    ///
    /// Type: `u64` counter. Unit: `"turn"`. Labels: `workspace`, `phase`.
    pub acp_turns_total: Counter<u64>,

    /// `akh_acp_turn_duration_seconds` — latency of an ACP turn from request
    /// receipt to response send.
    ///
    /// Type: `f64` histogram. Unit: `"s"`. Labels: `workspace`, `phase`.
    pub acp_turn_duration_seconds: Histogram<f64>,

    /// `akh_acp_turn_input_tokens` — number of prompt tokens submitted in a
    /// turn.
    ///
    /// Type: `u64` counter. Unit: `"token"`. Labels: `workspace`, `model`.
    pub acp_turn_input_tokens: Counter<u64>,

    /// `akh_acp_turn_output_tokens` — number of completion tokens returned in a
    /// turn.
    ///
    /// Type: `u64` counter. Unit: `"token"`. Labels: `workspace`, `model`.
    pub acp_turn_output_tokens: Counter<u64>,

    /// `akh_acp_turn_tool_calls_total` — number of native tool/function calls
    /// requested by the model in a turn.
    ///
    /// Type: `u64` counter. Unit: `"call"`. Labels: `workspace`, `model`.
    pub acp_turn_tool_calls_total: Counter<u64>,

    /// `akh_acp_turn_tool_results_total` — number of tool results delivered
    /// back to the model in a turn.
    ///
    /// Type: `u64` counter. Unit: `"result"`. Labels: `workspace`, `model`.
    pub acp_turn_tool_results_total: Counter<u64>,

    /// `akh_acp_turn_errors_total` — turns that ended in an error response.
    ///
    /// Type: `u64` counter. Unit: `"turn"`. Labels: `workspace`, `error_kind`.
    pub acp_turn_errors_total: Counter<u64>,

    /// `akh_acp_turn_retries_total` — retry attempts for a turn.
    ///
    /// Type: `u64` counter. Unit: `"retry"`. Labels: `workspace`, `retry_reason`.
    pub acp_turn_retries_total: Counter<u64>,
}

impl AkhomedMetrics {
    /// Build the registry from the supplied `Meter`.
    pub fn new(meter: Meter) -> Self {
        Self {
            meter: meter.clone(),

            ooda_cycles_total: meter
                .u64_counter("akh_ooda_cycles_total")
                .with_description("Completed OODA cycles")
                .with_unit("cycle")
                .build(),
            ooda_cycle_duration_seconds: meter
                .f64_histogram("akh_ooda_cycle_duration_seconds")
                .with_description("Wall-clock duration of a full OODA cycle")
                .with_unit("s")
                .build(),
            active_goals: meter
                .i64_observable_gauge("akh_active_goals")
                .with_description("Goals currently in working memory")
                .with_unit("goal")
                .build(),
            working_memory_size: meter
                .i64_observable_gauge("akh_working_memory_size")
                .with_description("Symbols/entities held in working memory")
                .with_unit("symbol")
                .build(),

            tool_invocations_total: meter
                .u64_counter("akh_tool_invocations_total")
                .with_description("Tool calls dispatched by the daemon")
                .with_unit("invocation")
                .build(),
            tool_duration_seconds: meter
                .f64_histogram("akh_tool_duration_seconds")
                .with_description("Wall-clock time spent inside a tool")
                .with_unit("s")
                .build(),
            tool_failures_total: meter
                .u64_counter("akh_tool_failures_total")
                .with_description("Tool calls that returned an error")
                .with_unit("failure")
                .build(),

            goals_activated_total: meter
                .u64_counter("akh_goals_activated_total")
                .with_description("Goals moved into active pursuit")
                .with_unit("goal")
                .build(),
            goals_completed_total: meter
                .u64_counter("akh_goals_completed_total")
                .with_description("Goals that reached a terminal success state")
                .with_unit("goal")
                .build(),
            goals_failed_total: meter
                .u64_counter("akh_goals_failed_total")
                .with_description("Goals that reached a terminal failure state")
                .with_unit("goal")
                .build(),

            kg_entities_total: meter
                .i64_observable_gauge("akh_kg_entities_total")
                .with_description("Entities in the workspace KG")
                .with_unit("entity")
                .build(),
            kg_triples_total: meter
                .i64_observable_gauge("akh_kg_triples_total")
                .with_description("Triples in the workspace KG")
                .with_unit("triple")
                .build(),
            kg_query_duration_seconds: meter
                .f64_histogram("akh_kg_query_duration_seconds")
                .with_description("SPARQL/similarity query latency")
                .with_unit("s")
                .build(),
            triples_created_total: meter
                .u64_counter("akh_triples_created_total")
                .with_description("Triples asserted by reflection or mutation")
                .with_unit("triple")
                .build(),

            reflection_cycles_total: meter
                .u64_counter("akh_reflection_cycles_total")
                .with_description("Completed reflection passes")
                .with_unit("cycle")
                .build(),
            reflection_duration_seconds: meter
                .f64_histogram("akh_reflection_duration_seconds")
                .with_description("Reflection pass latency")
                .with_unit("s")
                .build(),
            consolidation_cycles_total: meter
                .u64_counter("akh_consolidation_cycles_total")
                .with_description("Completed consolidation passes")
                .with_unit("cycle")
                .build(),
            consolidation_duration_seconds: meter
                .f64_histogram("akh_consolidation_duration_seconds")
                .with_description("Consolidation pass latency")
                .with_unit("s")
                .build(),

            acp_turns_total: meter
                .u64_counter("akh_acp_turns_total")
                .with_description("ACP request/response turns handled")
                .with_unit("turn")
                .build(),
            acp_turn_duration_seconds: meter
                .f64_histogram("akh_acp_turn_duration_seconds")
                .with_description("Latency of an ACP turn")
                .with_unit("s")
                .build(),
            acp_turn_input_tokens: meter
                .u64_counter("akh_acp_turn_input_tokens")
                .with_description("Prompt tokens submitted in a turn")
                .with_unit("token")
                .build(),
            acp_turn_output_tokens: meter
                .u64_counter("akh_acp_turn_output_tokens")
                .with_description("Completion tokens returned in a turn")
                .with_unit("token")
                .build(),
            acp_turn_tool_calls_total: meter
                .u64_counter("akh_acp_turn_tool_calls_total")
                .with_description("Native tool calls requested by the model")
                .with_unit("call")
                .build(),
            acp_turn_tool_results_total: meter
                .u64_counter("akh_acp_turn_tool_results_total")
                .with_description("Tool results delivered back to the model")
                .with_unit("result")
                .build(),
            acp_turn_errors_total: meter
                .u64_counter("akh_acp_turn_errors_total")
                .with_description("Turns that ended in an error response")
                .with_unit("turn")
                .build(),
            acp_turn_retries_total: meter
                .u64_counter("akh_acp_turn_retries_total")
                .with_description("Retry attempts for a turn")
                .with_unit("retry")
                .build(),
        }
    }

    /// Return the underlying meter for ad-hoc instruments in later slices.
    pub fn meter(&self) -> &Meter {
        &self.meter
    }
}

/// Convenience constructors for bounded label values. These are intentionally
/// narrow so callers cannot accidentally use raw IDs, goals, or queries as
/// labels.
pub mod labels {
    /// Allowed values for the `outcome` label.
    pub mod outcome {
        pub const SUCCESS: &str = "success";
        pub const FAILURE: &str = "failure";
        pub const TIMEOUT: &str = "timeout";
        pub const CANCELLED: &str = "cancelled";
    }

    /// Allowed values for the `phase` label (OODA or ACP turn phase).
    pub mod phase {
        pub const OBSERVE: &str = "observe";
        pub const ORIENT: &str = "orient";
        pub const DECIDE: &str = "decide";
        pub const ACT: &str = "act";
        pub const CHAT: &str = "chat";
        pub const TOOL: &str = "tool";
    }

    /// Allowed values for the `kind` label (goal or KG operation class).
    pub mod kind {
        pub const INFER: &str = "infer";
        pub const SEARCH: &str = "search";
        pub const SPARQL: &str = "sparql";
        pub const REFLECTION: &str = "reflection";
        pub const CONSOLIDATION: &str = "consolidation";
    }

    /// Allowed error-kind values.
    pub mod error_kind {
        pub const INTERNAL: &str = "internal";
        pub const TIMEOUT: &str = "timeout";
        pub const VALIDATION: &str = "validation";
        pub const UPSTREAM: &str = "upstream";
        pub const RATE_LIMIT: &str = "rate_limit";
    }

    /// Allowed retry-reason values.
    pub mod retry_reason {
        pub const UPSTREAM_ERROR: &str = "upstream_error";
        pub const RATE_LIMIT: &str = "rate_limit";
        pub const TIMEOUT: &str = "timeout";
    }
}

#[cfg(all(test, any(feature = "tracing-otel", feature = "metrics")))]
mod tests {
    use super::*;
    use opentelemetry::metrics::MeterProvider;
    use opentelemetry_sdk::metrics::{ManualReader, SdkMeterProvider};

    #[test]
    fn registry_constructs_all_instruments() {
        let reader = ManualReader::builder().build();
        let provider = SdkMeterProvider::builder().with_reader(reader).build();
        let meter = provider.meter("akh-test");
        let reg = AkhomedMetrics::new(meter);

        // Touch every instrument to ensure names/types/units compile.
        let _ = &reg.ooda_cycles_total;
        let _ = &reg.ooda_cycle_duration_seconds;
        let _ = &reg.active_goals;
        let _ = &reg.working_memory_size;
        let _ = &reg.tool_invocations_total;
        let _ = &reg.tool_duration_seconds;
        let _ = &reg.tool_failures_total;
        let _ = &reg.goals_activated_total;
        let _ = &reg.goals_completed_total;
        let _ = &reg.goals_failed_total;
        let _ = &reg.kg_entities_total;
        let _ = &reg.kg_triples_total;
        let _ = &reg.kg_query_duration_seconds;
        let _ = &reg.triples_created_total;
        let _ = &reg.reflection_cycles_total;
        let _ = &reg.reflection_duration_seconds;
        let _ = &reg.consolidation_cycles_total;
        let _ = &reg.consolidation_duration_seconds;
        let _ = &reg.acp_turns_total;
        let _ = &reg.acp_turn_duration_seconds;
        let _ = &reg.acp_turn_input_tokens;
        let _ = &reg.acp_turn_output_tokens;
        let _ = &reg.acp_turn_tool_calls_total;
        let _ = &reg.acp_turn_tool_results_total;
        let _ = &reg.acp_turn_errors_total;
        let _ = &reg.acp_turn_retries_total;

        drop(provider);
    }
}
