# OTLP Observability Research — akh-medu

- **Date:** 2026-04-26
- **Status:** Research draft (no implementation)
- **Scope:** akh-medu Rust workspace (`akhomed` daemon + `seshd` Seshat MCP server)
- **Target stack:** Talos cluster `talos.wegmueller.it`, Tempo (traces) / Loki (logs) / Prometheus or Mimir (metrics) via OpenTelemetry Collector
- **Author:** Claude (research agent)

---

## 1. Goal & scope

Make the akh-medu workspace observable end-to-end via the OpenTelemetry Protocol (OTLP). Both binaries — the synchronous OODA-loop daemon `akhomed` (`src/bin/akhomed.rs`) and the async Seshat MCP server `seshd` (`crates/seshat/src/main.rs`) — should emit traces, metrics, and (eventually) logs to an in-cluster OpenTelemetry Collector that fans out to **Tempo** (traces), **Loki** (logs) and **Prometheus / Mimir** (metrics) on the existing Grafana stack at `talos.wegmueller.it`. The goal is not "instrument everything" but "make the agent's interesting moments — OODA cycles, tool calls, sleep/consolidation, MCP requests, embed worker batches, pgvector searches, microtheory cache hits — show up as spans and metrics so the operator can debug a relocatable Akh mid-WAL-flush." Logs stay on stdout for the first slice (Loki scrapes via Promtail / Grafana Alloy) and only migrate to the OTel logs API if/when we want trace-correlated logs. No OpenTelemetry-native code exists in the repo today (verified with `rg 'opentelemetry|otlp'` — no hits outside `target/`), so this is greenfield. Implementation is split into three sub-phases (38i / 38j / 38k); this document is the research backing for the phase ticket, not the implementation.

## 2. Library picks (with rationale)

The OpenTelemetry Rust SDK is **famously version-fragile**: `opentelemetry`, `opentelemetry_sdk`, `opentelemetry-otlp`, and `tracing-opentelemetry` all need to release in lock-step, and minor mismatches surface as cryptic `Resource` or `Tracer` trait errors. This is the single biggest risk in the project. Mitigations:

1. Pin **exact** versions in `Cargo.toml` (`= "0.27.x"`, no caret).
2. Renovate-bot any OTel updates as a single PR that bumps every crate together — never partial.
3. Run `cargo tree -p opentelemetry` after every bump to make sure no transitive dependency drags in two majors.

### 2.1 Tracing

**Pick: `opentelemetry` + `opentelemetry_sdk` + `opentelemetry-otlp` + `tracing-opentelemetry`** (the canonical stack), not the `tracing-opentelemetry-instrumentation-sdk` wrapper. Reasons:

- The project already standardises on `tracing` 0.1 (declared in the workspace `Cargo.toml` line 76) and uses `tracing_subscriber::fmt()` in all three binaries (`src/main.rs:796`, `src/bin/akhomed.rs:3337`, `crates/seshat/src/main.rs:19`). `tracing-opentelemetry` is a `Layer` that plugs straight into the existing `Registry` — no source-level upheaval.
- `tracing-opentelemetry-instrumentation-sdk` is opinionated about HTTP-server middleware (axum, tower) and adds extra deps we don't need. Since `akhomed` and `seshd` only have a small number of axum / rmcp surface points, manual `#[tracing::instrument]` attributes give us cleaner control over span names, kinds and the OODA-loop attributes we actually care about (`cycle_count`, `goal_id`, `tool_name`).
- We need to instrument **synchronous** code (the OODA loop, see `src/agent/ooda.rs:130`), and `tracing-opentelemetry` works identically in sync and async contexts via spans on a thread-local stack — no async runtime required. Important because the OODA loop is *deliberately* synchronous (per `CLAUDE.md`: "the agent OODA loop is synchronous").

Crates and features:

```toml
opentelemetry         = "=0.27"                                  # API only
opentelemetry_sdk     = { version = "=0.27", features = ["rt-tokio"] }
opentelemetry-otlp    = { version = "=0.27", features = ["grpc-tonic", "trace", "metrics", "logs"] }
tracing-opentelemetry = "=0.28"                                  # versioned 1 ahead of opentelemetry
opentelemetry-semantic-conventions = "=0.27"
```

Notes:

- `grpc-tonic` is the OTLP transport. The Tempo / Mimir / Loki collectors in the cluster all accept OTLP/gRPC on `:4317`. The HTTP/protobuf transport (`http-proto`) is also viable but has a chunkier wire footprint.
- `rt-tokio` is needed for the **async batch span processor** that buffers + flushes to the collector — this is fine because both binaries already have a Tokio runtime (`#[tokio::main]` at `src/bin/akhomed.rs:3335` and `crates/seshat/src/main.rs:16`). For the synchronous OODA loop we still emit spans synchronously; the async batch processor only handles the *export* side.
- `opentelemetry-semantic-conventions` gives us pre-defined attribute keys (`service.name`, `service.version`, etc.) so we don't typo them.

### 2.2 Metrics

**Pick: `opentelemetry_sdk::metrics` directly** (not the `metrics` + `metrics-exporter-opentelemetry` bridge).

Rationale:

- The `metrics` facade crate is great when you've already standardised on `metrics::counter!()` etc. We haven't. Adding both `metrics` and `opentelemetry` as parallel APIs would be confusing — two macros to remember, two registries to debug.
- The OTel SDK's `Meter::u64_counter()` / `f64_histogram()` API is verbose but explicit, and it lets us share the same `Resource` (service.name, service.instance.id) between traces and metrics.
- Phase 38j-onwards we may want **exemplars** that link a histogram bucket back to a representative trace ID. Only the native OTel SDK supports that today; the `metrics`-bridge does not (as of 2026-Q1).
- Counter / gauge / histogram primitives map 1:1 to what we already track informally (cycle counts, WM size, embed batch latency).

If we later decide to expose Prometheus pull-style scraping (i.e. let Mimir scrape `seshd:9090/metrics` directly instead of going via the Collector), we can add `opentelemetry-prometheus` as a second exporter. Both can run side by side.

```toml
opentelemetry-otlp    = { version = "=0.27", features = ["grpc-tonic", "metrics"] }   # already pulled in for traces
# No additional crate — Meter is in opentelemetry_sdk::metrics.
```

### 2.3 Logs

**Recommendation for the first slice: keep stdout JSON logs and let Loki scrape via the existing Grafana Alloy / Promtail DaemonSet on the cluster.** Reasons:

- The existing logging story is `tracing_subscriber::fmt()` writing human-readable lines. Switching to a JSON formatter (`tracing_subscriber::fmt::json()`) gives Loki structured fields *for free* with no SDK changes.
- The OTel **logs** API (`opentelemetry-otlp` `logs` feature + `opentelemetry-appender-tracing` bridge) is the newest and least-stable part of the OTel-Rust ecosystem. As of 0.27.x it works but has had breaking changes between every minor.
- Trace-correlated log lines can be achieved cheaply by adding `trace_id` / `span_id` as `tracing` fields via `tracing_opentelemetry::OpenTelemetrySpanExt` — Loki indexes them as labels and Grafana can pivot from a Tempo trace to its log lines via `derived_fields`.

Phase 38k can revisit and add `opentelemetry-appender-tracing` if the operator decides exemplar-quality log correlation is worth the version churn.

```toml
# Phase 38i — keep what we have:
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }   # add "json"

# Phase 38k (in scope per Q3 resolution — OTel logs API → Alloy):
# opentelemetry-appender-tracing = "=0.27"
```

### 2.4 Version compatibility matrix

| Crate                                | Pin       | Notes                                                                                       |
| ------------------------------------ | --------- | ------------------------------------------------------------------------------------------- |
| `opentelemetry`                      | `=0.27`   | API surface; bumped together with SDK.                                                      |
| `opentelemetry_sdk`                  | `=0.27`   | Implements traces / metrics / logs SDKs. Needs `rt-tokio` feature for batch export.         |
| `opentelemetry-otlp`                 | `=0.27`   | Exporter. Features: `grpc-tonic`, `trace`, `metrics`, optionally `logs`.                    |
| `tracing-opentelemetry`              | `=0.28`   | **Always one minor ahead** of `opentelemetry` (per crate's release notes — 0.27 ↔ 0.28).    |
| `opentelemetry-semantic-conventions` | `=0.27`   | String constants only — pure-data crate, low risk.                                          |
| `tonic`                              | (transitive) | Pulled in by `grpc-tonic`. Must not conflict with `axum` (0.8 → tonic 0.12 OK).          |
| `tracing` / `tracing-subscriber`     | `0.1` / `0.3` | Already in use; add `json` feature.                                                     |

The Phase 26 ML deps (`candle-core` 0.10, `ort` 2.0.0-rc.11, `llama-cpp-2`) and Phase 37 deps (`sea-orm` 1.x, `rmcp` 0.17) do **not** transitively touch `opentelemetry`, so the OTel sub-tree is isolated. Verify with `cargo tree -p opentelemetry --invert` after adding.

### 2.5 Recommendation summary

| Concern  | Crate(s)                                                | Phase  |
| -------- | ------------------------------------------------------- | ------ |
| Traces   | `opentelemetry-otlp` + `tracing-opentelemetry`          | 38i    |
| Metrics  | `opentelemetry_sdk::metrics` + `opentelemetry-otlp`     | 38j    |
| Logs     | `tracing_subscriber::fmt::json()` (stdout → Loki)       | 38i    |
| Logs (OTel-native, optional) | `opentelemetry-appender-tracing`        | 38k (deferred) |

## 3. Existing logging story in this repo

### 3.1 Daemon entrypoint (`akhomed`)

`src/bin/akhomed.rs:3335-3343` — Tokio-driven `main`, initialises a `tracing_subscriber::fmt()` subscriber with an `EnvFilter`:

```rust
#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                tracing_subscriber::EnvFilter::new("info,egg=warn,hnsw_rs=warn")
            }),
        )
        .init();
    ...
}
```

The subscriber is the default human-readable one (no JSON, no file). The Helm values pass `RUST_LOG: "info"` (`charts/akh-medu/values.yaml:42`). `tracing::info!`/`warn!`/`error!` calls are scattered through the daemon — typical hot paths in this binary alone:

- `src/bin/akhomed.rs:175` — shared agent created
- `src/bin/akhomed.rs:218` — NLU pipeline created
- `src/bin/akhomed.rs:1434` — daemon task failure
- `src/bin/akhomed.rs:3853-3909` — server start/shutdown lifecycle
- `src/agent/daemon.rs:300, 332, 357, 381, 403, …` — every background-task method has structured `tracing::info!`/`warn!` lines (one per scheduled tick)

### 3.2 Seshat (`seshd`) entrypoint

`crates/seshat/src/main.rs:16-24` — same idiom:

```rust
#[tokio::main]
async fn main() -> miette::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "seshat=info".parse().unwrap()),
        )
        .init();
    ...
}
```

Hot paths:

- `crates/seshat/src/main.rs:46, 60, 68, 76, 88, 92, 97, 114, 147, 157` — startup lifecycle and migration progress
- `crates/seshat/src/embed_worker.rs:38-65` — embed worker batch progress (`embed batch complete`, `embed worker error`)

### 3.3 Other binaries (TUI etc.)

`src/main.rs:784-803` is the third entry point — the `akh` CLI / TUI binary (declared in the workspace `Cargo.toml` lines 159-161). Its `tracing_subscriber::fmt().init()` is identical to `akhomed`'s. Because the TUI lives in the same binary as the CLI dispatcher, instrumenting it is essentially free once `akhomed` is wired up — they share the same `lib.rs` (`src/lib.rs`).

### 3.4 Subscriber configuration

All three binaries use a global `tracing_subscriber::fmt()` and call `.init()` (which sets the *global* dispatcher). To layer in `tracing-opentelemetry` we'll need to switch from the convenience `.fmt().init()` chain to an explicit `Registry::default().with(fmt_layer).with(otel_layer).init()` pattern. This is a small refactor — about 10 lines per binary.

The default env-filters are repo-tuned to silence two noisy deps (`egg=warn`, `hnsw_rs=warn`). We will preserve those when we rewrite the subscriber wiring.

### 3.5 Gaps

- **No `#[tracing::instrument]` attributes anywhere.** `rg '#\[instrument' src/ crates/` returns zero hits. Every existing log line is a flat `tracing::info!(...)` — there are no spans, so today's traces would be a flat sea of events with no parent / child structure.
- **No structured field convention.** Some lines use `error = %e`, others use `"error: {e}"`-formatted strings. We should standardise on field-style (`tracing::warn!(error = %e, "embed worker error")`) so OTel attributes are populated cleanly. The Seshat code is mostly already there; the main repo is mixed.
- **No JSON / OTLP exporter.** Today log lines go to stdout in human-readable form; the Helm `Recreate` strategy means container logs are scraped by whatever runs on the node. We do not know if Promtail / Alloy is currently picking them up — verify against the Talos cluster (open question 3).
- **No correlation IDs.** A user request that spans `akh CLI → akhomed HTTP → akhomed agent OODA → seshd MCP → pgvector` cannot be reconstructed from existing logs.

## 4. Instrumentation surface area

### 4.1 `akhomed` — synchronous OODA agent

The interesting work happens inside one `Arc<Mutex<Agent>>` per workspace, scheduled by `AgentDaemon` (`src/agent/daemon.rs:117-260`). The daemon's main loop is a `tokio::select!` on 14 timer intervals (`daemon.rs:188-244`) — this is the natural top-level span boundary.

Concrete entry points worth a span:

| # | Span name (proposed)            | Code location                                 | Notes                                                                                                       |
|---|---------------------------------|-----------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| 1 | `agent.run`                     | `src/agent/daemon.rs:159` (`run`)             | Root span for the daemon; carries `workspace` attribute.                                                    |
| 2 | `agent.ooda_cycle`              | `src/agent/ooda.rs:130` (`run_ooda_cycle`)    | One span per cycle. Attributes: `cycle_count`, `active_goals`, `wm_size`, `chosen_tool`, `goal_progress`.   |
| 3 | `agent.observe`                 | `src/agent/ooda.rs:156` (`observe`)           | Child of (2). Records JITIR query stats.                                                                    |
| 4 | `agent.orient`                  | `src/agent/ooda.rs:234` (`orient`)            | Records `relevant_triples`, `inferences`, `memory_pressure`.                                                |
| 5 | `agent.decide`                  | `src/agent/ooda.rs:300` (`decide`)            | Records `chosen_tool`, `tool_score`, surfaces impasses as span events.                                      |
| 6 | `agent.act`                     | `src/agent/ooda.rs:1315` (`act`)              | Wraps tool dispatch. Use `span.record_error()` on failures so Tempo's "errors" tab lights up.               |
| 7 | `tool.execute`                  | `src/agent/tool.rs:182` (`ToolRegistry::execute`) | One span per tool invocation. Attribute: `tool_name`. This is the one most likely to dominate trace volume — sample aggressively. |
| 8 | `agent.consolidate`             | `src/agent/agent.rs:1023` (`consolidate`) and `src/agent/daemon.rs:340` (`run_consolidation_check`) | Records `entries_evicted`, `episodes_created`. |
| 9 | `agent.reflect`                 | `src/agent/daemon.rs:308` (`run_reflection`)  | Records `adjustments_count`, `applied`.                                                                     |
| 10| `agent.sleep_cycle`             | `src/agent/sleep.rs:145` (`run_sleep_cycle`) and `src/agent/daemon.rs:644` (`run_sleep_cycle`) | Three child spans for `consolidation`, `reorganization`, `dream_state` per `ConsolidationPhase` enum. |
| 11| `agent.equivalence_learning`    | `src/agent/daemon.rs:289` (`run_equivalence_learning`) | Counter for `count`.                                                                              |
| 12| `agent.rule_inference`          | `src/agent/daemon.rs:389` (`run_rule_inference`)  | Records `triples`, `iterations`.                                                                        |
| 13| `agent.gap_analysis`            | `src/agent/daemon.rs:415` (`run_gap_analysis`) | Records `gaps`.                                                                                            |
| 14| `agent.continuous_learning`     | `src/agent/daemon.rs:447` (`run_continuous_learning`) | Currently disabled (commit `913a549`) — span scaffolding still useful for when it returns.        |
| 15| `agent.goal_generation`         | `src/agent/daemon.rs:495` (`run_goal_generation`) | Long; multi-phase locking — annotate the lock acquire/release as events.                              |
| 16| `agent.knowledge_extraction`    | `src/agent/daemon.rs:763` (`run_knowledge_extraction`) | Wraps LLM/regex three-tier fallback.                                                              |
| 17| `agent.bridge_training`         | `src/agent/daemon.rs:863` (`run_bridge_training`) | Long; gate on `enabled`.                                                                              |
| 18| `agent.trigger_evaluation`      | `src/agent/daemon.rs:674` (`run_trigger_evaluation`) | Cheap; high-frequency — sample at ~5 %.                                                            |
| 19| `agent.persist`                 | `src/agent/daemon.rs:208` (tick) → `persist`  | Wraps redb write transaction.                                                                              |
| 20| `engine.infer`                  | `src/engine.rs:680` (`Engine::infer`)         | Spreading-activation. Attributes: `seeds_count`, `top_k`, `max_depth`.                                     |
| 21| `engine.search_similar`         | `src/engine.rs:596` (`search_similar`)        | HNSW search. Attributes: `top_k`, `result_count`.                                                          |
| 22| `engine.sparql_query`           | `src/engine.rs:1030` (`sparql_query`)         | Oxigraph query. Attribute: SPARQL hash (full text would balloon trace storage).                            |
| 23| `engine.run_rules`              | `src/engine.rs` (`run_rules`)                  | Rule engine over KG.                                                                                       |
| 24| `engine.learn_equivalences`     | `src/engine.rs` (`learn_equivalences`)         | Phase 9 equivalence learning.                                                                              |
| 25| `mcp.handler`                   | `src/mcp/mod.rs` (`get_engine`/`get_agent`)    | Per-MCP-tool span. Use `span_kind = SERVER`.                                                               |
| 26| `http.handler`                  | every axum route in `src/bin/akhomed.rs`       | Use `tower-http`'s `TraceLayer` once we depend on it; inject route as attribute.                          |

For (7) and (26), span kind matters: tools are `INTERNAL`, MCP & HTTP handlers are `SERVER`.

### 4.2 `seshd` — async Seshat MCP server

Smaller surface; mostly request/response.

| # | Span name (proposed)            | Code location                                                  | Notes                                                                                            |
|---|---------------------------------|----------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| 1 | `seshat.mcp.search`             | `crates/seshat/src/mcp/mod.rs:162` (`seshat_search`)           | SERVER. Attributes: `query_len`, `top_k`, `min_similarity`, `result_count`.                      |
| 2 | `seshat.mcp.wish`               | `crates/seshat/src/mcp/mod.rs:209` (`seshat_wish`)             | SERVER. Attributes: `cache_hit`, `chunks`, `synthesize_here`.                                    |
| 3 | `seshat.mcp.status`             | `crates/seshat/src/mcp/mod.rs:285` (`seshat_status`)           | SERVER. Cheap.                                                                                   |
| 4 | `seshat.mcp.documents`          | `crates/seshat/src/mcp/mod.rs:316` (`seshat_documents`)        | SERVER.                                                                                          |
| 5 | `seshat.mcp.ingest`             | `crates/seshat/src/mcp/mod.rs:352` (`seshat_ingest`)           | SERVER. Attributes: `format`, `body_chars`, `chunk_count`.                                       |
| 6 | `seshat.mcp.embed_now`          | `crates/seshat/src/mcp/mod.rs:391` (`seshat_embed_now`)        | SERVER.                                                                                          |
| 7 | `seshat.embed_worker.tick`      | `crates/seshat/src/embed_worker.rs:31` (`run_embed_worker`)    | One span per *batch*, not per chunk. Attribute: `batch_size`, `pending`.                         |
| 8 | `seshat.embed_worker.batch`     | `crates/seshat/src/embed_worker.rs:69` (`embed_pending_batch`) | INTERNAL child of (7). Attributes: `chunks_processed`, `embeddings_created`.                     |
| 9 | `seshat.embed.onnx_inference`   | inside `spawn_blocking` at `embed_worker.rs:97`                | The actual model forward pass. Latency-sensitive — ideal candidate for a histogram (see §5).     |
| 10| `seshat.query.search_similar`  | `crates/seshat/src/query.rs:29` (`search_similar`)             | pgvector `<->` operator. Attributes: `top_k`, `min_similarity`, `rows`.                          |
| 11| `seshat.query.surrounding`     | `crates/seshat/src/query.rs:106` (`surrounding_chunks`)        | INTERNAL.                                                                                        |
| 12| `seshat.ingest.document`       | `crates/seshat/src/ingest.rs:92` (`ingest_document`)           | INTERNAL. Attributes: `chunks`, `word_count`, `format`.                                          |
| 13| `seshat.ingest.chunk`          | `crates/seshat/src/ingest.rs:199` (`chunk_paragraph_aware`)    | Cheap; merge into (12) unless we need per-chunk timing.                                          |
| 14| `seshat.cache.lookup`          | `crates/seshat/src/mcp/mod.rs:420` (`check_cache`)             | Records `cache_hit` boolean as event for Tempo's hit-rate metric extraction.                     |
| 15| `seshat.migration`             | `crates/seshat/src/main.rs:42` (`Migrator::up`)                | Lifecycle span at startup; one-shot.                                                             |
| 16| `seshat.http.handler`          | the axum router at `crates/seshat/src/main.rs:138`             | Use `tower-http`'s `TraceLayer` for the rmcp-mounted `/mcp` route.                              |

Note that Seshat shells the embedding through `tokio::task::spawn_blocking`. The OTel context does **not** automatically propagate across `spawn_blocking` boundaries — we need to capture the current span and re-attach inside the blocking closure (`Span::current()` then `.in_scope(...)`). Document this gotcha in the implementation plan.

### 4.3 Cross-process spans (Akh ↔ Seshat ↔ Anima)

`akhomed` calls `seshd` over MCP-over-HTTP today. To get a single trace that spans both processes we need W3C `traceparent` propagation:

- Outbound from `akhomed`: when invoking `seshat_search` / `seshat_wish` / `seshat_ingest` over HTTP, inject `traceparent` and `tracestate` headers via `opentelemetry::global::get_text_map_propagator(...)`.
- Inbound on `seshd`: extract the same headers in the axum middleware (`crates/seshat/src/main.rs:138`) and use them as the parent of the SERVER span.

The Anima MCP server is *external* to this repo — it lives in a separate Forgejo project. Propagating into Anima requires a one-line change there too (open question 2). For Phase 38i we only need Akh→Seshat propagation; Anima is deferred to 38k.

The akh CLI ↔ akhomed connection (over WebSocket and over HTTP) is also a propagation candidate. The existing `client.rs` (`src/client.rs`) talks to akhomed via `ureq` — easy to inject headers. Lower priority because the CLI is short-lived and traces would be one-cycle deep.

## 5. Metrics worth exposing

All metrics carry the `service.name` resource attribute (`akhomed` or `seshd`) plus `workspace` (akhomed only — multi-tenant `WorkspaceManager`). Histograms use the OTel-default exponential bucket layout unless noted.

### 5.1 Agent-loop metrics (`akhomed`)

| Metric                                  | Type      | Source                                                                | Labels                  |
|-----------------------------------------|-----------|-----------------------------------------------------------------------|-------------------------|
| `akh_ooda_cycles_total`                 | Counter   | `Agent::cycle_count` (`src/agent/agent.rs:1065`); incremented at `src/agent/ooda.rs:131` | `workspace`     |
| `akh_ooda_cycle_duration_seconds`       | Histogram | wrap `run_ooda_cycle` (`src/agent/ooda.rs:130`)                       | `workspace`             |
| `akh_active_goals`                      | Gauge     | `Observation::active_goals.len()` (`src/agent/ooda.rs:42`)            | `workspace`             |
| `akh_working_memory_size`               | Gauge     | `WorkingMemory::len()` (`src/agent/memory.rs:158`)                    | `workspace`             |
| `akh_working_memory_pressure`           | Gauge     | `WorkingMemory::pressure()` (`src/agent/memory.rs:173`)               | `workspace`             |
| `akh_tool_invocations_total`            | Counter   | `ToolRegistry::execute` (`src/agent/tool.rs:182`)                     | `workspace`, `tool`     |
| `akh_tool_duration_seconds`             | Histogram | same                                                                  | `workspace`, `tool`     |
| `akh_tool_failures_total`               | Counter   | same; increment on `Err`                                              | `workspace`, `tool`     |
| `akh_decision_impasses_total`           | Counter   | `DecisionImpasse` emitted at `src/agent/ooda.rs:103-122`              | `workspace`, `kind`     |
| `akh_goal_progress_total`               | Counter   | `GoalProgress` enum branches (`src/agent/ooda.rs:90-95`)              | `workspace`, `progress` (`no_change`, `advanced`, `completed`, `failed`) |

### 5.2 Cognitive subsystem metrics

These map onto the `AgentDaemon::run_*` background tasks (one per scheduled tick):

| Metric                                         | Type      | Source                                                            |
|------------------------------------------------|-----------|-------------------------------------------------------------------|
| `akh_consolidation_runs_total`                 | Counter   | `run_consolidation_check` (`src/agent/daemon.rs:340`)             |
| `akh_consolidation_entries_evicted_total`      | Counter   | `ConsolidationResult.entries_evicted` (`daemon.rs:357`)           |
| `akh_consolidation_episodes_created_total`     | Counter   | `ConsolidationResult.episodes_created.len()` (`daemon.rs:357`)    |
| `akh_reflection_runs_total`                    | Counter   | `run_reflection` (`daemon.rs:308`)                                |
| `akh_reflection_adjustments_total`             | Counter   | `result.adjustments.len()` + applied (`daemon.rs:312, 326`)       |
| `akh_sleep_cycles_total`                       | Counter   | `run_sleep_cycle` (`daemon.rs:644`)                               |
| `akh_sleep_duration_seconds`                   | Histogram | wrap `run_sleep_cycle`                                            |
| `akh_sleep_duplicates_merged_total`            | Counter   | `SleepMetrics.duplicates_merged` (`src/agent/sleep.rs:79`)        |
| `akh_sleep_orphans_pruned_total`               | Counter   | `SleepMetrics.orphans_pruned` (`sleep.rs:81`)                     |
| `akh_sleep_dream_connections_total`            | Counter   | `SleepMetrics.dream_connections_found` (`sleep.rs:83`)            |
| `akh_equivalences_learned_total`               | Counter   | `run_equivalence_learning` (`daemon.rs:289`)                      |
| `akh_rule_inference_triples_total`             | Counter   | `run_rule_inference` (`daemon.rs:389`)                            |
| `akh_rule_inference_iterations`                | Histogram | same; one observation per run                                     |
| `akh_gap_analysis_gaps_found_total`            | Counter   | `run_gap_analysis` (`daemon.rs:415`)                              |
| `akh_goal_generation_proposals_total`          | Counter   | `run_goal_generation` (`daemon.rs:495`)                           |
| `akh_continuous_learning_concepts_ingested_total` | Counter | `run_continuous_learning` (`daemon.rs:447`)                       |
| `akh_knowledge_extraction_triples_total`       | Counter   | `run_knowledge_extraction` (`daemon.rs:763`)                      |
| `akh_trigger_firings_total`                    | Counter   | `run_trigger_evaluation` (`daemon.rs:674`); label by trigger name |

### 5.3 Storage / KG metrics

| Metric                              | Type      | Source                                                  |
|-------------------------------------|-----------|---------------------------------------------------------|
| `akh_kg_symbols`                    | Gauge     | sample from `Engine` symbol table (`src/symbol.rs`)     |
| `akh_kg_triples`                    | Gauge     | `DaemonStatus.kg_triples` (`src/bin/akhomed.rs:3413`)   |
| `akh_kg_inserts_total`              | Counter   | every `add_triple` call site                            |
| `akh_hnsw_search_duration_seconds`  | Histogram | wrap `Engine::search_similar` (`src/engine.rs:596`)     |
| `akh_hnsw_search_results`           | Histogram | result count from same                                  |
| `akh_sparql_query_duration_seconds` | Histogram | wrap `Engine::sparql_query` (`src/engine.rs:1030`)      |
| `akh_inference_duration_seconds`    | Histogram | wrap `Engine::infer` (`src/engine.rs:680`)              |
| `akh_inference_activations`         | Histogram | `result.activations.len()`                              |
| `akh_persist_duration_seconds`      | Histogram | wrap `daemon.rs:208` (`persist`); redb txn duration     |
| `akh_audit_entries_total`           | Counter   | from `AuditLedger` (`src/bin/akhomed.rs:124`)           |

### 5.4 Seshat metrics (`seshd`)

| Metric                                       | Type      | Source                                                                     |
|----------------------------------------------|-----------|----------------------------------------------------------------------------|
| `seshat_mcp_requests_total`                  | Counter   | every `#[tool]` handler (`crates/seshat/src/mcp/mod.rs:158-405`)           |
| `seshat_mcp_request_duration_seconds`        | Histogram | same                                                                       |
| `seshat_mcp_request_errors_total`            | Counter   | same; on `McpError`                                                        |
| `seshat_search_results`                      | Histogram | `seshat_search` result count (`mcp/mod.rs:190`)                            |
| `seshat_wish_cache_hits_total`               | Counter   | `check_cache` returning `Some` (`mcp/mod.rs:420`)                          |
| `seshat_wish_cache_misses_total`             | Counter   | `check_cache` returning `None`                                             |
| `seshat_documents_total`                     | Gauge     | `count_documents` (`crates/seshat/src/query.rs:65`)                        |
| `seshat_chunks_total`                        | Gauge     | `count_chunks` (`query.rs:74`)                                             |
| `seshat_embedded_chunks_total`               | Gauge     | `count_embedded_chunks` (`query.rs:83`)                                    |
| `seshat_pending_embedding_chunks`            | Gauge     | `count_pending` (`embed_worker.rs:147`) — **the key backlog metric**       |
| `seshat_microtheory_cache_size`              | Gauge     | `count_cached_microtheories` (`query.rs:95`)                               |
| `seshat_embed_batches_total`                 | Counter   | `run_embed_worker` (`embed_worker.rs:31`)                                  |
| `seshat_embed_batch_duration_seconds`        | Histogram | wrap `embed_pending_batch` (`embed_worker.rs:69`)                          |
| `seshat_embed_chunks_processed_total`        | Counter   | `EmbedStats.chunks_processed` (`embed_worker.rs:23`)                       |
| `seshat_pgvector_search_duration_seconds`    | Histogram | wrap `query::search_similar` (`query.rs:29`)                               |
| `seshat_ingest_documents_total`              | Counter   | `ingest_document` (`ingest.rs:92`)                                         |
| `seshat_ingest_chunk_count`                  | Histogram | `IngestSummary.chunk_count`                                                |
| `seshat_db_pool_active`                      | Gauge     | from SeaORM connection pool stats (if exposed)                             |

### 5.5 Process / runtime baseline

Both binaries should also emit standard process metrics. Easiest path: add `opentelemetry-process-metrics` (or write a tiny custom collector — there are only ~6 useful values):

| Metric                      | Type  | Source                       |
|-----------------------------|-------|------------------------------|
| `process_cpu_seconds_total` | Counter | `/proc/self/stat` on Linux  |
| `process_resident_memory_bytes` | Gauge | same                       |
| `process_open_fds`          | Gauge | `/proc/self/fd`              |
| `process_start_time_seconds`| Gauge | constant                     |
| `runtime_tokio_workers_busy` | Gauge | `tokio::runtime::Handle::metrics()` (Tokio 1.36+ — already on `tokio = "1"`) |
| `runtime_tokio_tasks_alive` | Gauge | same                         |

## 6. Helm chart changes

### 6.1 Current chart shape

`charts/akh-medu/` is a single chart with two top-level workloads:

- `templates/akhomed-deployment.yaml` — `Deployment` strategy `Recreate` (single writer to KG volume), one container, env from `values.yaml` `akhomed.env` (currently `RUST_LOG`, `AKH_SERVER_BIND`, `AKH_SERVER_PORT`, `AKH_AUTO_START`), one HTTP probe on `/status` (`charts/akh-medu/values.yaml:91-96`).
- `templates/seshd-deployment.yaml` — async deployment with an `initContainer` that runs `seshd migrate`, then `seshd` proper. ConfigMap mount at `/config/seshat.toml`, models PVC at `/models`, env-from secret for `DATABASE_URL`. Probe is `tcpSocket` on the rmcp HTTP port (chart comment explains `/mcp` returns 406 to plain GET — see `values.yaml:159-176`).

Both deployments accept arbitrary env via `range $key, $val := .Values.<binary>.env` (`akhomed-deployment.yaml:36-39`, `seshd-deployment.yaml:55-58`), so adding OTLP env vars is a values-only change with no template surgery.

### 6.2 Sidecar collector vs cluster-wide DaemonSet

Two well-known topologies:

**A) Cluster-wide OTel Collector DaemonSet / Deployment** (single endpoint per node or per cluster, owned by the platform team).
- Pros: one place to manage retention, sampling, auth, TLS to upstream Tempo / Loki / Mimir. Rust binaries just point at `http://otel-collector.observability:4317`. Smaller per-pod footprint. Aligned with the existing Grafana stack pattern (Loki / Tempo / Mimir already cluster-wide).
- Cons: tail sampling decisions become a shared concern; an Akh that emits 10k spans/s can shoulder-check other tenants on the same collector pod.

**B) Per-pod Collector sidecar** (every akhomed/seshd pod has a `otelcol-contrib` sidecar in the same pod spec, app talks to `localhost:4317`).
- Pros: failure isolation (collector OOM doesn't take out the agent or vice versa). Per-app sampling rules. Trace data is buffered in the pod's memory, so a brief collector outage in the cluster pipe doesn't drop traces.
- Cons: doubles the pod count, more YAML, harder to upgrade collector versions. Operator-style Sidecar injection (the OTel Operator's `Sidecar` CR) helps but adds another component.

### 6.3 Recommended deployment topology

**Pick A — cluster-wide collector — with one nuance.** The Talos cluster already runs the Grafana stack; presumably an OTel Collector lives next to Tempo / Loki / Mimir in the `observability` (or similar) namespace. Reuse it. Rationale: we currently run **one** akhomed and **one** seshd replica, and even at full duty cycle the trace volume is in the low hundreds of spans per second. Sidecar overhead is not worth it. If a future Phase introduces multi-tenant Akhs and trace volume spikes, revisit.

The one nuance: configure the cluster-wide collector with a **memory_limiter** processor + per-tenant rate limiter so a misbehaving Akh can't degrade other workloads. That is a collector-side change, not a chart change.

### 6.4 Env vars + config injection

Add to `charts/akh-medu/values.yaml` (both `akhomed.env` and `seshd.env` blocks):

```yaml
akhomed:
  env:
    # ... existing entries ...
    OTEL_EXPORTER_OTLP_ENDPOINT: "http://otel-collector.observability.svc.cluster.local:4317"
    OTEL_EXPORTER_OTLP_PROTOCOL: "grpc"
    OTEL_SERVICE_NAME: "akhomed"
    OTEL_RESOURCE_ATTRIBUTES: "service.namespace=akh-medu,deployment.environment=prod"
    OTEL_TRACES_SAMPLER: "parentbased_traceidratio"
    OTEL_TRACES_SAMPLER_ARG: "0.1"        # 10 % head sampling — see open question 1
    OTEL_LOG_LEVEL: "info"

seshd:
  env:
    # ... existing entries ...
    OTEL_EXPORTER_OTLP_ENDPOINT: "http://otel-collector.observability.svc.cluster.local:4317"
    OTEL_EXPORTER_OTLP_PROTOCOL: "grpc"
    OTEL_SERVICE_NAME: "seshd"
    OTEL_RESOURCE_ATTRIBUTES: "service.namespace=akh-medu,deployment.environment=prod"
    OTEL_TRACES_SAMPLER: "parentbased_traceidratio"
    OTEL_TRACES_SAMPLER_ARG: "1.0"        # accept whatever the parent sampled — Seshat is mostly downstream
```

The Rust SDK respects the standard OTel env vars at SDK init time, so we don't need a chart-side ConfigMap diff — the `range $key, $val := .Values.<binary>.env` already iterates everything.

`OTEL_SERVICE_INSTANCE_ID` should ideally come from the pod name; inject via the downward API in the deployment templates (small chart edit):

```yaml
env:
  - name: OTEL_SERVICE_INSTANCE_ID
    valueFrom:
      fieldRef:
        fieldPath: metadata.name
```

### 6.5 ServiceMonitor / PodMonitor (if Prom-Operator route is kept)

If Mimir is fronted by `kube-prometheus-stack` and uses `PodMonitor` CRs to discover scrape targets, we can side-step the collector entirely for metrics by enabling the Rust-side `opentelemetry-prometheus` exporter on a `/metrics` endpoint. Trade-off:

- **Push to OTel Collector** (default in §6.4): one config path, traces and metrics via the same wire, exemplar links work end-to-end.
- **Pull via PodMonitor**: simpler to debug ("just curl /metrics"), Mimir's scrape interval is the rate-control knob, but exemplars don't ship cleanly and we lose the "metrics derived from spans" capability.

**Recommendation**: push for Phase 38j, add `PodMonitor` only if the operator wants direct Mimir scraping later. Either way, no chart structure change beyond an optional `templates/podmonitor.yaml` keyed off `monitoring.podMonitor.enabled` in `values.yaml`.

## 7. Phasing

Three sub-phases, each demoable on its own. Each phase ends with a working Grafana panel.

### Phase 38i — Tracing skeleton + first OODA spans

**Goal:** one trace shows up in Tempo for a single OODA cycle on a deployed `akhomed` pod.

Scope:

1. Add OTel crates (versions pinned per §2.1) to root `Cargo.toml` and `crates/seshat/Cargo.toml`. Behind a `tracing-otel` Cargo feature in both crates so local dev can opt out if collector init flake-races on startup.
2. Refactor the three subscriber inits (`src/main.rs:796`, `src/bin/akhomed.rs:3337`, `crates/seshat/src/main.rs:19`) to a shared helper:
   - `akh_medu::observability::init_tracing(service_name)` returns a `TracerShutdown` guard the binary holds to drop-flush at exit.
   - Conditional on env var `OTEL_EXPORTER_OTLP_ENDPOINT` being set; otherwise fall back to today's `fmt().init()` behaviour.
3. Annotate the OODA loop in `src/agent/ooda.rs`:
   - `#[tracing::instrument(skip_all, fields(cycle = agent.cycle_count + 1))]` on `run_ooda_cycle`.
   - Same on `observe`, `orient`, `decide`, `act` — five spans total per cycle.
4. Annotate `ToolRegistry::execute` (`src/agent/tool.rs:182`) with a span carrying `tool = %tool_name`.
5. Helm: add the OTLP env vars from §6.4 to `values.yaml`. Add `OTEL_SERVICE_INSTANCE_ID` downward-API ref to both deployment templates.
6. Smoke test: deploy, trigger one OODA cycle (via existing HTTP `/agent/run-cycle` route or by waiting for the `idle_cycle_tick`), verify the trace appears in Tempo with the 5-span tree.

Acceptance: a Grafana Tempo screenshot showing one trace `agent.ooda_cycle` → `observe`/`orient`/`decide`/`act` → `tool.execute(<name>)`.

**This is the minimal demoable slice.** No metrics yet, no Seshat instrumentation yet, no log propagation yet.

Rough effort: 1–2 days.

### Phase 38j — Metrics + Seshat instrumentation

**Goal:** Grafana dashboard with the §5.1, §5.2, §5.4 metrics live.

Scope:

1. Add `opentelemetry_sdk::metrics::Meter` initialisation to the same `init_tracing` helper, rename it to `init_observability`. Returns both a `TracerShutdown` and `MeterShutdown`.
2. Wire counters / histograms / gauges in this order:
   - **akhomed**: `akh_ooda_cycles_total`, `akh_ooda_cycle_duration_seconds`, `akh_active_goals`, `akh_working_memory_size`, `akh_tool_invocations_total`, `akh_tool_duration_seconds`, `akh_tool_failures_total`. (§5.1)
   - **akhomed**: the cognitive subsystem counters (§5.2). One per `run_*` method in `src/agent/daemon.rs` — pure additions, no logic changes.
   - **akhomed**: the storage / KG histograms (§5.3) — wraps around `Engine::infer`, `Engine::search_similar`, `Engine::sparql_query`.
3. Instrument `seshd`:
   - All six `#[tool]` handlers in `crates/seshat/src/mcp/mod.rs` get a SERVER span + duration histogram + request counter.
   - Embed worker batch span + duration histogram + `seshat_pending_embedding_chunks` gauge sampled once per tick (`embed_worker.rs:31`). The pending gauge is the single most operationally useful metric — it tells the operator if the worker is keeping up.
   - pgvector search: `query::search_similar` (`query.rs:29`) gets a span + histogram.
   - Microtheory cache hit/miss counters at the `check_cache` site (`mcp/mod.rs:420`).
4. Provision a Grafana dashboard JSON with 12–15 panels grouped: "OODA loop", "Cognitive subsystems", "Knowledge graph", "Seshat — embedding pipeline", "Seshat — request rate / latency", "Process baseline". Check it in under `charts/akh-medu/grafana-dashboard.json` (or a Flux `ConfigMap` with the `grafana_dashboard` label per the existing Flux GitOps convention).
5. Optional: turn on **exemplars** on the histograms so a Mimir bucket spike links to the example trace in Tempo. Requires `OTLP/grpc` and the cluster collector to forward exemplars — verify both.

Acceptance: a Grafana dashboard with all panels green; a histogram bucket can be clicked through to a Tempo trace via exemplars.

Rough effort: 2–3 days.

### Phase 38k — Logs unification + cross-process propagation

**Goal:** a single trace ID stitches together a request from `akh CLI → akhomed → seshd → pgvector` and the corresponding log lines in Loki are clickable from Tempo.

Scope:

1. Switch `tracing_subscriber::fmt()` to `fmt::json()` in all three binaries — Loki ingests structured fields automatically. Add a `tracing_subscriber` field with `trace_id` + `span_id` from the OTel context (cookbook recipe in `tracing-opentelemetry` README).
2. **W3C traceparent propagation Akh ↔ Seshat:**
   - Outbound: in the akhomed Seshat client (whichever module calls `seshat_search` / `seshat_wish`), inject `traceparent` into the rmcp HTTP request headers via `opentelemetry::global::get_text_map_propagator(...)`.
   - Inbound: a tower middleware on the seshd axum router (`crates/seshat/src/main.rs:138`) that extracts the headers and seeds the SERVER span's parent. Pattern: a `TraceContextPropagator` registered globally.
3. **akh CLI → akhomed propagation:**
   - The CLI's `client.rs` uses `ureq`; inject `traceparent` on outbound calls. Lower priority — most CLI calls are one-shot.
4. (Optional) Akhomed → Anima propagation. Out of scope unless Anima's MCP surface is updated to accept trace context — file an Anima WI before doing this.
5. Configure Grafana's Tempo data source with `tracesToLogs` (`derived_fields` → `Loki: trace_id={{ .TraceId }}`) so the Tempo UI gets a "View related logs" button.
6. (Optional) migrate stdout logs to OTel logs API via `opentelemetry-appender-tracing`. Defer unless the operator decides log → trace correlation is too lossy via Loki scraping.

Acceptance: clicking a trace in Tempo, selecting "View related logs", lands in Loki on the right time range with matching `trace_id`.

Rough effort: 1–2 days (longer if option 6 is taken).

### Phase ordering rationale

Each sub-phase delivers a complete operator-visible win:

- 38i: "I can see what an OODA cycle does."
- 38j: "I can see how the agent is *behaving*."
- 38k: "I can debug a cross-service issue from one trace."

If 38j or 38k slip, 38i still has standalone value. If we have to revert, traces and metrics are independent — `init_observability` is feature-flagged.

## 8. Open questions for the user

### Resolutions log (2026-04-26)

User answers, locked in:

- **Q1 — Sampling strategy.** Default accepted: head sampling 10 % via `parentbased_traceidratio` on akhomed, accept-parent on seshd, force-keep on `error=true`. (No tail sampling for now.)
- **Q2 — Cross-process propagation.** Akh ↔ Seshat in 38k (we own both ends). akh CLI → akhomed in 38k (two-line change in `client.rs`). **Akhomed → Anima propagation is NOT filed from this side — Anima team will file their own OTLP work.**
- **Q3 — Logs.** **CHANGED FROM PROPOSED DEFAULT.** Logs go via the **OTel logs API → OTLP → Alloy**, not stdout → Loki. Rationale: out-of-cluster akhomed instances (e.g. M2 in cellar, future remote backends) need to ship logs into the same pipeline; only OTLP gives that uniformly. Talos already runs Alloy as the in-cluster log scraper, but Alloy also accepts incoming OTLP, so we ship to Alloy directly via `opentelemetry-appender-tracing`. This **promotes `opentelemetry-appender-tracing` from "deferred" to in-scope for Phase 38k**.
- **Q4 — Metrics exposition.** OTLP push only. No `opentelemetry-prometheus` pull exporter. Rationale: keep one wire, one config path. Phase 38j drops the "second exporter" optional path.
- **Q5 — Cardinality budget.** Default accepted: lock the §5 label list; reject `symbol_id`, `microtheory_id`, `goal_id`, raw SPARQL text as metric labels (span attributes only).
- **Q6 — Build feature flag.** Default accepted: `tracing-otel` Cargo feature, default-on under the `server` feature.
- **Q7 — `workspace` resource attribute.** Default accepted: span attribute + metric label. No per-workspace `Tracer`/`Meter`.
- **Q8 — TLS / distroless.** Default accepted: cluster-internal HTTP for collector reach. Revisit when/if Akhs federate across clusters.

### Phasing impact of resolutions

- **Phase 38i:** unchanged.
- **Phase 38j:** simpler — drop the "optional `opentelemetry-prometheus` second exporter" line item. OTLP push only.
- **Phase 38k:** scope shifts. Logs migrate to **OTel logs API via `opentelemetry-appender-tracing` (not deferred)**. Stdout JSON is still emitted as a debug fallback but the canonical log shipping path is OTLP → Alloy. M2 / remote akhomed instances ship to a public OTLP ingress on Alloy (separate ingress ticket if/when remote akhomed lands — out of scope for 38k, file when Phase 39 starts shipping remote backends).

### Open question text (kept for record)

1. **Sampling strategy.** Tempo storage budget = sampling decisions.
   - Proposed default: **head sampling at 10 % via `parentbased_traceidratio` on `akhomed`**, accept-parent on `seshd`. Burst-prone OODA tool calls (e.g. `pattern_mine`, `code_ingest`) get 100 % via a dedicated sampler rule. If a span is marked `error = true` (panicked tool), force-keep.
   - Alternative: tail sampling at the collector ("keep all error traces, 5 % of the rest, plus 100 % of traces over 500 ms"). Heavier collector config, but no decision lost. Pick this if the cluster collector's `tailsamplingprocessor` is already in use elsewhere.

2. **Trace propagation across Akh ↔ Seshat ↔ Anima.**
   - Akh ↔ Seshat is in scope for 38k (we own both ends).
   - Akh → Anima requires a one-line MCP-server change in the Anima repo (the rmcp axum router needs the same propagator middleware as seshd). Proposed default: **defer to Phase 39+**, file the Anima-side ticket now (`anima_create_work_item`) so it's tracked. Tag it `observability`.
   - akh CLI ↔ akhomed: nice-to-have. Proposed default: **add in 38k** because `client.rs` already exists and the change is two lines.

3. **Logs: OTel Logs API vs stdout-to-Loki.**
   - Proposed default: **stdout JSON → Loki via Promtail/Alloy** (Phase 38i scope), revisit only if the operator wants exemplar-quality log↔trace correlation. Loki indexes JSON fields, including a `trace_id` field we add via `tracing_opentelemetry`.
   - Counter-question to confirm: does Talos run a log scraper today (Promtail / Grafana Alloy / Vector)? If not, we either need to deploy one or go straight to OTel-native logs.

4. **Metric exposition: OTLP push vs Prometheus pull.**
   - Proposed default: **OTLP push** for unified pipeline. Adds zero scrape config to the cluster.
   - If the operator prefers debuggability (`curl /metrics`), add `opentelemetry-prometheus` as a *second* exporter alongside OTLP — both can run, no extra tradeoffs other than ~50 LOC.

5. **Cardinality budget — what becomes a metric label?**
   - Safe labels (low cardinality): `workspace` (≤10), `tool` (≤30), `goal_progress` enum (4 values), `kind` for impasses (small enum), `format` for ingest (≤6).
   - Unsafe — keep these as **span attributes only, never metric labels**:
     - `symbol_id` (unbounded — KG grows monotonically)
     - `microtheory_id`
     - `goal_id`
     - SPARQL query text (use a hash if it must be a label).
   - Proposed default: lock the labels listed in §5 tables; reject PRs that add new high-cardinality labels.

6. **Build-time feature flag vs always-on.**
   - Proposed default: **feature flag `tracing-otel`, default-on for the `server` feature** (which is what builds `akhomed`). Out-of-cluster local dev runs `cargo run --bin akh` without OTel, in-cluster runs `--features server,tracing-otel`. Keeps the link cost off the CLI binary.
   - If we want to keep things simpler: always-on, but skip exporter init when `OTEL_EXPORTER_OTLP_ENDPOINT` is unset. Marginal binary-size cost, simpler matrix.

7. **Workspace-level resource attribute.** akhomed is multi-tenant — one process, N workspaces. Should `workspace` be a span attribute (default), a metric label, **or** a resource attribute on a per-workspace tracer? Proposed default: span attribute + metric label. A resource-per-workspace would require multiple `Tracer`/`Meter` instances per process; not worth it.

8. **Distroless / static-musl image impact.** `opentelemetry-otlp` with `grpc-tonic` pulls in `tonic` → `hyper` → TLS. The current Docker image (verify in `docker/`) needs `ca-certificates` for outbound HTTPS to the collector if we ever go off-cluster. In-cluster (via `http://`) this is moot. Proposed default: keep collector reach as cluster-internal HTTP, don't add TLS until/unless we federate Akhs across clusters.

---

## Appendix A — Crate version pins (tentative)

```toml
# Add to root Cargo.toml [dependencies] and crates/seshat/Cargo.toml
opentelemetry         = "=0.27"
opentelemetry_sdk     = { version = "=0.27", features = ["rt-tokio"] }
opentelemetry-otlp    = { version = "=0.27", features = ["grpc-tonic", "trace", "metrics"] }
tracing-opentelemetry = "=0.28"
opentelemetry-semantic-conventions = "=0.27"

# Already present:
# tracing            = "0.1"
# tracing-subscriber = { version = "0.3", features = ["env-filter"] }
#   ↳ Phase 38i: add the "json" feature

# Phase 38k (in scope per Q3 resolution — OTel logs API → Alloy):
# opentelemetry-appender-tracing = "=0.27"
```

Verify after adding:
```sh
cargo tree -p opentelemetry --invert
cargo tree -p opentelemetry_sdk --invert
```
Both should show only the OTel crates and `tracing-opentelemetry` as direct consumers — no double versions.

---

## Appendix B — Code-citation index

Quick lookup table for the implementation plan; every line cited above with file:line.

| File                                                  | Line(s)        | What                                            |
|-------------------------------------------------------|----------------|-------------------------------------------------|
| `Cargo.toml`                                          | 76–79          | `tracing` 0.1 / `tracing-subscriber` 0.3 deps   |
| `Cargo.toml`                                          | 159–161        | `[[bin]] akh` → `src/main.rs`                   |
| `Cargo.toml`                                          | 163–165        | `[[bin]] akhomed` → `src/bin/akhomed.rs`, requires `server` feature |
| `crates/seshat/Cargo.toml`                            | 47–48          | `tracing` deps in seshat                        |
| `crates/seshat/Cargo.toml`                            | 79–81          | `[[bin]] seshd` → `crates/seshat/src/main.rs`   |
| `src/main.rs`                                         | 784–803        | CLI/TUI `fn main` + tracing init                |
| `src/bin/akhomed.rs`                                  | 3335–3343      | `akhomed` `#[tokio::main]` + tracing init       |
| `src/bin/akhomed.rs`                                  | 3383–3909      | startup / shutdown lifecycle log lines          |
| `crates/seshat/src/main.rs`                           | 16–24          | `seshd` `#[tokio::main]` + tracing init         |
| `crates/seshat/src/main.rs`                           | 105–115        | embed worker `tokio::spawn`                     |
| `crates/seshat/src/main.rs`                           | 138            | axum router mount of `/mcp`                     |
| `src/agent/daemon.rs`                                 | 117–260        | `AgentDaemon` struct + 14-tick `select!` loop   |
| `src/agent/daemon.rs`                                 | 289–909        | `run_*` background-task methods                 |
| `src/agent/ooda.rs`                                   | 130–153        | `run_ooda_cycle` (4-step span tree)             |
| `src/agent/ooda.rs`                                   | 156, 234, 300, 1315 | `observe` / `orient` / `decide` / `act`    |
| `src/agent/agent.rs`                                  | 599, 1023, 1065 | `run_cycle`, `consolidate`, `cycle_count`      |
| `src/agent/memory.rs`                                 | 158, 173       | `WorkingMemory::len()`, `pressure()`            |
| `src/agent/memory.rs`                                 | 263–274        | `auto_consolidate_at` config                    |
| `src/agent/sleep.rs`                                  | 22–32, 73–88, 145 | `ConsolidationPhase` enum, `SleepMetrics`, `run_sleep_cycle` |
| `src/agent/tool.rs`                                   | 182            | `ToolRegistry::execute`                         |
| `src/engine.rs`                                       | 596, 680, 1030 | `search_similar`, `infer`, `sparql_query`       |
| `crates/seshat/src/mcp/mod.rs`                        | 158–405        | six `#[tool]` handlers                          |
| `crates/seshat/src/mcp/mod.rs`                        | 420–445        | `check_cache` (microtheory cache hit/miss)      |
| `crates/seshat/src/embed_worker.rs`                   | 31–66          | `run_embed_worker` loop                         |
| `crates/seshat/src/embed_worker.rs`                   | 69–143         | `embed_pending_batch`                           |
| `crates/seshat/src/embed_worker.rs`                   | 147–153        | `count_pending`                                 |
| `crates/seshat/src/query.rs`                          | 29, 65, 74, 83, 95, 106 | `search_similar`, count helpers, `surrounding_chunks` |
| `crates/seshat/src/ingest.rs`                         | 92, 199        | `ingest_document`, `chunk_paragraph_aware`      |
| `charts/akh-medu/values.yaml`                         | 38–43          | `akhomed.env` block (where to add OTLP env)     |
| `charts/akh-medu/values.yaml`                         | 113–117        | `seshd.env` block                               |
| `charts/akh-medu/templates/akhomed-deployment.yaml`   | 36–39          | `range $key, $val := .Values.akhomed.env`       |
| `charts/akh-medu/templates/seshd-deployment.yaml`     | 55–58          | `range $key, $val := .Values.seshd.env`         |
| `charts/akh-medu/templates/seshd-deployment.yaml`     | 67–88          | probe configuration (TCP socket on rmcp)        |
