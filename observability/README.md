# Runtime sessions and turns

The session-service binaries export OTLP/HTTP protobuf logs, traces and metrics
when `OTEL_EXPORTER_OTLP_ENDPOINT` is set. Without it, stderr logging remains
available and no exporter is installed. The service identity comes from the
binary name and configured service ID; additional resource attributes are read
from `OTEL_RESOURCE_ATTRIBUTES`.

Example for existing cluster runtimes:

```sh
OTEL_EXPORTER_OTLP_ENDPOINT=http://alloy.observability.svc.cluster.local:4318
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
OTEL_RESOURCE_ATTRIBUTES=deployment.environment=homelab,service.namespace=akh-platform
OTEL_TRACES_SAMPLER=always_on
```

Local Roman Akhs retain their local runtime, SQLite stores, model bindings and
workspace placement. Their runtime exports to `http://127.0.0.1:19418`, through
the `dev.roman-akhs.otlp-tunnel` launch daemon to the same Alloy service. The
tunnel uses the existing Kubernetes credential. No provider credentials are
required for telemetry. The cluster Akhs continue running in the cluster.

## What is captured

- A trace per runtime turn, identified by session, turn, runtime, profile, model
  and job. Resumed turns retain the durable session ID and get a new trace.
- Nested model iterations, HTTP attempts, tools and human-input waits. A span
  is exported when it closes; logs show activity while a turn is still running.
- Timestamped text and exposed reasoning deltas, tool start/completion, token
  usage, rejected provider requests, retry waits and final operation outcome.
- OTLP logs include inherited session/turn fields and the trace/span IDs.
- Counters for tokens, stream bytes and operations, plus duration histograms.
  Session IDs are deliberately absent from metric labels.

`AKH_OTEL_CAPTURE_CONTENT=true` includes up to 2,048 characters per text or
reasoning delta. It is enabled in this deployment at the user's request.
Without it, only the delta kind and byte length are exported. Prompts, tool
arguments/results and complete provider error bodies are not exported. Text
and reasoning content can nevertheless contain private task information.

Internal chat-completion providers supply usage through an explicitly requested
SSE usage trailer. Codex reasoning/usage notifications and Claude exposed
thinking deltas/result usage are also observed. Unexposed provider information
remains unavailable, not zero. Input counts follow the provider input field;
Claude cache-read/cache-creation counts are not synthesized into that field. Delta
events are transport chunks, not individual token timestamps. Reasoning token
counts are a subset of output tokens and must not be summed with them.

Each span retains at most 4,096 events. Logs are exported independently through
the bounded SDK batch queue. Export is best effort, not a replacement for the
durable session ledger; collector outages or queue overflow can lose telemetry.

Provider diagnostics retain HTTP status, bounded code/type identifiers,
request ID and Retry-After. String-only errors are classified as rate limit,
quota/billing, usage limit, or unclassified without exporting the raw message.
Only explicit 429 responses retry: at most three attempts, 1/2-second fallback
delays, and integer Retry-After values up to 30 seconds. Longer or date-form
Retry-After values are not retried early. The existing deadline and cancellation
cover the request, error-body read and backoff. Partial streams, ambiguous
transport failures and tools are never replayed automatically. Existing
reconciliation and permission gates remain authoritative.

## Grafana

Import `runtime-dashboard.json`, or open the published dashboard:

https://grafana.lab.home/d/akh-runtime-turns

The dashboard includes operation latency/rate, usage, stream activity, filtered
session/turn logs and a trace table leading to the waterfall. Metrics use
VictoriaMetrics; logs use Loki; traces use Tempo. Loki's Alloy exporter nests
event fields under `attributes` and derives `service_name` from the namespaced
job, so its queries use `attributes_session_id` and `.*akhrtd`.

Matching Grafana provisioning is in the Talos GitOps change. Loki's `traceid`
derived field links to Tempo, and Tempo's trace-to-logs query selects runtime
logs containing that trace ID. No infrastructure placement changes are needed.

## Verification and rollout

```sh
cargo test -p akh-runtime -p akh-session-services --no-default-features \
  --features akh-session-services/runtime --lib --test providers --test telemetry

OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:19418 \
  cargo run -p akh-session-services --no-default-features --features runtime \
  --example runtime-telemetry-smoke
```

The smoke example uses synthetic spans/events and invokes no provider, tool or
session mutation. The wire test decodes real OTLP protobuf and checks the span
hierarchy, correlation, metrics, content opt-in and absence of session labels
on metrics. Provider fixtures cover 429 retry bounds/diagnostics and native
reasoning/usage callbacks.

Build the local binary from the Roman deployment branch, not the older main
checkout: the latter lacks its SQLite store support. Validate the existing
configuration with `akhrtd --config ... --check` before replacing anything.
Check the owning session API for active turns immediately before any restart;
do not interrupt a busy runtime or force its lifecycle state to make it idle.

`docker/runtime-observability.Dockerfile` layers the new runtime, its matching
measurement probe and the telemetry-enabled Claude bridge onto the specified
immutable deployed native base. Provider CLIs, SDK packages and helper tools
remain those of that base. Native conformance includes the runtime executable
hash: old measurements cannot certify a changed binary. Before deploying, run:

```sh
python3 scripts/session-platform/measure-runtime-observability.py \
  --image IMAGE --output NEW_EVIDENCE_DIRECTORY
```

This runs real installed Codex/Claude against synthetic local peers in a Linux
container with `--network none`, no credentials and no host mounts. Wire tests
cover native-write/exec denial, supplied tools, exact resume and hook suppression;
separate cancellation tests require descendant teardown. Receipts bind the
runtime, executor and probe hashes. Preserve the full evidence and use a new
immutable conformance ConfigMap. Its init container must use the same measured
image as the native worker; the existing certification and readiness commands
remain mandatory. Roll out one worker of each native kind first.

Deploy resulting images by digest, retaining node selectors, services, volumes,
credentials, bindings and replica counts. Internal workers do not use native
provider conformance and use the internal-base image.
