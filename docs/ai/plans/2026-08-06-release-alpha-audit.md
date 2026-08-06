# Release Alpha Audit — Full Sub-Phase Analysis

> **Date:** 2026-08-06  
> **Auditor:** tecton (session 14)  
> **WI:** `019f90cd-7609-7d63-ac70-3c566b6e83d9`  
> **Audit target:** WI-3 (`019db054-0da3-76a3-8725-0d1933aa32df`, "Release Alpha — Kubernetes deployment")  
> **Plan under review:** [`2026-02-23-release-alpha.md`](2026-02-23-release-alpha.md)  
> **Base commit:** `9965318` (main @ time of audit)  
> **HEAD:** `aa8aefb` (this PR's branch)  

## 0. Summary

Release Alpha is **largely shipped**, but **not fully**. A section-by-section audit of all
seven sub-phases against `main` at `9965318` reveals three genuinely-remaining gaps:

| Gap | Sub-phase | Severity |
|-----|-----------|----------|
| **A** | Alpha-4 (Metrics) | Medium — no Prometheus metrics or ServiceMonitor; OTLP tracing partially fills the observability gap but metrics are missing |
| **B** | Alpha-7 (Deployment docs) | Low — no K8s deployment guide, no getting-started, no configuration reference |
| **C** | Alpha-7 (README staleness) | Low — README still references retired REST/WS surface |

**Recommendation:** Cancel WI-3 as superseded. File two concrete follow-up work items
(Gap A and Gap B; Gap C folds into Gap B).

## 1. Sub-Phase Audit

### Alpha-1 — Build & Packaging ✅ Shipped

**Docker containerization:**
- 4 Dockerfiles in `docker/`: `akh.Dockerfile`, `akhomed.Dockerfile`, `seshd.Dockerfile`, `akhomed-cloud.Dockerfile`
- 3-stage builds (builder → ORT fetch → runtime), `debian:bookworm-slim` base
- Multi-arch: `linux/amd64` + `linux/arm64` (ADR-048 §Container Images)
- `.forgejo/workflows/release.yml` builds and pushes on tag `v*`: `akh`, `akhomed`, `seshd` to `code.aopc.cloud/toasterson/`
- Also produces Linux x86_64 tarball + illumos IPS package

**Binary packaging:**
- `cargo-binstall` artifacts (release.yml: `packaging/build-binstall-archive.sh`)
- Release profile with LTO (Cargo.toml release profile)
- Platform targets: x86_64-linux (primary), aarch64-linux (ARM), x86_64-illumos

**ARM NEON SIMD optimization:**
- `src/simd/neon.rs` exists (246 lines) — NEON kernel for aarch64 VSA operations
- Runtime dispatch: AVX2 / NEON / generic fallback (VsaKernel trait)

**NLU model packaging:**
- `akh setup models` command (SetupAction::Models) downloads NER + LLM models
- `akh setup onnxruntime` installs ONNX Runtime
- Models stored under XDG data dirs, not bundled in binary

**Configuration:**
- `src/akh_config.rs` — first-class `akh.toml` configuration (ADR-053)
- `src/config.rs` — global akhomed config (TOML at `$XDG_CONFIG_HOME/akh-medu/config.toml`)
- Environment variable overrides throughout (XDG_DATA_HOME, RUST_LOG, etc.)
- `akh setup check` verifies NLU setup status

**Verdict:** ✅ All Alpha-1 deliverables shipped. The `akh init` command was renamed to
`akh setup` (SetupAction), which is a superset. Feature flags are build-arg configurable
via Dockerfile `ARG FEATURES`.

### Alpha-2 — CLI Completeness ✅ Shipped (surface renamed)

**CLI commands audited against `src/main.rs`:**

| Plan command | Current surface | Status |
|---|---|---|
| `akh awaken parse` | `akh acp awaken` (AcpOp::Awaken) | ✅ Exists (renamed) |
| `akh awaken resolve` | `akh acp awaken` | ✅ Combined into awaken verb |
| `akh awaken expand` | `akh acp awaken` | ✅ Combined |
| `akh awaken prerequisite` | `akh acp awaken` | ✅ Combined |
| `akh agent daemon` | `akh daemon start/stop/status` | ✅ Exists (DaemonAction) |
| `akh status` | `akh acp status` (AcpOp::Status) | ✅ Exists (renamed) |
| `akh explain` | `akh acp explain` (AcpOp::Explain) | ✅ Exists |
| `akh init` | `akh setup models/onnxruntime/check` | ✅ Exists (renamed) |

**Error messages:**
- All error paths use `miette` diagnostics (verified: `miette::IntoDiagnostic`, `#[diagnostic]` attributes throughout `src/config.rs`, `src/akh_config.rs`, and across the codebase)
- `akh_config.rs` provides helpful diagnostics with `help("...")` pointing operators to fixes

**Verdict:** ✅ All Alpha-2 deliverables shipped. The CLI surface was restructured by ADR-061
(P8 pivot): the ~75 cognitive subcommands were dropped (they are agent tools, not CLI verbs),
and the remaining control-plane ops were routed over ACP. The awaken pipeline was consolidated
into a single `akh acp awaken` verb with sub-actions.

### Alpha-3 — Integration Test Suite ✅ Partially shipped

**Existing tests in `tests/` directory:**
- `integration.rs` — general integration tests
- `persistence.rs` — persistence layer
- `wal_live.rs` — WAL live tests
- `adr091_acceptance.rs` — ADR-091 acceptance
- `hermeticity.rs` — hermeticity verification
- `relocation.rs` — relocation/snapshot tests
- `snapshot_live.rs` — snapshot live tests
- `schema_registry_live.rs` — schema registry
- `phase33_aggregation.rs`, `phase34_communicative.rs` — phase-specific tests
- `benchmark_chat.rs`, `harvest_live.rs` — benchmarks and harvest
- `tests/vm/` — VM-based tests

**What's covered:** persistence, relocation, hermeticity, schema, WAL, aggregation,
communicative phases. The daemon stability scenario (60s idle, no panics) is partially
covered by the existing daemon tests.

**What may be missing:** The plan's specific end-to-end scenarios (full bootstrap:
purpose → identity → expansion → prerequisite → curriculum) are exercised by the
bootstrap modules' own unit tests, but a single end-to-end test chaining them is not
explicitly present in `tests/`. Mock HTTP server infrastructure for external APIs is
not present (though CI runs are network-isolated).

**Verdict:** ✅ Partially shipped. Integration test infrastructure exists and covers
core scenarios. The plan's specific mock-HTTP-server scenario is not present, but
the project has moved to a different testing strategy (live tests gated by environment,
hermeticity tests for isolation). This is a design evolution, not a gap.

### Alpha-4 — Observability & Metrics ⚠️ Partially shipped — REAL GAP (Gap A)

**Structured logging:**
- `tracing` crate used throughout (verified: `tracing::instrument`, `tracing_subscriber` in all binaries)
- Log levels via `RUST_LOG` env var (values.yaml sets `RUST_LOG: "info"`)
- OTLP tracing shipped: `src/observability.rs` (Phase 38i) — exports to OTel Collector
- Values.yaml configures `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`, sampler settings
- JSON format available via `tracing_subscriber` configuration

**Prometheus `/metrics` endpoint (12 named metrics):**
- ❌ **Does not exist.** No `/metrics` HTTP endpoint anywhere in the codebase.
- ❌ No `akh_kg_entities_total`, `akh_ooda_cycles_total`, or any of the 12 named metrics.
- No Prometheus scrape endpoint.

**Health & readiness endpoints:**
- ❌ No `/healthz`, `/readyz`, `/status`, `/metrics` HTTP endpoints.
- ADR-061 (P8, 2026-06-16) **deliberately removed the entire HTTP surface** — REST,
  WebSocket, web UI, and inbound MCP were all retired. akhomed's only inbound surface
  is raw ACP/TCP.
- K8s probes were reworked to `tcpSocket` on the ACP port (values.yaml lines 137-150,
  ADR-061 §7).

**ServiceMonitor chart template:**
- ❌ No `servicemonitor.yaml` in `charts/akh-medu/templates/` (12 template files, none is
  a ServiceMonitor).

**Reconciliation with ADR-061:**
ADR-061 supersedes the *HTTP endpoints* portion of Alpha-4 but does NOT supersede the
*metrics* portion. The ADR removed REST/WS/web-UI/inbound-MCP — it did not say "we no
longer need metrics." The OTLP observability research doc
(`docs/ai/research/2026-04-26-otlp-observability.md`) acknowledges metrics as a future
phase: "Phase 38j-onwards we may want exemplars" and section 2.2 discusses
`opentelemetry_sdk::metrics` as the intended metrics path.

**Current observability state:**
- Tracing: ✅ OTLP/gRPC to in-cluster Collector (Tempo) — shipped (Phase 38i)
- Logs: ✅ stdout via `tracing_subscriber::fmt` — scraped by Promtail/Alloy into Loki
- Metrics: ❌ **Not shipped.** No OTel metrics SDK, no Prometheus endpoint, no 12 named
  metrics, no ServiceMonitor.

**Verdict:** ⚠️ Partially shipped. The HTTP endpoints (`/healthz`, `/readyz`, `/status`)
are **superseded by ADR-061** (no HTTP surface — probes use `tcpSocket` on ACP port).
Structured logging and OTLP tracing are shipped. **Prometheus metrics are a REAL
remaining gap** — the 12 named metrics, the `/metrics` endpoint, and the ServiceMonitor
template do not exist. The project's intended path is OTLP-native metrics (not Prometheus
pull), per the observability research doc, but no implementation exists yet.

### Alpha-5 — Kubernetes Deployment ✅ Shipped (design changed per ADR-048)

**Helm chart:** `charts/akh-medu/` with 12 template files (including `_helpers.tpl`):
- `akhomed-deployment.yaml`, `akhomed-service.yaml`, `akhomed-pvc.yaml`
- `seshd-deployment.yaml`, `seshd-service.yaml`, `seshd-pvc.yaml`, `seshd-ingress.yaml`
- `seshd-configmap.yaml`, `seshd-secret.yaml`
- `akhomed-known-hosts-configmap.yaml`, `akhomed-ssh-secret.yaml`
- `_helpers.tpl`

**Chart version:** `0.1.6`, appVersion `"0.7.0"` (Chart.yaml)

**StatefulSet vs. Deployment:**
The plan specifies a `StatefulSet` ("not Deployment — needs stable persistent storage").
ADR-048 (2026-04-14, §Key Design Choices §3) overrides this: **Deployment with `Recreate`
strategy** — "the knowledge graph (oxigraph + redb + HNSW) is a single-writer store; no
rolling update possible." This is a deliberate, documented design decision: a single
replica Deployment with Recreate achieves the same single-writer semantics as a
StatefulSet without the operational complexity of ordered rollout. The chart implements
this (`akhomed-deployment.yaml` line 11: `type: Recreate`).

**CI publishing:**
- `.forgejo/workflows/ci.yml` — `publish-chart` job publishes to `oci://code.aopc.cloud/toasterson`
  on every main/develop push (idempotent: checks if version already exists)
- `.forgejo/workflows/ci.yml` — `docker-seshd` job builds and pushes seshd image on every push
  (SHA tag + branch tag)
- `.forgejo/workflows/release.yml` — builds and pushes `akh`, `akhomed`, `seshd` images on
  tag `v*`, plus Linux tarball and illumos IPS package

**Live deployment:** Homelab Flux HelmRelease pins to SHA tags; documented in
`.anima/wip/019dc4de.md` and referenced in WI-24's text.

**Verdict:** ✅ Shipped with a deliberate design change. ADR-048 supersedes the plan's
StatefulSet choice — Deployment + Recreate strategy is the implemented and documented
architecture. The chart is live in the homelab cluster.

### Alpha-6 — Autonomous Bootstrap & Specialization ✅ Shipped

**Startup sequence (bootstrap pipeline):**
- `src/bootstrap/purpose.rs` — Phase 14a: parse purpose statement ✅
- `src/bootstrap/identity.rs` — Phase 14b: resolve identity, construct Psyche ✅
- `src/bootstrap/expand.rs` — Phase 14c: expand domain → skeleton ontology ✅
- `src/bootstrap/prerequisite.rs` — Phase 14d: prerequisite discovery, ZPD classification ✅
- `src/bootstrap/resources.rs` — Phase 14e-f: resource discovery + ingestion ✅
- `src/bootstrap/competence.rs` — Phase 14g: competence assessment (Dreyfus) ✅
- `src/bootstrap/orchestrator.rs` — Phase 14h: orchestrator meta-OODA loop ✅
- `src/bootstrap/knowledge_elicit.rs` — knowledge elicitation ✅
- `src/bootstrap/recipe.rs` — bootstrap recipe ✅

**OODA loop:**
- `src/agent/ooda.rs` — the synchronous OODA loop (deliberately synchronous per CLAUDE.md)
- Drive system: curiosity, coherence, completeness, efficiency drives implemented
- Metacognition: ZPD assessment, strategy adjustment
- Reflection: stalled goal reformulation
- Consolidation: memory persistence, working memory management

**Daemon mode:**
- `akh daemon start` (DaemonAction::Start) — enters long-running OODA loop
- `akh daemon status` — shows running state, cycles, triggers
- Graceful shutdown via SIGINT/SIGTERM (verified in daemon implementation)

**Self-directed specialization:**
- Drive-generated goals every idle cycle (configurable via `idle_cycle_secs` in config.toml)
- Personality shapes exploration (archetype system in `src/compartment/psyche.rs`)

**Verdict:** ✅ All Alpha-6 deliverables shipped. The full bootstrap pipeline (Phases 14a-14h)
is implemented. The daemon enters autonomous operation after bootstrap. The observability
of this (metrics for `akh_dreyfus_level`, `akh_zpd_distribution`, etc.) depends on Alpha-4
metrics, which is the open gap.

### Alpha-7 — Documentation ❌ Mostly open (Gap B)

**User-facing documentation:**

| Planned doc | Status |
|---|---|
| `README.md` update | ⚠️ Stale — references retired REST/WebSocket surface ("Serve knowledge over REST and WebSocket APIs"), `akh-medu init`/`akh-medu seed` (pre-ACP commands) |
| `docs/getting-started.md` | ❌ Does not exist |
| `docs/configuration.md` | ❌ Does not exist |
| `docs/deployment.md` | ❌ Does not exist |

**Developer documentation:**
- `docs/ai/architecture.md` — comprehensive architecture doc (749 lines) ✅
- `CLAUDE.md` — developer guide with module inventory ✅
- No "how to add a new tool" or "how to add a new phase" doc (but CLAUDE.md covers tool patterns)

**Verdict:** ❌ Mostly open. The architecture doc is excellent, but user-facing deployment
and configuration docs are missing. The README is stale post-ADR-061 (still mentions
REST/WS surface and pre-ACP CLI commands).

## 2. What Genuinely Remains

### Gap A — Prometheus/OTLP Metrics (from Alpha-4)

**What's missing:** The 12 named Prometheus metrics (`akh_kg_entities_total`,
`akh_ooda_cycles_total`, etc.), a metrics export mechanism (either `/metrics` HTTP
endpoint or OTLP metrics to the Collector), and a `ServiceMonitor` chart template.

**Why it matters:** The plan's success criteria (#5, #6, #7, #8) all depend on metrics
being observable. OTLP tracing covers *latency/dependency* observability but not
*domain-level* observability (KG size, goal counts, drive strengths, Dreyfus level).

**Recommended approach:** Per the OTLP observability research doc, implement metrics
via `opentelemetry_sdk::metrics` (OTLP-native, no HTTP endpoint needed — aligns with
ADR-061's no-HTTP-surface decision). Add a `ServiceMonitor` template pointing at the
OTel Collector's metrics receiver.

### Gap B — K8s Deployment Guide + README Refresh (from Alpha-7)

**What's missing:** `docs/deployment.md` (or `docs/deploy/k8s.md`), `docs/getting-started.md`,
`docs/configuration.md`, and a README.md refresh to remove retired REST/WS references and
update CLI examples to ACP-based commands.

**Why it matters:** The chart is published and live, but no user-facing instructions exist
for `helm install`, values configuration, secrets setup, or upgrade procedures.

## 3. Version Reconciliation

WI-3 was written when the project version was 0.6.0-in-progress (2026-02-24). The
versioning story has fundamentally changed:

| Dimension | WI-3 era (Feb 2026) | Current (Aug 2026) |
|---|---|---|
| App version | 0.6.0-in-progress | 0.7.0 (Cargo.toml) |
| Chart version | Did not exist | 0.1.6 (SemVer, independent of app version) |
| Release mechanism | Manual | Tag-triggered CI (`.forgejo/workflows/release.yml`) |
| Image registry | `ghcr.io/toasty/` (planned) | `code.aopc.cloud/toasterson/` (Forgejo) |
| Transport | REST + WS + MCP + ACP | ACP-only (ADR-061) |

**"Release Alpha" is no longer a meaningful milestone.** The project ships continuously
via tag-triggered CI — every `git tag vX.Y.Z && git push` produces Docker images, tarballs,
and illumos packages. Chart versions are bumped manually alongside template changes and
published by CI. The plan's success criteria (e.g., "docker run akh-medu starts daemon",
"helm install akh deploy/helm/akh-medu/") are met by the existing CI pipeline and homelab
deployment. The plan was overtaken by incremental implementation across Phases 14-48 and
the P8 ACP pivot.

## 4. Recommendations — and where each was filed

1. **Cancel WI-3** (`019db054-0da3-76a3-8725-0d1933aa32df`) as superseded by events.
2. **File Gap A** as a concrete work item: "Implement OTLP metrics for akhomed + seshd".
3. **File Gap B** as a concrete work item: "Write K8s deployment guide + refresh README".
4. **Mark the original plan** (`2026-02-23-release-alpha.md`) as `Superseded` linking this
   audit, per repo convention. *(Done in this PR.)*

| Remainder | Anima work item | Notes |
|---|---|---|
| Gap A — OTLP metrics + instrumentation | `019dcaad-0db5-7962-8cb7-6b396dcc27b1` (Phase 38j — OTLP metrics + Seshat instrumentation, pre-existing, parked) | Already covers the metrics path this audit identifies; no duplicate filed. Companion: `019dcaad-aa6f` (Phase 38k — OTel logs). Unparking is an operator decision. |
| Gap B — K8s deploy guide + README refresh | `019fd897-7bcc-7b11-9323-4ea512cc0f4e` (filed 2026-08-06 with acceptance criteria) | Docs-only: `docs/deploy/k8s.md` (chart install, values incl. arm64 tmpfs + uid/gid 999, tcpSocket/ACP probes, ADR-048 Recreate) + fix README's stale REST/WebSocket claim. |
| WI-3 cancellation | `019db054-0da3` set `cancelled` as superseded by this audit | Closes the 50-failed-sessions epic. |

## 5. Why WI-3 Failed 50 Times

WI-3's description is three lines plus a link to the plan doc. It's a release epic with
no acceptance criteria and no definition of done. No implementer could close it because
"K8s deployment" was already live — there was nothing to implement, but no way to prove
completion. The triage plan also referenced the wrong repo (Anima, not akh-medu), sending
implementers to non-existent paths (`crates/anima-*`, `proto/anima/`).

---

## Appendix: Evidence Index

| Claim | Source file | Key detail |
|---|---|---|
| Docker images built and pushed | `.forgejo/workflows/release.yml` | 3 images on tag `v*` |
| Chart published to OCI | `.forgejo/workflows/ci.yml` (lines 141-214) | `publish-chart` job |
| Chart version 0.1.6/appVersion 0.7.0 | `charts/akh-medu/Chart.yaml` | |
| 12 template files | `charts/akh-medu/templates/` | `_helpers.tpl` + 11 resources |
| Deployment + Recreate (not StatefulSet) | `charts/akh-medu/templates/akhomed-deployment.yaml` line 11 | ADR-048 §3 |
| ACP-only, no HTTP surface | ADR-061 (`docs/ai/decisions/061-acp-only-retire-legacy-surfaces.md`) | P8 rip-out |
| Probes use tcpSocket | `charts/akh-medu/values.yaml` lines 137-150 | |
| OTLP tracing shipped | `src/observability.rs` | Phase 38i |
| OTLP env vars in chart | `charts/akh-medu/values.yaml` lines 40-45 | |
| No metrics endpoint | `grep -r 'metrics' src/` (no Prometheus/metrics HTTP) | |
| No ServiceMonitor | `charts/akh-medu/templates/` (12 files, none is ServiceMonitor) | |
| NEON SIMD | `src/simd/neon.rs` (246 lines) | |
| Bootstrap pipeline | `src/bootstrap/{purpose,identity,expand,prerequisite,resources,competence,orchestrator}.rs` | |
| OODA loop | `src/agent/ooda.rs` | |
| Config system | `src/config.rs`, `src/akh_config.rs` | ADR-053 |
| Integration tests | `tests/` (12+ test files) | |
| README stale | `README.md` line 17: "Serve knowledge over REST and WebSocket APIs" | Pre-ADR-061 |
| No deployment docs | `docs/deployment.md`, `docs/getting-started.md` — do not exist | |
| CI source of truth | `.forgejo/workflows/` (Solstice CI) | Not `.github/` mirror |