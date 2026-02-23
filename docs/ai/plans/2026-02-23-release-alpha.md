# Release Alpha — Continuously Running Autonomous Agent on Kubernetes

> Date: 2026-02-24

- **Status**: Planned
- **Phase**: Release milestone (after Phase 14 completion)
- **Depends on**: Phase 14i (all bootstrap sub-phases complete)

## Goal

Deploy akh-medu as a long-running Kubernetes workload that autonomously bootstraps
its identity, continuously expands domain knowledge, and decides on its own what
to specialize in. The operator provides an initial purpose statement; from there
the akh runs indefinitely — building knowledge via the OODA daemon loop, pursuing
drive-generated goals, ingesting resources, and eventually branching into
sub-domains it discovers relevant. Observable via structured logs, Prometheus
metrics, and a health endpoint.

## Scope

Everything through Phase 14 (engine, KG, VSA, e-graphs, agent OODA, drives,
metacognition, identity bootstrap). The akh uses `akh agent daemon` mode
(already implemented) with its drive system, goal generation pipeline,
metacognitive monitoring, and reflection — all running on scheduled intervals
inside a Kubernetes pod with persistent storage.

## Sub-phases

### Alpha-1 — Build & Packaging (~2 days)

**Docker containerization**:
- Multi-stage Dockerfile: builder (`rust:latest`) + runtime (`debian-slim`)
- `TARGETPLATFORM`-specific build caches (`--mount=type=cache,target=/usr/local/cargo/registry,id=cargo-${TARGETPLATFORM}`)
- Feature flags: `default` profile for core, `full` for email+oxifed+wiki
- Configurable at build time via `--build-arg FEATURES=...`
- Entrypoint: `akh agent daemon` (long-running mode by default)

**Binary packaging**:
- `cargo install` support with proper `[[bin]]` section
- Release profile with LTO + strip for minimal binary size
- Platform targets: x86_64-linux (primary), aarch64-linux (ARM)

**Configuration**:
- `akh.toml` configuration file: data directory, log level, feature toggles, API keys (Semantic Scholar, OpenAlex, ConceptNet), Oxigraph path, redb path
- Environment variable overrides (`AKH_DATA_DIR`, `AKH_LOG_LEVEL`, etc.)
- `akh init` command to create default config + data directories
- Secrets handling: API keys from env vars or Kubernetes Secrets, never hardcoded

### Alpha-2 — CLI Completeness Audit (~1 day)

**Audit all CLI commands for completeness**:
- `akh awaken parse` — verify clean output
- `akh awaken resolve` — verify identity resolution feedback
- `akh awaken expand` — verify domain expansion progress
- `akh awaken prerequisite` — verify curriculum output
- `akh agent daemon` — long-running OODA loop with graceful shutdown (SIGINT/SIGTERM)
- `akh status` — show current agent state, goals, knowledge stats, uptime, KG size
- `akh explain` — provenance chain queries

**Error messages**:
- All error paths produce actionable miette diagnostics
- Missing config → helpful "run `akh init`" message
- Network failures → retry advice with specific endpoint info
- Missing API keys → which key, where to get it

### Alpha-3 — Integration Test Suite (~2 days)

**End-to-end test scenarios**:
- Full bootstrap: purpose → identity → expansion → prerequisite → curriculum
- OODA cycle: goal creation → observation → decision → action → memory
- Provenance chain: action → derivation → explanation
- Knowledge graph: entity creation → relation → query → explain
- Daemon stability: start daemon, let idle cycles run for 60s, verify no panics

**Test infrastructure**:
- Mock HTTP server (for Wikidata/Wikipedia/ConceptNet/Semantic Scholar)
- Deterministic VSA seed for reproducible tests
- Temporary data directories cleaned after each test
- CI-compatible: no network calls in `cargo test`

### Alpha-4 — Observability & Metrics (~2 days)

**Structured logging**:
- `tracing` with JSON formatter for log aggregation (Loki, Elasticsearch, etc.)
- Log levels: ERROR for failures, WARN for degraded operation, INFO for milestones (goal completed, concept learned, competence assessed), DEBUG for decisions, TRACE for VSA operations
- OODA cycle logging: each phase logged with timing + goal context

**Prometheus metrics** (exposed on `/metrics` HTTP endpoint):
- `akh_kg_entities_total` — gauge: total entity count in knowledge graph
- `akh_kg_triples_total` — gauge: total triple count
- `akh_ooda_cycles_total` — counter: OODA cycles executed
- `akh_ooda_cycle_duration_seconds` — histogram: cycle timing
- `akh_goals_active` — gauge: currently active goals
- `akh_goals_completed_total` — counter: goals completed
- `akh_goals_failed_total` — counter: goals failed
- `akh_drive_strength` — gauge (labels: curiosity/coherence/completeness/efficiency)
- `akh_dreyfus_level` — gauge: current competence assessment (0-4)
- `akh_zpd_distribution` — gauge (labels: known/proximal/beyond): concept counts per zone
- `akh_wm_entries` — gauge: working memory utilization
- `akh_consolidation_total` — counter: memory consolidations performed
- `akh_bootstrap_stage` — gauge: current bootstrap pipeline stage (0-8)

**Health & readiness endpoints** (lightweight HTTP on configurable port):
- `GET /healthz` — liveness: process alive, stores accessible
- `GET /readyz` — readiness: bootstrap complete, daemon loop running
- `GET /metrics` — Prometheus scrape endpoint
- `GET /status` — JSON: current goals, drive strengths, KG stats, uptime

### Alpha-5 — Kubernetes Deployment (~2 days)

**Helm chart** (`deploy/helm/akh-medu/`):

```
deploy/helm/akh-medu/
├── Chart.yaml
├── values.yaml
├── templates/
│   ├── deployment.yaml      # or statefulset.yaml
│   ├── service.yaml         # metrics + health endpoints
│   ├── configmap.yaml       # akh.toml
│   ├── secret.yaml          # API keys
│   ├── pvc.yaml             # persistent volume for KG + redb
│   ├── servicemonitor.yaml  # Prometheus ServiceMonitor CRD
│   └── _helpers.tpl
```

**StatefulSet** (not Deployment — needs stable persistent storage):
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: akh-medu
spec:
  replicas: 1                    # single akh instance
  serviceName: akh-medu
  template:
    spec:
      containers:
      - name: akh
        image: ghcr.io/toasty/akh-medu:alpha
        command: ["akh", "agent", "daemon"]
        args: ["--purpose", "$(AKH_PURPOSE)"]
        env:
        - name: AKH_DATA_DIR
          value: /data
        - name: AKH_LOG_LEVEL
          value: info
        - name: AKH_LOG_FORMAT
          value: json
        - name: AKH_METRICS_PORT
          value: "9090"
        envFrom:
        - secretRef:
            name: akh-medu-api-keys
        ports:
        - containerPort: 9090
          name: metrics
        livenessProbe:
          httpGet:
            path: /healthz
            port: metrics
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /readyz
            port: metrics
          initialDelaySeconds: 120     # bootstrap takes time
          periodSeconds: 60
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2"
        volumeMounts:
        - name: data
          mountPath: /data
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi            # KG + redb + Oxigraph
```

**Persistent storage layout** (`/data/`):
```
/data/
├── akh.toml              # config (from ConfigMap mount or init)
├── oxigraph/             # SPARQL store
├── redb/                 # durable key-value store
│   ├── symbols.redb
│   ├── triples.redb
│   └── provenance.redb
├── hnsw/                 # HNSW indices (similarity search)
└── logs/                 # structured log files (optional, prefer stdout)
```

**Secrets** (Kubernetes Secret, mounted as env vars):
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: akh-medu-api-keys
type: Opaque
stringData:
  AKH_CONCEPTNET_URL: "https://api.conceptnet.io"
  AKH_SEMANTIC_SCHOLAR_KEY: "<key>"
  AKH_OPENAL_KEY: "<key>"
  AKH_OPEN_LIBRARY_URL: "https://openlibrary.org"
```

**ServiceMonitor** (for Prometheus Operator):
```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: akh-medu
spec:
  selector:
    matchLabels:
      app: akh-medu
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

### Alpha-6 — Autonomous Bootstrap & Specialization (~2 days)

**Startup sequence** (what happens when the pod starts):

```
1. akh init                        # create data dirs if missing
2. akh agent daemon --purpose "…"  # enter daemon mode

   Daemon internally:
   ├─ Phase 14a: parse purpose statement
   ├─ Phase 14b: resolve identity, construct Psyche, Ritual of Awakening
   ├─ Phase 14c: expand domain → skeleton ontology (~150 concepts)
   ├─ Phase 14d: prerequisite discovery → ZPD classification → curriculum
   ├─ Phase 14e-14f: resource discovery + ingestion (curriculum-ordered)
   ├─ Phase 14g: competence assessment (Dreyfus level check)
   ├─ Phase 14h: orchestrator meta-OODA loop
   │   └─ if competence < target: re-enter 14d-14g
   │   └─ if competence >= target: bootstrap complete (/readyz → 200)
   └─ Continuous autonomous operation:
       ├─ Drive-generated goals every 30s idle cycle
       ├─ Curiosity drive → explore adjacent domains
       ├─ Coherence drive → resolve contradictions
       ├─ Completeness drive → fill knowledge gaps
       ├─ Metacognition → assess ZPD, adjust strategy
       ├─ Reflection → reformulate stalled goals
       └─ Consolidation → persist insights, free WM
```

**Self-directed specialization**:
The akh decides its own sub-domain focus based on drive signals:

- **Curiosity drive** detects stagnation in current domain → proposes exploration goals in adjacent areas (e.g., "compilers" → "type theory", "LLVM IR", "formal verification")
- **Completeness drive** identifies high-value knowledge gaps → generates ingestion goals for Proximal-zone concepts
- **Coherence drive** notices contradictions between sources → generates investigation goals
- **Metacognition** tracks which sub-domains have highest competence gain rate → biases goal generation toward productive areas
- **Personality shapes exploration**: Creator archetype builds practical knowledge, Sage archetype pursues theoretical depth, Explorer archetype branches into cross-domain connections

Over time the akh autonomously becomes a specialist: it discovers what it's good at learning, doubles down on productive knowledge areas, and prunes unproductive branches — all without operator intervention after the initial purpose statement.

**Observable via metrics**:
- `akh_dreyfus_level` climbs from 0 (Novice) toward target
- `akh_zpd_distribution{zone="known"}` grows as concepts are learned
- `akh_goals_completed_total` accumulates as the akh pursues and achieves goals
- `akh_drive_strength{drive="curiosity"}` spikes when the akh discovers a new sub-domain

### Alpha-7 — Documentation & Getting Started (~1 day)

**User-facing documentation**:
- `README.md` update: installation, quickstart, purpose statement examples
- `docs/getting-started.md`: step-by-step from install to first awakening
- `docs/configuration.md`: all config options documented
- `docs/deployment.md`: Kubernetes deployment guide (helm install, secrets setup, Prometheus/Grafana dashboards)

**Developer documentation**:
- Architecture overview (point to `docs/ai/architecture.md`)
- How to add a new tool
- How to add a new phase

**Example purpose statements**:
```
"You are the Architect of the System based on Ptah"
"Be like Gandalf — a GCC compiler expert"
"You are a knowledge curator inspired by Thoth, specializing in formal methods"
"Be like Athena — expert in distributed systems and consensus algorithms"
```

## Files to Create/Modify

| File | Change |
|------|--------|
| `Dockerfile` | NEW — multi-stage build, `akh agent daemon` entrypoint |
| `deploy/helm/akh-medu/Chart.yaml` | NEW — Helm chart metadata |
| `deploy/helm/akh-medu/values.yaml` | NEW — default values (purpose, resources, storage) |
| `deploy/helm/akh-medu/templates/*.yaml` | NEW — StatefulSet, Service, ConfigMap, Secret, PVC, ServiceMonitor |
| `src/config.rs` | NEW — configuration loading (akh.toml + env vars) |
| `src/agent/metrics.rs` | NEW — Prometheus metrics registry + HTTP endpoint |
| `src/agent/health.rs` | NEW — /healthz, /readyz, /status HTTP handlers |
| `src/main.rs` | Add `Commands::Init`, `Commands::Status`, `Commands::Health`; wire metrics port |
| `.github/workflows/release.yml` | NEW — CI: build + test + Docker push to ghcr.io |
| `tests/integration/` | NEW — end-to-end test scenarios |
| `README.md` | Update with installation + quickstart + k8s deployment |
| `docs/deployment.md` | NEW — Kubernetes deployment guide |
| `docs/getting-started.md` | NEW — step-by-step quickstart |
| `docs/configuration.md` | NEW — all config options |

## Success Criteria

1. `docker build -t akh-medu .` completes successfully
2. `docker run akh-medu` starts daemon, runs bootstrap, enters autonomous loop
3. `helm install akh deploy/helm/akh-medu/ --set purpose="You are the Architect based on Ptah"` deploys to k8s
4. Pod reaches `Ready` state after bootstrap completes
5. `/metrics` endpoint returns Prometheus-format metrics
6. `akh_kg_triples_total` grows continuously over 24 hours (knowledge being built)
7. `akh_goals_completed_total` > 0 within first hour (akh is pursuing and achieving goals)
8. `akh_dreyfus_level` increases over first week (competence improving)
9. Pod survives 7-day continuous run without OOM, crash, or store corruption
10. `akh status` (via `kubectl exec`) shows meaningful goal history, drive activity, and knowledge stats
11. All error paths produce helpful miette diagnostics (no panics, no bare unwrap)
12. `cargo test` passes all existing + new integration tests

## Non-Goals

- Multi-akh cluster orchestration (single pod for alpha)
- Web UI (metrics + logs + CLI are sufficient)
- Windows support
- Production hardening (rate limiting, auth, multi-tenancy)
- Phase 15+ features (alpha exercises phases 1-14 only)
- Ingress / public exposure (internal cluster service only)
