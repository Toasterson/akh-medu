# Release Alpha — Deployable Test System

> Date: 2026-02-23

- **Status**: Planned
- **Phase**: Release milestone (after Phase 14 completion)
- **Depends on**: Phase 14i (all bootstrap sub-phases complete)

## Goal

Package phases 1-14 into a deployable, long-running test system. The akh should be able to: receive a purpose statement, construct an identity, expand domain knowledge, order a curriculum, and then run autonomously — all observable via TUI, CLI, and logs. This release milestone is not a new feature phase but an engineering checkpoint to ensure the system is testable, deployable, and monitorable in a real environment.

## Scope

Everything through Phase 14 (engine, KG, VSA, e-graphs, agent OODA, interaction, personal assistant, identity bootstrap). Phases 15+ are future work and not required for alpha.

## Sub-phases

### Alpha-1 — Build & Packaging (~2 days)

**Docker containerization**:
- Multi-stage Dockerfile: builder (rust:latest) + runtime (debian-slim)
- Platform-specific build caches for parallel CI
- Feature flags: `default` profile for core, `full` for email+oxifed+wiki
- Configurable at build time via `--build-arg FEATURES=...`

**Binary packaging**:
- `cargo install` support with proper `[[bin]]` section
- Release profile with LTO + strip for minimal binary size
- Platform targets: x86_64-linux (primary), aarch64-linux (ARM)

**Configuration**:
- `akh.toml` configuration file: data directory, log level, feature toggles, API keys (Semantic Scholar, OpenAlex, ConceptNet), Oxigraph path, redb path
- Environment variable overrides (`AKH_DATA_DIR`, `AKH_LOG_LEVEL`, etc.)
- `akh init` command to create default config + data directories
- Secrets handling: API keys from env vars or file, never hardcoded

### Alpha-2 — CLI Completeness Audit (~1 day)

**Audit all CLI commands for completeness**:
- `akh awaken parse` — verify clean output
- `akh awaken resolve` — verify identity resolution feedback
- `akh awaken expand` — verify domain expansion progress
- `akh awaken prerequisite` — verify curriculum output
- `akh run` — OODA loop with graceful shutdown (SIGINT/SIGTERM)
- `akh status` — show current agent state, goals, knowledge stats
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

**Test infrastructure**:
- Mock HTTP server (for Wikidata/Wikipedia/ConceptNet/Semantic Scholar)
- Deterministic VSA seed for reproducible tests
- Temporary data directories cleaned after each test
- CI-compatible: no network calls in `cargo test`

### Alpha-4 — Observability (~1 day)

**Logging**:
- Structured logging via `tracing` (already in use)
- Log levels: ERROR for failures, WARN for degraded operation, INFO for milestones, DEBUG for decisions, TRACE for VSA operations
- OODA cycle logging: each phase logged with timing

**Metrics** (optional, stretch):
- KG size (entities, relations, triples)
- VSA operations per cycle
- Memory usage (working memory entries)
- Cycle timing histogram

**Health check**:
- `akh health` command: verify stores are accessible, config is valid, API endpoints reachable

### Alpha-5 — Documentation & Getting Started (~1 day)

**User-facing documentation**:
- `README.md` update: installation, quickstart, purpose statement examples
- `docs/getting-started.md`: step-by-step from install to first awakening
- `docs/configuration.md`: all config options documented

**Developer documentation**:
- Architecture overview (point to `docs/ai/architecture.md`)
- How to add a new tool
- How to add a new phase

## Files to Create/Modify

| File | Change |
|------|--------|
| `Dockerfile` | NEW — multi-stage build with feature flags |
| `docker-compose.yml` | NEW — single-service compose for quick start |
| `src/config.rs` | NEW — configuration loading (akh.toml + env vars) |
| `src/main.rs` | Add `Commands::Init`, `Commands::Status`, `Commands::Health` |
| `.github/workflows/release.yml` | NEW — CI build + test + Docker push |
| `tests/integration/` | NEW — end-to-end test scenarios |
| `README.md` | Update with installation + quickstart |

## Success Criteria

1. `docker build -t akh-medu .` completes successfully
2. `docker run akh-medu awaken parse "You are the Architect based on Ptah"` produces expected output
3. `cargo test` passes all existing + new integration tests
4. `akh init && akh awaken parse "..."` works from a fresh install
5. `akh run` starts an OODA loop that runs for >1 hour without crashing
6. All error paths produce helpful miette diagnostics (no panics, no bare unwrap in user-facing code)

## Non-Goals

- Production hardening (rate limiting, auth, multi-tenancy)
- Web UI (TUI is sufficient for alpha)
- Windows support
- Distributed deployment
- Phase 15+ features
