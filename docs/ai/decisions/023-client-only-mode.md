# ADR-023: Client-Only Build Mode for `akh` Binary

> Date: 2026-02-28
> Status: Accepted
> Relates to: Release Alpha, Deployment Architecture

## Context

The `akh` binary currently creates local `Engine` and `Agent` instances for most commands,
with `AkhClient` providing a `Local`/`Remote` abstraction used by only some commands (skills,
library, daemon control, workspace role-assign). This means the `akh` binary must have the
full engine stack (HNSW, redb, egg, petgraph, oxigraph, etc.) compiled in, even when a user
only wants to talk to a running `akhomed` server.

For deployment scenarios — especially multi-user or containerized setups — operators need a
lightweight CLI that connects to an already-running `akhomed` server without bundling the
entire engine.

### Constraints

- Must not break the existing default build (`cargo build --bin akh` unchanged)
- Must work with the existing `akhomed` HTTP+WebSocket server
- Must provide rich diagnostics when the server is unreachable
- Must support all CLI subcommands (50+) through HTTP transport
- Binary size reduction is a secondary benefit; primary goal is architectural separation

## Decision

Add a compile-time **`client-only` feature flag** so that
`cargo build --features client-only --bin akh` produces a binary that:

1. **Requires a running `akhomed` server** (discovered via PID file + health check)
2. **Routes ALL commands through HTTP** to the server
3. **Never creates a local `Engine` or `Agent`**
4. **Provides clear diagnostics** via `ClientError::NoServer` with miette help text

### Key Design Choices

#### Feature gating strategy

- `AkhClient::Local(Arc<Engine>)` variant is gated with `#[cfg(not(feature = "client-only"))]`
- `require_server()` function replaces `resolve_client()` under `client-only`
- `main.rs` uses a top-level `#[cfg(feature = "client-only")] return run_client_only(cli);`
  early return, keeping the existing dispatch untouched behind `#[cfg(not(...))]`

#### Shared API types

Shared `src/api_types.rs` defines request/response structs used by both `akhomed` handlers
and `AkhClient` remote methods. This avoids duplicating serialization logic and ensures
wire-format compatibility.

#### Commands that cannot work remotely

Some commands are inherently local or require capabilities not yet exposed via HTTP:
- **CSV/Text ingest** — requires local file parsing; JSON label format works via HTTP
- **Headless chat** — requires local NLU pipeline; WebSocket chat works via TUI
- **CalDAV sync** — requires direct HTTP to CalDAV server; `cal import` works via HTTP
- **Library watch** — filesystem watcher is local-only

These bail with informative miette diagnostics explaining the limitation and suggesting
alternatives.

#### NLU in client-only mode

The TUI connects to `akhomed` via WebSocket. The akhomed WS handler initialises an
`NluPipeline` per session (restoring the VSA parse ranker from durable storage) and runs
the full 4-tier cascade server-side. When `classify_intent()` cannot categorize user input,
the NLU pipeline attempts a structured parse; if successful, the input is ingested as a
fact; if not, it is escalated to a goal for agent investigation.

This means client-only builds get full NLU capability without any local ML models. The
available tiers depend on the server's compile-time features (`nlu-ml`, `nlu-llm`).

## Alternatives Considered

### A. Separate `akh-client` crate

Split the library into `akh-medu-core` and `akh-medu-client` crates, with `akh` depending
on only the client crate.

**Rejected:** Premature crate splitting adds workspace complexity. The single-crate approach
with feature gating achieves the same binary separation. Dead code elimination by the linker
handles unused engine code. Can revisit if compile times become a problem.

### B. Runtime mode switch (no feature flag)

Use a runtime `--remote-only` flag instead of compile-time gating.

**Rejected:** Doesn't achieve binary size reduction. Still links all engine dependencies.
Feature flag gives the compiler/linker the information needed to strip unused code.

### C. Separate `akh-remote` binary

Create a completely separate binary with its own `main.rs`.

**Rejected:** Massive code duplication for CLI parsing, output formatting, and error handling.
Feature-gated conditional compilation in a single `main.rs` is cleaner.

## Consequences

### Positive

- Lightweight client binary for remote-only deployments
- Clean architectural separation between client and engine concerns
- Existing default build is completely unchanged (zero regression risk)
- Shared API types ensure wire-format consistency
- Foundation for future multi-client scenarios (web UI, mobile, etc.)

### Negative

- ~800 lines of `run_client_only()` dispatch code mirrors the local dispatch
- Server must expose ~35 new endpoints to cover all commands
- Some commands unavailable in client-only mode (with clear diagnostics)

### Neutral

- NLU pipeline now runs server-side in akhomed's WS handler (implemented alongside client-only mode)
- Binary size improvement depends on linker DCE effectiveness

## Files Changed

- `Cargo.toml` — `client-only = []` feature
- `src/lib.rs` — `pub mod api_types`
- `src/api_types.rs` — shared request/response types (~700 lines)
- `src/client.rs` — feature-gated `Local` variant, `require_server()`, ~35 new remote methods
- `src/main.rs` — `run_client_only()` dispatch, feature-gated imports and helpers
- Agent/bootstrap types — added `Serialize`/`Deserialize` derives where needed
