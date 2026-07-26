# ADR 080 — Lean `akh` client: cargo-binstall + remote ACP/TCP

> Date: 2026-06-30
> Status: Accepted (implemented + tested)
> Related: ADR 079 (release & packaging workflow — this extends it), ADR 061
>          (pure-ACP harness; ACP is the daemon's sole inbound surface), ADR 070
>          (selectable chat backing), `src/main.rs`, `src/client.rs`,
>          `src/acp/mod.rs`, `docker/akh.Dockerfile`

## Context

`akh` is a **pure ACP client** — the TUI/CLI that talks to an `akhomed` daemon;
all engine and LLM work happens daemon-side. Three gaps made it heavier and less
useful than it should be:

1. **It built the full LLM stack.** The crate's `default = ["nlu-candle", "acp"]`,
   so `cargo build --bin akh` dragged in candle-core/nn/transformers + the
   vendored candle fork — none of which the client uses. Slow to build, large
   binary, GPU/ML deps for a network client.
2. **No prebuilt distribution.** Installing `akh` meant compiling the whole crate.
   There was no `cargo binstall` path.
3. **No way to reach a *remote* daemon.** Every client path resolved the daemon
   from a **local PID file** (`discover_server`), so `akh` could only talk to an
   `akhomed` on the same host — even though the ACP TCP transport (`serve_tcp_*`,
   `ext_request_tcp`, `connect_tcp_retrying`) already existed and was tested.

A key finding de-risked all three: the library is **already correctly
feature-gated** — `cargo tree --no-default-features --features acp -i candle-core`
returns "no match", and the lean build compiles clean. So a lean client needs no
code surgery, only the right build flags.

## Decision

### Lean client build
Build `akh` with `--no-default-features --features client` everywhere it ships.
A new `client = ["acp"]` feature names the lean profile (ACP transport only, no
nlu-candle/ORT). Applied to `docker/akh.Dockerfile` and both release jobs. The
crate **default features are unchanged** (keeping `nlu-candle` for `akhomed` and
the existing test/CI surface) — the client is lean by explicit flag, not by
flipping a global default that would ripple into the daemon and develop CI.

### cargo-binstall
`[package.metadata.binstall]` points at the Forgejo release asset for the lean
`akh` (`akh-{version}-{target}.tar.gz`, binary at the archive root). Only `akh` is
binstalled: `akhomed` is excluded by `required-features = ["server"]` and
`generate_qa_dataset` is now gated behind `required-features = ["training-data"]`,
so under default features `akh` is the **sole** resolvable bin. The release builds
the per-target archives for `x86_64-unknown-linux-gnu` (extracted from the lean
`akh` image) and `x86_64-unknown-illumos` (native), via
`packaging/build-binstall-archive.sh`.

### Remote ACP/TCP
A global `--remote <host:port>` flag (default `AKH_REMOTE` env) makes the client
target a remote daemon directly: `ServerInfo::remote_tcp(addr)` builds a discovery
record with `acp_tcp = Some(addr)` and no socket, and a shared `resolve_server`
helper returns it (skipping local PID discovery) for every client path —
data-plane (`akh acp …`), control-plane (`akh daemon/config/…`), and chat (TUI +
headless). The remote daemon must serve ACP/TCP, which it already supports via
`AKH_ACP_TCP_PORT` / `AKH_ACP_TCP_BIND` (ADR 061, used by the Helm chart).

## Consequences

**Positive**
- `cargo binstall akh-medu` installs a small, prebuilt, candle-free `akh` from the
  Forgejo release on Linux and illumos.
- `akh --remote host:port chat` / `… acp status` drives an `akhomed` across the
  network — the client and daemon need not be co-located.
- The lean client builds much faster and ships without GPU/ML runtime deps.
- No change to daemon or develop-CI behaviour (default features untouched).

**Negative / caveats**
- **binstall source-fallback is heavy.** If no prebuilt matches the host triple,
  binstall falls back to `cargo install`, which uses *default* features (candle).
  Mitigated by shipping prebuilts for the common targets; documented.
- **`generate_qa_dataset` no longer builds by default.** It's a training tool;
  build it with `--features training-data`. (It was already uncaught by Solstice
  CI, which is scoped to `cargo check -p seshat`.)
- **`--remote` does no auth/TLS.** ACP/TCP is plaintext; expose it only on trusted
  networks or front it with an mTLS tunnel (consistent with the existing
  akh-pg-remote-mTLS guidance). A native ACP auth story is future work.

## Validation
- `cargo tree --no-default-features --features acp -i candle-core` → candle absent.
- `cargo check --no-default-features --features client --bin akh` → clean.
- New tests pass: `client::remote_tcp_*` (2) and `acp::acp_ext_data_plane_over_tcp`
  (drives `_akh/meta/capabilities` + an unknown-method error over a real TCP
  socket via `serve_tcp_loop` + `ext_request_tcp`).
- `cargo metadata` confirms `akh` is the only default-feature bin.
- Incidental: fixed a pre-existing broken unit test
  (`goal_generation::collect_signals_includes_watches` was missing the
  `predicates` arg) that the lean test build surfaced.
