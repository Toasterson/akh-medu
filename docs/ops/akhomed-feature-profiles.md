# akhomed feature profiles — which artifact each role gets

WI-339 removed `nlu-candle` from the root default features. Local Candle
inference (in-process GGUF via `candle-backend`, plus the `CandleTokenCounter`)
is now **opt-in per akhomed**, so a daemon that only serves remote-backed akhs
(Ollama cloud, OpenAI-compatible) builds without the vendored Candle stack,
tokenizers, and safetensors. This is the authoritative inventory of which
deployed/packaged akhomeds get the Candle-enabled artifact (`server-candle`)
and which get the lean one (`server`), per WI-339 scope items 1 and 5.

## The two daemon profiles

| Profile | Expands to | Local Candle inference | Use for |
|---------|-----------|------------------------|---------|
| `server` | `daemon + acp + postgres + tracing-otel + remote-executor` | **no** — candle/tokenizers/safetensors absent | Akhs whose chat/agentic backing is remote (`backend = "ollama"` or `"openai"`). The everyday/build-CI artifact. |
| `server-candle` | `server + candle-backend` | **yes** — in-process GGUF + Candle token counter | Any host that serves a `backend = "candle"` akh. The release-equivalent artifact. |

Optional GPU acceleration layers onto `server-candle`:

* `metal` — Apple Silicon (Mac mini). `server,candle-backend,metal`.
* `rocm` / `hip` — AMD GPU (ADR 068 / Phase 45). `server,candle-backend,rocm`.

### Fail-loud contract

A **lean** (`server`-only) akhomed refuses to start an akh whose `akh.toml`
selects `backend = "candle"` — it returns an explicit error naming
`server-candle` rather than silently routing the akh to Ollama/symbolic (the
"which model answered this stays answerable" invariant, ADR 070). So the
profile choice is enforced at runtime, not merely a packaging convention. See
the `cfg(not(feature = "candle-backend"))` arm in `src/bin/akhomed.rs`.

## Role inventory — which akhomed needs which profile

| Host / role | Where defined | Chat backing(s) served | Artifact / features |
|-------------|---------------|------------------------|---------------------|
| **Mac mini "armini"** (Apple Silicon, Metal) | `docs/ops/mac-akh-session-prompt.md` | `gemma-mac` → `backend = "candle"` (Metal GGUF); plus `thoth-mac` cloud | **Candle** — native: `--features server,candle-backend,metal` |
| **AMD box** (Gemma, three-akhs) | `docs/ops/three-akhs-postgres.md`, `examples/three-akhs/workspaces/gemma.toml` | Gemma → `backend = "candle"` | **Candle** — `--features server,candle-backend` (+`rocm` on the GPU box) |
| **Cloud container** `:cloud` | `docs/ops/akh-cloud-container.md` | `backend = "ollama"` (cloud model) only | **Lean** — `:cloud` build args drop candle; *note:* the shared `docker/akhomed.Dockerfile` **default** `FEATURES` still includes `nlu-candle`, so the candle-less `:cloud` variant must pass `--build-arg FEATURES=server,…` explicitly |
| **Cloud akhs in three-akhs** (Tecton, Thoth) | `docs/ops/three-akhs-postgres.md` | Ollama cloud only | **Lean** — `--features server` (runbook notes "Cloud akhs only need `server`") |
| **Release images / tarball / IPS** | `docker/akhomed.Dockerfile` (default `FEATURES`), `.forgejo/workflows/release.yml` | general-purpose release artifact — must be able to serve a candle akh | **Candle** — image builds keep `nlu-candle`; the illumos `illumos-pkg` job builds `--features server-candle` explicitly |
| **Routine CI** | `.forgejo/workflows/ci.yml` | type-check/test only | **Lean** — `cargo check --features server`, lean tests (fast lane) |

**Rule of thumb:** an akhomed that hosts a `backend = "candle"` akh gets
`server-candle`; everything else gets the lean `server`. GPU akhs add `metal` /
`rocm` / `hip` on top of `server-candle`.

## CI lanes

* **Lean (routine, fast):** `.forgejo/workflows/ci.yml` — `cargo check
  --features server` and the library/feature tests. Compiles nothing in the
  candle graph, so candle-side breakage is *not* caught here. This narrowing is
  an accepted WI-339 trade ("narrows routine builds, not release coverage").
* **Candle (release-equivalent, full):** `.forgejo/workflows/release.yml` and
  the `akhomed`/`seshd` image builds keep Candle in. The full candle-enabled
  build/test gate runs before every production release; it is preserved, not
  removed, by WI-339 (scope item 6).

## Where each profile is set

* Root feature definitions: `Cargo.toml` `[features]` (`server`, `server-candle`,
  `candle-backend`, `metal`, `rocm`, `hip`).
* Container image: `docker/akhomed.Dockerfile` `ARG FEATURES` (default includes
  `nlu-candle`; the `:cloud` lean variant overrides it).
* Release pipeline: `.forgejo/workflows/release.yml` (illumos `server-candle` +
  image builds).
* Hand-build runbook: `docs/ops/release.md` → "Building artifacts by hand".
