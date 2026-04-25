# Phase 39 — Distributed Inference Fabric

> Date: 2026-04-25
> Status: Planned
> Phase: 39
> Depends on: Phase 26b (Candle backend), Phase 28 (`n akh` semantic router)
> Related: Phase 38 (Relocatable Akh — same deployment topology), Phase 36 (delegation)
> ADR: [050-distributed-inference-fabric](../decisions/050-distributed-inference-fabric.md)
> Anima ticket: `019dc599-ff37-7cf0-b1dc-0ea5d774c20c`

## Executive Summary

Phase 38 separates *where the Akh's state lives* (Postgres truth, workspace-on-disk,
3-tier sleep) from *which host runs the daemon process*. Phase 39 takes the next
step: separate *where compute happens* from either of those. The daemon stays
where its state is — on a Talos node, on the M2 in the cellar, on a peer's
laptop — but its LLM and embedding inference can route to any reachable backend
that satisfies the request's capability profile.

The core artefact is a `LlmBackend` trait with three primary implementations
(`CandleLocal`, `CandleRemote`, `OpenRouter`) and one optional / conditional
backend (`Anthropic`, shipped if convenient and dropped if it adds maintenance
load), a reservation protocol so multiple Akhs don't fight over the 7900 XTX
desktop, a discovery + health loop, and a Phase 28 router extension that learns
latency / cost / availability per backend.

A non-negotiable property: **no native C++ in the inference path**. Candle
hosts everything, including the AMD ROCm / HIP variant. We accept that
Candle-on-ROCm is experimental today — see §[Candle on ROCm](#candle-on-rocm-the-hard-deliverable)
for the remediation plan — in exchange for a unified Rust toolchain and a
hidden-state extraction API that survives the wire.

## Motivation

### Where state lives ≠ where compute happens

The current architecture conflates the two. `llama-cpp-2` (and the in-progress
Candle backend at `src/nlu/candle_backend.rs`) live in-process. If the daemon
runs on the Talos cluster, all inference runs on the cluster's CPUs. If the
daemon runs on the M2, all inference runs on Apple Silicon. The 7900 XTX
desktop in the basement — 24 GB VRAM, the only host that can actually run
Qwen2.5-7B at interactive speed — sits idle unless we move the entire daemon
onto it, which loses the laptop's mobility and the cluster's redundancy.

Phase 39 inverts the relationship:

```
                Phase 38 says "the daemon goes where state goes."
                Phase 39 says "compute goes where the work is."

  ┌──────────────────────────────────────────────────────────────┐
  │  Akh daemon (M2, cellar)                                     │
  │   ├─ State: Postgres + workspace dir + warm/cold sleep       │
  │   └─ Compute requests (default chain):                      │
  │       ├─ Tier-1/2 NLU      → CandleLocal (Metal, M2 GPU)    │
  │       ├─ Heavy synthesis   → CandleRemote (7900 XTX desktop) │
  │       ├─ Cheap distillation → OpenRouter (paid, per-token)   │
  │       └─ (optional rung)   → Anthropic — conditional         │
  │                              fallback; ships if convenient,  │
  │                              dropped if it causes maintenance│
  │                              load (auth/caching/cost shifts) │
  └──────────────────────────────────────────────────────────────┘
```

### Why the 7900 XTX matters

A Radeon RX 7900 XTX has 24 GB of VRAM and is the sole homelab GPU capable of
running Qwen2.5-7B Q4_K_M with hidden-state extraction at interactive speed.
With Phase 26d the neural→VSA bridge needs that hidden state; with Phase 26c
PolarQuant we want a 7B model, not 1.5B. Today the desktop is unused because
moving the whole daemon there sacrifices the M2's mobility.

### Why OpenRouter

OpenRouter (`https://openrouter.ai`) provides a single API key against ~100
hosted OSS models — Llama-3.1-70B, Qwen2.5-72B, DeepSeek-V3 — billed per
token. For Phase 32 (knowledge extraction) and Phase 27 (training data
distillation), we want bulk OSS-LLM access without standing up our own
70B-class infra. The existing `src/extraction/backends/api.rs` already
points at an OpenAI-compatible endpoint and `ExternalApiConfig::base_url`
accepts `https://openrouter.ai/api/v1` — we generalise that pattern into a
proper backend.

### Why Anthropic is an optional / conditional fallback

Anthropic API gives access to Claude (the same family the operator uses
interactively). It is the most expensive option in the chain and is treated
as **optional**: it ships if convenient and is dropped if it causes
maintenance load (auth changes, prompt-caching API shifts, cost concerns).
The primary stack is Candle (local + remote) and OpenRouter; Anthropic is an
additional rung at the bottom of the fall-through chain rather than a
first-class backend. When present, it is used only when the on-prem chain is
unavailable *and* OpenRouter cannot satisfy the capability requirement (e.g.
when `n akh` decides only frontier reasoning will do). The cost ceiling is
enforced by `CostGuard` (existing pattern from Phase 36) and by the per-Akh
budget in `akh.toml`.

### No native C++

The C++ stack in the inference path (today: `llama-cpp-2`, transitively
llama.cpp + ggml) has bitten us repeatedly:

- `llama_get_embeddings()` only returns the final layer; per-layer hidden
  states require `llama_set_eval_callback()` (C API) which the Rust crate
  does not expose. Phase 26d cannot reach in.
- GGML assertion failures call `abort()`. Rust's `catch_unwind` does not
  intercept C aborts, so a malformed prompt or KV-cache corruption takes
  the whole daemon down.
- The CMake toolchain bloats the build — we ship to Talos via OCI, and a
  CMake-flavoured base image is far heavier than `cargo build`.

Phase 26b started the migration (`candle-backend` feature flag is already
in `Cargo.toml:117-119`); Phase 39 finishes it by making Candle the only
inference path and giving every backend — including the remote ones — the
same Rust-typed surface.

## Goals & Non-Goals

### Goals

- One `LlmBackend` trait, four implementations, callable from anywhere
  in the daemon that needs LLM inference.
- Reservation protocol with leases + auto-release for the 7900 XTX (and
  any other scarce backend the operator marks as such).
- Discovery + health-check loop so Akhs don't keep dispatching to a
  backend that's been offline for 30 minutes.
- Phase 28 bandit extended with backend-aware features.
- Cost tracking + per-Akh daily budget enforcement.
- Hidden-state extraction over the wire for `CandleRemote`.
- Acceptance criterion for the ROCm path: Qwen2.5-7B Q4_K_M at >5 tok/s
  on the 7900 XTX with hidden-state extraction available, falling back to
  CPU on runtime ROCm errors without crashing the daemon.

### Non-goals

- Hailo-8 / Coral / Edge TPU on Raspberry Pi. Those target embedding
  / NLU workloads (Phase 14, Phase 33), not LLM serving.
- Multi-tenant scheduling at FAANG scale. Concurrency is bounded
  (≤ ~5 concurrent Akhs, ≤ ~3 concurrent on a single backend).
- Training over the wire. Phase 27 (Burn LoRA training) stays local
  to whichever node owns the dataset.
- Fancy load balancing. The reservation protocol is "first-fit with
  fall-through chain", not a min-cost flow.

## Architecture Overview

```
                     ┌──────────────────────────────────────────────┐
                     │  n akh Router (Phase 28)                     │
                     │   ├─ classifies query → archetype features   │
                     │   ├─ LinUCB picks BackendId, not Tier        │
                     │   └─ context vector now includes:            │
                     │        latency_p95, cost_cents/tok, avail   │
                     └────────────────┬─────────────────────────────┘
                                      │
                                      ▼
                     ┌──────────────────────────────────────────────┐
                     │  BackendRegistry                             │
                     │   ├─ DashMap<BackendId, Box<dyn LlmBackend>> │
                     │   ├─ healthcheck loop (every 15s)            │
                     │   ├─ advertisement cache (capabilities, ETA) │
                     │   └─ ReservationManager (leases per backend) │
                     └────┬──────────────┬──────────────┬───────────┘
                          │              │              │
              ┌───────────▼──┐  ┌────────▼──────┐  ┌────▼──────────┐
              │ CandleLocal  │  │ CandleRemote  │  │ OpenRouter /  │
              │ Metal/CUDA/  │  │ gRPC over     │  │ Anthropic     │
              │ ROCm/CPU     │  │ TLS to peer   │  │ HTTPS/REST    │
              └──────────────┘  └───────────────┘  └───────────────┘
```

The Akh daemon owns the registry; the registry owns the backends. The
router consults the registry for capability + health + cost on every call.

### Component map (paths in the repo)

| Component | Path |
|---|---|
| Existing Candle skeleton | `src/nlu/candle_backend.rs` |
| Existing llama-cpp-2 (to remove in 39d) | `src/nlu/llm_translator.rs:53-56,83-85,206-211` |
| OpenAI-compatible HTTP backend (template for OpenRouter) | `src/extraction/backends/api.rs` |
| Triple-extraction backend selector (pattern reuse) | `src/extraction/mod.rs:67-141` |
| Cargo features for Candle | `Cargo.toml:113-150` |
| Phase 28 router target | `src/nlu/router.rs` (new in Phase 28) |
| New Phase 39 root module | `src/inference/` (new tree) |

## The `LlmBackend` Trait

Single trait, async where it matters, sync for cheap introspection.

```rust
//! src/inference/backend.rs

use std::time::Duration;

use async_trait::async_trait;
use futures::stream::BoxStream;
use miette::Diagnostic;
use thiserror::Error;

/// Unique identifier for a backend instance in the registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BackendId(pub u32);

/// Backend kinds — used for routing features and provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum BackendKind {
    CandleLocal,
    CandleRemote,
    OpenRouter,
    Anthropic,
}

/// Capabilities advertised by a backend at registration time and refreshed
/// on every healthcheck. Captures everything the router needs to decide.
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    pub kind: BackendKind,
    pub display_name: String,
    /// Supported model identifiers (e.g. "qwen2.5-7b-q4km", "claude-haiku-4").
    pub models: Vec<ModelId>,
    /// Largest model size this backend will accept, in billions of params.
    pub max_model_b_params: f32,
    /// Maximum context window across supported models, in tokens.
    pub max_context_tokens: u32,
    /// True when the backend can return per-layer or final hidden states.
    /// Required by Phase 26d neural-bridge probes.
    pub hidden_state_extraction: HiddenStateSupport,
    /// Quantisation levels accepted (`Q4_K_M`, `Q8_0`, `BF16`, …).
    pub quantizations: Vec<Quantization>,
    /// Cost in *thousandths of a cent* per input token. 0 for on-prem.
    pub cost_input_millicents_per_tok: u32,
    pub cost_output_millicents_per_tok: u32,
    /// Whether the backend requires a reservation lease before completion.
    pub requires_reservation: bool,
    /// Maximum lease duration the backend will grant.
    pub max_lease: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HiddenStateSupport {
    /// No hidden states available (e.g. `OpenRouter`, `Anthropic`).
    None,
    /// Final layer only, post-norm pre-lm_head. Cheap.
    FinalOnly,
    /// Any layer index in 0..n_layers. Most flexible; expensive on the wire.
    PerLayer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantization { F32, BF16, F16, Q8_0, Q4_K_M, Q4_0, Q3_K_M }

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelId(pub String);

/// Health snapshot. Updated by the healthcheck loop.
#[derive(Debug, Clone)]
pub struct Health {
    pub status: HealthStatus,
    pub last_seen_unix_ms: u64,
    pub current_load: f32,         // 0.0 idle, 1.0 saturated
    pub queue_depth: u32,
    pub eta_to_availability: Option<Duration>,
    pub rolling_p95_latency: Option<Duration>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// Reachable, accepting requests.
    Live,
    /// Reachable but reserved by another Akh.
    Reserved,
    /// Reachable but at capacity (queue full).
    Saturated,
    /// Last `n` healthchecks failed; mark stale, skip in router.
    Stale,
    /// Backend disabled by configuration.
    Disabled,
}

/// Lease handle for reservation. Drop = auto-release (best effort);
/// the registry's expiry watchdog is the source of truth.
#[derive(Debug)]
pub struct Lease {
    pub backend: BackendId,
    pub lease_id: u128,            // 128-bit random, encoded as Uuid for the wire
    pub expires_unix_ms: u64,
    pub holder: AkhId,             // who reserved it
    /// Sender used by the lease watcher to notify on expiry.
    pub(crate) on_expiry_tx: tokio::sync::oneshot::Sender<()>,
}

/// Akh identity — used for reservation provenance and cost attribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct AkhId(pub uuid::Uuid);

/// One inference request. Same shape for sync and stream.
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub model: ModelId,
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub stop: Vec<String>,
    /// If set, request hidden-state extraction at the given layer (or `None`
    /// for "last layer only"). Backend may reject if unsupported.
    pub hidden_states: Option<HiddenStateRequest>,
    /// Optional grammar / logits-processor pin for structured output.
    pub structured: Option<StructuredOutputSpec>,
    /// Lease the request is being served under (None for unreserved backends).
    pub lease: Option<Lease>,
    /// Caller's Akh id, for cost attribution and audit.
    pub akh: AkhId,
}

#[derive(Debug, Clone)]
pub struct HiddenStateRequest {
    pub layer: Option<usize>,      // None = final
    pub last_token_only: bool,     // true => single position [hidden_dim]
}

#[derive(Debug, Clone)]
pub enum StructuredOutputSpec {
    AbsTreeJson,                   // Phase 14 grammar-constrained
    JsonSchema(serde_json::Value),
    BraceDepth,                    // brace-balance heuristic (existing fallback)
}

#[derive(Debug, Clone)]
pub struct InferenceResponse {
    pub text: String,
    pub tokens_in: u32,
    pub tokens_out: u32,
    pub wall_time: Duration,
    pub model_used: ModelId,
    pub backend_used: BackendId,
    pub hidden_states: Option<HiddenStateBlob>,
    pub cost_millicents: u64,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone)]
pub enum FinishReason { Stop, MaxTokens, Eos, Cancelled, Error(String) }

/// Hidden states travel as a typed blob — see §"Hidden state wire format".
#[derive(Debug, Clone)]
pub struct HiddenStateBlob {
    pub layer: usize,
    pub n_tokens: u32,
    pub hidden_dim: u32,
    pub dtype: HiddenDType,         // F32 (default), BF16, F16
    pub data: bytes::Bytes,         // row-major [n_tokens, hidden_dim]
    pub source_backend: BackendId,
    pub schema_version: u16,        // protobuf schema version, see §wire format
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HiddenDType { F32, BF16, F16 }

/// One streamed delta — text chunk plus optional metadata.
#[derive(Debug, Clone)]
pub struct TokenDelta {
    pub text: String,
    pub tokens: u32,
    pub finish: Option<FinishReason>,
    pub error: Option<String>,
}

/// The trait every backend implements.
#[async_trait]
pub trait LlmBackend: Send + Sync {
    fn id(&self) -> BackendId;

    fn capabilities(&self) -> &BackendCapabilities;

    /// Cheap, synchronous snapshot of the last healthcheck's result.
    /// The actual probe runs in the registry loop, not in this method.
    fn health(&self) -> Health;

    /// Reserve the backend for `lease` duration. Backends that
    /// `requires_reservation == false` return an unconditional lease.
    async fn reserve(&self, holder: AkhId, lease: Duration) -> Result<Lease, BackendError>;

    /// Release a lease early. Called by RAII guards and by the agent on
    /// successful completion. Idempotent.
    async fn release(&self, lease: Lease) -> Result<(), BackendError>;

    /// Synchronous (single-shot) completion.
    async fn complete(&self, req: InferenceRequest) -> Result<InferenceResponse, BackendError>;

    /// Streaming completion. The stream item carries token deltas plus an
    /// optional terminal `HiddenStateBlob` on the final delta. If the wire
    /// dies mid-stream, the next item is `Err(BackendError::StreamInterrupted{..})`
    /// and the caller decides whether to fall through.
    async fn stream(
        &self,
        req: InferenceRequest,
    ) -> Result<BoxStream<'static, Result<TokenDelta, BackendError>>, BackendError>;
}

/// Errors are miette-diagnosable. Subsystems chain via `#[error(transparent)]`
/// per the project error pattern (CLAUDE.md).
#[derive(Debug, Error, Diagnostic)]
pub enum BackendError {
    #[error("backend {0:?} is offline")]
    #[diagnostic(
        code(akh::inference::offline),
        help("Check the backend's health endpoint or remove it from the chain in akh.toml")
    )]
    Offline(BackendId),

    #[error("backend {backend:?} reserved by {holder:?} until {expires_unix_ms}")]
    #[diagnostic(
        code(akh::inference::reserved),
        help("Wait for the lease to expire, or fall through to the next backend in the chain")
    )]
    Reserved { backend: BackendId, holder: AkhId, expires_unix_ms: u64 },

    #[error("model {0:?} not supported by backend {1:?}")]
    #[diagnostic(code(akh::inference::unsupported_model))]
    UnsupportedModel(ModelId, BackendId),

    #[error("hidden state extraction not supported by backend {0:?}")]
    #[diagnostic(
        code(akh::inference::no_hidden_states),
        help("Route this request to a CandleLocal or CandleRemote backend")
    )]
    HiddenStatesUnsupported(BackendId),

    #[error("daily cost budget exceeded for Akh {akh:?} on backend {backend:?}")]
    #[diagnostic(
        code(akh::inference::budget_exceeded),
        help("Increase the budget in akh.toml [inference.budget] or wait for daily reset")
    )]
    BudgetExceeded { akh: AkhId, backend: BackendId },

    #[error("stream interrupted at token {tokens_so_far}: {cause}")]
    #[diagnostic(code(akh::inference::stream_interrupted))]
    StreamInterrupted { tokens_so_far: u32, cause: String },

    #[error("backend {0:?} is at capacity (queue depth {1})")]
    #[diagnostic(code(akh::inference::saturated))]
    Saturated(BackendId, u32),

    #[error("rate limited by upstream provider, retry-after {retry_after_secs}s")]
    #[diagnostic(code(akh::inference::rate_limited))]
    RateLimited { retry_after_secs: u64 },

    #[error(transparent)]
    #[diagnostic(transparent)]
    Wire(#[from] WireError),
}

#[derive(Debug, Error, Diagnostic)]
pub enum WireError {
    #[error("HTTP {status} from {url}: {body}")]
    #[diagnostic(code(akh::inference::wire::http))]
    Http { status: u16, url: String, body: String },

    #[error("gRPC error: {0}")]
    #[diagnostic(code(akh::inference::wire::grpc))]
    Grpc(String),

    #[error("schema mismatch: expected version {expected}, got {got}")]
    #[diagnostic(
        code(akh::inference::wire::schema_mismatch),
        help("Update the candle-remote crate version on both ends")
    )]
    SchemaMismatch { expected: u16, got: u16 },
}
```

### Why these signatures

- `reserve` returns `Lease`, not `LeaseId`, so the caller carries a typed
  RAII guard. `Lease::Drop` triggers a best-effort release; the registry's
  watchdog enforces expiry deterministically (the wire might be down).
- `complete` and `stream` both take `InferenceRequest` (not `&str`) so
  hidden-state requests, leases, and akh attribution are *part* of the
  payload, not side channels.
- `BackendCapabilities` is a snapshot — backends rebuild it on every
  healthcheck, so model lists and quantisations can change without
  re-registering.
- `BackendError` is `thiserror + miette` per CLAUDE.md, with `code()` on
  every variant. The router pattern-matches `BackendError` to decide
  fall-through (e.g. `Offline | Reserved | RateLimited` → next chain entry,
  `UnsupportedModel` → bubble up because no fall-through will help).

## The Initial Backends

The three **primary** backends are `CandleLocal`, `CandleRemote`, and
`OpenRouter`. `Anthropic` is shipped as an **optional / conditional**
fallback rung — present if convenient, dropped if it causes maintenance load
(auth changes, prompt-caching API shifts, cost concerns). Phase 39 must
build cleanly with the Anthropic backend disabled.

### CandleLocal

In-process Candle on the daemon's host. The existing
`src/nlu/candle_backend.rs` becomes one impl behind the trait; `LocalLlm`
becomes the legacy adapter and is removed in sub-phase 39d.

| Field | Value |
|---|---|
| `kind` | `BackendKind::CandleLocal` |
| `models` | whatever's in `models/llm/`; default Qwen2.5-1.5B Q4_K_M |
| `max_model_b_params` | 7.0 (depends on host: 1.5 on M2 16GB, 7.0 on 7900 XTX) |
| `max_context_tokens` | model-dependent (32k for Qwen2.5) |
| `hidden_state_extraction` | `PerLayer` |
| `quantizations` | `[Q4_K_M, Q8_0, BF16, F16, F32]` |
| `cost_*_millicents_per_tok` | `0` |
| `requires_reservation` | `true` *only if* `policy.scarce == true` in `akh.toml` |

**Device selection** at construction:

```rust
let device = if cfg!(feature = "rocm") {
    Device::rocm_if_available(0)?
} else if cfg!(target_os = "macos") {
    Device::metal_if_available(0)?
} else if cfg!(feature = "cuda") {
    Device::cuda_if_available(0)?
} else {
    Device::Cpu
};
```

`metal_if_available` and `cuda_if_available` already exist in Candle 0.10
(see `Cargo.toml:117-119`). `rocm_if_available` is the gap — see
§[Candle on ROCm](#candle-on-rocm-the-hard-deliverable).

**Auth**: none (local).

**Cost metering**: zero per-token cost, but the registry still charges
*lease seconds* against the Akh's budget so we can express "the M2 is
worth $0.50/hr to this Akh" without per-token pricing.

### CandleRemote

A `CandleLocal` running on a different host, talking to its peer over gRPC
(tonic) with TLS. The peer exposes the same `LlmBackend` surface. The
remote endpoint is itself an Akh daemon running with `[server.candle_serve]
enabled = true`, or a stripped-down "candle-only" service binary.

| Field | Value |
|---|---|
| `kind` | `BackendKind::CandleRemote` |
| `models` | reported by the peer at handshake |
| `hidden_state_extraction` | `PerLayer` (transmits `HiddenStateBlob` on stream end) |
| `cost_*_millicents_per_tok` | `0` (on-prem peer) |
| `requires_reservation` | `true` (this is exactly where the 7900 XTX lives) |
| `max_lease` | `Duration::from_secs(600)` default, configurable per-peer |

**Auth**: mutual TLS with peer-pinned certificates plus an optional
shared-secret bearer token (HS256, expires per-call). Cert and token
configured per-peer in `akh.toml`. No PKI required for homelab; the
operator copies cert fingerprints into config.

**Cost metering**: lease-seconds charged against the Akh's budget. The
peer reports its own resource cost (electricity, lease conflicts) via the
healthcheck advertisement; this only matters if multiple Akhs share the
peer with different budget weights.

**Wire**: see §[Hidden state extraction over the wire](#hidden-state-extraction-over-the-wire).

### OpenRouter

OpenAI-compatible HTTPS to `https://openrouter.ai/api/v1/chat/completions`
and `…/completions`. Streaming uses Server-Sent Events. The existing
`src/extraction/backends/api.rs` already speaks this dialect — Phase 39
generalises it into a backend rather than an extraction-only path.

| Field | Value |
|---|---|
| `kind` | `BackendKind::OpenRouter` |
| `models` | configurable allowlist; defaults to `["meta-llama/llama-3.1-70b-instruct", "qwen/qwen-2.5-72b-instruct", "deepseek/deepseek-chat"]` |
| `hidden_state_extraction` | `None` |
| `cost_*_millicents_per_tok` | per-model from OpenRouter's `/models` endpoint, refreshed daily |
| `requires_reservation` | `false` |
| `max_lease` | n/a |

**Auth**: bearer token via `AKH_OPENROUTER_API_KEY` env var or
`[inference.openrouter] api_key` in `akh.toml`. Same resolver pattern as
`ExternalApiConfig::resolve_api_key` (`src/extraction/config.rs:65-69`).

**Cost metering**: every response carries `usage.prompt_tokens` and
`usage.completion_tokens`; multiply by the per-model rate from the
capability cache. `CostGuard::record_spend` updates the Akh's daily
counter atomically. If the backend returns 429, surface as
`BackendError::RateLimited{retry_after_secs}` and let the router fall
through.

### Anthropic — optional / conditional fallback

> **Status:** optional. Ships if convenient, dropped if it causes
> maintenance load (auth changes, prompt-caching API shifts, cost
> concerns). The build must succeed with Anthropic compiled out, and the
> default fall-through chain (CandleLocal → CandleRemote → OpenRouter)
> must work without it. This subsection is preserved so Anthropic can be
> reinstated cheaply if/when it is convenient.

Direct Anthropic API (`https://api.anthropic.com/v1/messages`). Used as
the last fall-through entry — most expensive, most capable.

| Field | Value |
|---|---|
| `kind` | `BackendKind::Anthropic` |
| `models` | `["claude-haiku-4-5", "claude-sonnet-4-7", "claude-opus-4-7"]` |
| `hidden_state_extraction` | `None` |
| `cost_*_millicents_per_tok` | hard-coded per-model (Anthropic publishes flat rates) |
| `requires_reservation` | `false` |

**Auth**: `AKH_ANTHROPIC_API_KEY` or `[inference.anthropic] api_key`.

**Cost metering**: same pattern as OpenRouter; the Anthropic response
includes `usage.input_tokens` / `usage.output_tokens`. Opus is **never
auto-selected** — only when the operator (or the n akh router with an
explicit `prefer_capability = "frontier"` flag) requests it.

**Tool use, system prompts, vision**: out of scope for the trait. We
flatten everything to a single user prompt at the boundary; richer
features can be added later by widening `InferenceRequest`.

## Candle on ROCm — the hard deliverable

This is the technically risky part. Candle's CUDA backend is mature; its
Metal backend is mature; its ROCm/HIP backend is *experimental* — the
upstream tracking issues (huggingface/candle #1392, #2103) call out
unstable kernels and missing operators.

### Status quo

- `candle-core/src/cuda_backend.rs` is feature-complete.
- `candle-core/src/metal_backend.rs` is feature-complete.
- `candle-core/src/hip_backend.rs` exists (behind a `hip` feature) but
  several operators panic or produce numerically incorrect results.
- The Quant kernels (`Q4_K_M`, `Q8_0`) are particularly under-tested on
  HIP — they were ported from CUDA mechanically.

### The plan

1. **Build phase**: stand up a ROCm 6.0 toolchain on the 7900 XTX
   desktop (Talos doesn't have ROCm; this host is a bare Arch Linux box).
   Build `candle-core` with `--features hip` and run the existing Candle
   test suite. Capture every failing test.

2. **Triage phase**: classify each failure:
   - **Numerical** (output drifts beyond eps) → diff against CPU
     reference, locate the divergent kernel, file an upstream PR if the
     fix is small (most are missing `__hip_atomic_*` intrinsics or HIP
     vector-load alignment).
   - **Crash** (segfault / abort) → usually missing operator dispatch;
     shim with a CPU fallback in `akh-medu`'s own ROCm wrapper crate
     (`crates/akh-candle-rocm/`) before the kernel exists upstream.
   - **Slow** (>5x slower than CUDA on equivalent hardware) → record but
     don't block; the goal is *correctness > 5 tok/s*, not parity with
     CUDA.

3. **Critical path for our use case**: Qwen2.5-7B Q4_K_M generation
   needs:
   - `quantized_qwen2.rs` forward pass (rope, rms_norm, qmatmul)
   - KV cache append + retrieve
   - softmax / attention
   - hidden-state extraction (just `Tensor::clone()` between layers —
     hits the same code as the regular forward)
   That's <20 distinct kernels. We don't need the full operator surface.

4. **Acceptance**: Qwen2.5-7B Q4_K_M at >5 tok/s sustained on the
   7900 XTX, with `extract_hidden=true, hidden_layer=None` enabled.
   Numerical drift vs CPU reference: <0.5% perplexity on a held-out
   set of 500 prompts.

5. **Runtime fallback**: every kernel call is wrapped:

   ```rust
   match candle_kernel(rocm_dev, args) {
       Ok(t) => t,
       Err(e) if matches!(e, CandleError::Hip(_)) => {
           tracing::warn!(?e, "ROCm kernel failed, falling back to CPU");
           candle_kernel(cpu_dev, args)?  // re-run on CPU
       }
       Err(e) => return Err(e.into()),
   }
   ```

   The daemon does *not* crash on ROCm errors. A persistent fault flips
   the device for the rest of the request to CPU; the next request
   re-tries ROCm.

### Risk acknowledgement

- We are doing kernel work in a research-stage upstream. Upstream may
  refuse a PR; we may have to maintain a fork.
- Performance is a secondary axis — *not crashing* and *producing
  correct hidden states* are the primary success criteria.
- If the ROCm path proves intractable, the fallback is unmodified: we
  ship without it, the desktop runs the same Candle build with the CPU
  backend (slower but correct), and the OpenRouter / Anthropic chain
  picks up heavy synthesis. Phase 39 still ships value.

## Reservation Protocol

Scarce backends — primarily `CandleRemote` to the 7900 XTX, optionally
`CandleLocal` when the host marks itself scarce — require a lease before
the request runs. The lease is requested out-of-band, before
`complete`/`stream`, and threaded through the `InferenceRequest`.

### Wire choice

Three options were considered; we pick gRPC.

| Option | Pros | Cons |
|---|---|---|
| HTTP+JSON | Easy, debuggable | Extra schema work for hidden states; binary blobs base64'd |
| **gRPC** (tonic) | Bidirectional streaming, protobuf for hidden states, tonic already in deps for Anima | Heavier deploy; needs proto codegen |
| MCP | Reuses Phase 36's bridge | MCP is request/response only; no streaming hidden states |

gRPC wins because hidden-state blobs are large (Qwen2.5-7B last-layer
hidden state for a 1k-token context is `1000 × 3584 × 4 bytes = 14 MB`)
and protobuf gives us a typed, versioned binary frame.

### Wire format sketch

```proto
// crates/akh-inference-proto/proto/inference.proto

syntax = "proto3";
package akh.inference.v1;

service Inference {
  rpc Capabilities(CapabilitiesRequest) returns (CapabilitiesResponse);
  rpc Health(HealthRequest) returns (HealthResponse);
  rpc Reserve(ReserveRequest) returns (ReserveResponse);
  rpc Release(ReleaseRequest) returns (ReleaseResponse);
  rpc Complete(InferenceRequest) returns (InferenceResponse);
  rpc Stream(InferenceRequest) returns (stream TokenDelta);
}

message ReserveRequest {
  bytes akh_id = 1;             // 16 bytes uuid
  uint64 lease_secs = 2;
  string desired_model = 3;
}

message ReserveResponse {
  oneof outcome {
    Lease granted = 1;
    Reserved already_reserved = 2;
    Unavailable unavailable = 3;
  }
}

message Lease {
  bytes lease_id = 1;           // 16 bytes
  uint64 expires_unix_ms = 2;
  bytes akh_id = 3;
}

message InferenceRequest {
  string model = 1;
  string prompt = 2;
  uint32 max_tokens = 3;
  float temperature = 4;
  repeated string stop = 5;
  optional HiddenStateRequest hidden_states = 6;
  optional StructuredOutputSpec structured = 7;
  optional Lease lease = 8;
  bytes akh_id = 9;
}

message HiddenStateRequest {
  optional uint32 layer = 1;     // None = final
  bool last_token_only = 2;
}

message HiddenStateBlob {
  uint32 layer = 1;
  uint32 n_tokens = 2;
  uint32 hidden_dim = 3;
  HiddenDType dtype = 4;
  bytes data = 5;
  uint16 schema_version = 6;
}

enum HiddenDType { F32 = 0; BF16 = 1; F16 = 2; }

message TokenDelta {
  string text = 1;
  uint32 tokens = 2;
  optional FinishReason finish = 3;
  optional string error = 4;
  optional HiddenStateBlob terminal_hidden = 5;  // only on the final delta
}
```

The proto file lives in a new crate, `crates/akh-inference-proto`, so
both daemon and standalone candle-server share it.

### Reservation flow with two contending Akhs

```
T+0     Akh A (M2, cellar):    BackendRouter.reserve(7900_XTX, 600s)
        ReservationManager:    free → grant lease L_A, expires T+600
T+0.1   Akh A:                 stream(req_A) ... starts streaming tokens

T+5     Akh B (Talos, cluster): BackendRouter.reserve(7900_XTX, 600s)
        ReservationManager:    held by L_A until T+600
                               policy.queue_depth = 1 → enqueue B
        Akh B:                 awaits queue slot

T+45    Akh A's request finishes (180 tokens, ~40s).
        Akh A:                 BackendRouter.release(L_A)
        ReservationManager:    pops queue → grant L_B to Akh B
        Akh B:                 stream(req_B) starts

T+645   (hypothetical) Akh A had not released; lease watchdog fires,
        L_A.on_expiry_tx.send(()) — release event broadcast
        ReservationManager:    pops queue → L_B granted
```

Policy is per-backend; the operator can also configure
`policy.on_contention = "fall_through"` for "if 7900 XTX busy, don't
queue, jump to OpenRouter." The Akh's archetype hint
(`prefer_capability = "frontier"` vs `"cheap_distillation"`) feeds the
choice.

### Auto-release on expiry

The `ReservationManager` spawns one watchdog task per active lease:

```rust
async fn watch_lease(mgr: Arc<ReservationManager>, lease: Lease) {
    let until = Duration::from_millis(
        lease.expires_unix_ms - now_unix_ms()
    );
    tokio::time::sleep(until).await;
    let _ = mgr.force_release(lease.lease_id).await;
    // notify holder so its next request fails fast rather than 504-ing
}
```

The grace period is 5s past expiry (clock drift). If the holder calls
`release` early, the watchdog is cancelled.

## Discovery + Health

Each backend publishes an advertisement; the daemon polls them every
15s (configurable). The advertisement is the same `BackendCapabilities`
struct plus a `Health` snapshot:

```rust
pub struct Advertisement {
    pub capabilities: BackendCapabilities,
    pub health: Health,
    pub published_unix_ms: u64,
}
```

For local backends, "publishing" is in-process — the registry queries
the `LlmBackend::health()` method directly. For remote backends, the
healthcheck calls `Inference::Health` over gRPC.

### Failure semantics

- 1 failed health probe → `last_seen` ages, but status stays `Live`.
- 3 consecutive failures over 45s → status flips to `Stale`.
- Stale backends are excluded from the router's chain *during* requests
  but not from the registry — we keep probing.
- After a single successful probe post-stale, status flips back to
  `Live` and the router re-includes it (with a +1 boost in exploration
  rate so the bandit notices the recovery).

### Manual chain ordering

`akh.toml` carries a per-Akh fall-through chain; the router reads it but
the bandit can override per-archetype if it has confident data. Example:

```toml
[inference.chain]
# Primary stack: Candle (local + remote) and OpenRouter. Anthropic is an
# optional rung — listed last when present, omitted entirely when absent.
default            = ["candle_local", "candle_remote_basement", "openrouter"]
frontier_synthesis = ["candle_remote_basement", "openrouter"]
cheap_distillation = ["openrouter", "candle_local"]

# Optional: append "anthropic" when the operator opts in.
# default = ["candle_local", "candle_remote_basement", "openrouter", "anthropic"]
```

## Latency-aware Routing in Phase 28

Phase 28 (`019db054-55d2`) defines a LinUCB bandit with VSA-derived
context features. Phase 39 extends the context vector with three
backend-aware terms:

| Feature | Source | Update cadence |
|---|---|---|
| `backend.latency_p95_ms` | rolling p95 from last N=64 requests | per request |
| `backend.availability` | 1.0 if `Live`, 0.0 if `Stale`, 0.5 if `Reserved/Saturated` | per healthcheck |
| `backend.cost_millicents_per_kilotok` | from capabilities | daily refresh |

The bandit's reward signal (already in Phase 28b) becomes:

```
reward = success ? 1.0 : 0.0
       + max(0, 1.0 - latency / latency_budget)         [latency bonus]
       - alpha_cost * cost_millicents / cost_budget    [cost penalty]
       - alpha_avail * (1 - availability)              [availability penalty]
```

`alpha_cost` and `alpha_avail` are tunable in `akh.toml`. Default:
`alpha_cost = 0.3`, `alpha_avail = 0.5`. The router gracefully avoids
backends that are slow, expensive, or absent — without ever being
explicitly told "the 7900 XTX is offline today."

## Cost Tracking + Per-Akh Budget

`CostGuard` is the existing pattern from Phase 36
(`src/agent/delegation/mod.rs:CostGuard`). Phase 39 reuses it with a
per-backend index:

```rust
pub struct InferenceBudget {
    pub akh: AkhId,
    pub daily_budget_millicents: u64,
    pub per_request_max_millicents: u64,
    pub spent_today: DashMap<BackendKind, AtomicU64>,
    pub reset_at_unix_ms: AtomicU64,
}

impl InferenceBudget {
    pub fn can_afford(
        &self,
        backend_kind: BackendKind,
        estimated_tokens: u32,
        rate_per_kilotok: u32,
    ) -> Result<(), BackendError> {
        self.maybe_reset();
        let est = (estimated_tokens as u64 * rate_per_kilotok as u64) / 1000;
        let total_today: u64 = self.spent_today.iter()
            .map(|e| e.value().load(Ordering::Relaxed))
            .sum();
        if total_today + est > self.daily_budget_millicents {
            return Err(BackendError::BudgetExceeded {
                akh: self.akh,
                backend: BackendId(0),  // not relevant — budget exceeded across all
            });
        }
        if est > self.per_request_max_millicents {
            return Err(BackendError::BudgetExceeded {
                akh: self.akh, backend: BackendId(0),
            });
        }
        Ok(())
    }

    pub fn record(&self, backend_kind: BackendKind, actual_millicents: u64) {
        self.spent_today.entry(backend_kind).or_default()
            .fetch_add(actual_millicents, Ordering::Relaxed);
    }
}
```

When the daily budget is exhausted, the router treats the paid backends
as `Stale` for the rest of the day; the chain falls through to
`CandleLocal` and `CandleRemote` only.

### Configuration

```toml
[inference.budget]
daily_millicents = 50_000           # $0.50/day default
per_request_max_millicents = 5_000  # $0.05/request default

[inference.budget.openrouter]
daily_millicents = 100_000          # $1.00/day for distillation jobs

# Optional: only meaningful if the Anthropic backend is compiled in and
# enabled. Anthropic is a conditional fallback rung; cost tracking still
# applies when it ships.
[inference.budget.anthropic]
daily_millicents = 20_000           # $0.20/day for frontier fallback only
```

## Hidden-State Extraction Over the Wire

This is the trickiest invariant. Phase 26d's neural→VSA bridge
(`src/vsa/neural_bridge.rs` post-26d) consumes a `Tensor [hidden_dim]`
or `Tensor [n_tokens, hidden_dim]` produced by Candle locally. When the
inference happens on a remote Candle, we need byte-identical semantics
on the *bridge input* without depending on Candle's internal
representations.

### Constraints

1. The hidden state is a contiguous f32 (or bf16) buffer of known shape.
2. Endianness must be specified; both ends agree on little-endian.
3. The schema must be versioned so future tensor metadata (e.g.
   per-token attention weights) can be added without breaking old peers.
4. The bridge currently expects `f32` — bf16/f16 must be widened before
   handing to the projector.

### Wire format

`HiddenStateBlob` (above) carries:

- `dtype`: F32 / BF16 / F16
- `data`: row-major bytes, `n_tokens × hidden_dim × sizeof(dtype)`
- `schema_version`: u16. Bumped on any incompatible change. Sender +
  receiver both check; mismatch → `WireError::SchemaMismatch`.

Decoding on the daemon side:

```rust
pub fn decode_hidden(blob: &HiddenStateBlob, target_device: &Device)
    -> Result<Tensor, BackendError>
{
    if blob.schema_version != HIDDEN_STATE_SCHEMA_V1 {
        return Err(WireError::SchemaMismatch {
            expected: HIDDEN_STATE_SCHEMA_V1, got: blob.schema_version,
        }.into());
    }
    let shape = (blob.n_tokens as usize, blob.hidden_dim as usize);
    let tensor = match blob.dtype {
        HiddenDType::F32 => {
            let floats = bytemuck::cast_slice::<u8, f32>(&blob.data);
            Tensor::from_slice(floats, shape, target_device)?
        }
        HiddenDType::BF16 => {
            let halves = bytemuck::cast_slice::<u8, half::bf16>(&blob.data);
            // Widen to f32 for the bridge.
            let f32s: Vec<f32> = halves.iter().map(|&x| x.to_f32()).collect();
            Tensor::from_slice(&f32s, shape, target_device)?
        }
        HiddenDType::F16 => {
            let halves = bytemuck::cast_slice::<u8, half::f16>(&blob.data);
            let f32s: Vec<f32> = halves.iter().map(|&x| x.to_f32()).collect();
            Tensor::from_slice(&f32s, shape, target_device)?
        }
    };
    Ok(tensor)
}
```

### Bandwidth budget

Qwen2.5-7B last-layer hidden state for one 1k-token context:
`1000 × 3584 × 4 = 14 MB` raw. Streamed alongside ~150 tokens of text
output, the wire payload is dominated by hidden states. Two mitigations:

- **`last_token_only = true`** (default for Phase 26d single-shot probe):
  cuts to `1 × 3584 × 4 = 14 KB`. Fine on a homelab LAN.
- **bf16 dtype**: halves the bytes for the rare `last_token_only =
  false` case.

We do *not* compress hidden states at the protocol layer; the bandwidth
is acceptable on a wired homelab LAN, and any compression scheme adds a
quality risk to the bridge. PolarQuant compression (Phase 26c) is for
KV-cache only, not for the hidden-state wire path.

## Failure Modes

Dedicated section because half the value of this phase is graceful
degradation.

| Failure | Detection | Response |
|---|---|---|
| Backend offline at request time | `BackendError::Offline` from `complete` | router falls to next chain entry; bandit penalises |
| Backend offline mid-stream | `Err(StreamInterrupted)` on the stream item | partial result returned to caller with `partial = true`; if archetype demands full result, retry via next chain entry |
| Lease expires while streaming | watchdog fires, peer cancels stream | same as "offline mid-stream" |
| 429 rate-limit (OpenRouter / Anthropic) | `BackendError::RateLimited` | router excludes backend for `retry_after_secs`, falls through |
| Budget exhausted mid-request | `CostGuard` rejects before dispatch | `BackendError::BudgetExceeded`; router skips paid backends rest of day |
| ROCm kernel error | wrapped in `CandleError::Hip` | per-request CPU fallback; persistent fault → mark device degraded |
| Schema-version mismatch | `WireError::SchemaMismatch` | mark backend stale, log loudly, page operator (this is a deploy issue) |
| Healthcheck cascading failures | 3 consecutive in 45s | mark `Stale`, exclude until next successful probe |
| Daemon crash mid-request | n/a | next daemon launch reconciles outstanding leases (server-side watchdog handles cleanup) |

## Sub-phase Breakdown

| Sub-phase | Lines (est.) | Complexity | Depends on |
|---|---|---|---|
| **39a** — `LlmBackend` trait + `CandleLocal` | ~1200 | Medium | Phase 26b |
| **39b** — `CandleRemote` + hidden-state wire | ~1500 | High | 39a |
| **39c** — `OpenRouter` + `Anthropic` + cost tracking | ~900 | Medium | 39a |
| **39d** — Candle on ROCm | ~600 (+ upstream PRs) | High (research) | 39a |
| **39e** — Reservation + health + discovery | ~700 | Medium | 39a |
| **39f** — Phase 28 router integration | ~400 | Low | 39a, 39c, 39e, Phase 28 |
| **Total** | **~5300** | | |

### 39a — Trait + CandleLocal (~1200 lines)

**New tree**: `src/inference/`

- `src/inference/mod.rs` — public surface, re-exports
- `src/inference/backend.rs` — trait, structs, errors (above)
- `src/inference/registry.rs` — `BackendRegistry` with `DashMap<BackendId, Arc<dyn LlmBackend>>`
- `src/inference/candle_local.rs` — adapter wrapping the existing `src/nlu/candle_backend.rs`
- `src/inference/budget.rs` — `InferenceBudget` reusing `CostGuard` patterns
- `tests/inference_local.rs` — integration test with the in-tree GGUF

**Cargo features**:

```toml
[features]
inference = ["async-trait", "tokio/full", "futures", "bytes"]
inference-candle-local = ["inference", "candle-backend"]
```

**Acceptance**:
- `BackendRegistry::register(Box::new(CandleLocal::new(...)))` works.
- `complete()` returns text + final hidden state when requested.
- Budget enforcement rejects overspend before dispatch.
- Existing NLU tier 3 path can call `registry.complete()` instead of
  `CandleBackend::generate()` directly.

### 39b — CandleRemote + hidden-state wire (~1500 lines)

**New crate**: `crates/akh-inference-proto/`
- `proto/inference.proto` — see §[wire format](#wire-format-sketch)
- `build.rs` — tonic-build codegen
- `src/lib.rs` — re-exports of generated types + manual conversion
  helpers (`Lease ↔ proto::Lease`, etc.)

**New crate**: `crates/akh-candle-server/`
- A standalone binary that hosts a `CandleLocal` and serves it over
  gRPC via tonic. Same endpoint shape an in-process Akh exposes when
  `[server.candle_serve] enabled = true`.

**New module**: `src/inference/candle_remote.rs`
- `CandleRemoteBackend` — implements `LlmBackend` by calling the
  generated tonic client.
- TLS configuration via rustls + per-peer pinned certs.

**Acceptance**:
- Two daemons on the same LAN: one runs candle-server, the other
  registers a `CandleRemoteBackend` pointing at it.
- `complete()` round-trips text *and* a hidden-state blob.
- Decoded tensor on the consumer side matches a local-only run to
  within fp32 epsilon (`max_abs_diff < 1e-5`).
- Stream interruption (kill the server mid-stream) surfaces as
  `BackendError::StreamInterrupted{tokens_so_far}` and the consumer
  sees the partial result.

### 39c — OpenRouter (+ optional Anthropic) + cost tracking (~900 lines)

**New module**: `src/inference/openrouter.rs` (primary)
- Reuses `ureq`/`reqwest` (already in deps for the existing
  `extraction/backends/api.rs`).
- SSE parser for streaming.
- Per-model rate cache, refreshed daily via a registry idle task.

**New module**: `src/inference/anthropic.rs` (optional / conditional)
- Same shape, different schema (Anthropic `/v1/messages`).
- Hard-coded rate table per model.
- Gated behind the `inference-anthropic` feature flag. Build must
  succeed with the feature off; the rest of 39c (OpenRouter + cost
  tracking) does not depend on it. Drop this module if Anthropic
  causes maintenance load.

**Modify**: `src/inference/budget.rs` to consume per-response `usage`.

**Acceptance**:
- `complete()` against OpenRouter returns text + token counts + cost.
- Budget exhaustion forces fall-through to free backends for the rest
  of the day.
- 429 response is surfaced as `BackendError::RateLimited` with the
  upstream's `Retry-After` header parsed.

### 39d — Candle on ROCm (~600 lines + upstream PRs)

**New crate**: `crates/akh-candle-rocm/` — a small wrapper around
`candle-core --features hip` that adds:
- `Device::rocm_if_available(ordinal)` (thin wrapper, eventually
  upstreamed into `candle-core`).
- Per-kernel CPU fallback wrapper (above).
- Numerical-correctness test harness against a CPU reference.

**Upstream contributions**: file at minimum:
- a fix for `q4_k_m` HIP kernel correctness (likely missing
  alignment / atomic),
- a fix for `rms_norm` HIP variant if it diverges from CPU,
- a `Device::rocm_if_available` constructor.

**Acceptance** (operator demonstrates on the 7900 XTX):
- Qwen2.5-7B Q4_K_M generation, 256-token output, sustained >5 tok/s.
- Hidden-state extraction enabled; bridge produces a HyperVec whose
  top-1 ItemMemory match equals the top-1 from the CPU reference run
  on the same prompt.
- Inject a synthetic ROCm fault → daemon logs the warning, completes
  on CPU, does not crash.

### 39e — Reservation + health + discovery (~700 lines)

**New module**: `src/inference/reservation.rs`
- `ReservationManager` with `DashMap<BackendId, ReservationState>` and
  per-lease watchdog tasks.
- `ReservationPolicy` — queue depth, timeout, fall-through hint.

**New module**: `src/inference/health.rs`
- 15s healthcheck loop spawned by the registry.
- Stale / live state machine (3-fail / 1-success).
- Advertisement cache.

**Acceptance**:
- Two `Akh` instances reserve the same backend; the second waits for
  the first's release, then proceeds.
- A backend killed externally is marked `Stale` within 45s and
  excluded from routing.
- A previously-stale backend that recovers is re-included on the next
  successful probe.

### 39f — Phase 28 router integration (~400 lines)

**Modify**: `src/nlu/router.rs` (the Phase 28 module — must exist for
this sub-phase).
- Extend `RouterContext` with `backend_features: BackendFeatures`.
- Reward shaping: latency bonus + cost penalty + availability penalty
  (above).
- Persist per-backend bandit arms across daemon restarts via the
  existing redb path used by the parse ranker.

**Acceptance**:
- Workload mix where 80% of queries route to the cheapest viable
  backend without operator intervention.
- After a forced 7900 XTX outage, traffic shifts to OpenRouter within
  ~10 requests; after recovery, traffic returns to 7900 XTX within
  ~20 requests (exploration kicks in).

### Implementation order

39a → (39b parallel with 39c) → 39e → 39f → 39d. ROCm is decoupled and
slow; the rest of the fabric should ship without waiting for it.

## Risk Section

### ROCm maturity

The single largest risk. Mitigations: (1) parallelism — kernel work
proceeds on a branch while the rest of the fabric merges; (2)
acceptance criterion is *correctness > 5 tok/s* not *parity with CUDA*;
(3) the daemon falls back to CPU on per-request faults so a partially
working ROCm path still adds value.

### Hidden-state wire-format compatibility

Schema versioning catches the easy case (mismatched daemon versions).
The harder case is *Candle* changing its internal layer numbering
between releases — a Qwen2.5 layer 24 hidden state from Candle 0.10
might not be byte-identical to the same layer from Candle 0.11. We
version the `HIDDEN_STATE_SCHEMA` against both the proto and the Candle
crate version, and we re-train the neural bridge if we ever bump
Candle past a minor version.

### OpenRouter quota changes

OpenRouter has occasionally changed per-model rates without notice. The
daily-refresh of the per-model cost cache mitigates pricing drift; the
bigger risk is a model being delisted mid-day. The router treats
`UnsupportedModel` as a permanent fall-through (no retry on the same
backend with the same model) — the operator updates the allowlist next
day.

### Network jitter affecting routing

p95 latency is computed over a rolling window of 64 requests. Short
spikes don't dominate; sustained increases shift traffic. The risk is
oscillation — the bandit thrashes between two roughly-equal backends
because each is "best" half the time. The exploration constant
(`alpha = 0.25`) damps this; if it doesn't, we add a per-archetype
hysteresis term in a follow-up.

### Cost-runaway via misconfigured budget

The default `[inference.budget] daily_millicents = 50_000` is $0.50/day.
The default for paid backends is *zero* — they're disabled until the
operator explicitly enables them. The same defensive posture as Phase
36 delegation.

### Lease leaks

A daemon crashing mid-request leaves the peer with a held lease. The
peer's watchdog handles expiry; the daemon's restart code reconciles
outstanding leases by querying each remote backend's reservation state
and forcing release on any lease whose holder == this Akh.

## Estimated Effort

| Sub-phase | Lines | Complexity | Risk |
|---|---|---|---|
| 39a | ~1200 | Medium | Low |
| 39b | ~1500 | High | Medium (gRPC schema) |
| 39c | ~900 | Medium | Low |
| 39d | ~600 + upstream | High | High (ROCm) |
| 39e | ~700 | Medium | Low |
| 39f | ~400 | Low | Low |
| **Total** | **~5300** | | |

## New Dependencies

```toml
# Async + streaming
async-trait = "0.1"
futures = "0.3"
bytes = "1"

# gRPC + protobuf
tonic = { version = "0.12", optional = true }
prost = { version = "0.13", optional = true }
tonic-build = { version = "0.12", optional = true }   # build-dep

# TLS for CandleRemote
rustls = { version = "0.23", optional = true }
rustls-pemfile = { version = "2", optional = true }

# Anthropic / OpenRouter — reuse existing reqwest if not already a dep
# (ureq is in extraction; we widen to reqwest for streaming SSE)
reqwest = { version = "0.12", features = ["json", "stream"], optional = true }
eventsource-stream = { version = "0.2", optional = true }

# Half-precision conversion for hidden-state widening
half = "2"
bytemuck = "1"

# ROCm — only on the 7900 XTX host build
[target.'cfg(target_os = "linux")'.dependencies]
candle-core = { version = "0.10", optional = true, features = ["hip"] }
```

## Feature Flags

```toml
[features]
inference                   = ["async-trait", "futures", "bytes"]
inference-candle-local      = ["inference", "candle-backend"]
inference-candle-remote     = ["inference", "tonic", "prost", "rustls", "rustls-pemfile"]
inference-openrouter        = ["inference", "reqwest", "eventsource-stream"]
inference-anthropic         = ["inference", "reqwest", "eventsource-stream"]
inference-rocm              = ["inference-candle-local"]   # gated to Linux+HIP
```

The default `inference` feature pulls in the trait but no concrete
backends — concrete inclusion is opt-in per build target.

## Relationship to Other Phases

| Phase | Relationship |
|---|---|
| Phase 26b | Provides the `CandleBackend` Phase 39a wraps |
| Phase 26c (PolarQuant) | KV-cache compression stays *inside* the backend; the wire only carries hidden states |
| Phase 26d (Neural bridge) | Consumes `HiddenStateBlob` regardless of backend kind |
| Phase 27 (Burn training) | Local-only; not affected — but training data can be *generated* via OpenRouter |
| Phase 28 (n akh router) | Phase 39f extends the bandit context |
| Phase 32 (Knowledge extraction) | OpenRouter becomes the cheap-distillation target |
| Phase 36 (Delegation) | Reuses `CostGuard` pattern; a delegation can pick a backend through the registry rather than hard-coding `claude --print` |
| Phase 38 (Relocatable Akh) | Daemons move freely — Phase 39 is what makes that move *not* require relocating compute |

## Open Questions

These are the questions the implementation will need to answer; we
record them here so they're not lost.

1. **Reservation queue policy across multiple Akhs**: when three Akhs
   contend for the 7900 XTX, do we use FIFO, priority by goal weight, or
   the bandit's confidence as a tiebreaker? Default to FIFO; revisit
   after multi-Akh deployments exist.
2. **Hidden-state schema evolution**: should we adopt a stable
   "tensor-frame" format (e.g. safetensors-over-the-wire) or keep the
   custom protobuf? Custom is simpler today; safetensors-over-grpc is
   future-friendly.
3. **bf16 vs f32 default on the wire**: bf16 halves bandwidth but loses
   ~2 bits/coord on the bridge input. The bridge's binarisation
   threshold is 0.0; values near zero are the noisy ones. Empirical
   answer needed once the bridge is trained.
4. **Multiplexing requests on a single Lease**: does a single lease
   permit one in-flight request, or many? Default to one (simpler);
   batched serving could relax this if it becomes a bottleneck.
5. **Anthropic prompt caching**: *Resolved (2026-04-25):* Anthropic is
   deprioritized to an optional / conditional fallback. If the
   Anthropic client creates a problem (auth changes, prompt-caching
   API shifts, cost concerns), it is dropped. Prompt-caching support
   is therefore **not** a Phase 39 deliverable. If we keep Anthropic
   and want caching later, surface system prompts through
   `InferenceRequest` then; until then, defer.

## References

- Anima ticket: `019dc599-ff37-7cf0-b1dc-0ea5d774c20c`
- Phase 26 plan: [2026-03-26-phase26-llm-vsa-deep-integration.md](./2026-03-26-phase26-llm-vsa-deep-integration.md)
- Phase 28 plan: [2026-03-26-phase28-semantic-router.md](./2026-03-26-phase28-semantic-router.md)
- Phase 36 plan: [2026-03-28-phase36-delegation-external-scheduling.md](./2026-03-28-phase36-delegation-external-scheduling.md)
- ADR 037: [llm-vsa-deep-integration.md](../decisions/037-llm-vsa-deep-integration.md)
- ADR 038: [semantic-router.md](../decisions/038-semantic-router.md)
- ADR 046: [delegation-external-scheduling.md](../decisions/046-delegation-external-scheduling.md)
- ADR 050: [distributed-inference-fabric.md](../decisions/050-distributed-inference-fabric.md) (this phase)
- Candle: https://github.com/huggingface/candle
- Candle ROCm/HIP tracking: https://github.com/huggingface/candle/issues?q=label%3Arocm
- OpenRouter API docs: https://openrouter.ai/docs
- Anthropic API: https://docs.anthropic.com/en/api
- tonic (gRPC for Rust): https://github.com/hyperium/tonic

## Resolutions log

### 2026-04-25

- **P39Q5** (Anthropic prompt-caching exposure): deprioritize Anthropic. Primary stack is `CandleLocal` + `CandleRemote` + `OpenRouter`; Anthropic is an optional / conditional fallback rung — shipped if convenient, dropped if it causes maintenance load. Default fall-through chain is CandleLocal → CandleRemote → OpenRouter; Anthropic is an additional optional rung. Build must succeed with Anthropic compiled out. Cost tracking still mentions Anthropic but flags it as conditional. Prompt caching is not a Phase 39 deliverable.
- **P39Q6** (Per-archetype budget overrides): dropped. Single global budget per Akh is enough until specific archetypes are identified that need heavy inference and aren't covered by routing decisions. Bullet removed from the plan.
