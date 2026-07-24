# Adversarial Gap Analysis — Phase 39: Distributed Inference Fabric

> **Reviewer:** tecton (sage/explorer)
> **Date:** 2026-07-24
> **Scope:** WI 39 (LlmBackend trait + Candle ROCm + reservation), cancelled 39a (CandleLocal — absorbed or abandoned?)
> **Method:** codebase audit against ADR 050 + 59KB plan doc (`docs/ai/plans/2026-04-25-phase39-distributed-inference.md`)
> **Verdict:** **NOT STARTED (0%) — recommend REVISE**

---

## Executive Summary

Phase 39 is **entirely unimplemented**. Zero of the six sub-phases (39a–39f) have landed. The `src/inference/` directory does not exist. No `LlmBackend` trait, no `BackendRegistry`, no reservation protocol, no gRPC wire, no OpenRouter backend, no ROCm path, no health/discovery loop, no cost tracking for inference, no `DerivationKind::DistributedInference` provenance variant.

What *does* exist is the **pre-Phase 39 substrate** the plan expected to build on:
- A working `CandleBackend` skeleton (`src/nlu/candle_backend.rs`, 707 lines) — Metal + CUDA device selection, Qwen2.5-1.5B GGUF, logits-as-pseudo-hidden-state extraction
- A Phase 28 `NakhRouter` with `BackendId` enum + LinUCB bandit (`src/nlu/router.rs`, 492 lines) — but with `BackendId` hardcoded to NLU-tier backends, no inference backend awareness
- A `CostGuard` pattern in the delegation subsystem (`src/agent/delegation/mod.rs:42`) — reusable but not yet applied to inference
- An OpenAI-compatible HTTP extraction backend (`src/extraction/backends/api.rs`) — template for OpenRouter, not wired as inference
- `llama-cpp-2` still present behind `nlu-llm` feature flag, coexisting with Candle as tier-3 alternatives

The plan referenced a "recent forward-port candle boundary + ROCm GPU" commit — **no such commit exists in git history** (the worktree has a single merge commit). The ROCm path is not partially landed; it's absent.

---

## Per-Subcomponent Assessment

### 39a — LlmBackend Trait + CandleLocal — **NOT STARTED**

| Planned Artefact | Status | Evidence |
|---|---|---|
| `src/inference/mod.rs` | Missing | Directory does not exist |
| `src/inference/backend.rs` (trait + structs) | Missing | No `trait LlmBackend` anywhere in `src/` |
| `src/inference/registry.rs` (`BackendRegistry`) | Missing | Zero references to `BackendRegistry` |
| `src/inference/candle_local.rs` (adapter) | Missing | Not created |
| `src/inference/budget.rs` (`InferenceBudget`) | Missing | Not created |
| `tests/inference_local.rs` | Missing | Not created |
| `inference` Cargo feature | Missing | No `inference` feature in `Cargo.toml` |
| `async-trait` dependency | Missing | Not in `Cargo.toml` |

**What exists instead:** `CandleBackend` in `src/nlu/candle_backend.rs` — a concrete struct with `load()`, `generate()`, `translate()`, `extract_hidden_state()`. It's called directly by `NluPipeline` (`src/nlu/mod.rs:84`, stored as `Option<Arc<Mutex<CandleBackend>>>`). No trait abstraction; the NLU pipeline has a hardcoded two-path switch: `llm_translator` (llama-cpp-2) OR `candle_backend` (Candle), mutually exclusive at runtime.

**39a absorption question:** 39a was "cancelled" — was it absorbed into existing code or abandoned? **Answer: Abandoned.** The existing `CandleBackend` is not a `LlmBackend` trait impl. It's a standalone struct with no async surface, no `reserve`/`release`/`health`/`capabilities` methods, no `InferenceRequest`/`InferenceResponse` types. The trait abstraction layer that 39a was supposed to introduce simply doesn't exist. The cancellation appears to have been a planning decision with no corresponding implementation.

### 39b — CandleRemote + gRPC Wire — **NOT STARTED**

| Planned Artefact | Status | Evidence |
|---|---|---|
| `crates/akh-inference-proto/` | Missing | Crate does not exist |
| `crates/akh-candle-server/` | Missing | Binary does not exist |
| `src/inference/candle_remote.rs` | Missing | Not created |
| `tonic` / `prost` dependency | Missing | Not in `Cargo.toml` |
| `HiddenStateBlob` wire type | Missing | No such type |
| TLS / pinned cert config | Missing | Not present |

**Zero gRPC infrastructure.** No `tonic`, no `prost`, no `.proto` files, no protobuf codegen. The wire format sketch in the plan doc is design-only.

### 39c — OpenRouter + Cost Tracking — **NOT STARTED** (template exists)

| Planned Artefact | Status | Evidence |
|---|---|---|
| `src/inference/openrouter.rs` | Missing | Not created |
| `src/inference/anthropic.rs` | Missing | Not created |
| SSE streaming parser | Missing | No SSE in inference path |
| `inference-anthropic` feature flag | Missing | Not in `Cargo.toml` |
| Per-response `usage` → budget | Missing | No inference budget system |
| `BackendError::RateLimited` | Missing | No `BackendError` type |

**Templates that exist but aren't wired as inference backends:**
- `src/extraction/backends/api.rs` — OpenAI-compatible HTTP client for *triple extraction only*, not inference. Has `ExternalApiConfig` with `api_base` pointing at `openrouter.ai/api/v1` (config comment at `src/extraction/config.rs:40`).
- `src/bench_phase41g/pattern_b.rs` — OpenRouter client for benchmarking, uses `ureq::post("https://openrouter.ai/api/v1/chat/completions")`. Not production inference path.
- `src/agent/ollama.rs` — Ollama HTTP client with `health_check()` and `generate()`. This is the *actual* production inference path for dialogue generation (`AgentModelManager` routes to Ollama first, then `fallback_api`). But it's not behind any `LlmBackend` trait.

**Cost tracking:** `CostGuard` exists in `src/agent/delegation/mod.rs:42` with `daily_budget_cents` + `per_task_max_cents`. It's used for delegation cost control, not inference. The plan says "reuses `CostGuard` patterns" — the pattern is there but the `InferenceBudget` adapter is not.

### 39d — Candle on ROCm — **NOT STARTED**

| Planned Artefact | Status | Evidence |
|---|---|---|
| `crates/akh-candle-rocm/` | Missing | Crate does not exist |
| `Device::rocm_if_available()` | Missing | Not in `select_device()` |
| Per-kernel CPU fallback wrapper | Missing | Not present |
| `hip` feature on candle-core | Missing | `Cargo.toml:121`: `candle-core = { version = "0.10", optional = true }` — no features specified |
| ROCm upstream PRs | Unknown | No evidence in repo |

**`select_device()` in `src/nlu/candle_backend.rs:535`:**
```rust
fn select_device() -> Device {
    #[cfg(feature = "metal")]     // Metal: YES
    #[cfg(feature = "cuda")]      // CUDA: YES
    Device::Cpu                   // fallback
}
```
No `#[cfg(feature = "hip")]` or ROCm branch. The `metal` and `cuda` features aren't even declared in `Cargo.toml` — they'd need to be forwarded to `candle-core`'s feature flags. Currently `candle-core` is pulled in with no features (`candle-core = { version = "0.10", optional = true }`), so even Metal/CUDA are inert unless the user manually enables them in a downstream build.

The plan's "recent forward-port candle boundary + ROCm GPU" commit **does not exist** in the worktree's git history. The worktree has a single merge commit (`439b77a`). No ROCm work has been started.

### 39e — Reservation + Health + Discovery — **NOT STARTED**

| Planned Artefact | Status | Evidence |
|---|---|---|
| `src/inference/reservation.rs` | Missing | Not created |
| `ReservationManager` | Missing | Zero references |
| `ReservationPolicy` | Missing | Not present |
| `src/inference/health.rs` | Missing | Not created |
| 15s healthcheck loop | Missing | No backend health loop |
| Stale/live state machine | Missing | Not present |
| `inference_status` MCP tool | Missing | Zero references |

No reservation, lease, or health-check infrastructure exists for inference. The only `reserve`/`lease` references in the codebase are in the delegation subsystem (`src/agent/delegation/`) for subprocess task management — structurally different from backend reservation.

### 39f — Phase 28 Router Integration — **NOT STARTED** (router exists, not extended)

| Planned Artefact | Status | Evidence |
|---|---|---|
| `RouterContext` with `backend_features` | Missing | `RouterContext` not found |
| Latency bonus in reward shaping | Missing | Router reward is accuracy-only |
| Cost penalty in reward shaping | Missing | Not present |
| Availability penalty | Missing | Not present |
| Bandit arm persistence via redb | Missing | `ParseRanker` persists via redb; bandit arms don't |

**The Phase 28 router exists** (`src/nlu/router.rs`, 492 lines) with a `NakhRouter` struct, `LinUcbBandit`, `BackendId` enum, `RouteDecision`, `report_reward()`. But:

1. `BackendId` is hardcoded to NLU-tier backends: `RuleParser`, `OnnxNer`, `CandleLlm`, `VsaDirect`, `CloudApi`, `BurnLocal`. There's no `BackendId` variant for inference backends (no `CandleLocal`, `CandleRemote`, `OpenRouter`, `Anthropic`).
2. The router routes *NLU parse queries*, not *LLM inference requests*. The semantic overlap is real (both pick a backend), but the router has no `InferenceRequest` path.
3. No `BackendFeatures` struct with latency/cost/availability fields.
4. Reward shaping is accuracy-only (`report_reward` takes a float, no multi-term signal).
5. No persistence of bandit arms across restarts (the `ParseRanker` uses redb, but the bandit doesn't).

---

## Cross-Cutting Gaps

### 1. No `src/inference/` module at all
The plan calls for `src/inference/` as the new top-level module. It doesn't exist. All inference today flows through:
- `src/nlu/` — NLU tier-3 (parse + NLG)
- `src/agent/model_manager.rs` — dialogue generation (Ollama + fallback API)
- `src/agent/ollama.rs` — Ollama HTTP client
- `src/extraction/backends/api.rs` — triple extraction via OpenAI-compatible API

These are siloed, not unified behind a trait.

### 2. `llama-cpp-2` still present
ADR 050 says "llama-cpp-2 is removed in Phase 39d." It's still in `Cargo.toml` (`llama-cpp-2 = { version = "0.1", optional = true }`) and `src/nlu/llm_translator.rs` (943 lines, actively feature-gated under `nlu-llm`). The plan's migration step 7 ("llama-cpp-2 is removed, nlu-llm feature retired") hasn't happened.

### 3. No `DerivationKind::DistributedInference` provenance
The plan calls for a new provenance variant recording backend kind, model id, lease id, tokens-in/out, cost, wall time. `src/provenance.rs:46` has 25+ `DerivationKind` variants — no `DistributedInference`. The existing `DerivationKind::AgentDecision` and `CliToolExecution` are the closest but lack backend/cost metadata.

### 4. No `akh.toml` inference config sections
The plan specifies `[inference.budget]`, `[inference.backends.candle_remote]`, `[inference.backends.openrouter]`, etc. No `akh.toml` exists in the repo (config is `AkhomedConfig` in `src/config.rs`), and there are no inference-related config sections.

### 5. Hidden-state extraction is a placeholder
`CandleBackend::extract_hidden_state()` (`src/nlu/candle_backend.rs:303`) explicitly documents that it returns *logits*, not true hidden states:

> "Candle's `quantized_qwen2::ModelWeights` does not expose per-layer hidden states — the layer loop is internal to the `forward()` method. Getting true hidden states would require forking `candle-transformers` to insert a clone between layers."

This is a known limitation acknowledged in Phase 26d. Phase 39's hidden-state-over-the-wire design depends on a real `HiddenStateBlob` type that doesn't exist. The neural→VSA bridge (`src/vsa/neural_bridge.rs`) consumes `&[f32]` but the current producer gives logits (vocab-sized ~151K f32), not hidden states (model-dim ~1536 f32). The wire format, bandwidth budget, and fp32 epsilon comparison test all assume real hidden states.

### 6. No WIP files for Phase 39
`.anima/wip/` has 31 files. Zero reference Phase 39, `LlmBackend`, `BackendRegistry`, `CandleRemote`, `CandleLocal`, `akh-candle`, "inference fabric", or "ROCm". The only adjacent WIP is `019dc599.md` (Phase 38 — Relocatable Akh) which mentions Phase 39 as a future dependency. Phase 39 has no implementation trail at all.

---

## Stale WIP Assessment

`.anima/wip/` contains 31 files. None are Phase 39 WIP. The relevant adjacent ones:

| WIP File | Phase | Status | Relevance to P39 |
|---|---|---|---|
| `019dc599.md` | Phase 38 (Relocatable Akh) | Active | **Blocking dependency.** P39 plan says "Phase 38 lets the daemon move" and "decouple where compute happens." P38 itself is not fully implemented (no Postgres, no relocation protocol in code). |
| `019dd122.md` | Phase 26f (Hidden-state fidelity) | Active | **Blocking dependency.** P39's hidden-state-over-wire depends on real hidden states, which P26f was supposed to deliver. The current `extract_hidden_state()` returns logits, not hidden states. |
| `019dc59a.md` | Phase 40 (E-graph persistence) | Active | Not blocking P39, but shares the "measure → report → decide" pattern. |

**Stale WIP verdict:** No Phase 39 WIP to clean up. The WIP directory is clean for Phase 39 — it simply was never started.

---

## Was 39a Absorbed or Abandoned?

**Abandoned.** The evidence is unambiguous:

1. No `LlmBackend` trait exists — the core deliverable of 39a.
2. No `src/inference/` directory — the planned module structure.
3. No `BackendRegistry` — the planned orchestration layer.
4. `CandleBackend` exists but is not a trait impl — it's a concrete struct called directly by `NluPipeline`.
5. No `InferenceRequest`/`InferenceResponse` types — no typed request/response boundary.
6. No `InferenceBudget` — no cost enforcement at the inference layer.
7. No `inference` Cargo feature — the feature gate was never added.
8. No integration test file (`tests/inference_local.rs`).

The "cancellation" of 39a was a planning artifact. The work it described was never done, and the existing `CandleBackend` doesn't satisfy the 39a acceptance criteria (no trait, no registry, no budget enforcement, no `registry.complete()` API).

---

## Recommendation: **REVISE**

### Why not "Revive"
The plan as written assumes a substrate that has evolved:
- Phase 38 (Relocatable Akh) is itself not implemented — no Postgres, no relocation, no workspace-based state. The "where state lives ≠ where compute happens" framing in ADR 050 has no implementation foundation.
- Phase 26f (hidden-state fidelity) is not done — `extract_hidden_state()` returns logits, not hidden states. The entire hidden-state-over-wire design (14 MB blobs, fp32 epsilon tests, `HIDDEN_STATE_SCHEMA_VERSION`) is built on a feature that doesn't exist yet.
- The Phase 28 router exists but routes NLU parse queries, not inference requests. Extending it requires rethinking `BackendId` and `RouterContext` from scratch.

### Why not "Retire"
The architectural direction is sound:
- The `LlmBackend` trait abstraction is correct — inference is currently siloed across 4 code paths with no unified surface.
- The `CostGuard` pattern from delegation is proven and reusable.
- The Phase 28 `NakhRouter` + LinUCB bandit is a real, working router that *could* be extended.
- `CandleBackend` works for Qwen2.5-1.5B on CPU/Metal/CUDA — the local backend foundation is real.
- The OpenAI-compatible HTTP template (`src/extraction/backends/api.rs`) is directly reusable for OpenRouter.

### What revision should do
1. **Re-scope 39a to be truly minimal:** Define `LlmBackend` trait with `async fn complete()` + `fn capabilities()` only. Skip `reserve`/`release`/`health` for 39a. Wrap existing `CandleBackend` and `OllamaClient` as the first two impls. This is ~300 lines, not 1200.
2. **Defer 39b (gRPC wire) until hidden states are real.** The wire format design is premature when the producer gives logits, not hidden states. Ship `CandleRemote` as HTTP+JSON first (like OpenRouter), upgrade to gRPC when hidden-state blobs actually exist.
3. **Defer 39d (ROCm) indefinitely.** No ROCm expertise in repo, no upstream PRs filed, no test hardware in CI. The 7900 XTX can run Candle-CPU until ROCm matures. Don't block the trait + registry + OpenRouter on ROCm.
4. **Re-scope 39f (router integration) as a separate phase.** The NakhRouter routes NLU queries, not inference. Conflating the two risks breaking the working NLU router. Inference routing should be a new `InferenceRouter` that consults `BackendRegistry`, not an extension of `NakhRouter`.
5. **Add `DerivationKind::DistributedInference` to `provenance.rs` now** — it's a 15-line change that costs nothing and prepares the provenance surface for when backends land.
6. **Add `inference` config sections to `AkhomedConfig`** — even if empty, the config spine should exist so backends can be registered from config on boot.

### Revised implementation order
39a (minimal trait + 2 impls) → 39c (OpenRouter only, skip Anthropic) → 39e (reservation + health, simplified) → 39b (remote, HTTP first) → 39f (new InferenceRouter, not NakhRouter extension) → 39d (ROCm, deferred)

---

## Summary Table

| Sub-phase | Status | Est. Completion | Blocking Deps |
|---|---|---|---|
| 39a — Trait + CandleLocal | **NOT STARTED** | 0% | None (but 39a was "cancelled" — actually abandoned) |
| 39b — CandleRemote + gRPC | **NOT STARTED** | 0% | 39a + Phase 26f (real hidden states) |
| 39c — OpenRouter + cost | **NOT STARTED** | 0% | 39a (template exists in `extraction/backends/api.rs`) |
| 39d — ROCm | **NOT STARTED** | 0% | 39a + upstream Candle ROCm work (no PRs filed) |
| 39e — Reservation + health | **NOT STARTED** | 0% | 39a + 39b |
| 39f — Router integration | **NOT STARTED** | 0% | 39a–39e (router exists but routes NLU, not inference) |
| **Overall** | **NOT STARTED** | **0%** | Phase 38 not implemented; Phase 26f not implemented |

**Bottom line:** Phase 39 is a 59KB design doc with zero implementation. The substrate it planned to build on (Phase 38 relocation, Phase 26f hidden states) is also unimplemented. The plan should be revised to a minimal viable scope: trait + 2 local impls + OpenRouter, deferring gRPC, ROCm, and router integration until the foundations exist.
