# ADR 050 — Distributed Inference Fabric: pluggable LlmBackend trait, Candle-on-ROCm, no native C++

> Date: 2026-04-25
> Status: Proposed
> Phase: 39 (Distributed Inference Fabric)
> Depends on: Phase 26b (Candle backend), Phase 28 (n akh router)
> Related: ADR 037 (LLM-VSA), ADR 038 (semantic router), ADR 046 (delegation)
> Anima ticket: `019dc599-ff37-7cf0-b1dc-0ea5d774c20c`

## Context

The akh-medu daemon currently does inference in-process. Today the binding
is `llama-cpp-2` (`src/nlu/llm_translator.rs:53-56,83-85,206-211`) with a
half-finished Candle alternative behind the `candle-backend` feature flag
(`src/nlu/candle_backend.rs`, registered in `Cargo.toml:117-119`). All paths
assume "where the daemon runs is where the inference runs."

That assumption is becoming expensive:

1. **LLM compute pressure on the M2.** The M2 16GB host runs Qwen2.5-1.5B
   comfortably but cannot fit Qwen2.5-7B with KV cache and engine state in
   parallel. Phase 26d (the neural→VSA bridge) and Phase 32 (knowledge
   extraction) both want a 7B-class model. Phase 38 lets the daemon move,
   but moving it to the only host that can run 7B (the 7900 XTX desktop)
   sacrifices the laptop's mobility.

2. **The 7900 XTX desktop sits unused.** A Radeon RX 7900 XTX with 24 GB of
   VRAM is the only homelab GPU capable of Qwen2.5-7B Q4_K_M with hidden
   states at interactive speed. It's powered on most of the time and the
   daemon can't reach it because nothing in our stack speaks across hosts.

3. **OpenRouter is a cheap distillation opportunity.** OpenRouter hosts
   Llama-3.1-70B / Qwen2.5-72B / DeepSeek-V3 behind one API key, billed per
   token. Phase 32 (LLM knowledge extraction) and Phase 27 (live training
   data distillation) want bulk OSS-LLM access. The existing
   `src/extraction/backends/api.rs` and `src/extraction/config.rs:40` already
   point at an OpenAI-compatible endpoint — but only for triple extraction,
   not as a general inference target.

4. **Anthropic API is the natural cloud fallback.** When the on-prem chain
   is degraded *and* OpenRouter cannot satisfy the capability requirement
   (frontier reasoning), we want a paid escape hatch. We already use
   Anthropic interactively; using its API as a backend means one auth path
   for both modes.

5. **The native-C++ removal goal.** ADR 037 set the direction: replace
   `llama-cpp-2` with Candle. The reasons are structural: per-layer hidden
   states aren't reachable through llama-cpp-2's Rust API; GGML
   `abort()`-on-assertion bypasses Rust's panic infrastructure; the CMake
   build bloats our Talos OCI images. Phase 26b started the migration.
   Phase 39 finishes it by ensuring *every* backend — local, remote, and
   cloud — speaks the same Rust-typed surface.

A related driver is **Phase 38 (Relocatable Session-Based Akh)**, which
decouples *where state lives* from *which host runs the daemon*. Phase 39
is the natural sequel: decouple *where compute happens* from either of
those. State stays where Postgres + the workspace dir are; the daemon
runs on whichever host the Akh wants today; inference goes to whatever
backend currently best satisfies the request's capability profile.

## Decision

Introduce a `LlmBackend` trait (Send + Sync, async where it matters) with
three primary implementations — `CandleLocal`, `CandleRemote`, `OpenRouter` —
plus one **optional / conditional** implementation, `Anthropic`, shipped if
convenient and dropped if it causes maintenance load (auth changes,
prompt-caching API shifts, cost concerns). All backends are managed by a
`BackendRegistry` that provides healthchecks, reservations, and cost
tracking. The `n akh` semantic router (Phase 28) is extended with
backend-aware features (rolling p95 latency, availability, cents-per-token
cost) so its LinUCB bandit naturally avoids reserved or expensive backends
for queries that don't need them.

Concretely:

- **One trait, three primary backends, one optional rung.**
  `LlmBackend::{capabilities, health, reserve, release, complete, stream}`.
  Hidden-state extraction is part of `InferenceRequest`, not a side
  channel. The default fall-through chain is
  `CandleLocal → CandleRemote → OpenRouter`; `Anthropic` is an additional
  optional rung at the bottom when present. The build must succeed with
  Anthropic compiled out.
- **Reservation protocol with leases + auto-release** for scarce backends
  (the 7900 XTX, M2 when its homing Akh is busy). Per-backend policy
  controls queue depth and fall-through chain.
- **Discovery + healthcheck loop** every 15s. Three failed probes flips a
  backend to `Stale`; one successful probe restores it.
- **gRPC over TLS for `CandleRemote`**, with a versioned protobuf schema
  for hidden-state blobs (in a new crate `crates/akh-inference-proto`).
- **Cost tracking** via the existing `CostGuard` pattern from Phase 36,
  per-Akh daily budget, configurable in `akh.toml`. Paid backends are
  disabled by default.
- **Candle-on-ROCm** for the 7900 XTX. Acceptance: Qwen2.5-7B Q4_K_M at
  >5 tok/s with hidden-state extraction, falling back to CPU on per-request
  ROCm errors without crashing the daemon.
- **No native C++ in the inference path.** `llama-cpp-2` is removed in
  Phase 39d. Candle hosts everything, including the AMD HIP variant.

The `BackendRegistry` is constructed at daemon boot from `akh.toml`. The
router consults it on every request; backends are added or removed without
restarting the daemon.

## Alternatives Considered and Rejected

### llama.cpp via its server mode (`llama-server` over HTTP)

Rejected.

- llama-server (built on llama.cpp) speaks an OpenAI-compatible HTTP API.
  Adopting it as a remote backend would have been the path of least
  resistance, but it preserves the C++ dependency we're trying to escape:
  every host running it needs CMake + GGML.
- More importantly, it loses **Phase 26d hidden-state extraction**.
  llama-server exposes only final-token embeddings via `/embedding`, not
  per-layer hidden states. The neural→VSA bridge depends on the post-norm
  pre-lm_head tensor, which llama.cpp's API does not expose without C-level
  callback hooks (and those callbacks are not surfaced through any HTTP or
  Rust-binding layer).
- Even if a per-layer endpoint were added upstream, we'd be coupled to a
  C++ project's release cadence for an architectural feature we depend on.

### vLLM

Rejected.

- vLLM is excellent at production serving (continuous batching, PagedAttention,
  tensor parallelism). Throughput is best-in-class.
- It's Python-stack. Running it as a peer service on the 7900 XTX is fine
  *operationally* — Python on Linux is not a deployment problem — but it
  means our Rust-typed surface (`LlmBackend`, `BackendCapabilities`,
  `InferenceRequest`) is meeting a Python implementation through a wire
  protocol. Hidden-state extraction in vLLM is research-only (you can hack
  it via `model.model.layers[i].forward_hook` but that's not in their stable
  API).
- We want a unified Rust toolchain end-to-end. Adopting vLLM as the remote
  backend means maintaining Python infrastructure, container builds, GPU
  driver stacks, etc., for a project whose value to us is a small slice
  (Qwen2.5-7B inference + hidden states). Candle gets us the slice we need
  in our preferred language.
- Reconsidering vLLM later, if Candle ROCm proves intractable *and*
  throughput becomes the binding constraint, is reasonable. Phase 39's
  trait surface is provider-agnostic; we'd add a `VllmRemote` backend
  alongside `CandleRemote`.

### Custom inference engine from scratch

Rejected.

- The "build our own" option exists in the abstract: write quantized matmul
  kernels, RoPE, RMSNorm, FlashAttention 2, KV cache, GGUF loading. Each is
  a research-grade subsystem.
- Candle is ~95% of where we'd end up. The 5% gap (ROCm operator
  completeness, some quant kernel tuning) is exactly what Phase 39d aims to
  close — and any work we do there can flow upstream rather than rotting in
  a private fork.
- Building from scratch means rebuilding the model zoo (Qwen2, Llama, T5).
  Candle has the model zoo already. The Phase 35 T5 work depends on it.

### Single fixed backend (no fall-through, no cost-aware routing)

Rejected.

- The simplest design — pick one backend and live with its limits — fails
  every requirement: no fallback, no cost optimisation, no way to use the
  7900 XTX *and* the M2 *and* OpenRouter, no graceful degradation when a
  backend is offline.
- It also forecloses on Phase 38 (Relocatable Akh): a daemon that moves
  hosts but is wedded to one backend either has to bring the backend with
  it (impossible for the 7900 XTX) or accept that mobility costs capability.

### Plain HTTP everywhere (skip gRPC)

Considered, partially rejected.

- HTTP+JSON is what `OpenRouter` and `Anthropic` already speak. Adding
  gRPC is one more wire format to maintain.
- For `CandleRemote` specifically, the hidden-state blobs are large
  (14 MB for Qwen2.5-7B last-layer over a 1k-token context). Base64'ing
  through JSON adds 33% bandwidth overhead and parsing cost. Bidirectional
  streaming is cleaner with gRPC's HTTP/2 multiplexing than with chunked
  HTTP + custom framing.
- Compromise: gRPC for `CandleRemote` (where the binary blob and
  bidi-stream pay off), HTTP for `OpenRouter` and `Anthropic` (where the
  upstream forces it).

## Consequences

### Positive

- **Unified Rust stack.** From the daemon's `BackendRegistry` to the
  `CandleRemote` peer to the standalone `akh-candle-server` binary, every
  host in the inference path runs Rust. The Talos OCI image shrinks
  (`cargo build` only); the laptop and the 7900 XTX desktop share build
  flow.
- **Hidden-state extraction works on remote backends.** Phase 26d's bridge
  consumes `HiddenStateBlob` regardless of whether the producer was a
  `CandleLocal` on Metal, a `CandleRemote` over gRPC, or (eventually) a
  `CandleRemote` running on the 7900 XTX over ROCm. The neural↔symbolic
  loop survives the network boundary.
- **The 7900 XTX becomes usable on demand.** Akhs running on the M2 or in
  the cluster reserve it via the lease protocol, run heavy synthesis,
  release. Multiple Akhs queue politely. The desktop's idle hours stop
  being wasted.
- **OpenRouter access for distillation jobs.** Phase 32 (knowledge
  extraction) and Phase 27 (training data) get a cheap path to OSS frontier
  models without standing up our own 70B-class infra.
- **Cost is bounded architecturally, not just operationally.** `CostGuard`
  enforces per-Akh daily budgets at the registry boundary. A misconfigured
  archetype cannot drain the OpenRouter budget *and* the Anthropic budget
  in the same incident.
- **Backend-aware bandit.** The router's existing reward signal extends
  with latency, availability, and cost terms. Traffic shifts naturally as
  conditions change — no operator intervention needed for "the 7900 XTX
  went offline this morning."
- **Graceful degradation.** Mid-stream failures, lease expiry, rate
  limits, schema mismatches all surface as typed `BackendError` variants
  the router pattern-matches into specific fall-through behaviours.

### Negative

- **Candle ROCm maturity risk.** This is the single largest implementation
  risk. Upstream HIP support is experimental; we may have to ship our own
  fork of `candle-core` for the 7900 XTX while upstream PRs are reviewed.
  Mitigation: per-kernel CPU fallback at runtime, so an incomplete ROCm
  path doesn't block the rest of the fabric from shipping.
- **gRPC wire-format coupling between Candle versions.** Hidden states
  from Candle 0.10 layer 24 might not be byte-identical to the same layer
  from Candle 0.11 (layer reorderings, normalisation tweaks, RoPE base
  changes). The `HIDDEN_STATE_SCHEMA_VERSION` protobuf field catches the
  obvious case (incompatible schema), but invisible numerical drift across
  Candle minor versions could silently degrade the neural bridge. We
  re-train the bridge on any non-patch Candle bump and pin the version
  explicitly in `Cargo.toml`.
- **More moving parts to monitor.** Today the daemon has one inference
  path; after Phase 39 it has up to four backends, each with its own health
  state, lease state, and cost meter. Telemetry burden grows. Mitigation:
  the registry exposes a single MCP tool (`inference_status`) that surfaces
  every backend's health, current load, and today's spend.
- **Wire latency for hidden-state blobs.** A 14 MB blob over a homelab LAN
  is ~100ms at gigabit. Acceptable for batch probes, painful for
  tight-loop interactive use. Default to `last_token_only = true` for
  bridge probes (drops to 14 KB); reserve full per-token blobs for
  occasional debug / introspection sessions.
- **Lease leaks on unclean shutdown.** A daemon that crashes mid-request
  leaves the peer with a held lease. The peer's watchdog handles expiry
  (5s grace past `expires_unix_ms`); the daemon's restart code reconciles
  any leases it can prove it owned. Net effect: at most a few minutes of
  spurious "reserved" state on restart, never permanent leak.
- **Operational complexity per host.** The 7900 XTX desktop now runs an
  `akh-candle-server` binary, not just the regular akh daemon. That's one
  more service to deploy, monitor, and update. Mitigation: the binary is
  ~5 MB, statically linked, deployed via the same OCI image as the daemon
  with a different entrypoint.

### Cost ceiling explicitly enforced

Default `[inference.budget] daily_millicents = 50_000` ($0.50/day) covers
moderate distillation. Paid backends (`OpenRouter`, `Anthropic`) are
**disabled by default**; the operator opts in per-backend in `akh.toml`,
with explicit per-backend daily ceilings. Same defensive posture as Phase
36 delegation.

### Provenance extension

A new `DerivationKind::DistributedInference` records, for every triple
asserted from an LLM call, the backend kind, model id, lease id (or
`None`), tokens-in, tokens-out, cost in millicents, and wall time. The
Phase 36 `Delegation` provenance variant remains for `claude --print`
subprocesses; the new variant covers the in-band inference path.

## Migration Path

Roughly the order in the plan doc:

1. Phase 39a lands the trait + `CandleLocal` adapter; the existing NLU
   tier-3 path begins consuming `BackendRegistry::complete()` instead of
   calling Candle directly. `llama-cpp-2` stays on as a fallback feature.
2. Phase 39b stands up `CandleRemote` and the gRPC server. Operators
   running both M2 and 7900 XTX can begin dispatching to the desktop.
3. Phase 39c adds `OpenRouter` (primary). `Anthropic` is added as an
   optional / conditional rung in the same sub-phase, gated behind its
   own feature flag; both are default-disabled, and Anthropic is dropped
   if it causes maintenance load. The rest of 39c does not depend on it.
4. Phase 39e adds reservations + healthchecks. Up to here, the M2 can use
   the desktop without contention because no other Akh exists yet.
5. Phase 39f extends the Phase 28 router. Traffic begins shifting based
   on observed latency / cost.
6. Phase 39d closes the ROCm path. Until this lands, the 7900 XTX runs
   Candle-CPU (correct but slower).
7. Once Phase 39d ships and the ROCm path is stable for a sustained
   period, `llama-cpp-2` is removed (the `nlu-llm` feature flag retired,
   the `LlmTranslator` trait collapsed into the Candle-only path).

## References

- Anima ticket: `019dc599-ff37-7cf0-b1dc-0ea5d774c20c`
- Phase 39 plan: [2026-04-25-phase39-distributed-inference.md](../plans/2026-04-25-phase39-distributed-inference.md)
- ADR 037 (LLM-VSA): [037-llm-vsa-deep-integration.md](037-llm-vsa-deep-integration.md)
- ADR 038 (semantic router): [038-semantic-router.md](038-semantic-router.md)
- ADR 046 (delegation): [046-delegation-external-scheduling.md](046-delegation-external-scheduling.md)
- Phase 28 plan: [2026-03-26-phase28-semantic-router.md](../plans/2026-03-26-phase28-semantic-router.md)
- Existing Candle skeleton: `src/nlu/candle_backend.rs`
- Existing llama-cpp-2 binding (to be removed): `src/nlu/llm_translator.rs`
- OpenAI-compatible HTTP backend (template for OpenRouter): `src/extraction/backends/api.rs`
- Cargo features for Candle: `Cargo.toml:113-150`
- Candle: https://github.com/huggingface/candle
- Candle ROCm tracking: https://github.com/huggingface/candle/issues?q=label%3Arocm
- OpenRouter API: https://openrouter.ai/docs
- Anthropic API: https://docs.anthropic.com/en/api
- tonic (Rust gRPC): https://github.com/hyperium/tonic
- vLLM (rejected alternative): https://github.com/vllm-project/vllm

## Resolutions log

### 2026-04-25

- **P39Q5** (Anthropic prompt-caching exposure): deprioritize Anthropic. Primary stack is `CandleLocal` + `CandleRemote` + `OpenRouter`; Anthropic is an optional / conditional fallback rung — shipped if convenient, dropped if it causes maintenance load. Default fall-through chain is CandleLocal → CandleRemote → OpenRouter; Anthropic is an additional optional rung. Build must succeed with Anthropic compiled out. Cost tracking still applies when Anthropic ships.
- **P39Q6** (Per-archetype budget overrides): dropped. Single global budget per Akh is enough until specific archetypes are identified that need heavy inference and aren't covered by routing decisions.
