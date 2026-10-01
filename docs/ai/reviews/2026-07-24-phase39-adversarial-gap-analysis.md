# Adversarial Gap Analysis — Phase 39: Distributed Inference Fabric

> **Reviewer:** tecton (sage/explorer)
> **Date:** 2026-07-24
> **Revision:** round 2 — re-audited against current tree (HEAD `30623df`). Round 1 was produced against a stale, shallow July checkout (`439b77a`, base ~2026-06-02) and asserted 39d/ROCm as "NOT STARTED 0%" against a `git log` that did not contain the ROCm/HIP work. That commit-existence claim and the derived ROCm verdict were **false**; this revision corrects them.
> **Audit base:** `30623df4d2ed39ab425cdf1c8dce1b49dfe8da32` (full history in this worktree; ROCm/HIP commits verified reachable).
> **Scope:** WI 39 (LlmBackend trait + Candle ROCm + reservation), cancelled 39a (CandleLocal — absorbed or abandoned?)
> **Method:** codebase audit against ADR 050 + 59KB plan doc (`docs/ai/plans/2026-04-25-phase39-distributed-inference.md`, 59,347 bytes / 1,407 lines)
> **Verdict:** **PARTIAL — recommend REVISE** (trait abstraction + orchestration layer NOT STARTED; the 39d/ROCm substrate is substantially LANDED).

---

## Executive Summary

Phase 39 is **partially implemented, in an asymmetric way the round-1 report got backwards.**

The **inference-fabric scaffolding** — the thing ADR 050's title is actually about ("pluggable LlmBackend trait") — is **still entirely unimplemented**: no `src/inference/` module, no `LlmBackend` trait, no `BackendRegistry`, no reservation protocol, no gRPC wire, no OpenRouter production backend, no health/discovery loop, no `InferenceBudget`, no `DerivationKind::DistributedInference` provenance variant, no `inference` Cargo feature, no `async-trait`/`tonic`/`prost` dependency.

But the round-1 verdict on **39d (Candle on ROCm) was flatly wrong.** Reviewer verification, re-confirmed against this tree, shows the ROCm/HIP path is substantially **LANDED**:

- The "forward-port candle boundary + ROCm GPU" commit the WI scope named **exists**: `5019f19` `feat(reconcile): forward-port candle boundary + ROCm GPU + Gemma4 audio; keep Ollama` (reachable from HEAD). It is accompanied by a body of Phase 45/46/51 work: `b84e5a6` (vendor candle-core fork + `Device::Hip` scaffold), `ec06b23` (real HIP device init + on-device memory), `8c05f1f` (merge: Phase 45 `Device::Hip` + Phase 46 VRAM/pool/graph foundation), `d80d632` (HIP-graph capture of the full decode forward — byte-exact), `d75a77f` (`quantized_qwen2` HIP device-gating: rms-norm/rope/softmax), `80ac8fb` (Phase 46b fused flash-attention decode kernel), `5c2ed72` (Phase 51: grug-train on ROCm/HIP via CubeCL).
- `Cargo.toml` declares `rocm` (:253), `metal` (:258), `hip` (:262) features and patches in a **vendored candle-core fork** (`[patch.crates-io]` at :344-349) that ships the HIP backend (ADR 073).
- `select_device()` at `src/nlu/candle_backend.rs:2713` has a real `#[cfg(feature = "hip")]` branch building `Device::Hip` via the vendored cuda-slot, preceded by Metal and CUDA branches, falling back to CPU; `report_rocm_status()` at :2768 does runtime ROCm probing via `src/nlu/gpu.rs`.
- `src/nlu/hip.rs` (69,760 bytes) implements a rocBLAS matmul `CustomOp2` (+ PCIe H2D/D2D crossing counters); `src/nlu/hip_kernels/dequant.hip` ships a custom dequant kernel.
- ADRs **067** (candle conversational boundary), **068** (ROCm/HIP GPU backend, dated 2026-06-04), **073** (Full Candle `Device::Hip`, dated 2026-06-21), **074** (static-shape inference + training), **098** (Qwen3.5-RDNA3 inference stack) all exist and document exactly this work.

One further **dependency-flip** the round-1 report missed: the hidden-state fidelity limitation it relied on ("`extract_hidden_state()` returns logits, not hidden states — Phase 26f not implemented") is **no longer true**. `extract_hidden_state()` at `src/nlu/candle_backend.rs:1969` now calls `self.model.forward_with_hidden()` and extracts the `hidden` tensor (`[batch, hidden_dim]`) via the `vendored` loaders — i.e., it returns **real hidden states**, with forward-verify / logits-parity instrumentation around it. This removes the supposed Phase 26f blocker for the 39b hidden-state-over-wire design.

What *still* holds from round 1 (verified unchanged against this tree): no `src/inference/`, no `LlmBackend` trait, no `BackendRegistry`, no `ReservationManager`, no `InferenceBudget`, no `DerivationKind::DistributedInference`, the Phase 28 `NakhRouter` routes NLU parse queries only (unextended), `llama-cpp-2` remains in `Cargo.toml` behind `nlu-llm` (migration step 7 incomplete), and 39a ("CandleLocal") was **abandoned, not absorbed**.

So the honest bottom line: **the GPU/distributed-hardware leg (39d) that ADR 050 deferred to last is the one leg that has made real progress — via an out-of-band Phases 45/46/51 effort — while the trait/orchestration backbone (39a/39e) that everything was supposed to build on has not started at all.**

---

## Per-Subcomponent Assessment

### 39a — LlmBackend Trait + CandleLocal — **NOT STARTED**

| Planned Artefact | Status | Evidence |
|---|---|---|
| `src/inference/mod.rs` | Missing | Directory does not exist |
| `src/inference/backend.rs` (trait + structs) | Missing | No `trait LlmBackend` anywhere in `src/` (grep: zero hits) |
| `src/inference/registry.rs` (`BackendRegistry`) | Missing | Zero references to `BackendRegistry` |
| `src/inference/candle_local.rs` (adapter) | Missing | Not created |
| `src/inference/budget.rs` (`InferenceBudget`) | Missing | Zero references |
| `tests/inference_local.rs` | Missing | Not created |
| `inference` Cargo feature | Missing | No `inference` feature in `Cargo.toml` |
| `async-trait` dependency | Missing | Not in `Cargo.toml` |

**What exists instead:** `CandleBackend` in `src/nlu/candle_backend.rs` (3,091 lines — round 1 said 707; the file has roughly quadrupled with the ROCm/HIP, hidden-state, KV-cache, and PolarQuant work). It is a concrete struct with `load()`/`try_load()`, `generate()`, `translate()`, `extract_hidden_state()`. It is called directly by `NluPipeline` (`src/nlu/mod.rs:99-101`, stored as `Option<Arc<Mutex<CandleBackend>>>`). No trait abstraction; the NLU pipeline keeps a hardcoded two-path switch — `llm_translator` (llama-cpp-2) OR `candle_backend` (Candle) — annotated "Mutually exclusive with llm_translator at runtime; Candle preferred when both available." A `LocalLlmBackend` struct exists at `src/extraction/backends/local.rs:13`, but it is a **triple-extraction** helper sharing `LlmTranslator`, **not** a 39a `LlmBackend` trait impl.

**39a absorption question:** 39a was "cancelled" — was it absorbed into existing code or abandoned? **Answer: Abandoned.** The existing `CandleBackend` is not a `LlmBackend` trait impl. It has no async surface, no `reserve`/`release`/`health`/`capabilities` methods, no `InferenceRequest`/`InferenceResponse` types, no `BackendRegistry`, no `registry.complete()` API. The trait abstraction that 39a was supposed to introduce simply does not exist. The "Candle/GGUF/LlmBackend seam already landed" claim in the `d1e086e` llama-retirement commit message refers to the Candle backend having taken over the `nlu-llm` *translation* path — not to a Phase 39 trait/registry, which grep confirms is absent.

### 39b — CandleRemote + gRPC Wire — **NOT STARTED**

| Planned Artefact | Status | Evidence |
|---|---|---|
| `crates/akh-inference-proto/` | Missing | Crate does not exist |
| `crates/akh-candle-server/` | Missing | Binary does not exist |
| `src/inference/candle_remote.rs` | Missing | Not created |
| `tonic` / `prost` dependency | Missing | Not in `Cargo.toml` |
| `HiddenStateBlob` wire type | Missing | No such type |
| TLS / pinned cert config | Missing | Not present |

**Zero gRPC infrastructure.** No `tonic`, no `prost`, no `.proto` files, no protobuf codegen. The wire-format sketch in the plan doc is design-only. **However**, the round-1 framing that 39b's hidden-state blob was blocked on Phase 26f ("returns logits") is **obsolete**: with `forward_with_hidden()` now returning true hidden states, the *producer* side of the wire design exists. The consumer/transport side (proto schema, `HiddenStateBlob`, server) is still entirely unstarted.

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
- `src/extraction/backends/api.rs` — OpenAI-compatible HTTP client for *triple extraction only*, not inference. (ADR 050 itself notes `src/extraction/config.rs:40` already points at an OpenAI-compatible endpoint.)
- Separately, `src/deepseek.rs` / the chat-backing work (ADR 070 selectable chat backing, ADR 095 OpenAI-compatible chat backing) shows an OpenAI-compatible HTTP path exists for *dialogue* — but not as a Phase 39 `src/inference/openrouter.rs` backend behind a trait.
- `src/agent/ollama.rs` — Ollama HTTP client with `health_check()`/`generate()`; the *actual* production inference path for dialogue generation, but not behind any `LlmBackend` trait.

**Cost tracking:** `CostGuard` exists at `src/agent/delegation/mod.rs:42` with `daily_budget_cents` + `per_task_max_cents` (`new()` at :54). It enforces *delegation* cost, not inference. The plan's "reuses `CostGuard` patterns" intent is unfulfilled — there is no `InferenceBudget` adapter.

### 39d — Candle on ROCm — **PARTIAL / SUBSTANTIALLY LANDED (major round-1 correction)**

This is the round-1 report's blocking failure. It claimed 39d was "NOT STARTED, 0%, no ROCm work has been started, no `#[cfg(feature = "hip")]` branch, metal/cuda features not even declared, the forward-port commit does not exist." **All false against this tree.**

| Planned Artefact | Status | Evidence (verified at `30623df`) |
|---|---|---|
| `hip` feature on candle-core | **DONE** | `Cargo.toml:262` `hip = ["rocm", "candle-core/hip"]`; `rocm` at :253; `metal` at :258 (all declared — round 1 said they weren't) |
| Vendored candle-core fork w/ HIP backend | **DONE** | `[patch.crates-io]` at `Cargo.toml:344-349`: `candle-core = { path = "vendor/candle-core" }` (ADR 073) |
| `Device::Hip` construction in `select_device()` | **DONE** | `src/nlu/candle_backend.rs:2713` — real `#[cfg(feature = "hip")]` branch at :2736 builds `Device::new_cuda(0)` → real `HipDevice` via the cuda-slot; Metal branch :2715; CUDA :2724; CPU fallback |
| ROCm runtime detection | **DONE** | `report_rocm_status()` at `src/nlu/candle_backend.rs:2768` → `src/nlu/gpu.rs` `detect_rocm()` (Ready / KernelReadyUserspaceMissing / Unavailable) |
| HIP compute kernel(s) | **DONE** | `src/nlu/hip.rs` (69.7 KB) rocBLAS matmul `CustomOp2` + PCIe-crossing counters; `src/nlu/hip_kernels/dequant.hip` custom dequant kernel |
| `quantized_qwen2` device gating | **DONE** | commit `d75a77f` (rms-norm/rope/softmax device-gated for HIP) |
| HIP-graph capture of decode forward | **DONE** | commit `d80d632` "works, byte-exact"; Phase 46b fused flash-attention `80ac8fb` |
| ROCm/HIP training (CubeCL) | **DONE** | commit `5c2ed72` Phase 51 grug-train on ROCm/HIP |
| `Device::rocm_if_available()` dedicated ctor | N/A — superseded | Design used `new_cuda` cuda-slot instead (ADR 073); functionally equivalent route |
| Per-kernel CPU fallback wrapper | **PARTIAL** | CPU fallback exists at the *device* level; the explicit per-kernel wrapper described in the plan is not the mechanism used |
| ROCm upstream PRs | Not filed (by design) | ADR 073 vendors the fork locally rather than upstreaming; "no upstream PRs" is true but is a *choice*, and in-repo HIP expertise clearly exists |

**Device selection, corrected** (`src/nlu/candle_backend.rs:2713-2760`):
```rust
fn select_device() -> Device {
    #[cfg(feature = "metal")] { Device::new_metal(0) … }   // Metal: YES
    #[cfg(feature = "cuda")]  { Device::new_cuda(0)  … }   // CUDA: YES
    #[cfg(feature = "hip")]   { Device::new_cuda(0) → HipDevice … }  // HIP: YES (cuda slot, ADR 073)
    Self::report_rocm_status();                            // probe + honest log if on CPU
    Device::Cpu                                            // fallback
}
```

**The named commit exists.** `5019f19 feat(reconcile): forward-port candle boundary + ROCm GPU + Gemma4 audio; keep Ollama` is reachable from HEAD. The round-1 report's "worktree has a single merge commit (`439b77a`)" was a stale-shallow-checkout artifact; it should have been recorded as an evidence gap, not asserted as a universal negative — especially because the WI scope explicitly named this commit as a key question.

**39d verdict correction:** NOT STARTED → **PARTIAL**. The ROCm/HIP backend is real, feature-gated (`hip`/`rocm`), and carries device init, custom kernels, graph capture/replay, and CubeCL training. What remains for a *complete* 39d: the `llama-cpp-2` removal (migration step 7) and any hardening/CI-hardware story — neither of which was the blocker round 1 imagined.

### 39e — Reservation + Health + Discovery — **NOT STARTED**

| Planned Artefact | Status | Evidence |
|---|---|---|
| `src/inference/reservation.rs` | Missing | Not created |
| `ReservationManager` | Missing | Zero references (grep) |
| `ReservationPolicy` | Missing | Not present |
| `src/inference/health.rs` | Missing | Not created |
| 15s healthcheck loop | Missing | No backend health loop for inference |
| Stale/live state machine | Missing | Not present |
| `inference_status` MCP tool | Missing | Zero references |

No reservation, lease, or health-check infrastructure exists for **inference**. (`reserve`/`lease` references elsewhere are the *delegation* subsystem's subprocess-task machinery — structurally different.) Note: daemon/host lease work from later waves (ADR 0037/0039 scheduling, session leases) exists, but none of it is wired to an inference-backend registry, because that registry does not exist.

### 39f — Phase 28 Router Integration — **NOT STARTED** (router exists, not extended)

| Planned Artefact | Status | Evidence |
|---|---|---|
| `RouterContext` with `backend_features` | Missing | `RouterContext`/`BackendFeatures` not found |
| Latency bonus in reward shaping | Missing | Reward is accuracy-only |
| Cost penalty in reward shaping | Missing | Not present |
| Availability penalty | Missing | Not present |
| Bandit arm persistence via redb | Missing | `ParseRanker` persists via redb; bandit arms don't |

**The Phase 28 router exists** (`src/nlu/router.rs`, 492 lines — this figure was correct) with `NakhRouter`, `LinUcbBandit`, `BackendId`, `RouteDecision`, `report_reward()`. But `BackendId` is hardcoded to NLU-tier backends (`RuleParser`, `OnnxNer`, `CandleLlm`, `VsaDirect`, `CloudApi`, `BurnLocal`); there is no inference-backend variant. The router routes *NLU parse queries*, not *LLM inference requests*, and has no inference path, cost/latency/availability features, or persistence of bandit arms.

---

## Cross-Cutting Gaps

### 1. No `src/inference/` module at all
Confirmed at `30623df`. Inference flows through siloed paths: `src/nlu/` (NLU tier-3 parse + NLG), `src/agent/model_manager.rs` + `src/agent/ollama.rs` (dialogue via Ollama + fallback API), `src/extraction/backends/api.rs` (triple extraction). None unified behind a trait.

### 2. `llama-cpp-2` still present (migration step 7 incomplete)
ADR 050: "llama-cpp-2 is removed in Phase 39d." It is still in `Cargo.toml:161` (`llama-cpp-2 = { version = "0.1", optional = true, features = ["sampler"] }`), still feature-gated under `nlu-llm` (:245), and `src/nlu/llm_translator.rs` is present (12 `llama` references). Commit `d1e086e` migrated the `nlu-llm` *path* onto Candle but did **not** remove the crate or retire the feature flag. So even though the ROCm leg of 39d landed, the coupled native-C++ removal has not.

### 3. No `DerivationKind::DistributedInference` provenance
`src/provenance.rs` has ~25 `DerivationKind` variants; grep confirms no `DistributedInference`. No backend-kind/model-id/lease-id/token-count/cost/wall-time recording for inference.

### 4. No inference config sections
No `akh.toml` and no `[inference.*]` sections in `AkhomedConfig` (`src/config.rs`). No spine to register backends from config on boot.

### 5. Hidden-state extraction — **corrected: now real hidden states**
Round 1 claimed `extract_hidden_state()` returns *logits* (Phase 26f incomplete, blocking 39b). At `src/nlu/candle_backend.rs:1969`, it now calls `self.model.forward_with_hidden()` and extracts the `hidden` tensor via the `vendored` loaders — **true hidden states** (`[batch, hidden_dim]`), with forward-verify / logits-parity instrumentation (e.g., around :236, :263, :312). The Phase 26f dependency that 39b's wire design assumes **has landed**; the wire itself (39b) has not.

### 6. No Phase 39 WIP files
`.anima/wip/` has **32** files (round 1 said 31 — trivial undercount). Zero reference Phase 39, `LlmBackend`, `BackendRegistry`, `CandleRemote`/`CandleLocal`, or "inference fabric." Adjacent: `019dc599.md` (Phase 38 Relocatable Akh — mentions Phase 39 as a future dependency). The fabric leg has no implementation trail; the ROCm leg's trail lives in Phases 45/46/51 + ADRs 067/068/073/098 instead.

---

## Stale WIP Assessment

`.anima/wip/` contains 32 files, none Phase 39. The relevant adjacent ones:

| WIP File | Phase | Status | Relevance to P39 |
|---|---|---|---|
| `019dc599.md` | Phase 38 (Relocatable Akh) | Active | Partially-blocking. P39 leans on "where state lives ≠ where compute happens." Later relocation-protocol work (Wave R2/R3, owner_host/capability tags, fail-safe boot) has landed since, so this is less of a hard blocker than round 1 implied. |
| `019dd122.md` (P26f hidden-state fidelity) | Phase 26f | Active | **No longer blocking.** `extract_hidden_state()` now returns real hidden states (`:1969`). |
| `019dc59a.md` | Phase 40 (E-graph persistence) | Active | Not blocking; shares the "measure → report → decide" pattern. |

**Stale WIP verdict:** No Phase 39 WIP to clean up; the fabric was never started. The claimed Phase 26f blocker is obsolete.

---

## Was 39a Absorbed or Abandoned?

**Abandoned.** Unambiguous at `30623df`:

1. No `LlmBackend` trait (the core 39a deliverable). 2. No `src/inference/` module. 3. No `BackendRegistry`. 4. `CandleBackend` is a concrete struct called directly, not a trait impl (no async, no `reserve`/`release`/`health`/`capabilities`, no `InferenceRequest`/`InferenceResponse`). 5. No `InferenceBudget`. 6. No `inference` Cargo feature. 7. No `tests/inference_local.rs`.

The "Candle/GGUF/LlmBackend seam" phrase in `d1e086e` describes the Candle backend absorbing the `nlu-llm` *translator* role — not the 39a trait/registry. 39a's cancellation was a planning artifact; its work was never done, and the existing code does not satisfy its acceptance criteria.

---

## Recommendation: **REVISE** (re-derived — the 39d premise changed)

Round 1 recommended "Defer 39d indefinitely; no ROCm expertise in repo, no upstream PRs filed." **That recommendation was built on the false ROCm premise and is reversed here.** Substantial in-repo HIP expertise demonstrably exists (vendored backend, custom kernels, HIP-graph capture/replay, CubeCL training across Phases 45–51). Re-deriving from the actual tree:

### Why not "Revive" (as-is)
- The trait/registry backbone (39a) — the load-bearing seam every other sub-phase references — does not exist. Reviving the plan verbatim would still require building it.
- 39b's gRPC wire is unstarted; though its hidden-state blocker is gone, the proto/server/transport half is a from-scratch build.
- `llama-cpp-2` removal (the 39d exit criterion in ADR 050's ordering) is not actually done.

### Why not "Retire"
- The architectural direction remains correct *and is now de-risked on the hardware side*: the 7900 XTX ROCm path (39d) is substantially built and tested against real kernels.
- The `LlmBackend` trait abstraction is still the right answer to today's 4-way siloed inference.
- `CostGuard`, the Phase 28 bandit router, and the real `CandleBackend` are all reusable foundations.
- Real hidden states now exist, unblocking the hidden-state-over-wire design when transport is built.

### What revision should do (corrected ordering)
1. **Re-scope 39a as truly minimal:** define `LlmBackend` with `async fn complete()` + `fn capabilities()` only; wrap `CandleBackend` and `OllamaClient` as the first two impls (~300 lines). Skip `reserve`/`release`/`health` for 39a.
2. **Re-plan 39d as "finish + remove," not "defer":** the ROCm/HIP backend is landed. The remaining 39d work is (a) completing the `llama-cpp-2`/`nlu-llm` removal, and (b) surfacing the HIP path through the 39a trait. Do **not** defer — this is now a strength to consolidate, not a risk.
3. **Defer only the gRPC wire (39b) to *after* the trait:** the hidden-state producer now exists, so 39b is unblocked technically; but it should still follow 39a's trait so `CandleRemote` has a trait to implement. Ship HTTP+JSON before gRPC if faster.
4. **Keep 39f as a new `InferenceRouter`, not an `NakhRouter` extension:** the NLU router works; conflating it with inference routing risks regressions.
5. **Add `DerivationKind::DistributedInference` + `[inference]` config spine now** — cheap, prepares surface for backends as they land.

### Revised implementation order (corrected)
39a (minimal trait + 2 impls) → 39d-finish (remove llama-cpp-2; surface HIP through trait) → 39c (OpenRouter only) → 39e (reservation + health, simplified) → 39b (remote transport; hidden-state producer ready, so gRPC is now justified) → 39f (new InferenceRouter).

---

## Summary Table

| Sub-phase | Round 1 (stale) | **This revision** | Est. Completion | Blocking Deps |
|---|---|---|---|---|
| 39a — Trait + CandleLocal | NOT STARTED | **NOT STARTED** | 0% | None (cancelled ⇒ abandoned) |
| 39b — CandleRemote + gRPC | NOT STARTED | **NOT STARTED** (hidden-state blocker now gone) | 0% | 39a; transport unstarted (producer ready) |
| 39c — OpenRouter + cost | NOT STARTED | **NOT STARTED** | 0% | 39a (templates exist elsewhere) |
| 39d — ROCm | NOT STARTED ❌ | **PARTIAL / substantially LANDED** ✅ | ~60–70% | llama-cpp-2 removal; trait surface for HIP |
| 39e — Reservation + health | NOT STARTED | **NOT STARTED** | 0% | 39a + 39b |
| 39f — Router integration | NOT STARTED | **NOT STARTED** (router exists, NLU-only) | 0% | 39a–39e |
| **Overall** | NOT STARTED (0%) | **PARTIAL** | **~15–25%** | 39a backbone missing; 39d hardware leg landed |

**Bottom line (corrected):** Phase 39 is *not* a 59KB design doc with zero implementation. Its **hardware/distributed leg (39d) made real, verified progress** via the Phases 45/46/51 effort and ADRs 067/068/073/098 — the round-1 report missed this entirely because it audited a stale shallow checkout and converted an evidence gap into a false universal negative. What remains genuinely unstarted is the **trait/orchestration backbone (39a/39e/39f) and the transport (39b)**. The revive/revise/retire call should therefore be **REVISE**: build the minimal `LlmBackend` trait first, *consolidate* (not defer) the landed ROCm work into it, finish the native-C++ removal, and only then add the transport and routing layers.
