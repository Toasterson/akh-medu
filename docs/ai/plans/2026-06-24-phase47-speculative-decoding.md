# Phase 47 — Speculative Decoding (Candle, gemma4-12b)

- **Status:** Active (47a–47d + productization done + validated on gfx1100; default-on toggle + 47e remain)
- **Anima work item:** Akh-Medu project `019d9844-ce74-7993-a9f0-c953b8f04a4e` (work-item id TBD — create + link)
- **ADR:** [075-speculative-decoding](../decisions/075-speculative-decoding.md)
- **Builds on:** Phase 46d-2 (HIP-graph per-token decode; device-pos static KV cache) — [plan](2026-06-22-phase46-candle-static-shape-inference.md), [ADR 074]
- **Target model:** **gemma4-12b** (the Gemma 4 series — NOT gemma3). This is a hard requirement from the user.

## Why

Candle decode on the HIP backend is **dispatch-bound**: gemma3/gemma4-12b run at ~8.8 % of the
bandwidth roofline (11.5 tok/s on an RX 7900 XTX), i.e. ~11× spare compute, with wall-time
dominated by per-token host-enqueue latency, not FLOPs. Phase 46d-2 attacked the dispatch cost
with HIP graphs (collapsing the per-token kernel storm into one `hipGraphLaunch`) and got +1.10×
net — diluted by per-token host-sync `embed`/sampling.

Speculative decoding attacks the same axis from a different direction: produce several candidate
tokens cheaply, then **verify them all in a single target forward**. Because we are dispatch-bound
with ~11× compute headroom, verifying `γ+1` tokens in one forward costs ≈ **one** decode-time
(the larger shapes hide in idle FLOPs) — so spec decode collapses up to `γ+1` sequential target
forwards into one. This regime makes spec decode *more* attractive for us than for compute-bound
setups.

Motivating workload: agentic / RAG / backtesting loops (the use case in the tweet that prompted
this) emit highly repetitive, context-echoing output — the ideal case for prompt-lookup drafting
(below).

## The gemma4 constraint (the crux) and its resolution

Classic spec decode needs a small **draft model** sharing the target's *exact* token IDs and
special tokens. For gemma4-12b this is **blocked**:

| Model | tokenizer md5 | vocab | Note |
|---|---|---|---|
| gemma3-4b / 12b / 27b | `47a57159…` | 262144 | byte-identical across the gemma3 family |
| **gemma4-12b** | `72b1044584…` | 262144 | **different tokenizer**; no smaller gemma4 on disk |

So no gemma3 model can draft for gemma4 (token IDs / specials won't align), and we have no
gemma4-mini. **Resolution: draft-model-free speculative decoding**, which uses the *target's own*
vocab and therefore sidesteps the tokenizer problem entirely:

- **Prompt-lookup decoding (PLD) / n-gram drafting — the MVP.** The "draft" is a lookup over
  `prompt + generated-so-far`: propose the span that follows the longest recent n-gram match. No
  model, no tokenizer issue. Wins big (often 1.5–3×) when output **echoes context** (tool-call
  args, RAG quotes, code edits, structured/JSON output — i.e. agentic workloads); near-neutral on
  free-form prose, and it never hurts much (a missed lookup falls back to one normal token), so it
  is safe to leave always-on.
- **Self-draft (Medusa heads / early-exit) — later.** Also target-vocab, but Medusa needs training
  and early-exit needs the model to be coherent shallow (gemma often isn't). Out of MVP scope.
- **A real gemma4-mini draft — only if obtained.** Requires a vocab-compatible (`72b1…`) small
  gemma4 we'd fetch + quantize. External dependency, not satisfiable from the repo today; recorded
  as an open option, not the plan.

## Architecture — most of it reuses existing code

Lossless-greedy round (target KV valid `[0, P)`, current token `cur`):

1. **Draft (PLD):** look up the longest match of the recent n-gram tail in `prompt + output`;
   propose its continuation span `d₁…d_γ` (γ ≈ 3–8, truncated at the lookup length). No forward.
2. **Verify:** target runs **one** `(γ+1)`-token forward over `[cur, d₁…d_γ]` at positions
   `[P … P+γ]`, returning logits at **all** positions. This is a `γ+1`-token *mini-prefill* — the
   existing `seq_len > 1` causal-mask path already does exactly this; the *only* change is to not
   slice off the last position (47a).
3. **Accept (greedy, lossless):** accept `dᵢ` while `argmax(target_logitsᵢ) == dᵢ`; on the first
   mismatch take the target's own argmax as the correction. Emit accepted drafts + 1
   correction/bonus → up to **γ+1 tokens per target forward**. Byte-identical to plain greedy.
4. **Rollback:** rewind the target KV to the last accepted position. **Free with our static
   device-pos KV cache (46d-2):** just move the write index back; over-written columns are
   re-written next round. (PLD has no draft KV to roll back.)

New code: an all-position verify forward, an n-gram lookup drafter, and the accept/rollback loop.
The mask, KV cache, and (later) graph machinery already exist.

## Phased breakdown (gate-per-step, measure-or-revert)

- **47a — all-position verify forward (gemma4).** Add a verify path to gemma4 `ModelWeights` that
  returns `[1, seq, vocab]` (drop the `i((.., seq_len-1, ..))` last-position slice). Shared by
  every drafter. **Gate:** last-position of the verify output == plain `forward` byte-exact; each
  intermediate position == a per-position reference forward. Low-risk, byte-exact-gated. *(begin
  here)*
- **47b — n-gram (prompt-lookup) drafter.** Maintain a rolling index over `prompt + generated`;
  propose the continuation of the longest matching recent n-gram tail (configurable max span).
  **Gate:** proposals are valid token spans; lookup hit-rate logged on a repetitive prompt.
- **47c — lossless greedy verify + accept + KV rollback.** Wire 47a + 47b into the decode loop.
  **Gate:** **byte-exact generated token sequence vs plain greedy gemma4 decode** + report tok/s,
  acceptance rate, and mean accepted-tokens/round. Keep iff it beats plain decode on a repetitive
  workload; otherwise record the finding.
- **47d (opt) — HIP-graph the verify forward.** The `(γ+1)`-token verify is a *fixed* shape →
  capturable with the same device-pos block-padding from 46d-2c. Composes spec decode with the
  graph-decode lever.
- **47e (opt) — sampling spec decode.** Probabilistic accept + residual resample, fixed RNG seed;
  lossless in distribution (not byte-exact). After the greedy MVP proves out.

## Honest speedup analysis

Dispatch-bound model: target verify of `γ+1` tokens ≈ one decode-time `T`. PLD draft cost ≈ 0
(no forward). So per round = `T` (one verify) producing `E[tokens/round]` tokens, where
`E ≈ (1 − α^{γ+1})/(1 − α)` and `α` is the per-token acceptance rate.

- **PLD is conditional:** `α` is high only where output reuses context spans. On agentic /
  RAG / code / structured output, `α` is often 0.6–0.9 → ~2–4 tokens/round → ~2–4× on those
  segments. On free-form prose `α ≈ 0` → ~neutral (one verify per token, same as plain decode).
- Because a missed lookup just yields a single normal token, PLD has **no real downside** —
  unlike a too-large draft model (gemma3-4b → gemma3-12b was ≈break-even and could lose).
- For a *guaranteed broad* speedup on gemma4 you would need a real gemma4-mini draft (the `72b1…`
  tokenizer) — external, out of scope here.

## Composition with 46d-2 (already shipped)

- The `(γ+1)`-token verify is a fixed shape → capturable via the 46d-2c device-pos block-padded
  attention + additive mask (generalize seq=1 → seq=γ+1).
- KV rollback is free because the cache is static + device-pos (46d-2b): the write index is just
  rewound.
- So Phase 47 stacks cleanly on the Phase 46 decode engine rather than replacing it.

## Risks / open items

- **`α` is empirical** — 47c must measure it on representative agentic traffic before claiming a win.
- **gemma4-mini draft** (for non-repetitive workloads) is an unmet external dependency; PLD is the
  only draft-free win and is workload-conditional. State this plainly to the user.

## RESULTS (2026-06-24) — 47a/47b/47c done; the verify COST, not acceptance, is the wall

- **47a verify forward (`73fb50a`)** — `gemma4 forward_verify -> [1,seq,vocab]`. Gate: n=1
  byte-exact; n=8/16/32 ~1e-2 logit drift (inherent m=1 GEMV vs m=n GEMM quant diff) but **100%
  argmax agreement** → greedy-lossless in practice. (The early "diverged at n=1" was a re-prefill
  confound — non-deterministic rocBLAS prefill GEMM — removed.)
- **47b drafter (`96f5b0a`)** — `NgramDrafter` + pure `accept_run`, 6 unit tests.
- **47c loop (`402718e`)** — `bench_spec_decode`. Tokens MATCH plain on every prompt. Measured
  gemma4-12b: free-form (photosynthesis) accept 6%, 1.23 tok/fwd → **0.47×**; high-echo (repeat
  an API list) accept 44%, 4.57 tok/fwd → **1.07×**.

### The premise was wrong: the verify is NOT ≈ one decode-time
The "dispatch-bound → γ+1 verify ≈ one decode" rationale is **false for the quantized matmuls**.
A verify over `m = γ+1` tokens routes EVERY quantized matmul (q/k/v/o + MLP + lm_head, ×48
layers) through the **dequant + GEMM** path — there is no fused GEMV for `m>1`, so it materializes
the full f32 weights (the exact 4× weight temp the 45h decode GEMV was built to avoid). **Measured
verify ≈ 4× a single-token decode.** Consequences:
- A round must reach `mean(accepted)+1 > ~4` just to break even.
- A REJECTED draft round costs ~4× a plain step for nothing → **PLD HURTS on low-echo**
  (0.47×). This corrects the earlier "PLD has no real downside" claim above — that only holds
  once the verify is cheap.

### Revised next step
- **47d (was "HIP-graph the verify"; SUPERSEDED) — batched fused quantized GEMV (m=n).** The
  verify's bottleneck is dequant **bandwidth**, not dispatch, so graphing it barely helps.
  Implement a batched fused quantized GEMV (m=n output rows, dequant each weight superblock once
  inline, accumulate n dot products, no f32 weight temp) in `vendor/candle-core/src/quantized/
  hip.rs`, extending the 45h m=1 GEMV; route the `m>1` quantized matmul through it. Target: verify
  → ~1× plain, at which point spec decode is a broad win (≈ tokens/forward ×) AND harmless on
  misses. This kernel also benefits any future batched/training forward.
- **47e (opt) — sampling spec decode** (unchanged): after the batched-GEMV win lands.

NB the foundation (verify forward, drafter, accept loop, free KV rollback) is correct and lives
behind `AKH_BENCH_SPEC` — not wired into production decode. The win is gated on 47d.
- **Verify compute at large γ / long context** could drift from "≈ one decode-time" toward
  compute-bound; keep γ small (≤ 8) and measure.

## RESULTS (2026-06-29) — 47d DONE: the batched GEMV lifts spec throughput 1.68×

**47d batched fused quantized GEMV (`7822d7f`).** Templated-on-`M` kernel
(`akh_gemv_batched_q{4,6}k_f32`) in `quant.hip`: warp-per-weight-row, **dequant each superblock
once**, carry `M` accumulators per lane (one per activation row) so each dequantized weight is
reused across all `M` rows **in registers** — quantized weights streamed exactly once, no f32
temp. A C dispatch macro picks the instantiation for runtime `m` (1..=16); larger `m` routes to
GEMM. Rust: `gemv_batched` FFI + method + a routing branch in `QCudaStorage::fwd` for
`2 ≤ b·m ≤ 16`, contiguous f32 activation, `k%256==0`, **opt-in `AKH_GEMV_BATCH=1`** (default off;
never touches the m=1 decode hot path).

Validated on gfx1100 (RX 7900 XTX) — **the box has the toolchain *and* the GPU** (the compiler is
`amdclang++`, not `hipcc`; earlier "off-GPU" was a mis-probe):
- `hip_gemv_batched_parity`: Q4_K/Q6_K, `m=2,4,7,9` match the dequant-f32 reference to **~1.5e-7
  rel** (f32 rounding — *tighter* than the rocBLAS GEMM path, because the batched GEMV reuses the
  m=1 per-row warp-dot accumulation; this also makes the verify track plain-greedy logits more
  closely than 47a's GEMM verify). m=1 parity tests unchanged.
- gemma4-12b end-to-end, **tokens MATCH plain greedy (lossless)**, apples-to-apples A/B (same
  prompt → identical accepted-token counts; only verify cost differs):

  | workload | accept | tok/fwd | spec tok/s 47c→47d | speedup 47c→47d |
  |---|---|---|---|---|
  | high-echo (JSON echo ×4) | 75% | 5.37 | 14.24 → **23.94** | 1.45× → **2.45×** |
  | free-form (photosynthesis) | 3% | 1.09 | 7.08 (→) | 0.47× → **0.71×** |

  Plain decode 9.79 tok/s in both arms (flag never touches m=1). Net: **batched GEMV lifts
  spec-decode throughput 1.68×**, cutting the verify from ~4× a decode to **~1.5×** (the residual
  is the non-quantized attention/norm/rope work that genuinely scales with `m`, not the weights).

### Conclusion + remaining levers
- **Spec decode is now a strong win on echo-heavy generation (2.45×)** and roughly half the penalty
  on free-form (0.71×). PLD stays **workload-conditional** — it still costs <1× when acceptance is
  near zero, so it is NOT safe to leave unconditionally on.
- **Productization (next):** wire spec decode into production decode (`run_agentic_turn`) for
  echo-heavy turns (agentic / RAG / structured / code), gated so low-echo turns fall back to plain
  decode. Cheapest gate: only draft when the matched suffix is ≥2–3 tokens (more predictive →
  fewer rejected verifies), and/or an adaptive-γ that backs off after consecutive misses. Then flip
  `AKH_GEMV_BATCH` default → on (also speeds short prefills, m≤16).
- **47e (opt) — sampling spec decode**: probabilistic accept + residual resample, now that the
  greedy win is real.

## RESULTS (2026-06-29 cont.) — productized into the production generate path

Spec decode is now wired into production text generation (`generate_inner`), not just the bench
(`22b99e5`):

- **Shared per-token path.** Extracted `DecodeState` + `process_token(token, eos, stop)` (detokenize
  + `StopMode` EOS/brace/newline/chat-end bookkeeping). The proven `decode_loop` now calls it and
  is byte-identical for every non-spec case; `spec_decode_loop` calls it once per accepted/bonus
  token, so stop + detokenization semantics match exactly.
- **`spec_decode_loop`** (gemma4, greedy only): n-gram draft → one `forward_verify` → greedy
  accept-run → emit accepted + bonus, with the implicit free KV rollback (device-pos static cache).
  Borrow-scoped so the `&mut model` verify releases before `process_token(&self)`.
- **Echo-gate.** The production drafter uses **`min_ngram=2`** so low-echo text falls back to cheap
  `m=1` verifies (== plain cost) instead of paying for rejected γ-rounds — this removes the
  free-form penalty the 47c bench saw at `min_ngram=1` (0.71×).
- **Eligibility** (`spec_decode_eligible`): greedy (`temperature==0`) **and** gemma4 **and**
  `AKH_SPEC_DECODE=1`. Sampling / other families always take plain decode (greedy-lossless only;
  sampling needs 47e). Warns if `AKH_GEMV_BATCH != 1` (correct but slow without the batched verify).

Validated on gfx1100, gemma4-12b (`AKH_SPEC_DECODE=1 AKH_GEMV_BATCH=1`), via the `AKH_BENCH_GEN=1`
probe over the real `generate()` path:

| workload | wall plain → spec (incl prefill) | output |
|---|---|---|
| high-echo (repeat a list) | 13.93s → **7.15s (1.95×)** | **byte-identical** (same md5) |
| free-form | ~neutral (+1.7%, noise) | byte-identical |

**Off by default** → zero change to existing decode behavior.

### Per-akh config + runtime tool (2026-06-29 cont., `e11115b`)
Enablement moved from the `AKH_SPEC_DECODE` env flag to typed, per-akh config that a tool flips
live:
- **`AkhConfig.spec_decode`** (`akh.toml` scalar, default false) — persists the choice.
- **`InferenceOptions`** (`src/inference_options.rs`): a lock-free `AtomicBool` bundle held behind
  an `Arc` by the akh's `Engine` and **shared** with the Candle backend, so a flip is observed next
  turn without locking the (busy) model mutex.
- `akhomed` seeds the engine flag from `AkhConfig.spec_decode` and shares the `Arc` into the
  `CandleBackend`; `spec_decode_eligible` reads it (env stays a fallback for standalone benches).
- **`inference_config` agent tool** (`get` / `set spec_decode on|off`): flips the shared `Arc` via
  `&Engine` — the bridge, since a `Tool` only sees `&Engine`, not the agent's `chat_backend`. Safe
  (no KG writes; greedy-lossless). Tests cover the live toggle, bool parsing, and bad input.

### Batched GEMV default-on (2026-06-29 cont., `877dc5a`)
`AKH_GEMV_BATCH` flipped from opt-in (`=1`) to **opt-out (`=0`)**: the validated batched fused
GEMV is now the default for small-`m` quantized matmuls (spec-decode verify + short prefills), so
`spec_decode=true` in `akh.toml` is fast with no second flag — removing the footgun where it was
silently slow without `AKH_GEMV_BATCH=1`. Validated on gfx1100: the candle-core `quantized::hip`
parity suite stays 4/4 (now default-routing `m>1` through the batched kernel), and gemma3-12b
short-prefill `generate()` is **byte-identical** batched-on vs `AKH_GEMV_BATCH=0` (the only path the
default change newly touches for non-gemma4 models; the m=1 decode kernel is separate and
unaffected). `AKH_GEMV_BATCH=0` remains the escape hatch.

Remaining: 47e (sampling spec decode) for non-greedy turns.
