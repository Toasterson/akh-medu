# ADR 075 — Speculative Decoding for the Candle gemma4 Boundary

- **Status:** Accepted — 47a–47d + productization into the generate path, validated on gfx1100 (2026-06-29); default-on toggle + 47e (sampling) remain
- **Date:** 2026-06-24 (updated 2026-06-29)
- **Context plan:** [Phase 47](../plans/2026-06-24-phase47-speculative-decoding.md)
- **Supersedes / relates:** builds on [ADR 074] (Phase 46d HIP-graph decode) and the device-pos
  static KV cache; complements ADR 070 (selectable chat backing) — spec decode is a Candle-side
  decode-speed feature, no effect on the Ollama backing.

## Context

Candle decode is dispatch-bound (~8.8 % of bandwidth roofline; ~11× spare compute). Phase 46d-2
attacked dispatch cost with HIP graphs (+1.10× net, diluted by per-token host-sync embed/sampling).
Speculative decoding is the orthogonal lever: verify several candidate tokens in **one** target
forward. In a dispatch-bound regime a `γ+1`-token verify costs ≈ one decode-time, so spec decode
collapses up to `γ+1` sequential target forwards into one — a better fit for us than for
compute-bound setups.

The user requires the **Gemma 4 series (gemma4-12b)**, not gemma3.

## Decision

1. **Implement speculative decoding in the Candle boundary, lossless-greedy first.** The MVP accepts
   a draft token iff it equals the target's greedy argmax, so the generated sequence is
   **byte-identical to plain greedy decode** — the correctness gate (same discipline as 46d-2).
2. **gemma4 uses a draft-model-free drafter: prompt-lookup / n-gram decoding (PLD).** gemma4-12b's
   tokenizer (`72b1044584…`) differs from the gemma3 family (`47a57159…`) despite a shared
   262 144 vocab, and no smaller gemma4 exists on disk — so **no separate draft model is viable**
   for gemma4. PLD drafts from the target's own vocab (looking up recent n-grams in
   `prompt + output`), eliminating the tokenizer-compatibility requirement entirely.
3. **One shared primitive: an all-position verify forward** returning `[1, seq, vocab]` (the
   existing prefill `seq_len>1` path minus the last-position slice). Every drafter (PLD now, a
   gemma4-mini draft or Medusa heads later) reuses it.
4. **Reuse the Phase 46d-2 static device-pos KV cache for free rollback** — on partial acceptance
   the write index is simply rewound; rejected columns are overwritten next round.

## Alternatives considered

- **Separate draft model (gemma3-4b → gemma4-12b).** Rejected: incompatible tokenizer (`72b1…` vs
  `47a5…`) — token IDs/specials don't align. (Even within gemma3, 4b → 12b was ≈break-even because
  the draft is too large; only 4b → 27b or a tiny 270m/1b draft pays. Moot for gemma4.)
- **Medusa self-draft heads.** Target-vocab (no tokenizer issue) and workload-agnostic, but needs
  training. Deferred — revisit if PLD's workload-conditional win is insufficient.
- **Early-exit self-speculation.** Target-vocab but depends on the model being coherent at shallow
  depth (gemma typically isn't). Rejected for the MVP.
- **Obtain a gemma4-mini draft.** The only route to a *guaranteed broad* (non-repetitive) speedup,
  but an unmet external dependency (needs a published vocab-`72b1…`-compatible small gemma4 to
  fetch + quantize). Recorded as a future option, not the plan.

## Consequences

- **gemma4 spec decode is workload-conditional.** PLD wins where output echoes context (agentic /
  RAG / code / structured output — the intended workload); it is ~neutral on free-form prose and
  never materially hurts (a missed lookup falls back to one normal token). This trade is explicit
  and acceptable; broad speedup needs a real draft model (above).
- **Byte-exact correctness** is preserved (greedy MVP) — verifiable against plain greedy decode.
- **Composes with, doesn't replace, Phase 46.** The fixed-shape `γ+1` verify is HIP-graph-capturable
  via 46d-2c; KV rollback is free via the device-pos cache.
- New surface: an all-position verify forward (gemma4), an n-gram drafter, and the accept/rollback
  loop. The mask/KV/graph machinery already exists.

## Update (2026-06-29) — empirical outcome; the verify-cost finding revised the plan

The "dispatch-bound → `γ+1` verify ≈ one decode-time" premise was **measured false for the
quantized matmuls** (47c): a verify over `m=γ+1` rows had no fused GEMV, so every quantized matmul
took the dequant→full-f32→rocBLAS-GEMM path — verify cost ≈ **4× a single decode**, which made PLD
a *loss* on low-echo workloads. Dispatch was never the verify's bottleneck; **dequant bandwidth**
was. So the originally-listed "HIP-graph the verify" optimization was dropped in favour of a
**batched fused quantized GEMV (m=n)** — keep the m=1 warp-per-row, dequant-each-superblock-once
structure but carry `M` accumulators per lane, reusing each dequantized weight across all `M`
activation rows in registers (weights streamed once, no f32 temp).

Measured on gfx1100 (RX 7900 XTX), gemma4-12b, lossless (tokens match plain greedy):

| workload | accept | tok/fwd | speedup 47c (GEMM verify) → 47d (batched GEMV) |
|---|---|---|---|
| high-echo | 75% | 5.37 | **1.45× → 2.45×** |
| free-form | 3% | 1.09 | **0.47× → 0.71×** |

The batched GEMV cut the verify from ~4× a decode to ~1.5× (the residual scales with `m` via the
*non-quantized* attention/norm/rope, not the weights) and lifted spec-decode throughput **1.68×**.
**Net decision:** PLD remains **workload-conditional** (still <1× at near-zero acceptance), so spec
decode must be **echo-gated** in production rather than always-on — but where output echoes context
(agentic / RAG / structured / code) it is now a clear win. The batched GEMV is opt-in
(`AKH_GEMV_BATCH=1`, default off) until spec decode is wired into production decode; it also
accelerates short prefills (m ≤ 16) and any future batched/training forward.

**Productized (2026-06-29).** Spec decode is now wired into production text generation
(`generate_inner` → `spec_decode_loop`), gated to **greedy + gemma4 + `AKH_SPEC_DECODE=1`** (off by
default). The plain and speculative loops share a `process_token` helper so detokenization/stop
semantics are byte-identical; the production drafter uses **`min_ngram=2`** as an echo-gate so
low-echo turns fall back to cheap `m=1` verifies and incur no penalty. Validated on gfx1100,
gemma4-12b: high-echo `generate()` wall 13.93s → 7.15s (**1.95×**, incl prefill) with
**byte-identical** output; free-form ~neutral (+1.7%), output byte-identical. Because spec decode is
greedy-lossless, enabling it cannot change *what* is generated — only how fast. Sampling turns still
need 47e; the `AKH_GEMV_BATCH` default-on flip (also speeding short prefills) is the remaining
toggle.
