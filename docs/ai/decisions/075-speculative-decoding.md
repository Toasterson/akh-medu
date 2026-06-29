# ADR 075 — Speculative Decoding for the Candle gemma4 Boundary

- **Status:** Accepted (implementation starting, Phase 47a)
- **Date:** 2026-06-24
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
