# Phase 47 — Speculative Decoding (Candle, gemma4-12b)

- **Status:** Active (47a in progress)
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
- **Verify compute at large γ / long context** could drift from "≈ one decode-time" toward
  compute-bound; keep γ small (≤ 8) and measure.
