# Phase 51 — Grug-Brain: From-Scratch BabyLLM Reasoning Organ (Burn)

- **Status:** Plan (grilling-locked design, not built). Numbering provisional — this worktree branches from a base at ADR 085; develop independently used 084/086, so Phase 51 / ADR 087 may need reconciliation on merge (cf. the 079/080 precedent).
- **Date:** 2026-07-05
- **Author:** Fable 5 via Claude Code, grilled by Till.
- **Decision record:** [ADR 087](../decisions/087-grug-brain-baby-llm-training.md)
- **Research basis:** [fable-traces / BabyLLM note](../2026-07-05-fable-traces-babyllm.md), [Tiny LLMs Huge Context](../research/2026-07-05-tiny-llm-huge-context.md), [Project Grug-Brain report](../research/2026-07-05-project-grug-brain.md), [High Grug reasoning-language](../research/2026-07-04-grug-reasoning-language.md).
- **Related engine work:** Phase 41 (VSA Thinking Engine — the peer reasoner), Phase 27 (Live KG→LLM training — Burn/LoRA), Phase 28 (semantic router — the KG bridge), Phase 44 (context assembly — message-level analogue of the KV managers), Phase 24 (sensory grounding — where modality lives, *not* here).

---

## 0. Vision

Train, from scratch, a sub-500M-parameter Tiny Language Model whose **weights hold reasoning + JSON tool-use + syntax**, not facts. All knowledge is injected at runtime from the KG and episodic memory. Its internal reasoning register is **High Grug** — a designed, token-dense language whose tokens *are* KG SymbolIds. It becomes a **dual-process peer** to the Phase 41 VSA Thinking Engine, not a replacement.

**The keystone (why this is an akh organ, not a finetune):** because High Grug tokens are SymbolIds, a reasoning step can **round-trip** — the LLM emits High Grug → the VSA loop reasons/verifies over the *same symbols* → back to the LLM. A closed quality-improvement loop across both substrates. A true **kludge-brain**: neural System-1 and symbolic System-2 sharing one representation, each covering the other's failure mode (notably: binary-VSA analogy is degenerate — the neural organ is exactly the tool for the reasoning VSA can't do; the VSA loop verifies the neural organ via the existing Phase 41 CompositeVerifier).

## 1. Locked decisions (grilling outcomes)

1. **Goal sequencing:** research spike first → reasoning organ (c) as eventual goal. Continuous testing throughout, not one end gate.
2. **Framework: Burn** for training — its `Autodiff<Backend>` + CubeCL generate *both* forward and backward ROCm kernels, the exact gap in the in-house candle fork (which is forward/inference-only; verified by source inspection — `vendor/candle-core` has autograd but the HIP slot ships only inference kernels). Candle stays inference/deploy. Two frameworks accepted, each doing what it is best at.
3. **Isolation:** Burn lives in a new workspace crate `crates/grug-train`, `default-features = false`, **no burn-dataset** (roll-your-own dataloader) — keeps the `rusqlite → libsqlite3-sys 0.35` conflict (the reason `Cargo.toml:143` stubs Burn out today) away from the 113K-LOC engine.
4. **Spike target:** JSON tool-calling with a standard tokenizer. High Grug **deferred** from the spike. Continuous-eval signal lives at the **boundary** (parseable JSON), never on internal reasoning.
5. **Spike data:** a **programmatic Rust generator** you own end-to-end (perfect labels, zero cost, infinite, auto-verifiable). Cloud-teacher (ii) and real Claude Code traces (iii) enter *after* the spike.
6. **Curriculum (organ) — remix of Qwen's order:** pretrain base → SFT cold-start (**tool-calling folded in here**, not late) → RL (**executable/Tool-Zero rewards**, not human-preference RLHF) → KG/tool expansion. **Multi-modal and the model-family are hard-dropped** from the first organ.
7. **High Grug placement:** fluency → **pretrain** (an invented language must be seeded in the pretrain corpus); format → **SFT cold-start**; reasoning quality/efficiency → **RL** (the true reasoning-forging stage). Pretrain corpus = **(i) KG-walk backbone + (iii) English↔High-Grug bilingual ramp** off the current KG. **High Grug must be designed as a language first** (own work item; blocks P3).
8. **Attention/context managers:** the spike uses **vanilla attention behind a clean `Attention` seam** (seed, not throwaway). Managers are a **late context-scaling phase**: RoPE-extrapolation + EM-LLM (pure inference-time, no retrain) first; DuoAttention (needs a gate-identification training stage) only if the inference-time two are insufficient.
9. **Eval harness (continuous):** valid-JSON rate + schema adherence + argument correctness + an **anti-collapse entropy/diversity alarm**, on a **frozen whole-tool holdout**, logged as a time series, all in-crate off the generator. Tool-calling eval in the spike; reasoning-quality eval in P3.
10. **Integration:** (C)+(B) — shared-SymbolId substrate + dual-process peer to the VSA Thinking Engine; the founding "reasoning stays in VSA space" thesis is **preserved**, not overturned.
11. **Ops:** factory = a dedicated AMD **gfx1100** (7900 XTX) box, kept off the live 3-akh fleet host; **Mac Mini = workbench** (author code, run the generator, 50-step CPU sanity runs). Burn `Recorder` checkpoint-every-N (resumable), **setsid-detached** runs, eval time-series co-located with checkpoints.

## 2. Phase spine

Cross-cutting from day one: the eval harness (§decision 9) and the `crates/grug-train` isolation (§decision 3).

- **P0 — Burn beachhead (Spike-0, plumbing).** `crates/grug-train`; Burn `default-features=false`, no burn-dataset, roll-your-own dataloader. Vanilla ~2–10M transformer *with the `Attention` seam*. Byte tokenizer. Programmatic tool-call generator. Train on **CPU (ndarray)** → overfit → loss↓ + valid-JSON; then flip to **CubeCL/ROCm**, measure tok/s + VRAM.
  **Gate:** Burn trains on the 7900 XTX at acceptable throughput. Knocks down R1 (Burn works here), R2 (a from-scratch transformer trains), R3 (fast enough on AMD) — in that order.
- **P1 — Foundation + task (Spike-1).** Real BPE tokenizer over Apertus. Pretrain foundation LM (comprehension) → SFT tool-calling on generator data.
  **Gate:** parses a natural-language request → correct JSON on held-out tools, above threshold, with the anti-collapse alarm green.
- **P2 — Data scaling.** Add (ii) cloud-teacher diversity + (iii) the real-trace extractor (deterministic, provenance-preserving, capability-filtered, collapse-eval-guarded per the fable-traces note). Add **reasoning-quality eval** to the harness here.
- **P3 — Reasoning curriculum (organ base).** Full remix: pretrain (Apertus + Phi-style reasoning textbooks + High Grug via KG-walk) → SFT cold-start (tool-calls + High Grug format + gap-token) → RL (executable/Tool-Zero rewards).
  **Blocked on:** High Grug language design.
- **P4 — KG / gap-token bridge.** Wire the High Grug **gap-token** ("I am missing concept X" in one token) → KG retrieval → context injection. Reuses the Phase 28 semantic router.
- **P5 — Context scaling.** RoPE-extrapolation + EM-LLM (inference-time) first; DuoAttention gate-ID stage only if needed. Enables big textbook/KG injection at runtime.
- **P6 — Integration as reasoning organ (c).** Shared-SymbolId round-trip with the VSA Thinking Engine; dual-process scheduling; VSA CompositeVerifier verifies neural output.

**Parallel work items (not blocking P0–P2):** High Grug *language design* (blocks P3); the real-trace extractor (feeds P2, §fable-traces note is the spec).

## 3. Open questions (carried, not yet resolved)

- **High Grug language design** — tokens, grammar, the gap-token, modifiers, custom-token-ID ↔ SymbolId mapping mechanics, and the English↔High-Grug bilingual alignment. Its own grilling/plan before P3.
- **Apertus** — license/coverage adequacy as the pretrain foundation; corpus prep pipeline.
- **Capacity risk** — does a <500M model retain enough syntactic competence (and eventually 3-language: Rust/C#/C) without collapse (the fable-traces warning)? P1 is the first real read.
- **Round-trip mechanics (P6)** — the exact protocol for High-Grug-token ↔ HyperVec hand-off and dual-process scheduling.
- **DuoAttention cost** — gate-identification training cost on the 7900 XTX, only assessed if P5's inference-time managers fall short.

## 4. Success criteria

- **Spike (P0–P1):** Burn trains from scratch on the 7900 XTX; the model parses NL → valid schema-adherent JSON tool calls on held-out tools; anti-collapse alarm stays green. This alone answers "can we train a reasoning-shaped model on our own sovereign hardware."
- **Organ (P3–P6):** a High-Grug-reasoning neural peer that round-trips with the VSA loop, fetches knowledge from the KG via the gap-token, and improves reasoning quality on tasks the VSA loop can't do alone — measured against the VSA-only baseline.
