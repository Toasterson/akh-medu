# ADR 087 — Grug-Brain: From-Scratch BabyLLM Training via Burn

- **Status:** Accepted (design; not built). Number provisional — reconcile on merge (worktree branches at ADR 085; develop used 084/086).
- **Date:** 2026-07-05
- **Deciders:** Till, Fable 5 (grilling session)
- **Plan:** [Phase 51 plan](../plans/2026-07-05-phase51-grug-brain-baby-llm.md)
- **Supersedes/relates:** extends the "no *cloud* LLM" thesis (ADR 022) and coexists with — does not replace — the Phase 41 VSA Thinking Engine (ADR 052/085).

## Context

We want a from-scratch, sovereign, sub-500M reasoning model whose weights hold reasoning + JSON tool-use + syntax, with facts injected at runtime from the KG. The viral `AliesTaha/fable-traces` "model" is a joke ("Egypt won" mode-collapse satire), but it encodes a real warning about reckless trace distillation. Two Gemini research threads (tiny-LLM context managers; Project Grug-Brain) supply the method. akh already has: a VSA reasoning engine (Phase 41), a candle inference fork with ROCm (Phase 45), a LoRA/bridge training scaffold (`src/training/`, `src/vsa/neural_bridge.rs`), and a semantic router (Phase 28).

## Decisions

1. **Train in Burn, not the candle fork.** Burn's `Autodiff<Backend>` + CubeCL emit both forward *and* backward ROCm kernels. Verified by source inspection: the in-repo "training" (`BridgeTrainer`) is a hand-rolled single-linear-layer SGD, not a transformer trainer; `vendor/candle-core` has autograd but its HIP slot ships **inference-only** kernels. Candle stays inference/deploy. Two frameworks, each at its strength.
2. **Isolate Burn in `crates/grug-train`** with `default-features = false` and no burn-dataset, quarantining the `libsqlite3-sys` conflict that currently stubs Burn out of the main crate.
3. **Spike on JSON tool-calling** with a standard tokenizer and a **programmatic data generator** (owned ground truth, zero cost, auto-verifiable). Boundary-only eval (parseable JSON). Cloud-teacher and real-trace data come after.
4. **Curriculum = remix of Qwen's order:** pretrain → SFT (tool-calling folded in) → RL (executable/Tool-Zero rewards) → KG/tool expansion. Multi-modal and the model-family are dropped.
5. **High Grug threads all stages:** fluency in pretrain (seeded via KG-walk + English↔High-Grug bilingual ramp), format in SFT cold-start, reasoning quality in RL. High Grug must be *designed as a language* first.
6. **Attention managers behind a seam, built late:** RoPE + EM-LLM (inference-time) first; DuoAttention (needs gate-ID training) only if needed.
7. **Continuous eval:** JSON validity + schema + arg-correctness + anti-collapse entropy alarm on a frozen holdout, in-crate.
8. **Integration = shared-SymbolId dual-process peer.** High Grug tokens are SymbolIds, enabling LLM↔VSA round-trip; the neural organ is a verifying peer to the VSA loop, preserving "reasoning stays in VSA space."
9. **Ops:** dedicated gfx1100 factory off the fleet host; Mac Mini workbench; checkpoint-resume + setsid-detach.

## Consequences

- **Positive:** sovereign (own data/weights/provenance, no cloud); Burn autodiff removes the GPU-backward gap; the kludge-brain design turns a known weakness (degenerate binary-VSA analogy) into a division of labor; the spike gives a cheap, honest feasibility gate before any multi-week pretrain.
- **Negative / risks:** two ML frameworks to maintain; Burn's young ecosystem means hand-writing transformer blocks; capacity risk (a <500M model may transfer style without capability — the fable-traces collapse); High Grug is an unbuilt language on the critical path to P3.
- **Rejected alternatives:** (A) replace the VSA thinking loop — abandons the founding thesis on an unproven model; (D) replace only the boundary translator — wastes the organ's point; making the candle fork train on GPU — reinvents what Burn gives free, on an inference-built fork; cloud-teacher data for the spike — cost + provenance + muddied signal.
