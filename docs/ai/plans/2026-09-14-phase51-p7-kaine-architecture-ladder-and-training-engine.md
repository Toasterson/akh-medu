# Phase 51 P7 — ADAM → Kaine architecture ladder and the training-engine memory program

- **Status:** Planned (2026-09-14), **gated on P6**: no P7 item starts until the P6.4/P6.5 window queue (`runs/p6-4/queue-20260914.txt`) and its release gate have finished (operator decision 2026-09-14). All items are HITL: `source=architect-session`, `draft` until the operator promotes them, never `ready`/`working`
- **ADRs:** [ADR 108](../decisions/108-kaine-architecture-ladder.md) (architecture ladder), [ADR 109](../decisions/109-training-engine-memory-program.md) (engine memory), [ADR 110](../decisions/110-kaine-character-guided-reward-live-environment.md) (character, guided reward, live environment — extension of 2026-09-14, §below); governance [ADR 105](../decisions/105-autonomous-run-governance.md); lane [ADR 106](../decisions/106-local-burn-metal-lane-and-engine-in-the-loop-training.md); attention memory [ADR 107](../decisions/107-training-memory-turboquant-vs-recompute-attention.md)
- **Anima:** project `019d9844-ce74-7993-a9f0-c953b8f04a4e`, plan `019f90cb-2e66-7f92-ba88-a4060be97c6e`
- **Predecessor:** [P6 plan](2026-09-13-phase51-p6-local-training-neurosymbolic.md) (active; P7 does not pre-empt the P6.4/P6.5 window queue)

## Goal

Decide Kaine's architecture from measurements taken on the ADAM ladder (33M → 350M → ≈1.1B) rather than from extrapolation, and bring
the training engine's memory down far enough that each rung trains on the cheapest hardware the literature says it can: 33M–125M on
the Mac, 350M and 1.1B on one pod GPU, Kaine on whatever P7.8 shows is needed.

## Starting facts

| fact | value | source |
|---|---|---|
| Current model | `small_8k`: d_model 512, 8 heads, 8 layers, d_ff 2048, RoPE, ctx 8192, chunk 1024, LayerNorm, GELU, no dropout | `crates/grug-train/src/transformer.rs` |
| Attention memory | O(L·d) retained per layer (`RecomputeAttention`), 8192×2 = 7.0 GB on the M2 | ADR 107 |
| Residual memory | `small_8k` 2.7 GB activations + 0.4 GB state; `350m_8k` ≈18 GB + 4.2 GB | ADR 107 |
| Checkpointing | per-op; Linear/embedding/matmul inputs always retained | `burn-autodiff-0.21.0/src/checkpoint/strategy.rs:62-96`, `src/ops/module.rs` |
| Fusion | not enabled (`metal = ["wgpu"]` lacks `fusion`) | `burn-0.21.0/Cargo.toml:147-157`, `burn-wgpu-0.21.0/src/lib.rs:63-64` |
| LayerNorm | statistics reduced in backend dtype, no f32 cast | `burn-backend-0.21.0/src/backend/ops/modules/base.rs:1103-1129` vs `burn-nn-0.21.0/src/modules/norm/rms.rs:74` |
| Cross-entropy | full `[b·s, vocab]` logits + log_softmax retained | `crates/grug-train/src/train.rs:500-527` |
| Optimizers in Burn | Adam, AdamW, Adagrad, RMSProp, SGD, LBFGS, Adan, Muon; no Adafactor/8-bit | `burn-optim-0.21.0/src/optim/mod.rs:9-30` |
| f16 lane | parked: loss 6.30 vs 3.76, 2× slower; not a loss-scale problem | `training/results/p6-1-metal-smoke-20260913-RESULT.md:242-249, 337-340` |
| Copy floor | fixed-state model needs ≥ L·log₂(D) bits to copy an L-token span | Jelassi 2402.01032 Cor. 2.8 |
| Hybrid ratio | 2–15 attention layers of 48 within 0.06 ppl at 350M; 3 global layers recover recall at 300M | Mamba-2 2405.21060, Hymba 2411.13676 |
| Small-scale verdict | 125M Mamba < OPT on BabyLM; 70M attention > 1.4B gated-conv on recall | 2510.19493, Zoology 2312.04927 |

## Work items

| WI | scope | exit condition |
|---|---|---|
| `01a0a1a2-aa69-7c72-9596-76b319e9cd9f` P7.1 fusion + f32-stats LayerNorm + fp32 softmax/CE + f16 gate rerun | enable Burn `fusion` on metal/gpu/cuda; hand-written LayerNorm with f32 statistics; fp32 accumulation for softmax and masked CE under f16; dynamic loss scaling | fusion tokens/s + memory table at 2048×1 / 8192×1; f16 loss@4 within 0.05 of f32 on the 2048×1 smoke **and** faster, else f16 stays parked with the measured reason |
| `01a0a1a2-cf26-78d3-853d-f1d8c0771074` P7.2 `RecomputeFeedForward` | custom `Backward` for up-proj → GELU → down-proj, row-chunked, saves only the block input; vendored patch as ADR 107 | CPU gradient equality to 1e-4; ≥ 50 % of the 18 GB removed at `350m_8k` 8192×1 (analytic + measured on the largest shape that fits); overhead < 30 % |
| `01a0a1a2-ef2c-7fd1-802f-aa20e74c51e5` P7.3 Cut Cross-Entropy | blockwise loss op with online logsumexp; reference path kept for tests | gradient equality at vocab 8000 / seq 512; memory delta at `small_8k` 8192×2; identical loss curve |
| `01a0a1a3-157a-76d1-a38b-424107f82fc7` P7.4 optimizer selector | `adamw` / `muon` (2-D weights) + AdamW (embeddings, norms, head) / Adafactor fallback | loss parity on the 2048×1 smoke and one remediation window on the unchanged sentinels; state bytes halved |
| `01a0a1a3-439f-7420-baa8-ed626623cbe8` P7.5 SWA + spaced global layers | `with_attention_layout(window, global_layers)`; windowed `RecomputeAttention`; presets `akh_tool_small_8k_swa` (w 1024, global {2, 5}) and 350M; serving KV cache honours the window | windowed gradient test; copy probe (8–64-token UUID/path spans at 1k–8k distance) and MQAR parity vs full attention; constrained gate on 401,577 after one window within 3 cases; memory + tokens/s at 8192×2 |
| `01a0a1a3-6c2b-78a0-abc4-3ff3d95230e8` P7.6 MobileLLM presets + NoPE rung | GQA, SwiGLU, block layer sharing; `adam_33m_deep`, `adam_33m_shared`, `nope` variant; matched-param, matched-token comparison at 2048/4096 | RESULT table (tool gate, knowledge gate, copy probe, tokens/s, memory) at 33M; winning shape becomes the ADAM small preset |
| `01a0a1a3-a8a2-7540-bc02-98db4507bbcd` P7.7 Gated DeltaNet parallel heads @ 350M (depends P7.5) | `GatedDeltaNet` (chunk 64, WY form) + `ParallelHead` gating; `adam_350m_hybrid`; pod run under ADR 105 | recurrence gradient test; MQAR + UUID/path probes + akh gates for hybrid vs SWA control in one table; go/no-go for 1.1B and Kaine |
| `01a0a1a3-d7c3-7602-a6d2-0fc09798199e` P7.8 spike: 1.1B fit (depends P7.2–P7.4) | `adam_1_1b_8k` preset (d_model 2048, 24 layers, GQA, SWA layout); 50-step pod smoke, levers on/off, cap ≤ 2 h A100 with in-thread approval; Burn multi-device audit | measured fit table; written decision "1.1B trains on one A100 at 8192: yes/no, batch"; the named blocker for the Kaine size |
| `01a0a1a4-03cb-75f3-b666-67eba03eb420` P7.9 spike: memory-layer seam (depends P7.5, informed by P7.7) | design for a product-key `MemoryLayer` whose keys/values are learned or projected from HyperVec/KG embeddings; episodic writes → `DerivationKind::AgentConsolidation`; copy-bound implications | ADR draft with the seam, parameter budget at Kaine size, go/no-go for a prototype rung |

## Sequence

1. **Engine first, on the Mac, after the P6 window queue and its gate have finished:** P7.1 → P7.2 → P7.3 → P7.4. Each lands behind a
   flag with its gradient test and enters a lane only after its gate. The P6 recipe (f32 + recompute attention) is never changed
   retroactively; P6's fine-tuning result is recorded before any P7 lever touches the lane.
2. **Architecture at 33M:** P7.5 and P7.6 on the Mac from the foundation corpus, scored on the unchanged sentinels. P7.5 is also an
   engine memory lever (L×w score chunks), so it may move ahead of P7.3/P7.4 if the P7.2 measurement says activations are still the floor.
3. **350M rung:** P7.7 on a pod (ADR 105 WI, dollar cap, exit condition) with the P7.5 350M preset as control. This is the first
   measurement that can change Kaine's backbone.
4. **1.1B rung:** P7.8 spike; only after P7.2–P7.4 so the arithmetic is measured, not assumed.
5. **Kaine:** size and backbone chosen from the P7.7 + P7.8 tables; P7.9 decides whether engine memory enters the model as a layer.

## Budget and gates

- Mac lane (ADR 106): a P7 engine change must not reduce tokens/s below the P6 gate (2048-context window ≤ 24 h) unless it buys the
  8192 fit for a shape that did not fit before.
- Pod runs (P7.7, P7.8): each has its own Anima WI thread with operator approval, a dollar cap (P7.8 ≤ 2 h A100 ≈ $4; P7.7 sized in its
  thread from the P7.8 tokens/s measurement) and an exit condition; events go to the Agent Board run thread, never a chat wake-up.
- Every WI writes a RESULT doc under `training/results/` and a thread post; architecture WIs also record the constrained-gate delta
  against candidate 401,577 so ADAM's release line is never regressed silently.

## What this plan does not do

- It does not change the ADAM release gate or the P6 remediation windows.
- It does not adopt Titans/TTT/ATLAS, activation quantisation, LOMO, FP8 or an engine swap (ADR 108 §Decision 6, ADR 109 §Decision 2, 4).
- It does not fix Kaine's parameter count; that is the output of P7.7 and P7.8.

## ADR 110 extension (2026-09-14): character, guided reward, live environment

Inputs: Cantrill, *Intelligence is not Enough* (Monktoberfest 2023); Hatcher on Gibson, *we act to perceive*; Oxide's principles and
values. Decisions are in [ADR 110](../decisions/110-kaine-character-guided-reward-live-environment.md): (1) the crew-of-8 slots carry
Oxide's 15 values as their public trained biases, the three principles are captain invariants enforced by the runtime; (2) reward is
guided and fine-grained — every step scored, every failed step yields a corrected sibling, a five-component reward vector
(effect / validity / provenance / disposition / budget) with a named reduction policy, weights in [0, 1], never a penalty, no
scalar-judge policy gradient; (3) episodes come from the real environment — `Engine` + the full `ToolRegistry` + Agent Board + Anima +
executor `Backend`, run as a sandboxed dojo tenancy, one dispatch per episode, end to end.

### New work items

| WI | scope | exit condition |
|---|---|---|
| `01a0a1b9-eb77-7390-9730-361a9d92faae` P7.10 `AkhEnvironment` dojo tenancy | stepped environment over the real Engine, full `ToolRegistry`, Agent Board (`DjehutyClient` + `akh-session-services` inbox intake, `coordination/*@v1`) and Anima (`get_work_item`, `submit_session_change`, `git_publication`, thread verdict) with the per-step recorder; full-registry `ToolContractSnapshot`; network tools on recorded fixtures | one authored dispatch runs end to end with every step recorded to JSONL; contract hash recorded; decontamination green against the four sentinel sets; dojo tenancy IDs documented; negative check shows nothing written outside the tenancy |
| `01a0a1ba-255a-7f70-afa8-85016faad6ab` P7.11 guided reward vector + trainer (depends P6.5, P7.3) | sidecar schema v1 (five components + `voice_called_for` + `corrected_sibling`), versioned `weight_policy`, `train_knowledge --record-weights use --weight-policy v1`, weighted loss through Cut CE | gradient equality of the weighted blockwise loss vs the naive weighted path; one A/B window from 401,577 (vector + corrected siblings vs E3 scalar) on the unchanged sentinels, delta recorded |
| `01a0a1bc-3abe-7042-9c17-5acdd9c1c551` P7.12 crew × Oxide corpus (depends P7.10, P7.11) | `slot` + `disposition` labels on the kata drills (15 values / 8 slots); emergent-property-bug and act-to-perceive families generated in the tenancy with passive controls; captain-selection labels; blinded social rubric; White→Yellow belt gate | corpus + generator hash; belt gate passed on the ADAM candidate without regressing the 1,272-case gate by more than 3 cases; RESULT with per-slot and per-family numbers |
| `01a0a1bc-c798-7003-9879-934fa3568507` P7.13 environment release gate (depends P7.12, P7.5) | frozen quarantined dispatch-episode set with deterministic final-state verifiers + reward-vector components + blinded rubric; evaluator under `crates/grug-train`; RESULT row every rung and every Kaine candidate emits | gate definition frozen (episode/verifier/rubric hashes); ADAM baseline recorded; runs within the ADR 106 local budget or the excess is measured and a pod run proposed under ADR 105 |

### Amendments to P7.1–P7.9 (recorded on the Anima items)

- P7.3: the blockwise loss accepts per-record weights; the gradient test covers the weighted path.
- P7.5–P7.8: each RESULT table gains the P7.13 environment-gate row once the gate exists, the P7.12 belt gate until then; P7.5 keeps
  board/Anima ID spans inside the global-layer copy path; P7.7's Kaine go/no-go is taken on all four gates; P7.8's memory table includes
  the weighted-loss path.
- P7.9: board/Anima episodic writes are the first use case for the memory-layer seam; `disposition`/`effect` are candidate write signals.

### Sequence and gating

Unchanged: nothing under this extension starts before the P6 window queue and gate are recorded, and every item stays `draft` +
`source=architect-session` until the operator promotes it. P7.10 is CPU-side harness code with no GPU or recipe contention; promoting
it earlier is an operator option, not a default. Order once open: P7.10 → P7.11 (after P6.5's E3 result and P7.3) → P7.12 → P7.13;
P7.13 then becomes a row in every architecture rung's RESULT table.

### Budget

- P7.10–P7.12 run on the Mac (CPU for the harness and corpus; the belt-gate window under the ADR 106 lane gate).
- P7.13's gate must fit the ≤ 12 h M2 evaluation budget or state the excess; any pod use is its own ADR 105 WI with a dollar cap.
- Dojo tenancy resources (Anima project, board tenancy tokens, dojo repository) are created once and named in the P7.10 RESULT.

