# ADR 106 — Local Burn-on-Metal training lane and engine-in-the-loop training

- **Status:** Accepted (2026-09-13)
- **Context:** Phase 51 P5 closed with the `small_8k` model uncertified after three cloud release gates (final 1257/1272 tools, three blocking families; [RESULT](../../training/results/tool-native-8k-lane-20260911-RESULT.md)). The follow-up work drifted into a Python MLX harness on the Mac ([inventory](../../training/results/akh-r1-local-mlx-inventory-20260913-RESULT.md)) which produced real neurosymbolic findings — symbolic completion repairs model proposals (17/24 → 24/24), event-calculus temporal gating blocks stale actions, causal schemas predict and verify effects — but no path back into the Rust training harness. Cloud pods cost ≈ $1.59/h and have burned ≈ $57 in three days for no certified model.
- **Plan:** [P6](../plans/2026-09-13-phase51-p6-local-training-neurosymbolic.md). Governance: [ADR 105](105-autonomous-run-governance.md). Related: ADR 102 (tool-native learning model), ADR 104 (effective context, copy-faithful decoding, train-short/fine-tune-long), ADR 096 (step-level rewards), ADR 032 (event calculus), ADR 087 (CPU-only default build).

## Decision

1. **`grug-train` gets a `metal` feature and the Mac becomes the primary training host.** `metal = ["burn/metal"]` selects Burn's wgpu backend with MSL kernels through CubeCL, which emits forward and backward kernels (unlike the inference-only Candle Metal path). `GpuBackend`/`GpuAutodiffBackend`/`gpu_device()` gain a third, mutually exclusive variant so examples stay vendor-agnostic. The default build stays CPU-only (ADR 087).
2. **Burn stays the harness even if the backend changes.** If wgpu/MSL misses the P6 throughput gate, the fallback is a forked or new Burn backend over MLX (`mlx-rs`), maintained as a dependency fork in this repository's `crates/` or `vendor/`. We will change and fork upstream crates when learning requires it; we will not move the training loop to Python.
3. **Remediation trains short, certifies long.** Tool-contract remediation windows run at 2048/4096 context on the RoPE-extended checkpoint; the 8192-context sentinel gate is unchanged (ADR 104 D3 lane, D5 schedule).
4. **akh-medu is the teacher and the judge.** Episode generation, target correction (symbolic completion), validity labels (event calculus `holds_at`), and step rewards (causal `predict_effects`/`verify_prediction`) come from the engine through the recovered Rust bridge examples. The Python MLX harness under `training/akh_r1/` is kept as reference tooling for cross-model comparison only.
5. **Cloud is the measured exception.** A pod restart requires a recorded local throughput miss (P6.1/P6.4), operator approval and a dollar cap in the WI thread.

## Workflow (out of the scheduler)

The P6 work items are driven from the interactive architect session. They must never be dispatched to worker akhs or reviewed automatically. Anima has no scheduler-exclusion flag today (its dispatch queries select solely on `current_phase IN ('ready','working')`), so the convention is: `kind=spike`, `source=architect-session`, `external_ref` naming this ADR, and phases restricted to `draft`/`testing`/`done`/`parked`. Operator holds are not used: they park the item and are meant for pausing lane work. The real fix — a `scheduler_excluded` flag with `exclude_from_scheduler`/`include_in_scheduler` verbs — is filed as Anima WI `01a09ac8-36a0-73d0-bb42-ec15c18dd0c3`; once it lands, the P6 items switch to it and regain `working`.

## Consequences

- One small, contained change to `grug-train` (feature, three type aliases, one device fn) unlocks GPU training on every Apple-silicon machine in the fleet; compile-time exclusivity checks extend to three vendors.
- The four recovered engine repairs and the three bridge examples become supported code with tests, not branch archaeology.
- Memory (16 GB unified) bounds batch size and context; gradient checkpointing (`grad_checkpointing`) and short-context windows are the levers, measured before any pod decision.
- Every experiment carries a control; every window writes a RESULT doc; events go to the board thread. The 2026-09-09..13 failure mode (self-paged research loop) is excluded by ADR 105.
