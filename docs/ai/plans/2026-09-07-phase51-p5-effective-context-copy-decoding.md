# Phase 51 P5 — Effective context and copy-faithful decoding for the tool-native BabyLLM

- **Status:** Active (2026-09-07)
- **ADR:** [ADR 104](../decisions/104-babyllm-effective-context-and-copy-faithful-decoding.md)
- **Anima:** project `019d9844-ce74-7993-a9f0-c953b8f04a4e`, plan
  `019f90cb-2e66-7f92-ba88-a4060be97c6e`
- **Predecessors:** [2026-09-03 adaptive exams / native multimodal](2026-09-03-phase51-adaptive-exams-native-multimodal.md),
  [4k lane result](../../training/results/tool-native-4k-lane-20260906-RESULT.md)

## Goal

Turn the measured 4096-context `small` result (five of seven workflows at
0.78–1.00, two blocked by hex-copy fidelity) into a certified text model by
(1) making the runtime guarantee what the grader demands, (2) removing the
tokenizer's contribution to the failure, (3) doubling the trained window
with an architecture that can be served longer, and (4) fixing the training
schedule the controller already told us about. Then measure BabyLLM against
the Phase 50 benchmark program instead of guessing.

## What is decided (ADR 104) and what is measured

| decision | mechanism | first measurement |
|---|---|---|
| D1 copy-faithful decoding | byte-level value constraint + grammar on the KV-cached path; runtime and evaluators share it | re-probe the existing 4096 checkpoint: `acquire_textbook`, `revise_from_new_edition` must show zero `SchemaViolation` |
| D2 identifier-aware tokenizer | digits split individually, ≥8-char hex runs split per char, recorded in the artifact | token count of a 64-hex string = 64; foundation perplexity at 8k comparable to 60.9 |
| D3 RoPE + chunked attention | `PositionalEncoding::Rope`, `attention_chunk`, `BalancedCheckpointing` knob, `akh_tool_small_8k` | fits 80 GB at batch 2 × 8192; CPU parity tests |
| D4 no 100k window | effective context via ADR 065 strategies + ADR 102 tools | `knowledge.length_generalization` sentinel at 4k/8k/12k/16k |
| D5 learning schedule | `--ordinary-learning-rate-scale 0.2`; frontier expansion over epochs | no healthy family regresses across windows |
| D6 serving budget | `AKH_GRUG_MAX_NEW_TOKENS` 768 | note-write actions parse |

## Work items

| WI | scope | status 2026-09-07 |
|---|---|---|
| `01a07a7e-6333-7ed2-b251-fb645d3aa3b3` P5.1 | D1 + D6: constrained decoder in `grug-data`/`grug-train`, wired into `KnowledgeModel`, `GrugChatBackend`, scorer, probe, sentinels, all-tool matrix | implementation dispatched (worktree) |
| `01a07a7e-7ac6-7d63-9819-58f19f79cc22` P5.2 | D2: `PreTokenization` in `bpe.rs`, `train_foundation` flags, manifest fields | implementation dispatched |
| `01a07a7e-9836-7e71-b914-fc1f1552a52c` P5.3 | D3: RoPE, chunked attention, checkpointing knob, 8k presets, shape/manifest compat | implementation dispatched |
| `01a07a7e-b81d-7be3-9af4-6a852b0ba3a0` P5.4 | D5: loop LR flags (done), `knowledge.length_generalization` family (open) | flags implemented; family open |
| `01a07a7e-de30-7963-92ee-7f164bb214b0` P5.5 | the 8k lane run + release gate + Phase 50 tiny-LLM entry | blocked on P5.1–P5.4 and on pod approval |

## Sequence

1. **Land P5.1–P5.4 on `phase51-p4-run`** (or a `phase51-p5` branch) as
   separate commits; all CPU tests green; `cargo clippy` clean. Every
   pipeline binary is rebuilt from that one commit on the pod (ADR 102
   schema-skew rule; `bootstrap-head.sh`).
2. **Re-probe the 4096 checkpoint with D1** (pod restart needs approval;
   ≈1–2 h). Command shape:
   `run_tool_native_adaptive_probe.py --bundle …/knowledge-model-small-v1 --max-new-tokens 640 --probe-dir adaptive-probe-constrained-v1`.
   Outcome A: both hex-copy families join their siblings (≥0.78) → the
   weights already have the workflow; proceed to 3 for the window and
   tokenizer, and expect certification on the 8k lane. Outcome B: a family
   stays red with zero schema violations → residual is workflow competence;
   read the per-step closed-loop diffs before deciding between more
   curriculum difficulty and the `350m_8k` rung. Write the numbers into a
   results doc that names the decoder.
3. **8k foundation** (`--model akh_tool_small_8k`, batch 2 × 50k steps,
   hex-aware tokenizer). First a 200-step memory/throughput smoke with and
   without `BalancedCheckpointing`; if 8192 does not fit at batch 2, open
   the sliding-window fallback WI before spending the run. Prune
   intermediate checkpoints as the run advances (150 GB volume).
4. **SFT + probe + loop** with `--max-new-tokens 768`,
   `--ordinary-learning-rate-scale 0.2`, eight families (the seven plus
   `length_generalization`). Controller stop remains the authority.
5. **Release gate → OCR → cross-host gate → install → publish** exactly as
   the 2026-09-04 runbook, with the offline gates' generation budget raised
   to 768.
6. **Serving length.** Set RoPE `scale` to the longest
   `length_generalization` bucket that passes; record it in the installed
   artifact's manifest.
7. **Phase 50 tiny-LLM track.** Run the certified model through the
   benchmark harness's agentic corpus; publish the leaderboard row. This is
   the coding-capability answer — measured, with the same verifiers as the
   capable models.

## Budget

Pod time on the existing A100 (`4ypc4li9y537f2`, $1.59/h): re-probe
≈ $3; 8k foundation ≈ 12–15 h, SFT ≈ 4 h, probe + two loop windows ≈ 4 h
→ ≈ $32–37 for the lane. No new billable resources. Every restart requires
the user's explicit approval.

## Risks

- Chunked attention reduces temporaries but autodiff still retains per-chunk
  softmax outputs for backward; only recompute (`BalancedCheckpointing`)
  changes total activation memory. The 200-step smoke decides.
- The value constraint changes what "grounded" measures; results before and
  after ADR 104 are not comparable. Documents must say which decoder.
- Hex-aware splitting lengthens prompts; the v2 corpus max (3,374 tokens at
  the old tokenizer) must be re-measured at the new tokenizer before SFT
  (`compacted=0` is the acceptance).
- If the `Gap` alternative is not representable in the grammar, abstention
  would silently vanish under the constraint; P5.1 has an explicit test for
  it.
