# Phase 51 P5 — Effective context and copy-faithful decoding for the tool-native BabyLLM

- **Status:** Closed — 8k lane NOT certified (2026-09-13); successor P5.6 `01a09a61-f4d2-7db3-bc2c-0f3ff513191e`. Results: [tool-native-8k-lane-20260911-RESULT](../../training/results/tool-native-8k-lane-20260911-RESULT.md); governance: [ADR 105](../decisions/105-autonomous-run-governance.md), [post-mortem](../2026-09-13-phase51-codex-session-postmortem.md)
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
| `01a07a7e-6333-7ed2-b251-fb645d3aa3b3` P5.1 | D1 + D6: constrained decoder in `grug-data`/`grug-train`, wired into `KnowledgeModel`, `GrugChatBackend`, scorer, probe, sentinels, all-tool matrix | landed (`ce31da65`, merged `eaf090c3`); `multimodal_runtime.rs` vision decoder still plain |
| `01a07a7e-7ac6-7d63-9819-58f19f79cc22` P5.2 | D2: `PreTokenization` in `bpe.rs`, `train_foundation` flags, manifest fields | landed (`9abeb4fc` + flags in `1fc01b74`) |
| `01a07a7e-9836-7e71-b914-fc1f1552a52c` P5.3 | D3: RoPE, chunked attention, checkpointing knob, 8k presets, shape/manifest compat | landed (`1fc01b74`); old-checkpoint load verified bitwise on CPU across commits; pod memory smoke still owed |
| `01a07a7e-b81d-7be3-9af4-6a852b0ba3a0` P5.4 | D5: loop LR flags, `knowledge.length_generalization` family | done (`7e11ddd4`, `37ece20d`) |
| `01a07a7e-de30-7963-92ee-7f164bb214b0` P5.5 | the 8k lane run + release gate + Phase 50 tiny-LLM entry | PARKED 2026-09-13 (terminal, ADR 105 rule 2): loops v5→v9 ran 2026-09-07→09-11 on pods `4ypc4li9y537f2`/`8ibnh0fwd0ezs4`; release gates v1/v2/v3 FAILED at 219,709 / 309,893 / 493,261 — final knowledge 800/800, tools 1257/1272, blockers `memory_write` 19/24, `reactor_manage` 1 cat, `structured_task_create` 1 cat. Both pods EXITED, volume `4wgri7h01a` retained. See RESULT doc. |
| `01a09a61-f4d2-7db3-bc2c-0f3ff513191e` P5.6 | remediation v4: fresh diagnosis of the three blockers, deploy decoder fix `659101db`, compare from 401,577 vs 493,261, one bounded loop stage → gate v4 | DRAFT 2026-09-13; needs operator budget + pod-restart approval; babysat per ADR 105 |

## Sequence

1. **Land P5.1–P5.4 on `phase51-p4-run`** (or a `phase51-p5` branch) as
   separate commits; all CPU tests green; `cargo clippy` clean. Every
   pipeline binary is rebuilt from that one commit on the pod (ADR 102
   schema-skew rule; `bootstrap-head.sh`).
2. **Re-probe the 4096 checkpoint with D1** (pod restart needs approval;
   ≈1–2 h). Command shape:
   `run_tool_native_adaptive_probe.py --bundle …/knowledge-model-small-v1 --max-new-tokens 640 --probe-dir adaptive-probe-constrained-v1 --knowledge-families <the seven v2 families>`
   (the eighth family's 8k/12k buckets cannot fit a 4096 checkpoint; the
   generator is now `akh-knowledge-workflow-v3`, so exams are fresh either way).
   Outcome A: both hex-copy families join their siblings (≥0.78) → the
   weights already have the workflow; proceed to 3 for the window and
   tokenizer, and expect certification on the 8k lane. Outcome B: a family
   stays red with zero schema violations → residual is workflow competence;
   read the per-step closed-loop diffs before deciding between more
   curriculum difficulty and the `350m_8k` rung. Write the numbers into a
   results doc that names the decoder.
3. **Foundation** — amended: `--model small_rope_4k`, batch 4 × 50k steps at 4096 (8192 does not fit; see ADR 104 amendment), hex-aware tokenizer; SFT extends to 8192 with `--extend-context`. First a 200-step memory/throughput smoke with and
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
- Burn's `Option<T>::load_record` silently drops a `Learned` checkpoint's
  embedding when the receiving model was built as `Rope` (no error). Matching
  configs load bitwise-exact (verified 2026-09-07 across `9abeb4fc`→HEAD).
  Follow-up WI filed to fail closed on encoding mismatch in `checkpoint.rs`.

## Paired decoder measurements (2026-09-07/08, 4096 checkpoint step 35207, window-2 sentinels, n=100 per family)

| decoder (commit) | acquire | recall | revise | fallback | abstain | inventory | navigate | all-tool pass / catastrophic |
|---|---|---|---|---|---|---|---|---|
| plain greedy (window 2) | 43 | 81 | 12 | 78 | 100 | 89 | 67 | 790/1272 / 435 |
| grammar + pattern copy (`ce31da65`) | 43 | 81 | 76 | 78 | 100 | 89 | 70 | 1006 / 179 |
| + array/object grammar (`eae9e849`) | 43 | 81 | 76 | 78 | 100 | 89 | 70 | 1027 / 130 |
| + copy bias v1 (`939df6eb`) — regression | 0 | 81 | 0 | – | – | – | – | – |
| + copy bias fixed, key dedup (`957589e1`) | 47 | 81 | 80 | 81 | 100 | 89 | 70 | 1041 / 80 |
| + identifier-suffix lock (`2f55af30`) | 47 | – | 80 | 81 | – | – | – | 1042 / 80 |
| + deadlock-only release, left-anchored runs (`a72b2837`) — regression | 54 | – | 26 | – | – | – | – | – |
| + run completion by run-byte children, strict escapes (`9b03b856`) | 54 | – | 80 | – | – | – | – | – |
| + lock from the first run byte after ':' or '/' (`20a33eaa`) — **lane decoder** | 69 | – | 80 | – | – | – | – | – |

Residual after `957589e1` (addressed by `a72b2837`/`9b03b856`, now 0 lock releases): identifiers embedded in composed values
(`synthetic://textbook:<id>:edition-2`, `MY-159`→`MY-15`) where neither the
whole-value bias nor the identifier lock engages → token-level lock on the
identifier-like suffix of a value (next). The all-tool note tools
(`study_note_revise` 0/24, `study_note_append` 3/24) fail on the synthetic
exam's exact-value reproduction and truncation, not on grammar; they are
re-measured on the 8k lane's model. Constrained evaluation is ~15× slower
than grammar-only (≈1 h per family) — performance pass in progress; it must
land before the loop stage.

## 8k lane — first measurements (2026-09-08)

- Foundation `small_rope_4k`: 49,999 steps, eval perplexity **40.1** (learned-position 4k run: 60.9), 33.4M params.
- SFT `small_8k --extend-context` (RoPE scale 2.0): p95 prompt 5,473 tokens, `compacted=0`, candidate step 43,841.
- Probe (n=20, decoder v9): abstention 20, navigation 20, recall 19, inventory 19, revise 18, length_generalization 7, acquire_textbook 0, provider_fallback 0; all-tool 517/636 (cat 36).
- Loop v1 window 1 (n=100, v9): abstention 100, recall 96, navigation 95, inventory 95, revise 90 (cat 9), length_generalization 34 (cat 66), acquire 0, fallback 0; all-tool 1027/1272 (cat 82). Ordinary training then failed on the resume shape check (`c4911cf9` fixes it); the loop refuses `--resume` after a binary change by design, so loop v2 restarts from the probe with decoder v10 (`1814e564`).
- Both zeros share one symptom: the model answers in free text at `distill_note`. Hypothesis: `length_generalization` padding steps make up 31% of SFT records → `--family-weight` (`5c0f6f39`) for the next curricula; decided after loop v2 window 1.
- **Loop v3 (decoder v10, ordinary windows at LR 2e-5 via `--allow-schedule-change`, `a05d5c3a`)**: window 1 (step 43,841) as above; window 2 (step 87,683): acquire_textbook 89 (cat 11), grounded_abstention 100, note_navigation 100, provider_fallback 100, recall 100, revise 100, textbook_inventory 99, length_generalization 34 (cat 66); all-tool 1209/1272 (cat 14). One reduced-LR epoch fixed both zero families with no regression (D5 confirmed). Controller stopped (stop: acquire catastrophic 11%, length_generalization, `reactor_manage`; steer: `memory_write`, `study_note_append`, `study_note_revise`). Root causes of the two knowledge stops are evaluator/decoder, not weights: (a) all 11 acquire failures emit `: edition 2` for `:edition-2` — the left anchor arms only after a run byte, so a space after ':' is never constrained (fix: arm at the delimiter); (b) long length_generalization episodes stop after step 4 because the closed-loop generator requires byte-exact equality with the expected action while the grader accepts semantic matches (`query_symbols` filler differs) — steps 5+ become `missing_prediction` (fix: one shared predicate). Loop v4 restarts from the window-2 candidate with both fixes.
- **Loop v4 (2026-09-09 01:31Z)**: same numbers as v3 window 2 and an immediate stop, because a loop started from a window's controller step inherits that window's stop verdicts. Two further root causes measured and fixed: length_generalization padding calls carry undecidable filler arguments (now `padding: true`, graded on tool + schema + grounding only, `14a393f7`); acquire composes `:edition-2` from a prompt that only says "edition 2" (URI-shaped values now forbid whitespace in the grammar, `6675db56`). **Loop v5** starts from candidate 87,683 with the probe's controller state (a second reduced-LR epoch is possible) and decoder v12.
- **Loop v5 (2026-09-08, decoder v12)**: restarts from candidate 87,683 with the probe's controller state (a second reduced-LR epoch is possible). Continuation beyond this point is not in the recovered operational record; the next dated evidence picks up mid-training at v7 window 2, checkpoint 175,367 (`runs/babyllm-recovery-20260909/takeover.md`).
- **Loop v7 window 2 (2026-09-09 15:28 UTC, decoder v13, pre-recovery)**: stops attempting a steered segment — the trainer rejects the parent because its already-extended RoPE scale (2.0) doesn't match the fresh `small_8k` preset's scale (1.0); the ordinary resume path had already adopted the verified shape, the steering path hadn't. Zero updates applied to the failed segment. Pre-recovery measurement at 175,367: knowledge 799/800 (0 cat), all-tool 1227/1272 (10 cat); `study_note_append` 12/24, `study_note_revise` 7/24. Fixed by `0c5ffd21a602e6ab49122935744ae510e3de9e9f` (adopts the verified shape on both resume and steering paths).
- **Loop v7-recovery (2026-09-09)**: completes only the pending second window, steps 175,368–219,709, from the exact preserved parent, executables and Adam state (LR 4e-6, batch 1, 43,842 base records + 500 preserved siblings). End checkpoint 219,709.
- **Release gate v1 (219,709)**: **FAILED**. Knowledge 799/800 (0 cat); tools 1244/1272 (7 cat). Five blocking families: `seshat_embed_now` 23/24 (1cat), `memory_write` 19/24 (2cat), `reactor_manage` 20/24 (2cat), `study_note_append` 20/24 (1cat), `study_note_revise` 12/24 (1cat).
- **Loop v8 window 1 (2026-09-10)**: fresh frontier diagnostic selects `study_note_revise`; ordinary continuation, steps 219,710–264,051 (44,342 updates). Interrupted by a Runpod volume-quota ENOSPC at step 231,999 (errno 122); volume expanded 150→300 GB and the window resumed from the last durable checkpoint with zero lost or forged updates. End checkpoint 264,051: knowledge 800/800 (0 cat), tools 1243/1272 (3 cat).
- **Loop v8-quota-tail / steering-recovery (2026-09-10)**: a corpus-preservation bug is caught before this window trains — ordinary resume had misclassified a prior sibling batch as permanent base records — fixed (`9e75797cd728c3e20f16a14635ed52a8b8be48c3`) and restarted with the correct 43,842 base + 2,000 `study_note_revise` siblings, steps 264,052–309,893 (45,842 updates). End checkpoint 309,893.
- **Release gate v2 (309,893)**: **FAILED**. Knowledge 800/800 (0 cat); tools 1254/1272 (15 precursor, 3 cat). Four blocking families: `code_ingest` 23/24 (1cat), `reactor_manage` 21/24 (2cat), `memory_write` 20/24 (0cat, 83.3%), `study_note_revise` 19/24 (0cat, 79.2%).
- **Loop v9 remediation (2026-09-10 → 2026-09-11, opt-in `release_recovery` policy, 4-window cap)**: prioritizes catastrophic-blocking families first. Steers `reactor_manage` twice (→355,735, →401,577 — same-checkpoint sentinel still 1 cat despite a clean 24/24 fresh frontier each time) then `mcp__seshat__seshat_wish` twice (→447,419, regressing to 5 blockers; →493,261). Best intermediate candidates 355,735 and 401,577 both measure 1259/1272 tools (1 cat, two blockers: `memory_write`/`reactor_manage`) — never independently release-gated, so not certified.
- **Release gate v3 (493,261, final)**: **FAILED**. Knowledge 800/800 (0 cat); tools 1257/1272 (13 precursor, 2 cat). Three blocking families: `memory_write` 19/24 (0cat), `reactor_manage` 23/24 (1cat), `structured_task_create` 23/24 (1cat). Four-window remediation cap exhausted; pod `8ibnh0fwd0ezs4` stopped 2026-09-11T12:53 UTC, volume `4wgri7h01a` retained.
- **Status 2026-09-13**: run stopped, four-window remediation cap exhausted, model **not certified**. Both pods (`4ypc4li9y537f2`, `8ibnh0fwd0ezs4`) EXITED/retained with volume `4wgri7h01a` intact; no GPU running, no pod/volume/checkpoint deleted. Next step is a fresh diagnosis of the three residual failure modes (nested-value copying in `memory_write`, tool-call-envelope emission in `reactor_manage`, create/update discrimination in `structured_task_create`) and a remediation-v4 go/no-go decision — including whether to deploy the undeployed grammar fix `659101db17a613e2768091f7cca79cd6fef78fb3` ("bind tool names to satisfied required arguments"; 290/291 grug-data tests locally, not yet GPU-evaluated or model-certified, source diverged from remote-pinned `242abd84`).
