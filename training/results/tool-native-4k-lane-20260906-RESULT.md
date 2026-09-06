# Tool-native BabyLLM — 4096-context lane (2026-09-05 → 2026-09-06) — RESULT

- **Verdict: no certified model; the adaptive controller stopped at window 2
  with clean evidence.** The `small` rung at 4096 context executes five of
  seven knowledge workflows well (0.78–1.00 sentinel pass at n=100) and
  fails two on a single, narrow capability: verbatim copying of long hex
  identifiers with doubled characters. Everything downstream of the loop
  (text release gate, native OCR, cross-host gate, install, publication)
  never ran and is unchanged.
- **Hardware:** the existing pod `4ypc4li9y537f2` (A100 SXM 80GB, US-WA-1,
  $1.59/h, volume `4wgri7h01a` 150GB). Resumed 2026-09-05 11:04Z, stopped
  (never deleted) 2026-09-06 21:47Z: ~34.7h ≈ $55. All artifacts remain on
  the volume under `/workspace/babyllm/tool-native-4k/`.
- **Code state:** `phase51-p4-run` `1aa6d85a`…`008343cc` (this lane's fixes:
  `94b2da89`, `e71983d4`, `ae50b963`, `8aa5dd05`, `1c30e8c9`, `ea750958`,
  `008343cc`). Companion to
  `tool-native-full-model-pipeline-20260904-RESULT.md` (the pipeline) and
  `p350m-RESULT.md` (the deduction organ).

## 1. What the 2026-09-04 run actually died of

The 2026-09-04 pipeline stalled not on the Mac (the crash-looping launchd
supervisors were a symptom) but remotely, in the adaptive loop's first
warm-start: `train_knowledge` refused the SFT bundle's `run.json`
(`missing field eval_every`) — a schema skew between the binary that wrote the
bundle and the newer binary reading it. Every downstream gate blocked behind
it. Recovering exposed four further, independent problems, each of which
would have blocked certification on its own:

| # | problem | evidence | fix |
|---|---|---|---|
| 1 | v1 (50-tool, five-family) corpus lane is unrecoverable | HEAD `knowledge_workflow` requires `study_note_revise` (v2 note-revision semantics); probe/loop/gate all *generate* fresh exams, so exams can only match a model trained on a v2 corpus; the v1 generator existed only as uncommitted working-tree state | regenerate v2 curricula (53 tools, seven families) deterministically on the pod: knowledge 32,188/6,440, dojo 476/42, tools 2,544/636; contract export reproduces `blake3:3d6e18bb…` byte-for-byte |
| 2 | 2048 context corrupts the long-workflow supervision | v2 corpus p95=2,315 / max=3,374 tokens; trainer compacted 4,701 records (13%), dropping the retrieval evidence the targets must quote; loss 4e-4 yet 100/100 catastrophic on intact closed-loop sentinels for exactly the long families | `akh_tool_small` context 2048→4096 (`1c30e8c9`; the 350m preset already carried 4096 for this stated reason); foundation retrained at batch 4 × 50k steps = identical 819.2M-token budget; validation then reports `compacted=0` |
| 3 | 128-token generation budget truncates the note-writing targets | `ingest_textbook` ~790 chars, `revise_note` ~660 (with a 64-hex revision token), `distill_note` ~525; at 128 tokens the output is not even a well-formed call (`wrong_action`, 100/100); the same weak checkpoint scores 0/100 at 128 and 10/100 at 640 | probe gains `--max-new-tokens` (`008343cc`); loop/gate already had it; lane runs at 640 |
| 4 | 4096 context at batch 8/4 saturates the A100 | batch 8: CubeCL "can't allocate buffer 5310889984" (5.3GB attention scores per layer); batch 4 SFT: 81,003 MiB, CubeCL "Memory page 0 doesn't exist", 63 corrupted (SKIPPED) updates by step 1.6k with grad norms ~2e10 | foundation batch 4 (74GB, zero skips); SFT and OCR batch 2 (41GB, zero skips); supervisor gains `--batch-size` (`ea750958`) |

Operational failures met on the way, all now recorded in the ops memory and
fixed in code where code was at fault: the 150GB network volume hit its quota
mid-checkpoint (looked like a mass kill; `du` first); the post-foundation
supervisor died on a torn `train.jsonl` tail (`8aa5dd05`); the foundation
trainer writes no pid file (the launcher must); `pkill -f` bracket patterns
still self-match when the plain string appears elsewhere on the same ssh
command line; Mac supervisors die at agent-session teardown unless launched
through a `setsid` shim.

## 2. Definitive numbers

Foundation `small` @ 4096 (35,517,248 params, vocab 8,000, tokenizer
`blake3:8ae8451e…` — byte-identical to the 2048 run): step 49,999, 819.2M
tokens, final loss 3.209, eval perplexity 60.86 (2048 run: 60.10). Model hash
`blake3:964b1e29…`. One SFT epoch on the v2 curricula at batch 2: 17,603
steps, zero skipped updates.

Closed-loop knowledge sentinels (n=100 per family, 640-token budget):

| family | probe (n=20) | window 1 (1 epoch) | window 2 (2 epochs) | verdict |
|---|---|---|---|---|
| grounded_abstention | 20/20 | 1.00 | 1.00 | green |
| textbook_inventory | 18/20 | 0.88 | 0.89 | green |
| recall_after_restart | 19/20 | 0.93 | 0.81 | green |
| provider_fallback | 17/20 | 0.83 | 0.78 | stop |
| note_navigation | 17/20 | 0.82 | 0.67 | stop |
| acquire_textbook | 7/20 | 0.39 | 0.43 | stop |
| revise_from_new_edition | 1/20 | 0.18 | 0.12 | stop |

All-tool contract sentinel (53 tools × 12): mean pass 0.51 (probe) → 0.53
(window 1) → 0.62 (window 2); 31 tools improved, 17 regressed between the
windows. Window-2 controller: 24 green / 4 watch / 33 stop (four knowledge
families plus 29 tool families with confirmed catastrophic rates). The
frontier chosen after window 1 was `tool_contract.similarity_search` at
difficulty 3; no sibling steering was triggered (`sibling_records: 0`).

Second-epoch effect: the two hex-copy families did not move (0.39→0.43,
0.18→0.12) while three previously healthy families regressed (recall
0.93→0.81, note_navigation 0.82→0.67, provider_fallback 0.83→0.78). This is
the P4 lesson again — the ordinary epoch trains well past the SFT floor —
and argues for lower-LR or shorter ordinary windows before any further loop
runs.

## 3. The residual failure is one capability

Every remaining catastrophic step in the two weak families is a
single-character drop while copying a hex identifier that contains a
doubled character, which then fails the schema `pattern`:

```
expected_revision  got  bb18784f…95353f41893c79279c6a4b        (63 hex)
                   want bb18784f…95353f41893c7927279c6a4b       (64 hex)
textbook id        got  48d282bd5d3d160    want 48d282bd5dd3d160
```

Tool choice, ordering, grounding and the evidence rules are otherwise
correct in those episodes. The BPE (trained on natural-language FineWeb)
tokenizes runs of repeated hex digits inconsistently, and greedy decoding
collapses them. The same signature dominates the weakest tool sentinels
(`seshat_ingest` 0/12, `study_note_revise` 0/12, `memory_write` 1/12,
`study_note_append` 1/12). Production revision tokens are real BLAKE3 hashes
with doubled characters, so this cannot be papered over in the generator.

## 4. Decisions this leaves open (not taken here)

1. **Engine-side copy constraint.** For schema fields with a `pattern`
   (and, more generally, for arguments that must equal a string present in
   the context), constrain decoding to copy from context — the
   grammar-enforcement stance of ADR 067/099 applied to the Grug runtime
   (`knowledge_runtime.rs`) and the evaluators alike. Cheapest fix; makes
   the sentinels measure workflow competence rather than BPE luck.
2. **Bigger rung.** `--model 350m` (already 4096 context) copies better but
   may retain the tokenizer failure mode; ~3–5× the compute.
3. **Ordinary-window schedule.** Lower LR / fewer steps per window, or stop
   at the SFT floor; the second epoch regressed three healthy families.
4. **Serving budget.** `AKH_GRUG_MAX_NEW_TOKENS` defaults to 384; note
   writes need ≥640 once a model is installed.

## 5. Artifacts (volume `4wgri7h01a`, base `/workspace/babyllm/tool-native-4k/`)

- `foundation-model-small-v1/` — 4096 foundation, `foundation-model.json`
  bound to `checkpoints/step_00049999` (intermediate checkpoints pruned).
- `knowledge-model-small-v1/` — SFT bundle (candidate step 17,603);
  `.failed-batch8-20260906/`, `.failed-batch4-20260906/` — the two OOM
  attempts, kept as evidence.
- `adaptive-probe-sft-epoch-1-v1/` — probe evidence, predictions, controller.
- `adaptive-loop-v1/` — two windows with per-step closed-loop scores,
  sentinels, controller states; terminal `stopped_by_controller`.
- `/workspace/babyllm/tool-native-v1/local-curricula-v2-20260905/` — the v2
  curricula and 53-tool contract; `/root/exp/` (container disk, gone on
  stop) held the 128-vs-640 experiment — numbers are in this document.
- Drivers: `/workspace/babyllm/chain-4k.sh`, `chain-4k-b.sh` (+ logs);
  bootstrap `bootstrap-head.sh`; binaries `target-head/release/examples/`.
