# Tool-native BabyLLM — 8192-context lane (2026-09-07 → 2026-09-11) — RESULT

Sources: `runs/babyllm-recovery-20260909/{takeover.md, release-failure.md, release-failure-summary.json, release-failure-v2.md, release-failure-v2-summary.json, release-v3-failure-analysis.md, remediation-v3-overview.md, event-9967a2ca-failure-analysis.json, budget-research-20260911-{pods,account}.json, budget-toolcalling-research-20260911.json}`; `docs/ai/plans/2026-09-07-phase51-p5-effective-context-copy-decoding.md` ("8k lane — first measurements").

## Summary

Between 2026-09-07 and 2026-09-11 the `small_8k` rung (33.4M params, RoPE-extended to 8192 context) was carried through loop v5 and an OAuth-interrupted v7, a full recovery (v7-recovery), a quota-interrupted v8, a corpus-bug-fixed v8-quota-tail, and a four-window `release_recovery` remediation (v9) — three independent release-gate attempts (v1 at checkpoint 219709, v2 at 309893, v3 at 493261), each evaluating all eight knowledge sentinel families and all 53 tool-contract families (24 samples/family) against the unchanged 85%-pass / 1%-catastrophic per-family bar. **Final verdict: NOT certified.** Knowledge sentinels are solved (800/800, zero catastrophic, all three gates) but tools plateaued at 1257/1272 with three still-blocking families at the final checkpoint: `memory_write` (nested-fact/field copying), `reactor_manage` (malformed tool-call envelope on at least one non-English case), and `structured_task_create` (create/update discrimination). The four-window remediation cap is now exhausted; both pods are stopped (not deleted) and no model was installed or promoted.

## Lane configuration

| Setting | Value | Source |
|---|---|---|
| Model | `small_8k`, 33,420,096 params (33.4M) | plan §"8k lane — first measurements"; `budget-toolcalling-research-20260911.json` (`parameter_count`) |
| Context | 8192 tokens, RoPE scale 2.0 (extended from a 4096 foundation) | `remediation-v3-overview.md`; `takeover.md` |
| Learning rate / batch | 4e-6, batch 1 | `remediation-v3-overview.md` |
| Sentinel families | 8 knowledge families × 100 samples (800 total); 53 tool-contract families × 24 samples (1272 total) | `release-failure-summary.json`, `release-failure-v2-summary.json`, `event-9967a2ca-failure-analysis.json` |
| Release thresholds | ≥85% pass rate per family, ≤1% catastrophic rate per family (unchanged across v1–v3) | `release-failure-summary.json` (`minimum_pass_rate_per_family`, `maximum_catastrophic_rate_per_family`) |

## Timeline

| Stage | Date | Start checkpoint | Windows | End checkpoint | Knowledge | Tools | Catastrophic | Outcome |
|---|---|---:|---|---:|---|---|---:|---|
| Loop v5 | 2026-09-08 | 87,683 | ordinary continuation, decoder v12 | — | — | — | — | Continuation beyond this point is not in these sources; next evidence is v7 window 2 (plan; `takeover.md`) |
| Loop v7 window 2 (pre-recovery) | 2026-09-09 15:28 UTC | 175,367 (parent, decoder v13) | 1 steered segment attempted, 0 updates applied | 175,367 (unchanged) | 799/800 | 1227/1272 | 10 | Trainer rejected the steered parent (RoPE scale 2.0 vs fresh preset 1.0); simultaneous OAuth loss stalled the operator (`takeover.md`) |
| v7-recovery | 2026-09-09 | 175,367 | 1 (completes the pending window, steps 175,368–219,709) | 219,709 | 799/800 | 1244/1272 | 7 | Release gate v1 **FAILED** (`release-failure-summary.json`) |
| v8 (window 1) | 2026-09-10 | 219,709 | 1 (steps 219,710–264,051, 44,342 updates; ENOSPC quota interruption at step 231,999, recovered in place) | 264,051 | 800/800 | 1243/1272 | 3 | Frontier diagnostic only, not a release gate (`takeover.md`, event `402ef900`) |
| v8-quota-tail (steering-recovery) | 2026-09-10 | 264,051 | 1 (steps 264,052–309,893, 45,842 updates; corpus-preservation bug fixed pre-training, 9e75797c) | 309,893 | 800/800 | 1254/1272 | 3 | Release gate v2 **FAILED** (`release-failure-v2-summary.json`) |
| v9 remediation | 2026-09-10 → 2026-09-11 | 309,893 | 4 (steps 309,894–493,261; steer reactor_manage ×2, seshat_wish ×2) | 493,261 | 800/800 | 1257/1272 | 2 | Release gate v3 **FAILED**, four-window cap exhausted (`event-9967a2ca-failure-analysis.json`) |

## Release gates v1 / v2 / v3

| Gate | Checkpoint | Knowledge | Tools | Catastrophic | Blocking families (pass/24, cat) |
|---|---:|---|---|---:|---|
| v1 | 219,709 | 799/800, 0 cat | 1244/1272 | 7 | `seshat_embed_now` 23/24 (1cat), `memory_write` 19/24 (2cat), `reactor_manage` 20/24 (2cat), `study_note_append` 20/24 (1cat), `study_note_revise` 12/24 (1cat) (`release-failure-summary.json`) |
| v2 | 309,893 | 800/800, 0 cat | 1254/1272 (15 precursor) | 3 | `code_ingest` 23/24 (1cat), `reactor_manage` 21/24 (2cat), `memory_write` 20/24 (0cat, 83.3%), `study_note_revise` 19/24 (0cat, 79.2%) (`release-failure-v2-summary.json`) |
| v3 (final) | 493,261 | 800/800, 0 cat | 1257/1272 (13 precursor) | 2 | `memory_write` 19/24 (0cat), `reactor_manage` 23/24 (1cat), `structured_task_create` 23/24 (1cat) (`event-9967a2ca-failure-analysis.json`) |

Certifiers exit on the first failing family (`seshat_embed_now`, then `code_ingest`, then `memory_write`); each `-summary`/`-failure-analysis` JSON carries the complete per-family report used above, not just the first error.

## Best measured candidates

Checkpoints **355,735** and **401,577** (both interior windows of v9) each measured **1259/1272 tools, 1 catastrophic, knowledge 800/800**, with only two blocking families (`memory_write`, `reactor_manage`) — the best tool scores observed anywhere in this lane (`event-9967a2ca-failure-analysis.json` history; `takeover.md` events `1a6b430a`, `3151e6f4`). **Neither was run through an independent release gate** (v1/v2/v3 evaluated 219709/309893/493261 only) — they are mid-loop sentinel measurements, not certified or fallback-ready candidates. Model and Adam bytes for both remain retained on the volume.

## Infrastructure incidents

| Incident | Detail | Fix / resolution |
|---|---|---|
| Steered-segment RoPE-scale rejection | Trainer rejected a steered parent because the ordinary-resume path adopted the extended RoPE scale (2.0) but the steering path still checked against the fresh preset's scale (1.0) | Commit `0c5ffd21a602e6ab49122935744ae510e3de9e9f` applies shape adoption to both paths (`takeover.md`) |
| Mac ENOSPC (×2) | 2026-09-09 ~19:39 UTC monitor logs showed ENOSPC; 2026-09-10 desktop free space fell to 4.02 GiB (briefly ~3 GiB) during concurrent Rust builds | Reclaimed ~9 GiB then ~17.54 GiB net from verified-inactive Cargo build caches only; no source/binaries/models touched (`takeover.md`, `event-7715d04e-*`) |
| Runpod volume quota | 2026-09-10 09:00 UTC, errno 122 disk quota exceeded at step 231,999 mid-checkpoint; checkpoint directory never created | Network volume `4wgri7h01a` expanded 150→300 GB; write/fsync/read probe re-verified before resume (`takeover.md`) |
| Pod host had no free GPU | Original pod `4ypc4li9y537f2` could not restart after the v2 gate failure | Replacement pod `8ibnh0fwd0ezs4` (A100 SXM 80GB, $1.59/h, US-WA-1) started on the same retained volume (`remediation-v3-overview.md`) |
| Cross-host waiter defect | Waiter inspected only the (not-yet-created) OCR status file when text certification failed, risking a false wait state | Fixed by `902987a46b0be8fd7176db3274797e2d23d2b24c`: checks the pinned text-release status first (`release-failure-v2.md`) |
| Corpus-preservation bug | Ordinary resume moved the prior 500 `study_note_append` siblings into `supplemental_jsonl`, misclassifying all 44,342 records as base for the next segment | Fixed by `9e75797cd728c3e20f16a14635ed52a8b8be48c3` before any update ran; verified 13 adaptive-loop tests on Mac and pod (`takeover.md`) |

## Cost

| Pod | Day | Cost (USD) | Disk billed (GB) | Time billed (h) |
|---|---|---:|---:|---:|
| `4ypc4li9y537f2` | 2026-09-09 | $9.38 | 3,500 | 5.9 |
| `4ypc4li9y537f2` | 2026-09-10 | $17.57 | 6,650 | 11.0 |
| `8ibnh0fwd0ezs4` | 2026-09-10 | $9.59 | 3,650 | 6.0 |
| `8ibnh0fwd0ezs4` | 2026-09-11 | $20.54 | 7,750 | 12.9 |

(`budget-research-20260911-pods.json`)

- **Total pod charges since 2026-09-09: $57.08** (`budget-toolcalling-research-20260911.json`, `pod_charges_since_20260909_usd`); network-volume charges in the same query window: $1.34 (`network_volume_charges_in_query_usd`).
- **Retained-pod rate** (if `8ibnh0fwd0ezs4` were resumed): $1.59/h (`remediation-v3-overview.md`).
- **Account balance at last check** (2026-09-11T15:14:55 UTC): $78.23, baseline drift $0.029/h, spend-limit setting $80 (API cap, not a project budget) (`budget-research-20260911-account.json`).

## Retained artifacts

- Both pods **EXITED, not deleted**: `4ypc4li9y537f2` (stopped 2026-09-09T23:06:50Z after gate v1) and `8ibnh0fwd0ezs4` (stopped 2026-09-11T12:53 UTC after gate v3) (`release-failure-summary.json`, `event-9967a2ca-failure-analysis.json`).
- Network volume `4wgri7h01a` (300 GB after resize) retained and attached to both pod records; all checkpoints remain on it.
- Checkpoint paths of record: `.../adaptive-loop-v7-recovery-20260909/segments/segment-0002/checkpoints/step_00219709`; `.../adaptive-loop-v8-recovery-20260910/.../step_00264051`; `.../adaptive-loop-v8-steering-recovery-20260910/.../step_00309893`; `.../adaptive-loop-v9-release-recovery-20260910/segments/segment-0001..0004/checkpoints/step_00355735`, `step_00401577`, `step_00447419`, `step_00493261` (`takeover.md`).
- Local evidence directory: `/Users/toasty/ws/akh-medu/runs/babyllm-recovery-20260909/` (chain scripts, monitor/bridge state, `evidence*/` archives, all JSON receipts cited above).

## Open hypotheses for the next remediation

From `release-v3-failure-analysis.md` and `event-9967a2ca-failure-analysis.json`:

1. **`memory_write`** — diagnose faithful copying of nested facts, optional-field presence and literal values with fresh, separated examples; never train on held-out sentinel prompts/answers.
2. **`reactor_manage`** — diagnose the tool-call envelope failure (malformed free-text output, e.g. the French `Derid-revision-381` case) with new multi-language examples and runtime-aligned decoder traces; any decoder change must preserve ordinary text answers.
3. **`structured_task_create`** — diagnose create-vs-update tool confusion with new contrastive cases that offer both tools together.
4. Use checkpoints 355,735 or 401,577 as explicit comparison baselines rather than blindly continuing from 493,261; evaluate any smaller candidate change against fresh diagnostics and the unchanged full release gates, and keep a better predecessor on regression.
5. **Undeployed decoder fix** `659101db17a613e2768091f7cca79cd6fef78fb3` ("bind tool names to satisfied required arguments") closes a grammar gap where a tool name remained selectable even when its required fields weren't satisfied (reproduced the `structured_task_update`-without-`id` failure mode). Validated locally — 290/291 grug-data tests, diff check passed — but **not yet GPU-evaluated, not deployed to the pod, and not model-certified**; source now differs from the remote-pinned `242abd84` (`budget-toolcalling-research-20260911.json`).
