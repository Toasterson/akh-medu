# Phase 51 P6.2 — decoder-fix rescore of the three retained release-recovery checkpoints

2026-09-13. Independent, local, Metal-GPU rescore of decoder fix `659101db`
("fix(training): bind tool names to satisfied required arguments") against
the three release-blocker tool families identified by the 2026-09-11 v3
release gate, on the three checkpoints retained from
`adaptive-loop-v9-release-recovery-20260910`. Anima work item: none created
for this measurement session per the "P6 items are session-driven" rule
(CLAUDE.md); this is a diagnostic pass, not an autonomous run.

## Setup

- Binary built from commit `95368800c21f5b4f550ec59f56e932cd22cd7598` (HEAD
  at build/run time). The worktree's HEAD advanced to `1e53cfdb` (P6.3
  episode generator) *while the long full-gate run was in flight*, from a
  concurrent agent session in the same shared worktree; that commit only
  adds `src/bin/akh_episode_generator.rs` + templates and touches neither
  `grug-data` nor `grug-train`, so it does not affect anything measured here.
  Decoder fix `659101db` (2026-09-11, timestamped *after* the v3 gate's
  12:53 UTC failure) is an ancestor of both.
- `cargo build --release -p grug-train --features metal --example eval_tool_contract_exam`
  built cleanly against `burn::backend::Metal<f32>` (the `metal` feature is a
  straight `cfg` match alongside `gpu`/`cuda` in `eval_tool_contract_exam.rs`
  and `grug-train/src/lib.rs` — no fallback needed).
- Device: Mac mini, Apple M2, 16 GB unified memory (the box named in ADR 106).
- GPU is shared with other P6 agent sessions; every run below took the
  `/tmp/p6-gpu.lock` directory lock immediately before invoking the binary
  and released it immediately after (including once by waiting ~3 minutes
  for another session's concurrent hold before the full 1272-case run could
  start).
- `--max-new-tokens 768`, matching `launch-config.json`'s `max_new_tokens`
  for the v3 gate.

## Bundle reconstruction

The three checkpoints were retrieved from the Runpod volume as bare
`meta.json` + `model.mpk` + `optim.mpk` directories (segment dirs never
carried a local `candidate-model.json`/`tokenizer.json`/`contracts.json` —
those only exist at the top of a bundle like `knowledge-model-small-v1`, and
the adaptive loop apparently assembled bundles in-process on the pod without
persisting them locally). `eval_tool_contract_exam` requires a full bundle
(`KnowledgeModel::load_candidate_gpu` → `verify_knowledge_bundle` in
`crates/grug-train/src/knowledge_runtime.rs`), so
`runs/p6-2-decoder-rescore-20260913/build_bundle.py` reconstructs one per
checkpoint under `runs/p6-2-decoder-rescore-20260913/bundle-<step>/`:

- `tokenizer.json` + `contracts.json` copied unchanged from
  `knowledge-model-small-v1` (the P5 SFT bundle this run started from). This
  is safe because the sentinel exam's `contract_hash` already matches this
  `contracts.json` byte-for-byte (both declare
  `blake3:3d6e18bb74765cab13d35901d0a10a7834077f302cc578941ca06ef56223700a`),
  so the tool contracts did not change across the adaptive loop.
- `model.mpk` + `meta.json` copied from each checkpoint's segment directory.
  `optim.mpk` (256 MB each) is **not** copied — `checkpoint_optimizer_hash` is
  omitted from the manifest, so `verify_knowledge_bundle` never looks for it;
  inference needs no Adam state.
- `candidate-model.json` is `knowledge-model-small-v1`'s own manifest with
  only `checkpoint_dir`/`checkpoint_step`/`checkpoint_model_hash` swapped to
  the target checkpoint's own `meta.json` values (which reproduce the
  pre-verified hashes `138592ba…`/`041d9d85…`/`0d4dfad1…` exactly).
  `checkpoint_optimizer_hash` and `certification` are dropped.
  `KnowledgeModelArtifact::verify()` (`crates/grug-train/src/knowledge_train.rs`)
  only checks that lineage hashes (`training_corpus_hash`, `tokenizer_hash`,
  ...) have a `blake3:` prefix — it never re-verifies them against a file —
  so `training_corpus_hash`/`training_records`/`capability_families`/
  `foundation`/... are inherited from the v1 bundle unchanged: **honest
  lineage metadata is not available for the mid-run checkpoints; what's
  checked (tokenizer, contracts, model weights) is exact.**

None of the originals under `runs/checkpoints/...` were modified.

The quarantined sentinel exam is `tier: "sentinel"`, so
`ToolContractExamSet::verify` skips the checkpoint-step/hash binding
entirely (a frozen sentinel's cases don't move with the checkpoint) and only
checks `contract_hash` — which matches. It does, however, still recompute
and check `exam_fingerprint` against the case list. Filtering the exam down
to the four families of interest therefore required *recomputing* the
fingerprint through the real Rust function rather than hand-editing JSON
(`ToolContractExamSet::compute_fingerprint`, `crates/grug-data/src/tool_curriculum.rs`,
hashes a canonicalized identity object including `cases`); a throwaway
standalone crate, `runs/p6-2-decoder-rescore-20260913/exam-filter/` (its own
`[workspace]` root, path-dependency on `grug-data`, never joins the main
workspace, and is not a modification of any crate file), filters
`all-tools.json` down to `memory_write` + `reactor_manage` +
`structured_task_create` + `structured_task_update` (24 cases each, 96
total), sets `target_tools` to those four names, and recomputes
`exam_fingerprint` for real. Output:
`runs/p6-2-decoder-rescore-20260913/blocker-families-exam.json`.

## Headline result: 659101db fixes exactly one of the three blocker families, deterministically, on all three checkpoints

Fixed decoder (`--decoding constrained`, current binary, includes 659101db),
blocker-families exam, all three checkpoints:

| Checkpoint | memory_write | reactor_manage | structured_task_create | structured_task_update |
|---|---|---|---|---|
| 355735 | 19/24 (5 precursor, 0 cat) | 23/24 (0 precursor, 1 cat) | 24/24 | 24/24 |
| 401577 | 19/24 (5 precursor, 0 cat) | 23/24 (0 precursor, 1 cat) | 24/24 | 24/24 |
| 493261 | 19/24 (5 precursor, 0 cat) | 23/24 (0 precursor, 1 cat) | 24/24 | 24/24 |

Identical across all three checkpoints — the fix's effect (and the
remaining gap) is checkpoint-independent, not an artifact of one particular
training window.

Checkpoint 493261, old (archived, pre-659101db) constrained decoder vs. new
(fixed) constrained decoder, same binary flavor (`constrained`, ADR 104 D1
grammar + value-copy), same `--max-new-tokens 768`:

| Family | Old (`text-release-gate-v3`, 2026-09-11) | New (fixed decoder, this run) | Changed? |
|---|---|---|---|
| memory_write | 19/24, 5 precursor, 0 cat | 19/24, 5 precursor, 0 cat | No — **all 5 failing predictions are byte-identical** to the archived ones |
| reactor_manage | 23/24, 0 precursor, 1 cat | 23/24, 0 precursor, 1 cat | No — **the 1 catastrophic prediction is byte-identical** |
| structured_task_create | 23/24, 0 precursor, 1 cat | **24/24, 0 precursor, 0 cat** | **Yes — the exact blocking case now passes** |
| structured_task_update (sibling, not itself a blocker) | 24/24 | 24/24 | No change |

The one case that flipped is exactly the failure mode `659101db`'s commit
message describes. Old prediction (archived,
`event-9967a2ca-failure-analysis.json`):
`{"arguments":{"recurrence":"recurrence-alpha","title":"title-revision-224"},"name":"structured_task_update"}`
— `structured_task_update` requires `id`, which is absent, so it's a schema
violation (`MissingRequired` at `$.id`) as well as the wrong tool. New
prediction (this run, `runs/p6-2-decoder-rescore-20260913/493261/blocker-families.constrained.predictions.json`):
same arguments, `"name":"structured_task_create"` — correct and schema-valid.
This is exactly what `close_arguments` (the new function replacing
`any_viable_required_satisfied`) does: closing the arguments object now
retains only tools whose *own* required fields are satisfied by what's been
emitted, so once `{"recurrence":...,"title":...}` closes without an `id`,
`structured_task_update` (which requires `id`) is no longer a legal `name`
choice — only `structured_task_create` is.

### `--decoding plain` does not reproduce the v3 gate's numbers — it isn't the right comparator

Per the task brief, `--decoding plain` was also run on 493261 against the
blocker-families exam to sanity-check the harness against archived numbers.
It does **not** match:

| Family | Archived (v3 gate, constrained, pre-fix) | `--decoding plain` (this run) |
|---|---|---|
| memory_write | 19/24, 5 precursor, 0 cat | 17/24, 3 precursor, **4 cat** |
| reactor_manage | 23/24, 0 precursor, 1 cat | 23/24, 0 precursor, 1 cat (matches) |
| structured_task_create | 23/24, 0 precursor, 1 cat | 22/24, 0 precursor, **2 cat** |
| structured_task_update | 24/24 | 24/24 (matches) |

This is expected, not a harness bug: `DecodingMode::Plain`'s own doc comment
(`crates/grug-train/src/knowledge_runtime.rs`) says it "reproduces the
pre-D1 unconstrained greedy decoder exactly ... for comparing against
historical certification numbers only — it is never the deployed runtime's
behavior." The v3 gate ran the *constrained* decoder (ADR 104 D1: runtime
`<tool_call>` grammar + value-copy constraints) — just without the
`659101db` name-binding fix, since that fix landed hours after the gate's
12:53 UTC failure. There is no flag on the current binary that reproduces
"constrained, pre-659101db" — the grammar change was an in-place patch to
`ToolCallGrammar`, not a gated alternate path — so `plain` (a strictly older
and more permissive decoding regime) is the wrong tool for that comparison;
the archived-vs-fixed **constrained** comparison above is the correct one,
and it is exact (byte-identical failing predictions where nothing changed).

## Case-level classification: decoder-side vs. model-side

Using the recovered decode reports (`--decode-reports-out`,
`crates/grug-data/src/constrain.rs` `DecodeReport`) alongside the byte-exact
before/after predictions:

**structured_task_create/update confusion — decoder-side, fixed.** Covered
above. `copy_fields` is empty for this case in both old and new predictions
(no value-copy field applies — `title`/`recurrence` are free-generated
strings, not identifier-shaped copy targets); the only thing that changed
was which `name` values the grammar considered legal at the moment
`arguments` closed. This is `659101db`'s exact, intended scope.

**memory_write (5/24 failing, unchanged) — model-side.** All 5 failing
predictions are byte-identical between the archived pre-fix run and this
fixed-decoder run. The failure mode ("arguments differ from the exact
requested values") is the model dropping/merging requested `facts` array
entries, substituting a different `subject` value than the one asked for,
or adding an unrequested `timestamp` field. None of these fields are
value-copy-constrained fields in the grammar (`copy_fields: []` in every
decode report for these cases — `subject`, `summary`, `tags` etc. are
free-generated strings the model composes, not context-anchored copies), so
there is no decoder mechanism in scope that could correct them;
`identifier_copy_locks` is 2 on two of the five (the decoder's
identifier-shaped-run heuristic fired on substrings that *look* like
identifiers, e.g. `subject-revision-855`) but never released and did not
change the (wrong) value chosen — the model's own logits picked the wrong
content from the start. `659101db` cannot and does not touch this; **P6.4
training work is needed here.**

**reactor_manage (1/24 catastrophic, unchanged) — model-side by the
decoder's own design, not a decoder bug.** The 1 catastrophic
prediction (`Derid-revision-381","operation":...`) is byte-identical
old vs. new. `RuntimeCallState` (`crates/grug-data/src/constrain.rs`) only
engages `ToolCallGrammar` once the model has typed the literal
`<tool_call>\n` tag byte-for-byte; the doc comment is explicit: "any
divergence [from the opening tag] commits to free text for the rest of
generation, which is exactly as permissive as the pre-existing unconstrained
decoder" — a deliberate design choice so a legitimate plain-text `Answer` or
abstention response is never blocked. In this case the model diverged from
the tag at the very first tokens (decode report: `constrained: true` but
`identifier_copy_locks: 0`, `anchored_run_bytes: 0` — none of the D1
mechanisms engaged at all), so the rest of the generation ran fully
unconstrained and produced the garbled, envelope-less text observed. Fixing
this would mean changing the *opening-tag* commitment logic (a different,
unexplored mechanism from `close_arguments`), not something in
`659101db`'s scope, or — more likely given the fix is genuinely working as
designed — training the model to reliably open the tag when tools are
offered. **Also P6.4 territory, and a distinct hypothesis from memory_write's
failure (wrong-content-after-correct-tag vs. never-opening-the-tag) worth
keeping separate in any remediation plan.**

## Full-gate number for 401577 (fixed decoder)

Ran the complete 1272-case exam (all 53 tool families, unfiltered
`all-tools.json`) against checkpoint 401577 with the fixed decoder:
`--max-new-tokens 768 --decoding constrained`, 43m46s wall clock (2075s user
+ 342s system CPU, 92% average utilization) on the M2 Metal backend — about
2.06s/case, consistent with the 96-case blocker-family runs (3m39s–4m58s,
2.3–3.1s/case including the smaller run's fixed startup overhead).

```
tools=53 samples=1272 pass=0.990 catastrophic=0.001
evidence=runs/p6-2-decoder-rescore-20260913/401577/full-1272.constrained.evidence.json
```

Aggregate: **1259/1272 pass, 12 precursor, 1 catastrophic.** Only two
families miss a gate (pass < 85% or catastrophic > 1%):
`tool_contract.memory_write` (19/24, 5 precursor, 0 catastrophic) and
`tool_contract.reactor_manage` (23/24, 0 precursor, 1 catastrophic) —
exactly the two families the blocker-family run already showed the fix does
not touch. `structured_task_create` and `structured_task_update` both pass
24/24.

This aggregate (1259/1272, 12 precursor, 1 catastrophic) is **exactly
identical** to 401577's archived pre-fix totals in
`event-9967a2ca-failure-analysis.json`'s history
(`{"pass_count": 1259, "precursor_count": 12, "catastrophic_count": 1}`,
`blockers: ["tool_contract.memory_write", "tool_contract.reactor_manage"]`).
That's an independent, whole-gate confirmation of the checkpoint-level
picture above: 401577 never exhibited the `structured_task_create`/
`structured_task_update` confusion in the first place (that pattern only
starts showing up by checkpoint 447419 in the historical record), so on
this checkpoint `659101db` has **zero net effect at the release-gate level**
— it neither breaks nor improves anything, because the one thing it fixes
never manifested here. The fix's value is checkpoint 493261-and-later
specific, though the underlying grammar behavior (and the two remaining
model-side failure modes) is identical across all three checkpoints tested.

## Verdict

Decoder fix `659101db` is real, checkpoint-independent, and precisely
targeted: it resolves the `structured_task_create`/`structured_task_update`
confusion (1 of the 3 blocking families, and the only blocker whose failure
mode was "wrong tool name given a satisfied-elsewhere argument set" — exactly
what the fix's grammar change addresses) on all three retained checkpoints,
with byte-exact reproduction of every other prediction. It does **not**
touch `memory_write`'s 5 value-copy-accuracy failures or `reactor_manage`'s
1 tag-divergence catastrophic failure — both are model-generation problems
(wrong content chosen by the model's own logits, in one case before the
constrained decoder's mechanisms even engage) that no decoder-side change
already in the tree addresses. Net effect on the 493261 release-gate
verdict: **2 of the 3 blockers survive unchanged; the model is closer to but
still not release-certifiable on tool_contract families without further
training (P6.4)** — deploying `659101db` alone would take checkpoint 493261
from "3 blocking families" to "2 blocking families, memory_write and
reactor_manage," not to certification.

## Files produced

- `runs/p6-2-decoder-rescore-20260913/build_bundle.py` — reconstructs a
  minimal candidate bundle per checkpoint (copies only; originals untouched).
- `runs/p6-2-decoder-rescore-20260913/bundle-{355735,401577,493261}/` — the
  three reconstructed bundles actually loaded by the eval binary.
- `runs/p6-2-decoder-rescore-20260913/exam-filter/` — standalone helper crate
  (own workspace root) that filters the sentinel exam and recomputes its
  fingerprint through the real `ToolContractExamSet::compute_fingerprint`.
- `runs/p6-2-decoder-rescore-20260913/blocker-families-exam.json` — the
  96-case (4 families × 24) filtered exam actually run.
- `runs/p6-2-decoder-rescore-20260913/{355735,401577,493261}/` — per-checkpoint
  evidence/predictions/observations/decode-reports for each run (constrained
  fixed decoder for all three; additionally `plain` for 493261; additionally
  the full 1272-case constrained run for 401577).
