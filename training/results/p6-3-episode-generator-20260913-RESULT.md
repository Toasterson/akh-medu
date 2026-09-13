# Phase 51 P6.3 — engine-in-the-loop episode generator (2026-09-13)

Scope: ADR 106 / P6 plan (`docs/ai/plans/2026-09-13-phase51-p6-local-training-neurosymbolic.md`), Anima WI P6.3
(`01a09ac7-c18c-7cd1-bb1a-3698b7256fa3`). Session-driven; no runner involved.

## What landed

`src/bin/akh_episode_generator.rs` (new binary, `required-features = ["daemon"]` in `Cargo.toml` — needed
for `agent::reactor`/`ReactorManageTool`). Drives akh-medu's own `Engine` + `MemoryWriteTool` /
`StructuredTaskCreateTool` / `StructuredTaskUpdateTool` / `ReactorManageTool` + `EventCalculusEngine` +
`CausalManager` (the same machinery `examples/akh_{symbolic,temporal_incident,causal}_episode_bridge.rs`
exercise) to generate, label, and decontaminate training episodes — not a template string generator.
Prompt phrasing lives entirely in `training/episodes/templates/*.toml` (data files); the Rust source
contains no authored natural-language constant, per the repo's language-agnostic hard rule.

Run: `cargo run --release --bin akh_episode_generator --features daemon -- --count 2000 --oracle-seeds 8`
Test: `cargo test --bin akh_episode_generator --features daemon` (12/12 pass)

## Counts (full run, seed 20260913)

| family | records | file |
|---|---:|---|
| `tool_contract.memory_write` | 2000 | `memory_write.jsonl` |
| `tool_contract.structured_task_create` | 2000 | `structured_task_create.jsonl` |
| `tool_contract.structured_task_update` | 2000 | `structured_task_update.jsonl` |
| `tool_contract.reactor_manage` (5 languages) | 2000 | `reactor_manage.jsonl` |
| oracle (3 R0-world scenarios x 8 seeds) | 24 | `oracle.jsonl` |

Total 8024 engine-generated records, all distinct engine states (unique doc/task/schedule labels per
index — 2000/family was genuinely reachable, no paraphrase-only padding). Wall clock: 4m34s on the M2
for the full run (`time` in the generation log). Generator git hash: `95368800c21f5b4f550ec59f56e932cd22cd7598`.

Decontamination: 0 sentinel files loaded (none are checked into this worktree — the release-gate sentinel
corpora live only on the Runpod volume / `runs/` local evidence directories, neither present here), so
0/8024 dropped. The mechanism itself (`--sentinel <path>...`, repeatable) mirrors
`crates/grug-data/examples/prepare_foundation.rs`'s decontamination exactly: NFKC-lowercase-alphanumeric
word normalization, exact-text BLAKE3 hash, and 13-word shingle BLAKE3 hashes, checked against every
record's `prompt` and `target`. Two unit tests (`sentinel_exact_text_is_excluded`,
`sentinel_thirteen_word_shingle_is_excluded`) prove both paths reject a synthetic sentinel.

## Decontamination against the real sentinels (architect session, after merge)

The quarantined release-gate exam sets are on this Mac since P6.4a (`runs/checkpoints/adaptive-loop-v9-release-recovery-20260910/sentinels/`).
They were flattened to `runs/p6-3-sentinels-20260913.jsonl` (1,272 tool cases + 800 knowledge episodes = 2,072 lines, 3,344 texts).
A first run with `--sentinel` alone dropped **all 2,000 `memory_write` records** on the 13-word shingle — the collision was the
`memory_write` tool's own description ("…the typed replacement for raw text journal ingestion in an LLM session"), which every one of
the 24 sentinel cases and every base-corpus record carries. That is contract boilerplate, not held-out content. The generator gained
`--sentinel-baseline <jsonl>` (repeatable): shingles/exact texts that also occur in a baseline file (the base SFT corpus
`local-curricula-v3-8k/sft/{tools,knowledge}.train.sft.jsonl`, 86,732 texts) are removed from the sentinel index first, so only
sentinel-private shingles remain. Result (`training/episodes/p6-3-20260913-decontaminated/manifest.json`):

| sentinel texts indexed | baseline texts | shingles removed as boilerplate | records checked | dropped exact | dropped shingle |
|---:|---:|---:|---:|---:|---:|
| 3,344 | 86,732 | 2,338 | 8,024 | 0 | 0 |

Determinism: regenerating in a second worktree gave byte-identical `*.jsonl` for every family; only `manifest.json` differs
(`generator_git_hash`). The decontaminated corpus is the one P6.4 trains on.

## Record schema

Primary per-family files (`<family>.jsonl`) use a local `EpisodeRecord` type whose base fields mirror
`grug_data::knowledge_workflow::KnowledgeSftRecord` field-for-field (`schema_version`, `id`, `episode_id`,
`step`, `stage`, `family`, `difficulty`, `language`, `contract_hash`, `tools`, `conversation`,
`system_context`, `prompt`, `target`), plus one additional `labels: Option<EpisodeLabels>` field carrying
the P6.3 experiment fields:

- **E1 (symbolic completion):** `e1_raw_target` (uncompleted/naive control — envelope-valid-but-wrong-value
  for `memory_write`/`structured_task_create`, or an unresolved placeholder id for
  `structured_task_update`, or a documented malformed-envelope failure for `reactor_manage`),
  `e1_raw_target_runtime_valid` (does the raw control even parse via
  `KnowledgeAction::parse_runtime`?), `e1_completed_changed_from_raw`.
- **E2 (temporal validity):** `e2_temporal_validity` (a real `EventCalculusEngine::holds_at` projection —
  every record gets a grant event and ~1-in-4/5/6 also get a revoke event before the decision time),
  `e2_raw_events` (the undigested event list — the control).
- **E3 (causal reward):** `e3_causal_reward_completed` / `e3_causal_reward_raw` — both computed by
  registering a real single-effect `ActionSchema` and calling
  `CausalManager::predict_effects` -> `verify_prediction` against the *actual* post-execution engine
  state. The completed target's effect was really asserted (reward 1.0); the raw target's effect never
  was (reward 0.0 in the overwhelming majority of cases — `structured_task_create`'s raw control can
  partially overlap since it under-specifies rather than corrupts, which is itself an honest signal, not
  a bug).

`grug-data` has no `#[serde(deny_unknown_fields)]` on `KnowledgeSftRecord`, so the extra `labels` field is
silently ignored by any reader that only wants the base SFT shape — verified directly (see "steering
sidecar" below).

## `train_knowledge` steering-sidecar compatibility (added mid-task per coordinator request)

Real pod artifacts on this Mac (`runs/checkpoints/adaptive-loop-v9-release-recovery-20260910/windows/
window-0002-attempt-0001/training-sibling/{manifest.json,separation.json,train.sft.jsonl,
training-sibling-plan.json}` plus `training-sibling.generate.log`, and the base corpus at
`runs/checkpoints/local-curricula-v3-8k/sft/{contracts.json,tools.train.sft.jsonl}`) were read (read-only,
never modified) to match `train_knowledge.rs`'s schema exactly. For each of the four tool-contract
families the generator now also writes `<family>/{train.sft.jsonl,manifest.json,separation.json}`:

- **`train.sft.jsonl`** is literally `Vec<grug_data::knowledge_workflow::KnowledgeSftRecord>` — the exact
  type `train_knowledge`'s `load_records` deserializes for `--train-jsonl`/`--eval-jsonl`. Verified two
  ways: (1) `to_sft_record` maps `EpisodeRecord` -> `KnowledgeSftRecord` and a unit test
  (`steering_sidecar_records_are_real_knowledge_sft_records`) round-trips one through
  `serde_json::to_string`/`from_str::<KnowledgeSftRecord>`; (2) the full-run sidecar files were produced by
  the same code path (`write_steering_sidecar`), so every line in every `<family>/train.sft.jsonl` is a
  real `KnowledgeSftRecord`, not my own shape.
- **`contract_hash`** is a *real* `grug_data::ToolContractSnapshot::from_serializable(&[four ToolManifests])
  .manifest_hash` — grug-data's own documented bridge for snapshotting runtime manifests without a
  reverse crate dependency (`crates/grug-data/src/tool_contract.rs`). It will **not** equal a specific pod
  run's `contracts.json` hash (e.g. `blake3:3d6e18bb...`), because that snapshots the akh's complete tool
  catalog, not just these four tools — documented in the source comment at the call site.
- **`manifest.json`** matches the real example's field set exactly (`schema_version`, `complete`,
  `contract_hash`, `difficulty`, `family`, `generator_version`, `kind`, `plan_hash`, `records`,
  `separation_hash`, `sft_hash`, `source_exam_file_hash`, `source_exam_fingerprint`, `tool`).
  `generator_version`/`kind` are set to this generator's own values (`akh-episode-generator-engine-v1` /
  `engine_in_the_loop_training_sibling`) so a reader can tell the two generation paths apart.
  `sft_hash`/`separation_hash` are real BLAKE3-of-file-bytes, computed the same way
  `fingerprint_artifact_file` does and verified equal in a unit test.
- **`separation.json`** matches the real example's field set exactly. `exact_user_request_overlap` and
  `exact_prompt_target_overlap` are genuinely measured (duplicate-prompt / duplicate-(prompt,target)-pair
  counts within the sidecar's own records — both 0 in the full run). `exam_family_cases` is honestly `0`
  and `max_transcript_ngram_jaccard` is `0.0` with a comment explaining it means "not computed against an
  exam" (no frontier exam exists in this path), not "zero measured overlap."

**Boundary, stated once rather than re-litigated per family:** the *full* `--steering-sibling-manifest`
mode in `train_knowledge.rs` (`verify_steering_inputs`) additionally requires a live `--steering-decision`
(`AdaptiveExamController` state) and `--steering-frontier-exam`
(`ToolContractExamSet`/`KnowledgeExamSet`), and it *regenerates* the sibling records from
`ToolCurriculumGenerator`/`KnowledgeWorkflowGenerator` to structurally cross-check them against that
frontier exam. Those two artifacts only exist mid-adaptive-loop on a training pod — this generator is a
genuinely different, engine-grounded generation path (ADR 106 decision 3), and does not fabricate them:
doing so would be either inert (fields that only pass the shallow `starts_with("blake3:")` check) or
actively wrong (claiming a frontier exam that was never run). `source_exam_file_hash`/
`source_exam_fingerprint` are therefore self-referential placeholders (`"blake3:<hash of
no-frontier-exam:<tool>>"` / `"engine-generated-no-frontier-exam:<tool>"`), documented as such in the
source and correctly **not** intended to pass `verify_steering_inputs`'s frontier-exam cross-check. What
*is* real and works today without any of the four steering flags: `train_knowledge --train-jsonl
<family>/train.sft.jsonl --eval-jsonl <held-out split of the same>` loads and trains on this corpus with
zero conversion.

## Known gaps

- **Italian/Spanish are not in the steering sidecar.** `grug_data::knowledge_workflow::WorkflowLanguage`
  has exactly 3 variants (English/German/French) and many existing generators/tests assume
  `WorkflowLanguage::all()` has 3 members; extending it was judged out of scope here (wide blast radius on
  a heavily-tested shared crate for a P6.3-scoped change). All 5 languages (English/German/French/
  Italian/Spanish) *are* present and tested in the primary `reactor_manage.jsonl` (400/language, envelope
  identical across all 5 — `reactor_manage_envelope_is_identical_across_languages` test); only the
  `KnowledgeSftRecord`-typed sidecar (1200/2000 records: en/de/fr) excludes it/es. A `WorkflowLanguage`
  extension is the natural follow-up if 5-language steering-mode training is wanted.
- **The 24 oracle episodes are newly engine-generated, not a byte-identical replay.** The P6 plan's
  starting-facts table cites "24 oracle episodes via the Engine bridge" from
  `training/results/akh-r1-*-RESULT.md` / `akh-r1-local-mlx-evidence-20260913/`; the underlying source
  data for those specific 24 episodes lives under `runs/` (local evidence, not checked into this
  worktree) and was not reachable. `generate_oracle` instead regenerates the *pattern* of R0's three
  reference worlds (`docs/ai/plans/2026-09-12-akh-R0-episode-contract.md`: acquire/act/restart/reuse,
  revise-graph-change-action, repair-interaction-through-changed-authority) through the identical
  generator path (same `Engine`/`EventCalculusEngine`/`CausalManager` machinery, same decontamination,
  same manifest), at 8 seeds x 3 scenarios = 24 episodes, so the generator hash covers an oracle set with
  the same shape and count. This is a deliberate, documented substitution, not a silent one.
- **`e3_causal_reward` is a post-hoc confirmation, not yet a multi-step plan reward.** Every "completed"
  target's effect really was asserted before verification (reward always 1.0), and every "raw" target's
  effect was deliberately never applied (reward 0.0, except `structured_task_create`'s under-specified —
  not corrupted — raw control, which can partially overlap). Extending this to score genuinely multi-step
  plans with some correct and some incorrect steps is P6.5/E3's fuller treatment, not attempted here.
- No full-workspace `cargo test` was run (this is a purely additive change — a new binary target plus one
  new `[[bin]]` Cargo.toml entry, touching no existing source file); `cargo test --bin
  akh_episode_generator --features daemon` (12/12) and `cargo clippy --bin akh_episode_generator --features
  daemon` (clean) are the verification performed.
