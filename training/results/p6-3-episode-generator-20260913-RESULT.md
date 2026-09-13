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

## Phase 51 P6.5 — experiment corpora: E1 raw-valid control, E2 prompt modes, E4 revision curriculum (2026-09-13)

Scope: `docs/ai/plans/2026-09-13-phase51-p6-local-training-neurosymbolic.md` §Experiments, Anima WI
`01a09ac7-f938-7e43-9044-94cdbebbc14e` (P6.5). Extends the P6.3 generator so the same engine-generated
corpus can serve E1, E2 and (as a stretch) E4 with real controls, rather than only carrying E1-E3 as
labels on a single always-completed corpus.

### E1 — `--e1-targets completed|raw-valid` (default `completed`)

The sidecar (`<family>/train.sft.jsonl`) target now swaps to `labels.e1_raw_target` under `raw-valid`, and
any record whose raw target is not runtime-valid (`e1_raw_target_runtime_valid == false`) is dropped from
the sidecar rather than swapped in — the control must be schema-valid-but-value-wrong, not
schema-invalid-garbage, so the treatment/control difference is exactly symbolic completion. Both the
per-family sidecar `manifest.json` (`e1_targets_mode`, `e1_raw_invalid_dropped`) and the top-level run
manifest (`e1_targets_mode`, `e1_raw_invalid_dropped_from_sidecar`) record the mode and the drop count.
The primary `<family>.jsonl` files are never affected — only the sidecar.

Measured (`--count 200`, seed 20260913, `--e1-targets raw-valid`): `memory_write` /
`structured_task_create` / `structured_task_update` keep all 200 (their raw targets are schema-valid,
value-wrong echoes/placeholders); `reactor_manage` drops all 200 (its raw target is the documented
malformed-envelope failure — free text outside `<tool_call>` tags — which is deliberately never
runtime-valid). Verified directly: the sidecar's `target` field equals `labels.e1_raw_target` and differs
from the primary record's completed `target` for every kept record (spot-checked programmatically, not
just asserted in a unit test).

### E2 — `--e2-prompt none|raw-events|projection` (default `none`)

`none` is byte-for-byte the P6.3 behavior (prompt never mentions temporal evidence, target is always
`add`/`update`). `raw-events` adds a template row (all 5 languages for `reactor_manage`; English-only for
`structured_task_update`, matching that family's existing single-language scope) that states the raw
grant/revoke event timestamps; `projection` instead states the event-calculus `holds_at` current-truth
projection. Both non-`none` modes gate the *target* on validity, not just the label: a revoked
authorization (`reactor_manage`, `i % 6 == 5`) or a closed task (`structured_task_update`, `i % 5 == 4`)
now produces a `KnowledgeAction::Gap` refusal — the same refusal envelope already proven runtime-accepted
by P6.3's `oracle_repair` episodes — instead of `add`/`update`, and the tool is never actually executed in
that branch (so the E3 causal-reward check on the same triple is a genuine negative signal, not a
post-hoc label on an action that ran anyway). Every reactor_manage/structured_task_update record now also
carries `labels.e2_prompt_mode` recording which mode produced it. New template kinds:
`add_raw_events_granted/revoked` and `add_projection_granted/revoked` (5 languages) in
`reactor_manage.toml`; `update_priority_raw_events_open/closed` and
`update_priority_projection_open/closed` (English) in `structured_task.toml`.

Measured (`--count 200`, seed 20260913): `--e2-prompt raw-events` on `reactor_manage` produced 33/200
revoked records (`i % 6 == 5` over 200) with `stage: "declare_gap"` and a target starting `Knowledge gap:
reactor-authorization` — confirmed programmatically that every revoked record's target does **not**
parse as a `ToolCall`, and every valid record's target does parse as a `ToolCall` with
`operation: "add"`. `--e2-prompt projection` shows the same split (same `i % 6` seed determinism). `none`
mode's target is `add` in both the valid and the revoked case (unchanged P6.3 behavior — confirmed the
`none`/`raw-events`/`projection` prompt texts for the same episode index are all mutually distinct).

### E4 (stretch) — `--e4-revision --e4-count N`, family `knowledge_revise`

Implemented and green. New family `knowledge_revise` (file stem, real grug-data family tag
`grug_data::REVISION_FAMILY` = `"knowledge.revise_from_new_edition"` — checked, not invented) generates
paired episodes per index: **treatment** builds an owned `(job, current-worker, old_worker)` fact, applies
the sourced correction through the engine's real exact-owned-edge revision path
(`Engine::retract_owned_rule_snapshot`, recovered P6.0, the same primitive `tests/exact_owned_revision.rs`
exercises — not `oracle_revise`'s plain `remove_triple`), then asserts the corrected fact and actually runs
`memory_write` against the revised graph (`tags: ["route-correction", "revision"]`, `source: job`);
**control** is append-only — the new fact is asserted via the same tool without ever retracting the stale
one, leaving both present (the unresolved contradiction that is the point of the control; no `tags`/
`source`, so the target text itself differs). Each pair runs in its own isolated engine (`ScratchDir` +
`build_engine`, mirroring the existing `generate_oracle_acquire` per-episode pattern) so
`retract_owned_rule_snapshot`'s exact-metadata match against `Engine::all_triples()` stays cheap
regardless of total corpus size. Prompt: new `training/episodes/templates/knowledge_revise.toml`
(English; distinct from `oracle.toml`'s `oracle_revise` row).

Measured (`--e4-revision --e4-count 200 --count 200`, seed 20260913): 400 `knowledge_revise` records (200
treatment + 200 control). Confirmed programmatically: every treatment record has
`e4_retraction_applied: true` / `e4_stale_edge_still_present: false`; every control record has the
reverse; every treatment/control pair's `target` differs; `family` is the real
`knowledge.revise_from_new_edition` tag on every record. Scope: `memory_write` only — a
`knowledge.revise`-family (data-to-text) episode was judged out of scope for this pass (`grug-data`'s
`WorkflowStage`/`KnowledgeAction` types have no dedicated revision-narration action beyond `ToolCall`/
`Answer`/`Gap`, and inventing one would be exactly the kind of unrequested new type CLAUDE.md's "strongly
typed idiomatic Rust" guidance argues against without a driving consumer). Not attempted: a
`structured_task_update`-after-close-and-owned-retraction variant, or multi-hop revision chains — both are
natural P6.6+ follow-ups if E4 shows a training effect worth deepening.

### Tests, lint, determinism

Five new unit tests (17/17 total): `e1_raw_valid_mode_drops_invalid_and_swaps_valid_targets`,
`e2_raw_events_and_projection_prompts_differ_and_gate_the_target`,
`e4_revised_graph_target_differs_from_append_only_control`,
`generate_revision_pairs_produces_treatment_and_control_siblings`, plus the pre-existing 12 all still
green. `cargo clippy --bin akh_episode_generator --features daemon` clean (one duplicate
`#[allow(clippy::too_many_arguments)]` introduced and immediately removed during development).
`cargo fmt` applied to this file only. All four `--count 200` smoke runs above used the same seed
(20260913, the CLI default) and reran once with `rm -rf` between runs on the `e1-raw-valid` output
directory with byte-identical resulting counts, confirming same-seed determinism holds under the new
modes.

### Deviations from the brief

- The brief's "least-privileged valid operation (e.g. `reactor_manage` `list`/no-op)" alternative was not
  used; `KnowledgeAction::Gap` was chosen instead because P6.3 already established it as the runtime's
  accepted refusal envelope (`oracle_repair`), so E2's revoked case is structurally identical to a case the
  training corpus and runtime already handle, rather than introducing a second refusal shape.
- New `reactor_manage.toml` translations (German/French/Italian/Spanish) for the four new
  `add_raw_events_*`/`add_projection_*` kinds are direct, non-professionally-reviewed translations
  following the vocabulary already established by the existing `add`/`enable` rows in the same file, not
  independently verified by a native speaker.
- `structured_task_create` was not E2-gated (only `structured_task_update`, per the brief's "and, if the
  engine state supports it cheaply, `structured_task_update`" — create has no precondition to revoke).
