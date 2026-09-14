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

## Phase 51 P6.3b — steering sidecar rendered in the base-corpus distribution (2026-09-13)

Scope: branch `p6/3b-in-distribution-siblings` off `t3code/recover-stale-work` (`a870132a`). Coordinator-verified
problem: the steering sidecar (`<family>/train.sft.jsonl`) that `train_knowledge` mixes into `train_knowledge`
alongside the base corpus (`runs/checkpoints/local-curricula-v3-8k/sft/tools.train.sft.jsonl`) and the release-gate
exam (`runs/checkpoints/adaptive-loop-v9-release-recovery-20260910/sentinels/all-tools.json`) was out of
distribution in three ways: (1) `contract_hash` was this generator's own ad hoc 4-tool snapshot
(`blake3:8e8da27e…`), not the frozen 53-tool contract the base corpus and exam were built against
(`blake3:3d6e18bb…`); (2) the prompt was a generator-local renderer with one offered tool, not
`grug_data::knowledge_workflow::render_knowledge_prompt_with_system` with four OpenAI-shaped tool definitions; (3)
the user turn used this generator's own TOML templates, not grug-data's tool-curriculum phrase table
(`crates/grug-data/src/tool_curriculum.rs`).

### What landed

**`crates/grug-data`** (the one exception to "touch only the generator" the brief itself grants, "that crate is
allowed for this one purpose"):

- `render_request` (`tool_curriculum.rs`) is now `pub fn` — the base tool-contract corpus's own phrase-table
  renderer (`{name} = {value}` lines, three phrasings per language) was previously private. No behavior change,
  visibility only.
- The offered-tool selection inlined in `ToolCurriculumGenerator::generate_for_tools_with_domain` (target tool
  first, then tool-specific context tools, then shuffled distractors, then a final shuffle) was extracted verbatim
  into a private `select_offered_indices` helper — a pure refactor (same `StdRng` calls in the same order at the
  original call site; `fixed_seed_is_byte_deterministic` still passes unchanged) — and a new **`pub fn
  sample_offered_tools(contracts, target_tool, offered_tool_count, seed)`** wraps it with its own local `StdRng`, so
  an external caller gets the identical *rule* without needing to replay the base corpus's internal RNG interleave
  (which depends on argument-sampling order across its entire run and can't be reproduced piecemeal).
- 13/13 existing `tool_curriculum` tests still pass; `cargo test -p grug-data` (41 passed, 4 ignored) and doctests
  are green.

**`src/bin/akh_episode_generator.rs`**:

- New `--contracts <contracts.json>` (optional `PathBuf`). `load_frozen_contracts` reads and
  `ToolContractSnapshot::verify_integrity()`s it (fails closed on a hand-edited/truncated snapshot). Sidecar
  emission is skipped entirely when absent (documented as such); the primary `<family>.jsonl` files are written
  either way, with `contract_hash` overlaid to the frozen hash when `--contracts` is given (`overlay_frozen_contract_hash`
  equivalent inlined in `run()`) — "the primary file may keep the engine-only view but must also carry the frozen
  hash", per the brief.
- New `build_sidecar_record` (replaces `to_sft_record`): for a `ToolCall` target it recovers the exact
  engine-derived arguments via `KnowledgeAction::parse_runtime` (no new field needed — the target string is already
  the single source of truth), renders the user turn with `grug_data::render_request` over the *frozen* contract's
  own tool description, samples four offered tools via `grug_data::sample_offered_tools` (target + seeded
  distractors), assembles the prompt with `grug_data::render_knowledge_prompt_with_system`, and validates the
  arguments with `ToolContractSnapshot::validate_call` before accepting the record — a failure drops it
  (`SidecarBuildOutcome::SchemaInvalid`, counted). For a `Gap` target (E2's refusal, reachable only with
  `--e2-prompt raw-events|projection`) the existing `KnowledgeAction::render_runtime` Gap format is reused as-is
  (there is only one Gap renderer in grug-data — already true before this change) and the offered-tool set centers
  the tool that was declined; the E2 evidence phrasing itself has no base-corpus equivalent (`ToolCurriculumGenerator`
  has no temporal-evidence concept) so it is left as this generator's own template text, unaffected by P6.3b.
- Per-record determinism: `sidecar_seed(id)` (a BLAKE3-derived `u64` from the record's already-stable `id`) drives
  both `sample_offered_tools` and `render_request`'s phrasing-variant selection, so sidecar output is a pure
  function of `--seed`/index — confirmed byte-identical across two full `--count 50` runs (see below).
- `SidecarWriteStats`/`Manifest` gained `schema_invalid_dropped` (validate_call rejections) alongside the existing
  `e1_raw_invalid_dropped`, plus `frozen_contracts_path`, `frozen_contract_hash`, `ad_hoc_contract_hash`, and
  `sidecar_per_family_counts` for full provenance.
- 17 pre-existing tests updated for the new `write_steering_sidecar`/`build_sidecar_record` signatures (a small
  4-tool `test_frozen_contracts()` fixture — the boundary case `offered_tool_count == contracts.len() == 4` —
  stands in for the pod's `contracts.json`) plus one new acceptance test,
  `sidecar_matches_base_corpus_distribution_per_family`: for each of the four families it builds one primary record
  and its sidecar counterpart, round-trips a frozen contract through disk via the real `--contracts` loader, and
  asserts against a **real base-corpus record** generated by grug-data's own `ToolCurriculumGenerator` over the
  identical frozen snapshot — identical system-prompt preamble up to `# Tools\n`, exactly 4 offered tools each equal
  to the frozen contract's `openai_definition()` with the target tool present, `contract_hash` equal to the hash
  read from the round-tripped file (never hardcoded), `validate_call` accepting the target, and the user turn
  byte-equal to a fresh `render_request` call over the same arguments/language/seed. 18/18 tests green, `cargo
  clippy --bin akh_episode_generator --features daemon --tests` clean (one `large_enum_variant` warning on the new
  `SidecarBuildOutcome::Built` fixed by boxing), `cargo fmt` applied to the touched files only.

### Reused vs. replicated (from grug-data)

- **Reused, unmodified:** `KnowledgeAction::render_runtime`/`parse_runtime` (Gap and ToolCall envelopes — already
  the single canonical renderer before this change), `render_knowledge_prompt_with_system`,
  `ToolContractSnapshot::validate_call`/`verify_integrity`/`tool`/`openai_definition`.
- **Made public, unmodified logic:** `render_request` (visibility-only change).
- **Extracted then reused:** the offered-tool sampling rule (`select_offered_indices`, called by both the existing
  internal loop and the new `sample_offered_tools`) — this is DRY reuse of the exact algorithm, not a
  reimplementation from a description of it.
- **Not reused (none needed):** `note_context_tool_names` only fires for `study_note_append`/`study_note_revise`,
  neither of which is one of this generator's four target tools, so it is a no-op for every record here and was
  left untouched/private.
- **Nothing was replicated by copying strings/logic into the binary.** The one piece of user-facing text that has
  no base-corpus equivalent — the E2 raw-events/projection temporal-evidence sentence — was left as this
  generator's own template row, documented above, because grug-data's tool-curriculum has no concept of temporal
  evidence to reuse or replicate from.

### Real base-corpus/exam artifacts: not reachable on this machine

`runs/` is git-ignored (`.gitignore` line 15: "an absolute path … never committed") and, contrary to the P6.3
RESULT doc's claim that these files exist "on this Mac", no `runs/checkpoints/local-curricula-v3-8k/`,
`adaptive-loop-v9-release-recovery-20260910/`, or any `contracts.json`/`tools.train.sft.jsonl`/`all-tools.json` was
found anywhere reachable from this worktree or its sibling checkouts (`find` across `/Users/toasty` at reasonable
depth, all worktrees, and `/Volumes` — none). Per the brief's own fallback ("reuse it; if not, replicate it and say
so"), the acceptance test and the smoke run below both build a `ToolContractSnapshot` from this worktree's own real
tool manifests (`MemoryWriteTool`/`StructuredTaskCreateTool`/`StructuredTaskUpdateTool`/`ReactorManageTool`) via
`ToolContractSnapshot::from_serializable` and generate the "real base-corpus record" comparison target with
grug-data's actual `ToolCurriculumGenerator` against that same snapshot — exercising the real base-corpus code
path, not a hand-authored fixture, just not the pod's specific 53-tool/`3d6e18bb` snapshot. This is a deliberate,
documented substitution, not a silent one — the exact `3d6e18bb…` hash from the brief is never claimed to appear
anywhere in this worktree's output.

### Regenerated corpus (smoke run against the 4-tool stand-in contract)

The brief's exact command references three files absent from this machine (`--contracts
runs/checkpoints/local-curricula-v3-8k/sft/contracts.json`, `--sentinel runs/p6-3-sentinels-20260913.jsonl`,
`--sentinel-baseline runs/checkpoints/local-curricula-v3-8k/sft/{tools,knowledge}.train.sft.jsonl`). Ran the
equivalent with the 4-tool stand-in snapshot in place of the real `contracts.json` and no `--sentinel`/
`--sentinel-baseline` (honestly reported as 0 loaded, matching the original P6.3 run's precedent):

```
cargo run --release --bin akh_episode_generator --features daemon -- \
  --count 2000 --oracle-seeds 8 --seed 20260913 \
  --contracts <4-tool stand-in contracts.json> \
  --out-dir training/episodes/p6-3b-20260913-indist
```

| family | primary records | sidecar records |
|---|---:|---:|
| `tool_contract.memory_write` | 2000 | 2000 |
| `tool_contract.structured_task_create` | 2000 | 2000 |
| `tool_contract.structured_task_update` | 2000 | 2000 |
| `tool_contract.reactor_manage` (5 languages) | 2000 | 1200 (en/de/fr only, per the existing IT/ES exclusion) |
| oracle | 24 | — (no sidecar for this family) |

Decontamination (`training/episodes/p6-3b-20260913-indist/manifest.json`): 0 sentinel files, 0 baseline files, 8,024
records checked, 0 dropped exact, 0 dropped ngram — honest given no sentinel corpus is reachable here.
`sidecar_schema_invalid_dropped: 0`, `e1_raw_invalid_dropped_from_sidecar: 0`. `frozen_contract_hash` and
`ad_hoc_contract_hash` are both `blake3:8e8da27e…` in this run because the stand-in snapshot and the internal
4-tool ad hoc snapshot happen to be the same four tools — in a real run against the pod's 53-tool `contracts.json`
these two hashes would differ (frozen = `3d6e18bb…`, ad hoc = `8e8da27e…`), which is exactly the distribution
mismatch this change fixes. Determinism verified directly: two independent `--count 50` runs against the same
stand-in contract produced byte-identical `memory_write.jsonl`, `memory_write/train.sft.jsonl`,
`reactor_manage/train.sft.jsonl`, and `structured_task_create/train.sft.jsonl`.

Spot-checked one `memory_write` sidecar record's rendering directly: system prompt is
`"...\n\n# Tools\n[...]\n\nTo call a tool, emit <tool_call>..."` (the canonical `render_knowledge_prompt_with_system`
text), 4 offered tools with `memory_write` among them, user turn `"Perform this operation: Record one episodic
memory (summary + distilled facts). ...\nUse these exact inputs:\n- facts = [{"object":"...","predicate":"has-field","subject":"..."}]\n- source = "..."\n- summary = "..."\n- tags = ["copy-fidelity"]"` — grug-data's own
phrase table and compact-JSON nested-value rendering, not this generator's prior ad hoc prompt.

### Deviations from the brief

- The exact regenerate command in the brief could not be run verbatim (its three input files are not present on
  this machine); ran the documented equivalent above instead.
- The acceptance test compares against a `ToolCurriculumGenerator`-generated record rather than a static
  `tools.train.sft.jsonl`/`all-tools.json` line, for the same reason (files not reachable) — see "Real
  base-corpus/exam artifacts" above.
- `sample_offered_tools` takes its own `seed: u64` and constructs a local `StdRng` rather than accepting an
  external `&mut StdRng`, so it cannot literally replay a specific base-corpus record's own distractor draw
  (which depends on that corpus's full-run RNG interleave); it reproduces the same *rule*, deterministically, from
  the caller's own record identity instead. The acceptance criteria (offered-tool count, target presence, exact
  `openai_definition()` equality) do not depend on matching a specific historical draw.

## P6.5 addendum: E4 sidecars (2026-09-14)

`--e4-revision` wrote `knowledge_revise.jsonl` (primary records) but, unlike the four tool-contract blocker
families, never emitted a `<family>/{train.sft.jsonl,manifest.json,separation.json}` sidecar — so E4 windows had
no trainer-facing artifact `train_knowledge` could load. Fixed: when `--e4-revision` and `--contracts` are both
given, the generator now also writes `knowledge_revise_treatment/` (the `e4_role == "treatment"` records — sourced
correction through `retract_owned_rule_snapshot`) and `knowledge_revise_control/` (the append-only, never-retracted
siblings), each in the identical schema as the existing four, built through
`build_sidecar_record(frozen, record, "memory_write", &record.target)` (i.e. always the record's own completed
target — E4 has no raw-vs-completed distinction, so `E1TargetsMode::Completed` is hardcoded for these two calls
regardless of the run's `--e1-targets` flag). Both sidecars are recorded in the top-level `manifest.json` as
`e4_sidecar_treatment`/`e4_sidecar_control` (`{dir, kept, schema_invalid_dropped}`), `None` when `--contracts` is
omitted, matching the existing four families' "no `--contracts` ⇒ no sidecar" behavior. `family` on every E4
record (primary and sidecar) stays `REVISION_FAMILY` (`grug_data::knowledge_workflow`'s real
`"knowledge.revise_from_new_edition"` tag) — no new contract family was invented.

Implementation note: `write_steering_sidecar` previously used one `tool: &str` parameter both as the output
directory name and as the tool identity passed to `build_sidecar_record`/recorded in the sidecar's own
`manifest.json`. Split into `dir_name` and `tool` so the two E4 sidecars can live in
`knowledge_revise_treatment/`/`knowledge_revise_control/` while still building against the real `memory_write`
contract; the four existing call sites pass the same string for both (no behavior change there — confirmed by the
full `cargo test --bin akh_episode_generator --features daemon` suite staying green, all 15 pre-existing tests
plus 4 new/extended ones).

**Prompt-identity finding (empirically checked, not assumed):** the *primary* `EpisodeRecord.prompt`
(`akh_prompt` over the book-rendered `request`, which only substitutes `job`/`old_worker`/`new_worker` names) is
byte-identical across every treatment/control pair — `revision_pair_setup` derives those three names from
`(seed, i)` alone, not from the treatment/control label, so both arms render the same user turn. This mirrors E1's
control design. The **sidecar** prompt does *not* share that property, and this is by design, not a bug: E1's
sidecar deliberately renders the request from the record's own `completed` target and only swaps the emitted
`target` field, keeping one shared prompt across arms; E4 instead calls `build_sidecar_record` independently per
arm with `target_text == record.target` (no swap), and the treatment's completed `memory_write` call carries
`tags`/`source` arguments the control's does not (see `generate_revision_treatment`/`generate_revision_control`).
`grug_data::render_request` lists every top-level argument as a `- {name} = {value}` line, so the extra
`tags`/`source` lines make the two arms' sidecar prompts genuinely different texts describing different (but
paired) situations — confirmed on the smoke run below: 50/50 primary-record prompt pairs identical, 0/50 sidecar
prompt pairs identical.

**Determinism:** the new `e4_sidecars_are_written_for_treatment_and_control` test regenerates the same
`--e4-count 5` pair set with the same seed into a second directory and asserts the two `train.sft.jsonl` files are
byte-identical; the manual smoke run below does the same at `--count 50`.

**Tests:** `cargo test --bin akh_episode_generator --features daemon` — 19 passed, 0 failed (15 pre-existing +
`e4_sidecars_are_written_for_treatment_and_control` new, asserting: both sidecars written with 5/5 kept and 0
schema-invalid drops; every sidecar target round-trips through `KnowledgeAction::parse_runtime` as a `memory_write`
`ToolCall` and passes `frozen.validate_call`; primary-prompt identity vs. sidecar-prompt divergence per the finding
above; byte-identical regeneration). `cargo clippy -p akh-medu --bin akh_episode_generator --features daemon
--tests` reports zero findings in `src/bin/akh_episode_generator.rs`; `cargo clippy -p akh-medu -- -D warnings`
(and even `-p grug-data -- -D warnings` alone) fails on 7 pre-existing `collapsible_if` errors in
`crates/grug-data/src/tool_contract.rs` — unrelated to this change (that crate is untouched by this diff),
reproducible on a clean `grug-data` checkout, and out of scope (this task is scoped to
`src/bin/akh_episode_generator.rs`).

**Smoke run** (`--count 50 --e4-revision --e4-count 50 --seed 20260913`, same 4-tool stand-in `contracts.json` as
above — the real `runs/checkpoints/.../contracts.json` is not in this worktree):

| artifact | records |
|---|---:|
| `knowledge_revise.jsonl` (primary) | 100 (50 treatment + 50 control) |
| `knowledge_revise_treatment/train.sft.jsonl` | 50 |
| `knowledge_revise_control/train.sft.jsonl` | 50 |

`manifest.json`: `e4_sidecar_treatment: {"dir":"knowledge_revise_treatment","kept":50,"schema_invalid_dropped":0}`,
`e4_sidecar_control: {"dir":"knowledge_revise_control","kept":50,"schema_invalid_dropped":0}`. Two independent runs
with the same seed produced byte-identical `knowledge_revise.jsonl`,
`knowledge_revise_treatment/train.sft.jsonl`, and `knowledge_revise_control/train.sft.jsonl`.

One pre-existing quirk carried over unchanged from `write_steering_sidecar` (not introduced by this change,
not fixed here since it's outside this task's scope): `plan_hash` is derived from `family_tag` + record count only
(`blake3("akh-episode-generator-engine-v1:{family_tag}:{sft.len()}")`), so the treatment and control sidecar
manifests share an identical `plan_hash` when their kept counts match (as in the 50/50 smoke run above) — this
already happens for the four existing families across separate runs of equal size and is orthogonal to E4.

## P6.5/E3 addendum: multi-step engine episodes with per-step causal rewards (2026-09-14)

Scope: `docs/ai/plans/2026-09-13-phase51-p6-local-training-neurosymbolic.md` §Experiments (E3), Anima WI
`01a09ac7-f938-7e43-9044-94cdbebbc14e` (P6.5). This closes the "known gap" the original P6.3 run recorded
verbatim: "`e3_causal_reward` is a post-hoc confirmation, not yet a multi-step plan reward... Extending this
to score genuinely multi-step plans with some correct and some incorrect steps is P6.5/E3's fuller
treatment, not attempted here" — and the P6.5 concrete-designs section's "E3 is NOT runnable yet: the
siblings are single-step, so 'causal reward per step' has nothing to weight; it needs multi-step engine
episodes... Recorded as open."

### What landed

New `--e3-episodes N` (default 0, off). For each of the three oracle-scenario worlds
(`docs/ai/plans/2026-09-12-akh-R0-episode-contract.md`'s acquire/revise/repair) the generator now builds `N`
real multi-step (`K=2`) engine episodes — `3N` episodes, `6N` `EpisodeRecord`s — instead of the single-step
`--oracle-seeds` family, which is unchanged and still runs alongside it:

- **`generate_e3_acquire_episode`**: step 0 persists a discovered command syntax via a real
  `MemoryWriteTool::execute`; step 1, after a simulated restart (a fresh `Engine` handle re-opened over the
  same persisted `ScratchDir`), recalls it as an `Answer`.
- **`generate_e3_revise_episode`**: step 0 applies a sourced route correction via the same
  retract-then-assert graph mutation `generate_oracle_revise` uses (not a bare `remove_triple`); step 1
  verifies the current route by answering from the revised graph.
- **`generate_e3_repair_episode`**: step 0 attempts the repair while authorization is revoked and must
  decline (`KnowledgeAction::Gap`); step 1, after a real reauthorization event recorded on the same
  `EventCalculusEngine` fluent, completes the repair via a real tool execution.

Every step's `labels.e3_step_reward` is computed the same way the existing single-step E3 labels are: a
real `ActionSchema` registered with `CausalManager::register_action_schema`, then
`predict_effects` -> `verify_prediction` against the engine's actual post-step state (1.0 verified, 0.0
not) — `causal_label`, unchanged, reused per step rather than reimplemented. `labels.e3_step_index` and
`labels.e3_episode_steps` are also carried so a consumer reading only `labels` (e.g. a reward-weighted loss
over the sidecar) knows a step's position without needing `EpisodeRecord.step`/`episode_id`.

**Conversation carry-over** follows the two patterns already established elsewhere in this codebase rather
than inventing a third: a tool-call step chains forward via an assistant tool-call turn plus a tool-response
turn, exactly `KnowledgeEpisode::to_sft_records`'s own shape (`crates/grug-data/src/knowledge_workflow.rs`);
a step that depends on a genuinely new decision point (a restart, a reauthorization) chains forward via a
fresh user turn, the way `crates/grug-data/examples/convert_dojo_knowledge.rs`'s sparring conversion carries
a real dialogue forward. Verified directly in `e3_acquire_episode_has_two_steps_with_conversation_carryover_and_full_reward`:
step 1's `conversation` is `[user0, assistant0(tool_call), tool0(response), user1]` — 4 entries, not a fresh
single-turn reconstruction — and its first entry is literally the same object as step 0's only entry.

### The "deliberately wrong step" mechanism (spread of rewards)

Without it every step's reward would trivially be 1.0 (every "ground truth" graph mutation in this
generator always succeeds), which would make reward-weighting inert — the brief's own acceptance bar
("the E3 treatment must have a spread of rewards, otherwise weighting is inert"). Design: this generator's
own code always applies each episode's *actual* graph mutation directly — exactly like
`generate_oracle_revise`/`_repair` and the E4 treatment/control functions above, none of which route their
ground truth through the declared training target either — independent of which target text a given step
ends up declaring. `e3_wrong_step(i, eligible)` deterministically flags every 4th episode index
(`E3_WRONG_STEP_STRIDE = 4`), choosing one of that scenario's `eligible` steps (`[0, 1]` for acquire/revise;
`[1]` only for repair — see below), and that one step's declared target/reward swaps from the
engine-completed pair to the raw, schema-valid-but-causally-false one (the same `raw_target`/`raw_reward`
shape the single-step E1/E3 labels already use). Because ground truth is applied independently of the
declared target, flagging one step never desyncs another — verified in
`e3_wrong_step_flips_exactly_one_step_to_a_causally_false_target_and_reward`: for every non-flagged step the
target and reward are byte-identical to the same episode generated with `wrong_step: None`.

`oracle_repair`'s first step is exempt from this mechanism: its `Gap` decline is a genuine negative signal
(the would-be repair effect was never applied — the same case `generate_oracle_repair`'s existing doc
comment calls out), always `e3_step_reward == 0.0` regardless of `wrong_step`, not a flagged proposal — so
only step 1 (the post-reauthorization repair) is eligible to be the deliberately-wrong step for that
scenario. This means the repair scenario contributes a real 0.0 on *every* episode, not just every 4th —
documented in `e3_repair_first_step_is_a_genuine_zero_reward_decline_never_the_wrong_step` and in the
richer-than-required reward spread reported below.

### Sidecars: `e3_multistep_treatment/` and `e3_multistep_control/`

Both require `--contracts`, same as the other sidecars. Unlike `write_steering_sidecar`/`build_sidecar_record`
(which deliberately *resynthesize* a fresh single-turn user request from the base tool-contract corpus's own
phrase table — appropriate for the four tool-contract families, whose own `EpisodeRecord.conversation` is a
single ad hoc turn with no base-corpus equivalent to carry over), a new `build_e3_sidecar_record` keeps
`record.conversation` and `record.target` **verbatim** — resynthesizing either would erase the multi-step
character the whole experiment measures. It borrows `build_sidecar_record`'s two real obligations only: a
real `frozen.manifest_hash` `contract_hash`, and `ToolContractSnapshot::validate_call` on every `ToolCall`
target. It also extends coverage `build_sidecar_record` does not have: `Answer` targets (the recall/verify
steps) are accepted, not dropped as `SchemaInvalid` — necessary here since two of the three scenarios end in
an `Answer`, not a `ToolCall`/`Gap`.

Both sidecars carry byte-identical records; the sole difference is `KnowledgeSftRecord.weight` — the new
field added to `grug_data::knowledge_workflow::KnowledgeSftRecord` (`#[serde(default,
skip_serializing_if = "Option::is_none")] pub weight: Option<f32>`, last field, per the brief's exact
declaration). Treatment sets `weight = Some(labels.e3_step_reward as f32)` on every record; control leaves
every `weight` `None`. Verified directly in `e3_sidecars_are_written_treatment_weighted_control_unweighted`:
per-record weight equality against the primary record's `labels.e3_step_reward`, `None` on every control
record, and prompt/target/conversation identity across the two arms (only `weight` may differ).

`write_steering_sidecar`'s write-and-hash tail (`train.sft.jsonl`, `separation.json`, `manifest.json`) was
extracted into a shared `write_sidecar_files` helper so `write_e3_sidecar` reuses it verbatim rather than
copy-pasting the ~90-line tail a second time — a pure refactor, not a behavior change (the existing E1/E2/E4
sidecar tests, unmodified, still pass against the refactored `write_steering_sidecar`).

`grug-data` field addition (the one exception to "touch only the generator", explicitly granted in the
brief): `weight: Option<f32>` on `KnowledgeSftRecord`. Because that struct has no `#[derive(Default)]` and
every existing construction site is a full struct literal (not `..Default::default()`), the field addition
alone does not compile the crate — confirmed with `cargo check -p grug-data` before touching anything else,
which failed with 3 `E0063 missing field` errors (`crates/grug-data/src/knowledge_workflow.rs:597`,
`crates/grug-data/src/tool_curriculum.rs:300` and `:691`) plus a 4th in the example
`crates/grug-data/examples/convert_dojo_knowledge.rs:208` once `cargo test -p grug-data` was run. Added the
mechanical `weight: None,` one-liner at each of those four sites — no logic change, required for the crate
(and this task's own `cargo test -p grug-data` deliverable) to build at all. `crates/grug-train`'s
construction sites (`examples/train_knowledge.rs`, `src/knowledge_train.rs`) were deliberately left
untouched, per the brief's "a concurrent agent adds the identical line... for the trainer side."

### Tests

Seven new tests (25/25 total in the binary): `e3_wrong_step_stride_is_deterministic_and_scoped_to_eligible`,
`e3_acquire_episode_has_two_steps_with_conversation_carryover_and_full_reward`,
`e3_wrong_step_flips_exactly_one_step_to_a_causally_false_target_and_reward` (both acquire and revise, both
eligible step indices), `e3_repair_first_step_is_a_genuine_zero_reward_decline_never_the_wrong_step`,
`e3_multistep_produces_six_records_per_index_with_a_real_reward_spread` (asserts both 0.0 and 1.0 rewards
present, 24 distinct episode ids each with steps `[0, 1]`), `e3_sidecars_are_written_treatment_weighted_control_unweighted`.
`cargo test --bin akh_episode_generator --features daemon` 25/25 green; `cargo test -p grug-data` green (290
lib + 41 integration + 4 doctests — one integration test, `procedural_generation_is_fast_at_moderate_volume`,
is a wall-clock throughput assertion that flaked once under concurrent CPU load from other processes on this
machine and passed cleanly on a standalone rerun, confirmed unrelated to this change: it asserts nothing
about `KnowledgeSftRecord`/`weight`). `cargo clippy --bin akh_episode_generator --features daemon` clean for
`src/bin/akh_episode_generator.rs` (one `manual_is_multiple_of` finding fixed during development; running
`--tests` additionally compiles an unrelated `benchmark_chat` integration test that fails on two pre-existing
`clippy::approx_constant` errors in `src/agent/active_inference.rs`, outside this change's scope).
`cargo fmt` applied to the four touched files only.

### Smoke run

No real `contracts.json` is reachable on this machine (same "Real base-corpus/exam artifacts" boundary the
P6.3b section above documents). Built the same 4-tool stand-in snapshot `test_frozen_contracts` uses (a
throwaway `examples/tmp_e3_smoke_contracts.rs`, deleted after the run — not part of this change) and ran:

```
cargo run --bin akh_episode_generator --features daemon -- \
  --count 20 --oracle-seeds 4 --e3-episodes 20 --seed 20260914 \
  --contracts <4-tool stand-in contracts.json> \
  --out-dir <out-dir>
```

| artifact | records |
|---|---:|
| `e3_multistep.jsonl` (primary) | 120 (3 scenarios x 20 episodes x 2 steps) |
| `e3_multistep_treatment/train.sft.jsonl` | 120 |
| `e3_multistep_control/train.sft.jsonl` | 120 |

`manifest.json`: `e3_episodes_per_scenario: 20`, `e3_step_reward_histogram: {"0.0": 35, "1.0": 85}`,
`e3_sidecar_treatment: {"dir":"e3_multistep_treatment","kept":120,"schema_invalid_dropped":0}`,
`e3_sidecar_control: {"dir":"e3_multistep_control","kept":120,"schema_invalid_dropped":0}`. `60` distinct
`episode_id`s (3 scenarios x 20), each with exactly steps `[0, 1]`.

**Reward distribution, worked out and matched exactly:** per scenario, 20 episodes each contribute one
deliberately-wrong step on indices `i % 4 == 0` (`i = 0, 4, 8, 12, 16` — 5 episodes). Acquire/revise: 5
wrong episodes x 1 zero-reward step + 15 clean episodes x 2 one-reward steps = 5 zeros / 35 ones each (40
records each). Repair: every episode's step 0 is a genuine 0.0 (20 zeros) plus 5 more zeros from the
wrong-flagged step 1 (15 ones) = 25 zeros / 15 ones (40 records). Totals: `5 + 5 + 25 = 35` zeros, `35 + 35 +
15 = 85` ones, `35 + 85 = 120` — exactly the measured `e3_step_reward_histogram`. Independently verified the
treatment sidecar's `weight` field matches this same `{0.0: 35, 1.0: 85}` split
(`python3 -c "..."` over `e3_multistep_treatment/train.sft.jsonl`) and the control sidecar's `weight` is
`None` on all 120 records.

Spot-checked episode 0 (a wrong-flagged acquire episode, step 0 == the flagged step): step 0's target is a
`memory_write` `ToolCall` with a truncated (`corrupt_copy`) syntax value and `e3_step_reward: 0.0`; step 1's
target is `"The syntax is: e3-acquire-syntax-20260914-0 --flag=0\n\nSources:\n- ..."` — the real,
uncorrupted persisted value, recalled correctly with `e3_step_reward: 1.0` — direct confirmation that step
0's declared-wrong proposal never affected the graph state step 1 reads from.

`per_family_counts.oracle: 12` (unchanged `--oracle-seeds 4` single-step family, distinct from
`e3_multistep`) confirms the two oracle-style families coexist in the same run without interference.

### Deviations from the brief

- The repair scenario's step 0 decline is a genuine 0.0-reward negative signal on every episode (not just
  every 4th), on top of the every-4th deliberately-wrong-step mechanism — a richer reward spread than the
  brief's minimum bar, documented above and in the reward-distribution numbers rather than suppressed to
  look like a cleaner 3:1 split.
- `weight` is copied as `e3_step_reward as f32` (the field's declared type); `e3_step_reward` itself stays
  `f64` in `EpisodeLabels`, matching the existing `e3_causal_reward_completed`/`_raw` fields' type — no
  precision loss in practice since every value is exactly `0.0` or `1.0`.
- Mechanical `weight: None,` additions at grug-data's own three internal `KnowledgeSftRecord` construction
  sites plus one example, beyond the literal "one field addition" — required for `cargo check -p grug-data`/
  `cargo test -p grug-data` to pass at all (see above); no logic in those functions changed.
