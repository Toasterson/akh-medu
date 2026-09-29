# Operating the adaptive exam and native text-vision slice

This runbook covers the locally executable foundation from [ADR
100](../decisions/100-adaptive-exams-native-text-vision.md). It does not turn a
synthetic smoke test into model-quality evidence.

## 1. Initialize controller state

```bash
cargo run -p grug-data --example adaptive_exam_controller -- init \
  --run-id p51-vision-001 \
  --generator-version babyllm-vision-text-v1 \
  --seed 42 \
  --capabilities vision.text.line:0:5:2,vision.text.layout:0:5:1,reason.chain.depth:0:6:3 \
  > state.json
```

The default decision floor is 200 items per family/window. Precursor steering
is disabled by default. `--min-samples-per-window` can set an explicit,
family-appropriate floor, and `--enable-precursor-steering` is the audited CLI
switch for turning early intervention on. Do not enable it for a visual
precursor until the checkpoint lead-time study shows that the selected signal
actually leads outcome failure. The narrower schema-valid/wrong-value tool
signal is governed separately by ADR 102.

## 2. Generate a rolling/frontier text-vision exam

```bash
cargo run -p grug-data --example generate_vision_exam -- \
  --out artifacts/exam-step-000100 \
  --seed 740059131 \
  --count 200 \
  --variants 7 \
  --difficulty 2 \
  --domain exam
```

The command writes `manifest.jsonl` and RGB PNGs. Seven variants cover clean,
low contrast, salt/pepper noise, box blur, occlusion, uneven lighting and
horizontal shear. Variants share a `semantic_id`, allowing exact
normalized-output metamorphic grading.

This generator is a rolling/frontier probe. A new seed from it is **not** the
generator-independent audit required for promotion.

## 3. Produce and score model predictions

Write one JSONL row per attempted case:

```json
{"case_id":"vtext-...","text":"recognized transcript","confidence":0.83}
```

Then score it:

```bash
cargo run -p grug-data --example score_vision_exam -- \
  --manifest artifacts/exam-step-000100/manifest.jsonl \
  --predictions artifacts/predictions-step-000100.jsonl \
  > artifacts/score-step-000100.json
```

The result includes exact/CER/WER/blank/near-miss metrics, language/layout/
corruption slices, exact-transcript metamorphic consistency, Wilson rate
intervals, deterministic case-bootstrap CER/WER intervals and controller-ready
`suggested_pass_count`, `suggested_precursor_count` and
`suggested_catastrophic_count` fields. Blank outputs are catastrophic outcomes;
near misses and consistency/collapse warnings remain predictive precursors.

Repeated output across different semantic items is reported as
`cross_semantic_collapse_rate` and contributes to the precursor count. Repeated
predictions across variants of the same semantic item do not.

## 4. Feed an observation and obtain the next plan

An observation is counts, not rounded rates:

```json
{"checkpoint":{"step":100,"artifact_hash":"sha256:..."},"tier":"frontier","family":"vision.text.line","difficulty":2,"sample_count":1000,"pass_count":731,"precursor_count":94,"catastrophic_count":3}
```

```bash
cargo run -p grug-data --example adaptive_exam_controller -- step \
  --state state.json \
  --observations observations.jsonl \
  --tier frontier \
  --checkpoint-step 200 \
  --checkpoint-hash sha256:... \
  --samples 200 \
  > decision.json

jq '.controller' decision.json > state-next.json
jq '.decisions, .next_plan' decision.json
```

Persist the complete prior state and decision artifact. Re-running from the
same state/checkpoint produces the same plan and seed.

Ineligible sub-floor windows are observable but do not update EWMA, trend,
difficulty or family priority. Predictive precursors remain shadow-only until
lead time is proven and do not affect family priority while disabled.
Confidence-qualified catastrophic hard outcomes stop after the configured
confirmation windows regardless of that precursor toggle. Each family's
sentinel difficulty is pinned at registration; sentinel item seeds and
fingerprints therefore survive both checkpoint changes and frontier-difficulty
adaptation. Its checkpoint field records when the same frozen case set was
scored.

## 5. Quarantine and audit a sibling training corpus

Only a `frontier` plan can derive a training sibling through
`derive_training_sibling`. Sentinel, rolling-probe and audit tiers fail closed.
After generation, audit the candidate manifests:

```bash
cargo run -p grug-data --example audit_vision_sibling -- \
  --exam artifacts/exam-step-000100/manifest.jsonl \
  --training artifacts/training-sibling-000100/manifest.jsonl
```

Admission requires:

- no exact transcript or pixel overlap;
- at least one held-out generator axis;
- maximum character-trigram Jaccard at or below 0.80;
- foreground text perceptual-hash distance at least 10 in the closest
  transcript neighborhood.

The command emits the evidence even when it exits nonzero. A failure is an
artifact to investigate, not a reason to remove the check.

For a knowledge-workflow frontier, materialize the controller's exact
`next_plan` as an exam first, then derive the training sibling from that frozen
exam:

```bash
cargo run -p grug-data --example knowledge_workflow -- exam \
  --contracts runs/knowledge/contracts.json \
  --family FAMILY_FROM_NEXT_PLAN --tier frontier \
  --run-id RUN_FROM_NEXT_PLAN --seed SEED_FROM_NEXT_PLAN \
  --plan-generator-version GENERATOR_FROM_NEXT_PLAN \
  --difficulty DIFFICULTY_FROM_NEXT_PLAN --samples SAMPLES_FROM_NEXT_PLAN \
  --checkpoint-step STEP --checkpoint-hash MODEL_HASH \
  > runs/knowledge/frontier-window-2.json

cargo run -p grug-data --example knowledge_workflow -- sibling \
  --contracts runs/knowledge/contracts.json \
  --exam runs/knowledge/frontier-window-2.json \
  --out runs/knowledge/training-sibling-window-2 --samples 500
```

The sibling command regenerates the exam from its declared plan, refuses every
protected tier, derives a separate seed namespace, rejects episode/source
overlap, measures only model-visible user prompts, and applies the 0.80 trigram
ceiling before writing `manifest.json` last. It emits SFT and complete hashes;
the source exam itself is not copied into the training directory.

For a controller-selected `tool_contract.NAME` frontier, materialize the tool
exam with the controller plan's run ID, checkpoint, difficulty, seed and sample
count, then derive the targeted sibling:

```bash
cargo run -p grug-data --example generate_tool_contract_exam -- \
  --contracts runs/knowledge/contracts.json \
  --out runs/knowledge/tool-frontier-window-2.json \
  --run-id RUN_FROM_NEXT_PLAN --checkpoint-step STEP \
  --checkpoint-hash MODEL_HASH --tier frontier \
  --difficulty DIFFICULTY_FROM_NEXT_PLAN --seed SEED_FROM_NEXT_PLAN \
  --samples-per-tool SAMPLES_FROM_NEXT_PLAN --offered-tools 4 \
  --family tool_contract.NAME_FROM_NEXT_PLAN

cargo run -p grug-data --example tool_contract_sibling -- \
  --contracts runs/knowledge/contracts.json \
  --exam runs/knowledge/tool-frontier-window-2.json \
  --family tool_contract.NAME_FROM_NEXT_PLAN \
  --out runs/knowledge/tool-sibling-window-2 --samples 500
```

The targeted frontier contains only the selected family's sample count while
each case still uses the complete live catalog for distractors. Omitting
`--family` preserves the all-tool sentinel format and fingerprint. Targeted
sentinel and audit generation is refused. This path replays the exam, rejects
sentinel/rolling/audit inputs, uses a
separate multilingual prompt bank, and records zero exact request and
prompt-target overlap plus trigram evidence. It works for zero-argument tools;
the historical targeted 50-tool `agent_list` smoke measured maximum Jaccard
0.293447.

Only a statistically eligible `steer` decision may branch an unpromoted
candidate into a new curriculum segment. Supply the full evidence set; the
sibling SFT is included automatically:

```bash
cargo run -p grug-train --release --features cuda \
  --example train_knowledge -- \
  --contracts runs/knowledge/contracts.json \
  --train-jsonl runs/knowledge/train-replay.sft.jsonl \
  --eval-jsonl runs/knowledge/eval-replay.sft.jsonl \
  --steer-from-bundle runs/knowledge/model-segment-1 \
  --steering-decision runs/knowledge/probe-1/controller.step.json \
  --steering-frontier-exam runs/knowledge/frontier-window-2.json \
  --steering-sibling-manifest runs/knowledge/sibling-window-2/manifest.json \
  --steering-sibling-jsonl runs/knowledge/sibling-window-2/train.sft.jsonl \
  --artifacts-dir runs/knowledge/model-segment-2 \
  --model small --epochs 1
```

The trainer reproduces `next_plan`, rejects any `stop`, replays the frontier
and sibling, checks adjacent separation evidence and every hash, then retains
both model and Adam state with monotonic checkpoint steps. The output remains
an uncertified candidate. `watch` and `green` never authorize a mixture change.

The sibling is limited to five percent of the segment's records using an exact
integer check. Base/sibling counts and the resulting parts-per-million share
are carried in `steered_from` lineage.

### 5a. Run bounded continuous windows

After the first rolling probe has written `controller.step.json`, the bounded
supervisor performs the complete middle loop without promoting anything:

The current source-head supervisor is the seven-family v2 contract. Do not
point it at the still-running, frozen five-family v1 directory; retain that
run's pinned supervisor and create a versioned v2 adaptation directory.
This source head exposes 53 tools. Browse metadata with `study_note_list`, read
one selected head with `study_note_read`, create with `study_note_append`, and
update with `study_note_revise` using the exact revision token returned by the
read. A conflict is a normal recoverable observation: read the head again and
retry against the new token; never strip the token or force an overwrite. The
all-tool corpus/exam generator version is v2 because it enforces and samples
`pattern` and `maxLength` constraints. Its note-write targets also retain the
earlier retrieval/head results required by the runtime gates while preserving
the four-tool decision budget.

```bash
python training/scripts/run_tool_native_adaptive_loop.py \
  --base /workspace/babyllm/tool-native-v2 \
  --out /workspace/babyllm/tool-native-v2/adaptive-loop-v2 \
  --train-binary /root/target-cuda/release/examples/train_knowledge \
  --knowledge-workflow-binary /root/target-cuda/release/examples/knowledge_workflow \
  --tool-exam-generator-binary /root/target-cuda/release/examples/generate_tool_contract_exam \
  --knowledge-eval-binary /root/target-cuda/release/examples/eval_knowledge_exam \
  --tool-eval-binary /root/target-cuda/release/examples/eval_tool_contract_exam \
  --controller-binary /root/target-cuda/release/examples/adaptive_exam_controller \
  --tool-sibling-binary /root/target-cuda/release/examples/tool_contract_sibling \
  --max-windows 8 --max-sibling-records 500
```

Each source-head v2 window evaluates seven frozen workflow sentinels, the
frozen complete tool sentinel, and one controller-selected targeted frontier.
A `watch` or `green` result continues one ordinary epoch with the exact existing corpus. A
statistically eligible frontier `steer` creates a separated sibling and a new
candidate segment while retaining model and Adam state. Any `stop` ends the
loop before an update. `windows.jsonl`, per-window evidence, controller states,
and segment bundles are replayable; `promotion_enabled` is always false. The
all-tool sentinel is created once with checkpoint step `0` and hash
`unbound:immutable-sentinel`; its evaluator binds each evidence report to the
candidate that actually executed it.

The supervisor records the exact executable hashes and launch budgets before
waiting. Restart it with the same command plus `--resume`: completed ledger
windows and content-hashed sentinels are reconstructed, while an interrupted
evaluation starts a new attempt directory without deleting partial evidence.
An interruption during model training or controller rebinding is deliberately
not retried automatically; the status remains fail-closed so a possibly
advanced candidate cannot receive a duplicate update. Terminal `stop` and
window-budget states are never reopened by `--resume`.

### 5b. Adapt to added, changed, or removed tools

Capture the next real runtime snapshot, including exact descriptors for tools
that are planned and implemented but not deployed yet. Compare it with the
contracts beside the active model:

```bash
cargo run -p grug-data --example prepare_tool_contract_delta -- \
  --previous-contracts runs/knowledge/model-small/contracts.json \
  --contracts runs/knowledge/contracts-next.json \
  --out runs/knowledge/tool-contract-delta-v2 \
  --run-id tool-contract-delta-v2 --offered-tools 4
```

The immutable bundle contains both endpoint snapshots, `contract-delta.json`,
train/eval SFT and a manifest written last. It replays every current tool,
overweights added and model-visible changed tools, and teaches a canonical gap
for removed or schema-incompatible legacy requests. Train/eval identity and
prompt-target overlap are refused. Metadata-only drift exits nonzero because
it should be handled by Akh policy, not by changing model weights.

A running Grug-backed workspace now captures these endpoints automatically in
`$XDG_DATA_HOME/akh-medu/workspaces/AKH/model-contract-drift/delta-HASH/`.
Each directory is named by the verified delta and contains
`previous-contracts.json`, `contracts.json`, `contract-delta.json` and a
last-written `manifest.json` with file identities. Feed its endpoint files to
the command above. Do not train directly from a log line or from the delta
summary without both complete snapshots. Planned-but-not-yet-deployed tools
still enter through an explicitly captured next snapshot; they are never
invented by the runtime journal.

Mix the delta with a small disjoint replay of the complete seven-family
knowledge workflow, then warm-start a new bundle from the certified active
model:

```bash
cargo run -p grug-train --release --features cuda \
  --example train_knowledge -- \
  --contracts runs/knowledge/contracts-next.json \
  --train-jsonl runs/knowledge/train-replay.sft.jsonl \
  --supplemental-jsonl runs/knowledge/tool-contract-delta-v2/train.sft.jsonl \
  --eval-jsonl runs/knowledge/eval-replay.sft.jsonl \
  --supplemental-eval-jsonl runs/knowledge/tool-contract-delta-v2/eval.sft.jsonl \
  --adapt-from-bundle runs/knowledge/model-small \
  --artifacts-dir runs/knowledge/model-small-v2 --model small --epochs 2
```

This uses the same active-bundle verifier as serving, preserves the tokenizer
and causal-foundation lineage, records the certified parent by content hash,
and resets Adam for the new immutable corpus. It cannot write back into the
parent bundle. The result remains a candidate and must pass fresh stateful and
all-tool sentinels for the new contract before activation.

### 5b. Freeze the independent promotion audit

The promotion audit uses `ab_glyph` outline rasterization and a content-hashed
OpenType font, not the bootstrap bitmap renderer. Keep the font outside the
corpus directory, verify its source separately, and record the exact identity
in every manifest row:

```bash
cargo run -p grug-data --example generate_vision_exam -- \
  --out runs/vision-training-sibling-v2-20260904 \
  --seed 51101 --count 150 --variants 7 --difficulty 2 \
  --domain training-sibling

cargo run -p grug-data --example generate_vision_audit -- \
  --out runs/vision-audit-noto-v1-20260904 \
  --font artifacts/fonts/NotoSansMono-Regular.ttf \
  --font-id notofonts/NotoSansMono \
  --font-revision v2.014-9b7310b8f99fcd2583c49606e6aefca13a391350 \
  --font-license OFL-1.1 \
  --seed 91700 --count 150 --variants 7 --difficulty 2 \
  > runs/vision-audit-noto-v1-20260904.report.json

cargo run -p grug-data --example validate_vision_corpus -- \
  --corpus runs/vision-audit-noto-v1-20260904 \
  --require-domain audit --min-per-cell 10 \
  --training-manifest runs/vision-training-sibling-v2-20260904/manifest.jsonl \
  > runs/vision-audit-noto-v1-20260904.validation.json
```

The pinned upstream is Noto Sans Mono v2.014, source commit
`9b7310b8f99fcd2583c49606e6aefca13a391350`, under OFL-1.1. The official
release ZIP SHA-256 is
`090cf6c5e03f337a755630ca888b1fef463e64ae7b33ee134e9309c05f978732`;
the selected font's BLAKE3 is
`blake3:ed7db3a5ba7dd62ccee116ffe532f550e495bbba1871b50ff908f73b8ee3abe8`.

The frozen local audit has 1,050 cases: exactly ten cases in every
EN/DE/FR x five-layout x seven-corruption cell. Its manifest hash is
`blake3:03d2a3cbbdd7e87fb4b417bb0eda30aa5c6fc25aa169990b833af31421e73fec`.
Validation recorded zero exact transcript/pixel overlap, held-out renderer and
content-bank axes, maximum trigram Jaccard 0.1896551724 and minimum perceptual
distance 35. These facts validate the audit corpus; they are not model scores.

## 6. Preflight visual-token cost

```bash
cargo run -p grug-train --example vision_token_budget -- \
  --max-seq 1024 --text-tokens 128 > visual-token-budget.json
```

The matrix covers line, receipt, screenshot and page inputs at patch sizes
8/16/24/32/48. Full pages cannot be treated as a late optimization: if
readable patches exceed the budget, tiling/resampling or a compact front-end is
a prerequisite.

### 6a. Run the stem-only resolution probe

```bash
cargo run -p grug-train --example vision_stem_probe -- \
  --steps 120 --learning-rate 0.003 > vision-stem-probe.json
```

This is 32-way pinned-glyph classification with held-out translations and no
decoder. Use `--glyph-scale` and `--patch-size` to repeat a single matrix cell.
Do not translate its accuracy into CER.

The raw-patch result is red. ADR 101 therefore adds an opt-in compact
convolutional comparison. Its default curriculum covers every translation in
a bounded 13x13 neighborhood except the two explicit held-out positions:

```bash
cargo run --release -p grug-train --features cuda \
  --example vision_conv_stem_probe -- \
  --steps 3000 --learning-rate 0.003 \
  > vision-conv-stem-probe.json
```

On Burn ndarray CPU, the recorded calibration is 23.7 seconds per 32-image
training step. Use CUDA/ROCm for the full gate; do not treat a short throughput
calibration as accuracy evidence. The captured A40 run cleared 100% on both
8px and 16px held-out positions. This unlocks the decoder rung but is not OCR
accuracy or independent-renderer evidence.

## 7. Native shared-decoder training boundary

The model path is:

```rust
use grug_train::{
    NativeVisionTransformerConfig, TinyTransformerConfig,
    VisionPatchStemConfig, init_vision_training, patchify_rgb,
    train_vision_step,
};

let config = NativeVisionTransformerConfig::new(
    TinyTransformerConfig::p1a(vocab_size),
    VisionPatchStemConfig::new(3, 16, 384, 64, 64),
);
let (model, mut optimizer) = init_vision_training(&config, &device);
let grid = patchify_rgb(&rgb, width, height, 16)?;
let rows = grid.rows;
let columns = grid.columns;
let patches = grid.into_tensor(&device);
let (model, loss) = train_vision_step(
    model,
    &mut optimizer,
    patches,
    rows,
    columns,
    &text_batch,
    learning_rate,
);
```

Visual patches are continuous prefix embeddings. Text targets and masks remain
ordinary tokenizer IDs, so this slice does not renumber the existing tokenizer
ABI. Loss is applied only to text positions while gradients flow through the
visual stem.

Run the end-to-end CPU overfit smoke before claiming that the wiring can learn:

```bash
cargo run -p grug-train --example vision_overfit_smoke -- \
  --cases 3 --steps 100 --learning-rate 0.001 \
  > vision-v0-cpu-smoke.json
```

A falling loss is necessary but insufficient. Treat `v1_gate_passed=false` as
a red capability gate; inspect exact rate, CER and
`cross_semantic_collapse_rate` rather than promoting on loss alone.

### 7a. Run the compact 2--10M clean-line gate

Once the stem gate is green, run the actual shared decoder on at least 128
training-sibling images. This command uses a 3.31M-parameter trunk and saves
both model and Adam state; the output gate still uses greedy transcription,
not teacher-forced loss:

```bash
cargo run --release -p grug-train --features cuda \
  --example vision_compact_overfit -- \
  --cases 128 --batch-size 32 --steps 4000 \
  --learning-rate 0.001 --min-learning-rate 0.00001 \
  --checkpoint-every 500 \
  --artifacts-dir artifacts/vision-v1-checkpoint \
  > artifacts/vision-v1-clean-line.json
```

The learning rate follows a deterministic cosine schedule between the two
recorded endpoints. Retain the stderr step/loss/rate log: a sustained loss
floor or oscillation is a steering signal, not a reason to erase the attempt.
When an artifacts directory is present, complete model/Adam snapshots are
written under `progress/step-NNNNNN/` every 500 steps; `meta.json` is written
last, so a partly written directory is not a valid stop/resume artifact. Use
`--checkpoint-every 0` only for a deliberately disposable calibration.

The rung passes only at `exact_rate >= 0.98`,
`cross_semantic_collapse_rate <= 0.01`, and at least 128 cases. The corpus is
explicitly `training_sibling`; passing demonstrates capacity to learn the
native pixel-to-text path, not held-out OCR generalization. Its 96 visual
tokens consume 18.75% of the configured 512-token context, below the 25%
preflight ceiling.

Materialize the exact image corpus used by this rung for provenance with the
same index stride as the training example:

```bash
cargo run -p grug-data --example generate_vision_exam -- \
  --out artifacts/compact-clean-line-training \
  --seed 51100 --count 128 --index-stride 15 --variants 1 \
  --difficulty 1 --domain training-sibling
```

This selection is intentionally English/line/clean. It is not the EN/DE/FR
generalization rung, and it is never relabelled as an exam.

Replay a saved checkpoint through the provider-free CPU path before publishing
it. `CARGO_NET_OFFLINE` also prevents Cargo from consulting the registry; the
binary itself contains no network or provider client:

```bash
CARGO_NET_OFFLINE=true cargo build --release -p grug-train \
  --example vision_compact_infer

# macOS: prove the already-built runtime succeeds under an OS network deny.
sandbox-exec -p '(version 1) (allow default) (deny network*)' \
  target/release/examples/vision_compact_infer \
  --checkpoint artifacts/vision-v1-checkpoint --cases 3 \
  > artifacts/vision-v1-local-offline.json
```

Record wall time and peak resident memory separately (for example with
`/usr/bin/time -l`) because capability success alone is not a local deployment
latency claim.

The captured A40 run passed 128/128 exact with zero CER/WER/collapse. Its
network-denied Mac replay passed 3/3 exact in 12.27 model seconds, with
44,253,184 bytes peak RSS. See
[`vision-v1-a40-RESULT.md`](../../../training/results/vision-v1-a40-RESULT.md)
for hashes and limitations.

### 7b. Warm-start the full tool-native model

Native OCR training starts only from the exact certified text/tool active
artifact. The visual stem is trained first while the text trunk is under
`no_grad`; joint training then alternates visual and quarantined text-retention
updates. The command is candidate-only and cannot write an active pointer:

```bash
cargo run --release -p grug-train --features cuda \
  --example train_native_ocr -- \
  --text-bundle runs/knowledge/model-small \
  --vision-train-manifest runs/vision-training-sibling-v2-20260904/manifest.jsonl \
  --vision-audit-manifest runs/vision-audit-noto-v1-20260904/manifest.jsonl \
  --audit-validation runs/vision-audit-noto-v1-20260904.validation.json \
  --retention-train-jsonl runs/knowledge/tool-native-v1/corpus/train.sft.jsonl \
  --retention-train-jsonl runs/knowledge/tool-native-v1/dojo/train.sft.jsonl \
  --retention-train-jsonl runs/knowledge/tool-native-v1/tools/train.sft.jsonl \
  --retention-eval-jsonl runs/knowledge/tool-native-v1/corpus/eval.sft.jsonl \
  --retention-eval-jsonl runs/knowledge/tool-native-v1/dojo/eval.sft.jsonl \
  --retention-eval-jsonl runs/knowledge/tool-native-v1/tools/eval.sft.jsonl \
  --artifacts-dir runs/knowledge/model-small-native-ocr \
  --canvas-width 480 --canvas-height 192 --batch-size 8 \
  --stem-warmup-updates 500 --joint-epochs 3 --retention-every 1
```

The 480x192 canvas and stride-16 stem produce 360 visual prefix tokens, leaving
1,688 positions of the 2,048-token text context. `multimodal-candidate.json`
binds the tokenizer, tool contracts, certified text base, train/audit corpora,
retention corpora, model checkpoint and optimizer state. An uncertified text
candidate can be used only with `--allow-uncertified-text-candidate`; that
lineage remains ineligible for promotion.

Periodic checkpoints are immutable direct children of `checkpoints/` and are
published by an atomic directory rename only after `model.mpk`, `optim.mpk`
and their hash-bearing `meta.json` are complete. Resume with the exact original
arguments plus one of those paths:

```bash
cargo run --release -p grug-train --features cuda \
  --example train_native_ocr -- \
  [the exact original arguments] \
  --resume runs/knowledge/model-small-native-ocr/checkpoints/update_XXXXXXXX
```

The command re-hashes the model and Adam state, requires the stored `run.json`
to match every training, evaluation and image-admission parameter, verifies
the copied tokenizer/contracts, and reconstructs the deterministic stem/joint
batch schedule from the saved counters. A completed candidate checkpoint uses
`candidate_update_XXXXXXXX` and can finish interrupted evaluation/output
publication idempotently. It cannot be resumed after `completion.json` or an
active pointer exists, and a partial or out-of-tree checkpoint is refused.

For an unattended chained run, use the candidate-only supervisor. It waits for
the separate text release gate, pins the executable and every manifest/corpus
by SHA-256, and never invokes a certifier:

```bash
python3 training/scripts/run_native_ocr_training.py \
  --text-release-gate runs/knowledge/text-release-gate-v1 \
  --out runs/knowledge/native-ocr-training-v1 \
  --artifacts-dir runs/knowledge/model-small-native-ocr \
  --train-binary target/release/examples/train_native_ocr \
  --vision-train-manifest runs/vision-training-sibling-v2-20260904/manifest.jsonl \
  --vision-audit-manifest runs/vision-audit-noto-v1-20260904/manifest.jsonl \
  --audit-validation runs/vision-audit-noto-v1-20260904.validation.json \
  --retention-train-jsonl runs/knowledge/tool-native-v1/corpus/train.sft.jsonl \
  --retention-train-jsonl runs/knowledge/tool-native-v1/dojo/train.sft.jsonl \
  --retention-train-jsonl runs/knowledge/tool-native-v1/tools/train.sft.jsonl \
  --retention-eval-jsonl runs/knowledge/tool-native-v1/corpus/eval.sft.jsonl \
  --retention-eval-jsonl runs/knowledge/tool-native-v1/dojo/eval.sft.jsonl \
  --retention-eval-jsonl runs/knowledge/tool-native-v1/tools/eval.sft.jsonl
```

If interrupted, repeat the exact command with `--resume`. The supervisor
selects only an immutable direct-child checkpoint, while the Rust trainer
revalidates its model/Adam hashes and exact run descriptor. Its terminal
success state is `candidate_ready_for_offline_gate`, never an active model.

### 7c. Audit, certify and activate

Build the provider-free binaries before entering the network-denied Linux
namespace. The evaluator loads the candidate through the same runtime loader as
serving, transcribes all 1,050 cases, compares text/tool behavior against the
exact certified base and replays a deterministic 30-case slice twice:

```bash
cargo build --release -p grug-train --features cuda \
  --example eval_native_ocr --example certify_multimodal_model

unshare --net --mount-proc \
  target/release/examples/eval_native_ocr \
  --candidate-bundle runs/knowledge/model-small-native-ocr \
  --base-text-bundle runs/knowledge/model-small \
  --audit-manifest runs/vision-audit-noto-v1-20260904/manifest.jsonl \
  --retention-eval-jsonl runs/knowledge/tool-native-v1/corpus/eval.sft.jsonl \
  --retention-eval-jsonl runs/knowledge/tool-native-v1/dojo/eval.sft.jsonl \
  --retention-eval-jsonl runs/knowledge/tool-native-v1/tools/eval.sft.jsonl \
  --output-dir runs/knowledge/model-small-native-ocr/evidence/release-1 \
  --network-replay-cases 30 --require-network-denied

target/release/examples/certify_multimodal_model \
  --candidate-bundle runs/knowledge/model-small-native-ocr \
  --audit-evidence runs/knowledge/model-small-native-ocr/evidence/release-1/multimodal-audit-evidence.json \
  --audit-manifest runs/vision-audit-noto-v1-20260904/manifest.jsonl \
  --audit-predictions runs/knowledge/model-small-native-ocr/evidence/release-1/audit-predictions.jsonl \
  --retention-evidence runs/knowledge/model-small-native-ocr/evidence/release-1/multimodal-text-retention-evidence.json \
  --retention-eval-jsonl runs/knowledge/tool-native-v1/corpus/eval.sft.jsonl \
  --retention-eval-jsonl runs/knowledge/tool-native-v1/dojo/eval.sft.jsonl \
  --retention-eval-jsonl runs/knowledge/tool-native-v1/tools/eval.sft.jsonl \
  --retention-before-outputs runs/knowledge/model-small-native-ocr/evidence/release-1/retention-before-outputs.jsonl \
  --retention-after-outputs runs/knowledge/model-small-native-ocr/evidence/release-1/retention-after-outputs.jsonl \
  --network-replay-evidence runs/knowledge/model-small-native-ocr/evidence/release-1/multimodal-network-denied-replay.json \
  --network-replay-first runs/knowledge/model-small-native-ocr/evidence/release-1/network-replay-first.jsonl \
  --network-replay-second runs/knowledge/model-small-native-ocr/evidence/release-1/network-replay-second.jsonl
```

When the GPU host cannot create a network namespace, build the pinned CPU
release image while networking is available, stage the candidate, certified
text base, audit corpus and three retention files beneath one local directory,
then run the whole evidence-and-certification gate with Docker networking
disabled:

```bash
docker build \
  -f training/docker/native-ocr-offline.Dockerfile \
  -t akh-medu/native-ocr-offline:phase51-v1 .

offline_root="$(cd runs/native-ocr-offline-v1 && pwd)"
docker run --rm \
  --network none --cap-drop ALL \
  --security-opt no-new-privileges --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --user "$(id -u):$(id -g)" \
  -v "$offline_root:/work" \
  akh-medu/native-ocr-offline:phase51-v1 \
  --candidate-bundle /work/model-small-native-ocr \
  --base-text-bundle /work/model-small-text \
  --audit-manifest /work/vision-audit/manifest.jsonl \
  --retention-eval-jsonl /work/retention/knowledge.eval.sft.jsonl \
  --retention-eval-jsonl /work/retention/dojo.eval.sft.jsonl \
  --retention-eval-jsonl /work/retention/tools.eval.sft.jsonl \
  --out /work/offline-gate-v1
```

The entrypoint always adds the exact evaluator and certifier binaries built
into the image. It always passes `--require-network-denied`; a normal bridge,
host or provider network cannot produce promotion evidence. Partial evaluation
directories are preserved as numbered attempts, and `--resume` reuses only an
attempt carrying a complete checkpoint-bound evidence marker. On a validated
Colima/Docker `--network none` run the container exposed exactly `lo` and zero
rows in `/proc/net/route`.

`run_native_ocr_cross_host_gate.py` automates the provider-to-local boundary
without transferring an entire training history. It waits for the remote
candidate-only status, pins SSH host keys and the local image ID, transfers the
two manifests plus only their referenced tokenizer, contracts and checkpoint
directories, then invokes the same Docker command. Remote paths are restricted
to descendants of the configured run root and retention file order is
preserved because it is part of the candidate corpus fingerprint. Use
`--resume` after a host interruption; completed staging and partial offline
attempts are revalidated rather than overwritten.

Pair a paid GPU run with `stop_runpod_after_pipeline.py`. The guard pins its
enrolled `~/.local/bin/agent-tools` wrapper and `runpodctl`, holds only the
`runpod` cache profile name while waiting, and obtains the actual key just in
time from OpenBao-backed delivery. Override the wrapper with `--agent-tools`
when its installed path differs. There is no 1Password fallback. Legacy guard
launches using `--credential-ref`/`--op-wrapper` must be recreated with the new
arguments; schema-v1 launches cannot resume under schema v2. It stops (never deletes) the
pod after the candidate is staged locally, after a terminal pipeline failure,
or when the cost deadline expires. This prevents a failed watcher from leaving
unbounded paid compute behind while preserving the stopped pod and its volume
for diagnosis.

Promotion requires at least 80% overall exact OCR, at most 8% CER, at most 1%
cross-semantic collapse, at least 80% metamorphic consistency, at least 70%
exact in every required language/layout/corruption slice, and no more than two
percentage points of text/tool action-exact regression. The replay must expose
only loopback, have zero kernel routes and reproduce all 30 outputs byte for
byte. The certifier recomputes every score and file hash; only then does it
write `multimodal-active.json`, last, without an optimizer-state dependency.

At runtime, an active multimodal pointer takes precedence over the text-only
pointer and any invalid multimodal pointer fails closed. ACP accepts at most
four PNG images (24 MiB each, 48 MiB total), normalizes common RGB/RGBA/gray
screenshot encodings in memory, performs OCR locally once per turn and labels
the transcript as untrusted evidence before the tool loop. The evidence is
retained with that session's operator turn so follow-up questions work without
rerunning a provider.

### 7d. Install the certified serving bundle

Build the probe that uses the same `GrugChatBackend::load` path as `akhomed`,
then place the cross-host gate behind the installer:

```bash
cargo build --locked --release --example grug_bundle_probe \
  --features grug-backend

python3 training/scripts/install_certified_babyllm.py \
  --cross-host-run runs/native-ocr-cross-host-runpod-v2 \
  --models-dir ~/.local/share/akh-medu/models \
  --model-name knowledge/model-small \
  --probe-binary target/release/examples/grug_bundle_probe \
  --out runs/certified-babyllm-install-v1
```

The installer waits for `active_multimodal_model_ready`; candidates and failed
gates cannot reach the models directory. It copies only the active manifest,
tokenizer, contracts, model record, checkpoint metadata and promotion report.
In particular, `optim.mpk` and the candidate pointer are not serving payloads.
Both the temporary copy and immutable release are deserialized by the real
backend before an atomic `knowledge/model-small` symlink update. The installer
does not modify any workspace config. To opt in after installation:

```toml
backend = "grug"
model = "knowledge/model-small"
```

Build `akhomed` with `--features server,grug-backend`; an invalid or missing
active bundle is a startup error and never falls through to a provider model.

### 7e. Review and publish the serving-only release

Start the publication watcher behind the installer, using only a 1Password
reference while it waits:

```bash
python3 training/scripts/publish_certified_babyllm.py \
  --install-run runs/certified-babyllm-install-v1 \
  --repo-id OpenFlowLabs/grug-rule-350m \
  --release-root releases/tool-native-small \
  --credential-ref 'op://Homelab/Hugging Face Access Token/credential' \
  --op-wrapper ~/.local/bin/op-service-account \
  --uv /opt/homebrew/Cellar/uv/0.12.8/bin/uv \
  --upload-helper runs/babyllm-publication-tooling-v1/hf_commit_certified_babyllm.py \
  --out runs/certified-babyllm-publication-v1
```

After installation, the watcher creates an immutable `review-packet/` and
`review-request.json`, then waits. Each exact reviewer writes its declared
JSON record under `reviews/`, with direct-child prompt and raw-response files.
The prompt and response hashes, Hugging Face parent, operation count and packet
SHA-256 must match; the response must repeat the packet hash and contain
`VERDICT: PASS` or `VERDICT: BLOCK`. The only accepted model IDs are
`z-ai/glm-5.3` and `moonshotai/kimi-k3`. A BLOCK terminates without an upload.

The reviewed upload contains the root model card plus a unique
`releases/tool-native-small/model-small-step-*` serving directory. The plan
must cover every payload file and forbids `optim.mpk`,
`multimodal-candidate.json`, audit predictions, text-retention outputs and
network-replay streams. If the public repository head changes after packet
creation, prepare and review a new packet; do not silently rebase it.

Only after both exact PASS records does 1Password inject `HF_TOKEN` into an
offline-pinned `huggingface_hub` helper. The token is neither placed in an
argument nor retained by the watcher. Success requires an anonymous download
and byte-for-byte SHA-256 verification of every operation at the returned
commit. Resolve Homebrew symlinks before launch and run the long-lived watcher
from a read-only tooling copy; publication pins reject mutable symlink paths.

## 8. Stop handling

On a `stop` decision, retain at least:

- model checkpoint and optimizer state;
- controller state before and after the decision;
- exam plan, manifests, predictions and slice report;
- generator/rasterizer version and environment;
- training logs and the triggering decision.

Do not silently clear a stop. A resume records its reason and begins a new run
segment so pre/post-intervention evidence cannot be confused.
