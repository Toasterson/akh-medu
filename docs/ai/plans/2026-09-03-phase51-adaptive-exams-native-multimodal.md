# Phase 51: adaptive exams and native multimodal BabyLLM

**Date:** 2026-09-03  
**Decision:** [ADR 100](../decisions/100-adaptive-exams-native-text-vision.md)  
**Research:** [adaptive exams and native text vision](../research/2026-09-03-adaptive-exams-native-multimodal-babyllm.md)

## Outcome

Build a continuous evaluation-and-steering loop and a native text-recognition
path for the modular BabyLLM. The delivered system must still work locally
when every external provider is unavailable.

```text
checkpoint
   |                         immutable
   +--> sentinel ----------------+----> release gate
   |
   +--> rolling synthetic probes +----> capability time series
   |                                    |
   +--> adaptive frontier --------------+--> warning/controller
                                             |
                                             +--> sibling training mixture
                                                  (new seed; exam quarantined)

image -> patches -> tiny visual stem -> shared decoder -> text/HG/action

challenge -> inspect notes/textbooks -> retrieve quoted evidence
          -> append immutable note revision -> cited answer
```

## Non-negotiable invariants

- Exact or near-duplicate exam instances never train.
- Seeds are domain-separated between exam and training siblings.
- A frozen sentinel remains for longitudinal comparison.
- Generated plans include generator and checkpoint identity.
- Required slices gate independently; an average cannot hide a zero.
- Text-only capability is measured throughout multimodal training.
- OCR inference is local and native; providers are optional oracles only.
- Audit data uses a different generator/renderer or real human-verified input.
- Existing tokenizer IDs are not changed by the first vision slice.
- Engine-side routing remains authoritative.
- A durable note write requires source ID, locator and quote from one
  successful retrieval result observed in an earlier model round.
- Note discovery is metadata-only; a body enters context only through an
  explicit stable-ID read or bounded archival retrieval.
- A captured training run remains immutable when the runtime tool contract
  grows; new tools enter through a versioned adaptation and full recertification.

## Work packages

### K0 — Akh-native textbook and note lifecycle

Delivered in source-head v2:

- [x] exact native/MCP tool-contract capture and append-only runtime drift
  journaling;
- [x] Seshat document inventory, passage search, provenance-bearing ingest and
  explicit embedding transitions in the learned workflow;
- [x] bounded `study_note_list` and `study_note_read` contracts that separate
  body-free filtered pagination from a selected stable note ID and head token;
- [x] separate create/revise contracts with fully expressible required fields,
  optimistic concurrency, idempotent retry and hard-case conflict recovery;
- [x] append-only note revisions whose supplemental frontmatter merges with
  retained challenge metadata;
- [x] mandatory supporting quotes plus a host-side earlier-round evidence gate
  before any `study_note_append` mutation;
- [x] strategy-independent `archival_search` for the default small-model path,
  without granting core-memory or graph mutation privileges;
- [x] seven separately gated EN/DE/FR workflows: acquisition, restart recall,
  revision, provider fallback, grounded abstention, textbook inventory and note
  navigation.

Acceptance:

- the exact source-head contract exports 53 tools and passes the learning-loop
  contract check;
- a generated v2 corpus covers all seven families, all three languages, all 14
  stages and all twelve learning-loop tools;
- textbook inventory lists editions before passage retrieval;
- note navigation filters metadata before a separate identity read;
- generic all-tool note-write examples contain matching prior retrieval
  evidence, revision examples contain the observed head, and every example
  retains the real four-tool decision budget;
- missing, trivial, invented or same-round citation evidence cannot mutate the
  durable note store;
- a stale note revision cannot advance the head, and the hard revision exam
  requires a fresh read before retrying;
- the frozen paid v1 run remains on its original 50-tool/five-family bytes, and
  v2 is admitted only as a later candidate through contract-delta adaptation.

### E0 — exam contract and early-warning controller

Delivered:

- [x] `grug-data::adaptive_exam` with serializable plans, observations and state;
- [x] deterministic, domain-separated plan seeds;
- [x] per-family sentinel difficulty pinned independently of frontier state;
- [x] `sentinel`, `rolling_probe`, `frontier`, and `audit` tiers;
- [x] Wilson intervals, EWMA/progress/precursor warnings and repeated windows;
- [x] hard catastrophic outcome counts separated from shadow-only predictive
  precursors, with under-sampled windows excluded from trend state;
- [x] Benjamini-Hochberg helper for family-matrix false-discovery control;
- [x] target-band difficulty selection and family priority;
- [x] refusal to derive training material from sentinel, rolling or audit plans;
- [x] held-out-axis, transcript n-gram and perceptual-hash sibling contract;
- [x] replay/leakage tests and a state/observation/plan CLI boundary.
- [x] bounded continuous supervisor with immutable sentinels, targeted
  frontiers, five-percent sibling admission, candidate/Adam continuation and
  fail-closed stop semantics; it cannot promote a model.

Acceptance:

- identical controller state produces identical next plans;
- a rising precursor rate emits `steer` before pass rate reaches zero;
- repeated catastrophic blank/invalid outcomes emit `stop` even while
  predictive precursor steering is disabled;
- a sentinel regression emits `stop`;
- a sentinel case-set seed/fingerprint remains identical across checkpoints
  and after intervening frontier-difficulty changes;
- a mastered family gets harder and a collapsed family gets easier;
- a training sibling never shares the exam seed.
- a sibling without held-out-axis and distance evidence is rejected.

### E1 — deterministic synthetic text-vision exams

Delivered:

- [x] a version-pinned compiled-bitmap, seeded RGB renderer;
- [x] separate Latin-script English, German and French exam/training banks;
- [x] line, paragraph, receipt, terminal and broken-API layouts;
- [x] controlled contrast, noise, blur, occlusion and scale difficulty;
- [x] manifest with transcript, Unicode normalization, glyph digest, geometry,
  corruption parameters, semantic sibling ID and generator version;
- [x] PNG output and JSONL prediction scorer;
- [x] CER, WER, exact, blank, near-miss, exact-render metamorphic consistency,
  per-slice summaries, Wilson intervals and deterministic case-bootstrap intervals.

Acceptance:

- the same seed/renderer/version produces identical pixels and manifest;
- generated files contain no blank images for supported glyphs;
- perfect transcripts score CER/WER 0 and exact 1;
- a one-character deletion produces a nonzero, bounded CER;
- train and exam commands use distinct seed domains;
- missing glyphs are recorded and fail the affected case rather than silently
  changing its transcript.

### V0 — native patch stem vertical slice

Delivered:

- [x] configurable RGB patchification with explicit padding metadata;
- [x] a Burn visual stem with one patch projection, factorized row/column
  embeddings and LayerNorm;
- [x] a `TinyTransformer` entry point accepting continuous prefix embeddings;
- [x] a native vision wrapper whose output length is `patches + text tokens`;
- [x] text-position logits and an exported teacher-forced optimization step;
- [x] gradient and shape tests proving the visual stem is on the training path;
- [x] a preflight visual-token budget command.

Acceptance:

- two images produce different projected prefixes;
- gradients reach the patch projection;
- causal text tokens can attend to all preceding image patches;
- existing text-only forward tests remain unchanged;
- overlong image-plus-text sequences fail clearly.

### V1 — OCR bootstrap experiment

Train small rungs before spending a 350M run:

| Rung | Purpose | Promotion gate |
|---|---|---|
| 2–10M | prove render, tokenize, loss and decoding path | overfit 128 clean cases; exact >= 0.98 |
| 10–30M | test unseen strings/fonts and three languages | CER <= 0.15 clean, <= 0.30 corrupted |
| 50–100M | add layout and API/screenshot domains | required slices nonzero and improving |
| 350M | joint language/reasoning/vision trunk | all gates below plus text retention |

Before the 2–10M rung, run a stem-only glyph-resolution probe and compute token
budgets at patch sizes 8/16/24/32/48. Do not assume the Gemma 4 reference patch
size is appropriate for small type. If every <=16 patch configuration exceeds
CER 0.05 on clean 6–10pt lines, or a receipt consumes more than 25% of context,
choose tiling/resampling or a compact convolutional stem in a follow-up ADR;
do not relax the gate.

Execution status (2026-09-04):

- [x] token-budget preflight: 16px lines/receipts fit; screenshots/pages require
  tiling or a compact front-end;
- [x] raw-patch stem/readout matrix: all held-out-position gates failed;
- [x] focused 500-step check: 100% memorized-position accuracy, 15.6% held-out;
- [x] [ADR 101](../decisions/101-compact-conv-vision-stem-experiment.md):
  compact stride-16 convolutional candidate implemented on the same decoder ABI;
- [x] CPU conv calibration: 23.7 seconds/training step, so accelerated validation
  is required;
- [x] accelerated stem gate: the dense two-hole translation curriculum reached
  100% held-out accuracy at both 8px and 16px on an A40;
- [x] 3.31M shared-decoder gate: cosine training reached 128/128 exact greedy
  transcripts with zero CER/WER/collapse on the declared clean training corpus;
- [x] local offline replay: a network-denied CPU run loaded the CUDA checkpoint
  and reproduced 3/3 exact outputs at 4.09 seconds/case.
- [x] full native-OCR trainer checkpoints model plus Adam state atomically and
  resumes only after exact run/input/hash/counter verification.

These results clear only the representation and clean-line overfit rungs. They
select the compact path for held-out experiments; they do not clear the
different-renderer, unseen-string/font, multilingual, corruption/layout,
metamorphic or text-retention gates. The plan still forbids advancing to a
larger decoder while those prerequisites are red or unrun.

### V2 — joint corpus and retention

Mixture buckets:

- 35% existing language/understanding and multilingual text;
- 25% High-Grug/reasoning/tool traces;
- 20% clean synthetic text vision;
- 10% corrupted/layout text vision;
- 5% image-grounded HG/tool actions;
- 5% replay buffer chosen from recent regression families.

These are starting priors, not fixed doctrine. The controller can move at most
5 percentage points per evaluation window and cannot reduce text-only below
30% without a new decision.

Every visual batch uses modality dropout on a configurable fraction so the
same textual answer path remains usable without pixels.

### V3 — layout and visual-token efficiency

After V1 succeeds:

- compare fixed patches with tiled/dynamic-resolution inputs;
- add coarse-to-fine region selection only if full-page token cost is the
  measured bottleneck;
- represent reading order and regions in High-Grug rather than inventing a
  second hidden protocol;
- add tables, forms, code, terminal logs and UI screenshots;
- measure latency/memory against crop count and patch count.

### V4 — grounded understanding

Move beyond transcription:

- image -> transcript + spans;
- spans -> entities/relations in the existing KG boundary;
- image -> structured fields;
- image -> tool arguments with evidence coordinates;
- mixed text/image clarification when confidence is low.

The engine must retain evidence and confidence so a perception error is not
misreported as a reasoning or tool-selection failure.

### V5 — Dojo publication and continuous operation

Publish versioned, provenance-complete artifacts to the public OpenFlowLabs
repositories:

- immutable sentinel manifest and hashes;
- generator source/version and content licence metadata;
- rolling/frontier plan schemas, not private answers from the sentinel;
- model card capability matrix and known failure slices;
- checkpoint/config/tokenizer/visual-stem compatibility tuple;
- replay command and environment summary.

The public repositories verified for this phase are:

- `OpenFlowLabs/babyllm-dojo-kata` (dataset);
- `OpenFlowLabs/grug-rule-350m` (model).

Publication is a separate action from local validation. A local diff, test or
checkpoint is not evidence that Hugging Face contains it.

Publication status (2026-09-04):

- [x] reviewed dataset/controller payload published and anonymously verified at
  [`54bd84b`](https://huggingface.co/datasets/OpenFlowLabs/babyllm-dojo-kata/commit/54bd84b63c8a10b577edcaa308836391e376f0a3);
- [x] experimental checkpoint/model+Adam payload published and anonymously
  verified at
  [`3908c23`](https://huggingface.co/OpenFlowLabs/grug-rule-350m/commit/3908c23bd3a2c9e33a26bb079989c9501017e187);
- [x] existing checkpoint/tokenizer checksums re-read unchanged;
- [x] public cards explicitly retain the red OCR gate and text-only checkpoint
  boundary;
- [x] the narrow V1 training-overfit gate passed and its experimental
  multimodal checkpoint was published under the versioned research path;
- [ ] promotion beyond that research path remains blocked on unseen OCR,
  independent-renderer/real audit and text-retention gates.

Execution status (2026-09-04):

- [x] resumable text adaptive loop and separate immutable-sentinel release gate
  are implemented and running behind the foundation/SFT chain;
- [x] 1,050-case training sibling and 1,050-case independent Noto audit are
  content-pinned on the GPU volume;
- [x] candidate-only native OCR training supervisor is implemented and waiting
  for the certified text release;
- [x] provider-free offline evaluator/certifier image is built locally and its
  `--network none` environment exposes only loopback and zero kernel routes;
- [x] fail-closed serving installer verifies the certified source and immutable
  copy through the real `GrugChatBackend`, omits optimizer state and atomically
  advances `knowledge/model-small`; its watcher is running behind the gate;
- [x] a credential-late Runpod guard stops but never deletes the paid GPU after
  local staging, terminal pipeline failure or the bounded wait deadline;
- [x] a parent-pinned, serving-only Hugging Face publication watcher requires
  byte-bound PASS evidence from exact GLM 5.3 and Kimi K3, injects its token
  only for the commit, and verifies every resulting file anonymously;
- [ ] no text or multimodal candidate is called active until its downstream
  certifier actually writes the corresponding append-only active pointer.

The consolidated implementation and live-execution boundary is recorded in
[the tool-native full-model pipeline result](../../../training/results/tool-native-full-model-pipeline-20260904-RESULT.md).

Exact upload and byte-verification evidence is recorded in
[adaptive-exams-v1-publication-RESULT.md](../../../training/results/adaptive-exams-v1-publication-RESULT.md).

The local commands and artifact flow are documented in the
[operator runbook](../manuals/2026-09-04-adaptive-exams-native-vision.md).

## Continuous checkpoint loop

At each evaluation window:

1. score the immutable sentinel without changing it;
2. derive a rolling plan from run/checkpoint/generator identity;
3. ask the controller for the highest-information frontier families;
4. generate and score cases; retain item-level outcomes;
5. emit warning level and a bounded mixture recommendation;
6. if `steer`, generate sibling training families with a new seed domain;
7. if `stop`, retain artifacts and hold promotion; do not silently resume;
8. periodically run a different-renderer and real/human-verified audit set to
   find generator blind spots; a new seed from the bootstrap renderer does not count.

Controller windows require at least 200 cases per family. `steer` and `stop`
require two consecutive confidence-qualified windows. In the first two real
runs, precursors are shadow signals only; they become active steering inputs
only after a retrospective lead-time table shows predictive value. A `stop`
retains model/optimizer/controller state and starts a new run segment only with
an operator-authored resume reason.

## Promotion gates

A checkpoint advances only when all required gates pass:

| Gate | Initial threshold |
|---|---:|
| sentinel regression | 95% intervals within 3 points of original baseline |
| invalid/blank output | <= 1% clean; <= 5% corrupted |
| English clean line CER | <= 0.10 |
| German/French clean line CER | <= 0.15 each |
| corrupted line CER | <= 0.30 per required severity |
| exact structured fields | >= 0.80 |
| metamorphic consistency | exact normalized equality across >=3 renders, >= 0.90 |
| unseen-string generalization | confidence-qualified improvement for 3 windows |
| text-only regression after vision update | <= 3 points/promotion and <= 5 cumulative |
| local offline inference | required |

Thresholds are versioned with the exam policy. Changing a threshold does not
rewrite old results. Every required promotion slice has at least 500 cases;
point estimates without their intervals do not pass. EN/DE/FR are explicitly a
Latin-script scope. Arabic and CJK are generator-independent audit probes and
must not be advertised as supported until deliberately trained and gated.

## Pre-350M kill tests

1. **Controller noise floor:** stationary, slow-drift and step-regression
   simulations; false-steer <= 10% at the selected sample size.
2. **Precursor lead time:** replay saved checkpoints; precursor-driven steering
   stays shadow-only if median useful lead time is zero.
3. **Visual-token arithmetic:** reject any crop/patch policy that overflows the
   trunk or consumes >25% of context for the target document class.
4. **Stem resolution:** isolate projection/patch representation with a linear
   readout before blaming decoder capacity.
5. **Planted leakage canary:** deliberately train one near-sibling family; the
   independent audit must flag the synthetic/audit divergence.
6. **Modality interference:** equal-text-token text-only versus joint runs at
   dropout 0/0.1/0.3; stop if all exceed the text regression budget.
7. **Renderer replay:** pixel-identical compiled renderer in CI; separately
   record Burn, driver and accelerator versions for statistically reproducible GPU runs.

## Failure-directed steering

| Signal | Interpretation | Bounded response |
|---|---|---|
| CER improves, exact remains flat | near-miss recognition | increase sequence correction siblings |
| clean passes, blur/noise collapses | visual robustness | add adjacent corruption severities |
| transcript passes, fields fail | layout/grounding | emphasize region/order HG examples |
| one language regresses | mixture imbalance | add that language's sibling families |
| validity rises before chain score falls | impending reasoning collapse | restore shallow/valid-trace replay |
| all rungs flat with more capacity | corpus/representation problem | stop scale-up; inspect generator/stem |
| text sentinel falls after vision | catastrophic interference | stop; restore text mixture/modality dropout |

## Validation commands

The implementation slice is validated independently:

```bash
cargo test -p grug-data adaptive_exam
cargo test -p grug-data knowledge_workflow
cargo test -p grug-data vision_text
cargo test --lib study_note
cargo test -p grug-train vision
cargo test -p grug-train transformer
cargo check -p grug-data --examples
cargo check -p grug-train --examples
python3 -m unittest \
  training.scripts.tests.test_tool_native_adaptive_loop \
  training.scripts.tests.test_tool_native_release_gate \
  training.scripts.tests.test_native_ocr_pipeline
rustfmt --check --edition 2024 --config skip_children=true \
  crates/grug-data/src/{adaptive_exam,vision_text}.rs \
  crates/grug-data/examples/{adaptive_exam_controller,audit_vision_sibling,generate_vision_exam,score_vision_exam}.rs \
  crates/grug-train/src/vision.rs crates/grug-train/examples/vision_*.rs
```

The first end-to-end smoke creates a temporary corpus, validates its manifest,
scores an identity prediction file, and runs a few CPU optimization steps. GPU
training and Hugging Face publication require separate captured evidence;
neither is implied by these local commands.

The first native CPU run is captured in
[vision-v0-cpu-smoke-RESULT.md](../../../training/results/vision-v0-cpu-smoke-RESULT.md):
loss decreased strongly, while greedy output collapsed to one transcript. V0
wiring is green, but that checkpoint remains capability-red.

The stem experiment and the measured fallback branch are captured in
[vision-stem-resolution-RESULT.md](../../../training/results/vision-stem-resolution-RESULT.md)
and [ADR 101](../decisions/101-compact-conv-vision-stem-experiment.md). The
passing clean-line run, intervention ledger, checkpoint hashes and offline
replay are captured in
[vision-v1-a40-RESULT.md](../../../training/results/vision-v1-a40-RESULT.md).

## Review and closure

GLM 5.3 and Kimi K3 independently reviewed the same full implementation packet.
Kimi's final audit caught a mutable-frontier dependency in the sentinel plan;
publication stayed held until it was fixed and covered by an interleaving
regression test. Both exact models then returned `VERDICT: PASS` on the same
amendment packet. The full audit trail and dispositions are recorded in
[the review note](../reviews/2026-09-04-adaptive-exams-native-multimodal.md).
Those verdicts bind that byte-specific packet only. The later K0 source-head v2
catalog, evidence-gate and seven-family review boundary is recorded separately
in [the tool-native learning review note](../reviews/2026-09-04-tool-native-learning-v2.md).
Any verdict there binds only its recorded source hashes and generated
identities; it does not certify the running or eventual trained weights.
