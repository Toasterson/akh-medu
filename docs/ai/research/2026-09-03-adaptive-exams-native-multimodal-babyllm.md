# Adaptive exams and native text vision for BabyLLM

**Date:** 2026-09-03  
**Status:** implementation basis for ADR 100 and the Phase 51 adaptive-exam plan

## Question

How can BabyLLM detect a failing training run early, keep evaluation useful as
the model improves, and add text recognition as a native modality without
turning a small local model into a wrapper around unreliable provider APIs?

The answer is a two-part control system:

1. replace the single frozen-exam idea with a layered, reproducible exam loop;
2. make pixels another input to the BabyLLM trunk, beginning with synthetic
   text images whose labels and difficulty are fully owned.

The fixed holdout is not deleted. It becomes the small audit instrument rather
than the only source of development signal.

## Evidence

### Evaluation must be both stable and adaptive

[CheckList](https://aclanthology.org/2020.acl-main.442/) treats evaluation as a
matrix of capabilities and behavioral test types rather than a single held-out
accuracy. Its templated and generated cases expose failures hidden by aggregate
scores. BabyLLM should therefore report a capability vector and failure modes,
not just one pass rate.

[Dynabench](https://arxiv.org/abs/2104.14337) closes the loop between model
evaluation and data collection: model-in-the-loop failures become new,
increasingly difficult data. [AutoBencher](https://arxiv.org/abs/2407.08351)
similarly searches for novel, salient and difficult evaluation topics instead
of repeatedly sampling an exhausted benchmark. These support a moving frontier
lane, but not silently rewriting the audit set.

[Automated Curriculum Learning for Neural Networks](https://proceedings.mlr.press/v70/graves17a/graves17a.pdf)
uses learning progress as the teaching signal. [Teacher Algorithms for
Curriculum Learning of Deep RL](https://proceedings.mlr.press/v100/portelas20a.html)
shows why absolute learning progress is useful: it prioritizes a learnable
frontier while moving attention away from both mastered and currently
unlearnable regions. This is a better controller input than raw loss alone.

[ATLAS](https://arxiv.org/abs/2511.04689) applies item-response theory and
Fisher-information selection to adaptive language-model testing, reporting that
similar rankings can be recovered with far fewer items. BabyLLM does not need
to adopt its exact estimator yet, but should log item difficulty and select
near-boundary cases so that an eventual IRT controller has the required data.

The result is four evaluation lanes:

| Lane | Changes? | Purpose | May exact cases enter training? |
|---|---:|---|---:|
| sentinel | no | comparable release and regression evidence | never |
| rolling probe | every checkpoint, deterministic | dense development telemetry | never |
| frontier | adaptively, deterministic from controller state | find the current learning boundary | never |
| audit/red-team | periodic, different renderer plus real/human-sourced data | detect generator blind spots | never |

An exam failure may create a **sibling training family** under a new seed. The
exact exam input, answer, rendered image, seed and corruption parameters remain
quarantined. That distinction is essential: otherwise an adaptive exam becomes
a delayed training set and stops measuring generalization.

A different seed is not sufficient separation. Before sibling material enters
training, the pipeline must prove at least one held-out generator axis and
reject excessive transcript n-gram similarity or image perceptual-hash
proximity. Audit data must be generator-independent; another seed from the same
renderer is only a rolling probe.

### Steering requires precursor signals

A controller that waits for end-to-end accuracy to collapse is late. Each
capability should emit both hard outcome and predictive precursor measurements:

- invalid/blank/non-finite output as a hard outcome;
- component correctness before whole-chain correctness;
- calibrated confidence or margin where available;
- metamorphic consistency across equivalent renderings or paraphrases;
- character and word edit distance before exact-match reaches zero;
- per-language, corruption, layout and reasoning-depth slopes;
- learning progress over checkpoints, not only current score.

The first controller uses EWMA trend, target-band position and precursor rate,
but only after a per-family sample floor. Pass/precursor decisions use Wilson
intervals, require consecutive significant windows, and apply false-discovery
control when testing a family matrix. CER/WER reports use deterministic
case-resampling intervals. The sentinel difficulty is pinned at family
registration and never follows the mutable frontier. Precursor steering,
including precursor-driven family priority, remains disabled for expensive
runs until retrospective checkpoint evidence shows useful lead time. A
separate confidence-qualified catastrophic count can still stop on hard
blank/invalid outcomes; shadowing predictive signals must not disable an
outcome safety gate. Later, item-response parameters can replace the heuristic
without changing the stored observation schema.

### Native vision can be small

Google's official [Gemma 4 12B announcement](https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12b/)
and [developer guide](https://developers.googleblog.com/gemma-4-12b-the-developer-guide/)
describe an encoder-free multimodal model: raw 48x48 image patches are linearly
projected, augmented with factorized X/Y position embeddings and passed to the
same decoder-only transformer as text. The published vision input path is about
35M parameters rather than a second large encoder. This is the correct
architectural reference for a native BabyLLM vision organ.

BabyLLM should copy the principle, not blindly freeze the 48x48 patch size.
Small document text can disappear inside coarse patches, and fine patches can
overflow a 1K context before a full page fits. Token-budget arithmetic and a
projection/readout resolution probe therefore precede language-model training.
Patch size, tiling and resolution are experimental variables, while the durable contract is
`pixels -> projected visual prefix -> shared decoder`.

If no encoder-free configuration represents clean small type at the 10–30M
rung, the next decision is explicit: defer native vision or introduce a compact
convolutional/resampling stem. Thresholds are not silently relaxed.

The OCR literature adds useful bounds:

- [Donut](https://arxiv.org/abs/2111.15664) demonstrates end-to-end,
  OCR-free document understanding and includes a synthetic document generator;
  it avoids error propagation from a separately frozen OCR system.
- [GOT-OCR2.0](https://arxiv.org/abs/2409.01704) shows that a 580M unified
  recognizer can cover plain text, formulae, tables and charts, supporting a
  staged rather than text-only definition of OCR.
- [PaddleOCR-VL](https://arxiv.org/abs/2510.14528) reports a compact 0.9B
  multilingual document model covering 109 languages. Its scale is above the
  initial BabyLLM target, but it is evidence that multilingual document parsing
  does not require frontier-scale models.
- [Florence-2](https://arxiv.org/abs/2311.06242) uses a prompt-driven shared
  representation across vision tasks and an iterative data-engineering loop.
- [SynthText](https://arxiv.org/abs/1604.06646) establishes scalable rendered
  text-in-scene supervision, while [OCRBench v2](https://arxiv.org/abs/2501.00321)
  shows why evaluation must include uncommon characters, fine-grained text,
  layout, complex parsing and reasoning rather than clean word crops alone.

Synthetic text is unusually suitable for continuous exams: the transcript,
reading order, layout, language, glyph font, geometry and corruption are all
known exactly. The generator can increase one factor at a time and distinguish
an unreadable item from a model regression.

The measured stem fallback also has a broader basis. [Early Convolutions Help
Transformers See Better](https://arxiv.org/abs/2106.14881) found that replacing
a single coarse patchification operation with a short stack of local
convolutions materially improved transformer optimization stability. However,
ordinary strided convolutions and pooling are not automatically translation
invariant: [Making Convolutional Networks Shift-Invariant
Again](https://proceedings.mlr.press/v97/zhang19a.html) identifies aliasing at
downsampling as the cause and evaluates low-pass filtering before subsampling.
For BabyLLM this means held-out translations remain a required empirical gate.
The bootstrap compact stem uses full-resolution local processing, pooling and
deterministic translation augmentation; an anti-aliased downsampler remains a
versioned comparison, not an assumed improvement.

## Proposed BabyLLM design

### Exam controller

Each exam item belongs to a stable capability family such as:

- `reason.chain.depth`;
- `tool.schema.selection`;
- `vision.text.line`;
- `vision.text.layout`;
- `vision.text.api-error`;
- `vision.grounded-action`.

An observation records checkpoint identity, tier, generator version, family,
difficulty, seed, sample count, pass rate and precursor rate. The controller:

1. updates a per-family EWMA and learning-progress delta;
2. raises `watch`, `steer` or `stop` warnings before aggregate collapse;
3. moves difficulty toward the target success band;
4. prioritizes families with regression, precursor failures or useful learning
   progress;
5. emits the complete next-exam plan so it is reproducible.

`stop` is reserved for audit-sentinel regression or catastrophic validity
failure. `steer` changes the next sibling-training mixture. Neither state is a
license to train on exam cases.

### Native text-vision stem

The first implementation is deliberately narrow:

```text
RGB image
  -> deterministic patchification
  -> linear patch projection
  + factorized row/column embeddings
  -> LayerNorm
  -> visual prefix embeddings
  -> existing BabyLLM decoder trunk
  -> existing text vocabulary / HG surface
```

This preserves the Dojo rule that the tokenizer and High-Grug surface are the
inter-organ ABI. The image does not need fake token IDs, so existing text
checkpoints and tokenizer IDs are not renumbered. Future checkpoint formats
must reserve modality-boundary metadata before trained visual checkpoints are
published.

Engine-side routing remains responsible for deciding when an image is present
and which organ/checkpoint is active. A provider may be used as a labelled
oracle, disagreement reviewer or optional escalation, never as the only OCR
path and never as a source of untraceable labels.

### Corpus ladder

1. glyphs, digits, punctuation and short strings;
2. Latin-script lines in English, German and French, with exact Unicode normalization;
3. paragraphs with font, size, contrast, blur, noise and occlusion variation;
4. receipts, forms, tables, code snippets, terminal output and broken API
   responses;
5. screenshots and scene text with controlled backgrounds and reading order;
6. image-to-HG facts, tool arguments and grounded actions;
7. mixed image/text dialogues and document reasoning.

Text-only examples remain in every batch, with modality dropout, so adding
vision cannot silently erase language and reasoning ability.

This first requirement is deliberately **Latin-script**, not a claim of broad
multilingual OCR. A generator-independent audit includes at least Arabic and
CJK probes before a model card may broaden that claim. Those scripts require
shaping, directionality, segmentation and font coverage that EN/DE/FR do not
exercise.

### Metrics and gates

Report at least:

- character error rate (CER), word error rate (WER), exact transcript match,
  sample counts and confidence intervals;
- reading-order and structured-field accuracy;
- grounded tool/action correctness;
- blank/invalid rate and near-miss rate;
- metamorphic consistency for semantically identical renderings;
- slices by language, font, layout, corruption type/severity and token length;
- per-promotion and cumulative text-only sentinel delta after multimodal updates;
- latency, visual-token count and memory by input resolution.

No single weighted aggregate is a release gate. The capability matrix and its
worst supported slice are first-class output.

For reproducibility, the bootstrap renderer is a compiled, version-pinned
bitmap font rather than an operating-system font. Production font packs must be
license-cleared, content-hashed artifacts. Generator/item plans and CPU-rendered
pixels are bit-reproducible; GPU optimization is only statistically
reproducible and records Burn, driver and accelerator versions.

## Execution addendum: the small-model stem result

The raw linear-patch baseline failed every held-out translation cell despite
memorizing observed positions. Failure-directed refinement selected a compact
full-resolution residual pre-block plus four local-convolution/max-pooling
stages, with content normalization before factorized positions. Under a dense
translation curriculum with two explicit holes, this stem reached 100% on both
8px and 16px held-out offsets on an A40.

The resulting 3.31M shared-decoder model then passed its deliberately narrow
128-case clean-line overfit gate: 128/128 exact greedy transcripts with zero
CER/WER/blank/collapse. A CUDA checkpoint loaded into the local ndarray backend
and produced 3/3 exact transcripts under an OS network-deny sandbox. These
results support the architectural seam and compact-stem choice; they do not
answer the paper-level generalization questions. Different-renderer/real
images, unseen strings/fonts, EN/DE/FR, corruptions/layouts, metamorphic groups
and text retention remain the next required evidence.

## Rejected shortcuts

- **Only keep a frozen exam.** Comparable, but increasingly sparse and easy to
  overfit organizationally.
- **Continuously mutate the sole benchmark.** Responsive, but destroys
  longitudinal evidence and invites leakage.
- **Train on failed exam cases.** Fast apparent improvement, invalid
  measurement.
- **Put Tesseract or a cloud OCR call in front and call the model multimodal.**
  Useful as a fallback baseline, but not native perception and it inherits
  provider/API failure modes.
- **Add a large frozen vision encoder first.** It may eventually win on
  accuracy, but obscures the smallest viable shared-decoder experiment and
  makes deployment heavier before the OCR objective is proven.
- **Use one OCR score.** Hides the distinction between recognition, layout,
  reasoning and action failures.

## Review questions

The independent GLM 5.3 and Kimi K3 reviews should challenge:

1. whether the quarantine rule is sufficient to prevent adaptive leakage;
2. whether precursor warnings can actually lead end-to-end failure;
3. whether encoder-free patches retain enough fine text at BabyLLM scale;
4. which multilingual/layout stages must precede grounded reasoning;
5. which stop gates prevent multimodal gains from masking text regression.
