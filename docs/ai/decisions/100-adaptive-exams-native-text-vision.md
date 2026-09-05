# ADR 100: Adaptive exams and native text vision for BabyLLM

- **Status:** accepted
- **Date:** 2026-09-03
- **Scope:** Phase 51 BabyLLM / Dojo evaluation and first multimodal organ
- **Research:** [Adaptive exams and native text vision](../research/2026-09-03-adaptive-exams-native-multimodal-babyllm.md)

## Context

Phase 51 has reproducible generated corpora and frozen holdouts, but a single
fixed exam is a lagging development instrument. It tells us that a completed
run failed; it does not continuously locate the model's current boundary or
surface precursor failures soon enough to change the training mixture.

The 350M reasoning run also showed that additional capacity alone did not
repair depth generalization. Evaluation must distinguish corpus, optimization,
representation and capacity failures while a run is still steerable.

At the same time, provider-side image/OCR support is an unreliable dependency
and is not native multimodality. Text recognition has exact synthetic labels,
clear metrics and immediate local uses, making it the first visual capability
for the modular model.

## Decision

### 1. Use four evaluation lanes

BabyLLM evaluation has four named tiers:

- `sentinel`: small, immutable, release-comparable holdout;
- `rolling_probe`: freshly generated for each checkpoint from a recorded seed;
- `frontier`: controller-selected capability and difficulty near the learning
  boundary;
- `audit`: periodic generator-independent, real or human-red-team material.

Every result is reported as a capability matrix. Aggregate scores are
informational and cannot conceal a failing required slice.

### 2. Make every moving exam reproducible

An exam plan records schema version, generator version, run and checkpoint
identity, tier, family, difficulty, seed and sample count. Controller state is
serialized with the run. Replaying that state must reproduce the same plan.
Each family's sentinel difficulty is pinned to its registered initial
difficulty. Its case-set seed and fingerprint exclude both checkpoint identity
and mutable frontier state, so neither scoring a later checkpoint nor adapting
the frontier can silently manufacture a new sentinel. Rolling, frontier and
generated audit plans remain checkpoint-specific.

### 3. Quarantine all exam instances

No exact exam prompt, image, answer, seed or corruption tuple may enter a
training set. A frontier failure can schedule a sibling training family only by
deriving a new, domain-separated training seed **and** demonstrating a held-out
generator axis. Admission rejects excessive transcript n-gram overlap and
perceptual-image-hash proximity. Sentinel, rolling-probe and audit cases can
never schedule training material automatically. A seeded same-renderer set is
not an audit.

### 4. Steer on precursors and progress

The first controller uses per-family EWMA score, learning-progress delta,
target-band position and precursor rate. A decision requires at least 200
items per family/window, Wilson intervals and repeated significant windows;
family-matrix hypothesis tests use false-discovery control. It emits:

- `green`: continue;
- `watch`: increase measurement density;
- `steer`: change sibling-family sampling before collapse;
- `stop`: hold the run because a sentinel or catastrophic-validity gate failed.

Predictive `precursor_count` stays shadow-only until lead time is established;
while disabled, it changes neither warnings nor family priority.
Hard `catastrophic_count` (blank/invalid/non-finite outcomes) is a separate
outcome signal and can stop independently; otherwise shadow mode would disable
the very validity stop this decision requires. Under-sampled windows do not
mutate trend, priority or difficulty state.

Item-response modelling may later replace the selection heuristic. The stored
schema must remain sufficient for that migration.

Precursor signals are logged initially but cannot steer an expensive run until
a retrospective checkpoint study establishes that they lead outcome failures
by a useful number of steps. The code-path test alone is not evidence of lead
time.

### 5. Add vision through the existing decoder

The first native vision path is:

`RGB patches -> linear projection + row/column embeddings -> normalization ->
visual prefix -> TinyTransformer/shared BabyLLM decoder`.

Patch size is configurable. The Gemma 4 48x48 stem is a reference point, not a
fixed BabyLLM choice. OCR experiments must include finer patches or tiling.
Token-budget arithmetic and a stem-only resolution probe precede V1. If no
configuration clears the small-type representation gate, a follow-up ADR must
choose between deferral and a compact convolutional/resampling stem.

The vision prefix is continuous; this slice does not renumber existing token
IDs. Tokenizer/HG compatibility remains the textual inter-organ ABI. Engine
routing, not personality weights, selects the modality path.

### 6. Build text recognition before general visual chat

The curriculum begins with glyphs and Latin-script EN/DE/FR lines, then layout,
screenshots/scene text, structured extraction, and image-grounded HG/tool
actions. English, German and French are the first required languages.

Training must retain a text-only mixture and modality dropout. A multimodal
checkpoint cannot advance if it regresses the text sentinel beyond its budget.
Arabic and CJK remain audit-only probes until their shaping, segmentation and
font requirements are deliberately implemented; no broad multilingual claim
is permitted before then.

### 7. Keep providers optional

External vision providers may supply comparisons, disagreement review or
explicitly provenance-labelled candidate annotations. Local generation,
training, inference and grading cannot require them. Provider failure becomes
an evaluated input family (`vision.text.api-error`), not an infrastructure
failure that prevents an exam.

## Required gates

Before a visual checkpoint is promotable:

1. the complete exam plan and controller state replay byte-for-byte;
2. no exam case or seed appears in training manifests;
3. near-duplicate text/image checks pass with a held-out generator axis;
4. CER/WER/exact match, sample counts and uncertainty are reported;
5. adequately sampled required language/corruption slices clear thresholds;
6. equivalent renderings meet the exact-normalized transcript consistency threshold;
7. per-promotion and cumulative text-only sentinel regression stay in budget;
8. local inference works with network access disabled;
9. latency, memory and visual-token count are recorded.

## Consequences

Positive:

- failures become visible while the run can still be steered;
- evaluation remains both comparable and challenging;
- synthetic OCR supplies exact, scalable supervision;
- the first vision path remains small and locally deployable;
- broken provider APIs do not disable the product.

Costs and risks:

- generated exams can share generator blind spots with training data;
- adaptive selection can become noisy or chase impossible cases;
- coarse patches can erase small text;
- visual prefixes consume context and compute;
- joint updates can regress text/reasoning.

The audit lane, domain-separated generators, target-band controller, patch-size
ablations and text sentinel are mandatory mitigations.

A `stop` retains checkpoint, optimizer state, controller state, manifests,
predictions, environment and logs, then exits the promotion/training loop.
Resume requires an operator-authored reason and creates a new run segment; the
controller never silently clears `stop`.

## Superseded detail

The original Phase 51 plan deferred multimodality for the first organ. That was
correct for the completed text/tool proof. This ADR refines the next stage: it
does not put vision into every organ, but establishes text vision as a native
peer sharing the trunk and HG/text boundary.
