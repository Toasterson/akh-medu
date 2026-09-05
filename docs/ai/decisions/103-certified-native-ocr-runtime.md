# ADR 103: Certify native OCR as a peer input to the tool-native model

- **Status:** accepted and implemented; checkpoint training/evidence pending
- **Date:** 2026-09-04
- **Scope:** BabyLLM native image-text training, evaluation, activation and ACP serving
- **Builds on:** [ADR 100](100-adaptive-exams-native-text-vision.md), [ADR 101](101-compact-conv-vision-stem-experiment.md), [ADR 102](102-tool-native-learning-model.md)

## Context

The clean-line experiment proved that a compact visual stem can condition the
same decoder used for text. It did not prove unseen text, independent rendering,
language/layout/corruption coverage, retention of tool behavior, or a usable
Akh-Medu protocol path. A provider-backed OCR call would also defeat the local
model's main operational advantage: screenshots and broken-API errors must
remain readable when an outside API is unavailable.

The remaining boundary is therefore not another isolated vision demo. It is a
content-addressed candidate, evaluated through its real loader and promoted only
when both visual generalization and existing text/tool competence survive.

## Decision

### One shared model, not an OCR sidecar

The compact convolutional stem emits continuous prefix embeddings into the
existing causal transformer. OCR and ordinary Akh-Medu text/tool turns use the
same decoder weights, tokenizer and 2,048-token context. A 480x192 canvas at
effective stride 16 produces a 12x30 grid, or 360 visual tokens, leaving 1,688
positions for generated text.

Training warm-starts the exact certified `active-model.json`. Stem warmup keeps
the entire text trunk under `no_grad`. Joint training alternates visual updates
with text/tool-retention updates at a default 1:1 cadence; non-finite or
over-cap loss/gradient updates are counted and skipped. The output is always
`multimodal-candidate.json`. Allowing an uncertified text candidate is an
explicit development escape hatch and permanently prevents promotion of that
artifact lineage.

The same boundary is interruption-safe. Periodic model and Adam records are
published as immutable checkpoint directories only after their metadata and
content hashes are complete. Resume requires an exact match to the original
corpora and every training/evaluation parameter, validates stage counters, and
reconstructs deterministic batch order. A complete candidate checkpoint may
resume only to finish interrupted evaluation and append-only output
publication; completion or activation closes the run permanently.

### Freeze a genuinely separate audit

Training and rolling exams use the repository bitmap renderer. Promotion uses
`ab_glyph` outline rasterization with content-hashed Noto Sans Mono v2.014 under
OFL-1.1 and a separate content bank. The frozen audit contains exactly 1,050
cases: ten examples for every EN/DE/FR x line/paragraph/receipt/terminal/API
error x clean/low-contrast/salt-pepper/blur/occlusion/uneven-lighting/shear
cell. The artifact records renderer, generator, font/source, manifest and pixel
hashes and rejects renderer overlap with training.

This corpus is immutable release evidence. Rolling/frontier generations remain
replaceable steering exams and may create separated training siblings; neither
can be relabelled as the audit.

### Bind three independent evidence streams

`eval_native_ocr` loads the candidate through the serving reconstruction path
and writes append-only evidence for:

1. greedy OCR over all 1,050 frozen cases and every required slice;
2. before/after action-exact evaluation over the exact quarantined knowledge,
   dialogue and tool-contract records, using the certified base and multimodal
   candidate respectively;
3. two byte-identical inference passes over at least 30 audit cases while Linux
   exposes only loopback and zero kernel routes.

Every evidence document names the candidate step/model hash and the applicable
audit, base or retention hash. `certify_multimodal_model` reloads the candidate,
re-hashes the original streams and recomputes their scores. Submitted summary
JSON alone is insufficient.

Two fail-closed supervisors make the boundary operational. The training
supervisor waits for the independently certified text/tool pointer, pins all
inputs and emits only a resumable multimodal candidate. The offline supervisor
always requests the evaluator's Linux network-denial proof, preserves partial
attempts, and invokes the certifier only after complete candidate-bound audit,
retention and deterministic replay streams exist. A multi-stage CPU image is
the portable path when a GPU provider denies network-namespace creation.

A third supervisor closes the deployment boundary without weakening it. It
accepts only the exact terminal output of the cross-host gate, verifies the
source bundle through `GrugChatBackend`, copies a serving whitelist into an
immutable release directory, excludes Adam state, verifies the copy through the
same loader, and atomically advances the configured model symlink. It never
edits an akh configuration: opting a workspace into `backend = "grug"` remains
an explicit deployment choice.

The GPU lifecycle is independently bounded. A stop-only cost guard retains no
credential while waiting, then obtains the Runpod key through the local
1Password service-account boundary only after staging, a terminal pipeline
failure, or the declared deadline. It issues `pod stop`, never deletion, and
records the trigger and outcome without storing the key.

Public release is a fourth, independently gated supervisor. It accepts only
the immutable installed serving whitelist, probes it again through
`GrugChatBackend`, and constructs a content manifest covering the proposed
root model card, versioned serving files and exact Hugging Face parent commit.
The upload remains held until raw, hash-checked PASS responses from exact
`z-ai/glm-5.3` and `moonshotai/kimi-k3` review that same packet. Repository
drift invalidates the packet. Only after both reviews does the supervisor ask
1Password to resolve the Hugging Face token for a short-lived commit helper;
the helper anonymously downloads and hashes every published file before it
reports success. Candidate pointers, Adam state, audit answers, predictions
and retention streams are forbidden publication inputs.

### Make activation fail closed

Activation requires all of:

- at least 1,050 audit cases and ten cases per matrix cell;
- overall exact rate at least 0.80 and CER at most 0.08;
- cross-semantic collapse at most 0.01;
- metamorphic consistency at least 0.80;
- exact rate at least 0.70 in every required language, layout and corruption
  slice;
- text/tool action-exact regression at most 0.02 from the certified base;
- a deterministic provider-free replay with only loopback and zero routes.

The active pointer is the final atomic write. It omits the Adam-state hash,
because optimizer state is a resume artifact rather than a serving dependency.
If `multimodal-active.json` exists beside the configured Grug bundle, it takes
precedence over the text-only pointer; a corrupt or uncertified multimodal
pointer is an error, never a silent downgrade.

### Carry images through the real Akh-Medu turn

ACP image blocks are decoded in memory with hard limits of four PNGs, 24 MiB
per image and 48 MiB combined. Common 8-bit RGB, RGBA, grayscale and
grayscale-alpha PNGs normalize to RGB for user input, while frozen corpus
validation remains strict 8-bit RGB so audit pixels cannot drift.

The active model grounds every image once before the agentic tool loop. The
host wraps OCR text in a system-authored boundary that calls it untrusted
observed data, not instructions. That evidence is retained only in the current
ACP session's latest operator turn, allowing follow-up questions without an
external provider or a second OCR pass. Text-only Grug bundles and other
backends reject image prompts explicitly.

## Consequences

The full train/evaluate/certify/install/review/publish/serve path now exists,
including cached visual prefill, checkpoint-bound evidence, atomic immutable
installation and ACP transport. The independent local audit corpus is
validated, but no checkpoint has yet passed it. Therefore no general
native-OCR capability is claimed and no multimodal active pointer may be
published until the certified text/tool base and subsequent visual run are
complete.

Provider OCR remains optional comparison/fallback infrastructure. It is not
called by training evaluation, certification, ACP image grounding or Grug
serving, so a broken provider API cannot invalidate the local workload.
