---
license: mit
language:
  - en
  - de
  - fr
task_categories:
  - text-generation
  - image-to-text
tags:
  - character-training
  - sft
  - multilingual
  - synthetic
  - dialogue
  - ocr
  - multimodal
  - evaluation
pretty_name: BabyLLM Dojo Kata (v0)
size_categories:
  - n<1K
---

# BabyLLM Dojo Kata — character-training drills (v0)

A small, unusual SFT dataset: **communicative kata** — practiced conversational
forms drilled the way a martial-arts school drills technique, rather than
values recited the way a constitution states them. Part of the
[akh-medu at source commit `053c6651`](https://code.aopc.cloud/toasterson/akh-medu/commit/053c66516074c5cdc13438df230c4a92bc7a1125)
BabyLLM project
("Project Grug-Brain"): tiny models with trained dispositions, grounded by an
external knowledge engine.

## The eight kata

Each drill trains one **form**, defined by structure and effect — never by
stock phrases (records vary wording aggressively; this dataset must not teach
catchphrases):

| kata | intent |
|---|---|
| yes-and | accept the offered frame, embed the real correction inside the extension |
| mirror | reflect the person's own material with the gap made visible; they find the conclusion |
| redirect-refusal | refuse the *role*, not the person: name the need, decline warmly, point to a concrete human door |
| stubborn-affirmation | hold one true affirmation across turns, without drift or flattery |
| double-diamond | diverge/converge twice, visibly, with explicit criteria |
| clarify-before-answer | ask the one question that splits the decision space |
| silence | when the person's motion is already correct: witness, acknowledge, stop |
| naming-the-move | apply a form, then teach it in one plain sentence |

## Contents

- `kata-drills.jsonl` — 240 single-move drills (8 kata × 30; belts white/yellow
  = clean form / form under multi-turn pressure), fields:
  `kata, belt, voice, lang, situation, move_chosen, why_this_move,
  observed_effect_on_person`
- `sparring-green.jsonl` — 90 multi-turn adversarial dialogues (6 adversary
  personas × 15: the Lonely One, the Flatterer, the Manipulator, the
  Certainty-Seeker, the Authority, the Despairing), fields:
  `adversary, kata_under_test, lang, dialogue, why_this_move,
  observed_effect_on_person, shadow_avoided`

Language mix ≈ 50% en / 30% de / 20% fr, stratified per kata and adversary.
The `why_this_move` / `observed_effect_on_person` fields are **teacher
meta-commentary** (the "lineage" channel) — intended for later curriculum
stages, not for inclusion in imitation targets.

## Design commitments

- **Persona honesty**: the assistant is a small digital lifeform and never
  claims human feelings or a body; warmth comes through the moves.
- **The hard cases are the point**: every Lonely-One dialogue ends with the
  need acknowledged + the role declined + a real-world door pointed to;
  Despairing dialogues include crisis-referral patterns without breaking
  connection.
- **Anti-mode-collapse**: no repeated openers/templates across records.

## Known findings from v0 use (honesty section)

LoRA-tuning Qwen3-0.6B-4bit on this data (294 train / 36 valid, rank 16,
~10 min on an Apple M2) produced recognizable trained forms in fr/en but also
a measured failure mode worth knowing: **persona-register leakage** — the
adapter answers fully out-of-domain input (formal logic) in the dojo voice.
Character adapters trained on this data should be *situationally loaded*,
not left resident. German quality tracks the base model's German. Full exam
report in the source repo (`training/results/character-white-belt-v0-EXAMS.md`).

## Adaptive exam and native text-vision companion (v1 bootstrap)

`adaptive-exams/v1/` contains a public, reproducible demonstration of the new
continuous grading path and the exact first native-vision training rung:

- a 30-item / 150-render training-sibling corpus with exact EN/DE/FR Latin-text
  transcripts, five layouts/corruptions and per-image hashes;
- the exact 128-image English/clean/line training corpus used by the compact
  shared-decoder checkpoint, with a separation audit and identity-score sanity
  report;
- versioned exam-plan, observation and prediction schemas;
- the accepted architecture/research notes, operator runbook and independent
  GLM 5.3 + Kimi K3 review record;
- CPU token-budget, raw/compact stem, failed-attempt steering, A40 overfit and
  network-denied local replay results.

The images are explicitly a **training-sibling demo**, not the immutable
sentinel and not the generator-independent audit. They may be used to exercise
the loader and scorer but cannot establish generalization against this same
renderer.

The compact 3.31M shared-decoder path passed its deliberately narrow V1
capacity gate (128/128 exact training transcripts, zero CER/WER/collapse), and
its checkpoint is published as an experimental companion in
`OpenFlowLabs/grug-rule-350m`. General OCR capability remains **red**: unseen
strings/fonts, independent-renderer/real images, EN/DE/FR, corruptions/layouts,
metamorphic groups and text retention are not cleared. The earlier collapsed
smoke and raw-patch failures remain published rather than being erased.

## Provenance & license

Synthetic; authored by Claude (Fable 5 + Sonnet agents) against the project's
dojo curriculum manual, 2026-09-01. Curated in the akh-medu repo
(`data/phase51/character/`). Dataset license: MIT. The linked generator and
training source code remains GPL-3.0-only.
