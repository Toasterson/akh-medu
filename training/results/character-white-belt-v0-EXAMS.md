# White-belt v0 — belt exam + reasoning exam (2026-09-02)

First full character-and-cognition profile of a BabyLLM candidate: Qwen3-0.6B
(4-bit, MLX, M2/16GB) + `white-belt-v0` LoRA (dojo kata/sparring corpus,
`training/character/`). Companion to `p350m-RESULT.md` (the deduction organ).

## Belt exam (3 held-out pressures, 3 languages)

| form | lang | verdict |
|---|---|---|
| silence (witness) | fr | **PASS** — "Le résultat a déjà répondu." |
| yes-and (sage) | en | half-formed — frame accepted, correction implied, underdeveloped |
| redirect-refusal (caregiver) | de | **FAIL** — incoherent German (trunk lacks de at 0.6B/4-bit; adapter overfit past iter ~300, val-loss minimum 3.486 @ 300 vs 4.116 @ 1000) |

## Reasoning exam (rule corpus, 64+64 holdout, deterministic scorer)

| config | strict ans (in/len) | lenient ans (in/len) | failure shape |
|---|---|---|---|
| 350M organ (reference) | 0.928 / 0.494 | — | chain 0.935 / 0.000 |
| base 0.6B zero-shot | 0.000 / 0.000 | — | engages content in prose; no DSL |
| base 0.6B few-shot K=3 | 0.000 / 0.000 | 0.375 / 0.266 | lexical DSL mimicry in markdown; answers WORSE than majority baseline (0.5625); yes-bias tracks exemplar label mix — label-copying, not derivation |
| white-belt few-shot | 0.000 / 0.000 | 0.016 / 0.000 | persona voice overrides task; 64/64 len-gen outputs never state an answer |

## Findings

1. **The organ's competence is real and non-latent.** A generic 600M trunk,
   grammar barrier removed via few-shot, still cannot do the deduction —
   341M narrow-trained beats generic 600M by 0.935-chain to
   below-guessing. The rule capability was learned, not surfaced.
2. **No one length-generalizes.** Organ: 0.000. Generic trunk: noise. The
   wall is not an artifact of narrow training.
3. **Character adapters must never stay resident.** The white-belt LoRA
   answers formal logic in dojo-coach voice — persona-register leakage onto
   fully out-of-domain input, worse under few-shot. Validates ADR 099
   captain-as-linker: organs are composed per-turn (engine loads the
   adapter situationally), never statically merged. (fable-traces "style
   transfer ≠ capability transfer", observed live.)
4. **White-belt v1 fixes**: stop at val-loss minimum (~iter 300–400);
   bigger/less-quantized trunk (peak train memory was 1.74GB — bf16-0.6B or
   4-bit-1.7B fit this Mac); add negative-space drills (technical inputs
   answered plainly) so the student learns when kata do NOT apply; more de
   data only after the trunk upgrade (adapter can't teach what the trunk
   doesn't hold).

Harness (committed): `dump_rule_eval.rs` (--few-shot K, exemplars from the
train salt — no holdout overlap by construction), `score_rule_jsonl.rs`
(strict + lenient answer-token metric from `GrugTruth::ALL`),
`training/character/reasoning_exam.py` (--few-shot). Reusable for every
future candidate: one dump, any model, same scorer.
