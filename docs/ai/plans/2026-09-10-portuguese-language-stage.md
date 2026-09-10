# Portuguese language acquisition for BabyLLM

Date: 2026-09-10. Status: deferred follow-on stage; source documentation and
local integration points reviewed. Corpus acquisition, implementation, training,
and capability measurements are pending.

Execution handoff: [combined preparation for Portuguese and process
consultation](../../../training/README.md#combined-preparation-portuguese-and-process-consultation).
Prepare both corpora together, then train Portuguese before consultation with
separate manifests, checkpoints, and evaluation evidence for each stage.

## Intended capability

Add Portuguese reading, writing, dialogue, and task execution to BabyLLM.
Use Brazilian Portuguese as the initial working target because the reference
project, Manacá, targets PT-BR. Broader Portuguese coverage needs separate
variant annotations and evaluation; a generic `pt` source is not proof of PT-BR
or European Portuguese coverage. Audio training is outside this text-stage scope.

This plan follows interest in the [LNCC AI Institute's Manacá
repositories](https://github.com/Instituto-IA-LNCC). Its corpus preparation and
evaluation provide references for a candidate trained in Akh-Medu's own model
lineage. Using Manacá weights would be a separate architecture/backend decision.

## Source shortlist

These are candidate sources, not a frozen corpus manifest. Per Till's direction,
accessible corpus components are eligible regardless of license category.
Select and weight data by Portuguese coverage, quality, duplication, evaluation
contamination, and the training budget. Pin dataset revisions, splits, selected
files, source IDs, and checksums for reproducibility.

| Source | Role | Selection and provenance |
| --- | --- | --- |
| [GigaVerbo](https://huggingface.co/datasets/TucanoBR/GigaVerbo) | General Portuguese text | All accessible components are eligible. Its `metadata` field identifies upstream sources; `label` and `probs` describe classifier quality judgments. Preserve component identities for sampling and evaluation. |
| [FineWeb-2 Portuguese](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | Additional web coverage | The published Portuguese path is `data/por_Latn/train`. Deduplicate against other web sources before mixing. |
| [Manacá's Wikipedia and Ulysses sources](https://github.com/Instituto-IA-LNCC/manaca-1b-base#the-corpus) | Encyclopedic and optional legal text | Manacá reports using Portuguese Wikipedia and Ulysses Tesemõ alongside GigaVerbo. Preserve the selected upstream release's identity. Keep legal text a bounded domain component. |
| [Manacá's instruction corpus](https://github.com/Instituto-IA-LNCC/manaca-1b-instruct#the-instruction-corpus) | Instruction following, conversation, translation, summarization, and Portuguese NLP tasks | Include accessible constituent datasets without license-based exclusions. Preserve component identities and deduplicate against prose sources and held-out tasks. |
| [Aya Dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset) | Portuguese instruction examples | Human-annotated multilingual data; Portuguese has language code `por`. Use training examples and reserve its Portuguese test examples for evaluation. This selection refers to `aya_dataset`, not the larger Aya Collection. |
| Akh-Medu Portuguese episodes | Tool use, knowledge workflows, Dojo, correction and repair | Author natural Portuguese situations and targets in the canonical runtime format. Review translations and cultural assumptions with fluent speakers. Preserve scenario and generator provenance. |

## Local integration points

- [`FoundationRecord`](../../../crates/grug-data/src/foundation.rs) already
  carries source, revision, license, language, and text. Preserve truthful
  descriptive metadata; do not use license categories to filter this corpus.
- [`import_foundation`](../../../crates/grug-data/examples/import_foundation.rs)
  imports local text/JSONL and refuses to overwrite an existing output. It sets
  source metadata and language per invocation: partition an upstream export by
  component first. A single import of mixed GigaVerbo records would lose the needed
  component distinctions. Keep upstream IDs and an acquisition manifest.
- [`prepare_foundation`](../../../crates/grug-data/examples/prepare_foundation.rs)
  accepts language allowlists and separate train/eval inputs. Keep `pt` for
  unspecified Portuguese and `pt-BR` only where supported by source evidence;
  report aggregate Portuguese coverage as well as variant coverage.
- [`WorkflowLanguage`](../../../crates/grug-data/src/knowledge_workflow.rs)
  currently enumerates English, German, and French. Add Portuguese across its
  serialization/code mapping, generators, language rotation, and evaluations,
  including the templates in
  [`tool_curriculum`](../../../crates/grug-data/src/tool_curriculum.rs).
  Preserve existing serialized language identities. Translate dialogue while
  retaining exact tool names, schema keys, paths, identifiers, and arguments.
- The local [`BPE implementation`](../../../crates/grug-train/src/bpe.rs) uses
  byte-level encoding. Character encodability does not establish efficient
  Portuguese tokenization or Portuguese capability in a particular checkpoint.
  Audit the actual selected parent's tokenizer artifact.

## Planned experiment sequence

1. Select a compatible parent after its existing terminal gates. Record exact
   checkpoint/config/tokenizer/contract hashes and its measured outcome. Follow
   the parent-selection rules in the [training runbook](../../../training/README.md).
   This document does not establish the current state of any training job.
2. Create a small, deterministic, stratified source sample. Check Portuguese
   quality, source identity, duplication, and tokenizer behavior before
   selecting a larger token budget. Do not use the beginning of a concatenated
   dataset as a representative sample. Preserve training/evaluation boundaries
   across documents, near duplicates, translated equivalents, and episodes.
3. Measure exact round trips for accents, decomposed Unicode, names, mixed
   Portuguese/code text, and identifiers. Report tokens per character/word and
   truncation at the parent's context limit. Keep that tokenizer for a compatible
   continuation. If it is too inefficient, design a separately versioned
   tokenizer/embedding migration or a new foundation run with its own gates.
4. Train a bounded language-adaptation candidate on Portuguese prose mixed with
   replay from the parent's existing language and task distribution. Freeze the
   token budget, sampling weights, seed, stop conditions, and retention tolerances
   before launch. Mixture ratios need an ablation; no optimum is assumed here.
5. Follow with Portuguese instruction and canonical task training: requests,
   clarification, knowledge retrieval, tool calls, corrections, and Dojo episodes.
   Include native Portuguese situations alongside aligned multilingual examples.
   Use the eligible Portuguese candidate as the parent for the deferred
   consultation curriculum, following the combined preparation checklist.
6. Evaluate generated behavior and retention before selecting a candidate for
   the existing certification process. Training loss alone is insufficient.

## Evaluation and completion evidence

Freeze held-out sample sizes, rubrics, and promotion thresholds before training.
Compare the exact parent and candidate on the same cases:

- Portuguese comprehension, natural responses, instruction following, and
  requested language retention across multiple turns; fluent-speaker review
  complements automatic scoring.
- Canonical tool validity and actual task outcomes from Portuguese requests,
  including exact copies of names, paths, numbers, and hashes.
- Dojo and consultation behavior: respect explicit corrections, revise a wrong
  interpretation, repair the interaction, and preserve participant ownership.
  Score actions and effects rather than Portuguese keyword matches.
- Portuguese evaluation sources kept outside training, including Aya's test
  partition and selected tasks from [Manacá's evaluation
  suite](https://github.com/Instituto-IA-LNCC/manaca-1b-base/tree/main/scripts/eval).
  Audit upstream training mixtures for benchmark leakage before claiming an
  uncontaminated result.
- Retention of English/German/French and the parent's knowledge, tool, and Dojo
  gates; report each language/task slice so an aggregate cannot conceal a loss.

Completion requires a versioned corpus manifest, tokenizer audit, reproducible
training record, exact checkpoint identity, and passing Portuguese/retention
evidence. The current deliverable is this plan. It schedules no job and
provides no trained Portuguese model, certification, activation, or publication.
