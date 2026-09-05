# Tool-native BabyLLM full-model pipeline result

**Date:** 2026-09-04  
**State:** implementation complete; foundation training in progress; no model certified or active

## Outcome

The repository and live supervisors now contain a continuous path from a
multilingual foundation checkpoint to an Akh-Medu-loadable, tool-native,
native-text-vision model:

```text
multilingual foundation
  -> knowledge / Dojo / live-tool SFT
  -> immutable sentinels + rolling and frontier exams
  -> bounded synthetic sibling steering
  -> text/tool release certification
  -> native OCR warmup + joint retention training
  -> independent-renderer OCR and text/tool retention certification
  -> provider-free cross-host replay
  -> immutable local installation through the real Grug loader
  -> explicit Akh-Medu runtime selection
```

Every downstream stage waits on the preceding certified artifact. A candidate,
local test, or training checkpoint cannot advance an active pointer.

## Model and corpus contract

The foundation is the deliberately small `small` rung, not the repository
name's prospective 350M weight class:

- 34,468,672 text-trunk parameters;
- vocabulary 8,000, model width 512, eight layers, eight attention heads,
  feed-forward width 2,048, context 2,048;
- English, German and French natural-language foundation sources;
- 709,917 training records, 7,000 evaluation records and 905,837,926 corpus
  tokens;
- tokenizer hash
  `blake3:8ae8451e19179db369edf2f01a4d6cf5ffef538f47928d8cda01cb26f5099aa0`;
- live 50-tool catalog hash
  `blake3:05b1ca2310b5e8352d7b3bfd5e8d3329fdca59805b4b12ff06bec3700cf0a285`.

The post-foundation curricula contain:

| Corpus | Train | Eval | Purpose |
|---|---:|---:|---|
| Knowledge workflows | 23,000 | 4,600 | Acquire, recall after restart, revise from a new edition, provider fallback and grounded abstention |
| BabyLLM Dojo | 476 | 42 | Understanding and reasoning kata |
| Runtime tool contracts | 2,400 | 600 | Exact current Akh-Medu tool descriptors, selection and argument formation |

The vision stage uses the compact convolutional prefix stem selected by the
translation and clean-line experiments. Its frozen promotion audit is 1,050
cases across three languages, five layouts and seven corruption conditions,
rendered independently with Noto Sans Mono. The separated training sibling is
also 1,050 cases. Their manifest SHA-256 values are respectively
`0805aced5c14d08031b849f451ce5ec71df947bba68e8cf0ec3954e9d8f3ac13`
and
`63e13f0a97e79f48d84b7d68fd450c973de3ee127507c9720884c94a1f783316`.

## Implemented operating path

- [x] deterministic frozen, rolling, frontier and independent-audit exam tiers;
- [x] early-warning controller with minimum samples, confidence bounds,
  catastrophic stops and per-family progression;
- [x] exam-to-training quarantine, held-out axes and exact/n-gram/pixel
  separation evidence;
- [x] bounded continuous adaptation with at most five percent newly generated
  sibling data and no promotion authority;
- [x] atomic, immutable, hash-checked model and Adam checkpoints with strict
  resume identity;
- [x] a separate text/tool release gate;
- [x] native OCR training on the same decoder with text/tool retention replay;
- [x] independent-renderer OCR, retention and deterministic offline evidence;
- [x] certification gates for all required language/layout/corruption slices;
- [x] ACP image transport and native grounding through the serving backend;
- [x] provider-free Docker evaluation with `--network none`;
- [x] a fail-closed installer that probes the source, temporary copy, immutable
  release and final alias through `GrugChatBackend`, excludes optimizer state,
  and atomically advances `knowledge/model-small`;
- [x] a credential-late Runpod cost guard that stops, but never deletes, the GPU
  after local staging, terminal failure or a 36-hour deadline.
- [x] a serving-only Hugging Face release packager that pins the public parent,
  requires byte-bound PASS responses from exact GLM 5.3 and Kimi K3, injects
  its token only into the upload helper and anonymously verifies every result.
- [x] an append-only per-workspace runtime tool-contract drift journal that
  retains both complete endpoint snapshots and their verified delta for the
  next adaptation run without altering the current 50-tool corpus.
- [x] strategy-independent, bounded `archival_search` registration, so the
  default small-model path can retrieve durable note bodies using its trained
  contract without gaining virtual-context core-memory mutation privileges.
- [x] separate `study_note_list` and `study_note_read` tools: listing is
  body-free, metadata-filtered and paginated, while reading requires one stable
  ID and returns its current revision token; supporting source quotes remain
  archival.
- [x] separate, schema-complete note creation and revision tools; revisions use
  optimistic concurrency, reject stale heads without mutation, preserve exact
  retry idempotency and have synthetic conflict/re-read/retry coverage.
- [x] a host-enforced durable-note evidence gate: every citation must carry a
  nontrivial quote matching the same earlier successful retrieval result as its
  source ID and locator, same-round retrieval cannot authorize a write, and a
  repaired retry is rechecked before execution.
- [x] merge-preserving note revisions, so supplemental frontmatter does not
  erase an existing challenge link or other retained metadata.

The installer deliberately does not edit `akh.toml`. Once a certified bundle
exists, selecting `backend = "grug"` and `model = "knowledge/model-small"`
remains an explicit deployment operation.

## Source-head v2 refinement, isolated from the running job

The paid run remains frozen at its original 50-tool contract
`blake3:05b1ca2310b5e8352d7b3bfd5e8d3329fdca59805b4b12ff06bec3700cf0a285`
and five knowledge families. No remote supervisor, corpus or checkpoint was
changed.

The exact current source export now has 53 tools and contract hash
`blake3:3d6e18bb74765cab13d35901d0a10a7834077f302cc578941ca06ef56223700a`.
`akh-knowledge-workflow-v2` adds independently gated textbook-inventory and
note-navigation families. Its real-contract smoke generated 21 train and 21
evaluation episodes, 92 SFT records in each split, seven immutable sentinel
families, all EN/DE/FR lanes and all 14 workflow stages. Textbook navigation
lists documents before retrieving the chosen edition; note navigation filters
body-free metadata before reading one stable note ID; every generated note
write is quote-backed. Revision cases read a head token before writing, and
harder cases recover from an injected concurrent update. A separate
`akh-tool-contract-sft-v2` smoke generated 212 train and 106 evaluation records
over all 53 tools. All sampled note IDs and revision tokens satisfy their live
patterns. Note-create records contain matching earlier search results;
note-revision records contain both an observed head/token and matching earlier
search result. Every record retains exactly four offered tools.
The reproducible smoke identities are:

- knowledge train episodes
  `blake3:b377ffc7a3f256ba3e00cdaf9fe8ca8daaa70ac2cc73d4d162e27ef8bb878133`
  and SFT
  `blake3:c404fa441d06a3a0a0afb6f7d3de74397c6553c026b3cc289cf8126830e7493f`;
- knowledge evaluation episodes
  `blake3:2f9d173091f20ef953d7701646c6d9a9002c5b21ca2b686e2f9d78723aa26550`
  and SFT
  `blake3:a5ec20cddfc6ba7f25b42a9e61b824227ab20ef76c49d1bdc6938bc8b46c7f8c`;
- all-tool train SFT
  `blake3:eef2fa1308b4c9375e567dc0ee654ad77cdb7f5c7d1b7164cb2f9c66807426d4`
  and evaluation SFT
  `blake3:e58109e4a84623c33820f8a9f9c8c341acc169fc5958a889767e1debf4c112e6`.

This is a ready next-adaptation input, not current-checkpoint capability. It
must flow through the contract-delta journal, a new immutable candidate, all
seven closed-loop workflow exams, the complete 53-tool matrix, and the existing
text/vision retention gates before activation.

## Live execution evidence

Snapshot: **2026-09-04T17:30:33Z**.

- Foundation process PID 8233 is running on pod `4ypc4li9y537f2`.
- The training log had reached step 40,480 and 663,240,704 optimizer-visible
  tokens: 80.96% of the 819,200,000-token / 50,000-step schedule.
- Checkpoint 39,999 is complete on disk with 655,360,000 tokens and a
  137,884,285-byte model file. Its canonical BLAKE3 was not recomputed during
  this read-only refresh. The latest independently content-hashed checkpoint in
  this report remains step 25,999 with model hash
  `blake3:ebb0283053136878290fe844dd07f31524ee75ec9d2d586838e45139d162faa3`.
- The post-foundation supervisor is `waiting_for_foundation`.
- The adaptive loop is `waiting_for_initial_probe`.
- The text release gate is `waiting_for_adaptive_loop`.
- Native OCR training is `waiting_for_text_release`.
- The cross-host offline gate is `waiting_for_remote_candidate`, with no last
  connection error.
- The local installer is `waiting_for_certified_multimodal_model`.
- The cost guard is `waiting_for_pipeline_terminal`.
- The exact-review/publication watcher is `waiting_for_certified_install`.

The paid Runpod remained `RUNNING/running` at the snapshot, with one A100-class
GPU, network volume `4wgri7h01a`, and a reported rate of USD 1.59/hour. The
guard's recorded operation is `stop_never_delete`; it does not hold the API key
while waiting.

## Offline and deployment evidence

The pinned evaluator image is
`akh-medu/native-ocr-offline:phase51-v1`, image ID
`sha256:e8807626809b048573850c07d13fff60579ab4117d29925f38842914e6d448da`
(51,398,973 bytes). Its network-denied smoke exposed only loopback and zero
rows in `/proc/net/route`.

The frozen real-loader probe is
`runs/babyllm-tooling-v1/grug_bundle_probe`, SHA-256
`d2417e4565c9beddc7cc80a9b9b5f3f976614a8f3a5c5055e9a4bef477f582ea`.
The live installer pins that immutable copy rather than a mutable Cargo target.

The publication watcher runs from the read-only
`runs/babyllm-publication-tooling-v1` closure. Its supervisor and upload-helper
SHA-256 values are respectively
`4c2c4a146c0ee25d3b5548432782b40d8136c124df9f7f1dfe0f7eb9c29c523f`
and `3c16f8d0b9d0dade5e489e687fa094635124c05c750d21c562b5794d1ad0518d`.
An initial launch correctly rejected the mutable `/opt/homebrew/bin/uv`
symlink; that log is preserved, no packet existed, and no credential or upload
was attempted. The live launch pins the resolved uv 0.12.8 executable.

## Validation

- 25 Python orchestration tests pass across the adaptive loop, text release,
  native OCR pipeline, certified installer, Runpod cost guard and reviewed
  Hugging Face publication boundary.
- Four `GrugChatBackend` unit tests pass with `grug-backend` enabled.
- Four tool-contract drift journal tests pass, covering replayable/idempotent
  writes, metadata-only classification, no-drift suppression and tamper refusal.
- `akhomed` checks successfully with `server,grug-backend`, including the
  per-workspace journal wiring.
- The production daemon-path regression proves a default `compacting` akh has
  `study_note_append`, `study_note_revise`, `study_note_list`,
  `study_note_read` and `archival_search` while lacking
  `core_memory_append`, `core_memory_replace` and `query_graph`.
- Eleven study-note tests and 37 evidence-filter tests pass, including mandatory quotes,
  metadata-only filtering, explicit reads, pagination bounds, revision metadata
  retention, stale-head refusal, split-contract completeness and rejection
  before evidence. Separate regressions prove the real
  archival output carries the stored source/locator/quote and that the full
  dispatch loop rejects same-round evidence before succeeding in a later round;
  nested result envelopes also preserve per-hit provenance boundaries.
- Thirteen knowledge-workflow tests pass. A generated source-head bundle against the
  exact exported contract covers seven families, three languages, 14 stages
  and all twelve learning-loop tools. Eight contract and thirteen all-tool
  curriculum tests cover nested schemas, string patterns, deterministic
  sampling and complete-tool exams.
- The deployment probe builds in release mode and passes `cargo check` through
  the production backend feature.
- Earlier component validation remains green: 19 adaptive-exam, 11
  tool-curriculum, ten knowledge-workflow, eight checkpoint, five native-OCR
  trainer, two native-OCR evaluator and two certifier tests, plus real
  image/session wiring checks.
- Python compilation, Rust formatting and `git diff --check` pass.

## Gates still red

- [ ] Foundation training has not reached step 50,000.
- [ ] Post-foundation SFT and the initial probe have not executed.
- [ ] The bounded adaptive run has not produced its terminal evidence.
- [ ] No certified text/tool active pointer exists.
- [ ] Native OCR joint training and the 1,050-case audit have not executed.
- [ ] No multimodal active pointer or locally installed model alias exists.
- [ ] Akh-Medu has not been switched to the Grug backend/model alias.
- [ ] The final certified serving bundle has not been published to Hugging Face.

Therefore this result proves the implemented and running pipeline, not a
finished model or production deployment.

## Review boundary

The exact source-head request, amendment, model identities, byte hashes,
verdicts and finding dispositions are recorded in the
[tool-native learning review note](../../docs/ai/reviews/2026-09-04-tool-native-learning-v2.md).
Those verdicts bind only the recorded source slices and generated identities;
they do not certify trained weights, the still-running full pipeline, an
installed model or a Hugging Face release. No substitute reviewer and no
borrowed Agent Board identity is used or claimed.

## References

- [Phase 51 implementation plan](../../docs/ai/plans/2026-09-03-phase51-adaptive-exams-native-multimodal.md)
- [Adaptive/native-vision operations manual](../../docs/ai/manuals/2026-09-04-adaptive-exams-native-vision.md)
- [Certified native-OCR runtime decision](../../docs/ai/decisions/103-certified-native-ocr-runtime.md)
- [`install_certified_babyllm.py`](../scripts/install_certified_babyllm.py)
- [`stop_runpod_after_pipeline.py`](../scripts/stop_runpod_after_pipeline.py)
- [`publish_certified_babyllm.py`](../scripts/publish_certified_babyllm.py)
- [`hf_commit_certified_babyllm.py`](../scripts/hf_commit_certified_babyllm.py)
