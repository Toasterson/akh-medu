# ADR 102: Tool-native learning model and certified activation

- **Status:** accepted; runtime and training boundary implemented
- **Date:** 2026-09-04
- **Scope:** Phase 51 BabyLLM integration with Akh-Medu knowledge, notes, tools and exams
- **Builds on:** [ADR 089](089-organ-kg-zeroshot-path.md), [ADR 100](100-adaptive-exams-native-text-vision.md), [ADR 101](101-compact-conv-vision-stem-experiment.md)

## Context

The BabyLLM becomes useful in Akh-Medu when it can understand a request, plan
against the tools actually available to that akh, acquire missing evidence,
maintain durable study notes and answer with provenance. It does not become
useful merely by making a synthetic benchmark loss decrease.

The model also cannot safely memorize Akh-Medu's changing textbooks or tool
schemas. Facts, editions, notes and provider availability change faster than a
small checkpoint should be retrained. Tool schemas can change independently,
and an upstream search or OCR API can be broken at the moment a challenge is
attempted.

## Decision

### 1. Split learned competence from changing knowledge

The weights learn language, intent, workflow state, tool selection, argument
construction, evidence discipline and when to abstain. Runtime stores retain
the changing material:

| Need | Runtime capability | Model behaviour |
|---|---|---|
| recall current workspace state | `memory_recall` | inspect before searching or writing |
| identify missing evidence | `gap_analysis` | state the gap rather than inventing a fact |
| inspect available textbooks and editions | `mcp__seshat__seshat_documents` | list metadata before selecting a source |
| find textbook passages | `mcp__seshat__seshat_search` | retrieve and preserve source locators |
| acquire a new textbook/edition | `mcp__seshat__seshat_ingest` | ingest provenance-bearing source text |
| make new material searchable | `mcp__seshat__seshat_embed_now` | complete the ingest/index transition |
| recover older local material | `archival_search` | use local evidence when a provider fails |
| browse an akh's notes | `study_note_list` | filter and paginate body-free metadata |
| read one selected note | `study_note_read` | load one stable ID and its current head token |
| create distilled knowledge | `study_note_append` | create only after observing matching quoted evidence |
| revise distilled knowledge | `study_note_revise` | update against the exact observed head or re-read after a conflict |
| search outside providers | `web_search` | optional acquisition route, not a dependency |

Seshat remains the textbook/archive source of truth. Each akh's note catalog is
per-workspace, append-only and revisioned. Notes carry source citations,
supporting quotes, language, challenge ID, tags and frontmatter. Supporting
quotes are archival: resident context and catalog listings redact them. The
catalog can filter note ID, title, tags and non-archival frontmatter without
loading bodies and reports bounded pagination explicitly. Listing,
identity-based read, creation and revision are four single-purpose model-facing
contracts, so required fields are unconditional and generic curricula cannot
synthesize cross-operation arguments. Current reads and listings expose a
content-addressed revision token. Revision requires that token and fails
atomically if another turn advanced the head. Exact retries remain idempotent,
and harder synthetic cases teach re-read and retry after a conflict. Adding
custom frontmatter to a revision merges it with the prior note metadata rather
than silently dropping the challenge link. `archival_search` pages both
knowledge-graph/archive results and the durable note bodies back into context.
It is a bounded read-only capability under every Akhomed context strategy;
small `compacting`/`passive` models therefore retain the retrieval contract they
learned without receiving `core_memory_append`, `core_memory_replace` or
`query_graph`, which remain gated to the capable-model `virtual` strategy.

### 2. Train against captured real contracts

`ToolContractSnapshot` is the inter-organ ABI. The snapshot is content
addressed, retains nested JSON Schema, and is produced from the same native and
MCP manifests the daemon exposes. Corpus records carry the contract hash and
the exact OpenAI-style tool set offered at that step.

One canonical prompt renderer is shared by corpus generation, evaluation and
serving. Generated calls are validated against the live offered schema before
execution. Contract drift is logged by added, unavailable and changed tools;
the daemon still uses the live schema and blocks invalid calls. The drift log
now carries a reproducible `ToolContractDelta` hash and distinguishes changes
to the exact model-visible OpenAI definition from Akh-only metadata changes.
Metadata-only drift does not justify changing weights.

An immutable delta curriculum turns a model-relevant old-to-new snapshot into
three streams: low-rate replay across every current tool, extra positives for
added or model-visible changed tools, and explicit gap responses for removed
or schema-incompatible legacy requests. Every record offers tools from the
complete current catalog. Removed tools are never offered, and a still-valid
backward-compatible request is never falsely labelled as a gap.

When the Grug backend is active, the daemon also persists runtime drift under
the owning workspace's `model-contract-drift/` directory. Each append-only,
content-addressed entry contains the exact trained snapshot, exact live
snapshot, verified `ToolContractDelta`, per-file hashes and a manifest written
last. Metadata-only changes are retained but marked
`model_adaptation_required=false`; they cannot manufacture a weight update.
Repeated observations are idempotent, tampered entries fail verification, and
a failed journal write is retried on the next turn rather than being hidden by
the in-process de-duplication guard.

### 3. Teach complete study workflows

The source-head v2 stateful curriculum has seven independently gated
capabilities:

1. acquire and index a textbook;
2. recall cited knowledge after restart;
3. revise a note from a newer edition without erasing its history, including
   recovery from an intervening concurrent revision;
4. recover locally when an outside provider fails;
5. abstain when evidence is missing and refuse an ungrounded note write;
6. inspect the textbook inventory and select the newest matching edition
   before retrieving a passage;
7. find a challenge note through body-free metadata and explicitly read it by
   stable ID.

Every durable-note citation includes `source_id`, `locator`, and a supporting
`quote`. The host admits either write tool only when one successful retrieval result
from an earlier model round contains all three. Retrieval and write calls in
the same parallel round cannot satisfy the gate, and arguments changed by a
repair/grounding retry are checked again before execution. Model-supplied
retrieval arguments do not count as provenance; the returned content itself
must carry the source ID, locator and quote. Full evidence remains turn-local
while the WAL records only compact tool success and
`evidence_verified`, preventing source text from being copied into routine
traces.

Every corpus spans English, German and French and records the workflow stage.
Multi-step episodes become target-masked SFT records containing prior tool
actions and results. Training and generation preserve both the stable
tool-policy head and newest conversation tail when a prompt exceeds context.

Language competence is trained before tool SFT rather than expected to emerge
from templated calls. The causal foundation consumes pinned, licensed natural
text plus a causal projection of the knowledge, Dojo dialogue, and all-tool SFT
streams. Its tokenizer and checkpoint hashes become immutable lineage carried
by the warm-started knowledge candidate. The first production acquisition is
700,000 training and 7,000 held-out documents: 350,000/3,500 English documents
from [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) and
175,000/1,750 each of German and French from
[FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ), all at exact
Hub commit revisions. These are record ratios, not asserted token ratios; the
packed-token manifest reports the measured mix.

The first deployable capacity rung is the canonical `small` shape (512 hidden
width, 8 layers, 8 heads, 2048 context): 26.8M parameters with the smoke
vocabulary and about 34.5M with the 8,000-token production ceiling. The 350M
shape is an experiment behind the same artifact and runtime contracts, not a
separate product architecture.

### 4. Keep steering exams and release evidence distinct

Ongoing teacher-forced evaluation locates the failing family and workflow
stage while training. Rolling and frontier exams provide fresh synthetic
steering data. The immutable sentinel remains longitudinally comparable, and
audit data must come from a different generator or human-verified source.

Release evaluation for stateful learning workflows is closed-loop. A fixture
returns its tool result only after the checkpoint emits the exact expected,
schema-valid action. A wrong or malformed action stops the episode, so later
gold history cannot launder an early failure. A separate checkpoint-executed
single-turn matrix covers every tool in the captured contract, with the same
four-tool decision budget as the daemon. A correct tool with schema-valid but
wrong arguments is a precursor signal; malformed output, an unavailable/wrong
tool, or schema-invalid arguments are catastrophic. Each tool produces its own
`ExamObservation`, so an aggregate average cannot conceal a failing capability.
The scorer and daemon use the same strict runtime action parser. Teacher-forced,
externally supplied, and partial all-tool results cannot certify a model.

### 5. Separate candidates from active models

Training atomically writes `candidate-model.json`. It contains the canonical
shape, parameter count, tokenizer and tool-contract hashes, corpus identities,
covered families/languages/stages, and model/optimizer checkpoint hashes.

Certification requires checkpoint-bound, closed-loop quarantined evidence for
all seven stateful capabilities plus checkpoint-executed evidence for every tool
name in the exact captured contract snapshot. By default each knowledge family
must provide at least 100 sentinel episodes and each tool family at least 24
four-way sentinel cases; all require at least 85% complete passes and no more
than 1% catastrophic outputs. Activation additionally requires verified causal
foundation lineage with the exact language set `en`/`de`/`fr` and at least 500
million optimizer-visible tokens, preventing a smoke checkpoint from being
promoted even if evaluation was misconfigured. Training metadata must cover the
same complete family union, EN/DE/FR, and every workflow stage. The runtime
reloads the contract snapshot and independently checks that no tool family was
omitted before accepting `active-model.json`. The active manifest omits the
optimizer hash because Adam state remains a training/resume artifact, not a
serving dependency.

`akhomed` loads only `active-model.json`, verifies all referenced hashes and
paths, reconstructs the declared Burn model, verifies checkpoint step and
parameter count, and refuses an uncertified model. It never silently falls
back to another provider when `backend = "grug"` was selected.

Continual SFT starts only from an exam-certified `active-model.json`, through
the same path/hash/contract verifier used by serving. The new run copies the
exact tokenizer, loads model weights, resets Adam, preserves causal-foundation
lineage, and records a content-addressed `adapted_from` link containing the
parent checkpoint, contract, corpus and certification evidence hashes. Resume,
foundation warm-start and active-model adaptation are mutually exclusive. The
adapted output is a candidate in a different bundle and must pass the complete
current-contract sentinel matrix before activation.

### 6. Keep multimodality native but independently gated

The compact visual stem remains a peer input to the shared decoder. OCR output
feeds the same Seshat ingest, search, citation and notes workflow; it does not
create a second knowledge store. Provider OCR can be a labelled comparison or
fallback, never a prerequisite for local evaluation or serving.

ADR 103 supplies the candidate/runtime boundary: a frozen independent-renderer
audit, guarded stem/joint updates, checkpoint-bound OCR and text-retention
evidence, network-denied replay, fail-closed activation and ACP image plumbing.
The published visual checkpoint still proves only narrow clean-line overfit.
It is not joined into an active knowledge model until the ADR 103 unseen
text/font, EN/DE/FR, layout/corruption, independent-renderer, provider-free
replay and text-retention gates pass.

## Reproducible local flow

Capture the real learning-loop contracts:

```bash
cargo run -p seshat --example export_mcp_tools --no-default-features |
  cargo run --example export_akh_learning_contracts --features daemon -- \
  --mcp-stdin seshat --check-learning-loop > runs/knowledge/contracts.json
```

Acquire and prepare the pinned natural-language foundation. The fetcher streams
rows, projects only required columns, applies bounded output/disk admission,
and writes the final manifest only after every source file is complete and
hash-verified:

```bash
uv run training/scripts/fetch_foundation_hf.py \
  --config training/configs/foundation-en-de-fr-v1.json \
  --out runs/knowledge/tool-native-v1/foundation-natural-v1

cargo run -p grug-data --release --example prepare_foundation -- \
  --train-jsonl NATURAL_EN_TRAIN --train-jsonl NATURAL_DE_TRAIN \
  --train-jsonl NATURAL_FR_TRAIN \
  --train-jsonl FOUNDATION_KNOWLEDGE --train-jsonl FOUNDATION_DOJO \
  --train-jsonl FOUNDATION_TOOLS \
  --eval-jsonl NATURAL_EN_EVAL --eval-jsonl NATURAL_DE_EVAL \
  --eval-jsonl NATURAL_FR_EVAL \
  --out runs/knowledge/tool-native-v1/foundation-corpus-v2 \
  --run-id babyllm-foundation-en-de-fr-tools-v2 \
  --allow-language en --allow-language de --allow-language fr

cargo run -p grug-train --release --features cuda \
  --example train_foundation -- \
  --manifest runs/knowledge/tool-native-v1/foundation-corpus-v2/foundation-corpus.json \
  --artifacts-dir runs/knowledge/tool-native-v1/foundation-model-small-v1 \
  --model small --steps 50000 --batch-size 8 --bpe-vocab-size 8000 \
  --warmup-steps 1000 --eval-every 500 --ckpt-every 1000
```

Generate disjoint stateful and all-tool SFT streams, then warm-start a
candidate from the verified causal foundation:

```bash
cargo run -p grug-data --example knowledge_workflow -- training \
  --contracts runs/knowledge/contracts.json --seed 42 --episodes 500 --sft \
  > runs/knowledge/train.sft.jsonl

cargo run -p grug-data --example knowledge_workflow -- training \
  --contracts runs/knowledge/contracts.json --seed 4242 --episodes 100 --sft \
  > runs/knowledge/eval.sft.jsonl

cargo run -p grug-data --example prepare_tool_curriculum -- \
  --contracts runs/knowledge/contracts.json \
  --out runs/knowledge/tool-contract-sft --run-id tool-contract-sft-v1 \
  --train-examples-per-tool 48 --eval-examples-per-tool 12 \
  --offered-tools 4

cargo run -p grug-train --release --features cuda \
  --example train_knowledge -- \
  --contracts runs/knowledge/contracts.json \
  --train-jsonl runs/knowledge/train.sft.jsonl \
  --supplemental-jsonl runs/knowledge/dojo.train.sft.jsonl \
  --supplemental-jsonl runs/knowledge/tool-contract-sft/train.sft.jsonl \
  --eval-jsonl runs/knowledge/eval.sft.jsonl \
  --supplemental-eval-jsonl runs/knowledge/dojo.eval.sft.jsonl \
  --supplemental-eval-jsonl runs/knowledge/tool-contract-sft/eval.sft.jsonl \
  --foundation-bundle runs/knowledge/tool-native-v1/foundation-model-small-v1 \
  --artifacts-dir runs/knowledge/model-small --model small --epochs 8
```

When the live or planned-to-ship tool ABI changes, first capture the new exact
descriptors (planned MCP tools may be supplied through the existing `--mcp`
descriptor input), then materialize a verified delta. This command refuses an
identical or metadata-only change instead of creating an unnecessary update:

```bash
cargo run -p grug-data --example prepare_tool_contract_delta -- \
  --previous-contracts runs/knowledge/model-small/contracts.json \
  --contracts runs/knowledge/contracts-next.json \
  --out runs/knowledge/tool-contract-delta-v2 \
  --run-id tool-contract-delta-v2 --offered-tools 4

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

For a drift seen by a running Grug-backed akh, the two inputs are already
materialized at:

```text
$XDG_DATA_HOME/akh-medu/workspaces/AKH/model-contract-drift/delta-HASH/
  previous-contracts.json
  contracts.json
  contract-delta.json
  manifest.json
```

Pass that entry's `previous-contracts.json` and `contracts.json` to the same
command. The generator recomputes the delta; it does not trust the journal's
summary alone.

The base replay files in that example retain all seven stateful knowledge
families, all workflow stages and EN/DE/FR. The delta bundle itself guarantees
all current tool families plus targeted/absence examples; it is supplemental,
not a substitute for knowledge-workflow retention. Certification is repeated
against `contracts-next.json` and the new candidate checkpoint.

For each canonical family, generate a quarantined sentinel and execute it
against the candidate. The exam's checkpoint hash is the
`checkpoint_model_hash` from `candidate-model.json`:

```bash
cargo run -p grug-data --example knowledge_workflow -- exam \
  --contracts runs/knowledge/contracts.json \
  --family knowledge.acquire_textbook --tier sentinel --run-id release-1 \
  --seed 9001 --difficulty 1 --samples 100 \
  --checkpoint-step STEP --checkpoint-hash MODEL_HASH > sentinel-acquire.json

cargo run -p grug-train --release --features cuda \
  --example eval_knowledge_exam -- \
  --bundle runs/knowledge/model-small --exam sentinel-acquire.json \
  --evidence-out evidence-acquire.json \
  --predictions-out predictions-acquire.json
```

Generate the all-tool sentinel for the same candidate and execute it through
the same runtime loader. `--observations-out` can be fed directly to
`adaptive_exam_controller step` during training:

```bash
cargo run -p grug-data --example adaptive_exam_controller -- init \
  --run-id tool-steering-1 --generator-version akh-tool-contract-exam-v2 \
  --seed 51043 --contracts runs/knowledge/contracts.json \
  --min-samples-per-window 24 --enable-precursor-steering \
  > controller-tools.json

cargo run -p grug-data --example generate_tool_contract_exam -- \
  --contracts runs/knowledge/contracts.json --out sentinel-tools.json \
  --run-id release-1 --checkpoint-step 0 \
  --checkpoint-hash unbound:immutable-sentinel \
  --tier sentinel --difficulty 3 --samples-per-tool 24 --offered-tools 4

cargo run -p grug-train --release --features cuda \
  --example eval_tool_contract_exam -- \
  --bundle runs/knowledge/model-small --exam sentinel-tools.json \
  --evidence-out evidence-tools.json \
  --observations-out observations-tools.jsonl \
  --predictions-out predictions-tools.json
```

The lower eligibility floor is explicit and matches the 24 cases emitted for
each tool; it does not weaken the 100-case stateful workflow gate. Precursor
steering is enabled here because its definition is narrow and directly
actionable: the checkpoint chose the right offered tool and emitted valid
arguments, but copied at least one requested value incorrectly. Promotion still
depends only on the separately quarantined sentinel thresholds.

When the controller selects a knowledge frontier, materialize that exact plan
with `knowledge_workflow exam`, then pass the resulting frontier exam to
`knowledge_workflow sibling`. The latter replays and verifies the exam, derives
a separate seed namespace, rejects episode/source overlap and prompt similarity
above 0.80, and writes a content-addressed targeted SFT bundle without copying
the exam into training. Sentinel, rolling-probe and audit inputs are rejected.

A selected `tool_contract.NAME` frontier has the equivalent
`tool_contract_sibling` path. Pass `--family tool_contract.NAME` to
`generate_tool_contract_exam`; that materializes only the selected family's
cases while retaining the complete live tool catalog as distractors. Omitting
the flag keeps the historical full-catalog moving-exam bytes/fingerprint. A
full sentinel instead uses the explicit `unbound:immutable-sentinel` checkpoint
and a checkpoint-independent fingerprint; evaluation binds its report to the
executing candidate. The generator refuses partial sentinel or audit suites.
The sibling path first
replays the complete materialized targeted exam from its declared generator
inputs, then targets only `NAME` using a
separate multilingual prompt bank and seed namespace. Exact request and
prompt-target overlap are zero-gated, the same 0.80 trigram ceiling applies,
and even a zero-argument tool remains separable. A historical live 50-tool
smoke targeting `agent_list` produced 12 records with maximum request trigram
Jaccard 0.293447 and zero exact overlap; this is generator evidence, not model
quality.

When a statistically eligible controller decision is `steer`, the sibling can
start a new bounded candidate segment. `train_knowledge` requires the complete
controller-step artifact, exact frontier exam, sibling manifest and sibling
SFT together. It reproduces the controller's `next_plan`, refuses any `stop`,
replays the exam and sibling, verifies every content hash/separation gate, and
then restores the candidate model **and Adam state**. Global checkpoint steps
remain monotonic. The new candidate carries a `steered_from` lineage binding
the parent, controller, frontier and sibling; a later same-segment `--resume`
preserves the original segment start. No `watch` or `green` decision can be
misrepresented as authorization to change the mixture. A sibling can occupy at
most five percent of the segment's records; exact base/sibling counts and the
parts-per-million share are verified again from candidate lineage.

Text checkpoints are published through an atomic directory rename and carry
their own model/Adam content hashes. The loader remains compatible with
already-written schema-v1 checkpoints, while new schema-v2 checkpoints verify
both hashes before deserialization. In-place resume is limited to the exact
checkpoint referenced by the still-uncertified candidate, preserves the prior
run descriptor under `run-history/`, and refuses changed corpus, optimizer or
evaluation parameters (increasing only the epoch horizon is allowed).

`run_tool_native_adaptive_loop.py` closes the bounded operational loop. Every
source-head v2 window scores one controller-selected frontier plus frozen
sentinels for all seven workflows and the complete tool snapshot. It continues
the exact corpus for `watch`/`green`, creates a new evidence-bound segment only for an eligible
frontier `steer`, and halts before training on any `stop`. Controller decisions
carry their source tier, preventing a sentinel warning from authorizing a
frontier sibling. The supervisor writes replayable window/segment artifacts and
contains no certification or active-pointer operation. Its launch manifest
pins executable hashes and budgets; `--resume` reconstructs only committed
windows or retries evaluation in a new append-only attempt directory. A model
update or controller rebind interrupted before ledger commit remains
fail-closed instead of risking a duplicate training step.

After equivalent workflow evidence exists for all seven families and the
all-tool matrix covers the complete snapshot:

```bash
cargo run -p grug-train --release --example certify_knowledge_model -- \
  --bundle runs/knowledge/model-small \
  --evidence evidence-acquire.json \
  --evidence evidence-restart.json \
  --evidence evidence-revision.json \
  --evidence evidence-fallback.json \
  --evidence evidence-abstain.json \
  --evidence evidence-textbook-inventory.json \
  --evidence evidence-note-navigation.json \
  --tool-evidence evidence-tools.json
```

An akh opts in explicitly:

```toml
backend = "grug"
model = "knowledge/model-small"
```

The daemon must be built with `--features server,grug-backend`. A relative
model path resolves below Akh-Medu's shared models directory; an absolute path
is supported for an explicit development run.

## Consequences and current boundary

The running paid pipeline is intentionally frozen at its captured v1 boundary:
50 tools with contract hash
`blake3:05b1ca2310b5e8352d7b3bfd5e8d3329fdca59805b4b12ff06bec3700cf0a285`
and five workflow families. It has not been restarted or rewritten. The exact
source-head export is the next adaptation boundary: 53 tools with contract hash
`blake3:3d6e18bb74765cab13d35901d0a10a7834077f302cc578941ca06ef56223700a`
and generator `akh-knowledge-workflow-v2` with seven families. A reproducible
21-episode smoke expanded to 92 train and 92 evaluation records, covered all
three languages and all 14 stages, and emitted seven separate immutable
sentinels. The companion `akh-tool-contract-sft-v2` smoke covered all 53 tools,
kept every decision surface at exactly four offered tools, generated only
schema-valid note IDs and revision tokens, and gives note-write targets the
earlier read/retrieval results required by the host gates. These bytes must
enter through the versioned contract-delta,
adaptation and full recertification path after the current run; they are not
evidence that the running checkpoint already knows the new tool.

The repository now contains the real acquisition, causal pretraining,
tool/knowledge SFT, adaptive evaluation, promotion, and daemon-loading path. A
live CUDA smoke proves that a causal checkpoint can warm-start tool SFT and load
through the shared runtime; it is not a capable or active checkpoint. The first
700,000/7,000 natural corpus acquisition has completed and is content verified.
After exact/near-duplicate quarantine and addition of the knowledge, Dojo and
all-tool bridges, the production v2 manifest contains 709,917 training records
and 7,000 held-out records. It observes exactly `en`, `de` and `fr`; aliases such
as `english`, `german` and `french` are rejected by both preparation and
certification. A usable model still requires the full accelerated run, seven
passing closed-loop workflow matrices, complete all-tool sentinel evidence, and
the separate multimodal gates above. The native OCR trainer, full-audit
evaluator, certifier and ACP runtime are implemented, and the 1,050-case Noto
audit is frozen and separation-validated. No multimodal checkpoint has passed
that audit yet, so the current boundary remains candidate infrastructure rather
than an active visual model.

Autoregressive inference now uses a runtime-only per-layer key/value cache when
the compacted prompt reserves the requested completion budget; generic callers
that need the old left-truncation semantics retain the uncached compatibility
path. Cached prefill/decode logits and greedy output are parity-tested against
full-sequence forward passes on both CPU and CUDA, and the cache adds no module
parameters or checkpoint fields. A live 34.5M-parameter checkpoint probe with
a 502-token prompt and 16-token completion measured a 9.0x CPU speedup (12.22 s
to 1.35 s); CUDA was only 1.06x faster on a 1,502-token/32-token probe because
small one-token kernels remain launch-bound. Those measurements justify the
CPU serving change but are not a general throughput claim; release evidence
must continue to record the exact backend and workload shape.
