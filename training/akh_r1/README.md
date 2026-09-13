# Akh R1 development episode exam

## Status 2026-09-13

This directory is a curated recovery of the Python MLX/episode harness that
was developed on the archived `feat/akh-r1-episode-exam` branch. It exercises
two frozen local base models — `mlx-community/Qwen3-0.6B-4bit` (MLX inference
and LoRA training) and `Qwen/Qwen3-0.6B` (tokenizer/reference parity) — against
the real Rust Engine bridge inside disposable, network-disabled fixtures. Local
runs land under `runs/akh-r1-*`; the receipts for the run that validated this
recovery live in
[`training/results/akh-r1-local-mlx-evidence-20260913/`](../results/akh-r1-local-mlx-evidence-20260913/).

This harness is **reference/bootstrap tooling**, not the ongoing training
driver. That role is moving to the Rust Burn harness (`grug-train`, see
`crates/grug-train`) per the
[P6 local-training/neurosymbolic plan](../../docs/ai/plans/2026-09-13-phase51-p6-local-training-neurosymbolic.md).
Keep these scripts around for MLX-side experiments, episode/world fixtures and
the Rust-engine bridge protocol they exercise, but new training-loop work
belongs in `grug-train`.

### What was dropped from the archived branch, and why

The archived branch carried ~450 files, most of them belonging to a retired
"research loop" that recorded checkpoints, allowances, freezes and other
governance bookkeeping for an autonomous-run process ADR 105 has since
replaced with board-routed Anima work items. Recovering that machinery would
just reintroduce the failure mode ADR 105 exists to prevent, so it was left
behind:

- **All ~200 `audit_*.py` review scripts** (except `audit_compact_preflight.py`,
  which `run_typed_registry.py` imports directly as a real runtime dependency,
  not bookkeeping — it is kept for that reason despite the name).
- **`research_program.json` and `research_monitor.py`** — the retired
  self-monitoring research loop; `research_monitor.py` in particular must
  never come back (see the 2026-09-13 Codex-session post-mortem and ADR 105).
- Scripts whose only purpose was bookkeeping for that loop (recording
  "checkpoints", "allowances", "freezes", "dispositions") and any module that
  only those scripts imported.
- Top-level contract/coverage-adjacent JSON files not referenced by any kept
  script and not matching the contract/coverage naming convention
  (`breadth-distributions.json`, `recovery-pack-mapping.json`,
  `temporal-assistance-profile.json`).

What's kept is every MLX-driving script, every script that talks to the Rust
engine bridge or symbolic/native/temporal/causal/world fixtures, the
curriculum/lesson builders they depend on, their transitive local imports (by
`import`/`from` closure), and the top-level `*.md`/`*.json` contract and
coverage documents. See the recovery commit for the full accounting.

The [public-interface breadth results](../results/akh-r1-public-breadth-20260912-RESULT.md)
add four authored domains and 14 executed reference phases. `run_public_breadth.py`
uses the same `--out`, `--bridge`, `--image` and `--lock` arguments as the R0
reference runner; `audit_public_breadth.py RUN_DIRECTORY` audits its artifacts.
Candidates communicate through `public_episode.py` over JSONL from isolated
containers. This is still reference evaluation, with no model capability claim.

The [research program](../../docs/ai/plans/2026-09-12-akh-capability-and-model-size-research-program.md)
controls new experiments. R0 now has a [twelve-family contract and coverage inventory](../../docs/ai/plans/2026-09-12-akh-R0-episode-contract.md)
and `run_r0_reference.py` for three authored integration controls. These execute
without a neural model and cannot be reported as model or release passes.

```sh
python3 training/akh_r1/run_r0_reference.py --out /path/to/new-run \
  --bridge /path/to/akh_symbolic_episode_bridge --image sha256:PINNED_IMAGE \
  --lock /path/to/r1-local-mlx.lock
```

This is the first executable slice of the [integrated R1 plan](../../docs/ai/plans/2026-09-11-akh-350m-graph-grounded-release-plan.md).
It exercises a frozen MLX model, the real Rust Engine and its persistent graph,
and real Unix subprocesses in a disposable, network-disabled Docker container.
The current world is a development fixture, **not a sealed release exam**.
The [first teaching/adapter results](../results/akh-r1-curriculum-20260911-RESULT.md)
record the executed corpus, observed improvements and remaining failures.

The candidate must discover two unfamiliar command contracts, diagnose a pending
backup, act on the correct worker, preserve archives and verify a fresh result.
An operator correction prohibits restarts and requests progress visibility.
Follow-on phases start fresh model and Engine processes: a different job with
documentation offline and the acquired graph, the same job with an empty graph,
and a changed command version with the acquired graph. There is no weight or
adapter update during a run.

## Run

Build the isolated Engine bridge (no daemon, tracker or existing workspace graph):

```sh
cargo build --locked --no-default-features --example akh_episode_bridge
AKH_EXAM_BRIDGE="$PWD/target/debug/examples/akh_episode_bridge" \
  python3 -m unittest discover -s training/akh_r1 -p 'test_*.py' -v

python training/akh_r1/run_exam.py \
  --model /path/to/local/mlx-model \
  --bridge "$PWD/target/debug/examples/akh_episode_bridge" \
  --image akh-medu/native-ocr-offline:recovery-20260910 \
  --out /path/to/new-run-directory --protocol flat
```

The Python environment needs MLX and MLX-LM. The fixture image must already be
present and provide Python 3, bash, timeout and the listed Unix filters. The
image ID, model/tokenizer/adapter bytes, runner sources and bridge binary are
hashed in the manifest. The output directory must be new; a rerun cannot
overwrite an earlier graph or score. `--adapter` selects a separate diagnostic
adapter; `--portuguese` changes the operator correction language. The current
cached Qwen3/Dojo models are external controls, not an adopted release foundation.

Sampling defaults follow the Qwen3 model card: non-thinking temperature 0.7,
top-p 0.8 and top-k 20; `--thinking` uses 0.6, 0.95 and 20. The sampling seed is
recorded separately from the world seed. Use `--temperature 0` only for an
explicit greedy control. Qwen warns against greedy thinking because it can loop.
Reasoning text stays in raw receipts but is excluded from later conversation
inputs. These bounded action budgets are not an exhaustive model benchmark.
See the [upstream guidance](https://huggingface.co/Qwen/Qwen3-0.6B#best-practices).

`--demonstrations` adds an explicitly synthetic interface lesson about a different
command. It is an assistance condition and never populates the real graph.
`--phases acquire` runs only acquisition. The default also runs both cold phases
and revision. Exact sources and bridge bytes are retained in each run's `source/`
directory, and child phases execute those frozen copies.

`--contract-family subject_first_colon --phases acquire,cold_graph,cold_empty`
selects one of the new pilot contract families. It requires `--protocol flat`.
These families vary argument grammar and inventory column order/delimiters.
They do not yet implement a changed-version phase. The original development
family still provides the revision control.

`--repeat-feedback` is a separately marked assistance condition. It reports
when the same request returns the same result, using only the public request
and response. It does not select a next tool, bind an argument, alter a receipt
or block execution. The raw response is retained beside any added feedback.

`--validate-sources` is a separate runtime condition: memory-write sources must
match an earlier visible tool-response `evidence_id` in the same episode. Invalid
links are rejected before the Engine is called, for both memory interfaces.
Nothing selects or rewrites a fact or its source. A link can resolve to an
irrelevant or failed observation, so this does **not** establish semantic support.
Memory metrics distinguish query API success, empty results, returned triples
and accepted writes with resolved links; returned triples can still be useless.
Use `--receipt-namespace adapter-heldout` in matched comparisons so output folder
names do not change the receipt IDs seen by the model. The default retains the
older output-basename behavior for compatibility.

## Whole-episode teaching pilot

`build_curriculum.py` executes authored oracle lessons through actual isolated
Unix processes and the real persistent Engine. The teacher knows the contract
family; these successes are teaching-data verification, not model performance.
Every acquisition is paired with a new Engine process and a new conversation
using its copied graph while command documentation is offline.

```sh
python training/akh_r1/build_curriculum.py --out /path/to/new-curriculum-run \
  --bridge /path/to/akh_episode_bridge

# Run with the installed MLX-LM Python environment.
python training/akh_r1/preflight_curriculum.py \
  --corpus /path/to/new-curriculum-run/corpus --tokenizer /path/to/local/mlx-model \
  --max-seq-length 4096 --out /path/to/new-curriculum-run/tokenizer-preflight.json

python training/akh_r1/train_pilot.py --model /path/to/local/mlx-model \
  --corpus /path/to/new-curriculum-run/corpus \
  --preflight /path/to/new-curriculum-run/tokenizer-preflight.json \
  --out /path/to/new-adapter-pilot --iters 80 --timeout-seconds 900
```

The initial pack has four command-contract families within **one backup-progress
workflow**, three seeds per family, and acquisition/cold pairs: 24 episodes.
Two complete families go to train, one to validation, one to pilot test. All cold
and language variants stay with their family. These are development splits,
not a sealed release benchmark. Seed parity chooses English or simple Portuguese
goals/corrections; it does not establish language competence.
The first pack reuses world seeds across its grammar families, so its teacher
validation/test entities overlap training entities. Its live model comparison
uses a different seed and audits entity overlap. The full release pack still
requires disjoint entity sets, documents and workflow compositions.

`episodes.jsonl` retains events, chronology, provenance, correction, curriculum
annotations and examiner effects. Each chat row contains only prior events and
one next assistant target. Examiner truth and annotations are never rendered.
Actual tool responses expose receipt IDs before later sourced memory writes;
the compiler rejects a teaching citation to an invisible or future receipt.
It rejects failed positive lessons, split changes and cold-graph lineage that
crosses a family boundary. The compiler checks structural evidence linkage;
it is not a general semantic entailment judge.

Preflight exercises the installed MLX chat dataset with prompt masking, verifies
the native/non-thinking prefix and supervised target, checks input hashes, and
rejects truncation. The pilot launcher requires those exact corpus/tokenizer
bytes, masks earlier messages, trains a separate rank-8 adapter, records input
and output hashes, and stops after the update or wall-time budget. The model
path is an external diagnostic control; this does not select a release foundation.
Loss reductions must be followed by actual held-out command-family episodes and
cold graph/empty-graph comparisons. Existing failed development examples remain
diagnostic evidence and are not copied into the teaching data.

`train_pilot.py --objective tool_name_balanced` tests a separate training
objective: half the loss is the mean cross-entropy on the target function name;
half is the mean over remaining supervised tokens. Target spans come from the
actual tokenized native call. Prompt tokens receive no weight. This uses the
installed MLX trainer's loss/iterator hooks in an isolated process, preserves its
batch schedule and supervised-token range, and does not edit installed packages.
The entrypoint and loss implementation are frozen in the pilot output. Its
reported loss is not numerically comparable to ordinary mean token loss.
Inference, tool execution and release gates are unchanged.

`probe_choices.py --model MODEL --adapter ADAPTER --corpus CORPUS --out NEW_DIR`
checks eight selected decisions from two training episodes using their correct
prior histories. It records generated tool/argument matches, per-token loss,
function-name loss and training/inference tokenization equality. This is a
teacher-context diagnostic; it executes no tools and is not an episode success
or held-out capability result. Use it to investigate a low-loss/failing-rollout
gap before attributing that gap to model size.

`build_repair_curriculum.py --base-corpus CORPUS --out NEW_DIR --bridge BRIDGE`
adds executed repair episodes only to the original training families. They show
how to continue after a redundant version query/report, store documented facts
after an empty graph lookup, retrieve a newly stored identifier mapping, and
correct a rejected command before making and verifying the repair. Every
injected failure must fail without changing the fixture. These are authored
recovery examples, not copies of model evaluation traces.

An assistant event with `train_target: false` remains in the chronological
history but never becomes a supervised next-action row. Its tool result stays
visible to subsequent decisions. The exclusion flag, annotations and examiner
fields are not rendered into messages. Prompt masking must remain enabled so
that earlier mistakes are context, not targets. Original `valid.jsonl` and
`test.jsonl` must remain byte-identical, and the parent corpus is hashed without
being modified. The compiler still requires every completed teaching episode
to pass the actual effect checks.

## Bounded exposure comparison

`train_pilot.py --full-coverage --iters 396 --timeout-seconds 3600` explicitly
enables a larger diagnostic budget, capped at 512 updates and one hour. It must
budget at least one pass over the supplied training rows. Ordinary pilots retain
their 128-update/1800-second caps. SIGTERM stops the owned training child and
preserves any completed checkpoints.

`run_exposure.py prepare` freezes a plan, source, model, corpus, preflight and
bridge hashes into a new output directory. `run_exposure.py execute PLAN` runs
exactly two 396-update conditions, first the original corpus and then the repair
corpus, each starting from the same base model. Each gets the eight original
teacher-context probes, raw incident comparisons at seeds 83 and 107, and a
separate source-validation comparison at seed 83. Both worlds must have no
training identifier overlap. Receipt namespace and sampling settings are fixed.

The supervisor uses an exclusive local lock and atomic status updates every
15 seconds. It stops on changed inputs, process/numerical failure, disk below
5 GiB, stage deadlines or a 9000-second total deadline. It refuses to replay an
already started plan. Model failures in a completed diagnostic are recorded and
do not cause retries. Each exam phase has an owned process group and deadline;
interruption cleans up its own fixture. Outputs remain local, with no Runpod
provisioning, automatic installation or promotion. Check `status.json` and its
PID before acting on a later hourly wake; a prepared plan or started PID alone
is not proof of training progress or completion.

## Interfaces and evidence

- `json`: custom strict JSON envelope with nested argv arrays.
- `native`: the model's installed native tool-call template and nested argv arrays.
- `flat`: the native template with a command string and a flat single-fact writer.
  The host parses quoting/pipes and maps the four fact fields into `memory_write`.
  It does not choose commands, identifiers, relations, values or missing arguments.

Every turn records the available messages, raw output, parsed action, any syntax
projection, real process exit codes and tool results. `wire_request` and
`engine_request` make wrappers inspectable. Malformed output is returned as an
error rather than repaired with domain assumptions. No expected action, hidden
fault label, grader judgment or later outcome is inserted into earlier inputs.
Direct graph writes go through production `MemoryWriteTool`; recall, query and
retraction use the corresponding production tools. `exam_export` is available
only to the examiner, not advertised to the model.

Each phase has `trace.jsonl`, before/after graph exports and fixture states,
`engine.log`, and `scores.json`. The run includes `manifest.json`, `status.json`
and `results.json`. Model and adapter hashes are rechecked at completion.

Effect success requires restored progress, correct target, no restarts,
preserved archives, unchanged unrelated jobs and a successful post-action watch.
Valid JSON counts do not assert correct arguments or a correct decision. A
no-op may satisfy preservation checks but still fails effect success. Pipeline
receipts include every exit code so a successful filter cannot conceal a failed
upstream command. Integration tests deliberately exercise these false-pass cases.

Semantic graph precision, source support, correction uptake and Dojo quality are
reviewed separately; unscored fields are null. An empty acquired graph makes the
cold ablation uninformative about memory benefit. MLX peak allocation is **not**
whole-Akh memory or a Mac mini performance certification.

Still required by the full plan: additional independent workflows, a sealed
release pack, High Grug versus plain-text pairing, the full eight-kata and
six-consultation curriculum/exams, broader Portuguese checks, own-foundation
capacity pilots, backend parity, quantization and complete release gates.

`probe_choices.py --split test --all-decisions --max-decisions 32 --episodes ...`
measures all selected next actions under the executed teacher contexts, including
later argument binding, repair and verification decisions. It preserves the
original eight-probe default. Expected next actions are withheld from generation;
no generated tool call executes and no episode success is implied. It refuses
silent truncation when the selected episodes exceed the declared decision budget.
This separates context-conditioned command transfer from autonomous discovery,
memory construction and retrieval. Existing development splits can share entity
IDs with training, so this condition is not a fresh-entity or release certificate.

## Command binding kata

`run_binding_kata.py prepare` creates six one-action cases in the two training
command families, using real help/version/job/worker receipts and persisted Engine
records. It reopens the KG and executes twelve oracle references through both
executors. `evaluate --out PATH` runs both frozen adapters across four conditions:
prose or structured records, each paired with raw commands or template invocation.
The fixed budget is 48 candidate actions, 160 generated tokens each, 900 seconds,
an exclusive local MLX lock, and a five-GiB disk floor. No weight updates occur.

`command_templates.py` extracts the selected literal help line, stores it as a
versioned record, looks up the candidate-selected subject in the real graph,
substitutes exactly the candidate-supplied bindings as individual argv tokens,
and rejects version mismatches. It does not select an operation or target. The
kata oracle selects documentation lines during setup; the candidate does not
learn or locate these records in this diagnostic. Only standalone placeholders
are supported. Exact source text does not independently prove a useful command.

Candidate calls run in the existing disposable, network-disabled fixture. Effect
checks require the requested bounded mutation or documented fresh read, preserving
unrelated jobs/workers and restart counts. Merely printing matching JSON cannot
pass a read check. Input messages exclude examiner targets and future state.
These are assisted component kata, not whole episodes, autonomous graph learning,
High Grug certification or release evidence. Keep source acquisition, subject
retrieval, operator correction and complete incident verification as separate
remaining gates. Read results by operation and condition, not just an aggregate.

`prepare-flat --parent COMPLETED_FOUR_ARM_RUN --out NEW_RUN` prepares an additional
24-action control using the exact same case files, observation receipts, KG
snapshots and frozen adapters. It replaces the nested bindings object with flat
`subject` and `argument` strings. The selected stored template must have exactly
one placeholder (or none, with an empty argument). Projecting that supplied value
to its sole slot does not choose the operation or infer the target. The parent
must be completed and all its input hashes must still match.

`prepare-untuned --parent COMPLETED_FOUR_ARM_RUN --out NEW_RUN` adds the unchanged
parent model without an adapter across all six conditions (36 decisions). It
reuses the exact case observations and KG snapshots. This distinguishes adapter
transfer/retention problems from parent-model and interface limitations.

## Executed binding lessons and bounded training

`build_binding_lessons.py --out NEW --bridge PATH --rehearsal ORIGINAL_CORPUS`
builds 60 executed reference cases: 48 from the two training command families,
six validation and six test cases from their existing separate families. The
affected entity rotates; new literal entity identifiers are disjoint across
splits. Opaque command subjects change between cases, and source receipt IDs
hide the examiner's operation label. The two flat-interface representations
yield 96 training decisions and 12 each for validation/test. The full original
corpus is preserved as rehearsal, giving 288/108/108 combined decisions for the
current inputs. Original source corpus files remain unchanged.

`audit_binding_lessons.py ROOT` checks teaching targets against executed actions,
actual effects, source reconstruction, split separation, complete rehearsal and
frozen files. Run `preflight_curriculum.py` on the combined corpus before training.
This remains assisted component teaching, not candidate-led graph acquisition.

`run_binding_training.py prepare --lessons ROOT --preflight JSON --model PATH
--out NEW --lock-file PATH` freezes one fresh local adapter run and seven serial
stages. Before and after training, it executes 36 binding decisions on identical
prepared cases, the original eight teacher-context probes, and the three raw
incident phases. Six training cases diagnose acquisition of the new interface;
the twelve validation/test cases use command families excluded from training.
Subject selection, target binding and actual effects are scored separately.

`run_binding_training.py execute NEW/plan.json` owns the existing exclusive local
MLX lock, reports a heartbeat every 15 seconds, refuses replay, and preserves
checkpoints on failure. Training is one pass over 288 rows using token-mean loss,
at most 3600 seconds; the complete experiment is capped at 6600 seconds with a
five-GiB free-space floor. Source, weights, corpus and evaluator hashes are frozen.
No retry or promotion follows negative results. The full R1 work and independent
release evaluation remain open, including graph acquisition and cold retrieval.

## Candidate-led registry cycle

`run_registry_cycle.py prepare --out NEW --model PATH --adapter PATH --bridge PATH
--lock-file PATH` prepares a three-phase frozen-weight development check. It first
executes reference acquisition and cold retrieval through the real Engine and
checks the empty-registry/docs-offline control. `execute --out NEW` runs candidate
acquisition, cold graph, and cold empty phases in separate processes, with 24 turns
per phase, 192 new tokens per turn, 8192 context tokens, a 600-second phase limit,
1800 seconds overall, the shared MLX lock and five-GiB disk floor.

The candidate receives only the incident goal and installed command names.
`learn_command(command, line_number)` copies the candidate-selected syntax line
from help/version receipts already observed through `terminal`. It allocates a
stable syntax/version subject ID and persists the exact record. Re-registration
keeps the first source and avoids ambiguous duplicate facts. `list_commands`
retrieves actual stored registry records, and `invoke_command(subject, argument)`
uses the already tested literal binding and installed-version check. The candidate
chooses source lines, operations and targets; no oracle graph or target observation
is inserted into its context. The helper reads the registry via structured Engine
export, not model-authored descriptions or the human-readable query renderer.

Raw effect success is reported separately from graph use. A registry success
requires an actual registry-backed mutation and later verification; the cold phase
also requires a nonempty registry read before mutation. Empty acquired knowledge
makes the cold contrast uninformative. This scoped registry is not a general KG
semantics evaluator, and the new registration/retrieval tools were not in the
binding training corpus. Preserve the traces as a test of transfer to this
interface, not a release or model-capacity verdict.

`prepare-seeded --parent COMPLETED_CYCLE --out NEW` isolates cold retrieval and
action when candidate acquisition produced an empty registry. It copies the
reference's persisted records into a fresh process, leaves target observations
absent, and runs the same cold-world goal with the same frozen adapter. The plan
and scores explicitly label the oracle-prepared memory. It is a separate assisted
control and cannot be reported as successful candidate knowledge acquisition.

## Executed registry workflow curriculum

`build_registry_lessons.py --out NEW --bridge PATH --rehearsal BINDING/corpus`
executes eight acquisition/cold-graph/cold-empty cycles across disjoint command
families. The cold graph opens a new Engine with the prior persisted command
records and different job/worker identifiers. Each workflow includes actual
discovery, registration, retrieval, target observation, bounded repair and fresh
verification. Failed prerequisite and invalid-subject calls are context only;
the supervised target is the subsequent corrective action. Empty memory with
offline help ends with an accurate blocked report. These are authored reference
lessons, not model successes or general emotional-intelligence certification.

`audit_registry_lessons.py NEW` checks effects, exact prior help/version sources,
causal histories, excluded error targets, cold persistence, family/entity split
separation, and complete previous rehearsal. Run the actual tokenizer/mask
preflight before training. Version 1 has 104 new training decisions plus all
288 prior decisions; each held-out split contains all 108 prior rows plus 52 new
workflow decisions. No previous evaluation artifact is overwritten.

`run_registry_training.py prepare --lessons NEW --binding-lessons BINDING
--baseline COMPLETED_REGISTRY --preflight FILE --model PATH --out RUN
--lock-file PATH` freezes one fresh 392-update token-mean adapter condition.
`execute RUN/plan.json` owns the shared MLX lock, caps training at 3600 seconds
and the whole experiment at 7800 seconds, and retains checkpoints on failure.
It runs unassisted acquisition/cold-memory checks, a separately labelled
oracle-seeded cold control, previous binding cases, teacher-context probes and
raw incident controls. The completed previous registry run is frozen as a
development baseline. There are no automatic retries, promotions or cloud GPU
allocations, and repeated development checks are not independent release tests.

## Typed registry choice diagnostic

`run_typed_registry.py prepare --out NEW --parent COMPLETED_REGISTRY --lock-file
PATH` freezes a comparison against a completed free-generation registry run.
`execute --out NEW` evaluates acquisition, cold graph, cold empty and a separate
oracle-seeded cold phase with the same frozen adapter and seed. All four phases
use separate processes and the real Engine/isolated fixture. Limits are 24 turns,
192 generated tokens, 8192 context tokens, 600 seconds per phase and 2400 seconds
overall, with the shared local MLX lock and five-GiB disk floor.

`TypedChoices` receives only installed command names, the public goal and actual
successful tool responses. Help/version observations supply possible source-line
registrations; registry responses supply callable records; JSON fields or typed
inventory headers supply all observed identifiers. It never receives examiner
state or chooses the affected worker. The model sees the complete current menu
and a token-prefix trie limits output to its canonical calls, preserving model
logits among allowed continuations. Every executed request must belong to the
prepared turn menu. Traces retain each menu and its prior evidence for audit.

This combines menu visibility, typed values and constrained decoding. It also
restricts terminal use to documentation/version discovery and uses generic
report/finish text; it does not test arbitrary Unix composition or verbal claim
quality. The free baseline and full release gates remain intact. Reference
actions are first executed and then checked against menus rebuilt only from
their preceding public observations. A successful constrained call is still
scored separately from correct target selection, actual repair, verification
and cold graph dependence. No optimizer, retry or automatic promotion is used.

Add `--context-mode compact` with a completed full-history typed run as parent
to test receipt deduplication and a literal relation index with the same frozen
weights, menus and budgets. Each phase starts with an empty evidence view; only
returned tool receipts enter it. All observed job/worker relations remain visible
with their sources, including conflicts, and every original request/response is
exactly reconstructible. Operator messages are retained verbatim. The model still
selects operations and targets. This is an assisted state representation, not the
High Grug implementation or a model-size verdict.

Preparation checks exact receipt reconstruction and compares tokenizer counts for
both renderings of each prior trajectory without model inference. Candidate traces
also retain both actual compact prompt lengths and the full-history lengths for
that same trajectory. The artifact audit reconstructs every preceding receipt,
operator message, menu and state at every turn. Report token savings separately
from autonomous repair, cold memory use and actual verification; a shorter input
does not establish successful reasoning.

`run_compact_probe.py prepare --parent COMPLETED_TYPED_HISTORY --out NEW
--lock-file PATH`, followed by `execute --out NEW`, isolates next-action choices
at prior seeded-control decision indices 5, 8, 11 and 15. Both presentations
receive the identical preceding receipts and current menu. Greedy and sampled
decoding yield 16 independent decisions, with a 600-second total deadline and
the same model, adapter, shared lock and disk floor. No tools execute, generated
choices do not become later context, and no outcome labels enter the inputs.
These are selected development failures, not independent held-out episodes.

## Matched memory-to-action teaching

`build_memory_action_lessons.py prepare --out NEW --bridge PATH --rehearsal
PREVIOUS_REGISTRY/corpus --tokenizer PATH`, then `execute --out NEW`, freezes and
executes 12 authored reference episodes across disjoint command families. The
900-second preparation supervisor builds both corpora, audits them and checks
the real tokenizer/loss masks at a declared 8192-token limit. It runs no model
inference or optimizer and uses only isolated fixtures and the existing Engine.

Each cycle acquires command records, reopens the graph for a new incident, and
checks an empty-memory/docs-offline control. The cold-graph episode contains one
deliberately wrong worker action selected from observed inventory. That action
stays in context and fails the strict whole-episode gate; it is excluded from
supervised targets. The operator correction is followed by retrieval, fresh
binding, correct repair and verification. The corrective suffix has a separate
score; it never overwrites the full-episode failure.

`corpus-history` and `corpus-compact` contain byte-identical prior rehearsal and
the same new positive targets, menus and preceding receipts. Only the new input
presentation differs. The typed stop target is generic, matching the current
diagnostic interface; truthful authored finish statements remain in the reference
episodes and earlier rehearsal. Verbal outcome quality still needs its full R1
evaluation. A passing preparation is evidence about corpus construction, not
learned behavior, training feasibility at 8192 tokens or model certification.

`compare_memory_action_tokens.py NEW` compares actual inference-prompt lengths
for matched new rows, excluding unchanged rehearsal. It verifies matched targets
and frozen source/corpus inputs and records its own source hash. Report productive
episode cost separately from the deduplication benefit on repeated-read loops.

### Resource profile and paired training

`run_memory_action_profile.py prepare --out NEW --lessons COMPLETED_LESSONS
--model LOCAL_MODEL --lock-file SHARED_LOCK`, then `execute --out NEW`, runs two
scratch optimizer steps on each format's longest complete training row. It uses
the longest validation row, records actual peak MLX allocation and elapsed time,
and hashes the initial adapter tensors. Scratch weights remain profiling evidence.
Acceptance requires both peaks at most 11 GiB, identical initialization and
unchanged inputs. These short profiles do not predict complete-run duration.

`run_memory_action_training.py prepare --out NEW --lessons COMPLETED_LESSONS
--profile PASSED_PROFILE --model LOCAL_MODEL --baseline COMPLETED_TYPED_HISTORY
--binding-lessons COMPLETED_BINDING_LESSONS --lock-file SHARED_LOCK`, then
`execute --out NEW`, freezes and runs two fresh diagnostic adapters. Each receives
one complete corpus pass, batch one, rank 8, seed 83, a token-mean objective and
the full 8192-token limit. Before optimization, initial tensors must match the
accepted profile hash. `paired_batches.py` uses the same permutation of row
indices and validation sample for both formats, independent of token length;
the final audit requires identical order and exactly one visit per training row.
Padding and loss masks match the native MLX iterator; overlong rows fail.

The 12 serialized stages train each arm and then evaluate it with both typed
evidence formats, the previous 36 binding cases, eight teacher-context decisions
and three raw incident phases. Actual effects and assistance remain separately
scored. These are repeated development controls, not sealed release exams.

The detached supervisor holds the shared MLX lock, checks a 5 GiB disk floor,
caps each training arm at 3600 seconds and the complete pipeline at four hours,
and stops on process, input-hash or numerical failures. A reported peak MLX
allocation above 11 GiB also stops training; this is a log-based guard, not a
hard allocation limit. Interval and final checkpoints are retained. There are
no automatic retries or promotions. Completion requires the final artifact audit,
including actual initial tensor hashes, exposure order and evaluation receipts.

`run_memory_action_probe.py prepare --parent COMPLETED_PAIRED_RUN --out NEW
--lock-file SHARED_LOCK`, then `execute --out NEW`, checks all 27 positive
decisions in a declared training-family reference cycle and 27 in a held-out
development cycle for each adapter, using its own evidence format. It reuses the
audited teacher-context evaluator: 108 greedy, unconstrained generated calls,
no executed tools and no optimizer. Source/adapter/corpus hashes, exact selection,
tokenization, finite losses and complete result coverage are audited. The shared
lock, 5 GiB disk floor, 600-second per-condition and 2400-second total limits
apply. Report selection by expected tool and exact call, and distinguish this
supplied-history diagnostic from sampled, constrained autonomous episodes.

## Native symbolic completion

Build `akh_symbolic_episode_bridge` with `--locked --no-default-features` as a
separate executable; keep the original bridge and all frozen experiment binaries.
It exposes bounded native `RuleEngine` execution over the persistent Engine,
returning rule names, literal conclusions and matched antecedents. Rules are
versioned input artifacts with four iterations and 64 new triples as hard caps.

`run_symbolic_completion.py prepare --parent COMPLETED_PAIRED_RUN --out NEW
--bridge SYMBOLIC_BRIDGE --lock-file SHARED_LOCK`, then `execute --out NEW`,
compares 72 recorded model proposals completed by native rules, 18 symbolic-only
actions and 54 controls disabling command knowledge, disabling rules or changing
the installed command version. It executes actual fixture operations, with no
new neural generation or training. The controller has a 900-second budget and
5 GiB disk floor. Existing outputs cannot be replayed or overwritten.

`symbolic_policy.py` receives only the public goal, supplied observation receipts
and command records. Its explicit domain compiler recognizes the three tested
goal forms and documented effect descriptions; it does not consult hidden world
state, examiner action labels or scores. Command names and target identifiers
come from receipts. Native rules derive the operation and binding; the host
checks versions and executes literal argv. Stored records and input facts are
reopened in a fresh Engine process before inference. The artifact audit checks
effects, persistence, interventions and an independent literal/join proof replay.

Report the integrated action separately from model correctness and symbolic-only
success. An engine-completed action is valid system evidence, with its source of
knowledge and policy identified. This controlled one-operation replay does not
establish autonomous acquisition, arbitrary-language planning, a complete incident
episode or High Grug. The next integration must earn those separate claims.

`run_goal_handoff.py prepare --parent COMPLETED_SYMBOLIC_COMPARISON --out NEW
--lock-file SHARED_LOCK`, then `execute --out NEW`, generates 36 actual neural
effect selections across the same 18 public goals and two preserved adapters.
The model chooses only `observe_worker`, `resume_worker` or `observe_job` using
greedy constrained decoding. Native rules select the command and derive the
target from the cold graph; operator constraints independently authorize writes.
Actual fixture effects and independent native proofs are audited. This runner
shares the 900-second deadline, 5 GiB disk floor and no-replay checks.

The default `--evidence goal_only` presents only the operator goal to the model.
A separately prepared `--evidence supplied_observations` run appends exactly the
public observation receipts without changing the system prompt, tool choices,
weights or decoder. No command records or examiner answers enter this added
message. This is a matched one-step evidence comparison, not iterative feedback
or runtime knowledge acquisition. See the [handoff results](../results/akh-r1-goal-handoff-20260912-RESULT.md).

`run_feedback_handoff.py prepare --parent COMPLETED_OBSERVED_HANDOFF --out NEW
--lock-file SHARED_LOCK`, then `execute --out NEW`, crosses actual native feedback
on/off with final symbolic goal completion on/off. Each of 144 cases receives up
to three model decisions. The model always proposes an intent; on the last
decision only, enabled completion can substitute a supported public goal intent.
Native command derivation and independent authorization remain in every arm.

Shown feedback includes execution receipts, native derivations and an explicit
literal-domain goal check that identifies the requested effect. Attribute that
semantic assistance to the symbolic policy. The withheld control retains the
same previous model call and operator goal with a feedback-withheld marker.
Actual fixture state persists across decisions; successful observations update
subsequent public inputs. Each decision uses a freshly populated and reopened
native graph. This is bounded replanning with supplied records, not autonomous
acquisition or a complete incident. The run caps prompts at 4096 tokens, new
tokens at 96, model decisions at 432 and elapsed time at 1800 seconds, with the
same shared lock, disk floor and artifact requirements. See the
[frozen run description](../results/akh-r1-feedback-handoff-20260912-RUN.md).
