# Akh capability and model-size research program
- **Status:** Superseded (2026-09-13) by [P6 plan](2026-09-13-phase51-p6-local-training-neurosymbolic.md); kept as backup of the archived feat/akh-r1-episode-exam design.

Date: 2026-09-12. Status: executable research plan; model size and release readiness unresolved.

This is the controlling experiment sequence for the [full R1 requirements](2026-09-11-akh-350m-graph-grounded-release-plan.md). It preserves that document's functional scope and certification thresholds while replacing its open-ended sequence of component probes. The [program ledger](../../../training/akh_r1/research_program.json) records dependencies, decision owners, budgets and next work. Local preparation is authorized; no foundation-training job is launched by this document.

## The question and the honest starting point

Find a model, curriculum and symbolic runtime that together perform the Akh function on a 16 GB Mac mini: understand a goal, acquire and revise knowledge, reason with the graph, learn unfamiliar commands, act, verify, repair, remember across restarts, and adapt appropriately to people. Generalization means new worlds, command semantics and task compositions, not a new identifier in a familiar command template. The minimum useful model size is an outcome of this research.

The work has drifted toward narrow tool/registry diagnostics. Those experiments exposed real defects and useful mechanisms, but repeated variants of the same fixture cannot decide the central size or training-method question. That is a sequencing problem, not evidence that the neuro-symbolic thesis is wrong. We will retain their evidence and make the next investments answer broader, falsifiable questions.

The strongest current observations are:

| Observation | What it establishes | What remains unknown |
|---|---|---|
| Owned 33.4M checkpoint at step 493261 failed its text release gate | Training and checkpoint recovery work; narrow tool errors remain | Its failure does not locate the minimum capacity for a complete Akh |
| Historical 341.9M rule model had unseen-depth chain score 0.000 | The tested synthetic curriculum did not transfer to that depth | General-language capacity and iterative engine-assisted transfer |
| 596M Qwen diagnostic, two adapters, bounded model/native loop: 18/18 effects including 6/6 repairs in the best conditions | Symbolic binding plus repeated model decisions can recover effects that one model decision misses | Only six development worlds; completion recognized for 12/18 goals; no independent broad generalization result |
| Three compact-adapter repairs recovered through explicit symbolic completion | The engine can be the repair mechanism | Its contribution must be measured against engine-only execution |
| Full-history/compact paired training did not produce autonomous incident repair | That representation/curriculum pair was insufficient under the tested budget | Neither a capacity verdict nor a test of actual High Grug; productive compact prompts were 0.85% larger |
| One new persistent-goal smoke performed documentation discovery, graph storage, repair and fresh verification | A symbolic scaffold can join these components with an explicit operator goal and known capability semantics | No model participation, cold restart, semantic novelty or ablations yet; parked as a prototype |

Evidence: [assembled experiment report](../../../training/results/akh-r1-system-experiments-20260912-REPORT.md), [bounded feedback comparison](../../../training/results/akh-r1-feedback-handoff-20260912-RESULT.md), and [persistent-goal smoke](../../../training/results/akh-r1-persistent-goal-smoke-20260912-RESULT.md). None certifies R1. The old cloud pod is stopped with its volume retained; no local model job was running when this program was prepared.

## What the external research supports

The [linked post](https://x.com/_yusufknl/status/2098724555963425208), retrieved through its public FxTwitter mirror on September 12, extrapolates a 96% pruning result to Claude/GPT-5. The cited [lottery-ticket paper](https://arxiv.org/html/1803.03635v5) studied sparse subnetworks in MNIST/CIFAR10 classification, including their initialization and retraining. Its limitations explicitly discuss the expense of iterative searches. It does not show that 96% of those LLMs' weights are useless, or provide a cheap way to choose a winning small network before training. Existence of a useful subnetwork and efficient discovery of it are different claims.

Direct LLM results make pruning a credible comparison:

| Primary source | Relevant result | Consequence for Akh |
|---|---|---|
| [Sheared LLaMA](https://arxiv.org/abs/2310.06694) | Structured pruning turns Llama2-7B into 1.3B/2.7B models, followed by continued training with dynamic data loading | Test actual smaller dense architectures and include recovery training; this does not promise lossless 27B→350M conversion |
| [Minitron](https://arxiv.org/abs/2407.14679) | Pruning and distillation derive 8B/4B models from a trained 15B parent with much less additional training than independent full runs | Compare inherited weights plus recovery with directly trained small models; account separately for the parent's training |
| [SlimQwen](https://arxiv.org/abs/2605.08738) | Studies pruning/distillation of a large MoE during pretraining, including Qwen3-Next-80A3B→23A2B | Total parameters, active parameters and deployment memory are distinct quantities; MoE evidence is not automatically dense-model evidence |
| [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644) | A 1.3B code model trained on curated and synthetic educational data demonstrates strong results on the paper's coding evaluations | Supports investigating data quality and teaching structure; does not establish general Akh competence or teacher-free equivalence |
| [BFCL methodology](https://proceedings.mlr.press/v267/patil25a.html) and [multi-turn evaluation](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html) | Evaluates tool selection, executable behavior and multi-turn state changes | Add an external tool-use check, but retain our graph/restart/unknown-command exam as the primary outcome |

[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) is a relevant strong reference: its official card describes a dense native vision-language model with controllable thinking and improvements in multi-step work. Vendor benchmarks motivate an independent Akh comparison; they do not prove equivalent sub-billion behavior. Pin a reference revision, backend and thinking budget before testing. Do not download or allocate a 27B model just to complete a size table. Four-bit storage for 27B parameters alone is about 13.5 decimal GB before metadata, KV cache, runtime and graph, so the 16 GB deployment constraint remains material.

Pruning removes parameters. Quantization changes their representation. Distillation transfers behavior. A KG externalizes attributed state. High Grug supplies typed, checkable meaning. A curriculum changes what and how the model learns. These can cooperate, but each needs its own intervention and measured effect. Sparse weights do not imply proportional Metal speedups; prefer deployable structured pruning for the first compression experiment.

## Preserve the architectural intent and distinguish lineages

The program connects [ADR 087](../decisions/087-grug-brain-baby-llm-training.md), [ADR 092](../decisions/092-high-grug-v1-reasoning-register.md), [ADR 093](../decisions/093-dual-process-trace-handoff.md), [ADR 099](../decisions/099-agentic-communicative-intelligence.md) and the [Dojo manual](../manuals/2026-09-01-babyllm-dojo-curriculum.md). Earlier neural/VSA designs are architectural context, not proof that every proposed bridge is implemented.

Keep two labeled research routes:

1. **Build useful reflexes from the beginning.** Owned general-language foundation, engine-generated grounded traces, early High Grug fluency, staged/interleaved kata, and later measured consolidation. This is the sovereign base route accepted in ADR 092.
2. **Inherit and compress a stronger organ.** An external or owned larger parent, structured pruning and optional verified distillation, followed by the same Akh curriculum. Parent lineage, licenses, original training unknowns, recovery cost and export support remain explicit. An external diagnostic is not silently relabeled an owned sovereign base.

The user has authorized exploring these mechanisms. A change to the shipped foundation's lineage needs an explicit recorded selection decision after evidence; it does not block the comparisons. We are testing whether early teaching is effective, not assuming it directly selects identifiable “good weights.” A LoRA experiment on an already pretrained model cannot establish the effect of early foundation-training order.

Neural responsibility: interpret unfamiliar language/evidence, identify missing information, propose useful abstractions and subgoals, and select among capabilities. Symbolic responsibility: identity, provenance, persistent goals, rule execution, permitted argument binding, verification and supported repair. The clutch chooses when reflexes suffice and when explicit graph reasoning or consultation is needed. Record each handoff, its trigger and its benefit; do not infer a good clutch from a plausible explanation.

## Frozen capability contract

Keep the twelve families and quantitative full-release targets in the R1 plan. Begin with a **100-world development pack covering every family**, then a separate **100-world validation pack** with different generators/documents/compositions. Allocate at least six worlds to each family; tags may overlap. Release claims require the existing larger sealed pack, including at least 100 independent worlds per family and 200 learn/restart/recall episodes. These are deliverables, not claims that those packs already exist.

The first pack must contain complete episodes for:

- Graph acquisition, identity, provenance, revision and dependent-belief invalidation.
- Cold restart and retrieval of knowledge learned with frozen weights; graph correction changes the subsequent action.
- Unfamiliar command discovery, learning a versioned skill, Unix-style composition, execution, observation and reuse after restart.
- Grounded joins, rule selection, deeper compositions, counterfactuals, ambiguous goals and appropriate clarification.
- Homelab alert response across backups, capacity, monitoring and dependencies; changed CLI semantics and recovery after interruption.
- Dojo and emotional repair during operational work and ordinary consultation; Portuguese basics plus English/German/French retention.
- Perception→graph→action, first with verified transcripts as a labeled control, then actual OCR/images.

New syntax alone is weak novelty. Include unseen effects, flags with changed meanings, distractor tools, held-out relation structures, unseen workflow combinations and conflicting/stale documents. Freeze symbolic policies before validation; a new per-case rule consumes that case as development evidence. Return correct uncertainty when no adequate capability or source exists. A model starting with no site facts must not receive the answer through a menu, host goal compiler or examiner callback.

Score each episode on actual final world state, correct target, authorization, evidence-backed graph delta, subsequent reuse, fresh verification, truthful completion and interaction repair. Count blocked bad proposals separately from unauthorized effects. JSON validity is a component metric. Success is not a sum of easy metrics that masks a failed central capability.

Store causal traces and the assistance profile: raw neural proposal, grammar repair, host serialization, retrieved evidence, native rule proof, tool receipt, verification, and human clarification. A symbolic repair counts as system success when independently verified. Engine-only success does not count as neural generalization.

## Experiments and decisions

Use sequential comparisons, not every combination of model, curriculum and runtime. One bounded experiment must specify a hypothesis, discriminating control, artifact, budget, decision threshold and next action before launch.

| ID / dependency | Work and comparison | Decision / exit artifact |
|---|---|---|
| **R0 — immediate** | Inventory reusable harness paths; freeze all-family development/validation specifications, scorers and assistance contracts. Complete the persistent-goal scaffold only as needed by the integrated exam. Build three end-to-end smoke worlds: unknown command + cold reuse; graph correction + changed action; operator correction + repaired interaction. | Executable exam and an oracle/reference demonstration that expected success is reachable without examiner leakage. A fixture validity pass is not model success. |
| **R1 — after R0** | On the same pack compare symbolic-only, current 596M baseline, its selected adapter, and one stronger compatible open reference. Run full Akh and a fixed-budget model-only control where evidence access is comparable. | Identify runtime versus model bottlenecks across families. If the stronger reference also fails, fix the failing seam or test design before interpreting small-model capacity. |
| **R2 — after R1** | Controlled owned architecture ladder around **350M, 600M, 1.1B**, exact parameter counts from manifests, same tokenizer/contract and curriculum. Start with cheap pilots and learning curves. Compare at matched tokens and separately matched compute. External small-model comparisons can guide pilots but are not a clean size experiment. | Capacity curve with confidence intervals and undertraining checks. Expand toward 1.7–3B only if 1.1B remains inadequate and a stronger reference demonstrates the missing capability. Never cut the function merely to keep a size label. |
| **R3 — after R1; anchor selected with R2** | At one anchor size compare shuffled matched teaching versus belt progression with spaced retrieval and interleaved whole-episode repair. Then compare actual early High Grug, late High Grug and matched natural-language evidence. Confirm useful effects at a second size and across three seeds for finalists. | Estimate whether teaching order, grounding and representation move the capability curve. Match task information, exposure, tokens/compute and replay; report remaining differences. Confirm early-HG claims in foundation training, not only adapter tuning. |
| **R4 — after R2/R3 shortlist** | For a parent already competent on Akh tasks, prune conservatively to one nearby deployable target, then recover. Compare (a) prune + recovery, (b) directly trained small target, (c) small target + verified distillation. Keep unpruned parent as reference. | Quality/cost frontier, before/after pruning and after recovery. Separate inherited pretraining cost from additional project cost. Stop compression if it erases core transfer or requires more recovery than the direct route. |
| **R5 — after shortlist** | Deploy selected candidates on the actual Mac: high-precision reference, 8-bit, affine 4-bit, supported FP4; exact tokenizer/backend parity, full graph load, restart, concurrency and 90-minute endurance. | Choose a capability/memory/latency frontier point. Retest cognition, graph revision and command use after each conversion; quantization is not a capability substitute. |
| **R6 — after R5** | Freeze candidate and run sealed text, integrated episode, perception, cross-host and installer gates. Package capability/assistance profile and reproducible recipe. | Full operational certification if gates pass; otherwise an explicitly partial research candidate with complete failure map and a specific next research decision. |

The next action is R0 breadth and end-to-end coverage. The unrun 44-case expansion of the persistent-goal microfixture is parked. Do not start it automatically merely because its prototype exists.

## Tests that can disprove our preferred explanation

| Hypothesis | Discriminating intervention | Evidence against it / response |
|---|---|---|
| **H1: changing knowledge can live in the graph while weights supply reflexes** | Learn randomized facts/commands with fixed weights; clear KV/conversation, restart, reuse; remove or coherently change graph facts; change versions | Successful reuse absent, or actions ignore valid changed knowledge: locate write/retrieval/interpretation failure. Within-session no-KG controls receive equivalent permitted receipts; cold no-memory controls should admit the gap. |
| **H2: kata shape transferable memory into action** | Matched examples and budget, shuffled versus staged/interleaved practice; unseen domain and role transfer; correction after a failed first attempt | More polished wording without better action/correction/ownership, or benefit confined to taught templates: revise curriculum rather than claiming emotional intelligence. |
| **H3: the clutch improves useful reasoning** | Same evidence with learned selection, fixed always-engine policy, bounded model-only, engine-only; hold total work budget constant | No gain over fixed rules, unnecessary handoffs or no model contribution: simplify the policy and retain useful symbolic behavior. |
| **H4: High Grug is a useful common reasoning register** | Same information in prose versus typed executable traces; early/late exposure; binding/reference/rule ablations | No improvement in success, grounding, transfer or total cost: keep any verified auditability benefit, stop claiming compression. Count glosses, tables, parsing, retrieval and retries. |
| **H5: sub-billion/1.1B capacity is adequate** | Same-family learning curves, sufficient exposure, successful stronger reference, three finalist seeds | 1.1B still fails core unseen cases while larger organ succeeds under the same runtime: expand capacity or revise training. If both fail, size is not yet the explanation. |
| **H6: useful capacity can be inherited more cheaply than trained directly** | Same target architecture and Akh evaluation, direct versus pruned versus distilled, with parent/recovery costs explicit | Transfer collapses or measured total cost loses: reject that compression route. Unstructured zeros alone are not a deployment win. |

Add retrieval-oracle and explicit-goal controls only to diagnose a failed seam, clearly marked assisted. They never replace the unassisted evaluation. A stop checker written for six known phrases is not evidence of broad goal comprehension.

## Kata, extended Dojo and knowledge consolidation

The curriculum retains all eight starter forms: **Yes-And, The Mirror, Redirect-Refusal, Stubborn Affirmation, Double Diamond, Clarify-Before-Answer, The Silence, Naming-the-Move**. It retains all six consultation forms: **Establish the Working Agreement, Observe and Inquire, Hold Competing Explanations, Make a Small Intervention, Read the Response and Repair, Return Ownership and Learning**. These are behavioral choices, not mandatory response templates.

White/Yellow practice and contrast individual forms; Green handles corrections; Brown chooses and combines forms during unfamiliar complete tasks; Black tests transmission to a separately trained student as a stretch experiment. Vary names, domains, languages, roles and command structure independently. Teach when to stop, when to ask, and when already-authorized execution should continue. Include adverse cases where a familiar kata is the wrong move.

Preserve whole episodes with only past observations visible at each decision. Use spaced cold retrieval, interleaving, demonstration removal, wrong-binding repair and explicit counterexamples. Every stored skill has purpose, preconditions, provenance, tested effects, versions, failure cases and invalidation rules. Compare immediate KG learning with later verified weight consolidation; a site's current state should remain revisable without retraining. Consultation outcomes require independent action evidence plus blinded human/judge rubrics; agreement or gratitude is not the success criterion.

Portuguese basics cover simple goals, negation, conditions, clarification and corrections with exact identifiers; separate PT-BR/PT-PT and comprehension/generation. Broad fluency remains optional. Emotionally appropriate behavior must survive a participant saying “you misunderstood” and changing the actual next action. Do not infer an enduring personality or emotion as a ground-truth label.

## Selection, statistical discipline and stopping

Select the **smallest system that meets the capability floors with useful headroom**, not the smallest checkpoint that can be made to pass a narrow exam. Report per-family task success, cold learning/reuse, semantic tool correctness, graph revision, interaction repair, model/engine work, latency, peak total memory and cost per completed task. Any new unauthorized effect or fabricated completion blocks promotion under existing gates. Useful partial capability remains a valid research outcome.

Use independent worlds as the sampling unit. Repeated prompts and 144 condition trials over six worlds do not become 144 independent tests. Report binomial intervals and paired, world-clustered bootstrap differences. Freeze seed sets and comparators; use three training seeds for decisive curriculum/capacity claims. Exploratory one-seed pilots screen costs, not significance. Keep the sealed exam closed until candidate freeze; failures used for tuning become development data and require new sealed cases.

For shortlisted candidates, predeclare a five-percentage-point noninferiority margin against the strongest affordable Akh reference on core families, while still satisfying the R1 absolute thresholds. Wide confidence intervals mean undecided, not equivalent; expand independent worlds when needed. Do not select by an overall average that hides absent graph learning or emotional repair. The exact full-release thresholds in the existing plan remain authoritative.

After **two component probes for one unresolved decision**, write the evidence and choose: integrate, change the hypothesis, or stop that branch. No third cosmetic variant without a new discriminating reason. A bug fix receives a regression and one affected integration rerun; unrelated deployment work does not become the next research task. Plateauing small-model performance warrants checking exposure and representation before either declaring impossibility or training indefinitely.

## Budget, schedule and execution manifests

Current ledger: cloud training stopped; preserved volume incurs storage charges. Last recorded account credit was **$77.85 at 2026-09-12 04:12 UTC**, with a **$0.029/hour** baseline. This is timestamped evidence, not a live balance or proof that a complete 1.1B foundation run is affordable. Existing diagnostic compute cap is **$10 total additional cloud compute**, not $10 per condition. No new GPU allocation was made for this plan.

Do R0 and available-baseline work locally first. Proposed allocation within the existing diagnostic ceiling: up to $2 for measured backend/throughput feasibility, up to $4 for the most informative size or curriculum pilot, up to $2 for an independent confirmation, and $2 reserve. These are ceilings, not estimates or prepaid work. Use local execution when it meets time/memory limits; record wall time, electricity if measured, and desktop contention instead of calling it free.

Before any paid manifest, refresh actual credit and rate; measure representative sequence length, batch and precision throughput; include evaluation, startup, transfer and storage. Estimate `hours = total processed tokens / observed tokens per second / 3600 + evaluation/setup hours`, then apply 25% contingency. Cap the aggregate reservation at the remaining authorized diagnostic ceiling. Checkpoint before the cap/deadline and preserve the cost guard. Larger foundation/compression programs need a separately costed funding decision; do not infer sufficient funds from a small adapter pilot.

Every run manifest records: program/hypothesis ID, dependent evidence, code/checkpoint/tokenizer/backend hashes, parameter count, lineage, corpus and split hashes, seed, curriculum schedule, exact runtime assistance, primary outcome, comparator, useful-effect threshold, max updates/tokens/wall time/dollars, disk/memory floors, artifact paths, progress heartbeat and stop conditions. A paid run whose throughput is unmeasured is not launch-ready. The [$10 ledger](../../../training/akh_r1/research_program.json) is a reservation plan, not an enforced scheduler; the execution wrapper must enforce each frozen run cap.

Planning estimates: R0 takes roughly 1–2 focused engineering days using the existing harness; first broad reference evidence should follow in another 1–2 days if backend access is ready. Candidate screening and controlled pilots then take roughly 1–2 weeks, conditional on measured compute. Full owned pretraining has no honest completion date until its token budget and throughput are known. End each stage with a decision report, not merely another status update.

## Monitor and continuity

The detached `t3_wake.py` process now uses the versioned [research policy](../../../training/akh_r1/research_monitor.py). It preserves existing cloud/disk/gate event alerts and durable delivery receipts. It reads `r1-execution.json` and its active experiment status; it never starts a pod or changes a gate.

- Completed/stopped runs produce no periodic wake. Their historical metrics remain available.
- Active progress produces at most one hourly digest, only when counters or stage change. New stages/results and actionable stale/deadline conditions notify once per transition.
- A research checkpoint fires once for an explicit action ID whose state is `ready` and whose due time has arrived, only when no experiment is active. It directs the next bounded program action; it is not an endless idle reminder.
- Unattempted stale hourly messages are superseded. Accepted or uncertain dispatches retain their original IDs for reconciliation.

Installation proof is in `runs/babyllm-recovery-20260909/research-monitor-install-proof-20260912.json`. The bridge remains a detached desktop process; this change does not make it a boot-persistent service. Its heartbeat is observable even when it emits no messages. Operators register a new experiment before launching and update its status/progress during work; the policy cannot infer unregistered jobs.

The next checkpoint is **R0: build the broad episode contract and three integrated smoke worlds**, using local resources and existing artifacts. It must not reactivate the retired cloud pipeline or interpret a completed prototype as permission to launch the deferred microexperiment matrix.

## Seeded symbolic world model comparison (September 13 update)

Use existing architecture and documented sources before adding another symbolic subsystem. The source catalog is [world_model_sources.json](../../../training/akh_r1/world_model_sources.json): local ontology/common-sense/computing packs; Cyc-inspired reasoning from Phase 9; Wikidata, ConceptNet, Wikipedia and DBpedia from Phase 14/35. Cyc is a tentative identification of the user's remembered source. External dataset contents are not yet imported or pinned by this research slice.

Add matched conditions to the R0 repair/reference exam: schema only; actual local seed packs; separately sourced operational rules; later, episode-learned rules with explicit lineage. Use the same hidden environment instances and action budget. Score verified repair, unsupported actions, source precision, correction/invalidation, cold transfer and memory/latency overhead. Predicting an effect must be checked against a fresh observation. Background taxonomy or common sense cannot establish live authority, current configuration or successful completion.

Required negatives include missing current record despite an authoritative version witness, missing/conflicting witness, misleading generic background, incorrect prerequisite direction and a scoped exception that supersedes a general rule. Preserve an unseeded control and all failed artifacts. Do not count seed conditions as additional independent worlds. Source acquisition requires pinned snapshots and provenance; the current no-download local slice uses available packs only. The native causal manager and event-calculus interfaces must be evaluated for reuse, not assumed covered by Python fixture checks.
