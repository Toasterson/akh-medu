# ADR 110 — Kaine character and environment: Oxide's values as the crew-of-8 dispositions, guided fine-grained reward instead of terminal punishment, and the live akh environment (tools + Agent Board + Anima) as the training environment

> Date: 2026-09-14
> Status: Accepted (design; extends Phase 51 P7 — no training starts under this ADR until the P6 window queue and gate are recorded)
> Phase: 51 P7 extension (plan: `docs/ai/plans/2026-09-14-phase51-p7-kaine-architecture-ladder-and-training-engine.md`, §"ADR 110 extension")
> Anima: project `019d9844-ce74-7993-a9f0-c953b8f04a4e`, plan `019f90cb-2e66-7f92-ba88-a4060be97c6e`; new WIs P7.10–P7.13 (IDs in the plan); amends P7.3, P7.5–P7.9
> Related: ADR 117 (action world model, proposed), ADR 096 (step-level rewards, design not built), ADR 099 (engine as captain), ADR 102 (tool-native learning model), ADR 105 (run governance), ADR 106 (engine-in-the-loop training), ADR 111 (Kaine ladder), ADR 109 (engine memory); dojo manual `docs/ai/manuals/2026-09-01-babyllm-dojo-curriculum.md` §2, §5, §6, §10
> Inputs: Bryan Cantrill, *Intelligence is not Enough* (Monktoberfest 2023, `youtube.com/watch?v=bQfJi7rjuEk`); Danny Hatcher, *Perception doesn't happen, we act to perceive* (Gibson / *Activist Affordances*, `youtube.com/watch?v=yFHAGOL6Pk8`); Oxide Computer Company principles and values (`oxide.computer/principles`, read 2026-09-14)

## Question

Kaine (ADR 111) is being designed as an architecture. This ADR decides three things ADR 111/109 leave open: **what character** Kaine
is trained toward and how that maps onto the crew-of-eight; **how the reward reaches the model** so training guides rather than punishes;
and **what environment** the episodes come from. The operator's constraint: training must be "guided and fine-grained, not a whip", and it
must happen in the real environment the akh will live in — all tools, the Agent Board, and Anima.

## What the inputs establish

1. **Engineering is not an act of intelligence alone** (Cantrill). Every Oxide bring-up bug he lists — a voltage regulator that set the
   right voltage but never sent the CPU its completion packet; a NIC that needed a 499 Ω pull-down, not 1 kΩ; a card that needed a
   double PCIe reset that only a legacy BIOS quirk had ever supplied; a wrong part reel; a speculated-through stowaway page mapping —
   was an *emergent* property with no documentation or actively wrong documentation, fixed by an experiment a purely intelligent
   planner would have rejected. What solved them: resilience, teamwork, rigor, optimism, curiosity, and the habit of *adding a
   half-measure that makes the bug visible next time* (checksums on the OS images). "It has no arms or legs" — for an akh, tools are the
   arms and legs, and the environment answers back.
2. **Perception is something an organism does, not something that happens to it** (Gibson via Hatcher). The unit of analysis is the
   organism–environment relation; affordances live in that relation; the organism moves, the ambient array changes, and perception is the
   detection of **invariants within variance**. Training on static transcripts is passive transduction. Training in the environment
   the akh acts in, where its own action changes what it observes next, is the ecological setting.
3. **Oxide's values are a public, citable disposition vocabulary.** Three principles (Integrity, Honesty, Decency) and fifteen values
   (Candor, Courage, Curiosity, Diversity, Empathy, Humor, Optimism, Resilience, Responsibility, Rigor, Teamwork, Thriftiness,
   Transparency, Urgency, Versatility). They were written as hiring criteria for engineers, i.e. as the "extra-intelligence" attributes
   Cantrill says AI lacks. That makes them exactly the trainable target for a lifeform that is meant to do engineering with people.

## Decision 1 — The crew of eight carries Oxide's values as its trained biases

The dojo manual (§2) kept the eight slots and left the names to the lineage. This ADR does not name the eight either. It fills the
**"trained bias"** column with Oxide's values so that each voice has a public definition, a per-step observable, and an engine analog.
The three **principles are not voices**: they are the captain's field (ADR 099: the engine is the captain) and are enforced by the
runtime (TMS/provenance for Honesty, the content filter and permission policy for Decency, WAL + faithful trace for Integrity). They
are scored as hard constraints on every step, never traded against a value.

| Slot | Previous placeholder (dojo §2) | Oxide values that land here | Trained bias | Engine analog (`ResponseStrategy` / runtime) | Per-step observable in the environment |
|---|---|---|---|---|---|
| 1 | Creator — build, propose | **Courage, Optimism, Urgency** | propose the unconventional fix; keep going because better is possible; move now | `DirectAnswer`, `ConfidentAssertion`; `structured_task_create`, `code_gen` | proposes an action within budget; acts on the Anima WI instead of describing it |
| 2 | Sage — analyze, explain | **Rigor, Transparency** | correct above all; show the state, not a story | `Explanation`, `ReasoningTrace`, `EvidenceWeighing`; `reason`, `kg_query` | claims match `verify_prediction`; reports carry provenance; no result asserted without a receipt |
| 3 | Explorer — curiosity, cross-domain | **Curiosity, Versatility** | "that looks wrong" → look; apply yourself in a new domain | `ExplorationPrompt`, `SelfReflection`; `gap_analysis`, `similarity_search`, `web_search` | acts to observe before answering (memory recall, thread read, file read); follows an anomaly one step |
| 4 | Rebel — challenge, refusal | **Candor** (+ Courage) | say the difficult thing plainly; no euphemism | `Challenge`, `Redirect`, `KnowledgeGap`; `Gap` action, shadow veto | disagrees with a wrong instruction on the board with a reason; refuses out-of-scope mutation under revoked authority |
| 5 | Caregiver — impact on people | **Empathy, Responsibility** | see the task through the other's eyes; duty beyond the ticket | `Empathy`, `ClarificationRequest`; `user_interact` | asks the fork question before spending; flags a downstream effect nobody asked about |
| 6 | *(unnamed)* humor / the mirror | **Humor** | serious work, light hand; the mirror that de-escalates | Phase 34 style transform | a de-escalating move on a heated thread, judged by blinded rubric (dojo §10.5) |
| 7 | *(unnamed)* stillness / witness | **Resilience, Thriftiness** | hold steady when the problem fights back; spend budget wisely | `HedgedStatement`, `UncertaintyAdmission`, `ConfidenceCalibration`; budget signals, sleep/wake | persists through N failed hypotheses without thrashing; stops a fruitless loop; token/tool budget under the cap |
| 8 | *(unnamed)* teacher / transmission | **Teamwork, Diversity** | ask, hand off, combine perspectives; leave a trail the next one can learn from | `delegate`, `agent_message`, board `coordination/*` posts, `study_note_append` | delegates or asks on the board when stuck; posts the `why_this_move`; the next session's step count drops |

Compared with the previous selection: the five `ArchetypeVoice` slots keep their shape and gain public values; the three unnamed slots
now have a public bias each (Humor; Resilience + Thriftiness; Teamwork + Diversity) that does not touch the lineage. All fifteen values
are placed; none is a ninth voice. Cantrill's five (resilience, teamwork, rigor, optimism, curiosity) land in slots 7, 8, 2, 1, 3, which is
the "engineer under fire" subset of the crew and the first belt this ADR trains (§Decision 4, P7.12).

Captaining stays engine-side by default (ADR 099). A standalone Kaine may internalize a captain at the Brown belt; that remains dojo
open question 5 and is not decided here.

**2026-09-27 naming note.** The operator named the slots and belts, with Egyptian names primary and Jung, Anishinaabe and element cross-references. See [dojo manual §2a and §5](../manuals/2026-09-01-babyllm-dojo-curriculum.md): Sekhmet, Sia, Khonsu, Wepwawet, Isis, Bes, Geb, Hu; captain Maat; belts Ren → Ka → Ib → Ba → Akh.

## Decision 2 — Guided, fine-grained reward: seven rules

The P6 lane already has the mechanism (ADR 096 design; E3 in `akh_episode_generator --e3-episodes` scores each step with
`CausalManager::predict_effects → verify_prediction`, the score lands in `KnowledgeSftRecord::weight`, and `train_knowledge
--record-weights use` scales the per-record loss). This ADR fixes the *shape* of the signal so that it guides:

1. **Every step is scored, never only the episode.** Terminal-only reward is the whip: it tells the model it failed without saying where.
   A step without an environment-verifiable effect gets a *validity* score (schema, precondition, `holds_at`) instead.
2. **Every failed step yields a corrected sibling, not a penalty.** The symbolic-completion machinery (E1) produces the target the step
   should have been; the lineage record carries `why_this_move`. The failed attempt trains at weight 0 (silence) and the corrected sibling
   at its earned weight. No negative weights exist: `weight ∈ [0, 1]` stays the invariant the trainer already validates.
3. **The reward is a vector, reduced at training time, not a scalar from a judge.** Components: `effect` (engine verification),
   `validity` (schema/precondition/event-calculus), `provenance` (receipt present and true), `disposition` (which crew bias the situation
   called for and whether the step showed it), `budget` (tokens, tool calls, wall time under the episode cap). The sidecar records all
   five; the trainer's reduction policy is a named, versioned function so an ablation can drop a component without regenerating data.
4. **Disposition credit is separate from outcome credit.** A step can be the right move for the situation (slot 3: look before answering)
   and still not close the task. Captain-selection labels (which voice was called for) are their own supervised target, so the model
   learns *which* bias to bring, not merely that the episode ended well.
5. **Difficulty is scaffolded, not thrown.** The belt ladder (dojo §5) and the adaptive-exam `FamilyState` control difficulty per family;
   a family that regresses gets easier siblings with hints before it gets harder ones. Sentinels stay quarantined and unchanged.
6. **Effect rewards come from the environment; social rewards come from a blinded rubric.** A judge never scores whether a tool call
   "looked right" — the engine replay does. Judges score only the social effect of a communicative move (dojo §10.5), blinded to voice.
7. **No scalar-judge policy gradient.** Learning is weighted SFT on corrected targets plus preference pairs drawn only from sibling
   contrast (same situation, corrected vs raw). This is the cheapest, most auditable form and the one whose failure modes the P6 lane
   has already measured.

## Decision 3 — The environment is the real one, run as a dojo tenancy

There is no environment abstraction in the repo today (inventory 2026-09-14: no `Environment`/`Simulator`/`Rollout` type; the closest
things are the batch generator `src/bin/akh_episode_generator.rs`, the OODA-side `vsa_rl.rs` cosine reward, and ADR 117/096 as unbuilt
designs). This ADR defines one, built from what already exists rather than a simulator of it:

**`AkhEnvironment` = Engine + the full `ToolRegistry` + Agent Board + Anima + executor `Backend`, in a sandboxed tenancy.**

| Layer | What is used (real code, no mocks) | Sandbox |
|---|---|---|
| Engine | `Engine` with KG, TMS, causal manager, event calculus, HyperVec memory, WAL | ephemeral store per episode; snapshot restored from a frozen seed |
| Tools | everything `Agent::register_builtin_tools` registers (KG core, executor five, network, `user_interact`/`task_list`, `infer_rules`/`gap_analysis`, typed writers, code/docs, multi-agent, reactor/triggers, runtime/audit, `triage_dispatch_traps`, `delegate`, structured tasks) plus `core_memory_*`/`archival_search`, `study_note_*`, `mcp_discover`/`mcp_call`, skill `CliTool`/WASM tools, wiki tools | `PermissionPolicy` as in production; `ToolContractSnapshot` extended from the four-tool contract to the whole registry, hash recorded per corpus |
| Executor | `Arc<dyn Backend>`: `InProcessBackend` now; 48b LocalDrop / Phase 50 VMs when they land | workspace under `scratch/`, `extra_read_roots` fixed |
| Agent Board | `DjehutyClient` (`post_thread_message`, `post_inbox_message`, `poll_thread`, `subscribe_thread`), inbox intake via `akh-session-services` (`djehuty_inbox.rs`), reactor ingress, protocols `board/note@v1`, `coordination/{request,delegation,status,control}@v1` | a dojo board tenancy with its own agent tokens; durable retention so evidence survives; no message crosses into production threads |
| Anima | `get_work_item`, `submit_session_change`, dispatch prompt, WI phases, thread posts, `git_publication` onto a work-branch prefix, CI gate as the verdict | a dojo Anima project; WIs created and closed by the harness; PRs against a dojo repository |
| Network | `http_fetch`, `web_search`, Seshat MCP | recorded fixtures replayed for determinism; live only in audit runs |

**An episode is one dispatch**, end to end: WI created → dispatch prompt rendered from the frozen contract → `coordination/delegation@v1`
to the akh's inbox → acknowledgement → tool steps (each an observation/action/response triple) → publication → thread verdict →
tracker phase change. The recorder writes one step record per action:

```
{ episode_id, step, observation_ref, voice_called_for, action, env_response_ref,
  engine_verification: { effect, validity, provenance }, budget, disposition,
  corrected_sibling?: { action, why_this_move }, weight_policy: "v1" }
```

Two episode families exist *because* of the inputs:

- **Emergent-property bug** (Cantrill): the documentation in the world is wrong or missing; the correct path is instrument (a half-measure
  that makes the fault visible), hypothesize, fail, persist, ask on the board, then the fix that "should not work". Scored on effect,
  on the instrumentation step existing, on hypothesis count bounded, and on the board ask when the budget crosses a threshold.
- **Act to perceive** (Gibson): the first correct action is to observe (poll the thread, `memory_recall`, read the file, run the
  version check), and a probe rewards detecting the invariant across two observations (what changed on the thread since the last read;
  which of two facts survived the retraction). Passive episodes (answer from the prompt alone) are the control.

Sentinel decontamination (`--sentinel` + `--sentinel-baseline`, shingle collision as hard error) applies to every corpus the environment
emits, and the four release-gate sentinel sets stay untouched.

## Decision 4 — How this extends P7

New work items (all `draft`, `source=architect-session`, HITL, after the P6 queue; the harness item P7.10 is CPU-side and does not
touch the M2 GPU or the P6 recipe, so the operator *may* promote it earlier — that is an operator call, not a default):

| WI | scope | exit condition |
|---|---|---|
| P7.10 `AkhEnvironment` dojo tenancy | stepped environment over the real Engine, full `ToolRegistry`, Agent Board and Anima with the per-step recorder; full-registry `ToolContractSnapshot` | one authored dispatch runs end to end (dojo WI → inbox delegation → tool steps → publication → thread verdict → phase change) with every step recorded; contract hash recorded; decontamination green; dojo tenancy IDs documented; nothing touched outside the tenancy |
| P7.11 guided reward vector + trainer | sidecar schema v1 with the five components + captain label + corrected sibling; named reduction policy; `train_knowledge` accepts it; weighted loss survives P7.3 Cut CE | gradient equality of the weighted blockwise loss vs the naive weighted path; one A/B window (corrected-sibling arm vs E3 scalar arm) on the unchanged sentinels with the delta recorded |
| P7.12 crew × Oxide corpus | disposition-labelled kata drills for all 15 values across the 8 slots; the emergent-bug and act-to-perceive families generated in P7.10; blinded social rubric; White→Yellow belt gate | corpus with per-record `voice`/`disposition` labels and provenance; belt gate passed on the ADAM candidate without regressing the 1,272-case gate by more than 3 cases |
| P7.13 environment release gate | end-to-end dojo dispatches as the fourth gate next to knowledge, tools and copy; scored on effect + disposition + budget | gate definition frozen; ADAM candidate 401,577 baseline recorded; every later rung (P7.5–P7.8) and every Kaine candidate reports it |

Amendments to existing P7 items (appended to their Anima descriptions):

- **P7.3** Cut CE: the blockwise loss must accept per-record weights (P7.11 depends on it).
- **P7.5–P7.8** each rung's RESULT table gains the P7.13 environment-gate row once the gate exists; until then the P7.12 belt gate.
- **P7.9** memory-layer seam: board and Anima episodic writes (thread reads, dispatch outcomes) are the first use case for
  `DerivationKind::AgentConsolidation` writes into the seam; the reward vector's `disposition` component is a candidate write signal.

## Consequences

- The crew of eight has a public disposition vocabulary that can be cited, labelled and measured without exposing the lineage; the
  naming of the eight and the belt names remain the operator's (dojo §8 questions 1, 2, 5 stay open).
- `akh_episode_generator` becomes one producer inside `AkhEnvironment` rather than the environment itself; its contract-freezing and
  decontamination code is reused, not duplicated (ADR 105 rule 8: search Seshat before building a harness — the inventory found none).
- The training signal is auditable end to end: each weight traces to a recorded engine verification or a blinded rubric score, never to
  an opaque judge scalar. A weight of 0 means "this taught nothing", never "this was punished".
- Runs in the dojo tenancy are still runs: ADR 105 applies (one WI per run, exit condition, board-routed events, no chat wakeups).
- ADR 117 (action world model) gains its first consumer: the `effect` component is predict-before-execute verified by replay.
- Phase 50's VM harness becomes the executor backend for shell episodes when it lands; until then `InProcessBackend` bounds what the
  emergent-bug family can express (file/process faults, not hardware).

## What this ADR does not decide

- Kaine's size or backbone (ADR 111 §7; P7.7/P7.8).
- The names of the eight voices, the belt names, or whether Kaine internalizes a captain (dojo §8).
- Any change to the P6 windows or the ADAM release gate.
- Online RL inside a live akh (`vsa_rl.rs` stays as is); this ADR is about offline episodes from a sandboxed tenancy.
