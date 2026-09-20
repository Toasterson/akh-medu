# ADR 110 — Kaine character and environment: Oxide's values as the crew-of-8 dispositions, guided fine-grained reward instead of terminal punishment, and the live akh environment (tools + Agent Board + Anima) as the training environment

> Date: 2026-09-14
> Status: Accepted (design; extends Phase 51 P7 — no training starts under this ADR until the P6 window queue and gate are recorded)
> Phase: 51 P7 extension (plan: `docs/ai/plans/2026-09-14-phase51-p7-kaine-architecture-ladder-and-training-engine.md`, §"ADR 110 extension")
> Anima: project `019d9844-ce74-7993-a9f0-c953b8f04a4e`, plan `019f90cb-2e66-7f92-ba88-a4060be97c6e`; new WIs P7.10–P7.13 (IDs in the plan); amends P7.3, P7.5–P7.9
> Related: ADR 088 (action world model, proposed), ADR 096 (step-level rewards, design not built), ADR 099 (engine as captain), ADR 102 (tool-native learning model), ADR 105 (run governance), ADR 106 (engine-in-the-loop training), ADR 111 (Kaine ladder), ADR 109 (engine memory); dojo manual `docs/ai/manuals/2026-09-01-babyllm-dojo-curriculum.md` §2, §5, §6, §10
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
things are the batch generator `src/bin/akh_episode_generator.rs`, the OODA-side `vsa_rl.rs` cosine reward, and ADR 088/096 as unbuilt
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
- ADR 088 (action world model) gains its first consumer: the `effect` component is predict-before-execute verified by replay.
- Phase 50's VM harness becomes the executor backend for shell episodes when it lands; until then `InProcessBackend` bounds what the
  emergent-bug family can express (file/process faults, not hardware).

## What this ADR does not decide

- Kaine's size or backbone (ADR 111 §7; P7.7/P7.8).
- The names of the eight voices, the belt names, or whether Kaine internalizes a captain (dojo §8).
- Any change to the P6 windows or the ADAM release gate.
- Online RL inside a live akh (`vsa_rl.rs` stays as is); this ADR is about offline episodes from a sandboxed tenancy.


# ADR 111 — The ADAM → Kaine architecture ladder: attention stays the backbone; hybrids and memory layers are measured at the rung where the literature validates them

> Date: 2026-09-14 (renumbered 108 → 111 on 2026-09-14: ADR 108 is the Sokar artifact daemon, merged to main first)
> Status: Accepted
> Phase: 51 P7 (plan: `docs/ai/plans/2026-09-14-phase51-p7-kaine-architecture-ladder-and-training-engine.md`)
> Anima: project `019d9844-ce74-7993-a9f0-c953b8f04a4e`, plan `019f90cb-2e66-7f92-ba88-a4060be97c6e`; WIs P7.5 `01a0a1a3-439f`, P7.6 `01a0a1a3-6c2b`, P7.7 `01a0a1a3-a8a2`, P7.9 `01a0a1a4-03cb`
> Related: ADR 037 (LLM–VSA integration), ADR 104 (copy-faithful decoding, RoPE + chunked attention), ADR 107 (training memory), ADR 109 (training-engine memory program)
> Evidence: 17 papers read in full text on 2026-09-14 (arXiv IDs inline)

## Naming

- **ADAM** is the current BabyLLM lineage: `small_8k` (33.4M) through the 350M preset up to a planned ≈1.1B rung.
- **Kaine** is the next lineage, larger than ADAM's top rung. Its size is an *output* of the 350M and 1.1B measurements, not fixed here.

## Question

The Phase 51 model is a plain decoder transformer (d_model 512, 8 heads, 8 layers, d_ff 2048, RoPE, 8192 context, 1024-row
chunked recompute attention, ≈3k–8k BPE vocabulary). Before designing Kaine: does the research on attention alternatives
(state-space models, linear attention / delta-rule layers, test-time memory) say we should leave the transformer, and if so
at which size?

## What the literature establishes

### Linear attention *is* a VSA store

- Schlag, Irie, Schmidhuber 2021 (2102.11174) prove the capacity limit of the linear-attention state `S = Σ vᵢ ⊗ φ(kᵢ)` by citing
  Smolensky's tensor-product-representation crosstalk theorems directly; plain linear attention fails past 64 associations at
  head dim 64. Their delta rule `W ← W + β(v − W·φ(k)) ⊗ φ(k)` first *unbinds* the key to read the current filler, then writes only
  the residual — a cleanup-corrected write in VSA terms.
- Hrrformer (2305.19534) rebuilds attention with holographic reduced representations: circular convolution as binding, superposition
  as the memory, and **softmax as an explicit cleanup step**.
- DeltaNet (2406.06484): `S_t = S_{t−1} − β_t(S_{t−1}k_t − v_t)k_tᵀ`, chunk-parallel via the WY form of Householder products; perfect
  MQAR at 512/64 pairs. Gated DeltaNet (2412.06464): `S_t = S_{t−1}(α_t(I − β_t k_t k_tᵀ)) + β_t v_t k_tᵀ`; on the S-NIAH test with
  **UUID values** Mamba-2 "drops quickly", Gated DeltaNet degrades more slowly but is not exact. RWKV-7 (2503.14456) measures its state
  at **0.547 bits per dimension** (8192-dim state, 256 KV pairs → 72.9 % recall).
- Consequence: a delta-rule layer and the engine's HyperVec memory are the same algebra with different binding operators. This is the
  concrete bridge Phase 26 (ADR 037) was missing; it does **not** by itself argue for replacing attention.

### Copying has a hard floor that no model size removes

Jelassi et al. 2024 (2402.01032), Corollary 2.8: any fixed-state recurrent model whose state holds fewer than **L·log₂(D)** bits fails to
copy a length-L string over a D-symbol alphabet more than half the time. This is a construction limit, not a training limit. Their
160M Mamba drops to zero accuracy past training length; a Hard-ALiBi transformer trained to 50 tokens copies 1000. The learned
mechanism is an ≈5-gram hash lookup. Copy-faithful decoding (ADR 104) therefore requires real KV-cache attention over the raw tokens
on at least some layers at *every* size.

### Hybrids work, and the ratio is forgiving — from 300M upward

| Source | Scale | Attention share | Finding |
|---|---|---|---|
| Mamba-2 ablation (2405.21060 §9.2.3) | 350M, 48 layers, 7B tokens | 0→8.60 ppl, 1→8.38, 2→8.32, 6→**8.26**, 15→8.34, 24→8.50, all-attention 8.68 | flat from 2 to 15 layers; space them, not first/last |
| Jamba (2403.19887) | 1.3B | 1:3 vs 1:7 | "no substantial difference" |
| Hymba (2411.13676) | 300M | 3 global layers of the stack | all-SWA loses > 20 % on recall; 3 global layers recover it at 2.7× throughput, 3.8× smaller cache |
| Hymba parallel heads | 300M | attention + SSM in one layer | recall **51.8 %** vs 40.0 % pure attention vs 19.2 % pure Mamba |
| Kimi Linear (2510.26692) | 48B/3B active | 3:1 linear:global | 25 % global at large scale |
| Zoology (2312.04927) / Based (2402.18668) | 70M–1.3B | — | 82 % of the gated-conv gap is associative recall; a 70M attention model beats a 1.4B gated-conv model on MQAR; a 64–128-token window + linear attention spans the recall/state Pareto front |

### At tiny scale the evidence says plain attention

- BabyLM 2025 controlled comparison (2510.19493): 125M Mamba vs OPT on the 100M-word track — Mamba loses on entity tracking and long
  reading; "architectural efficiency does not automatically translate into performance gains in low-data or small-scale contexts".
- No paper ablates a hybrid below 125M. MobileLLM (2402.14905) is the only controlled ≤350M architecture study: deep-and-thin beats
  wide at fixed params, SwiGLU +1.3 pts, GQA neutral, block layer sharing +0.4/+0.6 pts; embedding tying saves little at small vocab.
- NoPE beats RoPE/ALiBi on length generalisation in Kazemnejad et al. (2305.19466); RoPE's high frequencies carry position and its
  low frequencies carry content (Barbero et al. 2410.06205).

### Test-time memory is unproven below 125M and hard to parallelise

TTT (2407.04620) starts at 125M, Titans (2501.00663) at 170M, ATLAS (2505.23735) headline tables at 340M, TTT-E2E (2512.23675) proves its
scaling claim at 3B; gains widen with scale rather than appearing at it, and ATLAS/TTT-E2E state that their inner loops "cannot be
parallelized" without bespoke kernels. Miras (2504.13173) lists only vector / matrix / MLP memories — no external-store seam. The one
memory paper with gains at its smallest size is **Memory Layers at Scale** (2412.09764, 134M → 8B), a product-key top-k lookup table,
i.e. the same operation as querying an external vector index. Larimar (2403.11901) is the LLM-agnostic episodic-module pattern.

## Decision

1. **ADAM stays a transformer at every rung.** No SSM or recurrent backbone below 300M; the tiny-scale evidence is unanimous and the
   tool-calling skills (verbatim copy, entity tracking, recall) are exactly where recurrent models lose.
2. **Attention layout becomes sliding-window on most layers + 2–3 spaced global layers** (P7.5). Safe at all sizes per Hymba/Mamba-2,
   keeps the copy bound satisfied through the global KV cache, and shrinks the recomputed score chunks from L×L to L×w. It is the
   first Kaine-lineage change and is validated on `small_8k` against the unchanged release gate.
3. **The 33M rung takes the MobileLLM ablations** (P7.6): deep-and-thin at fixed params, GQA, block layer sharing, SwiGLU; NoPE as one
   ablation rung (it conflicts with the RoPE context-extension path of ADR 104 and must earn its place).
4. **The 350M rung runs the hybrid comparison** (P7.7): Gated DeltaNet as Hymba-style *parallel* heads (sequential stacking showed no
   small-scale gain) against the P7.5 SWA transformer as control, on the akh gates plus MQAR and UUID/path copy probes. Gated DeltaNet is
   the only recurrent candidate: best published UUID-needle behaviour, chunked WY form is matmuls + triangular solves (no custom kernels
   in Burn). If it does not win on the akh corpus at 350M, Kaine does not carry it.
5. **Global attention never goes away.** Whatever the 350M/1.1B results, Kaine keeps real KV-cache attention on the spaced global layers.
6. **No Titans / TTT / ATLAS.** Memory belongs in the engine (KG + HyperVec + TMS), reached through tools today. The published seam for
   moving it *inside* Kaine is a product-key memory layer (P7.9 design spike); the engine remains the store of record.
7. **Kaine's size is decided after P7.7 and P7.8**, from measured gate results, not chosen now.

## Consequences

- ADR 104's chunked attention gains a window parameter and a global-layer set; the Grug serving runtime must honour the window in its KV cache.
- ADR 037 (Phase 26) gains a concrete neural↔VSA bridge: a delta-rule layer's state is a bind/superpose store; HRR binding (Hrrformer)
  is the alternative algebra. Phase 26's "Neural→VSA bridge" item should be re-planned around that equivalence.
- Runs at 350M and above leave the Mac lane (ADR 106) and fall under ADR 105 pod governance (WI with exit condition, dollar cap,
  operator approval); the Runpod scope rule applies.
- Training-memory work needed to reach the 1.1B rung and Kaine is ADR 109.


# Phase 51 P7 — ADAM → Kaine architecture ladder and the training-engine memory program

- **Status:** Planned (2026-09-14), **gated on P6**: no P7 item starts until the P6.4/P6.5 window queue (`runs/p6-4/queue-20260914.txt`) and its release gate have finished (operator decision 2026-09-14). All items are HITL: `source=architect-session`, `draft` until the operator promotes them, never `ready`/`working`
- **ADRs:** [ADR 111](../decisions/111-kaine-architecture-ladder.md) (architecture ladder), [ADR 109](../decisions/109-training-engine-memory-program.md) (engine memory), [ADR 110](../decisions/110-kaine-character-guided-reward-live-environment.md) (character, guided reward, live environment — extension of 2026-09-14, §below); governance [ADR 105](../decisions/105-autonomous-run-governance.md); lane [ADR 106](../decisions/106-local-burn-metal-lane-and-engine-in-the-loop-training.md); attention memory [ADR 107](../decisions/107-training-memory-turboquant-vs-recompute-attention.md)
- **Anima:** project `019d9844-ce74-7993-a9f0-c953b8f04a4e`, plan `019f90cb-2e66-7f92-ba88-a4060be97c6e`
- **Predecessor:** [P6 plan](2026-09-13-phase51-p6-local-training-neurosymbolic.md) (active; P7 does not pre-empt the P6.4/P6.5 window queue)

## Goal

Decide Kaine's architecture from measurements taken on the ADAM ladder (33M → 350M → ≈1.1B) rather than from extrapolation, and bring
the training engine's memory down far enough that each rung trains on the cheapest hardware the literature says it can: 33M–125M on
the Mac, 350M and 1.1B on one pod GPU, Kaine on whatever P7.8 shows is needed.

## Starting facts

| fact | value | source |
|---|---|---|
| Current model | `small_8k`: d_model 512, 8 heads, 8 layers, d_ff 2048, RoPE, ctx 8192, chunk 1024, LayerNorm, GELU, no dropout | `crates/grug-train/src/transformer.rs` |
| Attention memory | O(L·d) retained per layer (`RecomputeAttention`), 8192×2 = 7.0 GB on the M2 | ADR 107 |
| Residual memory | `small_8k` 2.7 GB activations + 0.4 GB state; `350m_8k` ≈18 GB + 4.2 GB | ADR 107 |
| Checkpointing | per-op; Linear/embedding/matmul inputs always retained | `burn-autodiff-0.21.0/src/checkpoint/strategy.rs:62-96`, `src/ops/module.rs` |
| Fusion | not enabled (`metal = ["wgpu"]` lacks `fusion`) | `burn-0.21.0/Cargo.toml:147-157`, `burn-wgpu-0.21.0/src/lib.rs:63-64` |
| LayerNorm | statistics reduced in backend dtype, no f32 cast | `burn-backend-0.21.0/src/backend/ops/modules/base.rs:1103-1129` vs `burn-nn-0.21.0/src/modules/norm/rms.rs:74` |
| Cross-entropy | full `[b·s, vocab]` logits + log_softmax retained | `crates/grug-train/src/train.rs:500-527` |
| Optimizers in Burn | Adam, AdamW, Adagrad, RMSProp, SGD, LBFGS, Adan, Muon; no Adafactor/8-bit | `burn-optim-0.21.0/src/optim/mod.rs:9-30` |
| f16 lane | parked: loss 6.30 vs 3.76, 2× slower; not a loss-scale problem | `training/results/p6-1-metal-smoke-20260913-RESULT.md:242-249, 337-340` |
| Copy floor | fixed-state model needs ≥ L·log₂(D) bits to copy an L-token span | Jelassi 2402.01032 Cor. 2.8 |
| Hybrid ratio | 2–15 attention layers of 48 within 0.06 ppl at 350M; 3 global layers recover recall at 300M | Mamba-2 2405.21060, Hymba 2411.13676 |
| Small-scale verdict | 125M Mamba < OPT on BabyLM; 70M attention > 1.4B gated-conv on recall | 2510.19493, Zoology 2312.04927 |

## Work items

| WI | scope | exit condition |
|---|---|---|
| `01a0a1a2-aa69-7c72-9596-76b319e9cd9f` P7.1 fusion + f32-stats LayerNorm + fp32 softmax/CE + f16 gate rerun | enable Burn `fusion` on metal/gpu/cuda; hand-written LayerNorm with f32 statistics; fp32 accumulation for softmax and masked CE under f16; dynamic loss scaling | fusion tokens/s + memory table at 2048×1 / 8192×1; f16 loss@4 within 0.05 of f32 on the 2048×1 smoke **and** faster, else f16 stays parked with the measured reason |
| `01a0a1a2-cf26-78d3-853d-f1d8c0771074` P7.2 `RecomputeFeedForward` | custom `Backward` for up-proj → GELU → down-proj, row-chunked, saves only the block input; vendored patch as ADR 107 | CPU gradient equality to 1e-4; ≥ 50 % of the 18 GB removed at `350m_8k` 8192×1 (analytic + measured on the largest shape that fits); overhead < 30 % |
| `01a0a1a2-ef2c-7fd1-802f-aa20e74c51e5` P7.3 Cut Cross-Entropy | blockwise loss op with online logsumexp; reference path kept for tests | gradient equality at vocab 8000 / seq 512; memory delta at `small_8k` 8192×2; identical loss curve |
| `01a0a1a3-157a-76d1-a38b-424107f82fc7` P7.4 optimizer selector | `adamw` / `muon` (2-D weights) + AdamW (embeddings, norms, head) / Adafactor fallback | loss parity on the 2048×1 smoke and one remediation window on the unchanged sentinels; state bytes halved |
| `01a0a1a3-439f-7420-baa8-ed626623cbe8` P7.5 SWA + spaced global layers | `with_attention_layout(window, global_layers)`; windowed `RecomputeAttention`; presets `akh_tool_small_8k_swa` (w 1024, global {2, 5}) and 350M; serving KV cache honours the window | windowed gradient test; copy probe (8–64-token UUID/path spans at 1k–8k distance) and MQAR parity vs full attention; constrained gate on 401,577 after one window within 3 cases; memory + tokens/s at 8192×2 |
| `01a0a1a3-6c2b-78a0-abc4-3ff3d95230e8` P7.6 MobileLLM presets + NoPE rung | GQA, SwiGLU, block layer sharing; `adam_33m_deep`, `adam_33m_shared`, `nope` variant; matched-param, matched-token comparison at 2048/4096 | RESULT table (tool gate, knowledge gate, copy probe, tokens/s, memory) at 33M; winning shape becomes the ADAM small preset |
| `01a0a1a3-a8a2-7540-bc02-98db4507bbcd` P7.7 Gated DeltaNet parallel heads @ 350M (depends P7.5) | `GatedDeltaNet` (chunk 64, WY form) + `ParallelHead` gating; `adam_350m_hybrid`; pod run under ADR 105 | recurrence gradient test; MQAR + UUID/path probes + akh gates for hybrid vs SWA control in one table; go/no-go for 1.1B and Kaine |
| `01a0a1a3-d7c3-7602-a6d2-0fc09798199e` P7.8 spike: 1.1B fit (depends P7.2–P7.4) | `adam_1_1b_8k` preset (d_model 2048, 24 layers, GQA, SWA layout); 50-step pod smoke, levers on/off, cap ≤ 2 h A100 with in-thread approval; Burn multi-device audit | measured fit table; written decision "1.1B trains on one A100 at 8192: yes/no, batch"; the named blocker for the Kaine size |
| `01a0a1a4-03cb-75f3-b666-67eba03eb420` P7.9 spike: memory-layer seam (depends P7.5, informed by P7.7) | design for a product-key `MemoryLayer` whose keys/values are learned or projected from HyperVec/KG embeddings; episodic writes → `DerivationKind::AgentConsolidation`; copy-bound implications | ADR draft with the seam, parameter budget at Kaine size, go/no-go for a prototype rung |

## Sequence

1. **Engine first, on the Mac, after the P6 window queue and its gate have finished:** P7.1 → P7.2 → P7.3 → P7.4. Each lands behind a
   flag with its gradient test and enters a lane only after its gate. The P6 recipe (f32 + recompute attention) is never changed
   retroactively; P6's fine-tuning result is recorded before any P7 lever touches the lane.
2. **Architecture at 33M:** P7.5 and P7.6 on the Mac from the foundation corpus, scored on the unchanged sentinels. P7.5 is also an
   engine memory lever (L×w score chunks), so it may move ahead of P7.3/P7.4 if the P7.2 measurement says activations are still the floor.
3. **350M rung:** P7.7 on a pod (ADR 105 WI, dollar cap, exit condition) with the P7.5 350M preset as control. This is the first
   measurement that can change Kaine's backbone.
4. **1.1B rung:** P7.8 spike; only after P7.2–P7.4 so the arithmetic is measured, not assumed.
5. **Kaine:** size and backbone chosen from the P7.7 + P7.8 tables; P7.9 decides whether engine memory enters the model as a layer.

## Budget and gates

- Mac lane (ADR 106): a P7 engine change must not reduce tokens/s below the P6 gate (2048-context window ≤ 24 h) unless it buys the
  8192 fit for a shape that did not fit before.
- Pod runs (P7.7, P7.8): each has its own Anima WI thread with operator approval, a dollar cap (P7.8 ≤ 2 h A100 ≈ $4; P7.7 sized in its
  thread from the P7.8 tokens/s measurement) and an exit condition; events go to the Agent Board run thread, never a chat wake-up.
- Every WI writes a RESULT doc under `training/results/` and a thread post; architecture WIs also record the constrained-gate delta
  against candidate 401,577 so ADAM's release line is never regressed silently.

## What this plan does not do

- It does not change the ADAM release gate or the P6 remediation windows.
- It does not adopt Titans/TTT/ATLAS, activation quantisation, LOMO, FP8 or an engine swap (ADR 111 §Decision 6, ADR 109 §Decision 2, 4).
- It does not fix Kaine's parameter count; that is the output of P7.7 and P7.8.

## ADR 110 extension (2026-09-14): character, guided reward, live environment

Inputs: Cantrill, *Intelligence is not Enough* (Monktoberfest 2023); Hatcher on Gibson, *we act to perceive*; Oxide's principles and
values. Decisions are in [ADR 110](../decisions/110-kaine-character-guided-reward-live-environment.md): (1) the crew-of-8 slots carry
Oxide's 15 values as their public trained biases, the three principles are captain invariants enforced by the runtime; (2) reward is
guided and fine-grained — every step scored, every failed step yields a corrected sibling, a five-component reward vector
(effect / validity / provenance / disposition / budget) with a named reduction policy, weights in [0, 1], never a penalty, no
scalar-judge policy gradient; (3) episodes come from the real environment — `Engine` + the full `ToolRegistry` + Agent Board + Anima +
executor `Backend`, run as a sandboxed dojo tenancy, one dispatch per episode, end to end.

### New work items

| WI | scope | exit condition |
|---|---|---|
| `01a0a1b9-eb77-7390-9730-361a9d92faae` P7.10 `AkhEnvironment` dojo tenancy | stepped environment over the real Engine, full `ToolRegistry`, Agent Board (`DjehutyClient` + `akh-session-services` inbox intake, `coordination/*@v1`) and Anima (`get_work_item`, `submit_session_change`, `git_publication`, thread verdict) with the per-step recorder; full-registry `ToolContractSnapshot`; network tools on recorded fixtures | one authored dispatch runs end to end with every step recorded to JSONL; contract hash recorded; decontamination green against the four sentinel sets; dojo tenancy IDs documented; negative check shows nothing written outside the tenancy |
| `01a0a1ba-255a-7f70-afa8-85016faad6ab` P7.11 guided reward vector + trainer (depends P6.5, P7.3) | sidecar schema v1 (five components + `voice_called_for` + `corrected_sibling`), versioned `weight_policy`, `train_knowledge --record-weights use --weight-policy v1`, weighted loss through Cut CE | gradient equality of the weighted blockwise loss vs the naive weighted path; one A/B window from 401,577 (vector + corrected siblings vs E3 scalar) on the unchanged sentinels, delta recorded |
| `01a0a1bc-3abe-7042-9c17-5acdd9c1c551` P7.12 crew × Oxide corpus (depends P7.10, P7.11) | `slot` + `disposition` labels on the kata drills (15 values / 8 slots); emergent-property-bug and act-to-perceive families generated in the tenancy with passive controls; captain-selection labels; blinded social rubric; White→Yellow belt gate | corpus + generator hash; belt gate passed on the ADAM candidate without regressing the 1,272-case gate by more than 3 cases; RESULT with per-slot and per-family numbers |
| `01a0a1bc-c798-7003-9879-934fa3568507` P7.13 environment release gate (depends P7.12, P7.5) | frozen quarantined dispatch-episode set with deterministic final-state verifiers + reward-vector components + blinded rubric; evaluator under `crates/grug-train`; RESULT row every rung and every Kaine candidate emits | gate definition frozen (episode/verifier/rubric hashes); ADAM baseline recorded; runs within the ADR 106 local budget or the excess is measured and a pod run proposed under ADR 105 |

### Amendments to P7.1–P7.9 (recorded on the Anima items)

- P7.3: the blockwise loss accepts per-record weights; the gradient test covers the weighted path.
- P7.5–P7.8: each RESULT table gains the P7.13 environment-gate row once the gate exists, the P7.12 belt gate until then; P7.5 keeps
  board/Anima ID spans inside the global-layer copy path; P7.7's Kaine go/no-go is taken on all four gates; P7.8's memory table includes
  the weighted-loss path.
- P7.9: board/Anima episodic writes are the first use case for the memory-layer seam; `disposition`/`effect` are candidate write signals.

### Sequence and gating

Unchanged: nothing under this extension starts before the P6 window queue and gate are recorded, and every item stays `draft` +
`source=architect-session` until the operator promotes it. P7.10 is CPU-side harness code with no GPU or recipe contention; promoting
it earlier is an operator option, not a default. Order once open: P7.10 → P7.11 (after P6.5's E3 result and P7.3) → P7.12 → P7.13;
P7.13 then becomes a row in every architecture rung's RESULT table.

### Budget

- P7.10–P7.12 run on the Mac (CPU for the harness and corpus; the belt-gate window under the ADR 106 lane gate).
- P7.13's gate must fit the ≤ 12 h M2 evaluation budget or state the excess; any pod use is its own ADR 105 WI with a dollar cap.
- Dojo tenancy resources (Anima project, board tenancy tokens, dojo repository) are created once and named in the P7.10 RESULT.

