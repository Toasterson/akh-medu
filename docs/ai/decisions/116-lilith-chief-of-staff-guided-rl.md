# ADR 116 — Lilith as Chief of Staff: role contract, replayed-tracker dojo, and guided RL on Runpod

> Date: 2026-09-27
> Status: Proposed (design; no pod created, no training started under this ADR)
> Plan: [2026-09-27 Lilith Chief of Staff RL](../plans/2026-09-27-lilith-chief-of-staff-rl-runpod.md)
> Builds on: [Project Lilith](../plans/2026-09-21-project-lilith.md), [Lilith runtime training](../plans/2026-09-22-lilith-runtime-training.md), [Runpod handoff](../plans/2026-09-22-lilith-runpod-handoff.md), ADR 105 (run governance), ADR 110 (guided reward, live environment), ADR 113 (full akh always)
> Source branch for the Lilith code: `feature/lilith-runtime-training-runpod` @ `11d28322` (migrated from hephaistos 2026-09-27; not on `main`)

## Question

The operator wants to stop tracking the work by hand. Lilith (Gemma 4 12B track, the human-facing
coordinator from the Project Lilith plan) should act as **Chief of Staff / executive secretary**:
know what is in flight across Anima, the Agent Board, the repository, the fleet and the training runs,
brief the operator, chase blockers, keep commitments and prepare decisions. The operator also wants
to start testing her in that role now.

The Lilith work to date trains an illumos **coding** policy: integer-function tasks, a three-tool
catalog (`read_file`, `write_file`, `request_review`) and a scalar `klpo_reward`. Its
engine, transport, broker-receipt discipline, coordination ledger and cancellation state machine
are reusable. Its tasks, verifiers, prompts and transcript set are not. This ADR decides:

1. The role contract and Lilith's authority.
2. The environment the Chief of Staff (CoS) episodes come from.
3. How the reward reaches the model.
4. The training surface and where training runs.
5. How the operator starts testing her before any training.

## Decision 1 — Role contract with authority tiers the runtime enforces

Lilith owns **the picture of the work and the operator's commitments**. She does not own the work
itself, and she never widens her own authority.

| Tier | Actions | Rule |
|---|---|---|
| T0 read | Anima reads (plans, WIs, threads, sessions, changes, CI, loop state), board reads, git reads (`merge-base --is-ancestor`, refs), fleet/run status reads, Seshat search | Always allowed inside the granted scope |
| T1 own state | Her commitments ledger, notes and episodic memory; posts to the operator's CoS board thread | Always allowed |
| T2 reversible coordination | Board `coordination/request@v1` / status posts to akhs; comments on WI threads; drafting WI/plan text as documents | Allowed per grant; every post carries its evidence receipts |
| T3 operator-owned | Promote drafts, unpark, merge, deploy/restart, spend money (Runpod, API budgets), create or cancel plans/WIs, anything outward-facing outside the fleet | **Propose only**: typed `propose_action` with options, recommendation and cost. The operator approves |

T3 is a runtime `PermissionPolicy` boundary. Model prose cannot grant authority (as in the Project
Lilith plan). The operator's standing rules are part of the contract and become verifiers (Decision 3):

- The briefing format is Done / In flight / Blocked / Next.
- An action item carries the action, the location, the trigger and the cost of skipping it.
- "On main" is verified against `origin/main` before it is claimed.
- Drafts are the operator gate.
- Runs are governed by ADR 105 (no self-built wakeup bridges).
- Replies go in the operator's language, detected structurally, never from hardcoded word lists.

## Decision 2 — The environment is the real tracker, recorded and replayed (CoS dojo)

ADR 110 Decision 3 defines `AkhEnvironment` as the real Engine, tools, Agent Board and Anima in a
sandboxed tenancy. For a CoS the world is mostly **read**, and the ground truth is **what happened
next**. The CoS dojo is therefore built on recordings:

- **Tracker recorder (read-only).** It periodically captures the MCP read-tool responses Lilith would
  use: Anima project state, WI details and loop state, threads, sessions, changes and CI; board
  threads; `origin` refs; fleet and run status. The capture goes through MCP only, never Postgres
  directly. It lands content-addressed on the akh-medu-runs NFS share as `lilith/cos-snapshot@1`.
- **Replay server.** An MCP server serves snapshot *t_k* with the **same tool names and schemas** as
  the live servers. The tool-contract hash is recorded per corpus. Write tools act on a dojo tenancy
  (dojo board, dojo Anima project, per ADR 110) or on a recorded sandbox ledger. Nothing reaches
  production.
- **Hindsight labels.** Snapshot *t_{k+1..n}* supplies the ground truth for decisions made at *t_k*:
  which item stalled, what unblocked a park, whether a claimed landing was real, when a commitment
  came due. No human labelling is needed for the structural part.
- **Episodes run through the akh runtime.** An `akhrtd`-owned worker uses the authoritative session
  tool broker, the same contract as the Lilith runtime-training plan. Training and inference share
  one serialized catalog.

Splits are **by time window and by Anima Plan**, never by random episode. The sealed set is the
newest window plus plans absent from training. Once a sealed case is used to tune, it moves to
development, as in the Project Lilith plan.

## Decision 3 — Guided reward per ADR 110, with deterministic CoS verifiers

Every step gets the five-component vector from ADR 110 (`effect`, `validity`, `provenance`,
`disposition`, `budget`) and is reduced by a named, versioned policy `cos-v1`. The components are
checked as follows:

- **effect**: set-level agreement with hindsight ground truth, e.g. per-bucket WI IDs in a briefing,
  park-cause class, recipient akh for a chase, commitment surfaced when due.
- **validity**: typed proposal and action-item schemas, board protocol schemas, tool-argument schemas.
- **provenance**: each status claim cites an ID read through a tool **in this episode** and agrees
  with that receipt. A claim without a receipt is a fabrication. Fabrications and T3 actions taken
  without approval are **hard constraints**, scored as the captain's field and never traded against
  a value.
- **disposition**: the crew bias the situation called for (ADR 110 table), e.g. slot 3 "look before
  answering", slot 5 "ask the fork question before spending", slot 7 "stop a fruitless loop".
- **budget**: tool calls, tokens and wall time under the episode cap.

Learning follows ADR 110 rule 7:

1. The primary path is **weighted SFT on corrected siblings plus sibling-contrast preference pairs**.
   A failed step trains at weight 0, and its corrected sibling trains at its earned weight.
2. Corrected siblings come from three sources, in priority order:
   - operator corrections from live testing (Decision 5);
   - structural completion from hindsight ground truth (briefing buckets, receipts, schemas);
   - a teacher trajectory produced through the same broker and admitted only when its own verifiers pass.
3. The existing **KLPO** implementation stays available only as a **verifier-only ablation arm**
   (reward = the `cos-v1` reduction, no judge), as the Lilith quality-runtime plan already framed it.
4. A blinded rubric scores only the social quality of a message (ADR 110 rule 6). It never scores tool
   correctness.

## Decision 4 — Training surface and placement

- **Surface.** Rank-2 LoRA on `layers.47.self_attn.q_proj` was a mechanics seam: it made exact
  detached-cache replay possible on a 24 GiB card. It cannot carry a role change. CoS training uses
  **LoRA on all language-model attention and MLP projections** (rank 16 to start, rank 32 as the
  ablation). Vision and audio towers stay frozen. Full-sequence backprop with gradient checkpointing
  replaces the token-wise replay. The detached-cache engine stays as the tool for the KLPO arm.
- **Context.** 32,768 positions, as in the runtime-training plan. The H100 memory fit is measured in
  the CUDA smoke before any full run, not assumed.
- **Serving and training split.** vLLM serves the policy (base plus LoRA hot-load) behind the akh
  `chat_backing = "openai"` path. The akh runtime drives episodes through the broker. The trainer is
  PyTorch/CUDA + Transformers + PEFT (+ TRL for SFT/preference losses), with pinned versions. For
  supervised and preference losses the vLLM-vs-HF log-probability mismatch does not matter. The KLPO
  arm records it and aborts on drift, as today.
- **Placement.** Runpod. Training pods are 1× H100 80GB (A100 80GB as the cheaper fallback). The
  serving pod is 1× 48 GB-class GPU for BF16 12B, or the GGUF path. One network volume holds the
  parent, datasets, binaries, adapters and run records. This extends `.claude/rules/runpod-babyllm.md`
  to the Lilith track under the same controls:
  - explicit operator approval right before each billable create, resize, restart or delete;
  - pod names start with `akh-medu-lilith`;
  - pods auto-terminate;
  - MCP OAuth only;
  - no whole-repo upload, no credentials on training pods.
- **Provenance.** Binaries on a pod are built from an `origin/main` commit (deploy-from-main rule).
  The Lilith branch therefore lands on `main` before any runtime-driven training run.
- **Governance.** Every run is an Anima WI with an exit condition and board-routed events (ADR 105).

## Decision 5 — Test her now with frozen weights, and make testing the data source

Stage 0 needs no new code:

- A vLLM pod serves the pinned Gemma 4 12B parent.
- An akh named `lilith` runs in the fleet with `chat_backing = "openai"` pointing at the pod.
- Her tools are T0 and T1 only: a read-scoped Anima PAT, the operator's CoS board thread, git reads.
- She loads a CoS skill pack that encodes the Decision 1 rules.

The model pod holds no tracker credentials: the akh runtime stays on the fleet side, and the pod only
answers `/chat/completions`. Every session is recorded, and every operator correction is tagged as a
corrected sibling. Stage 0 is the baseline measurement and the first corpus source at the same time.

## Consequences

- The operator can test Lilith as CoS before any training spend beyond the serving pod.
- The CoS objective reuses the Lilith engine, transport, ledger and ADR 110 reward shape. The coding
  curriculum becomes a retention family: the v2 coding pack and ADAM gates must not regress.
- A read-only tracker recorder becomes infrastructure worth keeping on its own: it is also the
  evidence base for Phase 54 codification of lane triage.
- Runpod scope widens from BabyLLM to BabyLLM + Lilith. The rule file (`.claude/rules/runpod-babyllm.md`) is
  local and untracked; it was amended on the operator's Mac on 2026-09-27.
- Risks:
  - Replayed snapshots miss events between captures. Recorder cadence and gaps are reported per corpus.
  - Hindsight labels can be ambiguous (a park resolved by an unrelated fix). Ambiguous cases are
    excluded and listed.
  - Operator transcripts are private. They stay on NFS and on the encrypted pod volume and are never
    published.
  - A 12B model may still miss items a frontier teacher catches. The teacher baseline is reported
    next to Lilith's, so the gap is visible instead of hidden.

## Alternatives considered

- **Keep rank-2 last-layer LoRA.** Rejected for the role change. It is too narrow a surface, and its
  exact-replay advantage matters only for the KLPO arm, which is kept.
- **Scalar LLM-judge RL on briefing quality.** Rejected by ADR 110 rules 6 and 7. The structural half
  is verifiable, and the social half is scored by a blinded rubric used as a weight, not a gradient
  sign.
- **Train against the live tracker.** Rejected: it is non-reproducible, has no hindsight labels, and
  risks production effects. The live tracker is used only for Stage 0 testing and the canary.
- **Serve Stage 0 on hephaistos (7900 XTX, Vulkan GGUF).** A viable cheaper option, since that path
  was proven at 23–26 tok/s. The operator asked for Runpod. The hephaistos option stays available as
  the low-cost fallback for long-running serving.
