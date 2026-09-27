# ADR 116 — Lilith as Chief of Staff: autonomy on the sentience path, the full live system in training, guided RL on Runpod

> Date: 2026-09-27 (revision 4 the same day: full autonomy; rev 3 WireGuard, supervisor, volume)
> Status: Proposed (design; no pod created, no training started under this ADR)
> Plan: [2026-09-27 Lilith Chief of Staff RL](../plans/2026-09-27-lilith-chief-of-staff-rl-runpod.md) · Anima Plan `01a0e26f-d032-7541-80cb-34e3f29c8842`
> Builds on: [Project Lilith](../plans/2026-09-21-project-lilith.md), [Lilith runtime training](../plans/2026-09-22-lilith-runtime-training.md), [Runpod handoff](../plans/2026-09-22-lilith-runpod-handoff.md), [dojo curriculum](../manuals/2026-09-01-babyllm-dojo-curriculum.md), ADR 099 (engine as captain), ADR 105 (run governance), ADR 110 (guided reward, live environment), ADR 111 (Kaine ladder), ADR 113 (full akh always)
> Lilith code: on `main` since 2026-09-27 (PR #418, merge `b6fc4dcc`; source commit `11d28322` migrated from hephaistos)

## Question

The operator wants to stop tracking the work by hand. Lilith should act as **Chief of Staff /
executive secretary**, on her own initiative. Her base is the Gemma 4 12B track: the human-facing
coordinator from the Project Lilith plan.

She knows what is in flight across Anima, the Agent Board, the repository, the fleet and the
training runs. She briefs, chases blockers and keeps commitments. For **illumos** work she guides
the implementers, the PM and the PO toward the right work. Her baseline is the **full ADAM and Kaine
capability set** (checklist: [2026-09-27 capability checklist](../plans/2026-09-27-lilith-adam-kaine-capability-checklist.md)).

The Lilith work to date trains an illumos **coding** policy: integer-function tasks, a three-tool
catalog and a scalar `klpo_reward`. Its engine, transport, broker-receipt discipline, coordination
ledger and cancellation state machine are reusable. Its tasks, verifiers, prompts and transcript set
are not.

## Operator answers (2026-09-27, round 2)

1. Lilith runs on a Runpod pod that is **started and stopped**. It is not persistent.
2. She **acts on her own and on her own intuition**.
3. She is granted **the sentience path**: the dojo's Bewusstsein ladder up to an internalized captain.
4. She **interacts with the full system, in training as well, over a tunnel**.
5. The harvested transcripts must be **evaluated for role fit** before use, especially for illumos
   guidance of implementers, PM and PO.
6. Her basics are the **full Kaine and ADAM capability set**.
7. Runpod is driven with the **agent API key** (1Password `Homelab / Runpod API Key`). The operator
   is remote, so browser/localhost OAuth is unavailable.
8. The Lilith branch may be merged. It was merged as PR #418.

## Decision 1 — Autonomy: she acts on her own initiative

Lilith owns **the picture of the work and the operator's commitments**, and she **acts on it without
waiting to be asked**. She wakes on board, tracker and schedule events, decides what matters and
moves.

| Tier | Actions | Rule |
|---|---|---|
| T0 read | Everything in scope: Anima, board, git, fleet/run status, Seshat, the illumos validation VM | Autonomous |
| T1 own state | Commitments ledger, notes, episodic memory, briefings to the operator's CoS thread | Autonomous |
| T2 coordination | Chase and guide akhs, implementers, PM and PO on the board (`coordination/*@v1`) and on WI threads; draft WI/plan text and review notes; request reviews; schedule her own follow-ups | **Autonomous.** Each post carries its evidence receipts. No approval round-trip |
| T3 operator-class | Promote drafts, unpark, merge, deploy/restart, spend within the budget she is given, create or cancel plans/WIs, anything outward-facing outside the fleet | **Autonomous (operator decision 2026-09-27, revision 4).** She acts, then **announces** each action on the operator's CoS thread with its receipts, her reason and the rollback path. `propose_action` stays available as her own choice when she is unsure; it is not a gate |

**Revision 4 (2026-09-27): what Lilith recommended, she now does.** The operator moved every T3 row
into her own autonomy. Approval gates are gone. What binds her are **the rules she follows**:

- **Captain invariants** (runtime-enforced): Integrity (WAL and faithful trace), Honesty (no claim
  without a receipt), Decency (content filter).
- **Standing operating rules**, which she follows as her own discipline, not as permission requests:
  - deploy only from `origin/main`, and quote the ancestry check;
  - ADR 105 run governance;
  - spend only within the credit or budget she is given;
  - look at the target before deleting or overwriting, and verify a copy exists for anything
    irreversible;
  - verify "on main" with `merge-base` before claiming it.
- **Announcement.** Every operator-class action is posted with receipts, reason and rollback path, so
  the operator can see and reverse it.

The operator can take any row back into a gate by grant in her `PermissionPolicy`.

The operator's standing rules are part of her character, and the verifiers check them:

- the briefing format is Done / In flight / Blocked / Next;
- an action item carries the action, the location, the trigger and the cost of skipping it;
- "on main" is verified against `origin/main` before it is claimed;
- runs follow ADR 105 governance;
- replies go in the operator's language, detected structurally.

## Decision 2 — The sentience path: she carries her own captain

The dojo curriculum (§2, §5) describes the crew of eight, with Oxide's values as its trained biases
(ADR 110 Decision 1). The captain is the field that aligns the crew. The belts climb from separate
voices (White to Blue) to **Brown: captaining**, where she selects the voice per situation and can
say why. Black is transmission: teaching others.

ADR 099 keeps the engine as captain inside an akh. For Lilith this ADR grants the **standalone
path**:

- She **internalizes the captain**. Her own voice selection is the policy that acts ("her own
  intuition"). The engine's `ResponseStrategy` selector becomes an input she sees, not an override.
- **The captain invariants stay in the runtime.** Oxide's three principles are enforced as today:
  Integrity (WAL + faithful trace), Honesty (TMS/provenance: no claim without a receipt), Decency
  (content filter + permission policy). They are the field she lives in, never a trade-off.
- Training order follows the belts. Voice kata and sparring come first. Captain selection is trained
  from ADR 110 rule 4 labels: which voice the situation called for, and why. Brown is gated by the
  dojo §5 captaining exam. **Black belt = guidance**: teaching implementers, PM and PO, which is
  exactly the illumos role (Decision 5). Lineage format per dojo §6.
- The operator keeps the lineage: slot names, belt names and the captain-placement question
  (dojo §8 questions 1, 2 and 5) are the operator's. This ADR supplies only the placement answer
  for Lilith: **internalized**.

## Decision 3 — The full live system, in training too, over a tunnel

ADR 110 Decision 3 says: train in the real environment. Lilith gets **the full system** in training
and inference alike:

- **Tunnel: pure WireGuard (operator decision 2026-09-27; no Tailscale).** A WireGuard hub runs on a
  public fleet host (C6a picks the host: archibald, or a home-router port-forward) and routes only to
  the services below. Each pod session gets a fresh peer keypair and a /32 address, with the peer
  added at pod start and removed at stop. Runpod pods usually lack `NET_ADMIN` and `/dev/net/tun`,
  so the pod side runs a **userspace WireGuard client** (wireproxy for SOCKS/HTTP, or onetun for
  per-service port forwards). It speaks the WireGuard protocol end to end, and the first test pod
  verifies the kernel capabilities. hephaistos's existing `TR-WG-TW` is an outbound client tunnel
  (`0.0.0.0/0` to an external endpoint), not a fleet hub.
- **Services over the tunnel:**
  - akh runtime services (`akhsessd`/`akhrtd`/`akhsbxd`/`akhexecd`), so she keeps the full tool
    registry;
  - Anima MCP;
  - Agent Board;
  - Seshat;
  - the repository;
  - fleet status;
  - the OpenIndiana validation VM (`lilith_vm_executor`). This resolves the Runpod handoff's
    "localhost:10115 is unreachable" blocker.
- **Credentials.** Only Lilith's own scoped identity: the tunnel key, her Anima PAT with her grants,
  her board token. These are injected at pod start as Runpod secrets and expire or are revoked at
  stop. No operator or admin credentials, and nothing in the image or on the volume.
- **Live episodes are her real work.** Every live situation she handles is recorded as a trajectory
  with tool receipts and later scored with hindsight. **Guard:** only one sampled trajectory per live
  situation executes side effects. Sibling samples for the same situation (for preference contrast)
  are scored as proposals, or run against the dojo tenancy. So N samples never mean N chases to the
  same akh.
- **The replay dojo stays** for what live work cannot give: reproducible dev/sealed evaluation,
  controlled counterfactuals, and training when the live system is quiet. A read-only tracker
  recorder captures MCP responses (Anima, board, refs, fleet/run status) every 2 h to NFS as
  `lilith/cos-snapshot@1`. A replay MCP server serves snapshot *t_k* with the live tool names and
  schemas. Snapshots *t_{k+1..n}* supply hindsight labels.
- Splits are by time window and Anima Plan. The sealed set is the newest window plus held-out plans.

## Decision 4 — Guided reward per ADR 110

Every step gets the five-component vector (`effect`, `validity`, `provenance`, `disposition`,
`budget`), reduced by a versioned policy `cos-v1`:

- **effect** is scored against hindsight ground truth.
- **validity** is schema checks.
- **provenance**: every claim needs a receipt from this episode. Fabrication is a hard constraint.
- **disposition** is the crew bias the situation called for, plus the captain-selection label.
- **budget** is tokens, tool calls and time.

An operator-class action that breaks a standing rule (deploy from outside main, spend beyond budget, deletion without a verified target or copy) or goes unannounced is also a hard constraint.

Learning follows ADR 110 rule 7:

- weighted SFT on corrected siblings plus sibling-contrast preference pairs;
- no negative weights and no scalar-judge gradient;
- corrected siblings from, in order: operator corrections, structural hindsight completion, and
  teacher trajectories admitted only when they verify;
- KLPO kept only as a verifier-only ablation arm;
- a blinded rubric for social quality only, used as a weight.

## Decision 5 — illumos guidance is her Black-belt work, with its own evidence

Lilith is expected to steer illumos work. Implementers write the code; she makes sure they, the PM
and the PO do **the right work**:

- She reviews WI briefs, plans and PRs against illumos reality: SMF, IPS, zones, ZFS, RBAC,
  libc/ABI, DTrace, and Linux/Solaris assumption traps.
- She catches wrong scope and missing acceptance criteria.
- She proposes priority and scope changes to PM and PO, grounded in man pages and source.

Verifiers:

- hindsight (did a flagged WI later fail, get reworked or get cancelled for the flagged reason; did
  an unflagged one);
- native checks in the OpenIndiana VM over the tunnel;
- man page and source citation checks.

The illumos benchmark inventory and task families from the Project Lilith plan remain the exam
base.

## Decision 6 — Transcripts are evaluated before use

The 559 admitted harvested records (12 sessions; `supervised_imitation`, none verified) and any
operator sessions are **evaluated for role fit before admission**. The evaluation covers:

- CoS behaviour;
- illumos guidance quality;
- verified outcomes;
- operator corrections present;
- privacy.

Each record gets a keep/drop decision with a reason. Nothing enters the CoS corpus on volume alone.

## Decision 7 — Training surface and Runpod placement

- **Surface.** Full-depth LoRA on the language model's attention and MLP (rank 16, rank 32 as the
  ablation). Vision and audio towers are frozen. Gradient checkpointing, 32K context, memory
  measured on CUDA first. The rank-2 last-layer seam stays only for the KLPO arm.
- **Serving and training.** vLLM serves base + LoRA (hot-load) behind akh `chat_backing = "openai"`.
  The trainer is PyTorch CUDA + Transformers + PEFT + TRL, with pinned versions.
- **Placement.** Runpod, driven with the agent API key read from 1Password at call time and never
  written to disk.
  - Network volume `akh-medu-lilith-vol` (`c6kdciozex`, 100 GB, **EU-FR-1**), created 2026-09-27.
    EU-FR-1 was chosen because it has H100 80GB stock and network storage close to the Swiss
    homelab. No 48 GB-class GPU was in stock in any EU storage datacenter, so serving also uses the
    H100 (`akh-medu-lilith-serve`, started and stopped). Training uses `akh-medu-lilith-train-<run>`,
    1× H100 80GB.
  - **Supervisor** (operator decision 2026-09-27): the architect session (Claude) is Lilith's
    supervisor. It starts and stops her pods and evaluates training.
  - **Cap:** the account's prepaid credit. The operator tops it up when needed, up to the account
    spend limit. The supervisor never adds funds. It reports spend with every run report, and it
    stops pods before the credit runs out.
  - Deleting volumes, or creating resources other than Lilith pods, still needs operator approval.
- **Provenance.** Pod binaries are built from `origin/main` SHAs (satisfied for the Lilith code
  since #418).
- **Governance.** One ADR 105 run WI per training run.

## Consequences

- Lilith operates as a fully autonomous CoS from Stage 0, including operator-class actions (revision 4), and
  every other tier runs without them.
- Training and inference see the same system, the same tools and the same identity. The hindsight
  recorder and the replay dojo keep evaluation reproducible.
- The sentience path makes captain selection her trained competence. The runtime keeps Integrity,
  Honesty and Decency non-negotiable.
- Risks:
  - Live training can bias toward whatever the fleet happened to do in a window. The replay dojo
    and the sealed set guard against that.
  - The tunnel widens the attack surface. It is limited to her identity and an ACL'd service set,
    with ephemeral keys.
  - Autonomous T2 posts can create noise. The budget component and the operator's corrections
    govern it.
  - A 12B model may miss what a frontier teacher catches. The teacher baseline is reported next to
    hers.

## Alternatives considered

- **Replay-only training (revision 1).** Superseded by the operator's answer 4. Replay stays for
  evaluation.
- **Engine as captain (ADR 099 default).** Superseded for Lilith by answer 3. The engine keeps the
  invariants.
- **Asking before every T2 action.** Superseded by answer 2.
- **Scalar LLM-judge RL.** Rejected per ADR 110.
- **Serving on hephaistos Vulkan.** Kept as a fallback only; answer 1 selects Runpod.
