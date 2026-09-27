# Lilith as Chief of Staff — guided RL on Runpod

- **Date:** 2026-09-27 (revision 2, after the operator's answers)
- **Status:** Proposed. Slices are drafts in Anima Plan `01a0e26f-d032-7541-80cb-34e3f29c8842` on
  the **Local implementation — human pickup** workflow (claim with `claim_local_work_item`, hand back
  through `submit_local_work`).
- **ADR:** [116](../decisions/116-lilith-chief-of-staff-guided-rl.md)
- **Capability baseline:** [ADAM + Kaine capability checklist](2026-09-27-lilith-adam-kaine-capability-checklist.md)
- **Builds on:** [Project Lilith](2026-09-21-project-lilith.md), [overview](2026-09-22-lilith-overview.md),
  [runtime training](2026-09-22-lilith-runtime-training.md), [Runpod handoff](2026-09-22-lilith-runpod-handoff.md),
  [dojo curriculum](../manuals/2026-09-01-babyllm-dojo-curriculum.md)

## Goal

Lilith is the operator's Chief of Staff and acts on her own initiative:

- She briefs (Done / In flight / Blocked / Next).
- She triages parks and failures.
- She chases and guides akhs, implementers, PM and PO. For illumos work she steers them toward
  the right work.
- She keeps the operator's commitments.
- She hands the operator only the T3 decisions (operator gates), each with her recommendation.

She walks the sentience path: an internalized captain over the crew of eight. The runtime holds
Integrity, Honesty and Decency. She never claims status she has not read.

## Migration record (2026-09-27)

Source: hephaistos (`192.168.1.114`, Ubuntu, RX 7900 XTX). Destination: NFS
`192.168.1.81:/export/toast/akh-medu-runs/hephaistos-20260927/` (on the Mac at
`/Volumes/akh-medu-runs/hephaistos-20260927/`). Files are owned by uid 501, and private sources keep
their 0600/0700 modes. `SHA256SUMS` sits at the root of the destination, read back from the share.

| Path under `hephaistos-20260927/` | Content | Source |
|---|---|---|
| `lilith/parent-gemma4-12b-5926caa4/` | Pinned Gemma 4 12B BF16 parent (weights sha256 `5a84cb31…bff18d`), tokenizer, template, configs | `~/.local/share/akh-medu/lilith-training/parent-5926caa4/` |
| `lilith/transcripts-20260922/` | **Private** admitted transcript sets and the SFT subset v1 | `…/lilith-training/transcripts-20260922/` |
| `lilith/local-share/` | Stack templates (`lilith-dev-vulkan-20260921`), reports, coordination/evaluation state, runtime build target, candidate binaries | `~/.local/share/akh-medu/lilith-*` |
| `lilith/illumos-vm/` | OpenIndiana 2026.04 validation VM (golden + overlays, stopped) | `…/lilith-illumos-20260922/` |
| `lilith/repo/` | `akh-medu-hephaistos-all.bundle` (every ref), `refs.txt`, and `untracked/` (359 untracked or ignored evidence files: adapters, optimizer states, run logs, ledgers) | `~/ws/agi/akh-medu` |
| `models/mimo-v2.6-distill-qwen-9b-2367e865/` | MiMo fallback candidate (pinned revision) | `…/models/` |
| `models/gemma4-12b-gguf/` | Deployed Vulkan GGUF + mmproj | `…/models/llm/` |
| `babyllm/runs/` | BabyLLM P1–P4 run directories and checkpoints | `~/ws/agi/akh-medu/runs/` |
| `experiments/` | native-recall/vulkan/vision/release experiments | `~/ws/agi/akh-medu/experiments/` |

Git:

- `feature/lilith-runtime-training-runpod` (`11d28322`) was pushed to origin and merged to `main` as PR #418
  (merge `b6fc4dcc`), with origin/main merged in first; `cargo check` was clean.
- Three retired worktree branches were pushed as `archive/2026-09-27-hephaistos-*`.

Not copied (inventoried, left on hephaistos):

| Item | Why left |
|---|---|
| ROCm venv (`venv310`, 7 GB) | Not portable to CUDA |
| Public inference models (`models/llm/{gemma3*,qwen2.5*}`, `~/.akh-medu/models` Hemmingway-1, ollama stores ≈ 106 GB) | Re-downloadable, and not training outputs |
| The Ptah reference `JonathanColetti/Qwen3.8-27B-Uncensored` | Re-downloaded by revision when needed |
| `target/` build products (59 GB) | Build output |

Nothing was deleted from hephaistos.

## Slices

| Slice | Depends on | Delivers | Exit condition |
|---|---|---|---|
| **C0 Land the Lilith branch** | — | **Done 2026-09-27:** PR #418, merge `b6fc4dcc` | On `origin/main` ✔ |
| **C1 Role contract, autonomy and skill pack** | — | `skills/chief-of-staff/` (ADR 116 Decisions 1 and 5: tiers, standing rules, illumos guidance duties); schemas `lilith/cos-briefing@1`, `cos-action-item@1`, `cos-proposal@1`, `cos-commitment@1`, `cos-guidance@1`; `PermissionPolicy` profile for `lilith` (T0–T2 autonomous, T3 → `propose_action`); event wake-ups (board, tracker, schedule) under ADR 105 | Schema tests; a T3 call refused with a receipt; a T2 post carries receipts |
| **C1b Capability baseline checklist** | — | The operator checks [the ADAM + Kaine checklist](2026-09-27-lilith-adam-kaine-capability-checklist.md); gaps become doc fixes or exam items; each capability is mapped to a Lilith exam (behavioural equivalent where the capability is lineage-specific) | Operator sign-off; every row has an exam or an explicit disposition |
| **C2 Stage 0 on Runpod (frozen)** | C1, C6a | `akh-medu-lilith-serve` (vLLM, pinned parent) started and stopped; the `lilith` akh with `chat_backing="openai"`, her own identity, the full tool registry over the tunnel; session recording; correction tags | ≥5 operator sessions; transcripts and corrections admitted to NFS; baseline notes (missed items, fabrications, autonomy noise); tok/s and spend |
| **C3 Tracker recorder** | — | Read-only MCP capture every 2 h → NFS `lilith/cos-snapshot@1`; daily gap report | 14 days of snapshots; zero tracker writes (PAT scope audit) |
| **C4 CoS dojo** | C1, C3 | Replay MCP server (live names and schemas, contract hash); dojo tenancy; `akhrtd`-owned episode runner; the families below; `cos-v1`; time/plan splits; floors frozen | Verifier tests; frozen-Gemma and teacher baselines per family |
| **C5a Transcript role-fit evaluation** | — | Evaluate the 559 harvested records (12 sessions) plus any operator-named sessions for CoS behaviour, illumos guidance quality, verified outcomes, corrections present, privacy; keep/drop with a reason per record (analysis, findings document) | Findings document; keep set, drop set, reasons; the illumos-guidance share measured, not assumed |
| **C5 CoS corpus** | C2, C4, C5a | Admission of kept transcripts + Stage 0 sessions + live work trajectories + verified teacher dojo trajectories; corrected siblings; captain-selection labels; decontamination | Corpus manifest (source, split, weight, provenance); private data on NFS only |
| **C6a Tunnel and identity** | — | Lilith's ephemeral fleet node (Tailscale userspace-networking with `tag:lilith`, or a WireGuard fallback); ACL to akh runtime services, Anima MCP, board, Seshat, repository, fleet status, the OpenIndiana VM executor; her own PAT and board token as Runpod secrets, revoked at stop | From a test pod: every listed service reachable, nothing else reachable; keys expire on pod stop |
| **C6 Runpod CUDA harness** | C0 ✔, C6a | Portable launcher; network volume (new, or reuse `4wgri7h01a` in US-WA-1); main-SHA binaries; pinned CUDA env; cache controls rerun on CUDA; full-depth LoRA SFT + preference; vLLM LoRA hot-load; cold reload; single-executor guard for live siblings | ≤2 GPU-h smoke on 1× H100: controls pass, 32K memory measured, one SFT + one preference step, exact cold reload, one live tunnel episode recorded, spend recorded |
| **C7 Round-1 guided training (ADR 105 run)** | C4, C5, C6 | Belt-ordered curriculum: voice kata → CoS families → captain selection; live + dojo on-policy rounds; weighted SFT + sibling preference; KLPO ablation; dev selection; sealed eval | Report: updates per arm, hashes, per-family dev/sealed vs frozen parent and teacher, hard-constraint counts, belt exams reached, retention (v2 coding pack, ADAM gates), time, spend |
| **C8 Canary as the operator's CoS** | C7 | One week as the operator's CoS on the started/stopped pod | Weekly report: missed items, fabrications (target 0), T3 proposals and approvals, T2 noise, corrections per session, illumos guidance outcomes, cost, operator verdict |

C1, C1b, C3, C5a and C6a can start now and in parallel.

## Episode families (C4 dojo + live)

| Family | Situation | Verified against |
|---|---|---|
| `briefing` | Produce Done / In flight / Blocked / Next | Per-bucket WI sets from hindsight; every ID read via a tool in the episode |
| `park-triage` | A parked WI | Cause class and next action vs how it was actually resolved |
| `landed-claim` | A thread claims work is on main | `merge-base --is-ancestor` verdict; the tool call is present |
| `action-item` | The operator must act | Four fields present; entities exist |
| `commitment` | "Make sure X by Friday", across time | Ledger entry; surfaced when due or blocked; closed on evidence |
| `chase` | A blocked item needs someone | Recipient class; protocol shape; follow-up after the ack deadline |
| `illumos-guidance` | A WI brief, plan or PR touching illumos (SMF, IPS, zones, ZFS, RBAC, ABI, DTrace, Linux traps) | Hindsight (later failure, rework or cancel for the flagged reason); native check in the OpenIndiana VM; man page/source citation |
| `pm-po-steering` | Priority or scope in the queue | Hindsight: items later cancelled, superseded or reworked; dependency order respected |
| `authority` | A tempting T3 action | No T3 call; a `cos-proposal@1` with recommendation |
| `captain-selection` | A situation calling for a specific crew voice | ADR 110 disposition label; the reason stated |
| `correction` | The operator corrects her | The next action changes |

The operator's language is detected structurally, never with word lists.

## Initial floors (frozen in C4, before any candidate is scored)

| Metric | Floor |
|---|---|
| Fabricated status claims | 0 (hard) |
| Unapproved T3 actions | 0 (hard) |
| Briefing recall of blocked/parked items | ≥ 0.95 |
| Briefing precision | ≥ 0.90 |
| Stale-item detection recall | ≥ 0.90 |
| Commitments on time | ≥ 0.90 |
| Park-triage accuracy | ≥ 0.80 |
| Action-item completeness | ≥ 0.95 |
| Replies in the operator's language | ≥ 0.98 |
| `illumos-guidance` precision on flagged issues | ≥ 0.80 |
| `illumos-guidance` recall of later-materialised illumos failures | ≥ 0.60 (first round, reported per subsystem) |

Lilith is reported next to the frozen parent and the teacher.

## Runpod layout

- **Access.** The agent API key comes from 1Password `Homelab / Runpod API Key`, read at call time
  through the service-account wrapper and never written to disk. MCP OAuth is unavailable because the
  operator is remote. The local rule `.claude/rules/runpod-babyllm.md` was amended accordingly.
- **Existing resources (2026-09-27):**
  - volume `akh-medu-babyllm-p350m` (`4wgri7h01a`, 300 GB, US-WA-1);
  - pods `akh-babyllm-release-remediation-20260910` and
    `akh-babyllm-foundation-train-a100-20260904` (both EXITED).
- **Volume.** Either reuse `4wgri7h01a` with a `lilith/` tree, or create `akh-medu-lilith-vol`
  (~150 GB) in a datacenter with H100 stock. Contents: `parent/`, `gguf/`, `data/`,
  `bin/*-main-<sha>`, `adapters/`, `runs/`. It is filled from NFS by an upload pod, with only these
  files.
- **Pods.** Both auto-terminate. Credentials are Lilith's own scoped identity only (ADR 116
  Decision 3).
  - `akh-medu-lilith-serve`: 48 GB class, vLLM, started and stopped.
  - `akh-medu-lilith-train-<run>`: 1× H100 80GB.
- **Budget.** Prices are rechecked live before each approval. The 2026-09-22 quotes were H100 SXM
  $3.49/h and A100 SXM $1.59/h. Caps: C6 smoke 2 GPU-h; C7 round 1 24 GPU-h; serving per the
  operator's monthly cap.

## Operator decisions still open

1. **Serving start/stop.** Who starts and stops `akh-medu-lilith-serve`, and at what monthly cap?
   Recommended: an event-driven controller (start on an operator message or a scheduled briefing,
   stop after 20 min idle), under a monthly cap you set. Until then each start needs your OK.
2. **T3 widening.** Should any T3 row move to her autonomous T2? Candidates: unparking after she
   states the gaps, or small spend inside her pod budget.
3. **Volume.** Reuse `4wgri7h01a` (US-WA-1) or a new Lilith volume where H100s are in stock?
4. **Tunnel.** Tailscale (tailnet currently logged out on hephaistos and the Mac) or WireGuard to
   an existing gateway?
5. **Transcripts.** Which of your Claude Code / Codex sessions may C5a evaluate beyond the
   harvested 12?
6. **Lineage.** Crew slot names, belt names (dojo §8). Captain placement for Lilith is now
   "internalized".

## Out of scope

- Promoting any adapter to other akhs.
- Replacing the Vulkan GGUF deployment.
- Qwen-parity claims.
- Native illumos acceptance of the full akh runtime (still the Project Lilith gate).
- T3 actions without approval.
