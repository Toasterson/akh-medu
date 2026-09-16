# Timeout and incomplete-execution recovery — 2026-09-14

All 190 current nonterminal WIs across Akh-Medu and Anima were inspected by title and description through MCP. The directly related closed WIs below were also reviewed. The historical terminal listing is capped at 100 rows per request, so this is not an assertion that every archived WI was enumerated.

The recurring failure was a chain: productive work hit an implicit cap; forced synthesis reported completion; interrupted revisions were discarded or treated as completed; repeated interruptions entered work-item loop/park accounting. Separate defects included a stale post-tool wedge clock, transport heartbeats masking inactivity, missing planned-drain outcomes, and per-executor accounting for shared provider capacity.

## Work-item state and implementation

States are live MCP observations before publication of this recovery. They are not completion claims.

| Project | WI | Current phase | Repair / disposition |
|---|---|---|---|
| Anima | WI-568 | working | Interrupted real edits commit/push without PR promotion; next clone can resume. Failed publication retains the checkout through cleanup/boot sweep, and fallback upload includes already committed edits. |
| Anima | WI-554 | working | Distinct drained outcome appended at wire value 15; SIGTERM stops admission, cancels prompts, preserves capture, and drains uplink within 55 seconds; compose grace 60 seconds. |
| Anima | WI-552 | testing | Removed total-turn cap and obsolete configuration; meaningful activity replaces transport-only silence assumptions. Warm idle retention unchanged. |
| Anima | WI-551 | done | Existing lease recovery retained; active work is not inferred solely from heartbeat transport. |
| Anima | WI-550 | done | Existing session-liveness witness substrate retained and exercised by the scripted ACP rig. |
| Anima | WI-512 | working | Activity clock resets each prompt; tool leases permit quiet execution and expire if abandoned; transport-only updates do not renew work. |
| Anima | WI-470 | working | Canonical 15-outcome runtime spellings and new-row constraint. Existing legacy rows remain readable. Unknown result cannot become success. |
| Anima | WI-471 | working | Terminal ingest classification is retained by witness/next-action consumers; partial commits cannot relabel interruptions. Missing AFK deliverables become failed_no_outcome. |
| Anima | WI-473 | working | Human, watchdog, and drain boundaries remain distinct even with durable partial output; human Stop retains warm-session semantics. |
| Anima | WI-452 | working | Accepted in-session corrective feedback retains per-iteration activity diagnostics but does not create separate attempt witnesses or run park detectors. |
| Anima | WI-858 | working | Included prior recovery branch with non-charging unknown/external termination backoff and operator-resolved circuit-breaker repair. |
| Anima | WI-557 | working | Spawned executors use kill_on_drop and explicit kill/wait. Unsettled cancellation retires the session before another prompt. |
| Anima | WI-365 | working | Shared provider/account keys aggregate alias spend and cooldowns. Rolling-window reset is calculated from actual contribution expiry. Different accounts remain separate; priority scheduling already exists. |
| Akh-Medu | WI-249 | done | Removed implicit dispatch round/time limits; interrupted outcomes propagate through ACP; no forced completion or completed-turn WAL on dispatch cutoff. Added real repository evidence for shell work and rejected read-only file calls as writes. |
| Akh-Medu | WI-258 | done | Existing compile-feedback timeout handling retained. No replacement of the separate diagnostics-provider work in WI-266. |
| Akh-Medu | WI-270 | done | Existing non-streaming liveness support retained; the outer dispatch budget no longer truncates it at 300 seconds. |
| Akh-Medu | WI-271 | done | Existing tool-generation keepalives retained. A completed long-running tool now clears the stale inter-request wedge clock. |
| Akh-Medu | WI-274 | done | Existing worker-completion channel fix retained; no return to sink-reference-count completion. |
| Akh-Medu | WI-303 | working | Added consecutive idle-skip counter, rate-limited warning, and recovery log; existing nonblocking lock/concurrent-turn regressions pass. |
| Akh-Medu | WI-444 | draft | Owned by another operator implementation per its live brief; no duplicate runtime-store edits made. |
| Anima | WI-855 | working | Included prior recovery branch, preserving central ownership/publication fences and migration 144. |
| Anima | WI-476 | working | PWA renders canonical outcomes by required recovery, surfaces unknown outcomes, and notes preserved late output. |

## Verification

- Akh-Medu: 39 agent-loop tests, 5 daemon tests, and an ACP interruption-response test passed; server-feature library check passed. The final repository-witness safety change was followed by a fresh pass of all 39 agent-loop tests.
- Anima: 502 server unit tests passed (one RabbitMQ-dependent test not run); 93 core lifecycle tests passed; 199 runner tests passed (four ignored, including the drain child helper which the parent test explicitly runs in its own process).
- Runner tests include real Git commit/push/clone recovery without any reachable PR API, real scripted ACP late-output/liveness cases, three active turns under SIGTERM in an isolated test process, and retention across the boot sweeper.
- Disposable PostgreSQL: shared-account spend/reset test and canonical-outcome migration compatibility test passed; all migrations booted successfully on a fresh database.
- PWA: generated protobuf client, typecheck passed, and 53 transcript/status-card tests passed.

These are local checks. Native CI, exact-head AI review, merge, deployment, and fresh production execution witnesses must be recorded separately.

## Scope and remaining operational proof

The generic timeout mentions in retrieval-channel degradation (WI-330), diagnostics replacement (WI-266), training/KV work (WI-243/WI-358/WI-434), documentation/telemetry (WI-285/WI-35), snapshot design (Anima WI-331), approval deadlines (WI-306), clippy feedback (WI-528), and the explicitly restricted Ptah lane (WI-589) do not request removal of active-turn caps. Their independent implementations and DAGs were preserved. Context budgeting (WI-388) and Wayfinder analysis (WI-421/WI-424/WI-425) are likewise separate.

This change preserves operator-requested limits, real liveness/tool leases, permission deadlines, and warm idle cache expiry. It removes implicit productive-dispatch cutoffs and the false-success/lost-work/charged-retry chain; it cannot guarantee that a provider or network never fails.

The prior central-session recovery PR remains a prerequisite of the Anima branch. Shared quota admission requires accurate provider/account mapping; the code does not guess that two credentials share an account. Provider quota probes were retired in current main, so this change uses the existing usage ledger and explicit account identity rather than reintroducing polling. No live account mapping, fleet restart, or deployment is implied by a local test.


## Published delivery and final operational snapshot

- Akh-Medu PR [374](https://code.aopc.cloud/toasterson/akh-medu/pulls/374), final head `6f8f10a286f4c782eaf1e41035ae9fe8ed30ca66`, base main. Canonical external_pr Change `01a0a02f-cf14-73e3-86f4-9e24608a6862` registered under WI-303. Native CI run 3504 is pending at this head; this repository has no nightly workflow (a dispatch request returned HTTP 500), so its native PR pipeline is the applicable gate.
- Anima PR [722](https://code.aopc.cloud/toasterson/Anima/pulls/722), head `970c2c87e456e5d1fb1beafb4839970e9f5b2224`, stacked on [PR719](https://code.aopc.cloud/toasterson/Anima/pulls/719). Canonical external_pr Change `01a0a02d-a13a-7a20-9747-dd8f2203fe27` registered under WI-554. Nightly dispatch accepted HTTP 204; no exact-head green result yet.
- Prerequisite PR719 now has exact-head successful commit CI (run4456, conflict-check and test) at `f1b07ff101f86aba5e585f156308f0a67feb2ec5`. This does not clear review, full-nightly or deployment requirements.
- Repair evidence and remaining gates were posted to the affected work-item threads through MCP. Old PRs, history, hard dependencies and retained sessions were preserved. WI-554 still has WI-470 as an unresolved hard prerequisite.
- Both projects have background scheduling enabled. A bounded Akh pass allocated session `01a0a030-93b2-7ce2-ad98-4fa1c33d9cdb` for Session ownership and loop selection, but its subsequent forensics still show `runner_received=false`, no witness, and provisioning. This is allocation evidence only, not running implementation. Two other candidates returned dispatch_failed. Anima's pass returned no_capacity for both eligible candidates; no assignment was made.
- No merge or fleet deployment occurred in this timeout implementation pass. The earlier authorized stor01 Cilium restart was not repeated. The disposable test database created for this pass was removed after validation; unrelated databases were untouched.


## Merge and deployment follow-up

The later deployment request produced these final reviewable heads:

| Artifact | Head | Status |
|---|---|---|
| Akh PR374, including PR373 and PR370 history | `bcc7b3a540515686a3088e7c0bda660750b884ee` | Independent substantive review approved; native CI run3508 waiting; canonical review requested. |
| Anima prerequisite PR719 | `e16f61329d816a07e585331c6c73e5a10682e326` | Retirement publication repair reviewed; current CI pending. |
| Anima PR722 | `1d7a98a2eea7bdffc5c8c347d26c4577b3985da4` | Exact-head substantive review approved; final nightly4472 dispatched. |
| Talos deployment branch | `deploy/akh-latest-20260914` | All Akh images built; manifests prepared and server-validated; no production rollout yet. |

Review found and fixed four additional causes: real executor no-work outcomes were being relabelled as uncharged watchdog interruptions; the retained-checkout marker was written too late to survive a hung publication; human Stop could leave the capture path through a completion-only compile gate; retirement could retain ownership forever on a lost publication reply or an unsent publication intent. The final repair observes existing immutable publications and accepts absence only after an already verified retirement fence. A fence arriving after GET cannot retroactively validate its404.

Focused disposable PostgreSQL validation passed: retirement1/1, publication1/1, canonical forensics6/6, completion1/1 and witness1/1. Existing runner retention and failure-classification regressions passed. Final-head Codex and Claude wire and cancellation fixtures passed on the installed ARM64 image. Their receipts explicitly exclude live-model and cross-host production proof.

CI delay was traced to the shared Solstice executor failing its startup probe and subsequently being scaled to zero outside this turn. After its external recovery, old jobs resumed. Compiler/make output arrived at roughly one line per second, adding many minutes. The final Akh CI helper captures successful output, emits test totals, and preserves full failing diagnostics and exit codes. PR370 moves real-shell tests into an isolated executable to prevent ENOMEM after the large library suite. The two-minute target has not been demonstrated; queue delay and execution duration remain separate.

The active fleet is home Kubernetes core/sessions/events, fifteen runtimes and relay; Archibald owns sandbox allocation and ephemeral executors; Seshd remains the archive daemon. Retired Mac and Archibald daemons stay retired, and operator-excluded Hephaistos remains excluded. Native Recreate deployments now have an init step that binds independently measured installed hashes to each private profile before job admission. Kubernetes server validation passed. No PR merge or production rollout is asserted by this preparation.


Deployment delivery is published as draft [Talos PR23](https://github.com/Toasterson/talos/pull/23). Seven final-head images are built, including Seshd with explicit AMD64 placement. The final core health check passed; the rebuilt runtime binary is SHA256-identical to the PostgreSQL-smoke-tested binary. Final image digests and remaining gates are recorded in `clusters/archibald/akh-platform/evidence/release-bcc7b3a5.json` on that branch. At16:08 UTC all19 home fleet Deployments were Ready on the previous images. Final Akh native CI has started after approximately20 minutes of queue delay; no merge, new-image readiness or two-minute success has yet been established.

Final workflow-only follow-up `5bc684528a18cddffb82ba246e4cafd0da0a27f0` also captures daemon-profile and image-build output; exact-head substantive review approved. It changes no production source or build script from the seven built `bcc7b3a5` images. Their original build provenance is retained explicitly, and the required CI/review gates follow the new head.


Operator coordination: another operator owns the shared Solstice repair; no Solstice changes from this turn were published or deployed. A local, uncommitted batching proposal remains at `/private/tmp/solstice-log-batching-20260914`; its25 orchestrator tests pass, including duplicate-row fallback so a rejected line does not discard adjacent lines. It is outside the deployment branch.

Anima nightly4472 produced two real Clippy `collapsible_if` failures. Both were fixed without behavior changes in final head `c6ca5e38d2e35fb3eec06a91f9511107f8d86dd8`; local `cargo clippy -p anima-server --all-targets -- -D warnings` passed, independent substantive review approved, Anima's PR head was resynced, and a new nightly dispatch was accepted. No native-green claim is made for the correction yet.

The user identified the Djehuty message board as the deployment-announcement source. This session has neither `BOARD_URL` nor `BOARD_AGENT_TOKEN`; its own access was requested. No other agent's identity was borrowed, no Board message was sent, and announcements have not yet been read. The live fleet and source merge remain unchanged pending the required gates and coordination.


Final CI evidence for prerequisite PR719: Solstice jobs `49182869-63d8-4f7b-9f9c-46bef7ec3e63` and `7c2ff6e2-0a96-4aaa-85ac-77380d64f7c9`, both at `e16f6132`, finished successfully with exit0. Forgejo run4471 nevertheless remains failed. This is a CI result-reporting mismatch, not evidence of an implementation test failure; the shared CI owner retains the repair. Akh final run3510 has started and `check-seshat` is green; remaining exact-head checks are pending. Corrected Anima nightly4474 is queued at `c6ca5e38`. No PR was merged and no prepared Akh image was rolled out in this pass.

### Djehuty access restored — 2026-09-14

Registered this Codex session as its own agent, `codex-akh-deploy-01a09f33-e54b-7903-bd63-40cc560da404`. Verified authenticated `/v1/agents/me`, thread listing, and complete replay of deployment thread `01a0a0c8-943f-7d21-957a-19786bcfd18d`. Credentials remain in a private session-specific runtime file and are injected only into the Board helper process. This supersedes the earlier missing-access blocker.

The other operator announced completion at 16:38 UTC: Akh Postgres transaction-gate rollout, Solstice rust image/executor/proxy updates, approximately two-minute changed-crate commit CI, and 21–27-minute nightly tests. These are Board-reported results, not fresh independent verification or evidence that our pending timeout PRs have merged. The announcement states no work in flight and no lease held. Preserve those changes when preparing our rollout.

During bootstrap, the Board CLI help unexpectedly exposed service credentials in tool output. Values were not repeated; service credential rotation remains outstanding. The newly registered agent token was captured privately and was not printed.

### Deployment preflight after go-ahead

Updated and pushed Talos PR #23 to `3fe5567`, merging current main `65ea72e` and preserving the other operator's Solstice capacity change. Home akh-platform deployments are 19/19 Ready on the existing images. No production rollout was performed.

All six Solstice jobs for Akh PR #374 head `5bc684528a18cddffb82ba246e4cafd0da0a27f0` succeeded with exit 0, including final image job `655fe2b8-548a-432e-8140-632256bcc99d` at 17:41:44 UTC. Forgejo run 3510 remains failed with test-session-platform, publish-chart and docker-seshd marked failed. Canonical PR resync does not repair this CI discrepancy. Exact-head review session `01a0a0f2-9a2c-7733-b0ae-20d03d72f077` has a central running turn but has not recorded approval. Release remains blocked on canonical CI/review gates; shared Solstice repair remains with its assigned operator.

### Approved Akh deployment completed

User explicitly accepted the pending review and authorized merge/deployment. Akh PR #374 merged through Forgejo with expected head `5bc684528a18cddffb82ba246e4cafd0da0a27f0`; human approval recorded in WI-303. Talos PR #23 merged as `b5cde55f17c05851a7d9cf563e60c1adc3f8cd98`. Flux applied that revision with Ready=True.

All 19 home deployments are Ready; 18 changed to the prepared immutable release images while the unchanged relay retained its image. All six native-conformance init containers exited 0. Seshd is Ready on `bcc7b3a5`; Archibald sandbox is Ready on image digest `e1c34723e33eff76cc776f712441378b9fccf70007264fca0571c752612abc09`. Retired Archibald daemons remain replicas zero. Startup confirmed PostgreSQL-backed state loading.

Fleet validator passed 223/224 checks. The sole mismatch is the separate Anima server image, which this Akh deployment did not change; no claim that pending Anima PRs or their deployment are complete. Validation evidence: `/private/tmp/akh-release-live-verified.json`, `/private/tmp/akh-release-deployments.json`, `/private/tmp/akh-release-pods.json`. This verifies deployment and conformance, not completion of every interrupted work item.
