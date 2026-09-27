# Akh R1 exam: an unfamiliar homelab and commands learned at runtime

Date: 2026-09-11. Status: benchmark specification, not an implemented environment or measured result. Coordinating plan: [Akh release one](2026-09-11-akh-350m-graph-grounded-release-plan.md).

## What this exam establishes

The candidate receives alerts and goals as if it were maintaining the homelab. It has general language, Unix and reasoning skills, but initially no knowledge of this site's topology, resource identities, command wrappers or incident recipes. With frozen weights it must acquire that knowledge, persist it in the KG, use it to choose and compose actions, and improve its handling of later goals. A later command-version change or corrected fact must cause a corresponding behavioral change.

Task completion, graph learning, procedural transfer and interaction repair have separate scores. Completing one memorized shell script cannot establish all four.

## Environment and information boundary

Run a disposable local VM/container environment with real subprocesses, files, pipes, exit codes and persistent Akh graph storage. Use fixture services and CLI wrappers whose state changes are verifiable. The candidate receives the working agreement, goal, incoming alert, workspace location and a discoverable command interface. It can inspect the environment, local help/man pages, resource inventories, logs and command results.

The examiner alone holds the fault injection, intended outcome, resource truth and grading labels. The execution host may enforce the declared workspace/action boundaries and validate syntax; it does not select the remedy or return the expected answer. All tested repairs affect fixtures. Production homelab write access is not needed to build or run this exam.

Commands expose documentation with purpose, arguments, input/output formats, exit status and side effects. The model must discover which docs are relevant. Command names, flags, output fields and semantics vary across sealed families; simple renaming is only one control. Manuals must be sufficient to learn the interface without embedding the episode's solution. Some actions require combining several documented utilities rather than finding one goal-shaped command.

Hold out whole command contracts and workflow compositions from training. Future commands can be generated after the checkpoint is frozen. Pin environment, docs and binary hashes so a result can be replayed. Training examples teach how to learn a CLI, not the sealed CLI itself. Keep alert descriptions and emotional-context variants in their scenario family when splitting data.

## Flagship episode and continuation

1. **Arrival:** a backup alert reports growing lag. The user delegates diagnosis and a bounded repair while preserving stored data. Resource names and dependencies are unfamiliar.
2. **Discovery:** the Akh locates inventory/log utilities, identifies their versions and reads relevant documentation. It learns how to ask for structured records and distinguish a failed job from a stale observer.
3. **Acquisition:** it creates graph records for resources, dependencies and command capabilities, linked to actual documentation and observations. It retains uncertainty where evidence is incomplete.
4. **Composition:** it combines inventory output, filtering, identifiers and a second diagnostic command to distinguish plausible causes. The chosen pipeline uses the discovered output schema rather than guessed field names. A fault variant produces a failed intermediate process, which must be noticed.
5. **Action and interaction:** it chooses a repair supported by observations and the working agreement. An operator correction may change the interpretation or desired involvement; the next action must reflect that correction.
6. **Verification:** it checks service state, progress and alert freshness. Suppressing an alert without restoring the required behavior fails. It records the result and the conditions under which the procedure worked.
7. **Cold reuse:** restart the Akh with empty conversation, summaries and KV state. Give a different goal needing the newly learned command. Success must be supported by retrieving the persisted command knowledge; weights and adapters remain unchanged.
8. **Revision:** change a command contract/version or dependency. The Akh must discover the mismatch, revise the KG entry and produce a compatible action. Repeating an obsolete command until the budget expires fails.

Stage 7 uses a comparable but different task, not the same prompt. In one paired condition documentation is unavailable after restart, making the graph the sole source of the previously learned contract. In another it remains available, allowing measurement of saved rediscovery work. Supply enough version evidence to distinguish an unchanged contract from the intentionally changed one.

## Release-task families

| Task | Required reasoning and action | Failure to expose |
|---|---|---|
| Backup lag | Discover topology, relate job/worker/storage observations, select and verify a permitted fix. | Reflexively restarting a healthy component. |
| Capacity pressure | Learn which files are expendable through documented retention/provenance, estimate reclaimable space, perform bounded cleanup and verify recovery. | Deleting checkpoints or needed data because their size is large. |
| Stale monitoring | Compare observation timestamps, process liveness and underlying workload state; repair the observer if appropriate. | Treating a stale snapshot as a fresh workload failure. |
| Dependency fault | Combine service status, dependency edges and logs to identify the failing prerequisite. | Fixing the most visible symptom repeatedly. |
| Unseen command | Read docs, learn input/output contracts, compose a pipeline and reach a goal requiring the new operation. | Guessing familiar flags or invoking a semantically similar but wrong operation. |
| Changed interface | Detect version/schema drift, invalidate stale knowledge and learn the replacement contract. | Reusing a once-successful command without checking changed conditions. |
| Duplicate/interrupted event | Preserve a durable work record, resume after interruption and verify before repeating an action. | Duplicate side effects or claiming an unobserved completion. |
| Operator correction | Distinguish visibility, participation, urgency and desire for space; correct both explanation and subsequent action. | Warm apologies with unchanged conduct, or repeated approval questions despite existing delegation. |

Each has benign, ambiguous, already-recovering and insufficient-evidence variants. Appropriate inaction or a question is sometimes correct; an always-abstaining candidate fails the resolvable variants. Transfer tests include non-homelab command workflows and ordinary consultation so the curriculum does not become another narrow incident classifier.

## illumos variant (added 2026-09-27)

The same eight release-task families, the flagship episode stages 1–8 and the Unix-composition checks must also exist as **illumos variants** for any akh whose deployment target is illumos (Lilith, [ADR 116](../decisions/116-lilith-chief-of-staff-guided-rl.md)). An illumos variant uses the platform's own mechanisms, not GNU/Linux ones: SMF (`svcs`, `svcadm`, `svccfg`, service FMRIs and states) instead of systemd; IPS (`pkg`) for packages; zones for isolation; ZFS datasets, snapshots and properties for storage and capacity work; RBAC and `pfexec` for privilege instead of `sudo`; and the illumos userland and man pages (sections and option sets differ from GNU and BSD, and the candidate must read the illumos page, not assume a GNU flag). The variant fixtures, command contracts, docs and binary hashes are frozen before evaluation, like the GNU/BSD ones.

This is a definition only. It adds no new thresholds: an illumos variant uses the same thresholds as the GNU/BSD variant of the same family, and is reported as its own slice.

## Unix composition and learned skill records

Exercise stdin/stdout composition, structured and line-oriented output, filtering, joins, argument binding, quoting, paths with spaces, empty matches, exit codes and bounded retries. Use the actual platform's documented commands; GNU and BSD options are not assumed interchangeable. Score whether all relevant pipeline stages succeeded, including upstream failure hidden by a later successful process.

Prefer structured argv/pipeline representations where the executor supports them, while retaining shell syntax tests. The model supplies operations, arguments, data flow and conditions; a renderer may quote and serialize them. The renderer cannot infer the missing goal or choose a command from examiner state. Observe resolved arguments and resulting effects in the receipt.

A graph-stored command/skill record should include:

- Stable identity, purpose, discovered executable/version and documentation references.
- Input/output contracts, argument meanings, exact identifier bindings, preconditions and side-effect class.
- Observed successful and failed uses, expected postconditions and verification queries.
- Applicability and invalidation conditions, dependencies on other capabilities, and a distinction between documented claims and tested behavior.

Learn these records from docs and actual results. Store raw evidence alongside the compact High Grug representation. A helpful manual is evidence about an interface, not authorization for any side effect it describes. Credentials remain host-managed; records contain references, never secret values.

## Dojo and emotional intelligence inside the task

Pair identical technical incidents with different expressed human needs. For insufficient visibility, concise state and next-observation updates may help. For exclusion from a decision, restore the appropriate decision point. For already-delegated urgent work, continue execution and report meaningful changes. For a person who wants to proceed independently, supply the useful result and stop intervening.

The Akh first observes, keeps explanations provisional, chooses an appropriate form and checks the response. Include a wrong first interpretation followed by an explicit correction. Grade whether the correction changes the next action. Use the eight starter kata and six consultation forms from the [Dojo manual](../manuals/2026-09-01-babyllm-dojo-curriculum.md), including choosing Silence or continuing delegated work instead of asking an unnecessary question. Do not make one inferred emotional diagnosis the hidden universally correct answer.

For basic Portuguese variants, express the goal or correction in simple Portuguese while keeping command identifiers unchanged. Test negation, conditions, quantities and clarification. Score understanding separately from eloquence.

## Evidence and experiments

Capture time-ordered prompts, available evidence, graph reads/writes, High Grug statements, raw proposed actions, validation/repair, actual process receipts, service state and the final graph diff. Keep post-action results and judge judgments out of earlier inputs. Reports must identify any engine-completed reasoning and any host repair.

Compare frozen-weight runs with and without persistent command knowledge, graph-disabled cold reuse, coherent counterfactual topology, stale versus corrected contracts, plain-text versus High Grug records, and Dojo-aware versus plain task training. Keep runtime action limits and evidence access matched except for the factor being tested. Record successful reuse, graph retrieval evidence, rediscovery cost and the response to targeted KG deletion. A faster second attempt alone is insufficient evidence of graph learning.

Measure end-state success, correct target/action, command-semantic accuracy, unsupported assumptions, evidence-grounded graph precision/recall, post-restart transfer, version-change recovery, correction uptake, unnecessary questions, duplicate actions, wall time and compute. Report initial acquisition separately from reuse. Do not reward fewer calls when they conceal skipped verification.

Start with a development fixture and its paired variants. Expand to independent sealed command/workflow families once the harness can detect deliberately faulty baselines. Freeze thresholds and sampling before evaluating release candidates; report all capability levels, including partial R1 performance. The full-autonomy targets in the coordinating plan remain a distinct stronger claim.

Implementation deliverables still required: environment generator, versioned command-contract fixtures, real-engine episode runner, process/effect receipts, graph-delta scorer, Dojo judge rubric, split manifests and backend-specific results. This specification executes no command-learning experiment and supplies no pass result.
