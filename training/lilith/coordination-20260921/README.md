# Lilith first durable coordination slice — 2026-09-21

This is a local integration fixture, not a coding benchmark, Qwen parity result,
or native illumos validation. See [the operating guide](../../../examples/lilith/COORDINATION.md).

## End-to-end evidence

[run1/result.json](run1/result.json) passed with real Gemma Vulkan authoring,
a model-requested `request_review` client tool, independent host verification
(2,051 integer cases), a durable Ptah child session, and a final Lilith report.
The controller was recreated from its ledger after the authoring turn. The
single runtime PID remained 250294 across Lilith → Ptah → Lilith model changes.
All four fixture-owned daemons were stopped afterwards.

Artifact SHA-256:
`bd3e50a7d01f5a3b1ca4792a1162a4e311a4578a05962a28495d7796572f84f6`.

This first run used `ptah-validator` with read_file. Ptah tried to read a file
that was deliberately supplied only as an inline snapshot. Its logic review
was useful, but it added a missing-file caveat and questioned the test evidence.
Lilith accurately reported those caveats separately from the host test result.
The initial run did not use the subsequent explicit receipt envelope or the
new tool-free reviewer identity; its evidence is preserved unchanged.

Runtime turn durations from the service log were approximately 167.4 seconds
for authoring, 115.0 seconds for review, and 262.0 seconds for the final report.
Cold model/context processing remains a substantial usability limit; previous
warm decode TPS improvements do not imply equivalent whole-task latency.

## Handoff diagnosis

[handoff-check/result.json](handoff-check/result.json) records a failed focused
real-model probe: even an explicit inline-source/host-receipt prompt did not
prevent a read_file attempt. Prompt wording alone was insufficient.

The final configuration adds `ptah-inline-reviewer`, sharing the existing Ptah
runtime profile but granting no tools. The existing read-only `ptah-validator`
remains available for file-based reviews. Both the generator and the local
prepared stack include this identity. The controller now selects it by default.

`probe_handoff.py handoff-check2` checks that final profile with the same source,
a fresh independent verification receipt, and the current handoff renderer.
[The focused result](handoff-check2/result.json) passed in 110.3 seconds with
zero tool calls. Ptah found no defect, recognized the matching host receipt and
limited its conclusion to the integer fixture. All owned services stopped. This
is distinct from the initial end-to-end run, not a claim that the entire final
workflow was rerun.

## Recovery and contract tests

[unit-tests.log](unit-tests.log): 27 passing scoped tests. Coverage includes
response loss after acknowledgement, child creation, review submission and
return submission; reopening the ledger without duplicate work; immutable
request replay across renderer changes; stale/missing/symlinked artifacts;
invalid scope and undefined Python names; cancelled author and failed/unknown review; exclusive writer
ownership and concurrent read-only status; and generated profile grants.

[immutable-intent-before.log](immutable-intent-before.log) records the failing
regression test before persisting complete outbound request bodies. The final
implementation stores each body before submission and replays that exact body.

The controller ledger is local SQLite project state, not Anima. It stores
snapshots, receipts, stable request IDs, child references and status. Neither
credentials nor the ledger are exposed to model workspaces. No VSA learning,
second-specialist routing, running-child cancellation, or arbitrary code
execution is implemented by this bounded fixture.
