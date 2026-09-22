# Mixed-backend transition failure

The second model-backed coordination episode completed Gemma/Vulkan delegation
and patch generation, native guest tests, and Ptah/HIP review. The final return
to Gemma/Vulkan loaded the model but did not complete the human report.

`kernel.log` records a gfx ring timeout in akhrtd PID 60310, MES queue-unmap
failures, MODE1 reset, VRAM loss and successful GPU recovery. The application log
is preserved in `../model-run2/akhrtd.log`. This is a GPU fault, not evidence that
the patch or guest tests failed. All owned daemons were stopped; partial work and
the pending report remain in the durable ledger.

Ranked hypotheses, stated before the recovery test:

1. Backend/context lifecycle state surviving a same-process Vulkan→HIP→Vulkan
   transition. Prediction: the same saved report succeeds in a fresh Vulkan-only
   process. HIP unload explicitly synchronizes/releases cached allocations;
   Vulkan relies on normal object destruction. This is a seam to investigate,
   not a proven root cause.
2. Stale inference state or Vulkan command batching affected by the transition.
   Prediction: resetting process/model state removes the failure, while a future
   isolated lifecycle reproducer can distinguish state from queue teardown.
3. A Vulkan fault specific to the report input, or general driver/resource
   instability. Prediction: a fresh Vulkan process with the same saved input may
   also fail. Successful recovery would reduce, not eliminate, this possibility.

An earlier HIP→Vulkan→HIP same-process canary had passed; there is no basis to
claim all mixed-backend switches fail. Repeated deliberate driver hangs were not
used as a stress test on this shared host. The captured failed episode is the
red integration signal; fresh-process replay is the bounded differential test.
Model loading makes this loop minutes long rather than a fast unit test.

The mitigation being validated is to keep each backend residency in a separate
runtime process, with completed-work evidence and session/assignment state held
outside that process. This does not claim a driver or general lifecycle fix.
Recovery and subsequent trial receipts are recorded alongside the model runs.

## Recovery queue diagnosis

The original report becomes `unknown` after runtime restart. The session service
intentionally treats that state as nonterminal: `SessionService::next_job` selects
the first nonterminal turn and refuses dispatch when it is unknown. New report
inputs in that same session therefore remain queued. This is a safety fence, not
a new GPU fault. Queued retry inputs can be cancelled through the API; the
original unknown result must remain visible until reconciled.

A tool-free report in a new recovery session can safely consume the already
completed, hash-guarded patch/test/reviewer evidence without repeating those
operations. Such a recovery must be labelled as a separate-session handoff, not
as completion of the original unknown turn. Future process handoffs should occur
only between confirmed completed turns, avoiding this crash-recovery boundary.

## Fresh-process differential probes

`../model-run2/fresh-report-probe/` replays the exact saved report payload in a
new tool-free session and a Vulkan-only runtime. It reaches model load but
exceeds the 180-second turn deadline. `fresh-report-no-encoder/` repeats that
payload with encoder and pass batching disabled; it also exceeds the deadline.
No additional GPU reset was observed in these probes. The separate
`fresh-report-no-encoder-entry-failure/` is a driver-script preflight error, not a
model observation. Each actual probe stopped its owned services.

These results do not support process isolation alone as sufficient mitigation.
The approximately 1,414-token rendered report prompt is a performance/failure
case still to investigate; a timeout does not by itself identify a GPU hang.
Gemma currently forwards the complete prefill at once. There is no demonstrated
causal fix to that path in this run.

The next bounded workflow uses a versioned report capsule and a fresh tool-free
report session: complete objective and reviewer text, compact native-test facts,
and references to the full guarded patch/test evidence. Raw patch content and
opaque hashes remain in audit artifacts rather than the report prompt. This is
an explicit workflow contract change, not a same-input recovery or benchmark
rescore. The original unknown turn remains unresolved and visible.

The `fresh-report-capsule/` attempt completed generation up to its 192-token
limit and returned `model_incomplete`; it did not time out. The same 513-token
user capsule with the normal 768 output allowance completed in
`fresh-report-capsule-768/`. Manual comparison confirms that its objective,
guest, command, pass/exit facts and explicitly labelled reviewer opinion agree
with the saved evidence. All owned services stopped. The original unknown turn
is still unresolved; this is a separately completed report session.

The capsule protocol's generic `scope` wording was inherited from the exact-input
probe: its `capsule: true` field and frozen `prompt.txt` define the actual input,
which is intentionally different from the old report payload. The full guarded
evidence is in `evidence.json`; no failed experiment or quality score was replaced.
The `fresh-report-capsule-768-entry-failure/` directory records an earlier port
availability check failure before any service started; the check now allows
normal socket reuse after a prior owned server stops.

The next full episode (`../model-run3/`) uses a different capsule@2 prompt:
735 user tokens, 2,048 context budget, 768 output budget, and the launcher's
600-second runtime deadline. Both backend boundaries completed only after all
owned turns were terminal. The reporter was still running more than four minutes
after loading. A kernel check during that wait found no new GPU reset; it did
show an amdgpu SVM deferred-work CPU-hog warning. That observation does not identify
the cause. These settings differ from the successful 513-token/4,096-context
isolated probe, so prompt size alone has not been causally isolated.

## Final bounded outcome

Run 3 completed all four model turns and both idle process boundaries. Its final
735-token report completed in 452.148 seconds, within the frozen 600-second
runtime limit. The final report's objective, explicit reviewer attribution and
SSH guest pass/exit facts match the saved evidence. The original fixture is
unchanged and all owned services stopped. See `../model-run3/truth-review.json`
and `../model-run3/candidate.patch`. No fourth model trial was necessary.

This establishes the bounded workflow's completion. It does not establish an
acceptable interactive reporting latency or diagnose/fix the earlier GPU reset.
The earlier 180-second exact-payload timeouts remain timeouts, not proof that
those inputs could never finish under a longer deadline. Distinguishing slow
prefill, generation cost, and backend/driver state remains follow-up work.
