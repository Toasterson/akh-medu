# Lilith two-specialist coordination evidence — 2026-09-22

This slice adds a bounded repository patch/test flow beside the earlier artifact
review controller. A synthetic Lilith author delegates a one-file replacement to
the Khnum identity. The host applies that proposal only to a temporary repository
copy and executes an operator-fixed `python3 -m unittest -q` command. A passing
proposal is then sent to the Ptah reviewer identity and returned to Lilith with a
prompt that requires a truthful account of the evidence.

The executable integration covers successful two-specialist completion, a real
failing/passing subprocess test, a wrong-path proposal, a failed patch peer, an
operator correction that cancels and replaces an in-flight patch turn, and an
operator cancellation that leaves an unrelated parent turn untouched. The
original fixture repository remains byte-for-byte unchanged in the success case.

After safety review, production execution now refuses proposals without an
explicit executor. The local regression executor accepts only a single
undecorated expression-only Python function and applies CPU, memory, file-size,
descriptor, timeout and output bounds. The controller pins every repository file
before delegation, rejects symlinks and special files, and detects snapshot drift
before verification. Corrections during patching, review, or reporting cancel and
observe all owned turns, erase stale evidence, and bind every replay identity to
the correction revision.

## Receipts

The final CPU regression command was:

```sh
cd scripts/session-platform
python3 -m unittest test_lilith_coordination.py \
  test_lilith_repository_coordination.py test_lilith_runtime_boundary.py
```

All 50 tests passed. The initial 44-test receipt predates the fenced-proposal,
fresh-report-session, report-capsule, and backend-boundary regressions. These are
controller and local-process tests; they do not make model-quality claims.

The isolated executor is now configured and contract-pinned. It copies a pinned
regular-file repository snapshot into an unprivileged SSH guest, writes only the
fixed target, runs only the operator-selected argv, bounds time and captured
output, removes the unique guest workspace, and returns a durable receipt. The
final guest verifier ran one unittest method containing seven boundary
assertions. It passed with exit code zero, no timeout, and successful cleanup.

Three model episodes are retained rather than rewritten:

- `model-run1` failed before verification because the valid proposal was wrapped
  in one JSON fence. Its frozen controller source and failure remain evidence.
- `model-run2` accepted the versioned fenced form, passed the guest verifier, and
  completed Ptah review. Its original Lilith report turn remained unknown after
  a real GPU reset, so the run remains incomplete. Separate recovery attempts
  and the kernel diagnostic remain failures/diagnostics, not completion.
- `model-run3` completed the fresh workflow. Lilith delegated once, Khnum
  produced candidate SHA-256
  `c887cd10140c179a5abbf887d130ec77a4884421d1282e39d4225ea715b99c4f`,
  the SSH guest verifier passed, and Ptah reported no defect. The runtime changed
  Vulkan PID 70606 to HIP PID 71280 only after the author and patch turns were
  terminal, then changed HIP PID 71280 to fresh Vulkan PID 71789 only after the
  review was terminal. The fresh tool-free reporter completed from a 735-token
  capsule in 452.148 seconds under the frozen 600-second runtime timeout.

[`model-run3/candidate.patch`](model-run3/candidate.patch) is the readable patch.
[`model-run3/truth-review.json`](model-run3/truth-review.json) binds it to the
candidate, original fixture, frozen tests, guest receipt, terminal turns,
process-boundary PIDs, runtime logs, and manual report review. Direct post-run
hashes prove the original fixture solution and tests still match `protocol.json`.
The same candidate hash was independently checked over 1,728 Cartesian boundary
cases with strict Boolean return in
`model-run2/independent-native-oracle`; that is a separate unscored diagnostic,
not an inflation of the official seven boundary assertions.

The final model report faithfully states the objective, attributes Ptah's opinion,
and reports only the recorded SSH guest boundary and exit status. It omits some
receipt fields (cleanup, timeout, artifact links, and the explicit unchanged flag),
which remain in the capsule and truth-review receipt. No output claims general
repository competence, native illumos correctness, deployment readiness, GPU
driver correctness, or broad model quality.

`run_model_stack.py` now creates a fresh runtime process at each backend change
and refuses a switch until all owned turns are terminal. This is process
isolation, not a fix or root-cause claim for the earlier GPU reset. Two longer
fresh Vulkan report probes exceeded 180 seconds without another reset, while a
513-token report capsule completed; the final bounded capsule also completed but
was slow. Gemma's current Candle completion path still prefills the whole prompt
in one forward call. Prompt chunking exists only for the Qwen path and remains a
separate parity-tested engineering task.
