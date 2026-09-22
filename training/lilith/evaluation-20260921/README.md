# Lilith public family-separated evaluation

All 24 planned model observations completed in `run2`. Every project service was
stopped afterward, and ports 18420–18423 were confirmed closed. Fifteen CPU harness
tests pass. No weights changed.

| Arm | Strict tasks passed | Principal observed failures |
| --- | --- | --- |
| Gemma, no retrieval | 5/6 | 8-bit ring task: multiple definitions; omitted modulus upper bound |
| Gemma, text | 5/6 | Same failure and same empty lesson context |
| Gemma, VSA | 5/6 | Same failure and same empty lesson context |
| Independent Ptah derivative | 4/6 | 8-bit alignment arithmetic; 16-bit forbidden unary complement |

These are six public tasks in three families, not 24 independent tasks. All 12
retrieval attempts abstained, so **no retrieval benefit was tested or established**.
The raw strict count does not demonstrate Gemma superiority: the reference's
16-bit alignment rejection is solely an artificial grammar constraint, and its
normalized arithmetic passes. Qwen certification and native illumos remain open.

The runner in `run.py` executes the frozen
[apprenticeship v2 manifest](../../../benchmarks/lilith/apprenticeship-v2/manifest.json)
in a fresh private stack. The model/tokenizer artifacts and service binaries are
hashed before startup. It creates tool-free candidate/reference identities and
uses one shared runtime, Gemma on Vulkan and the local Ptah derivative on HIP.

The proposed rotating schedule is preserved within each model batch. Candidate
jobs run before reference jobs to avoid repeated model swaps. This deliberate
order change is recorded in `schedule.json` and rules out latency comparisons.
The seed field is a pairing label, not a claim of API-controlled random seeding.
There are no retries. Each request, complete session, retrieval result, normalized
record and partial aggregate is saved as it completes. Services launched by the
runner are stopped in its `finally` block.

`run1` stopped during artifact hashing before any daemon startup or model request
to correct elapsed-time accounting. It contains zero observations. `run2` is the
first model-execution attempt. Actual wall time is preserved, never capped to fit
the protocol; an over-budget record makes the aggregate fail closed.

Input/output token measurements are left null because the native session result
does not expose both reliably. Runtime limits are pinned and enforced, but
`budget_usage_complete` remains false. Full elapsed time includes loading and
polling. The checker rejects generated code outside the explicitly requested tiny
arithmetic language; this is a finite Python reference check, not arbitrary-code
execution or native illumos validation.

Retrieval excludes all held-out families and uses the existing verified lesson
store. None of its current concepts are symbolically applicable to these new
families, so abstention is expected. Equal empty lesson contexts cannot establish
text/VSA retrieval benefit. The public holdout concepts have reference examples in
the development curriculum; they are excluded from candidate training but are
not sealed or unseen. Ptah is an independently run pinned derivative, not a
certified upstream Qwen checkpoint. These results cannot establish release parity.

See `run2/result.json` for run completion/service cleanup, `run2/report.json` for
observed outcomes, and `run2/records.json` for the individual normalized records.

## Failure diagnosis, without changing the score

[`run2/failure-ledger.json`](run2/failure-ledger.json) preserves primary scores,
original hashes and separate diagnostic checks. `diagnose.py` reproduces those
post-hoc diagnostics without launching a model or rewriting primary records.

Gemma's 8-bit ring outputs contain two definitions of `ring_distance`. Checking
each definition separately still exposes a semantic contract miss: `(0,0,256)`
returns 0, while the task explicitly bounds **all three inputs**, including modulus,
to 0..255 and therefore expects -1. A conventional ring with 8-bit positions can have
modulus 256, but that differs from this task's parameter domain. Influence of that
convention is only a hypothesis; the observed omission does not establish its
cause. The 16-bit variant passing does not prove general understanding.

Ptah's 8-bit alignment expression computes the low-bit remainder rather than the
aligned value. At `(value=0, alignment=2)` it returns 1 instead of 0. Its 16-bit
expression uses the valid Python idiom `& ~(alignment-1)`, forbidden by this
artificial tiny-language protocol. An explicitly unscored AST rewrite using the
exact Python integer identity `~x == -x-1` passes all 169 reference cases. The
original strict score remains a failure, but this is not evidence of an arithmetic
or illumos-coding defect and does not warrant training merely to avoid `~`.

The next useful teaching material should use **fresh non-holdout families** to
practice checking each parameter's domain, distinguishing position width from a
modulus domain, and returning one final artifact. Do not feed these evaluation
solutions into the current candidate corpus or rerun this pack after tuning and
call it fresh evidence.

## Measurement limits

All observations returned within the configured deadline, with no runtime or
cancellation failures. Their recorded wall times sum to 557.98 seconds; this excludes
setup hashing and CPU scoring between turns. The first Gemma turn took 64.17 seconds
and the first Ptah turn 110.75 seconds including residency/load work. These numbers
are descriptive and are not a fair speed comparison. The checker executed 240,504
finite case evaluations across the primary outputs, including the first reference
counterexample; repeated cases across arms do not increase the independent sample
size. Token consumption remains unmeasured, so budget usage completeness is false.
