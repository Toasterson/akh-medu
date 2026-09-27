# Lilith artifact-contract slice — 2026-09-21

The local coordination controller now accepts operator-selected, versioned
artifact contracts. The default integer clamp remains supported. The new
`u64-range/1` contract produces `range_within_u64.py`, a Python reference for
unsigned 64-bit bounded-range arithmetic relevant to systems programming.

[Operating guide](../../../examples/lilith/COORDINATION.md#select-an-artifact-contract).

## Contract and evidence boundaries

New assignments pin the contract ID, artifact path, specification and the verifier
module's SHA-256. Review handoffs include that specification and independently
computed evidence. Resumption rejects a changed grant before dispatching further
work. The model cannot select another contract by choosing another filename.
Cancellation and status remain available without changing verification authority.

The verifier admits a bounded AST with no calls, imports, loops or attribute
access. It checks 2,197 boundary combinations and 729 dense small-range cases
(2,926 evaluations, including overlaps). Python's unbounded sum supplies the
oracle independently of the requested subtraction implementation. Empty ranges
at the limit are valid; negative and out-of-uint64 values are invalid.

This establishes sampled Python reference behavior. It does not validate C
integer promotion/overflow, ABI or kernel behavior, or native illumos execution.
It is neither a benchmark parity result nor evidence of learned VSA transfer.

## Failure diagnosis

[verifier-diagnosis.json](verifier-diagnosis.json) records deliberate defective
implementations and concrete counterexamples: empty-range off-by-one, missing
limit bound, negative start, and integer rather than boolean return values.
A correct implementation omitting a redundant length-bound check passes; the
verifier does not require one spelling of a correct algorithm.

[run1/result.json](run1/result.json) records the first real-model failure:
`model_incomplete`, because the existing 256-token output cap was insufficient
for a complete response/tool envelope. No code-verification success or delegated
review was claimed. This is classified as a runtime output-budget failure,
separate from algorithm correctness. That attempt predates the explicit
specification field in the grant; it never reached artifact verification.

Lilith's prepared output cap is now 512 tokens; the 2,048-token context and Ptah's
supplied settings are unchanged. The verifier and its acceptance criteria were
not weakened to get a passing model run.

## Validation

[unit-tests.log](unit-tests.log): 48 passing scoped tests, including both artifact
contracts, meaningful counterexamples, unsafe syntax rejection, contract scope,
verifier mismatch after ledger reopen, registered-verifier identity binding,
specification delivery, and the previous
coordination/cancellation recovery cases.

The real-model integration uses Gemma on Vulkan and Ptah on HIP, serially through
one runtime. `run2/` records the retry with the final contract metadata and the
512-token cap. The harness owns and stops the four project daemons.

[run2/result.json](run2/result.json) **passed**: real model-requested delegation,
2,926 independent checks, Ptah review, controller reopen, truthful final report,
and same-process model switching (PID 271469). All services stopped. Ptah found
no defect and identified a redundant empty-range branch. Lilith clearly separated
host evidence from that opinion and excluded native illumos claims.

Runtime author/review/report durations were 133.6s, 115.1s, 267.0s. Cold-context latency remains
a material usability limitation.

Artifact SHA-256:
`35c0c9b1fad048dee36690135c587756c82f96250805a369d268baf89e8a2c23`.


No second specialist, VSA lesson ingestion, native illumos validation, or Qwen
quality parity is claimed by this slice.
