# Compressed KV behind a task-level gate

Date: 2026-09-20.

Compressed KV clears no numerical parity gate: worst per-vector cosine on the
live paged path is 0.99788 at 8 bits, 0.98540 at 4, against a 0.9999 bar, and
greedy-token agreement was shown to mislead (same codes, different tokens on two
devices). It is off by default. The user's decision for when it may be enabled:
a **task-level gate** - allow it at a bit-width only when the five-daemon
canary's recall, eviction and isolation checks passed at that width, for these
weights, on this device, accepting that this is weaker than a numerical bound.

## The gate

`validate_kv_gate` (in the native runtime) mirrors the existing runtime
conformance idiom - fail-closed, pinned to the exact artifacts:

- If the process is not configured for compression (`configured_kv_bits() ==
  None`), the gate stands down. KV off is the default and needs no evidence.
- If compression is enabled at N bits, a `KvGateEvidence` document is required
  (path in `NativeCandle.kv_quant_evidence`). It must record `passed: true` and
  pin `bits == N`, `model_sha256`, `tokenizer_sha256` and `device` to this load.
  Absent, malformed, failed, or mismatched on any pin, the load is refused.

So compression cannot run in the daemon without a passing canary document for
exactly this width, model, tokenizer and device. The gate is enforced at model
load in the native completion path, before any generation.

`KvGateEvidence` is a shared wire type (`akh-session-types`): `passed`,
`model_sha256`, `tokenizer_sha256`, `device`, `bits`, `recent`, `min_cosine`,
`checks`. `min_cosine` records the worst measured fidelity at that width for the
record; the gate does not threshold on it (that is the deliberately-declined
numerical bound), it thresholds on the canary having passed.

## Tested

Eight unit tests (GPU-free): the gate stands down when off; refuses when enabled
without evidence; admits a matching passing document; and rejects a wider width
than the evidence covers, a failed run, a wrong model, and a wrong device. The
enabling mechanism itself (the env-configured `configured_kv_quant`) is
unchanged and remains process-global, which is correct for a single-model
daemon; the gate constrains it.

## Not done

Emitting the evidence. The canary run at a bit-width already produces a passing
summary with its checks; a recorder wrapping that plus the measured `min_cosine`
into a signed `KvGateEvidence` for a chosen (model, tokenizer, device, bits) is
the operational follow-on. Until such a document exists and is referenced,
compression stays off - which is the safe default and the current state.

Per-load config threading (replacing the process-global env `OnceLock` with a
`NativeCandle` bit-width field carried into the model) was considered and
declined for now: it needs signature changes down the attention forward chain,
and the gate - the actual decision the user asked for - does not require it.
