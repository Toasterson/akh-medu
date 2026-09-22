# Paired local reference comparison — 22 September 2026

The first complete same-pack deployment run collected all sixteen observations.
Independent CPU replay reproduces every saved verifier receipt.

| Pinned deployed model | Train-family tasks | Development-family tasks | Total |
|---|---:|---:|---:|
| Gemma 4 12B, quantized Vulkan | 1/4 | 3/4 | 4/8 |
| Ptah Qwen3.8 27B derivative, quantized HIP | 2/4 | 4/4 | 6/8 |

Lilith has not met the reference target on this pack. Ptah is specifically the
local `JonathanColetti/Qwen3.8-27B-Uncensored` derivative, not an unmodified upstream
reference. Eight public development tasks with one observation per model do not
establish statistical release parity in either direction.

Both models received the same prompts, 4096-token context and 512-token output
budget, in sequential model-resident batches on one GPU. All turns completed and
all owned services stopped. The runtime does not expose a per-request seed. This
is separate from the BF16 training-arm experiment and its different decoder
budget; the scores must not be pooled as if the runs were identical.

Gemma's failures split into two grammar violations and two genuine invalid-input
errors: division by zero for an invalid alignment, and accepting width zero for
an unsigned field. Ptah's two failures were grammar violations. Grammar rejection
is a failure of the frozen artifact contract, not proof of an incorrect algorithm.
No output was rewritten to improve these official scores.

Evidence: `run1/protocol.json` pins models, tokenizers, task manifest, runner and
binaries; `run1/records.json` stores exact outputs and finite verifier evidence;
`run1/*-session.json` preserves task sessions; `run1/result.json` records completion
and teardown; `independent-report.json` contains independent replay and per-task
counterexamples. Generated code was interpreted under the bounded AST contract
on the host, not executed there as Python. Native guest replay is recorded
separately under `../native-20260922/`.

The [independent provenance audit](independent-provenance-audit.md) additionally
binds all 16 records to their saved requests and sessions with zero errors. It
keeps the verifier and reporter limitations explicit: the JSON report is an
independent score replay, not a fully independent collection-provenance audit.
