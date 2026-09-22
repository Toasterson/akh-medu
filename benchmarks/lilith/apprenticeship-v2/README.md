# Apprenticeship evaluation v2: public family holdouts

This is a reproducible **development evaluation**, not a release exam. It separates
candidate training families from evaluation families and independently checks
model-generated programs. It does not launch services or models.

The pinned manifest declares six tasks, three families at two widths, and four
arms: no retrieval, text retrieval, VSA retrieval, and an independent Ptah
reference. All arms have one attempt, zero tools, a 512-token output cap, a
4,096-token context cap, 1,400 lesson bytes, and a 600-second deadline. Failures do
not get a second attempt. Changing the verifier, task prompt, shared instruction,
model/tokenizer pin, or budget requires a newly frozen manifest. Old observations
remain tied to their original canonical manifest hash.

| Partition | Families | Interpretation |
| --- | --- | --- |
| Exposed training | saturation, bounded_range, permission_masks, byte_order, error_conventions | Includes prior lesson sources and new curriculum candidates |
| Development evaluation | alignment, interval_intersection, modular_sequence_order | Entire families excluded from candidate export and retrieval |
| Sealed certification | None | Must be designed independently later |

The modular task measures forward distance on a ring; it does not establish all
serial-number ordering semantics. Width variants are related tasks within a
family, not independent families. The new curriculum also publishes reference
examples for these holdout concepts: holdout means excluded from candidate
training, **not unseen or secret**. None of these receipts establishes C integer
overflow behavior, ABI correctness, or native illumos correctness.

## Commands

```sh
python3 scripts/session-platform/lilith_eval.py \
  benchmarks/lilith/apprenticeship-v2/manifest.json > /tmp/lilith-eval-schedule.json

python3 scripts/session-platform/lilith_eval.py \
  benchmarks/lilith/apprenticeship-v2/manifest.json \
  --results /path/to/records.json > /tmp/lilith-eval-report.json

python3 -m unittest discover -s scripts/session-platform -p test_lilith_eval.py
```

A runner should send `manifest.common_instruction`, two newlines, and the exact
task prompt in a fresh tool-free session. Additional lesson context must be
recorded separately and attributed as lesson data. Text/VSA can legitimately
abstain: the existing two-concept retriever may find no symbolically applicable
lesson for these families. Do not fabricate retrieved context or copy a baseline
observation into another arm. Identical empty contexts mean there is no retrieval
intervention to compare even if all runs pass.

The proposed schedule rotates arm order across tasks. A shared-runtime runner may
batch the reference model to avoid repeated residency swaps, but must record the
actual order and its deviation from the proposed schedule. This does not support
a speed comparison; cold/warm state and tokenization differ. `seed` identifies a
paired repetition; a runtime must additionally record whether it actually controls
the decoder random seed. The current one-seed manifest provides no estimate of
sampling variance.

## Record interface

Pass a JSON array to `evaluate(manifest, records)`. Each record has:

```json
{
  "task": "alignment-8",
  "arm": "none",
  "seed": 0,
  "manifest_sha256": "SHA256 of canonical manifest JSON",
  "model_sha256": "manifest.models.candidate.sha256",
  "tokenizer_sha256": "manifest.models.candidate.tokenizer_sha256",
  "budget": {"attempts": 1, "tools": 0, "max_new_tokens": 512,
    "context_tokens": 4096, "lesson_bytes": 1400, "wall_seconds": 600},
  "usage": {"attempts": 1, "tools": 0, "input_tokens": null,
    "output_tokens": null, "wall_seconds": 10},
  "retrieval": {"context": "", "lessons": []},
  "source": "the exact generated Python source",
  "status": "completed",
  "finish_reason": "stop"
}
```

`independent_reference` uses the reference model/tokenizer pins and receives no
lessons. Each retrieval lesson records `id`, `family`, and `source_sha256`. Other
statuses are `timeout`, `runtime_error`, and `cancelled`; all remain observed
failures. `finish_reason: length` is classified as output-budget exhaustion.
Preserve full raw session/error evidence beside these normalized records.

Token usage may be `null` only when the runtime did not expose a trustworthy
measurement. The report then marks `budget_usage_complete: false`; never insert a
configured cap or a text-based estimate as an observed count. Configured limits
must still be identical and pinned by the runner. Where native runtime faults
report exhausted output budget, normalize that event as `finish_reason: length`
with `status: completed` only if a generated output was actually returned;
otherwise keep `runtime_error` and preserve the specific native fault externally.

## Verification and failure analysis

The checker interprets a bounded arithmetic AST. It never executes generated code
with Python `exec`, and disallows imports, calls, loops, nesting functions,
containers, strings, exponentiation and shifts. The source cap is 4 KiB, AST cap
512 nodes, and each intermediate integer is bounded to 128 bits. Correctness is
finite reference testing, not a proof for every integer in the domain.

Boundary products cover negative/out-of-range inputs, zero, one, adjacent powers
of two, midrange and maximum-domain values. Results preserve the first failing
input and expected/actual values. Categories distinguish syntax, unsupported
syntax, execution faults, domain rejection, boundary semantics, arithmetic
semantics, runtime/timeout/cancellation, and output-budget exhaustion. These are
observed failure locations, not inferred root causes. A human or subsequent
controlled intervention must diagnose why a model failed before tuning it.

Manifests reject overlapping family assignments and verifier/prompt drift. Records
reject duplicated attempts, different budgets/model pins, retrieval from any
held-out family, and generated source matching an exposed training source exactly
or after AST normalization. These are local consistency checks, not a defense
against dishonest metadata or undiscovered pretraining contamination. The runner
must separately verify lesson provenance and tokenizer/model artifact hashes.

Aggregates show observed and expected runs, pass counts, failure categories and
paired outcomes against the no-retrieval arm. Missing observations remain missing;
infra failures do not disappear from comparisons. No p-values, confidence intervals,
equivalence or superiority claims are emitted from this tiny public sample.
The local reference is the pinned Ptah Qwen derivative; equivalence to the requested
upstream Qwen checkpoint is unresolved. **Qwen parity and native illumos gates
remain false**, regardless of pass counts.
