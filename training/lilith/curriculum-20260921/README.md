# Replayable reference curriculum

This expands the candidate corpus with **assistant-authored reference exercises**.
These are not generated Gemma solutions, Ptah lessons, or historical agent episodes.
No model weights were updated. No native illumos validation was performed.

The replay checks 8 reference implementations over **143,121 finite cases**, and
rejects one deliberately incorrect implementation per exercise. Four training
candidates and four separate correction examples are exported. This small corpus
is still insufficient evidence for useful adapter training.

| Semantic family | Exercises | Export role |
| --- | --- | --- |
| permission_masks | All required flags must be present | Training candidate |
| byte_order | Decode and encode unsigned little-endian 16-bit values | Training candidate |
| error_conventions | Normalize abstract integer errors to nonpositive values | Training candidate |
| alignment | General alignment; positive power-of-two predicate | Development holdout |
| interval_intersection | Half-open interval overlap | Development holdout |
| modular_sequence_order | Unsigned 32-bit serial ordering | Development holdout |

Family assignment is semantic, so renaming a function does not move it across the
split. Both byte-order directions share one family. The entire alignment family
is excluded from training and corrections to avoid contaminating the independent
development evaluation slice. The earlier saturation/bounded-range apprenticeship
examples are already exposed and must not be described as unseen evaluation
families. All holdouts and their references are public development material; this
is **not a sealed certification set**. Family disjointness is a grouping rule,
not a guarantee that mathematical concepts do not transfer between families.

Each reference has a differently constructed oracle: byte conversions use Python's
byte APIs, mask checks use sets of bit positions, intervals use set intersections,
and alignment uses quotient/remainder arithmetic. Byte encoding and decoding are
exhaustive within their stated finite domains. Other tasks use bounded grids and
edge cases. Finite checks do not establish correctness over all integer inputs.
The integer errno convention is explicitly abstract and is not an illumos ABI claim.

`data/candidates.jsonl` contains verified reference targets. `data/corrections.jsonl`
contains erroneous code, an executable witness, an authored diagnosis and the
corrected reference. Do not use `incorrect_source` as a positive SFT target.
`data/development-holdouts.jsonl` exports prompts without targets. The manifest
pins exact verifier-module and case-list SHA-256 hashes, source hashes, receipts,
and null teacher identities. The reference code and public holdout answers remain
in the replay module; training dataset consumers must use the candidate export,
not scrape every exercise from that module.

## Replay

From repository root, choose a fresh output directory:

```sh
python3 scripts/session-platform/lilith_curriculum.py /tmp/lilith-curriculum-replay
python3 -m unittest discover -s scripts/session-platform -p 'test_lilith_curriculum.py'
```

The command refuses an existing output directory. Replays are deterministic with
the same module version. Five tests pass, covering references and negative
controls, family separation, truthful provenance, deterministic non-overwriting
exports, strict boolean outputs and rejection of common side-effect syntax.

The integration seam is `exercises()`, `verify(exercise, source)` and
`build(new_output_path)` in `scripts/session-platform/lilith_curriculum.py`.
`verify` is intended for these trusted curated references. Its restricted AST is
not a resource-isolated sandbox for adversarial model-generated code. Future live
model evaluation must use a bounded worker or the project sandbox. Admission into
the historical lesson store is deliberately separate: this dataset has no agent
assignment or teacher receipt and must not be passed off as one.
