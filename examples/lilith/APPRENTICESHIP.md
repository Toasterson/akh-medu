# First Lilith apprenticeship slice

This slice implements the frozen-weight phase of the Lilith plan: verified
experience becomes durable, retrievable teaching material. Lesson admission and
retrieval keep the deployed Gemma weights unchanged and require no optimizer or
adapter activation. Separate adapter mechanics experiments are documented in the
readiness report below; they do not constitute curriculum training or deployment.

## What is admitted

`lilith_apprenticeship.py` accepts completed, non-cancelled assignments for the two
registered Python fixture contracts. It checks source hashes and historical
receipts, reruns the verifier, and records the contract specification, verifier
hash, author/teacher session and turn IDs, exact source, and teacher opinion.

The teaching principles are operator-curated from these verified episodes.
Teacher prose remains an attributed opinion; it is not promoted wholesale to a
verified training target. The old clamp episode is explicitly marked as having
no historical pinned contract; its source is replayed under the current verifier.
Repeated source/contract observations count once and cannot reactivate revocations.

The private store on this machine is:

```
~/.local/share/akh-medu/lilith-training-20260921/lessons.sqlite
```

It is outside model workspaces. Admission and retrieval use the local store and
host verifiers; no Anima database access is involved. Checksums detect accidental
corruption, not a malicious local operator who can rewrite the database.

## Retrieval and corrections

`examples/lilith_lesson_vsa.rs` uses the repository's existing VSA operations:
role/filler XOR binding, majority bundling and normalized Hamming similarity over
8,192 bits. SHA-256 counter expansion supplies a portable symbolic codebook;
features are locally re-encoded on each query. Peer neural hidden states or opaque
foreign vectors are never treated as compatible with this symbolic space.

The first vocabulary covers domain, operation, boundary, policy and evidence
scope. Features are curated by the operator, not inferred by a learned encoder.
Both text and VSA retrieval receive them. Text retrieval is a transparent token
Jaccard baseline; neither method is a production search system. Applicability
checks prevent an incompatible domain/operation from becoming lesson context.
Only a complete principle plus provenance may fit within the 1,400-byte budget.
Retrieval does not include the original solution code or teacher prose.

Revocation preserves the envelope and its reason, excludes it from retrieval and
export, survives restart, and cannot be undone by re-importing the same episode.
A changed verifier fails closed. Automatic lesson synthesis, richer precondition
checking, supersession links, and automatic model-driven corrections remain open.

## Commands

From the repository root:

```sh
cargo build --release --locked --example lilith_lesson_vsa --features hip,vulkan

python3 scripts/session-platform/lilith_apprenticeship.py \
  --db "$HOME/.local/share/akh-medu/lilith-training-20260921/lessons.sqlite" \
  admit training/lilith/contracts-20260921/run2/assignments.json

python3 scripts/session-platform/lilith_apprenticeship.py \
  --db "$HOME/.local/share/akh-medu/lilith-training-20260921/lessons.sqlite" \
  retrieve --mode vsa --query 'Check whether a request fits remaining capacity' \
  --features '[["domain","integer"],["operation","capacity_check"],["boundary","empty_allowed"],["policy","reject_invalid"],["evidence","host_python"]]'

python3 scripts/session-platform/lilith_apprenticeship.py \
  --db "$HOME/.local/share/akh-medu/lilith-training-20260921/lessons.sqlite" \
  revoke LESSON_ID --reason 'Evidence no longer applies'

python3 scripts/session-platform/lilith_apprenticeship.py \
  --db "$HOME/.local/share/akh-medu/lilith-training-20260921/lessons.sqlite" \
  export NEW_OUTPUT.jsonl --exclude-family bounded_range
```

Export refuses an existing output file. The development candidate format records
input, verified source output, family, lesson ID and source hash. Family exclusions
apply to both retrieval and export. This is a candidate corpus, not a certified
training set. The current export excludes the transfer probe family and contains
only one clamp seed; it is insufficient for meaningful adapter training.

## Measured transfer probe

The [predeclared development protocol](../../training/lilith/apprenticeship-20260921/protocol.json)
compares no lesson, text retrieval and VSA retrieval on `can_reserve`, an unsigned
16-bit capacity task. The same Gemma artifact, output/context caps, one attempt,
and zero tools apply to all arms. Each arm uses a fresh `lilith-learner` session;
this tool-free identity shares Lilith's existing runtime profile.

All three arms passed 1,512 independent checks. Text and VSA selected the same
lesson and supplied identical context. **No coding-quality gain or VSA advantage
was demonstrated.** The test is a related-family development transfer probe, not
family-held-out or sealed certification. Generated probe solutions never enter
the lesson store or candidate export. The shared model was already warm for later
arms, so the observed latency differences are not a fair speed comparison.

[Results and provenance](../../training/lilith/apprenticeship-20260921/README.md).

## Continuous admission and expanded training preparation

The coordinator now accepts `--lesson-db PATH` to replay and admit a lesson after
successful completion. Coding completion and learning admission are separate
recorded outcomes. Admission survives a lost acknowledgement without duplication.
Receipts bind to a persistent store UUID; resuming against a different store fails
explicitly. Each reconciliation refreshes revocation state. Rejected or cancelled
work cannot teach, and revoked evidence cannot be reactivated by replay.

The [reference curriculum](../../training/lilith/curriculum-20260921/README.md)
adds four training targets and four diagnosed correction examples, with separate
semantic-family development holdouts. Eight authored references pass 143,121 finite
checks; eight deliberately wrong variants fail. These are authored exercises,
not fabricated historical agent lessons. The corpus remains too small for a
meaningful weight-training claim.

The [expanded evaluation protocol](../../benchmarks/lilith/apprenticeship-v2/README.md)
compares six family-held-out tasks across no-memory, text, VSA and independent
reference arms. Its bounded interpreter checks generated programs without executing
arbitrary Python. Public development holdouts are not sealed certification.

The [24-run local comparison](../../training/lilith/evaluation-20260921/README.md)
completed with Gemma passing 5/6 tasks in each arm and Ptah passing 4/6 under the
strict output contract. One Ptah failure is solely a grammar restriction: an
unscored, semantics-preserving normalization passes its arithmetic checks.
Gemma's failing task contains duplicate definitions and a missing input-domain
check. Retrieval abstained throughout, so this measures baseline behavior and
provides no retrieval-benefit evidence. The failure ledger separates observed
logic defects from artificial harness constraints; neither scores nor diagnostic
normalizations establish Qwen parity.

The [adapter readiness audit](../../training/lilith/adapter-readiness-20260921/README.md)
records the initial blockers and their follow-up. A matching, hash-pinned official
parent candidate is now downloaded into a separate private cache, with an isolated
ROCm/PyTorch/Transformers/PEFT environment. The [actual mechanics receipt](../../training/lilith/adapter-readiness-20260921/mechanics-run1/result.json)
passes one rank-2 LoRA optimizer step on 16 synthetic tokens at the last attention
query projection. It records finite gradients, changed adapter tensors, unchanged
frozen tensor mutation versions, an observed forward-logit effect, exact adapter
and optimizer reload, and zero reload-logit error in the same loaded parent.
Peak GPU allocation was 22.41 GiB; no CPU offload was needed.

This establishes a bounded ROCm training mechanism, not useful coding adaptation,
a cold-process reload in that initial receipt, or adapter application in the deployed Vulkan runtime.
Only the scratch adapter changed. The deployed GGUF remains unchanged. The legacy
trainer also no longer relabels opaque binaries as safetensors or presents
unevaluated checkpoints as deployment-ready.

Automatic lesson synthesis, broader task coverage, curriculum tuning and a
verified Vulkan adapter/export path remain open. Native illumos receipts remain deferred until the VM host is ready.

The [first bounded KLPO experiment](../../training/lilith/rl-20260921/README.md) now adds actual probability/auxiliary-token capture, a tested token-MC objective, one supervised update and one RL update across 24 completed responses. Both scratch adapters changed, but strict scores stayed at training 1/4 and public family holdout 0/4 per arm. All seven rejected responses per arm pass finite logic checks after unscored removal of their enclosing Markdown fence. This is training-mechanics evidence and a reward-design diagnosis, not a coding-quality gain. Original scores and failed trajectories remain frozen.

A separate fresh process exactly reloaded the saved KLPO candidate, including adapter tensor fingerprint and checked log probabilities. That closes cold reload for this bounded HF/ROCm configuration; deployed Vulkan compatibility remains open.
