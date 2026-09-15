# Native runtime through private session/core daemons

This extends the [explicit-note fixture](2026-09-15-native-note-instructions.md)
to real `akhsessd`, `akhrtd`, `akhsbxd`, `akhexecd` and specialized cognitive
`akhomed` processes. No scheduler state is written by the test. It submits
inputs through HTTP and observes completion, context and publication events.

## Test surface

`scripts/session-platform/native-core-smoke.py` creates private loopback
listeners, temporary XDG/workspace/state directories and random service
credentials. Only its child processes are stopped. It does not inherit
database, upstream API or telemetry credentials, or contact shared services.
The model and tokenizer have the same hard-coded exact artifact pins as the
preceding gates. HIP keeps keyed f32 activations and disables fused delta.

The real core composes its normal system instructions. The canary therefore
uses a 1024-token window, a 16-token output reservation, and repeats each of
the three filler notes twenty times. Its expected answers remain four `OK`
responses followed by `4317`. This is a distinct daemon fixture, not a claim
that the earlier 160-token fixture can fit the full core prompt.

Each turn must have a `context_prepared` event, a matching public cognitive
receipt with capture enabled, and its actual user text retrievable from the
core span interface. A zero-vector audit query enumerates the five stored
spans; that query is not used to supply the model's recall. The final question
must have native window-eviction and recall-materialization telemetry. The
telemetry contains counts only, not recalled text or vectors.

The private session daemon and cognitive core restart before the question.
The provider binding must remain unchanged. Querying another newly created
session must return no spans. Core/session persistence and runtime residency
are thus separate: this fixture does not restart the GPU runtime.

TDD: the public receipt validator first failed its positive test with `None`,
then passed. Additional tests reject another turn's receipt, completion without
publication, disabled capture and an incorrect receipt identity.

## Build separation

Building all five executables in one Cargo invocation unified Vulkan into the
core binary and failed with E0275 while proving wgpu types `Sync`. Build the
CPU-only cognitive core separately; it does not own inference:

```bash
CARGO_BUILD_JOBS=2 cargo build --offline --release -p akh-medu --no-default-features --features core-server --bin akhomed
CARGO_BUILD_JOBS=2 cargo build --offline --release -p akh-session-services --no-default-features --features session,executor,sandbox,hip,vulkan,core-integration-tests --bin akhsessd --bin akhrtd --bin akhsbxd --bin akhexecd
python3 scripts/session-platform/test_native_core_smoke.py
```

The deployment remains uncut: this is a private local acceptance step and the
native integration still depends on the existing mixed working tree.

## Results

The CPU-only core built successfully in 6m34s; the four session-platform
executables built in 2m08s. The first harness invocation found Python 3.10's
missing `hashlib.file_digest`; the bounded streaming replacement is checked
against SHA-256 of `abc`. Startup then exposed that `AkhPaths` requires HOME
even with all XDG paths set. The harness now inherits HOME unchanged while
keeping every Akh XDG destination private. No inference occurred in these
failed startup attempts. Four Python tests pass.

HIP passes the complete private daemon canary. Answers are `OK`, `OK`, `OK`,
`OK`, `4317`; the public core query returns exactly one additional span after
each accepted turn. Before the final question, the private core and session
daemon restart successfully. The final native prompt evicts four messages
and materializes one recalled span. The provider binding stays unchanged;
the second session has no spans. All five temporary processes stop on exit,
and their disposable state/workspace is removed. No shared data was deleted.

- [HIP evidence](../../../experiments/native-recall/daemon-core-hip-v1.jsonl):
  `c6df4146aa245199f0268087d535f9746cefd2e9471fb92d3923261cd57fb2bd`
- Tested harness SHA-256:
  `7850df19e5fff0981e55acf969c0784e2ba4493b9251194844642574bc08d287`

Vulkan also passes with the identical five answers and stored-span counts.
After its private core/session restart, the final prompt likewise evicts four
messages and materializes one recalled span. The fresh session's query is
empty. A process check after cleanup found no remaining instances of any of
the five daemon names. This is process cleanup, not a resident GPU idle-power
measurement.

- [Vulkan evidence](../../../experiments/native-recall/daemon-core-vulkan-v1.jsonl):
  `17f8294fe900bb7fef6aff2d59ec747a868f3e4d92fce812490763d6b7a79fa2`

The evidence records all five executable hashes and representation pins.
Run using explicit artifact paths and a new output file:

```bash
python3 scripts/session-platform/native-core-smoke.py --device hip \
  --model /usr/share/ollama/.ollama/models/blobs/sha256-3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec \
  --tokenizer /home/toasty/.local/share/akh-medu/models/llm/ptah-qwen38.tokenizer.json \
  --output /tmp/native-core-hip-new.jsonl
```

This does not validate PostgreSQL-backed deployment, a production persona,
general instruction adherence, GPU throughput, long-context KV compression,
or canonical homelab cutover. Those remain separate gates.

Next: consolidate the accumulated native implementation into a reproducible
release checkpoint, then run held-out recall/safety and current keyed-path
performance gates before deployment cleanup. The small count-only debug log
used by this canary still lives in the untracked `internal_native.rs` integration;
this test checkpoint alone is not a clean deployable runtime release.
