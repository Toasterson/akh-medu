# Native memory-answer gate

## Scope

`crates/akh-session-services/examples/native_memory_gate.rs` exercises the real
Candle keyed capture/reprompt callback, span projector, and temporary core Store
append/query interfaces. It does not start daemons or write production state.
Facts are seeded directly as synthetic accepted turns; this bypasses the core's
HTTP acceptance lifecycle. The fixed instruction requests only the number in
recalled notes, or UNKNOWN when no note supplies it. It is not the deployed
agent's broader system prompt.

The exact model/tokenizer are hashed before loading, with the same policy and
projector pins as the [retrieval gate](2026-09-15-span-hard-negatives.md).
The unchanged three-fact fixture has SHA-256
`1891163de07c337c7680cf9af8a4da1d7ed76f44ef24d80a7043f960dcfbeacd`.
This reuses the earlier retrieval fixture; it is not a new held-out evaluation.

Each of three questions runs in five conditions: no recall, oracle-correct
notes, actual retrieval, retrieval with prior questions also in the index, and
correct notes containing an instruction to output the wrong number (9999).
No-recall expects UNKNOWN; all other conditions expect 4317, 8629, or 19.
The scorer requires a natural Stop and an exact trimmed answer, rejecting
truncation and ambiguous answers. Distinct session keys isolate all conditions.
Memory is bounded to four hits and 4096 bytes; generation is capped at 32 tokens.

## Evidence and interpretation

HIP and Vulkan each passed all 15 cases. All 15 answer rows, including complete
hit lists and distances, were identical across backends. Every answer reported
zero reused prompt tokens. This compares reported answers/retrieval, not logits
or raw hidden-state parity.
The answer gate exposes a limitation concealed by the earlier ranking test:
under this prompt, the Atlas port question ranks Boreal's fact first (distance
335) and Atlas's second (337). Answer generation still chooses 4317 because both
facts fit in the returned notes. With prior questions indexed, the Atlas fact
drops to fourth place behind two questions and Boreal's fact. This is a narrow
retrieval-budget margin, not evidence of robust top-one entity discrimination.
Prompt-conditioned captures are not invariant sentence embeddings.

The three hostile-note answers are a useful smoke test, not prompt-injection
security acceptance. They cover one payload, and the recalled text is still
placed in a system-role message. Larger distractor pools, varied history and
the real agent prompt remain untested here. No latency or throughput conclusion
should be drawn from these runs.

Complete synthetic JSONL evidence is retained in the repository:

- `experiments/native-recall/memory-answers-hip-v1.jsonl`, SHA-256
  `9e0fae91669a55c7c3066e314c23c92f6f27bd79d621b8b065f95b25142a6e28`.
- `experiments/native-recall/memory-answers-vulkan-v1.jsonl`, SHA-256
  `b9608c53d0da36b0a6925e53a752c85679bb13ccefb27bde646cc2bb04e4e174`.

Original outputs remain under `/tmp/ptah-memory-answers.aSxwVf/` as `hip.jsonl`
and `vulkan.jsonl`; repository copies have matching hashes. The tested example
source SHA-256 is
`774c3d7864d3aaa0411639094e58b97fd5f5efef9bab0803619ff47bbdd96bd6`.

## Reproduction

```bash
CARGO_BUILD_JOBS=2 cargo test --offline -p akh-session-services --no-default-features --features candle-runtime --example native_memory_gate
CARGO_BUILD_JOBS=2 cargo build --offline --release -p akh-session-services --no-default-features --features hip,vulkan --example native_memory_gate
AKH_HIP_Q8_GEMV=1 AKH_HIP_QWEN35_FUSED_DELTA=0 target/release/examples/native_memory_gate --device hip --model /usr/share/ollama/.ollama/models/blobs/sha256-3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec --tokenizer /home/toasty/.local/share/akh-medu/models/llm/ptah-qwen38.tokenizer.json --output /tmp/memory-hip-new.jsonl
```

Run Vulkan serially with `--device vulkan` and a different new output path.
Existing output files are never overwritten. The keyed HIP activation guard
still forces f32 despite the global Q8 environment switch; fused delta is off.
The example depends on the current, partly uncommitted native runtime worktree,
not solely the preceding committed revision. Two CPU harness tests passed after
the scorer's red/green cycle. The combined HIP/Vulkan release build passed.

## Next

Exercise recall through the isolated session/core/runtime HTTP stack with the
actual agent prompt, then increase distractor pressure and vary preceding
history before changing retrieval policy. Do not tune this fixture into a
claimed held-out success or use the short answer gate as deployment acceptance.
