# Exact-model span projection: HIP/Vulkan fixture gate

Current step: measure the compact projector on the exact Ptah model and compare
representations across HIP and Vulkan. These are the existing three paraphrase
pairs, not a held-out benchmark or a memory-answer evaluation.

## Results

Both device runs passed. Every fresh-session repeat was bit-identical within
its backend. Comparing corresponding HIP/Vulkan vectors found **zero bit flips
in all six 10000-bit vectors**. Excluding the identical query from the opposite
backend's index, each sentence's unique nearest neighbor was its paraphrase:
6/6 HIP-to-Vulkan and 6/6 Vulkan-to-HIP.

| Pair | HIP distance | Vulkan distance |
| --- | ---: | ---: |
| Cat name/diet | 335 | 335 |
| Cold Docker cache/deployment failure | 380 | 380 |
| Portugal capital/population | 396 | 396 |

Unrelated-pair distances ranged from 593 to 755 in both runs. The maximum
within-pair distance was 396 and the minimum unrelated distance was 593.
These unrelated distances are still small relative to 10000 bits: passing this
fixture does not establish a general similarity threshold or semantic coverage.

HIP test elapsed 37.08 s, including 26.81 s load; Vulkan elapsed 54.43 s,
including 33.50 s load. These totals include capture repetition and diagnostics;
they are **not decode throughput benchmarks**. GPU tests ran serially after
compilation finished. No production/shared service was changed.

## Identity and reproduction

SHA-256 was recomputed before testing:

- Model: `3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec`.
- Tokenizer: `e271c182ffa63eebd6d0f60a688fb90d3e87a1141c99c4d87a0d4508badfefea`.
- Capture policy: `ed60b0f2dc2be049a25df4144da46bb10117314ff63ad7efb0642406a955e839`.
- Projector v2: `c53bdfeb9c2e601e1595564b1fd3d56f3aa1ef217396e82f8588d34764b73382`.

Build the current working tree (native runtime changes still include uncommitted
files; this report does not imply a clean-source release):

```bash
CARGO_BUILD_JOBS=2 cargo test --offline --release --lib --no-default-features --features hip,vulkan --no-run
```

Test executable reported by Cargo:
`target/release/deps/akh_medu-b17c3e317abfd74d`.
Filter: `installed_artifact_span_projection_ranks_paraphrases_above_unrelated_text`.
Arguments: `--ignored --nocapture --test-threads=1`.
Environment for each serial run:

```text
AKH_HIP_Q8_GEMV=1
AKH_HIP_QWEN35_FUSED_DELTA=0
AKH_QWEN35_SESSION_DEVICE=hip  (then vulkan)
AKH_QWEN35_TEST_MODEL=/usr/share/ollama/.ollama/models/blobs/sha256-3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec
AKH_QWEN35_TEST_TOKENIZER=/home/toasty/.local/share/akh-medu/models/llm/ptah-qwen38.tokenizer.json
AKH_SPAN_GATE_OUTPUT=<a new output filename per run>
```

Keyed HIP generation still scopes activation arithmetic to f32. The test-only
export writes only the bounded synthetic fixture, its projected vectors and
representation identifiers; never raw pooled means. It uses `create_new` and
refuses to overwrite existing evidence. The export addition is currently part
of the uncommitted `src/nlu/candle_backend.rs` runtime work.

Saved local outputs:

- `/tmp/ptah-span-gates.O6Mudi/hip.json`, SHA-256
  `05fe63d1342dbb4a45b914514ddfd6e1447ceb0fa6a8440433e13f77c761f185`.
- `/tmp/ptah-span-gates.O6Mudi/vulkan.json`, SHA-256
  `c5f400212e2951c1bb6acaad2942fd30cd3120e5d0cfc7d5efc1bd065b8d6f37`.

```bash
python3 scripts/session-platform/compare-span-gates.py /tmp/ptah-span-gates.O6Mudi/hip.json /tmp/ptah-span-gates.O6Mudi/vulkan.json
python3 -m unittest discover -s scripts/session-platform -p test_compare_span_gates.py -v
```

The comparison checks identity/fixture agreement and fixed byte lengths,
reports bit differences separately, and rejects tied nearest neighbors as
successes. Its three tests passed (known distances, ties, malformed/mismatched
inputs). Release projector tests (9) and capture/rewind tests (4) also passed.

## Next steps

Use held-out questions, hard negatives, multilingual inputs and varied preceding
history. Then test actual answers whose required fact is absent from visible
history but present in the accepted-span annex. Include no-recall controls and
recalled-text placement/instruction-safety cases. This six-sentence result is
not enough to declare arbitrary HIP/Vulkan representation compatibility or to
remove those gates. ContextManager, vision, live KV compression, performance and
deployment/cleanup remain unfinished.
