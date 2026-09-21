# Held-out fact/query retrieval fixture

Current step: require entity/setting discrimination beyond the earlier
[easy paraphrase gate](2026-09-15-span-hip-vulkan-gate.md).

## Fixed test and result

Fixture: `experiments/native-recall/fact-query-hard-negatives-v1.json`.
Its SHA-256, recorded before either GPU run, is
`1891163de07c337c7680cf9af8a4da1d7ed76f44ef24d80a7043f960dcfbeacd`.
The three synthetic facts concern Atlas's TCP port, Boreal's TCP port, and
Atlas's snapshot retention. Their paired questions omit the answer values.
The fixture, labels, projection and prompts were not tuned after seeing results.

Before running, the test criterion was fixed to query-to-fact retrieval:
only the three facts are candidates; the three questions are queries, not
evidence. Every question must uniquely rank its paired fact first. This is a
different task from requiring separation among all six paraphrase-style pairs.
The original default paraphrase criterion is retained when no external fixture
is supplied. The artifact schema now includes a task discriminator.

HIP and Vulkan each passed 3/3. Both cross-backend directions also passed 3/3.
All six corresponding vectors were bit-identical, and repeated fresh-session
captures were identical within each backend.

| Query | Atlas port | Boreal port | Atlas retention | Winning margin |
| --- | ---: | ---: | ---: | ---: |
| Atlas port | **444** | 460 | 594 | 16 |
| Boreal port | 437 | **415** | 583 | 22 |
| Atlas retention | 608 | 610 | **450** | 158 |

These Hamming distances are identical on HIP and Vulkan. The 16/22-bit margins
are narrow: this does not establish robustness to paraphrasing, preceding
history, multilingual input or a larger candidate pool. Notably the two port
questions are only 157 bits apart, nearer than their supporting facts. Prior
questions as candidate distractors are therefore an important next control;
this fixture does not claim the production index filters them out.

## Reproduction and evidence

Use the release build, exact model/tokenizer, policy/projector pins and test
filter documented in the earlier gate. This turn rebuilt the current working
tree with `--no-default-features --features hip,vulkan`.
Add `AKH_SPAN_GATE_FIXTURE=experiments/native-recall/fact-query-hard-negatives-v1.json`
and a fresh `AKH_SPAN_GATE_OUTPUT` filename to each serial invocation.
The optional fixture loader accepts exactly three pairs of nonempty strings,
at most 8192 bytes per text and 64 KiB per fixture. The loader/test exporter
remain in the uncommitted `src/nlu/candle_backend.rs` runtime work.

Saved output files:

- `/tmp/ptah-hard-negatives.1g4KCZ/hip.json`, SHA-256
  `b2bcbb977a93622f17ba3ceaf15c6feff22cdc5baa66b307d33389c92b7103f4`.
- `/tmp/ptah-hard-negatives.1g4KCZ/vulkan.json`, SHA-256
  `149420aea648cc53686f42b2e73736251a1add544bc03eb2fa4881924da6f383`.

```bash
python3 scripts/session-platform/compare-span-gates.py /tmp/ptah-hard-negatives.1g4KCZ/hip.json /tmp/ptah-hard-negatives.1g4KCZ/vulkan.json
python3 -m unittest discover -s scripts/session-platform -p test_compare_span_gates.py -v
```

The new scorer regression first failed because it evaluated all six vectors
as queries/candidates; after adding task-aware scoring all four tests passed.
The previous saved paraphrase exports still compare successfully, including
backward compatibility for exports without a task field. The generic report
field is now `all_expected_neighbors_preserved`, replacing the paraphrase-only
name. The nine release projector and four capture/rewind tests also passed.

Elapsed GPU tests: HIP 33.97 s (load 24.03 s); Vulkan 51.25 s (load 31.89 s).
These are fixture runtimes, not throughput measurements. No raw means, real
user data or production state was retained/changed. GPU tests ran serially.

## Next: answers, not just neighbors

Use a temporary core Store and actual append/query for the first answer gate.
Capture facts through the native model pipeline, then test fresh conversations
where those facts are absent from visible history. Compare fixed-decoder
no-recall, oracle-correct-memory and retrieved-memory conditions; record both
retrieved identities and answers. Include wrong-entity facts, prior-question
distractors, abstention and instruction-like recalled text. This first gate
would not replace a full daemon HTTP acceptance test or ContextManager work.

A tool-disabled Fable consultation was attempted with
`docs/ai/plans/2026-09-15-memory-answer-fable-brief.md`; the CLI returned a Fable
usage-limit error. No new Fable advice was obtained and no alternative model
was substituted. Local response metadata remains at
`/tmp/ptah-hard-negatives.1g4KCZ/fable-review.json`.
