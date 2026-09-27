# MiMo fallback trial preparation — 2026-09-22

This is a separate fallback evaluation for
`XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`, prompted by the bounded Gemma quality
result (4/8) and the 452-second coordination report. It does not replace or
rewrite any Gemma, Ptah, adapter, coordination, or VM evidence.

The official checkpoint revision is
`2367e865d009c13ac81713a2878291d33ab28177`. The upstream metadata snapshot and
hashes are in `upstream-metadata/`. The checkpoint is an 18,819,627,488-byte
BF16 `Qwen3_5ForConditionalGeneration` model with a vision tower and a 32-layer,
4096-wide text decoder. The four official shard hashes are fixed in
`scripts/session-platform/lilith_mimo_trial.py`.

The model card requires recent Qwen3.5 support, the included MiMo v2.6 chat
template, `--reasoning-parser mimo`, and thinking enabled. The existing native
Candle backend cannot satisfy that contract: it loads GGUF artifacts and renders
Qwen3.5 as ordinary ChatML. The trial therefore uses the already-installed
Transformers Qwen3.5 implementation, exact checkpoint processor/template, and
thinking enabled. It records reasoning and answer separately at the last literal
`</think>` marker before applying the unchanged v2 source verifier.

The initial fit smoke uses the exact same 512 generated-token cap as
`comparison-20260922`. With the required thinking mode, the first response used
all 512 tokens on an unterminated reasoning trace and produced no answer, so the
verifier correctly rejected it. The full quality trial therefore uses an
8,192-token reasoning budget and is explicitly **not cap-comparable** to the
Gemma/Ptah 512-token scores. Both trials use the same eight public v2 coding
prompts and semantic verifier. They use the checkpoint's official sampling
values (temperature 0.6, top-k 20, top-p 0.95) and record the seed, token counts,
wall time, raw output, reasoning,
answer, verifier counterexample, GPU allocation, model load time, and every
artifact hash. Backend, precision, and prompt rendering differ from Gemma/Ptah
and are declared in `protocol.json`; scores are useful observations rather than
release parity or a promotion decision.

Each full-trial task has a 600-second wall bound. A response that exhausts its
budget without closing `</think>` is retained as `completion_budget_exhausted`
and excluded from semantic scoring rather than mislabeled as an incorrect final
answer. The runner records time to first token, reasoning time, answer time, and
whether an EOS token ended generation. The 8,192-token choice is larger than the
official card's 2,048-token example so the reasoning model can complete; the
extra KV requirement fits the remaining device memory because only eight of the
32 text layers use full attention.

No GPU work starts without `LILITH_MIMO_GPU_HANDOFF=1`. `--check` verifies the
complete local snapshot and frozen task pack without loading the model or using
the GPU.

## Results

`fit-smoke1` is retained as a pre-model failure: Transformers 5.10.1's
`AutoProcessor` rejected the checkpoint's vision image-processor metadata. The
text-only path does not need a vision processor, so `fit-smoke2` used the exact
checkpoint tokenizer and standalone `chat_template.jinja`. The full BF16 model
fit, allocating 18,819,708,416 bytes after load and peaking at 18,930,840,576
bytes. Load took 77.43 seconds. Its one 512-token response exhausted the budget
inside an unclosed reasoning trace, so it has no semantic score.

`full8-thinking8192` completed all eight tasks. Every trace closed `</think>`,
every answer ended at an EOS token, and no task exhausted the 8,192-token or
600-second limits. The independent replay in `independent-audit.json` confirms:

- train: 3/4;
- public development holdout: 3/4;
- total: 6/8.

The two failures are concrete. `align-up-power2/2` omitted the required valid
power-of-two and zero check, producing division by zero at `(0, 0)`.
`signed-field-extract/2` produced plausible logic but used forbidden augmented
assignment (`x -= ...`), so the frozen grammar rejected the artifact. This is a
contract failure even though the intended arithmetic is recognizable.

The full run took 416.08 seconds including a 106.72-second model load. Generation
ranged from 3.76 to 120.90 seconds and from 74 to 2,456 output tokens. Warm time
to first token was about 0.19–0.24 seconds; the cold first task was 5.93 seconds.
These are BF16 Transformers/ROCm observations with the slow Torch fallback for
linear attention, not GGUF deployment throughput.

MiMo's numerical 6/8 equals the recorded Ptah result and exceeds Gemma's 4/8,
but the comparison is not controlled: MiMo received an 8,192-token thinking
budget and exact MiMo template, while Gemma/Ptah used 512-token quantized session
runtimes with different templates and backends. The result justifies further
controlled evaluation; it does not justify promotion.

## Native illumos replay

`native-illumos/` replays the seven grammar-admitted canonical artifacts in the
pinned OpenIndiana SSH guest. It uses the frozen task cases without changing the
extractor, grammar, or interpreter score. Six artifacts pass natively. The
`align-up-power2/2` artifact exits 1 on the same division by zero at `(0, 0)`
found by the frozen interpreter. `signed-field-extract/2` remains rejected by
the frozen grammar and was not sent to the native executor. The score therefore
remains 6/8; this replay is additional execution evidence rather than rescoring.

Each admitted answer has its own receipt with the raw-output and extracted-source
hashes, exact test-source hash, guest boundary, exit status, timeout status, and
cleanup result. Each receipt ran one `unittest` method that loops over that
answer's frozen cases. The executor contract hash is
`ce73f2c4da203e537a4c3899453bccec0093b127ae47b989ee2d9b5881217b59`.

The first native attempt is retained in `native-illumos-attempt1-fenced-source/`.
Its harness incorrectly wrote the model's Markdown fence into `solution.py`, so
all seven admitted answers failed with `SyntaxError`. The corrected replay uses
the already-frozen canonical artifact extractor; it does not hide or rewrite
that harness failure.
