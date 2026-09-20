---
type: prototype
status: closed
resolved: 2026-09-19
blocked-by: [event-normal-form.md]
---

## Question

What exactly does one rendered transcript line look like, and how much does it
actually save?

[The compact render decision](compact-render.md) settled the *why* and the family
(High-Grug, per ADR 088). This ticket settles the grammar and proves the claim.

Make it concrete, then measure:

1. Draft the line grammar for each event kind from
   [the normal form](event-normal-form.md) — how role, tool name, arguments,
   output and truncation are carried, and what escapes a delimiter collision when
   a tool result legitimately contains the separator.
2. Render a real sample: a mid-size Claude Code session and a mid-size Codex
   rollout from the local corpus.
3. **Measure tokens, not bytes** — raw JSON vs render, with the tokenizer an
   agent would actually use. ADR 088 measured a halving on tool-call payloads;
   the claim here is different content and must be re-earned.
3b. **Use the byte-share tiers.** [The read surface](read-mcp-surface.md)
   measured what the corpus is actually made of: prose (`reply`/`prompt`/
   `thought`) is only **14.2%** of bytes, tool call+output is **29.3%**, and
   Codex `event_msg` alone is **37.7%**. The render should be calibrated to that
   — aggressive on process output, generous on decisions and plans — and its
   token measurement should be reported per kind, not as one corpus-wide ratio.
4. Decide what the render is allowed to lose. It is derived and the verbatim row
   is retained, so lossiness is permitted — but it must be *stated*, because a
   machine reading only the render needs to know when to page the raw row.
5. Decide whether the render is stored as a column or computed at read time.
   Storage costs space on a corpus already at 68 MB and growing; computing costs
   latency on every read.

Link the prototype as an asset; it is throwaway code answering a measurement
question, not the implementation.

## Resolution (2026-09-19)

Prototype: `assets/scripts/render_prototype.py` (runnable, reproduces every number
below).

### Grammar

Structure carried by a leading `<seq> <kind>` and ` | ` separated `<field> <value>`
segments; a field splits once on its first space, so values contain spaces without
quoting. No braces, quotes, colons or commas — ADR 088's shape.

```
412 prompt | t <text>
413 thought | enc 1
414 tool_call | n Bash | a {"command":"cargo test"}
415 tool_result | o <head>…[12043 elided]…<tail>
416 lifecycle | t agent-a44585373b94cca44
```

### Measured — bytes, per kind

| kind | raw MB | rendered MB | ratio | lossy? |
|---|---|---|---|---|
| `tool_result` | 45.94 | 4.14 | 11.1x | truncated |
| `tool_call` | 17.16 | 3.00 | 5.7x | truncated |
| `thought` | 21.22 | 0.24 | 89.5x | **discards opaque data — see below** |
| `context` | 9.93 | 0.08 | 128.6x | truncated |
| `reply` | 5.99 | 1.83 | **3.3x** | lossless |
| `prompt` | 1.42 | 1.05 | **1.4x** | lossless |

- **Whole corpus: 182.0 MB → 10.4 MB (17.5x)** — but 80.0 MB of that is *dropped
  noise*, not rendering.
- **Rendered records only: 102.0 MB → 10.4 MB (9.8x)** — still inflated by
  tier-2 truncation, which is lossy by design.
- **Pure format tax, prose only, nothing truncated: 7.41 MB → 2.88 MB = 2.6x.**

That last figure is the one comparable to ADR 088, and it is the honest headline:
**dropping JSON's envelope and punctuation costs about 2.6x on prose.** The larger
ratios are the tier policy from [the read surface](read-mcp-surface.md) doing its
job, not the format.

### Caveat: bytes, not tokens

The ticket asked for tokens. **No tokenizer is available** in this environment —
`tiktoken`, `tokenizers` and `transformers` are all absent, and the only local
`tokenizer.json` files belong to Grug's own identifier-aware vocabulary, which is
not representative of a frontier model's. So the token claim is **unverified**.

Worse, the byte ratio probably **overstates** the token advantage: modern BPE
merges JSON punctuation (`{"`, `":`, `",`) into single tokens, so JSON's token cost
is less than proportional to its bytes. Treat 2.6x as an upper bound until someone
re-runs this with a real tokenizer.

### What the render loses

- Tier 2 (`tool_call`, `tool_result`, `context`) is truncated head-and-tail with an
  explicit elision count. The verbatim row is always retrievable.
- Field names are abbreviated (`t`, `n`, `a`, `o`).
- Nested structure in tool arguments is kept as JSON inside the value, because
  flattening it would lose the argument boundaries that make a call reproducible.

### Storage

Store the render as a **column on the normalized row**, not computed at read time.
At 10.4 MB for the entire corpus to date, storage is irrelevant and read latency
is not.

## Finding that overturns earlier decisions: reasoning content does not exist

The `thought` row's 89.5x is **not compression** — it is discarding opaque data.

- **Codex `reasoning`: 4,642 records, 100% `summary: []` with `encrypted_content`.**
  Zero readable reasoning. 11.8 MB of ciphertext.
- **Claude `thinking`: 2,203 blocks, only 180 (8.2%) carry text.** The other 91.8%
  have an empty `thinking` field and an opaque `signature`. Total readable
  reasoning across the whole corpus: **0.11 MB — 0.06%**.

[The normal form](event-normal-form.md) made `thought` first-class citing "6,668
records … substantial and relevant to Phase-51 reasoning work". That counted
*records*, not *content*, and the content is not there. Corrections recorded on
[the normal form](event-normal-form.md),
[the read surface](read-mcp-surface.md) and
[the embedding pipeline](embedding-pipeline.md).

The `thought` **kind** stays — that a model thought at a point in the stream is
real, ordered information even when its content is sealed. But it is a marker, not
a body.
