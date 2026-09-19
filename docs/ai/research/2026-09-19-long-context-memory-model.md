# Long-context memory model for the exact Ptah model

Goal item 6 requires quantization quality and long-context memory to be
*measured before activation*. This is the memory half, derived from the pinned
artifact's own metadata and the vendored implementation. Read-only; no GPU run,
no code changed.

## Only a quarter of the blocks grow with context

`src/nlu/vendored/quantized_qwen35.rs` decides per layer:

```rust
pub(crate) fn is_recurrent(&self, layer: usize) -> bool {
    layer < self.block_count && (layer + 1) % self.full_attention_interval != 0
}
```

`MixerState` is therefore either `Attention(Option<AttentionState>)`, holding a
key and value tensor that grow with position, or `GatedDelta(..)`, holding a
fixed-size recurrent state that does not.

From the model GGUF: `qwen35.block_count = 65`, `nextn_predict_layers = 1` (64
target blocks plus the MTP block), `full_attention_interval = 4`. So **16 of the
64 target blocks carry a growing KV cache** and 48 are constant-memory
recurrent. The MTP block falls outside the `layer < block_count` guard and is
also treated as attention.

## Per-token cost

`qwen35.attention.key_length = 256`, `value_length = 256`,
`head_count_kv = 4`. `AttentionState` holds plain `Tensor` keys and values and
the surrounding path is F32.

Per token, per attention block: (256 + 256) x 4 = 2,048 f32 = 8 KiB.
Across the 16 target attention blocks: **128 KiB per token**; including the MTP
block, 136 KiB.

## The runtime now reports this itself, and it matches

`CandleBackend::kv_bytes_per_token` derives the figure from the layers actually
loaded rather than from metadata, by summing `2 x kv_heads x head_dim x 4` over
the attention mixers present. The probe prints it at load. On the exact model it
returns **131,072 bytes**, matching 16 x 2 x 4 x 256 x 4 exactly, and projects
**34.36 GB** at the advertised 262,144-token context.

So the arithmetic below is no longer a hand calculation: the loaded model agrees
with it, and any future change to block layout or head geometry will move the
reported number with it.

## What that means on this card

Measured resident footprint with the model loaded is about 20.2 GB on a 24 GiB
RX 7900 XTX, leaving roughly 5 GiB before the peaks we already observe at 24.3
GB. At 128 KiB per token that is on the order of **40,000 tokens** of KV.

The model advertises `qwen35.context_length = 262144`. Storing that at F32 would
need about 34 GB of KV alone — more than the whole card, before weights. The
advertised context is therefore **not reachable on this device today**, and the
gap is roughly 6x.

This is arithmetic from metadata plus one measured footprint, not a measured
out-of-memory point. The daemon currently runs far below the limit: the canary
uses a 1,024-token window and the probe 2,048, which is why nothing has failed.

## What compression would buy, and what is unproven

`CandleBackend::compress_kv_vector` / `decompress_kv_vector` implement
PolarQuant — 3-bit Lloyd-Max with randomized Hadamard rotation, documented at
about 5.3x — and are **called from nowhere**. Their doc comment defers
integration because it "requires forking `candle-transformers` to wrap read/write
paths". That reason is now obsolete: the model is vendored in-tree and owns its
own `AttentionState`, so there is a place to hook without forking anything.

At 5.3x, 128 KiB per token becomes about 24 KiB, moving the resident ceiling from
roughly 40,000 tokens to roughly 220,000 — the advertised window becomes
plausible. That is the case for activation.

It is only a case. Nothing here measures what 3-bit KV does to output quality on
this model, and the existing exact-agreement gates (0.9999 cosine, bit-identical
hidden and logit streams) would necessarily fail against an F32 reference once
KV is quantized, so activation needs a *different* acceptance gate, agreed in
advance, rather than a relaxed version of the current one. Compressing only the
older pages and keeping a recent F32 window is the usual way to keep a
bit-exact recent path; that choice is unmade.

## Next

1. Measure PolarQuant round-trip error on real KV tensors captured from this
   model, per attention block, before any integration.
2. Decide the acceptance gate for a lossy KV path, since the bit-exact gate
   cannot apply to it.
3. Only then wire compression into `AttentionState`, oldest pages first.
