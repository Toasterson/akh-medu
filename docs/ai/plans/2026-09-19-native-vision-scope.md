# Exact vision path: what the artifact actually requires

Scoping for goal item 5, written 2026-09-19 against `470b373c`. Read-only
inspection of local artifacts; no code changed, no model downloaded, nothing
deployed.

## The exact model is multimodal, and the projector is already local

`registry.ollama.ai/library/ptah-qwen3.8/stable` carries four layers:

| Layer | Digest | Bytes |
| --- | --- | ---: |
| model | `sha256:3445102e9cde…daee3bec` | 16,810,714,496 |
| projector | `sha256:add205b7bfdb…ebc5288` | 931,145,984 |
| license | `sha256:2af71558e438…d53a608` | 10 |
| params | `sha256:dbef89b10ade…3167509` | 100 |

The projector blob is present on this machine. No artifact needs fetching, and
the exact model identity is preserved: this is the projector that ships with the
pinned Ptah model, not a substitute. A sibling manifest, `dflash2-100k`, carries
a `draft` layer instead of a projector; do not confuse the two.

## What the projector is

GGUF v3, 334 tensors, `general.type = mmproj`, `general.architecture = clip`,
`general.name = Qwen3.8 27B Abliterated`, tagged `image-text-to-text`:

| Key | Value |
| --- | --- |
| `clip.projector_type` | `qwen3vl_merger` |
| `clip.has_vision_encoder` | true |
| `clip.vision.block_count` | 27 |
| `clip.vision.embedding_length` | 1152 |
| `clip.vision.feed_forward_length` | 4304 |
| `clip.vision.attention.head_count` | 16 |
| `clip.vision.patch_size` | 16 |
| `clip.vision.image_size` | 768 |
| `clip.vision.spatial_merge_size` | 2 |
| `clip.vision.projection_dim` | 5120 |
| `clip.vision.image_mean` / `image_std` | 0.5 / 0.5 per channel |
| `clip.vision.is_deepstack_layers` | 27-entry boolean list |

Tensor inventory, 20 distinct shapes: `v.patch_embd.{weight,bias}` plus a second
`v.patch_embd.weight.1`, `v.position_embd.weight`, 27 blocks of
`v.blk.N.{attn_qkv, attn_out, ln1, ln2, ffn_up, ffn_down}` each with weight and
bias, `v.post_ln.{weight,bias}`, and the `mm.N.{weight,bias}` merger stack.

The second patch-embedding tensor is the signature of a temporal/3D patch
embedding, i.e. the video path, not only stills. I have not enumerated which of
the 27 deepstack flags are set; the first eight are false.

## Why none of the existing code applies

`CandleBackend::understand_audio` is the only multimodal path today. It requires
the **Gemma 4** family and loads `AudioProjector` from a `<model-stem>.mmproj.gguf`
sidecar derived by `mmproj_path()`. Three separate reasons it does not carry over:

1. Wrong family and wrong projector. Ptah is `ModelFamily::Qwen35` and the
   projector is `qwen3vl_merger`, not the Gemma audio projector.
2. Wrong artifact discovery. `mmproj_path()` appends `.mmproj.gguf` to the model
   stem; the projector here is a separate content-addressed ollama blob with no
   such relationship to the model blob's name.
3. The text model has no multimodal support at all. `quantized_qwen35.rs`
   contains no mrope, rope-section, deepstack or vision handling.

## What the slice actually needs

1. **Done** (`src/nlu/projector_artifact.rs`). The projector is located by
   manifest layer when the model is a content-addressed blob, falling back to
   the Gemma sidecar convention, which still wins when present. A manifest that
   lists the same weights without a projector does not end the search, and a
   manifest for different weights never contributes its projector. Four CPU
   tests plus an ignored artifact test that resolves the real pinned Ptah
   projector and checks its size.
2. **Preprocessing — resolved.** Read from the implementation that consumes
   this GGUF, `ggml-org/llama.cpp` `tools/mtmd`:

   - It is **dynamic resolution**, not a fixed square. The projector is driven by
     `mtmd_image_preprocessor_dyn_size`, and `clip.vision.image_size = 768` is
     not the resize target.
   - `calc_size_preserved_ratio` in `mtmd-image.cpp` is the transformers
     `smart_resize`: round each side to a multiple of `align_size`, then if the
     area exceeds `max_pixels` scale by `beta = sqrt(h*w / max_pixels)` and floor
     each side by the factor, or if it falls below `min_pixels` scale up by
     `beta = sqrt(min_pixels / (h*w))` and ceil by the factor. Sides are clamped
     to at least `align_size`.
   - `align_size = patch_size * n_merge` = 16 * 2 = **32**.
   - `patch_area = patch_size^2 * n_merge^2` = **1024**, and
     `image_{min,max}_pixels = n_tokens_{min,max} * patch_area`.
   - Our mmproj carries **no** `clip.vision.image_min_pixels` or
     `image_max_pixels` key, so the projector defaults apply. The Qwen3-VL branch
     of `clip.cpp` sets `set_limit_image_tokens(256, 16384)` alongside
     `FFN_GELU`, which matches this artifact's `clip.use_gelu = true`; the
     Qwen2/2.5 branch is the different one, at `(8, 4096)` with bicubic resize.
     That gives **min 262,144 px and max 16,777,216 px**.

   The token-limit pair was read from search fragments rather than a contiguous
   region, so confirm the two lines are in the `PROJECTOR_TYPE_QWEN3VL` case
   before relying on them, then pin the whole policy with a fixture whose patch
   count is asserted.

3. **Vision tower forward** in Candle: patch embed, position embed, 27
   pre-norm blocks with fused QKV, post-norm. Needs HIP and Vulkan coverage and
   the same exact-agreement gate the text path uses.
4. **Merger — smaller than assumed.** The `mm.*` stack is just
   `mm.0.{weight,bias}` and `mm.2.{weight,bias}`: a two-layer MLP with the GELU
   that `clip.use_gelu` announces between them, taking the merged patch width
   (1152 x merge^2 = 4608) to the LLM's 5120.

5. **Deepstack routing — not needed for this artifact.** All 27 entries of
   `clip.vision.is_deepstack_layers` are false, and the tensor inventory
   contains no `deepstack` tensors at all. llama.cpp loads them optionally
   (`TN_DEEPSTACK_NORM`, fetched with `required = false`). So this projector
   has no deepstack routing to implement, and the step that would have required
   changes inside `quantized_qwen35.rs` drops out.

6. **Positional handling** for image tokens in the text model, which does not
   exist today. The artifact already declares it: the model GGUF carries
   `qwen35.rope.dimension_sections = [11, 11, 10, 0]` alongside
   `rope.dimension_count = 64`, which is the Qwen-VL multimodal RoPE section
   split. `quantized_qwen35.rs` reads only `dimension_count` and applies plain
   1-D RoPE. That is why the text-only exact-agreement gates still pass — for
   text, the sections carry the same index — but image tokens need the sections
   honoured, and that edit lands in the position path the text gates cover.
7. **Plumbing** — image inputs through the session path and akhrtd's prompt
   rendering, and the chat template's image token placement.

Steps 1-4 are self-contained. Steps 5-6 touch the text model that currently
passes exact-agreement gates on both devices, so they need the same bit-exact
regression treatment the continuation and tap work used: an image-free prompt
must produce identical hidden states and logits before and after.

## Sequencing note

Step 6 overlaps goal item 6, live KV compression and long-context gates, because
both change position handling in the same file. Doing vision positions first and
compression second, against the same exact-agreement gate, avoids two
simultaneous edits to the position path.

## Revised size

With deepstack out and the merger reduced to two linear layers, the remaining
work is preprocessing, a 27-block ViT forward, a two-layer projection, and
honouring the mrope sections the model already declares. Only the last touches
`quantized_qwen35.rs` and therefore the existing exact-agreement gates.
