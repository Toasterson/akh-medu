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
2. **Preprocessing — resolved and implemented** (`src/nlu/vision_preprocess.rs`). Read from the implementation that consumes
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

   `ResizePolicy::qwen3vl()` carries the policy and six CPU tests pin it:
   alignment, aspect preservation within one alignment step, the maximum-area
   pull-down, the minimum-area push-up, a sliver keeping a full merge block, the
   bounds being independently disablable, and the byte-range normalization.

   The token-limit pair was read from search fragments rather than a contiguous
   region, so confirm the two lines sit in the `PROJECTOR_TYPE_QWEN3VL` case
   before trusting the exact bounds; the tests assert the *properties*, so a
   corrected bound changes one constructor rather than the algorithm.

3. **Configuration and weight validation — done** (`src/nlu/vision_tower.rs`).
   `VisionConfig::from_gguf` reads the declared geometry and rejects a
   non-`qwen3vl_merger` projector; `validate_tensors` checks that every tensor
   the forward pass needs exists at the shape the geometry implies. An artifact
   test runs this against the real 931 MB projector and confirms 27 blocks,
   width 1152, FFN 4304, 16 heads of 72, patch 16, merge 2, projection 5120,
   merged width 4608, mean and std 0.5, deepstack disabled, a 48x48 position
   grid covering the 768 px reference, and the optional second temporal patch
   embedding. It also asserts that the policy derived from the geometry equals
   the one `vision_preprocess` implements from llama.cpp, so the two cannot
   drift apart silently.

   Writing this caught a real error: candle reports GGUF dimensions reversed, so
   weights read as `(out, in, ...)` rather than the file's order. The first
   version of the expected shapes was wrong in exactly the way that would have
   produced a silently transposed forward pass.

3b. **Vision tower forward — implemented** (`VisionTower::load` / `forward`).
   Patch embedding by strided convolution, learned position embeddings resampled
   bilinearly onto the actual patch grid, 27 pre-norm blocks with fused QKV and
   bidirectional attention over patches, post-norm, spatial merge, then the
   two-layer merger with GELU. Weights are dequantized from BF16 to f32 once at
   load. An artifact test runs the real projector on CPU and gets finite,
   non-degenerate `(merged_patches, 5120)` embeddings in about five seconds;
   another asserts it refuses geometry it cannot patch evenly.

   **Not yet numerically validated, and not validatable locally.** Shape and
   finiteness are checked, not agreement with a reference implementation.

   An attempt to build a NumPy reference was abandoned: numpy is not installed
   and installing it is outside the standing rules. Pure Python is not an
   option either — one 4-patch forward through 27 blocks at width 1152 is on the
   order of 1.6 billion multiply-adds.

   A second implementation *in this repository* was considered and rejected on
   merit rather than effort. Written by the same author from the same reading of
   the same artifact, it would share every comprehension error: if the temporal
   patch weights should be averaged rather than summed, or the position grid
   resampled half-pixel rather than align-corners, both implementations would be
   wrong identically and agree. It would catch coding slips such as a
   transposition, which is worth something, but it would read as validation
   while proving much less, and that is worse than no check.

   Real validation needs an **external** consumer of this GGUF producing
   embeddings or logits for a known image — llama.cpp's `mtmd`, or the
   transformers implementation. Neither is available on this machine. Until one
   is, the three open modelling choices (summing the temporal patch weights,
   align-corners interpolation, tanh GELU) stay open, and the tower's output
   should be treated as structurally right and numerically unconfirmed. Three choices need confirming
   against llama.cpp before the output is trusted: that a still image should sum
   both temporal patch-embedding weights rather than using one, that position
   interpolation is align-corners bilinear, and that `use_gelu` means the tanh
   approximation candle's `gelu()` implements. Nor has it run on HIP or Vulkan.

3c. **Old step 3 text** in Candle: patch embed, position embed, 27
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
