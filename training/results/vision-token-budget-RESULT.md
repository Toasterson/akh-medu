# Native text-vision token-budget preflight

**Date:** 2026-09-04  
**Context:** 1,024 total positions with 128 positions reserved for text

## Command

```bash
cargo run -q -p grug-train --example vision_token_budget -- \
  --max-seq 1024 --text-tokens 128
```

## Smallest tested patch that stays within one quarter of context

| Input | Dimensions | Result |
|---|---:|---|
| line | 384x64 | 16px, 96 visual tokens (9.4%) |
| receipt | 320x192 | 16px, 240 visual tokens (23.4%) |
| screenshot | 1024x768 | none of 8/16/24/32/48px |
| page | 1024x1024 | none of 8/16/24/32/48px |

At 48px, a screenshot still uses 352 visual tokens (34.4%) and a page uses
484 (47.3%). At 32px, a screenshot fits the absolute context limit but consumes
768 visual tokens; a page does not fit at all after reserving text positions.

## Decision

Line and receipt OCR can proceed to the stem-resolution probe. Full screenshots
and pages cannot enter decoder-scale training through naive full-image patches.
They require tiling/resampling or a compact visual front-end, followed by a
small-type readability gate. This is a preflight result, not evidence that a
particular patch size preserves glyphs.
