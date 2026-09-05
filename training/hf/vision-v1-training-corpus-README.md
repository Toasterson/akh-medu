# Compact clean-line V1 training corpus

This directory materializes the exact 128 deterministic images consumed by
the compact shared-decoder bootstrap run.

- Domain: `training_sibling` (never an exam)
- Seed: `51100`
- Item indices: `0, 15, 30, ...` (`--index-stride 15`)
- Scope: English, Latin bitmap glyphs, line layout, clean rendering
- Semantic items / rendered cases: 128 / 128
- Unique normalized transcripts: 128
- Manifest SHA-256:
  `c1456f2701eaa012aa6d0ae6788e7e51989f6e00659e7c230a375b390c89527a`

The domain-separated same-renderer audit found zero exact transcript/pixel
overlap, max trigram Jaccard `0.3721`, and minimum foreground perceptual-hash
distance `19`; it admitted the corpus under the bootstrap policy. That audit
checks quarantine, not generalization. It is not the different-renderer or
real-image audit required for model promotion.

Passing an overfit gate on these images proves only that the compact stem and
shared causal decoder can learn a pixel-to-byte path at this scale. It does not
establish unseen-string, font, corruption, layout, or multilingual OCR.
