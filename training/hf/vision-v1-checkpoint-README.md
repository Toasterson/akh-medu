# compact-clean-line-v1 (experimental)

This is a 3,310,340-parameter native pixel-prefix + shared causal decoder
checkpoint. It passed one deliberately narrow **training-overfit** gate:

- 128/128 exact greedy transcripts;
- CER/WER/blank/cross-semantic collapse: 0;
- English, compiled bitmap font, 384x64 clean line images;
- all scored strings were present in training.

It is **not** a general OCR, multilingual, document-understanding, or
production checkpoint. Unseen strings/fonts, different-renderer and real
images, EN/DE/FR, corruptions/layouts, metamorphic groups, text retention and
grounded understanding remain ungated.

## Files

- `model.mpk` — Burn 0.21 full-precision named model record;
- `optim.mpk` — Adam state at step 4,000;
- `meta.json` — training configuration and measured V1 result;
- `compatibility.json` — exact byte-tokenizer, decoder, stem and image-input
  contract.

| File | Bytes | SHA-256 |
|---|---:|---|
| `model.mpk` | 13,250,031 | `95a131de80436d5ba3dd8b3c1cb3fa6a87cd446389e272cec8d8fbb9406c2dec` |
| `optim.mpk` | 26,499,107 | `4a31ec01182f69e7c43425a72555084c39a68873ac59e1043c474b8611a40e4c` |
| `meta.json` | 713 | `7fb4ec488b0616ab389aeb1ffbc2e29dc133b7f9e55fa824603c945f4f042562` |

The tokenizer is a fixed 260-ID byte mapping, not the BPE tokenizer at the
repository root: bytes 0--255, BOS 256, EOS 257, PAD 258, SEP 259. Do not load
this checkpoint with root `tokenizer.json`.

The matching public training images are in
`OpenFlowLabs/babyllm-dojo-kata/adaptive-exams/v1/compact-clean-line-training/`
(seed 51100, item-index stride 15). See the adjacent V1 result and provenance
bundle for the command, stopped attempts, exact source and environment.
