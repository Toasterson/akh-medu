# Adaptive exams v1 Hugging Face publication result

**Date:** 2026-09-04  
**Organization:** OpenFlowLabs  
**Visibility:** public, verified without authentication after upload

## Dataset

- Repository: <https://huggingface.co/datasets/OpenFlowLabs/babyllm-dojo-kata>
- Exact public head: [`3e7b0261d6ddf281e9ee9034339a1f526c475b3b`](https://huggingface.co/datasets/OpenFlowLabs/babyllm-dojo-kata/commit/3e7b0261d6ddf281e9ee9034339a1f526c475b3b)
- Artifact payload commit: [`54bd84b63c8a10b577edcaa308836391e376f0a3`](https://huggingface.co/datasets/OpenFlowLabs/babyllm-dojo-kata/commit/54bd84b63c8a10b577edcaa308836391e376f0a3)
- Parent before this release: `8bc665763ba4fe3701659479e4f07f5e7d64ae23`
- Published bundle: `adaptive-exams/v1/`
- Public repository files after the upload: 317 total; 311 under the bundle
- Bundle `SHA256SUMS` SHA-256:
  `ddac72ccd450938998e4dff51c6625fea032fbf35c7343a6eeec757090ff806d`
- Demo corpus: 30 semantic training-sibling items, five variants each, 150 PNGs
- Demo manifest SHA-256:
  `eda7e275f91fa2858a71e416a2ba705af18f2f3ae17f66625f6dac105477d014`
- Demo first PNG SHA-256:
  `1f2328a8957c122871e4984c539d5fcfee2d804d5c6dc836c1a227bebae8e2a5`
- Exact V1 training corpus: 128 English clean-line images and 128 unique
  transcripts, seed 51,100 with item-index stride 15
- V1 manifest SHA-256:
  `c1456f2701eaa012aa6d0ae6788e7e51989f6e00659e7c230a375b390c89527a`
- V1 first PNG SHA-256:
  `ebcf380a5e5101295607eb1b987ff55a736a5e251508e02d1636ec70b7b9d4e5`

Both remote manifests, both first PNGs, the dataset card, reviewer record and
bundle checksum manifest were downloaded at the exact commit with no token and
matched the staged bytes. The image sets are labelled training siblings, not
sentinel and not independent audit.

## Model repository

- Repository: <https://huggingface.co/OpenFlowLabs/grug-rule-350m>
- Exact public head: [`80794cd59ccd0134548181568a223f40c2e3c07d`](https://huggingface.co/OpenFlowLabs/grug-rule-350m/commit/80794cd59ccd0134548181568a223f40c2e3c07d)
- Checkpoint payload commit: [`3908c23bd3a2c9e33a26bb079989c9501017e187`](https://huggingface.co/OpenFlowLabs/grug-rule-350m/commit/3908c23bd3a2c9e33a26bb079989c9501017e187)
- Parent before this release: `fa172939c71ca7ccf6f93888430acfe7cdf764b3`
- Published bundle: `research/adaptive-vision-v1/`
- Public repository files after the upload: 62 total; 49 under the bundle
- Bundle `SHA256SUMS` SHA-256:
  `0e35e8146954d7b1cd280e2eb95aeb0ba24803bdbeb82480a93a644104c24107`

The upload changed the model card and added the experimental checkpoint,
optimizer, compatibility tuple, current source, exact-run source/environment,
failed-attempt logs, results and final independent review. It did not include
the root checkpoint, tokenizer, evaluations or logs as upload inputs.

Anonymous downloads at the exact commit matched:

- experimental `model.mpk` (13,250,031 bytes):
  `95a131de80436d5ba3dd8b3c1cb3fa6a87cd446389e272cec8d8fbb9406c2dec`
- experimental `optim.mpk` (26,499,107 bytes):
  `4a31ec01182f69e7c43425a72555084c39a68873ac59e1043c474b8611a40e4c`
- experimental `meta.json` (713 bytes):
  `7fb4ec488b0616ab389aeb1ffbc2e29dc133b7f9e55fa824603c945f4f042562`

The public API and an anonymous tokenizer download also prove the pre-existing
root artifacts are unchanged:

- `model.mpk`: `15e77802421cac37bfd0e9e66624fdbc4e332943b9c711a759d99b2cca2e020d`
- `tokenizer.json`: `ce4e96b9eee682a95be8947c46cf0afbb5545fd1151525019413302edd2ffb0b`

The card explicitly says the root checkpoint is text-only. The companion
checkpoint is labelled a 128-case English clean-line training-overfit result;
general OCR remains red.

## Review and release gate

The full implementation packet had SHA-256
`af943765442c18ef1578d663b22be05eb90f80858dbed2b55cb5763ee6008d93`.
Kimi K3 correctly blocked publication because sentinel difficulty still
followed mutable frontier state. The amended controller pins sentinel
difficulty at family registration, includes a frontier-interleaving regression
test, makes shadow precursors priority-neutral, validates pass/catastrophic
count disjointness and lowers the hard-outcome stop default to 5%.

Both exact reviewers independently passed the byte-identical amendment packet
with SHA-256
`bccce588b032129e684dfa0f73739f658f2989b7a04582fb73f7192f65b024bd`:

- `z-ai/glm-5.3`: `VERDICT: PASS`;
- `moonshotai/kimi-k3`: `VERDICT: PASS`.

The final local gate was 170 passing `grug-data` unit tests, 41 integration
tests and 4 doctests; 60 passing `grug-train` unit tests, 2 integration tests
and 2 compact-schedule tests; strict all-target clippy passed for both crates.
Five explicitly documented manual/performance probes remained ignored.

## Credential handling

The Hugging Face token was read from
`op://Homelab/Hugging Face Access Token/credential` through the machine-local
1Password service-account helper, held only in the upload process environment,
and never printed, committed or embedded in a remote URL.

All post-publication checks were deliberately anonymous. Both API records
reported `private: false` at the exact heads above.
