# Sources and extraction provenance

Harvested 2026-09-16 for the user's ADAM/Kaine research archive.

## Primary paper

Tong Zheng et al., *Dream-RSI: Recursive Self-Improvement through Evolving Worlds*, arXiv:2609.14858v1, submitted 14 September 2026.

- Abstract: https://arxiv.org/abs/2609.14858v1
- PDF: https://arxiv.org/pdf/2609.14858v1
- HTML: https://arxiv.org/html/2609.14858v1
- Local originals: `dream-rsi-2609.14858v1.pdf` and `.html`.
- Full extraction: `dream-rsi-2609.14858v1.txt`, generated with `pdftotext -layout`, without paraphrasing or editorial correction.
- Verified 36 PDF pages and 36 extracted page breaks, including references, prompts and discovered-program listings. First and last pages rendered and visually inspected. Poppler emitted a font-type mismatch warning; the inspected renders were legible. Extraction can alter equations, whitespace, ligatures and figure layout; the original PDF remains authoritative.
- Rights: PDF states © 2026 Google. All rights reserved. arXiv links its perpetual non-exclusive distribution license: http://arxiv.org/licenses/nonexclusive-distrib/1.0/ . This archive does not relicense the paper.

## Supporting source snapshots

- Original post: https://x.com/Dr_Singularity/status/2100204780010222076 . Direct X retrieval returned HTTP 403 during analysis; the post text was recovered through https://api.fxtwitter.com/Dr_Singularity/status/2100204780010222076 . `../sources/tweet-fxtwitter.json` preserves that third-party response, not an authenticated X export.
- Authors' project: https://dream-rsi.com/ ; `../sources/project.html` and derived `project.txt`.
- Authors' repository: https://github.com/zhengkid/Dream-RSI ; `../sources/repository-readme.md` is a snapshot of main, not a pinned code release. At capture, the full codebase and reproduction scripts remained in preparation.
- `../sources/retrieval-manifest.json` records retrieval times, resolved URLs, byte counts and hashes.

## Scope

This thread analyzes one paper, Dream-RSI. The previously archived Oxford foresight and Schmidhuber compression-progress papers remain in `../../2026-09-16-kaine-foresight-curiosity/`; see that archive's README for their separate Seshat records. References appearing in Dream-RSI's bibliography were not separately harvested.

Seshat receives complete extracted paper text; original PDF/HTML binaries remain in this repository. Conversation strings are preserved in JSON and Markdown, including citations, caveats and historical proposals. Private reasoning, system/developer instructions, credentials and raw tool logs are excluded. The transcript snapshot predates final persistence receipts.
