# Dream-RSI relevance to ADAM and Kaine

Research harvest requested by the user on 2026-09-16. Start with [ANALYSIS.md](ANALYSIS.md), the verbatim assessment, then the [conversation](transcripts/conversation.md) and [complete paper](papers/dream-rsi-2609.14858v1.pdf).

## Contents

- `ANALYSIS.md`: exact original assessment, not an architecture amendment.
- `transcripts/conversation.{json,md}`: exact user-facing thread snapshot, including the harvest request; source and cutoff in `context/transcript-provenance.json`.
- `papers/`: complete versioned PDF, HTML, full text, PDF metadata and [source/license notes](papers/SOURCES.md).
- `sources/`: tweet mirror, authors' project page, repository README, arXiv metadata and retrieval hashes.
- `context/`: working-tree snapshots of ADR 110, ADR 111 and the P7 plan, plus the repository HEAD. These are design context, not live implementation or tracker certification.
- `seshat/`: import bodies, receipts, document metadata and hash verification.
- `SHA256SUMS`: integrity manifest for every other archived file.

## Research decision and open questions

Prioritize a bounded replay-controller experiment around a frozen ADAM checkpoint. Compare fixed, hand-written adaptive and replay-improved policies on sandboxed repair tasks. Split by whole task, hide future outcomes, restore isolated branch snapshots, count total policy-development and execution costs, and require fresh online validation. Keep existing tool/knowledge/copy gates and engine authority intact.

For Kaine, investigate the combination of hypothesis formation, learning-progress curiosity and replay-based allocation of exploration. This synthesis is a proposal, not a result established by Dream-RSI. No training, deployment, work-item dispatch or architecture change is part of this harvest.

The 162x headline compares different model-backed systems. The controlled Lasso comparison is 550 versus 317 discovery-agent calls; replay improvement does not guarantee future improvement or equivalent total-cost savings. The original transcript preserves the full qualifications.

## Related archives

- [Kaine foresight and compression-progress curiosity](../2026-09-16-kaine-foresight-curiosity/README.md): includes both previously discussed papers and their Seshat IDs.
- [TypeSafe/Jev decisions for ADAM and Kaine](../2026-09-16-typesafe-jev-adam-kaine/README.md).

## Seshat retrieval

Search for `Dream-RSI ADAM Kaine replay exploration`. Record slugs use the prefix `dream-rsi-adam-kaine-2026-09-16-`. See `seshat/verification.json` for content-hash checks and `seshat/document-records.json` for Anima and Seshat identifiers. Anima metadata visibility is private; this does not assert isolation within the shared Seshat archive.
