# Kaine foresight and curiosity: Fable archive

Start with [HANDOFF.md](HANDOFF.md). Both conversation transcripts and both complete paper texts are preserved, with provenance and content-hash verification.

```text
2026-09-16-kaine-foresight-curiosity/
  README.md                  # this index and Seshat identifiers
  HANDOFF.md                 # synthesis, historical proposals, caveats, review questions
  transcripts/               # exact user-facing Claude and Codex text, Markdown + JSON
  papers/                    # complete paper text, available source PDFs, licenses, line map
  context/                   # ADR 005/110/111, P7 plan, thread metadata, retrieval evidence
  seshat/                    # import payloads, receipts, IDs, verification
  scripts/harvest.py          # read-only transcript harvesting
  SHA256SUMS                 # checksums for every other archived file
```

## Retrieval for Fable

Search Seshat for `Kaine foresight and compression-progress curiosity Fable handoff`, or use the exact slugs and IDs below. All slugs start with `kaine-foresight-curiosity-2026-09-16-`. Anima document metadata is marked private; Seshat itself is the existing shared archive, so this does not assert per-document isolation in Seshat.

| Record | Anima document ID | Seshat document ID | Slug suffix |
|---|---|---|---|
| Kaine foresight and compression-progress curiosity — Fable handoff | `01a0aba4-dc34-7592-9b58-c7d3058ffc4f` | `590fbd8f-9f95-4ab7-9303-bfb29ce80aa3` | `handoff` |
| Kaine Theory-Based Foresight — Claude t3code transcript 7e5cfbbd | `01a0aba5-093b-7d30-a779-129252421e06` | `900a7e00-3cfa-4667-acaa-7febed37f776` | `claude-transcript` |
| Kaine Oxford and Schmidhuber — Codex conversation 2026-09-16 | `01a0aba5-09cc-7081-a0c2-ca1c512a8bdd` | `f7805771-9982-46f9-bfe0-d99e1284983c` | `codex-transcript` |
| Theory Is All You Need — Felin and Holweg 2024 — complete published paper text | `01a0aba5-0a9c-74a2-bb2c-a22038ab62f9` | `2386f817-b0ef-43a5-89ed-c68bef4d517b` | `oxford-paper` |
| Driven by Compression Progress — Schmidhuber 2008/2009 — complete paper text | `01a0aba5-0b51-7fc0-82c2-0a1a8c913390` | `f019342e-ffcb-4988-8388-02348a06c9fe` | `compression-paper` |
| Kaine foresight curiosity — paper sources licenses and extraction provenance | `01a0aba5-0bd6-7690-abd5-87a4cbb9f661` | `89b402fc-a687-487d-9da0-4e21b758456f` | `sources` |
| Kaine foresight curiosity — ADR 110 ADR 111 and P7 context snapshot | `01a0aba5-0c75-79f0-af07-266341a2f62d` | `2f87a6b7-97cc-4325-85b8-558cd06b4fc3` | `architecture` |
| Kaine Theory-Based Foresight — Claude source retrieval evidence | `01a0aba5-0cfe-7f91-9080-10ba79c8af3f` | `38b9e019-70cb-49aa-92c9-db84d0a7b35a` | `claude-evidence` |

## Additional context discovered during verification

Seshat already contained ADR 005, *Bootstrapping Research*, document `b8543a3b-7034-4390-bbd8-535f632b4b63`. Its section 2 already cites this exact Schmidhuber paper and proposes a compression-progress-inspired curiosity bonus. A current working-tree copy is in `context/005-bootstrapping-research.md`. Fable should reconcile that earlier design with the Kaine proposal. KG coverage change is a proposed proxy, not automatically the same quantity as compressor learning progress.

## Source fidelity and limits

The Oxford **published edition's full 26-page text** was recovered through the web tool; its local binary PDF download was blocked. A separately labelled August preprint PDF is included locally. Schmidhuber's licensed Dagstuhl PDF contains the arXiv v2 title-page marking and all 35 pages. Paper text is preserved without paraphrase, but extraction artifacts and mathematical-layout limitations remain; see `papers/SOURCES.md`.

The Codex transcript is a snapshot through the progress message immediately before ingestion; it cannot include this archive's later verification receipt. Both transcripts retain prior mistakes and corrections as historical evidence. The handoff labels proposals and questions separately.

The eight primary Seshat ingestion receipts have SHA-256 hashes equal to the exact submitted local bodies, with 333 chunks total. No implementation, training, queue dispatch, or t3code thread-state changes were made.
