# Exact-review follow-up: reviewer-led provenance and corpus hardening

**Date:** 2026-09-04  
**Requested models:** `z-ai/glm-5.3`, `moonshotai/kimi-k3`  
**Decision requested:** independent `PASS` or `BLOCK` on this exact follow-up

## Parent boundary

The first amendment request is
`docs/ai/reviews/2026-09-04-tool-native-learning-v2-amendment-request.md`,
SHA-256
`bc0ccf220545b4572dee40924ed77fa390ca23c537bd9e2d73445fd8b1ef74a4`.
Both exact requested models returned `VERDICT: PASS` on that boundary. This
follow-up voluntarily closes their remaining actionable non-blocking
observations. Their earlier PASS responses do not cover these new bytes.

The source-head v2 contract is still a future adaptation input. No running
supervisor, paid-run corpus, checkpoint, active model alias or Hugging Face
artifact was changed.

## Follow-up changes

1. Plain-text `archival_search` results are split into one KG-prefix scope and
   one scope per renderer-authenticated `### ... (note:note-<32 hex>)` block.
   Source, locator and quote can no longer combine across returned notes.
2. Structured results inspect every recognized envelope key independently.
   A later valid envelope is accepted, but values from different envelopes or
   sibling hits are never flattened together.
3. Generic tool-curriculum `study_note_read` context now carries the full live
   receipt shape: timestamp, document metadata, frontmatter, body, wikilinks
   and tasks as well as note ID and revision.
4. Revision curriculum prose says `revise`/`revision`, not `append`, while the
   create tool retains append terminology.
5. The note-conflict diagnostic now directs callers to `study_note_read`
   instead of the compatibility-only combined catalog.

## Exact changed files and SHA-256

```text
176faa157c0ecd8dc3dba46af746b135c6851e0ea6cc3a1c1bc0a631bd27d592  src/library/error.rs
f4452ab35027a9a73641e972c6cfddc9634eaa893e2aa45184f8eb7ad7dbb1f0  src/agent/agentic_chat.rs
a68ceffcef6408aa120fd425c5e5808c38f2aa3af3860f915910a68a49ce178a  crates/grug-data/src/knowledge_workflow.rs
edce76781d8afd991772c75c678d10ca7adb782827bf89917653a497be60e9d0  crates/grug-data/src/tool_curriculum.rs
b84280c55cca91557fbadf0f9b9467edef580743cd03904e5f097c13b98d03f5  training/results/tool-native-full-model-pipeline-20260904-RESULT.md
```

All parent files absent from this list remain byte-identical to the hashes in
the first amendment or its initial request.

## Regenerated exact boundary

- exported contract SHA-256:
  `a019d30e2a597bc6b8395a28f3601eb8769803bf1fb0946224b26c04a0133c47`;
- 53 tools and unchanged contract hash
  `blake3:3d6e18bb74765cab13d35901d0a10a7834077f302cc578941ca06ef56223700a`;
- knowledge workflow: 21/21 episodes, 92/92 SFT records, seven families,
  EN/DE/FR, 14 stages and twelve tools;
- knowledge train episode/SFT hashes:
  `blake3:b377ffc7a3f256ba3e00cdaf9fe8ca8daaa70ac2cc73d4d162e27ef8bb878133`,
  `blake3:c404fa441d06a3a0a0afb6f7d3de74397c6553c026b3cc289cf8126830e7493f`;
- knowledge eval episode/SFT hashes:
  `blake3:2f9d173091f20ef953d7701646c6d9a9002c5b21ca2b686e2f9d78723aa26550`,
  `blake3:a5ec20cddfc6ba7f25b42a9e61b824227ab20ef76c49d1bdc6938bc8b46c7f8c`;
- all-tool curriculum: 212/106 records, 53 families, exactly four offered
  tools per record and zero exact prompt-target overlap;
- all-tool train/eval hashes:
  `blake3:eef2fa1308b4c9375e567dc0ee654ad77cdb7f5c7d1b7164cb2f9c66807426d4`,
  `blake3:e58109e4a84623c33820f8a9f9c8c341acc169fc5958a889767e1debf4c112e6`;
- knowledge/tool manifest SHA-256 values:
  `a7337b3b57161cdb3a4486bacb33eaf6cbf4cfdc2048adf69a6e34d67a984025`,
  `cfb6d4419282b91aa3da13888b33b5a5ddc22a0bb4f47eb483a9d49996cb9729`.

## Validation rerun

- `cargo test --lib evidence`: 37 passed, including cross-plain-note and
  multi-envelope regressions;
- `cargo test --lib study_note`: 11 passed;
- `cargo test --lib note_store`: 11 passed;
- `cargo test -p grug-data knowledge_workflow`: 13 passed;
- `cargo test -p grug-data tool_curriculum`: 13 passed, including the
  full-shaped read receipt;
- `cargo test -p grug-data tool_contract`: 8 passed;
- `cargo test -p grug-train --lib knowledge`: 11 passed;
- default-small-model daemon registration regression: 1 passed;
- `cargo check --features server,grug-backend --bin akhomed`: passed;
- `cargo check -p grug-data --examples`: passed;
- `cargo fmt --all -- --check` and `git diff --check`: passed.

## Verdict contract

Begin with exactly `VERDICT: PASS` or `VERDICT: BLOCK`. A blocker must identify
a concrete correctness, security, provenance, schema/runtime or corpus defect
in these follow-up bytes or their interaction with the parent packet. Confirm
whether the five changes close the cited observations without weakening the
earlier fail-closed guarantees. List any remaining non-blockers separately.
