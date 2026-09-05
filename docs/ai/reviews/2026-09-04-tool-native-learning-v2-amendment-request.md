# Exact-review amendment: source-head v2 tool-native learning slice

**Date:** 2026-09-04  
**Requested models:** `z-ai/glm-5.3`, `moonshotai/kimi-k3`  
**Decision requested:** independent `PASS` or `BLOCK` on this exact amendment

## Parent review boundary

The immutable initial request is
`docs/ai/reviews/2026-09-04-tool-native-learning-v2-review-request.md`, SHA-256
`b7a8dc738d8058d317594ee6d57285b181f3bfaa07dc9d0b00af2ebed429e474`.
Both requested exact models returned `VERDICT: PASS` on that packet and raised
only non-blocking hardening opportunities. This request binds the resulting
amendment. Review the changed files and their interaction with the unchanged
parent slice; do not treat the initial PASS as covering these new bytes.

The model remains untrained on this v2 contract, uncertified, inactive and
unpublished. The paid v1 run remains untouched on its original 50-tool,
five-family bytes. Those lifecycle facts are intentional and are not source
defects.

## Findings and dispositions

The amendment incorporates the two reviewers' non-blocking observations:

1. The injected optimistic-concurrency failure now names
   `study_note_revise`, not the create-only append tool.
2. Restart-recall synthetic evidence now has the real `ArchivalSearchTool`
   renderer shape (`Notes`, heading, `Frontmatter`, body) and no invented
   `Citations` section.
3. `study_note_append` rejects `note_id` and `expected_revision` at runtime;
   only the private constructor used by `study_note_revise` enables revision.
4. Citation objects reject unknown fields, matching the exported schema's
   `additionalProperties: false` behavior.
5. Low-difficulty note-write exams reserve their read/search prerequisites
   instead of truncating them; the ordinary corpus still offers exactly four
   tools per record.
6. Citation source IDs and locators use bounded identity matching. Structured
   retrieval responses preserve per-hit provenance boundaries, including
   nested `data.results` envelopes, so fields from sibling hits cannot combine
   to authorize a write.
7. The generic curriculum fails closed when its captured contract lacks the
   evidence-search or note-read prerequisite needed by a generated write.
8. Module documentation makes the runtime boundary explicit: model-originated
   note calls go through `Agent::execute_tool_call`, which applies the
   prior-round evidence gate; direct executor use is reserved for trusted host
   code and tests.

Exact stale retries remain intentionally idempotent. A stale request with
different content remains a conflict and cannot advance history.

## Exact amended files and SHA-256

```text
b9ce39a96fab4614fd477ecbb1ce887223ae99ad4271b6c29b7a9c9fa383601c  src/agent/tools/study_note.rs
c459f214d5697c6eff6c4559039921e1e881b15e48bc901837cc9c8f84759bd0  src/agent/agentic_chat.rs
0470096a000c9fba04cd855c556b9d6c97cd9219ceee918307c74a84b3cfd7a6  crates/grug-data/src/knowledge_workflow.rs
c6bc880e492f330fbd733562804568b840c8fc7964bda7505c3bd7daf80f8239  crates/grug-data/src/tool_curriculum.rs
6ae26dedfd76f85e373d2d14bfee1b3374918a31dca4dda3dac276fffd4d6be8  docs/ai/plans/2026-09-03-phase51-adaptive-exams-native-multimodal.md
dc92aae6022e4df14aa97f3690dd8f6c53df345b7009f286d9978fdc4c83163c  training/results/tool-native-full-model-pipeline-20260904-RESULT.md
```

Files pinned by the parent request and absent from this list are unchanged.

## Regenerated exact boundary

- exported contract file SHA-256:
  `a019d30e2a597bc6b8395a28f3601eb8769803bf1fb0946224b26c04a0133c47`;
- 53 tools; contract hash
  `blake3:3d6e18bb74765cab13d35901d0a10a7834077f302cc578941ca06ef56223700a`;
- only `study_note_list`, `study_note_read`, `study_note_append` and
  `study_note_revise` are model-facing note tools;
- knowledge workflow: 21/21 episodes, 92/92 SFT records, seven immutable
  sentinel families, EN/DE/FR, 14 stages and twelve learning-loop tools;
- knowledge train episode/SFT hashes:
  `blake3:92d8bb4f68dbc05f8c8557f3afb33e8f053070b02b2359b581c4902c9d462454`,
  `blake3:c404fa441d06a3a0a0afb6f7d3de74397c6553c026b3cc289cf8126830e7493f`;
- knowledge eval episode/SFT hashes:
  `blake3:302a30f7e4ceacbfbc8fad1a3be5a8372bcbd3eb8ba3a9c62c8993a6381a3ce7`,
  `blake3:a5ec20cddfc6ba7f25b42a9e61b824227ab20ef76c49d1bdc6938bc8b46c7f8c`;
- all-tool curriculum: 212/106 records, exactly four offered tools in every
  record and zero exact prompt-target overlap;
- all-tool train/eval hashes:
  `blake3:f2129aed99a05c3dcd7ddb65b998ffa3044a68551c541e72c6664f87e7b07202`,
  `blake3:0120951da23893a87d49ed25faf64640e7b7113430993157e96d527b53bf1a82`.

## Validation rerun after the amendment

- `cargo test --lib evidence`: 37 passed;
- `cargo test --lib study_note`: 11 passed;
- `cargo test --lib note_store`: 11 passed;
- `cargo test -p grug-data knowledge_workflow`: 13 passed;
- `cargo test -p grug-data tool_curriculum`: 13 passed;
- `cargo test -p grug-data tool_contract`: 8 passed;
- `cargo test -p grug-train --lib knowledge`: 11 passed;
- default-small-model daemon registration regression: 1 passed;
- `cargo check --features server,grug-backend --bin akhomed`: passed;
- `cargo check -p grug-data --examples`: passed;
- `cargo fmt --all -- --check` and `git diff --check`: passed.

## Verdict contract

Begin with exactly `VERDICT: PASS` or `VERDICT: BLOCK`. A blocker must identify
a concrete correctness, security, provenance, privacy, schema/runtime,
synthetic-data or training/runtime defect in the amended source slice. List
non-blocking observations separately and cite function-level evidence. Confirm
whether every initial observation above is adequately closed or explicitly
accepted by design.
