# Exact-review request: source-head v2 tool-native learning slice

**Date:** 2026-09-04  
**Requested models:** `z-ai/glm-5.3`, `moonshotai/kimi-k3`  
**Decision requested:** independent `PASS` or `BLOCK` on this exact source slice

## Scope

Review the implementation that lets the small Akh-Medu model discover, read,
create and conflict-safely revise source-backed study notes while learning from
the exact native/MCP tool contract. Focus on correctness, security, provenance,
privacy, optimistic concurrency, JSON-Schema/runtime agreement, synthetic-data
leakage, and training/runtime mismatch.

The model-facing note surface is deliberately four single-purpose tools:

- `study_note_list`: body-free, quote-redacted metadata filtering/pagination;
- `study_note_read`: one stable `^note-[0-9a-f]{32}$` identity and current head;
- `study_note_append`: create with mandatory source ID, locator and quote;
- `study_note_revise`: update with the current `^[0-9a-f]{64}$` revision token.

The host refuses durable note writes unless every citation matches source ID,
locator and quote in one earlier successful retrieval result. Same-round tool
calls do not authorize a write, and repaired retries are checked again. The note
store uses append-only immutable revisions and a lock-scoped compare-and-swap;
stale differing writes fail without advancing history and exact retries are
idempotent.

The generic all-tool generator now enforces `pattern` and `maxLength`, samples
valid note IDs/revision tokens, and gives note-create targets matching earlier
search output. Revision targets receive both an earlier `study_note_read` result
with the expected head token and matching search output. Prerequisites reserve
slots inside, rather than in addition to, the four-tool budget.

## Exact generated boundary

- exported tools: 53;
- contract hash:
  `blake3:3d6e18bb74765cab13d35901d0a10a7834077f302cc578941ca06ef56223700a`;
- knowledge workflow: 21/21 episodes, 92/92 SFT records, seven sentinel
  families, EN/DE/FR, 14 stages, twelve learning-loop tools;
- knowledge train episode/SFT hashes:
  `blake3:c4505191668e503a246d7b52305c2a54ad13d4b8d46b63be25ea75cb12448882`,
  `blake3:0dda77833a7fbb2918d9f471f2b942e93fa4580a7da1d2857b889c36431221d0`;
- knowledge eval episode/SFT hashes:
  `blake3:3a0278b9ec203d8888f856fe1ad1e1ee3069a815985a41fe59765775325945a7`,
  `blake3:ec2e1a7b2bed7a1d34e89bbe7b97b58dd80bff5e2a8a375a58a3c052a0a47057`;
- all-tool curriculum: 212 train / 106 eval, exactly four offered tools per
  record, zero exact prompt-target overlap;
- all-tool train/eval hashes:
  `blake3:f2129aed99a05c3dcd7ddb65b998ffa3044a68551c541e72c6664f87e7b07202`,
  `blake3:0120951da23893a87d49ed25faf64640e7b7113430993157e96d527b53bf1a82`.

The running paid v1 job remains deliberately unchanged on its 50-tool,
five-family contract. This source slice is a future contract-delta adaptation,
not evidence that the running weights know these tools.

## Files and SHA-256

```text
c8323cea47da3841cb091a7bdf6c3a8fd3dd3329e22566d6b1ac17a80094f95a  src/library/error.rs
77f166f673301e63e523ed4dc0544b00df884486ddc13e1bb4d02e4403bdda4b  src/library/note_store.rs
7bbffd69e89272cab02e2472424c5963f813b916068f3ab9b3d772dabeddb049  src/agent/tools/study_note.rs
1c0b96a22692b73c779dc5f53c006ba1ba6313790e5501951732b1fe0f447f63  src/agent/agentic_chat.rs
96750b892f16feec35ca281f1aec5cc0e85432904b5ec2daa1d58950563e066c  src/bin/akhomed.rs
2fc91b88cc7cc10b6f2936cae84797ff1f5e55ac639bd46b8315dc9cd45aeb46  examples/export_akh_learning_contracts.rs
d6c807ec1bb9953b93c1a3e9e28c4fb509a334f1fe0a4469235d5a1aa02bb851  crates/grug-data/Cargo.toml
5883fcccd1ad9150495527d804afe39a07dfddfb7f7184e4bd14401e14131f69  crates/grug-data/src/knowledge_workflow.rs
908ffb5113ef0d025325c23abb05a0eb00707302ddce4f2e4290e72df7d689ec  crates/grug-data/src/tool_contract.rs
494866b7074126f2403e025f34c15d368ed5d09d80f1087bd8f6ce3404ebf5b7  crates/grug-data/src/tool_curriculum.rs
8599e2f64321d66012cb0acc6934d54eaa5db26171b6769242bd81a62bb21f85  docs/ai/decisions/102-tool-native-learning-model.md
d7191c986842be8107347a5b3419dce0ed03b2091a3fba59f7b8a66b5e25214c  docs/ai/plans/2026-09-03-phase51-adaptive-exams-native-multimodal.md
65dec6d6b42593e0c4bf7c922906cfc281518b63baad0223fd1d21340fcd0b48  docs/ai/manuals/2026-09-04-adaptive-exams-native-vision.md
d616ca46ad06cc88d54c24792313777093495064f3b1fa8d7810070579549842  training/results/tool-native-full-model-pipeline-20260904-RESULT.md
```

## Validation already run

- `cargo test --lib study_note`: 10 passed;
- `cargo test --lib note_store`: 11 passed;
- `cargo test -p grug-data knowledge_workflow`: 12 passed;
- `cargo test -p grug-data tool_contract`: 8 passed;
- `cargo test -p grug-data tool_curriculum`: 13 passed;
- `cargo test -p grug-train --lib knowledge`: 11 passed;
- daemon default-small-model tool-registration regression: passed;
- `cargo check --features server,grug-backend --bin akhomed`: passed;
- `cargo check -p grug-data --examples`: passed;
- targeted rustfmt check and `git diff --check`: passed.

## Verdict contract

Begin with exactly `VERDICT: PASS` or `VERDICT: BLOCK`. A blocker must identify
a concrete defect that makes this source slice unsafe, internally inconsistent,
or unable to enter the later adaptation/certification path. Do not call the
absence of completed training a source blocker: untrained, uncertified and
inactive status is explicitly preserved. List non-blocking improvements
separately and cite file/line or function-level evidence.
