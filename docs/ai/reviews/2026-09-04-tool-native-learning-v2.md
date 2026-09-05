# Exact review: source-head v2 tool-native learning

**Date:** 2026-09-04  
**Final source verdict:** PASS  
**Reviewers:** exact `openrouter/z-ai/glm-5.3` and
`openrouter/moonshotai/kimi-k3`

## Boundary and outcome

Both requested models independently returned `VERDICT: PASS` for the final
reviewer-led hardening packet. They found no correctness, security,
provenance, privacy, schema/runtime or corpus blocker. The verdict applies to
the source and generated identities recorded below. It does **not** certify a
trained checkpoint, activate a model, complete the running pipeline or
authorize publication.

Reviews were invoked directly against the configured exact model IDs. Agent
Board was not used, no other agent identity was used, and reviewers made no
repository edits.

## Immutable review chain

1. Initial request:
   `2026-09-04-tool-native-learning-v2-review-request.md`, SHA-256
   `b7a8dc738d8058d317594ee6d57285b181f3bfaa07dc9d0b00af2ebed429e474`.
   Both exact models returned `VERDICT: PASS`. Their non-blocking observations
   led to stricter create/revise separation, schema-aligned citation parsing,
   prerequisite retention, per-hit evidence scoping and live-shaped workflow
   mocks.
2. First amendment:
   `2026-09-04-tool-native-learning-v2-amendment-request.md`, SHA-256
   `bc0ccf220545b4572dee40924ed77fa390ca23c537bd9e2d73445fd8b1ef74a4`.
   Both exact models returned `VERDICT: PASS`. GLM verified all eight
   dispositions and the complete generated boundary. Kimi independently
   checked every note-write record and identified a residual cross-note seam
   in plain-text archival evidence plus minor curriculum fidelity points.
3. Reviewer-led hardening:
   `2026-09-04-tool-native-learning-v2-reviewer-hardening-request.md`, SHA-256
   `20b696e316ff8c99030c0c3543de5ddcb8e973eb7df5b09a8e10edb0813363bd`.
   Workspace-only retries of exact GLM 5.3 and Kimi K3 both returned
   `VERDICT: PASS` after verifying the changes at function level. Earlier
   attempts that terminated at a read-only `/tmp` permission boundary without
   rendering a verdict were discarded and are not counted as reviews.

## Final reviewed source hashes

```text
176faa157c0ecd8dc3dba46af746b135c6851e0ea6cc3a1c1bc0a631bd27d592  src/library/error.rs
77f166f673301e63e523ed4dc0544b00df884486ddc13e1bb4d02e4403bdda4b  src/library/note_store.rs
b9ce39a96fab4614fd477ecbb1ce887223ae99ad4271b6c29b7a9c9fa383601c  src/agent/tools/study_note.rs
f4452ab35027a9a73641e972c6cfddc9634eaa893e2aa45184f8eb7ad7dbb1f0  src/agent/agentic_chat.rs
96750b892f16feec35ca281f1aec5cc0e85432904b5ec2daa1d58950563e066c  src/bin/akhomed.rs
2fc91b88cc7cc10b6f2936cae84797ff1f5e55ac639bd46b8315dc9cd45aeb46  examples/export_akh_learning_contracts.rs
d6c807ec1bb9953b93c1a3e9e28c4fb509a334f1fe0a4469235d5a1aa02bb851  crates/grug-data/Cargo.toml
a68ceffcef6408aa120fd425c5e5808c38f2aa3af3860f915910a68a49ce178a  crates/grug-data/src/knowledge_workflow.rs
908ffb5113ef0d025325c23abb05a0eb00707302ddce4f2e4290e72df7d689ec  crates/grug-data/src/tool_contract.rs
edce76781d8afd991772c75c678d10ca7adb782827bf89917653a497be60e9d0  crates/grug-data/src/tool_curriculum.rs
8599e2f64321d66012cb0acc6934d54eaa5db26171b6769242bd81a62bb21f85  docs/ai/decisions/102-tool-native-learning-model.md
6ae26dedfd76f85e373d2d14bfee1b3374918a31dca4dda3dac276fffd4d6be8  docs/ai/plans/2026-09-03-phase51-adaptive-exams-native-multimodal.md
65dec6d6b42593e0c4bf7c922906cfc281518b63baad0223fd1d21340fcd0b48  docs/ai/manuals/2026-09-04-adaptive-exams-native-vision.md
b84280c55cca91557fbadf0f9b9467edef580743cd03904e5f097c13b98d03f5  training/results/tool-native-full-model-pipeline-20260904-RESULT.md
```

## Final generated boundary

- contract export SHA-256:
  `a019d30e2a597bc6b8395a28f3601eb8769803bf1fb0946224b26c04a0133c47`;
- contract: 53 tools, with only four model-facing note tools, hash
  `blake3:3d6e18bb74765cab13d35901d0a10a7834077f302cc578941ca06ef56223700a`;
- knowledge manifest SHA-256:
  `a7337b3b57161cdb3a4486bacb33eaf6cbf4cfdc2048adf69a6e34d67a984025`;
- knowledge train episode/SFT:
  `blake3:b377ffc7a3f256ba3e00cdaf9fe8ca8daaa70ac2cc73d4d162e27ef8bb878133`,
  `blake3:c404fa441d06a3a0a0afb6f7d3de74397c6553c026b3cc289cf8126830e7493f`;
- knowledge eval episode/SFT:
  `blake3:2f9d173091f20ef953d7701646c6d9a9002c5b21ca2b686e2f9d78723aa26550`,
  `blake3:a5ec20cddfc6ba7f25b42a9e61b824227ab20ef76c49d1bdc6938bc8b46c7f8c`;
- tool manifest SHA-256:
  `cfb6d4419282b91aa3da13888b33b5a5ddc22a0bb4f47eb483a9d49996cb9729`;
- all-tool train/eval:
  `blake3:eef2fa1308b4c9375e567dc0ee654ad77cdb7f5c7d1b7164cb2f9c66807426d4`,
  `blake3:e58109e4a84623c33820f8a9f9c8c341acc169fc5958a889767e1debf4c112e6`.

The knowledge smoke has 21 train and 21 evaluation episodes, 92 SFT records
per split, seven families, three languages, all 14 stages and all twelve
learning-loop tools. The all-tool smoke has 212/106 records over all 53 tools,
exactly four offered tools per record and zero exact prompt-target overlap.

## Final reviewer conclusions

GLM 5.3 confirmed that plain-text archival scopes cannot combine provenance
across notes, structured envelopes remain independent, legitimate real
archival results still pass, full read receipts match runtime, revision prose
is corrected, and the same-round/whitelist/identity gates remain fail-closed.

Kimi K3 independently confirmed those properties, every parent hash, the live
serde shapes, and why the knowledge SFT hashes remain stable while episode
hashes changed. Its remaining observations were non-blocking:

- the live revise tool describes an append-only revision with the technically
  correct word "Append";
- generator version labels remain v2 while exact artifact hashes distinguish
  amendments;
- one test name retains "conflict safe append" terminology;
- malformed heading-less archival text is handled conservatively.

These do not change tool identity, authorize unsupported writes, coalesce
evidence, invalidate a schema or blur the lifecycle boundary. They are accepted
for this v2 source boundary; any later cleanup creates a new contract/corpus
identity and requires a new review.

## Lifecycle boundary

The paid v1 pipeline remains frozen at its captured 50-tool/five-family
contract. The v2 source slice is ready for a later contract-delta adaptation,
but no resulting v2 weights have been trained, evaluated, certified, installed,
activated or published. Promotion remains conditional on the implemented
adaptive text/tool, native OCR, cross-host offline and installer gates.
