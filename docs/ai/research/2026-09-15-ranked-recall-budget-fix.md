# Ranked recall within the native prompt budget

The [exact-model gate](2026-09-15-exact-native-core-gate.md) retrieved the
correct fact first on both HIP and Vulkan but omitted it from the final prompt.
Retrying all returned notes after eviction could exceed the window, causing
the whole recall block to be skipped.

## Fix

`native_recall::assemble` owns selection, deduplication, materialization and
window fitting as one decision. It tries the largest ranked prefix first,
then drops the lowest-ranked candidate until a complete prefix fits. It never
splits a note or bypasses an oversized top-ranked note to promote a lower one.
At most four hits are considered. The complete recalled block is bounded to
4096 bytes; the fully rendered prompt is checked against both its exact token
budget and byte budget. The caller reserves output tokens as before.

When adding recall evicts history, the assembler rematerializes all selected
hits instead of relying on their previous visibility. Every selected nonempty
hit must still appear in the final rendered prompt. Deliberately pruned
lower-ranked hits are not pinned. System instructions and the complete current
turn remain protected by the existing whole-turn window selector. If no
prefix fits, the original prompt is kept. Rendering errors propagate.

Recall metadata now lists only the turns actually materialized in the block,
not every hit returned by the core. The resident model, capture path, projection
recipe, quantization and GPU settings are unchanged.

## Verification

The top-fact budget test first failed with no recalled prompt, then passed
after ranked-prefix selection. Four focused tests cover that failure, deduped
evidence re-pinning after eviction, byte overflow/no note splitting, visible
no-op behavior and error propagation. The native integration uses the new
assembler; the former duplicated materialization logic was removed.

```bash
CARGO_BUILD_JOBS=2 cargo test --offline -p akh-session-services --no-default-features --lib native_recall
```

The exact-model fixture, narrow instruction, expected answers, 160-token
window and 16-token output reservation remain unchanged. This fix targets the
recall budget failure, not the separate filler acknowledgment mismatches.
The native wiring still lives in the existing mixed runtime worktree; a
module-only checkpoint is not a complete deployable runtime release.

### Exact-model results

The release library suite with `core-integration-tests,hip,vulkan` passed:
48 passed, zero failed, three artifact-dependent tests ignored. The exact-model
test was then explicitly run serially on HIP and Vulkan, without concurrent
GPU work. Both runs now preserve the Atlas port fact in the final recall block
and answer `4317`. The selected block contains the top three ranked notes;
the lowest-ranked counting note is omitted. All five captures are delivered,
and the final-history recall placement and retention assertions pass.

The complete five-answer gate still fails on both backends: answers are
`OK`, `12`, `UNKNOWN`, `OK`, `4317`, with two acknowledgment mismatches against
the unchanged expected `OK` responses. Both evidence summaries explicitly
record `answer_failures: 2` and `passed: false`. These are not deployment
acceptance passes. Retrieval text, distances and answers agree across the
backends. No shared service was restarted or changed.

Evidence (SHA-256):

- [HIP](../../../experiments/native-recall/ranked-budget-hip-v1.jsonl):
  `138c496b5f0c5c00347c3d209757c979c87a015d9637467db8da186a705153b5`
- [Vulkan](../../../experiments/native-recall/ranked-budget-vulkan-v1.jsonl):
  `73f54e799044ed5bd2a2d236845ab6c73b62b756955cd9042b75ee7c598617d2`
- Tested release binary `akh_session_services-4a42cd0664cf9d14`:
  `d530de17ed212be3c40e61af5d61af771a365570d78e14aa491ca9dabc8e1fbd`
- Tested `native_recall.rs`:
  `a81b7d70d5cd480faba0b6750bf39a1b93a2a61b3910128ff9bce6b82990b3b8`

Next: investigate the remaining note acknowledgment behavior without relaxing
the expected answers, then validate real daemon dispatch before any cutover.
