# Native temporal cache repair

Cached fluent state and its last initiation/termination witnesses now derive from timestamped history, matching `holds_at`. Late grants and revocations cannot overwrite newer evidence merely by arriving last. Equal-time termination wins truth-value ties; equal-time witnesses of the same kind use a stable symbol-ID tie break. All history receipts remain intact.

Restoration rebuilds cached fields from history, repairing old serialized caches without a schema change. Current means latest recorded event state, not a wall-clock scheduler; future-dated event admission remains outside this change.

Validation: 19 native event-calculus tests passed, including the new late-event, witness, holding-fluent, retained-history, equal-time and persisted legacy-cache regression. The regression uses an isolated temporary on-disk engine. The first attempt used the nondurable default test engine and failed its restore assertion (18 passed, 1 failed); that test setup error and frozen source remain in `runs/akh-r1-native-temporal-cache-20260913-v1`. Corrected evidence is in sibling `akh-r1-native-temporal-cache-20260913-v2` in the main checkout. This tests actual native metadata persistence/restoration, not process crash or power-loss durability.

The hypothesis now supported is that native temporal queries and their persisted cache can agree under delayed evidence. This is symbolic implementation evidence, not learned reasoning or a new independent world. No model, training, download, cloud allocation, certification or release change occurred. Prior executable experiment artifacts remain untouched.

Next bounded action: connect the verified temporal prerequisite to the existing public incident authority guard, using observed source receipts and a hidden revocation followed by correction across cold restart. Reuse the incident fixture and independently inspect actual effects; compare stale-cache misuse as a negative control. No new component inventory or cosmetic world-count credit.
