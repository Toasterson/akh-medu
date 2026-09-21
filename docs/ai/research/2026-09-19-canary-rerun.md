# Five-daemon canary rerun on the current checkpoint

Source: `028804f1`. All five daemons were rebuilt from committed sources before
the runs. Local private stack only: no shared service, production database or
remote host was involved, and no production daemon is installed. Both runs
passed and left no processes behind.

## Result

| Check | HIP | Vulkan |
| --- | --- | --- |
| real core context | pass | pass |
| real daemon dispatch | pass | pass |
| native spans published | pass | pass |
| core/session restart | pass | pass |
| provider continuity | pass | pass |
| history eviction | pass | pass |
| session isolation | pass | pass |

Both runs stored five spans across five turns, dropped four messages from
history, materialized at most one span, and answered the post-restart question
with the correct number `4317` after four notes had been evicted. The four note
turns replied exactly `OK`, so recalled notes were treated as data rather than
instructions.

## Identity

Rebuilt with existing warnings, then run without further compilation:

```sh
CARGO_BUILD_JOBS=2 cargo build --locked --offline --release -p akh-medu \
  --no-default-features --features core-server --bin akhomed
CARGO_BUILD_JOBS=2 cargo build --locked --offline --release -p akh-session-services \
  --no-default-features --features session,executor,sandbox,hip,vulkan,core-integration-tests \
  --bin akhsessd --bin akhrtd --bin akhsbxd --bin akhexecd
python3 scripts/session-platform/native-core-smoke.py --device <hip|vulkan> \
  --model <pinned Q4_K_M blob> --tokenizer <pinned tokenizer> --output <jsonl>
```

Identical binaries served both device runs:

| Binary | SHA-256 |
| --- | --- |
| akhomed | `03870e4f2a3475e948fc0d6ef6436784db14db39c8895fd1cfd074d292cc565b` |
| akhsessd | `07d62e6a7763b45e0f5522dcb131cece09047fc06a080fea0b3ec59eeda90484` |
| akhrtd | `3d7225429e2d02830f793c65600d7be386054f858749178cf37a3d0bdcfeaf10` |
| akhsbxd | `2ad22c6da23b45f4f8ef0ee5cbde130820cc44709e05bbea96b00bdc06915fd8` |
| akhexecd | `82b540e495bddee2be0f9da741f3dfd0ee07a9ad72e2c2af59a9ce4e3a62d1ff` |

Model, tokenizer, policy and projector digests match the pinned artifacts
recorded in each run's identity line. Context window 1024 tokens, 16 output
tokens per turn.

Transcripts: [HIP](../../../experiments/native-recall/canary-20260919/native-core-hip-20260919.jsonl),
[Vulkan](../../../experiments/native-recall/canary-20260919/native-core-vulkan-20260919.jsonl).

## Not established

This is a private local acceptance step. It does not validate PostgreSQL-backed
deployment, a production persona, general instruction adherence, throughput,
long-context KV compression, vision, or any shared-service cutover.
