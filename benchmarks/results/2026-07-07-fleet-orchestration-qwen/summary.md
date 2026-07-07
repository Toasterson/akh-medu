# fleet-orchestration × qwen — 2026-07-07 14:31 UTC

- **Harness git SHA**: `4bec9e105c8a7c096dc357f2678532d29c4cbef5`
- **Task-set hash**: `b6afa9f4d7a58d898759c8927b8cd2e19365590dcbe688af773c3588fc091541`
- **Model backing**: cloud (`qwen3.5:397b`)
- **Grading**: autonomous (deterministic `verify.sh` only, no LLM judge)
- **Host tier**: VM (`code.aopc.cloud/CloudNebulaProject`, recipe `c93aaef5952dd28b0c54127ae5ec77abc298d9fc894ab39ef6b27ffd2cd5523c`)

**2/2 tasks passed (100.0%), mean score 1.000, 187.7s total wall time**, 962030 tokens, 47 requests

| Task | Outcome | Score | Wall time (s) | Tokens | Requests | Notes |
|---|---|---|---|---|---|---|
| relay-group-3svc-control | pass | 1.000 | 111.8 | 445370 | 15 | verify: PASS: relay-a, relay-b, relay-c all online, relay-c app/greeting=relay-c-ready |
| relay-group-3svc-orchestrated | pass | 1.000 | 75.9 | 516660 | 32 | speedup=1.47 (control 111.8s / orchestrated 75.9s) | coordination tax: decompose 7469 tok / 1 req | decomposition: used orchestrator's own JSON subtask split | redundant commands: 0/28 | verify: PASS: relay-a, relay-b, relay-c all online, relay-c app/greeting=relay-c-ready |
