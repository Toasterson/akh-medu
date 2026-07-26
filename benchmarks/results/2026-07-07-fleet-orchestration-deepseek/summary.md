# fleet-orchestration × deepseek — 2026-07-07 14:27 UTC

- **Harness git SHA**: `4bec9e105c8a7c096dc357f2678532d29c4cbef5`
- **Task-set hash**: `b6afa9f4d7a58d898759c8927b8cd2e19365590dcbe688af773c3588fc091541`
- **Model backing**: cloud (`deepseek-v4-pro`)
- **Grading**: autonomous (deterministic `verify.sh` only, no LLM judge)
- **Host tier**: VM (`code.aopc.cloud/CloudNebulaProject`, recipe `c93aaef5952dd28b0c54127ae5ec77abc298d9fc894ab39ef6b27ffd2cd5523c`)

**2/2 tasks passed (100.0%), mean score 1.000, 89.3s total wall time**, 543646 tokens, 32 requests

| Task | Outcome | Score | Wall time (s) | Tokens | Requests | Notes |
|---|---|---|---|---|---|---|
| relay-group-3svc-control | pass | 1.000 | 55.3 | 202993 | 8 | verify: PASS: relay-a, relay-b, relay-c all online, relay-c app/greeting=relay-c-ready |
| relay-group-3svc-orchestrated | pass | 1.000 | 34.0 | 340653 | 24 | speedup=1.63 (control 55.3s / orchestrated 34.0s) | coordination tax: decompose 5677 tok / 1 req | decomposition: used orchestrator's own JSON subtask split | redundant commands: 0/36 | verify: PASS: relay-a, relay-b, relay-c all online, relay-c app/greeting=relay-c-ready |
