# fleet-orchestration × glm-5.2 — 2026-07-07 14:25 UTC

- **Harness git SHA**: `4bec9e105c8a7c096dc357f2678532d29c4cbef5`
- **Task-set hash**: `b6afa9f4d7a58d898759c8927b8cd2e19365590dcbe688af773c3588fc091541`
- **Model backing**: cloud (`glm-5.2`)
- **Grading**: autonomous (deterministic `verify.sh` only, no LLM judge)
- **Host tier**: VM (`code.aopc.cloud/CloudNebulaProject`, recipe `c93aaef5952dd28b0c54127ae5ec77abc298d9fc894ab39ef6b27ffd2cd5523c`)

**2/2 tasks passed (100.0%), mean score 1.000, 216.2s total wall time**, 656647 tokens, 44 requests

| Task | Outcome | Score | Wall time (s) | Tokens | Requests | Notes |
|---|---|---|---|---|---|---|
| relay-group-3svc-control | pass | 1.000 | 100.4 | 372222 | 8 | verify: PASS: relay-a, relay-b, relay-c all online, relay-c app/greeting=relay-c-ready |
| relay-group-3svc-orchestrated | pass | 1.000 | 115.8 | 284425 | 36 | speedup=0.87 (control 100.4s / orchestrated 115.8s) | coordination tax: decompose 5512 tok / 1 req | decomposition: used orchestrator's own JSON subtask split | redundant commands: 2/55 | verify: PASS: relay-a, relay-b, relay-c all online, relay-c app/greeting=relay-c-ready |
