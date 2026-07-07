# illumos-e2e × glm-5.2-v2 — 2026-07-07 20:16 UTC

- **Harness git SHA**: `e9cca738ce5dc863907c8f61db5b7269ede77a83`
- **Task-set hash**: `0c05049432a82e168b7bbc1516032861f1783dfb55a06bd7b3a934e9e62ef8ef`
- **Model backing**: cloud (`glm-5.2`)
- **Grading**: autonomous (deterministic `verify.sh` only, no LLM judge)
- **Host tier**: VM (`code.aopc.cloud/CloudNebulaProject`, recipe `c93aaef5952dd28b0c54127ae5ec77abc298d9fc894ab39ef6b27ffd2cd5523c`)

**2/6 tasks passed (33.3%), mean score 0.333, 396.9s total wall time**, 891166 tokens, 66 requests

| Task | Outcome | Score | Wall time (s) | Tokens | Requests | Notes |
|---|---|---|---|---|---|---|
| c-01-kstat-cpu-count | fail | 0.000 | 26.7 | 133002 | 11 | verify: FAIL: /root/cpucount.c does not appear to call kstat_open/kstat_lookup | manpages: 2/3 grounded, missing [libkstat] (commands confirmed via ExecEnvelope on the remote backend) |
| c-02-libscf-read-prop | pass | 1.000 | 24.6 | 90219 | 6 | verify: PASS: readgreeting matches live config/greeting value | manpages: 2/2 grounded (commands confirmed via ExecEnvelope on the remote backend) |
| c-03-libscf-smf-state | fail | 0.000 | 65.1 | 400360 | 23 | verify: FAIL: /root/svcstate.c does not appear to call scf_simple_prop_get | manpages: 2/3 grounded, missing [scf_simple_prop_get] (commands confirmed via ExecEnvelope on the remote backend) |
| rust-01-processor-info | fail | 0.000 | 21.6 | 54841 | 7 | verify: FAIL: /root/cpuclock.rs shells out (std::process::Command) instead of calling processor_info(2) directly | manpages: 1/2 grounded, missing [psrinfo] (commands confirmed via ExecEnvelope on the remote backend) |
| rust-02-door-roundtrip | fail | 0.000 | 216.9 | 135546 | 10 | verify: FAIL: /root/doorping.rs shells out (std::process::Command) instead of using the Doors API directly | manpages: 3/3 grounded (commands confirmed via ExecEnvelope on the remote backend) |
| rust-03-xattr | pass | 1.000 | 42.0 | 77198 | 9 | verify: PASS: xattr round trip matches expected content | manpages: 1/2 grounded, missing [fsattr] (commands confirmed via ExecEnvelope on the remote backend) |
