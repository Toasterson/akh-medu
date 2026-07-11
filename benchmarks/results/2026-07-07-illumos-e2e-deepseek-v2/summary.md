# illumos-e2e × deepseek-v2 — 2026-07-07 20:20 UTC

- **Harness git SHA**: `38a771676e27e34b4019c3d9e20af0b87d6e04f4`
- **Task-set hash**: `0c05049432a82e168b7bbc1516032861f1783dfb55a06bd7b3a934e9e62ef8ef`
- **Model backing**: cloud (`deepseek-v4-pro`)
- **Grading**: autonomous (deterministic `verify.sh` only, no LLM judge)
- **Host tier**: VM (`code.aopc.cloud/CloudNebulaProject`, recipe `c93aaef5952dd28b0c54127ae5ec77abc298d9fc894ab39ef6b27ffd2cd5523c`)

**1/6 tasks passed (16.7%), mean score 0.167, 286.3s total wall time**, 1454401 tokens, 71 requests

| Task | Outcome | Score | Wall time (s) | Tokens | Requests | Notes |
|---|---|---|---|---|---|---|
| c-01-kstat-cpu-count | fail | 0.000 | 19.1 | 116554 | 9 | verify: FAIL: /root/cpucount.c does not appear to call kstat_open/kstat_lookup | manpages: 2/3 grounded, missing [libkstat] (commands confirmed via ExecEnvelope on the remote backend) |
| c-02-libscf-read-prop | fail | 0.000 | 52.5 | 635452 | 25 | verify: FAIL: readgreeting printed 'HellofromillumosSMF!', expected 'orange-1735-5384' | manpages: 2/2 grounded (commands confirmed via ExecEnvelope on the remote backend) |
| c-03-libscf-smf-state | fail | 0.000 | 26.6 | 360078 | 9 | verify: FAIL: /root/svcstate.c does not appear to call scf_simple_prop_get | manpages: 2/3 grounded, missing [scf_simple_prop_get] (commands confirmed via ExecEnvelope on the remote backend) |
| rust-01-processor-info | fail | 0.000 | 12.9 | 52780 | 7 | verify: FAIL: /root/cpuclock.rs shells out (std::process::Command) instead of calling processor_info(2) directly | manpages: 1/2 grounded, missing [psrinfo] (commands confirmed via ExecEnvelope on the remote backend) |
| rust-02-door-roundtrip | fail | 0.000 | 119.0 | 143808 | 10 | verify: FAIL: /root/doorping.rs shells out (std::process::Command) instead of using the Doors API directly | manpages: 3/3 grounded (commands confirmed via ExecEnvelope on the remote backend) |
| rust-03-xattr | pass | 1.000 | 56.1 | 145729 | 11 | verify: PASS: xattr round trip matches expected content | manpages: 2/2 grounded (commands confirmed via ExecEnvelope on the remote backend) |
