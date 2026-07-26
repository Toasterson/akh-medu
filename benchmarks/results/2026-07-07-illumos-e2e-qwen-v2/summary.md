# illumos-e2e × qwen-v2 — 2026-07-07 20:27 UTC

- **Harness git SHA**: `38a771676e27e34b4019c3d9e20af0b87d6e04f4`
- **Task-set hash**: `0c05049432a82e168b7bbc1516032861f1783dfb55a06bd7b3a934e9e62ef8ef`
- **Model backing**: cloud (`qwen3.5:397b`)
- **Grading**: autonomous (deterministic `verify.sh` only, no LLM judge)
- **Host tier**: VM (`code.aopc.cloud/CloudNebulaProject`, recipe `c93aaef5952dd28b0c54127ae5ec77abc298d9fc894ab39ef6b27ffd2cd5523c`)

**0/6 tasks passed (0.0%), mean score 0.000, 381.7s total wall time**, 1369649 tokens, 65 requests

| Task | Outcome | Score | Wall time (s) | Tokens | Requests | Notes |
|---|---|---|---|---|---|---|
| c-01-kstat-cpu-count | fail | 0.000 | 34.4 | 157440 | 9 | verify: FAIL: /root/cpucount.c does not appear to call kstat_open/kstat_lookup | manpages: 2/3 grounded, missing [libkstat] (commands confirmed via ExecEnvelope on the remote backend) |
| c-02-libscf-read-prop | fail | 0.000 | 66.3 | 221544 | 8 | verify: FAIL: readgreeting printed 'HellofromillumosSMF!', expected 'Hellorom\illumos\SMF!' | manpages: 2/2 grounded (commands confirmed via ExecEnvelope on the remote backend) |
| c-03-libscf-smf-state | fail | 0.000 | 105.7 | 620536 | 22 | verify: FAIL: /root/svcstate.c does not appear to call scf_simple_prop_get | manpages: 2/3 grounded, missing [scf_simple_prop_get] (commands confirmed via ExecEnvelope on the remote backend) |
| rust-01-processor-info | fail | 0.000 | 22.6 | 64188 | 7 | verify: FAIL: /root/cpuclock.rs shells out (std::process::Command) instead of calling processor_info(2) directly | manpages: 1/2 grounded, missing [psrinfo] (commands confirmed via ExecEnvelope on the remote backend) |
| rust-02-door-roundtrip | fail | 0.000 | 69.7 | 167598 | 10 | verify: FAIL: /root/doorping.rs shells out (std::process::Command) instead of using the Doors API directly | manpages: 3/3 grounded (commands confirmed via ExecEnvelope on the remote backend) |
| rust-03-xattr | fail | 0.000 | 83.0 | 138343 | 9 | verify: FAIL: xattrdemo printed '', expected 'hello-xattr' | manpages: 1/2 grounded, missing [fsattr] (commands confirmed via ExecEnvelope on the remote backend) |
